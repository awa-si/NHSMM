# nhsmm/tools/convergence_vectorized.py
import json
import torch
import numpy as np
from threading import Lock
from typing import Callable, List, Optional, Protocol
import matplotlib.pyplot as plt
from nhsmm.constants import DTYPE, EPS, logger

class CallbackFn(Protocol):
    def __call__(
        self,
        monitor: "ConvergenceTracker",
        iteration: int,
        init_idx: int,
        score: float,
        delta_abs: float,
        delta_rel: float,
        converged: bool,
    ) -> None: ...

class ConvergenceTracker:
    """
    GPU-aware, vectorized EM-style convergence tracker.
    Tracks per-init likelihoods, absolute/relative deltas over a rolling window,
    best scores, optional early-stop, and callback hooks.
    """

    def __init__(
        self,
        n_init: int,
        max_iter: int,
        tol: float = 1e-5,
        rel_tol: float = 1e-5,
        patience: int = 3,
        early_stop: bool = True,
        callbacks: Optional[List[CallbackFn]] = None,
        verbose: bool = True,
        device: Optional[torch.device] = None
    ):
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rel_tol = float(rel_tol)
        self.patience = int(patience)
        self.early_stop = early_stop
        self.verbose = verbose
        self.device = device or torch.device("cpu")
        self.callbacks = callbacks or []

        # Main tracking tensors
        shape = (self.max_iter + 1, self.n_init)
        self.scores = torch.full(shape, float("nan"), dtype=DTYPE, device=self.device)
        self.deltas = torch.full_like(self.scores, float("nan"))
        self.rel_deltas = torch.full_like(self.scores, float("nan"))
        self.converged_flags = torch.zeros(self.n_init, dtype=torch.bool, device=self.device)

        # Rolling window buffers
        self._rolling_abs = torch.full((self.n_init, self.patience), float("nan"), dtype=DTYPE, device=self.device)
        self._rolling_rel = torch.full_like(self._rolling_abs, float("nan"))

        # Best scores
        self.best_scores = torch.full((self.n_init,), float("-inf"), dtype=DTYPE, device=self.device)
        self.best_iters = torch.full((self.n_init,), -1, dtype=torch.int32, device=self.device)

        # Thread lock
        self._lock = Lock()
        self.stop_training = False

    # ------------------------ Public API ------------------------

    def update(self, score: float | torch.Tensor, iteration: int, init_idx: int) -> bool:
        self._record_score(score, iteration, init_idx)
        return self._evaluate_convergence(iteration, init_idx)

    def add_callback(self, fn: CallbackFn):
        if fn not in self.callbacks:
            self.callbacks.append(fn)

    def reset(self):
        self.scores.fill_(float("nan"))
        self.deltas.fill_(float("nan"))
        self.rel_deltas.fill_(float("nan"))
        self.converged_flags.zero_()
        self._rolling_abs.fill_(float("nan"))
        self._rolling_rel.fill_(float("nan"))
        self.best_scores.fill_(float("-inf"))
        self.best_iters.fill_(-1)
        self.stop_training = False

    # ------------------------ Internal: scoring ------------------------

    def _record_score(self, score: float | torch.Tensor, iteration: int, init_idx: int):
        val = score.detach() if torch.is_tensor(score) else torch.tensor(float(score), dtype=DTYPE, device=self.device)
        self.scores[iteration, init_idx] = val

        # Update best
        if val > self.best_scores[init_idx]:
            self.best_scores[init_idx] = val
            self.best_iters[init_idx] = iteration

        # Skip delta calculation for first iteration
        if iteration == 0:
            return

        prev = self.scores[iteration - 1, init_idx]
        if not torch.isfinite(prev):
            return

        delta = val - prev
        rel_delta = delta / (prev.abs() + EPS)
        self.deltas[iteration, init_idx] = delta
        self.rel_deltas[iteration, init_idx] = rel_delta

        # Rolling buffers (vectorized)
        self._rolling_abs[init_idx, :-1] = self._rolling_abs[init_idx, 1:]
        self._rolling_abs[init_idx, -1] = delta
        self._rolling_rel[init_idx, :-1] = self._rolling_rel[init_idx, 1:]
        self._rolling_rel[init_idx, -1] = rel_delta

    # ------------------------ Internal: convergence ------------------------

    def _evaluate_convergence(self, iteration: int, init_idx: int) -> bool:
        if iteration < self.patience:
            self.converged_flags[init_idx] = False
            return False

        buf_abs = self._rolling_abs[init_idx]
        buf_rel = self._rolling_rel[init_idx]

        valid_abs = buf_abs[torch.isfinite(buf_abs)]
        valid_rel = buf_rel[torch.isfinite(buf_rel)]

        conv_abs = valid_abs.numel() == self.patience and (valid_abs.abs() < self.tol).all()
        conv_rel = valid_rel.numel() == self.patience and (valid_rel.abs() < self.rel_tol).all()
        converged = bool(conv_abs and conv_rel)

        self.converged_flags[init_idx] = converged

        # Callbacks
        self._run_callbacks(iteration, init_idx, converged)

        # Verbose logging
        if self.verbose:
            s = float(self.scores[iteration, init_idx])
            da = float(self.deltas[iteration, init_idx]) if torch.isfinite(self.deltas[iteration, init_idx]) else float("nan")
            dr = float(self.rel_deltas[iteration, init_idx]) if torch.isfinite(self.rel_deltas[iteration, init_idx]) else float("nan")
            icon = "✔️" if converged else ""
            logger.info(f"[Init {init_idx+1:02d}] Iter {iteration:03d} | Score: {s:.6f} | Δ: {da:.3e} | Δ%: {dr:.3e} {icon}")

        # Global early stop
        if self.early_stop and self.converged_flags.all():
            self.stop_training = True

        return converged

    # ------------------------ Callback ------------------------

    def _run_callbacks(self, iteration: int, init_idx: int, converged: bool):
        with self._lock:
            s = float(self.scores[iteration, init_idx])
            da = float(self.deltas[iteration, init_idx]) if torch.isfinite(self.deltas[iteration, init_idx]) else float("nan")
            dr = float(self.rel_deltas[iteration, init_idx]) if torch.isfinite(self.rel_deltas[iteration, init_idx]) else float("nan")
            for fn in self.callbacks:
                try:
                    fn(self, iteration, init_idx, s, da, dr, converged)
                except Exception as e:
                    import traceback
                    logger.warning(f"[Callback Error] {fn}: {traceback.format_exc()}")

    # ------------------------ Plot ------------------------

    def plot(self, show: bool = True, savepath: Optional[str] = None,
             title: str = "Convergence Progress", log_scale: bool = False):

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(9, 5))
        iters = torch.arange(self.max_iter + 1, device=self.device)

        for r in range(self.n_init):
            mask = torch.isfinite(self.scores[:, r])
            if mask.any():
                ax.plot(iters[mask].cpu(), self.scores[mask, r].cpu(), marker="o", lw=1.5, label=f"Init {r+1}")
                ax.scatter([self.best_iters[r].cpu()], [self.best_scores[r].cpu()], color="black", marker="x", s=60, zorder=5)

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score / Log-Likelihood")
        if log_scale:
            ax.set_yscale("log", nonpositive='clip')
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()

        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------ Export ------------------------

    def export(self, path: str):
        data = {
            "tol": self.tol,
            "rel_tol": self.rel_tol,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "scores": self._tensor_to_list(self.scores),
            "deltas": self._tensor_to_list(self.deltas),
            "rel_deltas": self._tensor_to_list(self.rel_deltas),
            "converged": self.converged_flags.cpu().tolist(),
            "best_scores": self.best_scores.cpu().tolist(),
            "best_iters": self.best_iters.cpu().tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor):
        arr = t.cpu().numpy()
        return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]
