# nhsmm/convergence.py

import json
import torch
import numpy as np
from threading import Lock
import matplotlib.pyplot as plt
from typing import List, Optional, Protocol

from nhsmm.constants import DTYPE, EPS, logger, NEG_INF


class CallbackFn(Protocol):

    def __call__(
        self,
        monitor: "ConvergenceTracker",
        iteration: int,
        init_idx: int,
        score: float,
        delta_abs: float,
        delta_rel: float,
        converged: bool
    ) -> None: ...


class ConvergenceTracker:

    __slots__ = (
        "n_init", "max_iter", "tol", "rel_tol", "patience", "early_stop",
        "callbacks", "verbose", "device", "scores", "deltas", "rel_deltas",
        "converged_flags", "_rolling_abs", "_rolling_rel", "best_scores",
        "best_iters", "_lock", "stop_training", "history"
    )

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
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.rel_tol = rel_tol
        self.patience = patience
        self.early_stop = early_stop
        self.verbose = verbose
        self.device = device or torch.device("cpu")
        self.callbacks = callbacks or []

        shape = (max_iter + 1, n_init)

        self.scores = torch.full(shape, float("nan"), dtype=DTYPE, device=self.device)
        self.deltas = torch.full_like(self.scores, float("nan"))
        self.rel_deltas = torch.full_like(self.scores, float("nan"))

        self.converged_flags = torch.zeros(n_init, dtype=torch.bool, device=self.device)

        self._rolling_abs = torch.full(
            (n_init, patience), float("nan"), dtype=DTYPE, device=self.device
        )
        self._rolling_rel = torch.full_like(self._rolling_abs, float("nan"))

        self.best_scores = torch.full(
            (n_init,), float("-inf"), dtype=DTYPE, device=self.device
        )
        self.best_iters = torch.full(
            (n_init,), -1, dtype=torch.int32, device=self.device
        )

        self._lock = Lock()
        self.stop_training = False
        self.history: List[List[Optional[float]]] = [[] for _ in range(n_init)]

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
        self.history = [[] for _ in range(self.n_init)]

    def _record_score(self, score, iteration: int, init_idx: int):
        if torch.is_tensor(score):
            val = score.detach().to(self.device, dtype=DTYPE)
        else:
            val = torch.tensor(score, dtype=DTYPE, device=self.device)

        self.scores[iteration, init_idx] = val
        self.history[init_idx].append(float(val))

        if val > self.best_scores[init_idx]:
            self.best_scores[init_idx] = val
            self.best_iters[init_idx] = iteration

        if iteration == 0:
            return

        prev = self.scores[iteration - 1, init_idx]
        if not torch.isfinite(prev):
            return

        delta = val - prev
        rel_delta = delta / (prev.abs() + EPS)

        self.deltas[iteration, init_idx] = delta
        self.rel_deltas[iteration, init_idx] = rel_delta

        self._rolling_abs[init_idx] = torch.cat(
            [self._rolling_abs[init_idx, 1:], delta.abs().view(1)]
        )
        self._rolling_rel[init_idx] = torch.cat(
            [self._rolling_rel[init_idx, 1:], rel_delta.abs().view(1)]
        )

    def _evaluate_convergence(self, iteration: int, init_idx: int) -> bool:
        if iteration < self.patience:
            self.converged_flags[init_idx] = False
            return False

        abs_buf = self._rolling_abs[init_idx]
        rel_buf = self._rolling_rel[init_idx]

        if not torch.isfinite(abs_buf).all() or not torch.isfinite(rel_buf).all():
            self.converged_flags[init_idx] = False
            return False

        converged = bool(
            (abs_buf < self.tol).all() and
            (rel_buf < self.rel_tol).all()
        )

        self.converged_flags[init_idx] = converged
        self._run_callbacks(iteration, init_idx, converged)

        if self.verbose:
            da = self.deltas[iteration, init_idx]
            dr = self.rel_deltas[iteration, init_idx]
            logger.info(
                f"[Init {init_idx+1:02d}] Iter {iteration:03d} | "
                f"Score {float(self.scores[iteration, init_idx]):.6f} | "
                f"Δ {float(da):.3e} | Δ% {float(dr):.3e}"
                + (" ✓" if converged else "")
            )

        if self.early_stop and self.converged_flags.all():
            self.stop_training = True

        return converged

    def _run_callbacks(self, iteration: int, init_idx: int, converged: bool):
        with self._lock:
            s = float(self.scores[iteration, init_idx])
            da = self.deltas[iteration, init_idx]
            dr = self.rel_deltas[iteration, init_idx]
            da = float(da) if torch.isfinite(da) else float("nan")
            dr = float(dr) if torch.isfinite(dr) else float("nan")

            for fn in self.callbacks:
                try:
                    fn(self, iteration, init_idx, s, da, dr, converged)
                except Exception as e:
                    logger.warning(f"[Callback Error] {fn}: {e}")

    def plot(self, show: bool = True, savepath: Optional[str] = None, title: str = "Convergence Progress", log_scale: bool = False):

        fig, ax = plt.subplots(figsize=(9, 5))
        iters = torch.arange(self.max_iter + 1)

        for i in range(self.n_init):
            mask = torch.isfinite(self.scores[:, i])
            if not mask.any():
                continue

            y = self.scores[mask, i].cpu()
            x = iters[mask].cpu()
            ax.plot(x, y, lw=1.5, marker="o", label=f"Init {i+1}")

            bi = self.best_iters[i].item()
            if bi >= 0:
                ax.scatter(bi, self.best_scores[i].cpu(), marker="x", s=60)

        ax.set(title=title, xlabel="Iteration", ylabel="Score")
        if log_scale:
            ax.set_yscale("log", nonpositive="clip")
        ax.legend(fontsize="small")
        fig.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ---------------- export ----------------

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
            "history": self.history
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor):
        arr = t.cpu().numpy()
        return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]
