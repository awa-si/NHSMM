# nhsmm/convergence.py
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock
from typing import List, Optional, Protocol

from nhsmm.config import DTYPE, EPS, logger


class CallbackFn(Protocol):
    def __call__(
        self,
        monitor: "Convergence",
        iteration: int,
        init_idx: int,
        score: float,
        delta_abs: float,
        delta_rel: float,
        converged: bool
    ) -> None: ...


class Convergence:

    def __init__(
        self,
        n_init: int,
        max_iter: int,
        patience: int = 3,
        tol: float = 1e-5,
        rel_tol: float = 1e-5,
        early_stop: bool = True,
        callbacks: Optional[List[CallbackFn]] = None,
        verbose: bool = True,
    ):
        self.n_init = n_init
        self.max_iter = max_iter
        self.patience = patience
        self.tol = tol
        self.rel_tol = rel_tol
        self.early_stop = early_stop
        self.callbacks = callbacks or []
        self.verbose = verbose

        shape = (max_iter + 1, n_init)
        self.scores = torch.full(shape, float("nan"), dtype=DTYPE)
        self.deltas = torch.full_like(self.scores, float("nan"))
        self.rel_deltas = torch.full_like(self.scores, float("nan"))

        self.best_scores = torch.full((n_init,), float("-inf"), dtype=DTYPE)
        self.best_iters = torch.full((n_init,), -1, dtype=torch.int32)
        self.converged_flags = torch.zeros(n_init, dtype=torch.bool)
        self.stop_training = False
        self._lock = Lock()

    def update(self, score: float | torch.Tensor, iteration: int, init_idx: int) -> bool:
        """Record score and check convergence."""
        self._record(score, iteration, init_idx)
        return self._check_convergence(iteration, init_idx)

    def reset(self):
        self.scores.fill_(float("nan"))
        self.deltas.fill_(float("nan"))
        self.rel_deltas.fill_(float("nan"))
        self.best_scores.fill_(float("-inf"))
        self.best_iters.fill_(-1)
        self.converged_flags.zero_()
        self.stop_training = False

    def _record(self, score, iteration: int, init_idx: int):
        val = score.item() if torch.is_tensor(score) else float(score)
        self.scores[iteration, init_idx] = val

        # Update best
        if val > self.best_scores[init_idx]:
            self.best_scores[init_idx] = val
            self.best_iters[init_idx] = iteration

        if iteration == 0:
            return

        prev = self.scores[iteration - 1, init_idx]
        if not np.isfinite(prev):
            return

        delta = val - prev
        rel_delta = delta / (abs(prev) + EPS)

        self.deltas[iteration, init_idx] = delta
        self.rel_deltas[iteration, init_idx] = rel_delta

    def _check_convergence(self, iteration: int, init_idx: int) -> bool:
        if iteration < self.patience:
            self.converged_flags[init_idx] = False
            return False

        da = self.deltas[iteration - self.patience + 1: iteration + 1, init_idx].abs()
        dr = self.rel_deltas[iteration - self.patience + 1: iteration + 1, init_idx].abs()

        if not torch.isfinite(da).all() or not torch.isfinite(dr).all():
            self.converged_flags[init_idx] = False
            return False

        converged = (da < self.tol).all() and (dr < self.rel_tol).all()
        self.converged_flags[init_idx] = converged

        self._run_callbacks(iteration, init_idx, converged)

        if self.verbose:
            logger.info(
                f"[Init {init_idx+1:02d}] Iter {iteration:03d} | "
                f"Score {self.scores[iteration, init_idx]:.6f} | "
                f"Δ {self.deltas[iteration, init_idx]:.3e} | "
                f"Δ% {self.rel_deltas[iteration, init_idx]:.3e}"
                + (" ✓" if converged else "")
            )

        if self.early_stop and self.converged_flags.all():
            self.stop_training = True

        return bool(converged)

    def _run_callbacks(self, iteration: int, init_idx: int, converged: bool):
        with self._lock:
            s = float(self.scores[iteration, init_idx])
            da = float(self.deltas[iteration, init_idx]) if torch.isfinite(self.deltas[iteration, init_idx]) else float("nan")
            dr = float(self.rel_deltas[iteration, init_idx]) if torch.isfinite(self.rel_deltas[iteration, init_idx]) else float("nan")
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

            ax.plot(
                iters[mask].numpy(),
                self.scores[mask, i].numpy(),
                lw=1.5,
                marker="o",
                label=f"Init {i+1}"
            )

            bi = self.best_iters[i].item()
            if bi >= 0:
                ax.scatter(bi, self.best_scores[i].item(), marker="x", s=60)

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

    def export(self, path: str):
        data = {
            "tol": self.tol,
            "rel_tol": self.rel_tol,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "scores": self._tensor_to_list(self.scores),
            "deltas": self._tensor_to_list(self.deltas),
            "rel_deltas": self._tensor_to_list(self.rel_deltas),
            "converged": self.converged_flags.numpy().tolist(),
            "best_scores": self.best_scores.numpy().tolist(),
            "best_iters": self.best_iters.numpy().tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor):
        arr = t.numpy()
        return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]

