from __future__ import annotations
import json
import torch
import numpy as np
from threading import Lock
from typing import List, Optional, Protocol, Literal

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
    """
    Tracks convergence of iterative optimization with optional LR-aware automatic patience.

    Modes
    -----
    delta    : convergence detected via absolute + relative delta
    plateau  : convergence detected when no improvement over a rolling window

    Automatic patience scaling increases patience as learning rate decreases.
    """

    def __init__(
        self,
        n_init: int,
        max_iter: int,
        patience: int = 3,
        tol: float = 1e-5,
        rel_tol: float = 1e-5,
        plateau_window: int = 5,
        plateau_tol: float = 1e-4,
        mode: Literal["delta", "plateau"] = "delta",
        callbacks: Optional[List[CallbackFn]] = None,
        patience_max: Optional[int] = None,
        patience_scale: float = 1.0,
        auto_patience: bool = True,
        early_stop: bool = True,
        verbose: bool = True,
    ):
        self.tol = tol
        self.mode = mode
        self.n_init = n_init
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.patience = patience
        self.plateau_tol = plateau_tol
        self.plateau_window = plateau_window
        self.callbacks = callbacks or []
        self.early_stop = early_stop
        self.verbose = verbose

        # auto patience
        self.patience_scale = patience_scale
        self.auto_patience = auto_patience
        self.patience_max = patience_max
        self.scheduler = None
        self._lr_ref = None

        # tracking arrays
        shape = (max_iter + 1, n_init)
        self.scores = torch.full(shape, float("nan"), dtype=DTYPE)
        self.deltas = torch.full_like(self.scores, float("nan"))
        self.rel_deltas = torch.full_like(self.scores, float("nan"))
        self.best_scores = torch.full((n_init,), float("-inf"), dtype=DTYPE)
        self.best_iters = torch.full((n_init,), -1, dtype=torch.int32)
        self.converged_flags = torch.zeros(n_init, dtype=torch.bool)
        self.stop_training = False
        self._lock = Lock()

    def attach_scheduler(self, scheduler) -> None:
        """Attach a ReduceLROnPlateau-style scheduler for LR-aware patience scaling."""
        self.scheduler = scheduler
        try:
            self._lr_ref = scheduler.optimizer.param_groups[0]["lr"]
        except Exception:
            self._lr_ref = None

    def _effective_patience(self) -> int:
        if not self.auto_patience or self.scheduler is None or self._lr_ref is None:
            return self.patience

        try:
            lr = self.scheduler.optimizer.param_groups[0]["lr"]
        except Exception:
            return self.patience

        ratio = max(lr / self._lr_ref, 1e-8)
        scale = 1.0 + self.patience_scale * (-torch.log10(torch.tensor(ratio))).item()
        p = int(round(self.patience * scale))
        if self.patience_max is not None:
            p = min(p, self.patience_max)
        return max(self.patience, p)

    # -------------------- Update / reset --------------------

    def reset(self) -> None:
        self.scores.fill_(float("nan"))
        self.deltas.fill_(float("nan"))
        self.rel_deltas.fill_(float("nan"))
        self.best_scores.fill_(float("-inf"))
        self.best_iters.fill_(-1)
        self.converged_flags.zero_()
        self.stop_training = False

    def update(self, score: float | torch.Tensor, iteration: int, init_idx: int) -> bool:
        """Record a new score and check convergence."""
        score_val = float(score.item()) if torch.is_tensor(score) else float(score)
        self._record_score(score_val, iteration, init_idx)
        converged = self._evaluate(iteration, init_idx)

        if self.scheduler is not None:
            self.scheduler.step(score_val)

        self._run_callbacks(iteration, init_idx, converged)
        if self.verbose:
            self._log(iteration, init_idx, converged)

        if self.early_stop and self.converged_flags.all():
            self.stop_training = True

        return converged

    # -------------------- Recording / evaluation --------------------

    def _record_score(self, score: float, it: int, i: int) -> None:
        self.scores[it, i] = score
        if score > self.best_scores[i]:
            self.best_scores[i] = score
            self.best_iters[i] = it
        if it == 0:
            return
        prev = self.scores[it - 1, i]
        if not torch.isfinite(prev):
            return
        delta = score - float(prev)
        self.deltas[it, i] = delta
        self.rel_deltas[it, i] = delta / (abs(float(prev)) + EPS)

    def _evaluate(self, it: int, i: int) -> bool:
        if self.mode == "delta":
            return self._delta_convergence(it, i)
        return self._plateau_convergence(it, i)

    def _delta_convergence(self, it: int, i: int) -> bool:
        p = self._effective_patience()
        if it < p:
            self.converged_flags[i] = False
            return False
        sl = slice(it - p + 1, it + 1)
        da = self.deltas[sl, i]
        dr = self.rel_deltas[sl, i]
        if not torch.isfinite(da).all() or not torch.isfinite(dr).all():
            self.converged_flags[i] = False
            return False
        ok = bool((da.abs() < self.tol).all() and (dr.abs() < self.rel_tol).all())
        self.converged_flags[i] = ok
        return ok

    def _plateau_convergence(self, it: int, i: int) -> bool:
        if it < self.plateau_window:
            self.converged_flags[i] = False
            return False
        sl = slice(it - self.plateau_window + 1, it + 1)
        window = self.scores[sl, i]
        if not torch.isfinite(window).all():
            return False
        improvement = window.max() - window.min()
        ok = bool(improvement < self.plateau_tol)
        self.converged_flags[i] = ok
        return ok

    # -------------------- Callbacks / logging --------------------

    def _run_callbacks(self, it: int, i: int, converged: bool) -> None:
        if not self.callbacks:
            return
        with self._lock:
            s = float(self.scores[it, i])
            da = float(self.deltas[it, i]) if torch.isfinite(self.deltas[it, i]) else float("nan")
            dr = float(self.rel_deltas[it, i]) if torch.isfinite(self.rel_deltas[it, i]) else float("nan")
            for fn in self.callbacks:
                try:
                    fn(self, it, i, s, da, dr, converged)
                except Exception as e:
                    logger.warning(f"[Callback Error] {fn}: {e}")

    def _log(self, it: int, i: int, converged: bool) -> None:
        logger.info(
            f"[Init {i+1:02d}] Iter {it:03d} | "
            f"Score {self.scores[it, i]:.6f} | "
            f"Δ {self.deltas[it, i]:.2e}"
            + (" ✓" if converged else "")
        )

    # -------------------- Properties / export --------------------

    @property
    def all_converged(self) -> bool:
        return bool(self.converged_flags.all())

    @property
    def any_converged(self) -> bool:
        return bool(self.converged_flags.any())

    def export(self, path: str) -> None:
        data = {
            "mode": self.mode,
            "tol": self.tol,
            "rel_tol": self.rel_tol,
            "plateau_window": self.plateau_window,
            "plateau_tol": self.plateau_tol,
            "scores": self._tensor_to_list(self.scores),
            "best_scores": self.best_scores.tolist(),
            "best_iters": self.best_iters.tolist(),
            "converged": self.converged_flags.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor) -> list:
        arr = t.cpu().numpy()
        return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]
