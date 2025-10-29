import json
import torch
import numpy as np
from threading import Lock
import matplotlib.pyplot as plt

from typing import Callable, List, Optional


class ConvergenceMonitor:
    """
    Tracks iterative convergence for EM-like algorithms.
    """
    def __init__(
        self,
        max_iter: int,
        n_init: int,
        tol: float = 1e-5,
        rel_tol: float = 1e-5,
        post_conv_iter: int = 3,
        verbose: bool = True,
        callbacks: Optional[List[Callable]] = None,
        early_stop: bool = True,
        device: Optional[torch.device] = None
    ):
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.rel_tol = float(rel_tol)
        self.post_conv_iter = int(post_conv_iter)
        self.verbose = verbose
        self.early_stop = early_stop
        self.callbacks = callbacks or []
        self.device = device or torch.device("cpu")

        shape = (self.max_iter + 1, self.n_init)
        self.score = torch.full(shape, float("nan"), dtype=torch.float64, device=self.device)
        self.delta = torch.full_like(self.score, float("nan"))
        self.rel_delta = torch.full_like(self.score, float("nan"))
        self.is_converged_per_init = torch.zeros(self.n_init, dtype=torch.bool, device=self.device)
        self.stop_training = False
        self._rolling_deltas = [
            torch.full((self.post_conv_iter,), float("nan"), dtype=torch.float64, device=self.device)
            for _ in range(self.n_init)
        ]
        self._rolling_rel_deltas = [
            torch.full((self.post_conv_iter,), float("nan"), dtype=torch.float64, device=self.device)
            for _ in range(self.n_init)
        ]
        self._lock = Lock()

    def push_pull(self, new_score: torch.Tensor, iteration: int, rank: int) -> bool:
        """Push new score and check convergence."""
        self.push(new_score, iteration, rank)
        return self.check_converged(iteration, rank)

    def push(self, new_score: torch.Tensor, iteration: int, rank: int):
        """Record new score, compute absolute and relative delta, and update rolling buffers."""
        val = new_score.detach() if torch.is_tensor(new_score) else torch.tensor(float(new_score), dtype=torch.float64, device=self.device)
        self.score[iteration, rank] = val

        if iteration > 0 and not torch.isnan(self.score[iteration - 1, rank]):
            delta = val - self.score[iteration - 1, rank]
            self.delta[iteration, rank] = delta

            # Relative change
            prev_val = self.score[iteration - 1, rank].abs()
            rel_delta = (delta / prev_val) if prev_val > 0 else float("nan")
            self.rel_delta[iteration, rank] = rel_delta

            # Rolling buffers
            buf = self._rolling_deltas[rank]
            buf[:-1] = buf[1:]
            buf[-1] = delta

            buf_rel = self._rolling_rel_deltas[rank]
            buf_rel[:-1] = buf_rel[1:]
            buf_rel[-1] = rel_delta

    def check_converged(self, iteration: int, rank: int) -> bool:
        """Check if convergence criteria are met and trigger callbacks."""
        buf = self._rolling_deltas[rank]
        buf_rel = self._rolling_rel_deltas[rank]

        valid_abs = buf[~torch.isnan(buf)]
        valid_rel = buf_rel[~torch.isnan(buf_rel)]

        conv_abs = valid_abs.numel() >= self.post_conv_iter and torch.all(torch.abs(valid_abs) < self.tol).item()
        conv_rel = valid_rel.numel() >= self.post_conv_iter and torch.all(torch.abs(valid_rel) < self.rel_tol).item()
        conv = conv_abs and conv_rel

        self.is_converged_per_init[rank] = conv

        self._trigger_callbacks(iteration, rank, conv)

        if self.verbose:
            s = float(self.score[iteration, rank].item())
            d_abs = float(self.delta[iteration, rank].item()) if not torch.isnan(self.delta[iteration, rank]) else float("nan")
            d_rel = float(self.rel_delta[iteration, rank].item()) if not torch.isnan(self.rel_delta[iteration, rank]) else float("nan")
            status = "✔️ Converged" if conv else ""
            print(f"[Init {rank+1:02d}] Iter {iteration:03d} | Score: {s:.6f} | Δ: {d_abs:.3e} | Δ%: {d_rel:.3e} {status}")

        if conv and self.early_stop:
            self.stop_training = True

        return conv

    def register_callback(self, fn: Callable):
        """Add a callback function triggered each iteration."""
        if fn not in self.callbacks:
            self.callbacks.append(fn)

    def _trigger_callbacks(self, iteration: int, rank: int, conv: bool):
        """Safely call all registered callbacks."""
        with self._lock:
            s = float(self.score[iteration, rank].item())
            d_abs = float(self.delta[iteration, rank].item()) if not torch.isnan(self.delta[iteration, rank]) else float("nan")
            d_rel = float(self.rel_delta[iteration, rank].item()) if not torch.isnan(self.rel_delta[iteration, rank]) else float("nan")
            for fn in self.callbacks:
                try:
                    fn(self, iteration, rank, s, d_abs, d_rel, conv)
                except Exception as e:
                    if self.verbose:
                        print(f"[Callback Error] {fn.__name__}: {e}")

    def plot_convergence(self, show: bool = True, savepath: Optional[str] = None,
                         title: str = "Convergence Curve", log_scale: bool = False):
        """Plot convergence curves for all initializations."""
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(9, 5))
        iters = torch.arange(self.max_iter + 1, device=self.device)
        for r in range(self.n_init):
            mask = ~torch.isnan(self.score[:, r])
            if mask.any():
                ax.plot(iters[mask].cpu(), self.score[mask, r].cpu(), marker="o", lw=1.5, label=f"Init {r+1}")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score / Log-Likelihood")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def export_log(self, path: str):
        """Export scores, deltas, relative deltas, and convergence status to JSON."""
        data = {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "tol": self.tol,
            "rel_tol": self.rel_tol,
            "scores": self._tensor_to_list(self.score),
            "delta": self._tensor_to_list(self.delta),
            "rel_delta": self._tensor_to_list(self.rel_delta),
            "converged": self.is_converged_per_init.cpu().tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor) -> List[List[Optional[float]]]:
        """Convert tensor to nested list, replacing NaNs with None."""
        arr = t.cpu().numpy()
        return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]

    def reset(self):
        """Reset all internal states."""
        self.score.fill_(float("nan"))
        self.delta.fill_(float("nan"))
        self.rel_delta.fill_(float("nan"))
        self.is_converged_per_init.fill_(False)
        self.stop_training = False
        for buf in self._rolling_deltas:
            buf.fill_(float("nan"))
        for buf in self._rolling_rel_deltas:
            buf.fill_(float("nan"))
