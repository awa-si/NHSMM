import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from typing import Callable, List, Optional, Dict, Any
from threading import Lock


class ConvergenceHandler:
    """
    GPU-compatible convergence monitor for EM or neural training loops.

    Tracks per-initialization log-likelihoods, deltas, and convergence status
    in a thread-safe, vectorized manner. Supports callback hooks, live plotting,
    and JSON export for post-analysis.

    Args:
        max_iter (int): Maximum number of iterations to track.
        n_init (int): Number of random initializations or runs.
        tol (float): Absolute tolerance for convergence detection.
        post_conv_iter (int): Number of consecutive small deltas to confirm convergence.
        verbose (bool): Whether to print progress each iteration.
        callbacks (list[Callable], optional): User-defined callback hooks.
        early_stop (bool): If True, training halts once any init converges.
        device (torch.device, optional): Device to store tensors on.
    """

    def __init__(
        self,
        max_iter: int,
        n_init: int,
        tol: float = 1e-5,
        post_conv_iter: int = 3,
        verbose: bool = True,
        callbacks: Optional[List[Callable]] = None,
        early_stop: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.post_conv_iter = int(post_conv_iter)
        self.verbose = verbose
        self.early_stop = early_stop
        self.callbacks = callbacks or []
        self.device = device or torch.device("cpu")

        shape = (self.max_iter + 1, self.n_init)
        self.score = torch.full(shape, float("nan"), dtype=torch.float64, device=self.device)
        self.delta = torch.full_like(self.score, float("nan"))
        self.is_converged_per_init = torch.zeros(self.n_init, dtype=torch.bool, device=self.device)
        self.stop_training = False
        self._rolling_deltas = [
            torch.full((self.post_conv_iter,), float("nan"), dtype=torch.float64, device=self.device)
            for _ in range(self.n_init)
        ]
        self._lock = Lock()

    # ----------------------------------------------------------------------
    # Core API
    # ----------------------------------------------------------------------
    def push_pull(self, new_score: torch.Tensor, iteration: int, rank: int) -> bool:
        """
        Push a new score and immediately check for convergence.

        Args:
            new_score (torch.Tensor | float): New log-likelihood value.
            iteration (int): Current iteration index.
            rank (int): Index of the current initialization.

        Returns:
            bool: True if this initialization has converged.
        """
        self.push(new_score, iteration, rank)
        return self.check_converged(iteration, rank)

    def push(self, new_score: torch.Tensor, iteration: int, rank: int):
        """Store a new score and update rolling deltas (GPU-safe)."""
        score_val = (
            new_score.detach()
            if torch.is_tensor(new_score)
            else torch.tensor(float(new_score), dtype=torch.float64, device=self.device)
        )
        self.score[iteration, rank] = score_val

        if iteration > 0 and not torch.isnan(self.score[iteration - 1, rank]):
            delta = score_val - self.score[iteration - 1, rank]
            self.delta[iteration, rank] = delta
            buf = self._rolling_deltas[rank]
            buf[:-1] = buf[1:]
            buf[-1] = delta

    def check_converged(self, iteration: int, rank: int) -> bool:
        """Evaluate convergence for one initialization based on recent deltas."""
        buf = self._rolling_deltas[rank]
        valid_deltas = buf[~torch.isnan(buf)]

        converged = (
            valid_deltas.numel() >= self.post_conv_iter
            and torch.all(torch.abs(valid_deltas) < self.tol).item()
        )

        self.is_converged_per_init[rank] = converged
        self._trigger_callbacks(iteration, rank, converged)

        if self.verbose:
            score = float(self.score[iteration, rank].cpu())
            delta = (
                float(self.delta[iteration, rank].cpu())
                if not torch.isnan(self.delta[iteration, rank])
                else float("nan")
            )
            status = "✔️ Converged" if converged else ""
            print(f"[Init {rank+1:02d}] Iter {iteration:03d} | Score: {score:.6f} | Δ: {delta:.3e} {status}")

        if converged and self.early_stop:
            self.stop_training = True

        return converged

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------
    def register_callback(self, fn: Callable):
        """Register a user-defined callback."""
        if fn not in self.callbacks:
            self.callbacks.append(fn)

    def _trigger_callbacks(self, iteration: int, rank: int, converged: bool):
        """Invoke all registered callbacks with thread safety."""
        with self._lock:
            score = self.score[iteration, rank].item()
            delta_val = self.delta[iteration, rank]
            delta = delta_val.item() if not torch.isnan(delta_val) else float("nan")
            for fn in self.callbacks:
                try:
                    fn(self, iteration, rank, score, delta, converged)
                except Exception as e:
                    if self.verbose:
                        print(f"[Callback Error] {fn.__name__}: {e}")

    # ----------------------------------------------------------------------
    # Visualization & Logging
    # ----------------------------------------------------------------------
    def plot_convergence(
        self,
        show: bool = True,
        savepath: Optional[str] = None,
        title: str = "Convergence Curve",
        log_scale: bool = False,
    ):
        """
        Plot per-initialization convergence curves.

        Args:
            show (bool): Display the plot interactively.
            savepath (str, optional): Path to save figure.
            title (str): Plot title.
            log_scale (bool): Use log scale on the y-axis.
        """
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(9, 5))
        iters = torch.arange(self.max_iter + 1, device=self.device)

        for r in range(self.n_init):
            mask = ~torch.isnan(self.score[:, r])
            if mask.any():
                ax.plot(iters[mask].cpu(), self.score[mask, r].cpu(), marker="o", lw=1.5, label=f"Init {r+1}")

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood")
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
        """Export convergence logs to a JSON file."""
        data = {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "tol": self.tol,
            "scores": self._tensor_to_list(self.score),
            "delta": self._tensor_to_list(self.delta),
            "converged": self.is_converged_per_init.cpu().tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor) -> List[List[Optional[float]]]:
        """Convert tensor to JSON-serializable nested list."""
        arr = t.cpu().numpy()
        return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]

    def reset(self):
        """Reset all convergence state."""
        self.score.fill_(float("nan"))
        self.delta.fill_(float("nan"))
        self.is_converged_per_init.fill_(False)
        self.stop_training = False
        for buf in self._rolling_deltas:
            buf.fill_(float("nan"))
