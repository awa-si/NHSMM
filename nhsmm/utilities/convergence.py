import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from typing import Callable, List, Optional
from threading import Lock


class ConvergenceHandler:
    """
    GPU-compatible convergence monitor for HMM/HSMM or neural training loops.

    Tracks log-likelihood scores, deltas, and convergence per initialization, fully
    vectorized and device-aware.
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
        device: Optional[torch.device] = None
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

        self._rolling_deltas = [torch.full((self.post_conv_iter,), float("nan"),
                                           dtype=torch.float64, device=self.device)
                                for _ in range(self.n_init)]
        self._lock = Lock()

    def push_pull(self, new_score: torch.Tensor, iteration: int, rank: int) -> bool:
        self.push(new_score, iteration, rank)
        return self.check_converged(iteration, rank)

    def push(self, new_score: torch.Tensor, iteration: int, rank: int):
        """Store a new score and compute delta (works with GPU tensors)."""
        score_val = new_score.detach() if torch.is_tensor(new_score) else torch.tensor(float(new_score), device=self.device)
        self.score[iteration, rank] = score_val

        if iteration > 0 and not torch.isnan(self.score[iteration - 1, rank]):
            self.delta[iteration, rank] = score_val - self.score[iteration - 1, rank]
            buf = self._rolling_deltas[rank]
            buf[:-1] = buf[1:]
            buf[-1] = self.delta[iteration, rank]

    def check_converged(self, iteration: int, rank: int) -> bool:
        buf = self._rolling_deltas[rank]
        valid_deltas = buf[~torch.isnan(buf)]
        converged = valid_deltas.numel() >= self.post_conv_iter and torch.all(torch.abs(valid_deltas) < self.tol).item()
        self.is_converged_per_init[rank] = converged

        if self.verbose:
            score = float(self.score[iteration, rank].cpu())
            delta = float(self.delta[iteration, rank].cpu()) if not torch.isnan(self.delta[iteration, rank]) else float("nan")
            status = "✔️ Converged" if converged else ""
            print(f"[Run {rank+1}] Iter {iteration:03d} | Score: {score:.6f} | Δ: {delta:.6f} {status}")

        self._trigger_callbacks(iteration, rank, converged)

        if converged and self.early_stop:
            self.stop_training = True

        return converged

    def _trigger_callbacks(self, iteration: int, rank: int, converged: bool):
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

    def register_callback(self, fn: Callable):
        self.callbacks.append(fn)

    # ----------------------------------------------------------------------

    def plot_convergence(self, show: bool = True, savepath: Optional[str] = None):
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(9, 5))
        iters = torch.arange(self.max_iter + 1, device=self.device)

        for r in range(self.n_init):
            mask = ~torch.isnan(self.score[:, r])
            if mask.any():
                ax.plot(iters[mask].cpu(), self.score[mask, r].cpu(), marker="o", lw=1.5, label=f"Run #{r+1}")

        ax.set_title("Log-Likelihood Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()

        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------------------------------------------------------------

    def export_log(self, path: str):
        score_data = self.score.cpu().numpy()
        delta_data = self.delta.cpu().numpy()
        data = {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "tol": self.tol,
            "scores": [[float(x) if np.isfinite(x) else None for x in row] for row in score_data],
            "delta": [[float(x) if np.isfinite(x) else None for x in row] for row in delta_data],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        self.score.fill_(float("nan"))
        self.delta.fill_(float("nan"))
        self.is_converged_per_init.fill_(False)
        self.stop_training = False
        for buf in self._rolling_deltas:
            buf.fill_(float("nan"))
