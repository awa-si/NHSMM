import torch
import matplotlib.pyplot as plt
import json
from typing import Callable, List, Optional


class ConvergenceHandler:
    """
    Robust convergence monitor for HMM/HSMM or neural training loops.

    Parameters
    ----------
    max_iter : int
        Maximum iterations per initialization.
    n_init : int
        Number of model restarts.
    tol : float
        Convergence tolerance for log-likelihood delta.
    post_conv_iter : int
        Number of iterations to continue after convergence before confirming.
    verbose : bool
        Print progress logs.
    callbacks : list of callables
        Each called as fn(handler, iter, rank, score, delta, converged).
    early_stop : bool
        Stop training once converged (if True).
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
    ):
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.post_conv_iter = int(post_conv_iter)
        self.verbose = verbose
        self.early_stop = early_stop
        self.callbacks = callbacks or []

        # Allocate convergence logs
        shape = (self.max_iter + 1, self.n_init)
        self.score = torch.full(shape, float("nan"), dtype=torch.float64)
        self.delta = torch.full_like(self.score, float("nan"))
        self.is_converged = False
        self.stop_training = False

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"max_iter={self.max_iter}, n_init={self.n_init}, "
                f"tol={self.tol}, post_conv_iter={self.post_conv_iter}, "
                f"early_stop={self.early_stop}, verbose={self.verbose})")

    # ----------------------------------------------------------------------

    def push_pull(self, new_score: float, iter: int, rank: int) -> bool:
        """Push a new score and immediately check convergence."""
        self.push(new_score, iter, rank)
        return self.check_converged(iter, rank)

    def push(self, new_score: float, iter: int, rank: int):
        """Store a new log-likelihood score and compute delta."""
        score_val = float(new_score.detach().item() if torch.is_tensor(new_score) else new_score)
        self.score[iter, rank] = score_val

        if iter > 0 and not torch.isnan(self.score[iter - 1, rank]):
            self.delta[iter, rank] = score_val - self.score[iter - 1, rank]

    def check_converged(self, iter: int, rank: int) -> bool:
        """Check convergence, trigger callbacks, and optionally stop training."""
        valid_deltas = self.delta[1:iter + 1, rank]
        valid_deltas = valid_deltas[~torch.isnan(valid_deltas)]

        self.is_converged = False
        if valid_deltas.numel() >= self.post_conv_iter:
            recent = valid_deltas[-self.post_conv_iter:]
            self.is_converged = torch.all(torch.abs(recent) < self.tol).item()

        # Print status
        if self.verbose:
            score = float(self.score[iter, rank])
            delta = float(self.delta[iter, rank]) if not torch.isnan(self.delta[iter, rank]) else float("nan")
            status = "✔️ Converged" if self.is_converged else ""
            print(f"[Run {rank+1}] Iter {iter:03d} | Score: {score:.6f} | Δ: {delta:.6f} {status}")

        # Callbacks
        self._trigger_callbacks(iter, rank)

        # Early stop
        if self.is_converged and self.early_stop:
            self.stop_training = True

        return self.is_converged

    # ----------------------------------------------------------------------

    def _trigger_callbacks(self, iter: int, rank: int):
        """Invoke all registered callbacks safely."""
        score = float(self.score[iter, rank])
        delta_val = self.delta[iter, rank]
        delta = float(delta_val) if not torch.isnan(delta_val) else float("nan")

        for fn in self.callbacks:
            try:
                fn(self, iter, rank, score, delta, self.is_converged)
            except Exception as e:
                if self.verbose:
                    print(f"[Callback Error] {fn.__name__}: {e}")

    def register_callback(self, fn: Callable):
        """Register a new callback dynamically."""
        self.callbacks.append(fn)

    # ----------------------------------------------------------------------

    def plot_convergence(self, show: bool = True, savepath: Optional[str] = None):
        """Visualize log-likelihood convergence across initializations."""
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(9, 5))
        iters = torch.arange(self.max_iter + 1)

        for r in range(self.n_init):
            mask = ~torch.isnan(self.score[:, r])
            ax.plot(iters[mask], self.score[mask, r].cpu(), marker="o", lw=1.5, label=f"Run #{r+1}")

        ax.set_title("Log-Likelihood Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()

        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        plt.close(fig)

    # ----------------------------------------------------------------------

    def export_log(self, path: str):
        """Export score and delta logs to JSON (handles non-finite values)."""
        data = {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "tol": self.tol,
            "scores": [[float(x) if torch.isfinite(torch.tensor(x)) else None for x in row]
                       for row in self.score.cpu().numpy().tolist()],
            "delta": [[float(x) if torch.isfinite(torch.tensor(x)) else None for x in row]
                      for row in self.delta.cpu().numpy().tolist()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Clear convergence history for reuse."""
        self.score.fill_(float("nan"))
        self.delta.fill_(float("nan"))
        self.is_converged = False
        self.stop_training = False
