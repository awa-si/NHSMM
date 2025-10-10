from typing import Optional, Union, Callable
import torch
from torch import nn
from torch.distributions import Multinomial

from nhsmm.hsmm.BaseHSMM import BaseHSMM
from nhsmm.utilities import utils, constraints


class MultinomialHSMM(BaseHSMM):
    """
    Neural-Compatible Multinomial Hidden Semi-Markov Model (HSMM)
    -------------------------------------------------------------
    HSMM with categorical (discrete) or neural emissions.

    Emission interpretation:
        n_trials = 1, n_features = 2 → Bernoulli
        n_trials = 1, n_features > 2 → Categorical
        n_trials > 1, n_features = 2 → Binomial
        n_trials > 1, n_features > 2 → Multinomial

    Optional Neural Integration:
        You can attach a neural encoder (e.g., CNN, LSTM, Transformer) to
        model complex contextual emissions. The encoder should output
        logits of shape [batch, n_features].

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_features : int
        Number of emission categories.
    max_duration : int
        Maximum duration per state.
    n_trials : int, default=1
        Number of multinomial trials.
    alpha : float, default=1.0
        Dirichlet concentration for priors.
    seed : int, optional
        Random seed.
    encoder : Optional[nn.Module], default=None
        Neural module that maps X → logits (N × n_features)
    encoder_fn : Optional[Callable], default=None
        Custom callable that produces logits(X).
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        n_trials: int = 1,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        encoder_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.n_features = n_features
        self.n_trials = n_trials
        self.encoder = encoder
        self.encoder_fn = encoder_fn
        super().__init__(n_states, max_duration, alpha, seed)

    @property
    def dof(self) -> int:
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1

    # -------------------------------------------------------------------------
    # --- EMISSION INITIALIZATION
    # -------------------------------------------------------------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Multinomial:
        eps = 1e-8
        if X is not None:
            if X.ndim == 2:
                freqs = X.float().mean(dim=0)
            else:
                freqs = torch.bincount(X.long(), minlength=self.n_features).float()
                freqs /= freqs.sum()
            logits = freqs.clamp_min(eps).log().expand(self.n_states, -1)
        else:
            probs = constraints.sample_probs(self.alpha, (self.n_states, self.n_features))
            logits = probs.clamp_min(eps).log()
        return Multinomial(total_count=self.n_trials, logits=logits)

    # -------------------------------------------------------------------------
    # --- EMISSION UPDATE
    # -------------------------------------------------------------------------
    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
    ) -> Multinomial:
        new_B = self._compute_B(X, posterior, theta).clamp_min(1e-8)
        return Multinomial(total_count=self.n_trials, logits=new_B.log())

    # -------------------------------------------------------------------------
    # --- EMISSION COMPUTATION
    # -------------------------------------------------------------------------
    def _compute_B(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
    ) -> torch.Tensor:
        """
        Compute emission probabilities for each hidden state.

        Automatically routes through neural encoder if available.
        """
        if self.encoder or self.encoder_fn:
            logits = self._neural_forward(X)
            probs = torch.softmax(logits, dim=-1)
            weighted_counts = posterior.T @ probs
            weighted_counts /= weighted_counts.sum(dim=1, keepdim=True)
            return weighted_counts

        if X.ndim == 1:
            X_onehot = torch.nn.functional.one_hot(X.long(), num_classes=self.n_features).float()
        else:
            X_onehot = X.float()

        weighted_counts = posterior.T @ X_onehot
        weighted_counts /= weighted_counts.sum(dim=1, keepdim=True)
        return weighted_counts

    # -------------------------------------------------------------------------
    # --- NEURAL INTERFACE
    # -------------------------------------------------------------------------
    def _neural_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attached encoder or encoder_fn.

        Returns
        -------
        logits : torch.Tensor
            [N × n_features] tensor of emission logits.
        """
        if self.encoder_fn is not None:
            logits = self.encoder_fn(X)
        elif self.encoder is not None:
            self.encoder.eval()  # inference mode
            with torch.no_grad():
                logits = self.encoder(X)
        else:
            raise RuntimeError("No encoder or encoder_fn attached to MultinomialHSMM.")

        if logits.shape[-1] != self.n_features:
            raise ValueError(
                f"Encoder output shape {logits.shape[-1]} != n_features ({self.n_features})"
            )
        return logits

    # -------------------------------------------------------------------------
    # --- MANAGEMENT
    # -------------------------------------------------------------------------
    def attach_encoder(self, encoder: nn.Module) -> None:
        """Attach a torch.nn.Module as neural emission encoder."""
        self.encoder = encoder

    def attach_encoder_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Attach a callable function that computes emission logits."""
        self.encoder_fn = fn

    def detach_encoder(self) -> None:
        """Detach any neural encoder or function."""
        self.encoder = None
        self.encoder_fn = None
