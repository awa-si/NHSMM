# nhsmm/hsmm/GaussianHSMM.py
from typing import Optional, Callable
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from sklearn.cluster import KMeans

from nhsmm.hsmm.BaseHSMM import BaseHSMM
from nhsmm.utilities import utils, constraints


class GaussianHSMM(BaseHSMM):
    """
    Neural-Compatible Gaussian Hidden Semi-Markov Model (HSMM)
    ----------------------------------------------------------
    Multivariate Normal emissions with optional neural/contextual adaptation.
    """

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 max_duration: int,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 min_covar: float = 1e-3,
                 covariance_type: constraints.CovarianceType = constraints.CovarianceType.FULL,
                 seed: Optional[int] = None,
                 encoder: Optional[nn.Module] = None,
                 encoder_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):

        self.n_features = n_features
        self.min_covar = min_covar
        self.k_means = k_means
        self.covariance_type = covariance_type
        self.encoder = encoder
        self.encoder_fn = encoder_fn

        super().__init__(n_states, max_duration, alpha, seed)

        # Register emission parameters for checkpointing
        self.register_buffer("_emission_means", self.pdf.loc.detach().clone())
        self.register_buffer("_emission_covs", self.pdf.covariance_matrix.detach().clone())

    @property
    def dof(self) -> int:
        return (
            self.n_states**2 - 1 +
            self.pdf.loc.numel() +
            self.pdf.covariance_matrix.numel()
        )

    # -------------------------------------------------------------------------
    # --- EMISSION INITIALIZATION
    # -------------------------------------------------------------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> MultivariateNormal:
        if X is not None:
            means = self._sample_kmeans(X) if self.k_means else X.mean(dim=0).expand(self.n_states, -1).clone()
            centered_data = X - X.mean(dim=0)
            covs = (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1)).expand(self.n_states, -1, -1).clone()
        else:
            means = torch.zeros((self.n_states, self.n_features), dtype=torch.float64)
            covs = self.min_covar * torch.eye(self.n_features, dtype=torch.float64).expand(
                self.n_states, self.n_features, self.n_features
            ).clone()

        return MultivariateNormal(means, covs)

    # -------------------------------------------------------------------------
    # --- EMISSION UPDATE
    # -------------------------------------------------------------------------
    def _estimate_emission_pdf(self,
                               X: torch.Tensor,
                               posterior: torch.Tensor,
                               theta: Optional[utils.ContextualVariables] = None) -> MultivariateNormal:
        if self.encoder or self.encoder_fn:
            means, covs = self._neural_emission(X, posterior)
        else:
            means = self._compute_means(X, posterior, theta)
            covs = self._compute_covs(X, posterior, means, theta)

        self._emission_means.copy_(means)
        self._emission_covs.copy_(covs)

        return MultivariateNormal(means, covs)

    # -------------------------------------------------------------------------
    # --- NEURAL EMISSION
    # -------------------------------------------------------------------------
    def _neural_emission(self, X: torch.Tensor, posterior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through neural encoder or encoder_fn to compute mean/cov per state.
        """
        if self.encoder_fn is not None:
            out = self.encoder_fn(X)
        elif self.encoder is not None:
            self.encoder.eval()
            with torch.no_grad():
                out = self.encoder(X)
        else:
            raise RuntimeError("No encoder attached to GaussianHSMM.")

        if out.ndim != 2 or out.shape[1] != self.n_features:
            raise ValueError(f"Encoder output shape {out.shape} must be (N, n_features={self.n_features})")

        # weighted means by posterior
        weighted_means = posterior.T @ out
        weighted_means /= posterior.T.sum(-1, keepdim=True).clamp_min(1e-10)

        # weighted covariances
        diff = out.unsqueeze(0) - weighted_means.unsqueeze(1)
        posterior_adj = posterior.T.unsqueeze(-1)
        covs = torch.matmul((posterior_adj * diff).transpose(-1, -2), diff)
        covs /= posterior_adj.sum(dim=1, keepdim=True).clamp_min(1e-10)
        covs += self.min_covar * torch.eye(self.n_features, dtype=torch.float64)

        return weighted_means, covs

    # -------------------------------------------------------------------------
    # --- ORIGINAL COMPUTATION HELPERS
    # -------------------------------------------------------------------------
    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        km = KMeans(n_clusters=self.n_states, random_state=seed or self.seed, n_init="auto").fit(X.cpu().numpy())
        return torch.from_numpy(km.cluster_centers_).to(torch.float64)

    def _compute_means(self, X: torch.Tensor, posterior: torch.Tensor, theta=None) -> torch.Tensor:
        weighted_sum = posterior.T @ X
        norm = posterior.T.sum(-1, keepdim=True).clamp_min(1e-10)
        return weighted_sum / norm

    def _compute_covs(self, X: torch.Tensor, posterior: torch.Tensor, means: torch.Tensor, theta=None) -> torch.Tensor:
        posterior_adj = posterior.T.unsqueeze(-1)
        diff = X.expand(self.n_states, -1, -1) - means.unsqueeze(1)
        covs = torch.matmul((posterior_adj * diff).transpose(-1, -2), diff)
        covs /= posterior_adj.sum(dim=1, keepdim=True).clamp_min(1e-10)
        covs += self.min_covar * torch.eye(self.n_features, dtype=torch.float64)
        return covs

    # -------------------------------------------------------------------------
    # --- PERSISTENCE
    # -------------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'means': self._emission_means,
            'covs': self._emission_covs,
            'config': {
                'n_states': self.n_states,
                'n_features': self.n_features,
                'max_duration': self.max_duration,
                'alpha': self.alpha,
                'min_covar': self.min_covar,
                'covariance_type': self.covariance_type,
                'k_means': self.k_means,
                'seed': self.seed,
            }
        }, path)

    @classmethod
    def load(cls, path: str) -> "GaussianHSMM":
        data = torch.load(path, map_location='cpu')
        model = cls(**data['config'])
        model.load_state_dict(data['state_dict'])
        model._emission_means.copy_(data['means'])
        model._emission_covs.copy_(data['covs'])
        model._params['emission_pdf'] = MultivariateNormal(model._emission_means, model._emission_covs)
        return model
