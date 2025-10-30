# nhsmm/emissions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Laplace, StudentT, Independent

from sklearn.cluster import KMeans
from typing import Optional, Dict

from nhsmm.defaults import DTYPE, EPS, logger


class EmissionSample(nn.Module):
    """
    Flexible emission model for HSMM supporting multiple distribution types:
    "gaussian", "categorical", "bernoulli", "poisson", "laplace", "studentt".
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        emission_type: str = "gaussian",
        k_means: bool = True,
        min_covar: float = 1e-6,
        init_spread: float = 1.0,
        device: Optional[torch.device] = None,
        seed: int = 0,
        student_df: float = 5.0
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_features = n_features
        self.emission_type = emission_type.lower()
        self.k_means = k_means
        self.min_covar = float(min_covar)
        self.init_spread = float(init_spread)
        self.seed = seed
        self.dof = student_df

        # Continuous distributions: mean & covariance/scale
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        eye = torch.eye(n_features, dtype=DTYPE, device=self.device)
        self.register_buffer("_emission_covs", eye.unsqueeze(0).repeat(n_states, 1, 1))

        # Discrete distributions: logits or raw params
        self.register_buffer("_emission_params", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))

    @torch.no_grad()
    def _spread_means(self, means: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        jitter = scale * torch.randn_like(means)
        for _ in range(3):
            dists = torch.cdist(means + jitter, means + jitter)
            if torch.all(dists + torch.eye(dists.size(0), device=dists.device) > 1e-3):
                break
            jitter += scale * 0.1 * torch.randn_like(jitter)
        return means + jitter

    @torch.no_grad()
    def initialize(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        emission_type: Optional[str] = None,
    ):
        """
        Initialize emission distributions with optional posterior weighting,
        k-means, or default global statistics.
        """
        emission_type = (emission_type or self.emission_type).lower()
        K, F = self.n_states, self.n_features
        device = self.device

        if X is not None:
            X = X.to(dtype=DTYPE, device=device)
            if X.std() < 1e-8:
                X = X + 1e-3 * torch.randn_like(X)

        # --- Continuous: Gaussian ---
        if emission_type == "gaussian":
            if X is not None:
                if posterior is not None:
                    norm = posterior.sum(0, keepdim=True).T.clamp_min(EPS)
                    means = (posterior.T @ X) / norm
                    covs = torch.stack([
                        ((posterior[:, k:k+1] * (X - means[k]).unsqueeze(0)).T @ (X - means[k]).unsqueeze(0)) / norm[k]
                        + self.min_covar * torch.eye(F, device=device, dtype=DTYPE)
                        for k in range(K)
                    ])
                elif self.k_means:
                    km = KMeans(n_clusters=K, n_init=10, random_state=self.seed)
                    km.fit(X.cpu().numpy())
                    labels = torch.tensor(km.labels_, dtype=torch.long, device=device)
                    means = torch.tensor(km.cluster_centers_, dtype=DTYPE, device=device)
                    covs = torch.zeros(K, F, F, dtype=DTYPE, device=device)
                    for k in range(K):
                        cluster_points = X[labels == k]
                        if cluster_points.shape[0] <= 1:
                            covs[k] = torch.eye(F, device=device, dtype=DTYPE) * self.min_covar
                        else:
                            centered = cluster_points - cluster_points.mean(0, keepdim=True)
                            cov = (centered.T @ centered) / (cluster_points.shape[0] - 1)
                            cov += torch.eye(F, device=device, dtype=DTYPE) * self.min_covar
                            covs[k] = cov
                else:
                    means = X.mean(0, keepdim=True).expand(K, -1)
                    covs = torch.stack([((X - X.mean(0))**2).mean(0).diag() + self.min_covar for _ in range(K)])
            else:
                means = self._emission_means.clone()
                covs = self._emission_covs.clone()

            if theta is not None:
                means += theta_scale * theta.unsqueeze(0).expand(K, -1)
            means = self._spread_means(means, scale=self.init_spread)
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)
            return MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)

        # --- Discrete: categorical / bernoulli / poisson ---
        elif emission_type in {"categorical", "bernoulli", "poisson"}:
            if X is not None:
                if emission_type == "categorical":
                    counts = torch.stack([torch.bincount(X[:, f].long(), minlength=F) for f in range(F)], dim=1).T.float()
                    probs = counts / counts.sum(-1, keepdim=True)
                else:
                    probs = X.float().mean(0, keepdim=True).expand(K, -1)
            else:
                probs = torch.full((K, F), 1/F, dtype=DTYPE, device=device)
            self._emission_params.copy_(probs)
            return probs

        # --- Continuous: Laplace / StudentT ---
        elif emission_type in {"laplace", "studentt"}:
            if X is not None:
                if posterior is not None:
                    norm = posterior.sum(0, keepdim=True).T.clamp_min(EPS)
                    means = (posterior.T @ X) / norm
                    scales = torch.stack([
                        ((posterior[:, k:k+1] * (X - means[k]).abs()).sum(0) / norm[k])
                        for k in range(K)
                    ])
                elif self.k_means:
                    km = KMeans(n_clusters=K, n_init=10, random_state=self.seed)
                    km.fit(X.cpu().numpy())
                    labels = torch.tensor(km.labels_, dtype=torch.long, device=device)
                    means = torch.tensor(km.cluster_centers_, dtype=DTYPE, device=device)
                    scales = torch.zeros(K, F, dtype=DTYPE, device=device)
                    for k in range(K):
                        cluster_points = X[labels == k]
                        if cluster_points.shape[0] == 0:
                            scales[k] = torch.ones(F, device=device) * EPS
                        else:
                            scales[k] = cluster_points.std(0, unbiased=True).clamp_min(EPS)
                else:
                    means = X.mean(0, keepdim=True).expand(K, -1)
                    scales = X.std(0, keepdim=True).expand(K, -1).clamp_min(EPS)
            else:
                means = self._emission_means.clone()
                scales = torch.ones_like(means)

            self._emission_means.copy_(means)
            self._emission_covs.copy_(torch.diag_embed(scales**2))

            if emission_type == "laplace":
                return Independent(Laplace(self._emission_means, scales), 1)
            else:  # studentt
                return Independent(StudentT(df=self.dof, loc=self._emission_means, scale=scales), 1)

        else:
            raise ValueError(f"Unsupported emission type: {emission_type}")
