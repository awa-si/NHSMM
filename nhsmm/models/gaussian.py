# models/gaussian.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from sklearn.cluster import KMeans
from typing import Optional, Literal
import numpy as np

from nhsmm.defaults import DTYPE, EPS, HSMMError
from nhsmm.models.hsmm import HSMM


class GaussianHSMM(HSMM):
    """
    Gaussian Hidden Semi-Markov Model (HSMM) with multivariate normal emissions.

    Features:
        - Robust emission initialization (k-means or moments)
        - Data-driven warm-start for pi, A, D
        - Device/dtype safe, with regularization and small-sample guards
        - decode() wrapper for convenience
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        k_means: bool = False,
        min_covar: float = 1e-3,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else torch.device('cpu')
        self.min_covar = float(min_covar)
        self.n_features = n_features
        self.k_means = k_means


        super().__init__(
            n_states=n_states,
            n_features=n_features,
            max_duration=max_duration,
            min_covar=min_covar,
            alpha=alpha,
            seed=seed,
        )

        # Initialize emission buffers
        means_buf = torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device)
        covs_buf = torch.stack([torch.eye(n_features, dtype=DTYPE, device=self.device) for _ in range(n_states)])
        self.register_buffer("_emission_means", means_buf)
        self.register_buffer("_emission_covs", covs_buf)

        # Initialize base emission PDF
        self._params['emission_pdf'] = MultivariateNormal(
            loc=self._emission_means,
            covariance_matrix=self._emission_covs
        )

    @property
    def dof(self) -> int:
        K, F = self.n_states, self.n_features
        trans_dof = K * (K - 1)
        mean_dof = K * F
        cov_dof = K * F * (F + 1) // 2
        return trans_dof + mean_dof + cov_dof

    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[dict] = None,
        theta_scale: float = 0.1
    ) -> MultivariateNormal:
        """
        Self-contained emission PDF sampler for Gaussian HSMM.
        Supports:
            - Optional observations X
            - Optional posterior weights per state
            - Optional context/hyperparameter theta
            - Safe per-state means and covariance regularization
        """
        device = getattr(self, "_emission_means", torch.zeros(1)).device if hasattr(self, "_emission_means") else "cpu"
        K, F = self.n_states, self.n_features
        eps = 1e-12

        # --- Initialize buffers if missing ---
        if not hasattr(self, "_emission_means") or not hasattr(self, "_emission_covs"):
            means = torch.zeros(K, F, dtype=DTYPE, device=device)
            covs = torch.stack([torch.eye(F, dtype=DTYPE, device=device) for _ in range(K)])
            self.register_buffer("_emission_means", means)
            self.register_buffer("_emission_covs", covs)
            return torch.distributions.MultivariateNormal(means, covs)

        # --- Compute means ---
        if X is not None:
            X = X.to(dtype=DTYPE, device=device)

            if posterior is not None:
                weighted_sum = posterior.T @ X
                norm = posterior.sum(dim=0, keepdim=True).T.clamp_min(eps)
                means = weighted_sum / norm
                means = torch.where(
                    torch.isnan(means) | torch.isinf(means),
                    X.mean(dim=0).expand_as(means),
                    means
                )
            else:
                if getattr(self, "k_means", False):
                    X_np = X.detach().cpu().numpy()
                    n_samples = X_np.shape[0]
                    if n_samples < K:
                        pad = K - n_samples
                        X_np = np.concatenate([X_np, X_np[np.random.choice(n_samples, pad, replace=True)]], axis=0)
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=K, n_init=10, random_state=getattr(self, "seed", 0))
                    km.fit(X_np)
                    means = torch.from_numpy(km.cluster_centers_).to(dtype=DTYPE, device=device)
                    if torch.linalg.norm(means - means.mean(dim=0, keepdim=True)) < 1e-8:
                        means += 1e-3 * torch.randn_like(means)
                else:
                    means = X.mean(dim=0, keepdim=True).expand(K, -1)

            if theta is not None:
                theta_tensor = torch.as_tensor(list(theta.values()), dtype=DTYPE, device=device).mean(dim=0, keepdim=True)
                means = means + theta_scale * theta_tensor.expand(K, -1)
        else:
            means = self._emission_means.clone()

        # Add tiny perturbation to avoid exact duplicates
        means += 1e-6 * torch.randn_like(means)

        # --- Compute covariances ---
        if X is not None:
            if posterior is not None:
                covs = torch.zeros(K, F, F, dtype=DTYPE, device=device)
                eye_F = torch.eye(F, dtype=DTYPE, device=device)
                for s in range(K):
                    w = posterior[:, s].clamp_min(eps).unsqueeze(-1)
                    diff = X - means[s]
                    denom = w.sum()
                    if denom < eps:
                        covs[s] = eye_F * self.min_covar
                        continue
                    C = (w * diff).T @ diff / denom
                    C = 0.5 * (C + C.T)
                    trace = torch.trace(C)
                    if not torch.isfinite(trace) or trace <= 0:
                        C = eye_F * self.min_covar
                    else:
                        C += self.min_covar * (1.0 + trace / F) * eye_F
                    covs[s] = C
            else:
                centered = X - X.mean(dim=0, keepdim=True)
                cov_base = (centered.T @ centered) / max(X.shape[0] - 1, 1)
                cov_base = 0.5 * (cov_base + cov_base.T)
                covs = cov_base.unsqueeze(0).expand(K, -1, -1).clone()
        else:
            covs = self._emission_covs.clone()

        # --- Regularize covariances ---
        eps_eye = self.min_covar * torch.eye(F, dtype=DTYPE, device=device).unsqueeze(0)
        covs = 0.5 * (covs + covs.transpose(-1, -2)) + eps_eye

        # --- Update buffers ---
        with torch.no_grad():
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        return torch.distributions.MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)


    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[dict] = None,
        theta_scale: float = 0.1
    ) -> torch.distributions.MultivariateNormal:
        """
        Estimate Gaussian emission PDF for HSMM from weighted observations.

        Args:
            X: Observations [T, F]
            posterior: State responsibilities [T, K]
            theta: Optional context modulation
            theta_scale: Scaling for context adjustment

        Returns:
            MultivariateNormal distribution with updated means and covariances
        """
        device = self._emission_means.device
        T, F = X.shape
        K = self.n_states
        eps = 1e-12

        X = X.to(dtype=DTYPE, device=device)
        posterior = posterior.to(dtype=DTYPE, device=device)

        # --- Weighted means ---
        weights_sum = posterior.sum(dim=0, keepdim=True).clamp_min(eps)  # [1, K]
        means = (posterior.T @ X) / weights_sum.T  # [K, F]

        # --- Context modulation ---
        if theta is not None and len(theta) > 0:
            theta_tensor = torch.as_tensor(list(theta.values()), dtype=DTYPE, device=device)
            if theta_tensor.ndim > 1:
                theta_tensor = theta_tensor.mean(dim=0, keepdim=True)  # collapse time dimension
            means = means + theta_scale * theta_tensor.expand(K, -1)

        # --- Handle NaNs/Infs in means ---
        nan_mask = ~torch.isfinite(means)
        if nan_mask.any():
            fallback = X.mean(dim=0).expand(K, -1)
            means = torch.where(nan_mask, fallback, means)

        # --- Tiny jitter to prevent collapse ---
        means += 1e-6 * torch.randn_like(means)

        # --- Weighted covariances ---
        covs = torch.zeros(K, F, F, dtype=DTYPE, device=device)
        eye_F = torch.eye(F, dtype=DTYPE, device=device)

        for s in range(K):
            w = posterior[:, s].clamp_min(eps).unsqueeze(-1)
            diff = X - means[s]
            denom = w.sum()
            if denom < eps:
                covs[s] = eye_F * self.min_covar
                continue

            C = (w * diff).T @ diff / denom
            C = 0.5 * (C + C.T)  # symmetrize

            # Regularize covariances
            trace = torch.trace(C)
            if not torch.isfinite(trace) or trace <= 0:
                C = eye_F * self.min_covar
            else:
                C += self.min_covar * (1.0 + trace / F) * eye_F
            covs[s] = C

        covs = 0.5 * (covs + covs.transpose(-1, -2))  # ensure symmetry
        covs += self.min_covar * eye_F.unsqueeze(0)

        # --- Update internal buffers ---
        with torch.no_grad():
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        return torch.distributions.MultivariateNormal(
            loc=self._emission_means,
            covariance_matrix=self._emission_covs
        )


    def _contextual_emission_pdf(self, X: Optional[torch.Tensor] = None, theta: Optional[dict] = None) -> Optional[MultivariateNormal]:
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if isinstance(pdf, MultivariateNormal):
            return pdf
        return None

    def initialize_emissions(
        self,
        X: torch.Tensor,
        method: Literal["kmeans", "moment"] = "kmeans",
        smooth_transition: float = 1e-2,
        smooth_duration: float = 1e-2,
    ):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=DTYPE)
        X = X.to(dtype=DTYPE, device=self._emission_means.device)
        T, F = X.shape
        K = self.n_states
        eps = 1e-12

        # --- Initialize emission parameters ---
        if method == "moment":
            mu = X.mean(dim=0, keepdim=True).repeat(K, 1)
            var = X.var(dim=0, unbiased=False, keepdim=True).clamp_min(self.min_covar)
            covs = torch.diag_embed(var.expand(K, F))
        else:
            labels = torch.as_tensor(
                KMeans(n_clusters=K, n_init=10, random_state=getattr(self, "seed", 0))
                .fit_predict(X.cpu().numpy()),
                device=X.device,
            )
            mu = torch.zeros(K, F, dtype=DTYPE, device=X.device)
            covs = torch.zeros(K, F, F, dtype=DTYPE, device=X.device)
            for k in range(K):
                mask = labels == k
                Nk = mask.sum().item()
                if Nk == 0:
                    # Fallback: pick a random point from X
                    Xk = X[torch.randint(0, T, (1,))]
                else:
                    Xk = X[mask]
                mu[k] = Xk.mean(dim=0)
                diff = Xk - mu[k]
                C = (diff.T @ diff) / max(Nk - 1, 1)
                # Symmetrize and regularize
                covs[k] = 0.5 * (C + C.T) + self.min_covar * torch.eye(F, device=X.device)
                # Optional shrinkage for tiny clusters
                covs[k] = 0.05 * torch.eye(F, device=X.device) + 0.95 * covs[k]

        self._emission_means.copy_(mu)
        self._emission_covs.copy_(covs)
        self._params["emission_pdf"] = MultivariateNormal(mu.clone(), covs.clone())

        # --- Warm-start initial state probabilities (pi) ---
        pi_counts = torch.bincount(labels[:min(5, T)], minlength=K).to(dtype=DTYPE)
        pi = (pi_counts + 1e-6) / (pi_counts.sum() + 1e-6 * K)
        self.init_logits.copy_(pi.log().to(self.init_logits.device))

        # --- Warm-start transitions ---
        lbls = labels.cpu().numpy()
        segments = []
        if T > 0:
            cur, length = lbls[0], 1
            for v in lbls[1:]:
                if v == cur:
                    length += 1
                else:
                    segments.append((cur, length))
                    cur, length = v, 1
            segments.append((cur, length))

        A_counts = torch.full((K, K), smooth_transition, dtype=DTYPE, device=self.transition_logits.device)
        prev = None
        for s, _ in segments:
            if prev is not None:
                A_counts[prev, s] += 1.0
            prev = s
        row_sums = A_counts.sum(dim=1, keepdim=True).clamp_min(eps)
        self.transition_logits.copy_((A_counts / row_sums).log())

        # --- Warm-start durations ---
        D_counts = torch.full((K, self.max_duration), smooth_duration, dtype=DTYPE, device=self.duration_logits.device)
        for s, l in segments:
            D_counts[s, min(l - 1, self.max_duration - 1)] += 1.0
        row_sums = D_counts.sum(dim=1, keepdim=True).clamp_min(eps)
        self.duration_logits.copy_((D_counts / row_sums).log())

    def decode(self, X: torch.Tensor, algorithm: Literal["viterbi", "map"] = "viterbi") -> np.ndarray:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=DTYPE)
        preds = self.predict(X, algorithm=algorithm, context=None)
        if isinstance(preds, list):
            preds = preds[0] if preds else torch.empty(0, dtype=torch.long)
        elif not torch.is_tensor(preds):
            preds = torch.as_tensor(preds, dtype=torch.long)
        return preds.cpu().numpy()
