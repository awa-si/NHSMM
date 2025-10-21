# models/gaussian.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from sklearn.cluster import KMeans
from typing import Optional, Literal
import numpy as np

from nhsmm.defaults import DTYPE
from nhsmm.models.hsmm import HSMM


class GaussianHSMM(HSMM, nn.Module):
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
    ):
        self.n_features = n_features
        self.min_covar = float(min_covar)
        self.k_means = k_means
        super().__init__(
            n_states=n_states,
            n_features=n_features,
            max_duration=max_duration,
            min_covar=min_covar,
            alpha=alpha,
            seed=seed,
        )

        # Register emission buffers
        means_buf = torch.zeros(n_states, n_features, dtype=DTYPE)
        covs_buf = torch.stack([torch.eye(n_features, dtype=DTYPE) for _ in range(n_states)])
        self.register_buffer("_emission_means", means_buf)
        self.register_buffer("_emission_covs", covs_buf)

        # Ensure base emission PDF
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    @property
    def dof(self) -> int:
        K, F = self.n_states, self.n_features
        trans_dof = K * K - 1
        mean_dof = K * F
        cov_dof = K * F * (F + 1) // 2
        return int(trans_dof + mean_dof + cov_dof)

    # -------------------------
    # --- EMISSION PDF METHODS
    # -------------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None, theta: Optional[dict] = None) -> MultivariateNormal:
        dev = 'cpu'
        if hasattr(self, "_emission_means"):
            dev = self._emission_means.device
        F, K = self.n_features, self.n_states

        # initialize buffers if missing
        if not hasattr(self, "_emission_means") or not hasattr(self, "_emission_covs"):
            means = torch.zeros(K, F, dtype=DTYPE, device=dev)
            covs = torch.stack([torch.eye(F, dtype=DTYPE, device=dev) for _ in range(K)])
            self.register_buffer("_emission_means", means)
            self.register_buffer("_emission_covs", covs)
            return MultivariateNormal(means, covs)

        # initialize from X
        if X is not None:
            X = X.to(dtype=DTYPE, device=dev)
            means = self._sample_kmeans(X) if self.k_means else X.mean(dim=0, keepdim=True).expand(K, -1)

            if theta is not None:
                # optional context/hyperparameter modulation
                theta_tensor = torch.as_tensor(list(theta.values()), dtype=DTYPE, device=dev).mean(dim=0, keepdim=True)
                means = means + 0.1 * theta_tensor.expand(K, -1)

            centered = X - X.mean(dim=0, keepdim=True)
            denom = max(X.shape[0] - 1, 1)
            covs = (centered.T @ centered) / denom
            covs = 0.5 * (covs + covs.T)
            covs = covs.unsqueeze(0).expand(K, -1, -1).clone()
        else:
            means = self._emission_means.clone()
            covs = self._emission_covs.clone()

        eps_eye = self.min_covar * torch.eye(F, dtype=DTYPE, device=dev).unsqueeze(0)
        covs = 0.5 * (covs + covs.transpose(-1, -2)) + eps_eye

        self._emission_means.copy_(means)
        self._emission_covs.copy_(covs)

        return MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)

    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta=None) -> MultivariateNormal:
        X = X.to(dtype=DTYPE, device=self._emission_means.device)
        posterior = posterior.to(dtype=DTYPE, device=self._emission_means.device)

        means = self._compute_means(X, posterior)
        covs = self._compute_covs(X, posterior, means)

        covs = 0.5 * (covs + covs.transpose(-1, -2))
        covs += self.min_covar * torch.eye(self.n_features, dtype=DTYPE, device=self._emission_covs.device).unsqueeze(0)

        with torch.no_grad():
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        return MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)

    # -------------------------
    # --- HELPER METHODS
    # -------------------------
    def _compute_means(self, X: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        weighted_sum = posterior.T @ X
        norm = posterior.sum(dim=0, keepdim=True).T.clamp_min(1e-12)
        means = weighted_sum / norm

        global_mean = X.mean(dim=0, keepdim=True)
        mask = torch.isnan(means) | torch.isinf(means)
        if mask.any():
            means = torch.where(mask, global_mean.expand_as(means), means)
        return means

    def _compute_covs(self, X: torch.Tensor, posterior: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        K, F = self.n_states, self.n_features
        covs = torch.zeros(K, F, F, dtype=DTYPE, device=self._emission_covs.device)
        eye_F = torch.eye(F, dtype=DTYPE, device=self._emission_covs.device)

        for s in range(K):
            w = posterior[:, s].clamp_min(1e-12).unsqueeze(-1)
            diff = X - means[s]
            denom = w.sum()

            if denom < 1e-8:
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

        return covs

    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if X.numel() == 0:
            raise ValueError("Cannot run KMeans on empty input tensor.")

        X_np = X.detach().cpu().numpy()
        K = self.n_states
        rng_seed = int(seed or getattr(self, "seed", None) or 0)

        n_samples = X_np.shape[0]
        if n_samples < K:
            pad = K - n_samples
            sampled = np.concatenate([X_np, X_np[np.random.choice(n_samples, pad, replace=True)]], axis=0)
            X_np = sampled

        try:
            km = KMeans(n_clusters=K, n_init=10, random_state=rng_seed)
            km.fit(X_np)
            centers = torch.from_numpy(km.cluster_centers_).to(dtype=DTYPE, device=self._emission_means.device)
        except Exception:
            centers = torch.randn(K, X.shape[1], dtype=DTYPE, device=self._emission_means.device) * 0.01 + X.mean(dim=0)
        
        # tiny jitter if degenerate
        diffs = centers - centers.mean(dim=0, keepdim=True)
        if torch.linalg.norm(diffs) < 1e-8:
            centers += 1e-3 * torch.randn_like(centers)
        return centers

    def _contextual_emission_pdf(self, X: Optional[torch.Tensor] = None, theta: Optional[dict] = None) -> Optional[MultivariateNormal]:
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if isinstance(pdf, MultivariateNormal):
            return pdf
        return None

    # -------------------------
    # --- INITIALIZATION
    # -------------------------
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

        if method == "moment":
            mu = X.mean(dim=0, keepdim=True).repeat(K, 1)
            var = X.var(dim=0, unbiased=False, keepdim=True).clamp_min(self.min_covar)
            covs = torch.diag_embed(var.expand(K, F))
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(covs)
            self._params["emission_pdf"] = MultivariateNormal(mu, covs)
            return

        km_seed = int(getattr(self, "seed", 0))
        km = KMeans(n_clusters=K, n_init=10, random_state=km_seed)
        labels = torch.as_tensor(km.fit_predict(X.cpu().numpy()), device=X.device)
        mus = torch.zeros(K, F, dtype=DTYPE, device=self._emission_means.device)
        covs = torch.zeros(K, F, F, dtype=DTYPE, device=self._emission_covs.device)

        for k in range(K):
            mask = labels == k
            Nk = mask.sum().item()
            if Nk == 0:
                mus[k] = X.mean(dim=0)
                covs[k] = torch.eye(F, dtype=DTYPE, device=X.device) * self.min_covar
                continue
            Xk = X[mask]
            mus[k] = Xk.mean(dim=0)
            diff = Xk - mus[k]
            C = (diff.T @ diff) / max(Nk - 1, 1)
            C = 0.5 * (C + C.T) + self.min_covar * torch.eye(F, dtype=DTYPE, device=X.device)
            covs[k] = C

        self._emission_means.copy_(mus)
        self._emission_covs.copy_(covs)
        self._params["emission_pdf"] = MultivariateNormal(self._emission_means, self._emission_covs)

        # Warm-start pi, A, D
        pi_counts = torch.bincount(labels[:1], minlength=K).to(dtype=DTYPE)
        pi = (pi_counts + 1e-6) / (pi_counts.sum() + 1e-6 * K)
        self._pi_logits.copy_(pi.log().to(self._pi_logits.device))

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

        A_counts = torch.full((K, K), smooth_transition, dtype=DTYPE, device=self._A_logits.device)
        prev = None
        for s, _ in segments:
            if prev is not None:
                A_counts[prev, s] += 1.0
            prev = s
        self._A_logits.copy_((A_counts / A_counts.sum(dim=1, keepdim=True)).log().to(self._A_logits.device))

        D_counts = torch.full((K, self.max_duration), smooth_duration, dtype=DTYPE, device=self._D_logits.device)
        for s, l in segments:
            D_counts[s, min(l, self.max_duration) - 1] += 1.0
        self._D_logits.copy_((D_counts / D_counts.sum(dim=1, keepdim=True)).log().to(self._D_logits.device))

    # -------------------------
    # --- DECODING
    # -------------------------
    def decode(
        self,
        X: torch.Tensor,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        duration_weight: float = 0.0
    ) -> np.ndarray:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=DTYPE)
        preds = self.predict(X, algorithm=algorithm, context=None)

        if isinstance(preds, list):
            preds = preds[0] if len(preds) > 0 else torch.empty(0, dtype=torch.long)
        elif not torch.is_tensor(preds):
            preds = torch.as_tensor(preds, dtype=torch.long)
        return preds.cpu().numpy()
