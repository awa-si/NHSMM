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

    @property
    def dof(self) -> int:
        K, F = self.n_states, self.n_features
        trans_dof = K * (K - 1)
        mean_dof = K * F
        cov_dof = K * F * (F + 1) // 2
        return trans_dof + mean_dof + cov_dof

    def sample_emission_pdf(
        self,
        X: torch.Tensor | None = None,
        posterior: torch.Tensor | None = None,
        theta: dict | None = None,
        theta_scale: float = 0.1,
    ) -> MultivariateNormal:

        self._emission_sample.to(self.device)
        if not self._params["emission_pdf"]:
            self._params["emission_pdf"] = self._emission_sample.initialize(
                X=X,
                posterior=posterior,
                theta=theta,
                theta_scale=theta_scale,
            )

        with torch.no_grad():
            self._emission_means.copy_(self._emission_sample._emission_means)
            self._emission_covs.copy_(self._emission_sample._emission_covs)

        return self._params["emission_pdf"]

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[dict] = None,
        theta_scale: float = 0.1
    ) -> MultivariateNormal:
        """
        Estimate Gaussian emission PDF for HSMM from weighted observations.

        Supports:
            - Optional context modulation (theta)
            - Robust handling of NaNs/Infs
            - Per-state regularization to prevent collapsed covariances
        """
        device = self._emission_means.device
        K = self.n_states
        T, F = X.shape

        X = X.to(dtype=DTYPE, device=device)
        posterior = posterior.to(dtype=DTYPE, device=device)

        # --- Weighted means ---
        weights_sum = posterior.sum(dim=0, keepdim=True).clamp_min(EPS)  # [1, K]
        means = (posterior.T @ X) / weights_sum.T  # [K, F]

        # --- Context modulation ---
        if theta:
            theta_tensor = torch.as_tensor(list(theta.values()), dtype=DTYPE, device=device)
            if theta_tensor.ndim > 1:
                theta_tensor = theta_tensor.mean(dim=0, keepdim=True)
            means += theta_scale * theta_tensor.expand(K, -1)

        # --- Handle NaNs/Infs and tiny jitter to prevent collapse ---
        means = torch.where(
            torch.isfinite(means),
            means,
            X.mean(dim=0).expand(K, -1)
        )
        means += 1e-6 * torch.randn_like(means)

        # --- Weighted covariances ---
        covs = torch.zeros(K, F, F, dtype=DTYPE, device=device)
        eye_F = torch.eye(F, dtype=DTYPE, device=device)

        for s in range(K):
            w = posterior[:, s].clamp_min(EPS).unsqueeze(-1)
            diff = X - means[s]
            denom = w.sum()
            if denom < EPS:
                covs[s] = eye_F * self.min_covar
                continue

            C = (w * diff).T @ diff / denom
            C = 0.5 * (C + C.T)  # symmetrize

            # Regularize covariances to avoid degenerate distributions
            trace = torch.trace(C)
            if not torch.isfinite(trace) or trace <= 0:
                C = eye_F * self.min_covar
            else:
                C += self.min_covar * (1.0 + trace / F) * eye_F

            covs[s] = C

        # Final symmetry & epsilon regularization
        covs = 0.5 * (covs + covs.transpose(-1, -2)) + self.min_covar * eye_F.unsqueeze(0)

        # --- Update internal buffers ---
        with torch.no_grad():
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        return MultivariateNormal(
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
        row_sums = A_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
        self.transition_logits.copy_((A_counts / row_sums).log())

        # --- Warm-start durations ---
        D_counts = torch.full((K, self.max_duration), smooth_duration, dtype=DTYPE, device=self.duration_logits.device)
        for s, l in segments:
            D_counts[s, min(l - 1, self.max_duration - 1)] += 1.0
        row_sums = D_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
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
