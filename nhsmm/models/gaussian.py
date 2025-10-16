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
    Pure Gaussian Hidden Semi-Markov Model (HSMM) with multivariate normal emissions.
    Enhancements:
      - initialize_emissions(X, method='kmeans'|'moment') to initialize means/covs and pi/A/D
      - data-driven warm-start for duration and transition logits
      - numeric/device/dtype safety and small-sample guards
      - convenience decode() wrapper using _viterbi or predict
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

        # Call base HSMM init
        super().__init__(
            n_states=n_states,
            n_features=n_features,
            max_duration=max_duration,
            min_covar=min_covar,
            alpha=alpha,
            seed=seed,
        )

        # Register emission buffers (mean and covariance per state)
        means_buf = torch.zeros(n_states, n_features, dtype=DTYPE)
        covs_buf = torch.stack([torch.eye(n_features, dtype=DTYPE) for _ in range(n_states)])
        self.register_buffer("_emission_means", means_buf)
        self.register_buffer("_emission_covs", covs_buf)

        # Ensure base emission pdf points to a valid object
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    @property
    def dof(self) -> int:
        """
        Total degrees of freedom for the model:
          - Transition matrix (A): n_states² - 1 (stochastic constraints)
          - Emission means: n_states × n_features
          - Emission covariances: depends on covariance structure (full/diag)
        """
        K, F = self.n_states, self.n_features

        # Transition degrees (subtract one due to normalization)
        trans_dof = K * K - 1

        # Emission means
        mean_dof = K * F

        # Emission covariances (assuming full covariance)
        cov_dof = K * F * (F + 1) // 2  # symmetric matrices

        return int(trans_dof + mean_dof + cov_dof)

    # -------------------------
    # --- EMISSION INITIALIZATION
    # -------------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> MultivariateNormal:
        """
        Construct or re-sample the emission distribution.

        If X is provided:
          - Initialize means via k-means (if enabled) or global mean.
          - Initialize covariances from sample covariance of X.
        If X is None:
          - Reuse current emission buffers (_emission_means / _emission_covs).
        """
        dev = getattr(self, "_emission_means", torch.tensor(0.)).device if hasattr(self, "_emission_means") else "cpu"
        F = self.n_features
        K = self.n_states

        # Fallback if buffers are not yet registered
        if not hasattr(self, "_emission_means") or not hasattr(self, "_emission_covs"):
            means = torch.zeros(K, F, dtype=DTYPE, device=dev)
            covs = torch.eye(F, dtype=DTYPE, device=dev).unsqueeze(0).repeat(K, 1, 1)
            return MultivariateNormal(means, covs)

        if X is not None:
            X = X.to(dtype=DTYPE, device=dev)
            means = (
                self._sample_kmeans(X)
                if self.k_means
                else X.mean(dim=0, keepdim=True).expand(K, -1)
            )

            # unbiased covariance estimate
            centered = X - X.mean(dim=0, keepdim=True)
            denom = max(X.shape[0] - 1, 1)
            base_cov = (centered.T @ centered) / denom
            base_cov = 0.5 * (base_cov + base_cov.T)  # ensure symmetry

            covs = base_cov.unsqueeze(0).expand(K, -1, -1).clone()
        else:
            means = self._emission_means.clone().to(dev)
            covs = self._emission_covs.clone().to(dev)

        # Regularize and enforce symmetry
        eps_eye = self.min_covar * torch.eye(F, dtype=DTYPE, device=dev).unsqueeze(0)
        covs = 0.5 * (covs + covs.transpose(-1, -2)) + eps_eye

        # Copy back to buffers to keep internal state consistent
        self._emission_means.copy_(means)
        self._emission_covs.copy_(covs)

        return MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)

    # -------------------------
    # --- EMISSION UPDATE
    # -------------------------
    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta=None) -> MultivariateNormal:
        """
        Update Gaussian emission parameters given data and posterior responsibilities.

        Args:
            X: (N, F) observation tensor
            posterior: (N, K) state posterior probabilities
            theta: optional context (unused here)
        Returns:
            Updated MultivariateNormal distribution over states
        """
        assert X.ndim == 2 and posterior.ndim == 2, "X and posterior must be 2D (N,F) and (N,K)"
        assert posterior.shape[0] == X.shape[0], "Mismatched sample count between X and posterior"
        assert posterior.shape[1] == self.n_states, "Posterior K must match n_states"

        dev = self._emission_means.device
        X = X.to(dtype=DTYPE, device=dev)
        posterior = posterior.to(dtype=DTYPE, device=dev)

        # Recompute weighted means and covariances
        means = self._compute_means(X, posterior)
        covs = self._compute_covs(X, posterior, means)

        # Symmetrize and regularize covariance matrices
        covs = 0.5 * (covs + covs.transpose(-1, -2))
        covs += self.min_covar * torch.eye(self.n_features, dtype=DTYPE, device=dev).unsqueeze(0)

        # Update internal buffers safely (no autograd tracking)
        with torch.no_grad():
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        # Return fresh MultivariateNormal tied to buffers
        return MultivariateNormal(
            loc=self._emission_means,
            covariance_matrix=self._emission_covs
        )

    # -------------------------
    # --- HELPERS
    # -------------------------
    def _compute_means(self, X: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        """
        Compute per-state weighted means.

        Args:
            X: (N, F) observation matrix
            posterior: (N, K) posterior responsibilities
        Returns:
            means: (K, F) tensor of state means
        """
        assert X.ndim == 2 and posterior.ndim == 2, "Expected X:(N,F), posterior:(N,K)"
        assert posterior.shape[0] == X.shape[0], "Mismatched sample count"
        assert posterior.shape[1] == self.n_states, f"posterior K={posterior.shape[1]} != n_states={self.n_states}"

        dev = self._emission_means.device
        X = X.to(dtype=DTYPE, device=dev)
        posterior = posterior.to(dtype=DTYPE, device=dev)

        # Weighted mean per state
        weighted_sum = posterior.T @ X  # (K,F)
        norm = posterior.sum(dim=0, keepdim=True).T.clamp_min(1e-12)  # (K,1)
        means = weighted_sum / norm

        # Replace potential NaNs (degenerate states) with global mean
        global_mean = X.mean(dim=0, keepdim=True)
        mask = torch.isnan(means) | torch.isinf(means)
        if mask.any():
            means = torch.where(mask, global_mean.expand_as(means), means)

        return means

    def _compute_covs(self, X: torch.Tensor, posterior: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """
        Compute per-state covariance matrices under posterior responsibilities.

        Args:
            X: (N, F) observations
            posterior: (N, K) state responsibilities
            means: (K, F) state means

        Returns:
            covs: (K, F, F) covariance matrices
        """
        N, F = X.shape
        dev = self._emission_covs.device
        X = X.to(dtype=DTYPE, device=dev)
        posterior = posterior.to(dtype=DTYPE, device=dev)
        means = means.to(dtype=DTYPE, device=dev)

        covs = torch.zeros(self.n_states, F, F, dtype=DTYPE, device=dev)
        eye_F = torch.eye(F, dtype=DTYPE, device=dev)

        for s in range(self.n_states):
            w = posterior[:, s].clamp_min(1e-12).unsqueeze(-1)  # (N,1)
            diff = X - means[s]                                 # (N,F)
            denom = w.sum()

            if denom < 1e-8:  # Empty or near-empty state
                covs[s] = eye_F * self.min_covar
                continue

            # Weighted covariance
            C = (w * diff).T @ diff / denom
            C = 0.5 * (C + C.T)  # enforce symmetry

            # Regularize to ensure positive-definiteness
            trace = torch.trace(C)
            if not torch.isfinite(trace) or trace <= 0:
                C = eye_F * self.min_covar
            else:
                reg = self.min_covar * (1.0 + trace / F)
                C += reg * eye_F

            covs[s] = C

        return covs

    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Run KMeans clustering on CPU to initialize emission means.
        Handles degenerate data gracefully and ensures numeric stability.

        Args:
            X: (N, F) observation tensor
            seed: optional random seed override

        Returns:
            centers: (K, F) tensor of cluster centers
        """
        if X.numel() == 0:
            raise ValueError("Cannot run KMeans on empty input tensor.")

        X_np = X.detach().cpu().numpy()
        K = self.n_states
        rng_seed = int(seed or getattr(self, "seed", None) or 0)

        # Guard against fewer samples than clusters
        n_samples = X_np.shape[0]
        if n_samples < K:
            # fall back to random subset + padding
            pad = K - n_samples
            sampled = np.concatenate([X_np, X_np[np.random.choice(n_samples, pad, replace=True)]], axis=0)
            X_np = sampled

        km = KMeans(
            n_clusters=K,
            n_init=10,
            random_state=rng_seed,
            algorithm="lloyd",
            max_iter=300,
        )

        try:
            km.fit(X_np)
            centers = torch.from_numpy(km.cluster_centers_).to(dtype=DTYPE)
        except Exception as e:
            # fallback to random Gaussian means if clustering fails
            centers = torch.randn(K, X.shape[1], dtype=DTYPE) * 0.01 + X.mean(dim=0)
            print(f"[GaussianHSMM._sample_kmeans] Warning: KMeans failed ({e}). Using random init.")

        # Add tiny jitter if duplicate centers exist
        diffs = centers - centers.mean(dim=0, keepdim=True)
        if torch.linalg.norm(diffs) < 1e-8:
            centers += 1e-3 * torch.randn_like(centers)

        return centers.to(device=self._emission_means.device, dtype=DTYPE)

    def _contextual_emission_pdf(self, X: Optional[torch.Tensor] = None, theta: Optional[dict] = None) -> Optional[MultivariateNormal]:
        """
        Return the current emission PDF, optionally modulated by context (theta).
        For standard Gaussian HSMMs, no contextual modulation is applied.

        Args:
            X: optional input tensor (unused here)
            theta: optional context or hyperparameter dict

        Returns:
            MultivariateNormal instance if available, otherwise None.
        """
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if isinstance(pdf, MultivariateNormal):
            return pdf
        if pdf is not None:
            # Defensive: handle stale or malformed entries
            print("[GaussianHSMM._contextual_emission_pdf] Warning: emission_pdf not a MultivariateNormal.")
        return None

    # -------------------------
    # --- Initialization helper (new)
    # -------------------------
    def initialize_emissions(
        self,
        X: torch.Tensor,
        method: Literal["kmeans", "moment"] = "kmeans",
        smooth_transition: float = 1e-2,
        smooth_duration: float = 1e-2,
    ):
        """
        Initialize emission parameters and warm-start π, A, D using simple clustering or moments.

        Args:
            X: (T, F) tensor of observations (single concatenated sequence)
            method: 'kmeans' for cluster-based init, 'moment' for global mean/var
            smooth_transition: Laplace smoothing for transition counts
            smooth_duration: Laplace smoothing for duration counts
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=DTYPE)
        X = X.to(dtype=DTYPE, device=self._emission_means.device)
        T, F = X.shape
        K = self.n_states

        # --- moment init ---
        if method == "moment":
            mu = X.mean(dim=0, keepdim=True).repeat(K, 1)
            var = X.var(dim=0, unbiased=False, keepdim=True).clamp_min(self.min_covar)
            covs = torch.diag_embed(var.expand(K, F))
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(covs)
            self._params["emission_pdf"] = MultivariateNormal(mu, covs)
            return

        # --- kmeans init ---
        km_seed = int(getattr(self, "seed", 0))
        km = KMeans(n_clusters=K, n_init=10, random_state=km_seed)
        labels = torch.as_tensor(km.fit_predict(X.cpu().numpy()), device=X.device)
        mus = torch.zeros(K, F, dtype=DTYPE, device=self._emission_means.device)
        covs = torch.zeros(K, F, F, dtype=DTYPE, device=self._emission_covs.device)

        for k in range(K):
            mask = labels == k
            Nk = mask.sum().item()
            if Nk == 0:
                # empty cluster fallback
                mus[k] = X.mean(dim=0)
                covs[k] = torch.eye(F, dtype=DTYPE, device=X.device) * self.min_covar
                continue
            Xk = X[mask]
            mus[k] = Xk.mean(dim=0)
            diff = Xk - mus[k]
            denom = max(Nk - 1, 1)
            C = (diff.T @ diff) / denom
            C = 0.5 * (C + C.T)
            C += self.min_covar * torch.eye(F, dtype=DTYPE, device=X.device)
            covs[k] = C

        self._emission_means.copy_(mus)
        self._emission_covs.copy_(covs)

        # --- warm-start π, A, D ---
        pi_counts = torch.bincount(labels[:1], minlength=K).to(dtype=DTYPE)
        pi = (pi_counts + 1e-6) / (pi_counts.sum() + 1e-6 * K)
        self._pi_logits.copy_(pi.log().to(self._pi_logits.device))

        # build label segments (for A and D)
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

        # transitions
        A_counts = torch.full((K, K), smooth_transition, dtype=DTYPE, device=self._A_logits.device)
        prev = None
        for s, _ in segments:
            if prev is not None:
                A_counts[prev, s] += 1.0
            prev = s
        A_probs = A_counts / A_counts.sum(dim=1, keepdim=True)
        self._A_logits.copy_(A_probs.log().to(self._A_logits.device))

        # durations
        D_counts = torch.full((K, self.max_duration), smooth_duration, dtype=DTYPE, device=self._D_logits.device)
        for s, l in segments:
            D_counts[s, min(l, self.max_duration) - 1] += 1.0
        D_probs = D_counts / D_counts.sum(dim=1, keepdim=True)
        self._D_logits.copy_(D_probs.log().to(self._D_logits.device))

        # --- finalize emission PDF ---
        self._params["emission_pdf"] = MultivariateNormal(self._emission_means, self._emission_covs)

    # -------------------------
    # --- Convenience decode
    # -------------------------
    def decode(
        self,
        X: torch.Tensor,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        duration_weight: float = 0.0
    ) -> np.ndarray:
        """
        Decode a single observation sequence X and return predicted state labels as a NumPy array.

        Args:
            X: (T, F) observation tensor.
            algorithm: decoding strategy, "viterbi" or "map".
            duration_weight: optional weight for duration modeling (currently ignored if predict doesn't use it).

        Returns:
            np.ndarray of predicted state indices, shape (T,)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=DTYPE)

        # Predict may return tensor, list of tensors, or other structures
        preds = self.predict(X, algorithm=algorithm, context=None)

        # Normalize output to single tensor
        if isinstance(preds, list):
            preds = preds[0] if len(preds) > 0 else torch.empty(0, dtype=torch.long)
        elif not torch.is_tensor(preds):
            preds = torch.as_tensor(preds, dtype=torch.long)

        return preds.cpu().numpy()
