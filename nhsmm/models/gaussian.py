# models/gaussian.py
import torch
from torch.distributions import MultivariateNormal
from sklearn.cluster import KMeans
from typing import Optional, Literal
import numpy as np

from nhsmm.defaults import DTYPE
from nhsmm.models import HSMM


class GaussianHSMM(HSMM):
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
        alpha: float = 1.0,
        min_covar: float = 1e-3,
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
            alpha=alpha,
            seed=seed,
            min_covar=min_covar,
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
        """Degrees of freedom: states, durations, emissions."""
        return self.n_states**2 - 1 + int(self._emission_means.numel()) + int(self._emission_covs.numel())

    # -------------------------
    # --- EMISSION INITIALIZATION
    # -------------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> MultivariateNormal:
        # Early-guard: if called before buffers exist, return safe identity-like PDF
        if not hasattr(self, "_emission_means") or not hasattr(self, "_emission_covs"):
            tmp_means = torch.zeros(self.n_states, self.n_features, dtype=DTYPE)
            tmp_covs = torch.stack([torch.eye(self.n_features, dtype=DTYPE) for _ in range(self.n_states)])
            return MultivariateNormal(tmp_means, tmp_covs)

        dev = self._emission_means.device

        if X is not None:
            X = X.to(dtype=DTYPE, device=dev)
            if self.k_means:
                means = self._sample_kmeans(X)
            else:
                means = X.mean(dim=0).expand(self.n_states, -1).to(dtype=DTYPE, device=dev)

            centered = X - X.mean(dim=0)
            denom = max(X.shape[0] - 1, 1)
            base_cov = (centered.T @ centered) / denom  # (F,F)
            covs = base_cov.expand(self.n_states, -1, -1).to(dtype=DTYPE, device=dev)
        else:
            means = self._emission_means.to(dev)
            covs = self._emission_covs.to(dev)

        # copy into registered buffers (keeps device/dtype consistent)
        self._emission_means.copy_(means.to(dev))
        self._emission_covs.copy_(covs.to(dev))

        # final safety: symmetrize & regularize (in-place copy back)
        covs_safe = 0.5 * (self._emission_covs + self._emission_covs.transpose(-1, -2))
        covs_safe += self.min_covar * torch.eye(self.n_features, dtype=DTYPE, device=dev).unsqueeze(0)
        self._emission_covs.copy_(covs_safe)

        return MultivariateNormal(self._emission_means, self._emission_covs)

    # -------------------------
    # --- EMISSION UPDATE
    # -------------------------
    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta=None) -> MultivariateNormal:
        X = X.to(dtype=DTYPE)
        posterior = posterior.to(dtype=DTYPE)
        means = self._compute_means(X, posterior)
        covs = self._compute_covs(X, posterior, means)

        # copy to registered buffers on correct device
        dev = self._emission_means.device
        self._emission_means.copy_(means.to(dtype=DTYPE, device=dev))
        self._emission_covs.copy_(covs.to(dtype=DTYPE, device=dev))

        return MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)

    # -------------------------
    # --- HELPERS
    # -------------------------
    def _compute_means(self, X: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        """
        X: (N, F), posterior: (N, K)
        Returns: (K, F)
        """
        assert X.ndim == 2 and posterior.ndim == 2, "X and posterior must be 2D (N, F) and (N, K)"
        assert posterior.shape[1] == self.n_states, "posterior second dim must equal n_states"

        weighted_sum = posterior.T @ X               # (K, F)
        norm = posterior.T.sum(dim=1, keepdim=True).clamp_min(1e-10)  # (K,1)
        return weighted_sum / norm

    def _compute_covs(self, X: torch.Tensor, posterior: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """
        Memory-safe per-state covariance computation.
        Returns covs: (K, F, F)
        """
        N, F = X.shape
        device = X.device
        covs = torch.zeros(self.n_states, F, F, dtype=DTYPE, device=device)

        for s in range(self.n_states):
            w = posterior[:, s].unsqueeze(-1).to(dtype=DTYPE, device=device)  # (N,1)
            diff = (X - means[s].to(device=device)).to(dtype=DTYPE, device=device)  # (N,F)
            denom = w.sum().clamp_min(1e-10)
            C = (w * diff).T @ diff / denom  # (F,F)
            C = 0.5 * (C + C.transpose(-1, -2))  # symmetrize
            C += self.min_covar * torch.eye(F, dtype=DTYPE, device=device)
            covs[s] = C

        return covs

    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        # sklearn on CPU, return dtype/device matching model buffers
        X_np = X.detach().cpu().numpy()
        km = KMeans(n_clusters=self.n_states, n_init=10, random_state=int(seed or (getattr(self, "_seed_gen", None) and self._seed_gen.seed) or 0))
        labels = km.fit_predict(X_np)
        centers = torch.from_numpy(km.cluster_centers_).to(dtype=DTYPE)
        return centers.to(dtype=DTYPE, device=self._emission_means.device)

    def _contextual_emission_pdf(self, X, theta=None):
        """No contextual modulation for pure Gaussian HSMM."""
        return self._params.get('emission_pdf', None)

    # -------------------------
    # --- Initialization helper (new)
    # -------------------------
    def initialize_emissions(
        self,
        X: torch.Tensor,
        method: Literal["kmeans", "moment"] = "kmeans",
        smooth_transition: float = 1e-2,
        smooth_duration: float = 1e-2
    ):
        """
        Initialize emissions and warm-start pi / A / D using kmeans assignments or moments.

        Args:
          X: (T, F) tensor of observations (single concatenated sequence).
          method: 'kmeans' or 'moment'
          smooth_transition: additive smoothing for transition counts (Laplace)
          smooth_duration: additive smoothing to duration counts
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=DTYPE)
        X = X.to(dtype=DTYPE, device=self._emission_means.device)

        T = X.shape[0]
        K = self.n_states

        if method == "moment":
            mu_init = X.mean(dim=0, keepdim=True).repeat(K, 1)
            var_init = X.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1).clamp_min(self.min_covar)
            cov_init = torch.diag_embed(var_init.squeeze(1))
            self._emission_means.copy_(mu_init.to(self._emission_means.device))
            # broadcast diagonal cov->full covariance per state
            covs = cov_init.expand(K, self.n_features, self.n_features).to(self._emission_covs.device)
            self._emission_covs.copy_(covs)
        elif method == "kmeans":
            centers = self._sample_kmeans(X)
            # compute per-cluster covariances and counts
            km = KMeans(n_clusters=K, n_init=10, random_state=int(getattr(self._seed_gen, "seed", 0)))
            labels = km.fit_predict(X.cpu().numpy())
            labels = torch.as_tensor(labels, device=X.device)
            mus = torch.zeros(K, self.n_features, dtype=DTYPE, device=self._emission_means.device)
            covs = torch.zeros(K, self.n_features, self.n_features, dtype=DTYPE, device=self._emission_covs.device)
            counts = torch.zeros(K, dtype=DTYPE, device=X.device)
            for k in range(K):
                mask = labels == k
                Nk = int(mask.sum().item())
                if Nk > 1:
                    Xk = X[mask]
                    mus[k] = Xk.mean(dim=0)
                    centered = Xk - Xk.mean(dim=0)
                    denom = max(Xk.shape[0] - 1, 1)
                    cov_k = (centered.T @ centered) / denom
                    cov_k = 0.5 * (cov_k + cov_k.transpose(-1, -2))
                    cov_k += self.min_covar * torch.eye(self.n_features, dtype=DTYPE, device=self._emission_covs.device)
                    covs[k] = cov_k
                    counts[k] = Nk
                elif Nk == 1:
                    mus[k] = X[mask].squeeze(0)
                    covs[k] = torch.eye(self.n_features, dtype=DTYPE, device=self._emission_covs.device) * (self.min_covar + 1e-6)
                    counts[k] = 1.0
                else:
                    # empty cluster fallback to global stats
                    mus[k] = X.mean(dim=0)
                    covs[k] = torch.eye(self.n_features, dtype=DTYPE, device=self._emission_covs.device) * (self.min_covar + 1e-6)
                    counts[k] = 1.0

            self._emission_means.copy_(mus)
            self._emission_covs.copy_(covs)
            # Warm-start pi, A, D from labels
            # pi by empirical first-state frequency (approx)
            first_labels = [labels[0].item()] if labels.numel() > 0 else []
            if len(first_labels) > 0:
                pi_counts = torch.bincount(torch.tensor(first_labels, device=labels.device), minlength=K).to(dtype=DTYPE)
                pi = (pi_counts + 1e-6) / (pi_counts.sum() + 1e-6 * K)
                self._pi_logits.copy_(torch.log(pi.to(self._pi_logits.device, dtype=DTYPE)))
            # transition counts and duration histogram
            # simple segmentation of consecutive equal labels
            segments = []
            if labels.numel() > 0:
                lbls = labels.cpu().numpy()
                cur = lbls[0]
                length = 1
                for v in lbls[1:]:
                    if v == cur:
                        length += 1
                    else:
                        segments.append((int(cur), int(length)))
                        cur = v
                        length = 1
                segments.append((int(cur), int(length)))
            # transition counts
            A_counts = torch.ones((K, K), dtype=DTYPE, device=self._A_logits.device) * smooth_transition
            prev = None
            for s, l in segments:
                if prev is not None:
                    A_counts[prev, s] += 1.0
                prev = s
            A_probs = A_counts / A_counts.sum(dim=1, keepdim=True)
            self._A_logits.copy_(torch.log(A_probs.to(self._A_logits.device)))
            # duration histogram
            D_counts = torch.ones((K, self.max_duration), dtype=DTYPE, device=self._D_logits.device) * smooth_duration
            for s, l in segments:
                d = min(l, self.max_duration)
                D_counts[s, d-1] += 1.0
            D_probs = D_counts / D_counts.sum(dim=1, keepdim=True)
            self._D_logits.copy_(torch.log(D_probs.to(self._D_logits.device)))
        else:
            raise ValueError("Unknown initialization method")

        # update emission pdf object
        self._params['emission_pdf'] = MultivariateNormal(self._emission_means, self._emission_covs)

    # -------------------------
    # --- Convenience decode
    # -------------------------
    def decode(self, X: torch.Tensor, algorithm: Literal["viterbi","map"]="viterbi", duration_weight: float = 0.0):
        """
        Run decoding and return numpy array of predicted labels for single sequence X.
        Uses self.predict / _viterbi depending on implementation.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=DTYPE)

        # Some predict implementations expect (T,F)
        preds = self.predict(X, algorithm=algorithm, context=None)
        if isinstance(preds, list) and len(preds) > 0:
            out = preds[0].cpu().numpy()
        elif torch.is_tensor(preds):
            out = preds.cpu().numpy()
        else:
            out = np.asarray(preds)
        return out

    # -------------------------
    # --- PERSISTENCE
    # -------------------------
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
                'k_means': self.k_means,
                'seed': getattr(self._seed_gen, "seed", None),
            }
        }, path)

    @classmethod
    def load(cls, path: str) -> "GaussianHSMM":
        data = torch.load(path, map_location='cpu')
        model = cls(**data['config'])
        model.load_state_dict(data['state_dict'])
        model._emission_means.copy_(data['means'])
        model._emission_covs.copy_(data['covs'])
        if not hasattr(model, "_params"):
            model._params = {}
        dev = model._emission_covs.device
        covs = 0.5 * (model._emission_covs + model._emission_covs.transpose(-1, -2))
        covs += model.min_covar * torch.eye(model.n_features, dtype=DTYPE, device=dev).unsqueeze(0)
        model._emission_covs.copy_(covs)
        model._params['emission_pdf'] = MultivariateNormal(model._emission_means, model._emission_covs)
        return model
