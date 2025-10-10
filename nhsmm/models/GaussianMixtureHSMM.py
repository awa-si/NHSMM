# nhsmm/hsmm/GaussianMixtureHSMM.py
from typing import Optional
import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from sklearn.cluster import KMeans

from nhsmm.hsmm.BaseHSMM import BaseHSMM
from nhsmm.utilities import utils, constraints


class GaussianMixtureHSMM(BaseHSMM):
    """
    Gaussian mixture emissions HSMM.

    Emission distribution is a MixtureSameFamily with
      - mixture_distribution: Categorical (batch_shape=(n_states,), logits shape (n_states, n_components))
      - component_distribution: MultivariateNormal (batch_shape=(n_states,n_components), event_shape=(n_features,))

    Buffers registered:
      - _emission_weights_logits : (n_states, n_components)  (logits in log-space)
      - _emission_means : (n_states, n_components, n_features)
      - _emission_covs  : (n_states, n_components, n_features, n_features)
    """

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 max_duration: int,
                 n_components: int = 1,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 covariance_type: constraints.CovarianceType = constraints.CovarianceType.FULL,
                 min_covar: float = 1e-3,
                 seed: Optional[int] = None):

        self.n_features = int(n_features)
        self.min_covar = float(min_covar)
        self.k_means = bool(k_means)
        self.covariance_type = covariance_type
        self.n_components = int(n_components)

        super().__init__(n_states, max_duration, alpha, seed)

        # Register emission buffers from the initialized pdf
        pdf = self._params.get('emission_pdf')
        if pdf is not None:
            # mixture_distribution.logits shape: (n_states, n_components)
            w_logits = pdf.mixture_distribution.logits.double()
            means = pdf.component_distribution.loc.double()  # (n_states, n_components, n_features)
            covs = pdf.component_distribution.covariance_matrix.double()  # (n_states, n_components, n_features, n_features)
        else:
            # fallback defaults (shouldn't happen because BaseHSMM called sample_emission_pdf)
            w_logits = torch.log(constraints.sample_probs(self.alpha, (self.n_states, self.n_components))).double()
            means = torch.zeros((self.n_states, self.n_components, self.n_features), dtype=torch.float64)
            covs = (self.min_covar * torch.eye(self.n_features, dtype=torch.float64)
                    .expand(self.n_states, self.n_components, self.n_features, self.n_features)
                    .clone())

        # register or copy into buffers
        if not hasattr(self, "_emission_weights_logits"):
            self.register_buffer("_emission_weights_logits", w_logits.clone())
            self.register_buffer("_emission_means", means.clone())
            self.register_buffer("_emission_covs", covs.clone())
        else:
            self._emission_weights_logits.copy_(w_logits)
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        # ensure _params['emission_pdf'] is consistent with buffers
        self._params['emission_pdf'] = self._build_pdf_from_buffers()

    @property
    def dof(self) -> int:
        # degrees of freedom: transitions + mixture weights + component params
        # approximate: n_states^2 - 1 + n_states*(n_components - 1) + params in means+covs
        comp_params = self._emission_means.numel() + self._emission_covs.numel()
        mix_weight_params = self.n_states * (self.n_components - 1)
        return self.n_states**2 - 1 + mix_weight_params + comp_params

    # ---------- emission PDF helpers ----------
    def _build_pdf_from_buffers(self) -> MixtureSameFamily:
        """Construct MixtureSameFamily from registered buffers."""
        logits = self._emission_weights_logits.double()
        means = self._emission_means.double()
        covs = self._emission_covs.double()
        mix = Categorical(logits=logits)  # batch_shape (n_states,)
        comp = MultivariateNormal(loc=means, covariance_matrix=covs)  # batch_shape (n_states,n_components)
        return MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)

    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> MixtureSameFamily:
        """Initialize mixture emission distribution (batch_shape=(n_states,))."""
        # mixture logits
        weights = torch.log(constraints.sample_probs(self.alpha, (self.n_states, self.n_components))).double()

        if X is not None:
            # compute per-state/component means
            if self.k_means:
                # run k-means per-state by clustering into (n_states*n_components) then reshape
                total_clusters = self.n_states * self.n_components
                km = KMeans(n_clusters=total_clusters, random_state=self.seed, n_init="auto")
                centers = torch.from_numpy(km.fit(X.cpu().numpy()).cluster_centers_).to(torch.float64)
                means = centers.reshape(self.n_states, self.n_components, self.n_features)
            else:
                # use global mean replicated
                means = X.mean(dim=0, keepdim=True).expand(self.n_states, self.n_components, -1).clone().double()
            centered = X - X.mean(dim=0)
            base_cov = (torch.mm(centered.T, centered) / (X.shape[0] - 1)).double()
            covs = base_cov.expand(self.n_states, self.n_components, self.n_features, self.n_features).clone()
        else:
            means = torch.zeros((self.n_states, self.n_components, self.n_features), dtype=torch.float64)
            covs = (self.min_covar * torch.eye(self.n_features, dtype=torch.float64)
                    .expand(self.n_states, self.n_components, self.n_features, self.n_features)
                    .clone())

        # write into buffers (so state_dict picks them up)
        if not hasattr(self, "_emission_weights_logits"):
            self.register_buffer("_emission_weights_logits", weights.clone())
            self.register_buffer("_emission_means", means.clone())
            self.register_buffer("_emission_covs", covs.clone())
        else:
            self._emission_weights_logits.copy_(weights)
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

        pdf = self._build_pdf_from_buffers()
        return pdf

    def _compute_log_responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log responsibilities: shape (n_states, n_components, n_samples)

        log r_{s,c,n} = log w_{s,c} + log N(x_n | mean_{s,c}, cov_{s,c}) - logsumexp_c(...)
        """
        pdf = self._params.get('emission_pdf')
        if pdf is None:
            pdf = self._build_pdf_from_buffers()
            self._params['emission_pdf'] = pdf

        # component_distribution.log_prob accepts X broadcasted -> returns (n_states, n_components, n_samples)
        comp_logp = pdf.component_distribution.log_prob(X)  # (s, c, n)
        mix_logits = pdf.mixture_distribution.logits.double().unsqueeze(-1)  # (s, c, 1)
        log_resp = constraints.log_normalize(mix_logits + comp_logp, dim=1)  # normalize across components
        return log_resp  # (s, c, n)

    # ---------- EM emission updates ----------
    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta: Optional[utils.ContextualVariables] = None) -> MixtureSameFamily:
        """
        Re-estimate mixture emission parameters given:
          - X: (n_samples, n_features)
          - posterior: (n_states, n_samples) (state posteriors gamma)
        Returns a new MixtureSameFamily and updates buffers.
        """
        # compute responsibilities p(component | state, x) in log-space
        log_resp = self._compute_log_responsibilities(X)  # (s, c, n)
        resp = log_resp.exp()  # (s, c, n)

        # posterior shape (s, n) -> expand to (s, c, n)
        posterior_exp = posterior.unsqueeze(1)  # (s,1,n)
        posterior_resp = resp * posterior_exp  # (s,c,n)

        # effective counts per state-component
        eff_counts = posterior_resp.sum(dim=-1)  # (s, c)
        eps = 1e-12
        eff_counts_safe = eff_counts.clamp_min(eps)

        # new mixture weights (logits) per state
        new_weights_logits = torch.log(eff_counts_safe)  # (s, c)
        new_weights_logits = constraints.log_normalize(new_weights_logits, dim=1)

        # new means: numerator = sum_n posterior_resp * x_n
        # use einsum for clarity: (s,c,n) @ (n,f) -> (s,c,f)
        new_means = torch.einsum('scn,nf->scf', posterior_resp, X) / eff_counts_safe.unsqueeze(-1)

        # new covariances
        # diff: (s,c,n,f)
        diff = X.unsqueeze(0).unsqueeze(0) - new_means.unsqueeze(2)
        posterior_adj = posterior_resp.unsqueeze(-1)  # (s,c,n,1)
        # covariance numerator: sum_n posterior_resp * diff^T diff -> (s,c,f,f)
        cov_num = torch.matmul((posterior_adj * diff).transpose(-1, -2), diff)  # (s,c,f,f)
        new_covs = cov_num / eff_counts_safe.unsqueeze(-1).unsqueeze(-1)
        # regularize
        new_covs = new_covs + self.min_covar * torch.eye(self.n_features, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

        # update buffers (so state_dict persists)
        self._emission_weights_logits.copy_(new_weights_logits.double())
        self._emission_means.copy_(new_means.double())
        self._emission_covs.copy_(new_covs.double())

        # rebuild distribution and store
        new_pdf = self._build_pdf_from_buffers()
        self._params['emission_pdf'] = new_pdf
        return new_pdf

    # ---------- helpers ----------
    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """KMeans initialization producing (n_states, n_components, n_features)."""
        total_clusters = self.n_states * self.n_components
        km = KMeans(n_clusters=total_clusters, random_state=seed or self.seed, n_init="auto")
        centers = torch.from_numpy(km.fit(X.cpu().numpy()).cluster_centers_).to(torch.float64)
        return centers.reshape(self.n_states, self.n_components, self.n_features)

    # optional convenience save/load (preserve old pattern)
    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'weights': self._emission_weights_logits,
            'means': self._emission_means,
            'covs': self._emission_covs,
            'config': {
                'n_states': self.n_states,
                'n_features': self.n_features,
                'max_duration': self.max_duration,
                'n_components': self.n_components,
                'k_means': self.k_means,
                'alpha': self.alpha,
                'min_covar': self.min_covar,
                'covariance_type': self.covariance_type,
                'seed': self.seed,
            }
        }, path)

    @classmethod
    def load(cls, path: str) -> "GaussianMixtureHSMM":
        data = torch.load(path, map_location='cpu')
        cfg = data['config']
        model = cls(
            n_states=cfg['n_states'],
            n_features=cfg['n_features'],
            max_duration=cfg['max_duration'],
            n_components=cfg.get('n_components', 1),
            k_means=cfg.get('k_means', False),
            alpha=cfg.get('alpha', 1.0),
            min_covar=cfg.get('min_covar', 1e-3),
            seed=cfg.get('seed', None)
        )
        model.load_state_dict(data['state_dict'])
        model._emission_weights_logits.copy_(data['weights'])
        model._emission_means.copy_(data['means'])
        model._emission_covs.copy_(data['covs'])
        model._params['emission_pdf'] = model._build_pdf_from_buffers()
        return model
