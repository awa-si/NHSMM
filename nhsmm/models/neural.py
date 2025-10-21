# nhsmm/models/neural.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Independent, Categorical, Poisson, LogNormal, MultivariateNormal, Dirichlet

from typing import Literal, Optional, Union, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass
import numpy as np

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition
from nhsmm.models.hsmm import HSMM
from nhsmm import utils


class NeuralEmission(nn.Module):
    """Batched, neural/contextual emission distribution."""

    def __init__(self, emission_type: str,
                 params: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 encoder: Optional[nn.Module] = None,
                 context_mode: str = "none",
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.context_mode = context_mode.lower()
        if self.context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")
        self.encoder = encoder
        self.emission_type = emission_type.lower()

        if self.emission_type == "gaussian":
            mu, cov = params
            self.n_states, self.n_features = mu.shape
            self.mu = nn.Parameter(mu.to(self.device, DTYPE))
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            self.cov = nn.Parameter(cov.to(self.device, DTYPE))
            self._gaussian_proj = nn.Linear(self.n_features, 2 * self.n_states * self.n_features, bias=False).to(self.device, DTYPE)
        else:
            logits = params
            self.n_states, self.n_features = logits.shape
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))

        self.temporal_adapter = nn.Conv1d(self.n_features, self.n_features, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE) \
            if self.context_mode == "temporal" else None
        self.spatial_adapter = nn.Linear(self.n_features, self.n_features, bias=False).to(self.device, DTYPE) \
            if self.context_mode == "spatial" else None

    @property
    def dof(self):
        total = self.mu.numel() + getattr(self, "cov", torch.tensor([])).numel() if self.emission_type=="gaussian" else self.logits.numel()
        for m in [self.encoder, self.temporal_adapter, self.spatial_adapter]:
            if m:
                total += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total

    def _apply_context(self, theta: Optional[torch.Tensor]):
        out = theta
        if self.encoder and theta is not None:
            out = self.encoder(theta)
        if self.temporal_adapter and out is not None and out.ndim == 3:
            out = self.temporal_adapter(out.transpose(1,2)).transpose(1,2)
        if self.spatial_adapter and out is not None:
            out = self.spatial_adapter(out)
        return out

    def contextual_params(self, theta: Optional[torch.Tensor] = None):
        out = self._apply_context(theta)
        if self.emission_type == "gaussian":
            if out is None: return self.mu, self.cov
            tmp = self._gaussian_proj(out).view(out.shape[0],2,self.n_states,self.n_features)
            mu = tmp[:,0]
            cov = torch.diag_embed(F.softplus(tmp[:,1]) + EPS)
            return mu.to(self.device, DTYPE), cov.to(self.device, DTYPE)
        logits = out if out is not None else self.logits
        if logits.ndim==2: logits = logits.unsqueeze(0).expand(theta.shape[0],-1,-1) if theta is not None else logits.unsqueeze(0)
        return F.log_softmax(logits, dim=-1).clamp_min(EPS)

    def forward(self, X: torch.Tensor, theta: Optional[torch.Tensor]=None, log: bool=False):
        X = X.to(self.device, DTYPE)
        batch_mode = X.ndim==3
        B, T, Fdim = (X.shape if batch_mode else (1,*X.shape))
        K = self.n_states

        if self.emission_type=="gaussian":
            mu, cov = self.contextual_params(theta)
            if mu.ndim==2: mu = mu.unsqueeze(0).expand(B,K,Fdim)
            if cov.ndim==3: cov = cov.unsqueeze(0).expand(B,K,Fdim,Fdim)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov),1)
            logp = dist.log_prob(X.unsqueeze(-2))
            return logp if log else logp.exp()
        else:
            logits = self.contextual_params(theta)
            probs = F.softmax(logits, dim=-1)
            X_idx = X.long()
            if X_idx.max()>=self.n_features or X_idx.min()<0:
                raise ValueError(f"Observation index out of bounds: 0 <= X < {self.n_features}")
            logp = torch.log(probs.gather(-1, X_idx.unsqueeze(-2).expand(-1,K,-1))+EPS) if batch_mode else torch.log(probs[:,X_idx]+EPS).T
            return logp if log else logp.exp()

    def log_prob(self, X: torch.Tensor, theta: Optional[torch.Tensor]=None):
        return self.forward(X, theta=theta, log=True)

    def sample(self, n_samples: int=1, theta: Optional[torch.Tensor]=None, state_indices: Optional[torch.Tensor]=None):
        if self.emission_type=="gaussian":
            mu, cov = self.contextual_params(theta)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov),1)
            samples = dist.rsample((n_samples,)).transpose(0,1)
        else:
            logits = self.contextual_params(theta)
            dist = Categorical(logits=logits)
            samples = dist.sample((n_samples,)).transpose(0,1)
        return samples if state_indices is None else samples[:,state_indices]

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace=True):
        posterior = posterior.to(self.device, DTYPE)
        X = X.to(self.device, DTYPE)
        X_flat = X.reshape(-1,X.shape[-1]) if X.ndim>2 else X
        posterior_flat = posterior.reshape(-1,posterior.shape[-1]) if posterior.ndim>2 else posterior

        if self.emission_type=="gaussian":
            counts = posterior_flat.sum(dim=0).clamp_min(EPS)[:,None]
            mu_new = (posterior_flat.T @ X_flat)/counts
            cov_new = torch.stack([((X_flat - mu_new[s]).T * posterior_flat[:,s])@(X_flat - mu_new[s])/counts[s,0] + torch.eye(self.n_features,device=self.device,dtype=DTYPE)*EPS for s in range(self.n_states)])
            if inplace:
                self.mu.data.copy_(mu_new)
                self.cov.data.copy_(cov_new)
                return self
            return NeuralEmission("gaussian",(mu_new,cov_new),encoder=self.encoder,context_mode=self.context_mode,device=self.device)
        else:
            X_onehot = F.one_hot(X_flat.long(),num_classes=self.n_features).float()
            weighted_counts = posterior_flat.T @ X_onehot
            logits_new = (weighted_counts/weighted_counts.sum(dim=1,keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
            if inplace:
                self.logits.data.copy_(logits_new)
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder, context_mode=self.context_mode, device=self.device)

    @classmethod
    def initialize(cls, emission_type, n_states, n_features=None, n_categories=None, alpha=1.0, encoder=None, device=None):
        device = device or torch.device("cpu")
        if emission_type.lower()=="gaussian":
            mu = torch.randn(n_states, n_features, device=device, dtype=DTYPE)*0.1
            cov = torch.stack([torch.ones(n_features,device=device,dtype=DTYPE) for _ in range(n_states)])
            return cls("gaussian",(mu,cov),encoder=encoder,device=device)
        else:
            logits = Dirichlet(torch.ones(n_categories,device=device,dtype=DTYPE)*alpha).sample([n_states]).clamp_min(EPS).log()
            return cls(emission_type.lower(), logits, encoder=encoder, device=device)

class NeuralDuration(nn.Module):
    """Batched neural/contextual duration distribution."""

    def __init__(self, n_states: int,
                 mode: str="poisson",
                 rate: Optional[torch.Tensor]=None,
                 mean: Optional[torch.Tensor]=None,
                 std: Optional[torch.Tensor]=None,
                 encoder: Optional[nn.Module]=None,
                 context_mode: str="none",
                 max_duration: int=20,
                 device: Optional[torch.device]=None,
                 scale: float=0.1):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_states = n_states
        self.mode = mode.lower()
        self.max_duration = max_duration
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = scale

        self.temporal_adapter = nn.Conv1d(n_states,n_states,kernel_size=3,padding=1,bias=False).to(self.device, DTYPE) if self.context_mode=="temporal" else None
        self.spatial_adapter = nn.Linear(n_states,n_states,bias=False).to(self.device, DTYPE) if self.context_mode=="spatial" else None

        if self.mode=="poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states,device=self.device,dtype=DTYPE))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states,device=self.device,dtype=DTYPE))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states,device=self.device,dtype=DTYPE))

    @property
    def dof(self):
        dof = self.rate.numel() if self.mode=="poisson" else self.mean.numel()+self.std.numel()
        for m in [self.encoder,self.temporal_adapter,self.spatial_adapter]:
            if m:
                dof += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return dof

    def _contextual_params(self, context: Optional[torch.Tensor]=None):
        X = getattr(context,"X",context) if hasattr(context,"X") else context
        if X is not None:
            X = X.to(self.device, DTYPE)
        encoded = self.encoder(X) if self.encoder and X is not None else None
        if encoded is not None:
            if self.temporal_adapter and encoded.ndim==3:
                encoded = encoded.transpose(1,2)
                encoded = self.temporal_adapter(encoded)
                encoded = encoded.transpose(1,2)
            if self.spatial_adapter:
                encoded = self.spatial_adapter(encoded)
        if self.mode=="poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        else:
            if encoded is not None:
                mean, log_std = torch.chunk(encoded,2,dim=-1)
            else:
                mean, log_std = self.mean, self.std.log()
            return mean, log_std.exp().clamp_min(EPS)

    def forward(self, context: Optional[torch.Tensor]=None, log: bool=False, return_dist: bool=False):
        durations = torch.arange(1,self.max_duration+1,device=self.device,dtype=DTYPE)
        if self.mode=="poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0)*torch.log(rate.unsqueeze(-1))-rate.unsqueeze(-1)-torch.lgamma(durations.unsqueeze(0)+1)
        else:
            mean, std = self._contextual_params(context)
            logits = -0.5*((durations.unsqueeze(0)-mean.unsqueeze(-1))/std.unsqueeze(-1))**2 - torch.log(std.unsqueeze(-1)) - 0.5*torch.log(2*torch.pi)
        logits = logits-logits.max(dim=-1,keepdim=True)[0]
        probs = F.softmax(logits,dim=-1)
        if return_dist:
            if self.mode=="poisson": return [Poisson(rate[k]) for k in range(self.n_states)]
            else: return [LogNormal(mean[k], std[k]) for k in range(self.n_states)]
        return torch.log(probs) if log else probs

    def log_prob(self, X: torch.Tensor, context: Optional[torch.Tensor]=None):
        return self.forward(context=context, log=True)

    def sample(self, n_samples: int=1, context: Optional[torch.Tensor]=None, state_indices: Optional[torch.Tensor]=None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists],dim=0)
        return samples if state_indices is None else samples[state_indices]

class NeuralTransition(nn.Module):
    """Batched neural/contextual transition distribution."""

    def __init__(self, n_states: int,
                 logits: Optional[torch.Tensor]=None,
                 encoder: Optional[nn.Module]=None,
                 context_mode: str="none",
                 device: Optional[torch.device]=None,
                 scale: float=0.1,
                 temperature: float=1.0):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_states = n_states
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = scale
        self.temperature = temperature

        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))
        else:
            uniform = torch.full((n_states,n_states),1.0/n_states,device=self.device,dtype=DTYPE)
            self.logits = nn.Parameter(torch.log(uniform.clamp_min(EPS)))

        self.temporal_adapter = nn.Conv1d(n_states,n_states,kernel_size=3,padding=1,bias=False).to(self.device, DTYPE) if self.context_mode=="temporal" else None
        self.spatial_adapter = nn.Linear(n_states,n_states,bias=False).to(self.device, DTYPE) if self.context_mode=="spatial" else None

    @property
    def dof(self):
        dof = self.n_states*(self.n_states-1)
        for m in [self.encoder,self.temporal_adapter,self.spatial_adapter]:
            if m:
                dof += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return dof

    def contextual_logits(self, theta: Optional[torch.Tensor]=None):
        base_logits = self.logits.clone()
        if self.encoder and theta is not None:
            X = getattr(theta,"X",theta)
            out = self.encoder(X)
            if isinstance(out, tuple): out = out[0]
            if out.ndim==2 and out.shape[1]==self.n_states**2:
                out = out.view(-1,self.n_states,self.n_states)
            elif out.ndim==3 and out.shape[1:]!=(self.n_states,self.n_states):
                out = out.view(-1,self.n_states,self.n_states)
            delta = torch.tanh(out)*self.scale
            base_logits = base_logits.unsqueeze(0).expand_as(delta)+delta

        if self.temporal_adapter and theta is not None:
            delta = theta.transpose(1,2)
            delta = self.temporal_adapter(delta).transpose(1,2)
            base_logits = base_logits+delta

        if self.spatial_adapter and theta is not None:
            delta = self.spatial_adapter(theta)
            base_logits = base_logits+delta

        safe_logits = F.log_softmax(base_logits/self.temperature, dim=-1)
        safe_logits = torch.where(torch.isnan(safe_logits)|torch.isinf(safe_logits),
                                  torch.log(torch.full_like(safe_logits,1.0/self.n_states)), safe_logits)
        return safe_logits

    def update(self, posterior: torch.Tensor, inplace=True):
        posterior = posterior/posterior.sum(dim=-1,keepdim=True).clamp_min(EPS)
        logits = torch.log(posterior.to(self.device, DTYPE))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(self.n_states, logits=logits, encoder=self.encoder, context_mode=self.context_mode,
                                device=self.device, scale=self.scale, temperature=self.temperature)

    def forward(self, theta: Optional[torch.Tensor]=None):
        return self.contextual_logits(theta)


@dataclass
class NHSMMConfig:
    n_states: int
    n_features: int
    max_duration: int
    alpha: float = 1.0
    seed: Optional[int] = None
    emission_type: str = "gaussian"
    encoder: Optional[nn.Module] = None
    device: Optional[torch.device] = None
    context_dim: Optional[int] = None
    min_covar: float = 1e-3
    emission_scale: float = 0.1
    transition_scale: float = 0.1
    duration_scale: float = 0.1
    emission_temp: float = 1.0
    transition_temp: float = 1.0
    duration_temp: float = 1.0
    n_context_states: Optional[int] = None

class NeuralHSMM(HSMM):
    """ NeuralHSMM with EM, Viterbi, context, and gradient support.
    """

    def __init__(self, config: NHSMMConfig):
        
        self.config = config
        nn.Module.__init__(self)

        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = config.encoder.to(self.device, dtype=DTYPE) if config.encoder else None
        self.n_features = config.n_features
        self.min_covar = config.min_covar


        # Neural modules
        self.emission_module = NeuralEmission.initialize(
            emission_type=config.emission_type,
            n_states=config.n_states,
            n_features=config.n_features,
            encoder=self.encoder,
            device=self.device,
        )
        self.duration_module = NeuralDuration(
            n_states=config.n_states,
            max_duration=config.max_duration,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim else "none",
            scale=config.duration_scale,
            device=self.device,
        )
        self.transition_module = NeuralTransition(
            n_states=config.n_states,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim else "none",
            scale=config.transition_scale,
            device=self.device,
        )

        # Context embeddings
        if config.n_context_states and config.context_dim:
            self.context_embedding = nn.Embedding(config.n_context_states, config.context_dim)
            nn.init.normal_(self.context_embedding.weight, 0.0, 1e-3)

            self.ctx_transition = nn.Linear(config.context_dim, config.n_states**2, device=self.device, dtype=DTYPE)
            self.ctx_duration = nn.Linear(config.context_dim, config.n_states*config.max_duration, device=self.device, dtype=DTYPE)
            self.ctx_emission = nn.Linear(config.context_dim, config.n_states*config.n_features, device=self.device, dtype=DTYPE)

            for m in (self.ctx_transition, self.ctx_duration, self.ctx_emission):
                nn.init.normal_(m.weight, 0.0, 1e-3)
                nn.init.normal_(m.bias, 0.0, 1e-3)
        else:
            self.context_embedding = None
            self.ctx_transition = None
            self.ctx_duration = None
            self.ctx_emission = None

        self._params: dict = {"emission_type": config.emission_type}

        super().__init__(
            n_states=config.n_states,
            n_features=config.n_features,
            max_duration=config.max_duration,
            context_dim=config.context_dim,
            min_covar=config.min_covar,
            alpha=config.alpha,
            device=self.device,
            seed=config.seed,
        )

        self._params["emission_pdf"] = self.sample_emission_pdf(
            temperature=config.emission_temp,
            scale=config.emission_scale,
        )
        self._context = None

    @property
    def dof(self) -> int:
        """
        Compute degrees of freedom (dof) for the HSMM model, including:
          - initial state probabilities
          - transition probabilities
          - duration probabilities
          - emission parameters
        """

        nS, nD, nF = self.n_states, self.max_duration, self.n_features
        dof = 0

        # Initial state probabilities
        dof += nS - 1

        # Transition matrix (row-stochastic)
        dof += nS * (nS - 1)

        # Duration distributions (row-stochastic)
        dof += nS * (nD - 1)

        # Emission parameters
        pdf = getattr(self, "pdf", None)
        if pdf is not None:
            if isinstance(pdf, Categorical):
                # Each state's categorical probs minus one (sum to 1)
                dof += nS * (pdf.logits.shape[-1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                # Mean and diagonal covariance for each state
                dof += nS * (nF + nF)  # mean + diag(cov)
            else:
                # Fallback: warn for unhandled pdf type
                import warnings
                warnings.warn(f"Unknown pdf type {type(pdf)}, emission dof not counted")

        return dof

    def _prepare_delta(
        self,
        delta: Optional[torch.Tensor],
        shape: Tuple[int, int, int],
        scale: float = 0.1,
        broadcast: bool = True
    ) -> torch.Tensor:
        """
        Safely reshape, pad/truncate, and scale delta tensor for contextual modulation.

        Args:
            delta: Tensor from encoder or external context (any shape), or None.
                   Can be [H], [B,H], [B,T,H], or already [B,n_states,feature_dim].
            shape: Target shape (B, n_states, feature_dim)
            scale: Scaling factor for modulation.
            broadcast: If True, automatically broadcast singleton batch to target_B.

        Returns:
            Tensor of shape (B, n_states, feature_dim), padded/truncated and scaled.
        """
        target_B, n_states, feature_dim = shape
        device, dtype = self.device, DTYPE

        # === 1. Missing delta → zero tensor ===
        if delta is None:
            return torch.zeros((target_B, n_states, feature_dim), dtype=dtype, device=device)

        delta = delta.to(device=device, dtype=dtype)

        # === 2. Handle already-correct shape ===
        if delta.shape == (target_B, n_states, feature_dim):
            return scale * torch.tanh(delta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0))

        # === 3. Remove temporal dimension or flatten arbitrary trailing dims ===
        if delta.ndim == 1:               # [H]
            delta = delta.unsqueeze(0)    # [1, H]
        elif delta.ndim == 3 and delta.shape[1] != n_states:
            # [B, T, H] → mean over time
            delta = delta.mean(dim=1)
        elif delta.ndim > 3:
            # Flatten anything extra beyond batch
            delta = delta.reshape(delta.shape[0], -1)

        # === 4. Infer shape info ===
        B, L = delta.shape[-2], delta.shape[-1]
        expected = n_states * feature_dim

        # === 5. Pad or truncate along feature axis ===
        if L < expected:
            delta = F.pad(delta, (0, expected - L))
        elif L > expected:
            delta = delta[..., :expected]

        # === 6. Reshape to (B, n_states, feature_dim) ===
        delta = delta.view(B, n_states, feature_dim)

        # === 7. Scale and clamp safely ===
        delta = scale * torch.tanh(delta)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        # === 8. Broadcast batch if needed ===
        if broadcast and B != target_B:
            if B == 1:
                delta = delta.expand(target_B, -1, -1)
            else:
                # Pad or crop batch dimension if partial mismatch
                if B < target_B:
                    extra = delta[:1].expand(target_B - B, -1, -1)
                    delta = torch.cat([delta, extra], dim=0)
                else:
                    delta = delta[:target_B]

        return delta

    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        scale: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Distribution:
        """
        Sample context-aware emission distribution (Gaussian or Categorical),
        aligned with NHSMM, supporting optional context theta and batch X.
        """
        nS, nF = self.n_states, self.n_features
        device, dtype = self.device, DTYPE
        scale = scale or getattr(self.config, "emission_scale", 0.1)

        # Set RNG for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Flatten batch if provided
        if X is not None:
            X = X.to(dtype=dtype, device=device)
            batch_mode = X.ndim == 3
            X_flat = X.reshape(-1, nF)
        else:
            batch_mode = False
            X_flat = None

        # Compute context delta
        combined = self.combine_context(theta)
        delta = None
        if combined is not None:
            delta = self.ctx_emission(combined) if getattr(self, "ctx_emission", None) else combined[..., :nS * nF]
        delta = self._prepare_delta(delta, shape=(1, nS, nF), scale=scale)
        delta_mean = delta.mean(dim=0)  # [nS, nF]

        # --- Categorical emission ---
        if self.emission_type.lower() == "categorical":
            if X_flat is not None:
                probs = X_flat.mean(dim=0, keepdim=True).expand(nS, -1)
            else:
                probs = torch.full((nS, nF), 1.0 / nF, dtype=dtype, device=device)

            logits = (probs + delta_mean) / max(temperature, 1e-6)
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            logits = torch.nan_to_num(logits)
            return Categorical(logits=logits)

        # --- Gaussian emission ---
        elif self.emission_type.lower() == "gaussian":
            if X_flat is not None:
                mean = X_flat.mean(dim=0, keepdim=True).expand(nS, -1)
                var = X_flat.var(dim=0, unbiased=False, keepdim=True).expand(nS, -1).clamp_min(getattr(self, "min_covar", 1e-6))
            else:
                mean = torch.zeros(nS, nF, dtype=dtype, device=device)
                var = torch.full((nS, nF), getattr(self, "min_covar", 1e-6), dtype=dtype, device=device)

            mean = mean + delta_mean
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> Distribution:
        """
        Estimate emission distribution from weighted data (posterior responsibilities),
        optionally modulated by context theta.

        Args:
            X: Observations [T, F] or [B*T, F].
            posterior: Responsibility matrix [T, K] or [B*T, K].
            theta: Optional context tensor [H] or [B, H].
            scale: Scaling for context modulation.

        Returns:
            MultivariateNormal (gaussian) or Categorical.
        """
        K, F = self.n_states, self.n_features
        device, dtype = self.device, DTYPE
        scale = scale or getattr(self.config, "emission_scale", 0.1)

        X = X.to(dtype=dtype, device=device)
        posterior = posterior.to(dtype=dtype, device=device)

        # --- Compute context delta ---
        combined = self.combine_context(theta)
        delta = None
        if combined is not None:
            if getattr(self, "ctx_emission", None):
                delta = self.ctx_emission(combined)
            else:
                delta = combined[..., :K*F]
        delta = self._prepare_delta(delta, shape=(1, K, F), scale=scale)
        delta_mean = delta.mean(dim=0)  # (K, F)

        # --- Categorical emission ---
        if self.emission_type == "categorical":
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                counts = torch.bincount(X.squeeze(-1).long(), minlength=F).to(dtype=dtype, device=device)
            else:
                counts = X.sum(dim=0).to(dtype=dtype, device=device)

            probs = counts / counts.sum()
            probs = (probs + delta_mean.mean(dim=0)).clamp_min(EPS)
            probs = probs / probs.sum()
            return Categorical(probs=probs.expand(K, -1))

        # --- Gaussian emission ---
        elif self.emission_type == "gaussian":
            Nk = posterior.sum(dim=0).clamp_min(EPS)  # (K,)
            mean = (posterior.T @ X) / Nk.unsqueeze(1)  # (K, F)

            # Weighted covariance
            diff = X.unsqueeze(1) - mean.unsqueeze(0)  # (T, K, F)
            weighted_diff = diff * posterior.unsqueeze(-1)
            cov = torch.einsum("tkf,tkh->kfh", weighted_diff, diff) / Nk.unsqueeze(-1).unsqueeze(-1)
            cov += torch.eye(F, dtype=dtype, device=device).unsqueeze(0) * getattr(self, "min_covar", 1e-6)

            mean = mean + delta_mean
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def initialize_emissions(self, X: torch.Tensor, method: str = "moment"):
        """
        Initialize emission parameters from raw data X (batched or unbatched),
        using context-consistent shifts. Supports Gaussian, Categorical, and Bernoulli emissions.
        """
        # --- 1. Ensure tensor on correct device/dtype ---
        X = X if torch.is_tensor(X) else torch.tensor(X)
        X = X.to(dtype=DTYPE, device=self.device)

        K = self.n_states
        F = X.shape[-1]
        emission_type = self.emission_type.lower()

        # Flatten batch if present
        X_flat = X.reshape(-1, F) if self.is_batch(X) else X

        # --- 2. Compute context delta using same logic as _contextual_emission_pdf ---
        delta = None
        combined_ctx = self.combine_context(None) if getattr(self, "combine_context", None) else None
        if getattr(self, "ctx_emission", None) and combined_ctx is not None:
            delta_raw = self.ctx_emission(combined_ctx)
            delta = self._prepare_delta(
                delta_raw,
                shape=(1, K, F),
                scale=getattr(self.config, "emission_scale", 1.0)
            )

        # --- 3. Gaussian emission ---
        if emission_type == "gaussian":
            if method == "moment":
                mu_init = X_flat.mean(dim=0, keepdim=True)  # [1, F]
                var_init = X_flat.var(dim=0, unbiased=False, keepdim=True)  # [1, F]
                mu_init = mu_init.expand(K, F)
                var_init = var_init.expand(K, F)
            elif method == "kmeans":
                labels = torch.tensor(
                    KMeans(n_clusters=K, n_init=10, random_state=0)
                    .fit_predict(X_flat.cpu().numpy()),
                    device=self.device, dtype=torch.long
                )
                mu_init = torch.stack([
                    X_flat[labels == k].mean(dim=0) if (labels == k).any() else X_flat.mean(dim=0)
                    for k in range(K)
                ])
                var_init = torch.stack([
                    X_flat[labels == k].var(dim=0, unbiased=False) if (labels == k).any() else X_flat.var(dim=0, unbiased=False)
                    for k in range(K)
                ])
            else:
                raise ValueError(f"Unknown initialization method '{method}'")

            # Apply context shift
            if delta is not None:
                mu_init = mu_init + delta.mean(dim=0)

            # Covariance
            var_init = var_init.clamp_min(getattr(self, "min_covar", 1e-6))
            cov_init = torch.diag_embed(var_init)

            # Assign parameters
            self.emission_module.mu = nn.Parameter(mu_init)
            self.emission_module.cov = nn.Parameter(cov_init)
            self._params["emission_pdf"] = MultivariateNormal(loc=mu_init, covariance_matrix=cov_init)

        # --- 4. Categorical / Bernoulli emission ---
        elif emission_type in ("categorical", "bernoulli"):
            if X_flat.ndim == 1 or X_flat.shape[1] == 1:
                counts = torch.bincount(X_flat.squeeze(-1).long(), minlength=self.n_features)
            else:
                counts = X_flat.sum(dim=0)
            counts = counts.to(dtype=DTYPE, device=self.device)

            probs = counts / counts.sum().clamp_min(EPS)

            # Context modulation in logits space for consistency
            logits = probs.log()[None, :]  # [1, F]
            if delta is not None:
                logits = logits + delta.mean(dim=0)  # broadcast across states
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            if logits.shape[0] == 1:
                logits = logits.expand(K, -1)  # [K, F]

            # Assign parameters
            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.cov = nn.Parameter(torch.zeros_like(logits))
            self._params["emission_pdf"] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

    def _contextual_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ):
        """
        Compute context-modulated emission PDF for HSMM using unified delta logic.

        Args:
            X: Optional observation tensor [B, T, F] or [T, F].
            theta: Optional contextual embedding [B, H], [B, T, H], or [H].
            scale: Optional scaling for contextual delta.

        Returns:
            torch.distributions.Distribution: Context-modulated emission PDF.
        """
        device, dtype = self.device, DTYPE
        pdf = self.pdf
        if pdf is None:
            return None

        scale = scale or getattr(self.config, "emission_scale", 1.0)
        K, F = self.n_states, self.n_features

        # --- 1. Compute delta using same logic as initialize_emissions ---
        combined_ctx = theta if theta is not None else self.combine_context(None)
        delta_raw = None
        if getattr(self, "ctx_emission", None) and combined_ctx is not None:
            delta_raw = self.ctx_emission(combined_ctx)

        # Target batch size
        B = X.shape[0] if X is not None and X.ndim == 3 else 1
        delta = self._prepare_delta(delta_raw, shape=(B, K, F), scale=scale)  # [B, K, F]

        # --- 2. Gaussian emission ---
        if isinstance(pdf, MultivariateNormal):
            mean = pdf.mean.to(device=device, dtype=dtype)
            cov = pdf.covariance_matrix.to(device=device, dtype=dtype)

            # Align shapes
            if mean.ndim == 2:  # [K, F]
                mean = mean.unsqueeze(0)
            if mean.shape[0] != delta.shape[0]:
                mean = mean.expand(delta.shape[0], -1, -1)
            mean = mean + delta

            # Covariance
            F_cov = cov.shape[-1]
            eps_eye = EPS * torch.eye(F_cov, device=device, dtype=dtype)
            if cov.ndim == 2:  # [F, F]
                cov = cov + eps_eye
                cov = cov.unsqueeze(0).unsqueeze(0).expand(B, K, F_cov, F_cov)
            elif cov.ndim == 3:  # [K, F, F]
                cov = cov.unsqueeze(0).expand(B, -1, -1, -1) + eps_eye.view(1, 1, F_cov, F_cov)
            elif cov.ndim == 4:  # [B, K, F, F]
                if cov.shape[0] != B and cov.shape[0] == 1:
                    cov = cov.expand(B, -1, -1, -1)
                cov = cov + eps_eye.view(1, 1, F_cov, F_cov)

            cov = torch.nan_to_num(cov, nan=0.0, posinf=1.0, neginf=1e-6)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        # --- 3. Categorical / Bernoulli emission ---
        elif isinstance(pdf, Categorical):
            logits = pdf.logits.to(device=device, dtype=dtype)
            if logits.ndim == 1:  # [F]
                logits = logits.unsqueeze(0).unsqueeze(0)
            elif logits.ndim == 2:  # [K, F]
                logits = logits.unsqueeze(0)

            if logits.shape[0] != delta.shape[0]:
                logits = logits.expand(delta.shape[0], -1, -1)

            logits = logits + delta.mean(dim=-1, keepdim=True)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            return Categorical(logits=logits)

        # --- 4. Fallback ---
        return pdf

    def _contextual_duration_pdf(
        self,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        temperature: Optional[float] = None
    ):
        """Return context-modulated duration distribution (softmax over max_duration)."""
        scale = scale or getattr(self.config, "duration_scale", 0.1)
        temperature = max(temperature or getattr(self.config, "duration_temp", 1.0), 1e-6)

        # Base logits from duration module
        logits = getattr(self.duration_module, "logits", None)
        if logits is None:
            # If duration module directly returns probabilities
            return self.duration_module.forward(log=False)

        # Context modulation
        combined = self.combine_context(theta)
        delta = None
        if combined is not None and getattr(self, "ctx_duration", None):
            delta = self.ctx_duration(combined)
        delta = self._prepare_delta(delta, shape=(1, self.n_states, self.max_duration), scale=scale)

        # Align logits batch with delta
        logits = self._align_logits(logits, delta.shape[0])
        combined_logits = logits + delta

        # Softmax with temperature and safe numeric handling
        return F.softmax(torch.nan_to_num(combined_logits) / temperature, dim=-1)

    def _contextual_transition_matrix(
        self,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        temperature: Optional[float] = None
    ):
        """Return context-modulated transition matrix (softmax over states)."""
        scale = scale or getattr(self.config, "transition_scale", 1.0)
        temperature = max(temperature or getattr(self.config, "transition_temp", 1.0), 1e-6)

        # Base logits from transition module
        logits = getattr(self.transition_module, "logits", None)
        if logits is None:
            # If transition module directly returns probabilities
            return self.transition_module.forward(log=False)

        # Context modulation
        combined = self.combine_context(theta)
        delta = None
        if combined is not None and getattr(self, "ctx_transition", None):
            delta = self.ctx_transition(combined)
        delta = self._prepare_delta(delta, shape=(1, self.n_states, self.n_states), scale=scale)

        # Align batch dimension
        logits = self._align_logits(logits, delta.shape[0])
        combined_logits = logits + delta

        return F.softmax(torch.nan_to_num(combined_logits) / temperature, dim=-1)

    def forward(
        self,
        X: torch.Tensor,
        return_pdf: bool = False,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None
    ):
        """
        Encode observations, merge optional context, and optionally return
        a contextually-modulated emission PDF.

        Args:
            X: Observation tensor [T, F] or [B, T, F].
            return_pdf: If True, return emission PDF instead of encoded theta.
            context: Optional raw context tensor.
            context_ids: Optional categorical context IDs for embedding.

        Returns:
            Theta tensor [B, latent_dim] or context-modulated emission PDF.
        """
        prev_ctx = getattr(self, "_context", None)
        ctx_list = []

        # --- Context embedding from IDs ---
        if context_ids is not None:
            if self.context_embedding is None:
                raise ValueError("context_ids provided but no context_embedding defined.")
            ctx_emb = self.context_embedding(context_ids.to(self.device))
            if ctx_emb.ndim == 3:  # [B, L, H] -> mean over L
                ctx_emb = ctx_emb.mean(dim=1)
            ctx_list.append(ctx_emb.to(dtype=DTYPE, device=self.device))

        # --- Raw context tensor ---
        if context is not None:
            ctx_list.append(context.to(dtype=DTYPE, device=self.device))

        # --- Merge context and set internal state ---
        merged_ctx = torch.cat(ctx_list, dim=-1) if ctx_list else None
        if merged_ctx is not None:
            self.set_context(merged_ctx)

        # --- Encode observations ---
        theta = self.encode_observations(X)

        # --- Return context-modulated emission PDF if requested ---
        if return_pdf:
            pdf_mod = self._contextual_emission_pdf(X=X, theta=theta)
            self._context = prev_ctx
            return pdf_mod

        # --- Return encoded latent theta ---
        self._context = prev_ctx
        return theta

    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> list[torch.Tensor]:
        """
        Context-aware HSMM prediction with dynamic PDFs and safe batching.

        Args:
            X: Observation tensor [T, F] or [B, T, F].
            lengths: Optional list of sequence lengths for batched sequences.
            algorithm: "map" or "viterbi".
            context: Optional raw context tensor.
            context_ids: Optional categorical context IDs for embedding.
            batch_size: Prediction batch size (unused if X fits in memory).

        Returns:
            List[torch.Tensor]: Predictions for each batch.
        """
        with torch.no_grad():
            # ---------------- Device + dtype ---------------- #
            if not torch.is_tensor(X):
                X = torch.as_tensor(X, dtype=DTYPE)
            X = X.to(dtype=DTYPE, device=self.device)
            B = X.shape[0] if X.ndim == 3 else 1

            # ---------------- Save previous context ---------------- #
            prev_ctx = getattr(self, "_context", None)
            prev_pdf = self._params.get("emission_pdf", None)
            prev_transition = getattr(self, "_transition_logits", None)
            prev_duration = getattr(self, "_duration_logits", None)

            # ---------------- Build context ---------------- #
            ctx_list = []

            if context_ids is not None:
                if self.context_embedding is None:
                    raise ValueError("context_ids provided but no context_embedding defined.")
                ctx_emb = self.context_embedding(context_ids.to(self.device))
                if ctx_emb.ndim == 3:
                    ctx_emb = ctx_emb.mean(dim=1)
                ctx_list.append(ctx_emb.to(dtype=DTYPE, device=self.device))

            if context is not None:
                ctx_list.append(context.to(dtype=DTYPE, device=self.device))

            ctx = torch.cat(ctx_list, dim=-1) if ctx_list else None
            if ctx is not None:
                self.set_context(ctx)

            # ---------------- Encode observations ---------------- #
            theta = self.encode_observations(X)

            # ---------------- Contextual PDFs ---------------- #
            try:
                pdf = self._contextual_emission_pdf(X, theta)
            except Exception as e:
                print(f"[predict] emission PDF creation failed: {e}")
                pdf = prev_pdf

            transition_logits = self._contextual_transition_matrix(theta)
            duration_logits = self._contextual_duration_pdf(theta)

            # ---------------- Ensure batch alignment ---------------- #
            def expand_to_batch(logits: torch.Tensor, target_B: int):
                if logits.ndim == 2:
                    return logits.unsqueeze(0).expand(target_B, *logits.shape)
                elif logits.ndim == 3:
                    if logits.shape[0] == 1:
                        return logits.expand(target_B, *logits.shape[1:])
                    elif logits.shape[0] == target_B:
                        return logits
                return logits

            transition_logits = expand_to_batch(transition_logits, B)
            duration_logits = expand_to_batch(duration_logits, B)

            # ---------------- Backup & assign dynamic parameters ---------------- #
            if pdf is not None:
                self._params["emission_pdf"] = pdf
                if hasattr(pdf, "dof"):
                    self._dynamic_dof = pdf.dof
            self._transition_logits = transition_logits
            self._duration_logits = duration_logits

            # ---------------- Base HSMM prediction ---------------- #
            preds = super().predict(X, lengths=lengths, algorithm=algorithm, batch_size=batch_size)

            # ---------------- Restore previous parameters ---------------- #
            self._params["emission_pdf"] = prev_pdf
            if prev_transition is not None:
                self._transition_logits = prev_transition
            elif hasattr(self, "_transition_logits"):
                del self._transition_logits

            if prev_duration is not None:
                self._duration_logits = prev_duration
            elif hasattr(self, "_duration_logits"):
                del self._duration_logits

            self._context = prev_ctx

            # ---------------- Ensure List[Tensor] ---------------- #
            if isinstance(preds, torch.Tensor):
                return [preds]
            return [torch.as_tensor(p, device=self.device) if not torch.is_tensor(p) else p for p in preds]

    def decode(
        self,
        X: torch.Tensor,
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        algorithm: Literal["viterbi", "map"] = "viterbi",
    ) -> np.ndarray:
        """
        Context-aware, batch-safe HSMM decoding.

        Args:
            X: Observations [T, F] or [B, T, F].
            duration_weight: Scaling for duration logits.
            context: Optional context tensor.
            context_ids: Optional context IDs for embedding.
            algorithm: "viterbi" or "map".

        Returns:
            Numpy array of predicted state sequences.
            - Single-sequence input -> shape [T]
            - Batched input -> shape [B, T]
        """
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE, device=self.device)
        else:
            X = X.to(dtype=DTYPE, device=self.device)

        # --- Encode observations and merge context ---
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # --- Backup original parameters ---
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        try:
            # --- Assign context-aware parameters ---
            pdf = self._contextual_emission_pdf(X, theta)
            self._params["emission_pdf"] = pdf
            if hasattr(pdf, "dof"):
                self._dynamic_dof = pdf.dof

            transition = self._contextual_transition_matrix(theta)
            batch_size = theta.shape[0] if theta.ndim == 2 else 1
            self._transition_logits = self._align_logits(transition, B=batch_size)

            duration = self._contextual_duration_pdf(theta, scale=duration_weight)
            self._duration_logits = self._align_logits(duration, B=batch_size)

            # --- Run base HSMM prediction ---
            preds_list = self.predict(X, algorithm=algorithm)

        finally:
            # --- Restore original parameters safely ---
            if prev_pdf is not None:
                self._params["emission_pdf"] = prev_pdf
            else:
                self._params.pop("emission_pdf", None)

            if prev_transition is not None:
                self._transition_logits = prev_transition
            elif hasattr(self, "_transition_logits"):
                del self._transition_logits

            if prev_duration is not None:
                self._duration_logits = prev_duration
            elif hasattr(self, "_duration_logits"):
                del self._duration_logits

        # --- Flatten / concatenate batch properly ---
        out_list = []
        for p in preds_list:
            if torch.is_tensor(p):
                out_list.append(p.detach().cpu())
            else:
                out_list.append(torch.as_tensor(p, device="cpu", dtype=torch.long))

        out = torch.cat(out_list, dim=0)

        # --- Squeeze singleton batch dimension if single sequence ---
        if out.ndim > 1 and out.shape[0] == 1:
            out = out.squeeze(0)

        return out.numpy()

    def tune(
        self,
        X: torch.Tensor,
        lengths: list[int] = None,
        configs: list[dict] = None,
        score_metric: str = "log_likelihood",
        verbose: bool = False
    ) -> dict:
        """
        GPU-batched hyperparameter tuning using `Observations` for vectorized handling of sequences,
        log_probs, and optional context. Supports dynamic assignment of parameters and safe scoring.
        """

        if not configs:
            raise ValueError("configs must be provided as a list of dicts.")

        # --- convert to device/dtype ---
        X = X.to(device=self.device, dtype=DTYPE)

        # --- build Observations ---
        if hasattr(self, "to_observations"):
            obs_X = self.to_observations(X, lengths)
        else:
            obs_X = Observations(sequence=[X] if X.ndim == 2 else [x for x in X], lengths=lengths)

        B = obs_X.n_sequences
        n_configs = len(configs)
        K, Dmax = self.n_states, self.max_duration
        scores: dict[int, float] = {}

        # --- prepare parameter stack ---
        param_keys = set().union(*[cfg.keys() for cfg in configs])
        param_stack = {k: [] for k in param_keys if k != "encoder_params"}

        for i, cfg in enumerate(configs):
            for k in list(param_stack.keys()):
                try:
                    val = cfg.get(k, getattr(self, k, None) or self._params.get.get(k, None))
                    if val is None:
                        if verbose:
                            print(f"[Config {i}] skipping field '{k}' (unhandled)")
                        continue
                    val = torch.as_tensor(val, device=self.device, dtype=DTYPE) if not isinstance(val, torch.Tensor) else val.to(device=self.device, dtype=DTYPE)
                    param_stack[k].append(val)
                except Exception as e:
                    if verbose:
                        print(f"[Config {i}] skipping field '{k}' due to: {e}")
                    continue

        # --- stack tensors ---
        for k in list(param_stack.keys()):
            try:
                param_stack[k] = torch.stack(param_stack[k], dim=0).to(dtype=DTYPE, device=self.device)
            except Exception as e:
                if verbose:
                    print(f"[tune] skipping field '{k}' (stacking failed: {e})")
                param_stack.pop(k, None)

        # --- normalize logits ---
        if "transition_logits" in param_stack:
            vals = param_stack["transition_logits"]
            if vals.ndim == 3 and vals.shape[1:] == (K, K):
                param_stack["transition_logits"] = vals - torch.logsumexp(vals, dim=-1, keepdim=True)
        if "duration_logits" in param_stack:
            vals = param_stack["duration_logits"]
            if vals.ndim == 3 and vals.shape[1:] == (K, Dmax):
                param_stack["duration_logits"] = vals - torch.logsumexp(vals, dim=-1, keepdim=True)

        # --- emission PDF check ---
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized for batched tuning.")

        # --- prepare batched tensor for vectorized scoring ---
        seq_batch, mask, _, _ = obs_X.to_batch(return_mask=True)
        try:
            lp_base = pdf.log_prob(seq_batch.unsqueeze(1)).to(dtype=DTYPE)
            if lp_base.ndim > 2:
                lp_base = lp_base.sum(dim=list(range(2, lp_base.ndim)))
            all_log_probs = lp_base.unsqueeze(0).repeat(n_configs, 1, 1)
        except Exception as e:
            raise RuntimeError(f"Emission log-prob computation failed: {e}")

        # --- assign parameters and compute scores ---
        with torch.no_grad():
            for i in range(n_configs):
                last_key, last_val = None, None
                try:
                    for k, vals in param_stack.items():
                        last_key, last_val = k, vals[i]
                        val = last_val
                        if hasattr(self, k):
                            try:
                                setattr(self, k, val)
                            except:
                                if verbose:
                                    print(f"[Config {i}] skipping field '{k}' (cannot set attribute)")
                        elif k in getattr(self, "_params", {}):
                            self._params[k] = val
                        else:
                            if verbose:
                                print(f"[Config {i}] skipping field '{k}' (unknown)")

                    # assign log_probs for current configuration
                    obs_X = obs_X.clone()
                    obs_X.log_probs = [all_log_probs[i, sum(obs_X.lengths[:j]):sum(obs_X.lengths[:j+1])] for j in range(B)]

                    score_tensor = self._compute_log_likelihood(obs_X)
                    score = float(torch.nan_to_num(score_tensor, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")).sum().item())
                    scores[i] = score

                    if verbose:
                        print(f"[Config {i}] score={score:.4f}")

                except Exception as e:
                    print(f"[Config {i}] Internal tuning failed in field '{last_key or 'unknown'}': {e}")
                    if last_val is not None:
                        try:
                            print(f"  param '{last_key}' shape={tuple(last_val.shape)}, dtype={last_val.dtype}, device={last_val.device}")
                        except:
                            pass
                    scores[i] = float('-inf')

        return scores
