# nhsmm/defaults.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli, Independent

import logging
from typing import Optional, Union, Literal, Tuple, Dict, Any


EPS = 1e-12
MAX_LOGITS = 50.0
DTYPE = torch.float32

# ---------------- Logger ----------------
logger = logging.getLogger("nhsmm")
if not logger.hasHandlers():
    # logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class HSMMError(ValueError):
    """Custom error class for HSMM module."""
    pass


class ContextEncoder(nn.Module):
    """
    Encoder wrapper for sequence models (HSMM/HMM/CRF/etc.).
    Supports variable-length sequences, pooling, attention, optional dropout & layer norm.
    Caches context in self._context and supports optional debug logging.
    """
    def __init__(
        self,
        encoder: nn.Module,
        batch_first: bool = True,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        device: Optional[torch.device] = None,
        n_heads: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        layer_norm: bool = True,
        debug: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.batch_first = batch_first
        self.pool = pool.lower().strip()
        self.device = device or next(encoder.parameters(), torch.tensor(0.0)).device
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self._context: Optional[torch.Tensor] = None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm_flag = layer_norm
        self.debug = debug

        # Attention params
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        self.encoder.to(device=self.device, dtype=DTYPE)
        self._dbg(f"ContextEncoder initialized: pool={self.pool}, device={self.device}")

    # ---------------- Debug helper ----------------
    def _dbg(self, *args):
        if self.debug:
            logger.debug(*args)

    # ---------------- Masking ----------------
    def _make_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor], pad_value: float = 0.0) -> Optional[torch.Tensor]:
        if lengths is None:
            return (x.sum(dim=-1) != pad_value) if x.ndim == 3 else None
        lengths = lengths.view(-1)
        max_T = x.shape[1] if self.batch_first else x.shape[0]
        idx = torch.arange(max_T, device=self.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        self._dbg(f"Mask created: {mask.shape}")
        return mask

    # ---------------- Forward ----------------
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:

        self._dbg(f"Forward input shape: {x.shape}")
        x = x.to(self.device, DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0 if self.batch_first else 1)
        elif x.ndim != 3:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}, expected (T,F) or (B,T,F)")

        if not self.batch_first:
            x = x.transpose(0, 1)

        if mask is None and lengths is not None:
            mask = self._make_mask(x, lengths)
        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)

        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Encoder must return a tensor, got {type(out)}")

        # Ensure (B,T,F)
        if out.ndim == 1:
            theta = out.unsqueeze(0).unsqueeze(1)
        elif out.ndim == 2:
            theta = out.unsqueeze(1) if out.shape[0] == x.shape[0] else out.unsqueeze(0)
        else:
            theta = out

        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        context, attn_weights = self._pool_context(theta, mask, return_attn_weights)
        if self.layer_norm_flag:
            context = F.layer_norm(context, context.shape[-1:])
        context = self.dropout(torch.clamp(context, -10.0, 10.0))
        self._context = context.detach()
        self._dbg(f"Forward output shape: theta={theta.shape}, context={context.shape}")

        if return_context:
            return (theta, context, attn_weights) if return_attn_weights else (theta, context)
        return theta

    # ---------------- Pooling ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        attn_weights = None
        if self.pool == "last":
            if mask is not None:
                idx = mask.sum(dim=1).clamp_min(1) - 1
                context = theta.gather(1, idx.unsqueeze(-1).expand(-1, -1, theta.shape[-1])).squeeze(1)
            else:
                context = theta[:, -1]
        elif self.pool == "mean":
            if mask is not None:
                context = (theta * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            else:
                context = theta.mean(dim=1)
        elif self.pool == "max":
            masked = theta.masked_fill(~mask.unsqueeze(-1), float('-inf')) if mask is not None else theta
            context = masked.max(dim=1).values
        elif self.pool == "attn":
            context, attn_weights = self._attention_context(theta, mask, return_attn=True)
        elif self.pool == "mha":
            context, attn_weights = self._multihead_context(theta, mask, return_attn=True)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pool}")

        self._dbg(f"Pooling done: context={context.shape}, attn_weights={None if attn_weights is None else attn_weights.shape}")
        return (context, attn_weights) if return_attn else (context, None)

    # ---------------- Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn=False):
        hidden = theta.shape[-1]
        if self._attn_vector is None or self._attn_vector.shape[-1] != hidden:
            self._attn_vector = nn.Parameter(torch.randn(hidden, device=self.device, dtype=DTYPE))
        attn_scores = theta @ self._attn_vector
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (attn_weights * theta).sum(dim=1)
        self._dbg(f"Attention computed: context={context.shape}, attn_weights={attn_weights.shape}")
        return (context, attn_weights) if return_attn else context

    # ---------------- Multihead Attention ----------------
    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn=False):
        B, T, d_model = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.n_heads,
                                              batch_first=True, device=self.device, dtype=DTYPE)
        attn_mask = ~mask if mask is not None else None
        attn_output, attn_weights = self._mha(theta, theta, theta, key_padding_mask=attn_mask)
        context = attn_output.mean(dim=1)
        self._dbg(f"Multihead attention: context={context.shape}, attn_weights={attn_weights.shape}")
        return (context, attn_weights) if return_attn else context

    # ---------------- Context Utilities ----------------
    def set_context(self, context: Optional[torch.Tensor], batch_size: Optional[int] = None):
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(device=self.device, dtype=DTYPE)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        if batch_size is not None and ctx.shape[0] == 1:
            ctx = ctx.expand(batch_size, -1)
        self._context = ctx
        self._dbg(f"Context set: {ctx.shape}")

    def reset_context(self):
        self._context = None
        self._dbg("Context reset")

    def get_context(self) -> Optional[torch.Tensor]:
        return self._context

    # ---------------- Vectorized delta ----------------
    def _prepare_delta(self, delta: Optional[torch.Tensor], shape: Tuple[int,int,int], scale: float = 0.1, broadcast: bool = True) -> torch.Tensor:
        B_target, n_states, F = shape
        device, dtype = self.device, DTYPE
        if delta is None:
            delta_tensor = torch.zeros((B_target, n_states, F), dtype=dtype, device=device)
            self._dbg(f"Delta prepared: {delta_tensor.shape}")
            return delta_tensor
        delta = delta.to(device=device, dtype=dtype)
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        elif delta.ndim > 2:
            delta = delta.view(delta.shape[0], -1)
        B, total_features = delta.shape
        expected_features = n_states * F
        if total_features < expected_features:
            delta = F.pad(delta, (0, expected_features - total_features))
        elif total_features > expected_features:
            delta = delta[..., :expected_features]
        delta = delta.view(B, n_states, F)
        delta = scale * torch.tanh(torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0))
        if broadcast and B != B_target:
            delta = delta.expand(B_target, -1, -1) if B == 1 else delta[:B_target]
        self._dbg(f"Delta prepared: {delta.shape}")
        return delta

    # ---------------- Vectorized combine ----------------
    def _combine_context(self, theta: Optional[torch.Tensor], allow_broadcast: bool = True) -> Optional[torch.Tensor]:
        ctx = self._context
        if theta is None and ctx is None:
            return None
        if theta is None:
            return ctx
        if ctx is None:
            return theta
        theta = theta.to(self.device, DTYPE)
        ctx = ctx.to(self.device, DTYPE)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        B_theta, B_ctx = theta.shape[0], ctx.shape[0]
        if B_theta != B_ctx:
            if allow_broadcast:
                if B_theta == 1:
                    theta = theta.expand(B_ctx, *theta.shape[1:])
                elif B_ctx == 1:
                    ctx = ctx.expand(B_theta, *ctx.shape[1:])
                else:
                    raise ValueError(f"Batch mismatch: theta {theta.shape} vs ctx {ctx.shape}")
            else:
                raise ValueError(f"Batch mismatch and broadcasting disabled: theta {theta.shape} vs ctx {ctx.shape}")
        if theta.shape[1:-1] != ctx.shape[1:-1]:
            raise ValueError(f"Sequence dimension mismatch: theta {theta.shape} vs ctx {ctx.shape}")
        combined = torch.cat([theta, ctx.expand(theta.shape[0], *ctx.shape[1:])], dim=-1)
        self._dbg(f"Context combined: {combined.shape}")
        return combined

    def get_config(self):
        return dict(
            pool=self.pool,
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0,
            batch_first=self.batch_first,
            device=str(self.device)
        )


class Contextual(nn.Module):
    """Base class for context-modulated parameters with optional aggregation and debug logging."""

    def __init__(
        self,
        context_dim: Optional[int],
        target_dim: int,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        hidden_dim: Optional[int] = None,
        activation: str = "tanh",
        device: Optional[torch.device] = None,
        debug: bool = False
    ):
        super().__init__()
        if context_dim is not None and context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if target_dim <= 0:
            raise ValueError(f"target_dim must be positive, got {target_dim}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.aggregate_context = aggregate_context
        self.aggregate_method = aggregate_method.lower()
        self.debug = debug

        if context_dim:
            hidden_dim = hidden_dim or max(context_dim, target_dim)
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, device=self.device, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, device=self.device, dtype=DTYPE),
                self._get_activation(activation),
                nn.Linear(hidden_dim, target_dim, device=self.device, dtype=DTYPE)
            )
            self._init_weights()
        else:
            self.context_net = None

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name.lower()]

    def _init_weights(self):
        if self.context_net:
            for m in self.context_net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None or self.context_net is None:
            return None
        if not torch.is_tensor(context):
            raise ValueError(f"Context must be a torch.Tensor, got {type(context)}")
        context = context.to(self.device, DTYPE)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"Context last dimension must be {self.context_dim}, got {context.shape[-1]}")
        return context

    def _contextual_modulation(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1) -> torch.Tensor:
        base = base.to(self.device, DTYPE)
        if self.context_net is None or context is None:
            return torch.zeros_like(base)

        context = self._validate_context(context)
        delta = self.context_net(context)

        # Vectorized broadcasting to match base dimensions
        while delta.ndim < base.ndim:
            delta = delta.unsqueeze(1)

        if self.aggregate_context:
            if self.aggregate_method == "mean":
                delta = delta.mean(dim=0, keepdim=True)
            elif self.aggregate_method == "sum":
                delta = delta.sum(dim=0, keepdim=True)
            elif self.aggregate_method == "max":
                delta, _ = delta.max(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unsupported aggregate_method: {self.aggregate_method}")

        delta = torch.tanh(delta) * scale

        if self.debug:
            logger.debug(f"_contextual_modulation: base.shape={base.shape}, delta.shape={delta.shape}, scale={scale}")

        return delta

    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1) -> torch.Tensor:
        modulated = base.to(self.device, DTYPE) + self._contextual_modulation(base, context, scale)
        modulated = torch.where(torch.isfinite(modulated), modulated, base)
        if self.debug:
            logger.debug(f"_apply_context: base.shape={base.shape}, modulated.shape={modulated.shape}")
        return modulated

    def get_config(self) -> Dict[str, Any]:
        return dict(
            context_dim=self.context_dim,
            target_dim=self.target_dim,
            aggregate_context=self.aggregate_context,
            aggregate_method=self.aggregate_method,
            device=str(self.device),
            dtype=str(DTYPE),
            debug=self.debug
        )

    def to(self, device, **kwargs):
        self.device = torch.device(device)
        super().to(device, **kwargs)
        if self.context_net:
            self.context_net.to(device, **kwargs)
        return self


class Emission(Contextual):
    """
    Contextual emission distribution supporting Gaussian, Categorical, or Bernoulli.
    Handles per-sequence or per-time-step context embeddings from ContextEncoder.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-3,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        emission_type: str = "gaussian",
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        device: Optional[torch.device] = None,
        scale: float = 0.1,
        modulate_var: bool = False,
        debug: bool = False
    ):
        super().__init__(
            context_dim=context_dim,
            target_dim=n_states * n_features,
            aggregate_context=aggregate_context,
            aggregate_method=aggregate_method,
            hidden_dim=hidden_dim,
            activation="tanh",
            device=device,
        )

        self.n_states = n_states
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.min_covar = min_covar
        self.scale = scale
        self.modulate_var = modulate_var
        self.emission_type = emission_type.lower()
        self.debug = debug

        self._init_parameters()
        if self.debug:
            logger.debug(f"Initialized Emission: {self.get_config()}")

    # ----------------------------------------------------------------------
    # Parameter initialization
    # ----------------------------------------------------------------------
    def _init_parameters(self):
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(
                torch.randn(self.n_states, self.n_features, device=self.device, dtype=DTYPE) * 0.1
            )
            self.log_var = nn.Parameter(
                torch.full((self.n_states, self.n_features), -1.0, device=self.device, dtype=DTYPE)
            )
        elif self.emission_type in ["categorical", "bernoulli"]:
            self.logits = nn.Parameter(
                torch.zeros(self.n_states, self.n_features, device=self.device, dtype=DTYPE)
            )
        else:
            raise HSMMError(f"Unsupported emission_type: {self.emission_type}")

    # ----------------------------------------------------------------------
    # Forward pass with context modulation
    # ----------------------------------------------------------------------
    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        if self.emission_type == "gaussian":
            mu = self._apply_context(self.mu, context, self.scale)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                var = self._apply_context(var, context, self.scale).abs() + self.min_covar
            if self.debug:
                logger.debug(f"Gaussian forward: mu.shape={mu.shape}, var.shape={var.shape}")
            if return_dist:
                return Independent(Normal(mu, var.sqrt()), 1)
            return mu, var

        logits = self._apply_context(self.logits, context, self.scale)
        if self.debug:
            logger.debug(f"{self.emission_type.capitalize()} forward: logits.shape={logits.shape}")
        if return_dist:
            if self.emission_type == "categorical":
                return Categorical(logits=logits)
            return Independent(Bernoulli(logits=logits), 1)
        return logits

    # ----------------------------------------------------------------------
    # Log-probability computation
    # ----------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        x = x.to(self.device, DTYPE)

        # Ensure compatible shape for event dimension
        if self.emission_type == "gaussian":
            if x.ndim == 2:  # [T, F] -> [1, T, F]
                x = x.unsqueeze(0)
            x = x.unsqueeze(2) if x.ndim == 3 else x
        else:
            if x.ndim == 2:  # [B, F] -> [B, 1, F]
                x = x.unsqueeze(1)
            elif x.ndim == 1:  # [F] -> [1, 1, F]
                x = x.unsqueeze(0).unsqueeze(0)

        logp = dist.log_prob(x)
        logp = logp.to(self.device, DTYPE)

        # Ensure output shape [B, T, K]
        if logp.ndim < 3:
            logp = logp.unsqueeze(0).unsqueeze(-1)
        elif logp.ndim == 3 and logp.shape[-1] != self.n_states:
            logp = logp.transpose(-1, -2)

        if self.debug:
            logger.debug(f"log_prob: x.shape={x.shape}, logp.shape={logp.shape}")
        return logp

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------
    def sample(
        self,
        n_samples: int = 1,
        context: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        samples = dist.rsample((n_samples,)) if self.emission_type == "gaussian" else dist.sample((n_samples,))
        # Shape: [n_samples, K, F] for Gaussian or categorical
        if self.debug:
            logger.debug(f"Sampled raw: samples.shape={samples.shape}")

        if samples.ndim == 3:
            samples = samples.permute(1, 0, 2)  # [K, n_samples, F]

        if state_indices is not None:
            samples = samples[state_indices]

        if self.debug:
            logger.debug(f"Sampled final: samples.shape={samples.shape}")
        return samples.to(self.device, DTYPE)

    # ----------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            n_features=self.n_features,
            min_covar=self.min_covar,
            scale=self.scale,
            modulate_var=self.modulate_var,
            emission_type=self.emission_type,
            hidden_dim=self.hidden_dim,
            aggregate_context=self.aggregate_context,
            aggregate_method=self.aggregate_method,
        ))
        return cfg


class Duration(Contextual):
    """Contextual duration distribution (categorical over discrete durations)."""

    def __init__(
        self,
        n_states: int,
        max_duration: int = 20,
        context_dim: Optional[int] = None,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        scale: float = 0.1,
        debug: bool = False
    ):
        target_dim = n_states * max_duration
        super().__init__(
            context_dim, target_dim, aggregate_context, aggregate_method,
            hidden_dim, "tanh", device
        )
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.debug = debug
        self.logits = nn.Parameter(
            torch.randn(n_states, max_duration, device=self.device, dtype=DTYPE) * 0.1
        )
        if self.debug:
            logger.debug(f"Duration initialized: n_states={n_states}, max_duration={max_duration}")

    # ---------------- Forward ----------------
    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        """
        Returns:
            - If return_dist=True: torch.distributions.Categorical
            - Else: [B, n_states, max_duration] probability or log-probability
        """
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)

        if self.debug:
            logger.debug(f"Forward logits shape: {logits.shape}, temperature={self.temperature}")

        if return_dist:
            return Categorical(logits=logits)

        probs = F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)
        if self.debug:
            logger.debug(f"Forward output shape: {probs.shape}")
        return probs

    # ---------------- Expected duration ----------------
    def expected_duration(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, return_dist=False)
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)
        expected = (probs * durations).sum(dim=-1)
        if self.debug:
            logger.debug(f"Expected duration shape: {expected.shape}")
        return expected

    # ---------------- Sampling ----------------
    def sample(
        self,
        n_samples: int = 1,
        context: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,))  # [n_samples, n_states]
        samples = samples.transpose(0, 1)  # [n_states, n_samples]

        if state_indices is not None:
            samples = samples[state_indices, ...]

        if self.debug:
            logger.debug(f"Sampled durations shape: {samples.shape}")
        return samples

    # ---------------- Configuration ----------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            max_duration=self.max_duration,
            temperature=self.temperature,
            scale=self.scale
        ))
        return cfg


class Transition(Contextual):
    """Contextual transition distribution with optional debug logging."""
    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        scale: float = 0.1,
        debug: bool = False
    ):
        target_dim = n_states * n_states
        super().__init__(context_dim, target_dim, aggregate_context, aggregate_method, hidden_dim, "tanh", device)
        self.logits = nn.Parameter(torch.randn(n_states, n_states, device=self.device, dtype=DTYPE) * 0.1)
        self.temperature = max(temperature, 1e-6)
        self.n_states = n_states
        self.scale = scale
        self.debug = debug
        if self.debug:
            logger.debug(f"Transition initialized: n_states={n_states}, temperature={self.temperature}, scale={scale}")

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if self.debug:
            logger.debug(f"Forward logits: {logits.shape}, min={logits.min().item()}, max={logits.max().item()}")
        if return_dist:
            dist = Categorical(logits=logits)
            if self.debug:
                logger.debug(f"Returning Categorical distribution with logits: {logits.shape}")
            return dist
        probs = F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)
        if self.debug:
            logger.debug(f"Forward probs/log_probs: {probs.shape}")
        return probs

    def expected_transitions(self, context: Optional[torch.Tensor] = None):
        probs = self.forward(context=context, log=False, return_dist=False)
        if self.debug:
            logger.debug(f"Expected transitions computed: {probs.shape}")
        return probs

    def sample(
        self,
        n_samples: int = 1,
        context: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None
    ):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).transpose(0,1)
        if state_indices is not None:
            samples = samples[state_indices, ...]
        if self.debug:
            logger.debug(f"Sampled transitions: {samples.shape}")
        return samples

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            temperature=self.temperature,
            scale=self.scale
        ))
        return cfg
