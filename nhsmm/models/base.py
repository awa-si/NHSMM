# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.context import ContextEncoder
from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger, MAX_LOGITS
from nhsmm.tools import utils, constraints, SeedGenerator, ConvergenceTracker
from nhsmm.distributions import Categorical, Initial, Emission, Duration, Transition


class HSMM(nn.Module):
    """
    Hidden Semi-Markov Model (HSMM) base class.

    Supports variable-length sequences with optional context or neural modulation 
    of initial, duration, transition, and emission parameters. Provides GPU-compatible, 
    batched computations for forward, backward, and Viterbi decoding.

    Core Components:
        - `initial_module`: Log-probabilities of initial states.
        - `duration_module`: Log-duration probabilities per state.
        - `transition_module`: State-to-state transition log-probabilities.
        - `emission_module`: Observation likelihoods (parametric or neural).
        - Optional `ContextEncoder` for context-conditioned parameters.

    Key Methods:
        - `_forward(X, theta)`: Log-probabilities α[t,k,d] for ending in state `k` at time `t` with duration `d`.
        - `_backward(X, theta)`: Log-probabilities β[t,k,d] for observations from time `t` onward.
        - `_compute_state_posteriors(X, theta)`: Returns γ (state marginals), ξ (transitions), η (state-duration posteriors).
        - `_viterbi(X, theta, duration_weight)`: Most likely state sequence, optionally weighting durations.

    Notes:
        - Handles discrete (Categorical) and continuous (Normal/MultivariateNormal) emissions.
        - Fully batched and GPU-ready.
        - Extensible for custom neural or context-modulated emission modules.

    Attributes:
        n_states: Number of hidden states (K)
        max_duration: Maximum allowed state duration (Dmax)
        device: Torch device (CPU/GPU)
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        modulate_var: bool = False,
        alpha: Optional[float] = 1.0,
        emission_type: str = "gaussian",
        transition_type: Any = constraints.Transitions.ERGODIC,
        min_covar: Optional[float] = 1e-6,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,

        # HSMM encoder params
        encoder: Optional[nn.Module] = None,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        embed_dim: Optional[int] = None,
        precompute: bool = True,
        dropout: float = 0.0,
        debug: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed or SeedGenerator(seed).seed
        self.emission_type = emission_type
        self.max_duration = max_duration
        self.temperature = temperature
        self.n_features = n_features
        self.precompute = precompute
        self.n_states = n_states
        self.alpha = alpha
        self.debug = debug

        self._params: Dict[str, Any] = {}

        # ---------------- Seed ----------------
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # ---------------- Encoder setup ----------------
        self.encoder: Optional[ContextEncoder] = None
        self._context: Optional[torch.Tensor] = None
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim or self.context_dim

        # --- Wrap encoder if needed ---
        if encoder is not None:
            if isinstance(encoder, ContextEncoder):
                self.encoder = encoder
            else:
                self.encoder = ContextEncoder(
                    encoder=encoder,
                    pool=pool,
                    n_heads=n_heads,
                    dropout=dropout,
                    debug=debug,
                )

            # --- Infer context/output dimension ---
            if hasattr(self.encoder, "out_dim") and getattr(self.encoder, "out_dim") is not None:
                self.context_dim = self.encoder.out_dim
            else:
                # Dummy forward to safely infer dimension
                device = getattr(self, "device", torch.device("cpu"))
                self.encoder.eval()
                try:
                    dummy_seq_len = 16
                    dummy_in = torch.zeros(
                        1, dummy_seq_len, n_features,
                        device=device, dtype=DTYPE
                    )
                    # Some encoders may not support return_context/return_sequence
                    try:
                        dummy_out, ctx, _ = self.encoder(dummy_in, return_context=True, return_sequence=True)
                    except TypeError:
                        dummy_out = self.encoder(dummy_in)
                        ctx = None
                    self.context_dim = ctx.shape[-1] if ctx is not None else dummy_out.shape[-1]
                finally:
                    # Restore training mode
                    if getattr(self.encoder, "training", True):
                        self.encoder.train()

            # --- Set hidden_dim consistently ---
            self.hidden_dim = self.context_dim if hidden_dim is None else hidden_dim

            if debug and logger:
                logger.debug(f"[Init] Inferred context_dim={self.context_dim}, hidden_dim={self.hidden_dim}")

        # ---------------- Initialize modules ----------------
        self._init_modules(
            transition_type=transition_type,
            emission_type=emission_type,
            modulate_var=modulate_var,
            max_duration=max_duration,
            temperature=temperature,
            min_covar=min_covar,
            device=self.device,
        )

        self.to(self.device)

    def _init_modules(
        self,
        device: torch.device | str,
        dof: float = 5.0,
        n_states: int = 4,
        scale: float = 1.0,
        n_features: int = 1,
        max_duration: int = 30,
        temperature: float = 1.0,
        modulate_var: bool = False,
        adaptive_scale: bool = True,
        spatial_adapter: bool = False,
        temporal_adapter: bool = False,
        min_covar: Optional[float] = 1e-6,
        init_mode_transition: str = "diag_bias",
        init_mode_duration: str = "uniform",
        init_mode_initial: str = "uniform",
        init_mode_emission: str = "kmeans",
        transition_type: str = "ergodic",
        emission_type: str = "gaussian",
        cache_limit: int = 32,
        debug: bool = False,
    ):
        """Initialize all HSMM modules with consistent device, context, and precomputation."""

        device, debug = self.device, self.debug

        # ---------------- Initial ----------------
        self.initial_module = Initial(
            n_states=self.n_states,
            init_mode=init_mode_initial,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            cache_limit=cache_limit,
            debug=debug
        ).to(device)

        # ---------------- Emission ----------------
        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            temporal_adapter=temporal_adapter,
            spatial_adapter=spatial_adapter,
            adaptive_scale=adaptive_scale,
            emission_type=emission_type,
            modulate_var=modulate_var,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            min_covar=min_covar,
            seed=self.seed,
            scale=scale,
            dof=dof,
        ).to(device)

        # ---------------- Duration ----------------
        self.duration_module = Duration(
            n_states=self.n_states,
            init_mode=init_mode_duration,
            max_duration=max_duration,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            cache_limit=cache_limit,
            debug=debug,
        ).to(device)

        # ---------------- Transition ----------------
        self.transition_module = Transition(
            n_states=self.n_states,
            init_mode=init_mode_transition,
            transition_type=transition_type,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            cache_limit=cache_limit,
            debug=debug,
        ).to(device)

        # ---------------- Optional Precompute ----------------
        # if self.precompute and self.context_dim is not None:
            # with torch.no_grad():
                # dummy_ctx = torch.zeros(1, self.context_dim, device=device)
                # self.initial_module(dummy_ctx)
                # self.duration_module(dummy_ctx)
                # self.transition_module(dummy_ctx)
                # self.emission_module(dummy_ctx)

        # ---------------- Initialize PDFs ----------------
        try:
            self._params.update({
                "initial_pdf": self.initial_module.initialize(mode=init_mode_initial),
                "duration_pdf": self.duration_module.initialize(mode=init_mode_duration),
                "transition_pdf": self.transition_module.initialize(mode=init_mode_transition),
                "emission_pdf": self.emission_module.initialize(mode=init_mode_emission),
            })
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {e}") from e

        # ---------------- Debug Logging ----------------
        if debug:
            logger.debug(
                f"HSMM modules initialized on {device}: "
                f"n_states={self.n_states}, n_features={self.n_features}, "
                f"context_dim={self.context_dim}, emission={self.emission_type}, "
                f"max_duration={self.max_duration}, precompute={self.precompute}"
            )

    def _ensure_dim(self, module, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Ensure that a distribution module returns a properly shaped log-matrix
        expanded to match batch and time dimensions. Adds debug logging instead of print.
        """
        if module is None:
            raise ValueError("_ensure_dim called with module=None")

        module_type = type(module).__name__
        if isinstance(module, (Initial, Transition, Duration, Emission, Categorical)) and getattr(self, "debug", False):
            logger.debug(f"[_ensure_dim] Processing module type: {module_type}")

        # ---------------- Prepare context ----------------
        ctx = context
        if ctx is not None:
            if ctx.ndim == 1:
                ctx = ctx[None, None, :]
            elif ctx.ndim == 2:
                ctx = ctx[None, :, :]
            if getattr(self, "debug", False):
                logger.debug(f"[_ensure_dim] Input context shape: {context.shape}, standardized to {ctx.shape}")

        # ---------------- Module forward ----------------
        x = module.log_matrix(context=ctx)
        if not torch.is_tensor(x):
            raise TypeError(f"log_matrix must return tensor, got {type(x)}")
        if getattr(self, "debug", False):
            logger.debug(f"[_ensure_dim] log_matrix output shape: {x.shape}")

        # ---------------- Match feature dims ----------------
        expected = getattr(module, "_shape", None)
        if expected is not None:
            feat_dims = expected if isinstance(expected, (list, tuple)) else [expected]
            while x.shape[-len(feat_dims):] != tuple(feat_dims):
                x = x.unsqueeze(0)
                if getattr(self, "debug", False):
                    logger.debug(f"[_ensure_dim] Unsqueezed to match feature dims: {x.shape}")

        # ---------------- Expand along batch/time if needed ----------------
        if ctx is not None:
            B, T = ctx.shape[0], ctx.shape[1]
            expand_shape = [B, T] + list(x.shape[2:])
            if x.shape[:2] == (1, 1):
                x = x.expand(*expand_shape)
            elif x.shape[0] == 1:
                x = x.expand(B, *x.shape[1:])
            elif x.shape[1] == 1:
                x = x.expand(x.shape[0], T, *x.shape[2:])
            if getattr(self, "debug", False):
                logger.debug(f"[_ensure_dim] Expanded along batch/time to: {x.shape}")

        return x

    def _encode_observations(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None,
        detach: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequences via the HSMM encoder (if any) and produce:
            - context_aligned: per-timestep context [B, T, H]
            - ctx_canonical: canonical context [B, 1, H]

        Handles batching, masking, pooling, and optional detachment.
        Strictly validates shapes and context_dim consistency.
        """
        if self.encoder is None or sequences.numel() == 0:
            self._sequence = None
            self._context = None
            self._ctx_canonical = None
            return None, None

        # Ensure [B,T,F]
        if sequences.ndim == 2:
            sequences = sequences.unsqueeze(0)
        B, T, F = sequences.shape
        device = sequences.device

        # Normalize mask → [B,T]
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        elif mask.ndim == 1:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 2:
            raise ValueError(f"mask must be [T] or [B,T], got {mask.shape}")

        # Temporary pool override
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            # Forward pass through encoder
            seq_out, ctx_canonical, _ = self.encoder(
                sequences,
                mask=mask,
                return_context=True,
                return_sequence=True,
                detach_context=detach
            )

            # ---------------- Canonical context ----------------
            if ctx_canonical is None:
                ctx_canonical = seq_out.mean(dim=1, keepdim=True)
            elif ctx_canonical.ndim == 2:
                ctx_canonical = ctx_canonical.unsqueeze(1)
            elif ctx_canonical.ndim == 3 and ctx_canonical.shape[1] != 1:
                ctx_canonical = ctx_canonical.mean(dim=1, keepdim=True)

            # ---------------- Per-timestep aligned context ----------------
            if seq_out.shape[1] == T:
                context_aligned = seq_out
            else:
                context_aligned = ctx_canonical.expand(-1, T, -1)

            # ---------------- Dimension verification ----------------
            if context_aligned.shape[-1] != self.context_dim:
                raise RuntimeError(
                    f"Encoder output dimension mismatch: expected {self.context_dim}, "
                    f"got {context_aligned.shape[-1]}"
                )

            if getattr(self, "debug", False) and logger:
                logger.debug(
                    f"[_encode_observations] sequences={sequences.shape}, "
                    f"seq_out={seq_out.shape}, mask={mask.shape}, "
                    f"ctx_canonical={ctx_canonical.shape}, context_aligned={context_aligned.shape}"
                )

        finally:
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        # Detach if requested
        if detach:
            seq_out = seq_out.detach()
            context_aligned = context_aligned.detach()
            ctx_canonical = ctx_canonical.detach()

        # Internal storage for forward pass
        self._sequence = seq_out
        self._context = context_aligned
        self._ctx_canonical = ctx_canonical

        return context_aligned, ctx_canonical

    def _prepare_observations(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> utils.SequenceSet:
        """
        Prepares a SequenceSet with fully batch-aligned tensors:
            - Computes canonical context (via encoder or provided theta)
            - Computes log-probs per timestep and state
            - Builds masks and lengths

        Returns:
            SequenceSet with:
                - sequences: list of [T,F] tensors
                - lengths: list of sequence lengths
                - log_probs: list of [T,K] tensors
                - contexts: [B,T,H] tensor
                - masks: [B,T,1] tensor
        """
        # Ensure batch dimension
        if X.ndim == 2:
            X = X.unsqueeze(0)
        B, T, F = X.shape
        device = X.device

        # Masks
        mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)

        # Context
        if theta is not None:
            context_aligned = theta
            if theta.ndim == 2:  # [T,H] -> [1,T,H]
                context_aligned = theta.unsqueeze(0)
            elif theta.shape[0] != B:  # expand if batch mismatch
                context_aligned = theta.expand(B, -1, -1)
            ctx_canonical = context_aligned[:, :1, :]
        else:
            context_aligned, ctx_canonical = self._encode_observations(X)
            if context_aligned is None:
                context_aligned = torch.zeros(B, T, self.context_dim, device=device)
                ctx_canonical = torch.zeros(B, 1, self.context_dim, device=device)

        # Emission parameters
        K = self.n_states
        dist_type = getattr(self.emission_module, "dist_type", None)

        if dist_type in (torch.distributions.Categorical, torch.distributions.Bernoulli):
            # Vectorized categorical log-probs
            emission_dist = self.emission_module.forward(context=context_aligned, return_dist=True)
            logits = emission_dist.logits  # [B,K,F]
            seq_cat = X[..., 0].long()     # [B,T]
            log_probs_all = F.log_softmax(logits, dim=-1)  # [B,K,F]

            idx = seq_cat.unsqueeze(1).expand(-1, K, -1).permute(0, 2, 1)  # [B,T,K]
            log_probs_list = torch.gather(
                log_probs_all.unsqueeze(1).expand(-1, T, -1, -1), -1, idx.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_list = [log_probs_list[b] for b in range(B)]

        else:
            # Gaussian / independent emissions fully vectorized
            means = self.emission_module._emission_means  # [K,F]
            covs = self.emission_module._emission_covs
            stds = covs.diagonal(dim1=-2, dim2=-1).sqrt()  # [K,F]

            # Expand for batch and time
            means_exp = means.unsqueeze(0).unsqueeze(0).expand(B, T, K, F)  # [B,T,K,F]
            stds_exp = stds.unsqueeze(0).unsqueeze(0).expand(B, T, K, F)    # [B,T,K,F]
            X_exp = X.unsqueeze(2).expand(B, T, K, F)                        # [B,T,K,F]

            log_probs = -0.5 * torch.log(2 * torch.pi * stds_exp**2)
            log_probs -= 0.5 * ((X_exp - means_exp)**2 / (stds_exp**2))
            log_probs = log_probs.sum(-1)  # [B,T,K]
            log_probs_list = [log_probs[b] for b in range(B)]

        # Build SequenceSet
        return utils.SequenceSet(
            sequences=[X[b] for b in range(B)],
            lengths=[T] * B,
            log_probs=log_probs_list,
            contexts=context_aligned,
            masks=mask
        )

    def _forward(
        self,
        X: utils.SequenceSet,
        theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None
    ) -> list[torch.Tensor]:
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        alpha_list = []

        for i, (log_emissions, seq_len) in enumerate(zip(X.log_probs, X.lengths)):
            device = log_emissions.device

            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, dtype=DTYPE, device=device))
                continue

            T = seq_len

            # ---------------- Per-sequence context ----------------
            ctx_seq = None
            if theta is not None:
                ctx_seq = theta[i] if isinstance(theta, list) else theta

            # ---------------- Module logits ----------------
            initial_logits = self._ensure_dim(self.initial_module, ctx_seq)
            duration_logits = self._ensure_dim(self.duration_module, ctx_seq)
            transition_logits = self._ensure_dim(self.transition_module, ctx_seq)

            initial_logits    = initial_logits.expand(1, T, K).squeeze(0)        # [T,K]
            duration_logits   = duration_logits.expand(1, T, K, Dmax).squeeze(0) # [T,K,Dmax]
            transition_logits = transition_logits.expand(1, T, K, K).squeeze(0)  # [T,K,K]

            # ---------------- Cumulative emission sums ----------------
            cumsum_emit = torch.zeros(T + 1, K, dtype=DTYPE, device=device)
            cumsum_emit[1:] = torch.cumsum(log_emissions, dim=0)

            dur_range = torch.arange(1, Dmax + 1, device=device).view(1, Dmax)  # [1,Dmax]
            time_idx = torch.arange(T, device=device).view(T, 1)               # [T,1]
            start_idx = (time_idx - dur_range + 1).clamp(min=0)                # [T,Dmax]

            emit_sums = torch.zeros(T, K, Dmax, dtype=DTYPE, device=device)
            for d in range(Dmax):
                start = start_idx[:, d]           # [T]
                end = time_idx[:, 0] + 1          # [T]
                emit_sums[:, :, d] = cumsum_emit[end, :] - cumsum_emit[start, :]

            # ---------------- Forward recursion ----------------
            alpha_tensor = torch.full((T, K, Dmax), neg_inf, dtype=DTYPE, device=device)

            for t in range(T):
                max_d = min(Dmax, t + 1)

                if t == 0:
                    alpha_tensor[t, :, :max_d] = (
                        initial_logits[t].unsqueeze(1)
                        + duration_logits[t, :, :max_d]
                        + emit_sums[t, :, :max_d]
                    )
                    continue

                prev_alpha = torch.full((max_d, K), neg_inf, dtype=DTYPE, device=device)
                valid_mask = start_idx[t, :max_d] > 0

                if valid_mask.any():
                    idx = valid_mask.nonzero(as_tuple=True)[0]
                    prev_vals = torch.logsumexp(alpha_tensor[start_idx[t, idx] - 1, :, :], dim=2)
                    prev_alpha[idx] = prev_vals

                if (~valid_mask).any():
                    idx = (~valid_mask).nonzero(as_tuple=True)[0]
                    prev_alpha[idx] = (
                        initial_logits[t].unsqueeze(0)
                        + duration_logits[t, :, :max_d][:, idx].T
                    )

                alpha_trans = torch.logsumexp(
                    prev_alpha.unsqueeze(2) + transition_logits[t].unsqueeze(0),
                    dim=1
                )

                alpha_tensor[t, :, :max_d] = (
                    alpha_trans.T
                    + duration_logits[t, :, :max_d]
                    + emit_sums[t, :, :max_d]
                )

            alpha_list.append(alpha_tensor)

        return alpha_list

    def _backward(
        self,
        X: utils.SequenceSet,
        theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None
    ) -> list[torch.Tensor]:
        """
        Vectorized backward pass for HSMM.
        Returns list of [T, K, Dmax] tensors per sequence.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        beta_list = []

        for i, (seq_logp, seq_len) in enumerate(zip(X.log_probs, X.lengths)):
            device = seq_logp.device

            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, dtype=DTYPE, device=device))
                continue

            T = seq_len

            # 1. Per-sequence context
            ctx_seq = theta[i] if isinstance(theta, list) and theta[i] is not None else theta

            # 2. Module logits
            init_logits  = self._ensure_dim(self.initial_module, ctx_seq).expand(1, T, K).squeeze(0)      # [T,K]
            dur_logits   = self._ensure_dim(self.duration_module, ctx_seq).expand(1, T, K, Dmax).squeeze(0) # [T,K,Dmax]
            trans_logits = self._ensure_dim(self.transition_module, ctx_seq).expand(1, T, K, K).squeeze(0) # [T,K,K]

            # 3. Cumulative emission sums
            cumsum_emit = torch.zeros(T + 1, K, dtype=DTYPE, device=device)
            cumsum_emit[1:] = torch.cumsum(seq_logp, dim=0)

            dur_range = torch.arange(1, Dmax + 1, device=device)        # [Dmax]
            ends = torch.arange(T, device=device).unsqueeze(1) + dur_range.unsqueeze(0)  # [T,Dmax]
            ends_clamped = ends.clamp(max=T)  # clip beyond sequence length

            start = (ends_clamped - dur_range).clamp(min=0)  # start indices for emission sums

            # emission sums: [T,K,Dmax]
            emit_sums = cumsum_emit[ends_clamped] - cumsum_emit[start]
            emit_sums = emit_sums.permute(0, 2, 1)  # [T,K,Dmax]

            # 4. Initialize β
            beta_tensor = torch.full((T, K, Dmax), neg_inf, dtype=DTYPE, device=device)
            beta_tensor[-1, :, 0] = 0.0  # 0-duration at final step

            # 5. Backward recursion
            for t in reversed(range(T)):
                max_d = min(Dmax, T - t)
                dur_scores = dur_logits[t, :, :max_d]  # [K,max_d]
                trans_t = trans_logits[t]              # [K,K]
                init_t = init_logits[t]                # [K]

                # β of next steps
                next_beta = torch.full((max_d, K), neg_inf, dtype=DTYPE, device=device)
                valid_mask = (t + torch.arange(max_d, device=device) < T)
                if valid_mask.any():
                    idx = valid_mask.nonzero(as_tuple=True)[0]
                    next_beta[idx] = torch.logsumexp(beta_tensor[t + torch.arange(max_d, device=device)[idx], :, :], dim=2)

                if (~valid_mask).any():
                    idx = (~valid_mask).nonzero(as_tuple=True)[0]
                    next_beta[idx] = init_t.unsqueeze(0)

                # Combine transitions
                beta_trans = torch.logsumexp(next_beta.unsqueeze(2) + trans_t.unsqueeze(0), dim=1)  # [max_d,K]

                # Update β
                beta_tensor[t, :, :max_d] = beta_trans.T + dur_scores + emit_sums[t, :, :max_d]

            beta_list.append(beta_tensor)

        return beta_list

    def _compute_state_posteriors(self, X: utils.SequenceSet, theta: Optional[ContextFeatures] = None):
        """
        Vectorized computation of HSMM state posteriors for a batch of sequences.
        Returns gamma, xi, eta as padded tensors.
        
        Shapes:
            gamma: [B, T_max, K]
            eta:   [B, T_max, K, Dmax]
            xi:    [B, T_max-1, K, K]
        """
        B = len(X.sequences)
        K, Dmax = self.n_states, self.max_duration
        T_max = max(X.lengths) if B > 0 else 0
        neg_inf = torch.finfo(DTYPE).min / 2.0
        device = X.log_probs[0].device if B > 0 else torch.device("cpu")

        # -------------------- Pad log_probs --------------------
        logp_padded = torch.full((B, T_max, K), neg_inf, dtype=DTYPE, device=device)
        mask = torch.zeros(B, T_max, dtype=DTYPE, device=device)
        for b, (logp, L) in enumerate(zip(X.log_probs, X.lengths)):
            if L > 0:
                logp_padded[b, :L, :] = logp
                mask[b, :L] = 1.0
        mask_bool = mask.bool()

        # -------------------- Align context --------------------
        ctx_aligned = None
        if theta is not None:
            if isinstance(theta, list):
                # Pad per sequence
                ctx_max_dim = theta[0].shape[-1]
                ctx_aligned = torch.zeros((B, T_max, ctx_max_dim), dtype=DTYPE, device=device)
                for b, ctx_seq in enumerate(theta):
                    L = X.lengths[b]
                    if L > 0:
                        ctx_aligned[b, :L] = ctx_seq
            else:
                ctx_aligned = theta
                if ctx_aligned.shape[1] != T_max:
                    raise ValueError("Context time dimension mismatch with max sequence length")

        # -------------------- Forward / Backward --------------------
        alpha_list = self._forward(X, theta=ctx_aligned)
        beta_list = self._backward(X, theta=ctx_aligned)

        # Pad alpha/beta to [B, T_max, K, Dmax]
        alpha_padded = torch.full((B, T_max, K, Dmax), neg_inf, dtype=DTYPE, device=device)
        beta_padded = torch.full((B, T_max, K, Dmax), neg_inf, dtype=DTYPE, device=device)
        for b, L in enumerate(X.lengths):
            if L > 0:
                alpha_padded[b, :L] = alpha_list[b]
                beta_padded[b, :L] = beta_list[b]

        # -------------------- Eta --------------------
        eta_log = alpha_padded + beta_padded
        eta_flat = eta_log.view(B, T_max, -1)
        eta_norm = torch.logsumexp(eta_flat, dim=-1, keepdim=True)
        eta = (eta_flat - eta_norm).view(B, T_max, K, Dmax).exp()
        eta = eta * mask.unsqueeze(-1).unsqueeze(-1)  # mask padding

        # -------------------- Gamma --------------------
        gamma = eta.sum(-1)
        gamma = gamma / gamma.sum(-1, keepdim=True).clamp_min(EPS)
        gamma = gamma * mask.unsqueeze(-1)

        # -------------------- Xi --------------------
        xi = torch.zeros((B, T_max - 1, K, K), dtype=DTYPE, device=device)
        for b, L in enumerate(X.lengths):
            if L <= 1:
                continue
            # Transition logits
            trans_logits = self._ensure_dim(self.transition_module, ctx_aligned[b] if ctx_aligned is not None else None)
            if trans_logits.shape[0] == 1:
                trans_logits = trans_logits.squeeze(0)
            if trans_logits.ndim == 2:  # static
                trans_seq = trans_logits.unsqueeze(0).expand(L-1, -1, -1)
            else:
                trans_seq = trans_logits[:L-1]

            a_prev = torch.logsumexp(alpha_list[b], dim=-1)
            b_next = torch.logsumexp(beta_list[b], dim=-1)
            xi_seq = []
            for t in range(L-1):
                log_xi = a_prev[t][:, None] + trans_seq[t] + b_next[t+1][None, :]
                log_xi = log_xi - torch.logsumexp(log_xi, dim=(0,1))
                xi_seq.append(log_xi.exp())
            xi[b, :L-1] = torch.stack(xi_seq)

        return gamma, xi, eta

    def _model_params(
        self,
        X: Optional[utils.SequenceSet] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
        max_iter: int = 50,
        iter_idx: int = 0,
    ) -> dict[str, Any]:
        """
        Compute HSMM model parameters with EM-style estimate, sample-mode collapse, and neural updates.
        Returns a dict of PDFs: emission, initial, duration, transition.
        """
        eps_collapse = 1e-3
        α_min, α_max = 0.03, self.alpha
        α = max(α_min, α_max * (1.0 - iter_idx / max_iter))

        def _check_context_hidden_dim(context: Optional[torch.Tensor] = None):
            """
            Ensure that the encoder context_dim matches hidden_dim assumptions.
            Raises an error if inconsistent.
            """
            ctx_dim = self.context_dim
            if context is not None:
                if context.ndim == 3:
                    ctx_dim = context.shape[-1]
                elif context.ndim == 2:
                    ctx_dim = context.shape[-1]
            if self.hidden_dim is not None and ctx_dim != self.hidden_dim:
                raise RuntimeError(
                    f"Context dimension mismatch: inferred context_dim={ctx_dim}, "
                    f"hidden_dim={self.hidden_dim}. Check encoder output or hidden_dim setting."
                )

        if theta is not None:
            _check_context_hidden_dim(theta)
        elif hasattr(self, "_context") and self._context is not None:
            _check_context_hidden_dim(self._context)
        else:
            _check_context_hidden_dim()

        # Flatten sequences for EM updates
        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequences", [X]) if s.numel() > 0], dim=0)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, dtype=DTYPE, device=self.device)
        else:
            all_X = torch.zeros(1, self.n_features, dtype=DTYPE, device=self.device)

        # Align or encode context
        if theta is not None:
            context_aligned = theta
            if theta.ndim == 2:  # [T,H] -> [1,T,H]
                context_aligned = theta.unsqueeze(0)
            elif X is not None and theta.shape[0] != len(X.sequences):
                context_aligned = theta.expand(len(X.sequences), -1, -1)
        else:
            context_aligned, _ = getattr(self, "_encode_observations", lambda x: (None, None))(all_X)
            if context_aligned is None:
                B = len(X.sequences) if X else 1
                context_aligned = torch.zeros(B, all_X.shape[0], self.context_dim, device=all_X.device)

        # -------- Sample mode: collapse logits --------
        def collapse_logits(logits: torch.Tensor, dim: int) -> torch.Tensor:
            logits = logits.clamp_min(-20.0)
            probs = logits.exp()
            probs /= probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            dir_alpha = probs * α + EPS
            shape = dir_alpha.shape
            perm = list(range(dir_alpha.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            flat = dir_alpha.permute(perm).reshape(-1, shape[dim])
            samples = torch.distributions.Dirichlet(flat).rsample()
            samples = samples.reshape(*[shape[i] for i in perm]).permute(*perm)
            samples = samples / samples.sum(dim=dim, keepdim=True)
            samples = torch.where(samples > eps_collapse, samples, torch.full_like(samples, 1.0 / shape[dim]))
            return torch.log(samples + EPS)

        def safe_sum(lst: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            tensors = [x for x in lst if x is not None and x.numel() > 0]
            return torch.cat(tensors, dim=0).sum(dim=0) if tensors else None

        if mode == "sample":
            with torch.no_grad():
                # Initial
                initial_pdf = self.initial_module.forward(context=context_aligned, return_dist=True)
                self.initial_module.update(
                    new_logits=collapse_logits(initial_pdf.logits, dim=0).exp(),
                    from_probs=True
                )
                # Transition
                transition_pdf = self.transition_module.forward(context=context_aligned, return_dist=True)
                self.transition_module.update(
                    new_logits=collapse_logits(transition_pdf.logits, dim=1).exp(),
                    from_probs=True
                )
                # Duration
                duration_pdf = self.duration_module.forward(context=context_aligned, return_dist=True)
                self.duration_module.update(
                    new_logits=collapse_logits(duration_pdf.logits, dim=1).exp(),
                    from_probs=True
                )
                # Emission
                try:
                    emission_pdf = self.emission_module.forward(context=context_aligned, return_dist=True)
                except Exception:
                    emission_pdf = self.emission_module.initialize(
                        X=all_X, context=context_aligned, theta=context_aligned, theta_scale=theta_scale
                    )

        # -------- Estimate mode: EM updates --------
        elif mode == "estimate":
            if X is None or not isinstance(X, utils.SequenceSet):
                raise RuntimeError("SequenceSet X required for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_state_posteriors(X, theta=context_aligned)

            # Posterior counts with fallback to module buffer
            init_counts = safe_sum([g[0] for g in gamma_list])
            init_counts = init_counts if init_counts is not None else self.initial_module._mod_logits_buffer.exp()

            trans_counts = safe_sum(xi_list)
            trans_counts = trans_counts if trans_counts is not None else self.transition_module._mod_logits_buffer.exp()

            dur_counts = safe_sum(eta_list)
            dur_counts = dur_counts if dur_counts is not None else self.duration_module._mod_logits_buffer.exp()

            # α-blended log normalization
            blend_init = constraints.log_normalize(
                torch.log(init_counts + EPS) * α + (1 - α) * self.initial_module._mod_logits_buffer, dim=0
            )
            blend_trans = constraints.log_normalize(
                torch.log(trans_counts + EPS) * α + (1 - α) * self.transition_module._mod_logits_buffer, dim=1
            )
            blend_dur = constraints.log_normalize(
                torch.log(dur_counts + EPS) * α + (1 - α) * self.duration_module._mod_logits_buffer, dim=1
            )

            # Update modules
            for module, logits in [
                (self.initial_module, blend_init),
                (self.transition_module, blend_trans),
                (self.duration_module, blend_dur)
            ]:
                posterior = logits.exp() if any(p.requires_grad for p in module.parameters()) else None
                module.update(new_logits=logits.exp(), posterior=posterior, from_probs=True)

            # Emission
            try:
                emission_pdf = self.emission_module.forward(context=context_aligned, return_dist=True)
            except Exception:
                emission_pdf = self.emission_module.initialize(
                    X=all_X, context=context_aligned, theta=context_aligned, theta_scale=theta_scale
                )

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # -------- Forward once for all modules to return PDFs --------
        initial_pdf = self.initial_module.forward(context=context_aligned, return_dist=True)
        duration_pdf = self.duration_module.forward(context=context_aligned, return_dist=True)
        transition_pdf = self.transition_module.forward(context=context_aligned, return_dist=True)

        return {
            "initial_pdf": initial_pdf,
            "duration_pdf": duration_pdf,
            "transition_pdf": transition_pdf,
            "emission_pdf": emission_pdf,
        }

    def _viterbi(
        self,
        X: utils.SequenceSet,
        theta: Optional[torch.Tensor] = None,
        duration_weight: float = 0.0
    ) -> list[torch.Tensor]:
        """
        Time-aware Viterbi for HSMM with per-timestep logits and duration weighting.
        Uses _ensure_dim to normalize context and module logits.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        predicted_sequences: list[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, dtype=torch.int64)

        for b, seq in enumerate(X.sequences):
            L = seq.shape[0]
            device = seq.device

            if L == 0:
                predicted_sequences.append(torch.empty(0, dtype=torch.int64, device=device))
                continue

            # --- Per-sequence context ---
            ctx_seq = theta[b] if isinstance(theta, list) else theta

            # --- Module logits, normalized to [1, L, ...] ---
            init_logits = self._ensure_dim(self.initial_module, ctx_seq)  # [1,L,K]
            dur_logits = self._ensure_dim(self.duration_module, ctx_seq)  # [1,L,K,Dmax]
            trans_logits = self._ensure_dim(self.transition_module, ctx_seq)  # [1,L,K,K]

            # Expand to [L,...] for convenience
            init_logits = init_logits.expand(L, K)
            dur_logits = dur_logits.expand(L, K, Dmax)
            trans_logits = trans_logits.expand(L, K, K)

            # --- Emission log-probs ---
            emit_log = X.log_probs[b].to(device)  # [L,K]
            cumsum_emit = torch.vstack((
                torch.zeros((1, K), device=device, dtype=DTYPE),
                torch.cumsum(emit_log, dim=0)
            ))  # [L+1,K]

            # --- Initialize Viterbi tensors ---
            V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.int64)
            best_durations = torch.zeros((L, K), dtype=torch.int64, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                # Current timestep logits
                ini = init_logits[t]             # [K]
                dur_scores = dur_logits[t, :, :max_d]  # [K, max_d]
                trans_t = trans_logits[t]        # [K, K]

                if duration_weight != 0.0:
                    dur_scores = dur_scores * (1.0 - duration_weight)

                if t == 0:
                    scores = ini.unsqueeze(1) + dur_scores + emit_sums  # [K, max_d]
                    best_score, best_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_idx]
                    back_ptr[t] = -1
                    continue

                # Previous scores per duration
                prev_scores_base = V[torch.clamp(starts - 1, min=0)]  # [max_d, K]
                mask_start0 = (starts == 0).unsqueeze(1).expand(-1, K)
                prev_scores_base = torch.where(mask_start0, ini.unsqueeze(0).expand_as(prev_scores_base), prev_scores_base)

                # Align for broadcasting: prev_state x duration x next_state
                prev_scores_base = prev_scores_base.T.unsqueeze(2)  # [K, max_d, 1]
                trans_exp = trans_t.unsqueeze(1)                    # [K, 1, K]
                prev_scores = prev_scores_base + trans_exp          # [K, max_d, K]

                # Max over previous states
                prev_max, prev_arg = prev_scores.max(dim=0)         # [max_d, K], [max_d, K]

                # Total scores including duration + emission
                scores = prev_max.T + dur_scores + emit_sums       # [K, max_d]
                best_score, best_d_idx = scores.max(dim=1)

                V[t] = best_score
                best_durations[t] = durations[best_d_idx]

                state_idx = torch.arange(K, device=device)
                prev_arg_selected = prev_arg[best_d_idx, state_idx]
                back_ptr[t] = torch.where(best_durations[t] == 1, torch.full_like(prev_arg_selected, -1), prev_arg_selected)

            # --- Backtrace ---
            t_cursor = L - 1
            cur_st = int(torch.argmax(V[t_cursor]).item())
            segments = []

            while t_cursor >= 0:
                d = int(best_durations[t_cursor, cur_st].item())
                start = max(0, t_cursor - d + 1)
                segments.append((start, t_cursor, cur_st))
                prev_state = int(back_ptr[t_cursor, cur_st].item())
                t_cursor = start - 1
                cur_st = prev_state if prev_state >= 0 else cur_st

            segments.reverse()
            seq_path = torch.cat([
                torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                for start, end, st in segments
            ])
            predicted_sequences.append(seq_path[:L])

        return predicted_sequences


    # HSMM EM
    @torch.no_grad()
    def _compute_emit_log(
        self,
        X: utils.SequenceSet,
        theta: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Compute per-sequence log-likelihoods for an HSMM with optional context.
        Handles variable-length sequences and zero-length sequences robustly.

        Args:
            X: SequenceSet object containing sequences and lengths.
            theta: Optional context tensor for sequences.
            verbose: If True, logs min/max/mean of log-likelihoods.

        Returns:
            Tensor of shape [B], one log-likelihood per sequence.
        """
        B = len(X.sequences)
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # Handle empty batch
        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, dtype=DTYPE)
            return X.log_likelihoods

        # Prepare context
        if theta is not None:
            context_aligned = theta
            if theta.ndim == 2:  # [T,H] -> [1,T,H]
                context_aligned = theta.unsqueeze(0)
            elif theta.shape[0] != B:
                context_aligned = theta.expand(B, -1, -1)
        else:
            context_aligned, _ = getattr(self, "_encode_observations", lambda x: (None, None))(torch.cat(X.sequences, dim=0))
            if context_aligned is None:
                max_len = max(X.lengths)
                context_aligned = torch.zeros(B, max_len, self.context_dim, device=self.device)

        # Forward pass
        alpha_list = self._forward(X, theta=context_aligned)

        # Determine dimensions
        max_len = max(X.lengths) if B > 0 else 0
        if alpha_list and alpha_list[0].ndim == 3:
            n_states, n_durations = alpha_list[0].shape[1:3]
        else:
            n_states, n_durations = self.n_states, self.max_duration

        # Preallocate padded alpha tensor
        alpha_padded = torch.full(
            (B, max_len, n_states, n_durations), neg_inf, dtype=DTYPE, device=self.device
        )

        # Pad sequences into batch tensor
        for b, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L > 0:
                alpha_padded[b, :L] = alpha[:L]

        # Compute per-sequence log-likelihood using logsumexp over states and durations
        lengths_tensor = torch.tensor(X.lengths, dtype=torch.long, device=self.device)
        valid_mask = lengths_tensor > 0
        ll = torch.full((B,), neg_inf, dtype=DTYPE, device=self.device)

        if valid_mask.any():
            final_alpha = alpha_padded[valid_mask, lengths_tensor[valid_mask] - 1]  # [valid_B, n_states, n_durations]
            ll[valid_mask] = torch.logsumexp(final_alpha.view(final_alpha.size(0), -1), dim=1)

        if verbose:
            logger.info(
                f"[compute_emit_log] seqs={B}, min={ll.min().item():.4f}, "
                f"max={ll.max().item():.4f}, mean={ll.mean().item():.4f}"
            )

        X.log_likelihoods = ll
        return X.log_likelihoods

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        patience: int = 1,
        max_iter: int = 20,
        ignore_conv: bool = False,
        theta: Optional[torch.Tensor] = None,
        update_rate_max: float = 0.8,
        update_rate_min: float = 0.1,
        adapt_factor: float = 10.0,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        """Fit HSMM using EM with optional context-modulated updates."""

        # ---------------- Encode context if needed ----------------
        if theta is None and getattr(self, "encoder", None) is not None:
            context_aligned, _ = self._encode_observations(X)
            theta = context_aligned

        # ---------------- Prepare observations ----------------
        X_valid = self._prepare_observations(X, theta=theta)
        B = len(X_valid.sequences)
        max_len = max(X_valid.lengths) if B > 0 else 0
        F_dim = X_valid.sequences[0].shape[-1] if B > 0 else 0
        device = X_valid.sequences[0].device if B > 0 else torch.device("cpu")

        seq_tensor = torch.zeros((B, max_len, F_dim), dtype=DTYPE, device=device)
        mask = torch.zeros(B, max_len, dtype=DTYPE, device=device)
        for b, seq in enumerate(X_valid.sequences):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq
            mask[b, :L] = 1.0
        mask_exp = mask.unsqueeze(-1)

        self._convergence = ConvergenceTracker(
            tol=tol, rel_tol=tol, n_init=n_init, max_iter=max_iter, patience=patience, verbose=verbose
        )
        best_score = -float("inf")

        neural_update = getattr(self, "encoder", None) is not None and any(
            p.requires_grad for p in self.emission_module.parameters()
        )

        # ---------------- EM initialization ----------------
        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            mode = "sample" if run_idx > 0 else "estimate"
            params = self._model_params(X_valid, theta=theta, mode=mode)

            init_pdf = params["initial_pdf"]
            transition_pdf = params["transition_pdf"]
            duration_pdf = params["duration_pdf"]
            emission_pdf = params["emission_pdf"]

            X_valid.log_probs = emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp
            prev_ll = self._compute_emit_log(X_valid).sum().item()
            self._convergence.update(prev_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # -------- Compute posteriors --------
                gamma_list, xi_list, eta_list = self._compute_state_posteriors(X_valid, theta=theta)

                # -------- Preallocate batch tensors --------
                K, Dmax = self.n_states, self.max_duration
                max_len_gamma = max([g.shape[0] for g in gamma_list], default=0)
                max_len_eta   = max([e.shape[0] for e in eta_list], default=0)
                max_len_xi    = max([x.shape[0] if x is not None else 0 for x in xi_list], default=0)

                gamma_tensor = torch.zeros(B, max_len_gamma, K, dtype=DTYPE, device=device)
                eta_tensor   = torch.zeros(B, max_len_eta, K, Dmax, dtype=DTYPE, device=device)
                xi_tensor    = torch.zeros(B, max_len_xi, K, K, dtype=DTYPE, device=device) if max_len_xi > 0 else None

                for b in range(B):
                    Lg = gamma_list[b].shape[0]
                    gamma_tensor[b, :Lg] = gamma_list[b]

                    Le = eta_list[b].shape[0]
                    eta_tensor[b, :Le] = eta_list[b]

                    if xi_tensor is not None and xi_list[b] is not None:
                        Lx = xi_list[b].shape[0]
                        xi_tensor[b, :Lx] = xi_list[b]

                # Apply mask to gamma/eta
                gamma_tensor *= mask_exp
                eta_tensor   *= mask_exp.unsqueeze(-1)

                # -------- EM sufficient statistics --------
                init_counts = gamma_tensor.sum((0, 1))
                dur_counts = eta_tensor.sum((0, 1)) if eta_tensor is not None else duration_pdf.expected_probs()
                trans_counts = xi_tensor.sum((0, 1)) if xi_tensor is not None else transition_pdf.expected_probs()

                # Normalize counts
                init_counts /= init_counts.sum().clamp_min(EPS)
                dur_counts /= dur_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
                trans_counts /= trans_counts.sum(dim=1, keepdim=True).clamp_min(EPS)

                # -------- EM alpha decay --------
                α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
                α = max(α_min, α_max * (1.0 - it / max_iter))

                # Update categorical distributions
                init_pdf = self.initial_module.dist_type(
                    probs=α * init_counts + (1 - α) * self.initial_module.expected_probs()
                )
                duration_pdf = self.duration_module.dist_type(
                    probs=α * dur_counts + (1 - α) * self.duration_module.expected_probs()
                )
                transition_pdf = self.transition_module.dist_type(
                    probs=α * trans_counts + (1 - α) * self.transition_module.expected_probs()
                )

                # -------- Flatten for emission updates --------
                flat_mask = mask.bool().reshape(-1)
                flat_X = seq_tensor.reshape(-1, F_dim)[flat_mask]
                flat_gamma = gamma_tensor.reshape(-1, K)[flat_mask]
                flat_theta = theta.reshape(-1, theta.shape[-1])[flat_mask] if theta is not None else None

                # Adaptive learning rate
                delta_ll = max((emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp / F_dim).sum().item() - prev_ll, 0.0)
                adaptive_rate = min(update_rate_max, max(update_rate_min, adapt_factor * delta_ll))
                update_rate_final = α * adaptive_rate + (1 - α) * update_rate_min

                # Neural or EM update for emission
                if neural_update:
                    self.emission_module.zero_grad()
                    emission_loss = -(flat_gamma * self.emission_module.log_prob(flat_X)).sum() / flat_gamma.sum()
                    emission_loss.backward()
                    with torch.no_grad():
                        for p in self.emission_module.parameters():
                            if p.grad is not None:
                                p.data.add_(update_rate_final * p.grad)
                    emission_pdf = self.emission_module.forward(context=flat_theta, return_dist=True)
                else:
                    emission_pdf = self.emission_module.update(
                        X=flat_X,
                        posterior=flat_gamma,
                        theta=flat_theta,
                        update_rate=update_rate_final
                    )

                self._params["emission_pdf"] = emission_pdf

                # -------- Likelihood & convergence --------
                X_valid.log_probs = emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp / F_dim
                curr_ll = self._compute_emit_log(X_valid).sum().item()
                delta = curr_ll - prev_ll

                if verbose:
                    print(f"[Iter {it:02d}] LL={curr_ll:.4f} Δ={delta:.3e} (update_rate={update_rate_final:.3f}, α={α:.3f})")

                if self._convergence.update(curr_ll, it, run_idx) and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

                prev_ll = curr_ll

            # -------- Store best parameters --------
            if curr_ll > best_score:
                best_score = curr_ll
                self._params.update({
                    "initial_pdf": init_pdf,
                    "transition_pdf": transition_pdf,
                    "duration_pdf": duration_pdf,
                    "emission_pdf": emission_pdf
                })

        if plot_conv:
            self._convergence.plot()

        return self

    def _map(self, X: utils.SequenceSet) -> list[torch.Tensor]:
        """
        MAP decoding of HSMM sequences using posterior state marginals.
        Returns a list of tensors, each of shape [T].
        """
        gamma_list, _, _ = self._compute_state_posteriors(X)
        results = []

        for gamma in gamma_list:
            if gamma is None or gamma.numel() == 0:
                results.append(torch.empty(0, dtype=torch.long))
                continue

            # Replace NaNs and infinities with large negative values
            gamma = torch.nan_to_num(gamma, nan=-1e9, posinf=-1e9, neginf=-1e9)

            # MAP decoding
            map_seq = gamma.argmax(dim=-1)
            results.append(map_seq.to(dtype=torch.long))

        return results

    def predict(
        self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        transition_temp: float = 1.0,
        duration_temp: float = 1.0,
    ) -> list[torch.Tensor]:
        """
        Predict hidden states for HSMM sequences using Viterbi or MAP decoding.
        Fully vectorized for batched sequences, supports variable-length sequences
        and optional context modulation.
        """
        # --- Prepare observations ---
        obs = self._prepare_observations(X, theta=context)
        B = len(obs.sequences)
        lengths = [seq.shape[0] for seq in obs.sequences]
        max_len = max(lengths) if lengths else 0
        K = self.n_states

        if max_len == 0:
            return [torch.empty(0, dtype=torch.long) for _ in range(B)]

        # --- Prepare per-sequence context ---
        if context is None:
            theta_list = [None] * B
        elif isinstance(context, list):
            theta_list = context
        else:  # single tensor
            theta_list = [context] * B

        # --- Decode ---
        preds: list[torch.Tensor] = []

        if algorithm.lower() == "viterbi":
            decoded_list = self._viterbi(obs, theta=theta_list)
            for seq_path in decoded_list:
                preds.append(seq_path.detach().to(dtype=torch.long))
            return preds

        elif algorithm.lower() == "map":
            # Precompute context-modulated logits for all sequences
            initial_logits = []
            transition_logits = []
            duration_logits = []

            for b in range(B):
                theta_b = theta_list[b]

                # Initial, transition, duration logits
                init = self.initial_module.log_matrix(context=theta_b)
                trans = self.transition_module.log_matrix(context=theta_b) / max(transition_temp, 1e-6)
                dur = self.duration_module.log_matrix(context=theta_b) / max(duration_temp, 1e-6)

                initial_logits.append(init.clamp(-MAX_LOGITS, MAX_LOGITS).squeeze(0))
                transition_logits.append(trans.clamp(-MAX_LOGITS, MAX_LOGITS).squeeze(0))
                duration_logits.append(dur.clamp(-MAX_LOGITS, MAX_LOGITS).squeeze(0))

            # Map decoding per sequence
            for b in range(B):
                L = lengths[b]
                if L == 0:
                    preds.append(torch.empty(0, dtype=torch.long))
                    continue

                log_probs_b = obs.log_probs[b]  # [T, K] or [T, K, F]
                decoded = self._map_decode(
                    log_probs_b,
                    initial_logits[b],
                    transition_logits[b],
                    duration_logits[b]
                )
                preds.append(decoded.detach().to(dtype=torch.long))

            return preds

        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

    @torch.no_grad()
    def score(self, X: torch.Tensor | list[torch.Tensor], theta: Optional[torch.Tensor | list[torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute log-likelihoods for sequences under the HSMM model with optional contextual modulation.
        Supports variable-length, time-varying context.
        Returns [B] log-likelihoods.
        """
        # --- Normalize X to list ---
        X_list = [X] if torch.is_tensor(X) else list(X)
        B = len(X_list)
        if B == 0:
            return torch.empty(0, dtype=DTYPE)

        # --- Normalize theta to list of tensors or None per sequence ---
        if theta is None:
            theta_list: list[Optional[torch.Tensor]] = [None] * B
        elif torch.is_tensor(theta):
            # split theta according to sequence lengths
            lengths = [seq.shape[0] for seq in X_list]
            cum_lengths = torch.cat([torch.zeros(1, dtype=torch.long),
                                     torch.tensor(lengths, dtype=torch.long).cumsum(0)])
            theta_list = [theta[cum_lengths[b]:cum_lengths[b + 1]].to(DTYPE) for b in range(B)]
        elif isinstance(theta, list):
            theta_list = [t.to(DTYPE) if isinstance(t, torch.Tensor) else None for t in theta]
            if len(theta_list) != B:
                raise ValueError(f"Length of theta list ({len(theta_list)}) does not match number of sequences ({B})")
        else:
            raise TypeError(f"Unsupported theta type {type(theta)}")

        # --- Compute emissions ---
        all_seq = torch.cat([seq.to(DTYPE) for seq in X_list], dim=0)
        non_none_theta = [t for t in theta_list if t is not None]
        all_theta = torch.cat(non_none_theta, dim=0) if non_none_theta else None
        log_B = self.emission_module.log_prob(all_seq, context=all_theta)  # [sum T_b, K]

        # Split emission log-probs per sequence
        seq_lengths = [seq.shape[0] for seq in X_list]
        log_probs_split = list(torch.split(log_B, seq_lengths, dim=0))
        obs = utils.SequenceSet(X_list, log_probs=log_probs_split)

        # --- Remove global_ctx, store per-sequence theta for future _forward use ---
        # _forward will handle theta_list directly
        alpha_list = self._forward(obs, theta=theta_list)

        # --- Compute log-likelihood per sequence ---
        log_likelihoods = []
        for alpha, L in zip(alpha_list, seq_lengths):
            if L == 0:
                log_likelihoods.append(torch.tensor(0.0, dtype=DTYPE))
            else:
                log_likelihoods.append(torch.logsumexp(alpha[L - 1], dim=(-2, -1)))

        return torch.stack(log_likelihoods)

    @torch.no_grad()
    def info(
        self,
        X: torch.Tensor,
        criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """
        Compute an information criterion (AIC, BIC, etc.) for the HSMM.

        Args:
            X: Input sequences, shape (B, T, F) or (T, F).
            criterion: Which information criterion to compute.
            lengths: Optional sequence lengths for variable-length batches.
            by_sample: Whether to return per-sequence values or a scalar sum.

        Returns:
            Tensor of information criterion values (per-sample or aggregated).
        """

        device = self.device

        # --- Compute per-sequence log-likelihood ---
        try:
            ll = self.score(X)  # Returns [B] tensor
            ll = ll.to(dtype=DTYPE, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to compute log-likelihood: {e}")

        # --- Count total observations ---
        if lengths is not None:
            n_obs = max(sum(lengths), 1)
        elif X.ndim == 3:  # (B, T, F)
            n_obs = max(X.shape[0] * X.shape[1], 1)
        elif X.ndim == 2:  # single sequence (T, F)
            n_obs = max(X.shape[0], 1)
        else:
            raise ValueError(f"Unsupported input shape {X.shape}")

        # --- Degrees of freedom ---
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError(
                "Model degrees of freedom ('dof') not defined. "
                "Please set 'self.dof' during initialization."
            )

        # --- Compute information criterion ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=ll,
            dof=dof,
            criterion=criterion
        )

        # --- Convert to tensor and sanitize ---
        if not isinstance(ic_value, torch.Tensor):
            ic_value = torch.tensor(ic_value, dtype=DTYPE, device=device)
        ic_value = ic_value.nan_to_num(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))

        # --- Ensure per-sample shape ---
        if by_sample and ic_value.ndim == 0:
            ic_value = ic_value.unsqueeze(0)

        return ic_value.detach().cpu()

    def decode(
        self,
        X: torch.Tensor | np.ndarray,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        first_only: bool = True
    ) -> np.ndarray | list[np.ndarray]:
        """
        Decode hidden states from input sequence(s) using Viterbi or MAP.

        Args:
            X: Input sequence(s), tensor or ndarray
            algorithm: "viterbi" or "map"
            first_only: If True, return only first sequence (single-sequence use)
                        If False, return list of arrays for all sequences
        Returns:
            NumPy array(s) of predicted states
        """
        # Ensure input tensor with consistent dtype
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE)
        else:
            X = X.to(dtype=DTYPE)

        # Get predicted states via HSMM predict method
        preds = self.predict(X, algorithm=algorithm, context=None)

        # Convert to NumPy arrays
        preds_np = [p.detach().cpu().numpy() for p in preds]

        return preds_np[0] if first_only and preds_np else preds_np

