# nhsmm/models/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.context import ContextEncoder
from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger, MAX_LOGITS
from nhsmm.tools import utils, constraints, SeedGenerator, ConvergenceTracker
from nhsmm.distributions import Categorical, Initial, Emission, Duration, Transition


class HSMM(nn.Module, ABC):
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

    def _align_theta(self, theta, seq_len: int):
        """
        Aligns and pads an optional context representation 'theta' to match a
        target sequence length.

        Preserves API:
          - Accepts None, list of tensors, or tensor.
          - Returns:
                list[tensor]  -> [1, seq_len, F_total]
                1D tensor     -> [seq_len, F]
                2D tensor     -> [seq_len, F]
                3D tensor     -> [B, seq_len, F]
        """
        if theta is None:
            return None

        # ================================================================
        # CASE 1 — list of tensors → concatenate along feature dim
        # ================================================================
        if isinstance(theta, list):
            if len(theta) == 0:
                return None
            if not all(torch.is_tensor(t) for t in theta):
                raise TypeError("All elements in theta list must be torch.Tensor")

            aligned_list = []
            for t in theta:
                if t.ndim == 1:
                    # [F] → [1,F]
                    t = t.unsqueeze(0)
                if t.ndim != 2:
                    raise TypeError(
                        f"List tensor must be 1D or 2D, got ndim={t.ndim} with shape {t.shape}"
                    )

                T, F = t.shape
                out = t.new_zeros(seq_len, F)
                out[:min(T, seq_len)] = t[:min(T, seq_len)]
                aligned_list.append(out)

            # concat on feature axis
            out = torch.cat(aligned_list, dim=-1)  # [seq_len, F_total]
            return out.unsqueeze(0)                # [1, seq_len, F_total]

        # ================================================================
        # CASE 2 — tensor input
        # ================================================================
        if not torch.is_tensor(theta):
            raise TypeError(f"theta must be Tensor or list of Tensor, got {type(theta)}")

        ndim = theta.ndim

        # ---------------- 1D: [F] -> [seq_len, F]
        if ndim == 1:
            return theta.unsqueeze(0).expand(seq_len, -1).contiguous()

        # ---------------- 2D: [T, F] -> pad to [seq_len, F]
        if ndim == 2:
            T, F = theta.shape
            out = theta.new_zeros(seq_len, F)
            out[:min(T, seq_len)] = theta[:min(T, seq_len)]
            return out

        # ---------------- 3D: [B, T, F] -> pad to [B, seq_len, F]
        if ndim == 3:
            B, T, F = theta.shape
            out = theta.new_zeros(B, seq_len, F)
            out[:, :min(T, seq_len)] = theta[:, :min(T, seq_len)]
            return out

        # ---------------- unsupported
        raise TypeError(f"Unsupported theta dimension {ndim}; expected 1,2,3 or list[tensor]")

    @torch.no_grad()
    def _encode_observations(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None,
        detach: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes observations into:
            context_aligned:  [B, T, H]
            context_canonical: [B, 1, H]

        Preserves existing API + behavior, but ensures:
            - Strict shape guarantees
            - Stable mask handling
            - Robust canonical context
            - Safe detach
        """
        # Early exit
        if self.encoder is None or sequences.numel() == 0:
            self._sequence = None
            self._context = None
            self._ctx_canonical = None
            return None, None

        # Ensure batch
        if sequences.ndim == 2:
            sequences = sequences.unsqueeze(0)  # [1, T, F]

        B, T, F = sequences.shape
        device = sequences.device

        # Normalize mask to [B,T]
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        elif mask.ndim == 1:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 2:
            raise ValueError("mask must be [B,T] or [T]")

        # Optional pool override
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            # ---- Encoder pass ----
            theta_out, ctx_canonical, _ = self.encoder(
                sequences,
                mask=mask,
                return_context=True,
                return_sequence=True,
                detach_context=detach
            )

            # --- Canonical context ---
            # Expect one context per sequence: [B,1,H]
            if ctx_canonical is None:
                # No canonical → mean pool over time
                ctx_canonical = theta_out.mean(dim=1, keepdim=True)

            elif ctx_canonical.ndim == 2:
                # [B,H] → [B,1,H]
                ctx_canonical = ctx_canonical.unsqueeze(1)

            elif ctx_canonical.ndim == 3 and ctx_canonical.shape[1] != 1:
                # [B,T,H] but encoder didn’t reduce → reduce now
                ctx_canonical = ctx_canonical.mean(dim=1, keepdim=True)

            # Final guarantee: canonical is [B,1,H]
            assert ctx_canonical.ndim == 3 and ctx_canonical.shape[1] == 1

            # --- Per-timestep aligned context ---
            if theta_out.shape[1] == T:
                context_aligned = theta_out
            else:
                # Encoder returned pooled context only → tile
                context_aligned = ctx_canonical.expand(-1, T, -1)

            # Optional debug
            if getattr(self, "debug", False):
                print("DEBUG _encode_observations:")
                print(" sequences:", sequences.shape)
                print(" mask:", mask.shape)
                print(" theta_out:", theta_out.shape)
                print(" ctx_canonical:", ctx_canonical.shape)
                print(" context_aligned:", context_aligned.shape)

        finally:
            # Restore pool
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        # ---- Safe detach ----
        if detach:
            theta_out = theta_out.detach()
            context_aligned = context_aligned.detach()
            ctx_canonical = ctx_canonical.detach()

        # Store for access by forward passes
        self._sequence = theta_out          # [B,T,H_seq]
        self._context = context_aligned     # [B,T,H]
        self._ctx_canonical = ctx_canonical # [B,1,H]

        return context_aligned, ctx_canonical

    def _prepare_observations(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> SequenceSet:
        """
        Prepares SequenceSet:
            - Computes aligned + canonical context (via encoder or provided theta)
            - Computes log-probs for each timestep and state
            - Builds masks + lengths
        """
        device = X.device
        dtype = X.dtype

        # ---- Ensure batch dimension ----
        if X.ndim == 2:
            X = X.unsqueeze(0)
        B, T, F = X.shape

        # ---- Masks: [B,T,1] ----
        mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)

        # ---- Context computation ----
        if theta is None:
            context_aligned = []
            ctx_canonical = []
            for b in range(B):
                ctx_t, ctx_c = self._encode_observations(X[b])
                context_aligned.append(ctx_t)    # [T,H]
                ctx_canonical.append(ctx_c)      # [1,H]
        else:
            aligned_theta = self._align_theta(theta, seq_len=T)  # [B,T,H]
            context_aligned = [aligned_theta[b] for b in range(B)]
            ctx_canonical = [aligned_theta[b, :1, :] for b in range(B)]

        # ---- Emission parameters (cached) ----
        dist_type = self.emission_module.dist_type
        K = self.n_states

        # optional Gaussian params (used below if needed)
        if not dist_type in (torch.distributions.Categorical, torch.distributions.Bernoulli):
            means = self.emission_module._emission_means.to(device)
            covs = self.emission_module._emission_covs.to(device)
            stds = covs.diagonal(dim1=-2, dim2=-1).sqrt()

        # ---- Log-probs per batch ----
        log_probs_list = []

        for b in range(B):
            seq_b = X[b]                  # [T,F]
            ctx_b = context_aligned[b]    # [T,H]

            # First try per-timestep context; fallback to canonical
            try:
                emission_dist = self.emission_module.forward(context=ctx_b, return_dist=True)
            except Exception:
                logger.warning(f"Emission module failed forward pass: {e}, forward...")
                emission_dist = self.emission_module.forward(context=ctx_canonical[b], return_dist=True)

            T_seq = seq_b.shape[0]

            # ---- Categorical/Bernoulli ----
            if dist_type in (torch.distributions.Categorical, torch.distributions.Bernoulli):
                logits = emission_dist.logits             # [K,F]
                seq_cat = seq_b[:, 0].long()              # [T]

                log_probs_all = F.log_softmax(logits, dim=-1)  # [K,F]
                log_probs = torch.gather(
                    log_probs_all.unsqueeze(0).expand(T_seq, K, -1),
                    -1,
                    seq_cat.view(T_seq, 1, 1).expand(-1, K, 1)
                ).squeeze(-1)                              # [T,K]

            # ---- Gaussian-like (log_prob available or Independent) ----
            else:
                seq_exp  = seq_b.unsqueeze(1).expand(T_seq, K, F)
                means_exp = means.unsqueeze(0).expand(T_seq, K, F)
                stds_exp  = stds.unsqueeze(0).expand(T_seq, K, F)

                log_probs = -0.5 * torch.log(2 * torch.pi * stds_exp**2)
                log_probs -= 0.5 * ((seq_exp - means_exp)**2 / (stds_exp**2))
                log_probs = log_probs.sum(-1)  # [T,K]

            log_probs_list.append(log_probs)

        # ---- Pack result ----
        return utils.SequenceSet(
            sequences=[X[b] for b in range(B)],
            lengths=[T] * B,
            log_probs=log_probs_list,
            contexts=context_aligned,
            masks=[mask[b] for b in range(B)],
        )

    def _ensure_time_dim(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Ensure tensor has shape [B, T, ...] for context/logits broadcasting.
        Returns None if input is None.
        """
        if x is None:
            return None
        if x.ndim == 1:  # [H] -> [1,1,H]
            return x.unsqueeze(0).unsqueeze(1)
        if x.ndim == 2:  # [T,H] -> [1,T,H]
            return x.unsqueeze(0)
        return x

    def _forward(self, X: utils.SequenceSet, theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None) -> list[torch.Tensor]:
        """
        Vectorized forward pass for HSMM with context-modulated logits.
        Args:
            X: SequenceSet container
            theta: Optional per-sequence or per-timestep context

        Returns:
            List of alpha tensors [T, K, Dmax] per sequence
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        alpha_list = []

        for i, (log_emissions, seq_len) in enumerate(zip(X.log_probs, X.lengths)):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, dtype=DTYPE, device=log_emissions.device))
                continue

            T = seq_len
            device = log_emissions.device

            # --- Ensure per-sequence context ---
            ctx_seq = None
            if theta is not None:
                ctx_seq = theta[i] if isinstance(theta, list) else theta
            ctx_seq = self._ensure_time_dim(ctx_seq)

            # --- Module logits ---
            initial_logits = self._ensure_time_dim(self.initial_module.log_matrix(context=ctx_seq))  # [1,T,K]
            duration_logits = self._ensure_time_dim(self.duration_module.log_matrix(context=ctx_seq))  # [1,T,K,Dmax]
            transition_logits = self._ensure_time_dim(self.transition_module.log_matrix(context=ctx_seq))  # [1,T,K,K]

            # Broadcast to [T,...]
            initial_logits = initial_logits.expand(1, T, K).squeeze(0)         # [T,K]
            duration_logits = duration_logits.expand(1, T, K, Dmax).squeeze(0) # [T,K,Dmax]
            transition_logits = transition_logits.expand(1, T, K, K).squeeze(0) # [T,K,K]

            # --- Cumulative emission sums ---
            # cumsum_emit: [T+1, K]
            cumsum_emit = torch.zeros(T + 1, K, dtype=DTYPE, device=device)
            cumsum_emit[1:] = torch.cumsum(log_emissions, dim=0)

            # Compute start indices for each duration: [T,Dmax]
            dur_range = torch.arange(1, Dmax + 1, device=device).unsqueeze(0).expand(T, -1)
            start_idx = (torch.arange(T, device=device).unsqueeze(1) - dur_range + 1).clamp(min=0)

            # Compute segment emission sums: [T,K,Dmax]
            emit_sums = torch.stack([cumsum_emit[t+1] - cumsum_emit[start_idx[t]] for t in range(T)], dim=0)
            # emit_sums has shape [T,Dmax,K], need [T,K,Dmax]
            emit_sums = emit_sums.permute(0,2,1)

            # --- Initialize alpha tensor ---
            alpha_tensor = torch.full((T, K, Dmax), neg_inf, dtype=DTYPE, device=device)

            # --- Forward recursion ---
            for t in range(T):
                max_d = min(Dmax, t + 1)

                if t == 0:
                    alpha_tensor[t, :, :max_d] = initial_logits[t].unsqueeze(1) + duration_logits[t, :, :max_d] + emit_sums[t, :, :max_d]
                    continue

                # Previous alpha contributions: [max_d,K]
                prev_alpha = torch.full((max_d, K), neg_inf, dtype=DTYPE, device=device)
                valid_mask = start_idx[t, :max_d] > 0

                if valid_mask.any():
                    idx = valid_mask.nonzero(as_tuple=True)[0]
                    prev_vals = torch.logsumexp(alpha_tensor[start_idx[t, idx] - 1, :, :], dim=2)  # [len(idx),K]
                    prev_alpha[idx] = prev_vals

                if (~valid_mask).any():
                    idx = (~valid_mask).nonzero(as_tuple=True)[0]
                    prev_alpha[idx] = initial_logits[t].unsqueeze(0) + duration_logits[t, :, :max_d][:, idx].T

                # Combine with transition logits: [max_d,K]
                alpha_trans = torch.logsumexp(prev_alpha.unsqueeze(2) + transition_logits[t].unsqueeze(0), dim=1)
                alpha_tensor[t, :, :max_d] = alpha_trans.T + duration_logits[t, :, :max_d] + emit_sums[t, :, :max_d]

            alpha_list.append(alpha_tensor)

        return alpha_list

    def _backward(self, X: utils.SequenceSet, theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None) -> list[torch.Tensor]:
        """
        Vectorized, batch-safe backward pass for HSMM.
        Returns list of [T, K, Dmax] tensors for each sequence.
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

            # --- Sequence context ---
            ctx_seq = None
            if theta is not None:
                ctx_seq = theta[i] if isinstance(theta, list) else theta
                ctx_seq = self._ensure_time_dim(ctx_seq)  # [1,T,H], [T,H], [1,1,H]

            # --- Module logits ---
            dur_raw = self.duration_module.log_matrix(context=ctx_seq)      # [K,Dmax] or [T,K,Dmax]
            trans_raw = self.transition_module.log_matrix(context=ctx_seq) # [K,K] or [T,K,K]
            init_raw = self.initial_module.log_matrix(context=ctx_seq)     # [K] or [T,K]

            # --- Standardize shapes safely to [T,...] ---
            if dur_raw.ndim == 2:       # [K,Dmax] static
                dur_time = dur_raw.unsqueeze(0).repeat(T, 1, 1)
            elif dur_raw.ndim == 3:     # [T,K,Dmax]
                dur_time = dur_raw
            else:                        # [1,T,K,Dmax]
                dur_time = dur_raw.view(T, K, Dmax)

            if trans_raw.ndim == 2:      # [K,K] static
                trans_time = trans_raw.unsqueeze(0).repeat(T, 1, 1)
            elif trans_raw.ndim == 3:    # [T,K,K]
                trans_time = trans_raw
            else:                        # [1,T,K,K]
                trans_time = trans_raw.view(T, K, K)

            if init_raw.ndim == 1:       # [K] static
                init_time = init_raw.unsqueeze(0).repeat(T, 1)
            elif init_raw.ndim == 2:     # [T,K]
                init_time = init_raw
            else:                        # [1,T,K]
                init_time = init_raw.view(T, K)

            # --- Precompute cumulative emission sums ---
            cumsum_emit = torch.zeros(T + 1, K, dtype=DTYPE, device=device)
            cumsum_emit[1:] = torch.cumsum(seq_logp, dim=0)

            # --- Initialize beta tensor ---
            beta_tensor = torch.full((T, K, Dmax), neg_inf, dtype=DTYPE, device=device)
            beta_tensor[-1, :, 0] = 0.0  # terminal state

            # --- Backward recursion ---
            for t in reversed(range(T)):
                max_d = min(Dmax, T - t)
                ends = t + torch.arange(1, max_d + 1, device=device)  # [max_d]

                # Duration, transition, and initial scores
                dur_scores = dur_time[t, :, :max_d]   # [K, max_d]
                trans_t = trans_time[t]               # [K,K]
                init_t = init_time[t]                 # [K]

                # Segment emission sums
                emit_sums = torch.stack([cumsum_emit[e] - cumsum_emit[t] for e in ends], dim=1)  # [K, max_d]

                # Contributions from next timestep
                prev_beta = torch.full((max_d, K), neg_inf, dtype=DTYPE, device=device)
                valid_mask = ends < T
                if valid_mask.any():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    next_beta = beta_tensor[ends[valid_idx] - 1, :, :].logsumexp(dim=2)  # [len(valid_idx), K]
                    prev_beta[valid_idx] = next_beta

                # Segments that end at T (starting at t=0)
                zero_mask = ~valid_mask
                if zero_mask.any():
                    idx = zero_mask.nonzero(as_tuple=True)[0]
                    prev_beta[idx] = init_t.unsqueeze(0)  # broadcast along len(idx)

                # Combine with transition
                alpha_trans = (prev_beta.unsqueeze(2) + trans_t.unsqueeze(0)).logsumexp(dim=1)  # [max_d,K]

                # Update beta tensor
                beta_tensor[t, :, :max_d] = alpha_trans.T + dur_scores + emit_sums

            beta_list.append(beta_tensor)

        return beta_list

    def _compute_state_posteriors(self, X: utils.SequenceSet, theta: Optional[ContextFeatures] = None):
        K, Dmax = self.n_states, self.max_duration
        B = len(X.sequences)

        gamma_list, xi_list, eta_list = [], [], []

        for b in range(B):
            L = X.lengths[b]
            logp = X.log_probs[b].to(dtype=DTYPE)

            if L == 0:
                z = lambda *s: torch.zeros(*s, dtype=DTYPE)
                gamma_list.append(z(0, K))
                eta_list.append(z(0, K, Dmax))
                xi_list.append(z(0, K, K))
                continue

            # -------- Context alignment ----------
            ctx_aligned = None
            if theta is not None:
                ctx_b = theta[b] if isinstance(theta, list) else theta
                if ctx_b.ndim == 3 and ctx_b.shape[0] == 1:
                    ctx_aligned = ctx_b[0]
                elif ctx_b.ndim == 3:
                    ctx_aligned = ctx_b[b]
                elif ctx_b.ndim == 2:
                    ctx_aligned = ctx_b
                else:
                    ctx_aligned = ctx_b.unsqueeze(0).expand(L, -1)

            # -------- Normalize shapes ----------
            def norm(x):
                return x[0] if isinstance(x, torch.Tensor) and x.ndim > 1 and x.shape[0] == 1 else x

            initial_logits   = norm(self.initial_module.log_matrix(context=ctx_aligned))
            transition_logits = norm(self.transition_module.log_matrix(context=ctx_aligned))
            duration_logits  = norm(self.duration_module.log_matrix(context=ctx_aligned))

            # -------- Forward/backward ----------
            obs_seq = utils.SequenceSet(
                sequences=[logp], log_probs=[logp], lengths=[L]
            )
            alpha = self._forward(obs_seq, theta=ctx_aligned)[0]
            beta  = self._backward(obs_seq, theta=ctx_aligned)[0]

            # -------- Eta ----------
            eta_log = alpha + beta
            flat = eta_log.view(L, -1)
            norm_term = torch.logsumexp(flat, dim=-1, keepdim=True)
            flat = flat - norm_term
            eta = flat.view(L, K, Dmax).exp()

            # -------- Gammas ----------
            gamma = eta.sum(-1)
            gamma = gamma / gamma.sum(-1, keepdim=True).clamp_min(EPS)

            # -------- Xi ----------
            if L <= 1:
                xi = torch.zeros((0, K, K), dtype=DTYPE)
            else:
                if transition_logits.ndim == 2:
                    trans = transition_logits.unsqueeze(0).expand(L-1, -1, -1)
                else:
                    trans = transition_logits[:L-1]

                xi_seq = []
                for t in range(L - 1):
                    a_prev = torch.logsumexp(alpha[t], dim=-1)
                    b_next = torch.logsumexp(beta[t+1], dim=-1)
                    log_xi = a_prev[:,None] + trans[t] + b_next[None,:]
                    log_xi = log_xi - torch.logsumexp(log_xi, dim=(0,1))
                    xi_seq.append(log_xi.exp())
                xi = torch.stack(xi_seq)

            gamma_list.append(gamma)
            eta_list.append(eta)
            xi_list.append(xi)

        return gamma_list, xi_list, eta_list

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
        seq_len = sum(getattr(X, "lengths", [1])) if X is not None else 1
        aligned_theta = self._align_theta(theta, seq_len) if theta is not None else None

        # Flatten sequences
        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequences", [X]) if s.numel() > 0], dim=0)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, dtype=DTYPE)
        else:
            all_X = torch.zeros(1, self.n_features, dtype=DTYPE)

        α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
        α = max(α_min, α_max * (1.0 - iter_idx / max_iter))

        # Helpers
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

        if aligned_theta is None:
            if hasattr(self, "_context") and self._context is not None:
                aligned_theta = self._context
            else:
                _, aligned_theta = self._encode_observations(all_X)

        # -------- Sample mode: collapse logits --------
        if mode == "sample":
            with torch.no_grad():
                initial_pdf = self.initial_module.forward(context=aligned_theta, return_dist=True)
                self.initial_module.update(new_logits=collapse_logits(initial_pdf.logits, dim=0).exp(), from_probs=True)

                transition_pdf = self.transition_module.forward(context=aligned_theta, return_dist=True)
                self.transition_module.update(new_logits=collapse_logits(transition_pdf.logits, dim=1).exp(), from_probs=True)

                duration_pdf = self.duration_module.forward(context=aligned_theta, return_dist=True)
                self.duration_module.update(new_logits=collapse_logits(duration_pdf.logits, dim=1).exp(), from_probs=True)

                try:
                    emission_pdf = self.emission_module.forward(context=aligned_theta, return_dist=True)
                except Exception:
                    emission_pdf = self.emission_module.initialize(
                        X=all_X, context=aligned_theta, theta=aligned_theta, theta_scale=theta_scale
                    )

        # -------- Estimate mode: EM updates --------
        elif mode == "estimate":
            if X is None or not isinstance(X, utils.SequenceSet):
                raise RuntimeError("SequenceSet X required for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_state_posteriors(X, theta=aligned_theta)

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

            # Update modules once
            for module, logits in [
                (self.initial_module, blend_init),
                (self.transition_module, blend_trans),
                (self.duration_module, blend_dur)
            ]:
                posterior = logits.exp() if any(p.requires_grad for p in module.parameters()) else None
                module.update(new_logits=logits.exp(), posterior=posterior, from_probs=True)

            # Emission
            try:
                emission_pdf = self.emission_module.forward(context=aligned_theta, return_dist=True)
            except Exception:
                emission_pdf = self.emission_module.initialize(
                    X=all_X, context=aligned_theta, theta=aligned_theta, theta_scale=theta_scale
                )

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # -------- Forward once for all modules to return PDFs --------
        initial_pdf = self.initial_module.forward(context=aligned_theta, return_dist=True)
        duration_pdf = self.duration_module.forward(context=aligned_theta, return_dist=True)
        transition_pdf = self.transition_module.forward(context=aligned_theta, return_dist=True)

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
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        B = len(X.sequences)
        predicted_sequences: list[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, dtype=torch.int64)

        def select_logits(logits, t, seq_len, default):
            """Helper: choose per-timestep or static logits safely."""
            if logits is None:
                return default
            if logits.ndim == 3 and logits.shape[0] == seq_len:
                return logits[t]
            if logits.ndim in [2, 3] and logits.shape[0] == 1:
                return logits.squeeze(0)
            return logits

        for b, seq in enumerate(X.sequences):
            L = seq.shape[0]
            device = seq.device

            if L == 0:
                predicted_sequences.append(torch.empty(0, dtype=torch.int64, device=device))
                continue

            # Context per sequence
            ctx_seq = None
            if theta is not None:
                ctx_seq = theta[b] if isinstance(theta, list) else theta
                ctx_seq = self._ensure_time_dim(ctx_seq)

            # Module logits
            init_raw = self.initial_module.log_matrix(context=ctx_seq)
            dur_raw = self.duration_module.log_matrix(context=ctx_seq)
            trans_raw = self.transition_module.log_matrix(context=ctx_seq)

            # Emission log-probs
            emit_log = X.log_probs[b].to(device)  # [L,K]
            cumsum_emit = torch.vstack((
                torch.zeros((1, K), device=device, dtype=DTYPE),
                torch.cumsum(emit_log, dim=0)
            ))  # [L+1,K]

            # Initialize Viterbi tensors
            V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.int64)
            best_durations = torch.zeros((L, K), dtype=torch.int64, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                # Select logits
                ini = select_logits(init_raw, t, L, self.initial_module.logits)  # [K]
                dur_scores = select_logits(dur_raw, t, L, self.duration_module.logits)  # [K, Dmax]
                trans_t = select_logits(trans_raw, t, L, self.transition_module.logits)  # [K, K]

                if duration_weight != 0.0:
                    dur_scores = dur_scores * (1.0 - duration_weight)

                if t == 0:
                    # First timestep: initial + duration + emission
                    scores = ini.unsqueeze(1) + dur_scores[:, :max_d] + emit_sums  # [K, max_d]
                    best_score, best_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_idx]
                    back_ptr[t] = -1
                    continue

                # Previous scores for each duration
                prev_scores_base = V[torch.clamp(starts - 1, min=0)]  # [max_d, K]
                mask_start0 = (starts == 0).unsqueeze(1).expand(-1, K)
                prev_scores_base = torch.where(mask_start0, ini.unsqueeze(0).expand_as(prev_scores_base), prev_scores_base)  # [max_d, K]

                # Align for broadcasting: prev_state x duration x next_state
                prev_scores_base = prev_scores_base.T.unsqueeze(2)  # [K, max_d, 1]
                trans_exp = trans_t.unsqueeze(1)  # [K, 1, K]
                prev_scores = prev_scores_base + trans_exp  # [K, max_d, K]

                # Max over previous states
                prev_max, prev_arg = prev_scores.max(dim=0)  # [max_d, K], [max_d, K]

                # Total scores including duration + emission
                scores = prev_max.T + dur_scores[:, :max_d] + emit_sums  # [K, max_d]
                best_score, best_d_idx = scores.max(dim=1)  # [K]

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
        verbose: bool = False,
        device_output: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute per-sequence log-likelihoods for an HSMM with optional context.
        Handles variable-length sequences and zero-length sequences robustly.

        Args:
            X: SequenceSet object containing sequences and lengths.
            theta: Optional context tensor for sequences.
            verbose: If True, logs min/max/mean of log-likelihoods.
            device_output: Optional device for output tensor.

        Returns:
            Tensor of shape [B], one log-likelihood per sequence.
        """
        device_output = device_output or torch.device("cpu")
        B = len(X.sequences)
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # Handle empty batch
        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, dtype=DTYPE, device=device_output)
            return X.log_likelihoods

        # Align context per sequence if provided
        aligned_theta = self._align_theta(theta, sum(X.lengths)) if theta is not None else None

        # Forward pass (vectorized)
        alpha_list = self._forward(X, theta=aligned_theta)

        # Determine max sequence length and state/duration dimensions
        max_len = max(X.lengths) if B > 0 else 0
        if alpha_list and alpha_list[0].ndim == 3:
            n_states, n_durations = alpha_list[0].shape[1:3]
        else:
            n_states, n_durations = self.n_states, self.max_duration

        # Preallocate padded alpha tensor
        alpha_padded = torch.full(
            (B, max_len, n_states, n_durations), neg_inf, dtype=DTYPE
        )

        # Pad sequences into batch tensor
        for b, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L > 0:
                alpha_padded[b, :L] = alpha[:L]

        # Compute per-sequence log-likelihood using logsumexp over states and durations
        lengths_tensor = torch.tensor(X.lengths, dtype=torch.long)
        valid_mask = lengths_tensor > 0
        ll = torch.full((B,), neg_inf, dtype=DTYPE)

        if valid_mask.any():
            final_alpha = alpha_padded[valid_mask, lengths_tensor[valid_mask] - 1]  # [valid_B, n_states, n_durations]
            ll[valid_mask] = torch.logsumexp(final_alpha.view(final_alpha.size(0), -1), dim=1)

        # Move to output device
        ll = ll.to(device_output)
        X.log_likelihoods = ll

        if verbose:
            logger.info(
                f"[compute_emit_log] seqs={B}, min={ll.min().item():.4f}, "
                f"max={ll.max().item():.4f}, mean={ll.mean().item():.4f}"
            )

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

        aligned_theta = self._align_theta(theta, max_len) if theta is not None else None

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
            params = self._model_params(X_valid, theta=aligned_theta, mode=mode)

            # Access distributions via forward/_get_dist for context handling
            init_pdf = params["initial_pdf"]
            transition_pdf = params["transition_pdf"]
            duration_pdf = params["duration_pdf"]
            emission_pdf = params["emission_pdf"]

            X_valid.log_probs = emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp
            prev_ll = self._compute_emit_log(X_valid).sum().item()
            self._convergence.update(prev_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # -------- Compute posteriors --------
                gamma_list, xi_list, eta_list = self._compute_state_posteriors(X_valid, theta=aligned_theta)

                gamma_tensor = torch.nn.utils.rnn.pad_sequence(gamma_list, batch_first=True) * mask_exp
                xi_tensor = torch.nn.utils.rnn.pad_sequence([x for x in xi_list if x is not None], batch_first=True) if xi_list else None
                eta_tensor = torch.nn.utils.rnn.pad_sequence([e for e in eta_list if e is not None], batch_first=True) if eta_list else None
                if eta_tensor is not None:
                    eta_tensor = eta_tensor * mask_exp.unsqueeze(-1)

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

                # Update categorical distributions using counts
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
                flat_gamma = gamma_tensor.reshape(-1, self.n_states)[flat_mask]
                flat_theta = self._context.reshape(-1, self._context.shape[-1])[flat_mask] if getattr(self, "_context", None) is not None else None

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
                results.append(torch.empty(0, dtype=torch.long, device=self.device))
                continue

            # Ensure tensor is on the correct device
            gamma = gamma.to(self.device)
            gamma = torch.nan_to_num(gamma, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))

            map_seq = gamma.argmax(dim=-1)
            results.append(map_seq)

        return results

    def predict(
        self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        transition_temp: float = 1.0,
        duration_temp: float = 1.0,
        device_output: Optional[torch.device] = None,
    ) -> list[torch.Tensor]:
        """
        Predict hidden states using Viterbi or MAP decoding with optional contextual modulation.
        Supports variable-length sequences and batched context.
        """
        device_output = device_output or torch.device("cpu")

        # --- Prepare observations ---
        obs = self._prepare_observations(X, theta=context)
        B = len(obs.sequences)
        lengths = [seq.shape[0] for seq in obs.sequences]
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            return [torch.empty(0, dtype=torch.int64, device=device_output) for _ in range(B)]

        # --- Batch tensors ---
        n_features = obs.sequences[0].shape[1] if obs.sequences[0].ndim > 1 else 1
        seq_tensor = torch.zeros(B, max_len, n_features, dtype=DTYPE)
        log_probs_tensor = torch.zeros(B, max_len, self.n_states, dtype=DTYPE)
        mask = torch.zeros(B, max_len, dtype=torch.bool)

        for b, seq in enumerate(obs.sequences):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq
            log_probs_tensor[b, :L] = obs.log_probs[b]
            mask[b, :L] = True

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
                preds.append(seq_path.detach().to(device_output))
        elif algorithm.lower() == "map":
            for b in range(B):
                L = lengths[b]
                if L == 0:
                    preds.append(torch.empty(0, dtype=torch.int64, device=device_output))
                    continue

                theta_b = theta_list[b]
                # Compute context-modulated logits
                initial_logits_b = self.initial_module.log_matrix(context=theta_b).squeeze(0)
                transition_logits_b = self.transition_module.log_matrix(context=theta_b).squeeze(0) / max(transition_temp, 1e-6)
                transition_logits_b = transition_logits_b.clamp(-MAX_LOGITS, MAX_LOGITS)
                duration_logits_b = self.duration_module.log_matrix(context=theta_b).squeeze(0) / max(duration_temp, 1e-6)
                duration_logits_b = duration_logits_b.clamp(-MAX_LOGITS, MAX_LOGITS)

                log_probs_b = obs.log_probs[b]
                decoded = self._map_decode(
                    log_probs_b, initial_logits_b, transition_logits_b, duration_logits_b
                )
                preds.append(decoded.detach().to(device_output))
        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

        return preds

    @torch.no_grad()
    def score(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute log-likelihoods for sequences under the HSMM model with optional contextual modulation.
        Supports variable-length, time-varying context.
        Returns [B] log-likelihoods.
        """
        # --- Normalize X to list ---
        if torch.is_tensor(X):
            X_list = [X]
        elif isinstance(X, (list, tuple)):
            X_list = list(X)
        else:
            raise TypeError(f"Unsupported input type {type(X)}")

        B = len(X_list)
        if B == 0:
            return torch.empty(0, dtype=DTYPE)

        # --- Normalize theta to list of tensors or None per sequence ---
        if theta is None:
            theta_list: list[Optional[torch.Tensor]] = [None] * B
        elif torch.is_tensor(theta):
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

        # --- Global context for uniform modules ---
        global_ctx = None
        for t in theta_list:
            if t is not None and t.numel() > 0:
                global_ctx = t[:1] if t.ndim > 1 else t.unsqueeze(0)
                break

        temperature = max(self.temperature, 1e-6)
        initial_logits = self.initial_module.log_matrix(context=global_ctx)
        duration_logits = (self.duration_module.log_matrix(context=global_ctx) / temperature).clamp(-MAX_LOGITS, MAX_LOGITS)
        transition_logits = (self.transition_module.log_matrix(context=global_ctx) / temperature).clamp(-MAX_LOGITS, MAX_LOGITS)

        # --- Forward algorithm for each sequence ---
        alpha_list = self._forward(obs, theta=theta_list)
        log_likelihoods = []
        for alpha, L in zip(alpha_list, seq_lengths):
            if L == 0:
                log_likelihoods.append(torch.tensor(0.0, dtype=DTYPE))
            else:
                # sum over hidden states and durations
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
            X: Input sequence(s)
            algorithm: "viterbi" or "map"
            first_only: If True, return only first sequence (for single-sequence use)
                        If False, return list of arrays for all sequences
        Returns:
            NumPy array(s) of predicted states
        """
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE)

        preds = self.predict(X, algorithm=algorithm, context=None)
        preds_np = [p.cpu().numpy() for p in preds]
        return preds_np[0] if first_only else preds_np


