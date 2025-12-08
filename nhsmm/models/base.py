# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.context import ContextEncoder
from nhsmm import utils, constraints, SeedGenerator, ConvergenceTracker
from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger, MAX_LOGITS
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
        - `_compute_posteriors(X, theta)`: Returns γ (state marginals), ξ (transitions), η (state-duration posteriors).
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

        if hidden_dim is None:
            hidden_dim = context_dim
        elif hidden_dim != context_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal context_dim ({context_dim}) "
                "unless all modules explicitly define projection layers."
            )

        self.hidden_dim = hidden_dim

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
                ).to(device=self.device, dtype=DTYPE)

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

        self.to(device=self.device, dtype=DTYPE)

    def _init_modules(
        self,
        device: torch.device | str,
        n_states: int = 4,
        n_features: int = 1,
        max_duration: int = 30,
        temperature: float = 1.0,
        modulate_var: bool = False,
        min_covar: Optional[float] = 1e-6,
        transition_type: str = "ergodic",
        emission_type: str = "gaussian",
        init_mode_transition: str = "diag_bias",
        init_mode_duration: str = "uniform",
        init_mode_initial: str = "uniform",
        init_mode_emission: str = "data",
        cache_limit: int = 32,
        debug: bool = False,
        adaptive_scale: bool = True,
        dof: float = 5.0,
    ):
        device, debug = self.device, self.debug

        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            emission_type=emission_type,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            modulate_var=modulate_var,
            temperature=temperature,
            min_covar=min_covar,
        ).to(device)

        self.initial_module = Initial(
            n_states=self.n_states,
            init_mode=init_mode_initial,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
        ).to(device)

        self.duration_module = Duration(
            n_states=self.n_states,
            init_mode=init_mode_duration,
            max_duration=max_duration,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
        ).to(device)

        self.transition_module = Transition(
            n_states=self.n_states,
            n_features=self.n_features,
            init_mode=init_mode_transition,
            transition_type=transition_type,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature
        ).to(device)

        # Initialize distributions
        try:
            self._params.update({
                "initial_dist": self.initial_module.initialize(),
                "duration_dist": self.duration_module.initialize(),
                "transition_dist": self.transition_module.initialize(),
                "emission_dist": self.emission_module.initialize(),
            })
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {e}") from e

        if debug:
            logger.debug(
                f"HSMM modules initialized on {device}: "
                f"n_states={self.n_states}, n_features={self.n_features}, "
                f"context_dim={self.context_dim}, emission={self.emission_type}, "
                f"max_duration={self.max_duration}, precompute={self.precompute}"
            )

    def _encode(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None,
        detach: bool = True,
        debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence_aligned: [B, T, H]
            context_canonical: [B, 1, H]
        """

        device = sequences.device

        # ---------------------------
        # Standardize sequences: [B,T,F]
        # ---------------------------
        if sequences.ndim == 2:
            sequences = sequences.unsqueeze(0)
        B, T, F_in = sequences.shape

        # ---------------------------
        # Standardize mask: [B,T]
        # ---------------------------
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            mask = mask.bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            mask = mask[:, :T]

        # ---------------------------
        # No encoder → return zeros
        # ---------------------------
        if self.encoder is None or sequences.numel() == 0:
            ctx_dim = self.context_dim
            sequence_aligned = torch.zeros(B, T, ctx_dim, device=device)
            context_canonical = torch.zeros(B, 1, ctx_dim, device=device)
            return sequence_aligned, context_canonical

        # ---------------------------
        # Run ContextEncoder forward
        # ---------------------------
        kwargs = dict(
            mask=mask,
            return_context=True,
            return_sequence=True,
            detach_context=detach,
        )
        if pool is not None:
            kwargs["pool"] = pool

        seq_out, ctx_out, _ = self.encoder(sequences, **kwargs)
        # seq_out : [B,T,H]
        # ctx_out : [B,1,H] or None

        # ---------------------------
        # Validate encoder output
        # ---------------------------
        if seq_out is None:
            raise RuntimeError("Encoder did not return sequence features.")

        if seq_out.shape[1] != T:
            # This should never happen with your encoder +
            # ContextEncoder(return_sequence=True),
            # but we guard it instead of silently broadcasting.
            raise RuntimeError(
                f"Encoder returned sequence length {seq_out.shape[1]} != {T}"
            )

        sequence_aligned = seq_out

        # ---------------------------
        # Canonical context fallback (rare)
        # ---------------------------
        if ctx_out is None:
            # No canonical context returned → compute pooled mean
            pooled = (sequence_aligned * mask.unsqueeze(-1)).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled = pooled / denom
            context_canonical = pooled.unsqueeze(1)
        else:
            # Ensure shape [B,1,H]
            if ctx_out.ndim != 3 or ctx_out.shape[1] != 1:
                # ContextEncoder should already guarantee this, but we enforce correctness
                ctx_out = ctx_out.mean(dim=1, keepdim=True)
            context_canonical = ctx_out

        # ---------------------------
        # Adjust feature dimensionality to context_dim (if set)
        # ---------------------------
        target_dim = self.context_dim
        feat_dim = sequence_aligned.shape[-1]

        if feat_dim != target_dim:
            if feat_dim < target_dim:
                pad = target_dim - feat_dim
                sequence_aligned = F.pad(sequence_aligned, (0, pad))
                context_canonical = F.pad(context_canonical, (0, pad))
            else:
                sequence_aligned = sequence_aligned[:, :, :target_dim]
                context_canonical = context_canonical[:, :, :target_dim]

        # ---------------------------
        # Optional detach
        # ---------------------------
        if detach:
            sequence_aligned = sequence_aligned.detach()
            context_canonical = context_canonical.detach()

        return sequence_aligned, context_canonical

    def _prepare(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        debug: bool = True) -> utils.SequenceSet:
        """
        Prepare a SequenceSet with batch-aligned tensors and log-probabilities per state.
        Fully vectorized across batch and states.

        Returns:
            SequenceSet with:
                - sequences: [T, F] per sequence
                - log_probs: [T, K] per sequence
                - contexts: [B, T, H] or [B, 1, H]
                - masks: [B, T, 1]
        """
        device = X.device
        if X.ndim == 2:
            X = X.unsqueeze(0)  # [B=1, T, F]
            if debug:
                logger.debug(f"[Prepare] Added batch dimension: {X.shape}")

        B, T, F = X.shape

        # ---------------- Mask ----------------
        mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)
        if debug:
            logger.debug(f"[Prepare] Created mask: {mask.shape}")

        # ---------------- Context ----------------
        if theta is not None:
            context = theta
            if theta.ndim == 2:
                context = theta.unsqueeze(0)
            elif theta.shape[0] != B:
                context = theta.expand(B, -1, -1)
            ctx_canonical = context[:, :1, :]
            if debug:
                logger.debug(f"[Prepare] Using provided context: {context.shape}")
        else:
            context, ctx_canonical = self._encode(X)
            if context is None:
                ctx_dim = self.context_dim or F
                context = torch.zeros(B, T, ctx_dim, device=device)
                ctx_canonical = torch.zeros(B, 1, ctx_dim, device=device)
                if debug:
                    logger.debug(f"[Prepare] Encoder missing, using zeros for context with ctx_dim={ctx_dim}")
            else:
                if debug:
                    logger.debug(f"[Prepare] Encoded context: {context.shape}, canonical: {ctx_canonical.shape}")

        # ---------------- Compute log-probabilities ----------------
        log_probs = None
        K = self.n_states
        F_em = self.n_features

        if self.emission_module.emission_type in {"gaussian", "laplace", "studentt"}:
            # Continuous distributions
            dist = self.emission_module._get_dist(context=context, return_dist=True)
            if self.emission_module.emission_type == "gaussian":
                loc = dist.mean.unsqueeze(0).unsqueeze(0).expand(B, T, K, F_em)
                cov = dist.covariance_matrix.unsqueeze(0).unsqueeze(0).expand(B, T, K, F_em, F_em)
                # Compute log_prob using multivariate normal formula
                diff = (X.unsqueeze(2) - loc)  # [B, T, K, F]
                var = torch.diagonal(cov, dim1=-2, dim2=-1)  # [B, T, K, F]
                log_probs = -0.5 * (diff ** 2 / var).sum(-1) \
                            -0.5 * var.log().sum(-1) \
                            -0.5 * F_em * math.log(2 * math.pi)
            else:  # Laplace or StudentT
                log_probs = dist.log_prob(X.unsqueeze(2).expand(B, T, K, F_em))
        else:
            dist = self.emission_module._get_dist(context=context, return_dist=True)
            logits = getattr(dist, "logits", getattr(dist, "rate", None))  # [K, F] or [B,K,F]
            logits = logits.view(1, 1, K, F_em).expand(B, T, K, F_em)

            if self.emission_module.emission_type == "categorical":
                x_exp = X.long().unsqueeze(2).expand(B, T, K)
                log_probs_all = F.log_softmax(logits, dim=2)
                log_probs = torch.gather(log_probs_all, 2, x_exp).sum(-1)  # [B,T,K]

            elif self.emission_module.emission_type == "bernoulli":
                x_exp = X.unsqueeze(2).expand(B, T, K, F_em)
                log_probs = dist.log_prob(x_exp)  # [B,T,K]

            elif self.emission_module.emission_type == "poisson":
                x_exp = X.unsqueeze(2).expand(B, T, K, F_em)
                log_probs = dist.log_prob(x_exp)  # [B,T,K]

        # ---------------- Build SequenceSet ----------------
        sequences_list = [X[b] for b in range(B)]
        log_probs_list = [log_probs[b] for b in range(B)]
        lengths = [T] * B

        if debug:
            logger.debug(f"[Prepare] Prepared SequenceSet with {B} sequences, length={T}, features={F}")

        return utils.SequenceSet(
            sequences=sequences_list,
            log_probs=log_probs_list,
            contexts=context,
            lengths=lengths,
            masks=mask
        )

    def _ensure_shape(self, module, context: Optional[torch.Tensor] = None, T: Optional[int] = None) -> torch.Tensor:
        """
        Normalize module.log_matrix outputs to expected shape.

        Args:
            module: HSMM module (Initial, Duration, Transition, etc.).
            context: None | Tensor ([H], [T,H], [B,T,H]) | list of Tensors per sequence.
            T: explicit time length for expansion.

        Returns:
            Tensor with trailing dims matching module target shape and leading dims consistent with time.
        """
        if module is None:
            raise ValueError("_ensure_shape called with module=None")

        # --- Helper: list of context -> padded tensor [B,T,H] ---
        def _context_list_to_tensor(ctx_list):
            if len(ctx_list) == 0:
                return None
            H, max_T = None, 0
            for c in ctx_list:
                if c is None:
                    continue
                if not torch.is_tensor(c):
                    raise TypeError("Context list elements must be tensors or None")
                t, h = (1, c.shape[0]) if c.ndim == 1 else (c.shape[0], c.shape[1]) if c.ndim == 2 else (None, None)
                if t is None:
                    raise ValueError("Context list elements must be 1D or 2D tensors")
                max_T = max(max_T, t)
                H = h if H is None else H
                if H != h:
                    raise ValueError("Incompatible context feature dims inside list")
            B = len(ctx_list)
            device = next((c.device for c in ctx_list if c is not None), torch.device("cpu"))
            out = torch.zeros((B, max_T, H), dtype=DTYPE, device=device)
            for i, c in enumerate(ctx_list):
                if c is None:
                    continue
                if c.ndim == 1:
                    out[i, 0] = c.to(dtype=DTYPE, device=device)
                else:
                    L = c.shape[0]
                    out[i, :L] = c.to(dtype=DTYPE, device=device)
            return out

        # --- Normalize context ---
        ctx = context
        if isinstance(ctx, list):
            ctx = _context_list_to_tensor(ctx)
        elif ctx is not None and not torch.is_tensor(ctx):
            raise TypeError("_ensure_shape context must be tensor, list, or None")

        # --- Determine target trailing shape ---
        if hasattr(module, "_shape") and module._shape is not None:
            target_shape = [int(module._shape)] if isinstance(module._shape, int) else [int(x) for x in module._shape]
        else:
            if isinstance(module, Initial):
                target_shape = [self.n_states]
            elif isinstance(module, Duration):
                target_shape = [self.n_states, self.max_duration]
            elif isinstance(module, Transition):
                target_shape = [self.n_states, self.n_states]

        tal = len(target_shape)

        # --- Determine time length T if not given ---
        if T is None and ctx is not None:
            if ctx.ndim == 1:
                T = 1
            elif ctx.ndim == 2:
                T = ctx.shape[0]
            elif ctx.ndim == 3:
                T = ctx.shape[1]
            else:
                raise ValueError("Unsupported context ndim")

        # --- Call module ---
        if isinstance(module, Duration):
            # Duration logits are either static [K,D] or batch [B,1,K,D], don't index timestep
            x = module.log_matrix(context=ctx)
        elif isinstance(module, Transition):
            # Only pass timestep if context is [T,H] or [B,T,H], not static/batch
            timestep_for_modulate = None
            if ctx is not None and ctx.ndim == 2:
                timestep_for_modulate = T
            x = module.log_matrix(context=ctx, timestep=timestep_for_modulate)
        else:
            # Initial or other modules: safe to pass T
            x = module.log_matrix(context=ctx, timestep=T)

        if not torch.is_tensor(x):
            raise TypeError(f"log_matrix must return tensor, got {type(x)}")

        # --- Minimal fix: collapse accidental time dimension ---
        # Prevent Duration/Transition from returning [T, ...] blocks when _forward
        # expects per-step static logits.
        if T is not None and x.ndim >= 2:
            if x.shape[0] == T:
                # Use T-1 slice (consistent with how timestep=T is interpreted in modulate)
                x = x[T - 1]

        # --- Verify trailing dims ---
        if list(x.shape[-tal:]) != target_shape:
            if x.numel() == int(torch.tensor(target_shape).prod().item()):
                x = x.view(*target_shape)
            else:
                raise RuntimeError(
                    f"{type(module).__name__}.log_matrix returned trailing dims "
                    f"{tuple(x.shape[-tal:])}, expected {tuple(target_shape)}"
                )

        # --- Ensure leading/time dim ---
        leading_nd = x.ndim - tal
        if leading_nd == 0:
            x = x.unsqueeze(0)
            leading_nd = 1

        if leading_nd == 1 and T is not None and x.shape[0] == 1:
            x = x.expand(T, *x.shape[1:])
        elif leading_nd >= 2 and T is not None:
            if x.shape[0] == 1 and x.shape[1] == T:
                x = x.squeeze(0)
            elif x.shape[0] == T:
                pass
            elif x.shape[0] == 1:
                x = x.expand(T, *x.shape[1:])
            else:
                raise RuntimeError(
                    f"{type(module).__name__}.log_matrix leading dims "
                    f"{tuple(x.shape[:-tal])} incompatible with T={T}"
                )

        # --- Final check ---
        if list(x.shape[-tal:]) != target_shape:
            raise RuntimeError(
                f"Post-processed {type(module).__name__} shape {tuple(x.shape)} "
                f"trailing dims != {tuple(target_shape)}"
            )

        return x

    def _forward(
        self,
        X: utils.SequenceSet,
        theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None) -> list[torch.Tensor]:

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
            initial_logits = self._ensure_shape(self.initial_module, ctx_seq, T)  # [T,K] or [K]
            duration_logits = self._ensure_shape(self.duration_module, ctx_seq, T)  # [T,K,Dmax]
            transition_logits = self._ensure_shape(self.transition_module, ctx_seq, T)  # [T,K,K]

            # ---------------- Cumulative emission sums ----------------
            cumsum_emit = torch.zeros(T + 1, K, dtype=DTYPE, device=device)
            cumsum_emit[1:] = torch.cumsum(log_emissions, dim=0)

            dur_range = torch.arange(1, Dmax + 1, device=device).view(1, Dmax)  # [1,Dmax]
            time_idx = torch.arange(T, device=device).view(T, 1)  # [T,1]
            start_idx = (time_idx - dur_range + 1).clamp(min=0)  # [T,Dmax]

            start_idx_exp = start_idx.unsqueeze(2).expand(T, Dmax, K)  # [T,Dmax,K]
            end_idx_exp = (time_idx + 1).unsqueeze(2).expand(T, 1, K)  # [T,1,K]
            cumsum_emit_exp = cumsum_emit.unsqueeze(0).expand(T, T + 1, K)  # [T,T+1,K]
            emit_sums = cumsum_emit_exp.gather(1, end_idx_exp) - cumsum_emit_exp.gather(1, start_idx_exp)
            emit_sums = emit_sums.permute(0, 2, 1)  # [T,K,Dmax]

            # ---------------- Forward recursion ----------------
            alpha_tensor = torch.full((T, K, Dmax), neg_inf, dtype=DTYPE, device=device)

            for t in range(T):
                max_d = min(Dmax, t + 1)

                if t == 0:
                    alpha_tensor[t, :, :max_d] = (
                        initial_logits[t].unsqueeze(1)  # [K,1]
                        + duration_logits[t, :, :max_d]  # [K,max_d]
                        + emit_sums[t, :, :max_d]        # [K,max_d]
                    )
                    continue

                prev_alpha = torch.full((max_d, K), neg_inf, dtype=DTYPE, device=device)
                valid_mask = start_idx[t, :max_d] > 0
                initial_mask = ~valid_mask

                # --- Case 1: valid previous segments ---
                if valid_mask.any():
                    idx = valid_mask.nonzero(as_tuple=True)[0]
                    prev_positions = start_idx[t, idx] - 1
                    prev_vals = torch.logsumexp(alpha_tensor[prev_positions, :, :], dim=2)
                    prev_alpha[idx] = prev_vals

                # --- Case 2: segment begins at t=0 ---
                if initial_mask.any():
                    idx = initial_mask.nonzero(as_tuple=True)[0]
                    prev_alpha[idx] = (
                        initial_logits[t].unsqueeze(0)                   # [1,K]
                        + duration_logits[t, :, :max_d][:, idx].T        # [len(idx),K]
                    )

                # --- Transition step ---
                alpha_trans = torch.logsumexp(
                    prev_alpha.unsqueeze(2) + transition_logits[t].unsqueeze(0), dim=1
                )  # [max_d,K]

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
        theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None) -> list[torch.Tensor]:
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
            init_logits = self._ensure_shape(self.initial_module, ctx_seq, T)     # [T,K]
            dur_logits = self._ensure_shape(self.duration_module, ctx_seq, T)     # [T,K,Dmax]
            trans_logits = self._ensure_shape(self.transition_module, ctx_seq, T) # [T,K,K]

            # 3. Cumulative emission sums
            cumsum_emit = torch.zeros(T + 1, K, dtype=DTYPE, device=device)
            cumsum_emit[1:] = torch.cumsum(seq_logp, dim=0)

            dur_range = torch.arange(1, Dmax + 1, device=device)                        # [Dmax]
            ends = torch.arange(T, device=device).unsqueeze(1) + dur_range.unsqueeze(0) # [T,Dmax]
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

    def _compute_posteriors(self, X: utils.SequenceSet, theta: Optional[ContextFeatures] = None):
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

        if B == 0 or T_max == 0:
            gamma = torch.zeros((B, T_max, K), dtype=DTYPE, device=device)
            eta = torch.zeros((B, T_max, K, Dmax), dtype=DTYPE, device=device)
            xi = torch.zeros((B, max(T_max-1,0), K, K), dtype=DTYPE, device=device)
            return gamma, xi, eta

        # -------------------- Pad log_probs --------------------
        logp_padded = torch.full((B, T_max, K), neg_inf, dtype=DTYPE, device=device)
        mask = torch.zeros(B, T_max, dtype=DTYPE, device=device)
        for b, (logp, L) in enumerate(zip(X.log_probs, X.lengths)):
            if L > 0:
                logp_padded[b, :L] = logp
                mask[b, :L] = 1.0
        mask_bool = mask.bool()

        # -------------------- Align context --------------------
        ctx_aligned = None
        if theta is not None:
            if isinstance(theta, list):
                ctx_dim = theta[0].shape[-1]
                ctx_aligned = torch.zeros((B, T_max, ctx_dim), dtype=DTYPE, device=device)
                for b, ctx_seq in enumerate(theta):
                    L = X.lengths[b]
                    if L > 0:
                        ctx_aligned[b, :L] = ctx_seq
            else:
                ctx_aligned = theta
                if ctx_aligned.shape[1] != T_max:
                    raise ValueError("Context time dimension mismatch with max sequence length")

        # -------------------- Forward / Backward --------------------
        alpha_list, beta_list = self._forward(X, theta=ctx_aligned), self._backward(X, theta=ctx_aligned)

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
        eta = eta * mask.unsqueeze(-1).unsqueeze(-1)

        # -------------------- Gamma --------------------
        gamma = eta.sum(-1)
        gamma = gamma / gamma.sum(-1, keepdim=True).clamp_min(EPS)
        gamma = gamma * mask.unsqueeze(-1)

        # -------------------- Xi --------------------
        xi = torch.zeros((B, max(T_max-1,0), K, K), dtype=DTYPE, device=device)
        for b, L in enumerate(X.lengths):
            if L <= 1:
                continue

            # Transition logits
            trans_logits = self._ensure_shape(self.transition_module, ctx_aligned[b] if ctx_aligned is not None else None)
            if trans_logits.ndim == 2:  # static
                trans_seq = trans_logits.unsqueeze(0).expand(L-1, -1, -1)
            else:
                trans_seq = trans_logits[:L-1]

            # Compute xi in log-space safely
            a_prev = torch.logsumexp(alpha_list[b], dim=-1)[:L-1]       # [T-1, K]
            b_next = torch.logsumexp(beta_list[b], dim=-1)[1:L]         # [T-1, K]
            log_xi = a_prev.unsqueeze(2) + trans_seq + b_next.unsqueeze(1)  # [T-1, K, K]
            log_xi -= torch.logsumexp(log_xi.reshape(-1), dim=0)  # normalize
            xi[b, :L-1] = log_xi.exp()

        return gamma, xi, eta

    def _model_params(
        self,
        X: Optional[utils.SequenceSet] = None,
        theta: Optional[torch.Tensor] = None,
        mode: str = "estimate",
        iter_idx: int = 0) -> dict[str, Any]:
        """
        Compute HSMM model parameters with EM-style estimate, sample-mode collapse, and neural updates.
        Returns a dict of distributions: emission, initial, duration, transition.
        """

        max_iter: int = 50
        α_min, α_max = 0.01, self.alpha
        α = max(α_min, α_max * (1.0 - iter_idx / max_iter))

        # -------------------- Context dimension checks --------------------
        def _check_context_hidden_dim(context: Optional[torch.Tensor] = None):
            ctx_dim = self.context_dim
            if context is not None:
                ctx_dim = context.shape[-1]  # last dimension is feature
            if self.hidden_dim is not None and ctx_dim != self.hidden_dim:
                raise RuntimeError(
                    f"Context dimension mismatch: inferred context_dim={ctx_dim}, "
                    f"hidden_dim={self.hidden_dim}. Check encoder output or hidden_dim setting."
                )

        # Determine context to use
        if theta is not None:
            _check_context_hidden_dim(theta)
        elif hasattr(self, "_context") and self._context is not None:
            _check_context_hidden_dim(self._context)
        else:
            _check_context_hidden_dim()

        # -------------------- Encode / Align context --------------------
        if theta is not None:
            context_aligned = theta
            if theta.ndim == 2:  # [T,H] -> [1,T,H]
                context_aligned = theta.unsqueeze(0)
            elif X is not None and theta.shape[0] != len(X.sequences):
                context_aligned = theta.expand(len(X.sequences), -1, -1)
            ctx_canonical = context_aligned.mean(dim=1, keepdim=True)

        elif X is not None:
            seqs = X.sequences
            B = len(seqs)
            lengths = [s.shape[0] for s in seqs]
            T_max = max(lengths)
            padded = torch.zeros(B, T_max, self.n_features, device=self.device, dtype=DTYPE)
            mask = torch.zeros(B, T_max, dtype=torch.bool, device=self.device)
            for b, s in enumerate(seqs):
                L = s.shape[0]
                padded[b, :L] = s.to(device=self.device, dtype=DTYPE)
                mask[b, :L] = 1

            context_aligned, ctx_canonical = self._encode(padded, mask=mask, detach=True)
            # context_aligned: [B, T, H], ctx_canonical: [B, 1, H]

        else:
            context_aligned = torch.zeros(1, 1, self.context_dim, device=getattr(self, "device", "cpu"))
            ctx_canonical = context_aligned

        # -------------------- Helper functions --------------------
        def collapse_logits(
            logits: torch.Tensor, 
            dim: int, 
            alpha_scale: float = 1.0, 
            softmax_temp: float = 0.85, 
            eps_collapse: float = 1e-9) -> torch.Tensor:

            probs = F.softmax(logits / softmax_temp, dim=dim)
            dir_alpha = (probs * alpha_scale).clamp_min(eps_collapse)

            # Flatten all dims except target for batch Dirichlet sampling
            permute_order = list(range(logits.ndim))
            permute_order[dim], permute_order[-1] = permute_order[-1], permute_order[dim]
            flat_alpha = dir_alpha.permute(permute_order).reshape(-1, dir_alpha.shape[dim])

            # Sample from Dirichlet
            samples = torch.distributions.Dirichlet(flat_alpha).rsample()
            samples = samples.reshape(*[dir_alpha.shape[i] for i in permute_order[:-1]], dir_alpha.shape[dim])
            samples = samples.permute(permute_order)  # restore original dim order

            # Log-normalize safely
            log_probs = torch.log(samples / samples.sum(dim=dim, keepdim=True).clamp_min(EPS))
            return log_probs

        def safe_sum(lst: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            tensors = [t for t in lst if t is not None and t.numel() > 0]
            if not tensors: return None
            return torch.cat(tensors, dim=0).sum(dim=0)

        if mode == "sample":
            with torch.no_grad():
                # Timestep-dependent emissions: use per-timestep context
                emission_dist = self.emission_module.forward(context=context_aligned, return_dist=True)

                # Stationary HSMM parameters: use sequence-level context
                initial_dist = self.initial_module.forward(context=ctx_canonical, return_dist=True)
                transition_dist = self.transition_module.forward(context=ctx_canonical, return_dist=True)
                duration_dist = self.duration_module.forward(context=ctx_canonical, return_dist=True)

                # Collapse logits for stability
                self.initial_module.update(
                    new_logits=collapse_logits(initial_dist.logits, dim=0).exp(),
                    from_probs=True
                )
                self.transition_module.update(
                    new_logits=collapse_logits(transition_dist.logits, dim=1).exp(),
                    from_probs=True
                )
                self.duration_module.update(
                    new_logits=collapse_logits(duration_dist.logits, dim=1).exp(),
                    from_probs=True
                )

        elif mode == "estimate":
            if X is None or not isinstance(X, utils.SequenceSet):
                raise RuntimeError("SequenceSet X required for estimate mode.")

            # Emissions: per-timestep
            try:
                emission_dist = self.emission_module.forward(context=context_aligned, return_dist=True)
            except Exception as err:
                logger.warning("Emission initialize fallback: %s", err)
                all_X = torch.cat([s for s in X.sequences], dim=0) if X else None
                emission_dist = self.emission_module.initialize(X=all_X, context=context_aligned)

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=context_aligned)

            # Posterior counts
            init_counts = safe_sum([g[0] for g in gamma_list])
            init_counts = init_counts if init_counts is not None else self.initial_module.logits.exp()

            trans_counts = safe_sum(xi_list)
            trans_counts = trans_counts if trans_counts is not None else self.transition_module.logits.exp()

            dur_counts = safe_sum(eta_list)
            dur_counts = dur_counts if dur_counts is not None else self.duration_module.logits.exp()

            # α-blended log normalization
            blend_init = constraints.log_normalize(
                torch.log(init_counts + EPS) * α + (1 - α) * self.initial_module.logits, dim=0
            )
            blend_trans = constraints.log_normalize(
                torch.log(trans_counts + EPS) * α + (1 - α) * self.transition_module.logits, dim=1
            )
            blend_dur = constraints.log_normalize(
                torch.log(dur_counts + EPS) * α + (1 - α) * self.duration_module.logits, dim=1
            )

            # Update stationary modules with sequence-level context
            self.initial_module.update(new_logits=blend_init.exp(), context=ctx_canonical, from_probs=True)
            self.transition_module.update(new_logits=blend_trans.exp(), context=ctx_canonical, from_probs=True)
            self.duration_module.update(new_logits=blend_dur.exp(), context=ctx_canonical, from_probs=True)

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        return {
            "initial_dist": self.initial_module.forward(context=ctx_canonical, return_dist=True),
            "duration_dist": self.duration_module.forward(context=ctx_canonical, return_dist=True),
            "transition_dist": self.transition_module.forward(context=ctx_canonical, return_dist=True),
            "emission_dist": emission_dist,
        }


    # HSMM EM
    def _map(self, X: utils.SequenceSet) -> list[torch.Tensor]:
        """
        MAP decoding of HSMM sequences using posterior state marginals.
        Returns a list of tensors, each of shape [T].
        """
        gamma_list, _, _ = self._compute_posteriors(X)
        results = []
        for gamma in gamma_list:
            if gamma is None or gamma.numel() == 0:
                results.append(torch.empty(0, dtype=torch.long))
                continue

            gamma = torch.nan_to_num(gamma, nan=-1e9, posinf=-1e9, neginf=-1e9)
            map_seq = gamma.argmax(dim=-1)
            results.append(map_seq.to(dtype=torch.long))

        return results

    def _viterbi(self, X: utils.SequenceSet, theta: Optional[torch.Tensor] = None, duration_weight: float = 0.0) -> list[torch.Tensor]:
        """
        Time-aware Viterbi for HSMM with per-timestep logits and optional duration weighting.
        Supports theta as a list of sequence tensors or a single tensor [B, T, H].
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        predicted_sequences: list[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, dtype=torch.int64)

        # Preprocess theta: list -> padded tensor
        if isinstance(theta, list) and len(theta) > 0 and isinstance(theta[0], torch.Tensor):
            max_len = max(t.shape[0] for t in theta)
            theta_tensor = torch.stack([torch.cat([t, torch.zeros((max_len - t.shape[0], t.shape[1]), dtype=t.dtype, device=t.device)]) if t.shape[0] < max_len else t for t in theta], dim=0)
        else:
            theta_tensor = theta

        for b, seq in enumerate(X.sequences):
            L = seq.shape[0]
            device = seq.device

            if L == 0:
                predicted_sequences.append(torch.empty(0, dtype=torch.int64, device=device))
                continue

            ctx_seq = theta_tensor[b] if theta_tensor is not None else None

            # --- Module logits ---
            init_logits = self._ensure_shape(self.initial_module, ctx_seq, L)     # [L, K]
            dur_logits = self._ensure_shape(self.duration_module, ctx_seq, L)     # [L, K, Dmax]
            trans_logits = self._ensure_shape(self.transition_module, ctx_seq, L) # [L, K, K]

            # --- Emission log-probs ---
            emit_log = X.log_probs[b].to(device)
            cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=DTYPE), torch.cumsum(emit_log, dim=0)))

            # --- Viterbi tensors ---
            V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.int64)
            best_durations = torch.zeros((L, K), dtype=torch.int64, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                ini = init_logits[t]
                dur_scores = dur_logits[t, :, :max_d] * (1.0 - duration_weight) if duration_weight != 0.0 else dur_logits[t, :, :max_d]
                trans_t = trans_logits[t]

                if t == 0:
                    scores = ini.unsqueeze(1) + dur_scores + emit_sums
                    best_score, best_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_idx]
                    back_ptr[t] = -1
                    continue

                prev_scores = V[torch.clamp(starts - 1, min=0)].T.unsqueeze(2) + trans_t.unsqueeze(1)
                # handle initial state when start == 0
                prev_scores[:, starts == 0, :] = ini.unsqueeze(0).expand_as(prev_scores[:, starts == 0, :])

                prev_max, prev_arg = prev_scores.max(dim=0)  # max over previous states
                scores = prev_max.T + dur_scores + emit_sums
                best_score, best_d_idx = scores.max(dim=1)

                V[t] = best_score
                best_durations[t] = durations[best_d_idx]
                back_ptr[t] = torch.where(best_durations[t] == 1, torch.full_like(best_durations[t], -1), prev_arg[best_d_idx, torch.arange(K, device=device)])

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
            seq_path = torch.cat([torch.full((end - start + 1,), st, dtype=torch.int64, device=device) for start, end, st in segments])
            predicted_sequences.append(seq_path[:L])

        return predicted_sequences

    @torch.no_grad()
    def _compute_emit_log(self, X: utils.SequenceSet, theta: Optional[torch.Tensor] = None, verbose: bool = False) -> torch.Tensor:
        """
        Compute per-sequence emission log-likelihoods for an HSMM with optional context.
        Handles variable-length sequences, zero-length sequences, and proper masking.
        """
        batch_size = len(X.sequences)
        neg_inf = torch.finfo(DTYPE).min / 2.0
        if batch_size == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, dtype=DTYPE, device=self.device)
            return X.log_likelihoods

        # Prepare per-sequence context
        if theta is not None:
            context_batch = theta.unsqueeze(0) if theta.ndim == 2 else theta
            if context_batch.shape[0] != batch_size:
                context_batch = context_batch.expand(batch_size, -1, -1)
        else:
            max_len = max(X.lengths)
            seq_dim = X.sequences[0].shape[-1] if X.sequences else self.n_features
            padded_seqs = torch.zeros(batch_size, max_len, seq_dim, device=self.device, dtype=DTYPE)
            mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
            for i, seq in enumerate(X.sequences):
                if seq.shape[0] > 0:
                    padded_seqs[i, :seq.shape[0]] = seq.to(self.device, dtype=DTYPE)
                    mask[i, :seq.shape[0]] = 1
            context_batch, _ = self._encode(padded_seqs, mask=mask, detach=True)

        # Forward pass
        alpha_list = self._forward(X, theta=context_batch)

        # Determine shapes
        max_len = max(X.lengths)
        n_states, n_durations = self.n_states, self.max_duration
        if alpha_list:
            alpha_sample = alpha_list[0]
            if alpha_sample.ndim == 3: n_states, n_durations = alpha_sample.shape[1], alpha_sample.shape[2]
            if alpha_sample.ndim == 4: n_states, n_durations = alpha_sample.shape[2], alpha_sample.shape[3]

        # Build padded alpha tensor
        alpha_padded = torch.full((batch_size, max_len, n_states, n_durations), neg_inf, dtype=DTYPE, device=self.device)
        for i, (alpha_seq, L) in enumerate(zip(alpha_list, X.lengths)):
            if L > 0:
                alpha_padded[i, :L] = alpha_seq[:L] if alpha_seq.ndim == 3 else alpha_seq[i, :L]

        # Compute per-sequence log-likelihood
        lengths_tensor = torch.tensor(X.lengths, dtype=torch.long, device=self.device)
        valid = lengths_tensor > 0
        log_likelihoods = torch.full((batch_size,), neg_inf, dtype=DTYPE, device=self.device)
        if valid.any():
            last_alpha = alpha_padded[valid, lengths_tensor[valid]-1]
            log_likelihoods[valid] = torch.logsumexp(last_alpha.reshape(last_alpha.size(0), -1), dim=1)

        if verbose:
            logger.info(f"[compute_emit_log] batch={batch_size}, min={log_likelihoods.min():.4f}, max={log_likelihoods.max():.4f}, mean={log_likelihoods.mean():.4f}")

        X.log_likelihoods = log_likelihoods
        return log_likelihoods

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        patience: int = 1,
        max_iter: int = 20,
        ignore_conv: bool = False,
        theta: Optional[torch.Tensor] = None,
        update_rate_max: float = 0.9,
        update_rate_min: float = 0.1,
        adapt_factor: float = 10.0,
        plot_conv: bool = False,
        verbose: bool = True):
        """Fit HSMM using EM with optional context-modulated updates."""

        # ---------------- Encode context if needed ----------------
        if theta is None and getattr(self, "encoder", None) is not None:
            theta, _ = self._encode(X)

        # ---------------- Prepare sequences ----------------
        X_valid = self._prepare(X, theta=theta)
        B = len(X_valid.sequences)
        max_len = max(X_valid.lengths) if B > 0 else 0
        F_dim = X_valid.sequences[0].shape[-1] if B > 0 else 0
        device = X_valid.sequences[0].device if B > 0 else torch.device("cpu")

        seq_tensor = torch.zeros((B, max_len, F_dim), dtype=DTYPE, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for b, seq in enumerate(X_valid.sequences):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq
            mask[b, :L] = True
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

            init_pdf = params["initial_dist"]
            duration_dist = params["duration_dist"]
            emission_dist = params["emission_dist"]
            transition_dist = params["transition_dist"]

            # Precompute initial log_probs and likelihood
            log_probs_tensor = emission_dist.log_prob(seq_tensor.unsqueeze(2)) * mask_exp
            prev_ll = log_probs_tensor.sum().item()
            self._convergence.update(prev_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # -------- Compute posteriors --------
                gamma_list, xi_list, eta_list = self._compute_posteriors(X_valid, theta=theta)

                # -------- Preallocate batch tensors --------
                K, Dmax = self.n_states, self.max_duration
                max_len_gamma = max([g.shape[0] for g in gamma_list], default=0)
                max_len_eta = max([e.shape[0] for e in eta_list], default=0)
                max_len_xi = max([x.shape[0] if x is not None else 0 for x in xi_list], default=0)

                gamma_tensor = torch.zeros(B, max_len_gamma, K, dtype=DTYPE, device=device)
                eta_tensor = torch.zeros(B, max_len_eta, K, Dmax, dtype=DTYPE, device=device)
                xi_tensor = torch.zeros(B, max_len_xi, K, K, dtype=DTYPE, device=device) if max_len_xi > 0 else None

                for b in range(B):
                    Lg = gamma_list[b].shape[0]
                    gamma_tensor[b, :Lg] = gamma_list[b]
                    Le = eta_list[b].shape[0]
                    eta_tensor[b, :Le] = eta_list[b]
                    if xi_tensor is not None and xi_list[b] is not None:
                        Lx = xi_list[b].shape[0]
                        xi_tensor[b, :Lx] = xi_list[b]

                gamma_tensor *= mask_exp
                eta_tensor *= mask_exp.unsqueeze(-1)

                # -------- EM sufficient statistics --------
                init_counts = gamma_tensor.sum((0, 1))
                dur_counts = eta_tensor.sum((0, 1)) if eta_tensor is not None else duration_dist.expected_probs()
                trans_counts = xi_tensor.sum((0, 1)) if xi_tensor is not None else transition_dist.expected_probs()

                # Normalize counts
                init_counts /= init_counts.sum().clamp_min(EPS)
                dur_counts /= dur_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
                trans_counts /= trans_counts.sum(dim=1, keepdim=True).clamp_min(EPS)

                # -------- EM alpha decay --------
                α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
                α = max(α_min, α_max * (1.0 - it / max_iter))

                # Update categorical distributions
                init_pdf = self.initial_module._dist(
                    probs=α * init_counts + (1 - α) * self.initial_module.expected_probs()
                )
                duration_dist = self.duration_module._dist(
                    probs=α * dur_counts + (1 - α) * self.duration_module.expected_probs()
                )
                transition_dist = self.transition_module._dist(
                    probs=α * trans_counts + (1 - α) * self.transition_module.expected_probs()
                )

                # -------- Flatten for emission updates --------
                flat_mask = mask.view(-1)
                flat_X = seq_tensor.reshape(-1, F_dim)[flat_mask]
                flat_gamma = gamma_tensor.reshape(-1, K)[flat_mask]
                flat_theta = theta.reshape(-1, theta.shape[-1])[flat_mask] if theta is not None else None

                # -------- Adaptive learning rate --------
                delta_ll = max(log_probs_tensor.sum().item() - prev_ll, 0.0)
                adaptive_rate = min(update_rate_max, max(update_rate_min, adapt_factor * delta_ll))
                update_rate_final = α * adaptive_rate + (1 - α) * update_rate_min

                # -------- Neural or EM emission update --------
                if neural_update:
                    self.emission_module.zero_grad()
                    emission_loss = -(flat_gamma * self.emission_module.log_prob(flat_X)).sum() / flat_gamma.sum()
                    emission_loss.backward()
                    with torch.no_grad():
                        for p in self.emission_module.parameters():
                            if p.grad is not None:
                                p.add_(update_rate_final * p.grad)
                    emission_dist = self.emission_module.forward(context=flat_theta, return_dist=True)
                else:
                    emission_dist = self.emission_module.update(
                        X=flat_X, posterior=flat_gamma, theta=flat_theta, update_rate=update_rate_final
                    )

                self._params["emission_dist"] = emission_dist

                # -------- Likelihood & convergence --------
                log_probs_tensor = emission_dist.log_prob(seq_tensor.unsqueeze(2)) * mask_exp / F_dim
                curr_ll = log_probs_tensor.sum().item()
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
                    "initial_dist": init_pdf,
                    "transition_dist": transition_dist,
                    "duration_dist": duration_dist,
                    "emission_dist": emission_dist
                })

        if plot_conv:
            self._convergence.plot()

        return self

    def predict(
        self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        transition_temp: float = 1.0,
        duration_temp: float = 1.0) -> list[torch.Tensor]:
        """
        Predict hidden states for HSMM sequences using Viterbi or MAP decoding.
        Supports batched, variable-length sequences with optional context modulation.

        Args:
            X: Observation tensor or list of tensors for each sequence.
            algorithm: Decoding method, either "viterbi" or "map".
            context: Optional context tensor or list for each sequence.
            transition_temp: Temperature scaling for transition probabilities.
            duration_temp: Temperature scaling for duration probabilities.

        Returns:
            List of torch.LongTensor, each tensor containing predicted states for a sequence.
        """
        # --- Prepare observations ---
        obs_set = self._prepare(X, theta=context)
        num_sequences = len(obs_set.sequences)
        sequence_lengths = [seq.shape[0] for seq in obs_set.sequences]
        max_seq_len = max(sequence_lengths) if sequence_lengths else 0
        num_states = self.n_states

        if max_seq_len == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(num_sequences)]

        # --- Prepare per-sequence context safely ---
        if context is None:
            context_list = [None] * num_sequences
        elif isinstance(context, list):
            context_list = context
        else:
            if context.ndim == 3 and context.shape[0] == num_sequences:
                context_list = [context[b].unsqueeze(0).clone() for b in range(num_sequences)]
            else:
                context_list = [context.clone() for _ in range(num_sequences)]

        # --- Viterbi decoding ---
        if algorithm.lower() == "viterbi":
            decoded_sequences = self._viterbi(obs_set, theta=context_list)
            return [seq.detach().to(dtype=torch.long) for seq in decoded_sequences]

        # --- MAP decoding ---
        elif algorithm.lower() == "map":
            decoded_sequences: list[torch.Tensor] = []

            for b, seq_len in enumerate(sequence_lengths):
                if seq_len == 0:
                    decoded_sequences.append(torch.empty(0, dtype=torch.long, device=self.device))
                    continue

                theta_b = context_list[b]

                # Compute logits for this sequence
                init_logits = self.initial_module.log_matrix(context=theta_b).clamp(-MAX_LOGITS, MAX_LOGITS)
                trans_logits = (self.transition_module.log_matrix(context=theta_b) / max(transition_temp, 1e-6)).clamp(-MAX_LOGITS, MAX_LOGITS)
                dur_logits = (self.duration_module.log_matrix(context=theta_b) / max(duration_temp, 1e-6)).clamp(-MAX_LOGITS, MAX_LOGITS)

                # Ensure correct shape: remove batch dim if present
                if init_logits.ndim > 1:
                    init_logits = init_logits.view(num_states)
                if trans_logits.ndim > 2:
                    trans_logits = trans_logits.view(num_states, num_states)
                if dur_logits.ndim > 2:
                    dur_logits = dur_logits.view(num_states, self.max_duration)

                log_probs_seq = obs_set.log_probs[b]  # [T, K] expected
                decoded_seq = self._map_decode(
                    log_probs_seq,
                    init_logits,
                    trans_logits,
                    dur_logits
                )
                decoded_sequences.append(decoded_seq.detach().to(dtype=torch.long))

            return decoded_sequences

        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

    @torch.no_grad()
    def score(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute log-likelihoods for HSMM sequences with optional context.
        
        Args:
            X: Tensor [T,F] or list of Tensors [T_b,F] representing sequences.
            theta: Optional context tensor or list of tensors, shape [T,H] or [B,T,H].
        
        Returns:
            Tensor of shape [B], log-likelihood per sequence.
        """
        # --- Normalize sequences to list ---
        sequences = [X] if torch.is_tensor(X) else list(X)
        num_sequences = len(sequences)

        if num_sequences == 0:
            return torch.empty(0, dtype=DTYPE)

        # --- Normalize context to per-sequence list ---
        if theta is None:
            context_list: list[Optional[torch.Tensor]] = [None] * num_sequences
        elif torch.is_tensor(theta):
            # split along time dimension if a single concatenated tensor is provided
            lengths = [seq.shape[0] for seq in sequences]
            cum_lengths = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(lengths, dtype=torch.long).cumsum(0)])
            context_list = [theta[cum_lengths[b]:cum_lengths[b+1]]
                            for b in range(num_sequences)]
        elif isinstance(theta, list):
            context_list = [
                t if isinstance(t, torch.Tensor) else None for t in theta
            ]
            if len(context_list) != num_sequences:
                raise ValueError(f"Length of theta list ({len(context_list)}) != number of sequences ({num_sequences})")
        else:
            raise TypeError(f"Unsupported theta type: {type(theta)}")

        # --- Concatenate all sequences for emission log-prob computation ---
        concat_sequences = torch.cat([seq for seq in sequences], dim=0)
        non_none_context = [c for c in context_list if c is not None]
        concat_context = torch.cat(non_none_context, dim=0) if non_none_context else None

        # Compute emission log-probs for all concatenated sequences
        emission_log_probs = self.emission_module.log_prob(concat_sequences, context=concat_context)

        # Split emission log-probs per sequence
        seq_lengths = [seq.shape[0] for seq in sequences]
        emission_log_probs_per_seq = list(torch.split(emission_log_probs, seq_lengths, dim=0))

        # Wrap into SequenceSet for forward pass
        obs_set = utils.SequenceSet(sequences, log_probs=emission_log_probs_per_seq)

        # Compute alpha (forward probabilities) per sequence
        alpha_list = self._forward(obs_set, theta=context_list)

        # --- Compute log-likelihood per sequence ---
        log_likelihoods = []
        for alpha, seq_len in zip(alpha_list, seq_lengths):
            if seq_len == 0:
                log_likelihoods.append(torch.tensor(0.0, dtype=DTYPE))
            else:
                # logsumexp over states and durations at final time step
                log_likelihoods.append(torch.logsumexp(alpha[seq_len-1], dim=(-2, -1)))

        return torch.stack(log_likelihoods)

    @torch.no_grad()
    def info(
        self,
        X: torch.Tensor,
        criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True) -> torch.Tensor:
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
        first_only: bool = True) -> np.ndarray | list[np.ndarray]:
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

