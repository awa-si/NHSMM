# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger, MAX_LOGITS
from nhsmm.distributions import Initial, Emission, Duration, Transition
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm import constraints, SeedGenerator, ConvergenceTracker


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
        - `_forward(X, theta)`: α[t,k,d] for ending in state k at time t with duration d.
        - `_backward(X, theta)`: β[t,k,d] for observations from time t onward.
        - `_compute_posteriors(X, theta)`: γ (state marginals), ξ (transitions), η (state-duration posteriors).
        - `_viterbi(X, theta, duration_weight)`: Most likely state sequence, optionally weighting durations.

    Notes:
        - Supports discrete (Categorical) and continuous (Normal/MultivariateNormal) emissions.
        - Fully batched and GPU-ready.
        - Extensible for custom neural/context-modulated emission modules.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        n_heads: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        min_covar: float = 1e-6,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        modulate_var: bool = False,
        emission_type: str = "gaussian",
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        transition_type: Any = constraints.Transitions.ERGODIC,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        precompute: bool = True,
        debug: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed or SeedGenerator(seed).seed
        self._params: Dict[str, Any] = {}

        super().__init__()

        self.emission_type = emission_type
        self.max_duration = max_duration
        self.temperature = temperature
        self.n_features = n_features
        self.precompute = precompute
        self.n_states = n_states
        self.alpha = alpha
        self.debug = debug

        # --- Set random seed ---
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # --- Handle hidden/context dimensions ---
        self.context_dim = context_dim
        if hidden_dim is None:
            hidden_dim = context_dim
        elif hidden_dim != context_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal context_dim ({context_dim}) "
                "unless all modules explicitly define projection layers."
            )
        self.hidden_dim = hidden_dim

        # --- Encoder setup ---
        self.encoder: Optional[ContextEncoder] = None
        if encoder is not None:
            self.encoder = (
                encoder if isinstance(encoder, ContextEncoder)
                else ContextEncoder(
                    encoder=encoder, pool=pool, n_heads=n_heads, dropout=dropout, debug=debug
                ).to(device=self.device, dtype=DTYPE)
            )
            # Infer context_dim
            self.context_dim = self._infer_encoder_dim(n_features)  # helper method
            self.hidden_dim = self.context_dim if hidden_dim is None else hidden_dim

        # --- Initialize modules ---
        self._init_modules(
            transition_type=transition_type,
            emission_type=emission_type,
            modulate_var=modulate_var,
            max_duration=max_duration,
            temperature=temperature,
            min_covar=min_covar,
        )

        self.to(device=self.device, dtype=DTYPE)

    def _infer_encoder_dim(self, n_features: int) -> int:
        """
        Runs a dummy forward pass to infer the output/context dimension of the encoder.
        """
        self.encoder.eval()
        try:
            dummy_in = torch.zeros(1, 16, n_features, device=self.device, dtype=DTYPE)
            try:
                _, ctx, _ = self.encoder(dummy_in, return_context=True, return_sequence=True)
            except TypeError:
                ctx = None
            return ctx.shape[-1] if ctx is not None else self.encoder(dummy_in).shape[-1]
        finally:
            self.encoder.train()

    def _init_modules(
        self,
        transition_type: str,
        emission_type: str,
        modulate_var: bool,
        max_duration: int,
        temperature: float,
        min_covar: float,
    ):
        """
        Initializes core HSMM modules: emission, initial, duration, transition.
        """
        device, debug = self.device, self.debug

        self.emission_module = Emission(
            n_states=self.n_states, n_features=self.n_features,
            emission_type=emission_type, context_dim=self.context_dim,
            hidden_dim=self.hidden_dim, modulate_var=modulate_var,
            temperature=temperature, min_covar=min_covar
        )

        self.initial_module = Initial(
            n_states=self.n_states, context_dim=self.context_dim,
            hidden_dim=self.hidden_dim
        )

        self.duration_module = Duration(
            n_states=self.n_states, max_duration=max_duration,
            context_dim=self.context_dim, hidden_dim=self.hidden_dim,
            temperature=temperature
        )

        self.transition_module = Transition(
            n_states=self.n_states, n_features=self.n_features,
            transition_type=transition_type, context_dim=self.context_dim,
            hidden_dim=self.hidden_dim, temperature=temperature
        )

        try:
            self._params.update({
                "initial_dist": self.initial_module.initialize(),
                "duration_dist": self.duration_module.initialize(),
                "transition_dist": self.transition_module.initialize(),
                "emission_dist": self.emission_module.initialize(),
            })
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {e}") from e

        if debug and logger:
            logger.debug(
                f"HSMM initialized on {device}: n_states={self.n_states}, "
                f"n_features={self.n_features}, context_dim={self.context_dim}, "
                f"emission={self.emission_type}, max_duration={self.max_duration}"
            )

    def _encode(self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None, detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode HSMM sequences into per-timestep features and a canonical context.

        Supports optional pooling ('mean', 'last', 'max', 'attn', 'mha') and
        deterministic projection to self.context_dim without truncation.

        Args:
            sequences: [B, T, F] input batch
            mask: optional [B, T] boolean mask (1=valid, 0=padded)
            pool: pooling method ('mean', 'last', 'max', 'attn', 'mha')
            detach: whether to detach the outputs from the computation graph

        Returns:
            seq_features: [B, T, H] per-timestep encoded features
            ctx_features: [B, 1, H] pooled canonical context
        """
        device = sequences.device
        B, T, F_in = sequences.shape

        # --- Normalize mask ---
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            mask = mask.bool().to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            mask = mask[:, :T]

        # --- Early fallback if no encoder or empty sequences ---
        if self.encoder is None or sequences.numel() == 0:
            H = self.context_dim
            seq_features = torch.zeros(B, T, H, device=device, dtype=sequences.dtype)
            ctx_features = torch.zeros(B, 1, H, device=device, dtype=sequences.dtype)
            return seq_features, ctx_features

        # --- Encoder forward pass ---
        encoder_kwargs = dict(mask=mask, return_sequence=True, return_context=True, detach_context=detach)
        if pool is not None:
            encoder_kwargs["pool"] = pool
        seq_out, ctx_out, _ = self.encoder(sequences, **encoder_kwargs)
        if seq_out is None:
            raise RuntimeError("Encoder did not return sequence features.")
        seq_features = seq_out

        # --- Canonical context fallback ---
        if ctx_out is None:
            mask_f = mask.unsqueeze(-1).to(seq_features.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (seq_features * mask_f).sum(dim=1) / denom
            ctx_features = pooled.unsqueeze(1)
        else:
            if ctx_out.ndim != 3 or ctx_out.shape[1] != 1:
                ctx_out = ctx_out.mean(dim=1, keepdim=True)
            ctx_features = ctx_out

        # --- Attention / Multi-head Attention pooling ---
        if pool in {"attn", "mha"}:
            seq_dim = seq_features.shape[-1]

            if pool == "attn":
                # Single-head dot-product attention
                if not hasattr(self, "_attn_proj"):
                    self._attn_proj = nn.Linear(seq_dim, seq_dim, bias=False).to(device)
                    with torch.no_grad():
                        self._attn_proj.weight.copy_(torch.eye(seq_dim, device=device))
                Q = self._attn_proj(ctx_features)    # [B, 1, H]
                K = self._attn_proj(seq_features)    # [B, T, H]
                V = seq_features                      # [B, T, H]
                attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(seq_dim)
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
                attn_weights = torch.softmax(attn_scores, dim=-1)
                ctx_features = torch.matmul(attn_weights, V)  # [B, 1, H]

            elif pool == "mha":
                n_heads = getattr(self, "_mha_heads", 4)
                if not hasattr(self, "_mha_layer"):
                    self._mha_layer = nn.MultiheadAttention(embed_dim=seq_dim, num_heads=n_heads, batch_first=True).to(device)
                key_padding_mask = ~mask
                ctx_features, _ = self._mha_layer(query=ctx_features, key=seq_features, value=seq_features, key_padding_mask=key_padding_mask)

        # --- Deterministic projection to context_dim ---
        target_dim = self.context_dim
        if seq_features.shape[-1] != target_dim:
            if not hasattr(self, "_encoder_proj") or self._encoder_proj.weight.shape[0] != seq_features.shape[-1]:
                proj_weight = torch.zeros(seq_features.shape[-1], target_dim, device=device)
                for i in range(min(seq_features.shape[-1], target_dim)):
                    proj_weight[i, i] = 1.0
                self._encoder_proj = nn.Linear(seq_features.shape[-1], target_dim, bias=False).to(device)
                with torch.no_grad():
                    self._encoder_proj.weight.copy_(proj_weight.T)
            seq_features = self._encoder_proj(seq_features)
            ctx_features = self._encoder_proj(ctx_features)

        # --- Optional detach ---
        if detach:
            seq_features = seq_features.detach()
            ctx_features = ctx_features.detach()

        # --- Debug prints ---
        if getattr(self, "debug", False):
            print(f"_encode output shapes: seq_features={seq_features.shape}, ctx_features={ctx_features.shape}")
            print(f"Mask sum per batch: {mask.sum(dim=1)}")
            if pool == "attn":
                print(f"Attention weights shape: {attn_weights.shape}")

        return seq_features, ctx_features

    def _prepare(self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        mask: Optional[torch.BoolTensor] = None) -> SequenceSet:
        """
        Returns:
            SequenceSet containing:
                - sequences: [B, T, F]
                - log_probs: [B, T, K]
                - contexts: [B, T, H]
                - canonical: [B, 1, H]
                - masks: [B, T, 1]
        """

        # --- Convert lists to padded tensors ---
        if isinstance(X, list):
            X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        if isinstance(theta, list):
            theta = torch.nn.utils.rnn.pad_sequence(theta, batch_first=True)

        device = X.device
        debug = self.debug

        # --- Ensure batch dimension ---
        if X.ndim == 2:
            X = X.unsqueeze(0)  # [1, T, F]
            if debug:
                print(f"[Prepare] Added batch dimension: {X.shape}")

        B, T, F = X.shape
        F_em = self.n_features
        K = self.n_states

        # --- Mask ---
        if mask is None:
            mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)
        else:
            mask = mask.bool().to(device)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            elif mask.ndim != 3 or mask.shape[-1] != 1:
                mask = mask.reshape(B, T, 1)

        if debug:
            print(f"[Prepare] Mask shape: {mask.shape}")

        # --- Context ---
        if theta is not None:
            context = theta
            if context.ndim == 2:
                context = context.unsqueeze(1).expand(B, T, -1)
            ctx_canonical = context[:, :1, :]
            if debug:
                print(f"[Prepare] Using provided context: {context.shape}")
        else:
            if T > 0:
                context, ctx_canonical = self._encode(X, mask=mask.squeeze(-1))
                if debug:
                    print(f"[Prepare] Encoded context: {context.shape}, canonical: {ctx_canonical.shape}")
            else:
                # For zero-length sequences, create empty context tensors
                H = self.context_dim
                context = torch.zeros(B, T, H, device=device, dtype=X.dtype)
                ctx_canonical = torch.zeros(B, 1, H, device=device, dtype=X.dtype)

        # --- Compute log-probabilities ---
        if T == 0:
            log_probs = torch.empty(B, 0, K, device=device, dtype=X.dtype)
        else:
            dist = self.emission_module._get_dist(context=context)
            if self.emission_module.emission_type == "gaussian":
                loc = dist.mean
                cov = dist.covariance_matrix
                if loc.ndim == 2:
                    loc = loc.unsqueeze(1).unsqueeze(2)
                elif loc.ndim == 3:
                    loc = loc.unsqueeze(2)
                loc = loc.expand(B, T, K, F_em)

                if cov.ndim == 2:
                    cov = cov.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                elif cov.ndim == 3:
                    cov = cov.unsqueeze(0).unsqueeze(0)
                cov = cov.expand(B, T, K, F_em, F_em)

                diff = X.unsqueeze(2) - loc
                var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-6)
                log_probs = -0.5 * (diff ** 2 / var).sum(-1) - 0.5 * var.log().sum(-1) - 0.5 * F_em * math.log(2 * math.pi)

            elif self.emission_module.emission_type in {"laplace", "studentt", "bernoulli", "poisson"}:
                X_exp = X.unsqueeze(2).expand(B, T, K, F_em)
                log_probs = dist.log_prob(X_exp)

            elif self.emission_module.emission_type == "categorical":
                logits = getattr(dist, "logits", None)
                if logits is None:
                    raise RuntimeError("Categorical distribution missing logits")
                logits = logits.view(1, 1, K, F_em).expand(B, T, K, F_em)
                X_exp = X.long().unsqueeze(2).expand(B, T, K)
                log_probs_all = F.log_softmax(logits, dim=2)
                log_probs = torch.gather(log_probs_all, 2, X_exp).sum(-1)

            else:
                raise NotImplementedError(f"Unsupported emission_type={self.emission_module.emission_type}")

            # --- Zero-out padded positions ---
            log_probs = log_probs.masked_fill(~mask.bool(), float("-inf"))

        lengths = mask.squeeze(-1).sum(dim=1).to(torch.long)

        if debug:
            print(f"[Prepare] log_probs shape: {log_probs.shape}")

        return SequenceSet(
            masks=mask,
            sequences=X,
            lengths=lengths,
            contexts=context,
            log_probs=log_probs,
            canonical=ctx_canonical,
        )

    def _forward(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> torch.Tensor:
        """
        Forward pass for HSMM with canonical and time-varying context.

        Args:
            X: SequenceSet containing sequences, log_probs, canonical/context tensors, masks
            theta: Optional tensor or ContextRouter to modulate/replace canonical/context/log_probs

        Returns:
            alpha: [B, T, K, Dmax] tensor of forward log-probabilities
        """
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta
        B, T, K = router.log_probs.shape[:3]
        Dmax = self.max_duration
        device = router.context.device
        dtype = router.context.dtype
        neg_inf = torch.finfo(dtype).min / 2.0

        # --- Module logits ---
        initial_logits = self.initial_module.log_matrix(context=router.canonical)  # [B,1,K]
        duration_logits = self.duration_module.log_matrix(context=router.context)  # [B,T,K,Dmax]
        transition_logits = self.transition_module.log_matrix(context=router.context)  # [B,T,K,K]

        # --- Precompute cumulative emission sums ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=dtype)
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)

        time_idx = torch.arange(T, device=device)
        emit_sums = torch.empty((B, T, K, Dmax), device=device, dtype=dtype)
        for d in range(1, Dmax + 1):
            start = (time_idx - d + 1).clamp(min=0)
            end = time_idx + 1
            emit_sums[:, :, :, d - 1] = cumsum_emit[:, end, :] - cumsum_emit[:, start, :]

        # --- Initialize alpha ---
        alpha = torch.full((B, T, K, Dmax), neg_inf, device=device, dtype=dtype)
        alpha[:, 0, :, 0] = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]

        # --- Recursion over time ---
        for t in range(1, T):
            max_d = min(Dmax, t + 1)

            # Previous alpha sums for all durations
            prev_alpha = torch.full((B, max_d, K), neg_inf, device=device, dtype=dtype)
            for d in range(1, max_d + 1):
                t_prev = t - d
                mask_valid = (t < X.lengths).view(B, 1)  # [B,1] valid timestep
                if t_prev >= 0:
                    prev_alpha[:, d - 1, :] = torch.logsumexp(alpha[:, t_prev, :, :d], dim=-1)
                else:
                    prev_alpha[:, d - 1, :] = initial_logits.squeeze(1) + duration_logits[:, 0, :, d - 1]
                prev_alpha[:, d - 1, :] = torch.where(mask_valid, prev_alpha[:, d - 1, :], neg_inf)

            # Transition & duration update
            alpha_trans = torch.logsumexp(prev_alpha.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2).transpose(1, 2)
            duration_logits_t = duration_logits[:, t, :, :max_d]
            alpha[:, t, :, :max_d] = alpha_trans + duration_logits_t + emit_sums[:, t, :, :max_d]

            # Invalidate durations exceeding current timestep
            invalid_d = torch.arange(Dmax, device=device) > t
            if invalid_d.any():
                alpha[:, t, :, invalid_d] = neg_inf

        # --- Invalidate timesteps beyond sequence length ---
        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)  # [B,T]
        alpha = alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), neg_inf)

        # --- Apply mask over padded timesteps ---
        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        alpha = alpha.masked_fill(~mask_exp, neg_inf)

        return alpha

    def _backward(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> torch.Tensor:
        """
        Batched backward pass for HSMM with canonical and time-varying context.

        Args:
            X: SequenceSet containing sequences, log_probs, canonical/context tensors, masks
            theta: Optional tensor or ContextRouter to modulate/replace canonical/context/log_probs

        Returns:
            beta: [B, T, K, Dmax] tensor of backward log-probabilities
        """
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta

        Dmax = self.max_duration
        dtype = router.context.dtype
        neg_inf = torch.finfo(dtype).min / 2.0
        B, T, K = router.log_probs.shape
        device = router.context.device

        # --- Module logits ---
        duration_logits = self.duration_module.log_matrix(context=router.context)
        transition_logits = self.transition_module.log_matrix(context=router.context)

        # --- Precompute cumulative emission sums ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=dtype)
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)

        emit_sums = torch.empty((B, T, K, Dmax), device=device, dtype=dtype)
        time_idx = torch.arange(T, device=device)
        for d in range(1, Dmax + 1):
            start = (time_idx - d + 1).clamp(min=0)
            end = time_idx + 1
            emit_sums[:, :, :, d - 1] = cumsum_emit[:, end, :] - cumsum_emit[:, start, :]

        # --- Initialize beta ---
        beta = torch.full((B, T, K, Dmax), neg_inf, device=device, dtype=dtype)
        beta[:, -1, :, 0] = 0.0  # last timestep, duration=1

        # --- Backward recursion over time ---
        for t in reversed(range(T)):
            max_d = min(Dmax, T - t)
            prev_beta = torch.full((B, max_d, K), neg_inf, device=device, dtype=dtype)

            for d in range(1, max_d + 1):
                t_next = t + d
                if t_next < T:
                    prev_beta[:, d - 1, :] = torch.logsumexp(beta[:, t_next, :, :d], dim=-1)
                    prev_beta[:, d - 1, :] = prev_beta[:, d - 1, :].masked_fill(
                        ~router.mask[:, t_next, 0].view(B, 1), neg_inf
                    )
                else:
                    prev_beta[:, d - 1, :] = 0.0

            beta_trans = torch.logsumexp(prev_beta.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2).transpose(1, 2)
            beta[:, t, :, :max_d] = beta_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]

            invalid_d = torch.arange(Dmax, device=device) >= (T - t)
            if invalid_d.any():
                beta[:, t, :, invalid_d] = neg_inf

        # --- Apply mask over padded timesteps ---
        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        beta = beta.masked_fill(~mask_exp, neg_inf)
        return beta

    def _compute_posteriors(self, X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute HSMM posteriors for a batch of sequences.

        Returns:
            gamma : [B, T, K]        state posteriors
            xi    : [B, T-1, K, K]   transition posteriors
            eta   : [B, T, K, Dmax]  duration posteriors
        """
        B = len(X.sequences)
        T_max = max(X.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration
        device = X.sequences.device if B > 0 else torch.device("cpu")

        if B == 0 or T_max == 0:
            gamma = torch.zeros((B, T_max, K), dtype=DTYPE, device=device)
            eta = torch.zeros((B, T_max, K, Dmax), dtype=DTYPE, device=device)
            xi = torch.zeros((B, max(T_max - 1, 0), K, K), dtype=DTYPE, device=device)
            return gamma, xi, eta

        # ---- Context handling ----
        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        mask = router.mask.squeeze(-1) if router.mask.ndim == 3 else router.mask
        mask_TK = mask.unsqueeze(-1)           # [B, T, 1]
        mask_TKD = mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]

        # ---- Forward / backward ----
        alpha = self._forward(X, theta=router)  # [B, T, K, Dmax]
        beta = self._backward(X, theta=router)  # [B, T, K, Dmax]

        # ---- Eta: duration posteriors ----
        eta_log = alpha + beta                  # [B, T, K, Dmax]
        eta_flat = eta_log.reshape(B, T_max, -1)
        eta_norm = torch.logsumexp(eta_flat, dim=-1, keepdim=True)
        eta = (eta_flat - eta_norm).reshape(B, T_max, K, Dmax).exp()
        eta = eta * mask_TKD

        # ---- Gamma: state posteriors ----
        gamma = eta.sum(-1)                     # [B, T, K]
        gamma = gamma * mask_TK
        gamma = gamma / gamma.sum(-1, keepdim=True).clamp_min(EPS)

        # ---- Xi: transition posteriors ----
        xi = torch.zeros((B, max(T_max - 1, 0), K, K), dtype=DTYPE, device=device)
        if T_max > 1:
            trans_logits = self.transition_module.log_matrix(router.context)  # [B, T, K, K]
            if trans_logits.ndim == 2:
                trans_logits = trans_logits.unsqueeze(0).unsqueeze(1).expand(B, T_max, K, K)

            # Compute log_xi for valid timesteps
            a_prev = torch.logsumexp(alpha[:, :T_max - 1], dim=-1)  # [B, T-1, K]
            b_next = torch.logsumexp(beta[:, 1:], dim=-1)           # [B, T-1, K]

            log_xi = (
                a_prev.unsqueeze(-1)        # [B, T-1, K, 1]
                + trans_logits[:, :T_max - 1]  # [B, T-1, K, K]
                + b_next.unsqueeze(-2)      # [B, T-1, 1, K]
            )
            log_xi = log_xi - torch.logsumexp(log_xi, dim=(2, 3), keepdim=True)
            xi = log_xi.exp()

            # Apply sequence-length mask
            lengths = torch.as_tensor(X.lengths, device=device)
            valid = torch.arange(T_max - 1, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T-1]
            xi = xi * valid.unsqueeze(-1).unsqueeze(-1)

        return gamma, xi, eta

    def _model_params(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        mode: str = "estimate", iter_idx: int = 0) -> dict:
        """
        Compute HSMM model parameters using SequenceSet X.
        Supports EM-style 'estimate' mode and 'sample' mode.
        Returns a dict of distributions: emission, initial, duration, transition.
        Fully router-aware: any theta (tensor or ContextRouter) is used consistently.
        """
        # --- Annealing alpha ---
        max_iter: int = 50
        α_min, α_max = 0.01, self.alpha
        α = max(α_min, α_max * (1.0 - iter_idx / max_iter))

        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)

        # --- Helper functions ---
        def collapse_logits(
            logits: torch.Tensor,
            dim: int,
            alpha_scale: float = 1.0,
            softmax_temp: float = 0.85,
            eps_collapse: float = 1e-9) -> torch.Tensor:
            """Softmax + Dirichlet sampling + renormalize log-probs."""
            probs = F.softmax(logits / softmax_temp, dim=dim)
            dir_alpha = (probs * alpha_scale).clamp_min(eps_collapse)
            perm = list(range(logits.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            flat_alpha = dir_alpha.permute(perm).reshape(-1, dir_alpha.shape[dim])
            samples = torch.distributions.Dirichlet(flat_alpha).rsample()
            samples = samples.reshape(*[dir_alpha.shape[i] for i in perm[:-1]], dir_alpha.shape[dim])
            samples = samples.permute(perm)
            return torch.log(samples / samples.sum(dim=dim, keepdim=True).clamp_min(EPS))

        def safe_sum(tensor_list: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            """Sum over non-empty tensors in a list; return None if empty."""
            tensors = [t for t in tensor_list if t is not None and t.numel() > 0]
            return torch.cat(tensors, dim=0).sum(dim=0) if tensors else None

        # --- Sample mode ---
        if mode == "sample":
            emission_dist = self.emission_module.forward(context=router.context, return_dist=True)
            initial_dist = self.initial_module.forward(context=router.canonical, return_dist=True)
            transition_dist = self.transition_module.forward(context=router.canonical, return_dist=True)
            duration_dist = self.duration_module.forward(context=router.canonical, return_dist=True)

            self.initial_module.update(
                new_logits=collapse_logits(initial_dist.logits, dim=0).exp(),
                context=router.canonical,
                from_probs=True
            )
            self.transition_module.update(
                new_logits=collapse_logits(transition_dist.logits, dim=1).exp(),
                context=router.canonical,
                from_probs=True
            )
            self.duration_module.update(
                new_logits=collapse_logits(duration_dist.logits, dim=1).exp(),
                context=router.canonical,
                from_probs=True
            )

        # --- Estimate mode ---
        elif mode == "estimate":
            emission_dist = self.emission_module.forward(context=router.context, return_dist=True)

            # Compute posteriors using router-aware theta
            gamma, xi, eta = self._compute_posteriors(X, theta=router)

            # Posterior counts
            init_counts = gamma[:, 0].sum(0)  # [K]
            trans_counts = xi.sum(1)          # [K, K]
            dur_counts = eta.sum(1)           # [K, Dmax]

            init_counts = init_counts if init_counts is not None else self.initial_module.logits.exp()
            trans_counts = trans_counts if trans_counts is not None else self.transition_module.logits.exp()
            dur_counts = dur_counts if dur_counts is not None else self.duration_module.logits.exp()

            # Blend with previous estimates
            blend_init = constraints.log_normalize(
                torch.log(init_counts + EPS) * α + (1 - α) * self.initial_module.logits, dim=0
            )
            blend_trans = constraints.log_normalize(
                torch.log(trans_counts + EPS) * α + (1 - α) * self.transition_module.logits, dim=1
            )
            blend_dur = constraints.log_normalize(
                torch.log(dur_counts + EPS) * α + (1 - α) * self.duration_module.logits, dim=1
            )

            # Update modules
            self.initial_module.update(new_logits=blend_init.exp(), context=router.canonical, from_probs=True)
            self.transition_module.update(new_logits=blend_trans.exp(), context=router.canonical, from_probs=True)
            self.duration_module.update(new_logits=blend_dur.exp(), context=router.canonical, from_probs=True)

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # --- Return distributions ---
        return {
            "initial_dist": self.initial_module.forward(context=router.canonical, return_dist=True),
            "duration_dist": self.duration_module.forward(context=router.canonical, return_dist=True),
            "transition_dist": self.transition_module.forward(context=router.canonical, return_dist=True),
            "emission_dist": emission_dist,
        }

    def _viterbi(
        self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        duration_weight: float = 0.0) -> list[torch.Tensor]:
        """
        Batched Viterbi decoding for HSMM with time-varying durations and context.
        Returns:
            predicted: list of tensors of shape [L_b] containing most likely state sequence per batch.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(X.log_probs.dtype).min / 2.0
        predicted: list[torch.Tensor] = []

        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        device = router.log_probs.device
        B, T_max, _ = router.log_probs.shape
        durations_full = torch.arange(1, Dmax + 1, device=device)

        for b in range(B):
            L = int(router.mask[b].sum().item())
            if L == 0:
                predicted.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            ctx_seq = router.context[b:b+1, :L]   # [1, L, H]
            canon_seq = router.canonical[b:b+1]   # [1, 1, H]

            # --- Module logits ---
            init_logits = self.initial_module.log_matrix(context=canon_seq, T=L)[0]    # [L, K]
            dur_logits = self.duration_module.log_matrix(context=ctx_seq, T=L)[0]      # [L, K, Dmax]
            trans_logits = self.transition_module.log_matrix(context=ctx_seq, T=L)[0]  # [L, K, K]

            if duration_weight != 0.0:
                dur_logits = dur_logits * (1.0 - duration_weight)

            # --- Emission sums ---
            emit_log = router.log_probs[b, :L]  # [L, K]
            cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=emit_log.dtype),
                                        torch.cumsum(emit_log, dim=0)))  # [L+1, K]

            # --- DP initialization ---
            V = torch.full((L, K), neg_inf, device=device, dtype=emit_log.dtype)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.long)
            best_dur = torch.zeros((L, K), device=device, dtype=torch.long)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1  # [max_d]

                # --- Batched emission sums for all durations ---
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                if t == 0:
                    scores = init_logits[t][:, None] + dur_logits[t, :, :max_d] + emit_sums
                    V[t], idx = scores.max(dim=1)
                    best_dur[t] = durations[idx]
                    continue

                # --- Transition scores ---
                prev_scores = V[torch.clamp(starts - 1, min=0)].T.unsqueeze(2) + trans_logits[t].unsqueeze(1)  # [K, max_d, K]
                mask_starts0 = (starts == 0)
                if mask_starts0.any():
                    prev_scores[:, mask_starts0, :] = init_logits[t].unsqueeze(0)

                prev_max, prev_arg = prev_scores.max(dim=0)  # [max_d, K]
                scores = prev_max.T + dur_logits[t, :, :max_d] + emit_sums  # [K, max_d]

                V[t], dur_idx = scores.max(dim=1)
                best_dur[t] = durations[dur_idx]
                back_ptr[t] = torch.where(
                    best_dur[t] == 1,
                    torch.full((K,), -1, device=device, dtype=torch.long),
                    prev_arg[dur_idx, torch.arange(K, device=device)]
                )

            # --- Backtrace ---
            t = L - 1
            state = int(V[t].argmax())
            segments = []

            while t >= 0:
                d = int(best_dur[t, state])
                start = max(0, t - d + 1)
                segments.append((start, t, state))
                prev = int(back_ptr[t, state])
                t = start - 1
                if prev >= 0:
                    state = prev

            segments.reverse()
            path = torch.cat([torch.full((end - start + 1,), st, device=device, dtype=torch.long)
                              for start, end, st in segments])
            predicted.append(path[:L])

        return predicted

    def _map(
        self,
        log_probs_seq: torch.Tensor,      # [T, n_features] or [T, n_states]
        init_logits: torch.Tensor,        # [n_states]
        trans_logits: torch.Tensor,       # [n_states, n_states]
        dur_logits: torch.Tensor) -> torch.Tensor:
        """
        Vectorized MAP decoding for a single sequence using HSMM parameters.
        Returns a tensor of shape [T] with predicted states.
        """
        T = log_probs_seq.shape[0]
        K = self.n_states
        D = self.max_duration
        device = log_probs_seq.device

        # --- Convert emissions to [T, K] if needed ---
        if log_probs_seq.shape[1] != K:
            logp_emit = self.emission_module.log_prob(log_probs_seq)  # [T, K]
        else:
            logp_emit = log_probs_seq

        # --- Initialize DP tables ---
        delta = torch.full((T, K), -float("inf"), device=device)
        dur_choice = torch.zeros((T, K), dtype=torch.long, device=device)
        psi = torch.zeros((T, K), dtype=torch.long, device=device)

        # Precompute cumulative sum of emission logs for fast duration sum
        cumsum_emit = torch.zeros((T + 1, K), device=device)
        cumsum_emit[1:] = torch.cumsum(logp_emit, dim=0)  # [1..T] contains sums

        for t in range(T):
            for k in range(K):
                # All possible durations for this position
                max_d = min(D, t + 1)
                d_range = torch.arange(1, max_d + 1, device=device)

                # Emission log for each duration
                emit_sum = cumsum_emit[t+1, k] - cumsum_emit[t+1-d_range, k]  # [max_d]

                # Duration log
                dur_log = dur_logits[k, d_range - 1]  # [max_d]

                # Previous delta
                if t - max_d < 0:
                    prev_vals = init_logits[k].expand(max_d)
                    prev_states = torch.zeros(max_d, dtype=torch.long, device=device)
                else:
                    prev_delta = delta[t - d_range]  # [max_d, K]
                    prev_vals, prev_states = torch.max(prev_delta + trans_logits[:, k], dim=1)  # [max_d]

                # Total score
                total_score = prev_vals + dur_log + emit_sum  # [max_d]

                # Max over duration
                max_val, max_idx = torch.max(total_score, dim=0)
                delta[t, k] = max_val
                dur_choice[t, k] = d_range[max_idx].item()
                psi[t, k] = prev_states[max_idx]

        # --- Backtrack ---
        path = torch.zeros(T, dtype=torch.long, device=device)
        t = T - 1
        state = torch.argmax(delta[t]).item()
        while t >= 0:
            d = dur_choice[t, state].item()
            path[t-d+1:t+1] = state
            t -= d
            if t >= 0:
                state = psi[t, state].item()

        return path

    def fit(self,
        X: torch.Tensor,
        n_init: int = 1,
        max_iter: int = 20,
        tol: float = 1e-4,
        patience: int = 1,
        theta: Optional[torch.Tensor] = None,
        update_rate_min: float = 0.1,
        update_rate_max: float = 0.9,
        adapt_factor: float = 10.0,
        verbose: bool = True):
        """
        Fit the HSMM using EM with optional neural emission updates.
        """
        X_valid = self._prepare(X, theta=theta)
        B = len(X_valid.sequences)
        if B == 0:
            return self

        device = X_valid.sequences[0].device
        F_dim = X_valid.sequences[0].shape[-1]
        max_len = max(X_valid.lengths)

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

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            mode = "sample" if run_idx > 0 else "estimate"
            params = self._model_params(X_valid, theta=theta, mode=mode)

            init_pdf = params["initial_dist"]
            duration_dist = params["duration_dist"]
            emission_dist = params["emission_dist"]
            transition_dist = params["transition_dist"]

            log_probs_tensor = emission_dist.log_prob(seq_tensor.unsqueeze(2)) * mask_exp
            prev_ll = log_probs_tensor.sum().item()
            self._convergence.update(prev_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                gamma, xi, eta = self._compute_posteriors(X_valid, theta=theta)

                # Sufficient statistics
                init_counts = gamma.sum((0, 1))
                dur_counts = eta.sum((0, 1))
                trans_counts = xi.sum((0, 1)) if xi is not None else transition_dist.expected_probs()

                init_counts /= init_counts.sum().clamp_min(EPS)
                dur_counts /= dur_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
                if xi is not None:
                    trans_counts /= trans_counts.sum(dim=1, keepdim=True).clamp_min(EPS)

                # Update distributions (alpha blending handled internally)
                init_pdf = self.initial_module._dist(probs=init_counts)
                duration_dist = self.duration_module._dist(probs=dur_counts)
                if xi is not None:
                    transition_dist = self.transition_module._dist(probs=trans_counts)

                # Flatten for emission updates
                flat_mask = mask.view(-1)
                flat_X = seq_tensor.reshape(-1, F_dim)[flat_mask]
                flat_gamma = gamma.reshape(-1, self.n_states)[flat_mask]
                flat_theta = theta.reshape(-1, theta.shape[-1])[flat_mask] if theta is not None else None

                if self.emission_module.emission_type in {"gaussian", "laplace", "studentt"}:
                    flat_gamma = flat_gamma.unsqueeze(-1).expand(-1, -1, F_dim)

                delta_ll = max(log_probs_tensor.sum().item() - prev_ll, 0.0)
                adaptive_rate = min(update_rate_max, max(update_rate_min, adapt_factor * delta_ll))

                if neural_update:
                    for p in self.emission_module.parameters():
                        p.requires_grad_(True)
                    if not hasattr(self, "_emission_optimizer") or self._emission_optimizer is None:
                        self._emission_optimizer = torch.optim.SGD(self.emission_module.parameters(), lr=adaptive_rate)
                    else:
                        for pg in self._emission_optimizer.param_groups:
                            pg['lr'] = adaptive_rate

                    self._emission_optimizer.zero_grad()
                    flat_gamma_safe = flat_gamma.detach()
                    loss = -(flat_gamma_safe * self.emission_module.log_prob(flat_X)).sum() / flat_gamma_safe.sum()
                    loss.backward()
                    self._emission_optimizer.step()
                    emission_dist = self.emission_module.forward(context=flat_theta, return_dist=True)
                else:
                    self.emission_module.update(posterior=flat_gamma, context=flat_theta, update_rate=adaptive_rate)
                    emission_dist = self.emission_module.forward(context=flat_theta, return_dist=True)

                self._params["emission_dist"] = emission_dist

                # Likelihood & convergence
                log_probs_tensor = emission_dist.log_prob(seq_tensor.unsqueeze(2)) * mask_exp / F_dim
                curr_ll = log_probs_tensor.sum().item()

                if verbose:
                    print(f"[Iter {it:02d}] LL={curr_ll:.4f} Δ={curr_ll - prev_ll:.3e}")

                if self._convergence.update(curr_ll, it, run_idx):
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

                prev_ll = curr_ll

            if curr_ll > best_score:
                best_score = curr_ll
                self._params.update({
                    "initial_dist": init_pdf,
                    "transition_dist": transition_dist,
                    "duration_dist": duration_dist,
                    "emission_dist": emission_dist
                })

        return self

    def predict(
        self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        transition_temp: float = 1.0,
        duration_temp: float = 1.0,
        duration_weight: float = 0.0,
        verbose: bool = True) -> list[torch.Tensor]:
        """
        Predict hidden states for HSMM sequences using Viterbi or MAP decoding.
        Supports variable-length and empty sequences, with optional context.
        """
        # --- Prepare sequences ---
        seq_set = self._prepare(X, theta=context)
        B = len(seq_set.sequences)
        if B == 0 or seq_set.total_timesteps == 0:
            # Entire batch empty
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(B)]

        device = self.device
        max_len = max(seq_set.lengths)
        if verbose:
            print(f"[Predict] Sequences: {B}, max_len: {max_len}, device: {device}")

        # --- Context routing ---
        router = ContextRouter.from_tensor(seq_set, theta=context)

        # --- Filter non-empty sequences ---
        nonzero_indices = [i for i, L in enumerate(seq_set.lengths) if L > 0]
        if len(nonzero_indices) < B:
            seq_set_nz = seq_set.index_select(torch.tensor(nonzero_indices, device=device))
            router_nz = router.select(nonzero_indices)
        else:
            seq_set_nz = seq_set
            router_nz = router

        results: list[torch.Tensor] = [torch.empty(0, dtype=torch.long, device=device) for _ in range(B)]

        if algorithm.lower() == "viterbi":
            decoded = self._viterbi(seq_set_nz, theta=router_nz, duration_weight=duration_weight)
            for idx, path in zip(nonzero_indices, decoded):
                results[idx] = path.detach().to(dtype=torch.long)
            return results

        elif algorithm.lower() == "map":
            gamma, _, _ = self._compute_posteriors(seq_set_nz, theta=router_nz)
            for i, b in enumerate(nonzero_indices):
                L = seq_set.lengths[b]
                gamma_b = gamma[i, :L]
                gamma_b = torch.nan_to_num(gamma_b, nan=-1e9, posinf=-1e9, neginf=-1e9)

                # --- Sequence-specific logits ---
                init_logits = self.initial_module.log_matrix(context=router_nz.canonical[i:i+1, :1])

                trans_logits_full = self.transition_module.log_matrix(context=router_nz.context[i:i+1, :L])
                dur_logits_full = self.duration_module.log_matrix(context=router_nz.context[i:i+1, :L])

                if init_logits.ndim > 1:
                    init_logits = init_logits.view(-1)

                # --- Handle empty or time-dependent logits safely ---
                trans_logits = trans_logits_full
                if trans_logits_full.ndim > 2 and trans_logits_full.shape[1] > 1:
                    trans_logits = trans_logits_full[0].mean(dim=0)
                elif trans_logits_full.ndim > 2:
                    trans_logits = trans_logits_full[0, 0]

                dur_logits = dur_logits_full
                if dur_logits_full.ndim > 2 and dur_logits_full.shape[1] > 1:
                    dur_logits = dur_logits_full[0].mean(dim=0)
                elif dur_logits_full.ndim > 2:
                    dur_logits = dur_logits_full[0, 0]

                log_probs_seq = seq_set.log_probs[b, :L]
                decoded_seq = self._map(
                    log_probs_seq,
                    init_logits,
                    trans_logits,
                    dur_logits
                )
                results[b] = decoded_seq.detach().to(dtype=torch.long)
            return results

        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

    def decode(
        self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["viterbi", "map"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        first_only: bool = True, verbose: bool = True) -> torch.Tensor | list[torch.Tensor]:
        """
        Decode hidden states from input sequence(s) using Viterbi or MAP.
        Supports variable-length sequences, batched inputs, and optional context.

        Args:
            X: Input sequence(s), tensor or list of tensors
            algorithm: "viterbi" or "map"
            first_only: If True, return only first sequence (single-sequence use)
                        If False, return list of tensors for all sequences
            context: Optional context tensor or list matching X sequences
            verbose: Print debug info

        Returns:
            torch.Tensor or list[torch.Tensor] of predicted states
        """
        # --- Normalize input to list of tensors ---
        if torch.is_tensor(X):
            X_list = [X] if X.ndim == 2 else [X[i] for i in range(X.shape[0])]
        elif isinstance(X, list):
            X_list = [torch.as_tensor(x, dtype=DTYPE) if not torch.is_tensor(x) else x.to(dtype=DTYPE)
                      for x in X]
        else:
            raise TypeError(f"Unsupported input type {type(X)} for decode()")

        # --- Normalize context to list of tensors ---
        context_list: Optional[list[torch.Tensor]] = None
        if context is not None:
            if torch.is_tensor(context):
                context_list = [context] if context.ndim == 2 else [context[i] for i in range(context.shape[0])]
            elif isinstance(context, list):
                context_list = [torch.as_tensor(c, dtype=DTYPE) if not torch.is_tensor(c) else c.to(dtype=DTYPE)
                                for c in context]
            else:
                raise TypeError(f"Unsupported context type {type(context)}")

        if verbose:
            print(f"[decode] algorithm={algorithm}, batch_size={len(X_list)}")

        # --- Run HSMM predict on variable-length sequences ---
        preds = self.predict(X_list, algorithm=algorithm, context=context_list, verbose=verbose)

        return preds[0] if first_only and preds else preds

    @torch.no_grad()
    def score(self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        verbose: bool = True) -> torch.Tensor:
        """
        Compute per-sequence log-likelihoods for HSMM sequences with optional context.

        This uses the same pipeline as `predict`, including SequenceSet preparation
        and emission computation, ensuring proper handling of variable-length
        sequences, zero-length sequences, and optional context modulation.

        Args:
            X: Tensor [T,F] or list of Tensors [T_b,F] representing sequences.
            theta: Optional context tensor or list of tensors, shape [T,H] or [B,T,H].
            verbose: If True, prints debug info.

        Returns:
            Tensor of shape [B], log-likelihood per sequence.
        """
        # --- Normalize input to list of tensors ---
        if torch.is_tensor(X):
            sequences = [X]
        elif isinstance(X, list):
            sequences = [torch.as_tensor(x, dtype=DTYPE) if not torch.is_tensor(x) else x.to(dtype=DTYPE) for x in X]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B = len(sequences)
        if B == 0:
            return torch.empty(0, dtype=DTYPE, device=self.device)

        # --- Normalize context ---
        context_list: Optional[list[torch.Tensor]] = None
        if theta is not None:
            if torch.is_tensor(theta):
                if theta.ndim == 2 and B > 1:
                    # Split context along time dimension based on sequence lengths
                    lengths = [seq.shape[0] for seq in sequences]
                    cum_lengths = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(lengths, dtype=torch.long).cumsum(0)])
                    context_list = [theta[cum_lengths[b]:cum_lengths[b+1]] for b in range(B)]
                else:
                    context_list = [theta] * B
            elif isinstance(theta, list):
                context_list = [
                    torch.as_tensor(t, dtype=DTYPE) if t is not None else None
                    for t in theta
                ]
                if len(context_list) != B:
                    raise ValueError(f"Length of theta list ({len(context_list)}) != number of sequences ({B})")
            else:
                raise TypeError(f"Unsupported theta type: {type(theta)}")

        # --- Prepare SequenceSet (handles padding, mask, and canonical context) ---
        seq_set = self._prepare(sequences, theta=context_list)

        if verbose:
            print(f"[score] Prepared SequenceSet with {B} sequences, max_len={max(seq_set.lengths)}")

        # --- Forward pass to compute alpha ---
        alpha_list = self._forward(seq_set, theta=None)

        # --- Compute per-sequence log-likelihoods ---
        log_likelihoods = []
        for alpha, L in zip(alpha_list, seq_set.lengths):
            if L == 0:
                log_likelihoods.append(torch.tensor(float("-inf"), dtype=DTYPE, device=self.device))
            else:
                # logsumexp over states and durations at last valid timestep
                log_likelihoods.append(torch.logsumexp(alpha[L-1], dim=(-2, -1)))

        return torch.stack(log_likelihoods)

    @torch.no_grad()
    def _compute_emit_log(self, X: SequenceSet, theta: Optional[torch.Tensor] = None, verbose: bool = False) -> torch.Tensor:
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

