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

    def _encode(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None,
        detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequences using self.encoder with optional pooling.

        Args:
            sequences: [B, T, F] input batch
            mask: optional [B, T] boolean mask (1=valid)
            pool: pooling method ('mean', 'last', 'max', 'attn', 'mha')
            detach: whether to detach outputs

        Returns:
            sequence_aligned: [B, T, H] per-timestep features
            context_canonical: [B, 1, H] pooled context
        """
        device = sequences.device
        B, T, F_in = sequences.shape

        # --- Normalize mask to [B, T] ---
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            mask = mask.bool().to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            mask = mask[:, :T]

        # --- No encoder fallback ---
        if self.encoder is None or sequences.numel() == 0:
            H = self.context_dim
            seq_aligned = torch.zeros(B, T, H, device=device, dtype=sequences.dtype)
            ctx_canonical = torch.zeros(B, 1, H, device=device, dtype=sequences.dtype)
            return seq_aligned, ctx_canonical

        # --- Call encoder safely ---
        kwargs = dict(mask=mask, return_sequence=True, return_context=True, detach_context=detach)
        if pool is not None:
            kwargs["pool"] = pool

        seq_out, ctx_out, _ = self.encoder(sequences, **kwargs)

        if seq_out is None:
            raise RuntimeError("Encoder did not return sequence features.")

        # --- Validate shapes ---
        if seq_out.shape[0] != B or seq_out.shape[1] != T:
            raise RuntimeError(f"Sequence output shape mismatch: {seq_out.shape}, expected ({B},{T},H)")

        seq_aligned = seq_out

        # --- Canonical context fallback ---
        if ctx_out is None:
            mask_f = mask.unsqueeze(-1).to(seq_aligned.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (seq_aligned * mask_f).sum(dim=1) / denom
            ctx_canonical = pooled.unsqueeze(1)
        else:
            if ctx_out.ndim != 3 or ctx_out.shape[1] != 1:
                ctx_out = ctx_out.mean(dim=1, keepdim=True)
            ctx_canonical = ctx_out

        # --- Adjust feature dimension ---
        target_dim = self.context_dim
        feat_dim = seq_aligned.shape[-1]
        if feat_dim < target_dim:
            pad = target_dim - feat_dim
            seq_aligned = F.pad(seq_aligned, (0, pad))
            ctx_canonical = F.pad(ctx_canonical, (0, pad))
        elif feat_dim > target_dim:
            seq_aligned = seq_aligned[:, :, :target_dim]
            ctx_canonical = ctx_canonical[:, :, :target_dim]

        # --- Optional detach ---
        if detach:
            seq_aligned = seq_aligned.detach()
            ctx_canonical = ctx_canonical.detach()

        if getattr(self, "debug", False):
            print(f"_encode output shapes: sequence={seq_aligned.shape}, context={ctx_canonical.shape}")

        return seq_aligned, ctx_canonical

    def _prepare(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        mask: Optional[torch.BoolTensor] = None) -> SequenceSet:
        """
        Prepare SequenceSet with sequences, log_probs, contexts, canonical context, and masks.

        Returns:
            SequenceSet containing:
                - sequences: [B, T, F]
                - log_probs: [B, T, K]
                - contexts: [B, T, H]
                - canonical: [B, 1, H]
                - masks: [B, T, 1]
        """
        debug = True
        device = X.device

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
            context, ctx_canonical = self._encode(X, mask=mask.squeeze(-1))
            if debug:
                print(f"[Prepare] Encoded context: {context.shape}, canonical: {ctx_canonical.shape}")

        # --- Compute log-probabilities ---
        dist = self.emission_module._get_dist(context=context)
        log_probs = None

        if self.emission_module.emission_type == "gaussian":
            loc = dist.mean
            cov = dist.covariance_matrix
            # --- Canonical shapes ---
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
            var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-6)  # prevent NaNs
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
        # --- Ensure router handles all context/log_probs/mask uniformly ---
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta

        B, T, K = router.log_probs.shape[0], router.log_probs.shape[1], router.log_probs.shape[2]
        Dmax = self.max_duration
        device = router.context.device
        neg_inf = torch.finfo(router.context.dtype).min / 2.0

        # --- Module logits ---
        initial_logits = self.initial_module.log_matrix(context=router.canonical)
        duration_logits = self.duration_module.log_matrix(context=router.context)
        transition_logits = self.transition_module.log_matrix(context=router.context)

        # --- Precompute cumulative emission sums ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=router.context.dtype)
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)

        emit_sums = torch.empty((B, T, K, Dmax), device=device, dtype=router.context.dtype)
        time_idx = torch.arange(T, device=device)
        for d in range(1, Dmax + 1):
            start = (time_idx - d + 1).clamp(min=0)
            end = time_idx + 1
            emit_sums[:, :, :, d - 1] = cumsum_emit[:, end, :] - cumsum_emit[:, start, :]

        # --- Initialize alpha ---
        alpha = torch.full((B, T, K, Dmax), neg_inf, device=device, dtype=router.context.dtype)
        alpha[:, 0, :, 0] = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]

        # --- Recursion over time ---
        for t in range(1, T):
            max_d = min(Dmax, t + 1)
            prev_alpha = torch.full((B, max_d, K), neg_inf, device=device, dtype=router.context.dtype)

            # Compute previous alpha sums
            for d in range(1, max_d + 1):
                t_prev = t - d
                if t_prev >= 0:
                    prev_alpha[:, d - 1, :] = torch.logsumexp(alpha[:, t_prev, :, :d], dim=-1)
                    prev_alpha[:, d - 1, :] = prev_alpha[:, d - 1, :].masked_fill(
                        ~router.mask[:, t_prev, 0].view(B, 1), neg_inf
                    )
                else:
                    prev_alpha[:, d - 1, :] = initial_logits.squeeze(1) + duration_logits[:, 0, :, d - 1]

            # Transition and duration update
            transition_logits_t = transition_logits[:, t, :, :]
            alpha_trans = torch.logsumexp(prev_alpha.unsqueeze(3) + transition_logits_t.unsqueeze(1), dim=2).transpose(1, 2)
            duration_logits_t = duration_logits[:, t, :, :max_d]
            alpha[:, t, :, :max_d] = alpha_trans + duration_logits_t + emit_sums[:, t, :, :max_d]

            # Invalidate durations exceeding current time
            invalid_d = torch.arange(Dmax, device=device) > t
            if invalid_d.any():
                alpha[:, t, :, invalid_d] = neg_inf

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
        # --- Ensure router handles all context/log_probs/mask uniformly ---
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta

        B, T, K = router.log_probs.shape
        Dmax = self.max_duration
        device = router.context.device
        dtype = router.context.dtype
        neg_inf = torch.finfo(dtype).min / 2.0

        # --- Module logits ---
        initial_logits = self.initial_module.log_matrix(context=router.canonical)
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
        beta[:, -1, :, 0] = 0.0  # last timestep, duration=1 ends naturally

        # --- Backward recursion over time ---
        for t in reversed(range(T - 1)):
            max_d = min(Dmax, T - t)
            next_beta = torch.full((B, max_d, K), neg_inf, device=device, dtype=dtype)

            # Compute contributions from next feasible durations
            for d in range(1, max_d + 1):
                next_t = t + d
                if next_t < T:
                    next_beta[:, d - 1, :] = torch.logsumexp(beta[:, next_t, :, :d], dim=-1)
                    next_beta[:, d - 1, :] = next_beta[:, d - 1, :].masked_fill(
                        ~router.mask[:, next_t, 0].view(B, 1), neg_inf
                    )
                else:
                    next_beta[:, d - 1, :] = 0.0

            # Apply transition: [B, max_d, K_from] + [B, K_from, K_to] -> [B, K_to, max_d]
            beta_trans = torch.logsumexp(next_beta.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2)
            beta_trans = beta_trans.transpose(1, 2)

            # Add duration + emission sums
            beta[:, t, :, :max_d] = beta_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]

            # Invalidate durations exceeding remaining time
            invalid_d = torch.arange(Dmax, device=device) >= (T - t)
            if invalid_d.any():
                beta[:, t, :, invalid_d] = neg_inf

        # --- Apply mask over padded timesteps ---
        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        beta = beta.masked_fill(~mask_exp, neg_inf)

        return beta

    def _compute_posteriors(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> Tuple:
        """
        Compute HSMM state posteriors (gamma, xi, eta) for a batch of sequences.

        Args:
            X: SequenceSet with sequences, log_probs, contexts, canonical, masks
            theta: Optional tensor or ContextRouter to modulate/replace context/log_probs

        Returns:
            gamma: [B, T, K] state posteriors
            xi:    [B, T-1, K, K] transition posteriors
            eta:   [B, T, K, Dmax] duration posteriors
        """
        B = len(X.sequences)
        T_max = max(X.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration
        device = X.sequences.device if B > 0 else torch.device("cpu")

        if B == 0 or T_max == 0:
            gamma = torch.zeros((B, T_max, K), dtype=DTYPE, device=device)
            eta = torch.zeros((B, T_max, K, Dmax), dtype=DTYPE, device=device)
            xi = torch.zeros((B, max(T_max-1,0), K, K), dtype=DTYPE, device=device)
            return gamma, xi, eta

        # --- Ensure ContextRouter handles theta consistently ---
        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        mask_exp = router.mask.unsqueeze(-1)  # [B, T, 1, 1]

        # --- Forward and backward pass ---
        alpha = self._forward(X, theta=router)  # [B, T, K, Dmax]
        beta = self._backward(X, theta=router)  # [B, T, K, Dmax]

        # --- Eta: duration posteriors ---
        eta_log = alpha + beta
        eta_flat = eta_log.view(B, T_max, -1)
        eta_norm = torch.logsumexp(eta_flat, dim=-1, keepdim=True)
        eta = (eta_flat - eta_norm).view(B, T_max, K, Dmax).exp()
        eta = eta * mask_exp

        # --- Gamma: state posteriors ---
        gamma = eta.sum(-1) * mask_exp.squeeze(-1)  # [B, T, K]
        gamma_sum = gamma.sum(-1, keepdim=True).clamp_min(EPS)
        gamma = gamma / gamma_sum

        # --- Xi: transition posteriors ---
        trans_logits = self.transition_module.log_matrix(router.context)  # [B, T, K, K] or [K, K]
        if trans_logits.ndim == 2:  # static transitions
            trans_logits = trans_logits.unsqueeze(0).unsqueeze(1).expand(B, T_max, K, K)

        xi = torch.zeros((B, T_max-1, K, K), dtype=DTYPE, device=device)
        lengths_tensor = torch.as_tensor(X.lengths, device=device)
        length_mask = torch.arange(T_max-1, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)  # [B, T-1]

        a_prev = torch.logsumexp(alpha[:, :T_max-1], dim=-1)  # [B, T-1, K]
        b_next = torch.logsumexp(beta[:, 1:], dim=-1)         # [B, T-1, K]

        log_xi = a_prev.unsqueeze(-1) + trans_logits[:, :T_max-1, :, :] + b_next.unsqueeze(-2)
        log_xi = log_xi - torch.logsumexp(log_xi, dim=(2, 3), keepdim=True)
        xi = log_xi.exp()

        # Apply sequence length mask
        xi = xi * length_mask.unsqueeze(-1).unsqueeze(-1)
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

    def _viterbi(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        duration_weight: float = 0.0) -> list[torch.Tensor]:
        """
        Viterbi decoding for HSMM with time-varying durations and context.

        Args:
            X: SequenceSet containing sequences, log_probs, contexts, canonical tensors, masks.
            theta: Optional ContextRouter or tensor to override context/log_probs/canonical/mask.
            duration_weight: Weight to scale duration logits (0: keep original, 1: ignore).

        Returns:
            predicted: list of tensors of shape [L_b] containing most likely state sequence per batch.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
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

            ctx_seq = router.context[b:b+1, :L]  # [1, L, H]
            canon_seq = router.canonical[b:b+1]   # [1, 1, H]

            # --- Module logits ---
            init_logits = self.initial_module.log_matrix(context=canon_seq, T=L)[0]    # [L, K]
            dur_logits = self.duration_module.log_matrix(context=ctx_seq, T=L)[0]      # [L, K, Dmax]
            trans_logits = self.transition_module.log_matrix(context=ctx_seq, T=L)[0]  # [L, K, K]

            if duration_weight != 0.0:
                dur_logits = dur_logits * (1.0 - duration_weight)

            # --- Cumulative emissions ---
            emit_log = router.log_probs[b, :L]  # [L, K]
            cumsum_emit = torch.vstack((
                torch.zeros((1, K), device=device, dtype=DTYPE),
                torch.cumsum(emit_log, dim=0)
            ))  # [L+1, K]

            # --- DP buffers ---
            V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.long)
            best_dur = torch.zeros((L, K), device=device, dtype=torch.long)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1

                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                if t == 0:
                    scores = init_logits[t][:, None] + dur_logits[t, :, :max_d] + emit_sums
                    V[t], idx = scores.max(dim=1)
                    best_dur[t] = durations[idx]
                    continue

                trans = trans_logits[t]  # [K, K]
                prev_scores = V[torch.clamp(starts - 1, min=0)].T.unsqueeze(2) + trans.unsqueeze(1)  # [K, max_d, K]

                # Handle t=0 starts
                mask_starts0 = (starts == 0)
                if mask_starts0.any():
                    prev_scores[:, mask_starts0, :] = init_logits[t].unsqueeze(0)

                prev_max, prev_arg = prev_scores.max(dim=0)  # max over previous states
                scores = prev_max.T + dur_logits[t, :, :max_d] + emit_sums

                V[t], d_idx = scores.max(dim=1)
                best_dur[t] = durations[d_idx]
                back_ptr[t] = torch.where(
                    best_dur[t] == 1,
                    torch.full((K,), -1, device=device, dtype=torch.long),
                    prev_arg[d_idx, torch.arange(K, device=device)]
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
            path = torch.cat([
                torch.full((end - start + 1,), st, device=device, dtype=torch.long)
                for start, end, st in segments
            ])
            predicted.append(path[:L])

        return predicted

    def _map(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> list[torch.Tensor]:
        """
        MAP decoding of HSMM sequences using posterior state marginals (gamma).
        Returns a list of tensors, each of shape [T], with state indices.

        Args:
            X: SequenceSet containing sequences and precomputed features.
            theta: Optional ContextRouter or tensor to override sequence context/log_probs.

        Returns:
            List of [T] tensors with MAP state assignments.
        """
        gamma, _, _ = self._compute_posteriors(X, theta=theta)
        B, T_max, K = gamma.shape
        results: list[torch.Tensor] = []

        for b in range(B):
            L = X.lengths[b]
            if L == 0:
                results.append(torch.empty(0, dtype=torch.long, device=gamma.device))
                continue

            # Clamp gamma to finite values for safety
            gamma_b = gamma[b, :L]
            gamma_b = torch.nan_to_num(gamma_b, nan=-1e9, posinf=-1e9, neginf=-1e9)

            map_seq = gamma_b.argmax(dim=-1).to(dtype=torch.long)
            results.append(map_seq)

        return results


    # HSMM EM

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

    @torch.no_grad()
    def score(self,
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
            context_list = [theta[cum_lengths[b]:cum_lengths[b+1]] for b in range(num_sequences)]
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
        obs_set = SequenceSet(sequences, log_probs=emission_log_probs_per_seq)

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

