# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.context import ContextEncoder
from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger, MAX_LOGITS
from nhsmm.tools import utils, constraints, SeedGenerator, ConvergenceMonitor
from nhsmm.distributions import Initial, Emission, Duration, Transition


class HSMM(nn.Module, ABC):
    """
        Hidden Semi-Markov Model (HSMM) base class.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        seed: Optional[int] = None,
        modulate_var: bool = False,
        transition_type: Any = None,
        alpha: Optional[float] = 1.0,
        emission_type: str = "gaussian",
        min_covar: Optional[float] = 1e-6,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_type = transition_type or constraints.Transitions.ERGODIC
        self.seed = seed or SeedGenerator(seed).seed
        self.emission_type = emission_type
        self.max_duration = max_duration
        self.modulate_var = modulate_var
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.min_covar = min_covar
        self.n_states = n_states
        self.alpha = alpha

        self._context: Optional[torch.Tensor] = None
        self.encoder: Optional[nn.Module] = None
        self._params: Dict[str, Any] = {}

        if encoder is not None:
            # If already a ContextEncoder, attach directly
            if isinstance(encoder, ContextEncoder):
                self.encoder = encoder
            else:
                # Wrap any raw nn.Module automatically
                self.encoder = ContextEncoder(
                    encoder=encoder,
                    pool=pool,
                    n_heads=n_heads,
                    dropout=dropout,
                    embed_dim=embed_dim,
                    device=self.device
                )

        # ---------------- Seed ----------------
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self._init_modules()

    def _init_modules(self) -> None:
        """Initialize HSMM modules (initial, emission, duration, transition) with consistent Contextual base."""

        device = self.device
        debug = getattr(self, "debug", DEBUG)

        self.initial_module = Initial(
            init_mode="uniform",
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            debug=debug,
        ).to(device)

        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            min_covar=self.min_covar,
            emission_type=self.emission_type,
            context_dim=self.context_dim,
            modulate_var=self.modulate_var,
            dof=getattr(self, "dof", 5.0),
            seed=self.seed,
            debug=debug,
        ).to(device)

        self.duration_module = Duration(
            init_mode="uniform",
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=getattr(self, "duration_temp", 1.0),
            scale=getattr(self, "duration_scale", 1.0),
            debug=debug,
        ).to(device)

        self.transition_module = Transition(
            init_mode="diag_bias",
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=getattr(self, "transition_temp", 1.0),
            scale=getattr(self, "transition_scale", 1.0),
            debug=debug,
        ).to(device)

        # ---------------- Initialize PDFs ----------------
        try:
            self._params.update({
                "initial_pdf": self.initial_module.initialize(),
                "duration_pdf": self.duration_module.initialize(),
                "transition_pdf": self.transition_module.initialize(),
                "emission_pdf": self.emission_module.initialize(),
            })
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {e}") from e

        # ---------------- Debug Logging ----------------
        if debug:
            logger.debug(
                f"HSMM modules initialized on {device}: "
                f"n_states={self.n_states}, n_features={self.n_features}, "
                f"context_dim={self.context_dim}, emission={self.emission_type}, "
                f"max_duration={self.max_duration}"
            )

    def _align_theta(self, theta: Optional[torch.Tensor | list[torch.Tensor]], seq_len: int) -> Optional[torch.Tensor]:
        """
        Align context theta to [seq_len, F] or [B, seq_len, F] for batching.
        Supports 1D, 2D, 3D tensors, or list of tensors. Pads shorter sequences with zeros.
        """
        if theta is None:
            return None

        device = self.device

        # Convert list of tensors to a 3D tensor
        if isinstance(theta, list):
            if not theta:
                return None
            if not all(torch.is_tensor(t) for t in theta):
                raise TypeError("All elements in theta list must be torch.Tensor")
            B, F = len(theta), theta[0].shape[-1]
            aligned = torch.zeros(B, seq_len, F, device=device, dtype=DTYPE)
            for i, t in enumerate(theta):
                t = t.to(device=device, dtype=DTYPE)
                L = min(t.shape[0], seq_len)
                aligned[i, :L] = t[:L]
            return aligned

        # Ensure tensor is on correct device/dtype
        theta = theta.to(device=device, dtype=DTYPE)
        ndim = theta.ndim

        # 1D: [F] -> [seq_len, F]
        if ndim == 1:
            return theta.unsqueeze(0).expand(seq_len, -1).contiguous()

        # 2D: [T, F] -> [seq_len, F]
        if ndim == 2:
            T, F = theta.shape
            padded = torch.zeros(seq_len, F, device=device, dtype=DTYPE)
            padded[:min(T, seq_len)] = theta[:min(T, seq_len)]
            return padded

        # 3D: [B, T, F] -> [B, seq_len, F]
        if ndim == 3:
            B, T, F = theta.shape
            padded = torch.zeros(B, seq_len, F, device=device, dtype=DTYPE)
            padded[:, :min(T, seq_len), :] = theta[:, :min(T, seq_len), :]
            return padded

        raise TypeError(f"Unsupported theta dimension {ndim}, expected 1, 2, 3, or list")

    def attach_encoder(
        self,
        encoder: nn.Module,
        dropout: float = 0.0,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        embed_dim: Optional[int] = None,
        n_heads: int = 4,
    ) -> ContextEncoder:
        """Attach or replace the model's ContextEncoder."""
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"encoder must be an nn.Module, got {type(encoder)}")
        if pool not in {"mean", "last", "max", "attn", "mha"}:
            raise ValueError(f"Invalid pool type '{pool}'.")

        self.encoder = ContextEncoder(
            encoder=encoder,
            pool=pool,
            n_heads=n_heads,
            dropout=dropout,
            embed_dim=embed_dim,
            device=self.device
        )
        return self.encoder

    def _validate(
        self,
        value: torch.Tensor,
        clamp: bool = False,
        context: Optional[torch.Tensor] = None,
        check_range: bool = True,
        allow_nan: bool = False,
    ) -> torch.Tensor:
        """
        Validate a tensor against the emission module:
          - Align device/dtype
          - Add batch/time dimension if missing
          - Check PDF support and optionally clamp
          - Verify event shape
          - Optional NaN/Inf checks
          - Optional value range checks (for discrete PDFs)
        
        Args:
            value: Tensor of observations to validate
            clamp: If True, clamp out-of-support values
            context: Optional context for contextual emissions
            check_range: If True, check values against PDF type (e.g., integer range for categorical)
            allow_nan: If False, raise on NaN/Inf values

        Returns:
            Validated tensor of same shape as input
        """
        if not hasattr(self, "emission_module"):
            raise RuntimeError("Emission module not found. Initialize `self.emission_module` first.")

        # --- Align device/dtype ---
        value = value.to(self.emission_module.device, dtype=DTYPE)

        # --- Forward to get distribution ---
        dist = self.emission_module.forward(context=context, return_dist=True)
        event_shape = dist.event_shape or ()

        # --- Add batch dimension if missing ---
        if event_shape and value.ndim == len(event_shape):
            value = value.unsqueeze(0)

        # --- Check for NaN/Inf ---
        if not allow_nan:
            if not torch.isfinite(value).all():
                raise ValueError("NaN or Inf detected in input values.")

        # --- Clamp/check support ---
        if hasattr(dist, "support"):
            support = dist.support
            if hasattr(support, "check"):
                mask = support.check(value)
                if not mask.all():
                    if clamp:
                        if hasattr(support, "clamp"):
                            value = torch.where(mask, value, support.clamp(value))
                        else:
                            lower = getattr(support, "lower_bound", -float("inf"))
                            upper = getattr(support, "upper_bound", float("inf"))
                            value = value.clamp(lower, upper)
                    else:
                        bad_vals = value[~mask].flatten().unique()
                        raise ValueError(f"Values outside PDF support: {bad_vals.tolist()}")

        # --- Event shape validation ---
        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(
                f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}"
            )

        # --- Additional range checks for discrete distributions ---
        if check_range:
            if isinstance(dist, Categorical):
                if not ((value >= 0) & (value < dist.logits.shape[-1])).all():
                    raise ValueError(f"Categorical values must be in [0, {dist.logits.shape[-1]-1}]")
            elif isinstance(dist, Bernoulli):
                if not ((value >= 0) & (value <= 1)).all():
                    raise ValueError("Bernoulli values must be 0 or 1")
            elif isinstance(dist, Poisson):
                if (value < 0).any():
                    raise ValueError("Poisson values must be non-negative integers")

        return value

    @torch.no_grad()
    def _encode_observations(
        self,
        sequences: torch.Tensor | list[torch.Tensor],
        pool: Optional[str] = None,
        detach: bool = True,
        store: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Vectorized encoding of sequences into context vectors using the attached encoder.

        Supports:
            - Single feature vector: [F]
            - Single sequence: [T, F]
            - Batched sequences: [B, T, F]
            - List of sequences (variable lengths)

        Returns:
            Tensor of shape [B, H] (context per sequence) or None if no encoder attached.
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = self.device
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            # --- Normalize to list of tensors ---
            if torch.is_tensor(sequences):
                if sequences.ndim == 1:  # [F]
                    seq_list = [sequences.unsqueeze(0)]
                elif sequences.ndim == 2:  # [T,F]
                    seq_list = [sequences]
                elif sequences.ndim == 3:  # [B,T,F]
                    seq_list = [sequences[b] for b in range(sequences.shape[0])]
                else:
                    raise TypeError(f"Unsupported tensor ndim={sequences.ndim}")
            elif isinstance(sequences, (list, tuple)):
                if not all(torch.is_tensor(seq) for seq in sequences):
                    raise TypeError("All elements must be torch.Tensor.")
                seq_list = list(sequences)
            else:
                raise TypeError(f"Unsupported input type {type(sequences)}")

            batch_size = len(seq_list)
            lengths = torch.tensor([seq.shape[0] for seq in seq_list], device=device)
            max_len = lengths.max().item()
            feature_dim = seq_list[0].shape[1] if seq_list[0].ndim > 1 else seq_list[0].shape[0]

            # --- Padded batch tensor ---
            padded = torch.zeros(batch_size, max_len, feature_dim, device=device, dtype=DTYPE)
            for i, seq in enumerate(seq_list):
                seq = seq.to(device=device, dtype=DTYPE)
                if seq.ndim == 1:
                    seq = seq.unsqueeze(-1)
                padded[i, :seq.shape[0], :] = seq

            # --- Boolean mask for variable lengths ---
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

            # --- Forward pass through encoder ---
            encoder_kwargs = {}
            if "mask" in self.encoder.forward.__code__.co_varnames:
                encoder_kwargs["mask"] = mask
            _ = self.encoder(padded, return_context=True, **encoder_kwargs)
            context_batch = self.encoder.get_context()
            if context_batch is None:
                raise RuntimeError("Encoder returned None context")

            if detach:
                context_batch = context_batch.detach()

            # --- Ensure batch size matches input ---
            if context_batch.shape[0] != batch_size:
                context_batch = context_batch[:batch_size]

        finally:
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        if store:
            self._context = context_batch

        return context_batch

    def _prepare_observations(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        chunk_size: int = 8,
    ) -> utils.Observations:
        """
        Convert raw sequences + optional context into Observations with emission log-probabilities.
        Memory-safe and vectorized, supports Gaussian, Laplace, StudentT, Categorical, Bernoulli, Poisson.
        
        Args:
            X: Input sequence(s) [T,F], [B,T,F], or list of [T_i,F].
            theta: Optional context(s) [D], [T,D], [B,T,D], or list of [T_i,D].
            chunk_size: Number of states processed per batch for VRAM control.

        Returns:
            utils.Observations containing sequences, log_probs, lengths, and context.
        """
        device = self.device

        # --- Normalize sequences to list ---
        if torch.is_tensor(X):
            X_list = [X[b] for b in range(X.shape[0])] if X.ndim == 3 else [X]
        elif isinstance(X, (list, tuple)):
            X_list = list(X)
        else:
            raise TypeError(f"Unsupported input type {type(X)}")

        B = len(X_list)
        lengths = [seq.shape[0] if seq.ndim > 1 else 1 for seq in X_list]
        n_features = X_list[0].shape[-1] if X_list[0].ndim > 1 else X_list[0].shape[0]
        max_len = max(lengths)

        # --- Pad sequences into tensor ---
        seq_tensor = torch.zeros(B, max_len, n_features, device=device, dtype=DTYPE)
        for b, seq in enumerate(X_list):
            seq = seq.to(device=device, dtype=DTYPE)
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq_tensor[b, :seq.shape[0], :] = seq

        # --- Align context ---
        theta_tensor = self._align_theta(theta or getattr(self, "_context", None), seq_len=max_len)
        if theta_tensor is not None and theta_tensor.ndim == 2:
            theta_tensor = theta_tensor.unsqueeze(0).expand(B, max_len, -1)

        # --- Get emission distribution ---
        dist = self.emission_module.forward(context=theta_tensor, return_dist=True)
        if not isinstance(dist, torch.distributions.Distribution):
            dist = torch.distributions.Independent(dist, 1)

        K = self.n_states
        T_max = seq_tensor.shape[1]
        log_probs_full = torch.empty(B, T_max, K, device=device, dtype=DTYPE)
        etype = self.emission_module.emission_type

        # --- Vectorized emission log-prob computation ---
        if etype in {"gaussian", "laplace", "studentt"}:
            means = self.emission_module._emission_means  # [K, F]
            stds = self.emission_module._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()  # [K, F]

            for k_start in range(0, K, chunk_size):
                k_end = min(K, k_start + chunk_size)
                seq_exp = seq_tensor.unsqueeze(2).expand(-1, T_max, k_end - k_start, n_features)  # [B,T,K_chunk,F]
                chunk_means = means[k_start:k_end].unsqueeze(0).unsqueeze(0)  # [1,1,K_chunk,F]
                chunk_stds = stds[k_start:k_end].unsqueeze(0).unsqueeze(0)    # [1,1,K_chunk,F]

                log_norm = -0.5 * torch.log(2 * torch.pi * chunk_stds**2)
                log_exp = -0.5 * ((seq_exp - chunk_means)**2 / (chunk_stds**2))
                log_probs_full[:, :, k_start:k_end] = (log_norm + log_exp).sum(-1)

        elif etype == "categorical":
            seq_cat = seq_tensor[..., 0].long()
            logits = getattr(self.emission_module, "logits", None)
            if logits is None:
                raise RuntimeError("Emission module has no logits.")
            for k_start in range(0, K, chunk_size):
                k_end = min(K, k_start + chunk_size)
                log_probs_chunk = F.log_softmax(logits[k_start:k_end], dim=-1)  # [K_chunk,F]
                one_hot = F.one_hot(seq_cat, num_classes=log_probs_chunk.shape[-1]).float()  # [B,T,F]
                log_probs_full[:, :, k_start:k_end] = torch.einsum("btf,kf->btk", one_hot, log_probs_chunk)

        elif etype in {"bernoulli", "poisson"}:
            for k_start in range(0, K, chunk_size):
                k_end = min(K, k_start + chunk_size)
                sub_dist = dist[k_start:k_end]
                seq_exp = seq_tensor.unsqueeze(2)  # [B,T,1,F]
                log_probs_full[:, :, k_start:k_end] = sub_dist.log_prob(
                    seq_exp.expand(-1, -1, k_end - k_start, -1)
                )

        else:
            raise TypeError(f"Unsupported emission type: {etype}")

        # --- Split back into lists ---
        sequences = [seq_tensor[b, :lengths[b]] for b in range(B)]
        log_probs_list = [log_probs_full[b, :lengths[b]] for b in range(B)]
        context_list = [theta_tensor[b, :lengths[b]] if theta_tensor is not None else None for b in range(B)]

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=lengths,
            context=context_list,
        )

    # hsmm.py HSMM precessing
    def _forward(
        self,
        X: utils.Observations,
        theta: Optional[ContextualVariables] = None
    ) -> list[torch.Tensor]:
        """
        Vectorized Forward algorithm for multiple sequences using context-modulated
        Initial, Duration, and Transition distributions via log_matrix.

        Each output α[t, k, d] represents the log-probability of ending in state k
        at time t with duration d. Contextual modules adapt parameters based on θ.

        Returns:
            list[Tensor]: One tensor per sequence of shape [T, K, Dmax].
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Context-modulated log-prob matrices ---
        init_logits = self.initial_module.log_matrix(context=theta)       # [K]
        dur_logits = self.duration_module.log_matrix(context=theta)       # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta)   # [K, K]

        alpha_list = []

        for log_emissions, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            log_emissions = log_emissions.to(device=device, dtype=DTYPE)
            T = seq_len

            # --- Cumulative emissions for efficient segment sums ---
            cumsum_emit = torch.zeros(T + 1, K, device=device, dtype=DTYPE)
            cumsum_emit[1:] = torch.cumsum(log_emissions, dim=0)

            alpha_tensor = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, device=device)
                starts = t - durations + 1  # segment start indices (inclusive)

                # Segment emission sums [K, max_d]
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                if t == 0:
                    alpha_tensor[t, :, :max_d] = init_logits.unsqueeze(1) + dur_logits[:, :max_d] + emit_sums
                    continue

                prev_alpha = torch.full((max_d, K), neg_inf, device=device, dtype=DTYPE)
                valid_idx = torch.nonzero(starts > 0, as_tuple=False).squeeze(-1)

                if valid_idx.numel() > 0:
                    prev_vals = torch.logsumexp(alpha_tensor[starts[valid_idx] - 1, :, :], dim=2)
                    prev_alpha[valid_idx, :] = prev_vals

                if (starts == 0).any():
                    prev_alpha[starts == 0, :] = init_logits

                # Combine with transitions and durations
                trans_summed = torch.logsumexp(prev_alpha.unsqueeze(2) + trans_logits.unsqueeze(0), dim=1)
                alpha_tensor[t, :, :max_d] = trans_summed.T + dur_logits[:, :max_d] + emit_sums

            alpha_list.append(alpha_tensor)

        # if all(l == X.lengths[0] for l in X.lengths):
            # return torch.stack(alpha_list, dim=0)

        return alpha_list

    def _backward(
        self,
        X: utils.Observations,
        theta: Optional[ContextualVariables] = None
    ) -> list[torch.Tensor]:
        """
        Vectorized Backward algorithm for HSMM using context-modulated log matrices.

        Each β[t, k, d] represents the log-probability of generating observations 
        from time t onward, assuming state k started at time t-d+1 with duration d.

        Returns:
            list[Tensor]: One tensor per sequence of shape [T, K, Dmax].
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Context-modulated parameters ---
        init_logits = self.initial_module.log_matrix(context=theta)     # [K]
        dur_logits = self.duration_module.log_matrix(context=theta)     # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta) # [K, K]

        beta_list = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)
            T = seq_len

            log_beta = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1, :, 0] = 0.0  # terminal condition

            # Precompute cumulative emission sums for segment likelihoods
            cumsum_emit = torch.zeros(T + 1, K, device=device, dtype=DTYPE)
            cumsum_emit[1:] = torch.cumsum(seq_logp, dim=0)

            for t in reversed(range(T - 1)):
                max_d = min(Dmax, T - t)
                durations = torch.arange(1, max_d + 1, device=device)
                ends = t + durations  # inclusive segment ends

                # Segment emission sums [K, max_d]
                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0)).T  # [K, max_d]
                dur_scores = dur_logits[:, :max_d]  # [K, max_d]

                # Transition contribution from next states
                next_beta = log_beta[ends - 1, :, 0]  # [max_d, K]
                beta_next = torch.logsumexp(trans_logits.unsqueeze(0) + next_beta.unsqueeze(1), dim=2).T  # [K, max_d]

                # Combine segment, duration, and transition terms
                segment_scores = emit_sums + dur_scores + beta_next
                log_beta[t, :, 0] = torch.logsumexp(segment_scores, dim=1)

                # Handle overlapping durations (carry forward shorter segments)
                if max_d > 1:
                    shift_len = min(max_d - 1, T - t - 1)
                    if shift_len > 0:
                        log_beta[t, :, 1:shift_len + 1] = (
                            log_beta[t + 1, :, :shift_len] + seq_logp[t + 1].unsqueeze(-1)
                        )

            beta_list.append(log_beta)

        # if all(l == X.lengths[0] for l in X.lengths):
            # return torch.stack(beta_list, dim=0)

        return beta_list

    def _compute_state_posteriors(
        self,
        X: utils.Observations,
        theta: Optional[ContextualVariables] = None
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute HSMM posteriors with context-modulated modules.

        Returns:
            gamma_vec: [T, K] state marginals
            xi_vec: [T-1, K, K] state-to-state transitions
            eta_vec: [T, K, Dmax] state-duration joint
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        B = len(X.sequence)

        gamma_vec = [None] * B
        xi_vec = [None] * B
        eta_vec = [None] * B

        # Group sequences by length for batching
        length_to_indices = {}
        for b, L in enumerate(X.lengths):
            length_to_indices.setdefault(L, []).append(b)

        # Context-modulated logits
        init_logits = self.initial_module.log_matrix(context=theta)      # [K]
        dur_logits = self.duration_module.log_matrix(context=theta)      # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta)  # [K, K]

        for L, batch_indices in length_to_indices.items():
            B_l = len(batch_indices)
            batch_logp = torch.stack([X.log_probs[b].to(device=device, dtype=DTYPE) for b in batch_indices], dim=0)

            # Forward and backward
            alpha_batch = torch.stack([
                self._forward(utils.Observations(sequence=[batch_logp[i]], log_probs=[batch_logp[i]], lengths=[L]), theta)[0]
                for i in range(B_l)
            ], dim=0)  # [B_l, L, K, Dmax]

            beta_batch = torch.stack([
                self._backward(utils.Observations(sequence=[batch_logp[i]], log_probs=[batch_logp[i]], lengths=[L]), theta)[0]
                for i in range(B_l)
            ], dim=0)  # [B_l, L, K, Dmax]

            # η (state-duration joint)
            eta_log = alpha_batch + beta_batch
            eta_log_flat = eta_log.view(B_l, L, -1)
            eta_log_flat -= torch.logsumexp(eta_log_flat, dim=-1, keepdim=True)
            eta_soft = eta_log_flat.view(B_l, L, K, Dmax).exp()

            # γ (state marginal)
            gamma_batch = eta_soft.sum(dim=-1)
            gamma_batch /= gamma_batch.sum(dim=-1, keepdim=True).clamp_min(EPS)

            # ξ (state-to-state transitions) fully vectorized
            if L <= 1:
                xi_batch = [torch.zeros((0, K, K), device=device, dtype=DTYPE)] * B_l
            else:
                alpha_prev = torch.logsumexp(alpha_batch[:, :-1], dim=3)  # [B_l, L-1, K]
                beta_next = torch.logsumexp(beta_batch[:, 1:], dim=3)     # [B_l, L-1, K]
                # Expand dims for broadcasting: [B_l, L-1, K, 1] + [1, K] + [B_l, L-1, 1, K]
                log_xi = alpha_prev.unsqueeze(3) + trans_logits.unsqueeze(0).unsqueeze(0) + beta_next.unsqueeze(2)
                log_xi = log_xi - torch.logsumexp(log_xi.view(B_l, L-1, K*K), dim=-1, keepdim=True).view(B_l, L-1, 1, 1)
                xi_batch = [log_xi[b].exp() for b in range(B_l)]

            # Assign results back to original sequence indices
            for idx, b in enumerate(batch_indices):
                gamma_vec[b] = gamma_batch[idx]
                eta_vec[b] = eta_soft[idx]
                xi_vec[b] = xi_batch[idx]

        return gamma_vec, xi_vec, eta_vec

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """
        MAP decoding of a single HSMM sequence using posterior state marginals.
        Returns a list containing one tensor of shape [T].
        """
        gamma_list, _, _ = self._compute_state_posteriors(X)

        if not gamma_list or gamma_list[0] is None or gamma_list[0].numel() == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device)]

        # Take first sequence
        gamma = gamma_list[0]

        # Replace NaN or infinite values with very negative log-prob
        gamma = torch.nan_to_num(gamma, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))

        # MAP decoding: select state with maximum posterior probability at each time
        map_seq = gamma.argmax(dim=-1).to(dtype=torch.long, device=self.device)

        return [map_seq]

    def _viterbi(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        duration_weight: float = 0.0
    ) -> list[torch.Tensor]:
        """
        Fully vectorized Viterbi decoding for HSMM sequences using the fitted emission_pdf.
        Handles variable-length sequences in a batch. Returns one predicted state sequence per input.
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        B = len(X.sequence)

        # --- Context-modulated logits ---
        init_logits = self.initial_module.log_matrix(context=theta)       # [K]
        dur_logits = self.duration_module.log_matrix(context=theta)       # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta)   # [K, K]

        initial_pdf = self._params["initial_pdf"]
        duration_pdf = self._params["duration_pdf"]
        transition_pdf = self._params["transition_pdf"]
        emission_pdf = self._params["emission_pdf"]

        # init_logits = initial_pdf.logits
        # dur_logits = duration_pdf.logits
        # trans_logits = transition_pdf.logits

        # Optional duration weighting
        if duration_weight > 0:
            dur_idx = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE).unsqueeze(0)  # [1, Dmax]
            dur_mean = (dur_logits.exp() * dur_idx).sum(dim=1, keepdim=True)              # [K, 1]
            dur_penalty = -((dur_idx - dur_mean) ** 2) / (2 * (Dmax / 3) ** 2)
            dur_logits = (1 - duration_weight) * dur_logits + duration_weight * dur_penalty

        dur_logits = dur_logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS)
        trans_logits = trans_logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS)

        # --- Prepare emissions ---
        emit_logs = []
        max_L = max(seq.shape[0] for seq in X.sequence)
        if isinstance(emission_pdf, torch.distributions.Independent):
            means = self.emission_module._emission_means        # [K, D]
            stds = self.emission_module._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()  # [K, D]

            for seq in X.sequence:
                L = seq.shape[0]
                seq_exp = seq.unsqueeze(1).expand(-1, K, -1)          # [L, K, D]
                var = stds**2
                log_norm = -0.5 * torch.log(2 * torch.pi * var)
                log_exp = -0.5 * ((seq_exp - means) ** 2 / var)
                emit_log = (log_norm + log_exp).sum(-1)               # [L, K]
                emit_logs.append(emit_log)

        elif isinstance(emission_pdf, torch.distributions.MultivariateNormal):
            means = self.emission_module._emission_means           # [K, D]
            covs = self.emission_module._emission_covs            # [K, D, D]

            for seq in X.sequence:
                L = seq.shape[0]
                seq_exp = seq.unsqueeze(1).expand(-1, K, -1)       # [L, K, D]
                mvn_batch = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
                emit_log = mvn_batch.log_prob(seq_exp)            # [L, K]
                emit_logs.append(emit_log)

        elif isinstance(emission_pdf, torch.distributions.Categorical):
            log_probs_all = F.log_softmax(emission_pdf.logits, dim=-1)
            for seq in X.sequence:
                seq_long = seq.long() if seq.ndim == 1 else seq
                emit_log = log_probs_all[:, seq_long].T           # [L, K]
                emit_logs.append(emit_log)

        else:
            raise TypeError(f"Unsupported emission type: {type(emission_pdf)}")

        predicted_sequences: list[torch.Tensor] = []
        for b, seq in enumerate(X.sequence):
            L = seq.shape[0]
            if L == 0:
                predicted_sequences.append(torch.empty(0, dtype=torch.int64, device=device))
                continue

            emit_log = emit_logs[b].clamp(min=-MAX_LOGITS)
            cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=DTYPE),
                                        torch.cumsum(emit_log, dim=0)))  # [L+1, K]

            # --- DP tables ---
            V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.int64)
            best_durations = torch.zeros((L, K), dtype=torch.int64, device=device)
            durations_full = torch.arange(1, Dmax + 1, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1

                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]
                dur_scores = dur_logits[:, :max_d]

                if t == 0:
                    scores = init_logits.unsqueeze(1) + dur_scores + emit_sums
                    best_score, best_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_idx]
                    back_ptr[t] = -1
                else:
                    prev_scores_base = V[torch.clamp(starts - 1, min=0)]
                    mask_start0 = starts == 0
                    prev_scores_base = prev_scores_base.clone()
                    prev_scores_base[mask_start0] = 0

                    prev_scores = prev_scores_base.unsqueeze(2) + trans_logits.unsqueeze(0)
                    prev_max, prev_arg = prev_scores.max(dim=1)

                    scores = prev_max.T + dur_scores + emit_sums
                    best_score, best_d_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_d_idx]

                    state_idx = torch.arange(K, device=device)
                    prev_arg_selected = prev_arg[best_d_idx, state_idx]
                    back_ptr[t] = torch.where(mask_start0[best_d_idx], torch.full_like(prev_arg_selected, -1), prev_arg_selected)

            # --- Backtrace ---
            t = L - 1
            cur_state = int(torch.argmax(V[t]).item())
            segments = []

            while t >= 0:
                d = int(best_durations[t, cur_state].item())
                start = max(0, t - d + 1)
                segments.append((start, t, cur_state))
                prev_state = int(back_ptr[t, cur_state].item())
                t = start - 1
                cur_state = prev_state if prev_state >= 0 else cur_state

            segments.reverse()
            seq_path = torch.cat([torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                                  for start, end, st in segments])
            predicted_sequences.append(seq_path[:L])

        return predicted_sequences

    def _model_params(
        self,
        X: Optional[utils.Observations] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
        max_iter: int = 50,
        iter_idx: int = 0,
    ) -> dict[str, Any]:
        device = self.device
        eps_collapse = 1e-3
        seq_len = sum(getattr(X, "lengths", [1]))
        aligned_theta = self._align_theta(theta, seq_len)

        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequence", [X]) if s.numel() > 0], dim=0).to(device, DTYPE)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, device=device, dtype=DTYPE)
        else:
            all_X = torch.zeros(1, self.n_features, device=device, dtype=DTYPE)

        # Adaptive α decay
        α_min, α_max = 0.05, self.alpha
        α = float(max(α_min, α_max * (1.0 - iter_idx / max_iter)))

        # -------- Collapse helper integrated into update --------
        def collapse_logits(logits: torch.Tensor, dim: int) -> torch.Tensor:
            probs = logits.exp()
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            dir_alpha = probs * α + EPS
            shape = dir_alpha.shape
            perm = list(range(dir_alpha.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            flat = dir_alpha.permute(perm).reshape(-1, shape[dim])
            dir_dist = torch.distributions.Dirichlet(flat)
            samples = dir_dist.rsample()
            samples = samples.reshape(*[shape[i] for i in perm]).permute(*perm)
            samples = samples / samples.sum(dim=dim, keepdim=True)
            samples = torch.where(samples > eps_collapse, samples, torch.full_like(samples, 1.0 / shape[dim]))
            return torch.log(samples + EPS)

        # -------- Mode: sample --------
        if mode == "sample":
            with torch.no_grad():
                # Initial
                init_pdf = self.initial_module.forward(context=aligned_theta, return_dist=True)
                init_logits = collapse_logits(init_pdf.logits, dim=0)
                self.initial_module.update(init_logits.exp(), from_probs=True)

                # Transition
                trans_pdf = self.transition_module.forward(context=aligned_theta, return_dist=True)
                transition_logits = collapse_logits(trans_pdf.logits, dim=1)
                self.transition_module.update(transition_logits.exp(), from_probs=True)

                # Duration
                dur_pdf = self.duration_module.forward(context=aligned_theta, return_dist=True)
                duration_logits = collapse_logits(dur_pdf.logits, dim=1)
                self.duration_module.update(duration_logits.exp(), from_probs=True)

                # Emission
                try:
                    emission_pdf = self.emission_module.forward(context=aligned_theta, return_dist=True)
                except Exception:
                    emission_pdf = self.emission_module.initialize(
                        X=all_X, context=aligned_theta, theta=aligned_theta, theta_scale=theta_scale
                    )

        # -------- Mode: estimate --------
        elif mode == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Observations X required for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_state_posteriors(X, theta=aligned_theta)

            def safe_sum(lst, dim=0):
                if not lst:
                    return None
                tensors = [x for x in lst if x is not None and x.numel() > 0]
                return torch.stack(tensors).sum(dim=0) if tensors else None

            # Safe counts
            init_counts_tmp = safe_sum([g[0] for g in gamma_list])
            init_counts = init_counts_tmp if init_counts_tmp is not None else self.initial_module._mod_logits_buffer.exp()

            trans_counts_tmp = safe_sum(xi_list)
            trans_counts = trans_counts_tmp if trans_counts_tmp is not None else self.transition_module._mod_logits_buffer.exp()

            dur_counts_tmp = safe_sum(eta_list)
            dur_counts = dur_counts_tmp if dur_counts_tmp is not None else self.duration_module._mod_logits_buffer.exp()

            # α-blended log-normalization
            blend_init = constraints.log_normalize(torch.log(init_counts + EPS) * α + (1 - α) * self.initial_module._mod_logits_buffer, dim=0)
            blend_trans = constraints.log_normalize(torch.log(trans_counts + EPS) * α + (1 - α) * self.transition_module._mod_logits_buffer, dim=1)
            blend_dur = constraints.log_normalize(torch.log(dur_counts + EPS) * α + (1 - α) * self.duration_module._mod_logits_buffer, dim=1)

            # Collapse and update modules if α > 0.5
            if α > 0.5:
                init_logits = collapse_logits(blend_init, dim=0)
                transition_logits = collapse_logits(blend_trans, dim=1)
                duration_logits = collapse_logits(blend_dur, dim=1)

                self.initial_module.update(init_logits.exp(), from_probs=True)
                self.duration_module.update(duration_logits.exp(), from_probs=True)
                self.transition_module.update(transition_logits.exp(), from_probs=True)

            else:
                init_logits, transition_logits, duration_logits = blend_init, blend_trans, blend_dur
                self.initial_module.update(blend_init.exp(), from_probs=True)
                self.duration_module.update(blend_dur.exp(), from_probs=True)
                self.transition_module.update(blend_trans.exp(), from_probs=True)

            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0)
            try:
                emission_pdf = self.emission_module.forward(context=aligned_theta, return_dist=True)
            except Exception:
                emission_pdf = self.emission_module.initialize(
                    X=all_X, posterior=all_gamma, theta=aligned_theta, context=aligned_theta, theta_scale=theta_scale
                )
        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        return {
            "emission_pdf": emission_pdf,
            "initial_pdf": torch.distributions.Categorical(logits=self.initial_module._mod_logits_buffer),
            "duration_pdf": torch.distributions.Categorical(logits=self.duration_module._mod_logits_buffer),
            "transition_pdf": torch.distributions.Categorical(logits=self.transition_module._mod_logits_buffer),
        }

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 20,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        theta: Optional[torch.Tensor] = None,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        X = X.to(self.device, dtype=DTYPE) if torch.is_tensor(X) else X
        if theta is None and getattr(self, "encoder", None):
            theta = self._encode_observations(X)

        X_valid = self._prepare_observations(X, theta=theta)
        max_len, B = max(X_valid.lengths), len(X_valid.sequence)
        aligned_theta = self._align_theta(theta, sum(X_valid.lengths))

        # Prepare padded tensors
        seq_tensor = torch.zeros((B, max_len, X_valid.sequence[0].shape[-1]), device=self.device, dtype=DTYPE)
        mask = torch.zeros(B, max_len, device=self.device, dtype=DTYPE)
        for b, seq in enumerate(X_valid.sequence):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq.to(self.device, dtype=DTYPE)
            mask[b, :L] = 1.0
        seq_exp_base = seq_tensor.unsqueeze(2)  # [B, T, 1, D]
        mask_exp = mask.unsqueeze(2)

        # Initialize convergence monitor
        self._convergence = ConvergenceMonitor(
            tol=tol, n_init=n_init, max_iter=max_iter, post_conv_iter=post_conv_iter, verbose=verbose
        )

        best_score = -float("inf")

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            mode = "sample" if (run_idx > 0) else "estimate"
            params = self._model_params(X_valid, theta=aligned_theta, mode=mode)

            init_pdf = params["initial_pdf"]
            duration_pdf = params["duration_pdf"]
            emission_pdf = params["emission_pdf"]
            transition_pdf = params["transition_pdf"]

            seq_exp = seq_exp_base.expand(-1, -1, self.n_states, -1)
            X_valid.log_probs = emission_pdf.log_prob(seq_exp) * mask_exp
            base_ll = self._compute_emit_log(X_valid).sum()
            self._convergence.push_pull(base_ll, 0, run_idx)
            curr_ll = base_ll

            for it in range(1, max_iter + 1):
                gamma_list, xi_list, eta_list = self._compute_state_posteriors(X_valid, theta=aligned_theta)

                # Concatenate safely, fallback to zeros if empty
                gamma_tensor = torch.cat([g for g in gamma_list if g.numel() > 0], dim=0) \
                    if gamma_list else torch.zeros(1, self.n_states, device=self.device, dtype=DTYPE)
                xi_tensor = torch.cat([x for x in xi_list if x is not None], dim=0) if xi_list else None
                eta_tensor = torch.cat([e for e in eta_list if e is not None], dim=0) if eta_list else None

                # Counts with safe fallback
                init_counts = gamma_tensor.sum(0)
                dur_counts = eta_tensor.sum(0) if eta_tensor is not None else duration_pdf.logits.exp()
                trans_counts = xi_tensor.sum(0) if xi_tensor is not None else transition_pdf.logits.exp()

                # Normalize counts
                init_counts = init_counts / init_counts.sum().clamp_min(EPS)
                dur_counts = dur_counts / dur_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
                trans_counts = trans_counts / trans_counts.sum(dim=1, keepdim=True).clamp_min(EPS)

                α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
                α = float(max(α_min, α_max * (1.0 - it / max_iter)))

                # Blend distributions
                init_pdf = torch.distributions.Categorical(probs=α * init_counts + (1 - α) * init_pdf.probs)
                transition_pdf = torch.distributions.Categorical(probs=α * trans_counts + (1 - α) * transition_pdf.probs)
                duration_pdf = torch.distributions.Categorical(probs=α * dur_counts + (1 - α) * duration_pdf.probs)

                # Update emission in batch-consistent manner
                emission_pdf = self.emission_module.initialize(
                    X=seq_tensor.reshape(-1, seq_tensor.shape[-1]),
                    posterior=gamma_tensor,
                    theta=aligned_theta,
                    context=aligned_theta
                )
                self._params["emission_pdf"] = emission_pdf

                # Update log-probs
                X_valid.log_probs = emission_pdf.log_prob(seq_exp) * mask_exp / seq_exp.shape[-1]

                curr_ll = self._compute_emit_log(X_valid).sum()
                converged = self._convergence.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv and verbose:
                    print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            if float(curr_ll.item()) > best_score:
                best_score = float(curr_ll.item())
                self._params.update({
                    "initial_pdf": torch.distributions.Categorical(probs=init_pdf.probs.clone()),
                    "transition_pdf": torch.distributions.Categorical(probs=transition_pdf.probs.clone()),
                    "duration_pdf": torch.distributions.Categorical(probs=duration_pdf.probs.clone()),
                    "emission_pdf": emission_pdf
                })

        if plot_conv and hasattr(self, "_convergence"):
            self._convergence.plot_convergence()

        return self

    def predict(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0,
    ) -> List[torch.Tensor]:
        """Predict hidden states using Viterbi or MAP decoding with contextual modulation."""

        if torch.is_tensor(X):
            X = [X]
        B = len(X)

        # --- Handle context ---
        if context is not None:
            if torch.is_tensor(context):
                lengths = torch.tensor([s.shape[0] for s in X], device=context.device)
                cum_lengths = torch.cat([torch.zeros(1, dtype=torch.long, device=context.device), lengths.cumsum(0)])
                context_splits = [context[cum_lengths[b]:cum_lengths[b + 1]] for b in range(B)]
            elif isinstance(context, list):
                context_splits = context
            else:
                raise ValueError("Unsupported context type")
        else:
            context_splits = [None] * B

        # --- Clamp temperatures ---
        temp_transition = max(temp_transition, 1e-6)
        temp_duration = max(temp_duration, 1e-6)

        preds: List[torch.Tensor] = []

        for b, seq in enumerate(X):
            seq = seq.to(self.device, DTYPE)
            T = seq.shape[0]
            theta_b = None if context_splits[b] is None else self._align_theta(context_splits[b], T)

            if T == 0:
                preds.append(torch.empty(0, dtype=torch.int64, device="cpu"))
                continue

            # --- Contextual logits ---
            init_logits = getattr(self._params.get("initial_pdf"), "logits", None)
            if init_logits is None:
                init_logits = self.initial_module.log_matrix(context=theta_b)

            trans_logits = getattr(self._params.get("transition_pdf"), "logits", None)
            if trans_logits is None:
                trans_logits = self.transition_module.log_matrix(context=theta_b)

            dur_logits = getattr(self._params.get("duration_pdf"), "logits", None)
            if dur_logits is None:
                dur_logits = self.duration_module.log_matrix(context=theta_b)

            trans_logits /= temp_transition
            dur_logits /= temp_duration
            trans_logits = trans_logits.clamp(-MAX_LOGITS, MAX_LOGITS)
            dur_logits = dur_logits.clamp(-MAX_LOGITS, MAX_LOGITS)

            # --- Compute emission log-probs ---
            emission_pdf = self._params.get("emission_pdf") or self.emission_module.forward(context=theta_b, return_dist=True)

            if isinstance(emission_pdf, (torch.distributions.Independent, torch.distributions.MultivariateNormal)):
                log_probs = emission_pdf.log_prob(seq.unsqueeze(1))  # [T,K]
            elif isinstance(emission_pdf, torch.distributions.Categorical):
                logits = emission_pdf.logits  # [K,C]
                if seq.ndim == 1:
                    seq_idx = seq.long()
                else:
                    seq_idx = seq.argmax(-1)
                # Gather log-probs per observation
                log_probs = logits.gather(-1, seq_idx.unsqueeze(-1)).squeeze(-1)  # [T,K]
            else:
                raise TypeError(f"Unsupported emission type: {type(emission_pdf)}")

            log_probs = log_probs.clamp(min=-MAX_LOGITS)

            # --- Observations container ---
            obs = utils.Observations([seq], log_probs=[log_probs])

            # --- Decoding ---
            if algorithm.lower() == "viterbi":
                decoded_list = self._viterbi(obs, theta=theta_b)
                for decoded in decoded_list:
                    preds.append(decoded.detach().cpu())
            elif algorithm.lower() == "map":
                decoded = self._map_decode(log_probs, init_logits, trans_logits, dur_logits)
                preds.append(decoded.detach().cpu())
            else:
                raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

        return preds

    @torch.no_grad()
    def _compute_emit_log(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Vectorized per-sequence log-likelihood computation for an HSMM with optional context.
        Handles variable-length sequences, empty sequences, and zero-length context.

        Returns:
            Tensor of shape [B], one log-likelihood per sequence.
        """
        device, B = self.device, len(X.sequence)
        neg_inf = torch.finfo(DTYPE).min / 2.0

        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, device=device, dtype=DTYPE)
            return X.log_likelihoods

        # ---------------- Align theta for batch ----------------
        aligned_theta: Optional[torch.Tensor] = None
        total_len = sum(X.lengths)
        if theta is not None:
            aligned_theta = self._align_theta(theta, total_len)

        # ---------------- Forward pass ----------------
        alpha_list = self._forward(X, theta=aligned_theta)

        # ---------------- Flatten and pad alphas ----------------
        max_len = max(X.lengths)
        n_states, n_durations = alpha_list[0].shape[1:3] if alpha_list and alpha_list[0].ndim == 3 else (self.n_states, 1)
        alpha_padded = torch.full((B, max_len, n_states, n_durations), neg_inf, device=device, dtype=DTYPE)

        for b, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L > 0:
                alpha_padded[b, :L] = alpha[:L]

        # ---------------- Logsumexp over states & durations at final step ----------------
        ll = torch.full((B,), neg_inf, device=device, dtype=DTYPE)
        for b, L in enumerate(X.lengths):
            if L > 0:
                alpha_final = alpha_padded[b, L-1]
                ll[b] = torch.logsumexp(alpha_final.flatten(), dim=0)

        X.log_likelihoods = ll

        # ---------------- Verbose logging ----------------
        if verbose:
            logger.info(f"[compute_log_likelihood] seqs={B}, min={ll.min():.4f}, max={ll.max():.4f}, mean={ll.mean():.4f}")

        return X.log_likelihoods

    @torch.no_grad()
    def score(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0
    ) -> torch.Tensor:
        """
        Compute log-likelihoods for sequences using context-modulated HSMM modules.

        Supports time-varying context per sequence.

        Args:
            X: Tensor [T,F] or list of [T_b,F].
            theta: Optional context tensor or list of tensors.
            temp_transition: Temperature scaling for transition logits.
            temp_duration: Temperature scaling for duration logits.

        Returns:
            Tensor [B] of log-likelihoods per sequence.
        """
        if torch.is_tensor(X):
            X = [X]

        B = len(X)
        if B == 0:
            return torch.tensor([], device=self.device, dtype=DTYPE)

        # Compute total length for aligning context
        seq_lengths = [seq.shape[0] for seq in X]
        total_len = sum(seq_lengths)
        aligned_theta = self._align_theta(theta, total_len) if theta is not None else None

        # Concatenate sequences into batch
        all_seq = torch.cat([seq.to(self.device, dtype=DTYPE) for seq in X], dim=0)

        # --- Compute emission log-probs for the concatenated sequences ---
        log_B = self.emission_module.log_prob(all_seq, context=aligned_theta)  # [total_len, K]

        # --- Handle per-timestep context for discrete modules ---
        init_logits = self.initial_module.log_matrix(context=self._align_theta(theta, 1))
        dur_logits = self.duration_module.log_matrix(context=self._align_theta(theta, 1)) / max(temp_duration, 1e-6)
        trans_logits = self.transition_module.log_matrix(context=self._align_theta(theta, 1)) / max(temp_transition, 1e-6)

        trans_logits = trans_logits.clamp(-MAX_LOGITS, MAX_LOGITS)
        dur_logits = dur_logits.clamp(-MAX_LOGITS, MAX_LOGITS)

        # Split log_B per sequence for Observations container
        log_probs_split = []
        start = 0
        for L in seq_lengths:
            log_probs_split.append(log_B[start:start+L])
            start += L

        obs = utils.Observations(X, log_probs=log_probs_split)

        # Compute log-likelihoods using vectorized forward algorithm
        ll = self._compute_emit_log(obs, theta=aligned_theta)
        return ll

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

