# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import copy
import math

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
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            init_mode="uniform",
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
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            init_mode="uniform",
            temperature=getattr(self, "duration_temp", 1.0),
            scale=getattr(self, "duration_scale", 1.0),
            debug=debug,
        ).to(device)

        self.transition_module = Transition(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            init_mode="diag_bias",
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

        Supports:
            - 1D: [F] -> [seq_len, F]
            - 2D: [1, F] -> [seq_len, F], [seq_len, F] -> [seq_len, F]
            - 3D: [B, 1, F] -> [B, seq_len, F], [B, seq_len, F] -> [B, seq_len, F]
            - List of tensors of shape [T_i, F] -> padded [B, seq_len, F]

        Automatically pads shorter sequences with zeros.

        Args:
            theta: Context tensor or list of tensors.
            seq_len: Target sequence length.

        Returns:
            Aligned tensor on correct device and dtype.
        """
        if theta is None:
            return None

        device = self.device

        # --- Convert list input to padded 3D tensor ---
        if isinstance(theta, list):
            if not theta:  # empty list
                return None
            # Ensure all elements are tensors
            if not all(torch.is_tensor(t) for t in theta):
                raise TypeError("All elements in theta list must be torch.Tensor")
            B = len(theta)
            F = theta[0].shape[-1]
            aligned = torch.zeros(B, seq_len, F, device=device, dtype=DTYPE)
            for i, t in enumerate(theta):
                t = t.to(device=device, dtype=DTYPE)
                L = min(t.shape[0], seq_len)
                aligned[i, :L] = t[:L]
            return aligned

        # --- Tensor input ---
        theta = theta.to(device=device, dtype=DTYPE)

        if theta.ndim == 1:  # [F] -> [seq_len, F]
            return theta.unsqueeze(0).expand(seq_len, -1).contiguous()

        if theta.ndim == 2:  # [T, F]
            T, F = theta.shape
            if T == 1:
                return theta.expand(seq_len, F).contiguous()
            elif T == seq_len:
                return theta
            else:
                padded = torch.zeros(seq_len, F, device=device, dtype=DTYPE)
                L = min(T, seq_len)
                padded[:L] = theta[:L]
                return padded

        if theta.ndim == 3:  # [B, T, F]
            B, T, F = theta.shape
            if T == seq_len:
                return theta
            else:
                padded = torch.zeros(B, seq_len, F, device=device, dtype=DTYPE)
                L = min(T, seq_len)
                padded[:, :L, :] = theta[:, :L, :]
                return padded

        raise TypeError(f"Unsupported theta dimension {theta.ndim}, expected 1, 2, 3, or list")

    def attach_encoder(
        self,
        encoder: nn.Module,
        dropout: float = 0.0,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        embed_dim: Optional[int] = None,
        n_heads: int = 4,
    ) -> ContextEncoder:
        """
        Attach a ContextEncoder wrapper to the model for contextual HSMM parameter modulation.

        Args:
            encoder: Base nn.Module for encoding raw inputs (e.g., MLP, Transformer).
            n_heads: Number of attention heads (used if pool='mha' or 'attn').
            pool: Pooling method for sequence embedding. Options:
                'mean', 'last', 'max', 'attn', 'mha'.
            dropout: Optional dropout rate for attention layers.
            embed_dim: Optional embedding dimension override for attention pooling.

        Returns:
            The attached ContextEncoder instance.
        """
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"encoder must be an nn.Module, got {type(encoder)}")

        if pool not in {"mean", "last", "max", "attn", "mha"}:
            raise ValueError(f"Invalid pool type '{pool}'. Must be one of 'mean', 'last', 'max', 'attn', 'mha'.")

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
    def encode_observations(
        self,
        sequences: torch.Tensor | list[torch.Tensor],
        pool: Optional[str] = None,
        detach: bool = True,
        store: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Encode sequences into context vectors using the attached encoder.

        Supports:
            - Single feature vector: [F]
            - Single sequence: [T, F]
            - Batched sequences: [B, T, F]
            - List of sequences (variable lengths)

        Args:
            sequences: Input sequence(s)
            pool: Optional pooling method for encoder
            detach: If True, detach the returned context tensor
            store: If True, store the resulting context in self._context

        Returns:
            Tensor of shape [B, H] (batch of context vectors), or None if no encoder attached.
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
            # --- Normalize input to list of tensors ---
            if torch.is_tensor(sequences):
                if sequences.ndim == 1:
                    sequence_list = [sequences.unsqueeze(0)]  # [F] -> [1,F]
                elif sequences.ndim == 2:
                    sequence_list = [sequences]  # [T,F] -> single sequence
                elif sequences.ndim == 3:
                    sequence_list = [sequences[b] for b in range(sequences.shape[0])]  # [B,T,F]
                else:
                    raise TypeError(f"Unsupported tensor ndim={sequences.ndim}")
            elif isinstance(sequences, (list, tuple)):
                if not all(torch.is_tensor(seq) for seq in sequences):
                    raise TypeError("All elements in the list must be torch.Tensor.")
                sequence_list = list(sequences)
            else:
                raise TypeError(f"Unsupported input type {type(sequences)}")

            batch_size = len(sequence_list)
            seq_lengths = [seq.shape[0] if seq.ndim > 1 else 1 for seq in sequence_list]
            max_length = max(seq_lengths)
            feature_dim = sequence_list[0].shape[1] if sequence_list[0].ndim > 1 else sequence_list[0].shape[0]

            # --- Create padded sequences tensor ---
            padded_sequences = torch.zeros(batch_size, max_length, feature_dim, device=device, dtype=DTYPE)
            for i, seq in enumerate(sequence_list):
                seq = seq.to(device=device, dtype=DTYPE)
                if seq.ndim == 1:
                    seq = seq.unsqueeze(-1)
                padded_sequences[i, :seq.shape[0], :] = seq

            # --- Optional mask for variable-length sequences ---
            encoder_kwargs = {}
            if "mask" in self.encoder.forward.__code__.co_varnames:
                mask = torch.zeros(batch_size, max_length, device=device, dtype=DTYPE)
                for i, length in enumerate(seq_lengths):
                    mask[i, :length] = 1.0
                encoder_kwargs["mask"] = mask

            # --- Forward pass through encoder ---
            _ = self.encoder(padded_sequences, return_context=True, **encoder_kwargs)
            context_batch = self.encoder.get_context()
            if context_batch is None:
                raise RuntimeError("Encoder returned None context")

            if detach:
                context_batch = context_batch.detach().clone()

            # --- Ensure correct batch size ---
            if context_batch.shape[0] > batch_size:
                context_batch = context_batch[:batch_size]

        finally:
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        if store:
            self._context = context_batch

        return context_batch

    def to_observations(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        chunk_size: int = 8,
    ) -> utils.Observations:
        """
        Vectorized conversion of sequences into Observations using emission_module.
        Uses _align_theta for context alignment. Memory-safe version with lazy per-state log-prob computation.

        Args:
            X: Input sequence(s): [T,F], [B,T,F], or list of [T_i,F].
            theta: Context(s): [D], [T,D], [B,T,D], or list of [T_i,D].
            chunk_size: States processed per forward pass (reduces VRAM usage).
        """
        device = self.device

        # --- Normalize sequences to list ---
        if torch.is_tensor(X):
            if X.ndim == 3:
                X_list = [X[b] for b in range(X.shape[0])]
            else:
                X_list = [X]
        elif isinstance(X, (list, tuple)):
            X_list = list(X)
        else:
            raise TypeError(f"Unsupported input type {type(X)}")

        B = len(X_list)
        lengths = [seq.shape[0] if seq.ndim > 1 else 1 for seq in X_list]
        n_features = X_list[0].shape[-1] if X_list[0].ndim > 1 else X_list[0].shape[0]
        max_len = max(lengths)

        # --- Stack sequences into padded tensor ---
        seq_tensor = torch.zeros(B, max_len, n_features, device=device, dtype=DTYPE)
        for b, seq in enumerate(X_list):
            seq = seq.to(device=device, dtype=DTYPE)
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq_tensor[b, :seq.shape[0], :] = seq

        # --- Align context using _align_theta ---
        theta_tensor = self._align_theta(theta or getattr(self, "_context", None), seq_len=max_len)
        if theta_tensor is not None and theta_tensor.ndim == 2:  # [T,D] -> [B,T,D]
            theta_tensor = theta_tensor.unsqueeze(0).expand(B, max_len, -1)

        # --- Get emission distribution ---
        dist = self.emission_module.forward(context=theta_tensor, return_dist=True)
        if not hasattr(dist, "log_prob"):
            dist = torch.distributions.Independent(dist, 1)

        K = self.n_states
        T_max = seq_tensor.shape[1]

        # --- Memory-safe log-prob computation ---
        log_probs_full = torch.empty(B, T_max, K, device=device, dtype=DTYPE)
        if self.emission_module.emission_type in {"gaussian", "laplace", "studentt"}:
            for k_start in range(0, K, chunk_size):
                k_end = min(K, k_start + chunk_size)
                sub_dist = dist[k_start:k_end] if isinstance(dist, (list, tuple)) else dist
                seq_exp = seq_tensor.unsqueeze(2)
                logp = sub_dist.log_prob(seq_exp.expand(-1, -1, k_end - k_start, -1))
                log_probs_full[:, :, k_start:k_end] = logp

        elif self.emission_module.emission_type == "categorical":
            seq_cat = seq_tensor[..., 0].long()
            logits = getattr(self.emission_module, "logits", None)
            if logits is None:
                raise RuntimeError("Emission module has no logits.")
            for k_start in range(0, K, chunk_size):
                k_end = min(K, k_start + chunk_size)
                logits_chunk = logits[k_start:k_end]
                logp = torch.nn.functional.log_softmax(logits_chunk, dim=-1)
                one_hot = torch.nn.functional.one_hot(seq_cat, num_classes=logp.shape[-1]).float()
                log_probs_full[:, :, k_start:k_end] = torch.einsum("btf,kf->btk", one_hot, logp)

        elif self.emission_module.emission_type in {"bernoulli", "poisson"}:
            for k_start in range(0, K, chunk_size):
                k_end = min(K, k_start + chunk_size)
                sub_dist = dist[k_start:k_end]
                seq_exp = seq_tensor.unsqueeze(2)
                log_probs_full[:, :, k_start:k_end] = sub_dist.log_prob(
                    seq_exp.expand(-1, -1, k_end - k_start, -1)
                )

        else:
            raise TypeError(f"Unsupported emission type: {type(dist)}")

        # --- Split back into per-sequence lists ---
        sequences = [seq_tensor[b, :lengths[b]] for b in range(B)]
        log_probs_list = [log_probs_full[b, :lengths[b]] for b in range(B)]
        context_list = (
            [theta_tensor[b, :lengths[b]] if theta_tensor is not None else None for b in range(B)]
        )

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=lengths,
            context=context_list,
        )


    # hsmm.py HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        """
        Vectorized Forward algorithm for multiple sequences using context-modulated
        Initial, Duration, and Transition distributions via log_matrix.
        Returns a list of tensors [T, K, Dmax] for each sequence.
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Get context-modulated log-prob matrices ---
        init_logits = self.initial_module.log_matrix(context=theta)    # [K]
        dur_logits = self.duration_module.log_matrix(context=theta)    # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta)  # [K, K]

        alpha_list: list[torch.Tensor] = []

        for log_emissions, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            log_emissions = log_emissions.to(device=device, dtype=DTYPE)  # [T, K]
            T = seq_len

            # --- Precompute cumulative emission sums for all durations ---
            cumsum_emit = torch.cat([torch.zeros(1, K, device=device, dtype=DTYPE), torch.cumsum(log_emissions, dim=0)], dim=0)  # [T+1, K]

            # --- DP table ---
            alpha_tensor = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, device=device)  # [max_d]
                starts = t - durations + 1                              # [max_d]

                # emission sums for each duration [K, max_d]
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_d]

                if t == 0:
                    # only initial state possible
                    alpha_tensor[t, :, :max_d] = init_logits.unsqueeze(1) + dur_logits[:, :max_d] + emit_sums
                    continue

                # previous alpha for valid starts
                prev_alpha = torch.full((max_d, K), neg_inf, device=device, dtype=DTYPE)
                valid_mask = starts > 0
                if valid_mask.any():
                    prev_vals = torch.logsumexp(alpha_tensor[starts[valid_mask] - 1, :, :], dim=2)  # [num_valid, K]
                    prev_alpha[valid_mask] = prev_vals
                # starts at 0 -> use initial logits
                prev_alpha[starts == 0] = init_logits

                # transition + duration + emission
                summed = torch.logsumexp(prev_alpha.unsqueeze(2) + trans_logits.unsqueeze(0), dim=1)  # [max_d, K]
                alpha_tensor[t, :, :max_d] = summed.T + dur_logits[:, :max_d] + emit_sums

            alpha_list.append(alpha_tensor)

        return alpha_list

    def _backward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        """
        Vectorized Backward algorithm for HSMM using context-modulated log matrices.
        Returns list of [T, K, Dmax] log-beta tensors for each sequence.
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Contextual log-prob matrices ---
        init_logits = self.initial_module.log_matrix(context=theta)       # [K]
        dur_logits = self.duration_module.log_matrix(context=theta)       # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta)   # [K, K]

        beta_list: list[torch.Tensor] = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)  # [T, K]
            T = seq_len

            # DP table
            log_beta = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1, :, 0] = 0.0  # terminal step for duration=1

            # Precompute cumulative emission sums for durations
            cumsum_emit = torch.cat([torch.zeros(1, K, device=device, dtype=DTYPE),
                                     torch.cumsum(seq_logp, dim=0)], dim=0)  # [T+1, K]

            for t in reversed(range(T - 1)):
                max_dur = min(Dmax, T - t)
                durations = torch.arange(1, max_dur + 1, device=device)
                ends = t + durations  # segment ends

                # Emission sums: [K, max_dur]
                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0)).T

                # Duration scores: [K, max_dur]
                dur_scores = dur_logits[:, :max_dur]

                # Next beta contributions: [max_dur, K] → [K, max_dur]
                next_beta = log_beta[ends - 1, :, 0]             # [max_dur, K]
                beta_next = torch.logsumexp(trans_logits.unsqueeze(0) + next_beta.unsqueeze(1), dim=2).T  # [K, max_dur]

                # Segment score aggregation
                segment_scores = emit_sums + dur_scores + beta_next
                log_beta[t, :, 0] = torch.logsumexp(segment_scores, dim=1)

                # Propagate within segment
                if max_dur > 1:
                    shift_len = min(max_dur - 1, T - t - 1)
                    if shift_len > 0:
                        log_beta[t, :, 1:shift_len + 1] = log_beta[t + 1, :, :shift_len] + seq_logp[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Fully vectorized posterior computation for HSMM using contextual modules.
        Returns:
            gamma_vec: [T, K] state marginals
            xi_vec: [T-1, K, K] state-to-state transitions
            eta_vec: [T, K, Dmax] state-duration joint
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0
        B = len(X.sequence)

        # --- Fetch context-modulated logits from modules ---
        init_logits = self.initial_module.log_matrix(context=theta)           # [K]
        dur_logits  = self.duration_module.log_matrix(context=theta)          # [K, Dmax]
        trans_logits = self.transition_module.log_matrix(context=theta)       # [K, K]

        # --- Group sequences by length for batching ---
        length_to_indices: dict[int, list[int]] = {}
        for b, L in enumerate(X.lengths):
            length_to_indices.setdefault(L, []).append(b)

        gamma_vec: list[torch.Tensor] = [None] * B
        xi_vec: list[torch.Tensor]    = [None] * B
        eta_vec: list[torch.Tensor]   = [None] * B

        for L, batch_indices in length_to_indices.items():
            B_l = len(batch_indices)
            # Stack log-probs [B_l, T, K]
            batch_logp = torch.stack([X.log_probs[b].to(device=device, dtype=DTYPE) for b in batch_indices], dim=0)

            # --- Forward & Backward in log-space ---
            alpha_batch = torch.stack([self._forward(utils.Observations(sequence=[batch_logp[i]],
                                                                        log_probs=[batch_logp[i]],
                                                                        lengths=[L]),
                                                     theta)[0]
                                       for i in range(B_l)], dim=0)  # [B_l, T, K, Dmax]
            beta_batch = torch.stack([self._backward(utils.Observations(sequence=[batch_logp[i]],
                                                                       log_probs=[batch_logp[i]],
                                                                       lengths=[L]),
                                                      theta)[0]
                                      for i in range(B_l)], dim=0)

            # --- η (state-duration joint) ---
            eta_log = alpha_batch + beta_batch                     # still in log-space
            eta_log_flat = eta_log.view(B_l, L, -1)
            eta_log_flat = eta_log_flat - torch.logsumexp(eta_log_flat, dim=-1, keepdim=True)
            eta_soft = eta_log_flat.exp().view(B_l, L, K, Dmax)

            # --- γ (state marginal) ---
            gamma_batch = eta_soft.sum(dim=-1)
            gamma_batch = gamma_batch / gamma_batch.sum(dim=-1, keepdim=True).clamp_min(EPS)

            # --- ξ (state-to-state transitions) ---
            xi_batch = []
            for i in range(B_l):
                if L <= 1:
                    xi_batch.append(torch.zeros((0, K, K), device=device, dtype=DTYPE))
                    continue

                alpha_prev = torch.logsumexp(alpha_batch[i][:-1], dim=2)  # [T-1, K]
                beta_next = torch.logsumexp(beta_batch[i][1:], dim=2)     # [T-1, K]

                log_xi_i = alpha_prev.unsqueeze(2) + trans_logits.unsqueeze(0) + beta_next.unsqueeze(1)  # [T-1, K, K]
                log_xi_i = log_xi_i - torch.logsumexp(log_xi_i.view(L-1, K*K), dim=-1, keepdim=True).view(L-1, 1, 1)
                xi_batch.append(log_xi_i.exp())

            # --- Assign back to original sequence indices ---
            for idx, b in enumerate(batch_indices):
                gamma_vec[b] = gamma_batch[idx]
                eta_vec[b] = eta_soft[idx]
                xi_vec[b] = xi_batch[idx]

        return gamma_vec, xi_vec, eta_vec

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """MAP decoding of a single HSMM sequence from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if not gamma_list or gamma_list[0] is None or gamma_list[0].numel() == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device)]

        gamma = gamma_list[0]  # single sequence
        gamma = torch.nan_to_num(gamma, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))

        # MAP decoding: argmax along states
        map_seq = gamma.argmax(dim=-1).to(dtype=torch.long, device=self.device)
        return [map_seq]

    def _viterbi(
        self, 
        X: utils.Observations, 
        theta: Optional[torch.Tensor] = None, 
        duration_weight: float = 0.0
    ) -> torch.Tensor:
        """
        Viterbi decoding for a single HSMM sequence using context-modulated parameters.
        Supports Gaussian, Laplace, StudentT, Categorical, Bernoulli, and Poisson emissions.
        Returns predicted states tensor of shape [T].
        """
        device = self.device
        seq = X.sequence[0].to(device, DTYPE)
        L, K, Dmax = seq.shape[0], self.n_states, self.max_duration
        if L == 0:
            return torch.empty(0, dtype=torch.int64, device=device)

        neg_inf = torch.finfo(DTYPE).min / 2

        # --- Contextual log-probabilities ---
        init_logits = self.initial_module.log_matrix(context=theta)        # [K]
        trans_logits = self.transition_module.log_matrix(context=theta)    # [K,K]
        dur_logits = self.duration_module.log_matrix(context=theta)        # [K,Dmax]

        # --- Optional duration weighting ---
        if duration_weight > 0:
            dur_idx = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE).unsqueeze(0)  # [1,Dmax]
            dur_mean = (dur_logits.exp() * dur_idx).sum(dim=1, keepdim=True)              # [K,1]
            dur_penalty = -((dur_idx - dur_mean) ** 2) / (2 * (Dmax / 3) ** 2)
            dur_logits = (1 - duration_weight) * dur_logits + duration_weight * dur_penalty
        dur_logits = dur_logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS)
        trans_logits = trans_logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS)

        # --- Emission log-probs ---
        emission_dist = self._params.get("emission_pdf") or self.emission_module.forward(context=theta, return_dist=True)
        
        if isinstance(emission_dist, torch.distributions.Independent) or isinstance(emission_dist, torch.distributions.MultivariateNormal):
            # Continuous distributions: [L,K]
            emit_log = emission_dist.log_prob(seq.unsqueeze(1))
        elif isinstance(emission_dist, torch.distributions.Categorical):
            # Categorical: [L,K]
            log_probs_all = F.log_softmax(emission_dist.logits, dim=-1)  # [K,C]
            seq_long = seq.long() if seq.ndim == 1 else seq              # [L] or [L,F]
            one_hot = F.one_hot(seq_long, num_classes=log_probs_all.shape[-1]).float()  # [L,F,C]
            emit_log = torch.einsum("lfc,kc->lk", one_hot, log_probs_all)
        else:
            raise TypeError(f"Unsupported emission distribution type: {type(emission_dist)}")

        emit_log = emit_log.clamp(min=-MAX_LOGITS)
        cumsum_emit = torch.vstack((torch.zeros((1,K), device=device, dtype=DTYPE), torch.cumsum(emit_log, dim=0)))  # [L+1,K]

        # --- DP tables ---
        V = torch.full((L,K), neg_inf, device=device, dtype=DTYPE)
        back_ptr = torch.full((L,K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((L,K), dtype=torch.int64, device=device)
        durations_full = torch.arange(1,Dmax+1, device=device)

        for t in range(L):
            max_d = min(Dmax, t+1)
            durations = durations_full[:max_d]
            starts = t - durations + 1

            emit_sums = (cumsum_emit[t+1].unsqueeze(0) - cumsum_emit[starts]).permute(1,0)  # [K,max_d]
            dur_scores = dur_logits[:, :max_d] if dur_logits.ndim == 2 else dur_logits[:max_d].unsqueeze(0).expand(K,-1)

            if t == 0:
                scores = init_logits.unsqueeze(1) + dur_scores + emit_sums
                best_score, best_idx = scores.max(dim=1)
                V[t] = best_score
                best_dur[t] = durations[best_idx]
            else:
                starts_clip = torch.clamp(starts-1, min=0)
                prev_scores_base = V[starts_clip]                # [max_d,K]
                prev_scores_base[starts==0] = 0
                prev_scores = prev_scores_base.unsqueeze(2) + trans_logits.unsqueeze(0)  # [max_d,K,K]
                prev_max, prev_arg = prev_scores.max(dim=1)                             # [max_d,K]
                scores = prev_max.permute(1,0) + dur_scores + emit_sums                 # [K,max_d]
                best_score, best_d_idx = scores.max(dim=1)
                V[t] = best_score
                best_dur[t] = durations[best_d_idx]

                idx_range = torch.arange(K, device=device)
                prev_arg_selected = prev_arg[best_d_idx, idx_range]
                valid_mask = starts[best_d_idx] > 0
                back_ptr[t] = torch.where(valid_mask, prev_arg_selected.to(torch.int64),
                                          torch.full_like(prev_arg_selected, -1, dtype=torch.int64))

        # --- Backtrace ---
        t = L-1
        cur_state = int(torch.argmax(V[t]).item())
        segments = []
        while t >= 0:
            d = int(best_dur[t, cur_state].item())
            start = max(0, t-d+1)
            segments.append((start, t, cur_state))
            prev_state = int(back_ptr[t, cur_state].item())
            t = start-1
            cur_state = prev_state if prev_state >= 0 else cur_state

        segments.reverse()
        seq_path = torch.cat([torch.full((end-start+1,), st, dtype=torch.int64, device=device)
                              for start,end,st in segments])
        return seq_path[:L]

    def _model_params(
        self,
        X: Optional[utils.Observations] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
    ) -> dict[str, Any]:
        """
        Compute or sample HSMM parameters using module-level caching.
        Supports 'estimate' and 'sample' modes with optional contextual modulation.
        Returns distribution objects (Categorical, emission) for all HSMM parameters.
        """
        K, Dmax = self.n_states, self.max_duration
        device = self.device
        eps_collapse = 1e-3
        max_iter: int = 50
        iter_idx: int = 0

        seq_len = sum(getattr(X, "lengths", [1]))
        aligned_theta = self._align_theta(theta, seq_len)

        # ---------------- Observations tensor ----------------
        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequence", [X]) if s.numel() > 0], dim=0).to(device, DTYPE)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, device=device, dtype=DTYPE)
        else:
            all_X = torch.zeros(1, self.n_features, device=device, dtype=DTYPE)

        # ---------------- Adaptive α decay ----------------
        α_min, α_max = 0.05, self.alpha
        α = float(max(α_min, α_max * (1.0 - iter_idx / max_iter)))

        # ---------------- Helper: log-probs with module cache ----------------
        def _get_log_probs(module: nn.Module, expected_shape: tuple) -> torch.Tensor:
            try:
                lp = module.forward(context=aligned_theta, log=True)
                if lp.ndim > len(expected_shape):
                    lp = lp.mean(dim=tuple(range(lp.ndim - len(expected_shape))))
                if lp.shape != expected_shape and lp.numel() == math.prod(expected_shape):
                    lp = lp.view(expected_shape)
                return lp.to(device, DTYPE)
            except Exception:
                logger.warning("Exception occured in _get_log_probs")
                return torch.full(expected_shape, -math.log(expected_shape[-1]), device=device, dtype=DTYPE)

        init_ctx = _get_log_probs(self.initial_module, (K,))
        trans_ctx = _get_log_probs(self.transition_module, (K, K))
        dur_ctx = _get_log_probs(self.duration_module, (K, Dmax))

        # ---------------- Helper: Dirichlet smoothing ----------------
        def collapse_safe(logits: torch.Tensor, dim: int) -> torch.Tensor:
            probs = logits.exp()
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            dir_alpha = probs * α + EPS
            n = probs.shape[dim]

            # Vectorized flatten for Dirichlet sampling
            perm = list(range(dir_alpha.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            flat = dir_alpha.permute(perm).reshape(-1, n)
            samples = torch.stack([torch.distributions.Dirichlet(row).rsample() for row in flat], dim=0)
            samples = samples.reshape(dir_alpha.shape).permute(perm)
            samples = samples / samples.sum(dim=dim, keepdim=True)
            samples = torch.where(samples > eps_collapse, samples, torch.full_like(samples, 1.0 / n))
            return torch.log(samples + EPS)

        # ---------------- Mode: sample ----------------
        if mode.lower() == "sample":
            with torch.no_grad():
                init_logits = collapse_safe(init_ctx, dim=0)
                transition_logits = collapse_safe(trans_ctx, dim=1)
                duration_logits = collapse_safe(dur_ctx, dim=1)
                try:
                    emission_pdf = self._params.get('emission_pdf')
                    # self.emission_module.initialize(
                        # X=all_X,
                        # posterior=None,
                        # theta=aligned_theta,
                        # context=aligned_theta,
                        # theta_scale=theta_scale,
                        # emission_type=self.emission_type
                    # )
                    # emission_pdf = self.emission_module.forward(context=aligned_theta, return_dist=True)
                except Exception:
                    try:
                        emission_pdf = self.emission_module.initialize(
                            X=None,
                            emission_type=self.emission_module.emission_type
                        )
                    except Exception:
                        # Last-resort: build a simple gaussian with zeros (keeps API)
                        emission_pdf = torch.distributions.Independent(
                            torch.distributions.Normal(
                                torch.zeros(self.n_states, self.n_features, device=device, dtype=DTYPE),
                                torch.ones(self.n_states, self.n_features, device=device, dtype=DTYPE)
                            ), 1
                        )

        # ---------------- Mode: estimate ----------------
        elif mode.lower() == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Observations X required for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=aligned_theta)

            def safe_sum(lst, dim=0):
                if not lst: return None
                tensors = [x for x in lst if x is not None and x.numel() > 0]
                return torch.stack(tensors).sum(dim=0) if tensors else None

            init_counts = safe_sum([g[0] for g in gamma_list]) or init_ctx.exp()
            trans_counts = safe_sum(xi_list) or trans_ctx.exp()
            dur_counts = safe_sum(eta_list) or dur_ctx.exp()

            blend_init = constraints.log_normalize(torch.log(init_counts + EPS) * α + (1 - α) * init_ctx, dim=0)
            blend_trans = constraints.log_normalize(torch.log(trans_counts + EPS) * α + (1 - α) * trans_ctx, dim=1)
            blend_dur = constraints.log_normalize(torch.log(dur_counts + EPS) * α + (1 - α) * dur_ctx, dim=1)

            if α > 0.5:
                init_logits = collapse_safe(blend_init, dim=0)
                transition_logits = collapse_safe(blend_trans, dim=1)
                duration_logits = collapse_safe(blend_dur, dim=1)
            else:
                init_logits, transition_logits, duration_logits = blend_init, blend_trans, blend_dur

            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0)
            emission_pdf = self.emission_module.initialize(
                X=all_X,
                posterior=all_gamma,
                theta=aligned_theta,
                context=aligned_theta,
                theta_scale=theta_scale
            )

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        return {
            "emission_pdf": emission_pdf,
            "initial_pdf": torch.distributions.Categorical(logits=init_logits),
            "duration_pdf": torch.distributions.Categorical(logits=duration_logits),
            "transition_pdf": torch.distributions.Categorical(logits=transition_logits),
        }

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 15,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_D_from_X: bool = False,
        theta: Optional[torch.Tensor] = None,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        X = X.to(self.device, dtype=DTYPE) if torch.is_tensor(X) else X
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, theta=theta)
        max_len = max(X_valid.lengths)
        B = len(X_valid.sequence)

        self._convergence = ConvergenceMonitor(
            tol=tol,
            n_init=n_init,
            max_iter=max_iter,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        aligned_theta = self._align_theta(theta, sum(X_valid.lengths))

        # --- Prebuild padded tensors for batching ---
        seq_tensor = torch.zeros((B, max_len, X_valid.sequence[0].shape[-1]), device=self.device, dtype=DTYPE)
        mask = torch.zeros(B, max_len, device=self.device, dtype=DTYPE)
        for b, seq in enumerate(X_valid.sequence):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq.to(self.device, dtype=DTYPE)
            mask[b, :L] = 1.0
        seq_exp_base = seq_tensor.unsqueeze(2)  # [B, T, 1, D]

        # --- Main EM loop over initializations ---
        best_score, best_state = -float("inf"), None
        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # --- Initialize / sample parameters ---
            mode = "sample" if run_idx > 0 or sample_D_from_X else "estimate"
            params = self._model_params(X_valid, theta=aligned_theta, mode=mode)

            init_pdf = params["initial_pdf"]
            transition_pdf = params["transition_pdf"]
            duration_pdf = params["duration_pdf"]
            emission_pdf = params["emission_pdf"]

            seq_exp = seq_exp_base.expand(-1, -1, self.n_states, -1)
            log_probs_full = emission_pdf.log_prob(seq_exp)
            mask_exp = mask.unsqueeze(2)  # [B, T, 1]
            X_valid.log_probs = log_probs_full * mask_exp

            base_ll = self._compute_log_likelihood(X_valid).sum()
            self._convergence.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                gamma_list, xi_list, eta_list = self._compute_posteriors(X_valid, theta=aligned_theta)

                gamma_tensor = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0) \
                    if gamma_list else torch.zeros(1, self.n_states, device=self.device, dtype=DTYPE)

                xi_tensor = torch.cat([x for x in xi_list if x is not None], dim=0) if xi_list else None
                eta_tensor = torch.cat([e for e in eta_list if e is not None], dim=0) if eta_list else None

                init_counts = gamma_tensor.sum(0)
                trans_counts = xi_tensor.sum(0) if xi_tensor is not None else None
                dur_counts = eta_tensor.sum(0) if eta_tensor is not None else None

                init_counts = init_counts if (init_counts >= EPS).all() else init_pdf.probs
                trans_counts = trans_counts if (trans_counts.sum(-1) >= EPS).all() else transition_pdf.probs
                dur_counts = dur_counts if (dur_counts.sum(-1) >= EPS).all() else duration_pdf.probs

                init_counts = init_counts / init_counts.sum().clamp_min(EPS)
                trans_counts = trans_counts / trans_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
                dur_counts = dur_counts / dur_counts.sum(dim=1, keepdim=True).clamp_min(EPS)

                α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
                α = float(max(α_min, α_max * (1.0 - it / max_iter)))
                init_pdf = torch.distributions.Categorical(probs=α * init_counts + (1 - α) * init_pdf.probs)
                transition_pdf = torch.distributions.Categorical(probs=α * trans_counts + (1 - α) * transition_pdf.probs)
                duration_pdf = torch.distributions.Categorical(probs=α * dur_counts + (1 - α) * duration_pdf.probs)

                emission_pdf = self.emission_module.initialize(
                    X=seq_tensor.view(-1, seq_tensor.shape[-1]),
                    posterior=gamma_tensor,
                    theta=aligned_theta,
                    context=aligned_theta
                )
                self._params["emission_pdf"] = emission_pdf

                log_probs_full = emission_pdf.log_prob(seq_exp) / seq_exp.shape[-1]
                X_valid.log_probs = log_probs_full * mask_exp

                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self._convergence.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # --- Track best model ---
            run_score = float(curr_ll.item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    "initial_pdf": torch.distributions.Categorical(probs=init_pdf.probs.clone()),
                    "duration_pdf": torch.distributions.Categorical(probs=duration_pdf.probs.clone()),
                    "transition_pdf": torch.distributions.Categorical(probs=transition_pdf.probs.clone()),
                    "emission_pdf": emission_pdf
                }

        # --- Restore best parameters ---
        if best_state:
            self._params.update(best_state)

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
                context_splits = [context[cum_lengths[b]:cum_lengths[b+1]] for b in range(B)]
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
            init_logits = self.initial_module.log_matrix(context=theta_b) \
                if "initial_pdf" not in self._params else self._params["initial_pdf"].logits
            trans_logits = self.transition_module.log_matrix(context=theta_b) \
                if "transition_pdf" not in self._params else self._params["transition_pdf"].logits
            dur_logits = self.duration_module.log_matrix(context=theta_b) \
                if "duration_pdf" not in self._params else self._params["duration_pdf"].logits

            trans_logits /= temp_transition
            dur_logits /= temp_duration
            trans_logits = trans_logits.clamp(-MAX_LOGITS, MAX_LOGITS)
            dur_logits = dur_logits.clamp(-MAX_LOGITS, MAX_LOGITS)

            # --- Compute emission log-probs ---
            emission_dist = self._params.get("emission_pdf") or self.emission_module.forward(context=theta_b, return_dist=True)

            if isinstance(emission_dist, torch.distributions.Independent) or isinstance(emission_dist, torch.distributions.MultivariateNormal):
                log_probs = emission_dist.log_prob(seq.unsqueeze(1))  # [T,K]
            elif isinstance(emission_dist, torch.distributions.Categorical):
                log_probs_all = F.log_softmax(emission_dist.logits, dim=-1)  # [K,C]
                seq_long = seq.long() if seq.ndim == 1 else seq
                one_hot = F.one_hot(seq_long, num_classes=log_probs_all.shape[-1]).float()  # [T,F,C]
                log_probs = (one_hot.unsqueeze(1) * log_probs_all.unsqueeze(0).unsqueeze(0)).sum([-1, -2])  # [T,K]
            else:
                raise TypeError(f"Unsupported emission type: {type(emission_dist)}")

            log_probs = log_probs.clamp(min=-MAX_LOGITS)

            # --- Observations container ---
            obs = utils.Observations([seq], log_probs=[log_probs])

            # --- Decoding ---
            if algorithm.lower() == "viterbi":
                decoded = self._viterbi(obs, theta=theta_b)
            elif algorithm.lower() == "map":
                decoded = self._map_decode(log_probs, init_logits, trans_logits, dur_logits)
            else:
                raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

            preds.append(decoded.detach().cpu())

        return preds

    @torch.no_grad()
    def _compute_log_likelihood(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Vectorized computation of per-sequence log-likelihoods: log P(X | model, theta).
        Supports variable-length sequences and context modulation.
        Returns a tensor of shape [B] containing log-likelihoods for each sequence.
        """
        neg_inf = torch.finfo(DTYPE).min / 2.0
        device = self.device
        B = len(X.sequence)

        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, device=device, dtype=DTYPE)
            return X.log_likelihoods

        # ---------------- Align theta for entire batch ----------------
        total_len = sum(X.lengths)
        aligned_theta: Optional[torch.Tensor] = None
        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            if theta.ndim == 1:
                aligned_theta = theta.unsqueeze(0).expand(total_len, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == B:  # per-sequence
                    aligned_theta = torch.cat([self._align_theta(theta[b], L) for b, L in enumerate(X.lengths)], dim=0)
                elif theta.shape[0] == total_len:  # already per-time-step
                    aligned_theta = theta
                else:
                    raise ValueError(f"Cannot align theta of shape {theta.shape} for B={B}")
            elif theta.ndim == 3 and theta.shape[0] == B:
                aligned_theta = torch.cat([theta[b, :L] for b, L in enumerate(X.lengths)], dim=0)
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape} for B={B}")

        # ---------------- Forward pass ----------------
        alpha_list = self._forward(X, theta=aligned_theta)

        # ---------------- Pad alpha for batching ----------------
        T_max = max(X.lengths)
        K, Dmax = self.n_states, self.max_duration
        alpha_pad = torch.full((B, T_max, K, Dmax), neg_inf, device=device, dtype=DTYPE)

        for b, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L == 0:
                continue
            if alpha.ndim == 3:  # [T, K, D]
                alpha_pad[b, :L, :, :] = alpha[:L]
            else:  # fallback [T, K]
                alpha_pad[b, :L, :, 0] = alpha[:L]

        # ---------------- Apply optional sequence masks ----------------
        if hasattr(X, "masks") and X.masks is not None:
            mask_pad = torch.zeros(B, T_max, device=device, dtype=DTYPE)
            for b, L in enumerate(X.lengths):
                mask_pad[b, :L] = X.masks[b].to(device=device, dtype=DTYPE)
            alpha_pad += torch.log(mask_pad.unsqueeze(-1).unsqueeze(-1) + EPS)

        # ---------------- Compute per-sequence log-likelihoods ----------------
        ll_pad = torch.logsumexp(alpha_pad, dim=(2, 3))  # sum over states and durations
        ll_list = [ll_pad[b, :X.lengths[b]].max().clone() for b in range(B)]
        X.log_likelihoods = torch.stack(ll_list)

        # ---------------- Verbose logging ----------------
        if verbose:
            ll = X.log_likelihoods
            msg = f"[compute_log_likelihood] seqs={B}, min={ll.min():.4f}, max={ll.max():.4f}, mean={ll.mean():.4f}"
            if 'logger' in globals() and logger is not None:
                logger.info(msg)
            else:
                print(msg)

        return X.log_likelihoods

    @torch.no_grad()
    def score(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0
    ) -> torch.Tensor:
        """
        Compute log-likelihoods for one or more sequences using context-modulated modules.

        Args:
            X: Tensor [T,F] or list of [T_b,F].
            theta: Optional context for sequence-dependent parameters.
            temp_transition: Temperature scaling for transition logits.
            temp_duration: Temperature scaling for duration logits.

        Returns:
            Tensor [B] of log-likelihoods.
        """

        # --- Wrap single sequence into list ---
        if torch.is_tensor(X):
            X = [X]

        B = len(X)
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        device = self.device

        if B == 0:
            return torch.tensor([], device=device, dtype=DTYPE)

        ll_list = []

        for b, seq in enumerate(X):
            seq = seq.to(dtype=DTYPE, device=device)
            T = seq.shape[0]
            if T == 0:
                ll_list.append(torch.tensor(neg_inf, device=device, dtype=DTYPE))
                continue

            # --- Emission log-probs ---
            log_B = self.emission_module.log_prob(seq, context=self._align_theta(theta, T))  # [T, K]

            # --- Module logits ---
            init_logits = self.initial_module.log_matrix(context=self._align_theta(theta, 1))  # [K]
            dur_logits = self.duration_module.log_matrix(context=self._align_theta(theta, 1))  # [K, Dmax]
            trans_logits = self.transition_module.log_matrix(context=self._align_theta(theta, 1))  # [K, K]

            # Temperature scaling
            dur_logits = dur_logits / max(temp_duration, self.min_covar)
            trans_logits = trans_logits / max(temp_transition, self.min_covar)

            # --- Forward DP ---
            alpha = torch.full((T, K), neg_inf, device=device, dtype=DTYPE)
            alpha[0] = log_B[0] + torch.log_softmax(init_logits, dim=0)

            for t in range(1, T):
                max_d = min(Dmax, t + 1)
                dur_idx = torch.arange(max_d, device=device)
                emit_sums = torch.stack([log_B[t - d + 1:t + 1].sum(dim=0) for d in dur_idx], dim=0)  # [max_d, K]

                if dur_logits.ndim == 2:
                    dur_scores = dur_logits[:, :max_d].T  # [max_d, K]
                else:
                    dur_scores = dur_logits[:max_d].unsqueeze(1).expand(-1, K)

                scores = []
                for i, d in enumerate(dur_idx):
                    start = t - d + 1
                    if start == 0:
                        trans_score = torch.zeros(K, device=device, dtype=DTYPE)
                    else:
                        trans_score = torch.logsumexp(alpha[start - 1].unsqueeze(1) + trans_logits, dim=0)
                    scores.append(trans_score + dur_scores[i] + emit_sums[i])

                alpha[t] = torch.logsumexp(torch.stack(scores, dim=0), dim=0)

            ll_list.append(torch.logsumexp(alpha[-1], dim=-1).detach())

        return torch.stack(ll_list)


    # Context
    def set_context(self, ctx: torch.Tensor):
        self._context = ctx
        if self.encoder is not None:
            self.encoder.set_context(ctx)

    def reset_context(self):
        self._context = None
        if self.encoder is not None:
            self.encoder.reset_context()

    def combine_context(self, theta: Optional[torch.Tensor], allow_broadcast: bool = True):
        if self.encoder is None:
            return theta
        return self.encoder._combine_context(theta, allow_broadcast)

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute context-modulated emission log-probabilities [B, T, K].
        Always returns numeric log-probs, never a Distribution.
        Supports inputs: [F], [T,F], [B,F], [B,T,F].

        Tiny random jitter is added to prevent collapsed states during EM.
        """

        # --- Normalize input to [B,T,F] ---
        if X.ndim == 1:           # [F] -> [1,1,F]
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:
            if X.shape[1] == self.n_features:  # [T,F] -> [1,T,F]
                X = X.unsqueeze(0)
            else:                              # [B,F] -> [B,1,F]
                X = X.unsqueeze(1)
        elif X.ndim != 3:
            raise ValueError(f"Unsupported input shape {X.shape}")

        B, T, F = X.shape

        # --- Align context ---
        theta_batch = self.combine_context(theta)
        if theta_batch is not None:
            if theta_batch.ndim == 2:            # [B,H] -> [B,T,H]
                theta_batch = theta_batch.unsqueeze(1).expand(-1, T, -1)
            elif theta_batch.ndim == 3:          # [B,T,H] already ok
                if theta_batch.shape[0] != B or theta_batch.shape[1] != T:
                    theta_batch = theta_batch.expand(B, T, -1)
            else:
                raise ValueError(f"Unsupported theta shape {theta_batch.shape}")

        # --- Compute log-probs vectorized ---
        logp_tensor = self.emission_module.log_prob(X, context=theta_batch)  # [B,T,K] or [B,T]
        if logp_tensor.ndim == 2:                 # [B,T] -> [B,T,1]
            logp_tensor = logp_tensor.unsqueeze(-1)

        # --- Sanitize numerics to avoid EM collapse ---
        dtype_info = torch.finfo(logp_tensor.dtype)
        min_val = dtype_info.min / 2
        logp_tensor = torch.nan_to_num(logp_tensor, nan=min_val, neginf=min_val, posinf=dtype_info.max / 2)

        # --- Optional tiny jitter to prevent collapsed states ---
        if B > 1 and T > 1:
            logp_tensor = logp_tensor + 1e-6 * torch.rand_like(logp_tensor)

        return logp_tensor

    def _contextual_duration_pdf(self, theta: Optional[ContextualVariables] = None) -> torch.Tensor:
        """
        Compute context-modulated duration log-probabilities [K, Dmax] without batching.

        Tiny random jitter is added to avoid collapsed duration states during EM.
        """
        base_logits = self._duration_logits  # [K, Dmax]

        # Apply context if provided
        if theta is not None:
            log_duration = self.duration_module._apply_context(base_logits, theta)
            if not torch.is_tensor(log_duration):
                raise TypeError(f"Expected tensor from _apply_context, got {type(log_duration)}")
        else:
            log_duration = base_logits.clone()

        # Clean NaNs/Infs and add tiny jitter
        dtype_info = torch.finfo(log_duration.dtype)
        log_duration = torch.nan_to_num(log_duration, nan=dtype_info.min / 2, posinf=dtype_info.max / 2)
        log_duration = log_duration + 1e-8 * torch.rand_like(log_duration)

        # Normalize along duration dimension
        log_duration = log_duration - torch.logsumexp(log_duration, dim=-1, keepdim=True)

        return log_duration  # [K, Dmax]

    def _contextual_transition_matrix(self, theta: Optional[ContextualVariables] = None) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities [K, K] (non-batched).

        Tiny random jitter is added to prevent collapsed transitions during EM.
        Structural transition constraints are enforced.
        """
        log_transition = self._transition_logits.clone()  # [K, K]

        if theta is not None:
            log_transition = self.transition_module._apply_context(log_transition, theta)
            if not torch.is_tensor(log_transition):
                raise TypeError(f"Expected tensor from _apply_context, got {type(log_transition)}")

        # Apply structural mask
        if hasattr(self, "transition_type"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_type).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask, float("-inf"))

        # Clean and normalize
        dtype_info = torch.finfo(log_transition.dtype)
        log_transition = torch.nan_to_num(log_transition, nan=dtype_info.min / 2, posinf=dtype_info.max / 2)
        log_transition = log_transition + 1e-8 * torch.rand_like(log_transition)
        log_transition = log_transition - torch.logsumexp(log_transition, dim=-1, keepdim=True)

        return log_transition  # [K, K]


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

