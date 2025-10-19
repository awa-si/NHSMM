# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from nhsmm.defaults import Emission, Duration, Transition, DTYPE, EPS, HSMMError
from nhsmm import utils, constraints, SeedGenerator, ConvergenceHandler, ContextEncoder


class HSMM(ABC):
    """
    Hidden Semi-Markov Model (HSMM) base class.

    Supports:
        - Emissions: state-dependent observation likelihoods.
        - Transitions: contextual transition probabilities.
        - Durations: explicit duration modeling per state.
        - Optional neural context encoding.

    Key methods to implement/override:
        - forward(X)
        - _compute_log_likelihood(X)
        - score(X, lengths=None, by_sample=True)
        - sample(T)
        - information(X, criterion, lengths=None)
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        alpha: float = None,
        min_covar: float = None,
        seed: Optional[int] = None,
        transition_constraint: Any = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_constraint = transition_constraint or constraints.Transitions.SEMI
        self._context: Optional[torch.Tensor] = None
        self.encoder: Optional[nn.Module] = None
        self._seed_gen = SeedGenerator(seed)
        self._params: Dict[str, Any] = {}
        self.max_duration = max_duration
        self.context_dim = context_dim
        self.n_features = n_features
        self.min_covar = min_covar
        self.n_states = n_states
        self.alpha = alpha

        self._init_buffers()
        self._init_modules()
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    def _init_modules(self):
        self.emission_module = Emission(self.n_states, self.n_features, self.min_covar, self.context_dim)
        self.duration_module = Duration(self.n_states, self.max_duration, self.context_dim)
        self.transition_module = Transition(self.n_states, self.context_dim)

    def _init_buffers(self):
        device = next(self.buffers(), torch.zeros(1, dtype=DTYPE)).device

        def sample_logits(shape):
            if len(shape) == 1:
                probs = constraints.sample_probs(self.alpha, shape)
            elif len(shape) == 2:
                if shape[1] == self.max_duration:
                    probs = constraints.sample_probs(self.alpha, shape)
                else:
                    probs = constraints.sample_transition(self.alpha, shape[0], self.transition_constraint)
            else:
                raise ValueError(f"Unsupported shape {shape} for logits sampling.")

            logits = torch.log(probs.clamp_min(1e-12)).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        # Base parameters
        init_logits, init_shape = sample_logits((self.n_states,))
        transition_logits, transition_shape = sample_logits((self.n_states, self.n_states))
        duration_logits, duration_shape = sample_logits((self.n_states, self.max_duration))

        # Register buffers
        self.register_buffer("_init_logits", init_logits)
        self.register_buffer("_transition_logits", transition_logits)
        self.register_buffer("_duration_logits", duration_logits)

        # Store shapes for later
        self._init_shape = init_shape
        self._duration_shape = duration_shape
        self._transition_shape = transition_shape

        # Optional super-states
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_init_logits, super_init_shape = sample_logits((n_super_states,))
            super_transition_logits, super_transition_shape = sample_logits((n_super_states, n_super_states))

            self.register_buffer("_super_init_logits", super_init_logits)
            self.register_buffer("_super_transition_logits", super_transition_logits)

            self._super_init_shape = super_init_shape
            self._super_transition_shape = super_transition_shape

        # Snapshot of initialization means
        summary = [
            init_logits.mean(),
            transition_logits.mean(),
            duration_logits.mean()
        ]
        if n_super_states > 1:
            summary += [super_init_logits.mean(), super_transition_logits.mean()]

        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def init_logits(self) -> torch.Tensor:
        return self._init_logits

    @init_logits.setter
    def init_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._init_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states,):
            raise ValueError(f"init_logits logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")
        
        norm_val = logits.logsumexp(dim=0)
        zero = torch.tensor(0., dtype=DTYPE, device=logits.device)
        if not torch.allclose(norm_val, zero, atol=1e-6):
            raise ValueError(f"init_logits logits must normalize (logsumexp==0); got {norm_val.item():.3e}")

        self._init_logits.copy_(logits)

    @property
    def init_probs(self) -> torch.Tensor:
        return self._init_logits.softmax(0)

    @property
    def transition_logits(self) -> torch.Tensor:
        return self._transition_logits

    @transition_logits.setter
    def transition_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._transition_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"transition_logits logits must have shape ({self.n_states},{self.n_states})")
        
        row_norm = logits.logsumexp(dim=1)
        zeros = torch.zeros_like(row_norm)
        if not torch.allclose(row_norm, zeros, atol=1e-6):
            raise ValueError(f"Rows of transition_logits logits must normalize (logsumexp==0); got {row_norm.tolist()}")

        if not constraints.is_valid_transition(logits, self.transition_constraint):
            raise ValueError("transition_logits logits do not satisfy transition constraints")

        self._transition_logits.copy_(logits)

    @property
    def transition_probs(self) -> torch.Tensor:
        return self._transition_logits.softmax(-1)

    @property
    def duration_logits(self) -> torch.Tensor:
        return self._duration_logits

    @duration_logits.setter
    def duration_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._duration_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"duration_logits logits must have shape ({self.n_states},{self.max_duration})")
        
        row_norm = logits.logsumexp(dim=1)
        zeros = torch.zeros_like(row_norm)
        if not torch.allclose(row_norm, zeros, atol=1e-6):
            raise ValueError(f"Rows of duration_logits logits must normalize (logsumexp==0); got {row_norm.tolist()}")

        self._duration_logits.copy_(logits)

    @property
    def duration_probs(self) -> torch.Tensor:
        return self._duration_logits.softmax(-1)

    @property
    def super_init(self):
        if hasattr(self, "_super_init_logits"):
            return self._super_init_logits
        return None

    @property
    def super_transition(self):
        if hasattr(self, "_super_transition_logits"):
            return self._super_transition_logits
        return None

    @property
    def pdf(self) -> Any:
        return self._params.get('emission_pdf')

    @property
    @abstractmethod
    def dof(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        raise NotImplementedError

    @abstractmethod
    def initialize_emissions(self, X: torch.Tensor, method: str = "moment") -> None:
        """
        Abstract method to initialize emission parameters from raw data.
        """
        raise NotImplementedError("Subclasses must implement initialize_emissions().")

    @abstractmethod
    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None
    ) -> Distribution:
        raise NotImplementedError(
            "Subclasses must implement _estimate_emission_pdf. "
            "It should return a Distribution supporting `.log_prob` and handle optional context theta."
        )

    def attach_encoder(
        self,
        encoder: nn.Module,
        batch_first: bool = True,
        pool: str = "mean",
        n_heads: int = 4,
        max_seq_len: int = 1024
    ):
        """
        Attach a ContextEncoder wrapper to the model.
        - encoder: a nn.Module that outputs (B,T,F) or (B,F)
        - batch_first: True if input shape is (B,T,F)
        - pool: "mean", "last", "max", "attn", "mha"
        - n_heads: number of heads if pool="mha"
        - max_seq_len: maximum expected sequence length (for relative positions)
        """
        self.encoder = ContextEncoder(
            encoder=encoder,
            batch_first=batch_first,
            pool=pool,
            device=self.device,
            n_heads=n_heads,
            max_seq_len=max_seq_len
        )

    def encode_observations(self, X: torch.Tensor, pool: Optional[str] = None, store: bool = True) -> Optional[torch.Tensor]:
        """
        Encode observations into context vectors θ using the attached encoder.
        Automatically uses the encoder's pooling mode unless overridden.
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        # Normalize input to (B,T,F)
        if X.ndim == 1:  # single feature vector (F,)
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:  # (T,F)
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected input of shape (F,), (T,F) or (B,T,F), got {X.shape}")

        # Forward with temporary pooling override
        original_pool = getattr(self.encoder, 'pool', None)
        if pool is not None:
            self.encoder.pool = pool

        try:
            out = self.encoder(X)
        except Exception as e:
            raise RuntimeError(f"Encoder forward() failed: {e}")
        finally:
            # Restore original pool
            if pool is not None and original_pool is not None:
                self.encoder.pool = original_pool

        # The encoder wrapper should already set _context
        vec = getattr(self, "_context", None)

        if vec is None and store:
            # Fallback mean pooling
            if out.ndim == 3:
                vec = out.mean(dim=1)
            else:
                vec = out.unsqueeze(0)
            vec = nn.functional.layer_norm(vec, vec.shape[-1:])
            vec = torch.clamp(vec, -10.0, 10.0)
            self._context = vec.detach()

        return vec

    def check_constraints(self, value: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """
        Validate observations against the emission PDF support and event shape.
        Optionally clamps values outside the PDF support.

        Args:
            value (torch.Tensor): Input tensor of shape (batch, *event_shape) or (*event_shape) for single sample.
            clamp (bool): If True, clip values to the valid support instead of raising errors.

        Returns:
            torch.Tensor: Validated (and possibly clamped) tensor.

        Raises:
            RuntimeError: If the emission PDF is not initialized.
            ValueError: If input shape does not match the PDF event shape.
        """
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized. Ensure `sample_emission_pdf()` or `encode_observations()` has been called."
            )

        # Align device and dtype
        if hasattr(pdf, "mean") and isinstance(pdf.mean, torch.Tensor):
            value = value.to(device=pdf.mean.device, dtype=pdf.mean.dtype)

        # Add batch dimension if missing
        event_shape = pdf.event_shape or ()
        if value.ndim == len(event_shape):
            value = value.unsqueeze(0)  # treat as single batch

        # Vectorized support check
        support_mask = pdf.support.check(value)
        if clamp:
            if hasattr(pdf.support, "clamp"):
                value = torch.where(support_mask, value, pdf.support.clamp(value))
            else:
                # fallback: clip to min/max if available
                if hasattr(pdf.support, "lower_bound") and hasattr(pdf.support, "upper_bound"):
                    value = value.clamp(min=pdf.support.lower_bound, max=pdf.support.upper_bound)
        elif not torch.all(support_mask):
            bad_vals = value[~support_mask].flatten().unique()
            raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Validate event shape
        expected_ndim = 1 + len(event_shape)  # batch + event dims
        if value.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim} input dimensions (batch + event dims), got {value.ndim}."
            )

        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(
                f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}."
            )

        return value

    def _combine_context(
        self,
        X: Optional[Observations] = None,
        theta: Optional[ContextualVariables] = None,
        reduce: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Combine observation features with contextual variables into a consistent batch tensor.

        Args:
            X: Observations (optional)
            theta: ContextualVariables (optional)
            reduce: If True, collapse time dimension via mean features

        Returns:
            Tensor [B, T, F+H] or [B, 1, F+H] if reduced, None if no input
        """
        if X is None and theta is None:
            return None

        # Pad sequences and extract masks
        X_batch, mask_X = (X.pad_sequences(return_mask=True) if X else (None, None))
        if theta:
            if getattr(theta, "time_dependent", False):
                theta_batch, mask_theta = theta.pad_sequences(return_mask=True)
            else:
                theta_tensor = theta.cat(dim=-1).unsqueeze(1)
                theta_batch, mask_theta = theta_tensor, None
        else:
            theta_batch, mask_theta = None, None

        if X_batch is None and theta_batch is None:
            return None

        # Determine batch and time dimensions
        B = X_batch.shape[0] if X_batch is not None else theta_batch.shape[0]
        T = max(X_batch.shape[1] if X_batch is not None else 0,
                theta_batch.shape[1] if theta_batch is not None else 0)

        # Pad sequences to same length
        if X_batch is not None and X_batch.shape[1] != T:
            X_batch = F.pad(X_batch, (0, 0, 0, T - X_batch.shape[1]))
        if theta_batch is not None and theta_batch.shape[1] != T:
            theta_batch = F.pad(theta_batch, (0, 0, 0, T - theta_batch.shape[1]))

        # Concatenate along feature dimension
        combined = torch.cat([t for t in [X_batch, theta_batch] if t is not None], dim=-1)

        # Optional reduction along time dimension
        if reduce:
            mask = mask_X if mask_X is not None else torch.ones(B, T, dtype=torch.bool, device=combined.device)
            combined = combined.masked_fill(~mask.unsqueeze(-1), 0.0)
            lengths = mask.sum(1).unsqueeze(-1).clamp(min=1)
            combined = combined.sum(1, keepdim=True) / lengths

        return torch.nan_to_num(combined, nan=0.0, posinf=1e8, neginf=-1e8)

    def _contextual_emission_pdf(
        self,
        X: Observations,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated emission log-probabilities for batched sequences.

        Returns: [B, T, K]
        """
        X_batch, mask = X.pad_sequences(return_mask=True)
        theta_batch = self._combine_context(X=None, theta=theta)

        base_logp = self.emission_module.log_prob(X_batch, theta_batch)
        base_logp = torch.clamp(torch.nan_to_num(base_logp, nan=-1e8, posinf=-1e8, neginf=-1e8), -1e6, 1e6)

        # Apply context delta if available
        if theta is not None:
            delta = self.emission_module._apply_context(base_logp, theta_batch)
            logp = base_logp + delta
        else:
            logp = base_logp

        # Mask padding positions
        logp = logp.masked_fill(~mask.unsqueeze(-1), -1e8)
        return F.log_softmax(logp, dim=-1)

    def _contextual_duration_pdf(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated duration log-probabilities.

        Returns: [B, K, max_duration]
        """
        base_duration = self._duration_logits.unsqueeze(0)  # [1, K, max_duration]
        theta_batch = self._combine_context(X=None, theta=theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        log_duration = self.duration_module._apply_context(
            base_duration.expand(B, *base_duration.shape[1:]), theta_batch
        )
        log_duration = torch.nan_to_num(log_duration, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return log_duration - torch.logsumexp(log_duration, dim=2, keepdim=True)

    def _contextual_transition_matrix(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities.

        Returns: [B, K, K]
        """
        base_transition = self._transition_logits.unsqueeze(0)  # [1, K, K]
        theta_batch = self._combine_context(X=None, theta=theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        log_transition = self.transition_module._apply_context(
            base_transition.expand(B, *base_transition.shape[1:]), theta_batch
        )

        # Apply optional transition constraints
        if hasattr(self, "transition_constraint"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_constraint).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask.unsqueeze(0), -1e8)

        log_transition = torch.nan_to_num(log_transition, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return log_transition - torch.logsumexp(log_transition, dim=2, keepdim=True)

    def to_observations(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Converts raw input X (and optional context theta) into a structured Observations object.
        Supports both batched and flattened inputs, and automatically computes log-probabilities.

        Returns:
            utils.Observations(sequence, log_probs, lengths)
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        X_valid = self.check_constraints(X).to(device=device, dtype=DTYPE)

        # ---- Context alignment ----
        if theta is None and hasattr(self, "_context") and self._context is not None:
            theta = self._context

        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            if X_valid.ndim == 3 and theta.ndim == 2:
                # Expand per-batch context across time dimension
                theta = theta.unsqueeze(1).expand(-1, X_valid.shape[1], -1)
            elif X_valid.ndim == 2 and theta.ndim == 3 and theta.shape[1] == 1:
                theta = theta.squeeze(1)

        # ---- Sequence length validation ----
        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]
        if lengths is None:
            seq_lengths = [total_len]
        else:
            if not isinstance(lengths, (list, tuple)) or not all(isinstance(l, int) for l in lengths):
                raise TypeError("`lengths` must be a list of integers.")
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths

        # ---- Split into sequences ----
        if X_valid.ndim == 2:
            sequences = list(torch.split(X_valid, seq_lengths))
        elif X_valid.ndim == 3:
            if len(seq_lengths) == X_valid.shape[0]:
                sequences = [X_valid[i] for i in range(X_valid.shape[0])]
            else:
                sequences = list(torch.split(X_valid.reshape(-1, X_valid.shape[-1]), seq_lengths))
        else:
            raise ValueError(f"Unsupported input shape {tuple(X_valid.shape)}")

        # ---- Compute vectorized log-probs ----
        log_probs_list = []
        for i, seq in enumerate(sequences):
            # Select contextual theta for this sequence
            seq_theta = None
            if theta is not None:
                if theta.ndim == 2 and theta.shape[0] == len(sequences):
                    seq_theta = theta[i].unsqueeze(0)
                elif theta.ndim == 3 and i < theta.shape[0]:
                    seq_theta = theta[i]
                else:
                    seq_theta = theta

            # Compute emission log-probs or retrieve distribution
            logp_or_dist = self._contextual_emission_pdf(seq, seq_theta)

            if isinstance(logp_or_dist, torch.distributions.Distribution):
                # Fully vectorized log_prob computation
                log_probs = logp_or_dist.log_prob(seq.unsqueeze(-2))
                if log_probs.ndim > 2:
                    log_probs = log_probs.sum(dim=list(range(2, log_probs.ndim)))
            elif torch.is_tensor(logp_or_dist):
                log_probs = logp_or_dist
                if log_probs.ndim == 1:
                    log_probs = log_probs.unsqueeze(-1)
            else:
                raise TypeError(f"Unsupported return type from _contextual_emission_pdf: {type(logp_or_dist)}")

            log_probs_list.append(log_probs.to(dtype=DTYPE, device=device))

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=seq_lengths
        )

    def _estimate_model_params(
        self,
        X: utils.Observations,
        theta: Optional[utils.ContextualVariables] = None
    ) -> dict[str, Any]:
        """
        M-step: Estimate updated HSMM parameters from posterior expectations (gamma, xi, eta).
        Ensures log-domain normalization, device consistency, and optional contextual emission updates.
        """
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)
        device = self.device
        K = self.n_states

        # -------------------------------
        # Initial state distribution π
        # -------------------------------
        init_terms = []
        for g in gamma_list:
            if g is not None and g.numel() > 0:
                init_terms.append(g[0])
        if init_terms:
            pi_counts = torch.stack(init_terms, dim=0).sum(dim=0).clamp_min(EPS)
            new_init = constraints.log_normalize(torch.log(pi_counts), dim=0)
        else:
            new_init = self.init_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Transition matrix A
        # -------------------------------
        xi_valid = [x for x in xi_list if x is not None and x.numel() > 0]
        if xi_valid:
            xi_cat = torch.cat(xi_valid, dim=0).to(device=device, dtype=DTYPE)
            trans_counts = xi_cat.sum(dim=0).clamp_min(EPS)
            new_transition = constraints.log_normalize(torch.log(trans_counts), dim=1)
        else:
            new_transition = self.transition_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Duration distributions P(d | z)
        # -------------------------------
        eta_valid = [e for e in eta_list if e is not None and e.numel() > 0]
        if eta_valid:
            eta_cat = torch.cat(eta_valid, dim=0).to(device=device, dtype=DTYPE)
            dur_counts = eta_cat.sum(dim=0).clamp_min(EPS)
            new_duration = constraints.log_normalize(torch.log(dur_counts), dim=1)
        else:
            new_duration = self.duration_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Emission PDF (contextual / static)
        # -------------------------------
        new_pdf = self._params.get("emission_pdf", None)

        if theta is None:
            theta = getattr(self, "_context", None)

        if X.sequence and any(seq.numel() > 0 for seq in X.sequence):
            all_X = torch.cat(X.sequence, dim=0).to(device=device, dtype=DTYPE)
            all_gamma = torch.cat(
                [g for g in gamma_list if g is not None and g.numel() > 0], dim=0
            ).to(device=device, dtype=DTYPE)

            # --- Align contextual variable θ ---
            theta_full = None
            if theta is not None:
                theta = theta.to(device=device, dtype=DTYPE)
                if theta.ndim == 1:
                    theta_full = theta.unsqueeze(0).expand(all_X.shape[0], -1)
                elif theta.ndim == 2:
                    if theta.shape[0] != all_X.shape[0]:
                        repeats = all_X.shape[0] // theta.shape[0]
                        if all_X.shape[0] % theta.shape[0] != 0:
                            raise ValueError(
                                f"Theta shape {tuple(theta.shape)} cannot align with sequence length {all_X.shape[0]}"
                            )
                        theta_full = theta.repeat_interleave(repeats, dim=0)
                    else:
                        theta_full = theta
                elif theta.ndim == 3:
                    theta_full = theta.reshape(-1, theta.shape[-1])
                    if theta_full.shape[0] != all_X.shape[0]:
                        raise ValueError(
                            f"Context θ has shape {tuple(theta.shape)} but expected total length {all_X.shape[0]}"
                        )
                else:
                    raise ValueError(f"Unsupported θ shape: {tuple(theta.shape)}")

            # --- Update emission model ---
            if hasattr(self, "_estimate_emission_pdf"):
                new_pdf = self._estimate_emission_pdf(all_X, all_gamma, theta_full)
            elif hasattr(self, "_contextual_emission_pdf"):
                new_pdf = self._contextual_emission_pdf(all_X, theta_full)
            else:
                raise RuntimeError(
                    "No emission estimator available — model missing _estimate_emission_pdf or _contextual_emission_pdf."
                )

        # -------------------------------
        # Assemble updated parameters
        # -------------------------------
        return {
            "init_logits": new_init.to(device=device, dtype=DTYPE),
            "transition_logits": new_transition.to(device=device, dtype=DTYPE),
            "duration_logits": new_duration.to(device=device, dtype=DTYPE),
            "emission_pdf": new_pdf,
        }

    def sample_model_params(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        inplace: bool = True
    ) -> Dict[str, Any]:
        """
        Sample HSMM model parameters (π, transition_logits, duration_logits, emission_pdf) in log-space.
        Optionally uses context-conditioned sampling via theta, strictly aligned to X.
        """
        α = getattr(self, "alpha", 1.0)

        # --- Initial state distribution π ---
        init_logits = torch.log(
            constraints.sample_probs(α, (self.n_states,)).to(dtype=DTYPE, device=self.device).clamp_min(EPS)
        )

        # --- Align theta to X sequences if both are provided ---
        aligned_theta = None
        if theta is not None and X is not None:
            T = X.shape[0] if X.ndim == 2 else X.shape[1]  # total sequence length
            if theta.ndim == 1:
                aligned_theta = theta.view(1, -1).expand(T, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == T:
                    aligned_theta = theta
                elif T % theta.shape[0] == 0:
                    aligned_theta = theta.repeat(T // theta.shape[0], 1)
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot align with sequence length {T}")
            elif theta.ndim == 3:
                if theta.shape[0] == T:
                    aligned_theta = theta
                elif T % theta.shape[0] == 0:
                    aligned_theta = theta[:, -1, :].repeat(T // theta.shape[0], 1)
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot align with sequence length {T}")
            else:
                raise ValueError(f"Unexpected theta shape {theta.shape}")
        else:
            aligned_theta = theta

        # --- Transition logits ---
        if aligned_theta is not None and hasattr(self, "_contextual_transition_matrix"):
            transition_logits = torch.log(self._contextual_transition_matrix(aligned_theta).clamp_min(EPS))
        else:
            transition_logits = torch.log(
                constraints.sample_transition(α, self.n_states, constraints.Transitions.SEMI)
                .to(dtype=DTYPE, device=self.device)
                .clamp_min(EPS)
            )

        # --- Duration logits ---
        if aligned_theta is not None and hasattr(self, "_contextual_duration_pdf"):
            duration_logits = torch.log(self._contextual_duration_pdf(aligned_theta).clamp_min(EPS))
        else:
            duration_logits = torch.log(
                constraints.sample_probs(α, (self.n_states, self.max_duration))
                .to(dtype=DTYPE, device=self.device)
                .clamp_min(EPS)
            )

        # --- Emission PDF ---
        if aligned_theta is not None and hasattr(self, "_contextual_emission_pdf") and X is not None:
            emission_pdf = self._contextual_emission_pdf(X, aligned_theta)
        else:
            emission_pdf = self.sample_emission_pdf(X)

        params = {
            "init_logits": init_logits,
            "transition_logits": transition_logits,
            "duration_logits": duration_logits,
            "emission_pdf": emission_pdf
        }

        if inplace:
            if hasattr(self, "init_logits"):
                self.init_logits.data.copy_(init_logits)
            if hasattr(self, "transition_logits"):
                self.transition_logits.data.copy_(transition_logits)
            if hasattr(self, "duration_logits"):
                self.duration_logits.data.copy_(duration_logits)
            self._params["emission_pdf"] = emission_pdf

        return params


    # HSMM precessing
    def _forward(self, X: utils.Observations) -> List[torch.Tensor]:
        K, Dmax = self.n_states, self.max_duration
        init_logits = self.init_logits.to(self.device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(self.device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(self.device, dtype=DTYPE)
        neg_inf = -torch.inf

        alpha_list = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=self.device, dtype=DTYPE))
                continue

            seq_probs = seq_probs.to(self.device, dtype=DTYPE)
            log_alpha = torch.full((seq_len, K, Dmax), neg_inf, device=self.device, dtype=DTYPE)

            # cumulative sums for fast segment sums
            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=self.device),
                                        torch.cumsum(seq_probs, dim=0)))

            # t = 0 initialization
            max_d0 = min(Dmax, seq_len)
            durations0 = torch.arange(1, max_d0 + 1, device=self.device)
            emit_sums0 = (cumsum_emit[durations0] - cumsum_emit[0]).T  # (K, max_d0)
            log_alpha[0, :, :max_d0] = init_logits.unsqueeze(1) + duration_logits[:, :max_d0] + emit_sums0

            init_row = torch.full((K, Dmax), neg_inf, device=self.device, dtype=DTYPE)
            init_row[:, 0] = init_logits

            for t in range(1, seq_len):
                max_dt = min(Dmax, t + 1)
                durations = torch.arange(1, max_dt + 1, device=self.device)
                starts = t - durations + 1  # (max_dt,)

                # --- emission sums for all durations ending at t ---
                # gather the cumsum values at start indices
                starts_exp = starts.unsqueeze(1).expand(-1, K)  # (max_dt, K)
                end_cumsum = cumsum_emit[t + 1].unsqueeze(0).expand(max_dt, -1)  # (max_dt, K)
                start_cumsum = torch.gather(cumsum_emit, 0, starts_exp)          # (max_dt, K)
                emit_sums_t = (end_cumsum - start_cumsum).T                       # (K, max_dt)

                # --- previous alpha for durations ---
                idx = torch.clamp(starts - 1, min=0)
                prev_alpha_first = log_alpha[idx, :, 0]  # (max_dt, K)
                mask = (starts == 0).unsqueeze(1)
                prev_alpha_first = torch.where(mask, init_logits.unsqueeze(0), prev_alpha_first)

                prev_alpha_sum = torch.logsumexp(prev_alpha_first.unsqueeze(2) + transition_logits.unsqueeze(0), dim=1).T  # (K, max_dt)

                log_alpha[t, :, :max_dt] = prev_alpha_sum + duration_logits[:, :max_dt] + emit_sums_t

            alpha_list.append(log_alpha)
        return alpha_list

    def _backward(self, X: utils.Observations) -> List[torch.Tensor]:
        K, Dmax = self.n_states, self.max_duration
        transition_logits = self.transition_logits.to(self.device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(self.device, dtype=DTYPE)
        neg_inf = -torch.inf

        beta_list = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=self.device, dtype=DTYPE))
                continue

            seq_probs = seq_probs.to(self.device, dtype=DTYPE)
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=self.device, dtype=DTYPE)
            log_beta[-1].fill_(0.0)

            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=self.device),
                                        torch.cumsum(seq_probs, dim=0)))

            for t in reversed(range(seq_len - 1)):
                max_dt = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dt + 1, device=self.device)
                ends = t + durations  # end indices for each duration

                # --- emission sums ---
                start_idx = t * torch.ones(max_dt, dtype=torch.long, device=self.device)
                end_idx = ends
                # gather cumsum
                start_cumsum = cumsum_emit[start_idx].T       # (K, max_dt)
                end_cumsum = cumsum_emit[end_idx].T           # (K, max_dt)
                emit_sums = end_cumsum - start_cumsum        # (K, max_dt)

                # --- beta_next contributions ---
                beta_next = log_beta[ends - 1, :, 0].T       # (K, max_dt)
                dur_scores = duration_logits[:, :max_dt]     # (K, max_dt)

                combined = emit_sums + dur_scores + beta_next
                log_beta[t, :, 0] = torch.logsumexp(combined, dim=1)

                # shift for durations > 1
                if max_dt > 1:
                    shift_len = min(max_dt - 1, seq_len - t - 1)
                    log_beta[t, :, 1:shift_len + 1] = log_beta[t + 1, :, :shift_len] + seq_probs[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)
        return beta_list

    def _compute_posteriors(self, X: utils.Observations) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        K, Dmax = self.n_states, self.max_duration

        alpha_list = self._forward(X)
        beta_list = self._backward(X)

        gamma_vec: List[torch.Tensor] = []
        xi_vec: List[torch.Tensor] = []
        eta_vec: List[torch.Tensor] = []

        init_logits = self.init_logits.to(device=self.device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=self.device, dtype=DTYPE)

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            if seq_len == 0:
                gamma_vec.append(torch.zeros((seq_len, K), dtype=DTYPE, device=self.device))
                eta_vec.append(torch.zeros((seq_len, K, Dmax), dtype=DTYPE, device=self.device))
                xi_vec.append(torch.zeros((max(seq_len - 1, 0), K, K), dtype=DTYPE, device=self.device))
                continue

            alpha = alpha.to(self.device, dtype=DTYPE)
            beta = beta.to(self.device, dtype=DTYPE)
            seq_probs = seq_probs.to(self.device, dtype=DTYPE)

            # --- gamma ---
            log_gamma = torch.logsumexp(alpha + beta, dim=2)
            log_gamma_norm = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
            gamma = torch.clamp(log_gamma_norm.exp(), min=EPS)
            gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(EPS)
            gamma_vec.append(gamma)

            # --- eta ---
            log_eta = alpha + beta
            log_eta_flat = log_eta.view(seq_len, -1)
            log_eta_flat -= torch.logsumexp(log_eta_flat, dim=1, keepdim=True)
            eta = torch.clamp(log_eta_flat.exp().view(seq_len, K, Dmax), min=EPS)
            eta = eta / eta.view(seq_len, -1).sum(dim=1, keepdim=True).view(seq_len, 1, 1).clamp_min(EPS)
            eta_vec.append(eta)

            # --- xi ---
            if seq_len > 1:
                # Compute cumsum for emission sums
                cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=self.device),
                                            torch.cumsum(seq_probs, dim=0)))
                max_durations = torch.arange(1, Dmax + 1, device=self.device)  # (Dmax,)

                # Precompute all prev_alpha_first for each t and each duration
                # shape: (T-1, Dmax, K)
                prev_alpha_first = torch.full((seq_len - 1, Dmax, K), -torch.inf, device=self.device, dtype=DTYPE)

                for d in max_durations:
                    t_indices = torch.arange(seq_len - 1, device=self.device)
                    start_idx = t_indices - d + 1
                    valid_mask = start_idx >= 0
                    start_idx_clamped = start_idx.clamp(min=0)
                    prev_alpha_first[valid_mask, d - 1] = alpha[start_idx_clamped[valid_mask], :, 0]
                    # start_idx < 0: use init_logits
                    prev_alpha_first[~valid_mask, d - 1] = init_logits

                # logsumexp over durations for previous states
                alpha_sum = torch.logsumexp(prev_alpha_first, dim=1)  # (T-1, K)

                # logsumexp over beta for next states and durations
                beta_next = torch.logsumexp(beta[1:, :, :Dmax], dim=2)  # (T-1, K)

                # compute xi: log_xi[t,i,j] = alpha_sum[t,i] + transition[i,j] + beta_next[t,j]
                log_xi = alpha_sum.unsqueeze(2) + transition_logits.unsqueeze(0) + beta_next.unsqueeze(1)  # (T-1,K,K)
                log_xi -= torch.logsumexp(log_xi.view(seq_len - 1, -1), dim=1, keepdim=True).unsqueeze(2)
                xi_seq = torch.clamp(log_xi.exp(), min=EPS)
                xi_seq /= xi_seq.sum(dim=(1, 2), keepdim=True).clamp_min(EPS)
                xi_vec.append(xi_seq)
            else:
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=self.device))

        return gamma_vec, xi_vec, eta_vec

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """Vectorized MAP decoding of HSMM sequences from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if gamma_list is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        seq_lengths = [g.shape[0] if g is not None else 0 for g in gamma_list]
        B = len(seq_lengths)
        max_T = max(seq_lengths) if seq_lengths else 0
        K = self.n_states
        neg_inf = -float("inf")

        if max_T == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(B)]

        # Prepare padded tensor
        gamma_padded = torch.full((B, max_T, K), neg_inf, device=self.device, dtype=DTYPE)
        for i, g in enumerate(gamma_list):
            if g is not None and g.numel() > 0:
                gamma_padded[i, :g.shape[0]] = torch.nan_to_num(g, nan=neg_inf, posinf=neg_inf, neginf=neg_inf)

        # MAP decoding: argmax along states
        map_padded = gamma_padded.argmax(dim=-1)

        # Split back into list per original sequence lengths
        map_sequences = [
            map_padded[i, :L].to(dtype=torch.long, device=self.device) if L > 0 else torch.empty(0, dtype=torch.long, device=self.device)
            for i, L in enumerate(seq_lengths)
        ]

        return map_sequences

    def _viterbi(self, X: utils.Observations, duration_weight: float = 0.0) -> list[torch.Tensor]:
        K, Dmax = self.n_states, self.max_duration
        B = len(X.log_probs)
        lengths = X.lengths
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            return [torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(B)]

        init_logits = self.init_logits.to(self.device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(self.device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(self.device, dtype=DTYPE)

        neg_inf = -torch.inf
        dur_indices = torch.arange(1, Dmax + 1, device=self.device)

        if duration_weight > 0.0:
            dur_mean = (torch.softmax(duration_logits, dim=1) * dur_indices).sum(dim=1)
            dur_penalty = -((dur_indices.unsqueeze(0) - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
            dur_lp = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty
        else:
            dur_lp = duration_logits

        # Precompute cumulative emission sums
        cumsum_emit = [
            torch.cat([torch.zeros((1, K), dtype=DTYPE, device=self.device),
                       torch.cumsum(seq.to(self.device, dtype=DTYPE), dim=0)])
            for seq in X.log_probs
        ]

        V = torch.full((B, max_len, K), neg_inf, dtype=DTYPE, device=self.device)
        best_prev = torch.full((B, max_len, K), -1, dtype=torch.int64, device=self.device)
        best_dur = torch.zeros((B, max_len, K), dtype=torch.int64, device=self.device)

        for t in range(max_len):
            active_mask = torch.tensor([t < L for L in lengths], device=self.device)
            if not active_mask.any():
                continue
            active_idx = active_mask.nonzero(as_tuple=True)[0]

            for b in active_idx:
                T = lengths[b]
                max_d = min(Dmax, t + 1)
                durations = dur_indices[:max_d]
                starts = t - durations + 1  # (max_d,)

                # --- vectorized emission sums ---
                end_idx = (t + 1) * torch.ones(max_d, dtype=torch.long, device=self.device)
                start_idx = starts
                # Gather start and end cumulative sums
                start_cumsum = cumsum_emit[b][start_idx]       # (max_d, K)
                end_cumsum = cumsum_emit[b][end_idx]           # (max_d, K)
                emit_sums = (end_cumsum - start_cumsum).T      # (K, max_d)

                dur_scores = dur_lp[:, :max_d]                 # (K, max_d)

                if t == 0:
                    scores = init_logits.unsqueeze(1) + dur_scores + emit_sums
                    prev_idx = torch.full(scores.shape, -1, dtype=torch.int64, device=self.device)
                else:
                    # previous V for all durations at once
                    prev_idx_raw = (starts - 1).clamp(min=0)  # (max_d,)
                    prev_V = V[b, prev_idx_raw, :]           # (max_d, K)
                    # For durations starting at 0, use init_logits instead of V
                    mask_start0 = starts == 0
                    if mask_start0.any():
                        prev_V[mask_start0] = init_logits

                    # scores_plus_trans: (max_d, K_prev, K_curr)
                    scores_plus_trans = prev_V.unsqueeze(2) + transition_logits.unsqueeze(0)
                    scores_max, argmax_prev = torch.max(scores_plus_trans, dim=1)  # (max_d, K)
                    scores = (scores_max.T + dur_scores + emit_sums)                # (K, max_d)
                    prev_idx = argmax_prev.T                                        # (K, max_d)
                    # mask durations starting at 0
                    mask2d = mask_start0.unsqueeze(0).expand(K, -1)
                    prev_idx = prev_idx.masked_fill(mask2d, -1)

                # choose best duration per current state
                best_score, best_d_idx = torch.max(scores, dim=1)
                V[b, t] = best_score
                best_dur[b, t] = durations[best_d_idx]
                best_prev[b, t] = prev_idx[torch.arange(K, device=self.device), best_d_idx]

        # --- backtrace ---
        paths: list[torch.Tensor] = []
        for b, T in enumerate(lengths):
            if T == 0:
                paths.append(torch.empty((0,), dtype=torch.int64, device=self.device))
                continue

            t = T - 1
            cur_state = int(torch.argmax(V[b, t]).item())
            segments = []

            while t >= 0:
                d = int(best_dur[b, t, cur_state].item())
                if d <= 0:
                    break
                start = max(0, t - d + 1)
                segments.append((start, t, cur_state))
                prev_state = int(best_prev[b, t, cur_state].item())
                t = start - 1
                cur_state = prev_state if prev_state >= 0 else cur_state

            segments.reverse()
            if not segments:
                paths.append(torch.full((T,), cur_state, dtype=torch.int64, device=self.device))
                continue

            seq_path = torch.cat([
                torch.full((end - start + 1,), st, dtype=torch.int64, device=self.device)
                for start, end, st in segments
            ])
            if seq_path.shape[0] < T:
                seq_path = torch.cat([seq_path, torch.full((T - seq_path.shape[0],), seq_path[-1], dtype=torch.int64, device=self.device)])
            paths.append(seq_path)

        return paths

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 15,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_D_from_X: bool = False,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        """
        Vectorized, GPU-friendly EM fitting for HSMM with context-aware emissions.
        """

        # Optional initialization
        if sample_D_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # Encode observations if needed
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths, theta)

        # Align theta strictly
        if theta is not None:
            if theta.ndim != 2:
                raise ValueError("Theta must be 2D [T_total, D].")
            if theta.shape[0] != sum(X_valid.lengths):
                # Expand or repeat to match total time steps
                if theta.shape[0] == len(X_valid.sequence):
                    theta = torch.cat([theta[i].expand(l, -1) for i, l in enumerate(X_valid.lengths)], dim=0)
                elif sum(X_valid.lengths) % theta.shape[0] == 0:
                    theta = theta.repeat(sum(X_valid.lengths) // theta.shape[0], 1)
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot be aligned with total length {sum(X_valid.lengths)}")
            valid_theta = theta
        else:
            valid_theta = None

        # Initialize convergence tracker
        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        best_score = -float("inf")
        best_state = None

        # Flatten all sequences for vectorized operations
        seq_tensor = torch.cat([s for s in X_valid.sequence if s.numel() > 0], dim=0).to(dtype=DTYPE, device=self.device)
        seq_offsets = torch.cumsum(torch.tensor([0] + X_valid.lengths, device=self.device), dim=0)

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            if run_idx > 0:
                # Resample parameters
                sampled = self.sample_model_params(X, valid_theta)
                self._init_logits.copy_(sampled['init_logits'])
                self._transition_logits.copy_(sampled['transition_logits'])
                self._duration_logits.copy_(sampled['duration_logits'])
                self._params['emission_pdf'] = sampled['emission_pdf']

            # Compute initial log-likelihood
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # --- E-step: vectorized log_probs ---
                pdf = self._params['emission_pdf']
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or incompatible with vectorized log_prob.")
                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))

                # Split into sequences
                X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(len(X_valid.lengths))]

                # --- M-step: estimate model parameters ---
                new_params = self._estimate_model_params(X_valid, valid_theta)
                for key, attr in [('init_logits', '_init_logits'),
                                  ('transition_logits', '_transition_logits'),
                                  ('duration_logits', '_duration_logits'),
                                  ('emission_pdf', None)]:
                    if key in new_params:
                        if key == 'emission_pdf':
                            self._params[key] = new_params[key]
                        else:
                            getattr(self, attr).copy_(new_params[key])

                # Evaluate log-likelihood
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv and verbose:
                    print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # Track best run
            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'init_logits': self._init_logits.clone(),
                    'transition_logits': self._transition_logits.clone(),
                    'duration_logits': self._duration_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        # Restore best parameters
        if best_state is not None:
            self._init_logits.copy_(best_state['init_logits'])
            self._transition_logits.copy_(best_state['transition_logits'])
            self._duration_logits.copy_(best_state['duration_logits'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        if plot_conv and hasattr(self, 'conv'):
            self.conv.plot_convergence()

        return self

    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> list[torch.Tensor]:
        """
        Decode sequences for HSMM using MAP or Viterbi with safe duration handling.
        """
        X_valid = self.to_observations(X, lengths)
        algorithm = algorithm.lower()
        B = len(X_valid.lengths)
        total_T = sum(X_valid.lengths)
        K = getattr(self, "n_states", None)

        # --- Context ---
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0)
            elif context.ndim > 2:
                context = context.mean(dim=0, keepdim=True)
            context = context.to(device=self.device, dtype=DTYPE)

        # --- Flatten observations ---
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(device=self.device, dtype=DTYPE)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths, device=self.device), dim=0).tolist())

        # --- Emission log-probs ---
        log_probs_chunks = []
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is not None and hasattr(pdf, "log_prob"):
            for start in range(0, total_T, batch_size):
                end = min(start + batch_size, total_T)
                chunk = seq_tensor[start:end].unsqueeze(1)
                lp = pdf.log_prob(chunk)
                if lp.ndim > 2:
                    lp = lp.sum(dim=list(range(2, lp.ndim)))
                log_probs_chunks.append(lp)
            all_log_probs = torch.cat(log_probs_chunks, dim=0)
        else:
            if not hasattr(self, "emission_module"):
                raise RuntimeError("No emission module found for NeuralHSMM.")
            output = self.emission_module(context)
            if isinstance(output, tuple):
                mu, var = output
                var = var.clamp(min=1e-6)
                dist = torch.distributions.Normal(mu, var.sqrt())
                for start in range(0, total_T, batch_size):
                    end = min(start + batch_size, total_T)
                    chunk = seq_tensor[start:end].unsqueeze(1)
                    lp = dist.log_prob(chunk).sum(dim=-1)
                    log_probs_chunks.append(lp)
                all_log_probs = torch.cat(log_probs_chunks, dim=0)
            else:
                logits = output
                all_log_probs = F.log_softmax(logits, dim=-1)
                if seq_tensor.dtype != torch.long:
                    raise TypeError("seq_tensor must contain integer indices for categorical emission.")
                all_log_probs = all_log_probs[torch.arange(seq_tensor.shape[0], device=self.device), seq_tensor.long()]

        X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(B)]
        K = all_log_probs.shape[-1]

        # --- Transition logits ---
        if hasattr(self, "transition_module"):
            if hasattr(self.transition_module, "log_probs"):
                transition_logits = self.transition_module.log_probs(context=context)
            else:
                transition_logits = torch.log_softmax(self.transition_module(context), dim=-1)
        else:
            transition_logits = getattr(self._params, "log_transition", None)
            if transition_logits is None:
                raise RuntimeError("Transition parameters missing for HSMM.")
            transition_logits = transition_logits.to(device=self.device)

        # --- Duration logits ---
        log_duration_global = None
        log_duration_batch = None
        if hasattr(self, "duration_module"):
            if hasattr(self.duration_module, "log_probs"):
                log_duration_batch = self.duration_module.log_probs(context=context)
            elif hasattr(self.duration_module, "logits"):
                log_duration_global = F.log_softmax(self.duration_module.logits, dim=-1)
        else:
            log_duration_global = getattr(self._params, "log_duration", None)
            if log_duration_global is not None:
                log_duration_global = log_duration_global.to(device=self.device)

        max_T = max(X_valid.lengths)
        log_B = torch.full((B, max_T, K), -torch.inf, device=self.device)
        for i, seq_lp in enumerate(X_valid.log_probs):
            log_B[i, :seq_lp.shape[0]] = seq_lp
        mask = torch.arange(max_T, device=self.device).unsqueeze(0) < torch.tensor(X_valid.lengths, device=self.device).unsqueeze(1)

        # --- Helper for safe duration indexing ---
        def get_dur_term(t: int) -> torch.Tensor:
            if log_duration_global is not None:
                dur_idx = min(t, log_duration_global.shape[1] - 1)
                return log_duration_global[:, dur_idx].unsqueeze(0).expand(B, -1)
            if log_duration_batch is not None:
                dur_idx = min(t, log_duration_batch.shape[2] - 1)
                return log_duration_batch[:, :, dur_idx]
            return torch.zeros(B, K, device=self.device, dtype=DTYPE)

        # --- Decoding ---
        if algorithm == "map":
            alpha = torch.full((B, max_T, K), -torch.inf, device=self.device)
            alpha[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                prev = alpha[:, t - 1].unsqueeze(2) + transition_logits
                prev += get_dur_term(t).unsqueeze(1)
                alpha[:, t] = torch.logsumexp(prev, dim=1) + log_B[:, t]
                alpha[:, t] = torch.where(mask[:, t].unsqueeze(-1), alpha[:, t], alpha[:, t - 1])
            decoded = torch.argmax(alpha, dim=-1)
        elif algorithm == "viterbi":
            delta = torch.full((B, max_T, K), -torch.inf, device=self.device)
            psi = torch.zeros((B, max_T, K), dtype=torch.long, device=self.device)
            delta[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                scores = delta[:, t - 1].unsqueeze(2) + transition_logits
                scores += get_dur_term(t).unsqueeze(1)
                psi[:, t] = torch.argmax(scores, dim=1)
                delta[:, t] = torch.max(scores, dim=1).values + log_B[:, t]
                delta[:, t] = torch.where(mask[:, t].unsqueeze(-1), delta[:, t], delta[:, t - 1])

            decoded = torch.zeros((B, max_T), dtype=torch.long, device=self.device)
            decoded[:, -1] = torch.argmax(delta[:, -1], dim=-1)
            for t in range(max_T - 2, -1, -1):
                decoded[:, t] = torch.gather(psi[:, t + 1], 1, decoded[:, t + 1].unsqueeze(1)).squeeze(1)
                decoded[:, t] = torch.where(mask[:, t], decoded[:, t], decoded[:, t + 1])
        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

        return [decoded[i, :L].detach().cpu() for i, L in enumerate(X_valid.lengths)]

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True,
        theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Vectorized, duration-aware log-likelihood for HSMM with optional context.

        Args:
            X: Input sequences (B, T, F) or flattened.
            lengths: Optional sequence lengths.
            by_sample: If True, returns per-sequence log-likelihood; otherwise sum.
            theta: Optional context tensor for conditional emissions.

        Returns:
            Tensor of log-likelihoods.
        """
        obs = self.to_observations(X, lengths, theta)
        B = len(obs.lengths)
        max_T = max(obs.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration

        # -------- Emission log-probabilities --------
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")

        all_seq = torch.cat(obs.sequence, dim=0).to(dtype=DTYPE, device=self.device)
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)

        # Map to batch/sequence
        seq_offsets = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(torch.tensor(obs.lengths, device=self.device), dim=0)])
        log_B = torch.full((B, max_T, K), -torch.inf, device=self.device, dtype=DTYPE)
        for i in range(B):
            start, end = seq_offsets[i].item(), seq_offsets[i + 1].item()
            L = obs.lengths[i]
            log_B[i, :L] = all_log_probs[start:end]

        # -------- Transition and duration --------
        if hasattr(self, "transition_module"):
            if hasattr(self.transition_module, "log_probs"):
                log_transition = self.transition_module.log_probs(context=theta)
            elif hasattr(self.transition_module, "logits"):
                log_transition = F.log_softmax(self.transition_module.logits, dim=-1)
            else:
                raise RuntimeError("Transition module missing logits or log_probs method.")
        else:
            log_transition = F.log_softmax(self._transition_logits, dim=-1) if hasattr(self, "_transition_logits") else torch.full((K, K), 1.0 / K, device=self.device).log()

        log_duration = F.log_softmax(self._duration_logits, dim=-1) if hasattr(self, "_duration_logits") else torch.zeros(K, 1, device=self.device, dtype=DTYPE)

        # -------- Forward algorithm (log-space) --------
        V = torch.full((B, max_T, K), -torch.inf, device=self.device, dtype=DTYPE)
        V[:, 0] = log_B[:, 0] + self.init_logits.to(device=self.device)

        dur_idx = torch.arange(Dmax, device=self.device)
        for t in range(1, max_T):
            max_d = min(t + 1, Dmax)
            durations = dur_idx[:max_d]

            prev_V_list = []
            emit_list = []
            for d in durations:
                prev_V_list.append(V[:, t - d])
                emit_list.append(log_B[:, t - d + 1:t + 1].sum(dim=1))
            prev_V = torch.stack(prev_V_list, dim=1)  # (B, max_d, K)
            emit_sums = torch.stack(emit_list, dim=1)  # (B, max_d, K)
            dur_scores = log_duration[:, :max_d].T.unsqueeze(0)  # (1, max_d, K)

            trans_scores = prev_V.unsqueeze(3) + log_transition  # (B, max_d, K, K)
            trans_max = torch.logsumexp(trans_scores, dim=2)  # (B, max_d, K)

            V[:, t] = torch.logsumexp(trans_max + dur_scores + emit_sums, dim=1)

        # Mask invalid positions
        mask = torch.arange(max_T, device=self.device).unsqueeze(0) < torch.tensor(obs.lengths, device=self.device).unsqueeze(1)
        V = torch.where(mask.unsqueeze(-1), V, torch.tensor(-torch.inf, device=self.device))

        # Sequence log-likelihood
        end_idx = torch.tensor(obs.lengths, device=self.device) - 1
        seq_ll = torch.logsumexp(V[torch.arange(B), end_idx], dim=-1)

        return seq_ll.detach().cpu() if by_sample else seq_ll.sum().detach().cpu()

    def _compute_log_likelihood(self, X: utils.Observations) -> torch.Tensor:
        """Fully vectorized log-likelihood computation for HSMM sequences."""
        alpha_list = self._forward(X)  # list of (T_i, K, Dmax)
        if not alpha_list:
            raise RuntimeError("Forward pass returned empty results. Model may be uninitialized.")

        B = len(alpha_list)
        K, Dmax = self.n_states, self.max_duration
        max_T = max([a.shape[0] for a in alpha_list])
        neg_inf = -torch.inf

        # Prepare a padded tensor for all sequences
        alpha_padded = torch.full((B, max_T, K, Dmax), neg_inf, device=self.device, dtype=DTYPE)
        mask = torch.zeros((B, max_T), device=self.device, dtype=torch.bool)

        for i, alpha in enumerate(alpha_list):
            T_i = alpha.shape[0]
            if T_i > 0:
                alpha_padded[i, :T_i] = alpha
                mask[i, :T_i] = True

        # Clamp invalid entries
        alpha_padded = alpha_padded.nan_to_num(nan=neg_inf, posinf=neg_inf, neginf=neg_inf)

        # Compute logsumexp over states and durations at last valid timestep per sequence
        seq_lengths = torch.tensor([a.shape[0] for a in alpha_list], device=self.device)
        last_idx = (seq_lengths - 1).clamp(min=0)  # ensure non-negative indices
        ll_list = torch.logsumexp(
            alpha_padded[torch.arange(B, device=self.device), last_idx], dim=(-1, -2)
        )

        # Sequences with length 0 -> -inf
        ll_list = torch.where(seq_lengths > 0, ll_list, torch.full_like(ll_list, neg_inf))

        return ll_list.detach()

    def information(
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

        # --- Compute per-sequence log-likelihood safely ---
        try:
            log_likelihood = self.score(X, lengths=lengths, by_sample=by_sample).to(device=self.device, dtype=DTYPE)
        except Exception as e:
            raise RuntimeError(f"Failed to compute log-likelihood: {e}")

        # --- Count total observations (time steps, not features) ---
        if lengths is not None:
            n_obs = max(int(sum(lengths)), 1)
        elif X.ndim == 3:  # (B, T, F)
            n_obs = max(int(X.shape[0] * X.shape[1]), 1)
        else:  # (T, F) or flat
            n_obs = max(int(X.shape[0]), 1)

        # --- Degrees of freedom check ---
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError(
                "Model degrees of freedom ('dof') not defined. "
                "Please set 'self.dof' during initialization."
            )

        # --- Compute IC ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=log_likelihood,
            dof=dof,
            criterion=criterion
        )

        # --- Ensure tensor, handle NaN/Inf ---
        if not isinstance(ic_value, torch.Tensor):
            ic_value = torch.tensor(ic_value, dtype=DTYPE, device=self.device)
        ic_value = ic_value.nan_to_num(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))

        # --- Ensure per-sample shape if requested ---
        if by_sample and ic_value.ndim == 0:
            ic_value = ic_value.unsqueeze(0)

        return ic_value.detach().cpu()
