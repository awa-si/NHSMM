# models/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict

from torch.distributions import Distribution, Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch

from nhsmm.utilities import utils, constraints, SeedGenerator, ConvergenceHandler

DTYPE = torch.float64


class BaseHSMM(nn.Module, ABC):
    """
    Base HSMM core (probabilistic). 
    - Implements EM (Baum-Welch for semi-Markov), forward/backward, Viterbi, scoring, IC.
    - Keeps pi/A/D logits registered as buffers for persistence.
    - Delegates emission parameterization to subclasses via `sample_emission_pdf` and `_estimate_emission_pdf`.
    - Provides small, well-defined hooks for neural/contextual integration:
        - attach_encoder / encode_observations
        - _contextual_emission_pdf, _contextual_transition_matrix, _contextual_duration_pdf
    """

    def __init__(self,
                 n_states: int,
                 max_duration: int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__()
        self.n_states = int(n_states)
        self.max_duration = int(max_duration)
        self.alpha = float(alpha)
        self._seed_gen = SeedGenerator(seed)

        # container for emitted distribution and other non-buffer parameters
        self._params: Dict[str, Any] = {}

        # optional external encoder (neural) for contextualization
        self.encoder: Optional[nn.Module] = None

        # initialize & register pi/A/D logits as buffers (log-space)
        self._init_buffers()

        # initialize emission pdf by calling subclass hook (may be based on no data)
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

        self._context: Optional[torch.Tensor] = None

    # ----------------------
    # Persistence buffers
    # ----------------------
    def _init_buffers(self):
        """
        Initialize log-probability buffers for HSMM priors, including optional hierarchical super-states.

        Behavior:
            - Base-level π, A, D are initialized using Dirichlet-style sampling.
            - Log-space tensors are clamped to prevent numerical underflow.
            - If `n_super_states` exists (>1), hierarchical π and A are also initialized.
            - Stores a diagnostic snapshot `_init_prior_snapshot` for debugging and monitoring.

        Notes:
            - Buffers are device- and dtype-aligned with module parameters.
            - Subclasses may override or adapt these buffers via contextual hooks.
        """
        device = next(self.parameters(), torch.tensor(0., dtype=DTYPE)).device

        # -------------------------------
        # Base-level prior initialization
        # -------------------------------
        sampled_pi = constraints.sample_probs(self.alpha, (self.n_states,))
        sampled_A = constraints.sample_A(self.alpha, self.n_states, constraints.Transitions.SEMI)
        sampled_D = constraints.sample_probs(self.alpha, (self.n_states, self.max_duration))

        pi_logits = torch.log(sampled_pi.clamp_min(1e-8)).to(device=device, dtype=DTYPE)
        A_logits = torch.log(sampled_A.clamp_min(1e-8)).to(device=device, dtype=DTYPE)
        D_logits = torch.log(sampled_D.clamp_min(1e-8)).to(device=device, dtype=DTYPE)

        # -------------------------------
        # Hierarchical / super-state extensions
        # -------------------------------
        n_super_states = getattr(self, "n_super_states", None)
        if n_super_states is not None and n_super_states > 1:
            super_pi = constraints.sample_probs(self.alpha, (n_super_states,))
            super_A = constraints.sample_A(self.alpha, n_super_states, constraints.Transitions.SEMI)

            super_pi_logits = torch.log(super_pi.clamp_min(1e-8)).to(device=device, dtype=DTYPE)
            super_A_logits = torch.log(super_A.clamp_min(1e-8)).to(device=device, dtype=DTYPE)

            self.register_buffer("_super_pi_logits", super_pi_logits)
            self.register_buffer("_super_A_logits", super_A_logits)

        # -------------------------------
        # Register base buffers
        # -------------------------------
        self.register_buffer("_pi_logits", pi_logits)
        self.register_buffer("_A_logits", A_logits)
        self.register_buffer("_D_logits", D_logits)

        # -------------------------------
        # Diagnostic snapshot for debugging
        # -------------------------------
        summary = [pi_logits.mean(), A_logits.mean(), D_logits.mean()]
        if n_super_states is not None and n_super_states > 1:
            summary += [super_pi_logits.mean(), super_A_logits.mean()]
        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

    # ----------------------
    # Properties (log-space tensors)
    # ----------------------
    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def pi(self) -> torch.Tensor:
        return self._pi_logits

    @pi.setter
    def pi(self, logits: torch.Tensor):
        logits = logits.to(device=self._pi_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states,):
            raise ValueError(f"pi logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")

        norm_val = logits.logsumexp(0)
        if not torch.allclose(norm_val, torch.tensor(0.0, dtype=DTYPE, device=logits.device), atol=1e-8):
            raise ValueError(f"pi logits must normalize (logsumexp==0); got {norm_val.item():.3e}")

        self._pi_logits.copy_(logits)

    @property
    def A(self) -> torch.Tensor:
        return self._A_logits

    @A.setter
    def A(self, logits: torch.Tensor):
        logits = logits.to(device=self._A_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"A logits must have shape ({self.n_states},{self.n_states})")

        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-8):
            raise ValueError(f"Rows of A logits must normalize (logsumexp==0); got {row_norm}")

        if not constraints.is_valid_A(logits, constraints.Transitions.SEMI):
            raise ValueError("A logits do not satisfy SEMI transition constraints")

        self._A_logits.copy_(logits)

    @property
    def D(self) -> torch.Tensor:
        return self._D_logits

    @D.setter
    def D(self, logits: torch.Tensor):
        logits = logits.to(device=self._D_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"D logits must have shape ({self.n_states},{self.max_duration})")

        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-8):
            raise ValueError(f"Rows of D logits must normalize (logsumexp==0); got {row_norm}")

        self._D_logits.copy_(logits)


    @property
    def pdf(self) -> Any:
        """Emission distribution (torch.distributions.Distribution). Managed by subclass via hooks."""
        return self._params.get('emission_pdf')

    # ----------------------
    # Subclass API (abstract)
    # ----------------------
    @property
    @abstractmethod
    def dof(self) -> int:
        """Degrees of freedom (required for IC computations)."""
        raise NotImplementedError

    @abstractmethod
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        """
        Create and return an initial emission Distribution.
        Called at construction and optionally with data X when sample_B_from_X=True.
        """
        raise NotImplementedError

    @abstractmethod
    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,                       # shape: (n_samples, *event_shape)
        posterior: torch.Tensor,               # shape: (n_samples, n_states)
        theta: Optional[utils.ContextualVariables] = None
    ) -> Distribution:
        """
        Update the emission distribution (PDF) based on posterior state probabilities.

        Parameters
        ----------
        X : torch.Tensor
            Observations stacked over all sequences. Shape: (n_samples, *event_shape).
        posterior : torch.Tensor
            Posterior state probabilities γ_t(s). Shape: (n_samples, n_states).
        theta : Optional[utils.ContextualVariables], default=None
            Optional contextual variables from neural encoder or external source.

        Returns
        -------
        Distribution
            Updated emission distribution. Should reflect the per-state
            probabilities or parameters for all hidden states.

        Notes
        -----
        - Subclasses must implement this method according to their emission type
          (e.g., categorical, Gaussian, or neural-adapted PDF).
        - Use `posterior` as weights to compute expected sufficient statistics.
        - Must be fully vectorized for stability and GPU efficiency.
        """
        raise NotImplementedError

    # ----------------------
    # Neural/context hooks (override as needed)
    # ----------------------
    def attach_encoder(self, encoder: nn.Module, batch_first: bool = True):
        """
        Attach a neural encoder module (e.g., CNN+LSTM) and auto-integrate its output
        as `theta` for contextual HSMM hooks.

        Args:
            encoder: nn.Module producing context embeddings.
            batch_first: Whether encoder expects input of shape (B, T, F). If False, expects (T, B, F).

        Enhancements:
            - Device-aligned.
            - Batch-safe (adds batch dim if missing).
            - Auto-detects sequence-level vs. time-level output.
            - Integrates output into `self.theta` via `to_contextuals`.
        """
        device = next(self.parameters()).device
        encoder.to(device)

        class EncoderWrapper(nn.Module):
            def __init__(self, enc, batch_first, parent):
                super().__init__()
                self.enc = enc
                self.batch_first = batch_first
                self.parent = parent

            def forward(self, x):
                # Ensure batch dimension
                if x.dim() == 2:  # (T, F)
                    x = x.unsqueeze(0 if self.batch_first else 1)
                out = self.enc(x)
                if not torch.is_tensor(out):
                    raise TypeError(f"Encoder must return a tensor, got {type(out)}")

                # Auto-detect output shape
                if out.dim() == 1:  # (hidden_dim,) sequence-level
                    theta = out.unsqueeze(1)  # make (hidden_dim, 1)
                elif out.dim() == 2:
                    if out.shape[1] == 1 or out.shape[1] == x.shape[1]:
                        theta = out  # already time-distributed
                    else:
                        theta = out.T  # transpose to (hidden_dim, T)
                else:
                    raise ValueError(f"Unexpected encoder output shape {out.shape}")

                # Integrate into parent context
                self.parent.theta = self.parent.to_contextuals(theta)
                return out

        self.encoder = EncoderWrapper(encoder, batch_first, self)

    def encode_observations(
        self,
        X: torch.Tensor,
        pool: str = "last",
        store: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Encode observations into a context vector θ using the attached encoder.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (T, F) for a single sequence or (B, T, F) for a batch.
        pool : str, default="last"
            Temporal pooling strategy for sequence encoders:
                - "last": take the last timestep (typical for RNNs/LSTMs)
                - "mean": average over all timesteps
                - "max": max-pooling over timesteps
        store : bool, default=True
            Whether to store the resulting context vector in `self._context`.

        Returns
        -------
        torch.Tensor or None
            Encoded context tensor of shape (B, H) or (1, H) for single sequence.
            Returns None if no encoder is attached.
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        # Ensure 3D input: (B, T, F)
        if X.ndim == 2:  # single sequence (T, F)
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Unsupported input shape {X.shape}, expected (T,F) or (B,T,F)")

        # Forward through encoder
        out = self.encoder(X)
        if isinstance(out, tuple):  # handle RNN/LSTM returning (output, hidden)
            out = out[0]
        out = out.detach().to(dtype=DTYPE, device=device)

        # Temporal pooling
        if out.ndim == 3:  # (B, T, H)
            if pool == "last":
                vec = out[:, -1, :]
            elif pool == "mean":
                vec = out.mean(dim=1)
            elif pool == "max":
                vec, _ = out.max(dim=1)
            else:
                raise ValueError(f"Unsupported pooling mode '{pool}'")
        elif out.ndim == 2:  # (B, H)
            vec = out
        else:
            raise ValueError(f"Unsupported encoder output shape {out.shape}")

        # Normalize for stability
        vec = nn.functional.layer_norm(vec, vec.shape[-1:])

        # Store for contextual adaptation
        if store:
            self._context = vec.detach()

        return vec

    # ----------------------
    # Contextual hooks
    # ----------------------
    def _combine_context(self, theta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Merge encoder output and stored context into a single context vector.

        - Preserves batch/time dimensions where possible.
        - Applies mean-pooling over time for 2D/3D tensors.
        - Concatenates stored context (`_context`) along feature dimension.
        - Returns shape (H_total, 1) for downstream contextual adaptation.
        """
        theta_combined: Optional[torch.Tensor] = None
        device, dtype = self.device, DTYPE

        # -------------------------
        # Process encoder output
        # -------------------------
        if theta is not None:
            if theta.dim() == 1:  # (H,)
                theta_combined = theta.unsqueeze(0)
            elif theta.dim() == 2:  # (B, H) or (H, T)
                if theta.shape[0] == 1 or theta.shape[0] == theta.shape[1]:  # heuristic: time along dim=1?
                    theta_combined = theta.mean(dim=1, keepdim=True)
                else:
                    theta_combined = theta
            elif theta.dim() == 3:  # (B, T, H)
                theta_combined = theta[:, -1, :]  # take last timestep
            else:
                raise ValueError(f"Unsupported encoder output shape {theta.shape}")

            theta_combined = theta_combined.to(device=device, dtype=dtype)

        # -------------------------
        # Process stored context
        # -------------------------
        if getattr(self, "_context", None) is not None:
            ctx_vec = self._context.to(device=device, dtype=dtype)
            if ctx_vec.dim() > 2:
                ctx_vec = ctx_vec.mean(dim=1)  # reduce time dimension if present
            if theta_combined is None:
                theta_combined = ctx_vec
            else:
                theta_combined = torch.cat([theta_combined, ctx_vec], dim=-1)

        if theta_combined is not None:
            return theta_combined.unsqueeze(-1)  # shape (H_total, 1)
        return None

    def _contextual_emission_pdf(self, X: utils.Observations, theta: Optional[utils.ContextualVariables]) -> Distribution:
        """
        Return a context-adapted emission PDF.

        - Uses encoder output and optional stored context combined via `_combine_context`.
        - Neural subclasses can override this to map `theta_combined` -> new emission distribution.
        - Default: return stored emission PDF if no context is available.

        Args:
            X: Observations object containing sequences and log_probs
            theta: Optional contextual variables from encoder or external source

        Returns:
            Distribution: A torch.distributions.Distribution representing emission probabilities
        """
        # Combine encoder output and stored context
        theta_combined = self._combine_context(theta)

        # If no context, return base emission PDF
        if theta_combined is None:
            return self._params.get('emission_pdf')

        # Default behavior: for neural subclasses, map theta_combined -> new emission PDF
        # Placeholder: return stored emission PDF unchanged
        # Subclasses should override this method to implement context-aware emissions
        return self._params.get('emission_pdf')

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return a context-adapted duration distribution (log-space).

        - If `theta` is None, returns base `_D_logits`.
        - If `theta` is provided, maps it through a small neural adapter
          to produce context-conditioned durations per state.
        - Ensures log-probabilities sum to 1 per state (logsumexp normalization).
        
        Args:
            theta: Optional context tensor of shape (H_total, 1) or (batch, H_total)

        Returns:
            log_D: Tensor of shape (n_states, max_duration)
        """
        base_D = self._D_logits  # (n_states, max_duration)
        device = base_D.device
        dtype = DTYPE

        if theta is None:
            return base_D

        # Ensure 2D: (batch, H_total)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        elif theta.dim() == 3:
            theta = theta[:, -1, :]  # take last timestep
        theta = theta.to(device=device, dtype=dtype)

        # ----------------------------
        # Neural adaptation
        # ----------------------------
        # Simple adapter: small MLP mapping context → duration logits per state
        if not hasattr(self, "_duration_adapter"):
            hidden_dim = theta.shape[-1]
            self._duration_adapter = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_states * self.max_duration)
            ).to(device=device, dtype=dtype)

        # Predict delta logits (adjustment to base duration)
        delta_logits = self._duration_adapter(theta)  # (batch, n_states * max_duration)
        delta_logits = delta_logits.view(-1, self.n_states, self.max_duration)  # (batch, n_states, max_duration)

        # If multiple sequences, average predicted adjustments
        delta_mean = delta_logits.mean(dim=0)  # (n_states, max_duration)

        # Combine with base logits
        log_D = base_D + delta_mean

        # Normalize log-probabilities per state
        log_D = log_D - torch.logsumexp(log_D, dim=1, keepdim=True)
        return log_D

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return a context-adapted transition matrix (log-space).

        - If `theta` is None, returns the base transition `_A_logits`.
        - If `theta` is provided, maps it through a small neural adapter to produce
          context-conditioned transition probabilities per state.
        - Ensures each row is a valid probability distribution in log-space.

        Args:
            theta: Optional context tensor of shape (H_total, 1) or (batch, H_total)

        Returns:
            log_A: Tensor of shape (n_states, n_states) in log-space
        """
        base_A = self._A_logits  # (n_states, n_states)
        device = base_A.device
        dtype = DTYPE

        if theta is None:
            return base_A

        # Ensure 2D: (batch, H_total)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        elif theta.dim() == 3:
            theta = theta[:, -1, :]  # take last timestep
        theta = theta.to(device=device, dtype=dtype)

        # ----------------------------
        # Neural adaptation
        # ----------------------------
        # Simple adapter: MLP mapping context -> delta logits for transitions
        if not hasattr(self, "_transition_adapter"):
            hidden_dim = theta.shape[-1]
            self._transition_adapter = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_states * self.n_states)
            ).to(device=device, dtype=dtype)

        # Predict delta logits (adjustment to base transition)
        delta_logits = self._transition_adapter(theta)  # (batch, n_states * n_states)
        delta_logits = delta_logits.view(-1, self.n_states, self.n_states)  # (batch, n_states, n_states)

        # Average over batch to get a single adjusted transition matrix
        delta_mean = delta_logits.mean(dim=0)  # (n_states, n_states)

        # Combine with base logits
        log_A = base_A + delta_mean

        # Apply transition constraints (e.g., semi-Markov)
        if hasattr(constraints, "mask_invalid_transitions"):
            mask = constraints.mask_invalid_transitions(self.n_states, constraints.Transitions.SEMI).to(device)
            log_A = log_A.masked_fill(~mask, -torch.inf)

        # Normalize rows in log-space
        log_A = log_A - torch.logsumexp(log_A, dim=1, keepdim=True)
        return log_A

    # ----------------------
    # Emission mapping & validation
    # ----------------------
    def map_emission(self, x: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute per-state emission log-probabilities for a sequence, optionally using context.

        Enhancements:
        - Fully integrates context vector `theta` via `_contextual_emission_pdf`.
        - Supports batch and single-sequence inputs.
        - Ensures broadcasting and numerical stability in log-space.
        - Raises clear errors for mismatched event shapes or uninitialized PDFs.

        Args:
            x (torch.Tensor): Observation sequence of shape (T, ...) or (B, T, ...).
            theta (Optional[torch.Tensor]): Context vector (H_total, 1) or batch (B, H_total).

        Returns:
            torch.Tensor: Log-probabilities of shape (T, n_states) or (B, T, n_states).
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        dtype = DTYPE

        # Ensure tensor is on correct device & type
        x = x.to(device=device, dtype=dtype)

        # Get context-conditioned emission PDF
        pdf = self._contextual_emission_pdf(x, theta)
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized. Ensure `sample_emission_pdf()` is called.")

        # Ensure event shape compatibility
        if pdf.event_shape and x.shape[-len(pdf.event_shape):] != pdf.event_shape:
            raise ValueError(f"Input shape {x.shape[-len(pdf.event_shape):]} "
                             f"does not match PDF event shape {pdf.event_shape}.")

        # Expand for broadcasting over states
        if x.ndim == len(pdf.event_shape) + 1:  # (T, F...)
            x_exp = x.unsqueeze(1)  # (T, 1, F...)
        elif x.ndim == len(pdf.event_shape) + 2:  # (B, T, F...)
            x_exp = x.unsqueeze(2)  # (B, T, 1, F...)
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        # Compute log-probabilities
        try:
            log_probs = pdf.log_prob(x_exp)  # (T, n_states, ...) or (B, T, n_states, ...)
        except Exception as e:
            raise RuntimeError(f"Error computing log_prob: {e}")

        # If the PDF returns extra dimensions, sum over event dimensions
        if log_probs.ndim > x_exp.ndim:
            log_probs = log_probs.sum(dim=-1)

        # Ensure final shape is (T, n_states) or (B, T, n_states)
        return log_probs

    def check_constraints(self, value: torch.Tensor) -> torch.Tensor:
        """Validate observations against the emission PDF support and event shape."""
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized.")

        # Support validation
        support_mask = pdf.support.check(value)
        if not torch.all(support_mask):
            bad_vals = value[~support_mask].unique()
            raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Shape validation
        event_shape = pdf.event_shape
        expected_ndim = len(event_shape) + 1  # batch + event dims
        if value.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D input (batch + event), got {value.ndim}D.")
        if event_shape and value.shape[1:] != event_shape:
            raise ValueError(f"PDF event shape mismatch: expected {event_shape}, got {value.shape[1:]}.")

        return value

    def to_observations(self, X: torch.Tensor, lengths: Optional[List[int]] = None, theta: Optional[torch.Tensor] = None) -> utils.Observations:
        """
        Convert tensor X into a utils.Observations object for HSMM.

        Enhancements:
        - Fully vectorized per-state log-probabilities using `map_emission`.
        - Supports optional context vector `theta` for neural adaptation.
        - Handles batched and single-sequence inputs.
        - Provides clear validation of lengths and event shapes.

        Args:
            X (torch.Tensor): Input tensor of shape (N, F) or (B, T, F) where N/T is
                              total time steps and F is feature dimension.
            lengths (Optional[List[int]]): Sequence lengths summing to N (optional).
            theta (Optional[torch.Tensor]): Optional context vector (H_total, 1) or batch (B, H_total)
                                            for contextual emissions.

        Returns:
            utils.Observations: Container with attributes:
                - sequence: list of tensors (split by lengths)
                - log_probs: list of log-probabilities per state for each sequence
                - lengths: list of sequence lengths
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        dtype = DTYPE

        # Validate and cast input
        X_valid = self.check_constraints(X).to(device=device, dtype=dtype)
        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]

        # Infer or validate sequence lengths
        if lengths is not None:
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths
        else:
            seq_lengths = [total_len]

        # Split sequences
        if X_valid.ndim == 2:
            sequences = list(torch.split(X_valid, seq_lengths))  # (T, F)
        elif X_valid.ndim == 3:
            # (B, T, F) -> split along batch dimension if lengths correspond to T?
            if len(seq_lengths) == X_valid.shape[0]:
                sequences = [X_valid[i] for i in range(X_valid.shape[0])]
            else:
                # Flatten batch for splitting
                sequences = list(torch.split(X_valid.reshape(-1, X_valid.shape[-1]), seq_lengths))
        else:
            raise ValueError(f"Unsupported input shape {X_valid.shape}")

        # Compute per-sequence log-probabilities using enhanced map_emission
        log_probs_list = []
        for idx, seq in enumerate(sequences):
            # Determine per-sequence context
            seq_theta = None
            if theta is not None:
                if theta.ndim == 2 and theta.shape[0] == len(sequences):
                    seq_theta = theta[idx].unsqueeze(-1)  # (H_total, 1)
                else:
                    seq_theta = theta

            log_probs = self.map_emission(seq, seq_theta)  # (T, n_states)
            log_probs_list.append(log_probs)

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=seq_lengths
        )

    # ----------------------
    # Parameter sampling & EM loop
    # ----------------------
    def sample_model_params(self, X: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Randomly sample initial model parameters (π, A, D, emission_pdf)
        in log-space for EM initialization.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        α = self.alpha

        sampled_pi = torch.log(constraints.sample_probs(α, (self.n_states,))).to(dtype=DTYPE, device=device)
        sampled_A = torch.log(constraints.sample_A(α, self.n_states, constraints.Transitions.SEMI)).to(dtype=DTYPE, device=device)
        sampled_D = torch.log(constraints.sample_probs(α, (self.n_states, self.max_duration))).to(dtype=DTYPE, device=device)

        sampled_pdf = self.sample_emission_pdf(X)

        return {
            "pi": sampled_pi,
            "A": sampled_A,
            "D": sampled_D,
            "emission_pdf": sampled_pdf
        }

    def fit(
        self,
        X: torch.Tensor,
        tol: float = 1e-4,
        max_iter: int = 15,
        n_init: int = 1,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_B_from_X: bool = False,
        verbose: bool = True,
        plot_conv: bool = False,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ):
        """
        Fit the HSMM using EM with full context-awareness and vectorized emissions.

        Enhancements:
        - Leverages `_contextual_emission_pdf`, `_contextual_transition_matrix`, `_contextual_duration_pdf`.
        - Supports per-sequence context vector `theta`.
        - Vectorized log-probability computation across all sequences.
        - Optional multiple EM initializations with best-score selection.
        - Convergence monitored via `ConvergenceHandler`.

        Args:
            X: Input tensor of shape (T, F) or (B, T, F).
            lengths: Optional sequence lengths.
            theta: Optional context tensor (per sequence or global).
            Other args: EM and convergence parameters.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        dtype = DTYPE

        # Initial emission sampling if requested
        if sample_B_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # Encode observations if no explicit context provided
        if theta is None and self.encoder is not None:
            theta = self.encode_observations(X)

        # Convert X to Observations container with per-sequence log-probs
        X_valid = self.to_observations(X, lengths, theta)

        # Align context per sequence
        valid_theta = None
        if theta is not None:
            if theta.ndim == 2 and theta.shape[0] == len(X_valid.sequence):
                valid_theta = theta
            else:
                valid_theta = theta.repeat(len(X_valid.sequence), 1)  # broadcast global context

        # Convergence handler
        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        best_state, best_score = None, -float("inf")

        # Concatenate all sequences for vectorized emissions
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=dtype, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths), dim=0).numpy())

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # Resample model parameters for each initialization after the first
            if run_idx > 0:
                sampled = self.sample_model_params(X)
                self._pi_logits.copy_(sampled['pi'])
                self._A_logits.copy_(sampled['A'])
                self._D_logits.copy_(sampled['D'])
                self._params['emission_pdf'] = sampled['emission_pdf']

            # Initial likelihood
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # Contextual adaptation for this iteration
                self._params['emission_pdf'] = self._contextual_emission_pdf(X_valid, valid_theta)
                A_logits = self._contextual_transition_matrix(valid_theta)
                D_logits = self._contextual_duration_pdf(valid_theta)

                if A_logits is not None:
                    self._A_logits.copy_(A_logits)
                if D_logits is not None:
                    self._D_logits.copy_(D_logits)

                # M-step: estimate updated parameters
                new_params = self._estimate_model_params(X_valid, valid_theta)
                if 'emission_pdf' in new_params:
                    self._params['emission_pdf'] = new_params['emission_pdf']
                if 'pi' in new_params:
                    self._pi_logits.copy_(new_params['pi'])
                if 'A' in new_params:
                    self._A_logits.copy_(new_params['A'])
                if 'D' in new_params:
                    self._D_logits.copy_(new_params['D'])

                # Vectorized recomputation of emission log-probs
                pdf = self._params['emission_pdf']
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or unsupported for vectorized log_prob.")
                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (total_T, K)
                X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i+1]] for i in range(len(X_valid.lengths))]

                # Compute current log-likelihood
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # Track best scoring run
            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'pi': self._pi_logits.clone(),
                    'A': self._A_logits.clone(),
                    'D': self._D_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        # Restore best scoring parameters
        if best_state is not None:
            self._pi_logits.copy_(best_state['pi'])
            self._A_logits.copy_(best_state['A'])
            self._D_logits.copy_(best_state['D'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        # Optional convergence plotting
        if plot_conv and hasattr(self, 'conv'):
            self.conv.plot_convergence()

        return self

    # ----------------------
    # Predict / Score / IC
    # ----------------------
    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi"
    ) -> List[torch.Tensor]:
        """
        Decode most likely hidden state sequences using MAP or Viterbi in a vectorized manner.

        Args:
            X: Input tensor of shape (T, F) or (B, T, F).
            lengths: Optional list of sequence lengths for batched inputs.
            algorithm: 'map' for posterior decoding, 'viterbi' for max-likelihood path.

        Returns:
            List[torch.Tensor]: Decoded state sequences per input.
        """
        # --- normalize input ---
        X_valid = self.to_observations(X, lengths)
        device = next(self.parameters(), torch.tensor(0.0)).device
        algorithm = algorithm.lower()

        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=DTYPE, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths), dim=0).numpy())
        B, K = len(X_valid.lengths), getattr(self, "n_states", None)

        # --- emissions ---
        pdf = self._params.get("emission_pdf", None)
        if pdf is None:
            raise RuntimeError("Emission PDF is not initialized.")
        if not hasattr(pdf, "log_prob"):
            raise NotImplementedError("Vectorized log_prob not implemented for this emission PDF.")
        all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (total_T, K)
        X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(B)]
        K = K or all_log_probs.shape[-1]

        # --- transition matrix (contextual or static) ---
        if hasattr(self, "transition") and hasattr(self.transition, "contextual_logits"):
            A = self.transition.contextual_logits()
        elif hasattr(self, "transition") and hasattr(self.transition, "logits"):
            A = torch.log_softmax(self.transition.logits, dim=-1)
        elif "transition_logits" in self._params:
            A = torch.log_softmax(self._params["transition_logits"], dim=-1)
        else:
            A = torch.full((K, K), 1.0 / K, device=device).log()  # fallback

        # --- batched emission alignment ---
        max_T = max(X_valid.lengths)
        log_B = torch.full((B, max_T, K), -torch.inf, device=device)
        for i, seq_log_probs in enumerate(X_valid.log_probs):
            log_B[i, :seq_log_probs.shape[0]] = seq_log_probs

        # --- mask for valid timesteps ---
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(X_valid.lengths, device=device).unsqueeze(1)

        # --- decoding ---
        if algorithm == "map":
            alpha = torch.full((B, max_T, K), -torch.inf, device=device)
            alpha[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                prev = alpha[:, t - 1].unsqueeze(2) + A  # (B, K, K)
                alpha[:, t] = torch.logsumexp(prev, dim=1) + log_B[:, t]
                alpha[:, t] = torch.where(mask[:, t].unsqueeze(-1), alpha[:, t], alpha[:, t - 1])
            decoded = torch.argmax(alpha, dim=-1)

        elif algorithm == "viterbi":
            delta = torch.full((B, max_T, K), -torch.inf, device=device)
            psi = torch.zeros((B, max_T, K), dtype=torch.long, device=device)
            delta[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                scores = delta[:, t - 1].unsqueeze(2) + A  # (B, K, K)
                psi[:, t] = torch.argmax(scores, dim=1)
                delta[:, t] = torch.max(scores, dim=1).values + log_B[:, t]
                delta[:, t] = torch.where(mask[:, t].unsqueeze(-1), delta[:, t], delta[:, t - 1])

            decoded = torch.zeros((B, max_T), dtype=torch.long, device=device)
            decoded[:, -1] = torch.argmax(delta[:, -1], dim=-1)
            for t in range(max_T - 2, -1, -1):
                decoded[:, t] = torch.gather(psi[:, t + 1], 1, decoded[:, t + 1].unsqueeze(1)).squeeze(1)
                decoded[:, t] = torch.where(mask[:, t], decoded[:, t], decoded[:, t + 1])

        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'. Use 'map' or 'viterbi'.")

        # --- unpack decoded sequences ---
        decoded_list = [decoded[i, :L].detach().cpu() for i, L in enumerate(X_valid.lengths)]
        return decoded_list

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """
        Compute log-likelihood(s) of input sequence(s) under the model.

        Args:
            X: Input tensor (T, F) or (B, T, F).
            lengths: Optional list of valid sequence lengths for each batch.
            by_sample: If True, return per-sequence log-likelihoods;
                       else return total log-likelihood (summed).

        Returns:
            torch.Tensor: Log-likelihoods (B,) or scalar tensor.
        """
        # --- Normalize & validate inputs ---
        obs = self.to_observations(X, lengths)
        device = next(self.parameters(), torch.tensor(0.0)).device
        dtype = getattr(self, "DTYPE", torch.float32)
        B = len(obs.lengths)
        K = getattr(self, "n_states", None)
        max_T = max(obs.lengths)

        # --- Emission log-probabilities ---
        pdf = self._params.get("emission_pdf", None)
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized.")
        if not hasattr(pdf, "log_prob"):
            raise NotImplementedError("Emission PDF lacks log_prob().")

        all_seq = torch.cat(obs.sequence, dim=0).to(dtype=dtype, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(obs.lengths), dim=0).numpy())
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)
        log_B = torch.full((B, max_T, all_log_probs.shape[-1]), -torch.inf, device=device)
        for i, L in enumerate(obs.lengths):
            log_B[i, :L] = all_log_probs[seq_offsets[i]:seq_offsets[i + 1]]
        K = all_log_probs.shape[-1]

        # --- Transition probabilities ---
        if hasattr(self, "transition") and hasattr(self.transition, "contextual_logits"):
            log_A = self.transition.contextual_logits()
        elif hasattr(self, "transition") and hasattr(self.transition, "logits"):
            log_A = torch.log_softmax(self.transition.logits, dim=-1)
        elif "transition_logits" in self._params:
            log_A = torch.log_softmax(self._params["transition_logits"], dim=-1)
        else:
            log_A = torch.full((K, K), (1.0 / K), device=device).log()

        # --- Duration probabilities (if present) ---
        if "duration_pdf" in self._params and self._params["duration_pdf"] is not None:
            dur_pdf = self._params["duration_pdf"]
            if hasattr(dur_pdf, "log_prob"):
                log_D = dur_pdf.log_prob(torch.arange(1, max_T + 1, device=device))
            else:
                log_D = torch.zeros(max_T, device=device)
        else:
            log_D = torch.zeros(max_T, device=device)

        # --- Forward recursion in log-domain ---
        log_alpha = torch.full((B, max_T, K), -torch.inf, device=device)
        log_alpha[:, 0] = log_B[:, 0]
        for t in range(1, max_T):
            prev = log_alpha[:, t - 1].unsqueeze(2) + log_A  # (B, K, K)
            log_alpha[:, t] = torch.logsumexp(prev, dim=1) + log_B[:, t] + log_D[t - 1]

        # --- Mask invalid steps ---
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(obs.lengths, device=device).unsqueeze(1)
        log_alpha = log_alpha * mask.unsqueeze(-1) + (~mask).unsqueeze(-1) * (-torch.inf)

        # --- Compute sequence log-likelihoods ---
        seq_ll = torch.logsumexp(log_alpha.gather(1, (torch.tensor(obs.lengths, device=device) - 1).view(-1, 1, 1).expand(-1, 1, K)), dim=-1).squeeze(1)

        if by_sample:
            return seq_ll.detach().cpu()
        else:
            return seq_ll.sum(dim=0, keepdim=True).detach().cpu()

    def ic(
        self,
        X: torch.Tensor,
        criterion: "constraints.InformCriteria" = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """
        Compute model information criterion (AIC, BIC, etc.) based on log-likelihoods.

        Args:
            X: Input tensor of shape (T, F) or (B, T, F).
            criterion: Information criterion type (AIC, BIC, or custom).
            lengths: Optional list of sequence lengths.
            by_sample: Whether to compute criterion per-sample or aggregate.

        Returns:
            torch.Tensor: Information criterion value(s).
        """
        # --- Compute log-likelihood ---
        log_likelihood = self.score(X, lengths=lengths, by_sample=by_sample)
        if isinstance(log_likelihood, (list, tuple)):
            log_likelihood = torch.tensor(log_likelihood, device=next(self.parameters()).device)
        elif not isinstance(log_likelihood, torch.Tensor):
            log_likelihood = torch.as_tensor(log_likelihood, device=next(self.parameters()).device)

        # --- Determine number of observations ---
        if lengths is not None:
            n_obs = sum(lengths)
        elif X.ndim == 3:
            n_obs = X.shape[0] * X.shape[1]
        else:
            n_obs = X.shape[0]

        n_obs = max(int(n_obs), 1)
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError("Model degrees of freedom ('dof') not set. Define self.dof in initialization.")

        # --- Dispatch computation ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=log_likelihood,
            dof=dof,
            criterion=criterion
        )

        # --- Postprocess output ---
        if isinstance(ic_value, torch.Tensor):
            return ic_value.detach().cpu()
        return torch.as_tensor(ic_value, dtype=torch.float32)

    # ----------------------
    # Forward / Backward / Posteriors
    # ----------------------
    def _forward(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        Log-domain forward recursion for explicit-duration HSMM.
        Returns: list of tensors (T, n_states, max_duration)
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        pi, A, D = self.pi, self.A, self.D
        neg_inf = -torch.inf
        alpha_vec: List[torch.Tensor] = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            log_alpha = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device), torch.cumsum(seq_probs, dim=0)))

            # t = 0 initialization
            max_d = min(self.max_duration, seq_len)
            durations = torch.arange(1, max_d + 1, device=device)
            emit_sums = (cumsum_emit[durations] - cumsum_emit[0]).T  # (n_states, durations)
            log_alpha[0, :, :max_d] = pi.unsqueeze(-1) + D[:, :max_d] + emit_sums

            # Recursion
            for t in range(1, seq_len):
                prev_alpha = log_alpha[t - 1]

                # Continuation of durations
                shifted = torch.cat([
                    prev_alpha[:, 1:],
                    torch.full((self.n_states, 1), neg_inf, dtype=DTYPE, device=device)
                ], dim=1)

                # Transition to new durations
                trans = torch.logsumexp(prev_alpha[:, 0].unsqueeze(1) + A, dim=0)
                log_alpha[t] = torch.logsumexp(
                    torch.stack([
                        shifted + seq_probs[t].unsqueeze(-1),
                        D + trans.unsqueeze(-1)
                    ]),
                    dim=0
                )

            alpha_vec.append(log_alpha)

        return alpha_vec

    def _backward(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        Log-domain backward recursion for explicit-duration HSMM.
        Returns: list of tensors (T, n_states, max_duration)
        """
        beta_vec: List[torch.Tensor] = []
        A, D = self.A, self.D
        neg_inf = -torch.inf

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            device = seq_probs.device
            log_beta = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)
            log_beta[-1].fill_(0.)

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((
                torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                torch.cumsum(seq_probs, dim=0)
            ))

            durations_all = torch.arange(1, self.max_duration + 1, device=device)

            for t in reversed(range(seq_len - 1)):
                max_d = min(self.max_duration, seq_len - t)
                durations = durations_all[:max_d]

                # Emission sums over durations [t, t+d)
                emit_sums = cumsum_emit[t + durations] - cumsum_emit[t]  # (durations, n_states)
                emit_sums = emit_sums.T  # (n_states, durations)

                dur_lp = D[:, :max_d]
                beta_next = log_beta[t + durations - 1, :, 0].T  # (n_states, durations)

                # Combine emissions, duration, and next beta
                contrib = emit_sums + dur_lp + beta_next
                log_beta[t, :, 0] = torch.logsumexp(contrib, dim=1)

                # Continue existing durations
                if self.max_duration > 1:
                    log_beta[t, :, 1:] = log_beta[t + 1, :, :-1] + seq_probs[t + 1].unsqueeze(-1)

            beta_vec.append(log_beta)

        return beta_vec

    def _compute_posteriors(self, X: "utils.Observations") -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute posterior expectations for HSMM sequences:
          γ_t(s)    = state marginals
          ξ_t(s,s') = transition marginals
          η_t(s,d)  = duration marginals

        Returns lists aligned with X.sequences.
        """
        alpha_list = self._forward(X)
        beta_list = self._backward(X)

        gamma_vec, xi_vec, eta_vec = [], [], []

        for seq_log_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            device = alpha.device
            n_states, max_dur = self.n_states, self.max_duration

            if alpha is None or beta is None or alpha.numel() == 0 or beta.numel() == 0:
                # Empty or degenerate sequence
                gamma_vec.append(torch.zeros((seq_len, n_states), device=device))
                eta_vec.append(torch.zeros((seq_len, n_states, max_dur), device=device))
                xi_vec.append(torch.zeros((max(seq_len-1, 0), n_states, n_states), device=device))
                continue

            # --- State marginals γ ---
            gamma = torch.logsumexp(alpha + beta, dim=2)  # sum over durations
            gamma = constraints.log_normalize(gamma, dim=1).exp()  # normalize across states
            gamma_vec.append(gamma)

            # --- Duration marginals η ---
            log_eta = alpha + beta
            eta = constraints.log_normalize(log_eta, dim=(1, 2)).exp()  # normalize across states & durations
            eta_vec.append(eta)

            # --- Transition marginals ξ ---
            if seq_len > 1:
                # α_t(s) for t = 0..T-2, durations = 1 (start of next state)
                trans_alpha = alpha[:-1, :, 0].unsqueeze(2) + self.A.unsqueeze(0)  # (T-1, s, s')

                # sum over durations for β_{t+1} and D
                beta_next = beta[1:] + self.D.unsqueeze(0)  # (T-1, s', max_dur)
                dur_beta_sum = torch.logsumexp(beta_next, dim=2)  # (T-1, s')

                log_xi = trans_alpha + dur_beta_sum.unsqueeze(1)  # broadcast to (T-1, s, s')
                xi = constraints.log_normalize(log_xi, dim=(1, 2)).exp()
                xi_vec.append(xi)
            else:
                xi_vec.append(torch.zeros((0, n_states, n_states), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _estimate_model_params(self, X: utils.Observations, theta: Optional[utils.ContextualVariables] = None) -> Dict[str, Any]:
        """
        M-step: Estimate updated HSMM parameters from posterior expectations.

        Parameters
        ----------
        X : utils.Observations
            Batched observation sequences with precomputed log-probabilities.
        theta : Optional[utils.ContextualVariables], default=None
            Optional contextual variables from encoder or external source.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing updated model parameters:
                - 'pi' : Initial state probabilities (log-space)
                - 'A'  : State transition matrix (log-space)
                - 'D'  : Duration distributions (log-space)
                - 'emission_pdf' : Updated emission distribution (torch.distributions.Distribution)
        
        Notes
        -----
        - Fully vectorized across sequences for efficiency.
        - Posterior expectations γ, ξ, η are used to compute expected sufficient statistics.
        - Emission update is delegated to `_estimate_emission_pdf`, which can leverage
          optional contextual variables for neural HSMMs.
        """
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)
        device = next(self.parameters(), torch.tensor(0.0)).device
        dtype = DTYPE

        # -------------------------------
        # π (initial state probabilities)
        # -------------------------------
        pi_stack = torch.stack([g[0] for g in gamma_list], dim=1).to(device=device, dtype=dtype)  # (n_states, n_sequences)
        new_pi = constraints.log_normalize(torch.log(pi_stack.sum(dim=1)), dim=0)

        # -------------------------------
        # A (state transition matrix)
        # -------------------------------
        xi_valid = [x for x in xi_list if x.numel() > 0]
        if xi_valid:
            xi_cat = torch.cat(xi_valid, dim=0).to(device=device, dtype=dtype)
            new_A = constraints.log_normalize(torch.logsumexp(xi_cat, dim=0), dim=1)
        else:
            new_A = self.A.clone()

        # -------------------------------
        # D (duration distributions)
        # -------------------------------
        eta_cat = torch.cat([e for e in eta_list], dim=0).to(device=device, dtype=dtype)
        new_D = constraints.log_normalize(torch.logsumexp(eta_cat, dim=0), dim=1)

        # -------------------------------
        # Emission PDF (contextual or base)
        # -------------------------------
        if X.sequence:
            all_X = torch.cat(X.sequence, dim=0).to(device=device, dtype=dtype)
            all_gamma = torch.cat([g for g in gamma_list], dim=0).to(device=device, dtype=dtype)
            new_pdf = self._estimate_emission_pdf(all_X, all_gamma, theta)
        else:
            # fallback to existing emission PDF if no data
            new_pdf = self._params.get('emission_pdf')

        return {
            'pi': new_pi,
            'A': new_A,
            'D': new_D,
            'emission_pdf': new_pdf
        }

    # ----------------------
    # Viterbi (semi-Markov)
    # ----------------------
    @torch.no_grad()
    def _viterbi(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        Vectorized Viterbi decoding for multiple sequences (semi-Markov HSMM).
        Optimized for memory and numerical stability.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        K, Dmax = self.n_states, self.max_duration
        A, pi, D = self.A.to(device), self.pi.to(device), self.D.to(device)
        neg_inf = -torch.inf

        B = [seq.to(dtype=DTYPE, device=device) for seq in X.log_probs]
        lengths = X.lengths
        max_len = max(lengths)

        # Initialize score and backtrack tensors
        V = torch.full((len(B), max_len, K), neg_inf, dtype=DTYPE, device=device)
        best_prev = torch.full((len(B), max_len, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((len(B), max_len, K), dtype=torch.int64, device=device)

        for seq_idx, (seq_probs, T) in enumerate(zip(B, lengths)):
            if T == 0:
                continue

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((
                torch.zeros((1, K), dtype=DTYPE, device=device),
                torch.cumsum(seq_probs, dim=0)
            ))

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, dtype=torch.int64, device=device)
                starts = t - durations + 1

                # Sum of emissions for each possible duration
                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # (K, durations)
                dur_lp = D[:, :max_d]  # (K, durations)

                if t == 0:
                    # Initialization: starting at t=0 with any duration
                    scores = pi.unsqueeze(1) + dur_lp + emit_sums
                    prev_idx = torch.full_like(scores, -1, dtype=torch.int64)
                else:
                    # Previous scores for each possible duration
                    V_prev = V[seq_idx, t - durations]  # (durations, K)
                    scores_plus_trans = V_prev.unsqueeze(2) + A.unsqueeze(0)  # (durations, K, K)
                    scores_max, argmax_prev = scores_plus_trans.max(1)  # (durations, K)
                    scores = scores_max.T + dur_lp + emit_sums  # (K, durations)
                    prev_idx = argmax_prev.T  # (K, durations)

                # Select best duration for each state
                best_score_dur, best_d_idx = scores.max(dim=1)
                V[seq_idx, t] = best_score_dur
                best_dur[seq_idx, t] = durations[best_d_idx]
                best_prev[seq_idx, t] = prev_idx[torch.arange(K), best_d_idx]

        # Backtracking
        paths = []
        for seq_idx, T in enumerate(lengths):
            if T == 0:
                paths.append(torch.empty((0,), dtype=torch.int64, device=device))
                continue

            t = T - 1
            cur_state = int(V[seq_idx, t].argmax().item())
            decoded_segments = []

            while t >= 0:
                d = int(best_dur[seq_idx, t, cur_state].item())
                start = t - d + 1
                decoded_segments.append((start, t, cur_state))
                prev_state = int(best_prev[seq_idx, t, cur_state].item())
                if prev_state < 0:
                    break
                t = start - 1
                cur_state = prev_state

            decoded_segments.reverse()
            seq_path = torch.cat([torch.full((e - s + 1,), st, dtype=torch.int64, device=device)
                                  for s, e, st in decoded_segments])
            paths.append(seq_path)

        return paths

    def _map(self, X: "utils.Observations") -> List[torch.Tensor]:
        """
        Compute Maximum A Posteriori (MAP) state sequence estimates.

        Args:
            X: utils.Observations — preprocessed observation container
               containing log-probabilities, lengths, and contextual metadata.

        Returns:
            List[torch.Tensor]: Each tensor is of shape (T_i,) giving the
            MAP-decoded state index sequence for the i-th sample.
        """
        # --- Compute posteriors (log or normalized) ---
        gamma, _, _ = self._compute_posteriors(X)
        if gamma is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        map_sequences = []
        for seq_idx, tens in enumerate(gamma):
            if tens is None or tens.numel() == 0:
                map_sequences.append(torch.empty(0, dtype=torch.long, device=next(self.parameters()).device))
                continue

            # Numerical stabilization
            tens = tens.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure proper normalization if not already probabilities
            if not torch.allclose(tens.sum(dim=1), torch.ones_like(tens.sum(dim=1)), atol=1e-3):
                tens = torch.softmax(tens, dim=1)

            # Decode state index sequence
            seq_map = tens.argmax(dim=1)
            map_sequences.append(seq_map)

        return map_sequences

    def _compute_log_likelihood(self, X: "utils.Observations") -> torch.Tensor:
        """
        Compute sequence log-likelihoods using the forward algorithm.

        Args:
            X: utils.Observations
                Preprocessed observation container (with log-probs, lengths, etc.)

        Returns:
            torch.Tensor: Vector of log-likelihoods for each sequence (shape: [n_sequences])
        """
        log_alpha_vec = self._forward(X)
        if not log_alpha_vec:
            raise RuntimeError("Forward pass returned empty results. Model may be uninitialized.")

        device = next(self.parameters(), torch.tensor(0.0)).device
        neg_inf = -torch.inf
        log_likelihoods = []

        for log_alpha in log_alpha_vec:
            if log_alpha is None or log_alpha.numel() == 0:
                log_likelihoods.append(torch.tensor(neg_inf, device=device))
                continue

            log_alpha = log_alpha.nan_to_num(nan=-1e8, posinf=-1e8, neginf=-1e8)
            ll = torch.logsumexp(log_alpha[-1], dim=0)

            # Correct finite-check reduction
            if not torch.isfinite(ll).all():
                ll = torch.full_like(ll, float("-inf"))

            # Aggregate over state dimension
            if ll.ndim > 0:
                ll = torch.logsumexp(ll, dim=0)

            log_likelihoods.append(ll)

        return torch.stack(log_likelihoods)
