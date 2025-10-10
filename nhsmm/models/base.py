# nhsmm/hsmm/BaseHSMM.py
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
    def attach_encoder(self, encoder: nn.Module):
        """
        Attach a neural encoder module (e.g. CNN+LSTM). Encoder must accept a single sequence
        shaped (1, T, F) or (T, F) and return either:
          - a tensor of shape (hidden_dim,) representing sequence-level context, or
          - a tensor of shape (hidden_dim, T) or (hidden_dim, 1) compatible with to_contextuals().
        The BaseHSMM will transform encoder output into `theta` used by contextual hooks.
        """
        self.encoder = encoder

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
    # Contextual hooks (enhanced)
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

    def _contextual_transition_matrix(self, theta: Optional[utils.ContextualVariables]) -> torch.Tensor:
        """
        Return a context-adapted transition matrix (log-space).

        - Combines encoder output and stored context.
        - Neural subclasses can override this to produce theta-conditioned transitions.
        - Default: returns current transition buffer `_A_logits`.

        Args:
            theta: Optional contextual variables

        Returns:
            torch.Tensor: log-space transition matrix of shape (n_states, n_states)
        """
        theta_combined = self._combine_context(theta)

        # Default: return current A logits if no context is available
        if theta_combined is None:
            return self._A_logits

        # Placeholder: subclasses can implement mapping from theta_combined -> A_logits
        return self._A_logits

    def _contextual_duration_pdf(self, theta: Optional[utils.ContextualVariables]) -> torch.Tensor:
        """
        Return a context-adapted duration distribution (log-space).

        - Combines encoder output and stored context.
        - Neural subclasses can override this to produce theta-conditioned durations.
        - Default: returns current duration buffer `_D_logits`.

        Args:
            theta: Optional contextual variables

        Returns:
            torch.Tensor: log-space duration matrix of shape (n_states, max_duration)
        """
        theta_combined = self._combine_context(theta)

        # Default: return current D logits if no context is available
        if theta_combined is None:
            return self._D_logits

        # Placeholder: subclasses can implement mapping from theta_combined -> D_logits
        return self._D_logits

    # ----------------------
    # Emission mapping & validation
    # ----------------------
    def map_emission(self, x: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute per-state emission log-probabilities for a sequence, optionally using context.

        This method maps each observation in `x` to its log-probability under each
        hidden state according to the current emission distribution. If a context
        vector `theta` is provided, it uses the `_contextual_emission_pdf` hook to
        compute context-adapted emissions.

        Args:
            x (torch.Tensor): Observation sequence of shape (T, ...) where T is
                              sequence length and ... matches the emission event shape.
            theta (Optional[torch.Tensor]): Optional context vector of shape (H_total, 1)
                                            used to condition emission probabilities.

        Returns:
            torch.Tensor: Log-probabilities of shape (T, n_states), where each entry
                          log_probs[t, s] = log P(x[t] | state=s, theta).

        Raises:
            RuntimeError: If the emission PDF is not initialized.
            ValueError: If the observation shape does not match the emission PDF's event shape.
        """
        # Get context-adapted emission distribution if theta is provided
        pdf = self._contextual_emission_pdf(x, theta) if theta is not None else self.pdf

        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized; subclass must implement `sample_emission_pdf()`."
            )

        assert pdf.event_shape == x.shape[1:], f"PDF event shape mismatch {pdf.event_shape} vs {x.shape[1:]}"

        # Ensure input matches expected event shape
        event_shape = pdf.event_shape
        if event_shape and x.shape[1:] != event_shape:
            raise ValueError(
                f"Input event shape {x.shape[1:]} does not match PDF event shape {event_shape}."
            )

        # Expand state dimension for broadcasting across states
        x_expanded = x.unsqueeze(1)  # (T, 1, ...)
        log_probs = pdf.log_prob(x_expanded)  # Broadcast over states

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

    def to_observations(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Convert tensor X into a utils.Observations object for HSMM.

        This method:
            - Splits the input tensor `X` into sequences according to `lengths`.
            - Computes per-state log-probabilities for each sequence.
            - Supports optional context vector `theta` for neural or hierarchical HSMMs.

        Args:
            X (torch.Tensor): Input tensor of shape (N, F) or (T, F) where N/T is
                              total time steps and F is feature dimension.
            lengths (Optional[List[int]]): Optional sequence lengths summing to N.
            theta (Optional[torch.Tensor]): Optional context vector (H_total, 1)
                                            to condition emission probabilities.

        Returns:
            utils.Observations: Container with attributes:
                - sequence: list of tensors (split by lengths)
                - log_probs: list of log-probabilities per state for each sequence
                - lengths: list of sequence lengths

        Raises:
            ValueError: If sum(lengths) does not match X.size(0).
            RuntimeError: If emission PDF is not initialized.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        X_valid = self.check_constraints(X).to(dtype=DTYPE, device=device)
        n_samples = X_valid.size(0)

        # Validate or infer sequence lengths
        if lengths is not None:
            if sum(lengths) != n_samples:
                raise ValueError(
                    f"Sum of lengths ({sum(lengths)}) does not match total samples ({n_samples})."
                )
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]

        # Split sequences
        tensor_list = list(torch.split(X_valid, seq_lengths))

        # Compute per-sequence emission log-probabilities
        tensor_probs = [self.map_emission(seq, theta) for seq in tensor_list]

        return utils.Observations(sequence=tensor_list, log_probs=tensor_probs, lengths=seq_lengths)

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

    def fit(self,
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
            theta: Optional[torch.Tensor] = None):
        """
        Fit the HSMM using EM with vectorized emissions for maximal efficiency.

        Vectorized across sequences to avoid Python loops, preserving full contextual adaptation.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device

        if sample_B_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        if theta is None and self.encoder is not None:
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths)
        valid_theta = self.to_contextuals(theta, X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        best_state, best_score = None, -float("inf")

        # Pre-convert sequences to a batched tensor for vectorized emissions
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=DTYPE, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths), dim=0).numpy())

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            if run_idx > 0:
                sampled = self.sample_model_params(X)
                self._pi_logits.copy_(sampled['pi'].to(dtype=DTYPE, device=device))
                self._A_logits.copy_(sampled['A'].to(dtype=DTYPE, device=device))
                self._D_logits.copy_(sampled['D'].to(dtype=DTYPE, device=device))
                self._params['emission_pdf'] = sampled['emission_pdf']

            # Initial likelihood
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # Contextual adaptation
                self._params['emission_pdf'] = self._contextual_emission_pdf(X_valid, valid_theta)
                A_logits = self._contextual_transition_matrix(valid_theta)
                D_logits = self._contextual_duration_pdf(valid_theta)

                if A_logits is not None:
                    self._A_logits.copy_(A_logits.to(dtype=DTYPE, device=device))
                if D_logits is not None:
                    self._D_logits.copy_(D_logits.to(dtype=DTYPE, device=device))

                # M-step
                new_params = self._estimate_model_params(X_valid, valid_theta)
                if 'emission_pdf' in new_params:
                    self._params['emission_pdf'] = new_params['emission_pdf']
                if 'pi' in new_params:
                    self._pi_logits.copy_(new_params['pi'].to(dtype=DTYPE, device=device))
                if 'A' in new_params:
                    self._A_logits.copy_(new_params['A'].to(dtype=DTYPE, device=device))
                if 'D' in new_params:
                    self._D_logits.copy_(new_params['D'].to(dtype=DTYPE, device=device))

                # --- Vectorized emission log-probabilities ---
                # seq_tensor: (total_T, F), emission_pdf: (K, ...)
                pdf = self._params['emission_pdf']
                if pdf is None:
                    raise RuntimeError("Emission PDF not initialized.")
                if hasattr(pdf, 'log_prob'):
                    # Vectorized computation across all sequences
                    x_exp = seq_tensor.unsqueeze(1)  # (total_T, 1, F)
                    all_log_probs = pdf.log_prob(x_exp)  # (total_T, K) broadcasted
                else:
                    raise NotImplementedError("Vectorized log_prob not implemented for this PDF type.")

                # Split back to sequences
                X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i+1]] for i in range(len(X_valid.lengths))]

                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'pi': self._pi_logits.clone(),
                    'A': self._A_logits.clone(),
                    'D': self._D_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        # Restore best
        if best_state is not None:
            self._pi_logits.copy_(best_state['pi'])
            self._A_logits.copy_(best_state['A'])
            self._D_logits.copy_(best_state['D'])
            self._params['emission_pdf'] = best_state['emission_pdf']

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
        X_valid = self.to_observations(X, lengths)

        # Vectorized log-probabilities for all sequences at once
        device = next(self.parameters(), torch.tensor(0.0)).device
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=DTYPE, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths), dim=0).numpy())

        # Compute emission log-probs once for all sequences
        pdf = self._params['emission_pdf']
        if pdf is None:
            raise RuntimeError("Emission PDF is not initialized.")
        if hasattr(pdf, 'log_prob'):
            all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (total_T, K)
        else:
            raise NotImplementedError("Vectorized log_prob not implemented for this PDF type.")

        # Split back per sequence
        X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i+1]] for i in range(len(X_valid.lengths))]

        # Decoder selection
        algorithm = algorithm.lower()
        if algorithm == "map":
            return self._map(X_valid)
        elif algorithm == "viterbi":
            return self._viterbi(X_valid)
        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'. Use 'map' or 'viterbi'.")

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """Compute log-likelihood(s) of input sequence(s)."""
        obs = self.to_observations(X, lengths)
        log_likelihoods = self._compute_log_likelihood(obs)
        return log_likelihoods if by_sample else log_likelihoods.sum(dim=0, keepdim=True)

    def ic(
        self,
        X: torch.Tensor,
        criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        log_likelihood = self.score(X, lengths, by_sample)
        n_obs = len(lengths) if lengths is not None else X.shape[0]
        return constraints.compute_information_criteria(n_obs, log_likelihood, self.dof, criterion)

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
            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

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
            log_beta = torch.full((seq_len, self.n_states, self.max_duration),
                                  neg_inf, dtype=DTYPE, device=device)
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

    def _compute_posteriors(self, X: utils.Observations) -> Tuple[List[torch.Tensor], ...]:
        """
        Compute posterior expectations for HSMM sequences:
          γ_t(s)  = state marginals
          ξ_t(s,s') = transition marginals
          η_t(s,d) = duration marginals

        Returns lists aligned with X.sequences.
        """
        alpha_list = self._forward(X)
        beta_list = self._backward(X)

        gamma_vec, xi_vec, eta_vec = [], [], []

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            device = alpha.device
            n_states, max_dur = self.n_states, self.max_duration

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
                # alpha_t(s) for t = 0..T-2, durations = 1 (start of next state)
                trans_alpha = alpha[:-1, :, 0].unsqueeze(2) + self.A.unsqueeze(0)  # (T-1, s, s')

                # sum over durations for beta_t+1 and D
                beta_next = beta[1:] + self.D.unsqueeze(0)  # (T-1, s, max_dur)
                dur_beta_sum = torch.logsumexp(beta_next, dim=2)  # (T-1, s')

                log_xi = trans_alpha + dur_beta_sum.unsqueeze(1)  # broadcast to (T-1, s, s')
                xi = constraints.log_normalize(log_xi, dim=(1, 2)).exp()
                xi_vec.append(xi)
            else:
                xi_vec.append(torch.empty((0, n_states, n_states), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _estimate_model_params(
        self,
        X: utils.Observations,
        theta: Optional[utils.ContextualVariables] = None
    ) -> Dict[str, Any]:
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

    def _map(self, X: utils.Observations) -> List[torch.Tensor]:
        gamma, _, _ = self._compute_posteriors(X)
        return [tens.argmax(1) for tens in gamma]

    def _compute_log_likelihood(self, X: utils.Observations) -> torch.Tensor:
        log_alpha_vec = self._forward(X)
        # return vector of shape (n_sequences,)
        return torch.stack([log_alpha[-1].logsumexp(0) for log_alpha in log_alpha_vec])

