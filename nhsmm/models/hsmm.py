# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Independent

from nhsmm.context import ContextEncoder
from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger
from nhsmm.distributions import Initial, Emission, Duration, Transition
from nhsmm import utils, constraints, SeedGenerator, ConvergenceMonitor


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
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self._seed_gen = SeedGenerator(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_type = transition_type or constraints.Transitions.ERGODIC
        self.emission_type = emission_type
        self.max_duration = max_duration
        self.modulate_var = modulate_var
        self.context_dim = context_dim
        self.n_features = n_features
        self.min_covar = min_covar
        self.n_states = n_states
        self.alpha = alpha

        self._params: Dict[str, Any] = {
            'initial_pdf': None,
            'emission_pdf': None,
            'duration_pdf': None,
            'transition_pdf': None,
        }
        self._context: Optional[torch.Tensor] = None
        self.encoder: Optional[nn.Module] = None

        self._init_modules()

    def _init_modules(self, seed: Optional[int] = None) -> None:
        """Initialize HSMM modules: initial, emission, duration, and transition using their `initialize()` methods."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        device = self.device
        hidden_dim = getattr(self, "hidden_dim", None)

        # ---------------- Initialize Modules ----------------
        self.initial_module = Initial(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            init_mode="uniform",
            debug=getattr(self, "debug", False),
            device=device,
        ).to(device)

        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            emission_type=self.emission_type,
            context_dim=self.context_dim,
            modulate_var=self.modulate_var,
            min_covar=self.min_covar,
            dof=getattr(self, "dof", 5.0),
            seed=seed or getattr(self, "seed", 0),
            debug=getattr(self, "debug", False),
        ).to(device)

        self.duration_module = Duration(
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            init_mode="uniform",
            temperature=getattr(self, "duration_temp", 1.0),
            scale=getattr(self, "duration_scale", 1.0),
            debug=getattr(self, "debug", False),
            device=device,
        ).to(device)

        self.transition_module = Transition(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            init_mode="uniform",
            temperature=getattr(self, "transition_temp", 1.0),
            scale=getattr(self, "transition_scale", 1.0),
            debug=getattr(self, "debug", False),
            device=device,
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
        if getattr(self, "debug", False):
            logger.debug(
                f"HSMM modules initialized on {device}: "
                f"n_states={self.n_states}, n_features={self.n_features}, "
                f"context_dim={self.context_dim}, emission={self.emission_type}, "
                f"max_duration={self.max_duration}"
            )

    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    def get_model_params(
        self,
        X: Optional["Observations"] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
        max_iter: int = 50,
        iter_idx: int = 0,
    ) -> dict[str, Any]:
        """
        Compute or sample HSMM parameters with optional contextual modulation.
        Parameters are returned as distribution objects (Categorical, etc.).
        """
        eps_collapse: float = 1e-3
        device, dtype = self.device, DTYPE
        K, Dmax = self.n_states, self.max_duration

        # ---------------- Align theta ----------------
        def _align_theta(theta: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
            if theta is None: return None
            if theta.ndim == 1: return theta.unsqueeze(0).expand(seq_len, -1).to(device, dtype)
            if theta.ndim == 2:
                if theta.shape[0] in [1, seq_len]:
                    return theta.expand(seq_len, -1).to(device, dtype)
            raise ValueError(f"Cannot align theta {theta.shape} with seq_len={seq_len}")

        seq_len = sum(getattr(X, "lengths", [1])) if X is not None else 1
        aligned_theta = _align_theta(theta, seq_len)

        # ---------------- Observation tensor ----------------
        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequence", [X]) if s.numel() > 0], dim=0).to(device, dtype)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, device=device, dtype=dtype)
        else:
            all_X = torch.zeros(1, self.n_features, device=device, dtype=dtype)

        # ---------------- Adaptive α decay ----------------
        α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
        α = float(max(α_min, α_max * (1.0 - iter_idx / max_iter)))

        # ---------------- Module log-probs ----------------
        def _get_log_probs(key: str, module: nn.Module, expected_shape: tuple):
            cached = getattr(self, "_params", {}).get(key, None)
            if cached is not None:
                logits = getattr(cached, "logits", None)
                if logits is not None and logits.shape == expected_shape:
                    return logits.to(device, dtype)
            try:
                lp = module.forward(context=aligned_theta, log=True)
                if torch.is_tensor(lp) and lp.shape != expected_shape and lp.numel() == math.prod(expected_shape):
                    lp = lp.view(expected_shape)
                return lp.to(device, dtype)
            except Exception:
                return torch.full(expected_shape, -math.log(expected_shape[-1]), device=device, dtype=dtype)

        init_ctx = _get_log_probs("initial_pdf", self.initial_module, (K,))
        trans_ctx = _get_log_probs("transition_pdf", self.transition_module, (K, K))
        dur_ctx = _get_log_probs("duration_pdf", self.duration_module, (K, Dmax))

        # ---------------- Collapse-safe Dirichlet ----------------
        def collapse_safe(logits: torch.Tensor, dim: int) -> torch.Tensor:
            probs = logits.exp()
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            dir_alpha = probs * α + EPS
            # Vectorized Dirichlet sampling
            for _ in range(5):
                noise = torch.stack([torch.distributions.Dirichlet(a).rsample() for a in dir_alpha]) if dir_alpha.ndim == 2 else torch.distributions.Dirichlet(dir_alpha).rsample()
                noise = noise / noise.sum(dim=dim, keepdim=True)
                if (noise.max(dim=dim).values < 1.0 - eps_collapse).all():
                    break
            return torch.log(noise + EPS)

        # ---------------- Mode: sample ----------------
        if mode.lower() == "sample":
            init_logits = collapse_safe(init_ctx, dim=0)
            transition_logits = collapse_safe(trans_ctx, dim=1)
            duration_logits = collapse_safe(dur_ctx, dim=1)

            try:
                emission_pdf = self.emission_module.initialize(
                    X=all_X,
                    posterior=None,
                    theta=aligned_theta,
                    context=aligned_theta,
                    theta_scale=theta_scale,
                )
            except Exception:
                emission_pdf = self.emission_module.initialize(mode="uniform")

        # ---------------- Mode: estimate ----------------
        elif mode.lower() == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Observations X required for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=aligned_theta)

            def safe_sum(lst: list[torch.Tensor], dim=0) -> Optional[torch.Tensor]:
                vals = [x.sum(dim=dim) for x in lst if x is not None and x.numel() > 0]
                return torch.stack(vals).sum(0) if vals else None

            init_counts = safe_sum([g[0] for g in gamma_list])
            trans_counts = safe_sum(xi_list)
            dur_counts = safe_sum(eta_list)

            init_counts = init_counts if init_counts is not None and (init_counts >= EPS).all() else init_ctx.exp()
            trans_counts = trans_counts if trans_counts is not None and (trans_counts.sum(-1) >= EPS).all() else trans_ctx.exp()
            dur_counts = dur_counts if dur_counts is not None and (dur_counts.sum(-1) >= EPS).all() else dur_ctx.exp()

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
            try:
                emission_pdf = self.emission_module.initialize(
                    X=all_X,
                    posterior=all_gamma,
                    theta=aligned_theta,
                    context=aligned_theta,
                    theta_scale=theta_scale,
                )
            except Exception:
                emission_pdf = self.emission_module.initialize(mode="uniform")

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # ---------------- Wrap as Distribution ----------------
        return {
            "initial_pdf": Categorical(logits=init_logits),
            "transition_pdf": Categorical(logits=transition_logits),
            "duration_pdf": Categorical(logits=duration_logits),
            "emission_pdf": emission_pdf,
        }

    def attach_encoder(
        self,
        encoder: nn.Module,
        n_heads: int = 4,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
    ) -> ContextEncoder:
        """Attach a ContextEncoder wrapper to the model."""
        self.encoder = ContextEncoder(
            pool=pool,
            n_heads=n_heads,
            encoder=encoder,
            device=self.device
        )
        return self.encoder

    def _validate(self, value: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """
        Validate a tensor against the emission PDF:
          - Align device and dtype
          - Add batch/time dimension if missing
          - Check PDF support (optional clamp)
          - Verify event shape
        """
        pdf = self._params.get('emission_pdf')
        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized. Call `sample_emission_pdf()` or `encode_observations()` first."
            )

        # Align device and dtype
        if hasattr(pdf, "mean") and isinstance(pdf.mean, torch.Tensor):
            value = value.to(device=pdf.mean.device, dtype=pdf.mean.dtype)

        # Flatten single time step if needed
        event_shape = pdf.event_shape or ()
        if event_shape and value.ndim == len(event_shape):
            value = value.unsqueeze(0)

        # Support check
        if hasattr(pdf.support, "check"):
            support_mask = pdf.support.check(value)
            if not torch.all(support_mask):
                if clamp:
                    if hasattr(pdf.support, "clamp"):
                        value = torch.where(support_mask, value, pdf.support.clamp(value))
                    else:
                        min_val = getattr(pdf.support, "lower_bound", -float("inf"))
                        max_val = getattr(pdf.support, "upper_bound", float("inf"))
                        value = value.clamp(min=min_val, max=max_val)
                else:
                    bad_vals = value[~support_mask].flatten().unique()
                    raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Event shape validation
        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(
                f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}"
            )

        return value

    @torch.no_grad()
    def encode_observations(
        self,
        X: torch.Tensor | list[torch.Tensor],
        pool: Optional[str] = None,
        detach_return: bool = True,
        store: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Encode sequences into context vectors using the attached encoder.

        Returns:
            Tensor of shape [B,H] for B sequences or [1,H] for single sequence.
            None if no encoder is attached.
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = self.device

        # Normalize input to list of tensors
        if isinstance(X, torch.Tensor):
            if X.ndim == 1:
                X = [X.unsqueeze(0)]          # [F] -> [1,F]
            elif X.ndim == 2:
                X = [X]                       # [T,F]
            elif X.ndim == 3:
                X = [x for x in X]            # [B,T,F]
            else:
                raise ValueError(f"Unsupported tensor shape {X.shape}")
        elif isinstance(X, list):
            if not all(torch.is_tensor(x) for x in X):
                raise TypeError("All elements in list X must be torch.Tensor.")
        else:
            raise TypeError(f"Unsupported input type {type(X)}")

        contexts = []
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            for seq in X:
                seq = seq.to(device=device, dtype=DTYPE)
                if seq.numel() == 0:
                    raise ValueError(f"Empty sequence encountered: {seq.shape}")

                # Ensure shape [1,T,F] for encoder
                if seq.ndim == 1:
                    seq = seq.unsqueeze(0).unsqueeze(0)
                elif seq.ndim == 2:
                    seq = seq.unsqueeze(0)
                elif seq.ndim != 3:
                    raise ValueError(f"Sequence must be (F,) or (T,F), got {seq.shape}")

                _ = self.encoder(seq, return_context=True)
                vec = self.encoder.get_context()
                if vec is None:
                    raise RuntimeError("Encoder returned None context")

                vec = vec.to(device=device, dtype=DTYPE)
                if detach_return:
                    vec = vec.detach().clone()
                if vec.ndim == 1:
                    vec = vec.unsqueeze(0)  # [1,H]

                contexts.append(vec)

        finally:
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        out = torch.cat(contexts, dim=0)  # [B,H]
        if store:
            self._context = out

        return out

    def to_observations(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None
    ) -> utils.Observations:
        """
        Vectorized conversion of sequences into Observations using _params.

        Supports:
            - Single sequence [T,F] or [F]
            - Batch [B,T,F] or list of [T_i,F]
            - Context: [D], [T,D], [B,T,D] or list of [T_i,D]
            - Emission PDFs: MultivariateNormal or Categorical
        Returns:
            utils.Observations(sequence, log_probs, lengths, context)
        """
        device = self.device

        # --- Normalize sequences to list ---
        if torch.is_tensor(X):
            if X.ndim == 3:  # [B,T,F]
                X_list = [x for x in X]
            else:  # [T,F] or [F]
                X_list = [X]
        else:
            X_list = list(X)
        B = len(X_list)

        lengths = [seq.shape[0] if seq.ndim > 1 else 1 for seq in X_list]
        max_len = max(lengths)
        F = X_list[0].shape[1] if X_list[0].ndim > 1 else X_list[0].shape[0]

        # --- Stack sequences into padded tensor ---
        seq_tensor = torch.zeros(B, max_len, F, device=device, dtype=DTYPE)
        mask = torch.zeros(B, max_len, device=device, dtype=DTYPE)
        for b, seq in enumerate(X_list):
            seq = seq.to(device=device, dtype=DTYPE)
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            T = seq.shape[0]
            seq_tensor[b, :T] = seq
            mask[b, :T] = 1.0

        # --- Normalize context ---
        if theta is not None:
            if torch.is_tensor(theta):
                if theta.ndim == 1:  # [D]
                    theta_tensor = theta.unsqueeze(0).expand(B, max_len, -1)
                elif theta.ndim == 2:  # [T,D] -> broadcast to batch
                    theta_tensor = torch.stack([theta.expand(L, -1) for L in lengths], dim=0)
                elif theta.ndim == 3:  # [B,T,D]
                    theta_tensor = theta
                else:
                    raise ValueError(f"Unsupported theta shape {theta.shape}")
            elif isinstance(theta, list):
                if len(theta) != B:
                    raise ValueError(f"Context list length {len(theta)} != number of sequences {B}")
                theta_tensor = torch.zeros(B, max_len, theta[0].shape[-1], device=device, dtype=DTYPE)
                for b, ctx in enumerate(theta):
                    T = ctx.shape[0]
                    theta_tensor[b, :T] = ctx.to(device=device, dtype=DTYPE)
            else:
                raise TypeError(f"Unsupported theta type: {type(theta)}")
        else:
            theta_tensor = None

        # --- Compute emission log-probs ---
        emission_pdf = self._params.get("emission_pdf")
        if isinstance(emission_pdf, torch.distributions.MultivariateNormal):
            seq_exp = seq_tensor.unsqueeze(2)  # [B,T,1,F]
            log_probs_full = emission_pdf.log_prob(seq_exp)  # [B,T,K]
        elif isinstance(emission_pdf, torch.distributions.Categorical):
            seq_cat = seq_tensor[..., 0].long()  # [B,T]
            logit_probs = F.log_softmax(emission_pdf.logits, dim=-1)  # [K, n_classes]
            one_hot = F.one_hot(seq_cat, num_classes=logit_probs.shape[-1]).float()  # [B,T,n_classes]
            log_probs_full = torch.einsum('btn,kn->btk', one_hot, logit_probs)  # [B,T,K]
        else:
            raise TypeError(f"Unsupported emission_pdf type: {type(emission_pdf)}")

        # --- Split padded tensor back into per-sequence lists ---
        sequences = [seq_tensor[b, :lengths[b]] for b in range(B)]
        log_probs_list = [log_probs_full[b, :lengths[b]] for b in range(B)]
        contexts = [theta_tensor[b, :lengths[b]] if theta_tensor is not None else getattr(self, "_context", None)
                    for b in range(B)]

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=lengths,
            context=contexts
        )

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


    # hsmm.py HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        """
        Vectorized Forward algorithm for multiple sequences with optimized duration handling.
        Returns a list of tensors [T, K, Dmax] for each sequence.
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Use parameters from _params ---
        init_logits = self._params["initial_pdf"].logits.to(device=device, dtype=DTYPE)     # [K]
        dur_logits = self._params["duration_pdf"].logits.to(device=device, dtype=DTYPE)     # [K, Dmax] or [Dmax]
        trans_logits = self._params["transition_pdf"].logits.to(device=device, dtype=DTYPE) # [K, K]

        # --- Safe normalization ---
        def _safe_log_rows(logits: torch.Tensor, dim=-1, min_prob=1e-6):
            probs = logits.exp().clamp_min(min_prob)
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            return probs.log()

        init_logits = _safe_log_rows(init_logits, dim=0)
        dur_logits = _safe_log_rows(dur_logits, dim=-1)
        trans_logits = _safe_log_rows(trans_logits, dim=-1)

        alpha_list: list[torch.Tensor] = []

        for log_emissions, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            log_emissions = log_emissions.to(device=device, dtype=DTYPE)  # [T, K]
            T = seq_len

            # --- Precompute cumulative emission sums for all durations ---
            cumsum_emit = torch.cat([torch.zeros(1, K, device=device, dtype=DTYPE),
                                     torch.cumsum(log_emissions, dim=0)], dim=0)  # [T+1, K]

            # --- DP table ---
            alpha_tensor = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, device=device)  # [max_d]
                starts = t - durations + 1                             # [max_d]

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
        Backward algorithm for HSMM using stored _params.
        Returns list of log-beta tensors per sequence: [T, K, Dmax].
        Preserves original logic with clearer naming and safe bounds.
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Fetch parameters from _params ---
        init_logits = self._params["initial_pdf"].logits.to(device=device, dtype=DTYPE)     # [K]
        dur_logits = self._params["duration_pdf"].logits.to(device=device, dtype=DTYPE)     # [K, Dmax] or [Dmax]
        trans_logits = self._params["transition_pdf"].logits.to(device=device, dtype=DTYPE) # [K, K]

        # Safe normalization
        dur_logits = torch.log(F.softmax(dur_logits, dim=-1).clamp_min(EPS))
        trans_logits = torch.log(F.softmax(trans_logits, dim=-1).clamp_min(EPS))

        beta_list = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)  # [T, K]
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1, :, 0] = 0.0  # terminal step for duration=1

            # Precompute cumulative emission sums for efficiency
            cumsum_emit = torch.cat([torch.zeros(1, K, device=device, dtype=DTYPE),
                                     torch.cumsum(seq_logp, dim=0)], dim=0)  # [T+1, K]

            for t in reversed(range(seq_len - 1)):
                max_dur = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dur + 1, device=device)
                ends = t + durations  # segment end indices

                # Emission sums per duration
                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0)).T  # [K, max_dur]

                # Duration logits
                dur_scores = dur_logits[:, :max_dur]  # [K, max_dur]

                # Contribution from next segments
                next_beta = log_beta[ends - 1, :, 0]  # [max_dur, K]
                beta_next = torch.logsumexp(trans_logits.unsqueeze(0) + next_beta.unsqueeze(1), dim=2).T  # [K, max_dur]

                # Combine emission, duration, and next-segment contributions
                segment_scores = emit_sums + dur_scores + beta_next
                log_beta[t, :, 0] = torch.logsumexp(segment_scores, dim=1)

                # Within-segment propagation
                if max_dur > 1:
                    shift_len = min(max_dur - 1, seq_len - t - 1)
                    if shift_len > 0:
                        log_beta[t, :, 1:shift_len + 1] = log_beta[t + 1, :, :shift_len] + seq_logp[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(
        self, 
        X: utils.Observations, 
        theta: Optional[ContextualVariables] = None
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Computes posterior distributions for HSMM:
            - gamma_vec: per-time step state marginal [T, K]
            - xi_vec: per-time step state-to-state joint [T-1, K, K]
            - eta_vec: per-time step state-duration joint [T, K, Dmax]

        Supports variable-length sequences and optional contextual variables.
        """
        neg_inf = torch.finfo(DTYPE).min / 2.0
        K, Dmax, device = self.n_states, self.max_duration, self.device
        B = len(X.sequence)

        init_logits = self._params["initial_pdf"].logits.to(device=device, dtype=DTYPE)
        trans_logits = self._params["transition_pdf"].logits.to(device=device, dtype=DTYPE)
        dur_logits = self._params["duration_pdf"].logits.to(device=device, dtype=DTYPE)

        trans_logits = torch.log_softmax(trans_logits, dim=-1)
        dur_logits = torch.log_softmax(dur_logits, dim=-1)

        # --- Group sequences by length for efficient semi-batching ---
        length_to_indices: dict[int, list[int]] = {}
        for b, L in enumerate(X.lengths):
            length_to_indices.setdefault(L, []).append(b)

        gamma_vec: list[torch.Tensor] = [None] * B
        xi_vec: list[torch.Tensor] = [None] * B
        eta_vec: list[torch.Tensor] = [None] * B

        for L, batch_indices in length_to_indices.items():
            B_l = len(batch_indices)

            # --- Collect log-probs for batch ---
            batch_logp = torch.stack([X.log_probs[b].to(device=device, dtype=DTYPE) for b in batch_indices], dim=0)  # [B_l, T, K]

            # --- Forward ---
            alpha_batch = torch.zeros(B_l, L, K, Dmax, device=device, dtype=DTYPE)
            for i in range(B_l):
                obs_i = utils.Observations(sequence=[batch_logp[i]], log_probs=[batch_logp[i]], lengths=[L])
                alpha_batch[i] = self._forward(obs_i, theta)[0]

            # --- Backward ---
            beta_batch = torch.zeros(B_l, L, K, Dmax, device=device, dtype=DTYPE)
            for i in range(B_l):
                obs_i = utils.Observations(sequence=[batch_logp[i]], log_probs=[batch_logp[i]], lengths=[L])
                beta_batch[i] = self._backward(obs_i, theta)[0]

            # --- Compute η (state-duration posterior) ---
            eta_batch = alpha_batch + beta_batch  # [B_l, T, K, Dmax]
            eta_flat = eta_batch.view(B_l, L, -1)
            eta_soft = F.softmax(eta_flat, dim=-1).view(B_l, L, K, Dmax)

            # --- Compute γ (state marginal) ---
            gamma_batch = eta_soft.sum(dim=-1)
            gamma_batch = gamma_batch / gamma_batch.sum(dim=-1, keepdim=True).clamp_min(EPS)

            # --- Compute ξ (state-to-state transitions) ---
            xi_batch = []
            for i in range(B_l):
                if L <= 1:
                    xi_batch.append(torch.zeros((0, K, K), device=device, dtype=DTYPE))
                    continue

                alpha_prev = torch.logsumexp(alpha_batch[i][:-1], dim=2)  # [T-1, K]
                beta_next = torch.logsumexp(beta_batch[i][1:], dim=2)     # [T-1, K]
                log_xi_i = alpha_prev.unsqueeze(2) + trans_logits.unsqueeze(0) + beta_next.unsqueeze(1)
                xi_i_flat = log_xi_i.reshape(L-1, -1)
                xi_i = F.softmax(xi_i_flat, dim=1).reshape(L-1, K, K).clamp_min(EPS)
                xi_batch.append(xi_i)

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
        Viterbi decoding for HSMM using contextual parameters.
        Uses _params distributions for initial, transition, duration, and emission.
        """
        device = self.device
        neg_inf = torch.finfo(DTYPE).min / 2
        seq = X.sequence[0].to(device, DTYPE)
        L, K, Dmax = seq.shape[0], self.n_states, self.max_duration
        if L == 0:
            return torch.empty(0, dtype=torch.int64, device=device)

        # --- Extract distributions from _params ---
        init_logits = self._params["initial_pdf"].logits        # [K]
        trans_logits = self._params["transition_pdf"].logits   # [K, K]
        dur_logits = self._params["duration_pdf"].logits       # [K, Dmax]
        emission_pdf = self._params.get("emission_pdf")        # Distribution

        # --- Optional duration weighting ---
        if duration_weight > 0:
            dur_indices = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE).unsqueeze(0)  # [1, Dmax]
            dur_mean = (dur_logits.exp() * dur_indices).sum(dim=1, keepdim=True)               # [K,1]
            dur_penalty = -((dur_indices - dur_mean) ** 2) / (2 * (Dmax / 3) ** 2)             # [K, Dmax]
            dur_logits = (1 - duration_weight) * dur_logits + duration_weight * dur_penalty
            dur_logits = torch.clamp(dur_logits, min=-MAX_LOGITS, max=MAX_LOGITS)

        # --- Compute emission log-probs ---
        if emission_pdf is not None:
            if isinstance(emission_pdf, torch.distributions.MultivariateNormal):
                seq_exp = seq.unsqueeze(1)          # [L, 1, D]
                emit_log = emission_pdf.log_prob(seq_exp)   # [L, K]
            elif isinstance(emission_pdf, torch.distributions.Categorical):
                seq_cat = seq.long() if seq.ndim == 1 else seq[:, 0].long()
                log_probs_all = F.log_softmax(emission_pdf.logits, dim=-1)  # [K, n_classes]
                emit_log = (F.one_hot(seq_cat, num_classes=log_probs_all.shape[-1])
                            .float()
                            .unsqueeze(1) * log_probs_all.unsqueeze(0)).sum(dim=-1)  # [L, K]
            else:
                emit_log = X.log_probs[0].to(device, DTYPE)
        else:
            emit_log = X.log_probs[0].to(device, DTYPE)

        # --- Cumulative sums for duration-efficient emission sums ---
        cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=DTYPE),
                                    torch.cumsum(emit_log, dim=0)))  # [L+1, K]

        # --- DP tables ---
        V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
        back_ptr = torch.full((L, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((L, K), dtype=torch.int64, device=device)

        durations_full = torch.arange(1, Dmax + 1, device=device)  # [Dmax]

        for t in range(L):
            max_d = min(Dmax, t + 1)
            durations = durations_full[:max_d]          # [max_d]
            starts = t - durations + 1                  # [max_d]

            # --- Compute emission sums for candidate durations ---
            emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).permute(1, 0)  # [K, max_d]

            # --- Duration scores ---
            dur_scores = dur_logits[:, :max_d] if dur_logits.ndim == 2 else dur_logits[:max_d].unsqueeze(0).expand(K, -1)

            if t == 0:
                scores = init_logits.unsqueeze(1) + dur_scores + emit_sums
                best_score, best_idx = scores.max(dim=1)
                best_dur[t] = durations[best_idx]
                V[t] = best_score
                back_ptr[t] = -1
            else:
                prev_scores_base = torch.full((max_d, K), neg_inf, device=device, dtype=DTYPE)
                valid_mask = starts > 0
                if valid_mask.any():
                    prev_scores_base[valid_mask] = V[starts[valid_mask] - 1]

                prev_scores = prev_scores_base.unsqueeze(2) + trans_logits.unsqueeze(0)  # [max_d, K, K]
                prev_max, prev_arg = prev_scores.max(dim=1)  # [max_d, K]

                scores = prev_max.permute(1, 0) + dur_scores + emit_sums
                best_score, best_d_idx = scores.max(dim=1)
                V[t] = best_score
                best_dur[t] = durations[best_d_idx]

                idx_range = torch.arange(K, device=device)
                prev_arg_selected = prev_arg[best_d_idx, idx_range]
                chosen_d_valid = valid_mask[best_d_idx]
                back_ptr[t] = torch.where(chosen_d_valid, prev_arg_selected.to(torch.int64),
                                          torch.full_like(prev_arg_selected, -1, dtype=torch.int64))

        # --- Backtrace ---
        t = L - 1
        cur_state = int(torch.argmax(V[t]).item())
        segments = []

        while t >= 0:
            d = int(best_dur[t, cur_state].item())
            start = max(0, t - d + 1)
            segments.append((start, t, cur_state))
            prev_state = int(back_ptr[t, cur_state].item())
            t = start - 1
            cur_state = prev_state if prev_state >= 0 else cur_state

        segments.reverse()
        seq_path = torch.cat([torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                              for start, end, st in segments])

        return seq_path[:L]

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
        # --- Prepare observations ---
        X = X.to(self.device, dtype=DTYPE) if torch.is_tensor(X) else X
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)
        X_valid = self.to_observations(X, theta=theta)
        B = len(X_valid.sequence)
        max_len = max(X_valid.lengths)
        total_len = sum(X_valid.lengths)

        # --- Vectorized theta alignment ---
        aligned_theta = None
        if theta is not None:
            if theta.ndim == 1:
                aligned_theta = theta.unsqueeze(0).expand(total_len, -1)
            elif theta.shape[0] == B:
                repeats = torch.tensor(X_valid.lengths, device=theta.device)
                aligned_theta = torch.repeat_interleave(theta, repeats, dim=0)
            elif theta.shape[0] == total_len:
                aligned_theta = theta
            else:
                raise ValueError(f"Cannot align theta of shape {tuple(theta.shape)}")
            aligned_theta = aligned_theta.to(self.device, dtype=DTYPE)

        # --- Convergence monitor ---
        self.conv = ConvergenceMonitor(
            tol=tol, max_iter=max_iter, n_init=n_init,
            post_conv_iter=post_conv_iter, verbose=verbose
        )

        best_score, best_state = -float("inf"), None

        # --- Prebuild padded tensors for batching ---
        seq_tensor = torch.zeros((B, max_len, X_valid.sequence[0].shape[-1]),
                                 device=self.device, dtype=DTYPE)
        mask = torch.zeros(B, max_len, device=self.device, dtype=DTYPE)
        lengths_tensor = torch.tensor(X_valid.lengths, device=self.device, dtype=torch.long)
        for b, seq in enumerate(X_valid.sequence):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq.to(self.device, dtype=DTYPE)
            mask[b, :L] = 1.0
        seq_exp_base = seq_tensor.unsqueeze(2)  # [B, T, 1, D]

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # --- Initialize / sample parameters ---
            mode = "sample" if run_idx > 0 or sample_D_from_X else "estimate"
            params = self.get_model_params(X_valid, theta=aligned_theta, mode=mode)

            init_pdf = params["initial_pdf"]
            transition_pdf = params["transition_pdf"]
            duration_pdf = params["duration_pdf"]
            emission_pdf = params["emission_pdf"]

            # --- Compute emission log-probs ---
            seq_exp = seq_exp_base.expand(-1, -1, self.n_states, -1)
            log_probs_full = emission_pdf.log_prob(seq_exp) / seq_exp.shape[-1]

            # Vectorized slicing to per-sequence lengths
            mask_exp = mask.unsqueeze(2)  # [B, T, 1]
            X_valid.log_probs = log_probs_full * mask_exp  # masked log-probs

            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # --- E-step ---
                gamma_list, xi_list, eta_list = self._compute_posteriors(X_valid, theta=aligned_theta)

                # --- Vectorized accumulation ---
                gamma_tensor = torch.cat([g for g in gamma_list if g.numel() > 0], dim=0)
                xi_tensor = torch.cat([x for x in xi_list if x is not None], dim=0) if xi_list else None
                eta_tensor = torch.cat([e for e in eta_list if e is not None], dim=0) if eta_list else None

                init_counts = gamma_tensor.sum(0)
                trans_counts = xi_tensor.sum(0) if xi_tensor is not None else None
                dur_counts = eta_tensor.sum(0) if eta_tensor is not None else None

                # Fallbacks
                init_counts = init_counts if (init_counts >= EPS).all() else init_pdf.probs
                trans_counts = trans_counts if (trans_counts.sum(-1) >= EPS).all() else transition_pdf.probs
                dur_counts = dur_counts if (dur_counts.sum(-1) >= EPS).all() else duration_pdf.probs

                # --- Blend with prior ---
                α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
                α = float(max(α_min, α_max * (1.0 - it / max_iter)))
                init_probs = α * init_counts + (1 - α) * init_pdf.probs
                trans_probs = α * trans_counts + (1 - α) * transition_pdf.probs
                dur_probs = α * dur_counts + (1 - α) * duration_pdf.probs

                init_pdf = Categorical(probs=init_probs)
                transition_pdf = Categorical(probs=trans_probs)
                duration_pdf = Categorical(probs=dur_probs)

                # --- Update emissions ---
                emission_pdf = self.emission_module.initialize(
                    X=seq_tensor.view(-1, seq_tensor.shape[-1]),
                    posterior=gamma_tensor,
                    theta=aligned_theta,
                    context=aligned_theta
                )
                log_probs_full = emission_pdf.log_prob(seq_exp) / seq_exp.shape[-1]
                X_valid.log_probs = log_probs_full * mask_exp

                # --- Convergence check ---
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # --- Track best model (clone tensors instead of deepcopy) ---
            run_score = float(curr_ll.item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    "initial_pdf": Categorical(probs=init_pdf.probs.clone()),
                    "transition_pdf": Categorical(probs=transition_pdf.probs.clone()),
                    "duration_pdf": Categorical(probs=duration_pdf.probs.clone()),
                }
                if isinstance(emission_pdf, torch.distributions.Categorical):
                    best_state["emission_pdf"] = Categorical(probs=emission_pdf.probs.clone())
                elif isinstance(emission_pdf, torch.distributions.MultivariateNormal):
                    best_state["emission_pdf"] = torch.distributions.MultivariateNormal(
                        loc=emission_pdf.loc.clone(),
                        covariance_matrix=emission_pdf.covariance_matrix.clone()
                    )
                else:
                    best_state["emission_pdf"] = emission_pdf  # fallback

        # --- Restore best parameters ---
        if best_state:
            self._params.update(best_state)

        if plot_conv and hasattr(self, "conv"):
            self.conv.plot_convergence()

        return self

    def predict(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        Predict hidden states for one or multiple sequences using MAP or Viterbi decoding.
        Relies on `_params` distributions for consistent emissions, transitions, and durations.
        """
        if torch.is_tensor(X):
            X = [X]

        B = len(X)
        preds: List[torch.Tensor] = []

        # --- Handle context ---
        if context is not None and torch.is_tensor(context):
            cum_lengths = [0] + list(torch.cumsum(torch.tensor([s.shape[0] for s in X]), dim=0).cpu().numpy())
            context_splits = [context[cum_lengths[b]:cum_lengths[b+1]] for b in range(B)]
        elif isinstance(context, list):
            context_splits = context
        else:
            context_splits = [None] * B

        # Clamp temperatures
        temp_transition = max(temp_transition, 1e-6)
        temp_duration = max(temp_duration, 1e-6)

        # --- Get PDFs from trained parameters ---
        init_pdf = self._params["initial_pdf"]
        trans_pdf = self._params["transition_pdf"]
        dur_pdf = self._params["duration_pdf"]
        emission_pdf = self._params.get("emission_pdf")

        # Scale logits
        trans_logits = trans_pdf.logits / temp_transition
        dur_logits = dur_pdf.logits / temp_duration

        for b, seq in enumerate(X):
            seq = seq.to(self.device, DTYPE)
            T = seq.shape[0]
            ctx = None if context_splits[b] is None else context_splits[b].to(self.device, DTYPE)

            if T == 0:
                preds.append(torch.empty(0, dtype=torch.int64, device="cpu"))
                continue

            # --- Emission log-probs ---
            if emission_pdf is not None:
                if isinstance(emission_pdf, torch.distributions.MultivariateNormal):
                    seq_exp = seq.unsqueeze(1) if seq.ndim == 2 else seq  # [T, 1, D]
                    log_probs = emission_pdf.log_prob(seq_exp)           # [T, K]
                elif isinstance(emission_pdf, torch.distributions.Categorical):
                    seq_cat = seq.long() if seq.ndim == 1 else seq[:, 0].long()
                    log_probs_all = F.log_softmax(emission_pdf.logits, dim=-1)  # [K, n_classes]
                    log_probs = (F.one_hot(seq_cat, num_classes=log_probs_all.shape[-1])
                                 .float().unsqueeze(1) * log_probs_all.unsqueeze(0)).sum(dim=-1)
                else:
                    log_probs = seq.to(self.device, DTYPE)
            else:
                log_probs = seq.to(self.device, DTYPE)

            log_probs = log_probs.view(T, self.n_states)
            obs = utils.Observations([seq], log_probs=[log_probs])

            # --- Decoding ---
            if algorithm.lower() == "viterbi":
                decoded = self._viterbi(obs, theta=ctx)
            elif algorithm.lower() == "map":
                K = self.n_states
                delta = torch.full((T, K), -torch.inf, device=self.device, dtype=DTYPE)
                psi = torch.full((T, K), -1, dtype=torch.int64, device=self.device)
                delta[0] = log_probs[0]

                Dmax = dur_logits.shape[-1]
                for t in range(1, T):
                    dur_term = dur_logits[:, min(t, Dmax - 1)] if dur_logits.ndim == 2 else dur_logits[min(t, Dmax - 1)]
                    scores = delta[t - 1].unsqueeze(1) + trans_logits + dur_term.unsqueeze(0)
                    psi[t] = torch.argmax(scores, dim=0)
                    delta[t] = torch.max(scores, dim=0).values + log_probs[t]

                decoded = torch.zeros(T, dtype=torch.int64, device=self.device)
                decoded[-1] = torch.argmax(delta[-1])
                for t in reversed(range(T - 1)):
                    decoded[t] = psi[t + 1, decoded[t + 1]]
            else:
                raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

            preds.append(decoded.detach().cpu())

        return preds

    @torch.no_grad()
    def _compute_log_likelihood(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Vectorized computation of per-sequence log-likelihoods: log P(X | model, theta).
        Supports variable-length sequences and context modulation.
        Returns a tensor of shape [B] containing log-likelihoods for each sequence.
        """
        device = self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0
        B = len(X.sequence)

        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, device=device, dtype=DTYPE)
            return X.log_likelihoods

        # --- Align context across sequences ---
        total_len = sum(X.lengths)
        aligned_theta: Optional[torch.Tensor] = None
        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            if theta.ndim == 1:
                aligned_theta = theta.unsqueeze(0).expand(total_len, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == B:  # per-sequence context
                    aligned_theta = torch.cat([theta[b].expand(L, -1) for b, L in enumerate(X.lengths)], dim=0)
                elif theta.shape[0] == total_len:
                    aligned_theta = theta
                else:
                    raise ValueError(f"Cannot align theta of shape {tuple(theta.shape)} for B={B}")
            elif theta.ndim == 3 and theta.shape[0] == B:
                aligned_theta = torch.cat([theta[b, :L] for b, L in enumerate(X.lengths)], dim=0)
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape} for B={B}")

        # --- Forward pass ---
        alpha_list = self._forward(X, theta=aligned_theta)

        # --- Pad alpha for batching ---
        T_max = max(X.lengths)
        K, Dmax = self.n_states, self.max_duration
        alpha_pad = torch.full((B, T_max, K, Dmax), neg_inf, device=device, dtype=DTYPE)

        for b, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L == 0:
                continue
            t_slice = slice(0, L)
            if alpha.ndim == 3:  # [T, K, D]
                alpha_pad[b, t_slice, :, :] = alpha[:L]
            else:  # fallback [T, K]
                alpha_pad[b, t_slice, :, 0] = alpha[:L]

        # --- Apply optional sequence masks ---
        if hasattr(X, "masks") and X.masks is not None:
            mask_pad = torch.zeros(B, T_max, device=device, dtype=DTYPE)
            for b, L in enumerate(X.lengths):
                mask_pad[b, :L] = X.masks[b].to(device=device, dtype=DTYPE)
            alpha_pad += torch.log(mask_pad.unsqueeze(-1).unsqueeze(-1) + EPS)

        # --- Compute log-likelihoods per sequence ---
        ll_pad = torch.logsumexp(alpha_pad, dim=(2, 3))  # sum over states and durations
        ll_list = [ll_pad[b, :X.lengths[b]].max().clone() for b in range(B)]
        X.log_likelihoods = torch.stack(ll_list)

        # --- Verbose logging ---
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
        Compute log-likelihoods for one or more sequences.

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

        if B == 0:
            return torch.tensor([], device="cpu", dtype=DTYPE)

        ll_list = []

        for b, seq in enumerate(X):
            seq = seq.to(dtype=DTYPE, device=self.device)
            T = seq.shape[0]
            if T == 0:
                ll_list.append(torch.tensor(neg_inf, device="cpu", dtype=DTYPE))
                continue

            # --- Emission log-probs ---
            pdf = self._params.get("emission_pdf")
            if pdf is not None and hasattr(pdf, "log_prob"):
                x_input = seq.unsqueeze(1) if seq.ndim == 2 else seq
                log_B = pdf.log_prob(x_input)
                if log_B.ndim > 2:
                    log_B = log_B.flatten(start_dim=1).sum(dim=1)
                if log_B.ndim == 1:
                    log_B = log_B.unsqueeze(-1)
            elif hasattr(self, "emission_module"):
                output = self.emission_module(seq.unsqueeze(0) if seq.ndim == 2 else seq)
                log_B = output.squeeze(0) if output.ndim > 2 else output
            else:
                raise RuntimeError("No emission PDF or module initialized for scoring.")

            # --- Module logits ---
            init_logits = self.initial_module.log_matrix(context=theta).to(self.device, DTYPE)
            dur_logits = self.duration_module.log_matrix(context=theta).to(self.device, DTYPE)
            trans_logits = self.transition_module.log_matrix(context=theta).to(self.device, DTYPE)

            # Temperature scaling
            trans_logits = trans_logits / max(temp_transition, self.min_covar)
            dur_logits = dur_logits / max(temp_duration, self.min_covar)

            # --- Forward DP ---
            V = torch.full((T, K), neg_inf, device=self.device, dtype=DTYPE)
            V[0] = log_B[0] + torch.log_softmax(init_logits, dim=0)

            for t in range(1, T):
                max_d = min(Dmax, t + 1)
                dur_idx = torch.arange(max_d, device=self.device)
                emit_sums = torch.stack([log_B[t - d + 1:t + 1].sum(dim=0) for d in dur_idx], dim=0)

                if dur_logits.ndim == 2:
                    dur_scores = dur_logits[:, :max_d].T  # [max_d, K]
                else:
                    dur_scores = dur_logits[:max_d].unsqueeze(1).expand(-1, K)

                scores = []
                for i, d in enumerate(dur_idx):
                    start = t - d + 1
                    if start == 0:
                        trans_score = torch.zeros(K, device=self.device, dtype=DTYPE)
                    else:
                        trans_score = torch.logsumexp(V[start - 1].unsqueeze(1) + trans_logits, dim=0)
                    scores.append(trans_score + dur_scores[i] + emit_sums[i])

                V[t] = torch.logsumexp(torch.stack(scores, dim=0), dim=0)

            ll_seq = torch.logsumexp(V[-1], dim=-1).detach().cpu()
            ll_list.append(ll_seq)

        return torch.stack(ll_list)

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

