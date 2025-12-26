# nhsmm/models/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.constants import DEBUG, DTYPE, EPS, logger, MAX_LOGITS, NEG_INF
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm import constraints, ConvergenceTracker

# torch.autograd.set_detect_anomaly(True)

class HSMM(nn.Module, ABC):

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        min_covar: float = 1e-6,
        temperature: float = 1.0,
        modulate_var: bool = False,
        emission_type: str = "gaussian",
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        transition_type: Any = constraints.Transitions.ERGODIC,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        seed: Optional[int] = None, debug: bool = False):

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._params: Dict[str, Any] = {}

        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.debug = debug
        self.n_states = n_states
        self.n_features = n_features
        self.temperature = temperature
        self.max_duration = max_duration
        self.emission_type = emission_type

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
                    encoder=encoder,
                    pool=pool,
                    n_heads=n_heads,
                    dropout=dropout,
                    debug=debug
                ).to(device=self.device, dtype=DTYPE)
            )

            # --- Infer context dimension safely ---
            self.encoder.eval()
            try:
                dummy = torch.zeros(1, 16, n_features, device=self.device, dtype=DTYPE)
                try:
                    _, ctx, _ = self.encoder(dummy, return_context=True, return_sequence=True)
                except TypeError:
                    ctx = None
                self.context_dim = ctx.shape[-1] if ctx is not None else self.encoder(dummy).shape[-1]
                self.hidden_dim = self.context_dim if hidden_dim is None else hidden_dim
            finally:
                self.encoder.train()

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


    def _init_modules(self, # ToDo init externaly like encoder then pass
        transition_type: str,
        emission_type: str,
        modulate_var: bool,
        max_duration: int,
        temperature: float,
        min_covar: float):

        device, debug = self.device, self.debug

        self.initial_module = Initial(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim
        )
        self.duration_module = Duration(
            n_states=self.n_states,
            max_duration=max_duration,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature
        )
        self.transition_module = Transition(
            n_states=self.n_states,
            n_features=self.n_features,
            transition_type=transition_type,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature
        )
        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            emission_type=emission_type,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            modulate_var=modulate_var,
            temperature=temperature,
            min_covar=min_covar
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

        if debug:
            logger.debug(
                f"HSMM initialized on {device}: n_states={self.n_states}, "
                f"n_features={self.n_features}, context_dim={self.context_dim}, "
                f"emission={self.emission_type}, max_duration={self.max_duration}"
            )


    def _prepare(self,
        X: torch.Tensor | list,
        theta: Optional[torch.Tensor | list] = None,
        mask: Optional[torch.BoolTensor] = None) -> SequenceSet:

        device = X[0].device if isinstance(X, list) else X.device

        if isinstance(X, list):
            X = torch.nn.utils.rnn.pad_sequence(
                [x.to(dtype=DTYPE, device=device) for x in X],
                batch_first=True
            )
        if isinstance(theta, list):
            theta = torch.nn.utils.rnn.pad_sequence(
                [t.to(dtype=DTYPE, device=device) for t in theta],
                batch_first=True
            )

        # --- ensure batch dimension ---
        if X.ndim == 2: X = X.unsqueeze(0)
        B, T, F = X.shape
        K = self.n_states

        if mask is None:
            mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device).bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            elif mask.ndim != 3:
                mask = mask.view(B, T, 1)
        lengths = mask.squeeze(-1).sum(dim=1).to(torch.long)

        if theta is not None:
            # broadcast theta to [B,T,H]
            if theta.ndim == 2:
                theta = theta.unsqueeze(1).expand(B, T, -1)
            context = theta
            canonical = theta[:, :1, :]
        else:
            # use encoder to compute context + canonical consistently
            context, canonical = self.encoder.encode(
                sequences=X,
                mask=mask.squeeze(-1),
                detach=False
            )

        if T == 0:
            log_probs = torch.empty(B, 0, K, device=device, dtype=DTYPE)
        else:
            dist = self.emission_module.forward(context=context, return_dist=True)
            X_exp = X.unsqueeze(2).expand(B, T, K, F) # expand X for broadcasting
            log_probs = dist.log_prob(X_exp)
            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        if getattr(self, "debug", False):
            print(f"[Prepare] X={X.shape}, context={context.shape}, canonical={canonical.shape}, log_probs={log_probs.shape}")

        return SequenceSet(
            sequences=X,
            lengths=lengths,
            masks=mask,
            contexts=context,
            canonical=canonical,
            log_probs=log_probs
        )

    def _forward(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> torch.Tensor:
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta
        B, T, K = router.log_probs.shape[:3]
        Dmax = self.max_duration
        device = router.context.device
        dtype = router.context.dtype

        initial_logits = self.initial_module.log_matrix(context=router.canonical)  # [B,1,K]
        duration_logits = self.duration_module.log_matrix(context=router.context)  # [B,T,K,Dmax]
        transition_logits = self.transition_module.log_matrix(context=router.context)  # [B,T,K,K]

        # cumsum_emit: [B, T+1, K]
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=DTYPE)
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)
        d_range = torch.arange(1, Dmax + 1, device=device)  # [Dmax]
        t_range = torch.arange(T, device=device).unsqueeze(1)  # [T,1]
        start_idx = (t_range - d_range + 1).clamp(min=0)  # [T,Dmax]
        end_idx = (t_range + 1).expand(-1, Dmax)  # [T,Dmax]

        # Add batch and state dimensions: B x T x K x Dmax
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, T, K, Dmax)
        k_idx = torch.arange(K, device=device).view(1, 1, K, 1).expand(B, T, K, Dmax)
        t_idx_start = start_idx.view(1, T, 1, Dmax).expand(B, T, K, Dmax)
        t_idx_end = end_idx.view(1, T, 1, Dmax).expand(B, T, K, Dmax)
        emit_sums = cumsum_emit[b_idx, t_idx_end, k_idx] - cumsum_emit[b_idx, t_idx_start, k_idx]  # [B,T,K,Dmax]

        # --- Initialize alpha ---
        alpha = torch.full((B, T, K, Dmax), NEG_INF, device=device, dtype=DTYPE)
        alpha0 = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]
        alpha[:, 0, :, 0] = alpha0  # safe because alpha0 is fresh, not part of graph yet

        # --- Recursion over time ---
        for t in range(1, T):
            max_d = min(Dmax, t + 1)

            # Gather previous alpha contributions for all valid durations
            prev_alpha_vals = []
            mask_valid = (t < X.lengths).view(B, 1)  # [B,1]
            for d in range(1, max_d + 1):
                t_prev = t - d
                if t_prev >= 0:
                    # Clone slice to prevent inplace modification issues
                    prev_slice = alpha[:, t_prev, :, :d].clone()  # [B, K, d]
                    val = torch.logsumexp(prev_slice, dim=-1)     # [B, K]
                else:
                    val = initial_logits[:, 0, :] + duration_logits[:, 0, :, d - 1]  # [B,K]

                val = val.view(B, K)
                val = torch.where(
                    mask_valid,
                    val,
                    torch.full((B, K), NEG_INF, device=device, dtype=DTYPE)
                )
                prev_alpha_vals.append(val)

            prev_alpha = torch.stack(prev_alpha_vals, dim=1)  # [B, max_d, K]

            # Transition & duration update
            alpha_trans = torch.logsumexp(
                prev_alpha.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1),
                dim=2
            ).transpose(1, 2)  # [B, K, max_d]

            duration_logits_t = duration_logits[:, t, :, :max_d]  # [B, K, max_d]
            alpha_t = alpha_trans + duration_logits_t + emit_sums[:, t, :, :max_d]  # [B,K,max_d]

            # Assign fully computed alpha for this timestep
            alpha[:, t, :, :max_d] = alpha_t

            # Mask out invalid durations > t
            invalid_mask = torch.arange(Dmax, device=device).view(1, 1, -1) > t
            if invalid_mask.any():
                alpha[:, t, :, :] = torch.where(
                    invalid_mask,
                    torch.full_like(alpha[:, t, :, :], NEG_INF),
                    alpha[:, t, :, :]
                )

        # --- Invalidate timesteps beyond sequence length ---
        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)  # [B,T]
        alpha = torch.where(length_mask.unsqueeze(-1).unsqueeze(-1), alpha, torch.full_like(alpha, NEG_INF))

        # --- Apply mask over padded timesteps ---
        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        alpha = torch.where(mask_exp, alpha, torch.full_like(alpha, NEG_INF))
        return alpha

    def _backward(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> torch.Tensor:
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta

        Dmax = self.max_duration
        dtype = router.context.dtype
        B, T, K = router.log_probs.shape
        device = router.context.device

        initial_logits = self.initial_module.log_matrix(context=router.canonical)  # [B,1,K]
        duration_logits = self.duration_module.log_matrix(context=router.context)  # [B,T,K,Dmax]
        transition_logits = self.transition_module.log_matrix(context=router.context)  # [B,T,K,K]

        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=DTYPE) # cumsum_emit: [B, T+1, K]
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)
        d_range = torch.arange(1, Dmax + 1, device=device)  # [Dmax]
        t_range = torch.arange(T, device=device).unsqueeze(1)  # [T,1]
        start_idx = (t_range - d_range + 1).clamp(min=0)  # [T,Dmax]
        end_idx = (t_range + 1).expand(-1, Dmax)  # [T,Dmax]

        # Add batch and state dimensions: B x T x K x Dmax
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, T, K, Dmax)
        k_idx = torch.arange(K, device=device).view(1, 1, K, 1).expand(B, T, K, Dmax)
        t_idx_start = start_idx.view(1, T, 1, Dmax).expand(B, T, K, Dmax)
        t_idx_end = end_idx.view(1, T, 1, Dmax).expand(B, T, K, Dmax)
        emit_sums = cumsum_emit[b_idx, t_idx_end, k_idx] - cumsum_emit[b_idx, t_idx_start, k_idx]  # [B,T,K,Dmax]

        # --- Initialize beta ---
        beta = torch.full((B, T, K, Dmax), NEG_INF, device=device, dtype=DTYPE)
        beta[:, -1, :, 0] = 0.0  # last timestep, duration=1

        # --- Backward recursion over time ---
        for t in reversed(range(T)):
            max_d = min(Dmax, T - t)
            prev_beta = torch.full((B, max_d, K), NEG_INF, device=device, dtype=DTYPE)

            for d in range(1, max_d + 1):
                t_next = t + d
                if t_next < T:
                    prev_beta[:, d - 1, :] = torch.logsumexp(beta[:, t_next, :, :d], dim=-1)
                    prev_beta[:, d - 1, :] = prev_beta[:, d - 1, :].masked_fill(
                        ~router.mask[:, t_next, 0].view(B, 1), NEG_INF
                    )
                else:
                    prev_beta[:, d - 1, :] = 0.0

            beta_trans = torch.logsumexp(prev_beta.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2).transpose(1, 2)
            beta[:, t, :, :max_d] = beta_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]

            invalid_d = torch.arange(Dmax, device=device) >= (T - t)
            if invalid_d.any():
                beta[:, t, :, invalid_d] = NEG_INF

        # --- Apply mask over padded timesteps ---
        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        beta = beta.masked_fill(~mask_exp, NEG_INF)
        return beta

    def _model_params(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        detach: bool = False) -> dict:

        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        canonical = router.canonical.detach() if detach else router.canonical
        context = router.context.detach() if detach else router.context
        params = {
            "initial_dist": self.initial_module.forward(
                context=canonical,
                return_dist=True
            ),
            "duration_dist": self.duration_module.forward(
                context=canonical,
                return_dist=True
            ),
            "transition_dist": self.transition_module.forward(
                context=canonical,
                return_dist=True
            ),
            "emission_dist": self.emission_module.forward(
                context=context,
                return_dist=True
            ),
        }
        self._params.update(params)
        return params

    def _viterbi(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        duration_temp: float = 0.0, transition_temp: float = 0.0,
        duration_weight: float = 0.0,) -> list[torch.Tensor]:

        K, Dmax = self.n_states, self.max_duration
        predicted: list[torch.Tensor] = []

        # --- context router ---
        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        device = router.log_probs.device
        dtype = router.log_probs.dtype
        B, T_max, _ = router.log_probs.shape
        durations_full = torch.arange(1, Dmax + 1, device=device)

        for b in range(B):
            L = int(router.mask[b].sum())
            if L == 0:
                predicted.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            ctx = router.context[b:b+1, :L]       # [1, L, H]
            canon = router.canonical[b:b+1]       # [1, 1, H]

            init_logits = self.initial_module.log_matrix(context=canon)[0, 0]       # [K]
            dur_logits = self.duration_module.log_matrix(context=ctx)[0]            # [L, K, Dmax]
            trans_logits = self.transition_module.log_matrix(context=ctx)[0]        # [L, K, K]

            if duration_weight != 0.0:
                dur_logits = dur_logits * (1.0 - duration_weight)

            emit_log = router.log_probs[b, :L]  # [L, K]
            cumsum_emit = torch.zeros((L + 1, K), device=device, dtype=dtype)
            cumsum_emit[1:] = torch.cumsum(emit_log, dim=0)

            # DP arrays
            V = torch.full((L, K), NEG_INF, device=device, dtype=dtype)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.long)
            best_dur = torch.zeros((L, K), device=device, dtype=torch.long)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1

                # Sum emissions for each duration
                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # [K, max_d]
                scores_dur = dur_logits[t, :, :max_d] + emit_sums          # [K, max_d]

                if t == 0:
                    # First timestep: combine with initial logits
                    scores = init_logits[:, None] + scores_dur
                    V[t], idx = scores.max(dim=1)
                    idx = idx.clamp(max=len(durations) - 1)
                    best_dur[t] = durations[idx]
                    continue

                # Previous V + transition
                prev_t = torch.clamp(starts - 1, min=0)
                prev_V = V[prev_t].T.unsqueeze(2)                        # [K, max_d, 1]
                trans = trans_logits[t].unsqueeze(1)                     # [K, 1, K]
                prev_scores = prev_V + trans                              # [K, max_d, K]

                # Handle start-of-sequence
                mask_start0 = (starts == 0)
                if mask_start0.any():
                    prev_scores[:, mask_start0, :] = init_logits[:, None]

                prev_max, prev_arg = prev_scores.max(dim=0)              # [max_d, K]
                scores = prev_max.T + scores_dur                         # [K, max_d]

                # Update DP
                V[t], dur_idx = scores.max(dim=1)
                best_dur[t] = durations[dur_idx]
                back_ptr[t] = torch.where(best_dur[t] == 1, -1, prev_arg[dur_idx, torch.arange(K, device=device)])

            # --- Backtracking ---
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

    def fit(self,
        X: torch.Tensor | list[torch.Tensor],
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 100,
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        lr: float = 1e-2, verbose: bool = True):

        self._convergence = ConvergenceTracker(
            tol=tol,
            rel_tol=tol,
            n_init=n_init,
            max_iter=max_iter,
            patience=1,
            verbose=verbose,
        )
        best_score = -float("inf")
        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            prev_ll = self._reset_parameters(run_idx, context=theta)

            # Collect all trainable parameters
            params = []
            for module in (
                self.initial_module,
                self.transition_module,
                self.duration_module,
                self.emission_module,
            ):
                if module is not None:
                    params += [p for p in module.parameters() if p.requires_grad]

            optimizer = torch.optim.Adam(params, lr=lr)

            for it in range(max_iter):
                optimizer.zero_grad()

                # Differentiable HSMM log-likelihood
                ll = self.score(X, theta=theta, reduce=True)
                loss = -ll

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()

                ll_val = ll.item()
                self._convergence.update(ll_val, it, run_idx)

                if verbose:
                    delta = ll_val - prev_ll if prev_ll is not None else float("nan")
                    print(f"[Iter {it:03d}] LL={ll_val:.6f} Δ={delta:.3e}")

                if self._convergence.converged_flags[run_idx]:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

                prev_ll = ll_val

            if ll_val > best_score:
                best_score = ll_val
                self._snapshot_best_params()
        if n_init > 1:
            self._restore_best_params()

        return self

    def score(self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        reduce: bool = False) -> torch.Tensor:

        device = X[0].device if isinstance(X, list) else X.device

        # --- normalize X into list[Tensor[T,F]] ---
        if torch.is_tensor(X):
            if X.ndim == 2:
                sequences = [X.to(dtype=DTYPE, device=device)]
            elif X.ndim == 3:
                sequences = [X[i].to(dtype=DTYPE, device=device) for i in range(X.shape[0])]
            else:
                raise ValueError(f"Unsupported X shape {X.shape}")
        elif isinstance(X, list):
            sequences = [
                torch.as_tensor(x, dtype=DTYPE, device=device) if not torch.is_tensor(x) else x.to(dtype=DTYPE, device=device)
                for x in X
            ]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B = len(sequences)
        if B == 0:
            return torch.empty(0, dtype=DTYPE, device=device)

        # --- normalize theta into list[Tensor[T,H]] or None ---
        context_list: Optional[list[torch.Tensor]] = None
        if theta is not None:
            if torch.is_tensor(theta):
                if theta.ndim == 3:
                    if theta.shape[0] != B:
                        raise ValueError("theta batch dimension mismatch")
                    context_list = [theta[i].to(dtype=DTYPE, device=device) for i in range(B)]
                elif theta.ndim == 2:
                    context_list = [theta.to(dtype=DTYPE, device=device) for _ in range(B)]
                else:
                    raise ValueError(f"Unsupported theta shape {theta.shape}")
            elif isinstance(theta, list):
                if len(theta) != B:
                    raise ValueError("theta list length mismatch")
                context_list = [
                    torch.as_tensor(t, dtype=DTYPE, device=device) if not torch.is_tensor(t) else t.to(dtype=DTYPE, device=device)
                    for t in theta
                ]
            else:
                raise TypeError(f"Unsupported theta type: {type(theta)}")

        # --- prepare padded SequenceSet ---
        seq_set = self._prepare(sequences, theta=context_list)
        
        # --- forward algorithm (vectorized) ---
        alpha = self._forward(seq_set)  # [B, T, K, D]

        lengths = seq_set.lengths.to(dtype=torch.long, device=device)
        
        # --- compute log-likelihoods ---
        log_likelihoods = torch.full((B,), NEG_INF, dtype=DTYPE, device=device)
        valid = lengths > 0
        if valid.any():
            last_alpha = alpha[valid, lengths[valid] - 1]  # [N_valid, K, D]
            log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

        log_likelihoods = torch.nan_to_num(
            log_likelihoods,
            nan=NEG_INF,
            neginf=NEG_INF,
            posinf=MAX_LOGITS
        )
        return log_likelihoods.sum() if reduce else log_likelihoods

    def _reset_parameters(self, run_idx: int, context: Optional[torch.Tensor] = None, preserve_best: bool = True) -> None:
        modules = ["initial", "transition", "duration", "emission"]
        for name in modules:
            module = getattr(self, f"{name}_module")
            if preserve_best and getattr(self, "_best_state", None) and name in self._best_state:
                module.load_state_dict(self._best_state[name])
            else:
                module.initialize(context=context)

        if self.encoder is not None:
            if preserve_best and getattr(self, "_best_state", None) and "encoder" in self._best_state:
                self.encoder.load_state_dict(self._best_state["encoder"])
            else:
                self.encoder.reset()

        if getattr(self, "_optimizer", None):
            for pg in self._optimizer.param_groups:
                pg['lr'] = 0.0

        if getattr(self, "_convergence", None):
            if len(self._convergence.converged_flags) <= run_idx:
                self._convergence.converged_flags.extend([False] * (run_idx + 1 - len(self._convergence.converged_flags)))
            else:
                self._convergence.converged_flags[run_idx] = False

    def _snapshot_best_params(self, context: Optional[torch.Tensor] = None):
        self._params.update({
            "initial_dist": self.initial_module.forward(context=context, return_dist=True),
            "duration_dist": self.duration_module.forward(context=context, return_dist=True),
            "transition_dist": self.transition_module.forward(context=context, return_dist=True),
            "emission_dist": self.emission_module.forward(context=context, return_dist=True),
        })

        self._best_state = {
            name: getattr(self, f"{name}_module").state_dict()
            for name in ["initial", "duration", "transition", "emission"]
        }
        if self.encoder is not None:
            self._best_state["encoder"] = self.encoder.state_dict()

    def _restore_best_params(self):
        if not hasattr(self, "_best_state"):
            raise RuntimeError("No best parameters have been snapshotted")

        for name in ["initial", "transition", "duration", "emission"]:
            getattr(self, f"{name}_module").load_state_dict(self._best_state[name])

        if self.encoder is not None and "encoder" in self._best_state:
            self.encoder.load_state_dict(self._best_state["encoder"])

        for name in ["initial", "transition", "duration", "emission"]:
            module = getattr(self, f"{name}_module")
            module._reset_buffers()
            module._invalidate_cache()

    def predict(self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["viterbi", "score"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        duration_temp: float = 1.0, transition_temp: float = 1.0,
        duration_weight: float = 0.0, verbose: bool = True) -> list[torch.Tensor] | torch.Tensor:

        seq_set = self._prepare(X, theta=context)
        B = len(seq_set.sequences)
        if B == 0 or seq_set.total_timesteps == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(B)]

        device = self.device
        if verbose:
            print(f"[Predict] Sequences: {B}, max_len: {max(seq_set.lengths)}, device: {device}")

        router = ContextRouter.from_tensor(seq_set, theta=context)

        if algorithm == "viterbi":
            nonzero_indices = [i for i, L in enumerate(seq_set.lengths) if L > 0]
            if len(nonzero_indices) < B:
                seq_set_nz = seq_set.index_select(torch.tensor(nonzero_indices, device=device))
                router_nz = router.select(nonzero_indices)
            else:
                seq_set_nz = seq_set
                router_nz = router

            results: list[torch.Tensor] = [torch.empty(0, dtype=torch.long, device=device) for _ in range(B)]
            decoded_paths = self._viterbi(
                seq_set_nz,
                theta=router_nz,
                duration_weight=duration_weight,
                transition_temp=transition_temp,
                duration_temp=duration_temp
            )
            for idx, path in zip(nonzero_indices, decoded_paths):
                results[idx] = path.detach().to(dtype=torch.long)
            return results

        elif algorithm == "score":
            # --- compute differentiable log-likelihoods ---
            return self.score(X, theta=context, reduce=False)

        else:
            raise ValueError(f"Unsupported decoding algorithm '{algorithm}'")

    def decode(self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        first_only: bool = True,
        verbose: bool = True) -> torch.Tensor | list[torch.Tensor]:

        if verbose:
            B = len(X) if isinstance(X, list) else X.shape[0] if X.ndim == 3 else 1
            print(f"[decode] algorithm={algorithm}, batch_size={B}")

        preds = self.predict(X, algorithm=algorithm, context=context, verbose=verbose)
        return preds[0] if first_only and preds else preds

