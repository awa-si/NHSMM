# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.constants import DEBUG, DTYPE, EPS, logger, MAX_LOGITS, NEG_INF
from nhsmm.distributions import Initial, Emission, Duration, Transition
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm import constraints, SeedGenerator, ConvergenceTracker


class HSMM(nn.Module):

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
        debug: bool = False):

        super().__init__()

        self.alpha = alpha
        self.debug = debug
        self.n_states = n_states
        self.n_features = n_features
        self.temperature = temperature
        self.max_duration = max_duration
        self.emission_type = emission_type

        self._params: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Set random seed ---
        self.seed = seed or SeedGenerator(seed).seed
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

        if isinstance(X, list):
            X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        if isinstance(theta, list):
            theta = torch.nn.utils.rnn.pad_sequence(theta, batch_first=True)

        if X.ndim == 2:
            X = X.unsqueeze(0)

        B, T, F = X.shape
        K = self.n_states
        device = X.device
        debug = self.debug

        # --- mask ---
        if mask is None:
            mask = torch.ones(B, T, 1, dtype=torch.bool, device=device)
        else:
            mask = mask.bool().to(device)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            elif mask.ndim != 3:
                mask = mask.view(B, T, 1)

        lengths = mask.squeeze(-1).sum(dim=1).to(torch.long)

        # --- context ---
        if theta is not None:
            if theta.ndim == 2:
                theta = theta.unsqueeze(1).expand(B, T, -1)
            context = theta
            canonical = theta[:, :1, :]
        else:
            if T == 0:
                H = self.context_dim
                context = torch.zeros(B, 0, H, device=device, dtype=X.dtype)
                canonical = torch.zeros(B, 1, H, device=device, dtype=X.dtype)
            else:
                context, canonical = self.encoder._emcode(
                    X,
                    mask=mask.squeeze(-1),
                    detach=False
                )

        if debug:
            print(f"[Prepare] context={context.shape}, canonical={canonical.shape}")

        # --- emission log-probs ---
        if T == 0:
            log_probs = torch.empty(B, 0, K, device=device, dtype=X.dtype)
        else:
            dist = self.emission_module.forward(context=context, return_dist=True)

            if hasattr(dist, "log_prob"):
                X_exp = X.unsqueeze(2).expand(B, T, K, F)
                log_probs = dist.log_prob(X_exp)
            else:
                raise RuntimeError("Emission distribution does not support log_prob")

            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        if debug:
            print(f"[Prepare] log_probs={log_probs.shape}")

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
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=dtype)
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
        alpha = torch.full((B, T, K, Dmax), NEG_INF, device=device, dtype=dtype)
        alpha[:, 0, :, 0] = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]

        # --- Recursion over time ---
        for t in range(1, T):
            max_d = min(Dmax, t + 1)

            # Previous alpha sums for all durations
            prev_alpha = torch.full((B, max_d, K), NEG_INF, device=device, dtype=dtype)
            for d in range(1, max_d + 1):
                t_prev = t - d
                mask_valid = (t < X.lengths).view(B, 1)  # [B,1] valid timestep
                if t_prev >= 0:
                    prev_alpha[:, d - 1, :] = torch.logsumexp(alpha[:, t_prev, :, :d], dim=-1)
                else:
                    prev_alpha[:, d - 1, :] = initial_logits.squeeze(1) + duration_logits[:, 0, :, d - 1]
                prev_alpha[:, d - 1, :] = torch.where(mask_valid, prev_alpha[:, d - 1, :], NEG_INF)

            # Transition & duration update
            alpha_trans = torch.logsumexp(prev_alpha.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2).transpose(1, 2)
            duration_logits_t = duration_logits[:, t, :, :max_d]
            alpha[:, t, :, :max_d] = alpha_trans + duration_logits_t + emit_sums[:, t, :, :max_d]

            # Invalidate durations exceeding current timestep
            invalid_d = torch.arange(Dmax, device=device) > t
            if invalid_d.any():
                alpha[:, t, :, invalid_d] = NEG_INF

        # --- Invalidate timesteps beyond sequence length ---
        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)  # [B,T]
        alpha = alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), NEG_INF)

        # --- Apply mask over padded timesteps ---
        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        alpha = alpha.masked_fill(~mask_exp, NEG_INF)

        return alpha

    def _backward(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> torch.Tensor:
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta

        Dmax = self.max_duration
        dtype = router.context.dtype
        B, T, K = router.log_probs.shape
        device = router.context.device

        duration_logits = self.duration_module.log_matrix(context=router.context)
        transition_logits = self.transition_module.log_matrix(context=router.context)

        # --- Precompute cumulative emission sums ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=dtype) # cumsum_emit: [B, T+1, K]
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
        beta = torch.full((B, T, K, Dmax), NEG_INF, device=device, dtype=dtype)
        beta[:, -1, :, 0] = 0.0  # last timestep, duration=1

        # --- Backward recursion over time ---
        for t in reversed(range(T)):
            max_d = min(Dmax, T - t)
            prev_beta = torch.full((B, max_d, K), NEG_INF, device=device, dtype=dtype)

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

    def _compute_posteriors(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> Tuple:

        B = len(X.sequences)
        T_max = max(X.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration
        device = X.sequences.device if B > 0 else torch.device("cpu")

        if B == 0 or T_max == 0:
            gamma = torch.zeros((B, T_max, K), dtype=DTYPE, device=device)
            eta = torch.zeros((B, T_max, K, Dmax), dtype=DTYPE, device=device)
            xi = torch.zeros((B, max(T_max - 1, 0), K, K), dtype=DTYPE, device=device)
            return gamma, xi, eta

        # --- Context router ---
        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)

        # --- Forward / backward ---
        alpha = self._forward(X, theta=router)      # [B,T,K,Dmax]
        beta = self._backward(X, theta=router)      # [B,T,K,Dmax]

        # --- Masks ---
        mask = router.mask.bool().squeeze(-1) if router.mask.ndim == 3 else router.mask.bool()
        mask_TK = mask.unsqueeze(-1)                # [B,T,1]
        mask_TKD = mask_TK.unsqueeze(-1)            # [B,T,1,1]

        # --- Duration posteriors eta ---
        eta_log = alpha + beta                       # [B,T,K,Dmax]
        eta_flat = eta_log.view(B, T_max, -1)
        eta_norm = torch.logsumexp(eta_flat, dim=-1, keepdim=True)
        eta = (eta_flat - eta_norm).view(B, T_max, K, Dmax).exp()
        eta = eta * mask_TKD

        # --- State posteriors gamma ---
        gamma = eta.sum(dim=-1)                      # [B,T,K]
        gamma = gamma * mask_TK
        gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(EPS)

        # --- Transition posteriors xi (vectorized HSMM) ---
        xi = torch.zeros((B, max(T_max - 1, 0), K, K), dtype=DTYPE, device=device)

        if T_max > 1:
            lengths = torch.as_tensor(X.lengths, device=device)
            beta_marg = torch.logsumexp(beta, dim=-1).exp()  # [B,T,K]

            # Duration-shifted beta for valid continuation
            t_idx = torch.arange(T_max, device=device)       # [T]
            d_idx = torch.arange(Dmax, device=device) + 1   # [D]
            t_plus_d = t_idx[:, None] + d_idx[None, :]      # [T,D]

            # Valid timesteps mask
            valid_td = t_plus_d < T_max
            valid_td = valid_td[None, :, :] & (t_plus_d[None, :, :] < lengths[:, None, None])

            # Gather beta_shift safely
            beta_shift = torch.zeros(B, T_max, Dmax, K, dtype=DTYPE, device=device)
            for b in range(B):
                for t in range(T_max):
                    for d in range(Dmax):
                        tp = t + d + 1
                        if tp < lengths[b]:
                            beta_shift[b, t, d] = beta_marg[b, tp]

            # Duration-weighted continuation: cont[b,t,i,j] = sum_d eta[b,t,i,d] * beta_shift[b,t,d,j]
            cont = torch.einsum("btid,btdj->btij", eta, beta_shift)

            # --- Apply transitions ---
            if hasattr(self.transition_module, "log_factors"):
                # Low-rank transitions: A[i,r], B[r,j]
                A, Bf = self.transition_module.log_factors(router.context)
                if A.ndim == 3:
                    A = A.unsqueeze(0).expand(B, T_max, K, -1)
                    Bf = Bf.unsqueeze(0).expand(B, T_max, -1, K)
                tmp = torch.einsum("btir,btrj->btij", cont, Bf.exp())
                xi[:, :T_max - 1] = torch.einsum("btij,btir->btij", tmp[:, :T_max - 1], A[:, :T_max - 1].exp())
            else:
                # Dense transitions
                trans_logits = self.transition_module.log_matrix(router.context)
                if trans_logits.ndim == 2:
                    trans_logits = trans_logits.unsqueeze(0).unsqueeze(1).expand(B, T_max, K, K)
                xi[:, :T_max - 1] = cont[:, :T_max - 1] * trans_logits[:, :T_max - 1].exp()

            # --- Mask invalid timesteps ---
            valid_t = torch.arange(T_max - 1, device=device).unsqueeze(0) < (lengths - 1).unsqueeze(1)
            xi = xi * valid_t.unsqueeze(-1).unsqueeze(-1)

            # --- Normalize ---
            xi = xi / xi.sum(dim=(2, 3), keepdim=True).clamp_min(EPS)

        return gamma, xi, eta

    def _model_params(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> dict:

        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        gamma, xi, eta = self._compute_posteriors(X, theta=router)

        # Initial state distribution: p(z0 | canonical context)
        self.initial_module.update(
            posterior=gamma[:, 0],          # [B, K]
            context=router.canonical         # [B, H]
        )

        # Duration distribution: p(d | z, context_t)
        self.duration_module.update(
            posterior=eta,                  # [B, T, K, D]
            context=router.context           # [B, T, H]
        )

        # Transition distribution: p(z_t | z_{t-1}, context_t)
        self.transition_module.update(
            posterior=xi,                   # [B, T, K, K]
            context=router.context           # [B, T, H]
        )

        # --- Prepare learned conditional distributions ---
        params = {
            "initial_dist": self.initial_module.forward(
                context=router.canonical,
                return_dist=True
            ),
            "duration_dist": self.duration_module.forward(
                context=router.canonical,
                return_dist=True
            ),
            "transition_dist": self.transition_module.forward(
                context=router.canonical,
                return_dist=True
            ),
            "emission_dist": self.emission_module.forward(
                context=router.context,
                return_dist=True
            ),
        }

        return params

    def _viterbi(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        duration_temp: float = 0.0, transition_temp: float = 0.0, duration_weight: float = 0.0) -> list[torch.Tensor]:

        K, Dmax = self.n_states, self.max_duration
        predicted: list[torch.Tensor] = []

        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        device = router.log_probs.device
        B, T_max, _ = router.log_probs.shape
        durations_full = torch.arange(1, Dmax + 1, device=device)

        for b in range(B):
            L = int(router.mask[b].sum())
            print(f"[DEBUG][Batch {b}] sequence length L={L}, max_duration={Dmax}")
            if L == 0:
                predicted.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            ctx = router.context[b:b+1, :L]
            canon = router.canonical[b:b+1]

            init_logits = self.initial_module.log_matrix(context=canon)[0, 0]
            dur_logits = self.duration_module.log_matrix(context=ctx)[0]
            trans_logits = self.transition_module.log_matrix(context=ctx)[0]

            if duration_weight != 0.0:
                dur_logits = dur_logits * (1.0 - duration_weight)

            emit_log = router.log_probs[b, :L]
            cumsum_emit = torch.zeros((L + 1, K), device=device, dtype=emit_log.dtype)
            cumsum_emit[1:] = torch.cumsum(emit_log, dim=0)

            V = torch.full((L, K), NEG_INF, device=device, dtype=emit_log.dtype)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.long)
            best_dur = torch.zeros((L, K), device=device, dtype=torch.long)

            log_every = 50  # only log every 10 steps
            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1

                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # [K, max_d]
                scores_dur = dur_logits[t, :, :max_d] + emit_sums          # [K, max_d]

                if (t % log_every == 0 or t == L-1):
                    print(f"[DEBUG][Batch {b}][t={t}] max_d={max_d}, "
                          f"durations={durations.tolist()}, "
                          f"scores_dur max={scores_dur.max().item():.2f}, "
                          f"min={scores_dur.min().item():.2f}")

                if t == 0:
                    scores = init_logits[:, None] + scores_dur
                    V[t], idx = scores.max(dim=1)
                    idx = idx.clamp(max=len(durations)-1)
                    best_dur[t] = durations[idx]
                    continue

                prev_t = torch.clamp(starts - 1, min=0)
                prev_V = V[prev_t].T.unsqueeze(2)
                trans = trans_logits[t].unsqueeze(1)
                prev_scores = prev_V + trans

                mask_start0 = (starts == 0)
                if mask_start0.any():
                    prev_scores[:, mask_start0, :] = init_logits[None, None, :]

                prev_max, prev_arg = prev_scores.max(dim=0)
                prev_arg = prev_arg.clamp(max=K-1)
                scores = prev_max.T + scores_dur
                V[t], dur_idx = scores.max(dim=1)
                dur_idx = dur_idx.clamp(max=len(durations)-1)
                best_dur[t] = durations[dur_idx]
                back_ptr[t] = torch.where(best_dur[t] == 1, -1, prev_arg[dur_idx, torch.arange(K, device=device)])

            # Backtracking
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

        return predicted

    def _map(self,
        log_probs_seq: torch.Tensor,      # [T, n_features] or [T, n_states]
        init_logits: torch.Tensor,        # [n_states]
        trans_logits: torch.Tensor,       # [n_states, n_states]
        dur_logits: torch.Tensor) -> torch.Tensor:

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
        tol: float = 1e-4,
        max_iter: int = 20,
        theta: Optional[torch.Tensor] = None, verbose: bool = True):

        patience = 1
        adapt_factor = 10.0
        best_score = -float("inf")
        B = X.shape[0] if X.ndim == 3 else 1
        update_rate_min, update_rate_max = 1e-3, 1.0  # lower min LR for Adam stability

        self._convergence = ConvergenceTracker(
            tol=tol, rel_tol=tol, n_init=n_init, max_iter=max_iter, patience=patience, verbose=verbose
        )
        neural_update = getattr(self, "encoder", None) is not None and any(
            p.requires_grad for p in self.emission_module.parameters()
        )

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            prev_ll = self._reset_parameters(run_idx)

            for it in range(max_iter + 1):
                seq_set = self._prepare(X, theta=theta)
                seq_tensor, mask, F_dim = seq_set.sequences, seq_set.masks, seq_set.sequences.shape[-1]

                gamma, xi, eta = self._compute_posteriors(seq_set, theta=theta)

                flat_mask = mask.view(-1)
                flat_X = seq_tensor.reshape(-1, F_dim)[flat_mask]
                flat_gamma = gamma.reshape(-1, self.n_states)[flat_mask]
                flat_theta = theta.reshape(-1, theta.shape[-1])[flat_mask] if theta is not None else None

                # --- Compute log-likelihood ---
                log_probs_tensor = seq_set.log_probs
                ll_val = float(log_probs_tensor.sum(dim=(0, 1)) if B > 1 else log_probs_tensor.detach().sum())
                self._convergence.update(ll_val, it, run_idx)

                if verbose and it > 0:
                    delta_ll = ll_val - prev_ll if prev_ll is not None else float("nan")
                    print(f"[Iter {it:02d}] LL={ll_val:.6f} Δ={delta_ll:.3e}")

                adaptive_lr = (
                    min(update_rate_max, max(update_rate_min, adapt_factor * abs(ll_val - prev_ll)))
                    if prev_ll is not None else update_rate_max
                )
                prev_ll = ll_val

                if it == max_iter or self._convergence.converged_flags[run_idx]:
                    if verbose and self._convergence.converged_flags[run_idx]:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

                if neural_update:
                    for p in self.emission_module.parameters():
                        p.requires_grad_(True)

                    if not hasattr(self, "_emission_optimizer") or self._emission_optimizer is None:
                        self._emission_optimizer = torch.optim.SGD(self.emission_module.parameters(), lr=adaptive_lr)
                    else:
                        for pg in self._emission_optimizer.param_groups: pg["lr"] = adaptive_lr

                    self._emission_optimizer.zero_grad()
                    log_probs = self.emission_module.log_prob(flat_X)
                    loss = -(flat_gamma.detach() * log_probs).sum() / flat_gamma.detach().sum().clamp_min(EPS)
                    loss.backward()
                    self._emission_optimizer.step()
                else:
                    self.emission_module.update(
                        posterior=flat_gamma, context=flat_theta, update_rate=adaptive_lr
                    )

                self.initial_module.update(new_logits=gamma[:, 0].sum(0), from_probs=True)
                self.duration_module.update(new_logits=eta.sum(dim=0), from_probs=True)
                self.transition_module.update(new_logits=xi.sum(dim=(0, 1)), from_probs=True)

                if ll_val > best_score:
                    best_score = ll_val
                    self._params.update({
                        "initial_dist": self.initial_module.forward(return_dist=True),
                        "duration_dist": self.duration_module.forward(return_dist=True),
                        "transition_dist": self.transition_module.forward(return_dist=True),
                        "emission_dist": self.emission_module.forward(return_dist=True),
                    })
                    self._snapshot_best_params()

        if n_init > 1:
            self._restore_best_params()

        return self

    def _reset_parameters(self, run_idx: int, preserve_best: bool = True) -> None:
        for module_name in ["initial_module", "transition_module", "duration_module", "emission_module"]:
            module = getattr(self, module_name)
            if preserve_best and hasattr(self, "_best_state") and module_name.replace("_module","") in self._best_state:
                module.load_state_dict(self._best_state[module_name.replace("_module","")])
            else:
                module.initialize()

        if self.encoder is not None:
            if preserve_best and hasattr(self, "_best_state") and "encoder" in self._best_state:
                self.encoder.load_state_dict(self._best_state["encoder"])
            else:
                self.encoder.reset()

        # Reset optimizer LR to 0 (will be set adaptively later)
        if hasattr(self, "_emission_optimizer") and self._emission_optimizer is not None:
            for pg in self._emission_optimizer.param_groups:
                pg['lr'] = 0.0

        # Reset convergence flags
        if hasattr(self, "_convergence") and self._convergence is not None:
            self._convergence.converged_flags[run_idx] = False

    def _snapshot_best_params(self):
        self._best_state = {
            "initial": self.initial_module.state_dict(),
            "duration": self.duration_module.state_dict(),
            "transition": self.transition_module.state_dict(),
            "emission": self.emission_module.state_dict(),
        }
        if self.encoder is not None:
            self._best_state["encoder"] = self.encoder.state_dict()

    def _restore_best_params(self):
        if not hasattr(self, "_best_state"):
            raise RuntimeError("No best parameters have been snapshotted")

        self.initial_module.load_state_dict(self._best_state["initial"])
        self.transition_module.load_state_dict(self._best_state["transition"])
        self.duration_module.load_state_dict(self._best_state["duration"])
        self.emission_module.load_state_dict(self._best_state["emission"])

        if self.encoder is not None and "encoder" in self._best_state:
            self.encoder.load_state_dict(self._best_state["encoder"])

        for m in (
            self.initial_module,
            self.transition_module,
            self.duration_module,
            self.emission_module,
        ):
            m._reset_buffers()
            m._invalidate_cache()

    def predict(self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        transition_temp: float = 1.0,
        duration_temp: float = 1.0,
        duration_weight: float = 0.0,
        verbose: bool = True) -> list[torch.Tensor]:
        """
        Decode most likely state sequence for given input sequences.

        Args:
            X: [T,F] tensor or list of tensors [Ti,F]
            algorithm: "viterbi" (HSMM) or "map" (state-wise MAP)
            context: optional context tensor or list
            transition_temp: temperature for transition logits
            duration_temp: temperature for duration logits
            duration_weight: weight for duration bias in viterbi
            verbose: enable debug prints

        Returns:
            List of tensors [Ti] with predicted states
        """
        seq_set = self._prepare(X, theta=context)
        B = len(seq_set.sequences)
        if B == 0 or seq_set.total_timesteps == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(B)]

        device = self.device
        max_len = max(seq_set.lengths)
        if verbose:
            print(f"[Predict] Sequences: {B}, max_len: {max_len}, device: {device}")

        router = ContextRouter.from_tensor(seq_set, theta=context)

        # Handle sequences with non-zero length
        nonzero_indices = [i for i, L in enumerate(seq_set.lengths) if L > 0]
        if len(nonzero_indices) < B:
            seq_set_nz = seq_set.index_select(torch.tensor(nonzero_indices, device=device))
            router_nz = router.select(nonzero_indices)
        else:
            seq_set_nz = seq_set
            router_nz = router

        results: list[torch.Tensor] = [torch.empty(0, dtype=torch.long, device=device) for _ in range(B)]

        if algorithm.lower() == "viterbi":
            # HSMM Viterbi decoding
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

        elif algorithm.lower() == "map":
            # Compute posteriors
            gamma, _, _ = self._compute_posteriors(seq_set_nz, theta=router_nz)
            for i, b in enumerate(nonzero_indices):
                L = seq_set.lengths[b]
                gamma_b = gamma[i, :L]
                gamma_b = torch.nan_to_num(gamma_b, nan=-1e9, posinf=-1e9, neginf=-1e9)

                # Initial / transition / duration logits for MAP decoding
                init_logits = self.initial_module.log_matrix(context=router_nz.canonical[i:i+1, :1]).view(-1)
                trans_logits = self.transition_module.log_matrix(context=router_nz.context[i:i+1, :L])
                dur_logits = self.duration_module.log_matrix(context=router_nz.context[i:i+1, :L])

                # Safely reduce time dimension if needed
                if trans_logits.ndim > 2 and trans_logits.shape[1] > 1:
                    trans_logits = trans_logits.mean(dim=1)
                elif trans_logits.ndim > 2:
                    trans_logits = trans_logits[0]

                if dur_logits.ndim > 2 and dur_logits.shape[1] > 1:
                    dur_logits = dur_logits.mean(dim=1)
                elif dur_logits.ndim > 2:
                    dur_logits = dur_logits[0]

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
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'")

    def decode(self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["viterbi", "map"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        first_only: bool = True,
        verbose: bool = True) -> torch.Tensor | list[torch.Tensor]:

        if torch.is_tensor(X):
            if X.ndim == 2:
                X_list = [X]
            elif X.ndim == 3:
                X_list = [X[i] for i in range(X.shape[0])]
            else:
                raise ValueError(f"Unexpected tensor shape {X.shape} for X")
        elif isinstance(X, list):
            X_list = [torch.as_tensor(x, dtype=DTYPE) if not torch.is_tensor(x) else x.to(dtype=DTYPE)
                      for x in X]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        # --- Normalize context into list ---
        context_list: Optional[list[torch.Tensor]] = None
        if context is not None:
            if torch.is_tensor(context):
                if context.ndim == 2:
                    context_list = [context]
                elif context.ndim == 3:
                    context_list = [context[i] for i in range(context.shape[0])]
                else:
                    raise ValueError(f"Unexpected context tensor shape {context.shape}")
            elif isinstance(context, list):
                context_list = [torch.as_tensor(c, dtype=DTYPE) if not torch.is_tensor(c) else c.to(dtype=DTYPE)
                                for c in context]
            else:
                raise TypeError(f"Unsupported context type: {type(context)}")

        if verbose:
            print(f"[decode] algorithm={algorithm}, batch_size={len(X_list)}")

        # --- Predict sequences ---
        preds = self.predict(
            X_list,
            algorithm=algorithm,
            context=context_list,
            verbose=verbose
        )

        # --- Return first sequence if requested ---
        return preds[0] if first_only and preds else preds

    @torch.no_grad()
    def score(self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
        verbose: bool = True) -> torch.Tensor:

        if torch.is_tensor(X):
            sequences = [X]
        elif isinstance(X, list):
            sequences = [torch.as_tensor(x, dtype=DTYPE) if not torch.is_tensor(x) else x.to(dtype=DTYPE) for x in X]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B = len(sequences)
        if B == 0:
            return torch.empty(0, dtype=DTYPE, device=self.device)

        # --- normalize context ---
        context_list: Optional[list[torch.Tensor]] = None
        if theta is not None:
            if torch.is_tensor(theta):
                if theta.ndim == 2 and B > 1:
                    lengths = [seq.shape[0] for seq in sequences]
                    cum_lengths = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(lengths, dtype=torch.long).cumsum(0)])
                    context_list = [theta[cum_lengths[b]:cum_lengths[b+1]] for b in range(B)]
                else:
                    context_list = [theta] * B
            elif isinstance(theta, list):
                context_list = [torch.as_tensor(t, dtype=DTYPE) if t is not None else None for t in theta]
                if len(context_list) != B:
                    raise ValueError(f"Length of theta list ({len(context_list)}) != number of sequences ({B})")
            else:
                raise TypeError(f"Unsupported theta type: {type(theta)}")

        # --- prepare SequenceSet (padding, mask, canonical context, log_probs) ---
        seq_set = self._prepare(sequences, theta=context_list)

        if verbose:
            print(f"[score] Prepared SequenceSet with {B} sequences, max_len={max(seq_set.lengths)}")

        # --- forward pass (vectorized alpha) ---
        alpha = self._forward(seq_set, theta=None)  # [B,T,K,Dmax]

        # --- compute per-sequence log-likelihood ---
        if isinstance(seq_set.lengths, torch.Tensor):
            lengths_tensor = seq_set.lengths.detach().clone().to(dtype=torch.long, device=self.device)
        else:
            lengths_tensor = torch.tensor(seq_set.lengths, dtype=torch.long, device=self.device)

        max_len = alpha.shape[1]
        log_likelihoods = torch.full((B,), NEG_INF, dtype=DTYPE, device=self.device)

        if B > 0:
            idx = lengths_tensor > 0
            last_alpha = alpha[idx, lengths_tensor[idx]-1]  # [N_valid, K, Dmax]
            log_likelihoods[idx] = torch.logsumexp(last_alpha.reshape(last_alpha.size(0), -1), dim=1)

        if verbose:
            logger.info(f"[score] batch={B}, min={log_likelihoods.min():.4f}, max={log_likelihoods.max():.4f}, mean={log_likelihoods.mean():.4f}")

        return log_likelihoods

