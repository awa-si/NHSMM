# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from nhsmm.constants import DEBUG, DTYPE, EPS, logger, MAX_LOGITS, NEG_INF
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm import Convergence, DefaultEncoder


@dataclass
class HSMMConfig:
    n_states: int
    n_features: int
    max_duration: int
    n_heads: int = 4
    dropout: float = 0.0
    min_covar: float = 1e-6
    cnn_channels: float = 5
    temperature: float = 1.0
    modulate_var: bool = False
    emission_type: str = "gaussian"
    hidden_dim: Optional[int] = None
    context_dim: Optional[int] = None
    pool: Literal["mean", "last", "max", "attn", "mha"] = "mean"
    transition_type: Literal["ergodic", "semi", "left-to-right"] = "ergodic"
    seed: Optional[int] = None
    debug: bool = False


class DefaultDistribution(nn.Module):
    """Container for HSMM distribution modules."""

    def __init__(
        self,
        initial: Optional[nn.Module] = None,
        duration: Optional[nn.Module] = None,
        transition: Optional[nn.Module] = None,
        emission: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.initial = initial
        self.duration = duration
        self.transition = transition
        self.emission = emission

    def initialize(self) -> Dict[str, Any]:
        return {
            "initial_dist": self.initial.initialize(),
            "duration_dist": self.duration.initialize(),
            "transition_dist": self.transition.initialize(),
            "emission_dist": self.emission.initialize(),
        }


class HSMM(nn.Module):

    def __init__(
        self,
        config: HSMMConfig,
        encoder: Optional[nn.Module] = None,
        dist: Optional[DefaultDistribution] = None
    ):
        super().__init__()
        self.config = config
        self._params: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        self.debug = config.debug
        self.n_states = config.n_states
        self.n_features = config.n_features

        self._init_enc(config=config, encoder=encoder)
        self._init_dist(config=config, dist=dist)
        self.to(device=self.device, dtype=DTYPE)

        try:
            self._params.update(self.dist.initialize())
        except Exception as err:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {err}") from err

    def _init_enc(self, config: HSMMConfig, encoder: Optional[nn.Module]) -> None:

        self.context_dim = config.context_dim
        self.hidden_dim = config.hidden_dim

        if encoder is None:
            hidden_dim = max(32, min(64, self.n_features * 2))
            encoder = DefaultEncoder(
                n_features=self.n_features,
                cnn_channels=config.cnn_channels,
                hidden_dim=hidden_dim,
            )
        self.encoder = encoder if isinstance(encoder, ContextEncoder) else ContextEncoder(
            encoder=encoder,
            pool=config.pool,
            n_heads=config.n_heads,
            dropout=config.dropout,
            debug=self.debug
        )
        self.encoder.eval()

        try:
            dummy = torch.zeros(1, 16, self.n_features, device=self.device, dtype=DTYPE)
            try:
                _, ctx, _ = self.encoder(dummy, return_context=True, return_sequence=True)
                inferred_dim = ctx.shape[-1]
            except TypeError:
                inferred_dim = self.encoder(dummy).shape[-1]

            if self.context_dim is None:
                self.context_dim = inferred_dim
            if self.hidden_dim is None:
                self.hidden_dim = self.context_dim
            elif self.hidden_dim != self.context_dim:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must equal context_dim "
                    f"({self.context_dim}) unless projections are explicitly defined."
                )
        finally:
            self.encoder.train()

    def _init_dist(self, config: HSMMConfig, dist: Optional[DefaultDistribution] = None) -> None:

        self.temperature = config.temperature
        self.max_duration = config.max_duration
        self.emission_type = config.emission_type

        self.dist = dist or DefaultDistribution(
            initial=Initial(
                n_states=self.n_states,
                context_dim=self.context_dim,
                hidden_dim=self.hidden_dim
            ),
            duration=Duration(
                n_states=self.n_states,
                max_duration=config.max_duration,
                context_dim=self.context_dim,
                hidden_dim=self.hidden_dim,
                temperature=config.temperature
            ),
            transition=Transition(
                n_states=self.n_states,
                n_features=self.n_features,
                transition_type=config.transition_type,
                context_dim=self.context_dim,
                hidden_dim=self.hidden_dim,
                temperature=config.temperature
            ),
            emission=Emission(
                n_states=self.n_states,
                n_features=self.n_features,
                emission_type=config.emission_type,
                context_dim=self.context_dim,
                hidden_dim=self.hidden_dim,
                modulate_var=config.modulate_var,
                temperature=config.temperature,
                min_covar=config.min_covar
            )
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
            dist = self.dist.emission.forward(context=context, return_dist=True)
            X_exp = X.unsqueeze(2).expand(B, T, K, F) # expand X for broadcasting
            log_probs = dist.log_prob(X_exp)
            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        if getattr(self, "debug", False):
            print(f"[Prepare] X={X.shape}, context={context.shape}, canonical={canonical.shape}, log_probs={log_probs.shape}")

        return SequenceSet(
            masks=mask,
            sequences=X,
            lengths=lengths,
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

        # --- Module logits ---
        initial_logits = self.dist.initial.log_matrix(context=router.canonical)  # [B,1,K]
        duration_logits = self.dist.duration.log_matrix(context=router.context)  # [B,T,K,Dmax]
        transition_logits = self.dist.transition.log_matrix(context=router.context)  # [B,T,K,K]

        # --- Cumulative emission sums for all durations ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device, dtype=dtype)
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)

        # Create start/end indices for all durations
        d_range = torch.arange(1, Dmax + 1, device=device)  # [Dmax]
        t_range = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1)  # [T,1]
        start_idx = (t_range - d_range + 1).clamp(min=0)  # [T,Dmax]
        end_idx = (t_range + 1).expand(-1, Dmax)          # [T,Dmax]

        # Broadcast for batch/state dimensions
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, T, K, Dmax)
        k_idx = torch.arange(K, device=device).view(1, 1, K, 1).expand(B, T, K, Dmax)
        t_idx_start = start_idx.view(1, T, 1, Dmax).expand(B, T, K, Dmax)
        t_idx_end = end_idx.view(1, T, 1, Dmax).expand(B, T, K, Dmax)

        # Compute emission sums for all durations at once
        emit_sums = cumsum_emit[b_idx, t_idx_end, k_idx] - cumsum_emit[b_idx, t_idx_start, k_idx]  # [B,T,K,Dmax]

        # --- Initialize alpha ---
        alpha = torch.full((B, T, K, Dmax), NEG_INF, device=device, dtype=dtype)
        alpha[:, 0, :, 0] = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]

        # Precompute duration masks: mask invalid durations for each timestep
        duration_mask = torch.arange(Dmax, device=device).view(1, 1, 1, Dmax) <= torch.arange(T, device=device).view(1, T, 1, 1)

        # --- Forward recursion vectorized over durations ---
        for t in range(1, T):
            max_d = min(Dmax, t + 1)

            # Gather previous alphas for all possible durations
            idx_prev = t - d_range[:max_d]  # [max_d]
            idx_prev = idx_prev.clamp(min=0)

            # alpha_prev: [B, max_d, K]
            alpha_prev = alpha[:, idx_prev, :, :max_d]
            # Sum over past duration dimension if needed
            alpha_prev = torch.logsumexp(alpha_prev, dim=-1)  # [B, max_d, K]

            # Compute transition + previous alpha
            alpha_trans = torch.logsumexp(alpha_prev.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2).transpose(1, 2)  # [B,K,max_d]

            # Add duration logits and emission sums
            alpha_t = alpha_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]  # [B,K,max_d]

            # Apply duration mask
            full_alpha = torch.full((B, K, Dmax), NEG_INF, device=device, dtype=dtype)
            full_alpha[:, :, :max_d] = alpha_t
            full_alpha = torch.where(duration_mask[:, t:t+1, :, :], full_alpha, NEG_INF)
            alpha[:, t] = full_alpha

        # Mask out timesteps beyond sequence lengths
        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)  # [B,T]
        alpha = torch.where(length_mask.unsqueeze(-1).unsqueeze(-1), alpha, torch.full_like(alpha, NEG_INF))

        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        alpha = torch.where(mask_exp, alpha, torch.full_like(alpha, NEG_INF))
        return alpha

    def _backward(self, X: SequenceSet, theta: Optional[Union[torch.Tensor, ContextRouter]] = None) -> torch.Tensor:
        router = ContextRouter.from_tensor(X, theta=theta) if not isinstance(theta, ContextRouter) else theta

        Dmax = self.max_duration
        dtype = router.context.dtype
        B, T, K = router.log_probs.shape
        device = router.context.device

        initial_logits = self.dist.initial.log_matrix(context=router.canonical)  # [B,1,K]
        duration_logits = self.dist.duration.log_matrix(context=router.context)  # [B,T,K,Dmax]
        transition_logits = self.dist.transition.log_matrix(context=router.context)  # [B,T,K,K]

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
            "initial_dist": self.dist.initial.forward(
                context=canonical,
                return_dist=True
            ),
            "duration_dist": self.dist.duration.forward(
                context=canonical,
                return_dist=True
            ),
            "transition_dist": self.dist.transition.forward(
                context=canonical,
                return_dist=True
            ),
            "emission_dist": self.dist.emission.forward(
                context=context,
                return_dist=True
            ),
        }
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

            init_logits = self.dist.initial.log_matrix(context=canon)[0, 0]       # [K]
            dur_logits = self.dist.duration.log_matrix(context=ctx)[0]            # [L, K, Dmax]
            trans_logits = self.dist.transition.log_matrix(context=ctx)[0]        # [L, K, K]

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

        self._convergence = Convergence(
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

            # Collect all trainable parameters maybe _model_params?
            params = []
            for name in ["initial", "transition", "duration", "emission"]:
                module = getattr(self.dist, name)
                if module is not None:
                    params += [p for p in module.parameters() if p.requires_grad]

            if not getattr(self, "_optimizer", None):
                self._optimizer = torch.optim.Adam(params, lr=lr)

            for it in range(max_iter):
                self. _optimizer.zero_grad()

                ll = self.score(X, theta=theta, reduce=True)
                loss = -ll

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                self._optimizer.step()

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

        seq_set = self._prepare(sequences, theta=context_list)
        alpha = self._forward(seq_set)  # [B, T, K, D]
        lengths = seq_set.lengths.to(dtype=torch.long, device=device)
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
            module = getattr(self.dist, name)
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
            "initial_dist": self.dist.initial.forward(context=context, return_dist=True),
            "duration_dist": self.dist.duration.forward(context=context, return_dist=True),
            "transition_dist": self.dist.transition.forward(context=context, return_dist=True),
            "emission_dist": self.dist.emission.forward(context=context, return_dist=True),
        })

        self._best_state = {
            name: getattr(self.dist, name).state_dict()
            for name in ["initial", "duration", "transition", "emission"]
        }
        if self.encoder is not None:
            self._best_state["encoder"] = self.encoder.state_dict()

    def _restore_best_params(self):
        if not hasattr(self, "_best_state"):
            raise RuntimeError("No best parameters have been snapshotted")

        for name in ["initial", "transition", "duration", "emission"]:
            getattr(self.dist, name).load_state_dict(self._best_state[name])

        if self.encoder is not None and "encoder" in self._best_state:
            self.encoder.load_state_dict(self._best_state["encoder"])

        for name in ["initial", "transition", "duration", "emission"]:
            module = getattr(self.dist, name)
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

