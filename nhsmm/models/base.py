# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from nhsmm.constants import (
    DEBUG, DTYPE, EPS, logger, MAX_LOGITS, NEG_INF, HSMMConfig
)
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm import Convergence, DefaultEncoder, DefaultDistribution


class HSMM(nn.Module):

    def __init__(self, config: HSMMConfig, encoder: Optional[nn.Module] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

        self.config = config
        self._params: Dict[str, Any] = {}

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        self.debug = self.config.debug
        self.n_states = self.config.n_states
        self.n_features = self.config.n_features
        self.dist: Optional[DefaultDistribution] = None

        self._init_enc(encoder=encoder)
        self.to(device=self.device, dtype=DTYPE)

    def _init_enc(self, encoder: Optional[nn.Module] = None) -> None:

        self.context_dim = self.config.context_dim
        self.hidden_dim = self.config.hidden_dim

        if encoder is None:
            hidden_dim = max(32, min(64, self.n_features * 2))
            encoder = DefaultEncoder(
                n_features=self.n_features,
                cnn_channels=self.config.cnn_channels,
                hidden_dim=hidden_dim,
            )
        self.encoder = encoder if isinstance(encoder, ContextEncoder) else ContextEncoder(
            encoder=encoder,
            pool=self.config.pool,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            debug=self.debug
        ).to(device=self.device, dtype=DTYPE)
        self.encoder.eval()

        try:
            dummy = torch.zeros(1, 16, self.n_features)
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

    def _init_dist(self, context: Optional[torch.Tensor] = None, dist: Optional[DefaultDistribution] = None) -> None:

        if dist is not None and isinstance(dist, DefaultDistribution):
            self.dist = dist

        elif self.dist is None:
            self.dist = DefaultDistribution(
                initial=Initial(
                    n_states=self.n_states,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    init_mode=self.config.init_mode,
                ),
                duration=Duration(
                    n_states=self.n_states,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    max_duration=self.config.max_duration,
                    init_mode=self.config.init_mode,
                ),
                transition=Transition(
                    n_states=self.n_states,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    transition_type=self.config.transition_type,
                    init_mode=self.config.init_mode,
                ),
                emission=Emission(
                    n_states=self.n_states,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    min_covar=self.config.min_covar,
                    emission_type=self.config.emission_type,
                    init_mode=self.config.init_mode,
                )
            )

        self.dist.to(device=self.device, dtype=DTYPE)

        try:
            self._params.update(self.dist.initialize(context))
        except Exception as err:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {err}") from err

    def _prepare(self,
        X: torch.Tensor | list,
        context: Optional[torch.Tensor | list] = None,
        mask: Optional[torch.BoolTensor] = None) -> SequenceSet:

        # --- Normalize X into [B, T, F] ---
        if isinstance(X, list):
            if not X:
                raise ValueError("X must contain at least one sequence")
            X = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x) for x in X], batch_first=True)
        elif not torch.is_tensor(X):
            raise TypeError(f"Unsupported X type: {type(X)}")
        if X.ndim == 2:
            X = X.unsqueeze(0)

        B, T, F = X.shape
        if F != self.n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.n_features}, got {F}")

        # --- Normalize context into [B, T, H] if provided ---
        if context is not None:
            if isinstance(context, list):
                context = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(t) for t in context], batch_first=True)
            elif not torch.is_tensor(context):
                raise TypeError(f"Unsupported context type: {type(context)}")
            if context.ndim == 2:
                context = context.unsqueeze(1).expand(B, T, -1)
            canonical = context[:, :1]
        else:
            canonical = None
            context, canonical = self.encoder.encode(
                sequences=X,
                mask=(mask.squeeze(-1) if mask is not None else None),
                detach=False
            )

        # --- Mask ---
        if mask is None:
            mask = torch.ones(B, T, 1, dtype=torch.bool, device=X.device)
        else:
            mask = mask.bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            elif mask.ndim != 3:
                mask = mask.view(B, T, 1)
        lengths = mask.squeeze(-1).sum(dim=1)

        # --- Compute emission log-probabilities ---
        K = self.n_states
        if T == 0:
            log_probs = X.new_empty(B, 0, K)
        else:
            dist = self.dist.emission.forward(context=context, return_dist=True)
            log_probs = dist.log_prob(X.unsqueeze(2).expand(B, T, K, F))
            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        return SequenceSet(
            masks=mask,
            sequences=X,
            lengths=lengths,
            contexts=context,
            canonical=canonical,
            log_probs=log_probs
        )

    def _forward(self,
        X: SequenceSet,
        context: Optional[Union[torch.Tensor, ContextRouter]] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None) -> torch.Tensor:

        router = ContextRouter.from_tensor(X, context=context) if not isinstance(context, ContextRouter) else context
        Dmax = self.dist.duration.max_duration
        B, T, K = router.log_probs.shape[:3]
        device = router.log_probs.device

        initial_logits = self.dist.initial.log_matrix(
            context=router.canonical, temperature=temperature, timestep=timestep
        )  # [B,1,K]
        duration_logits = self.dist.duration.log_matrix(
            context=router.context, temperature=temperature, timestep=timestep
        )  # [B,T,K,Dmax]
        transition_logits = self.dist.transition.log_matrix(
            context=router.context, temperature=temperature, timestep=timestep
        )  # [B,T,K,K]

        # --- Cumulative emission sums ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device)
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)

        # Indices for duration sums
        d_range = torch.arange(1, Dmax + 1, device=device)
        t_range = torch.arange(T, device=device).view(T, 1)
        start_idx = (t_range - d_range + 1).clamp(min=0).view(1, T, 1, Dmax)
        end_idx = (t_range + 1).expand(T, Dmax).view(1, T, 1, Dmax)
        k_idx = torch.arange(K, device=device).view(1, 1, K, 1)
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
        start_idx = start_idx.expand(B, T, K, Dmax)
        end_idx = end_idx.expand(B, T, K, Dmax)
        k_idx = k_idx.expand(B, T, K, Dmax)
        b_idx = b_idx.expand(B, T, K, Dmax)
        emit_sums = cumsum_emit[b_idx, end_idx, k_idx] - cumsum_emit[b_idx, start_idx, k_idx]

        # --- Initialize alpha ---
        alpha = torch.full((B, T, K, Dmax), NEG_INF, device=device)
        alpha[:, 0, :, 0] = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]

        duration_mask = torch.arange(Dmax, device=device).view(1, 1, 1, Dmax) <= torch.arange(T, device=device).view(1, T, 1, 1)

        for t in range(1, T):
            max_d = min(Dmax, t + 1)
            idx_prev = (t - d_range[:max_d]).clamp(min=0)

            alpha_prev = alpha[:, idx_prev, :, :max_d]
            alpha_prev = torch.logsumexp(alpha_prev, dim=-1)

            alpha_trans = torch.logsumexp(
                alpha_prev.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1),
                dim=2
            )
            alpha_trans = alpha_trans.permute(0, 2, 1)

            alpha_t = alpha_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]

            full_alpha = torch.full((B, K, Dmax), NEG_INF, device=device)
            full_alpha[..., :max_d] = alpha_t
            alpha[:, t] = full_alpha * duration_mask[:, t:t+1, :, :]

        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)
        alpha = alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), NEG_INF)

        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        alpha = alpha * mask_exp
        return alpha

    def fit(self,
        X: torch.Tensor | list[torch.Tensor],
        n_init: int = 1, tol: float = 1e-4, max_iter: int = 10,
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        lr: float = 1e-2, verbose: bool = True, use_scheduler: bool = True):

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

            prev_ll = self._reset_parameters(run_idx, context=context)
            params = [
                p
                for name in ["initial", "transition", "duration", "emission"]
                for p in getattr(self.dist, name).parameters()
                if p.requires_grad
            ]

            self._optimizer = torch.optim.Adam(params, lr=lr)
            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, mode="max", factor=0.5, patience=5
                )
                if use_scheduler else None
            )

            for it in range(max_iter):
                self._optimizer.zero_grad()

                seq_set = self._prepare(X, context=context)
                temperature = max(0.5, 1.0 - it / max_iter)
                alpha = self._forward(seq_set, temperature=temperature)  # [B, T, K, D]
                lengths = seq_set.lengths

                log_likelihoods = alpha.new_full((len(seq_set.sequences),), NEG_INF)
                valid = lengths > 0
                if valid.any():
                    last_alpha = alpha[valid, lengths[valid] - 1]  # [N_valid, K, D]
                    log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

                ll = log_likelihoods.sum()
                loss = -ll

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                self._optimizer.step()

                ll_val = ll.item()
                self._convergence.update(ll_val, it, run_idx)

                if scheduler is not None:
                    scheduler.step(ll_val)

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

    def _viterbi(self,
        X: SequenceSet,
        context: Optional[Union[torch.Tensor, ContextRouter]] = None,
        duration_weight: float = 0.0) -> list[torch.Tensor]:

        K = self.n_states
        Dmax = self.dist.duration.max_duration
        predicted: list[torch.Tensor] = []

        router = context if isinstance(context, ContextRouter) else ContextRouter.from_tensor(X, theta=context)
        B, T_max, _ = router.log_probs.shape

        durations_full = torch.arange(1, Dmax + 1)

        for b in range(B):
            L = int(router.mask[b].sum())
            if L == 0:
                predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
                continue

            init_logits = self.dist.initial.log_matrix(context=router.canonical[b:b + 1])[0, 0]
            dur_logits = self.dist.duration.log_matrix(context=router.context[b:b + 1, :L])[0]
            trans_logits = self.dist.transition.log_matrix(context=router.context[b:b + 1, :L])[0]

            if duration_weight != 0.0:
                dur_logits = dur_logits * (1.0 - duration_weight)

            emit_log = router.log_probs[b, :L]
            cumsum_emit = emit_log.new_zeros((L + 1, K))
            cumsum_emit[1:] = torch.cumsum(emit_log, dim=0)

            V = emit_log.new_full((L, K), NEG_INF)
            back_ptr = emit_log.new_full((L, K), -1, dtype=torch.long)
            best_dur = emit_log.new_zeros((L, K), dtype=torch.long)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1

                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T
                scores_dur = dur_logits[t, :, :max_d] + emit_sums

                if t == 0:
                    scores = init_logits[:, None] + scores_dur
                    V[t], idx = scores.max(dim=1)
                    idx = idx.clamp(max=max_d - 1)
                    best_dur[t] = durations[idx]
                    continue

                prev_t = torch.clamp(starts - 1, min=0)
                prev_V = V[prev_t].T.unsqueeze(2)
                trans = trans_logits[t].unsqueeze(1)
                prev_scores = prev_V + trans

                mask_start0 = (starts == 0)
                if mask_start0.any():
                    init_logits_exp = init_logits.view(-1, 1, 1).expand(-1, 1, prev_scores.size(2))
                    prev_scores[:, mask_start0, :] = init_logits_exp

                prev_max, prev_arg = prev_scores.max(dim=0)
                scores = prev_max.T + scores_dur

                V[t], dur_idx = scores.max(dim=1)
                best_dur[t] = durations[dur_idx]
                back_ptr[t] = torch.where(
                    best_dur[t] == 1,
                    back_ptr.new_full((K,), -1),
                    prev_arg[dur_idx, torch.arange(K)]
                )

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
                router.log_probs.new_full((end - start + 1,), st, dtype=torch.long)
                for start, end, st in segments
            ])

            predicted.append(path[:L])

        return predicted

    def score(self,
        X: torch.Tensor | list[torch.Tensor],
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        reduce: bool = False) -> torch.Tensor:

        # --- Normalize X into list[Tensor[T,F]] ---
        if torch.is_tensor(X):
            if X.ndim == 2:
                sequences = [X]
            elif X.ndim == 3:
                sequences = [x for x in X]  # convert batch tensor to list for padding
            else:
                raise ValueError(f"Unsupported X shape {X.shape}")
        elif isinstance(X, list):
            if not X:
                return torch.empty(0)
            sequences = [torch.as_tensor(x) if not torch.is_tensor(x) else x for x in X]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B = len(sequences)
        if B == 0:
            return torch.empty(0)

        # --- Normalize context ---
        context_list: Optional[list[torch.Tensor]] = None
        if context is not None:
            if torch.is_tensor(context):
                if context.ndim == 3:
                    if context.shape[0] != B:
                        raise ValueError("context batch dimension mismatch")
                    context_list = [t for t in context]
                elif context.ndim == 2:
                    context_list = [context] * B
                else:
                    raise ValueError(f"Unsupported context shape {context.shape}")
            elif isinstance(context, list):
                if len(context) != B:
                    raise ValueError("context list length mismatch")
                context_list = [torch.as_tensor(t) if not torch.is_tensor(t) else t for t in context]
            else:
                raise TypeError(f"Unsupported context type: {type(context)}")

        seq_set = self._prepare(sequences, context=context_list)
        alpha = self._forward(seq_set)  # [B, T, K, D]

        lengths = seq_set.lengths
        log_likelihoods = alpha.new_full((B,), NEG_INF)

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

        if getattr(self, "_convergence", None):
            if len(self._convergence.converged_flags) <= run_idx:
                self._convergence.converged_flags.extend([False] * (run_idx + 1 - len(self._convergence.converged_flags)))
            else:
                self._convergence.converged_flags[run_idx] = False

    def _snapshot_best_params(self, context: Optional[torch.Tensor] = None):
        # self._params.update({
            # "initial_dist": self.dist.initial.forward(context=context, return_dist=True),
            # "duration_dist": self.dist.duration.forward(context=context, return_dist=True),
            # "transition_dist": self.dist.transition.forward(context=context, return_dist=True),
            # "emission_dist": self.dist.emission.forward(context=context, return_dist=True),
        # })

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
        duration_weight: float = 0.0, verbose: bool = True) -> list[torch.Tensor] | torch.Tensor:

        seq_set = self._prepare(X, context=context)
        B = len(seq_set.sequences)

        if B == 0 or seq_set.total_timesteps == 0:
            return [torch.empty(0, dtype=torch.long) for _ in range(B)]

        if verbose:
            print(f"[Predict] Sequences: {B}, max_len: {int(seq_set.lengths.max())}")

        router = ContextRouter.from_tensor(seq_set, context=context)

        if algorithm == "viterbi":
            nonzero_indices = [i for i, L in enumerate(seq_set.lengths) if L > 0]

            if len(nonzero_indices) < B:
                idx = torch.as_tensor(nonzero_indices, dtype=torch.long)
                seq_set_nz = seq_set.index_select(idx)
                router_nz = router.select(nonzero_indices)
            else:
                seq_set_nz = seq_set
                router_nz = router

            results: list[torch.Tensor] = [
                torch.empty(0, dtype=torch.long) for _ in range(B)
            ]

            decoded_paths = self._viterbi(
                seq_set_nz,
                context=router_nz,
                duration_weight=duration_weight
            )

            for i, path in zip(nonzero_indices, decoded_paths):
                results[i] = path.to(dtype=torch.long)

            return results

        if algorithm == "score":
            return self.score(X, context=context, reduce=False)

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

