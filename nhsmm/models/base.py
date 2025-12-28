# nhsmm/models/base.py

from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from nhsmm.constants import DEBUG, DTYPE, EPS, logger, MAX_LOGITS, NEG_INF, HSMMConfig
from nhsmm.distributions import Initial, Duration, Transition, Emission, DefaultDistribution
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm import Convergence, DefaultEncoder


class HSMM(nn.Module):

    def __init__(self, config: HSMMConfig, encoder: Optional[nn.Module] = None):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(device=self.device, dtype=DTYPE)
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
        )
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

    def _init_dist(self, dist: Optional[DefaultDistribution] = None) -> None:

        if dist is not None:
            self.dist = dist

        elif self.dist is None:
            self.dist = DefaultDistribution(
                initial=Initial(
                    n_states=self.n_states,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                ),
                duration=Duration(
                    n_states=self.n_states,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    max_duration=self.config.max_duration,
                    temperature=self.config.temperature
                ),
                transition=Transition(
                    n_states=self.n_states,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    transition_type=self.config.transition_type,
                    temperature=self.config.temperature
                ),
                emission=Emission(
                    n_states=self.n_states,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    min_covar=self.config.min_covar,
                    modulate_var=self.config.modulate_var,
                    emission_type=self.config.emission_type,
                    temperature=self.config.temperature,
                )
            )

        self.dist.to(device=self.device, dtype=DTYPE)

        try:
            self._params.update(self.dist.initialize())
        except Exception as err:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {err}") from err

    def _prepare(self,
        X: torch.Tensor | list,
        theta: Optional[torch.Tensor | list] = None,
        mask: Optional[torch.BoolTensor] = None) -> SequenceSet:

        if isinstance(X, list):
            if not X:
                raise ValueError("X must contain at least one sequence")
            X = torch.nn.utils.rnn.pad_sequence(
                [x if torch.is_tensor(x) else torch.as_tensor(x) for x in X], batch_first=True
            )
        elif not torch.is_tensor(X):
            raise TypeError(f"Unsupported X type: {type(X)}")

        if X.ndim == 2:
            X = X.unsqueeze(0)

        B, T, F = X.shape
        K = self.n_states

        if F != self.n_features:
            raise ValueError()

        # --- normalize theta ---
        if isinstance(theta, list):
            theta = torch.nn.utils.rnn.pad_sequence(
                [t if torch.is_tensor(t) else torch.as_tensor(t) for t in theta],
                batch_first=True
            )

        # --- mask ---
        if mask is None:
            mask = torch.ones(B, T, 1, dtype=torch.bool, device=X.device)
        else:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            elif mask.ndim != 3:
                mask = mask.view(B, T, 1)
            mask = mask.bool()

        lengths = mask.squeeze(-1).sum(dim=1)

        # --- context handling ---
        if theta is not None:
            if theta.ndim == 2:
                theta = theta.unsqueeze(1).expand(B, T, -1)
            context = theta
            canonical = theta[:, :1]
        else:
            context, canonical = self.encoder.encode(
                sequences=X,
                mask=mask.squeeze(-1),
                detach=False
            )

        # --- emission log-probabilities ---
        if T == 0:
            log_probs = X.new_empty(B, 0, K)
        else:
            dist = self.dist.emission.forward(context=context, return_dist=True)
            X_exp = X.unsqueeze(2).expand(B, T, K, F)
            log_probs = dist.log_prob(X_exp)
            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        if getattr(self, "debug", False):
            print(
                f"[Prepare] X={tuple(X.shape)}, "
                f"context={tuple(context.shape)}, "
                f"canonical={tuple(canonical.shape)}, "
                f"log_probs={tuple(log_probs.shape)}"
            )

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
        Dmax = self.dist.duration.max_duration
        B, T, K = router.log_probs.shape[:3]

        initial_logits = self.dist.initial.log_matrix(context=router.canonical)  # [B,1,K]
        duration_logits = self.dist.duration.log_matrix(context=router.context)  # [B,T,K,Dmax]
        transition_logits = self.dist.transition.log_matrix(context=router.context)  # [B,T,K,K]

        # --- Cumulative emission sums for all durations ---
        cumsum_emit = torch.zeros((B, T + 1, K))
        cumsum_emit[:, 1:, :] = torch.cumsum(router.log_probs, dim=1)

        d_range = torch.arange(1, Dmax + 1)          # [Dmax]
        t_range = torch.arange(T).view(T, 1)         # [T,1]

        # Compute start/end indices [T,Dmax]
        start_idx = (t_range - d_range + 1).clamp(min=0)            # [T,Dmax]
        end_idx = (t_range + 1).expand(-1, Dmax)                  # [T,Dmax]

        start_idx = start_idx.view(1, T, 1, Dmax)                   # [1,T,1,Dmax]
        end_idx = end_idx.view(1, T, 1, Dmax)                     # [1,T,1,Dmax]
        k_idx = torch.arange(K).view(1,1,K,1)    # [1,1,K,1]

        # Broadcast to [B,T,K,Dmax]
        start_idx = start_idx.expand(B, T, K, Dmax)
        end_idx = end_idx.expand(B, T, K, Dmax)
        k_idx = k_idx.expand(B, T, K, Dmax)

        # Use advanced indexing instead of gather
        b_idx = torch.arange(B).view(B,1,1,1).expand(B,T,K,Dmax)
        emit_sums = cumsum_emit[b_idx, end_idx, k_idx] - cumsum_emit[b_idx, start_idx, k_idx]

        # --- Initialize alpha ---
        alpha = torch.full((B, T, K, Dmax), NEG_INF)
        alpha[:, 0, :, 0] = initial_logits.squeeze(1) + duration_logits[:, 0, :, 0] + emit_sums[:, 0, :, 0]

        # Precompute duration masks: mask invalid durations for each timestep
        duration_mask = torch.arange(Dmax).view(1, 1, 1, Dmax) <= torch.arange(T).view(1, T, 1, 1)

        # --- Forward recursion vectorized over durations ---
        for t in range(1, T):
            max_d = min(Dmax, t + 1)

            # --- Indices for previous timesteps ---
            idx_prev = t - d_range[:max_d]       # [max_d]
            idx_prev = idx_prev.clamp(min=0)     # prevent negative indices

            # Gather previous alpha values for all possible durations
            # alpha[:, idx_prev, :, :max_d] -> [B, max_d, K, max_d] due to broadcasting
            alpha_prev = alpha[:, idx_prev, :, :max_d]  
            # Collapse last duration dimension to sum over overlapping contributions
            alpha_prev = torch.logsumexp(alpha_prev, dim=-1)  # [B, max_d, K]

            # Compute alpha contribution from transitions
            # transition_logits[:, t, :, :] -> [B, K, K]
            # alpha_prev.unsqueeze(3) -> [B, max_d, K, 1]
            alpha_trans = torch.logsumexp(alpha_prev.unsqueeze(3) + transition_logits[:, t, :, :].unsqueeze(1), dim=2)
            # [B, max_d, K] -> transpose to [B, K, max_d] for next step
            alpha_trans = alpha_trans.transpose(1, 2)

            # Add duration logits and emission sums
            alpha_t = alpha_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]  # [B,K,max_d]

            # Initialize full alpha slice for this timestep
            full_alpha = torch.full((B, K, Dmax), NEG_INF)
            full_alpha[:, :, :max_d] = alpha_t

            # Apply duration mask to invalidate impossible durations
            full_alpha = torch.where(duration_mask[:, t:t+1, :, :], full_alpha, NEG_INF)

            # Update alpha tensor
            alpha[:, t] = full_alpha

        # Mask out timesteps beyond sequence lengths
        length_mask = torch.arange(T).unsqueeze(0) < X.lengths.unsqueeze(1)  # [B,T]
        alpha = torch.where(length_mask.unsqueeze(-1).unsqueeze(-1), alpha, torch.full_like(alpha, NEG_INF))

        mask_exp = router.mask.bool().unsqueeze(-1).expand(B, T, K, Dmax)
        alpha = torch.where(mask_exp, alpha, torch.full_like(alpha, NEG_INF))
        return alpha

    def _viterbi(self,
        X: SequenceSet,
        theta: Optional[Union[torch.Tensor, ContextRouter]] = None,
        duration_weight: float = 0.0) -> list[torch.Tensor]:

        K = self.n_states
        Dmax = self.dist.duration.max_duration
        predicted: list[torch.Tensor] = []

        router = theta if isinstance(theta, ContextRouter) else ContextRouter.from_tensor(X, theta=theta)
        B, T_max, _ = router.log_probs.shape

        durations_full = torch.arange(1, Dmax + 1)

        for b in range(B):
            L = int(router.mask[b].sum())
            if L == 0:
                predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
                continue

            ctx = router.context[b:b + 1, :L]
            canon = router.canonical[b:b + 1]

            init_logits = self.dist.initial.log_matrix(context=canon)[0, 0]
            dur_logits = self.dist.duration.log_matrix(context=ctx)[0]
            trans_logits = self.dist.transition.log_matrix(context=ctx)[0]

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
                    prev_scores[:, mask_start0, :] = init_logits[:, None]

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

        # --- normalize X into list[Tensor[T,F]] ---
        if torch.is_tensor(X):
            if X.ndim == 2:
                sequences = [X]
            elif X.ndim == 3:
                sequences = list(X)
            else:
                raise ValueError(f"Unsupported X shape {X.shape}")
        elif isinstance(X, list):
            if not X:
                return torch.empty(0)
            sequences = [
                x if torch.is_tensor(x) else torch.as_tensor(x)
                for x in X
            ]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B = len(sequences)
        if B == 0:
            return torch.empty(0)

        # --- normalize theta into list[Tensor[T,H]] or None ---
        context_list: Optional[list[torch.Tensor]] = None
        if theta is not None:
            if torch.is_tensor(theta):
                if theta.ndim == 3:
                    if theta.shape[0] != B:
                        raise ValueError("theta batch dimension mismatch")
                    context_list = list(theta)
                elif theta.ndim == 2:
                    context_list = [theta] * B
                else:
                    raise ValueError(f"Unsupported theta shape {theta.shape}")
            elif isinstance(theta, list):
                if len(theta) != B:
                    raise ValueError("theta list length mismatch")
                context_list = [
                    t if torch.is_tensor(t) else torch.as_tensor(t)
                    for t in theta
                ]
            else:
                raise TypeError(f"Unsupported theta type: {type(theta)}")

        # --- forward algorithm ---
        seq_set = self._prepare(sequences, theta=context_list)
        alpha = self._forward(seq_set)  # [B, T, K, D]

        lengths = seq_set.lengths
        log_likelihoods = alpha.new_full((B,), NEG_INF)

        valid = lengths > 0
        if valid.any():
            last_alpha = alpha[valid, lengths[valid] - 1]  # [N_valid, K, D]
            log_likelihoods[valid] = torch.logsumexp(
                last_alpha.flatten(1), dim=1
            )

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
        duration_weight: float = 0.0, verbose: bool = True) -> list[torch.Tensor] | torch.Tensor:

        seq_set = self._prepare(X, theta=context)
        B = len(seq_set.sequences)

        if B == 0 or seq_set.total_timesteps == 0:
            return [torch.empty(0, dtype=torch.long) for _ in range(B)]

        if verbose:
            print(f"[Predict] Sequences: {B}, max_len: {int(seq_set.lengths.max())}")

        router = ContextRouter.from_tensor(seq_set, theta=context)

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
                theta=router_nz,
                duration_weight=duration_weight
            )

            for i, path in zip(nonzero_indices, decoded_paths):
                results[i] = path.to(dtype=torch.long)

            return results

        if algorithm == "score":
            return self.score(X, theta=context, reduce=False)

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

