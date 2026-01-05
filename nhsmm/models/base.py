from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from nhsmm import Convergence, DefaultDistribution, DefaultEncoder, HSMMConfig
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm.config import DTYPE, EPS, logger, MAX_LOGITS, NEG_INF


class HSMM(nn.Module):

    def __init__(self, config: HSMMConfig, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        self.debug = config.debug
        self.dist: Optional[DefaultDistribution] = None
        self.soft_dmax = nn.Parameter(torch.ones(config.n_states, config.max_duration))

        self.init_enc(encoder=encoder)
        self.to(device=self.device, dtype=DTYPE)

    def init_enc(self, encoder: Optional[nn.Module] = None) -> None:
        self.context_dim = self.config.context_dim
        self.hidden_dim = self.config.hidden_dim

        if encoder is None:
            hidden_dim = max(32, min(64, self.config.n_features * 2))
            encoder = DefaultEncoder(
                n_features=self.config.n_features,
                cnn_channels=self.config.cnn_channels,
                hidden_dim=hidden_dim,
            )
        self.encoder = encoder if isinstance(encoder, ContextEncoder) else ContextEncoder(
            encoder=encoder,
            pool=self.config.pool,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
        ).to(device=self.device, dtype=DTYPE)

        try:
            self.encoder.eval()
            dummy = torch.zeros(1, 16, self.config.n_features)
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

    def init_dist(self,
        context: Optional[torch.Tensor] = None,
        dist: Optional[DefaultDistribution] = None) -> None:

        if dist is not None:
            self.dist = dist

        elif self.dist is None:
            self.dist = DefaultDistribution(
                initial=Initial(
                    n_states=self.config.n_states,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    init_mode=self.config.init_mode,
                ),
                duration=Duration(
                    n_states=self.config.n_states,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    max_duration=self.config.max_duration,
                    init_mode=self.config.init_mode,
                ),
                transition=Transition(
                    n_states=self.config.n_states,
                    n_features=self.config.n_features,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    transition_type=self.config.transition_type,
                    max_duration=self.config.max_duration, # if None, standard HMM
                    init_mode=self.config.init_mode,
                ),
                emission=Emission(
                    n_states=self.config.n_states,
                    n_features=self.config.n_features,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    min_covar=self.config.min_covar,
                    emission_type=self.config.emission_type,
                    init_mode=self.config.init_mode,
                )
            )


        try:
            self.dist.to(device=self.device, dtype=DTYPE)
            self.dist.initialize(context)
        except Exception as err:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {err}") from err

    def _prepare_data(self,
        X: torch.Tensor | list[torch.Tensor],
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        mask: Optional[torch.BoolTensor] = None) -> SequenceSet:

        if isinstance(X, list):
            if len(X) == 0:
                raise ValueError("X must contain at least one sequence")
            X = [x if x.ndim > 1 else x.unsqueeze(-1) for x in X]
            X = torch.nn.utils.rnn.pad_sequence(
                X, batch_first=True, padding_value=0.0
            )
        elif torch.is_tensor(X):
            if X.ndim == 2:
                X = X.unsqueeze(0)
            elif X.ndim != 3:
                raise ValueError(f"X must be [B,T,F] or list of [T,F], got {X.shape}")
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B, T, F = X.shape

        if F != self.config.n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.n_features}, got {F}")

        if mask is None:
            mask_tensor = torch.ones(B, T, 1, dtype=torch.bool)
            lengths = torch.full((B,), T, dtype=torch.long)
        else:
            if not torch.is_tensor(mask):
                raise TypeError("mask must be a torch.Tensor")

            mask_tensor = mask.bool()
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(-1)
            if mask_tensor.ndim != 3:
                raise ValueError(f"mask must be [B,T] or [B,T,1], got {mask.shape}")
            if mask_tensor.shape[:2] != (B, T) or mask_tensor.shape[2] != 1:
                raise ValueError(
                    f"mask incompatible with X: expected [B,T,1]=({B},{T},1), got {mask_tensor.shape}"
                )
            lengths = mask_tensor.squeeze(-1).sum(dim=1)

        if context is None:
            context_tensor, canonical = self.encoder.encode(sequences=X, mask=mask_tensor.squeeze(-1))
        else:
            if isinstance(context, list):
                if len(context) != B:
                    raise ValueError(f"context list length {len(context)} != batch size {B}")

                ctx = []
                for i, c in enumerate(context):
                    if c is None:
                        tmp = X[i:i+1]
                    else:
                        if not torch.is_tensor(c):
                            raise TypeError("context elements must be tensors or None")

                        if c.ndim == 2:
                            tmp = c.unsqueeze(0)
                        elif c.ndim == 3:
                            tmp = c
                        else:
                            raise ValueError(f"Unsupported context ndim {c.ndim}")

                        if tmp.shape[1] == 1:
                            tmp = tmp.expand(1, T, -1)

                    if tmp.shape[1] != T:
                        raise ValueError(
                            f"context time mismatch at batch {i}: {tmp.shape[1]} != {T}"
                        )
                    ctx.append(tmp)
                context_tensor = torch.cat(ctx, dim=0)

            elif torch.is_tensor(context):
                if context.ndim == 2:
                    context_tensor = context.unsqueeze(0).expand(B, T, -1)
                elif context.ndim == 3:
                    if context.shape[:2] != (B, T):
                        raise ValueError(
                            f"context shape {context.shape} incompatible with (B,T)=({B},{T})"
                        )
                    context_tensor = context
                else:
                    raise ValueError(f"Unsupported context ndim {context.ndim}")
            else:
                raise TypeError(f"Unsupported context type: {type(context)}")

            canonical = context_tensor[:, :1]

        # ---------- emissions ----------
        K = self.config.n_states

        if T == 0:
            log_probs = X.new_empty(B, 0, K)
        else:
            dist = self.dist.emission.forward(context=context_tensor, return_dist=True)
            log_probs = dist.log_prob(X.unsqueeze(2).expand(-1, -1, K, -1))
            log_probs = log_probs.masked_fill(~mask_tensor, float("-inf"))

        return SequenceSet(
            sequences=X,
            lengths=lengths,
            masks=mask_tensor,
            contexts=context_tensor,
            canonical=canonical,
            log_probs=log_probs
        )

    def forward(self,
        X: SequenceSet,
        context: Optional[Union[torch.Tensor, ContextRouter]] = None,
        temperature: Optional[float] = None, timestep: Optional[int] = None) -> torch.Tensor:

        router = ContextRouter.from_tensor(X, context=context) if not isinstance(context, ContextRouter) else context
        Dmax = self.dist.duration.max_duration
        B, T, K = router.log_probs.shape[:3]
        device = router.log_probs.device

        kwargs = dict(
            soft_dmax=self.soft_dmax,
            temperature=temperature,
            timestep=timestep,
            T=T,
        )
        with torch.autocast(device_type=device.type):
            initial_logits = self.dist.initial.log_matrix(context=router.canonical, **kwargs)       # [B,1,K]        
            duration_logits = self.dist.duration.log_matrix(context=router.context, **kwargs)       # [B,T,K,Dmax]
            transition_logits = self.dist.transition.log_matrix(context=router.context, **kwargs)   # [B,T,K,K]

        # --- Cumulative emission sums ---
        cumsum_emit = torch.zeros((B, T + 1, K), device=device)
        cumsum_emit[:, 1:] = torch.cumsum(router.log_probs, dim=1)  # [B, T+1, K]

        # Duration range
        d_range = torch.arange(1, Dmax + 1, device=device)  # [Dmax]
        # Create [T, Dmax] indices for start and end
        t_range = torch.arange(T, device=device).unsqueeze(1)  # [T,1]
        start_idx = (t_range - d_range + 1).clamp(min=0)      # [T,Dmax]
        end_idx = t_range + 1                                 # [T,1] -> will broadcast

        # Expand to [B, T, K, Dmax] via broadcasting
        start_idx = start_idx.unsqueeze(0).unsqueeze(2).expand(B, T, K, Dmax)  # [B,T,K,Dmax]
        end_idx = end_idx.unsqueeze(0).unsqueeze(2).expand(B, T, K, Dmax)      # [B,T,K,Dmax]

        # Expand cumsum_emit for gather: [B,T+1,K] -> [B,T+1,K,1]
        cumsum_expand = cumsum_emit.unsqueeze(-1).expand(B, T+1, K, Dmax)
        emit_sums = cumsum_expand.gather(1, end_idx) - cumsum_expand.gather(1, start_idx)  # [B,T,K,Dmax]

        # --- Initialize alpha tensor ---
        alpha = torch.full((B, T, K, Dmax), NEG_INF, device=device)

        # Compute alpha[:, 0, :, :Dmax] in a vectorized way
        max_d0 = min(Dmax, T)
        alpha[:, 0, :, :max_d0] = (
            initial_logits.squeeze(1).unsqueeze(-1)      # [B, K, 1]
            + duration_logits[:, 0, :, :max_d0]         # [B, K, max_d0]
            + emit_sums[:, 0, :, :max_d0]              # [B, K, max_d0]
        )

        # --- Duration mask ---
        d_idx = torch.arange(1, Dmax + 1, device=device).view(1, 1, 1, Dmax)  # [1,1,1,Dmax]
        t_idx = torch.arange(T, device=device).view(1, T, 1, 1)              # [1,T,1,1]
        duration_mask = d_idx <= (t_idx + 1)                                  # [1,T,1,Dmax]

        # Combine with router sequence mask and expand
        duration_mask = duration_mask.expand(B, T, K, Dmax) & router.mask.unsqueeze(-1)

        for t in range(1, T):
            max_d = min(Dmax, t + 1)
            valid_d = d_idx[0,0,0,:max_d]  # [max_d]
            idx_prev = (t - valid_d).clamp(min=0)  # [max_d]

            alpha_prev = alpha[:, idx_prev, :, :max_d]  # [B, max_d, K, max_d]
            if self.dist.transition.max_duration is None:
                # standard HMM transitions: [B, T, K, K]
                alpha_prev = torch.logsumexp(alpha_prev, dim=-1)
                alpha_trans = torch.logsumexp(alpha_prev.unsqueeze(-1) + transition_logits[:, t], dim=2)
            else:
                # duration-dependent transitions: [B, T, K, D, K]
                alpha_prev = alpha_prev  # shape already [B, max_d, K, D]
                trans_t = transition_logits[:, t, :, :max_d, :]  # [B, K, D, K]
                alpha_trans = torch.logsumexp(alpha_prev.unsqueeze(-1) + trans_t.unsqueeze(1), dim=(2,3))

            # --- Permute and add duration + emission logits ---
            alpha_trans = alpha_trans.permute(0, 2, 1)  # [B, K, max_d]
            alpha_t = alpha_trans + duration_logits[:, t, :, :max_d] + emit_sums[:, t, :, :max_d]

            # --- Allocate full alpha for current timestep ---
            full_alpha = torch.full((B, K, Dmax), NEG_INF, device=device)
            full_alpha[..., :max_d] = alpha_t

            # --- Update alpha for current timestep and enforce duration mask ---
            alpha[:, t] = full_alpha
            alpha[:, t] = alpha[:, t].masked_fill(~duration_mask[:, t], NEG_INF)

        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)
        alpha = alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), NEG_INF)
        return alpha

    def fit(self,
        X: torch.Tensor | list[torch.Tensor],
        n_init: int = 1, tol: float = 1e-4, max_iter: int = 20,
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        loss_bias: float = 1e-4, lr: float = 1e-2, verbose: bool = True, use_scheduler: bool = True):

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
            ] + [self.soft_dmax]

            self._optimizer = torch.optim.Adam(params, lr=lr)
            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, mode="max", factor=0.5, patience=5
                )
                if use_scheduler else None
            )

            for it in range(max_iter):
                self._optimizer.zero_grad()

                seq_set = self._prepare_data(X, context=context)
                temperature = max(0.5, 1.0 - it / max_iter)
                alpha = self.forward(seq_set, temperature=temperature)  # [B, T, K, D]
                lengths = seq_set.lengths

                log_likelihoods = alpha.new_full((len(seq_set.sequences),), NEG_INF)
                valid = lengths > 0
                if valid.any():
                    last_alpha = alpha[valid, lengths[valid] - 1]  # [N_valid, K, D]
                    log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

                ll = log_likelihoods.sum()
                loss = -ll + loss_bias * nnF.relu(
                    self.soft_dmax[:, 1:] - self.soft_dmax[:, :-1]
                ).mean()

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
        context: Optional[Union[torch.Tensor, ContextRouter]] = None) -> list[torch.Tensor]:

        K = self.config.n_states
        Dmax = self.dist.duration.max_duration

        router = context if isinstance(context, ContextRouter) else ContextRouter.from_tensor(X, theta=context)
        B, T_max, _ = router.log_probs.shape

        predicted: list[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, device=router.log_probs.device)
        for b in range(B):
            L = int(router.mask[b].sum())
            if L == 0:
                predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
                continue

            with torch.autocast(device_type=router.log_probs.device.type):
                initial_logits = self.dist.initial.log_matrix(
                    context=router.canonical[b:b + 1], T=L
                )[0, 0]
                duration_logits = self.dist.duration.log_matrix(
                    context=router.context[b:b + 1, :L], T=L,
                    soft_dmax=self.soft_dmax
                )[0]
                transition_logits = self.dist.transition.log_matrix(
                    context=router.context[b:b + 1, :L], T=L,
                    soft_dmax=self.soft_dmax
                )[0]

            emit_log = router.log_probs[b, :L]
            cumsum_emit = torch.zeros((L + 1, K), device=emit_log.device)
            cumsum_emit[1:] = torch.cumsum(emit_log, dim=0)

            V = torch.full((L, K), NEG_INF, device=emit_log.device)
            back_ptr = torch.full((L, K), -1, dtype=torch.long, device=emit_log.device)
            best_dur = torch.zeros((L, K), dtype=torch.long, device=emit_log.device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1

                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts.clamp_min(0)]).T
                scores_dur = duration_logits[t, :, :max_d] + emit_sums

                if t == 0:
                    scores = initial_logits[:, None] + scores_dur
                    V[t], idx = scores.max(dim=1)
                    best_dur[t] = durations[idx]
                    continue

                prev_t = torch.clamp(starts - 1, min=0)
                if self.dist.transition.max_duration is None:
                    prev_scores = V[prev_t].T.unsqueeze(2) + transition_logits[t].unsqueeze(1)
                else:
                    prev_scores = V[prev_t].T.unsqueeze(2) + transition_logits[t, :, :max_d, :]

                mask_start0 = (starts == 0)
                if mask_start0.any():
                    init_logits_exp = initial_logits.view(-1, 1, 1).expand(-1, 1, prev_scores.size(2))
                    prev_scores[:, mask_start0, :] = init_logits_exp

                prev_max, prev_arg = prev_scores.max(dim=0)
                scores = prev_max.T + scores_dur

                V[t], dur_idx = scores.max(dim=1)
                best_dur[t] = durations[dur_idx]
                back_ptr[t] = prev_arg[dur_idx, torch.arange(K)]

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
        context: Optional[torch.Tensor | list[torch.Tensor]] = None, reduce: bool = False) -> torch.Tensor:

        if torch.is_tensor(X):
            if X.ndim == 2:
                sequences = [X]
            elif X.ndim == 3:
                sequences = [x for x in X]
            else:
                raise ValueError(f"Unsupported X shape {X.shape}")
        elif isinstance(X, list):
            if not X:
                return torch.empty(0)
            sequences = [torch.as_tensor(x) if not torch.is_tensor(x) else x for x in X]
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        B = len(sequences)
        if B == 0: return torch.empty(0)

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

        seq_set = self._prepare_data(sequences, context=context_list)
        alpha = self.forward(seq_set, context=context_list)  # [B, T, K, D]

        lengths = seq_set.lengths
        log_likelihoods = alpha.new_full((B,), NEG_INF)

        mask = lengths > 0
        if mask.any():
            last_alpha = alpha[mask, lengths[mask] - 1]  # [N_valid, K, D]
            log_likelihoods[mask] = torch.logsumexp(last_alpha.flatten(1), dim=1)

        log_likelihoods = torch.nan_to_num(
            log_likelihoods,
            nan=NEG_INF,
            neginf=NEG_INF,
            posinf=MAX_LOGITS
        )
        return log_likelihoods.sum() if reduce else log_likelihoods

    def _reset_parameters(self, run_idx: int, context: Optional[torch.Tensor] = None) -> None:
        for name in ["initial", "transition", "duration", "emission"]:
            module = getattr(self.dist, name)
            if getattr(self, "_best_state", None) and name in self._best_state:
                module.load_state_dict(self._best_state[name])
            else:
                module.initialize(context=context)

        if self.encoder is not None:
            if getattr(self, "_best_state", None) and "encoder" in self._best_state:
                self.encoder.load_state_dict(self._best_state["encoder"])
            else:
                self.encoder.reset()

        if getattr(self, "_convergence", None):
            if len(self._convergence.converged_flags) <= run_idx:
                self._convergence.converged_flags.extend([False] * (run_idx + 1 - len(self._convergence.converged_flags)))
            else:
                self._convergence.converged_flags[run_idx] = False

    def _snapshot_best_params(self, context: Optional[torch.Tensor] = None):
        self._best_state = {
            name: getattr(self.dist, name).state_dict()
            for name in ["initial", "duration", "transition", "emission"]
        }
        if self.encoder is not None:
            self._best_state["encoder"] = self.encoder.state_dict()

    def _restore_best_params(self):
        if not hasattr(self, "_best_state"):
            raise RuntimeError("No best parameters have been snapshotted")

        if self.encoder is not None and "encoder" in self._best_state:
            self.encoder.load_state_dict(self._best_state["encoder"])

        for name in ["initial", "transition", "duration", "emission"]:
            module = getattr(self.dist, name)
            if self._best_state.get(name):
                module.load_state_dict(self._best_state[name])

    def predict(self,
        X: torch.Tensor | list[torch.Tensor],
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        algorithm: Literal["viterbi", "score"] = "viterbi", verbose: bool = True) -> list[torch.Tensor] | torch.Tensor:

        seq_set = self._prepare_data(X, context=context)
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

            decoded_paths = self._viterbi(seq_set_nz, context=router_nz)

            for i, path in zip(nonzero_indices, decoded_paths):
                results[i] = path.to(dtype=torch.long)

            return results

        if algorithm == "score":
            return self.score(X, context=context, reduce=False)

        raise ValueError(f"Unsupported decoding algorithm '{algorithm}'")

    def decode(self,
        X: torch.Tensor | list[torch.Tensor],
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        algorithm: Literal["viterbi"] = "viterbi", first_only: bool = True, verbose: bool = True) -> torch.Tensor | list[torch.Tensor]:

        if verbose:
            B = len(X) if isinstance(X, list) else X.shape[0] if X.ndim == 3 else 1
            logger.debug(f"[decode] algorithm={algorithm}, batch_size={B}")

        preds = self.predict(X, algorithm=algorithm, context=context, verbose=verbose)
        return preds[0] if first_only and preds else preds

