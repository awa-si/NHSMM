from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils.rnn import pad_sequence

from nhsmm import Convergence, DistributionSet, DefaultEncoder, ModelConfig
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm.config import DTYPE, EPS, logger, MAX_LOGITS, NEG_INF


class NHSMM(nn.Module):

    def __init__(self, config: ModelConfig, encoder: Optional[nn.Module] = None):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        self.debug: bool = config.debug
        self.dist: Optional[DistributionSet] = None
        self.duration_logits_bias = nn.Parameter(torch.ones(config.n_states, config.max_duration))

        self.initialize_encoder(encoder=encoder)
        self.to(device=self.device, dtype=DTYPE)

    def initialize_encoder(self, encoder: Optional[nn.Module] = None) -> None:
        self.context_dim = self.config.context_dim
        self.hidden_dim = self.config.hidden_dim

        if encoder is None:
            hidden_dim = max(32, min(64, self.config.n_features * 2))
            encoder = DefaultEncoder(
                n_features=self.config.n_features,
                cnn_channels=self.config.cnn_channels,
                cnn_kernel=self.config.cnn_kernel,
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

    def initialize_distributions(self,
        context: Optional[torch.Tensor] = None,
        dist: Optional[DistributionSet] = None) -> None:

        if dist is not None:
            self.dist = dist

        elif self.dist is None:
            self.dist = DistributionSet(
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
            raise RuntimeError(f"Failed to initialize NHSMM PDFs: {err}") from err

    def _build_sequence_set(self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[torch.Tensor] = None) -> SequenceSet:

        X, mask = self._ensure_tensor(X, return_mask=True)  # X: [B,T,F], mask: [B,T]
        B, T, F = X.shape
        device = X.device

        if F != self.config.n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.config.n_features}, got {F}")

        # --- Build context ---
        if context is None:
            context_tensor, canonical = self.encoder.encode(sequences=X, mask=mask)
        else:
            context_tensor = self._ensure_tensor(context)  # [B,T,C] or broadcasted
            if context_tensor.ndim == 2:
                context_tensor = context_tensor.unsqueeze(0).expand(B, T, -1)
            elif context_tensor.ndim == 3:
                if context_tensor.shape[:2] != (B, T):
                    raise ValueError(f"context shape {context_tensor.shape} incompatible with batch {B}, seq {T}")
            else:
                raise ValueError(f"Unsupported context ndim {context_tensor.ndim}")
            canonical = context_tensor[:, :1]

        # --- Compute emission log-probs ---
        K = self.config.n_states
        if T == 0:
            log_probs = X.new_empty(B, 0, K)
        else:
            dist = self.dist.emission.forward(context=context_tensor, return_dist=True)
            X_exp = X.unsqueeze(2).expand(-1, -1, K, -1)  # [B,T,K,F]
            log_probs = dist.log_prob(X_exp)               # [B,T,K]
            log_probs = log_probs.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        return SequenceSet(
            sequences=X,
            lengths=mask.sum(dim=1),
            masks=mask.unsqueeze(-1),
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
            soft_dmax=self.duration_logits_bias,
            temperature=temperature,
            timestep=timestep,
            T=T,
        )
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
        emit_sums = emit_sums.clamp(min=NEG_INF, max=MAX_LOGITS)

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

    def _compute_loss(self,
        X: torch.Tensor, context: torch.Tensor = None,
        loss_bias: float = 1e-3, it: int = 0, max_iter: int = 20,
        t_min: float = 0.5, t_max: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:

        k = 10 / max_iter
        temperature = t_min + (t_max - t_min) / (1 + math.exp(k * (it - max_iter / 2)))

        seq_set = self._build_sequence_set(X, context=context)
        alpha = self.forward(seq_set, temperature=temperature)  # [B, T, K, D]
        lengths = seq_set.lengths

        log_likelihoods = alpha.new_full((len(seq_set.sequences),), NEG_INF)
        valid = lengths > 0
        if valid.any():
            last_alpha = alpha[valid, lengths[valid] - 1]  # [N_valid, K, D]
            log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

        ll = log_likelihoods.sum()

        # --- Duration bias regularization ---
        loss = -ll
        loss += loss_bias * nnF.relu(
            self.duration_logits_bias[:, 1:] - self.duration_logits_bias[:, :-1]
        ).mean()

        return ll, loss

    def optimize(self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 20,
        loss_bias: float = 1e-3,
        lr: float = 1e-2,
        verbose: bool = True,
        use_scheduler: bool = True):

        if self.dist is None:
            raise RuntimeError("Distributions not initialized")

        # convert X and context to padded tensors
        X, context = self._ensure_tensor(X), self._ensure_tensor(context)

        self._convergence = Convergence(
            tol=tol,
            patience=1,
            rel_tol=tol,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
        )

        best_score = -float("inf")
        for run_idx in range(n_init):
            prev_ll = self._initialize_run_state(run_idx, context=context)
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            params = [
                p
                for name in ["initial", "transition", "duration", "emission"]
                for p in getattr(self.dist, name).parameters()
                if p.requires_grad
            ] + [self.duration_logits_bias]

            self._optimizer = torch.optim.Adam(params, lr=lr)
            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, mode="max", factor=0.5, patience=5
                )
                if use_scheduler else None
            )

            for it in range(max_iter):
                self._optimizer.zero_grad()
                ll, loss = self._compute_loss(
                    X, context=context, loss_bias=loss_bias, it=it, max_iter=max_iter
                )
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
        context: Optional[Union[torch.Tensor, ContextRouter]] = None) -> List[torch.Tensor]:

        K = self.config.n_states
        Dmax = self.dist.duration.max_duration

        router = context if isinstance(context, ContextRouter) else ContextRouter.from_tensor(X, context=context)
        B, T_max, _ = router.log_probs.shape
        device = router.log_probs.device

        predicted: List[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, device=device)
        for b in range(B):
            L = int(router.mask[b].sum())
            if L == 0:
                predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
                continue

            initial_logits = self.dist.initial.log_matrix(
                context=router.canonical[b:b + 1], T=L
            )[0, 0]
            duration_logits = self.dist.duration.log_matrix(
                context=router.context[b:b + 1, :L], T=L,
                soft_dmax=self.duration_logits_bias
            )[0]
            transition_logits = self.dist.transition.log_matrix(
                context=router.context[b:b + 1, :L], T=L,
                soft_dmax=self.duration_logits_bias
            )[0]

            emit_log = router.log_probs[b, :L]
            cumsum_emit = torch.zeros((L + 1, K), device=device)
            cumsum_emit[1:] = torch.cumsum(emit_log, dim=0)

            V = torch.full((L, K), NEG_INF, device=device)
            back_ptr = torch.full((L, K), -1, dtype=torch.long, device=device)
            best_dur = torch.zeros((L, K), dtype=torch.long, device=device)

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

    def log_likelihood(self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reduce: bool = False) -> torch.Tensor:

        # --- Ensure tensors and mask ---
        X, context = self._ensure_tensor(X), self._ensure_tensor(context)
        B, T, F = X.shape

        if F != self.config.n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.config.n_features}, got {F}")

        # --- Build SequenceSet and forward ---
        seq_set = self._build_sequence_set(X, context=context)
        alpha = self.forward(seq_set, context=context)  # [B, T, K, D]

        # --- Log-likelihood per sequence ---
        lengths = seq_set.lengths
        log_likelihoods = alpha.new_full((B,), NEG_INF)
        valid = lengths > 0
        if valid.any():
            last_alpha = alpha[valid, lengths[valid]-1]        # [N_valid, K, D]
            log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

        # --- Safety against NaN / Inf ---
        log_likelihoods = torch.nan_to_num(
            log_likelihoods,
            nan=NEG_INF,
            neginf=NEG_INF,
            posinf=MAX_LOGITS
        )

        return log_likelihoods.sum() if reduce else log_likelihoods

    def _initialize_run_state(self, run_idx: int, context: Optional[torch.Tensor] = None) -> None:
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
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mode: Literal["viterbi", "log_likelihood"] = "viterbi",
        verbose: bool = True) -> torch.Tensor | list[torch.Tensor]:

        # --- Ensure X and context are padded tensors ---
        X, context = self._ensure_tensor(X), self._ensure_tensor(context)
        B, T, F = X.shape

        if B == 0 or T == 0:
            return [torch.empty(0, dtype=torch.long, device=X.device) for _ in range(B)]

        if verbose:
            print(f"[Predict] Sequences: {B}, max_len: {T}")

        seq_set = self._build_sequence_set(X, context=context)
        router = ContextRouter.from_tensor(seq_set, context=context)

        # --- Viterbi decoding ---
        if mode == "viterbi":
            results = [torch.empty(0, dtype=torch.long, device=X.device) for _ in range(B)]
            nonzero_idx = torch.nonzero(seq_set.lengths, as_tuple=False).squeeze(-1)

            if len(nonzero_idx) > 0:
                seq_set_nz = seq_set.select(nonzero_idx)
                router_nz = router.select(nonzero_idx)
                decoded_paths = self._viterbi(seq_set_nz, context=router_nz)

                for i, path in zip(nonzero_idx.tolist(), decoded_paths):
                    results[i] = path.to(dtype=torch.long)
            return results

        if mode == "log_likelihood":
            return self.log_likelihood(X, context=context, reduce=False)

        raise ValueError(f"Unsupported decoding mode '{mode}'")

    def decode(self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mode: Literal["viterbi"] = "viterbi",
        first_only: bool = True, verbose: bool = True) -> Union[torch.Tensor, list[torch.Tensor]]:

        B = X.shape[0] if X.ndim == 3 else 1
        if verbose:
            logger.debug(f"[decode] mode={mode}, batch_size={B}")

        preds = self.predict(X, mode=mode, context=context, verbose=verbose)
        return preds[0] if first_only and B == 1 else preds

    def _ensure_tensor(self,
        X: Union[torch.Tensor, List[torch.Tensor], None],
        pad_value: float = 0.0, return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor], None]:

        if X is None:
            return (None, None) if return_mask else None

        device = self.duration_logits_bias.device

        if torch.is_tensor(X):
            if X.ndim == 2:  # [T, F] -> batch of 1
                X = X.unsqueeze(0)
            elif X.ndim != 3:
                raise ValueError(f"Unsupported X shape {X.shape}")
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=device)
            return (X.to(device), mask) if return_mask else X.to(device)

        if isinstance(X, list):
            if not X:
                if return_mask:
                    return torch.empty(0, 0, 0, device=device), torch.empty(0, 0, dtype=torch.bool, device=device)
                return torch.empty(0, 0, 0, device=device)

            X_tensors = [torch.as_tensor(x, device=device) for x in X]
            lengths = [x.shape[0] for x in X_tensors]
            X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=pad_value)
            mask = torch.zeros(X_padded.shape[:2], dtype=torch.bool, device=device)
            for i, L in enumerate(lengths):
                mask[i, :L] = 1
            return (X_padded, mask) if return_mask else X_padded

        raise TypeError(f"Unsupported type: {type(X)}")



