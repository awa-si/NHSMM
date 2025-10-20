#!/usr/bin/env python3
"""
Enhanced Optuna tuning script for NeuralHSMM (OHLCV / multi-feature).

Features:
- Configurable regimes/durations
- Multi-feature + optional context support
- Robust internal tuning with candidate sampling
- Picklable trial.user_attrs (state_means, duration_logits, model_state, model_build_params)
- Atomic save/load of best trial + model reconstruction
- Logging and reproducibility (seeds)
"""

import os
import pickle
import logging
import traceback
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models.neural import NeuralHSMM, NHSMMConfig
from nhsmm.utilities import constraints, loader
from nhsmm.defaults import DTYPE

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------------
# Global params
# -------------------------
MODEL_LOAD = False
PATH_TRIAL = "./best_trial.pkl"
PATH_MODEL = "./best_model.pt"

DEFAULT_N_FEATURES = 5
DEFAULT_CONTEXT_DIM = 2
MIN_DURATION = 6
MAX_DURATION = 30
MIN_N_STATES = 3
MAX_N_STATES = 5
DEFAULT_SEED = 0
N_TRIALS = 40

# -------------------------
# Reproducibility
# -------------------------
torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(DEFAULT_SEED)

# -------------------------
# Utilities
# -------------------------
def best_permutation_accuracy(true, pred, n_classes: int) -> Tuple[float, np.ndarray]:
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = float((mapped_pred == true).mean())
    return acc, mapped_pred

# -------------------------
# CNN + LSTM Encoder with Context
# -------------------------
class CNN_LSTM_Encoder(nn.Module):
    """
    CNN + LSTM encoder with optional normalization, dropout, bidirectional LSTM,
    context feature concatenation, and chunked processing for long sequences.
    """
    def __init__(self, n_features: int, hidden_dim: int = 32, cnn_channels: int = 16,
                 kernel_size: int = 3, dropout: float = 0.1, bidirectional: bool = True,
                 normalize: bool = True, context_dim: Optional[int] = None,
                 max_seq_len: Optional[int] = None):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.cnn_channels = cnn_channels
        self.bidirectional = bidirectional
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len
        self.padding = kernel_size // 2

        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=self.padding)
        self.norm = nn.LayerNorm(cnn_channels) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        if context_dim:
            self.out_dim += context_dim  # concat context later

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] tensor of features
            context: [B, T, C] optional context features (broadcast if needed)
        Returns:
            Tensor: [B, out_dim] sequence embeddings
        """
        B, T, F = x.shape

        # Chunked processing for very long sequences
        if self.max_seq_len and T > self.max_seq_len:
            chunks = []
            for start in range(0, T, self.max_seq_len):
                end = start + self.max_seq_len
                chunk_out = self._forward_chunk(x[:, start:end, :])
                chunks.append(chunk_out)
            out = torch.cat(chunks, dim=1)  # [B, T, out_dim]
            out = out[:, -1, :]  # last time step
        else:
            out = self._forward_chunk(x)  # [B, out_dim]

        # Concatenate context if provided
        if context is not None:
            if context.shape[1] == 1 or context.shape[1] != out.shape[1]:
                context_vec = context[:, -1, :]  # take last step
            else:
                context_vec = context
            out = torch.cat([out, context_vec], dim=-1)

        return out

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, F, T]
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)  # [B, T, cnn_channels]
        x = self.norm(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return out[:, -1, :]  # last time step embedding

    def reset_parameters(self):
        """Re-initialize weights for stability"""
        nn.init.kaiming_uniform_(self.conv1.weight, a=np.sqrt(5))
        if self.conv1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.conv1.bias, -bound, bound)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

# -------------------------
# Internal candidate tuning
# -------------------------
def internal_tune(model: NeuralHSMM, X: torch.Tensor, n_states: int, max_duration: int,
                  hidden_dim: int, cnn_channels: int, dropout: float, bidirectional: bool,
                  trial: optuna.trial.Trial, n_candidates: int = 8):
    device = X.device

    def init_transition_logits(n_candidates_local: int):
        K = model.n_states
        logits_list = []
        for i in range(n_candidates_local):
            try:
                probs = constraints.sample_transition(
                    model.alpha, K,
                    getattr(model, "transition_type", constraints.Transitions.SEMI),
                    device=device
                )
                logits_list.append(torch.log(probs.clamp_min(1e-12)))
            except Exception as e:
                logger.warning(f"[internal_tune] Transition logits candidate {i} failed: {e}")
                logits_list.append(torch.zeros((K, K), device=device, dtype=DTYPE))
        return torch.stack(logits_list, dim=0)

    transition_logits_batch = init_transition_logits(n_candidates)
    duration_logits_batch = torch.randn(n_candidates, n_states, max_duration, device=device, dtype=DTYPE)
    duration_logits_batch -= torch.logsumexp(duration_logits_batch, dim=-1, keepdim=True)

    encoder_params_batch = []
    for i in range(n_candidates):
        encoder_params_batch.append({
            "hidden_dim": max(8, hidden_dim + trial.suggest_int(f"hidden_delta_{i}", -8, 8)),
            "cnn_channels": max(4, cnn_channels + trial.suggest_int(f"cnn_delta_{i}", -4, 4)),
            "dropout": float(np.clip(dropout + trial.suggest_float(f"dropout_delta_{i}", -0.1, 0.1), 0.0, 0.5)),
            "bidirectional": trial.suggest_categorical(f"bidirectional_opt_{i}", [True, False])
        })

    configs = [
        {
            "encoder_params": encoder_params_batch[i],
            "duration_logits": duration_logits_batch[i],
            "transition_logits": transition_logits_batch[i],
        }
        for i in range(n_candidates)
    ]

    try:
        scores = model.tune(X, lengths=[len(X)], configs=configs, verbose=False)
        best_idx = max(scores, key=scores.get)
        best_cfg = configs[best_idx]
        logger.info(f"[internal_tune] Candidate {best_idx} applied (score={scores[best_idx]:.4f})")

        for field in ["duration_logits", "transition_logits"]:
            if hasattr(model, field):
                val = best_cfg[field].to(device=device, dtype=DTYPE)
                param = getattr(model, field)
                if param.shape == val.shape:
                    val = val - torch.logsumexp(val, dim=-1, keepdim=True)
                    param.data.copy_(val)

        enc_params = best_cfg["encoder_params"]
        kernel_size = getattr(getattr(model, "encoder", None), "conv1", nn.Conv1d(1,1,3)).kernel_size[0]
        new_encoder = CNN_LSTM_Encoder(
            n_features=model.n_features,
            hidden_dim=enc_params.get("hidden_dim", hidden_dim),
            cnn_channels=enc_params.get("cnn_channels", cnn_channels),
            kernel_size=kernel_size,
            dropout=enc_params.get("dropout", dropout),
            bidirectional=enc_params.get("bidirectional", bidirectional)
        ).to(device, dtype=DTYPE)
        model.attach_encoder(new_encoder, batch_first=True, pool="mean", n_heads=4)

    except Exception as e:
        logger.error(f"[internal_tune] Failed tuning: {e}\n{traceback.format_exc()}")

# -------------------------
# Market tie-breaker
# -------------------------
def market_tie_breaker_from_attrs(trial_obj: optuna.trial.FrozenTrial) -> float:
    means = trial_obj.user_attrs["state_means"]
    durations = trial_obj.user_attrs.get("duration_logits", np.ones_like(means))
    min_dist = np.min([np.linalg.norm(means[i]-means[j]) for i in range(len(means)) for j in range(i+1, len(means))]) \
        if len(means) > 1 else float("inf")
    duration_penalty = np.sum(durations.sum(axis=-1) < 3)
    return float(min_dist - 0.1 * duration_penalty)

# -------------------------
# Load OHLCV
# -------------------------
SYMBOL = "BTC/USDT:USDT"
DATA_DIR = "/opt/trader/user_data/data/bybit/futures"
# dataframe = loader.load_dataframe(DATA_DIR, SYMBOL, "5m")

# -------------------------
# Optuna objective
# -------------------------
def objective(trial: optuna.trial.Trial, dataframe=None) -> float:
    # Hyperparameters
    n_states = trial.suggest_int("n_states", MIN_N_STATES, MAX_N_STATES)
    max_duration = trial.suggest_int("max_duration", MIN_DURATION, MAX_DURATION)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    noise_scale = trial.suggest_float("noise_scale", 0.01, 0.2)
    duration_weight = trial.suggest_float("duration_weight", 0.0, 0.2)
    n_segments_per_state = trial.suggest_int("n_segments_per_state", 1, 4)
    n_init = trial.suggest_int("n_init", 1, 3)
    n_features = DEFAULT_N_FEATURES
    context_dim = DEFAULT_CONTEXT_DIM

    hidden_dim = trial.suggest_int("hidden_dim", 16, 64, step=16)
    cnn_channels = trial.suggest_int("cnn_channels", 8, 32, step=8)
    kernel_size = trial.suggest_int("kernel_size", 1, 5, step=2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    # Generate synthetic or OHLCV sequence
    true_states, X, C = loader.generate_gaussian_sequence(
        n_states=n_states,
        n_features=n_features,
        seg_len_range=(5, 20),
        n_segments_per_state=n_segments_per_state,
        seed=DEFAULT_SEED,
        noise_scale=noise_scale,
        context_dim=context_dim,
        dataframe=dataframe
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(X, list):
        X = torch.cat(X, dim=0)
    X = X.to(device=device, dtype=DTYPE)
    if C is not None:
        if isinstance(C, list):
            C = torch.cat(C, dim=0)
        C = C.to(device=device, dtype=DTYPE)

    encoder = CNN_LSTM_Encoder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device, dtype=DTYPE)

    config = NHSMMConfig(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        alpha=alpha,
        seed=DEFAULT_SEED,
        emission_type="gaussian",
        encoder=encoder,
        device=device,
        context_dim=context_dim,
        min_covar=1e-6
    )

    model = NeuralHSMM(config)
    model.attach_encoder(encoder, batch_first=True, pool="mean", n_heads=4)
    model.to(device)

    try:
        model.initialize_emissions(X.detach().cpu().numpy(), method="kmeans")
    except Exception as e:
        logger.warning(f"[objective] initialize_emissions failed: {e}", exc_info=True)

    try:
        model.fit(X=X, max_iter=20, n_init=n_init, tol=1e-4, verbose=False, sample_D_from_X=True)
    except Exception as e:
        logger.error(f"[objective] model.fit failed: {e}\n{traceback.format_exc()}")
        raise

    internal_tune(model, X, n_states, max_duration, hidden_dim, cnn_channels, dropout, bidirectional, trial)

    try:
        pred = model.decode(X, context=C, duration_weight=duration_weight, algorithm="viterbi")
    except TypeError:
        pred = model.decode(X, duration_weight=duration_weight, algorithm="viterbi")

    acc, _ = best_permutation_accuracy(true_states, np.asarray(pred), n_classes=n_states)

    # Save picklable trial attributes
    try:
        trial.set_user_attr("state_means", model.emission_module.mu.detach().cpu().numpy())
        if hasattr(model, "duration_logits"):
            trial.set_user_attr("duration_logits", model.duration_logits.detach().cpu().numpy())
        trial.set_user_attr("model_state", {k: v.cpu() for k, v in model.state_dict().items()})
        trial.set_user_attr("model_build_params", {
            "n_states": n_states,
            "n_features": n_features,
            "max_duration": max_duration,
            "alpha": alpha,
            "encoder_params": {
                "hidden_dim": hidden_dim,
                "cnn_channels": cnn_channels,
                "kernel_size": kernel_size,
                "dropout": dropout,
                "bidirectional": bidirectional
            },
            "context_dim": context_dim,
            "min_covar": 1e-6
        })
    except Exception as e:
        logger.warning(f"[objective] could not save trial attributes: {e}", exc_info=True)

    trial.report(acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    logger.info(f"[objective] Trial finished (acc={acc:.4f})")
    return acc

# -------------------------
# Safe partial load
# -------------------------
def safe_load_state(model, state_dict):
    model_dict = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped: logger.warning(f"Skipped {len(skipped)} mismatched keys: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    if unexpected: logger.warning(f"Ignored unexpected keys: {unexpected}")
    if missing: logger.warning(f"Missing keys: {missing}")
    return missing, unexpected, skipped

# -------------------------
# Main study runner
# -------------------------
def run_study(n_trials: int = N_TRIALS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_LOAD and os.path.exists(PATH_TRIAL) and os.path.exists(PATH_MODEL):
        logger.info(f"Loading saved trial/model from {PATH_TRIAL}, {PATH_MODEL}")
        with open(PATH_TRIAL, "rb") as f:
            best_trial = pickle.load(f)
        try:
            model_state = torch.load(PATH_MODEL, map_location=device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return best_trial, None

        build = best_trial.get("user_attrs", {}).get("model_build_params")
        if build is None: return best_trial, None
        try:
            enc_p = build["encoder_params"]
            encoder = CNN_LSTM_Encoder(
                n_features=build.get("n_features", DEFAULT_N_FEATURES),
                hidden_dim=enc_p["hidden_dim"],
                cnn_channels=enc_p["cnn_channels"],
                kernel_size=enc_p["kernel_size"],
                dropout=enc_p["dropout"],
                bidirectional=enc_p["bidirectional"]
            ).to(device, dtype=DTYPE)

            cfg = NHSMMConfig(
                n_states=build["n_states"],
                n_features=build["n_features"],
                max_duration=build["max_duration"],
                alpha=build["alpha"],
                seed=DEFAULT_SEED,
                emission_type="gaussian",
                encoder=encoder,
                device=device,
                context_dim=build.get("context_dim"),
                min_covar=build.get("min_covar", 1e-6)
            )
            model = NeuralHSMM(cfg)
            model.attach_encoder(encoder, batch_first=True, pool="mean", n_heads=4)
            safe_load_state(model, model_state)
            logger.info("Model successfully rebuilt and weights loaded.")
            return best_trial, model
        except Exception as e:
            logger.error(f"Failed rebuild: {e}\n{traceback.format_exc()}")
            return best_trial, None

    # Run optimization
    study = optuna.create_study(direction="maximize")
    logger.info("Starting Optuna study...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_value = study.best_value
    best_trials = [t for t in study.trials if t.value == best_value]
    best_trial_obj = max(best_trials, key=market_tie_breaker_from_attrs)

    best_trial = {
        "value": best_trial_obj.value,
        "params": best_trial_obj.params,
        "user_attrs": best_trial_obj.user_attrs
    }

    # Atomic save
    os.makedirs(os.path.dirname(PATH_TRIAL) or ".", exist_ok=True)
    tmp_model = PATH_MODEL + ".tmp"
    tmp_trial = PATH_TRIAL + ".tmp"
    model_state = best_trial_obj.user_attrs.get("model_state", {})

    try:
        torch.save(model_state, tmp_model)
        os.replace(tmp_model, PATH_MODEL)
    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
        if os.path.exists(tmp_model): os.remove(tmp_model)

    try:
        with open(tmp_trial, "wb") as f: pickle.dump(best_trial, f)
        os.replace(tmp_trial, PATH_TRIAL)
    except Exception as e:
        logger.error(f"Failed to save trial: {e}", exc_info=True)
        if os.path.exists(tmp_trial): os.remove(tmp_trial)

    logger.info(f"Saved best trial to {PATH_TRIAL}")
    logger.info(f"Saved model weights to {PATH_MODEL}")
    logger.info(f"Best trial accuracy: {best_trial['value']:.4f}")
    logger.info(f"Params: {best_trial['params']}")

    # Attempt rebuild
    build = best_trial.get("user_attrs", {}).get("model_build_params")
    if build is None: return best_trial, model_state
    try:
        enc_p = build["encoder_params"]
        encoder = CNN_LSTM_Encoder(
            n_features=build.get("n_features", DEFAULT_N_FEATURES),
            hidden_dim=enc_p["hidden_dim"],
            cnn_channels=enc_p["cnn_channels"],
            kernel_size=enc_p["kernel_size"],
            dropout=enc_p["dropout"],
            bidirectional=enc_p["bidirectional"]
        ).to(device, dtype=DTYPE)

        cfg = NHSMMConfig(
            n_states=build["n_states"],
            n_features=build["n_features"],
            max_duration=build["max_duration"],
            alpha=build["alpha"],
            seed=DEFAULT_SEED,
            emission_type="gaussian",
            encoder=encoder,
            device=device,
            context_dim=build.get("context_dim"),
            min_covar=build.get("min_covar", 1e-6)
        )
        model = NeuralHSMM(cfg)
        model.attach_encoder(encoder, batch_first=True, pool="mean", n_heads=4)
        safe_load_state(model, model_state)
        logger.info("Rebuilt model from best trial (safe load successful).")
        return best_trial, model
    except Exception as e:
        logger.error(f"Failed to rebuild best model: {e}\n{traceback.format_exc()}")
        return best_trial, model_state

if __name__ == "__main__":
    run_study()
