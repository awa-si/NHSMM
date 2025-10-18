# tests/tune_neural.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models.neural import NeuralHSMM
from nhsmm.defaults import DTYPE

# -------------------------
# Best permutation accuracy
# -------------------------
def best_permutation_accuracy(true, pred, n_classes):
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    return acc, mapped_pred

# -------------------------
# CNN+LSTM encoder
# -------------------------
class CNN_LSTM_Encoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        cnn_channels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(cnn_channels) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return out[:, -1, :]  # last timestep aggregation

# -------------------------
# Synthetic Gaussian sequence
# -------------------------
def generate_gaussian_sequence(
    n_states=3,
    n_features=1,
    seg_len_range=(5, 20),
    n_segments_per_state=3,
    seed=0,
    noise_scale=0.05,
    context_dim: int | None = None,
):
    rng = np.random.default_rng(seed)
    states_list, X_list, C_list = [], [], []
    base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
    for s in range(n_states):
        for _ in range(n_segments_per_state):
            L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
            noise = rng.normal(scale=noise_scale, size=(L, n_features))
            segment_obs = base_means[s] + noise
            X_list.append(segment_obs)
            states_list.extend([s] * L)
            if context_dim is not None and context_dim > 0:
                ctxt = rng.normal(scale=0.5, size=(L, context_dim))
                C_list.append(ctxt)
    X = np.vstack(X_list)
    states = np.array(states_list)
    if context_dim is not None and context_dim > 0:
        C = np.vstack(C_list)
        return states, torch.tensor(X, dtype=DTYPE), torch.tensor(C, dtype=DTYPE)
    return states, torch.tensor(X, dtype=DTYPE), None

# -------------------------
# Optuna objective
# -------------------------
def objective(trial):
    # --- HSMM & data hyperparameters ---
    n_states = trial.suggest_int("n_states", 2, 6)
    n_features = trial.suggest_int("n_features", 1, 3)
    max_duration = trial.suggest_int("max_duration", 10, 60)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    n_init = trial.suggest_int("n_init", 1, 3)
    k_means_init = trial.suggest_categorical("k_means_init", [True, False])
    context_dim = trial.suggest_int("context_dim", 0, 6)
    noise_scale = trial.suggest_float("noise_scale", 0.01, 0.2)
    duration_weight = trial.suggest_float("duration_weight", 0.0, 0.2)
    n_segments_per_state = trial.suggest_int("n_segments_per_state", 1, 4)

    # --- Encoder hyperparameters ---
    hidden_dim = trial.suggest_int("hidden_dim", 16, 64, step=16)
    cnn_channels = trial.suggest_int("cnn_channels", 8, 32, step=8)
    kernel_size = trial.suggest_int("kernel_size", 1, 5, step=2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    # --- Generate synthetic data ---
    try:
        true_states, X, C = generate_gaussian_sequence(
            n_states=n_states,
            n_features=n_features,
            seg_len_range=(5, 20),
            n_segments_per_state=n_segments_per_state,
            seed=0,
            noise_scale=noise_scale,
            context_dim=context_dim if context_dim > 0 else None,
        )
    except Exception as e:
        print("Data generation failed:", e)
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device=device, dtype=torch.float64)
    if C is not None:
        C = C.to(device=device, dtype=torch.float64)

    # --- Encoder ---
    encoder = CNN_LSTM_Encoder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    # --- Instantiate NeuralHSMM ---
    model = NeuralHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        alpha=alpha,
        seed=0,
        emission_type="gaussian",
        encoder=encoder,
        device=device,
        context_dim=context_dim if context_dim > 0 else None,
        min_covar=1e-6,
    )
    model.to(device)

    # --- Initialize emissions ---
    init_method = "kmeans" if k_means_init else "moment"
    X_init = X.detach().cpu().numpy() if init_method == "kmeans" else X
    try:
        model.initialize_emissions(X_init, method=init_method)
    except Exception as e:
        print("Emission initialization failed:", e)
        return 0.0

    # --- Fit model ---
    fit_kwargs = dict(
        X=X,
        max_iter=20,
        n_init=n_init,
        tol=1e-4,
        verbose=False,
    )
    if C is not None:
        fit_kwargs["context"] = C

    try:
        model.fit(**fit_kwargs)
    except TypeError:
        try:
            model.fit(X, max_iter=20, n_init=n_init, tol=1e-4, verbose=False)
        except Exception as e:
            print("Fit failed:", e)
            return 0.0
    except Exception as e:
        print("Fit failed:", e)
        return 0.0

    # --- Decode ---
    try:
        pred = model.decode(X, context=C, duration_weight=duration_weight, algorithm="viterbi") if C is not None else model.decode(X, duration_weight=duration_weight, algorithm="viterbi")
    except TypeError:
        pred = model.decode(X, duration_weight=duration_weight, algorithm="viterbi")

    # --- Compute best-permutation accuracy ---
    try:
        acc, _ = best_permutation_accuracy(true_states, np.asarray(pred), n_classes=n_states)
    except Exception as e:
        print("Accuracy computation failed:", e)
        return 0.0

    # --- Pruning ---
    trial.report(acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return acc

# -------------------------
# Run Optuna study
# -------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("\nBest trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
