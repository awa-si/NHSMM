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


# ---------------------------------------------------------
# CNN+LSTM encoder
# ---------------------------------------------------------
class CNN_LSTM_Encoder(nn.Module):
    """
    Hybrid CNN–LSTM encoder for sequence feature extraction.

    Designed for compact contextual embeddings from time-series or sequential data.
    The CNN layer captures short-range temporal patterns; the LSTM models longer dependencies.

    Args:
        n_features (int): Number of input features per time step.
        hidden_dim (int): LSTM hidden size.
        cnn_channels (int): Number of convolutional output channels.
        kernel_size (int): Convolution kernel size.
        dropout (float): Dropout probability applied after CNN and LSTM.
        bidirectional (bool): Use bidirectional LSTM for richer context.
        normalize (bool): Apply LayerNorm to stabilize training.
    """

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

        padding = kernel_size // 2  # keep temporal length constant
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.LayerNorm(cnn_channels) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.out_dim = lstm_out_dim  # external reference for downstream modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F)
        Returns:
            torch.Tensor: Encoded feature vector of shape (B, H)
        """
        # CNN block
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.dropout(x)

        # LSTM block
        out, _ = self.lstm(x)
        out = self.dropout(out)

        # Aggregate: last time step (or mean pooling as fallback)
        last_output = out[:, -1, :]
        return last_output

# -------------------------
# Synthetic Gaussian sequence (with optional context)
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

    # per-state base means
    base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))

    for s in range(n_states):
        for _ in range(n_segments_per_state):
            L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
            noise = rng.normal(scale=noise_scale, size=(L, n_features))
            segment_obs = base_means[s] + noise
            X_list.append(segment_obs)
            states_list.extend([s] * L)

            if context_dim is not None and context_dim > 0:
                # simple random context per time-step (can be changed to structured context)
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
    # data/model hyperparameters
    n_states = trial.suggest_int("n_states", 2, 6)
    n_features = trial.suggest_int("n_features", 1, 3)
    max_duration = trial.suggest_int("max_duration", 10, 60)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    n_init = trial.suggest_int("n_init", 1, 3)
    k_means_init = trial.suggest_categorical("k_means_init", [True, False])

    # neural/context hyperparameters to explore
    context_dim = trial.suggest_int("context_dim", 0, 6)
    # training/noise hyperparams for synthetic data
    noise_scale = trial.suggest_float("noise_scale", 0.01, 0.2)
    n_segments_per_state = trial.suggest_int("n_segments_per_state", 1, 4)

    # generate synthetic data (optionally contextual)
    true_states, X, C = generate_gaussian_sequence(
        n_states=n_states,
        n_features=n_features,
        seg_len_range=(5, 20),
        n_segments_per_state=n_segments_per_state,
        seed=0,
        noise_scale=noise_scale,
        context_dim=context_dim if context_dim > 0 else None,
    )

    # device & dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    if C is not None:
        C = C.to(device)

    encoder = CNN_LSTM_Encoder(n_features, hidden_dim=16)

    # instantiate NeuralHSMM
    model = NeuralHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        alpha=alpha,
        seed=0,
        emission_type="gaussian",
        encoder=encoder,            # you can pass an encoder nn.Module here if available
        device=device,
        context_dim=context_dim if context_dim > 0 else None,
        min_covar=1e-6,
    )

    # move model to device/dtype if necessary (NeuralHSMM uses device argument internally)
    model.to(device)

    # initialize emissions
    init_method = "kmeans" if k_means_init else "moment"
    # convert X to cpu/double for kmeans if sklearn expects numpy (we already used torch above)
    X_init = X.detach().cpu().numpy() if init_method == "kmeans" else X
    model.initialize_emissions(X_init if init_method == "kmeans" else X, method=init_method)

    # fit EM - many HSMM implementations accept context in fit, if yours does not, remove context param
    fit_kwargs = dict(
        X=X,
        max_iter=50,
        n_init=n_init,
        tol=1e-4,
        verbose=False,
    )
    # pass context where supported
    try:
        if C is not None:
            fit_kwargs["context"] = C
    except Exception:
        # fail-safe: ignore context if model.fit doesn't accept it
        pass

    # call fit (wrap in try/except to return a poor trial if fit fails)
    try:
        model.fit(**fit_kwargs)
    except TypeError:
        # maybe your fit signature is fit(X, max_iter=..., n_init=..., ...) without context kwarg
        try:
            model.fit(X, max_iter=50, n_init=n_init, tol=1e-4, verbose=False)
        except Exception as e:
            # mark trial as failed (return low score)
            print("Fit failed:", e)
            return 0.0

    # decode: attempt to pass context if supported
    try:
        pred = model.decode(X, context=C, algorithm="viterbi") if C is not None else model.decode(X, algorithm="viterbi")
    except TypeError:
        pred = model.decode(X, algorithm="viterbi")

    # compute best-permutation accuracy
    acc, _ = best_permutation_accuracy(true_states, np.asarray(pred), n_classes=n_states)
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
