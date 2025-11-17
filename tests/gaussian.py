import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import polars as pl

from typing import Optional

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from nhsmm.constants import DEBUG, DTYPE, EPS, logger
from nhsmm.models import HSMM

DEFAULT_RNG_SEED = 0
DEFAULT_LABELS = ["range", "bull", "bear"]

# ============================================================
# Synthetic OHLCV generator (robust 2D output)
# ============================================================
def generate_ohlcv(n_segments=12, seg_len_low=15, seg_len_high=40, rng_seed=None):
    rng_seed = DEFAULT_RNG_SEED if rng_seed is None else rng_seed
    rng = np.random.default_rng(rng_seed)
    states, obs = [], []

    # Distinct means for each regime/state (keep order consistent with DEFAULT_LABELS)
    means = [
        np.array([140, 145, 135, 140, 2e6]),  # range
        np.array([200, 210, 190, 200, 5e5]),  # bull
        np.array([60, 65, 55, 60, 2e5]),      # bear
    ]
    cov = np.diag([2.0, 2.0, 2.0, 2.0, 5e4])

    for _ in range(n_segments):
        s = int(rng.integers(0, len(means)))
        L = int(rng.integers(seg_len_low, seg_len_high + 1))
        seg = rng.multivariate_normal(means[s], cov, size=L)
        seg = np.atleast_2d(seg)
        obs.append(seg)
        states.extend([s] * L)

    X_np = np.vstack(obs) if obs else np.empty((0, len(means[0])))
    states_arr = np.array(states, dtype=int)
    label_map = {i: lbl for i, lbl in enumerate(DEFAULT_LABELS)}
    return states_arr, X_np, label_map

# ============================================================
# Data Loading
# ============================================================
def load_ohlcv_tensor(
    data_dir: str,
    symbol: str,
    max_rows: int = 3000,
    timeframe: str = "5m",
    feature_cols: list[str] = ["open", "high", "low", "close"],
    state_col: str = "state",
    default_labels: list[str] = DEFAULT_LABELS,
    rng_seed: int = DEFAULT_RNG_SEED,
):
    """
    Load OHLCV data from a feather/ipc file. Optional state labels are encoded.
    If missing, synthetic/default label_map is used.
    """
    symbol_sanitized = symbol.replace("/", "_").replace(":", "_")
    filename = f"{symbol_sanitized}-{timeframe}-futures.feather"
    path = os.path.join(data_dir, filename)

    if os.path.exists(path):
        df = pl.read_ipc(path, memory_map=False).sort("date")[:max_rows]
        X = torch.tensor(df.select(feature_cols).to_numpy(), dtype=DTYPE)

        if state_col in df.columns:
            encoder = LabelEncoder()
            true_states = encoder.fit_transform(df[state_col].to_list())
            label_map = {i: lbl for i, lbl in enumerate(encoder.classes_)}
            logger.info(f"Loaded {len(df)} rows with provided state column.")
        else:
            true_states = None
            label_map = {i: lbl for i, lbl in enumerate(default_labels)}
            logger.info(f"No state column found — using default label map: {label_map}")
    else:
        # fallback synthetic data
        true_states, X_np, label_map = generate_ohlcv(rng_seed=rng_seed)
        X = torch.tensor(X_np, dtype=DTYPE)
        true_states = np.array(true_states)
        logger.info(f"No data file found — using synthetic data with label map: {label_map}")

    return X, true_states, label_map

# ============================================================
# Hungarian permutation alignment
# ============================================================
def best_permutation_accuracy(true, pred, n_classes, label_map=None):
    true = np.array(true)
    pred = np.array(pred)
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    readable = (
        {
            f"model_{m} ({label_map.get(m, m)})": f"true_{t} ({label_map.get(t, t)})"
            for m, t in mapping.items()
        }
        if label_map
        else mapping
    )
    return acc, mapped_pred, mapping, readable

# ============================================================
# Duration and variance summary
# ============================================================
def print_duration_summary(model):
    with torch.no_grad():
        D = torch.exp(model.duration_module.log_matrix()).cpu().numpy()
        V = (
            torch.exp(model.duration_module.log_var()).cpu().numpy()
            if hasattr(model.duration_module, "log_var")
            else None
        )

    logger.info("Learned duration statistics:")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        if V is not None:
            var_dur = float((np.arange(1, len(row) + 1) ** 2 * V[i]).sum())
            logger.info(f" state {i}: mode={mode}, mean={mean_dur:.2f}, var={var_dur:.2f}")
        else:
            logger.info(f" state {i}: mode={mode}, mean={mean_dur:.2f}")

# -------------------------
# CNN+LSTM Encoder
# -------------------------
class CNN_LSTM_Encoder(nn.Module):
    """
    CNN + LSTM feature encoder, fully future-safe.

    Input:
        x: [B, T, F]
        mask: optional [B, T] (1 for valid, 0 for padding)

    Output:
        if return_mode="sequence": [B, T, out_dim]
        if return_mode="last":     [B, out_dim]

    Notes:
        - _context always holds pooled sequence-level representation [B, out_dim]
        - Fully compatible with ContextEncoder wrapper
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 16,
        cnn_channels: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
        return_mode: str = "sequence",  # "sequence" or "last"
    ):
        super().__init__()
        assert return_mode in ("sequence", "last")
        self.return_mode = return_mode
        self._context: Optional[torch.Tensor] = None

        # CNN
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=padding)
        self.cnn_norm = nn.LayerNorm(cnn_channels)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, F]
            mask: optional [B, T], 1 for valid, 0 for padding

        Returns:
            [B, T, out_dim] if return_mode="sequence"
            [B, out_dim] if return_mode="last"
        """
        B, T, F_in = x.shape
        if T == 0:
            raise ValueError("Input sequence has zero length")

        # ---- CNN ----
        x_cnn = x.transpose(1, 2)          # [B, F, T]
        x_cnn = F.relu(self.conv1(x_cnn))  # [B, C, T]
        x_cnn = x_cnn.transpose(1, 2)      # [B, T, C]
        x_cnn = self.cnn_norm(x_cnn)
        x_cnn = self.dropout(x_cnn)

        # ---- LSTM ----
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x_cnn, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x_cnn)
        out = self.dropout(out)

        # ---- Pooled context ----
        if mask is not None:
            mask_f = mask.unsqueeze(-1)
            pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)
        else:
            pooled = out.mean(dim=1)
        self._context = pooled  # always [B, out_dim]

        # ---- Return ----
        if self.return_mode == "last":
            if mask is not None:
                idx = mask.sum(dim=1).clamp_min(1) - 1
                return out[torch.arange(B), idx]  # last valid timestep per sequence
            return out[:, -1, :]
        return out

    def get_context(self, detach: bool = True) -> Optional[torch.Tensor]:
        """Return pooled context [B, out_dim]"""
        if self._context is None:
            return None
        return self._context.detach() if detach else self._context

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    MAX_ITER = 3
    MAX_DURATION = 50
    SYMBOL = "BTC/USDT:USDT"
    DATA_DIR = "/opt/trader/user_data/data/bybit/futures_"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load or generate data ---
    X, true_states, label_map = load_ohlcv_tensor(DATA_DIR, SYMBOL)
    if X.numel() == 0:
        raise RuntimeError("No data available after load/generate — aborting.")

    X_torch = X.detach().clone() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=DTYPE)
    n_states = len(label_map)
    n_features = X.shape[1]

    logger.info(f"[Config] n_states={n_states}, n_features={n_features}, max_duration={MAX_DURATION}")

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_torch)
    X_torch = torch.tensor(X_scaled, dtype=DTYPE)

    # Build encoder and NHSMM
    encoder = CNN_LSTM_Encoder(n_features=n_features, cnn_channels=5, hidden_dim=16)
    encoder.to(device)

    # --- Initialize HSMM ---
    model = HSMM(
        # encoder=encoder,
        n_states=n_states,
        n_features=n_features,
        max_duration=MAX_DURATION,
        seed=DEFAULT_RNG_SEED,
        min_covar=1e-3,
        alpha=1.0,
    ).to(device)
    # model.emission_module.initialize(X=X_torch)

    print("[Init] Duration logits differentiated per state.")

    # --- EM Training ---
    print("\n=== EM Training ===")

    t0 = time.time()
    model.fit(
        X_torch,
        n_init=3,
        tol=1e-4,
        max_iter=MAX_ITER,
        verbose=True,
    )
    elapsed = time.time() - t0

    # --- Decode ---
    print("\n=== Decoding ===")
    v_path = model.decode(X_torch, algorithm="viterbi")

    # --- Accuracy metrics ---
    if true_states is not None:
        acc, mapped_pred, mapping, readable = best_permutation_accuracy(
            true_states, v_path, n_classes=n_states, label_map=label_map
        )
        print(f"\nBest-permutation accuracy: {acc:.4f}")
        print("Confusion matrix (permuted):")
        print(confusion_matrix(true_states, mapped_pred))
        print("Mapping (model→true):")
        for k, v in readable.items():
            print(f"  {k} → {v}")

        f1 = f1_score(true_states, mapped_pred, average="macro", zero_division=0)
        prec = precision_score(true_states, mapped_pred, average="macro", zero_division=0)
        rec = recall_score(true_states, mapped_pred, average="macro", zero_division=0)
        ll = model.score(X_torch).item()

        print("\nMetrics:")
        print(f" F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f" Log-likelihood: {ll:.2f} | EM time: {elapsed:.2f}s")
    else:
        print("⚠ No true state labels found — skipping accuracy evaluation.")

    # --- Duration summary ---
    print_duration_summary(model)

    # --- State occupancy & transition diagnostics ---
    with torch.no_grad():
        trans = torch.exp(model.transition_module.log_matrix()).cpu().numpy()
        init = torch.exp(model.initial_module.log_matrix()).cpu().numpy()
        print("\nInitial state distribution:")
        for i, p in enumerate(init):
            print(f"  {label_map[i]}: {p:.3f}")
        print("\nTransition matrix (row=from, col=to):")
        for i, row in enumerate(trans):
            print(f"  {label_map[i]}: {' '.join(f'{v:.3f}' for v in row)}")


        # print("\n----- Duration -----")
        # dur_logits = model.duration_module.log_matrix()
        # print("test: dur_logits.requires_grad", dur_logits.requires_grad)
        # print("test: dur_logits.mean(dim=-1)", dur_logits.mean(dim=-1))

        # print("\n----- Transition -----")
        # transition_logits = model.transition_module.log_matrix()
        # print("test: transition_logits.requires_grad", transition_logits.requires_grad)
        # print("test: transition_logits.mean(dim=-1)", transition_logits.mean(dim=-1))

    # --- Inferred state occupancy ---
    unique, counts = np.unique(v_path, return_counts=True)
    print("\nInferred state occupancies:")
    for s, c in zip(unique, counts):
        print(f"  {label_map[s]}: {c} frames ({c / len(v_path):.1%})")

    # --- Visual inspection ---
    if DEBUG:
        try:
            plt.figure(figsize=(12, 3))
            plt.plot(X[:, 3], color="gray", lw=0.8, label="close")
            plt.scatter(np.arange(len(v_path)), X[:, 3], c=v_path, cmap="viridis", s=6)
            plt.title("HSMM Viterbi decoded regimes")
            plt.legend()
            plt.tight_layout()
            # plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot")

    # --- Save model ---
    # torch.save(model.state_dict(), "gaussianhsmm_debug_state.pt")
    print("\n✅ Model state saved to gaussianhsmm_debug_state.pt")

