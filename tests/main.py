
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import polars as pl
from typing import Optional, Dict, Tuple

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
from nhsmm.context import CNN_LSTM_Encoder
from nhsmm.models import HSMM

DEFAULT_RNG_SEED = 0
DEFAULT_LABELS = ["range", "bull", "bear"]

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


def load_ohlcv_tensor(
    data_dir: str,
    symbol: str,
    max_rows: int = 3000,
    timeframe: str = "5m",
    feature_cols: list[str] = ["open", "high", "low", "close"],
    state_col: str = "state",
    default_labels: list[str] = DEFAULT_LABELS,
    rng_seed: int = DEFAULT_RNG_SEED,):
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


def best_permutation_accuracy(
    true: np.ndarray | list,
    pred: torch.Tensor,
    n_classes: int,
    label_map: Optional[Dict[int, str]] = None) -> Tuple[float, np.ndarray, Dict[int, int], Dict[str, str]]:

    true = np.array(true)
    pred = pred.detach().cpu().numpy()

    # Compute confusion matrix
    C = confusion_matrix(true, pred, labels=np.arange(n_classes))
    
    # Solve assignment problem for best matching
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Apply mapping to predictions
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = float((mapped_pred == true).mean())

    # Build readable mapping if label_map provided
    readable = {}
    if label_map:
        for m, t in mapping.items():
            readable[f"model_{m} ({label_map.get(m, m)})"] = f"true_{t} ({label_map.get(t, t)})"
    else:
        readable = mapping.copy()

    return acc, mapped_pred, mapping, readable


def print_duration_summary(model):
    with torch.no_grad():
        # log_matrix returns [B, T, K, Dmax]; squeeze to [K, Dmax]
        log_D = model.duration_module.log_matrix()
        if log_D.ndim > 2:
            log_D = log_D.squeeze(0).squeeze(0)
        D = torch.exp(log_D).cpu().numpy()

        # Check for Gaussian-like log_var for variance
        V = getattr(model.duration_module, "log_var", None)
        if V is not None:
            V = torch.exp(V).cpu().numpy()  # variance in original scale

    print("Learned duration statistics:")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        if V is not None:
            var_dur = float((np.arange(1, len(row) + 1) ** 2 * V[i]).sum())
            print(f" state {i}: mode={mode}, mean={mean_dur:.2f}, var={var_dur:.2f}")
        else:
            print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    MAX_ITER = 9
    MAX_DURATION = 30
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
    hidden_dim = max(32, min(64, n_features * 2))
    encoder = CNN_LSTM_Encoder(n_features=n_features, cnn_channels=5, hidden_dim=hidden_dim)

    # --- Initialize HSMM ---
    model = HSMM(
        encoder=encoder,
        n_states=n_states,
        n_features=n_features,
        emission_type="gaussian",
        max_duration=MAX_DURATION,
        seed=DEFAULT_RNG_SEED,
        min_covar=1e-6,
        alpha=1.0,
    )
    print("[Init] Duration logits differentiated per state.")

    t0 = time.time()
    print("\n=== EM Training ===")
    model.fit(X_torch, n_init=3, tol=1e-4, max_iter=MAX_ITER, verbose=True)
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
    print("\n ===== Duration summary =====")
    print_duration_summary(model)

    # --- State occupancy & transition diagnostics ---
    with torch.no_grad():
        # ---- Initial distribution ----
        log_init = model.initial_module.log_matrix()    # [B,T,K]
        log_init = log_init.mean(dim=(0, 1))             # [K]

        init = torch.softmax(log_init, dim=-1).cpu().numpy()

        print("\nInitial state distribution:")
        for i, p in enumerate(init):
            print(f"  {i:02d} ({label_map[i]}): {p:.4f}")

        # ---- Transition matrix ----
        log_trans = model.transition_module.log_matrix()  # [B,T,K,K]
        log_trans = log_trans.mean(dim=(0, 1))             # [K,K]

        trans = torch.softmax(log_trans, dim=-1).cpu().numpy()

        print("\nTransition matrix (row = from, col = to):")
        for i, row in enumerate(trans):
            row_fmt = " ".join(f"{v:8.4f}" for v in row)
            print(f"  {i:02d} ({label_map[i]:>6})  {row_fmt}")

        row_sums = trans.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            print("\n[WARN] Transition rows not normalized:")
            for i, s in enumerate(row_sums):
                print(f"  {i:02d} ({label_map[i]}): sum={s:.6f}")


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
    # torch.save(model.state_dict(), "hsmm_debug_state.pt")
    print("\n✅ Model state saved to gaussianhsmm_debug_state.pt")

