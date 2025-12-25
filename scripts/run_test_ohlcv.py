"""
NHSMM Example: Market Regime Detection on OHLCV Data
====================================================

This script demonstrates:
1. Synthetic or real OHLCV data generation/loading.
2. Contextual HSMM initialization with neural encoder.
3. EM-style training for regime detection.
4. Viterbi decoding and evaluation.
5. Diagnostic inspection of durations, transitions, and occupancy.
6. Optional visualization for debugging.

Dependencies:
- torch, numpy, polars, sklearn, matplotlib
- NHSMM library (https://github.com/awa-si/nhsmm)
"""

import os
import time
import numpy as np
import polars as pl
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from nhsmm.constants import DEBUG, DTYPE, EPS, logger
from nhsmm.context import CNN_LSTM_Encoder
from nhsmm.models import HSMM


DEFAULT_RNG_SEED = 0
DEFAULT_LABELS = ["range", "bull", "bear"]

# -----------------------------
# Synthetic OHLCV generator
# -----------------------------
def generate_ohlcv(
    n_segments: int = 12,
    seg_len_low: int = 15,
    seg_len_high: int = 40,
    rng_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Generate synthetic OHLCV-like sequences with distinct regimes/states.
    Returns:
        states_arr: Array of ground-truth states per frame
        X_np: Observations [T, F]
        label_map: dict mapping state indices to labels
    """
    rng_seed = DEFAULT_RNG_SEED if rng_seed is None else rng_seed
    rng = np.random.default_rng(rng_seed)

    states, obs = [], []

    # Define per-state means (open, high, low, close, volume)
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
        obs.append(seg)
        states.extend([s] * L)

    X_np = np.vstack(obs) if obs else np.empty((0, len(means[0])))
    states_arr = np.array(states, dtype=int)
    label_map = {i: lbl for i, lbl in enumerate(DEFAULT_LABELS)}
    return states_arr, X_np, label_map


# -----------------------------
# Load OHLCV tensor from feather / IPC
# -----------------------------
def load_ohlcv_tensor(
    data_dir: str,
    symbol: str,
    max_rows: int = 3000,
    timeframe: str = "5m",
    feature_cols: list[str] = ["open", "high", "low", "close"],
    state_col: str = "state",
    default_labels: list[str] = DEFAULT_LABELS,
    rng_seed: int = DEFAULT_RNG_SEED) -> Tuple[torch.Tensor, Optional[np.ndarray], Dict[int, str]]:
    """
    Load OHLCV data or generate synthetic if not found.
    Returns:
        X: Torch tensor [T, F]
        true_states: Optional ground-truth states array
        label_map: mapping state indices -> labels
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
        true_states, X_np, label_map = generate_ohlcv(rng_seed=rng_seed)
        X = torch.tensor(X_np, dtype=DTYPE)
        logger.info(f"No data file found — using synthetic data with label map: {label_map}")

    return X, true_states, label_map


# -----------------------------
# Accuracy / permutation metrics
# -----------------------------
def best_permutation_accuracy(
    true: np.ndarray | list,
    pred: torch.Tensor,
    n_classes: int,
    label_map: Optional[Dict[int, str]] = None) -> Tuple[float, np.ndarray, Dict[int, int], Dict[str, str]]:
    """
    Match predicted states to true states via Hungarian algorithm.
    Returns accuracy, remapped predictions, mapping dict, readable mapping.
    """
    true = np.array(true)
    pred = pred.detach().cpu().numpy()

    C = confusion_matrix(true, pred, labels=np.arange(n_classes))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = float((mapped_pred == true).mean())

    readable = {}
    if label_map:
        for m, t in mapping.items():
            readable[f"model_{m} ({label_map.get(m, m)})"] = f"true_{t} ({label_map.get(t, t)})"
    else:
        readable = mapping.copy()

    return acc, mapped_pred, mapping, readable


# -----------------------------
# Duration inspection
# -----------------------------
def print_duration_summary(model: HSMM):
    """
    Prints mean, mode, and optional variance for learned durations per state.
    """
    with torch.no_grad():
        log_D = model.duration_module.log_matrix()
        if log_D.ndim > 2:
            log_D = log_D.squeeze(0).squeeze(0)
        D = torch.exp(log_D).cpu().numpy()

        V = getattr(model.duration_module, "log_var", None)
        if V is not None:
            V = torch.exp(V).cpu().numpy()

    print("Learned duration statistics:")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        if V is not None:
            var_dur = float((np.arange(1, len(row) + 1) ** 2 * V[i]).sum())
            print(f" state {i}: mode={mode}, mean={mean_dur:.2f}, var={var_dur:.2f}")
        else:
            print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")


# -----------------------------
# Visual Diagnostics
# -----------------------------
def plot_hsmm_results(X: torch.Tensor, v_path: np.ndarray, model: HSMM, label_map: Dict[int, str]):
    """
    Plot HSMM decoded regimes, durations, and transition matrix.
    
    Args:
        X: Observations [T, F]
        v_path: Viterbi-decoded state path [T]
        model: trained HSMM model
        label_map: dict mapping state indices -> labels
    """
    T = X.shape[0]
    n_states = len(label_map)

    import matplotlib.pyplot as plt

    # --- 1. Plot decoded regimes over a feature (e.g., close price) ---
    plt.figure(figsize=(14, 4))
    plt.plot(X[:, 3].cpu().numpy(), color="gray", lw=0.8, label="Close Price")
    plt.scatter(np.arange(T), X[:, 3].cpu().numpy(), c=v_path, cmap="viridis", s=8)
    plt.title("HSMM Viterbi Decoded Regimes")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.colorbar(label="State")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2. Plot duration distributions per state ---
    with torch.no_grad():
        log_D = model.duration_module.log_matrix()
        if log_D.ndim > 2:
            log_D = log_D.squeeze(0).squeeze(0)
        D = torch.exp(log_D).cpu().numpy()  # [K, Dmax]

    plt.figure(figsize=(12, 3))
    for i, row in enumerate(D):
        plt.plot(np.arange(1, len(row) + 1), row, lw=2, label=f"{label_map[i]} ({i})")
    plt.title("Learned Duration Distributions per State")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # --- 3. Plot transition matrix heatmap ---
    with torch.no_grad():
        log_trans = model.transition_module.log_matrix()
        if log_trans.ndim > 2:
            log_trans = log_trans.mean(dim=(0, 1))  # [K, K]
        trans = torch.softmax(log_trans, dim=-1).cpu().numpy()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(trans, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, label="Transition Probability")
    plt.title("HSMM Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.xticks(ticks=np.arange(n_states), labels=[label_map[i] for i in range(n_states)], rotation=45)
    plt.yticks(ticks=np.arange(n_states), labels=[label_map[i] for i in range(n_states)])
    plt.tight_layout()
    # plt.show()

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(DEFAULT_RNG_SEED)
    np.random.seed(DEFAULT_RNG_SEED)

    INIT_MAX = 3
    MAX_ITER = 3
    MAX_DURATION = 35
    SYMBOL = "BTC/USDT:USDT"
    DATA_DIR = "/opt/trader/user_data/data/bybit/futures_"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load or generate data ---
    X, true_states, label_map = load_ohlcv_tensor(DATA_DIR, SYMBOL)
    if X.numel() == 0:
        raise RuntimeError("No data available after load/generate — aborting.")

    n_states = len(label_map)
    n_features = X.shape[1]

    logger.info(f"[Config] n_states={n_states}, n_features={n_features}, max_duration={MAX_DURATION}")

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_torch = torch.tensor(X_scaled, dtype=DTYPE)

    # --- Build context encoder ---
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
        modulate_var=True,
        min_covar=1e-6,
        alpha=1.0,
    )
    print("[Init] Duration logits differentiated per state.")

    # --- EM Training ---
    t0 = time.time()
    print("\n=== EM Training ===")
    model.fit(X_torch, n_init=INIT_MAX, tol=1e-4, max_iter=MAX_ITER, verbose=True)
    elapsed = time.time() - t0

    # --- Decode hidden states ---
    print("\n=== Decoding ===")
    v_path = model.decode(X_torch, algorithm="viterbi")

    # --- Evaluate accuracy if labels available ---
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
    print("\n===== Duration summary =====")
    print_duration_summary(model)

    # --- State occupancy & transition diagnostics ---
    with torch.no_grad():
        # ---- Initial state distribution (t=0 only) ----
        log_init = model.initial_module.log_matrix()  # [B,T,K]
        init_probs = torch.softmax(log_init[:, 0], dim=-1).mean(dim=0)  # only timestep 0
        init_probs = init_probs.cpu().numpy().flatten()

        print("\n=== Initial State Distribution ===")
        for i, p in enumerate(init_probs):
            print(f"  {i:02d} ({label_map[i]}): {p:.4f}")
        print(f"  Sum: {init_probs.sum():.4f}")

        # ---- Transition matrix ----
        log_trans = model.transition_module.log_matrix()  # [B,T,K,K]
        log_trans_mean = log_trans.mean(dim=(0, 1))       # [K,K]
        trans_probs = torch.softmax(log_trans_mean, dim=-1).cpu().numpy()

        print("\n=== Transition Matrix (row = from, col = to) ===")
        for i, row in enumerate(trans_probs):
            row_fmt = " ".join(f"{v:8.4f}" for v in row)
            print(f"  {i:02d} ({label_map[i]:>6})  {row_fmt}")
        row_sums = trans_probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            print("\n[WARN] Transition rows not normalized:")
            for i, s in enumerate(row_sums):
                print(f"  {i:02d} ({label_map[i]}): sum={s:.6f}")
        else:
            print("  All transition rows sum to 1 ✅")

        # ---- Duration distributions ----
        dur_logits = model.duration_module.log_matrix()  # [B,T,K,Dmax] or [K,Dmax]
        if dur_logits.ndim > 2:
            dur_logits = dur_logits.mean(dim=(0, 1))
        dur_probs = torch.exp(dur_logits).cpu().numpy()

        print("\n=== Duration Distributions per State ===")
        for i, row in enumerate(dur_probs):
            mode_dur = int(np.argmax(row)) + 1
            mean_dur = float((np.arange(1, len(row)+1) * row).sum())
            print(f"  {label_map[i]:<6} | mode={mode_dur}, mean={mean_dur:.2f}, total_prob={row.sum():.4f}")

    # --- Inferred state occupancy from Viterbi ---
    unique, counts = np.unique(v_path, return_counts=True)
    print("\n=== Inferred State Occupancies ===")
    total_frames = len(v_path)
    for s, c in zip(unique, counts):
        pct = c / total_frames * 100
        print(f"  {label_map[s]:<6}: {c} frames ({pct:.2f}%)")
    print(f"  Total frames: {total_frames}")


    if DEBUG:
        try:
            print("\n=== Visual Diagnostics ===")
            plot_hsmm_results(X_torch, v_path, model, label_map)
        except ImportError:
            print("matplotlib not installed — skipping plots")