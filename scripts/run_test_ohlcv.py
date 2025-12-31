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
- NHSMM library (https://github.com/awa-si/NHSMM)
"""

import os
import time
import numpy as np
import polars as pl
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as nnF

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from nhsmm import HSMM, HSMMConfig, DefaultDistribution
from nhsmm.config import DTYPE, EPS, logger


DEFAULT_RNG_SEED = 0
DEFAULT_LABELS = ("range", "bull", "bear")

# -----------------------------
# Synthetic OHLCV generator
# -----------------------------
def generate_ohlcv(
    n_segments: int = 10,
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
def generate_pseudo_states(X: np.ndarray, bull_thresh: float = 0.001, bear_thresh: float = -0.001, window: int = 5):
    """
    Generate heuristic pseudo-states from OHLCV data.
    
    Args:
        X: [T, F] array, columns = [open, high, low, close, ...]
        bull_thresh: min normalized gain to call 'bull'
        bear_thresh: max normalized loss to call 'bear'
        window: look-back window for rolling returns
    
    Returns:
        true_states: np.ndarray of shape [T], int-coded states
                     0 = range, 1 = bull, 2 = bear
    """
    close = X[:, 3]  # close price
    returns = (close[window:] - close[:-window]) / close[:-window]  # simple N-bar returns

    true_states = np.zeros(len(close), dtype=int)  # default = range
    # pad first `window` entries with range (0)
    true_states[window:] = np.where(returns > bull_thresh, 1, np.where(returns < bear_thresh, 2, 0))
    
    return true_states

def load_ohlcv_tensor(
    data_dir: str,
    symbol: str,
    max_rows: int = 3000,
    timeframe: str = "5m",
    state_col: str = "state",
    default_labels: list[str] = DEFAULT_LABELS,
    feature_cols: list[str] = ["open", "high", "low", "close", "volume"],
    rng_seed: int = DEFAULT_RNG_SEED) -> Tuple[torch.Tensor, Optional[np.ndarray], Dict[int, str]]:
    """
    Load OHLCV data or generate synthetic if not found.
    Generates pseudo-states if no true states column exists.
    
    Returns:
        X: Torch tensor [T, F]
        true_states: np.ndarray of shape [T] (heuristic if real states missing)
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
            X_np = X.cpu().numpy()
            true_states = generate_pseudo_states(X_np)
            label_map = {0: "range", 1: "bull", 2: "bear"}
            logger.info(f"No state column found — generated pseudo-states with label map: {label_map}")
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
# Main execution
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(DEFAULT_RNG_SEED)
    np.random.seed(DEFAULT_RNG_SEED)

    INIT_MAX = 3
    MAX_ITER = 5
    MAX_DURATION = 30
    SYMBOL = "BTC/USDT:USDT"
    DATA_DIR = "/opt/trader/user_data/data/bybit/futures_"

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

    # --- Create HSMM configuration ---
    config = HSMMConfig(
        n_states=n_states,
        n_features=n_features,
        max_duration=MAX_DURATION,
        emission_type="gaussian",
        seed=DEFAULT_RNG_SEED,
        modulate_var=True,
        min_covar=1e-6,
    )
    # --- Optionally create a custom distribution (or leave None to use defaults) ---
    dist = None  # or pass a pre-built DefaultDistribution()
    # --- Initialize HSMM ---
    model = HSMM(config=config)
    model.init_dist(dist=dist)

    # --- Training ---
    t0 = time.time()
    print("\n=== Model Training ===")
    model.fit(X_torch, n_init=INIT_MAX, tol=1e-5, max_iter=MAX_ITER, verbose=True)
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
        for k, v in readable.items(): print(f"  {k} → {v}")

        f1 = f1_score(true_states, mapped_pred, average="macro", zero_division=0)
        prec = precision_score(true_states, mapped_pred, average="macro", zero_division=0)
        rec = recall_score(true_states, mapped_pred, average="macro", zero_division=0)
        ll = model.score(X_torch).item()

        print("\nMetrics:")
        print(f" F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f" Log-likelihood: {ll:.2f} | EM time: {elapsed:.2f}s")
    else:
        print("⚠ No true state labels found — skipping accuracy evaluation.")

    # --- State occupancy & transition diagnostics ---
    with torch.no_grad():

        # ---- Initial state distribution (t=0 only) ----
        log_init = model.dist.initial.log_matrix()  # [B,T,K]
        init_probs = torch.softmax(log_init[:, 0], dim=-1).mean(dim=0)  # only timestep 0
        init_probs = init_probs.cpu().numpy().flatten()

        print("\n=== Initial Distribution per State ===")
        for i, p in enumerate(init_probs):
            print(f"  {i:02d} ({label_map[i]}): {p:.4f}")
        print(f"  All initial rows sum to {init_probs.sum():.2f} ✅")


        # ---- Duration distributions ----
        dur_logits = model.dist.duration.log_matrix()  # [B,T,K,Dmax] or [K,Dmax]
        if dur_logits.ndim > 2:
            dur_logits = dur_logits.mean(dim=(0, 1))
        dur_probs = torch.exp(dur_logits).cpu().numpy()

        print("\n=== Duration Distributions per State ===")
        for i, row in enumerate(dur_probs):
            mode_dur = int(np.argmax(row)) + 1
            mean_dur = float((np.arange(1, len(row)+1) * row).sum())
            print(f"  {label_map[i]:<6} | mode={mode_dur}, mean={mean_dur:.2f}, total_prob={row.sum():.2f}")

        # ---- Transition matrix ----
        log_trans = model.dist.transition.log_matrix()  # [B,T,K,K] or [B,T,K,D,K]
        if log_trans.ndim == 5:  # [B,T,K,D,K]
            log_trans_mean = log_trans.mean(dim=(0, 1, 3))  # average over batch, time, duration → [K,K]
        else:  # [B,T,K,K]
            log_trans_mean = log_trans.mean(dim=(0, 1))     # average over batch and time → [K,K]
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
            print(f"  All transition rows sum to 1.00 ✅")


    # --- Inferred state occupancy from Viterbi ---
    total_frames = len(v_path)
    unique, counts = np.unique(v_path, return_counts=True)
    print(f"\n=== Inferred State Occupancies ({total_frames} frames) ===")
    for s, c in zip(unique, counts):
        pct = c / total_frames * 100
        print(f"  {label_map[s]:<6}: {c} frames ({pct:.2f}%)")
    print("\n")

