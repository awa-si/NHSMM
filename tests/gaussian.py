import os
import time
import torch
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from nhsmm.constants import DEBUG, DTYPE, EPS, logger
from nhsmm.models import GaussianHSMM

# ============================================================
# Synthetic OHLCV generator (robust 2D output)
# ============================================================
def generate_ohlcv(n_segments=12, seg_len_low=15, seg_len_high=40, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    states, obs = [], []

    # Define distinct means for each regime/state
    means = [
        np.array([140, 145, 135, 140, 2e6]),  # range
        np.array([60, 65, 55, 60, 2e5]),      # bear
        np.array([200, 210, 190, 200, 5e5]),  # bull
    ]
    cov = np.diag([2.0, 2.0, 2.0, 2.0, 5e4])

    for _ in range(n_segments):
        s = int(rng.integers(0, len(means)))
        L = int(rng.integers(seg_len_low, seg_len_high + 1))
        seg = rng.multivariate_normal(means[s], cov, size=L)

        # Ensure segment is 2D
        if seg.ndim == 1:
            seg = seg.reshape(1, -1)

        obs.append(seg)
        states.extend([s] * L)

    # Stack all segments safely
    if len(obs) == 0:
        X_np = np.empty((0, len(means[0])))
    else:
        X_np = np.vstack(obs)

    states_arr = np.array(states, dtype=int)
    label_map = {0: "range", 1: "bear", 2: "bull"}

    return states_arr, X_np, label_map


# ============================================================
# Data Loading
# ============================================================
def load_ohlcv_tensor(
    data_dir: str,
    symbol: str,
    max_rows: int = 5000,
    timeframe: str = "5m",
    feature_cols: list[str] = ["open", "high", "low", "close", "volume"],
    state_col: str = "state",
    default_labels: list[str] = ["range", "bull", "bear"],
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
        else:
            true_states = None
            label_map = {i: lbl for i, lbl in enumerate(default_labels)}
            print(f"No state column found — using default label map: {label_map}")
    else:
        # fallback synthetic data
        true_states, X_np, label_map = generate_ohlcv()
        X = torch.tensor(X_np, dtype=DTYPE)
        true_states = np.array(true_states)
        print(f"No data file found — using synthetic data with label map: {label_map}")

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

    if label_map:
        readable = {
            f"model_{m} ({label_map.get(m, m)})": f"true_{t} ({label_map.get(t, t)})"
            for m, t in mapping.items()
        }
    else:
        readable = mapping

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

    print("\nLearned duration statistics:")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        if V is not None:
            var_dur = float((np.arange(1, len(row) + 1) ** 2 * V[i]).sum())
            print(f" state {i}: mode={mode}, mean={mean_dur:.2f}, var={var_dur:.2f}")
        else:
            print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    SYMBOL = "BTC/USDT:USDT"
    DATA_DIR = "/opt/trader/user_data/data/bybit/futures_"

    # --- Load or generate data ---
    X, true_states, label_map = load_ohlcv_tensor(DATA_DIR, SYMBOL)
    X_torch = X.detach().clone() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=DTYPE)
    n_states = len(label_map)
    n_features = X.shape[1]
    max_duration = 50

    print(f"\n[Config] n_states={n_states}, n_features={n_features}, max_duration={max_duration}")

    # --- Initialize HSMM ---
    model = GaussianHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        min_covar=1e-3,
        k_means=True,
        alpha=1.0,
        seed=0,
    )
    # model.emission_module.initialize(X=X_torch)

    # --- EM Training ---
    print("\n=== EM Training ===")

    t0 = time.time()
    model.fit(X_torch, n_init=3, max_iter=9, sample_D_from_X=True, verbose=True, tol=1e-4)
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
    torch.save(model.state_dict(), "gaussianhsmm_debug_state.pt")
    print("\n✅ Model state saved to gaussianhsmm_debug_state.pt")
