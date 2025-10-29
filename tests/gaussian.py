import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment

from nhsmm.defaults import DTYPE, EPS
from nhsmm.models import GaussianHSMM

# -------------------------
# Synthetic OHLCV generator (all states more distinct)
# -------------------------
def generate_ohlcv(
    n_segments=12,
    seg_len_low=15,
    seg_len_high=40,
    n_features=5,
    rng_seed=42
):
    rng = np.random.default_rng(rng_seed)
    states, obs = [], []

    # Strongly separated means to avoid collapse
    means = [
        np.array([140.0, 145.0, 135.0, 140.0, 2e6]),   # state 0
        np.array([60.0, 65.0, 55.0, 60.0, 2e5]),       # state 1
        np.array([200.0, 210.0, 190.0, 200.0, 5e5]),  # state 2 (moved away from others)
    ]
    cov = np.diag([2.0, 2.0, 2.0, 2.0, 5e4])

    for _ in range(n_segments):
        s = int(rng.integers(0, len(means)))
        L = int(rng.integers(seg_len_low, seg_len_high + 1))
        seg = rng.multivariate_normal(means[s], cov, size=L)
        obs.append(seg)
        states.extend([s] * L)

    return np.array(states), np.vstack(obs)

# -------------------------
# Hungarian permutation alignment
# -------------------------
def best_permutation_accuracy(true, pred, n_classes):
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    return acc, mapped_pred

# -------------------------
# Duration summary
# -------------------------
def print_duration_summary(model):
    with torch.no_grad():
        D = torch.exp(model.duration_logits).cpu().numpy()
    print("\nLearned duration modes (per state):")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    true_states, X = generate_ohlcv(n_segments=12, seg_len_low=15, seg_len_high=40)
    X_torch = torch.tensor(X, dtype=torch.float64)

    n_states = 3
    n_features = X.shape[1]
    max_duration = 40  # slightly higher to accommodate long segments

    print(f"\nConfig: n_states={n_states}, max_duration={max_duration}, n_features={n_features}")

    # Initialize Gaussian HSMM
    model = GaussianHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        k_means=True,
        alpha=1.0,
        seed=0
    )

    # Initialize emissions with mild smoothing
    model.initialize_emissions(
        X_torch,
        method="kmeans",
        smooth_transition=0.05,
        smooth_duration=0.05
    )

    # Fit EM
    print("\n=== Running EM for GaussianHSMM ===")
    import time
    t0 = time.time()
    model.fit(
        X_torch,
        max_iter=50,
        n_init=3,
        sample_D_from_X=True,
        verbose=True,
        tol=1e-4
    )
    elapsed = time.time() - t0

    # Decode using Viterbi
    print("\n=== Decoding ===")
    v_path = model.decode(X_torch, algorithm="viterbi")

    acc, mapped_pred = best_permutation_accuracy(true_states, v_path, n_classes=n_states)
    print(f"Best-permutation accuracy: {acc:.4f}")
    print("Optimally permuted confusion matrix (rows=true, cols=mapped_pred):")
    print(confusion_matrix(true_states, mapped_pred))

    # Extra metrics
    f1 = f1_score(true_states, mapped_pred, average="macro", zero_division=0)
    prec = precision_score(true_states, mapped_pred, average="macro", zero_division=0)
    rec = recall_score(true_states, mapped_pred, average="macro", zero_division=0)
    ll = model.score(X_torch).item()

    print("\nExtra metrics:")
    print(f" F1 (macro): {f1:.4f}")
    print(f" Precision (macro): {prec:.4f}")
    print(f" Recall (macro): {rec:.4f}")
    print(f" Log-likelihood: {ll:.2f}")
    print(f" EM elapsed: {elapsed:.2f}s")

    # Per-state coverage & accuracy
    unique_pred, counts_pred = np.unique(v_path, return_counts=True)
    print("\nPredicted state coverage:")
    for s, c in zip(unique_pred, counts_pred):
        print(f" state {s}: {c} frames ({100*c/len(v_path):.1f}%)")

    print("\nPer-state accuracy (after permutation):")
    for s in range(n_states):
        mask = true_states == s
        acc_s = (mapped_pred[mask] == s).mean()
        print(f" state {s}: {acc_s:.3f}")
        if acc_s < 0.05:
            print(f"  ⚠ State {s} may have collapsed during training!")

    print_duration_summary(model)

    torch.save(model.state_dict(), "gaussianhsmm_debug_state.pt")
    print("\nModel state saved to gaussianhsmm_debug_state.pt")
