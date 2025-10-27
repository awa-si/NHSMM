import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models import GaussianHSMM
from nhsmm.defaults import DTYPE, EPS

# -----------------------------
# Synthetic OHLCV generator
# -----------------------------
def generate_ohlcv(n_segments=8, seg_len_low=10, seg_len_high=40, rng_seed=42):
    """Generate synthetic OHLCV-like data with segment-level regimes."""
    rng = np.random.default_rng(rng_seed)
    means = [
        np.array([140.0, 145.0, 135.0, 140.0, 2e6]),
        np.array([60.0, 65.0, 55.0, 60.0, 2e5]),
        np.array([95.0, 98.0, 92.0, 95.0, 8e5]),
    ]
    cov = np.diag([2.0, 2.0, 2.0, 2.0, 5e4])

    obs, states = [], []
    for _ in range(n_segments):
        s = int(rng.integers(0, len(means)))
        L = int(rng.integers(seg_len_low, seg_len_high + 1))
        obs.append(rng.multivariate_normal(means[s], cov, size=L))
        states.extend([s] * L)

    return np.array(states), np.vstack(obs)


# -----------------------------
# Label alignment via Hungarian assignment
# -----------------------------
def best_permutation_accuracy(true, pred, n_classes):
    """Align predicted labels with true labels via Hungarian assignment."""
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    return acc, mapped_pred


# -----------------------------
# Duration summary
# -----------------------------
def print_duration_summary(model):
    with torch.no_grad():
        D = torch.exp(model.duration_logits).cpu().numpy()
    print("\nLearned duration modes (per state):")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # --- Seeds for reproducibility ---
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Generate synthetic data ---
    true_states, X = generate_ohlcv(n_segments=10, seg_len_low=8, seg_len_high=30)
    X_torch = torch.tensor(X, dtype=DTYPE)
    lengths = [X_torch.shape[0]]

    n_states = 3
    n_features = X.shape[1]
    max_duration = 60

    print(f"\nConfig: n_states={n_states}, max_duration={max_duration}, n_features={n_features}")

    # --- Initialize GaussianHSMM ---
    model = GaussianHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        k_means=True,
        alpha=1.0,
        seed=0
    )

    # --- Initialize emissions from data ---
    print("\n=== Initializing emissions ===")
    model.initialize_emissions(
        X_torch,
        method="kmeans",       # or "moment"
        smooth_transition=1e-2,
        smooth_duration=1e-2
    )

    # --- Fit model ---
    print("\n=== Running EM for GaussianHSMM ===")
    model.fit(
        X_torch,
        lengths=lengths,
        max_iter=30,
        n_init=3,
        sample_D_from_X=True,
        verbose=True,
        tol=EPS,
        theta=None
    )

    # Debug
    with torch.no_grad():
        mus = model._emission_means
        covs = model._emission_covs
    print("Emission means:\n", mus)
    print("Emission covariances:\n", covs)

    with torch.no_grad():
        pi = torch.softmax(model._init_logits, dim=0)
        A = torch.softmax(model._transition_logits, dim=1)
        D = torch.softmax(model._duration_logits, dim=1)

    print("Initial probs:", pi)
    print("Transition matrix:\n", A)
    print("Duration distributions:\n", D)



    # --- Decode ---
    print("\n=== Decoding ===")
    v_path = model.predict(
        X_torch,
        lengths=lengths,
        algorithm="viterbi",
    )[0].numpy()

    # --- Accuracy & confusion matrix ---
    acc, mapped_pred = best_permutation_accuracy(true_states, v_path, n_classes=n_states)
    print(f"Best-permutation accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=mapped_pred):")
    print(confusion_matrix(true_states, mapped_pred))

    # --- Duration summary ---
    print_duration_summary(model)

    # --- Save model ---
    torch.save(model.state_dict(), "gaussianhsmm_debug_state.pt")
    print("\nModel state saved to gaussianhsmm_debug_state.pt")
