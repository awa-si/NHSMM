import torch
import numpy as np

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models import GaussianHSMM


# ---------------------------------------------------------
# Synthetic OHLCV generator
# ---------------------------------------------------------
def generate_ohlcv(
    n_segments: int = 8,
    seg_len_low: int = 10,
    seg_len_high: int = 40,
    n_features: int = 5,
    rng_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic OHLCV-like data with segment-level regimes."""
    rng = np.random.default_rng(rng_seed)
    states, obs = [], []

    means = [
        np.array([140.0, 145.0, 135.0, 140.0, 2e6]),
        np.array([60.0, 65.0, 55.0, 60.0, 2e5]),
        np.array([95.0, 98.0, 92.0, 95.0, 8e5]),
    ]
    cov = np.diag([2.0, 2.0, 2.0, 2.0, 5e4])

    for _ in range(n_segments):
        s = int(rng.integers(0, len(means)))
        L = int(rng.integers(seg_len_low, seg_len_high + 1))
        seg = rng.multivariate_normal(means[s], cov, size=L)
        obs.append(seg)
        states.extend([s] * L)

    return np.array(states), np.vstack(obs)


# ---------------------------------------------------------
# Label alignment via Hungarian assignment
# ---------------------------------------------------------
def best_permutation_accuracy(true: np.ndarray, pred: np.ndarray, n_classes: int):
    """Align predicted labels with true labels via Hungarian assignment."""
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    # Map predicted class -> best matching true class
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    return acc, mapped_pred


# ---------------------------------------------------------
# Duration distribution summary
# ---------------------------------------------------------
def print_duration_summary(model):
    with torch.no_grad():
        D = torch.exp(model.duration_logits).cpu().numpy()
    print("\nLearned duration modes (per state):")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row) + 1) * row).sum())
        print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    true_states, X = generate_ohlcv(n_segments=10, seg_len_low=8, seg_len_high=30)
    X_torch = torch.tensor(X, dtype=torch.float64)  # GaussianHSMM expects float64

    n_states = 3
    n_features = X.shape[1]
    max_duration = 60

    print(f"\nConfig: n_states={n_states}, max_duration={max_duration}, n_features={n_features}")

    # --- Initialize Gaussian HSMM ---
    model = GaussianHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        k_means=True,
        alpha=1.0,
        seed=0
    )

    # Initialize emission parameters from data (optional if k_means=True)
    model.sample_emission_pdf(X_torch)

    print("\n=== Running EM for GaussianHSMM ===")
    model.fit(
        X_torch,
        max_iter=50,
        n_init=3,
        sample_D_from_X=True,
        verbose=True,
        tol=1e-4
    )

    print("\n=== Decoding ===")
    v_path = model.predict(X_torch, algorithm="viterbi")[0].numpy()

    acc, mapped_pred = best_permutation_accuracy(true_states, v_path, n_classes=n_states)
    print(f"Best-permutation accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=mapped_pred):")
    print(confusion_matrix(true_states, mapped_pred))

    print_duration_summary(model)

    torch.save(model.state_dict(), "gaussianhsmm_debug_state.pt")
    print("\nModel state saved to gaussianhsmm_debug_state.pt")
