import numpy as np
import torch.nn as nn
import torch

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from nhsmm.models import NeuralHSMM

# ---------------------------------------------------------
# Synthetic OHLCV generator
# ---------------------------------------------------------
def generate_ohlcv(n_segments=8, seg_len_low=10, seg_len_high=40, n_features=5, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    states, obs = [], []

    means = [
        np.array([140.0, 145.0, 135.0, 140.0, 2e6]),
        np.array([60.0, 65.0, 55.0, 60.0, 2e5]),
        np.array([95.0, 98.0, 92.0, 95.0, 8e5])
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
    def __init__(self, n_features, hidden_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (B,T,F)
        x = x.transpose(1, 2)       # (B,F,T) for Conv1d
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)       # (B,T,H)
        out, _ = self.lstm(x)
        return out[:, -1, :]        # last time step embedding


# ---------------------------------------------------------
# Duration distribution summary
# ---------------------------------------------------------
def print_duration_summary(model):
    with torch.no_grad():
        D = torch.exp(model.D).cpu().numpy()
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
    X_torch = torch.tensor(X, dtype=torch.float32)

    n_states = 3
    n_features = X.shape[1]
    max_duration = 60

    encoder = CNN_LSTM_Encoder(n_features, hidden_dim=16)

    model = NeuralHSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        alpha=1.0,
        seed=0,
        encoder=encoder
    )

    print("\n=== Training NeuralHSMM ===")
    model.fit(
        X_torch,
        max_iter=50,
        n_init=3,
        sample_B_from_X=True,
        verbose=True,
        tol=1e-4
    )

    print("\n=== Decoding ===")
    v_path = model.predict(X_torch, algorithm="viterbi")[0].numpy()

    acc, mapped_pred = best_permutation_accuracy(true_states, v_path, n_classes=n_states)
    print(f"Best-permutation accuracy: {acc:.4f}")
    print("Confusion matrix (mapped_pred vs true):")
    print(confusion_matrix(true_states, mapped_pred))

    print_duration_summary(model)

    torch.save(model.state_dict(), "neuralhsmm_debug_state.pt")
    print("\nModel state saved to neuralhsmm_debug_state.pt")
