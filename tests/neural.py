import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models.neural import NeuralHSMM, NHSMMConfig
from nhsmm.defaults import DTYPE

# -------------------------
# Synthetic Gaussian sequence
# -------------------------
def generate_gaussian_sequence(n_states=3, n_features=1, seg_len_range=(5, 20),
                               n_segments_per_state=3, seed=0, noise_scale=0.05,
                               context_dim: int | None = None):
    rng = np.random.default_rng(seed)
    states_list, X_list, C_list = [], [], []
    base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
    for s in range(n_states):
        for _ in range(n_segments_per_state):
            L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
            noise = rng.normal(scale=noise_scale, size=(L, n_features))
            X_list.append(base_means[s] + noise)
            states_list.extend([s] * L)
            if context_dim and context_dim > 0:
                C_list.append(rng.normal(scale=0.5, size=(L, context_dim)))
    X = np.vstack(X_list)
    states = np.array(states_list)
    if context_dim and context_dim > 0:
        C = np.vstack(C_list)
        return states, torch.tensor(X, dtype=DTYPE), torch.tensor(C, dtype=DTYPE)
    return states, torch.tensor(X, dtype=DTYPE), None

# -------------------------
# Best permutation accuracy
# -------------------------
def best_permutation_accuracy(true, pred, n_classes):
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    return (mapped_pred == true).mean(), mapped_pred

# -------------------------
# CNN+LSTM Encoder
# -------------------------
class DefaultEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim=16, cnn_channels=8, kernel_size=3, dropout=0.1, bidirectional=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return out[:, -1, :]

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
# Main
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_states = 3
    n_features = 5
    context_dim = 2
    max_duration = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic data
    true_states, X, C = generate_gaussian_sequence(
        n_states=n_states, n_features=n_features,
        seg_len_range=(8, 30), n_segments_per_state=3,
        seed=0, noise_scale=0.05, context_dim=context_dim
    )

    X_torch = X.to(device)
    C_torch = C.to(device) if C is not None else None

    # Build encoder and NHSMM
    encoder = DefaultEncoder(n_features, hidden_dim=16)
    encoder.to(device)

    config = NHSMMConfig(
        n_states=n_states,
        n_features=n_features,
        emission_type="gaussian",
        max_duration=max_duration,
        min_covar=1e-6,
        device=device,
        alpha=1.0,
        seed=0,
        encoder=encoder,
        context_dim=context_dim if context_dim > 0 else None
    )

    model = NHSMM(config)
    model.attach_encoder(
        encoder=encoder,
        batch_first=True,  # True if input shape is (B, T, F)
        pool="mean",
        n_heads=4
    )
    model.to(device)

    # Initialize emissions
    model.initialize_emissions(X, method="kmeans")

    # -------------------------
    # Training
    # -------------------------
    print("\n=== Training NeuralHSMM ===")
    fit_kwargs = dict(
        X=X_torch,
        max_iter=50,
        n_init=3,
        sample_D_from_X=True,
        verbose=True,
        tol=1e-4
    )
    model.fit(**fit_kwargs)

    # -------------------------
    # Decoding & evaluation
    # -------------------------
    print("\n=== Decoding ===")
    v_path = model.decode(X_torch, context=C_torch, algorithm="viterbi", duration_weight=0.0)

    acc, mapped_pred = best_permutation_accuracy(true_states, v_path, n_classes=n_states)
    print(f"Best-permutation accuracy: {acc:.4f}")
    print("Confusion matrix (mapped_pred vs true):")
    print(confusion_matrix(true_states, mapped_pred))

    # -------------------------
    # Duration summary
    # -------------------------
    print_duration_summary(model)

    # Save model safely
    torch.save(model.state_dict(), "neuralhsmm_debug_state.pt")
    print("\nModel state saved to neuralhsmm_debug_state.pt")
