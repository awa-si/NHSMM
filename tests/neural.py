import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import warnings

from nhsmm.models.neural import NeuralHSMM, NHSMMConfig
from nhsmm.utilities.loader import SequenceDataset
from nhsmm.defaults import DTYPE

# -------------------------
# Utility: Best-permutation accuracy
# -------------------------
def best_permutation_accuracy(true: np.ndarray, pred: np.ndarray, n_classes: int):
    true_flat = true.ravel()
    pred_flat = pred.ravel()
    unique_labels = np.unique(true_flat)
    
    if len(unique_labels) == 1:
        # Only one class in batch
        return (pred_flat == true_flat).mean(), pred.reshape(true.shape)
    
    C = confusion_matrix(true_flat, pred_flat, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_flat = np.array([mapping.get(p, p) for p in pred_flat], dtype=true_flat.dtype)
    return (mapped_flat == true_flat).mean(), mapped_flat.reshape(true.shape)

# -------------------------
# CNN+LSTM Encoder
# -------------------------
class CNN_LSTM_Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim=16, cnn_channels=8, kernel_size=3, dropout=0.1, bidirectional=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_channels, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1, :])  # last timestep representation

# -------------------------
# Duration summary
# -------------------------
def print_duration_summary(model):
    with torch.no_grad():
        D = torch.exp(model.duration_logits).cpu().numpy()
    print("\nLearned duration modes (per state):")
    for i, row in enumerate(D):
        mode = int(np.argmax(row)) + 1
        mean_dur = float((np.arange(1, len(row)+1) * row).sum())
        print(f" state {i}: mode={mode}, mean={mean_dur:.2f}")

# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------------
    # Hyperparameters
    # -------------------------
    n_states = 3
    n_features = 5
    max_duration = 60
    context_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load OHLCV
    # -------------------------
    SYMBOL = "BTC/USDT:USDT"
    DATA_DIR = "/opt/trader/user_data/data/bybit/futures"
    # dataframe = loader.load_dataframe(DATA_DIR, SYMBOL, "5m")

    # -------------------------
    # Dataset
    # -------------------------
    dataset = SequenceDataset(
        n_states=n_states,
        n_features=n_features,
        seed=0,
        seg_len_range=(8, 30),
        n_segments_per_state=3,
        noise_scale=0.05,
        context_dim=context_dim,
        variable_length=True
    )
    loader = dataset.loader(batch_size=2, shuffle=False)
    first_batch = next(iter(loader))
    X_batch, C_batch, state_batch, lengths = first_batch
    print(f"Loaded batch shapes: X={X_batch.shape}, C={C_batch.shape}, states={state_batch.shape}")

    # -------------------------
    # Model + encoder
    # -------------------------
    encoder = CNN_LSTM_Encoder(n_features).to(device)
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
        context_dim=context_dim
    )
    model = NeuralHSMM(config)
    model.attach_encoder(encoder, batch_first=True, pool="mean", n_heads=4)
    model.to(device)

    # -------------------------
    # Initialization
    # -------------------------
    model.initialize_emissions(X_batch.to(device), method="kmeans")
    with torch.no_grad():
        theta = model.encode_observations(X_batch.to(device))
        pdf = model._contextual_emission_pdf(X_batch.to(device), theta)
        transition = model._contextual_transition_matrix(theta)
        duration = model._contextual_duration_pdf(theta)
        print("θ range:", (theta.min().item(), theta.max().item()))
        print("Shapes:", theta.shape, transition.shape, duration.shape)

    # -------------------------
    # Training
    # -------------------------
    print("\n=== Training NeuralHSMM (batched) ===")
    n_epochs = 3
    for epoch in range(n_epochs):
        for X_pad, C_pad, states_pad, lengths in loader:
            X_pad = X_pad.to(device, dtype=DTYPE)
            C_pad = C_pad.to(device, dtype=DTYPE) if C_pad is not None else None
            lengths_list = lengths.tolist()

            # Flatten sequences
            X_flat = torch.cat([X_pad[i, :lengths_list[i]] for i in range(len(lengths_list))], dim=0)
            C_flat = torch.cat([C_pad[i, :lengths_list[i]] for i in range(len(lengths_list))], dim=0) if C_pad is not None else None

            if C_flat is not None:
                model.encoder.set_context(C_flat)
            else:
                model.encoder.reset_context()

            theta = model.encode_observations(X_flat) if getattr(model, "encoder", None) else None

            model.fit(X_flat, theta=theta, lengths=lengths_list, max_iter=10, verbose=False)

        print(f"Epoch {epoch+1}/{n_epochs} done")

    # -------------------------
    # Decoding
    # -------------------------
    print("\n=== Decoding ===")
    X_test, C_test, states_true, lengths_test = first_batch
    X_test = X_test.to(device)
    C_test = C_test.to(device) if C_test is not None else None
    lengths_list = lengths_test.tolist()

    # Flatten sequences
    X_flat = torch.cat([X_test[i, :lengths_list[i]] for i in range(len(lengths_list))], dim=0)
    C_flat = torch.cat([C_test[i, :lengths_list[i]] for i in range(len(lengths_list))], dim=0) if C_test is not None else None

    if C_flat is not None:
        model.encoder.set_context(C_flat)
    else:
        model.encoder.reset_context()

    theta = model.encode_observations(X_flat) if getattr(model, "encoder", None) else None

    preds_list = model.predict(X_flat, lengths=lengths_list, algorithm="viterbi")
    v_path = torch.cat(preds_list, dim=0).cpu().numpy()

    # Flatten true states
    states_true_flat = torch.cat([states_true[i, :lengths_list[i]] for i in range(len(lengths_list))], dim=0).cpu().numpy()

    # Evaluate accuracy
    acc, mapped_pred = best_permutation_accuracy(states_true_flat, v_path, n_classes=n_states)
    print(f"Best-permutation accuracy: {acc:.4f}")
    print("Confusion matrix (mapped_pred vs true):")
    print(confusion_matrix(states_true_flat, mapped_pred, labels=list(range(n_states))))

    # -------------------------
    # Debug: Detailed learned model parameters
    # -------------------------
    with torch.no_grad():
        print("\n=== Debug: Learned Model Parameters ===\n")

        # Duration distributions
        D = torch.softmax(model.duration_logits, dim=1).cpu()
        print("Duration probabilities per state (top 5 durations + first 10 values):")
        for i, row in enumerate(D):
            mode = int(torch.argmax(row).item()) + 1
            mean_dur = float((torch.arange(1, row.shape[0]+1) * row).sum())
            topk_vals, topk_idx = torch.topk(row, 5)
            print(f" State {i}: mode={mode}, mean={mean_dur:.2f}, top-5 durations={[(idx.item()+1, round(val.item(),4)) for idx,val in zip(topk_idx, topk_vals)]}")
            print(f"  probs[:10]={row[:10].numpy()}")

        # Transition probabilities
        T = torch.softmax(model.transition_logits, dim=1).cpu()
        print("\nTransition probabilities per state:")
        for i, row in enumerate(T):
            top_trans_idx = torch.argsort(row, descending=True)
            arrows = " -> ".join([f"S{idx.item()}({row[idx]:.3f})" for idx in top_trans_idx])
            print(f" State {i}: {arrows}")

        # Emission stats (Gaussian: mean + covariance)
        if hasattr(model, "emissions") and model.emission_type == "gaussian":
            print("\nEmission stats per state (mean + cov diag):")
            for i in range(model.n_states):
                mu = model.emissions[i].mean.cpu().numpy()
                cov_diag = np.diag(model.emissions[i].cov.cpu().numpy())
                print(f" State {i}: mean={np.round(mu,3)}, cov diag={np.round(cov_diag,3)}")

        # Encoder output
        if getattr(model, "encoder", None):
            theta = model.encode_observations(X_batch)
            print("\nEncoder output θ range:", (theta.min().item(), theta.max().item()))
            print("Encoder output θ shape:", theta.shape)


    # -------------------------
    # Duration summary
    # -------------------------
    print_duration_summary(model)

    # -------------------------
    # Save model
    # -------------------------
    torch.save(model.state_dict(), "neuralhsmm_debug_state.pt")
    print("\nModel state saved to neuralhsmm_debug_state.pt")
