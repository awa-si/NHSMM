import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models.neural import NeuralHSMM, NHSMMConfig
from nhsmm.defaults import DTYPE
from nhsmm import constraints

# -------------------------
# Best permutation accuracy
# -------------------------
def best_permutation_accuracy(true, pred, n_classes):
    C = confusion_matrix(true, pred, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(p, p) for p in pred])
    acc = (mapped_pred == true).mean()
    return acc, mapped_pred

# -------------------------
# CNN+LSTM encoder
# -------------------------
class CNN_LSTM_Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32, cnn_channels=16, kernel_size=3,
                 dropout=0.1, bidirectional=True, normalize=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(cnn_channels) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return out[:, -1, :]

# -------------------------
# Synthetic Gaussian sequence
# -------------------------
def generate_gaussian_sequence(n_states=3, n_features=1, seg_len_range=(5,20),
                               n_segments_per_state=3, seed=0, noise_scale=0.05,
                               context_dim=None):
    rng = np.random.default_rng(seed)
    states_list, X_list, C_list = [], [], []
    base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
    for s in range(n_states):
        for _ in range(n_segments_per_state):
            L = int(rng.integers(seg_len_range[0], seg_len_range[1]+1))
            noise = rng.normal(scale=noise_scale, size=(L, n_features))
            segment_obs = base_means[s] + noise
            X_list.append(segment_obs)
            states_list.extend([s]*L)
            if context_dim is not None and context_dim > 0:
                C_list.append(rng.normal(scale=0.5, size=(L, context_dim)))
    X = np.vstack(X_list)
    states = np.array(states_list)
    if context_dim is not None and context_dim > 0:
        C = np.vstack(C_list)
        return states, torch.tensor(X, dtype=DTYPE), torch.tensor(C, dtype=DTYPE)
    return states, torch.tensor(X, dtype=DTYPE), None

# -------------------------
# Robust internal GPU-parallel tuning
# -------------------------
def internal_tune(model, X, n_states, max_duration, hidden_dim, cnn_channels, dropout,
                  bidirectional, trial, n_candidates=8, debug=False):
    device = X.device
    dtype = DTYPE  # float64 enforced globally

    # --- Helper: generate valid transition logits candidates ---
    def init_masked_transition_logits(model, n_candidates, debug=True):
        K = model.n_states
        transition_type = getattr(model, "transition_type", constraints.Transitions.ERGODIC)
        logits_list = []
        for i in range(n_candidates):
            try:
                probs = constraints.sample_transition(model.alpha, K, transition_type, device=device)
                probs = probs.to(dtype=dtype)
                if debug:
                    print(f"[internal_tune][debug] Candidate {i} transition matrix:\n{probs}")
                logits = torch.log(probs.clamp_min(1e-12))
                logits_list.append(logits)
            except Exception as e:
                print(f"[internal_tune][warn] Transition sampling failed for candidate {i}: {e}")
                logits_list.append(torch.zeros((K, K), device=device, dtype=dtype))
        return torch.stack(logits_list, dim=0).to(dtype=dtype, device=device)

    # --- Duration logits normalized ---
    duration_logits_batch = torch.randn(n_candidates, n_states, max_duration, device=device, dtype=dtype)
    duration_logits_batch -= torch.logsumexp(duration_logits_batch, dim=-1, keepdim=True)

    # --- Transition logits strictly valid ---
    transition_logits_batch = init_masked_transition_logits(model, n_candidates, debug=debug)

    # --- Encoder parameters per candidate ---
    encoder_params_batch = []
    for i in range(n_candidates):
        encoder_params_batch.append({
            "hidden_dim": max(8, hidden_dim + trial.suggest_int(f"hidden_delta_{i}", -8, 8)),
            "cnn_channels": max(4, cnn_channels + trial.suggest_int(f"cnn_delta_{i}", -4, 4)),
            "dropout": float(np.clip(dropout + trial.suggest_float(f"dropout_delta_{i}", -0.1, 0.1), 0.0, 0.5)),
            "bidirectional": trial.suggest_categorical(f"bidirectional_opt_{i}", [True, False])
        })

    configs = []
    for i in range(n_candidates):
        configs.append({
            "encoder_params": encoder_params_batch[i],
            "duration_logits": duration_logits_batch[i].to(dtype=dtype, device=device),
            "transition_logits": transition_logits_batch[i].to(dtype=dtype, device=device),
        })

    # --- Try tuning ---
    try:
        scores = model.tune(X, lengths=[len(X)], configs=configs, verbose=debug)
        best_idx = max(scores, key=scores.get)
        best_cfg = configs[best_idx]

        # --- Safe assign helper ---
        def safe_assign(field, value):
            if not hasattr(model, field):
                print(f"[internal_tune][skip] Field '{field}' not found — skipping.")
                return False
            try:
                param = getattr(model, field)
                if value.dtype != dtype:
                    if debug:
                        print(f"[internal_tune][cast] Auto-upcasting {field} from {value.dtype} → {dtype}")
                    value = value.to(dtype=dtype)
                if value.device != device:
                    value = value.to(device)
                if param.shape != value.shape:
                    print(f"[internal_tune][skip] Shape mismatch for '{field}': {param.shape} vs {value.shape}")
                    return False
                # Optional: normalize if not row-stochastic
                if "transition" in field:
                    value = value - torch.logsumexp(value, dim=-1, keepdim=True)
                elif "duration" in field:
                    value = value - torch.logsumexp(value, dim=-1, keepdim=True)
                param.data.copy_(value)
                return True
            except Exception as e:
                print(f"[internal_tune][skip] Could not assign '{field}': {e}")
                return False

        # Assign best params
        safe_assign("duration_logits", best_cfg["duration_logits"])
        safe_assign("transition_logits", best_cfg["transition_logits"])

        # --- Encoder rebuild ---
        enc_params = best_cfg.get("encoder_params", {})
        kernel_size = 3
        if hasattr(model, "encoder") and hasattr(model.encoder, "conv1"):
            kernel_size = getattr(model.encoder.conv1, "kernel_size", (3,))[0]

        try:
            new_encoder = CNN_LSTM_Encoder(
                n_features=model.n_features,
                hidden_dim=enc_params.get("hidden_dim", hidden_dim),
                cnn_channels=enc_params.get("cnn_channels", cnn_channels),
                kernel_size=kernel_size,
                dropout=enc_params.get("dropout", dropout),
                bidirectional=enc_params.get("bidirectional", bidirectional)
            ).to(device, dtype=dtype)
            model.attach_encoder(new_encoder, batch_first=True, pool="mean", n_heads=4)
        except Exception as e:
            print(f"[internal_tune][warn] Encoder rebuild failed: {e}")

        best_score = scores.get(best_idx, float('-inf'))
        print(f"[internal_tune] Applied best candidate #{best_idx} with score={best_score:.4f}")

    except Exception as e:
        print(f"[internal_tune] Failed during tuning: {e}")
        import traceback
        traceback.print_exc()

# -------------------------
# Optuna objective
# -------------------------
def objective(trial):
    n_states = trial.suggest_int("n_states", 2, 6)
    n_features = trial.suggest_int("n_features", 1, 3)
    max_duration = trial.suggest_int("max_duration", 10, 60)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    n_init = trial.suggest_int("n_init", 1, 3)
    k_means_init = trial.suggest_categorical("k_means_init", [True, False])
    context_dim = trial.suggest_int("context_dim", 0, 6)
    noise_scale = trial.suggest_float("noise_scale", 0.01, 0.2)
    duration_weight = trial.suggest_float("duration_weight", 0.0, 0.2)
    n_segments_per_state = trial.suggest_int("n_segments_per_state", 1, 4)

    hidden_dim = trial.suggest_int("hidden_dim", 16, 64, step=16)
    cnn_channels = trial.suggest_int("cnn_channels", 8, 32, step=8)
    kernel_size = trial.suggest_int("kernel_size", 1, 5, step=2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    true_states, X, C = generate_gaussian_sequence(
        n_states=n_states,
        n_features=n_features,
        seg_len_range=(5,20),
        n_segments_per_state=n_segments_per_state,
        seed=0,
        noise_scale=noise_scale,
        context_dim=context_dim if context_dim>0 else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device=device, dtype=DTYPE)
    if C is not None:
        C = C.to(device=device, dtype=DTYPE)

    encoder = CNN_LSTM_Encoder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device, dtype=DTYPE)

    config = NHSMMConfig(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        alpha=alpha,
        seed=0,
        emission_type="gaussian",
        encoder=encoder,
        device=device,
        context_dim=context_dim if context_dim>0 else None,
        min_covar=1e-6
    )

    model = NeuralHSMM(config)
    model.attach_encoder(encoder, batch_first=True, pool="mean", n_heads=4)
    model.to(device)

    try:
        init_method = "kmeans" if k_means_init else "moment"
        X_init = X.detach().cpu().numpy() if init_method=="kmeans" else X
        model.initialize_emissions(X_init, method=init_method)
    except Exception:
        pass

    model.fit(X=X, max_iter=20, n_init=n_init, tol=1e-4, verbose=False, sample_D_from_X=True)
    internal_tune(model, X, n_states, max_duration, hidden_dim, cnn_channels, dropout, bidirectional, trial, n_candidates=8)

    try:
        pred = model.decode(X, context=C, duration_weight=duration_weight, algorithm="viterbi") if C is not None else model.decode(X, duration_weight=duration_weight, algorithm="viterbi")
    except TypeError:
        pred = model.decode(X, duration_weight=duration_weight, algorithm="viterbi")

    acc, _ = best_permutation_accuracy(true_states, np.asarray(pred), n_classes=n_states)
    trial.report(acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return acc

# -------------------------
# Run study
# -------------------------
if __name__=="__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("\nBest trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
