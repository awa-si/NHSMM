import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from nhsmm.models.neural import NeuralHSMM, NHSMMConfig
from nhsmm.utilities.loader import SequenceLoader
from nhsmm.defaults import DTYPE


def best_permutation_accuracy(true: np.ndarray, pred: np.ndarray, n_classes: int):
    true_flat = true.ravel()
    pred_flat = pred.ravel()
    
    unique_true, counts_true = np.unique(true_flat, return_counts=True)
    print(f"[Debug] Label distribution: {dict(zip(unique_true, counts_true))}")
    unique_pred, counts_pred = np.unique(pred_flat, return_counts=True)
    print(f"[Debug] Predicted state occupancy: {dict(zip(unique_pred, counts_pred))}")

    if len(unique_true) == 1:
        acc = (pred_flat == true_flat).mean()
        print(f"[Debug] Only one true label ({unique_true[0]}). Raw accuracy: {acc:.4f}")
        return acc, pred.reshape(true.shape)

    C = confusion_matrix(true_flat, pred_flat, labels=list(range(n_classes)))
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_flat = np.array([mapping.get(p, p) for p in pred_flat], dtype=true_flat.dtype)
    acc = (mapped_flat == true_flat).mean()

    print(f"[Debug] State mapping: {mapping}")
    print(f"[Debug] Raw confusion matrix:\n{C}")
    return acc, mapped_flat.reshape(true.shape)


class CNN_LSTM_Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32, cnn_channels=[16,32], kernel_sizes=[3,3],
                 bidirectional=True, dropout=0.1, n_heads=4, pool=None, activation=F.relu):
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes), "CNN channels/kernel sizes must match"
        self.pool = pool
        self.activation = activation
        
        # --- CNN stack ---
        self.convs = nn.ModuleList()
        in_ch = n_features
        for out_ch, k in zip(cnn_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_ch, out_ch, k, padding=k//2))
            in_ch = out_ch
        self.norms = nn.ModuleList([nn.LayerNorm(ch) for ch in cnn_channels])
        self.dropout = nn.Dropout(dropout)

        # --- LSTM ---
        self.lstm = nn.LSTM(cnn_channels[-1], hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        # --- Multi-head temporal attention ---
        self.n_heads = n_heads
        self.attn_proj = nn.Linear(self.out_dim, self.out_dim)
        self.attn_heads = nn.Linear(self.out_dim, n_heads)

    def forward(self, x):
        """
        x: [B, T, F]
        returns: [B, T, H] per-timestep embedding for HSMM
        """
        B, T, F_in = x.shape
        x = x.transpose(1, 2)  # [B, F, T]
        
        # CNN + LayerNorm + residuals
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = self.activation(conv(x))
            x = x.transpose(1,2)  # [B, T, C]
            x = norm(x)
            x = self.dropout(x)
            x = x.transpose(1,2)  # [B, C, T]
            if x.shape == x_res.shape:
                x = x + x_res
        
        x = x.transpose(1,2)  # [B, T, C]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, T, H]

        # Multi-head attention
        attn_scores = self.attn_heads(torch.tanh(self.attn_proj(lstm_out)))  # [B, T, n_heads]
        attn_weights = F.softmax(attn_scores, dim=1)  # softmax over time
        attn_out = torch.einsum("bth,btd->btd", attn_weights, lstm_out)  # weighted sum over time

        # Optional pooling over feature dimension
        if self.pool == "mean":
            out = attn_out.mean(dim=-1)  # [B, T]
        elif self.pool == "max":
            out, _ = attn_out.max(dim=-1)  # [B, T]
        else:
            out = attn_out  # [B, T, H]

        return self.dropout(out)


def print_duration_summary(model, topk=5):
    with torch.no_grad():
        D = torch.exp(model.duration_logits).cpu().numpy()
    print("\n=== Duration Distribution Summary ===\n")
    for i, row in enumerate(D):
        row = row / row.sum()
        mode = int(np.argmax(row)) + 1
        mean = float(np.sum(np.arange(1,len(row)+1)*row))
        std = float(np.sqrt(np.sum(((np.arange(1,len(row)+1)-mean)**2)*row)))
        entropy = float(-np.sum(row*np.log(row+1e-12)))
        top_idx = np.argsort(row)[::-1][:topk]
        topk_str = [f"{d+1}({row[d]:.4f})" for d in top_idx]
        warn = ""
        if entropy > 3.5:
            warn = "[Warn] High entropy: nearly uniform duration distribution!"
        elif entropy < 0.5:
            warn = "[Note] Low entropy: duration sharply peaked."
        print(f"State {i}: mode={mode}, mean±std={mean:.2f}±{std:.2f}, entropy={entropy:.4f} {warn}")
        print(f"  Top-{topk} durations: {topk_str}")
        print(f"  Sum(probabilities): {row.sum():.4f}")
        print(f"  First 10 probs    : {np.round(row[:10],5)}\n")
    print("=====================================\n")

if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)

    n_states = 3
    context_dim = 3
    max_duration = 20
    feature_cols = ['open','high','low','close','volume','return','volatility','trend']
    n_features = len(feature_cols)  # OHLCV + return + volatility + trend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SYMBOL = "BTC/USDT:USDT"
    DATA_DIR = "/opt/trader/user_data/data/bybit/futures"

    dataset = SequenceLoader(
        n_states=n_states,
        n_features=n_features,
        seg_len_range=(10, 100),
        n_segments_per_state=100,
        noise_scale=0.05,
        context_dim=context_dim,
        variable_length=True,
        normalize=True,
        device=device,
        data_dir=DATA_DIR,
        pair=SYMBOL,
        timeframe="15m",
        feature_cols=feature_cols,
        max_chunk=20000,
        seed=0,
    )

    loader = dataset.loader(batch_size=8, shuffle=False)

    # --- fetch first batch ---
    batch = next(iter(loader))
    if len(batch) == 4:
        X_batch, C_batch, state_batch, lengths = batch
    elif len(batch) == 3:
        X_batch, state_batch, lengths = batch
        C_batch = None
    else:
        raise ValueError(f"Unexpected batch structure: {len(batch)} elements")

    X_batch = X_batch.to(dtype=DTYPE, device=device)
    state_batch = state_batch.to(dtype=torch.long, device=device)
    C_batch = C_batch.to(dtype=DTYPE, device=device) if C_batch is not None else None
    lengths = lengths.to(dtype=torch.long, device=device)

    print(f"Loaded batch shapes:\n  X: {X_batch.shape}")
    if C_batch is not None:
        print(f"  C: {C_batch.shape}")
    print(f"  states: {state_batch.shape}, lengths: {lengths.shape} "
          f"(min={lengths.min().item()}, max={lengths.max().item()})")

    # --- initialize model ---
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

    # --- training ---
    print("\n=== Training NeuralHSMM (batched) ===")
    max_iter = 25
    n_epochs = 3

    for epoch in range(n_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True, ncols=100)
        for X_pad, C_pad, states_pad, lengths in pbar:
            X_pad = X_pad.to(device, dtype=DTYPE)
            C_pad = C_pad.to(device, dtype=DTYPE) if C_pad is not None else None
            lengths_list = lengths.tolist()

            # Flatten sequences for model
            X_flat = torch.cat([X_pad[i, :lengths_list[i]] for i in range(len(lengths_list))], 0)
            C_flat = torch.cat([C_pad[i, :lengths_list[i]] for i in range(len(lengths_list))], 0) \
                     if C_pad is not None else None

            if C_flat is not None:
                model.encoder.set_context(C_flat)
            else:
                model.encoder.reset_context()

            theta = model.encode_observations(X_flat) if getattr(model, "encoder", None) else None
            model.fit(X_flat, theta=theta, lengths=lengths_list, max_iter=max_iter, verbose=False)

            pbar.set_postfix({"batch_len": int(np.mean(lengths_list)), "n_samples": X_flat.shape[0]})
        print(f"Epoch {epoch+1}/{n_epochs} done")

    # --- decoding ---
    print("\n=== Decoding ===")
    t0 = time.time()

    X_test, C_test, states_true, lengths_test = batch
    X_test = X_test.to(device, dtype=DTYPE)
    C_test = C_test.to(device, dtype=DTYPE) if C_test is not None else None
    lengths_list = lengths_test.tolist()

    X_flat = torch.cat([X_test[i, :lengths_list[i]] for i in range(len(lengths_list))], 0)
    C_flat = torch.cat([C_test[i, :lengths_list[i]] for i in range(len(lengths_list))], 0) \
             if C_test is not None else None

    if C_flat is not None:
        model.encoder.set_context(C_flat)
        print(f"[Debug] Context set for {C_flat.shape[0]} timesteps.")
    else:
        model.encoder.reset_context()
        print("[Debug] Context reset (no context provided).")

    theta = model.encode_observations(X_flat) if getattr(model,"encoder",None) else None
    if theta is not None:
        print(f"[Debug] Encoded θ shape: {theta.shape}, range=({theta.min().item():.4f},{theta.max().item():.4f})")

    preds_list = model.predict(X_flat, lengths=lengths_list, algorithm="viterbi")
    v_path = torch.cat(preds_list,0).cpu().numpy()
    states_true_flat = torch.cat([states_true[i,:lengths_list[i]] for i in range(len(lengths_list))],0).cpu().numpy()

    decode_time = time.time()-t0
    print(f"[Timing] Decoding completed in {decode_time:.2f}s for {len(v_path)} timesteps.")

    acc, mapped_pred = best_permutation_accuracy(states_true_flat, v_path, n_classes=n_states)
    print(f"\nBest-permutation accuracy: {acc:.4f}")
    print("Confusion matrix (mapped_pred vs true):")
    print(confusion_matrix(states_true_flat, mapped_pred, labels=list(range(n_states))))

    unique_pred, counts_pred = np.unique(mapped_pred, return_counts=True)
    dist_str = ", ".join([f"S{s}:{c}" for s,c in zip(unique_pred, counts_pred)])
    pred_entropy = -np.sum((counts_pred/counts_pred.sum())*np.log(counts_pred/counts_pred.sum()+1e-12))
    print(f"\nDecoded state occupancy: {dist_str}, Prediction entropy: {pred_entropy:.4f}")
    if len(unique_pred)==1: print("[Warn] Decoder collapsed to single state")
    elif pred_entropy>np.log(n_states)*0.9: print("[Warn] Decoder near-uniform distribution")

    if hasattr(model,"transition_logits"):
        T = torch.softmax(model.transition_logits, dim=1).cpu().numpy()
        entropy_T = -np.sum(T*np.log(T+1e-12),axis=1)
        print(f"[Debug] Mean transition entropy: {entropy_T.mean():.4f}")

    print(f"[Timing] {decode_time:.2f}s total — {len(v_path)/decode_time:.1f} samples/sec")
    print_duration_summary(model)

    ts = time.strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"neuralhsmm_debug_state_{ts}.pt")
    print(f"\nModel state saved to neuralhsmm_debug_state_{ts}.pt")
