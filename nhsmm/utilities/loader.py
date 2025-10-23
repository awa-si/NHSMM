# utilities/loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import polars as pl
from typing import Tuple, Optional, List, Union

from nhsmm.defaults import DTYPE


# ============================================================
# Load external dataframe
# ============================================================
def load_dataframe(data_dir: str, pair: str, timeframe: str) -> pl.DataFrame:
    """
    Load a Feather file with price or feature data, ensuring sorted timestamps.
    """
    symbol = pair.replace("/", "_").replace(":", "_")
    filename = f"{symbol}-{timeframe}-futures.feather"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Freqtrade data not found for {pair} @ {timeframe}: {path}")
    return pl.read_ipc(path, memory_map=False).sort("date")


# ============================================================
# Synthetic Gaussian generator (now segment-aware)
# ============================================================
def generate_gaussian_sequence(
    n_states: int,
    n_features: int,
    seed: int,
    seg_len_range: Tuple[int, int] = (5, 20),
    n_segments_per_state: int = 3,
    noise_scale: float = 0.05,
    context_dim: Optional[int] = None,
    context_noise_scale: float = 0.05,
    normalize: bool = False,
    dataframe: Optional[pl.DataFrame] = None,
):
    """
    Generate synthetic Gaussian data for HSMMs with optional context.

    Returns:
        segments_X: list[Tensor[T_i, F]]
        segments_states: list[Tensor[T_i]]
        segments_C: list[Tensor[T_i, C]] or None
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    segments_X, segments_states, segments_C = [], [], []

    if dataframe is not None:
        # Use real data as one long sequence
        feature_cols = dataframe.columns[-n_features:]
        X_np = dataframe.select(feature_cols).to_numpy()
        n_samples = X_np.shape[0]
        X_tensor = torch.tensor(X_np, dtype=DTYPE)
        state_seq = torch.zeros(n_samples, dtype=torch.long)
        segments_X = [X_tensor]
        segments_states = [state_seq]
        if context_dim:
            C_tensor = torch.tensor(
                rng.normal(scale=context_noise_scale, size=(n_samples, context_dim)),
                dtype=DTYPE,
            )
            segments_C = [C_tensor]
    else:
        # Pure synthetic generation
        base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
        for s in range(n_states):
            for _ in range(n_segments_per_state):
                L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
                X_seg = base_means[s] + rng.normal(scale=noise_scale, size=(L, n_features))
                S_seg = np.full(L, s, dtype=int)
                segments_X.append(torch.tensor(X_seg, dtype=DTYPE))
                segments_states.append(torch.tensor(S_seg, dtype=torch.long))
                if context_dim:
                    C_seg = rng.normal(scale=context_noise_scale, size=(L, context_dim))
                    segments_C.append(torch.tensor(C_seg, dtype=DTYPE))

    # Optional normalization
    if normalize:
        all_X = torch.cat(segments_X, dim=0)
        mean, std = all_X.mean(0, keepdim=True), all_X.std(0, keepdim=True) + 1e-8
        segments_X = [(x - mean) / std for x in segments_X]

    return segments_X, segments_states, segments_C if segments_C else None


# ============================================================
# Sequence Dataset
# ============================================================
class SequenceDataset(Dataset):
    """
    Dataset for training/decoding variable-length sequences in HSMMs.
    Supports both synthetic Gaussian sequences and external dataframe inputs.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        seed: int = 42,
        n_segments_per_state: int = 3,
        seg_len_range: Tuple[int, int] = (5, 20),
        context_dim: Optional[int] = None,
        context_noise_scale: float = 0.05,
        variable_length: bool = False,
        noise_scale: float = 0.05,
        normalize: bool = False,
        dataframe: Optional[pl.DataFrame] = None,
    ):
        self.variable_length = variable_length

        seg_X, seg_states, seg_C = generate_gaussian_sequence(
            n_states=n_states,
            n_features=n_features,
            seed=seed,
            seg_len_range=seg_len_range,
            n_segments_per_state=n_segments_per_state,
            noise_scale=noise_scale,
            context_dim=context_dim,
            context_noise_scale=context_noise_scale,
            normalize=normalize,
            dataframe=dataframe,
        )

        # Store as list of sequences for variable-length mode
        if variable_length:
            self.X = seg_X
            self.states = seg_states
            self.C = seg_C
        else:
            # Flatten all segments into one big sequence
            self.X = torch.cat(seg_X, dim=0)
            self.states = torch.cat(seg_states, dim=0)
            self.C = torch.cat(seg_C, dim=0) if seg_C else None

    def __len__(self):
        return len(self.X) if self.variable_length else self.X.shape[0]

    def __getitem__(self, idx):
        if self.variable_length:
            if self.C is not None:
                return self.X[idx], self.C[idx], self.states[idx]
            else:
                return self.X[idx], self.states[idx]
        else:
            if self.C is not None:
                return self.X[idx], self.C[idx], self.states[idx]
            else:
                return self.X[idx], self.states[idx]

    # ========================================================
    # Collation and loader
    # ========================================================
    def collate_fn(self, batch):
        """
        Pads variable-length sequences into uniform tensors.
        """
        X_list = [b[0] for b in batch]
        lengths = torch.tensor([len(x) for x in X_list], dtype=torch.long)
        X_padded = torch.nn.utils.rnn.pad_sequence(X_list, batch_first=True)

        if self.C is not None:
            C_list = [b[1] for b in batch]
            C_padded = torch.nn.utils.rnn.pad_sequence(C_list, batch_first=True)
            states_list = [b[2] for b in batch]
            states_padded = torch.nn.utils.rnn.pad_sequence(states_list, batch_first=True)
            return X_padded, C_padded, states_padded, lengths
        else:
            states_list = [b[1] for b in batch]
            states_padded = torch.nn.utils.rnn.pad_sequence(states_list, batch_first=True)
            return X_padded, states_padded, lengths

    def loader(self, batch_size=64, shuffle=True):
        """
        Build DataLoader with correct collation.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn if self.variable_length else None,
        )
