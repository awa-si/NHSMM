# utilities/loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import polars as pl
from typing import Tuple, Optional, List

from nhsmm.config import DTYPE


def load_dataframe(data_dir: str, pair: str, timeframe: str) -> pl.DataFrame:
    symbol = pair.replace("/", "_").replace(":", "_")
    filename = f"{symbol}-{timeframe}-futures.feather"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Freqtrade data not found for {pair} @ {timeframe}: {path}")
    return pl.read_ipc(path, memory_map=False).sort("date")

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
    dataframe: Optional[pl.DataFrame] = None
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X_list, states_list, C_list = [], [], []

    if dataframe is not None:
        feature_cols = dataframe.columns[-n_features:]
        X_np = dataframe.select(feature_cols).to_numpy()
        n_samples = X_np.shape[0]
        X_list.append(X_np)
        states_list = np.zeros(n_samples, dtype=int)
        if context_dim:
            C_list.append(rng.normal(scale=context_noise_scale, size=(n_samples, context_dim)))
    else:
        base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
        for s in range(n_states):
            for _ in range(n_segments_per_state):
                L = int(rng.integers(*seg_len_range))
                segment = base_means[s] + rng.normal(scale=noise_scale, size=(L, n_features))
                X_list.append(segment)
                states_list.extend([s] * L)
                if context_dim:
                    C_list.append(rng.normal(scale=context_noise_scale, size=(L, context_dim)))

    X = np.vstack(X_list)
    states = np.array(states_list, dtype=int)
    C = np.vstack(C_list) if C_list else None

    if normalize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    X_tensor = torch.tensor(X, dtype=DTYPE)
    C_tensor = torch.tensor(C, dtype=DTYPE) if C is not None else None

    return states, X_tensor, C_tensor


class SequenceDataset(Dataset):
    def __init__(
        self,
        n_states: int,
        n_features: int,
        seed: int = 42,
        seg_len_range: Tuple[int, int] = (5, 20),
        n_segments_per_state: int = 3,
        noise_scale: float = 0.05,
        context_dim: Optional[int] = None,
        context_noise_scale: float = 0.05,
        normalize: bool = False,
        dataframe: Optional[pl.DataFrame] = None,
        variable_length: bool = False
    ):
        self.variable_length = variable_length
        self.states, X, C = generate_gaussian_sequence(
            n_states=n_states,
            n_features=n_features,
            seed=seed,
            seg_len_range=seg_len_range,
            n_segments_per_state=n_segments_per_state,
            noise_scale=noise_scale,
            context_dim=context_dim,
            context_noise_scale=context_noise_scale,
            normalize=normalize,
            dataframe=dataframe
        )

        # Convert to tensor
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=DTYPE)
        self.C = C if isinstance(C, torch.Tensor) or C is None else torch.tensor(C, dtype=DTYPE)
        self.states = torch.tensor(self.states, dtype=torch.long)

        # Split into sequences if variable_length
        if self.variable_length:
            self.seq_lengths = [len(self.X)] if dataframe is not None else [len(seg) for seg in X] if isinstance(X, list) else [len(X)]
            if isinstance(X, torch.Tensor) and not isinstance(X, list):
                self.X = [self.X]  # wrap in list for uniform processing
                if self.C is not None:
                    self.C = [self.C]

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

    def collate_fn(self, batch):
        # batch: list of tuples (X_seq, C_seq, state_seq) or (X_seq, state_seq)
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
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn if self.variable_length else None)
