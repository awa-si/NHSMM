# utilities/loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import os
import talib as ta
import numpy as np
import polars as pl
from typing import Tuple, Optional, List

from nhsmm.defaults import DTYPE


class SequenceLoader(Dataset):
    """
    Dataset for HSMM training/decoding.
    Automatically chunks long sequences (>1000 timesteps) for OHLCV data.
    Supports variable-length sequences, optional context, and deterministic output.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        seed: int = 0,
        n_segments_per_state: int = 3,
        seg_len_range: Tuple[int, int] = (5, 20),
        context_dim: Optional[int] = None,
        context_noise_scale: float = 0.05,
        variable_length: bool = False,
        noise_scale: float = 0.05,
        normalize: bool = False,
        # OHLCV options
        data_dir: Optional[str] = None,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        max_chunk: int = 1000,
        device: Optional[torch.device] = None,
    ):
        self.variable_length = variable_length
        self.device = device or torch.device("cpu")
        self.max_chunk = max_chunk

        dataframe = None
        if data_dir and pair and timeframe:
            dataframe = self.load_from_dataframe(data_dir, pair, timeframe)

        seg_X, seg_states, seg_C = self.generate_sequence(
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
            feature_cols=feature_cols,
        )

        if dataframe is not None:
            seg_X, seg_states, seg_C = self.chunk_sequences(seg_X, seg_states, seg_C)

        if variable_length:
            self.X = [x.to(self.device) for x in seg_X]
            self.states = [s.to(self.device) for s in seg_states]
            self.C = [c.to(self.device) for c in seg_C] if seg_C else None
        else:
            self.X = torch.cat(seg_X, dim=0).to(self.device)
            self.states = torch.cat(seg_states, dim=0).to(self.device)
            self.C = torch.cat(seg_C, dim=0).to(self.device) if seg_C else None

    # ----------------------
    # Chunking helper
    # ----------------------
    def chunk_sequences(self, X_list, states_list, C_list):
        new_X, new_states, new_C = [], [], []
        for i, X in enumerate(X_list):
            states = states_list[i]
            C = C_list[i] if C_list else None
            n = X.shape[0]
            if n <= self.max_chunk:
                new_X.append(X)
                new_states.append(states)
                if C is not None:
                    new_C.append(C)
                continue
            for start in range(0, n, self.max_chunk):
                end = min(start + self.max_chunk, n)
                new_X.append(X[start:end])
                new_states.append(states[start:end])
                if C is not None:
                    new_C.append(C[start:end])
        return new_X, new_states, new_C if C_list else None

    # ----------------------
    # Dataset / DataLoader
    # ----------------------
    def __len__(self):
        return len(self.X) if self.variable_length else self.X.shape[0]

    def __getitem__(self, idx):
        if self.variable_length:
            items = (self.X[idx],)
            if self.C is not None:
                items += (self.C[idx],)
            items += (self.states[idx],)
            return items
        else:
            if self.C is not None:
                return self.X[idx], self.C[idx], self.states[idx]
            return self.X[idx], self.states[idx]

    def collate_fn(self, batch):
        X_list = [b[0] for b in batch]
        lengths = torch.tensor([len(x) for x in X_list], dtype=torch.long, device=self.device)
        X_padded = pad_sequence(X_list, batch_first=True).to(dtype=DTYPE, device=self.device)

        if self.C is not None:
            C_list = [b[1] for b in batch]
            C_padded = pad_sequence(C_list, batch_first=True).to(dtype=DTYPE, device=self.device)
            states_list = [b[2] for b in batch]
            states_padded = pad_sequence(states_list, batch_first=True).to(self.device)
            return X_padded, C_padded, states_padded, lengths
        else:
            states_list = [b[1] for b in batch]
            states_padded = pad_sequence(states_list, batch_first=True).to(self.device)
            return X_padded, states_padded, lengths

    def loader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn if self.variable_length else None,
            drop_last=drop_last,
        )

    # ----------------------
    # Load OHLCV
    # ----------------------
    def load_from_dataframe(self, data_dir: str, pair: str, timeframe: str) -> pl.DataFrame:
        symbol = pair.replace("/", "_").replace(":", "_")
        filename = f"{symbol}-{timeframe}-futures.feather"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found for {pair} @ {timeframe}: {path}")
        df = pl.read_ipc(path, memory_map=False).sort("date")
        print(f"Loading {pair} OHLCV from dataframe {timeframe}...")
        return df

    # ----------------------
    # Add indicators for pseudo-states and context
    # ----------------------
    def _add_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()

        # Returns
        ret = np.zeros_like(close)
        ret[1:] = np.log(close[1:] / close[:-1])

        # Volatility (rolling std of returns)
        window = min(14, len(close))
        vol = np.zeros_like(close)
        vol[window-1:] = np.array([ret[i-window+1:i+1].std() for i in range(window-1, len(close))])

        # Trend: SMA fast vs slow
        sma_fast = ta.SMA(close, timeperiod=10)
        sma_slow = ta.SMA(close, timeperiod=30)
        trend = np.zeros_like(close)
        trend[sma_fast > sma_slow] = 1  # bull
        trend[sma_fast < sma_slow] = 2  # bear
        # trend == 0 -> range

        df = df.with_columns([
            pl.Series('return', ret),
            pl.Series('volatility', vol),
            pl.Series('trend', trend.astype(int))
        ])
        return df

    # ----------------------
    # Generate sequences (synthetic or from dataframe)
    # ----------------------
    def generate_sequence(
        self,
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
        feature_cols: Optional[List[str]] = None,
    ):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        segments_X, segments_states, segments_C = [], [], []

        if dataframe is not None:
            df = self._add_indicators(dataframe)

            # Auto-select feature columns if not provided
            if feature_cols is None:
                base_cols = ['open', 'high', 'low', 'close', 'volume']
                indicator_cols = ['return', 'volatility', 'trend']
                feature_cols = [c for c in base_cols + indicator_cols if c in df.columns][-n_features:]

            X_np = df.select(feature_cols).to_numpy()
            n_samples = X_np.shape[0]

            # Context features
            if context_dim:
                ctx_cols = ['return', 'volatility', 'trend'][:context_dim]
                ctx_cols = [c for c in ctx_cols if c in df.columns]
                C_np = df.select(ctx_cols).to_numpy()
                segments_C.append(torch.tensor(C_np, dtype=DTYPE))

            # Pseudo-states: 0=range, 1=bull, 2=bear
            pseudo_states = df['trend'].to_numpy().copy()
            vol = df['volatility'].to_numpy()
            high_vol_mask = vol > np.percentile(vol, 75)
            pseudo_states[high_vol_mask & (pseudo_states == 0)] = 0  # keep range
            segments_states.append(torch.tensor(pseudo_states, dtype=torch.long))
            segments_X.append(torch.tensor(X_np, dtype=DTYPE))

        else:
            # Synthetic sequences
            base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
            for s in range(n_states):
                for _ in range(n_segments_per_state):
                    L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
                    X_seg = base_means[s] + rng.normal(scale=noise_scale, size=(L, n_features))
                    S_seg = np.full(L, s, dtype=int)
                    segments_X.append(torch.tensor(X_seg, dtype=DTYPE))
                    segments_states.append(torch.tensor(S_seg, dtype=torch.long))
                    if context_dim:
                        segments_C.append(torch.tensor(
                            rng.normal(scale=context_noise_scale, size=(L, context_dim)), dtype=DTYPE
                        ))

        # Normalize
        if normalize and segments_X:
            device = segments_X[0].device
            all_X = torch.cat(segments_X, dim=0).to(device)
            mean, std = all_X.mean(0, keepdim=True), all_X.std(0, keepdim=True) + 1e-8
            segments_X = [(x - mean) / std for x in segments_X]

        return segments_X, segments_states, (segments_C if segments_C else None)
