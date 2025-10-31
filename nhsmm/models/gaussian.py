# models/gaussian.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Literal

from nhsmm.constants import DTYPE, EPS, HSMMError, logger
from nhsmm.models.hsmm import HSMM


class GaussianHSMM(HSMM):
    """
    Gaussian Hidden Semi-Markov Model (HSMM) with multivariate normal emissions.

    Features:
        - Robust emission initialization (k-means or moments)
        - Data-driven warm-start for pi, A, D
        - Device/dtype safe, with regularization and small-sample guards
        - decode() wrapper for convenience
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        k_means: bool = False,
        min_covar: float = 1e-3,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else torch.device('cpu')
        self.min_covar = float(min_covar)
        self.n_features = n_features
        self.k_means = k_means

        super().__init__(
            n_states=n_states,
            n_features=n_features,
            max_duration=max_duration,
            min_covar=min_covar,
            alpha=alpha,
            seed=seed,
        )

    @property
    def dof(self) -> int:
        K, F = self.n_states, self.n_features
        trans_dof = K * (K - 1)
        mean_dof = K * F
        cov_dof = K * F * (F + 1) // 2
        return trans_dof + mean_dof + cov_dof

    def _contextual_emission_pdf(self, X: Optional[torch.Tensor] = None, theta: Optional[dict] = None) -> Optional[MultivariateNormal]:
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if isinstance(pdf, MultivariateNormal):
            return pdf
        return None

    def decode(self, X: torch.Tensor, algorithm: Literal["viterbi", "map"] = "viterbi") -> np.ndarray:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=DTYPE)
        preds = self.predict(X, algorithm=algorithm, context=None)
        if isinstance(preds, list):
            preds = preds[0] if preds else torch.empty(0, dtype=torch.long)
        elif not torch.is_tensor(preds):
            preds = torch.as_tensor(preds, dtype=torch.long)
        return preds.cpu().numpy()
