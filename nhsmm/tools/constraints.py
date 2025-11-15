import torch
from enum import Enum
from typing import Union
from nhsmm.constants import DTYPE, EPS, logger

# -------------------------------
# Enums
# -------------------------------
class Transitions(Enum):
    SEMI = "semi"
    ERGODIC = "ergodic"
    LEFT_TO_RIGHT = "left-to-right"

class InformCriteria(Enum):
    AIC = "AIC"
    BIC = "BIC"
    HQC = "HQC"

class CovarianceType(Enum):
    FULL = "full"
    DIAG = "diag"
    TIED = "tied"
    SPHERICAL = "spherical"

# -------------------------------
# Utilities
# -------------------------------
def _resolve_type(val, enum_type) -> str:
    if isinstance(val, enum_type):
        return val.value.lower()
    if isinstance(val, str):
        return val.lower()
    raise ValueError(f"Expected {enum_type} or str, got {type(val)}")

def log_normalize(matrix: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return matrix - torch.logsumexp(matrix, dim=dim, keepdim=True)

def is_valid_transition(probs: torch.Tensor, A_type: Union[str, Transitions], atol: float = 1e-6) -> bool:
    t = _resolve_type(A_type, Transitions)
    if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
        logger.error("Transition matrix must be square.")
        return False
    if not torch.isfinite(probs).all() or (probs < 0).any():
        logger.error("Transition matrix invalid: contains NaN or negative entries.")
        return False
    if not torch.allclose(probs.sum(-1), torch.ones(probs.shape[0], device=probs.device), atol=atol):
        logger.error("Transition matrix invalid: rows not normalized.")
        return False
    if t == "semi" and not torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal()), atol=atol):
        logger.error("SEMI transition invalid: diagonal must be zero.")
        return False
    if t == "left-to-right" and not torch.allclose(probs, torch.triu(probs), atol=atol):
        logger.error("LEFT_TO_RIGHT transition invalid: lower-triangular entries non-zero.")
        return False
    return True

def compute_information_criteria(n_samples: int, log_likelihood: torch.Tensor, dof: int, criterion: Union[str, InformCriteria]) -> torch.Tensor:
    c = _resolve_type(criterion, InformCriteria)
    log_likelihood = torch.as_tensor(log_likelihood, dtype=torch.float64)
    log_n = torch.log(torch.as_tensor(n_samples, dtype=torch.float64, device=log_likelihood.device))
    penalties = {"aic": 2*dof, "bic": dof*log_n, "hqc": 2*dof*torch.log(log_n)}
    if c not in penalties:
        raise ValueError(f"Unsupported criterion: {c}")
    return -2*log_likelihood + penalties[c]

# -------------------------------
# Covariance Utilities
# -------------------------------
def _assert_spd(matrix: torch.Tensor, label: str = "Matrix", eps: float = EPS):
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{label} not symmetric.")
    stable = 0.5*(matrix + matrix.T) + torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)*eps
    try:
        _ = torch.linalg.cholesky(stable)
    except RuntimeError as e:
        raise ValueError(f"{label} not positive-definite: {e}")

def validate_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int, eps: float = EPS, auto_correct: bool = True) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c in ("spherical", "diag"):
        return covars.clamp_min(eps) if auto_correct else covars
    if c == "tied":
        _assert_spd(covars, "TIED", eps)
        return covars
    if c == "full":
        if covars.shape != (n_states, n_features, n_features):
            raise ValueError(f"FULL covars shape mismatch: expected {(n_states, n_features, n_features)}, got {covars.shape}")
        for i in range(n_states):
            _assert_spd(covars[i], f"FULL state {i}", eps)
        return covars
    raise NotImplementedError(f"Unsupported cov_type: {c}")

def init_covars(base_cov: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int, eps: float = EPS, device=None) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    device = device or base_cov.device
    if base_cov.ndim == 0:
        base_cov = base_cov.expand(n_features)
    elif base_cov.ndim == 1 and base_cov.numel() == n_features:
        base_cov = torch.diag(base_cov)
    base_cov = base_cov.to(device=device, dtype=DTYPE)
    if c == "spherical":
        val = base_cov.mean().clamp_min(eps)
        return val.repeat(n_states)
    if c == "diag":
        return torch.diag(base_cov).clamp_min(eps).unsqueeze(0).expand(n_states, -1)
    if c == "tied":
        _assert_spd(base_cov, "TIED", eps)
        return base_cov
    if c == "full":
        _assert_spd(base_cov, "FULL", eps)
        return base_cov.unsqueeze(0).expand(n_states, -1, -1)
    raise NotImplementedError(f"Unsupported cov_type: {c}")

def fill_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int, eps: float = EPS) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c == "full":
        return covars
    if c == "diag":
        return torch.diag_embed(covars.clamp_min(eps))
    if c == "tied":
        _assert_spd(covars, "TIED", eps)
        return covars.unsqueeze(0).expand(n_states, -1, -1)
    if c == "spherical":
        val = covars.clamp_min(eps)
        eye = torch.eye(n_features, dtype=val.dtype, device=val.device)
        return eye.unsqueeze(0).expand(n_states, -1, -1) * val.view(-1, 1, 1)
    raise NotImplementedError(f"Unsupported cov_type: {c}")

def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int, eps: float = EPS) -> torch.Tensor:
    if lambdas.shape != (n_states, n_features):
        raise ValueError(f"Invalid lambdas shape: expected {(n_states, n_features)}, got {lambdas.shape}")
    if not torch.isfinite(lambdas).all() or (lambdas <= eps).any():
        raise ValueError(f"Lambdas must be > {eps} and finite.")
    return lambdas
