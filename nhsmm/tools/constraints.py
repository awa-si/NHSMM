# tools/constraints.py
import torch
from enum import Enum
from typing import Union, Optional
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
    """
    Resolve input to string value of Enum.

    Args:
        val: Enum instance or string.
        enum_type: Enum class.

    Returns:
        str: lowercased Enum value.
    """
    if isinstance(val, enum_type):
        return val.value
    if isinstance(val, str):
        return val.lower()
    raise ValueError(f"Expected {enum_type} or str, got {type(val)}")


def log_normalize(matrix: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    Numerically stable log-space normalization along a given axis.

    Args:
        matrix: Logits tensor.
        dim: Dimension to normalize over.

    Returns:
        Normalized log probabilities.
    """
    return matrix - torch.logsumexp(matrix, dim=dim, keepdim=True)


def is_valid_transition(probs: torch.Tensor, A_type: Union[str, Transitions], atol: float = 1e-6) -> bool:
    """
    Validate a transition matrix according to its type.

    Args:
        probs: [K, K] transition matrix (probabilities).
        A_type: Transition structure (semi, ergodic, left-to-right).
        atol: Absolute tolerance for floating-point comparisons.

    Returns:
        True if valid, False otherwise.
    """
    t = _resolve_type(A_type, Transitions)
    if not torch.isfinite(probs).all() or (probs < 0).any():
        logger.error("Transition matrix invalid: contains NaN or negative entries.")
        return False
    if not torch.allclose(probs.sum(-1), torch.ones(probs.shape[0], device=probs.device), atol=atol):
        logger.error("Transition matrix invalid: rows not normalized.")
        return False
    if t == Transitions.SEMI.value and not torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal()), atol=atol):
        logger.error("SEMI transition invalid: diagonal must be zero.")
        return False
    if t == Transitions.LEFT_TO_RIGHT.value and not torch.allclose(probs, torch.triu(probs), atol=atol):
        logger.error("LEFT_TO_RIGHT transition invalid: lower-triangular entries non-zero.")
        return False
    return True


# -------------------------------
# Information Criteria
# -------------------------------
def compute_information_criteria(
    n_samples: int, log_likelihood: torch.Tensor, dof: int, criterion: Union[str, InformCriteria]
) -> torch.Tensor:
    """
    Compute AIC, BIC, or HQC information criteria.

    Args:
        n_samples: Number of observations (time steps or sequences).
        log_likelihood: Log-likelihood value(s) of shape [] or [B].
        dof: Degrees of freedom of the model.
        criterion: Criterion type (AIC, BIC, HQC).

    Returns:
        IC tensor of same shape as log_likelihood.
    """
    c = _resolve_type(criterion, InformCriteria)
    log_n = torch.log(torch.as_tensor(n_samples, dtype=log_likelihood.dtype, device=log_likelihood.device))
    penalties = {
        InformCriteria.AIC.value: 2 * dof,
        InformCriteria.BIC.value: dof * log_n,
        InformCriteria.HQC.value: 2 * dof * torch.log(log_n),
    }
    if c not in penalties:
        raise ValueError(f"Unsupported information criterion: {c}")
    return -2 * log_likelihood + penalties[c]


# -------------------------------
# Covariance Utilities
# -------------------------------
def _assert_spd(matrix: torch.Tensor, label: str = "Matrix", eps: float = EPS):
    """
    Check that a matrix is symmetric positive-definite (SPD).

    Args:
        matrix: Tensor of shape [F, F].
        label: Name for error messages.
        eps: Minimum eigenvalue threshold for numerical stability.
    """
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{label} is not symmetric.")
    # Add tiny diagonal for numerical stability
    matrix_stable = matrix + torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype) * eps
    try:
        _ = torch.linalg.cholesky(matrix_stable)
    except RuntimeError as e:
        raise ValueError(f"{label} is not positive definite: {e}")


def validate_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int, eps: float = EPS, auto_correct: bool = True) -> torch.Tensor:
    """
    Validate or auto-correct covariance matrices according to type.

    Args:
        covars: Covariance tensor.
        cov_type: Type of covariance (full, diag, tied, spherical).
        n_states: Number of states.
        n_features: Number of features.
        eps: Minimal eigenvalue / clamp for numerical stability.
        auto_correct: If True, apply minimal correction.

    Returns:
        Validated covariance tensor.
    """
    c = _resolve_type(cov_type, CovarianceType)

    if c == CovarianceType.SPHERICAL.value:
        return covars.clamp_min(eps) if auto_correct else covars
    if c == CovarianceType.DIAG.value:
        return covars.clamp_min(eps) if auto_correct else covars
    if c == CovarianceType.TIED.value:
        _assert_spd(covars, "TIED covariance", eps)
        return covars
    if c == CovarianceType.FULL.value:
        if covars.shape != (n_states, n_features, n_features):
            raise ValueError(f"FULL covars shape mismatch: expected {(n_states,n_features,n_features)}, got {covars.shape}")
        for i in range(n_states):
            _assert_spd(covars[i], f"FULL covars state {i}", eps)
        return covars
    raise NotImplementedError(f"Unsupported covariance type: {c}")


def init_covars(base_cov: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int, eps: float = EPS) -> torch.Tensor:
    """
    Initialize covariance matrices for HSMM emissions.

    Args:
        base_cov: Base covariance or variance tensor.
        cov_type: Covariance type.
        n_states: Number of hidden states.
        n_features: Number of features.
        eps: Minimal value for numerical stability.

    Returns:
        Initialized covariance tensor of appropriate shape.
    """
    c = _resolve_type(cov_type, CovarianceType)

    if c == CovarianceType.SPHERICAL.value:
        val = base_cov.mean().clamp_min(eps)
        return val.repeat(n_states)
    if c == CovarianceType.DIAG.value:
        return torch.diag(base_cov).clamp_min(eps).unsqueeze(0).repeat(n_states, 1)
    if c == CovarianceType.TIED.value:
        _assert_spd(base_cov, "TIED covariance", eps)
        return base_cov
    if c == CovarianceType.FULL.value:
        _assert_spd(base_cov, "FULL covariance", eps)
        return base_cov.unsqueeze(0).repeat(n_states, 1, 1)
    raise NotImplementedError(f"Unsupported covariance type: {c}")


def fill_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int, eps: float = EPS) -> torch.Tensor:
    """
    Expand covariance parameters into full [n_states, F, F] matrices for emission use.

    Args:
        covars: Covariance tensor.
        cov_type: Type of covariance.
        n_states: Number of states.
        n_features: Number of features.
        eps: Minimal value for numerical stability.

    Returns:
        Covariance tensor [n_states, F, F].
    """
    c = _resolve_type(cov_type, CovarianceType)

    if c == CovarianceType.FULL.value:
        return covars
    if c == CovarianceType.DIAG.value:
        return torch.diag_embed(covars.clamp_min(eps))
    if c == CovarianceType.TIED.value:
        _assert_spd(covars, "TIED covariance", eps)
        return covars.unsqueeze(0).repeat(n_states, 1, 1)
    if c == CovarianceType.SPHERICAL.value:
        val = covars.clamp_min(eps)
        eye = torch.eye(n_features, dtype=val.dtype, device=val.device)
        return eye.unsqueeze(0).repeat(n_states, 1, 1) * val.view(-1, 1, 1)
    raise NotImplementedError(f"Unsupported covariance type: {c}")


def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int, eps: float = EPS) -> torch.Tensor:
    """
    Validate Poisson rate parameters (must be positive finite).

    Args:
        lambdas: Tensor of shape [n_states, n_features].
        n_states: Number of states.
        n_features: Number of features.
        eps: Minimal allowed value.

    Returns:
        Validated lambdas tensor.
    """
    if lambdas.shape != (n_states, n_features) or not torch.isfinite(lambdas).all() or (lambdas <= eps).any():
        raise ValueError(f"Invalid lambdas: shape {lambdas.shape}, must be > {eps} and finite.")
    return lambdas
