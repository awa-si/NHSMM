# utilities/constraints.py
import torch
from enum import Enum
from typing import Tuple, Optional, Union

from nhsmm.defaults import DTYPE, EPS


# -------------------------
# Transition types
# -------------------------
class Transitions(Enum):
    SEMI = "semi"
    ERGODIC = "ergodic"
    LEFT_TO_RIGHT = "left-to-right"


# -------------------------
# Information criteria
# -------------------------
class InformCriteria(Enum):
    AIC = "AIC"
    BIC = "BIC"
    HQC = "HQC"


# -------------------------
# Covariance types
# -------------------------
class CovarianceType(Enum):
    FULL = "full"
    DIAG = "diag"
    TIED = "tied"
    SPHERICAL = "spherical"


# -------------------------
# Utilities
# -------------------------
def _resolve_type(val, enum_type) -> str:
    """Convert string or enum instance to string value."""
    if isinstance(val, enum_type):
        return val.value
    if isinstance(val, str):
        return val
    raise ValueError(f"Expected {enum_type} or str, got {type(val)}")


def log_normalize(matrix: torch.Tensor, dim: Union[int, Tuple[int, ...]] = -1) -> torch.Tensor:
    return matrix - torch.logsumexp(matrix, dim=dim, keepdim=True)


# -------------------------
# Dirichlet / probability sampling
# -------------------------
def sample_probs(prior: float, target_size: Union[Tuple[int, ...], torch.Size], dtype=DTYPE, device=None) -> torch.Tensor:
    prior = max(prior, EPS)
    alphas = torch.full(target_size, prior, dtype=dtype, device=device)
    return torch.distributions.Dirichlet(alphas).sample()


def sample_transition(
    prior: float,
    n_states: int,
    A_type: Union[str, Transitions],
    device=None,
    dtype=DTYPE,
    verbose: bool = False
) -> torch.Tensor:
    t = _resolve_type(A_type, Transitions)
    probs = sample_probs(prior, (n_states, n_states), dtype=dtype, device=device)

    if t == Transitions.SEMI.value:
        probs = probs.clone()
        probs.fill_diagonal_(0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = row_sums.squeeze(-1) == 0
        if zero_rows.any():
            if verbose: print("[sample_transition] Zero rows detected in SEMI; replacing with uniform off-diagonal")
            for i in torch.nonzero(zero_rows, as_tuple=False).squeeze(-1):
                probs[i] = 1.0
                probs[i, i] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)

    elif t == Transitions.LEFT_TO_RIGHT.value:
        mask = torch.triu(torch.ones_like(probs, dtype=dtype, device=device))
        probs = probs * mask
        row_sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = row_sums.squeeze(-1) == 0
        if zero_rows.any():
            if verbose: print("[sample_transition] Zero rows detected in LEFT_TO_RIGHT; filling upper-triangle")
            for i in torch.nonzero(zero_rows, as_tuple=False).squeeze(-1):
                probs[i] = mask[i]
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)

    elif t == Transitions.ERGODIC.value:
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    else:
        raise NotImplementedError(f"Unsupported Transition type: {t}")

    return probs


def is_valid_transition(probs: torch.Tensor, A_type: Union[str, Transitions], atol: float = 1e-6) -> bool:
    t = _resolve_type(A_type, Transitions)
    if not torch.isfinite(probs).all() or (probs < 0).any():
        return False
    if not torch.allclose(probs.sum(-1), torch.ones(probs.size(0), device=probs.device), atol=atol):
        return False
    if t == Transitions.ERGODIC.value:
        return True
    if t == Transitions.SEMI.value:
        return torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal()), atol=atol)
    if t == Transitions.LEFT_TO_RIGHT.value:
        return torch.allclose(probs, torch.triu(probs), atol=atol)
    return False


# -------------------------
# Information criteria
# -------------------------
def compute_information_criteria(n_samples: int, log_likelihood: torch.Tensor, dof: int,
                                 criterion: Union[str, InformCriteria]) -> torch.Tensor:
    c = _resolve_type(criterion, InformCriteria)
    n_samples = float(n_samples)
    log_n = torch.log(torch.tensor(n_samples, dtype=log_likelihood.dtype, device=log_likelihood.device))
    penalties = {
        InformCriteria.AIC.value: 2.0 * dof,
        InformCriteria.BIC.value: dof * log_n,
        InformCriteria.HQC.value: 2.0 * dof * torch.log(log_n)
    }
    if c not in penalties:
        raise ValueError(f"Invalid information criterion: {c}")
    return -2.0 * log_likelihood + penalties[c]


# -------------------------
# Covariance utilities
# -------------------------
def _assert_spd(matrix: torch.Tensor, label: str = "Matrix"):
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{label} not symmetric")
    _, info = torch.linalg.cholesky_ex(matrix)
    if info != 0:
        raise ValueError(f"{label} not positive-definite")


def validate_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType],
                    n_states: int, n_features: int, n_components: Optional[int] = None,
                    eps: float = 1e-6, auto_correct: bool = True) -> torch.Tensor:
    """
    Validate and optionally auto-correct covariance matrices for different types.
    - SPHERICAL/DIAG: small negative or zero entries are replaced with eps.
    - TIED/FULL: must remain SPD, no auto-correction applied.
    """
    c = _resolve_type(cov_type, CovarianceType)

    if c == CovarianceType.SPHERICAL.value:
        if covars.numel() != n_features:
            raise ValueError("Invalid SPHERICAL covars shape")
        if auto_correct:
            covars = covars.clamp_min(eps)
        elif (covars <= 0).any():
            raise ValueError("SPHERICAL covars contain non-positive entries")
        return covars

    if c == CovarianceType.DIAG.value:
        if covars.shape != (n_states, n_features):
            raise ValueError("Invalid DIAG covars shape")
        if auto_correct:
            covars = covars.clamp_min(eps)
        elif (covars <= 0).any():
            raise ValueError("DIAG covars contain non-positive entries")
        return covars

    if c == CovarianceType.TIED.value:
        _assert_spd(covars)
        return covars

    if c == CovarianceType.FULL.value:
        expected_shape = (n_states, n_features, n_features)
        if n_components:
            expected_shape = (n_states, n_components, n_features, n_features)
        if covars.shape != expected_shape:
            raise ValueError("Invalid FULL covars shape")
        flat_covs = covars.view(-1, n_features, n_features)
        if not torch.allclose(flat_covs, flat_covs.transpose(-2, -1), atol=1e-6):
            raise ValueError("Some FULL covars not symmetric")
        _, info = torch.linalg.cholesky_ex(flat_covs)
        if info.any():
            idx = torch.nonzero(info, as_tuple=True)[0]
            raise ValueError(f"FULL covars not PD at indices {idx.tolist()}")
        return covars

    raise NotImplementedError(f"Unsupported covariance type: {c}")


def init_covars(base_cov: torch.Tensor, cov_type: Union[str, CovarianceType],
                n_states: int, n_features: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Initialize covariances for all states based on a base covariance.
    Small negative values are clamped to eps for SPHERICAL/DIAG types.
    """
    c = _resolve_type(cov_type, CovarianceType)

    if c == CovarianceType.SPHERICAL.value:
        val = base_cov.mean().clamp_min(eps)
        return val.expand(n_states)

    if c == CovarianceType.TIED.value:
        _assert_spd(base_cov)
        return base_cov

    if c == CovarianceType.DIAG.value:
        diag = torch.diag(base_cov).clamp_min(eps)
        return diag.unsqueeze(0).expand(n_states, -1)

    if c == CovarianceType.FULL.value:
        _assert_spd(base_cov)
        return base_cov.unsqueeze(0).expand(n_states, -1, -1)

    raise NotImplementedError(f"Unsupported covariance type: {c}")


def fill_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType],
                n_states: int, n_features: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Expand covariance tensors into full shape expected by emission distributions.
    Applies auto-correction for SPHERICAL/DIAG covars.
    """
    c = _resolve_type(cov_type, CovarianceType)

    if c == CovarianceType.FULL.value:
        return covars

    if c == CovarianceType.DIAG.value:
        diag = covars.clamp_min(eps)
        return torch.diag_embed(diag)

    if c == CovarianceType.TIED.value:
        _assert_spd(covars)
        return covars.unsqueeze(0).expand(n_states, -1, -1)

    if c == CovarianceType.SPHERICAL.value:
        val = covars.clamp_min(eps)
        eye = torch.eye(n_features, dtype=val.dtype, device=val.device)
        return eye.unsqueeze(0) * val.view(-1, 1, 1)

    raise NotImplementedError(f"Unsupported covariance type: {c}")


def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int) -> torch.Tensor:
    if lambdas.shape != (n_states, n_features) or not torch.isfinite(lambdas).all() or (lambdas <= 0).any():
        raise ValueError("Invalid lambdas")
    return lambdas
