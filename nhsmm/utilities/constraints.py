# constraints.py
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import torch

from nhsmm.defaults import DTYPE, EPS


@dataclass(frozen=True)
class Transitions:
    SEMI: str = "semi"
    ERGODIC: str = "ergodic"
    LEFT_TO_RIGHT: str = "left-to-right"

@dataclass(frozen=True)
class InformCriteria:
    AIC: str = "AIC"
    BIC: str = "BIC"
    HQC: str = "HQC"

@dataclass(frozen=True)
class CovarianceType:
    FULL: str = "full"
    DIAG: str = "diag"
    TIED: str = "tied"
    SPHERICAL: str = "spherical"

def _resolve_type(val, dataclass_type) -> str:
    if isinstance(val, dataclass_type):
        return val.__dict__[list(val.__dataclass_fields__.keys())[0]]
    if isinstance(val, str):
        return val
    raise ValueError(f"Expected {dataclass_type} or str, got {type(val)}")


def sample_probs(prior: float, target_size: Union[Tuple[int, ...], torch.Size],
                 dtype=DTYPE, device=None) -> torch.Tensor:
    alphas = torch.full(target_size, prior, dtype=dtype, device=device)
    return torch.distributions.Dirichlet(alphas).sample()


def sample_transition(
    prior: float,
    n_states: int,
    A_type: Union[str, Transitions],
    device=None,
    dtype=DTYPE,
) -> torch.Tensor:
    t = _resolve_type(A_type, Transitions)
    
    # Sample base probabilities
    probs = sample_probs(prior, (n_states, n_states), dtype=dtype, device=device)
    
    if t == Transitions().SEMI:
        probs = probs.clone()
        probs.fill_diagonal_(0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        # Replace zero rows with uniform probability
        zero_rows = (row_sums == 0)
        if zero_rows.any():
            probs[zero_rows.expand_as(probs)] = 1.0
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    
    elif t == Transitions().LEFT_TO_RIGHT:
        mask = torch.triu(torch.ones_like(probs, dtype=dtype, device=device))
        probs = probs * mask
        row_sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = (row_sums == 0)
        if zero_rows.any():
            probs[zero_rows.expand_as(probs)] = mask[zero_rows.expand_as(mask)]
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    
    elif t == Transitions().ERGODIC:
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    
    else:
        raise NotImplementedError(f"Unsupported Transition type: {t}")
    
    return probs


def compute_information_criteria(n_samples: int, log_likelihood: torch.Tensor, dof: int,
                                 criterion: Union[str, InformCriteria]) -> torch.Tensor:
    c = _resolve_type(criterion, InformCriteria)
    log_n = torch.log(torch.tensor(float(n_samples), dtype=log_likelihood.dtype, device=log_likelihood.device))
    penalties = {InformCriteria().AIC: 2.0*dof, InformCriteria().BIC: dof*log_n, InformCriteria().HQC: 2.0*dof*torch.log(log_n)}
    if c not in penalties:
        raise ValueError(f"Invalid information criterion: {c}")
    return -2.0*log_likelihood + penalties[c]

def is_valid_transition(probs: torch.Tensor, A_type: Union[str, Transitions], atol: float = 1e-6) -> bool:
    t = _resolve_type(A_type, Transitions)
    if not torch.isfinite(probs).all() or (probs < 0).any():
        return False
    if not torch.allclose(probs.sum(-1), torch.ones(probs.size(0), device=probs.device), atol=atol):
        return False
    if t == Transitions().ERGODIC:
        return True
    if t == Transitions().SEMI:
        return torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal()), atol=atol)
    if t == Transitions().LEFT_TO_RIGHT:
        return bool(torch.allclose(probs, torch.triu(probs), atol=atol))
    raise NotImplementedError(f"Unsupported Transition type: {t}")

def log_normalize(matrix: torch.Tensor, dim: Union[int, Tuple[int, ...]] = -1) -> torch.Tensor:
    return matrix - torch.logsumexp(matrix, dim=dim, keepdim=True)

def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int) -> torch.Tensor:
    if lambdas.shape != (n_states, n_features) or not torch.isfinite(lambdas).all() or (lambdas <= 0).any():
        raise ValueError("Invalid lambdas")
    return lambdas

def _assert_spd(matrix: torch.Tensor, label: str = "Matrix"):
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{label} not symmetric")
    _, info = torch.linalg.cholesky_ex(matrix)
    if info != 0:
        raise ValueError(f"{label} not positive-definite")

def validate_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int,
                    n_features: int, n_components: Optional[int] = None) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c == CovarianceType().SPHERICAL:
        if covars.numel() != n_features or (covars <= 0).any(): raise ValueError("Invalid SPHERICAL covars")
        return covars
    if c == CovarianceType().TIED:
        if covars.shape != (n_features, n_features): raise ValueError("Invalid TIED covars")
        _assert_spd(covars)
        return covars
    if c == CovarianceType().DIAG:
        if covars.shape != (n_states, n_features) or (covars <= 0).any(): raise ValueError("Invalid DIAG covars")
        return covars
    if c == CovarianceType().FULL:
        expected_shape = (n_states, n_features, n_features)
        if n_components: expected_shape = (n_states, n_components, n_features, n_features)
        if covars.shape != expected_shape: raise ValueError("Invalid FULL covars")
        flat_covs = covars.view(-1, n_features, n_features)
        if not torch.allclose(flat_covs, flat_covs.transpose(-2, -1), atol=1e-6):
            raise ValueError("Some FULL covars not symmetric")
        _, info = torch.linalg.cholesky_ex(flat_covs)
        if info.any():
            idx = torch.nonzero(info, as_tuple=True)[0]
            raise ValueError(f"FULL covars not PD at indices {idx.tolist()}")
        return covars
    raise NotImplementedError(f"Unsupported covariance type: {c}")

def init_covars(base_cov: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c == CovarianceType().SPHERICAL: return base_cov.mean().expand(n_states)
    if c == CovarianceType().TIED: return base_cov
    if c == CovarianceType().DIAG: return base_cov.diag().unsqueeze(0).expand(n_states, -1)
    if c == CovarianceType().FULL: return base_cov.unsqueeze(0).expand(n_states, -1, -1)
    raise NotImplementedError(f"Unsupported covariance type: {c}")

def fill_covars(covars: torch.Tensor, cov_type: Union[str, CovarianceType], n_states: int, n_features: int) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c == CovarianceType().FULL: return covars
    if c == CovarianceType().DIAG: return torch.diag_embed(covars)
    if c == CovarianceType().TIED: return covars.unsqueeze(0).expand(n_states, -1, -1)
    if c == CovarianceType().SPHERICAL:
        eye = torch.eye(n_features, dtype=covars.dtype, device=covars.device)
        return eye.unsqueeze(0) * covars.view(-1, 1, 1)
    raise NotImplementedError(f"Unsupported covariance type: {c}")
