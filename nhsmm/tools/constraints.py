import torch
from enum import Enum
from typing import Tuple, Optional, Union

from nhsmm.defaults import DTYPE, EPS, logger


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


def _resolve_type(val, enum_type) -> str:
    if isinstance(val, enum_type): return val.value
    if isinstance(val, str): return val
    logger.error(f"Invalid type for _resolve_type: {type(val)}")
    raise ValueError(f"Expected {enum_type} or str, got {type(val)}")

def log_normalize(matrix: torch.Tensor, dim=-1) -> torch.Tensor:
    return matrix - torch.logsumexp(matrix, dim=dim, keepdim=True)

# -------------------------
# Probability Sampling
# -------------------------
def sample_probs(prior: float,
                 target_size: Tuple[int, ...],
                 dtype=DTYPE,
                 device=None,
                 seed: Optional[int] = None,
                 eps_collapse: float = 1e-3,
                 max_resample: int = 5) -> torch.Tensor:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = max(prior, EPS)
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
    for attempt in range(max_resample):
        alphas = torch.full(target_size, prior, dtype=dtype, device=device)
        probs = torch.distributions.Dirichlet(alphas).sample().clamp_min(EPS)
        if not (probs > 1.0 - eps_collapse).any():
            break
        logger.debug(f"Collapsed probs detected attempt {attempt+1}")
    else:
        logger.error("Max resample attempts reached; some probs may be nearly deterministic")
    if seed is not None:
        torch.set_rng_state(rng_state)
    return probs

# -------------------------
# Transition Sampling
# -------------------------
def sample_transition(prior: float,
                      n_states: int,
                      A_type: Union[str, Transitions],
                      device=None,
                      dtype=DTYPE,
                      seed: Optional[int] = None,
                      eps_collapse: float = 1e-3,
                      max_resample: int = 5) -> torch.Tensor:
    t = _resolve_type(A_type, Transitions)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    for attempt in range(max_resample):
        probs = sample_probs(prior, (n_states, n_states), dtype=dtype, device=device, eps_collapse=eps_collapse)
        if t == Transitions.SEMI.value:
            probs.fill_diagonal_(0.0)
        elif t == Transitions.LEFT_TO_RIGHT.value:
            probs *= torch.triu(torch.ones_like(probs, dtype=dtype, device=device))
        probs /= probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
        if (probs.max(dim=-1).values > 1.0 - eps_collapse).any():
            for i, row_max in enumerate(probs.max(dim=-1).values):
                if row_max > 1.0 - eps_collapse:
                    row = torch.ones(n_states, dtype=dtype, device=device)
                    if t == Transitions.SEMI.value: row[i] = 0
                    if t == Transitions.LEFT_TO_RIGHT.value: row[:i] = 0
                    probs[i] = row / row.sum().clamp_min(EPS)
        else:
            break
    else:
        logger.error("Max resample attempts reached; some transition rows may be nearly deterministic")

    if seed is not None:
        torch.set_rng_state(rng_state)
    return probs

# -------------------------
# Validation
# -------------------------
def is_valid_transition(probs: torch.Tensor, A_type: Union[str, Transitions], atol: float = 1e-6) -> bool:
    t = _resolve_type(A_type, Transitions)
    if not torch.isfinite(probs).all() or (probs < 0).any():
        logger.error("Transition matrix invalid: NaN or negatives")
        return False
    if not torch.allclose(probs.sum(-1), torch.ones(probs.shape[0], device=probs.device), atol=atol):
        logger.error("Transition matrix invalid: rows not normalized")
        return False
    if t == Transitions.SEMI.value and not torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal()), atol=atol):
        logger.error("SEMI transition invalid: diagonal nonzero")
        return False
    if t == Transitions.LEFT_TO_RIGHT.value and not torch.allclose(probs, torch.triu(probs), atol=atol):
        logger.error("L2R transition invalid: lower entries nonzero")
        return False
    return True

def compute_information_criteria(n_samples:int, log_likelihood:torch.Tensor, dof:int, criterion:Union[str,InformCriteria]) -> torch.Tensor:
    c = _resolve_type(criterion, InformCriteria)
    log_n = torch.log(torch.tensor(float(n_samples), dtype=log_likelihood.dtype, device=log_likelihood.device))
    penalties = {
        InformCriteria.AIC.value: 2*dof,
        InformCriteria.BIC.value: dof*log_n,
        InformCriteria.HQC.value: 2*dof*torch.log(log_n)
    }
    if c not in penalties: 
        logger.error(f"Invalid information criterion: {c}")
        raise ValueError(f"Invalid info criterion: {c}")
    return -2*log_likelihood + penalties[c]

def _assert_spd(matrix: torch.Tensor, label: str = "Matrix"):
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        logger.error(f"{label} not symmetric")
        raise ValueError(f"{label} not symmetric")
    _, info = torch.linalg.cholesky_ex(matrix)
    if info != 0:
        logger.error(f"{label} not positive-definite")
        raise ValueError(f"{label} not PD")

def validate_covars(covars: torch.Tensor, cov_type: Union[str,CovarianceType], n_states:int, n_features:int, eps:float=1e-6, auto_correct:bool=True) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c in (CovarianceType.SPHERICAL.value, CovarianceType.DIAG.value):
        return covars.clamp_min(eps) if auto_correct else covars
    if c == CovarianceType.TIED.value:
        _assert_spd(covars)
        return covars
    if c == CovarianceType.FULL.value:
        if covars.shape != (n_states, n_features, n_features):
            logger.error(f"FULL covars shape mismatch: expected {(n_states,n_features,n_features)}, got {covars.shape}")
            raise ValueError("Invalid FULL covars shape")
        for i in range(n_states):
            _assert_spd(covars[i], f"FULL covars state {i}")
        return covars
    raise NotImplementedError(f"Unsupported covar type: {c}")

def init_covars(base_cov: torch.Tensor, cov_type: Union[str,CovarianceType], n_states:int, n_features:int, eps:float=1e-6) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c == CovarianceType.SPHERICAL.value:
        return base_cov.mean().clamp_min(eps).repeat(n_states)
    if c == CovarianceType.DIAG.value:
        return torch.diag(base_cov).clamp_min(eps).unsqueeze(0).repeat(n_states,1)
    if c == CovarianceType.TIED.value:
        _assert_spd(base_cov)
        return base_cov
    if c == CovarianceType.FULL.value:
        _assert_spd(base_cov)
        return base_cov.unsqueeze(0).repeat(n_states,1,1)
    logger.error(f"Unsupported covar type: {c}")
    raise NotImplementedError(f"Unsupported covariance type: {c}")

def fill_covars(covars: torch.Tensor, cov_type: Union[str,CovarianceType], n_states:int, n_features:int, eps:float=1e-6) -> torch.Tensor:
    c = _resolve_type(cov_type, CovarianceType)
    if c == CovarianceType.FULL.value:
        return covars
    if c == CovarianceType.DIAG.value:
        return torch.diag_embed(covars.clamp_min(eps))
    if c == CovarianceType.TIED.value:
        _assert_spd(covars)
        return covars.unsqueeze(0).repeat(n_states,1,1)
    if c == CovarianceType.SPHERICAL.value:
        val = covars.clamp_min(eps)
        return torch.eye(n_features,dtype=val.dtype,device=val.device).unsqueeze(0) * val.view(-1,1,1)
    logger.error(f"Unsupported covar type: {c}")
    raise NotImplementedError(f"Unsupported covariance type: {c}")

def validate_lambdas(lambdas: torch.Tensor, n_states:int, n_features:int) -> torch.Tensor:
    if lambdas.shape != (n_states,n_features) or not torch.isfinite(lambdas).all() or (lambdas<=0).any():
        logger.error("Invalid lambdas tensor")
        raise ValueError("Invalid lambdas")
    return lambdas
