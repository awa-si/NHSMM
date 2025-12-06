# tests/dist_emission.py

import math
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Bernoulli, Poisson, Laplace, StudentT, MultivariateNormal
from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger
from nhsmm.distributions import Emission, Categorical

def set_seed(seed: int = 42):
    torch.manual_seed(seed)

# ---------------- Basic Functionality ----------------
def test_basic(em_type="gaussian"):
    print(f"\n=== TEST: Basic Functionality ({em_type}) ===")
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 3
    em = Emission(n_states=3, n_features=n_features, emission_type=em_type)
    
    base = getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None)))
    print("Base param shape:", base.shape)
    
    log_var = getattr(em, "log_var", None)
    if log_var is not None:
        print("log_var shape:", log_var.shape)
    
    scale_param = getattr(em, "scale_param", None)
    if scale_param is not None:
        print("Scale param shape:", scale_param.shape)
    
    dist = em._get_dist()
    print("Distribution type:", type(dist))
    
    param = None
    if hasattr(dist, "logits"):
        param = dist.logits
    elif hasattr(dist, "rate"):
        param = dist.rate
    elif isinstance(dist, torch.distributions.Independent):
        base_dist = dist.base_dist
        param = getattr(base_dist, "logits", getattr(base_dist, "rate", None))
    print("Distribution params shape:", None if param is None else param.shape)

# ---------------- Temperature Scaling ----------------
def test_temperature(em_type="gaussian"):
    print(f"\n=== TEST: Temperature Scaling ({em_type}) ===")
    em = Emission(n_states=3, n_features=3, emission_type=em_type)
    base = getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None)))
    cold = em._modulate(base, temperature=0.1).detach()
    hot = em._modulate(base, temperature=5.0).detach()
    print("Cold min/max:", cold.min().item(), cold.max().item())
    print("Hot min/max:", hot.min().item(), hot.max().item())
    print("Shape:", cold.shape, hot.shape)

# ---------------- Context Tests ----------------
def test_context_single(em_type="gaussian"):
    print(f"\n=== TEST: Context Single ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=4, hidden_dim=8)
    ctx = torch.randn(4, requires_grad=True)
    base = getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None)))
    mod = em._modulate(base, context=ctx)
    print("Context:", ctx)
    print("Modulated shape:", mod.shape)

def test_context_batch(em_type="gaussian"):
    print(f"\n=== TEST: Context Batch ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=4, hidden_dim=8)
    ctx = torch.randn(5, 4, requires_grad=True)
    base = getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None)))
    mod = em._modulate(base, context=ctx)
    print("Batch context shape:", ctx.shape)
    print("Modulated shape:", mod.shape)

# ---------------- Sampling ----------------
def test_sampling(em_type="gaussian"):
    print(f"\n=== TEST: Sampling ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    samples = em.sample(n_samples=5)
    print("Samples shape:", samples.shape)
    print("Samples dtype:", samples.dtype)

# ---------------- Log-Probability ----------------
def test_log_prob(em_type="gaussian"):
    print(f"\n=== TEST: Log-Probability ({em_type}) ===")
    n_features = 1 if em_type == "categorical" else 2
    em = Emission(n_states=2, n_features=n_features, emission_type=em_type)
    x = torch.randint(0, 2, (3, n_features)) if em_type in {"categorical"} else torch.randn(3, n_features)
    logp = em.log_prob(x)
    print("Input shape:", x.shape)
    print("Log-probs shape:", logp.shape)
    print("Log-probs min/max:", logp.min().item(), logp.max().item())

# ---------------- Gradient Flow ----------------
def test_gradient_flow(em_type="gaussian"):
    print(f"\n=== TEST: Gradient Flow ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=3, hidden_dim=6)
    ctx = torch.randn(1, 3, requires_grad=True)
    base = getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None)))
    mod = em._modulate(base, context=ctx)
    out = mod.sum()
    out.backward()
    grad_base = base.grad.norm().item() if base.grad is not None else None
    print("Gradient base norm:", grad_base)
    print("Gradient context:", ctx.grad)

# ---------------- Update ----------------
def test_update(em_type="gaussian"):
    print(f"\n=== TEST: update() ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    old_base = getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None))).clone()
    new_base = torch.randn_like(old_base)
    em.update(new_base=new_base, update_rate=0.5)
    print("Updated base diff norm:", (getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None))) - old_base).norm().item())
    posterior = torch.ones_like(old_base)/old_base.numel()
    em.update(posterior=posterior, update_rate=0.5)
    print("After posterior update base diff norm:", (getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", None))) - old_base).norm().item())

# ---------------- Initialize ----------------
def test_initialize(em_type="gaussian"):
    print(f"\n=== TEST: initialize() ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    X = torch.randn(10, 2)
    dist = em.initialize(X=X)
    loc = getattr(dist, "mean", getattr(dist, "loc", None))
    print("Initialized loc:", loc.shape if loc is not None else None)
    if hasattr(dist, "covariance_matrix"):
        print("Initialized cov:", dist.covariance_matrix.shape)

# ---------------- Main ----------------
if __name__ == "__main__":
    set_seed()
    for em_type in ["gaussian", "laplace", "studentt", "categorical", "bernoulli", "poisson"]:
        test_basic(em_type)
        test_temperature(em_type)
        test_context_single(em_type)
        test_context_batch(em_type)
        test_sampling(em_type)
        test_log_prob(em_type)
        test_gradient_flow(em_type)
        test_update(em_type)
        test_initialize(em_type)
    print("\n✓ All Emission enhanced tests completed.")
