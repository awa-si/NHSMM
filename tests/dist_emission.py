import torch
import torch.nn.functional as F
from nhsmm.constants import DTYPE, EPS
from nhsmm.distributions import Emission
from nhsmm.context import DefaultEncoder, ContextEncoder

def set_seed(seed: int = 42):
    torch.manual_seed(seed)

def get_base(em: Emission):
    return getattr(em, "base")

def assert_tensor_safe(tensor: torch.Tensor, name="tensor"):
    assert not torch.isnan(tensor).any(), f"NaNs detected in {name}"
    assert not torch.isinf(tensor).any(), f"Infs detected in {name}"

@torch.no_grad()
def get_emission_params(em: Emission):
    if em.emission_type == "gaussian":
        mu = em.mu.detach().clone()
        var = F.softplus(em.log_var).detach().clone()
        return mu, var, None
    else:
        loc = em.loc.detach().clone()
        scale = F.softplus(em.scale_param).detach().clone()
        df = F.softplus(em.dof).detach().clone() + 2.0
        return loc, scale, df

def test_basic(em_type="gaussian"):
    print(f"\n=== TEST: Basic Functionality ({em_type}) ===")
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 3
    em = Emission(n_states=3, n_features=n_features, emission_type=em_type)
    base = get_base(em)
    print("Base param shape:", base.shape)
    dist = em._get_dist()
    param = getattr(dist, "logits", getattr(dist, "rate", getattr(dist, "loc", None)))
    print("Distribution type:", type(dist))
    print("Distribution params shape:", None if param is None else param.shape)
    if em_type == "studentt" and hasattr(dist, "df"):
        print("Degrees of freedom shape:", dist.df.shape)

def test_temperature(em_type="gaussian"):
    print(f"\n=== TEST: Temperature Scaling ({em_type}) ===")
    em = Emission(n_states=3, n_features=3, emission_type=em_type)
    base = get_base(em)
    cold = em._modulate(base, temperature=0.1).detach()
    hot = em._modulate(base, temperature=5.0).detach()
    default = em._modulate(base, temperature=None).detach()
    print("Cold min/max:", cold.min().item(), cold.max().item())
    print("Hot min/max:", hot.min().item(), hot.max().item())
    print("Default min/max:", default.min().item(), default.max().item())
    print("Shapes:", cold.shape, hot.shape, default.shape)
    for tensor, name in zip([cold, hot, default], ["cold", "hot", "default"]):
        assert_tensor_safe(tensor, name)

def test_context_modulation(em_type="gaussian"):
    print(f"\n=== TEST: Context Modulation ({em_type}) ===")
    em = Emission(
        n_states=2,
        n_features=2,
        emission_type=em_type,
        context_dim=4,
        hidden_dim=8,
        modulate_var=True,
    )
    ctx = torch.randn(4, requires_grad=True)
    mod = em._modulate(context=ctx)
    print("Modulated shape:", mod.shape)
    assert mod.ndim in (3, 4)
    assert mod.shape[-2:] == (2, 2)

def test_sampling(em_type="gaussian"):
    print(f"\n=== TEST: Sampling ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    with torch.random.fork_rng():
        set_seed()
        samples = em.sample()
    print("Samples shape:", samples.shape, "dtype:", samples.dtype)
    assert_tensor_safe(samples)
    if em_type in {"categorical", "bernoulli", "poisson"}:
        assert samples.min() >= 0

def test_expected_probs(em_type="gaussian"):
    print(f"\n=== TEST: Expect Probabilities ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=4, hidden_dim=8)
    probs = em.expected_probs()
    print("Expect probs shape:", probs.shape)
    assert_tensor_safe(probs)
    if em_type in {"categorical", "bernoulli"}:
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums)), "Probabilities do not sum to 1"

def test_log_prob(em_type="gaussian"):
    print(f"\n=== TEST: Log-Probability ({em_type}) ===")
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 2
    em = Emission(n_states=2, n_features=n_features, emission_type=em_type)
    B, F = 3, n_features
    x = torch.randint(0, em.n_states, (B, F))
    logp = em.log_prob(x)
    print("Input shape:", x.shape)
    print("Log-probs shape:", logp.shape)
    print("Log-probs min/max:", logp.min().item(), logp.max().item())

def test_gradient_flow(em_type="gaussian"):
    print(f"\n=== TEST: Gradient Flow ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=3, hidden_dim=6)
    ctx = torch.randn(1, 3, requires_grad=True)
    base = get_base(em)
    mod = em._modulate(context=ctx)
    out = mod.sum()
    out.backward()
    grad_base_norm = base.grad.norm().item() if base.grad is not None else None
    print("Gradient base norm:", grad_base_norm, "Context grad:", ctx.grad)

def test_initialize(em_type="gaussian", context=None):
    print(f"\n=== TEST: initialize() ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    dist = em.initialize(context=context)
    loc, var, df = get_emission_params(em)
    print("Initialized loc:", None if loc is None else loc.shape)
    if var is not None:
        print("Initialized variance/scale:", var.shape)
    if df is not None:
        print("Initialized df:", df.shape)

def test_update(em_type="gaussian"):
    print(f"\n=== TEST: update() ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    X = torch.randn(6, 2)
    base_before = get_base(em).detach().clone()
    posterior = torch.full((6, 2), 0.5)
    em.update(X=X, posterior=posterior, update_rate=0.5)
    base_after = get_base(em).detach()
    diff_norm = (base_after - base_before).norm().item()
    print("Posterior update diff norm:", diff_norm)
    assert diff_norm > 0

def test_em_e_step(em_type="gaussian"):
    print(f"\n=== TEST: test_em_e_step() ({em_type}) ===")
    em = Emission(n_states=3, n_features=2, emission_type=em_type)
    X = torch.randn(4, 2)
    logp = em.log_prob(X)
    posterior = F.softmax(logp, dim=-1)
    print("E-step posterior shape:", posterior.shape)
    base_before = get_base(em).detach().clone()
    em.update(X=X, posterior=posterior)
    diff = (get_base(em).detach() - base_before).norm().item()
    print("M-step parameter diff norm:", diff)
    assert diff > 0

def test_full_em_iteration(em_type="gaussian"):
    print(f"\n=== TEST: test_full_em_iteration() ({em_type}) ===")
    em = Emission(n_states=3, n_features=2, emission_type=em_type)
    X = torch.randn(32, 2)
    with torch.no_grad():
        for p in em.parameters():
            p += 1e-3 * torch.randn_like(p)
    base_before = get_base(em).detach().clone()
    print("Initial base params:\n", base_before)
    logp = em.log_prob(X)
    posterior = F.softmax(logp, dim=-1)
    mean_ll_before = (logp * posterior).mean().item()
    print("Initial mean log-likelihood:", mean_ll_before)
    em.update(X=X, posterior=posterior)
    base_after = get_base(em).detach()
    print("Base params after EM step:\n", base_after)
    delta = (base_after - base_before).abs().sum().item()
    logp_after = em.log_prob(X)
    mean_ll_after = (logp_after * posterior).mean().item()
    ll_delta = mean_ll_after - mean_ll_before
    print("Parameter delta (L1 sum):", delta)
    print("Post-EM mean log-likelihood:", mean_ll_after)
    print("Likelihood delta:", ll_delta)
    assert delta > 0

def test_em_likelihood_monotonicity():
    set_seed()
    B, T, F = 2, 4, 3
    K = 3
    X = torch.randn(B, T, F)
    em = Emission(n_states=K, n_features=F, emission_type="gaussian")
    em.initialize()
    def compute_weighted_ll(em, X, posterior):
        logp = em.log_prob(X)
        return (posterior * logp).sum().item(), logp.min().item(), logp.max().item()
    with torch.no_grad():
        logp = em.log_prob(X)
        posterior = F.softmax(logp, dim=-1)
    ll_prev, logp_min, logp_max = compute_weighted_ll(em, X, posterior)
    print(f"Initial weighted log-likelihood: {ll_prev:.6f}")
    for step in range(5):
        em.update(posterior=posterior, update_rate=1.0)
        ll_new, logp_min, logp_max = compute_weighted_ll(em, X, posterior)
        print(f"After M-step {step+1} Weighted log-likelihood: {ll_new:.6f}")
        assert ll_new + EPS >= ll_prev
        ll_prev = ll_new
    print("✓ EM likelihood monotonicity test passed.")

def test_em_variance_positive(em_type="gaussian"):
    print(f"\n=== TEST: Variance/Scale Positive ({em_type}) ===")
    set_seed()
    B, T, F = 3, 3, 4
    K = 3
    X = torch.randn(B, T, F) * 10.0

    em = Emission(n_states=K, n_features=F, emission_type=em_type)
    with torch.no_grad():
        posterior = torch.nn.functional.softmax(em.log_prob(X), dim=-1)

    for step in range(5):
        em.update(posterior=posterior)
        loc, var, df = get_emission_params(em)
        if var is not None:
            assert torch.all(var > 0), "Scale/variance contains non-positive values"
            assert torch.isfinite(var).all()
        if df is not None:
            assert torch.all(df > 0), "Degrees of freedom contains non-positive values"
        print(f"Step {step+1}: loc min/max {loc.min().item():.6f}/{loc.max().item():.6f}, "
              f"var/scale min/max {var.min().item():.6f}/{var.max().item():.6f}" +
              (f", df min/max {df.min().item():.6f}/{df.max().item():.6f}" if df is not None else ""))

def test_em_parameters_move(em_type="gaussian"):
    print(f"\n=== TEST: EM Parameter Movement ({em_type}) ===")
    set_seed()
    B, T, F = 3, 5, 2
    K = 2
    X = torch.randn(B, T, F)

    em = Emission(n_states=K, n_features=F, emission_type=em_type)
    
    # Initial posterior (can start uniform or skewed)
    posterior = torch.zeros(B, T, K)
    posterior[..., 0] = 0.75
    posterior[..., 1] = 0.25

    loc_before, var_before, df_before = get_emission_params(em)
    loc_moved = False
    var_moved = False

    for step in range(3):
        # --- M-step ---
        em.update(X=X, posterior=posterior)
        loc_curr, var_curr, df_curr = get_emission_params(em)

        print(f"\nAfter M-step {step+1}: loc min/max {loc_curr.min().item():.6f}/{loc_curr.max().item():.6f}, "
              f"var/scale min/max {var_curr.min().item():.6f}/{var_curr.max().item():.6f}" +
              (f", df min/max {df_curr.min().item():.6f}/{df_curr.max().item():.6f}" if df_curr is not None else ""))

        # --- Check for movement ---
        if not torch.allclose(loc_curr, loc_before):
            loc_moved = True
        if var_curr is not None and not torch.allclose(var_curr, var_before):
            var_moved = True

        # Update 'posterior' to mimic EM E-step recomputation
        # Here we slightly perturb posterior for testing, in real EM you'd recompute it
        posterior = posterior + torch.randn_like(posterior) * 0.01
        posterior = posterior.clamp_min(1e-6)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)

        loc_before = loc_curr.clone()
        if var_curr is not None:
            var_before = var_curr.clone()

    assert loc_moved, f"Location did not change at all after EM updates for {em_type}"
    if var_curr is not None:
        assert var_moved, f"Variance/scale did not change at all after EM updates for {em_type}"

    print("EM parameter movement test passed.")

def test_em_respects_posterior_weights(em_type="gaussian"):
    print(f"\n=== TEST: Posterior Weighting ({em_type}) ===")
    set_seed()
    B, T, F = 2, 4, 2
    K = 2
    X = torch.randn(B, T, F)

    em = Emission(n_states=K, n_features=F, emission_type=em_type)
    loc_before, var_before, df_before = get_emission_params(em)

    posterior = torch.zeros(B, T, K)
    posterior[..., 0] = 1.0  # all weight to state 0

    print("Before update loc:\n", loc_before)
    em.update(X=X, posterior=posterior)
    loc_after, var_after, df_after = get_emission_params(em)

    print("After update loc:\n", loc_after)
    print("After update var/scale:\n", var_after)
    if df_after is not None:
        print("After update df:\n", df_after)

    # Check state 0 moved
    moved_norm = torch.norm(loc_after[0] - loc_before[0])
    assert moved_norm > 1e-6, f"State 0 loc not updated correctly, movement {moved_norm:.6f}"

    # Check state 1 did NOT move
    assert torch.allclose(loc_after[1], loc_before[1], atol=1e-6), "State 1 loc incorrectly updated"

    # Validate positive variance/scale
    if var_after is not None:
        assert torch.all(var_after > 0), "Variance/scale contains non-positive values"
    if df_after is not None:
        assert torch.all(df_after > 0), "Degrees of freedom contains non-positive values"

    print("✓ Posterior weighting respected in M-step")

# Include in main test run
if __name__ == "__main__":
    set_seed()
    for em_type in ["gaussian", "studentt"]:
        test_basic(em_type)
        test_initialize(em_type)
        test_temperature(em_type)
        test_sampling(em_type)
        test_log_prob(em_type)
        test_expected_probs(em_type)
        test_gradient_flow(em_type)
        test_update(em_type)
        test_em_e_step(em_type)
        test_full_em_iteration(em_type)
        test_em_variance_positive(em_type)
        test_em_parameters_move(em_type)
        test_em_respects_posterior_weights(em_type)

    print("\n✓ All Emission tests completed.")
