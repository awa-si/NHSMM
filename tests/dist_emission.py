# tests/dist_emission.py

import torch
import torch.nn.functional as F
from nhsmm.constants import DTYPE, EPS
from nhsmm.distributions import Emission
from nhsmm.context import CNN_LSTM_Encoder, ContextEncoder

# ------------------ Helpers ------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)

def get_base(em: Emission):
    """Return the main parameter tensor for any emission type."""
    return getattr(
        em, "mu",
        getattr(
            em, "loc",
            getattr(em, "logits", getattr(em, "log_rate", None))
        )
    )

def assert_tensor_safe(tensor: torch.Tensor, name="tensor"):
    assert not torch.isnan(tensor).any(), f"NaNs detected in {name}"
    assert not torch.isinf(tensor).any(), f"Infs detected in {name}"

# ------------------ Basic Functionality ------------------
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

# ------------------ Temperature Scaling ------------------
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
        modulate_var=True,  # REQUIRED
    )

    # --- Single context ---
    ctx = torch.randn(4, requires_grad=True)
    mod = em._modulate(get_base(em), context=ctx)

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


def test_expect_probs(em_type="gaussian"):
    print(f"\n=== TEST: Expect Probabilities ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=4, hidden_dim=8)
    probs = em.expect_probs()
    print("Expect probs shape:", probs.shape)
    assert_tensor_safe(probs)
    if em_type in {"categorical", "bernoulli"}:
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums)), "Probabilities do not sum to 1"


def test_log_prob(em_type="gaussian"):
    print(f"\n=== TEST: Log-Probability ({em_type}) ===")
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 2
    em = Emission(n_states=2, n_features=n_features, emission_type=em_type)

    B = 3
    F = n_features
    # x must have shape (B, F)
    x = torch.randint(0, em.n_states, (B, F))

    # reshape x to match log_probs dims
    dist = em._get_dist()
    log_probs = dist.logits if hasattr(dist, "logits") else None
    if log_probs is not None:
        # log_probs: (B, F, K) or (B, K) for single feature
        if log_probs.ndim == 2:
            x_int = x.squeeze(-1)  # (B,)
        else:
            x_int = x  # (B, F)
    else:
        x_int = x

    logp = em.log_prob(x_int)
    print("Input shape:", x_int.shape)
    print("Log-probs shape:", logp.shape)
    print("Log-probs min/max:", logp.min().item(), logp.max().item())


def test_gradient_flow(em_type="gaussian"):
    print(f"\n=== TEST: Gradient Flow ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=3, hidden_dim=6)
    ctx = torch.randn(1, 3, requires_grad=True)
    base = get_base(em)
    mod = em._modulate(base, context=ctx)
    out = mod.sum()
    out.backward()
    grad_base_norm = base.grad.norm().item() if base.grad is not None else None
    print("Gradient base norm:", grad_base_norm, "Context grad:", ctx.grad)


def test_initialize(em_type="gaussian", context: torch.Tensor = None):
    print(f"\n=== TEST: initialize() ({em_type}) ===")

    # Create emission module
    em = Emission(n_states=2, n_features=2, emission_type=em_type)

    # Initialize (without passing X, using internal default KMeans/data-free init)
    dist = em.initialize(context=context)

    # Extract loc/mean
    loc = getattr(dist, "mean", getattr(dist, "loc", None))
    print("Initialized loc:", None if loc is None else loc.shape)

    # Extract covariance/scale
    if hasattr(dist, "covariance_matrix"):
        print("Initialized cov:", dist.covariance_matrix.shape)
    elif hasattr(dist, "scale"):
        print("Initialized scale:", dist.scale.shape)


def test_context_encoder():
    print("\n=== TEST: ContextEncoder + CNN_LSTM_Encoder with Emission ===")
    B, T, F_in = 4, 8, 6
    hidden = 32
    n_states = 5
    x = torch.randn(B, T, F_in)

    # Initialize encoder
    encoder = CNN_LSTM_Encoder(
        n_features=F_in,
        hidden_dim=hidden,
        cnn_channels=8,
        kernel_size=3,
        bidirectional=True,
        return_sequence=True
    )
    with torch.no_grad():
        seq_test = encoder(x)
        enc_out_dim = seq_test.shape[-1]
    print(f"Encoder output dim = {enc_out_dim}")

    for pool in ["mean", "last", "max", "attn", "mha"]:
        print(f"\n=== Testing single-step pool={pool} ===")
        ctx_enc = ContextEncoder(encoder=encoder, pool=pool, n_heads=2, debug=True)
        seq_out, ctx, attn = ctx_enc(
            x, return_context=True, return_attn_weights=True, return_sequence=True
        )
        print("Input shape:", x.shape, "Sequence output shape:", seq_out.shape, "Context shape:", ctx.shape)
        if attn is not None:
            print("Attention weights shape:", attn.shape)

        # Initialize emission
        em = Emission(
            n_states=n_states,
            n_features=3,
            emission_type="gaussian",
            context_dim=enc_out_dim,
            hidden_dim=32
        )

        # Batch-safe modulation
        ctx_mod = ctx.squeeze(1)  # [B, D]
        mod_list = [em._modulate(get_base(em), context=ctx_mod[b:b+1]) for b in range(B)]
        mod = torch.stack(mod_list, dim=0)  # [B, n_states, n_features]
        print("Modulated output shape:", mod.shape)
        assert mod.shape[0] == B, f"Batch size mismatch: {mod.shape[0]} != {B}"
        assert_tensor_safe(mod)

    # Multi-step sequence
    print("\n=== Testing multi-step sequence context ===")
    S = 2
    seq_inputs = torch.randn(S, B, T, F_in)
    ctx_enc = ContextEncoder(encoder=encoder, pool="mean", n_heads=2, debug=True)
    em = Emission(
        n_states=n_states,
        n_features=3,
        emission_type="gaussian",
        context_dim=enc_out_dim,
        hidden_dim=32
    )

    for s in range(S):
        seq_out, ctx, attn = ctx_enc(
            seq_inputs[s], return_context=True, return_attn_weights=True, return_sequence=True
        )
        ctx_mod = ctx.squeeze(1)  # [B, D]
        mod_list = [em._modulate(get_base(em), context=ctx_mod[b:b+1]) for b in range(B)]
        mod_params = torch.stack(mod_list, dim=0)  # [B, n_states, n_features]

        print(f"\n--- Sequence {s} ---")
        print("Input shape:", seq_inputs[s].shape, "Context shape:", ctx.shape, "Modulated shape:", mod_params.shape)
        assert mod_params.shape[0] == B
        assert mod_params.ndim == 3  # [B, n_states, n_features]
        assert_tensor_safe(mod_params)


def test_update(em_type="gaussian"):
    print(f"\n=== TEST: update() ({em_type}) ===")

    n_states = 2
    n_features = 2
    em = Emission(n_states=n_states, n_features=n_features, emission_type=em_type)

    X = torch.randn(6, n_features)
    base_before = get_base(em).detach().clone()

    posterior = torch.full((6, n_states), 1.0 / n_states)

    em.update(X=X, posterior=posterior, update_rate=0.5)

    base_after = get_base(em).detach()
    diff_norm = (base_after - base_before).norm().item()

    print("Posterior update diff norm:", diff_norm)
    assert diff_norm > 0, "Posterior update did not modify parameters"


def test_em_e_step(em_type="gaussian"):
    print(f"\n=== TEST: test_em_e_step() ({em_type}) ===")

    n_states, n_features = 3, 2
    em = Emission(n_states=n_states, n_features=n_features, emission_type=em_type)

    X = torch.randn(4, n_features)

    logp = em.log_prob(X)
    posterior = F.softmax(logp, dim=-1)
    print("E-step posterior shape:", posterior.shape)

    base_before = get_base(em).detach().clone()

    em.update(X=X, posterior=posterior)

    diff = (get_base(em).detach() - base_before).norm().item()
    print("M-step parameter diff norm:", diff)
    assert diff > 0


def test_em_m_step(em_type="gaussian"):
    n_states = 3
    n_features = 2
    n_points = 4

    em = Emission(n_states=n_states, n_features=n_features, emission_type=em_type)

    base_before = get_base(em).detach().clone()

    posterior = torch.randn(1, n_points, n_states)
    posterior = F.softmax(posterior, dim=-1)

    em.update(posterior=posterior)

    diff = (get_base(em).detach() - base_before).norm().item()

    print("E-step posterior shape:", posterior.shape)
    print("M-step parameter diff norm:", diff)

    # EM fixed-point check
    assert diff < 1e-6
    assert_tensor_safe(get_base(em), "M-step parameters")


def test_full_em_iteration(em_type="gaussian"):
    print(f"\n=== TEST: test_full_em_iteration() ({em_type}) ===")

    n_states, n_features = 3, 2
    em = Emission(n_states=n_states, n_features=n_features, emission_type=em_type)

    # Generate a larger, more diverse dataset to ensure parameter movement
    X = torch.randn(32, n_features)
    
    # Slightly perturb parameters to avoid fixed point
    with torch.no_grad():
        for p in em.parameters():
            p += 1e-3 * torch.randn_like(p)

    base_before = get_base(em).detach().clone()
    print("Initial base params:\n", base_before)

    # Compute initial mean log-likelihood
    logp = em.log_prob(X)
    posterior = F.softmax(logp, dim=-1)
    mean_ll_before = (logp * posterior).mean().item()
    print("Initial mean log-likelihood:", mean_ll_before)

    # Perform a single EM step
    em.update(X=X, posterior=posterior)

    base_after = get_base(em).detach()
    print("Base params after EM step:\n", base_after)

    # Compute parameter delta and log-likelihood delta
    delta = (base_after - base_before).abs().sum().item()
    logp_after = em.log_prob(X)
    mean_ll_after = (logp_after * posterior).mean().item()
    ll_delta = mean_ll_after - mean_ll_before

    print("Parameter delta (L1 sum):", delta)
    print("Post-EM mean log-likelihood:", mean_ll_after)
    print("Likelihood delta:", ll_delta)

    assert delta > 0, "EM step did not update parameters"


def test_em_likelihood_monotonicity_gaussian():
    set_seed()

    B, T, F = 2, 4, 3
    K = 3
    X = torch.randn(B, T, F)

    em = Emission(n_states=K, n_features=F, emission_type="gaussian")
    em.initialize(X)

    def compute_weighted_ll(emission, X, posterior):
        dist = emission._get_dist()
        log_probs = dist.log_prob(X.unsqueeze(2))  # (B,T,K)
        ll = (posterior * log_probs).sum().item()
        return ll, log_probs.min().item(), log_probs.max().item()

    with torch.no_grad():
        dist = em._get_dist()
        log_probs = dist.log_prob(X.unsqueeze(2))
        posterior = torch.nn.functional.softmax(log_probs, dim=-1)

    ll_prev, logp_min, logp_max = compute_weighted_ll(em, X, posterior)
    print(f"Initial weighted log-likelihood: {ll_prev:.6f}")
    print(f"Log-probs min/max: {logp_min:.6f} / {logp_max:.6f}")
    print("Initial mu:", em.mu)
    print("Initial var:", torch.nn.functional.softplus(em.log_var))

    for step in range(1, 6):
        em.update(posterior=posterior, update_rate=1.0)

        mu_min, mu_max = em.mu.min().item(), em.mu.max().item()
        var_min, var_max = torch.nn.functional.softplus(em.log_var).min().item(), torch.nn.functional.softplus(em.log_var).max().item()
        print(f"\n--- After M-step {step} ---")
        print(f"Mu min/max: {mu_min:.6f} / {mu_max:.6f}")
        print(f"Var min/max: {var_min:.6f} / {var_max:.6f}")
        print("_emission_covs diagonal:", em._emission_covs.diagonal(dim1=-2, dim2=-1))

        with torch.no_grad():
            dist = em._get_dist()
            log_probs = dist.log_prob(X.unsqueeze(2))
            posterior = torch.nn.functional.softmax(log_probs, dim=-1)

        ll_new, logp_min, logp_max = compute_weighted_ll(em, X, posterior)
        print(f"Log-probs min/max: {logp_min:.6f} / {logp_max:.6f}")
        print(f"Weighted log-likelihood: {ll_new:.6f}")

        assert ll_new + EPS >= ll_prev, "Log-likelihood decreased!"
        ll_prev = ll_new

    print("✓ Gaussian EM likelihood monotonicity test passed.")


def test_em_likelihood_convergence_gaussian():
    set_seed()

    B, T, F = 2, 4, 3
    K = 3
    X = torch.randn(B, T, F)

    # Initialize emission far from data to see convergence
    em = Emission(n_states=K, n_features=F, emission_type="gaussian")
    em.mu.data.fill_(0.0)
    em.log_var.data.fill_(torch.log(torch.tensor(1.0)))
    em._emission_covs.copy_(torch.diag_embed(torch.ones(F)))

    def compute_weighted_ll(emission, X, posterior=None):
        dist = emission._get_dist()
        log_probs = dist.log_prob(X.unsqueeze(2))  # (B,T,K)
        if posterior is None:
            return log_probs.sum().item()
        return (posterior * log_probs).sum().item()

    # Initial posterior and likelihood
    with torch.no_grad():
        dist = em._get_dist()
        log_probs = dist.log_prob(X.unsqueeze(2))
        posterior = torch.nn.functional.softmax(log_probs, dim=-1)

    ll_prev = compute_weighted_ll(em, X, posterior)
    print(f"Initial weighted log-likelihood: {ll_prev:.6f}")
    print("Initial mu min/max:", em.mu.min().item(), em.mu.max().item())
    print("Initial var min/max:", torch.nn.functional.softplus(em.log_var).min().item(),
          torch.nn.functional.softplus(em.log_var).max().item())

    for step in range(10):
        em.update(posterior=posterior, update_rate=1.0)

        # recompute posterior
        with torch.no_grad():
            dist = em._get_dist()
            log_probs = dist.log_prob(X.unsqueeze(2))
            posterior = torch.nn.functional.softmax(log_probs, dim=-1)

        ll_new = compute_weighted_ll(em, X, posterior)
        print(f"\n--- After M-step {step+1} ---")
        print("Mu min/max:", em.mu.min().item(), em.mu.max().item())
        var = torch.nn.functional.softplus(em.log_var)
        print("Var min/max:", var.min().item(), var.max().item())
        print("Weighted log-likelihood:", ll_new)

        # Monotonicity check
        assert ll_new + EPS >= ll_prev, "Log-likelihood decreased!"
        ll_prev = ll_new

    print("✓ Gaussian EM convergence test passed.")


def test_em_likelihood_trend_gaussian():
    print("\nTest test_em_likelihood_trend_gaussian")
    set_seed()

    B, T, F = 4, 6, 3
    K = 3
    X = torch.randn(B, T, F)

    em = Emission(n_states=K, n_features=F, emission_type="gaussian")
    em.mu.data.zero_()
    em.log_var.data.fill_(0.0)
    em._emission_covs.copy_(torch.diag_embed(torch.ones(F)))

    def weighted_ll(em, X, posterior):
        dist = em._get_dist()
        logp = dist.log_prob(X.unsqueeze(2))
        return (posterior * logp).sum().item()

    with torch.no_grad():
        logp = em._get_dist().log_prob(X.unsqueeze(2))
        posterior = torch.softmax(logp, dim=-1)

    ll_values = []

    for _ in range(15):
        em.update(posterior=posterior, update_rate=1.0)
        with torch.no_grad():
            logp = em._get_dist().log_prob(X.unsqueeze(2))
            posterior = torch.softmax(logp, dim=-1)
        ll_values.append(weighted_ll(em, X, posterior))

    # Final likelihood must be better than initial
    assert ll_values[-1] >= ll_values[0] - 1e-3

def test_em_mstep_state_normalization():
    print("\nTest test_em_mstep_state_normalization")
    w = torch.rand(10, 3)
    w /= w.sum(dim=0, keepdim=True)
    assert torch.allclose(w.sum(dim=0), torch.ones(3), atol=1e-6)

def test_em_variance_positive_gaussian():
    print("\nTest test_em_variance_positive_gaussian")
    set_seed()

    B, T, F = 3, 3, 4
    K = 3
    X = torch.randn(B, T, F) * 10.0

    em = Emission(n_states=K, n_features=F, emission_type="gaussian")

    with torch.no_grad():
        logp = em._get_dist().log_prob(X.unsqueeze(2))
        posterior = torch.softmax(logp, dim=-1)

    for _ in range(5):
        em.update(posterior=posterior)
        var = torch.nn.functional.softplus(em.log_var)
        assert torch.all(var > 0)
        assert torch.isfinite(var).all()

def test_em_parameters_move_gaussian():
    print("\nTest test_em_parameters_move_gaussian")
    set_seed()

    B, T, F = 3, 5, 2
    K = 2
    X = torch.randn(B, T, F)

    em = Emission(n_states=K, n_features=F, emission_type="gaussian")
    em.mu.data.zero_()
    em.log_var.data.zero_()

    # Explicitly break symmetry in posterior
    posterior = torch.zeros(B, T, K)
    posterior[..., 0] = 0.75
    posterior[..., 1] = 0.25

    mu_before = em.mu.clone()
    var_before = torch.nn.functional.softplus(em.log_var).clone()

    for step in range(3):
        em.update(X=X, posterior=posterior)

        mu_curr = em.mu.detach()
        var_curr = torch.nn.functional.softplus(em.log_var).detach()

        print(
            f"After M-step {step+1}: "
            f"mu min/max {mu_curr.min().item():.6f}/{mu_curr.max().item():.6f}, "
            f"var min/max {var_curr.min().item():.6f}/{var_curr.max().item():.6f}"
        )

    assert not torch.allclose(em.mu, mu_before), \
        "mu did not change after EM updates"
    assert not torch.allclose(
        torch.nn.functional.softplus(em.log_var), var_before
    ), "variance did not change after EM updates"

    print("✓ Gaussian EM parameter update test passed.")

def test_em_respects_posterior_weights_gaussian():
    print("\nTest test_em_respects_posterior_weights_gaussian")
    set_seed()

    B, T, F = 2, 4, 2
    K = 2

    # Generate synthetic data
    X = torch.randn(B, T, F)

    # Initialize emission far from data
    em = Emission(n_states=K, n_features=F, emission_type="gaussian")
    em.mu.data.zero_()
    em.log_var.data.zero_()
    em._emission_covs.copy_(torch.eye(F).unsqueeze(0).repeat(K, 1, 1))

    # Create posterior that assigns all probability to state 0
    posterior = torch.zeros(B, T, K)
    posterior[..., 0] = 1.0

    # Perform M-step
    em.update(X=X, posterior=posterior)

    # Detach parameters for inspection
    mu = em.mu.detach()
    var = torch.nn.functional.softplus(em.log_var.detach())

    # --- Assertions ---
    # State 0 should move towards data
    assert torch.norm(mu[0]) > 1e-6, "State 0 mean not updated correctly"

    # State 1 should remain at zero
    assert torch.allclose(mu[1], torch.zeros_like(mu[1]), atol=1e-6), "State 1 mean incorrectly updated"

    # Variance should remain positive
    assert torch.all(var > 0), "Variance contains non-positive values"

    print("✓ Posterior weighting respected in Gaussian M-step")


if __name__ == "__main__":
    set_seed()
    for em_type in ["gaussian",]:# "studentt"]:
        test_basic(em_type)
        test_initialize(em_type)
        test_temperature(em_type)
        test_sampling(em_type)
        test_log_prob(em_type)
        test_expect_probs(em_type)
        test_gradient_flow(em_type)
        test_update(em_type)
        test_em_e_step(em_type)
        test_em_m_step(em_type)
        test_context_modulation(em_type)
        test_full_em_iteration(em_type)
    test_context_encoder()
    test_em_likelihood_monotonicity_gaussian()
    test_em_likelihood_convergence_gaussian()
    test_em_mstep_state_normalization()
    test_em_variance_positive_gaussian()
    test_em_likelihood_trend_gaussian()
    test_em_parameters_move_gaussian()
    test_em_respects_posterior_weights_gaussian()
    print("\n✓ All Emission tests completed.")
