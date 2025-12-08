# tests/dist_emission.py

import math
import torch
import torch.nn.functional as F
from nhsmm.constants import DTYPE, EPS
from nhsmm.distributions import Emission
from nhsmm.context import CNN_LSTM_Encoder, ContextEncoder

# ------------------ ContextEncoder + CNN_LSTM_Encoder Integration ------------------
def test_context_encoder():
    print("\n=== TEST: ContextEncoder + CNN_LSTM_Encoder with Emission ===")

    B, T, F_in = 4, 8, 6
    hidden = 32
    n_states = 5

    x = torch.randn(B, T, F_in)

    # --- CNN + LSTM Encoder ---
    encoder = CNN_LSTM_Encoder(
        n_features=F_in,
        hidden_dim=hidden,
        cnn_channels=8,
        kernel_size=3,
        bidirectional=True,
        return_sequence=True
    )

    # Dynamically obtain encoder output dim
    with torch.no_grad():
        seq_test = encoder(x)
        enc_out_dim = seq_test.shape[-1]

    print(f"Encoder output dim = {enc_out_dim}")

    # --- Test each pooling method with single-step context ---
    for pool in ["mean", "last", "max", "attn", "mha"]:
        print(f"\n=== Testing single-step pool={pool} ===")

        ctx_enc = ContextEncoder(
            encoder=encoder,
            pool=pool,
            n_heads=2,
            debug=True
        )

        seq_out, ctx, attn = ctx_enc(
            x,
            return_context=True,
            return_attn_weights=True,
            return_sequence=True
        )

        print("Input shape:", x.shape)
        print("Sequence output shape:", seq_out.shape)
        print("Context shape:", ctx.shape)

        if attn is not None:
            print("Attention weights shape:", attn.shape)

        # Emission uses encoder output dim
        em = Emission(
            n_states=n_states,
            n_features=3,
            emission_type="gaussian",
            context_dim=enc_out_dim,
            hidden_dim=32
        )

        base = get_base(em)
        mod = em._modulate(base, context=ctx.squeeze(1))

        print("Modulated emission param shape:", mod.shape)
        assert mod.shape[0] == B

    # ------------------------------------------------------------------
    # MULTI-STEP SEQUENCE CONTEXT TEST
    # ------------------------------------------------------------------
    print("\n=== Testing multi-step sequence context ===")
    S = 2
    seq_inputs = torch.randn(S, B, T, F_in)

    # Create fresh pooling encoder for multi-step
    ctx_enc = ContextEncoder(
        encoder=encoder,
        pool="mean",     # irrelevant; context per sequence still works
        n_heads=2,
        debug=True
    )

    em = Emission(
        n_states=n_states,
        n_features=3,
        emission_type="gaussian",
        context_dim=enc_out_dim,
        hidden_dim=32
    )

    for s in range(S):
        seq_out, ctx, attn = ctx_enc(
            seq_inputs[s],
            return_context=True,
            return_attn_weights=True,
            return_sequence=True
        )

        mod_params = em._modulate(get_base(em), context=ctx.squeeze(1))

        print(f"\n--- Sequence {s} ---")
        print("Input shape:", seq_inputs[s].shape)
        print("Context shape:", ctx.shape)
        print("Modulated emission param shape:", mod_params.shape)

        # Ensure (B, 1, param_dim)
        assert mod_params.shape[0] == B
        assert mod_params.ndim in (3, 4)

# ------------------ Helpers ------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)

def get_base(em: Emission):
    """Return the main parameter tensor for any emission type."""
    return getattr(em, "mu", getattr(em, "loc", getattr(em, "logits", getattr(em, "log_rate", None))))

# ------------------ Basic Functionality ------------------
def test_basic(em_type="gaussian"):
    print(f"\n=== TEST: Basic Functionality ({em_type}) ===")
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 3
    em = Emission(n_states=3, n_features=n_features, emission_type=em_type)

    base = get_base(em)
    print("Base param shape:", base.shape)

    # Test distribution construction
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

# ------------------ Context Modulation ------------------
def test_context_single(em_type="gaussian"):
    print(f"\n=== TEST: Context Single ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=4, hidden_dim=8)
    ctx = torch.randn(4, requires_grad=True)
    mod = em._modulate(get_base(em), context=ctx)
    print("Context shape:", ctx.shape)
    print("Modulated shape:", mod.shape)

def test_context_batch(em_type="gaussian"):
    print(f"\n=== TEST: Context Batch ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=4, hidden_dim=8)
    ctx = torch.randn(5, 4, requires_grad=True)
    mod = em._modulate(get_base(em), context=ctx)
    print("Batch context shape:", ctx.shape)
    print("Modulated shape:", mod.shape)

def test_context_sequence(em_type="gaussian"):
    print(f"\n=== TEST: Context Sequence ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=3, hidden_dim=6)
    ctx = torch.randn(2, 4, 3, requires_grad=True)  # (B,T,H)
    mod = em._modulate(get_base(em), context=ctx)
    print("Sequence context shape:", ctx.shape)
    print("Modulated shape:", mod.shape)

# ------------------ Sampling ------------------
def test_sampling(em_type="gaussian"):
    print(f"\n=== TEST: Sampling ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    samples = em.sample(context=None)
    print("Samples shape:", samples.shape)
    print("Samples dtype:", samples.dtype)
    if em_type in {"categorical", "bernoulli", "poisson"}:
        print("Samples min/max:", samples.min().item(), samples.max().item())

# ------------------ Expect Probabilities (Discrete) ------------------
def test_expect_probs(em_type="categorical"):
    if em_type not in {"categorical", "bernoulli"}:
        return
    print(f"\n=== TEST: Expect Probabilities ({em_type}) ===")
    em = Emission(n_states=2, n_features=1, emission_type=em_type)
    dist = em._get_dist()
    
    if isinstance(dist, torch.distributions.Independent):
        base_dist = dist.base_dist
    else:
        base_dist = dist

    if em_type == "categorical":
        probs = F.softmax(base_dist.logits, dim=-1)
    elif em_type == "bernoulli":
        probs = base_dist.probs

    print("Expect probs shape:", probs.shape)
    print("Expect probs min/max:", probs.min().item(), probs.max().item())
    print("Sum across categories:", probs.sum(dim=-1))

# ------------------ Log-Probability ------------------
def test_log_prob(em_type="gaussian"):
    print(f"\n=== TEST: Log-Probability ({em_type}) ===")
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 2
    em = Emission(n_states=2, n_features=n_features, emission_type=em_type)

    if em_type in {"categorical", "bernoulli"}:
        x = torch.randint(0, 2, (3, n_features))
    elif em_type == "poisson":
        x = torch.randint(0, 5, (3, n_features))
    else:
        x = torch.randn(3, n_features)

    logp = em.log_prob(x)
    print("Input shape:", x.shape)
    print("Log-probs shape:", logp.shape)
    print("Log-probs min/max:", logp.min().item(), logp.max().item())

# ------------------ Gradient Flow ------------------
def test_gradient_flow(em_type="gaussian"):
    print(f"\n=== TEST: Gradient Flow ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type, context_dim=3, hidden_dim=6)
    ctx = torch.randn(1, 3, requires_grad=True)
    base = get_base(em)
    mod = em._modulate(base, context=ctx)
    out = mod.sum()
    out.backward()
    grad_base_norm = base.grad.norm().item() if base.grad is not None else None
    print("Gradient base norm:", grad_base_norm)
    print("Gradient context:", ctx.grad)

# ------------------ Update ------------------
def test_update(em_type="gaussian"):
    print(f"\n=== TEST: update() ({em_type}) ===")
    n_states = 2
    n_features = 1 if em_type in {"categorical", "bernoulli", "poisson"} else 2
    em = Emission(n_states=n_states, n_features=n_features, emission_type=em_type)

    base = get_base(em).clone()

    # --- Direct base update ---
    new_base = torch.randn_like(base)
    em.update(new_logits=new_base, update_rate=0.5)
    print("Direct update diff norm:", (get_base(em) - base).norm().item())

    # --- Posterior update ---
    base_shape = get_base(em).shape
    if em_type in {"categorical", "bernoulli"}:
        posterior = torch.rand(*base_shape)
        posterior = F.softmax(posterior, dim=-1)
    elif em_type == "poisson":
        posterior = torch.rand(*base_shape).clamp_min(EPS)
    else:  # Gaussian, Laplace, StudentT
        posterior = torch.ones_like(get_base(em)) / get_base(em).numel()

    em.update(posterior=posterior, update_rate=0.5)
    print("Posterior update diff norm:", (get_base(em) - base).norm().item())

# ------------------ Initialize ------------------
def test_initialize(em_type="gaussian"):
    print(f"\n=== TEST: initialize() ({em_type}) ===")
    em = Emission(n_states=2, n_features=2, emission_type=em_type)
    X = torch.randn(10, 2)
    dist = em.initialize(X=X)
    loc = getattr(dist, "mean", getattr(dist, "loc", None))
    print("Initialized loc:", None if loc is None else loc.shape)
    if hasattr(dist, "covariance_matrix"):
        print("Initialized cov:", dist.covariance_matrix.shape)

# ------------------ Main ------------------
if __name__ == "__main__":
    set_seed()
    for em_type in ["gaussian", "laplace", "studentt", "categorical", "bernoulli", "poisson"]:
        test_basic(em_type)
        test_temperature(em_type)
        test_context_single(em_type)
        test_context_batch(em_type)
        test_context_sequence(em_type)
        test_sampling(em_type)
        test_log_prob(em_type)
        test_expect_probs(em_type)  # new test
        test_gradient_flow(em_type)
        test_initialize(em_type)
        test_update(em_type)
    test_context_encoder()
    print("\n✓ All Emission tests completed.")
