# tests/dist_transition.py

import torch
import torch.nn.functional as F
from nhsmm.distributions import Transition
from nhsmm.constants import DTYPE, EPS
from nhsmm.context import DefaultEncoder, ContextEncoder

def set_seed(seed: int = 42):
    torch.manual_seed(seed)

def test_context_encoder():
    print("\n=== TEST: ContextEncoder + DefaultEncoder with Transition ===")
    B, T, F_in = 4, 8, 6
    n_states = 5
    context_dim = 16

    x = torch.randn(B, T, F_in)

    # --- CNN_LSTM Encoder ---
    encoder = DefaultEncoder(
        n_features=F_in,
        hidden_dim=context_dim,
        cnn_channels=8,
        kernel_size=3,
        bidirectional=True,
        return_sequence=True
    )

    # Test each pooling method
    for pool in ["mean", "last", "max", "attn", "mha"]:
        print(f"\n=== Testing pool={pool} ===")
        ctx_enc = ContextEncoder(encoder=encoder, pool=pool, n_heads=2, debug=True)
        seq_out, ctx, attn = ctx_enc(x, return_context=True, return_attn_weights=True, return_sequence=True)
        print("Input shape:", x.shape)
        print("Sequence output shape:", seq_out.shape)
        print("Context shape:", ctx.shape)
        if attn is not None:
            print("Attention weights shape:", attn.shape)

        # Pass context to Transition
        tr = Transition(n_states=n_states, n_features=n_states, context_dim=seq_out.shape[-1], hidden_dim=32)
        probs = tr.expected_probs(context=ctx.squeeze(1))
        print("Transition probs shape:", probs.shape)
        print("Sum over rows per batch:", probs.sum(dim=-1))

        # Verify row sums = 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B, n_states), atol=1e-5)

def test_context_encoder_sequence():
    print("\n=== TEST: Sequence Context + ContextEncoder + DefaultEncoder with Transition ===")
    S, B, T, F_in = 2, 3, 5, 6
    n_states = 4
    context_dim = 16

    x_seq = torch.randn(S, B, T, F_in)

    # --- CNN_LSTM Encoder ---
    encoder = DefaultEncoder(
        n_features=F_in,
        hidden_dim=context_dim,
        cnn_channels=8,
        kernel_size=3,
        bidirectional=True,
        return_sequence=True
    )

    for pool in ["mean", "last", "max", "attn", "mha"]:
        print(f"\n=== Testing pool={pool} ===")
        ctx_enc = ContextEncoder(encoder=encoder, pool=pool, n_heads=2, debug=True)

        for s in range(S):
            seq_out, ctx, attn = ctx_enc(x_seq[s], return_context=True, return_attn_weights=True, return_sequence=True)
            print(f"Sequence {s} - Input shape:", x_seq[s].shape)
            print("Sequence output shape:", seq_out.shape)
            print("Context shape:", ctx.shape)
            if attn is not None:
                print("Attention weights shape:", attn.shape)

            # Pass context to Transition
            tr = Transition(n_states=n_states, n_features=seq_out.shape[-1], context_dim=seq_out.shape[-1], hidden_dim=32)
            probs = tr.expected_probs(context=ctx.squeeze(1))
            print("Transition probs shape:", probs.shape)
            sums = probs.sum(dim=-1)
            print("Sum over rows per batch:", sums)
            # Verify each row sums to 1
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    tr = Transition(n_states=3, n_features=3)
    print("Logits shape:", tr.logits.shape)

    probs = tr.expected_probs().detach()
    print("Probs shape:", probs.shape)
    print("Rows sum to 1:", probs.sum(dim=-1))
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-5)

    sample_vec = tr.sample()
    print("Sample vector shape:", sample_vec.shape)
    print("Sample argmax per row:", sample_vec.argmax(dim=-1))

def test_initialize():
    print("\n=== TEST: Transition.initialize() for all init modes ===")

    n_states = 5
    modes = ["uniform", "biased", "normal"]

    for mode in modes:
        print(f"\n--- init_mode={mode} ---")
        transition = Transition(n_states=n_states, n_features=n_states, init_mode=mode)

        dist = transition.initialize(mode=mode)
        logits = transition.logits.detach()

        print("Logits:\n", logits.cpu().numpy())
        print("Dist:", dist)

        assert logits.shape == (n_states, n_states)
        assert torch.isfinite(logits).all()

        probs = torch.softmax(logits, dim=-1)
        print("Probs:\n", probs.cpu().numpy())
        print("Row sums:", probs.sum(dim=-1).cpu().numpy())

        assert torch.allclose(probs.sum(dim=-1), torch.ones(n_states), atol=1e-5)
        assert torch.all(probs >= 0)

    print("\n=== TEST: initialize() with context ===")

    context_dim = 4
    hidden_dim = 16
    batch_size = 3
    ctx = torch.randn(batch_size, context_dim)

    transition = Transition(
        n_states=n_states,
        n_features=n_states,
        init_mode="normal",
        context_dim=context_dim,
        hidden_dim=hidden_dim
    )

    # Context-conditioned initialization
    dist = transition.initialize(mode="normal", context=ctx)
    logits = transition.logits.detach()

    print("Context shape:", ctx.shape)
    print("Context-conditioned logits:\n", logits.cpu().numpy())

    assert logits.shape == (n_states, n_states)
    assert torch.isfinite(logits).all()

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n_states), atol=1e-5)

    print("\n✓ Transition.initialize() passed all modes and context cases")

def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    tr = Transition(n_states=3, n_features=3)
    cold = tr._modulate(temperature=0.1).detach()
    hot = tr._modulate(temperature=5.0).detach()
    print("Cold logits shape:", cold.shape)
    print("Hot logits shape:", hot.shape)

    # Check only last two dims against n_states
    assert cold.shape[-2:] == (tr.n_states, tr.n_states)
    assert hot.shape[-2:] == (tr.n_states, tr.n_states)

def test_context_single():
    print("\n=== TEST: Context Single ===")
    tr = Transition(n_states=2, n_features=2, context_dim=4, hidden_dim=8)
    ctx = torch.randn(4)
    probs = tr.expected_probs(context=ctx).detach()
    print("Single context probs shape:", probs.shape)
    print("Row sums:", probs.sum(dim=-1))
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-5)

def test_context_batch():
    print("\n=== TEST: Context Batch ===")
    tr = Transition(n_states=2, n_features=2, context_dim=4, hidden_dim=8)
    ctx = torch.randn(5, 4)
    probs = tr.expected_probs(context=ctx).detach()
    print("Batch context probs shape:", probs.shape)
    print("Row sums per batch:", probs.sum(dim=-1))
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-5)

def test_timestep():
    print("\n=== TEST: Timestep Handling ===")
    tr = Transition(n_states=2, n_features=2)
    mod = tr._modulate(timestep=3)
    print("mod(t=3) shape:", mod.shape)
    assert mod.shape[-2:] == (tr.n_states, tr.n_states)

def test_constraints():
    print("\n=== TEST: Constraint Types ===")
    tr_ltr = Transition(n_states=3, n_features=3, transition_type="left-to-right")
    print("LTR mask logits:\n", tr_ltr.log_matrix().detach())

    tr_semi = Transition(n_states=3, n_features=3, transition_type="semi")
    print("Semi-Markov mask logits:\n", tr_semi.log_matrix().detach())

def test_sampling_correctness(N: int = 5000):
    print("\n=== TEST: Sampling Correctness ===")
    tr = Transition(n_states=2, n_features=2, init_mode="uniform")
    probs = tr.expected_probs().detach()  # (1,1,S,S)
    counts = torch.zeros_like(probs)

    for _ in range(N):
        s = tr.sample()  # one-hot
        counts += s.float()

    empirical = counts / counts.sum(dim=-1, keepdim=True)
    print("Empirical freq:\n", empirical)
    print("Target probs:\n", probs)
    print("Difference:\n", empirical - probs)
    assert torch.allclose(empirical, probs, atol=0.05)

def test_gradient_flow():
    print("\n=== TEST: Gradient Flow ===")
    tr = Transition(n_states=2, n_features=2, context_dim=3, hidden_dim=6)

    ctx_scalar = torch.randn(3, requires_grad=True, dtype=DTYPE)
    tr.expected_probs(context=ctx_scalar).sum().backward()
    print("Scalar context grad:", ctx_scalar.grad)
    ctx_scalar.grad.zero_()

    ctx_batch = torch.randn(2, 3, requires_grad=True, dtype=DTYPE)
    tr.expected_probs(context=ctx_batch).sum().backward()
    print("Batch context grad shape:", ctx_batch.grad.shape)
    print("Batch context grad norm:", ctx_batch.grad.norm().item())
    ctx_batch.grad.zero_()

    tr.logits.grad = None
    tr.expected_probs().sum().backward()
    print("Logits grad norm (no context):", tr.logits.grad.norm().item())

def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    tr1 = Transition(n_states=1, n_features=1)
    print("n_states=1, n_features=1, probs:", tr1.expected_probs().cpu().detach().numpy())

    tr_large = Transition(n_states=5, n_features=5)
    out = tr_large.sample()
    print("Large batch sample shape:", out.shape)

def test_update_cache():
    print("\n=== TEST: Update and Cache ===")
    tr = Transition(n_states=2, n_features=2, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3)

    out1 = tr.sample(context=ctx)
    out2 = tr.sample(context=ctx)
    print("Cache sample shapes:", out1.shape, out2.shape)

    posterior = torch.ones_like(tr._modulate(context=ctx)) / tr._modulate(context=ctx).numel()
    tr.update(new_logits=posterior, context=ctx, update_rate=0.5)
    out3 = tr.sample(context=ctx)
    print("Updated sample shape:", out3.shape)

def test_batch_timestep_modulation():
    print("\n=== TEST: Batch + Timestep Modulation ===")
    tr = Transition(n_states=3, n_features=3)
    B, T = 4, 3
    ctx = torch.randn(B, 3)
    timesteps = torch.arange(1, T + 1).repeat(B, 1)

    mod_logits = torch.stack([tr._modulate(context=ctx[b], timestep=timesteps[b, t])
                              for b in range(B) for t in range(T)], dim=0)
    print("Batched modulated logits shape:", mod_logits.shape)
    assert mod_logits.shape[-2:] == (tr.n_states, tr.n_states)

    single_mod = torch.stack([tr._modulate(timestep=t) for t in range(2, 4)], dim=0)
    print("Single batch modulated logits shape:", single_mod.shape)

def test_log_matrix():
    print("\n=== TEST: Transition Log Matrix ===")

    tr = Transition(n_states=3, n_features=3)

    # Raw logits
    L = tr.log_matrix()
    print("log_matrix (logits) shape:", L.shape)
    print(L)

    # Convert logits → probabilities
    probs = L.softmax(dim=-1)

    print("Softmax probs:", probs)
    print("Row sums:", probs.sum(-1))

    # Each row of transition matrix must sum to 1
    assert torch.allclose(
        probs.sum(-1),
        torch.ones_like(probs[..., 0]),
        atol=1e-6
    )


if __name__ == "__main__":
    set_seed()
    test_basic()
    test_initialize()
    test_temperature()
    test_context_single()
    test_context_batch()
    test_timestep()
    test_constraints()
    test_log_matrix()
    test_sampling_correctness()
    test_gradient_flow()
    test_edge_cases()
    test_update_cache()
    test_batch_timestep_modulation()
    test_context_encoder()
    test_context_encoder_sequence()
    print("\n✓ All Transition tests completed.")
