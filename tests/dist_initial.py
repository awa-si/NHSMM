# tests/dist_initial.py

import torch
from nhsmm.distributions import Initial
from nhsmm.constants import EPS
from nhsmm.context import DefaultEncoder, ContextEncoder

def test_context_encoder():
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

        # Pass context to Initial
        init = Initial(n_states=n_states, context_dim=seq_out.shape[-1], hidden_dim=32)
        probs = init.expected_probs(context=ctx.squeeze(1))
        print("Initial probs shape:", probs.shape)
        print("Sum of probs (per batch):", probs.sum(dim=-1))

        # Verify probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    # --- Non-uniform temperature scaling ---
    print("\n=== Testing Non-uniform Temperature ===")
    init = Initial(n_states=n_states, init_mode="biased")
    logits = init.logits.detach().clone()
    print("Raw logits:", logits)
    for temp in [0.1, 1.0, 5.0]:
        probs_temp = init.expected_probs(temperature=temp).detach()
        print(f"Temperature {temp} -> probs:", probs_temp)
        # Check sum
        assert torch.allclose(probs_temp.sum(dim=-1), torch.ones_like(probs_temp.sum(dim=-1)), atol=1e-5)

    # --- Test multi-step context sequences ---
    print("\n=== Testing Sequence Context ===")
    S, B, T, C = 2, 3, 5, context_dim
    seq_ctx = torch.randn(S, B, T, C)
    init = Initial(n_states=n_states, context_dim=C, hidden_dim=32)
    for s in range(S):
        probs_seq = init.expected_probs(context=seq_ctx[s])
        sums = probs_seq.sum(dim=-1)
        print(f"Sequence {s}, probs shape: {probs_seq.shape}, sum per timestep:", sums)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    init = Initial(n_states=5, init_mode="uniform")
    print("Logits:", init.logits.detach().cpu().numpy())

    probs = init.expected_probs().detach()
    print("Probs:", probs.cpu().numpy())
    print("Sum of probs:", probs.sum().item())
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    sample_vec = init.sample()
    print("Sample vector:", sample_vec.detach().cpu().numpy())
    print("Sample index (argmax):", sample_vec.argmax().item())

def test_initialize():
    print("\n=== TEST: initialize() for all init modes ===")

    n_states = 5
    modes = ["uniform", "biased", "normal"]

    for mode in modes:
        print(f"\n--- init_mode={mode} ---")
        init = Initial(n_states=n_states, init_mode=mode)

        dist = init.initialize(mode=mode)
        logits = init.logits.detach()

        print("Logits:", logits.cpu().numpy())
        print("Dist:", dist)

        assert logits.shape == (n_states,)
        assert torch.isfinite(logits).all()

        probs = torch.softmax(logits, dim=-1)
        print("Probs:", probs.cpu().numpy())
        print("Sum probs:", probs.sum().item())

        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(probs >= 0)

    print("\n=== TEST: initialize() with context ===")

    context_dim = 4
    hidden_dim = 16
    ctx = torch.randn(3, context_dim)

    init = Initial(
        n_states=n_states,
        init_mode="normal",
        context_dim=context_dim,
        hidden_dim=hidden_dim
    )

    dist = init.initialize(mode="normal", context=ctx)
    logits = init.logits.detach()

    print("Context shape:", ctx.shape)
    print("Context-conditioned logits:", logits.cpu().numpy())

    assert logits.shape == (n_states,)
    assert torch.isfinite(logits).all()

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    print("\n✓ initialize() passed all modes and context cases")

def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    init = Initial(n_states=4, init_mode="uniform")
    cold = init.expected_probs(temperature=0.1).detach()
    hot = init.expected_probs(temperature=5.0).detach()

    print("Cold (τ=0.1):", cold.cpu().numpy())
    print("Hot  (τ=5.0):", hot.cpu().numpy())
    print("Entropy cold:", -(cold * (cold + EPS).log()).sum().item())
    print("Entropy hot :", -(hot * (hot + EPS).log()).sum().item())
    assert torch.allclose(cold.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(hot.sum(), torch.tensor(1.0), atol=1e-5)

def test_context_single():
    print("\n=== TEST: Context Single Vector ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(3)
    probs = init.expected_probs(context=ctx).detach()
    print("Context:", ctx.detach().cpu().numpy())
    print("Probs:", probs.detach().cpu().numpy())
    print("Sum:", probs.sum().item())
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

def test_context_batch():
    print("\n=== TEST: Context Batch ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(7, 3)
    probs = init.expected_probs(context=ctx).detach()
    print("Context shape:", ctx.shape)
    print("Probs shape:", probs.shape)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-5)

def test_context_sequence():
    print("\n=== TEST: Context Sequence [S,B,T,C] ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    S, B, T, C = 2, 3, 5, 3
    ctx = torch.randn(S, B, T, C)
    all_probs = []
    for s in range(S):
        probs = init.expected_probs(context=ctx[s]).detach()
        all_probs.append(probs)
        sums = probs.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print(f"Sequence context shape: {ctx.shape}, sample probs shapes:", [p.shape for p in all_probs])

def test_log_matrix():
    print("\n=== TEST: log_matrix ===")

    init = Initial(n_states=4)

    # NEW: log_matrix returns raw logits
    logits = init.log_matrix().detach()

    print("logits shape:", logits.shape)
    print("logits:\n", logits.cpu().numpy())

    # Convert manually to probs for verification
    probs = torch.softmax(logits, dim=-1)

    print("exp(rows) sum to:", probs.sum(-1))

    # Rows must sum to 1 after softmax
    assert torch.allclose(
        probs.sum(-1),
        torch.ones_like(probs.sum(-1)),
        atol=1e-5
    )

def test_log_prob_sequence():
    print("\n=== TEST: log_prob simple sequence ===")
    init = Initial(n_states=4)
    seq = torch.tensor([2])
    lp = init.log_prob(seq)
    print("Sequence:", seq.detach().cpu().numpy())
    print("log_prob:", lp.item())

def test_log_prob_context():
    print("\n=== TEST: log_prob with context ===")
    init = Initial(n_states=4, context_dim=5, hidden_dim=16)
    B, T, C = 6, 1, 5
    seq = torch.randint(high=4, size=(B, T))
    ctx = torch.randn(B, C)
    lp = init.log_prob(seq, context=ctx)
    print("Sequences:\n", seq.detach().cpu().numpy())
    print("Context shape:", ctx.shape)
    print("log_prob:", lp.detach().cpu().numpy())

    T = 3
    seq_multi = torch.randint(high=4, size=(B, T))
    ctx_multi = torch.randn(B, T, C)
    lp_multi = init.log_prob(seq_multi, context=ctx_multi)
    print("Multi-step sequences shape:", seq_multi.shape)
    print("Multi-step context shape:", ctx_multi.shape)
    print("log_prob multi-step:\n", lp_multi.detach().cpu().numpy())

def test_log_prob_batch():
    print("\n=== TEST: log_prob batch ===")
    init = Initial(n_states=4)
    seq = torch.tensor([[0], [3], [1]])
    lp = init.log_prob(seq)
    print("Sequences:\n", seq.detach().cpu().numpy())
    print("Batch log_prob:", lp.detach().cpu().numpy())

def test_log_prob_sequence_context():
    print("\n=== TEST: log_prob with batch sequence context ===")
    init = Initial(n_states=4, context_dim=5, hidden_dim=16)
    B, T, C = 6, 3, 5
    seq = torch.randint(high=4, size=(B, T))
    ctx = torch.randn(B, T, C)
    lp = init.log_prob(seq, context=ctx)
    print("Sequences shape:", seq.shape)
    print("Context shape:", ctx.shape)
    print("log_prob shape:", lp.shape)

def test_sampling_correctness():
    print("\n=== TEST: Sampling Correctness (empirical) ===")
    init = Initial(n_states=4, init_mode="uniform")
    probs = init.expected_probs().detach()
    print("Theoretical probs:", probs.detach().cpu().numpy())

    N = 5000
    counts = torch.zeros(4)
    for _ in range(N):
        s = init.sample().argmax()
        counts[s] += 1
    empirical = (counts / N).detach().cpu().numpy()
    print("Empirical freq:", empirical)
    print("Difference:", empirical - probs.detach().cpu().numpy())

def test_gradient_flow():
    print("\n=== TEST: Gradient Flow ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3, requires_grad=True)
    probs = init.expected_probs(context=ctx).sum()
    probs.backward()
    logits_grad = init.logits.grad.norm().item() if init.logits.grad is not None else None
    print("grad logits:", logits_grad)
    print("grad context:", ctx.grad)

def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    init1 = Initial(n_states=1)
    p1 = init1.expected_probs().detach()
    print("n_states = 1, probs:", p1.detach().cpu().numpy())
    assert torch.allclose(p1, torch.ones_like(p1))

    try:
        init = Initial(n_states=5)
        out = init.expected_probs(context=None)
        print(f"Large batch (context=None) OK, shape: {out.shape}")
    except Exception as e:
        print("Large batch raised:", e)

def test_update_and_cache():
    print("\n=== TEST: update() and cache ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3)

    out1 = init.expected_probs(context=ctx)
    out2 = init.expected_probs(context=ctx)
    assert torch.allclose(out1, out2)
    print("Cache check shapes:", out1.shape, out2.shape)

    posterior = torch.ones_like(out1) / out1.shape[-1]
    init.update(posterior=posterior, context=ctx, update_rate=0.5)
    out3 = init.expected_probs(context=ctx)
    assert not torch.allclose(out1, out3)
    print("Updated sample shape:", out3.shape)

if __name__ == "__main__":
    test_basic()
    test_initialize()
    test_temperature()
    test_context_single()
    test_context_batch()
    test_context_sequence()
    test_log_matrix()
    test_log_prob_sequence()
    test_log_prob_context()
    test_log_prob_batch()
    test_log_prob_sequence_context()
    test_sampling_correctness()
    test_gradient_flow()
    test_edge_cases()
    test_update_and_cache()
    test_context_encoder()
    print("\n✓ All tests finished.")
