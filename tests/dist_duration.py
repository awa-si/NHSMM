# tests/dist_duration.py

import torch
import torch.nn.functional as F
from nhsmm.distributions import Duration
from nhsmm.constants import DTYPE, EPS
from nhsmm.context import CNN_LSTM_Encoder, ContextEncoder

def set_seed(seed: int = 42):
    torch.manual_seed(seed)

# ---------------- ContextEncoder + CNN_LSTM_Encoder Integration ----------------
def test_context_encoder():
    print("\n=== TEST: ContextEncoder + CNN_LSTM_Encoder with Duration ===")
    B, T, F_in = 4, 8, 6
    n_states = 5
    context_dim = 16

    x = torch.randn(B, T, F_in)

    # --- CNN_LSTM Encoder ---
    encoder = CNN_LSTM_Encoder(
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

        # Pass context to Duration
        dur = Duration(n_states=n_states, max_duration=7, context_dim=seq_out.shape[-1], hidden_dim=32)
        probs = dur.expected_probs(context=ctx.squeeze(1))
        print("Duration probs shape:", probs.shape)
        print("Sum of probs (per state):", probs.sum(dim=-1))

        # Verify probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B, n_states), atol=1e-5)

    # --- Non-uniform temperature scaling ---
    print("\n=== Testing Non-uniform Temperature ===")
    dur = Duration(n_states=n_states, max_duration=7, init_mode="short_bias")
    logits = dur.logits.detach().clone()
    print("Raw logits:", logits)
    for temp in [0.1, 1.0, 5.0]:
        probs_temp = dur.expected_probs(temperature=temp).detach()
        print(f"Temperature {temp} -> probs:", probs_temp)
        assert torch.allclose(probs_temp.sum(dim=-1), torch.ones_like(probs_temp.sum(dim=-1)), atol=1e-5)

    # --- Test multi-step sequence contexts ---
    print("\n=== Testing Sequence Context ===")
    S, B, T, C = 2, 3, 5, context_dim
    seq_ctx = torch.randn(S, B, T, C)
    dur = Duration(n_states=n_states, max_duration=7, context_dim=C, hidden_dim=32)
    for s in range(S):
        probs_seq = dur.expected_probs(context=seq_ctx[s])
        sums = probs_seq.sum(dim=-1)
        print(f"Sequence {s}, probs shape: {probs_seq.shape}, sum per timestep:", sums)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

# ---------------- Basic Functionality ----------------
def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    dur = Duration(n_states=3, max_duration=5, init_mode="uniform")
    print("Logits shape:", dur.logits.shape)

    probs = dur.expected_probs().detach()
    print("Probs shape:", probs.shape)
    print("Sum over durations per state:", probs.sum(dim=-1))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(dur.n_states), atol=1e-5)

    sample_vec = dur.sample()
    print("Sample vector shape:", sample_vec.shape)
    print("Sample index (argmax) shape:", sample_vec.argmax(dim=-1).shape)

# ---------------- Temperature Scaling ----------------
def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    dur = Duration(n_states=2, max_duration=4, init_mode="uniform")
    cold = dur._modulate(temperature=0.1).detach()
    hot = dur._modulate(temperature=5.0).detach()
    print("Cold logits shape:", cold.shape)
    print("Hot logits shape:", hot.shape)

# ---------------- Context Tests ----------------
def test_context():
    print("\n=== TEST: Context ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    
    # Single vector
    ctx_single = torch.randn(3)
    probs_single = dur.expected_probs(context=ctx_single).detach()
    print("Single context probs shape:", probs_single.shape)
    print("Sum over durations per state:", probs_single.sum(dim=-1))
    assert torch.allclose(probs_single.sum(dim=-1), torch.ones(dur.n_states), atol=1e-5)

    # Batch context
    ctx_batch = torch.randn(5, 3)
    probs_batch = dur.expected_probs(context=ctx_batch).detach()
    print("Batch context probs shape:", probs_batch.shape)
    # Sum over durations (last dim)
    sums = probs_batch.sum(dim=-1)  # shape: [5, 1, n_states]
    print("Sum over durations per state:", sums)
    # Check sums = 1
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

# ---------------- Log Matrix ----------------
def test_log_matrix():
    print("\n=== TEST: Log Matrix ===")

    dur = Duration(n_states=2, max_duration=4)

    # Raw logits
    L = dur.log_matrix()
    print("log_matrix (logits) shape:", L.shape)
    print(L)

    # Softmax should produce a valid distribution per state
    probs = L.softmax(dim=-1)

    print("Softmax probs:", probs)
    print("Row sums:", probs.sum(-1))

    # Validate rows sum to 1
    assert torch.allclose(
        probs.sum(-1),
        torch.ones_like(probs[..., 0]),
        atol=1e-6
    )

# ---------------- Sequence Log Prob ----------------
def test_sequence_log_prob():
    print("\n=== TEST: Sequence Log Prob ===")
    dur = Duration(n_states=2, max_duration=4)
    seqs = torch.tensor([[1, 2, 0], [0, 1, 1]])
    mod_logits = dur._modulate()
    log_probs = F.log_softmax(mod_logits, dim=-1)
    # Gather per sequence
    idx_log_probs = torch.gather(log_probs, -1, seqs)
    print("Sequences shape:", seqs.shape)
    print("log_probs shape:", log_probs.shape)
    print("Indexed log_probs shape:", idx_log_probs.shape)

# ---------------- Sampling Correctness ----------------
def test_sampling_correctness(N: int = 5000):
    print("\n=== TEST: Sampling Correctness ===")
    dur = Duration(n_states=2, max_duration=4, init_mode="uniform")
    probs = dur.expected_probs().squeeze(0).squeeze(0)
    counts = torch.zeros_like(probs[0])
    for _ in range(N):
        s = dur.sample()
        s_idx = s.argmax(dim=-1) if s.ndim > 1 else s
        counts += torch.bincount(s_idx, minlength=probs.shape[-1]).float()
    empirical = counts / counts.sum()
    print("Empirical freq:", empirical)
    print("Difference:", empirical - probs[0])
    assert torch.allclose(empirical, probs[0], atol=0.05)

# ---------------- Gradient Flow ----------------
def test_gradient_flow():
    print("\n=== TEST: Gradient Flow ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)

    # Scalar context
    ctx_scalar = torch.randn(3, requires_grad=True, dtype=DTYPE)
    dur.expected_probs(context=ctx_scalar).sum().backward()
    print("Scalar context grad:", ctx_scalar.grad)
    ctx_scalar.grad.zero_()

    # Batch context (B, T, H)
    ctx_batch = torch.randn(2, 3, 3, requires_grad=True, dtype=DTYPE)
    dur.expected_probs(context=ctx_batch).sum().backward()
    print("Batch context grad shape:", ctx_batch.grad.shape)
    print("Batch context grad norm:", ctx_batch.grad.norm().item())
    ctx_batch.grad.zero_()

    # Logits gradient (no context)
    dur.logits.grad = None
    dur.expected_probs().sum().backward()
    print("Logits grad norm (no context):", dur.logits.grad.norm().item())

# ---------------- Edge Cases ----------------
def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    dur1 = Duration(n_states=1, max_duration=1)
    print("n_states=1, max_duration=1, probs:", dur1.expected_probs().cpu().detach().numpy())

    dur = Duration(n_states=5, max_duration=10)
    out = dur.sample()
    print(f"Large batch (context=None) OK, sample shape: {out.shape}")

# ---------------- Update and Cache ----------------
def test_update_cache():
    print("\n=== TEST: Update and Cache ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3)

    out1 = dur.sample(context=ctx)
    out2 = dur.sample(context=ctx)
    print("Cache check shapes:", out1.shape, out2.shape)

    # EM-style pseudo update
    posterior = torch.ones_like(dur._modulate(context=ctx)) / dur._modulate(context=ctx).numel()
    dur.update(posterior=posterior, context=ctx, update_rate=0.5)
    out3 = dur.sample(context=ctx)
    print("Updated sample shape:", out3.shape)

# ---------------- Timestep & Batch Modulation ----------------
def test_batch_timestep_modulation():
    print("\n=== TEST: Batch + Timestep Modulation ===")
    dur = Duration(n_states=3, max_duration=5)
    B, T = 4, 3
    ctx = torch.randn(B, 3)
    timesteps = torch.arange(1, T+1).repeat(B, 1)  # B x T

    mod_logits = torch.stack([dur._modulate(context=ctx[b], timestep=timesteps[b, t])
                              for b in range(B) for t in range(T)], dim=0)
    print("Batched modulated logits shape:", mod_logits.shape)

    # Single batch with timestep > 1
    single_mod = torch.stack([dur._modulate(timestep=t) for t in range(2, 4)], dim=0)
    print("Single batch modulated logits shape:", single_mod.shape)

# ---------------- Main ----------------
if __name__ == "__main__":
    set_seed()
    test_basic()
    test_temperature()
    test_context()
    test_log_matrix()
    test_sequence_log_prob()
    test_sampling_correctness()
    test_gradient_flow()
    test_edge_cases()
    test_update_cache()
    test_batch_timestep_modulation()
    test_context_encoder()
    print("\n✓ All Duration tests finished.")
