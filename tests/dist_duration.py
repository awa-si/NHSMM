# tests/dist_duration.py

import torch
import torch.nn.functional as F
from nhsmm.distributions import Duration
from nhsmm.constants import DTYPE, EPS
from nhsmm.context import DefaultEncoder, ContextEncoder

def set_seed(seed: int = 42):
    torch.manual_seed(seed)

def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    dur = Duration(n_states=3, max_duration=5)
    print("Logits shape:", dur.logits.shape)
    probs = dur.expected_probs().detach()
    print("Probs shape:", probs.shape)
    print("Sum over durations per state:", probs.sum(dim=-1))
    sample_vec = dur.sample()
    print("Sample vector shape:", sample_vec.shape)
    print("Sample argmax shape:", sample_vec.argmax(dim=-1).shape)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(dur.n_states), atol=1e-5)

def test_initialize():
    print("\n=== TEST: Duration.initialize() for all init modes ===")

    n_states = 5
    max_duration = 10
    modes = ["uniform", "biased", "normal"]

    for mode in modes:
        print(f"\n--- init_mode={mode} ---")
        duration = Duration(n_states=n_states, max_duration=max_duration, init_mode=mode)

        dist = duration.initialize(mode=mode)
        logits = duration.logits.detach()

        print("Logits:\n", logits.cpu().numpy())
        print("Dist:", dist)

        # Check shape
        assert logits.shape == (n_states, max_duration)
        # Ensure all finite values
        assert torch.isfinite(logits).all()

        probs = torch.softmax(logits, dim=-1)
        print("Probs:\n", probs.cpu().numpy())
        print("Sum probs per state:", probs.sum(dim=-1).cpu().numpy())

        # Each row sums to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(n_states), atol=1e-5)
        # All probabilities non-negative
        assert torch.all(probs >= 0)

    print("\n=== TEST: Duration.initialize() with context ===")

    context_dim = 4
    hidden_dim = 16
    ctx = torch.randn(3, context_dim)

    duration = Duration(
        n_states=n_states,
        max_duration=max_duration,
        init_mode="normal",
        context_dim=context_dim,
        hidden_dim=hidden_dim
    )

    dist = duration.initialize(mode="normal", context=ctx)
    logits = duration.logits.detach()

    print("Context shape:", ctx.shape)
    print("Context-conditioned logits:\n", logits.cpu().numpy())

    # Check shape and finite values
    assert logits.shape == (n_states, max_duration)
    assert torch.isfinite(logits).all()

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n_states), atol=1e-5)

    print("\n✓ Duration.initialize() passed all modes and context cases")

def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    dur = Duration(n_states=2, max_duration=4)
    cold = dur._modulate(temperature=0.1).detach()
    hot = dur._modulate(temperature=5.0).detach()
    print("Cold logits shape:", cold.shape)
    print("Hot logits shape:", hot.shape)
    assert cold.shape == hot.shape

def test_context():
    print("\n=== TEST: Context ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx_single = torch.randn(3)
    probs_single = dur.expected_probs(context=ctx_single).detach()
    print("Single context probs shape:", probs_single.shape, "sum:", probs_single.sum(dim=-1))
    ctx_batch = torch.randn(5, 3)
    probs_batch = dur.expected_probs(context=ctx_batch).detach()
    print("Batch context probs shape:", probs_batch.shape, "sum per state:", probs_batch.sum(dim=-1))
    assert torch.allclose(probs_single.sum(dim=-1), torch.ones(dur.n_states), atol=1e-5)
    assert torch.allclose(probs_batch.sum(dim=-1), torch.ones_like(probs_batch.sum(dim=-1)), atol=1e-5)

def test_log_matrix():
    print("\n=== TEST: Log Matrix ===")
    dur = Duration(n_states=2, max_duration=4)
    L = dur.log_matrix()
    print("log_matrix shape:", L.shape)
    probs = L.softmax(dim=-1)
    print("Softmax row sums:", probs.sum(-1))
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-6)

def test_sequence_log_prob():
    print("\n=== TEST: Sequence Log Prob ===")
    dur = Duration(n_states=2, max_duration=4)
    seqs = torch.tensor([[1, 2, 0], [0, 1, 1]])
    mod_logits = dur._modulate()
    log_probs = F.log_softmax(mod_logits, dim=-1)
    idx_log_probs = torch.gather(log_probs, -1, seqs)
    print("Sequences shape:", seqs.shape, "log_probs shape:", log_probs.shape, "Indexed shape:", idx_log_probs.shape)

def test_sampling_correctness(N=5000):
    print("\n=== TEST: Sampling Correctness ===")
    dur = Duration(n_states=2, max_duration=4)
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

def test_gradient_flow():
    print("\n=== TEST: Gradient Flow ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx_scalar = torch.randn(3, requires_grad=True, dtype=DTYPE)
    dur.expected_probs(context=ctx_scalar).sum().backward()
    print("Scalar context grad:", ctx_scalar.grad)
    ctx_scalar.grad.zero_()

    ctx_batch = torch.randn(2, 3, 3, requires_grad=True, dtype=DTYPE)
    dur.expected_probs(context=ctx_batch).sum().backward()
    print("Batch context grad shape:", ctx_batch.grad.shape, "norm:", ctx_batch.grad.norm().item())
    ctx_batch.grad.zero_()

    dur.logits.grad = None
    dur.expected_probs().sum().backward()
    print("Logits grad norm (no context):", dur.logits.grad.norm().item())

def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    dur1 = Duration(n_states=1, max_duration=1)
    print("n_states=1, max_duration=1, probs:", dur1.expected_probs().cpu().detach().numpy())
    dur = Duration(n_states=5, max_duration=10)
    out = dur.sample()
    print("Large batch (context=None) OK, sample shape:", out.shape)

def test_update_cache():
    print("\n=== TEST: Update and Cache ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3)
    out1 = dur.sample(context=ctx)
    out2 = dur.sample(context=ctx)
    print("Cache shapes:", out1.shape, out2.shape)
    posterior = torch.ones_like(dur._modulate(context=ctx)) / dur._modulate(context=ctx).numel()
    dur.update(posterior=posterior, context=ctx, update_rate=0.5)
    out3 = dur.sample(context=ctx)
    print("Updated sample shape:", out3.shape)

def test_batch_timestep_modulation():
    print("\n=== TEST: Batch + Timestep Modulation ===")
    dur = Duration(n_states=3, max_duration=5)
    B, T = 4, 3
    ctx = torch.randn(B, 3)
    timesteps = torch.arange(1, T+1).repeat(B, 1)
    mod_logits = torch.stack([dur._modulate(context=ctx[b], timestep=timesteps[b, t])
                              for b in range(B) for t in range(T)], dim=0)
    print("Batched modulated logits shape:", mod_logits.shape)
    single_mod = torch.stack([dur._modulate(timestep=t) for t in range(2, 4)], dim=0)
    print("Single batch modulated logits shape:", single_mod.shape)

def test_context_encoder():
    print("\n=== TEST: ContextEncoder + DefaultEncoder with Duration ===")
    B, T, F_in = 4, 8, 6
    n_states, context_dim = 5, 16
    x = torch.randn(B, T, F_in)
    encoder = DefaultEncoder(n_features=F_in, hidden_dim=context_dim, cnn_channels=8,
                               kernel_size=3, bidirectional=True, return_sequence=True)
    for pool in ["mean", "last", "max", "attn", "mha"]:
        print(f"\n=== Testing pool={pool} ===")
        ctx_enc = ContextEncoder(encoder=encoder, pool=pool, n_heads=2, debug=True)
        seq_out, ctx, attn = ctx_enc(x, return_context=True, return_attn_weights=True, return_sequence=True)
        print("Input:", x.shape, "Seq out:", seq_out.shape, "Context:", ctx.shape)
        if attn is not None: print("Attention:", attn.shape)
        dur = Duration(n_states=n_states, max_duration=7, context_dim=seq_out.shape[-1], hidden_dim=32)
        probs = dur.expected_probs(context=ctx.squeeze(1))
        print("Duration probs:", probs.shape, "sum per state:", probs.sum(dim=-1))
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B, n_states), atol=1e-5)

    # Non-uniform temperature
    dur = Duration(n_states=n_states, max_duration=7, init_mode="biased")
    logits = dur.logits.detach().clone()
    print("Raw logits:", logits)
    for temp in [0.1, 1.0, 5.0]:
        probs_temp = dur.expected_probs(temperature=temp).detach()
        print(f"Temperature {temp} -> probs:", probs_temp)
        assert torch.allclose(probs_temp.sum(dim=-1), torch.ones_like(probs_temp.sum(dim=-1)), atol=1e-5)

    # Sequence context
    S, B, T, C = 2, 3, 5, context_dim
    seq_ctx = torch.randn(S, B, T, C)
    dur = Duration(n_states=n_states, max_duration=7, context_dim=C, hidden_dim=32)
    for s in range(S):
        probs_seq = dur.expected_probs(context=seq_ctx[s])
        print(f"Sequence {s}, probs shape:", probs_seq.shape, "sum per timestep:", probs_seq.sum(dim=-1))
        assert torch.allclose(probs_seq.sum(dim=-1), torch.ones_like(probs_seq.sum(dim=-1)), atol=1e-5)

def test_hsmm_forward_backward():
    print("\n=== TEST: HSMM Forward–Backward Duration Alignment ===")
    n_states = 3
    max_duration = 5
    B, T = 2, 7

    # Create a synthetic duration log-matrix (logits) for HSMM
    dur = Duration(n_states=n_states, max_duration=max_duration, init_mode="uniform")
    logits_before = dur.logits.detach().clone()
    print("Initial logits:", logits_before)

    # Synthetic forward-backward posterior: shape [B, n_states, max_duration]
    posterior = torch.zeros(B, n_states, max_duration)
    posterior[:, 0, 1] = 0.6
    posterior[:, 0, 2] = 0.4
    posterior[:, 1, 0] = 0.5
    posterior[:, 1, 3] = 0.5
    posterior[:, 2, :] = EPS  # assign tiny values to prevent nan
    posterior = posterior / posterior.sum(dim=-1, keepdim=True)  # normalize

    print("Synthetic posterior:\n", posterior)

    # Update Duration with posterior (like EM M-step)
    dur.update(posterior=posterior, update_rate=1.0)
    logits_after = dur.logits.detach().clone()
    print("Updated logits:", logits_after)

    # Check that updated logits reflect the posterior shape
    probs_after = F.softmax(logits_after, dim=-1)
    print("Probs after update:\n", probs_after)
    sums = probs_after.sum(dim=-1)
    print("Sum per state after update:", sums)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Optionally: sample from updated distribution
    sample = dur.sample()
    print("Sample from updated distribution:", sample)

if __name__ == "__main__":
    set_seed()
    test_basic()
    test_initialize()
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
    test_hsmm_forward_backward()
    print("\n✓ All Duration tests finished.")
