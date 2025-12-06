# tests/dist_duration.py

import torch
import torch.nn.functional as F
from nhsmm.distributions import Duration
from nhsmm.constants import DTYPE

def set_seed(seed: int = 42):
    torch.manual_seed(seed)

# ---------------- Basic Functionality ----------------
def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    dur = Duration(n_states=3, max_duration=5, init_mode="uniform")
    print("Logits shape:", dur.logits.shape)

    probs = dur.expected_probs().detach()
    print("Probs shape:", probs.shape)
    print("Sum over durations per state:", probs.sum(dim=-1))

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

    # Batch context
    ctx_batch = torch.randn(5, 3)
    probs_batch = dur.expected_probs(context=ctx_batch).detach()
    print("Batch context probs shape:", probs_batch.shape)
    print("Sum over durations per state:", probs_batch.sum(dim=-1))

# ---------------- Log Matrix ----------------
def test_log_matrix():
    print("\n=== TEST: Log Matrix ===")
    dur = Duration(n_states=2, max_duration=4)
    L = dur.log_matrix()
    print("log_matrix shape:", L.shape)
    print("exp(log_matrix):", L.exp().detach().cpu().numpy())

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

    # Sequence batch gradient
    seq_ctx = torch.randn(2, 3, 3, requires_grad=True, dtype=DTYPE)
    dur.expected_probs(context=seq_ctx).sum().backward()
    print("Sequence batch context grad norm:", seq_ctx.grad.norm().item())
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

    # Modulate each timestep without flattening
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
    print("\n✓ All Duration tests finished.")
