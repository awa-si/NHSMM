# tests/dist_duration.py

import torch
from nhsmm.distributions import Duration
import torch.nn.functional as F

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

def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    dur = Duration(n_states=2, max_duration=4, init_mode="uniform")
    cold = dur._modulate(temperature=0.1).detach()
    hot = dur._modulate(temperature=5.0).detach()
    print("Cold logits shape:", cold.shape)
    print("Hot logits shape:", hot.shape)

def test_context_single():
    print("\n=== TEST: Context Single Vector ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(3)
    probs = dur.expected_probs(context=ctx).detach()
    print("Context:", ctx.detach().cpu().numpy())
    print("Probs shape:", probs.shape)
    print("Sum over durations per state:", probs.sum(dim=-1))

def test_context_batch():
    print("\n=== TEST: Context Batch ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(5, 3)
    probs = dur.expected_probs(context=ctx).detach()
    print("Context shape:", ctx.shape)
    print("Probs shape:", probs.shape)
    print("Sum over durations per state:", probs.sum(dim=-1))

def test_log_matrix():
    print("\n=== TEST: log_matrix ===")
    dur = Duration(n_states=2, max_duration=4)
    L = dur.log_matrix()
    print("log_matrix shape:", L.shape)
    print(L.detach().cpu().numpy())
    print("exp(log_matrix):", L.exp().detach().cpu().numpy())

def test_log_prob_sequence():
    print("\n=== TEST: log_prob simple sequence ===")
    dur = Duration(n_states=2, max_duration=4)
    seq = torch.tensor([[1, 2]])
    mod_logits = dur._modulate()
    log_probs = F.log_softmax(mod_logits, dim=-1)
    print("Sequence:", seq.detach().cpu().numpy())
    print("log_probs shape:", log_probs.shape)

def test_sampling_correctness():
    print("\n=== TEST: Sampling Correctness (empirical) ===")
    dur = Duration(n_states=2, max_duration=4, init_mode="uniform")
    probs = dur.expected_probs().detach()
    print("Theoretical probs shape:", probs.shape)

    N = 5000
    counts = torch.zeros(probs.shape[-1], dtype=torch.float32)
    for _ in range(N):
        s = dur.sample()
        s_idx = s.argmax(dim=-1).reshape(-1)  # flatten batch/time dimensions
        counts += torch.bincount(s_idx, minlength=probs.shape[-1]).float()

    empirical = counts / counts.sum()
    print("Empirical freq shape:", empirical.shape)
    print("Empirical freq:", empirical)

def test_gradient_flow():
    print("\n=== TEST: Gradient Flow ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3, requires_grad=True)
    probs = dur.expected_probs(context=ctx).sum()
    probs.backward()
    print("grad logits:", None if dur.logits.grad is None else dur.logits.grad.norm().item())
    print("grad context:", ctx.grad)

def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    dur1 = Duration(n_states=1, max_duration=1)
    p1 = dur1.expected_probs().detach()
    print("n_states=1, max_duration=1, probs:", p1.detach().cpu().numpy())

    try:
        dur = Duration(n_states=5, max_duration=10)
        out = dur.sample(context=None)
        print(f"Large batch (context=None) OK, sample shape: {out.shape}")
    except Exception as e:
        print("Large batch raised:", e)

def test_update_and_cache():
    print("\n=== TEST: update() and cache ===")
    dur = Duration(n_states=2, max_duration=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3)

    # --- Initial sampling ---
    out1 = dur.sample(context=ctx)
    out2 = dur.sample(context=ctx)
    print("Cache check - identical samples shapes:", out1.shape, out2.shape)
    assert out1.shape == out2.shape

    # --- Pseudo posterior update (EM-style, no grad required) ---
    mod_logits = dur._modulate(context=ctx)
    posterior = torch.ones_like(mod_logits) / mod_logits.numel()  # shape matches logits

    dur.update(posterior=posterior, context=ctx, update_rate=0.5)

    # --- Check sampling after update ---
    out3 = dur.sample(context=ctx)
    print("Updated sample shape:", out3.shape)
    assert out3.shape == out1.shape

def test_timestep_handling():
    print("\n=== TEST: Timestep Handling ===")
    dur = Duration(n_states=2, max_duration=4)
    mod_logits = dur._modulate(timestep=3)
    print("Modulated logits with timestep=3 shape:", mod_logits.shape)

if __name__ == "__main__":
    test_basic()
    test_temperature()
    test_context_single()
    test_context_batch()
    test_log_matrix()
    test_log_prob_sequence()
    test_sampling_correctness()
    test_gradient_flow()
    test_edge_cases()
    test_timestep_handling()
    test_update_and_cache()
    print("\n✓ All Duration tests finished.")
