# tests/dist_transition.py

import torch
import torch.nn.functional as F
from nhsmm.distributions import Transition
from nhsmm.constants import DTYPE


def set_seed(seed: int = 42):
    torch.manual_seed(seed)


# ---------------- Basic Functionality ----------------
def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    tr = Transition(n_states=3, n_features=3)
    print("Logits shape:", tr.logits.shape)

    probs = tr.expected_probs().detach()
    print("Probs shape:", probs.shape)
    print("Rows sum to 1:", probs.sum(dim=-1))


# ---------------- Temperature Scaling ----------------
def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    tr = Transition(n_states=3, n_features=3)
    cold = tr._modulate(temperature=0.1).detach()
    hot = tr._modulate(temperature=5.0).detach()
    print("Cold shape:", cold.shape)
    print("Hot shape:", hot.shape)


# ---------------- Context Tests ----------------
def test_context_single():
    print("\n=== TEST: Context Single ===")
    tr = Transition(n_states=2, n_features=2, context_dim=4, hidden_dim=8)
    ctx = torch.randn(4)
    probs = tr.expected_probs(context=ctx).detach()
    print("Context:", ctx)
    print("Probs shape:", probs.shape)
    print("Row sums:", probs.sum(dim=-1))


def test_context_batch():
    print("\n=== TEST: Context Batch ===")
    tr = Transition(n_states=2, n_features=2, context_dim=4, hidden_dim=8)
    ctx = torch.randn(5, 4)
    probs = tr.expected_probs(context=ctx).detach()
    print("Context shape:", ctx.shape)
    print("Probs shape:", probs.shape)
    print("Row sums per batch:", probs.sum(dim=-1))


# ---------------- Timestep Handling ----------------
def test_timestep():
    print("\n=== TEST: Timestep Handling ===")
    tr = Transition(n_states=2, n_features=2)
    mod = tr._modulate(timestep=3)
    print("mod(t=3) shape:", mod.shape)


# ---------------- Constraint Tests ----------------
def test_constraints():
    print("\n=== TEST: Constraint Types ===")
    tr_ltr = Transition(n_states=3, n_features=3, transition_type="left-to-right")
    print("LTR mask logits:\n", tr_ltr.log_matrix().detach())

    tr_semi = Transition(n_states=3, n_features=3, transition_type="semi")
    print("Semi-Markov mask logits:\n", tr_semi.log_matrix().detach())


# ---------------- Sampling Correctness ----------------
def test_sampling_correctness(N: int = 5000):
    print("\n=== TEST: Sampling Correctness ===")
    tr = Transition(n_states=2, n_features=2, init_mode="uniform")
    probs = tr.expected_probs().detach()  # (n_states, n_features)
    counts = torch.zeros_like(probs)

    for _ in range(N):
        s = tr.sample()  # returns one-hot per row
        counts += s.float()

    empirical = counts / counts.sum(dim=-1, keepdim=True)
    print("Empirical freq:\n", empirical)
    assert torch.allclose(empirical, probs, atol=0.05)


# ---------------- Gradient Flow ----------------
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


# ---------------- Edge Cases ----------------
def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    tr1 = Transition(n_states=1, n_features=1)
    print("n_states=1, n_features=1, probs:", tr1.expected_probs().cpu().detach().numpy())

    tr_large = Transition(n_states=5, n_features=5)
    out = tr_large.sample()
    print("Large batch sample shape:", out.shape)


# ---------------- Update and Cache ----------------
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


# ---------------- Batch + Timestep Modulation ----------------
def test_batch_timestep_modulation():
    print("\n=== TEST: Batch + Timestep Modulation ===")
    tr = Transition(n_states=3, n_features=3)
    B, T = 4, 3
    ctx = torch.randn(B, 3)
    timesteps = torch.arange(1, T+1).repeat(B, 1)

    mod_logits = torch.stack([tr._modulate(context=ctx[b], timestep=timesteps[b, t])
                              for b in range(B) for t in range(T)], dim=0)
    print("Batched modulated logits shape:", mod_logits.shape)

    single_mod = torch.stack([tr._modulate(timestep=t) for t in range(2, 4)], dim=0)
    print("Single batch modulated logits shape:", single_mod.shape)


# ---------------- Log Matrix ----------------
def test_log_matrix():
    print("\n=== TEST: log_matrix ===")
    tr = Transition(n_states=2, n_features=2)
    L = tr.log_matrix()
    print("log_matrix:\n", L)
    print("exp rows sum:", L.exp().sum(-1))


# ---------------- Main ----------------
if __name__ == "__main__":
    set_seed()
    test_basic()
    test_temperature()
    test_context_single()
    test_context_batch()
    test_timestep()
    test_constraints()
    test_sampling_correctness()
    test_gradient_flow()
    test_edge_cases()
    test_update_cache()
    test_batch_timestep_modulation()
    test_log_matrix()
    print("\n✓ All Transition tests completed.")
