# tests/dist_initial.py

import torch
from nhsmm.distributions import Initial

def test_basic():
    print("\n=== TEST: Basic Functionality ===")
    init = Initial(n_states=5, init_mode="uniform")
    print("Logits:", init.logits.detach().cpu().numpy())

    probs = init().detach()
    print("Probs:", probs.cpu().numpy())
    print("Sum of probs:", probs.sum().item())

    sample_vec = init.sample()
    print("Sample vector:", sample_vec.detach().cpu().numpy())
    print("Sample index (argmax):", sample_vec.argmax().item())

def test_temperature():
    print("\n=== TEST: Temperature Scaling ===")
    init = Initial(n_states=4, init_mode="uniform")

    cold = init(temperature=0.1).detach()
    hot = init(temperature=5.0).detach()
    print("Cold (τ=0.1):", cold.cpu().numpy())
    print("Hot  (τ=5.0):", hot.cpu().numpy())
    print("Entropy cold:", -(cold * cold.log()).sum().item())
    print("Entropy hot :", -(hot * hot.log()).sum().item())

def test_context_single():
    print("\n=== TEST: Context Single Vector ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(3)
    probs = init(context=ctx).detach()
    print("Context:", ctx.detach().cpu().numpy())
    print("Probs:", probs.detach().cpu().numpy())
    print("Sum:", probs.sum().item())

def test_context_batch():
    print("\n=== TEST: Context Batch ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(7, 3)
    probs = init(context=ctx).detach()
    print("Context shape:", ctx.shape)
    print("Probs shape:", probs.shape)
    print("Probs:", probs.detach().cpu().numpy())

def test_log_matrix():
    print("\n=== TEST: log_matrix ===")
    init = Initial(n_states=4)
    L = init.log_matrix()
    print("log_matrix shape:", L.shape)
    print(L.detach().cpu().numpy())
    print("exp(log_matrix):", L.exp().detach().cpu().numpy())

def test_log_prob_sequence():
    print("\n=== TEST: log_prob simple sequence ===")
    init = Initial(n_states=4)
    seq = torch.tensor([2])
    lp = init.log_prob(seq)
    print("Sequence:", seq.detach().cpu().numpy())
    print("log_prob:", lp.item())

def test_log_prob_batch():
    print("\n=== TEST: log_prob batch ===")
    init = Initial(n_states=4)
    seq = torch.tensor([[0], [3], [1]])
    lp = init.log_prob(seq)
    print("Sequences:\n", seq.detach().cpu().numpy())
    print("Batch log_prob:", lp.detach().cpu().numpy())

def test_log_prob_context():
    print("\n=== TEST: log_prob with context ===")
    init = Initial(n_states=4, context_dim=5, hidden_dim=16)
    batch = 6
    seq = torch.randint(high=4, size=(batch, 1))
    ctx = torch.randn(batch, 5)
    lp = init.log_prob(seq, context=ctx)
    print("Sequences:\n", seq.detach().cpu().numpy())
    print("Context shape:", ctx.shape)
    print("log_prob:", lp.detach().cpu().numpy())

def test_sampling_correctness():
    print("\n=== TEST: Sampling Correctness (empirical) ===")
    init = Initial(n_states=4, init_mode="uniform")
    probs = init().detach()
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
    probs = init(context=ctx).sum()
    probs.backward()
    print("grad logits:", None if init.logits.grad is None else init.logits.grad.norm().item())
    print("grad context:", ctx.grad)

def test_edge_cases():
    print("\n=== TEST: Edge Cases ===")
    init1 = Initial(n_states=1)
    p1 = init1().detach()
    print("n_states = 1, probs:", p1.detach().cpu().numpy())

    # Large batch with context=None
    try:
        init = Initial(n_states=5)
        out = init(context=None)
        print(f"Large batch (context=None) OK, shape: {out.shape}")
    except Exception as e:
        print("Large batch raised:", e)

def test_update_and_cache():
    print("\n=== TEST: update() and cache ===")
    init = Initial(n_states=4, context_dim=3, hidden_dim=8)
    ctx = torch.randn(1, 3)
    
    # Initial forward pass
    out1 = init(context=ctx)
    # Call again to check cache usage
    out2 = init(context=ctx)
    assert torch.allclose(out1, out2), "Cache mismatch"

    # Update logits via pseudo-posterior
    posterior = torch.ones_like(out1) / out1.shape[-1]
    init.update(posterior=posterior, context=ctx, update_rate=0.5)
    out3 = init(context=ctx)
    assert not torch.allclose(out1, out3), "Logits did not update"

if __name__ == "__main__":
    test_basic()
    test_temperature()
    test_context_single()
    test_context_batch()
    test_log_matrix()
    test_log_prob_sequence()
    test_log_prob_context()
    test_log_prob_batch()
    test_sampling_correctness()
    test_gradient_flow()
    test_edge_cases()
    test_update_and_cache()
    print("\n✓ All tests finished.")
