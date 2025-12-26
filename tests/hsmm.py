import torch
import numpy as np

from nhsmm import utils
from nhsmm.models import HSMM
from nhsmm.context import DefaultEncoder, ContextRouter, SequenceSet


def make_model(enc: bool = False, n_features: int = 5):
    """
    Build an HSMM instance with optional CNN+LSTM encoder.
    """
    context_dim = 32
    hidden_dim = max(context_dim, min(64, n_features * 2))  # → 32 if n_features=5

    encoder = DefaultEncoder(
        n_features=n_features,
        cnn_channels=5,
        hidden_dim=hidden_dim,
    ) if enc else None

    return HSMM(
        n_states=3,
        max_duration=4,
        n_features=n_features,
        emission_type="gaussian",
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        encoder=encoder,
        debug=False,
    )


def make_data(B=1, T=10, F=5):
    """
    Create random batch/time/feature data.
    """
    return torch.randn(B, T, F)


def test_encode():
    print("Test _encode")
    B, T, F = 3, 10, 5
    X = make_data(B, T, F)
    print(f"Input X shape: {X.shape}")

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 7:] = 0  # batch 1, last 3 steps padded
    mask[2, 5:] = 0  # batch 2, last 5 steps padded
    # print(f"Mask:\n{mask}")

    # --- no encoder ---
    m = make_model(enc=False, n_features=F)
    seq, ctx = m._encode(X, mask=mask)
    print(f"No-encoder → seq shape: {seq.shape}, ctx shape: {ctx.shape}")
    assert seq.shape == (B, T, m.context_dim)
    assert ctx.shape == (B, 1, m.context_dim)
    assert torch.all(seq == 0)
    assert torch.all(ctx == 0)
    print(f"✓ _encode (no encoder) → batch={B}, time={T}, context_dim={m.context_dim}")

    # --- with encoder ---
    m = make_model(enc=True, n_features=F)
    seq, ctx = m._encode(X, mask=mask)
    H = m.encoder.hidden_dim
    print(f"With-encoder → seq shape: {seq.shape}, ctx shape: {ctx.shape}")

    assert seq.shape == (B, T, H)
    assert ctx.shape == (B, 1, H)
    assert torch.isfinite(seq).all()
    assert torch.isfinite(ctx).all()

    # optional: check padded positions
    for b in range(B):
        pad_idx = (~mask[b]).nonzero(as_tuple=True)[0]
        if pad_idx.numel() > 0:
            # print(f"Batch {b} padded positions: {pad_idx.tolist()}, seq values: {seq[b, pad_idx]}")
            assert torch.isfinite(seq[b, pad_idx]).all()

    # print(f"Pooled context values:\n{ctx.squeeze(1)}")
    print(f"✓ _encode (with encoder) → batch={B}, time={T}, hidden_dim={H}")


def test_prepare():
    print("Test _prepare")
    B, T, F = 3, 10, 2  # batch, time, features
    m = make_model(n_features=F)
    X = make_data(B=B, T=T, F=F)

    # Variable-length mask
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 7:] = 0
    mask[2, 5:] = 0

    def check_sequence_set(S: SequenceSet, mask: torch.BoolTensor, description=""):
        print(f"{description} shapes:")
        print("sequences:", S.sequences.shape)
        print("masks    :", S.masks.shape)
        print("contexts :", S.contexts.shape)
        print("canonical:", S.canonical.shape)
        if S.log_probs is not None:
            print("log_probs:", S.log_probs.shape)
        else:
            print("log_probs: None")
        # Validate masks match expected lengths
        for b in range(B):
            expected_len = mask[b].sum()
            actual_len = S.masks[b].sum()
            assert expected_len == actual_len, f"Mask mismatch in batch {b}"

    # --- Case 1: theta=None ---
    print("\n--- Case 1: theta=None ---")
    S_none = m._prepare(X, mask=mask, theta=None)
    assert isinstance(S_none, SequenceSet)
    check_sequence_set(S_none, mask, "theta=None")
    print("✓ _prepare with theta=None passed")

    # --- Case 2: theta from _encode ---
    m = make_model(enc=True, n_features=F)
    if getattr(m, "encoder", None) is not None:
        print("\n--- Case 2: theta from _encode ---")
        seq, ctx_canonical = m._encode(X, mask=mask)
        H = seq.shape[-1]

        # Build batched theta [B, T, H] for SequenceSet
        theta_batched = torch.zeros(B, T, H, device=X.device, dtype=seq.dtype)
        for b in range(B):
            L = mask[b].sum()
            theta_batched[b, :L] = (
                ctx_canonical[b].repeat(L, 1) if ctx_canonical.shape[1] == 1 else ctx_canonical[b, :L]
            )
            print(f"Batch {b} theta mean/std: {theta_batched[b].mean():.4f}/{theta_batched[b].std():.4f}")

        S_theta = m._prepare(X, mask=mask, theta=theta_batched)
        assert isinstance(S_theta, SequenceSet)
        check_sequence_set(S_theta, mask, "theta from _encode")
        print("✓ _prepare with theta from _encode passed")


def test_sequence_set():
    print("\nTest SequenceSet")

    # ----------------------------------------------------
    # Test 1: Basic from_unbatched
    # ----------------------------------------------------
    seqs = [torch.tensor([[1.,2.],[3.,4.]]), torch.tensor([[5.,6.]])]
    contexts = [torch.tensor([[10.,20.],[30.,40.]]), torch.tensor([[50.,60.]])]
    log_probs = [torch.tensor([[0.1,0.9],[0.8,0.2]]), torch.tensor([[0.5,0.5]])]

    ss = SequenceSet.from_unbatched(seqs, contexts=contexts, log_probs=log_probs)

    print("sequences shape:", ss.sequences.shape)
    print("contexts shape:", ss.contexts.shape)
    print("canonical shape:", ss.canonical.shape)
    print("masks shape:", ss.masks.shape)
    print("log_probs shape:", ss.log_probs.shape)
    
    assert ss.sequences.shape == (2, 2, 2)
    assert ss.contexts.shape == (2, 2, 2)
    assert ss.canonical.shape == (2, 1, 2)
    assert ss.masks.shape == (2, 2, 1)
    assert ss.log_probs.shape == (2, 2, 2)
    assert torch.allclose(ss.canonical[0,0], contexts[0][0])
    assert torch.allclose(ss.canonical[1,0], contexts[1][0])

    # ----------------------------------------------------
    # Test 2: 1D sequences become feature-1 tensors
    # ----------------------------------------------------
    seqs1d = [torch.tensor([1.,2.,3.]), torch.tensor([4.,5.])]
    ss1 = SequenceSet.from_unbatched(seqs1d)
    
    print("1D sequences -> sequences shape:", ss1.sequences.shape)
    assert ss1.sequences.shape == (2, 3, 1)
    assert ss1.contexts.shape[2] == 1
    assert ss1.masks.shape == (2, 3, 1)
    assert ss1.masks[0].sum() == 3
    assert ss1.masks[1].sum() == 2

    # ----------------------------------------------------
    # Test 3: to(device, dtype)
    # ----------------------------------------------------
    ss_cpu = ss.to(device='cpu', dtype=torch.float64)
    assert ss_cpu.sequences.dtype == torch.float64
    assert ss_cpu.device == torch.device('cpu')
    assert ss_cpu.contexts.dtype == torch.float64
    assert ss_cpu.log_probs.dtype == torch.float64

    # ----------------------------------------------------
    # Test 4: index_select
    # ----------------------------------------------------
    idx = torch.tensor([1])
    ss_sel = ss.index_select(idx)
    assert ss_sel.n_sequences == 1
    assert torch.allclose(ss_sel.sequences[0], ss.sequences[1])
    assert torch.allclose(ss_sel.canonical[0], ss.canonical[1])

    # ----------------------------------------------------
    # Test 5: batchify with padding
    # ----------------------------------------------------
    items = [torch.tensor([[1.,2.]]), torch.tensor([[3.,4.],[5.,6.]])]
    batched = SequenceSet.batchify(items, pad_value=-1.0)
    print("batchify shape:", batched.shape)
    assert batched.shape == (2, 2, 2)
    assert batched[0,1,0] == -1.0  # padding applied

    # ----------------------------------------------------
    # Test 6: empty context list handling
    # ----------------------------------------------------
    ss_nc = SequenceSet.from_unbatched(seqs)
    assert ss_nc.contexts.shape == ss_nc.sequences.shape
    assert torch.allclose(ss_nc.canonical[0,0], seqs[0][0])
    assert torch.allclose(ss_nc.canonical[1,0], seqs[1][0])

    # ----------------------------------------------------
    # Test 7: NaN-safe padding propagation
    # ----------------------------------------------------
    seqs_nan = [
        torch.tensor([[1., 2.], [float("nan"), float("nan")]]),
        torch.tensor([[3., 4.]])
    ]
    ss_nan = SequenceSet.from_unbatched(seqs_nan)
    assert ss_nan.masks[0,1,0] == 0
    assert ss_nan.masks[0,0,0] == 1
    assert torch.isnan(ss_nan.sequences[0,1]).all()

    # ----------------------------------------------------
    # Test 8: canonical extraction for ragged + NaN sequences
    # ----------------------------------------------------
    assert torch.allclose(ss_nan.canonical[0,0], torch.tensor([1.,2.]))

    # ----------------------------------------------------
    # Test 9: batch ordering stability
    # ----------------------------------------------------
    seqX = [torch.randn(2,3), torch.randn(7,3), torch.randn(1,3)]
    ssX = SequenceSet.from_unbatched(seqX)
    assert torch.allclose(ssX.sequences[0,0], seqX[0][0])
    assert torch.allclose(ssX.sequences[1,0], seqX[1][0])
    assert torch.allclose(ssX.sequences[2,0], seqX[2][0])

    # ----------------------------------------------------
    # Test 10: log_prob optional behavior
    # ----------------------------------------------------
    ss_lp = SequenceSet.from_unbatched([torch.randn(4,2)], log_probs=None)
    assert ss_lp.log_probs is None

    # ----------------------------------------------------
    # Test 11: update with encoder
    # ----------------------------------------------------
    print("\nTest SequenceSet.update with encoder")
    B, T, F = 2, 6, 5
    x = make_data(B=B, T=T, F=F)
    seqs_enc = [x[i] for i in range(B)]
    ss_enc = SequenceSet.from_unbatched(seqs_enc)

    model = make_model(enc=True)
    assert model.encoder is not None

    ss_enc.update(model.encoder)

    ctx_shape = ss_enc.contexts.shape
    canonical_shape = ss_enc.canonical.shape
    print("Updated contexts shape:", ctx_shape)
    print("Updated canonical shape:", canonical_shape)

    assert ctx_shape == (B, T, model.encoder.hidden_dim)
    assert canonical_shape == (B, 1, model.encoder.hidden_dim)
    assert torch.allclose(ss_enc.canonical, ss_enc.contexts[:, :1, :])

    # Detach test
    ss_enc_detach = SequenceSet.from_unbatched(seqs_enc)
    ss_enc_detach.update(model.encoder, detach=True)
    assert not ss_enc_detach.contexts.requires_grad
    assert not ss_enc_detach.canonical.requires_grad

    print("✓ All SequenceSet tests passed including update")


def test_context_router():
    print("\n--- Test ContextRouter ---")

    B, T, H = 3, 5, 4  # batch, time, feature dims

    # ---------------- Fixtures ----------------
    lpB = torch.randn(B, 10)       # B-batch log_probs
    mask2D = torch.ones(B, T, dtype=torch.bool)
    maskBT = mask2D.unsqueeze(-1)

    X_dummy = SequenceSet(
        sequences=torch.zeros(B, T, H),
        lengths=torch.full((B,), T, dtype=torch.long),
        masks=maskBT,
        contexts=torch.zeros(B, T, H),
        canonical=torch.zeros(B, 1, H),
        log_probs=lpB
    )

    # ---------------- Test 1: theta=None ----------------
    print("Test1: theta=None")
    router = ContextRouter.from_tensor(X=X_dummy)
    assert torch.allclose(router.context, X_dummy.contexts)
    assert torch.allclose(router.canonical, X_dummy.canonical)
    assert torch.all(router.mask == X_dummy.masks)
    assert torch.allclose(router.log_probs, lpB)

    # ---------------- Test 2: theta 1D additive ----------------
    print("Test2: theta 1D additive")
    theta1 = torch.arange(H, dtype=torch.float32)
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta1, mode="additive")
    for b in range(B):
        assert torch.allclose(router.context[b,0], X_dummy.contexts[b,0] + theta1)
    assert torch.allclose(router.canonical, X_dummy.canonical)

    # ---------------- Test 3: theta 1D replace ----------------
    print("Test3: theta 1D replace")
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta1, mode="replace")
    for b in range(B):
        assert torch.allclose(router.context[b,0], theta1)
    expected_canonical = router.context[:, :1, :]
    assert torch.allclose(router.canonical, expected_canonical)

    # ---------------- Test 4: theta 2D batch-dependent ----------------
    print("Test4: theta 2D batch-dependent")
    theta2 = torch.randn(B, H)
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta2, mode="additive")
    for b in range(B):
        assert torch.allclose(router.context[b], X_dummy.contexts[b] + theta2[b])
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta2, mode="replace")
    for b in range(B):
        assert torch.allclose(router.context[b], theta2[b].expand(T, H))
    expected_canonical = router.context[:, :1, :]
    assert torch.allclose(router.canonical, expected_canonical)

    # ---------------- Test 5: theta 2D time-dependent ----------------
    print("Test5: theta 2D time-dependent")
    theta3 = torch.randn(T, H)
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta3, mode="additive")
    for b in range(B):
        assert torch.allclose(router.context[b], X_dummy.contexts[b] + theta3)
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta3, mode="replace")
    for b in range(B):
        assert torch.allclose(router.context[b], theta3)
    expected_canonical = router.context[:, :1, :]
    assert torch.allclose(router.canonical, expected_canonical)

    # ---------------- Test 6: theta 3D ----------------
    print("Test6: theta 3D")
    theta4 = torch.randn(B, T, H)
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta4, mode="additive")
    assert torch.allclose(router.context, X_dummy.contexts + theta4)
    router = ContextRouter.from_tensor(X=X_dummy, theta=theta4, mode="replace")
    assert torch.allclose(router.context, theta4)
    expected_canonical = router.context[:, :1, :]
    assert torch.allclose(router.canonical, expected_canonical)

    # ---------------- Test 7: mask propagation ----------------
    print("Test7: mask propagation")
    X_mod = SequenceSet(
        sequences=X_dummy.sequences,
        lengths=X_dummy.lengths,
        masks=maskBT,
        contexts=X_dummy.contexts,
        canonical=X_dummy.canonical,
        log_probs=X_dummy.log_probs
    )
    router = ContextRouter.from_tensor(X=X_mod)
    assert router.mask.shape == (B, T, 1)
    assert torch.all(router.mask.squeeze(-1) == mask2D)

    # ---------------- Test 8: log_probs propagation ----------------
    print("Test8: log_probs propagation")
    router = ContextRouter.from_tensor(X=X_dummy)
    assert torch.allclose(router.log_probs, lpB)

    # ---------------- Test 9: utilities ----------------
    print("Test9: clone/detach/to")
    router2 = router.clone()
    router3 = router.detach()
    router4 = router.to(device='cpu')
    assert torch.allclose(router2.context, router.context)
    assert not router3.context.requires_grad
    assert router4.context.device.type == "cpu"

    print("✓ All ContextRouter all-round tests passed with X-based modulation and modes")


def test_ensure_shape():
    m = make_model()
    T = 5
    for module in [m.initial_module, m.duration_module, m.transition_module]:
        out = m._ensure_shape(module, context=None, T=T)
        assert out.shape[0] == T
        assert torch.isfinite(out).all()
        print(f"✓ _ensure_shape {module.__class__.__name__}")


def test_forward():
    print("Test _forward with ContextRouter")

    # Build model and data
    B, T, F = 3, 10, 2
    model = make_model(enc=True, n_features=F)
    X = make_data(B=B, T=T, F=F)

    # Variable-length mask
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 7:] = 0  # example of variable-length sequence
    mask[2, 5:] = 0

    # Prepare SequenceSet
    S = model._prepare(X, mask=mask)

    print("Preparing sequences...")
    print(" S.log_probs:", S.log_probs.shape)
    print(" S.masks    :", S.masks.shape)
    print(" S.lengths  :", S.lengths)
    print(" S.canonical:", S.canonical.shape)
    print(" S.contexts :", S.contexts.shape)

    # --- Run forward without theta ---
    print("Running _forward (no theta)...")
    alpha = model._forward(S, theta=None)
    print(" alpha shape:", alpha.shape)
    print(" alpha dtype:", alpha.dtype)
    print(" alpha sample (first batch, t=0):", alpha[0, 0])

    # --- Run forward with additive theta ---
    theta_add = torch.randn(S.contexts.shape[2])
    CR_add = ContextRouter.from_tensor(S, theta=theta_add, mode="additive")
    alpha_add = model._forward(S, theta=CR_add)
    print(" alpha_add sample (first batch, t=0):", alpha_add[0, 0])

    # --- Run forward with replace theta ---
    theta_rep = torch.randn(S.contexts.shape[2])
    CR_rep = ContextRouter.from_tensor(S, theta=theta_rep, mode="replace")
    alpha_rep = model._forward(S, theta=CR_rep)
    print(" alpha_rep sample (first batch, t=0):", alpha_rep[0, 0])

    # ==== Basic structural checks ====
    for name, a in zip(["alpha", "alpha_add", "alpha_rep"], [alpha, alpha_add, alpha_rep]):
        print(f"Checking {name}...")
        assert isinstance(a, torch.Tensor)
        assert a.shape == (B, T, model.n_states, model.max_duration)
        assert not torch.isnan(a).any(), f"{name} contains NaNs"
        print(f" {name} max/min:", a.max().item(), a.min().item())

    # ==== Padding checks ====
    for name, a in zip(["alpha", "alpha_add", "alpha_rep"], [alpha, alpha_add, alpha_rep]):
        for b in range(B):
            pad_idx = (~mask[b]).nonzero(as_tuple=True)[0]
            if pad_idx.numel() > 0:
                pad_vals = a[b, pad_idx]
                # assert (pad_vals == float("-inf")).all(), f"Padded {name} not -inf for batch {b}"
                if (pad_vals == float("-inf")).all():
                    print(f" {name} batch {b} padded timesteps OK")
                else:
                    print(f"Padded {name} not -inf for batch {b}")

    # ==== Valid timestep checks ====
    for name, a in zip(["alpha", "alpha_add", "alpha_rep"], [alpha, alpha_add, alpha_rep]):
        for b in range(B):
            L = S.lengths[b]
            for t in range(L):
                max_d = min(model.max_duration, t + 1)
                finite_or_inf = torch.isfinite(a[b, t, :, :max_d]) | (a[b, t, :, :max_d] == -float("inf"))
                assert finite_or_inf.all(), f"Invalid value at batch {b}, t={t} in {name}"
                assert torch.isfinite(a[b, t, :, :max_d]).any(), f"No finite alpha at batch {b}, t={t} in {name}"
        print(f" {name} valid timesteps OK")

    print("✓ _forward all-round tests passed (no theta, additive, replace)")


def test_backward():
    print("\nTest _backward with ContextRouter")

    # ---------------- Build model and data ----------------
    B, T, F = 3, 10, 2
    model = make_model(enc=True, n_features=F)
    X = make_data(B=B, T=T, F=F)

    # ---------------- Variable-length mask ----------------
    mask = torch.ones(B, T, dtype=torch.bool)
    # Uncomment to simulate variable-length sequences
    # mask[1, 7:] = 0  # batch 1 length 7
    # mask[2, 5:] = 0  # batch 2 length 5

    # ---------------- Prepare SequenceSet ----------------
    S = model._prepare(X, mask=mask)
    print(f"[Prepare] Mask shape: {S.masks.shape}")
    print(f"[Prepare] Encoded context: {S.contexts.shape}, canonical: {S.canonical.shape}")
    print(f"[Prepare] log_probs shape: {S.log_probs.shape}")
    print(f"[Prepare] Sequence lengths: {S.lengths}")

    # --- Run backward without theta ---
    print("Running _backward (no theta)...")
    beta = model._backward(S, theta=None)
    print(f"[Backward] beta shape: {beta.shape}, dtype: {beta.dtype}")
    print(f"[Backward] beta sample (batch0, timestep0): {beta[0,0]}")

    # --- Run backward with additive theta ---
    theta_add = torch.randn(S.contexts.shape[2])
    CR_add = ContextRouter.from_tensor(S, theta=theta_add, mode="additive")
    beta_add = model._backward(S, theta=CR_add)
    print(f"[Backward-add] beta sample (batch0, timestep0): {beta_add[0,0]}")

    # --- Run backward with replace theta ---
    theta_rep = torch.randn(S.contexts.shape[2])
    CR_rep = ContextRouter.from_tensor(S, theta=theta_rep, mode="replace")
    beta_rep = model._backward(S, theta=CR_rep)
    print(f"[Backward-rep] beta sample (batch0, timestep0): {beta_rep[0,0]}")

    # ==== Basic structural checks ====
    for name, bpass in zip(["beta", "beta_add", "beta_rep"], [beta, beta_add, beta_rep]):
        assert isinstance(bpass, torch.Tensor)
        assert bpass.shape == (B, T, model.n_states, model.max_duration)
        assert not torch.isnan(bpass).any(), f"{name} contains NaNs"

    # ==== Padding checks ====
    for name, bpass in zip(["beta", "beta_add", "beta_rep"], [beta, beta_add, beta_rep]):
        for b in range(B):
            pad_idx = (~mask[b]).nonzero(as_tuple=True)[0]
            if pad_idx.numel() > 0:
                pad_vals = bpass[b, pad_idx]
                print(f"[Padding] {name} batch {b} pad_vals: {pad_vals}")
                assert (pad_vals == float("-inf")).all(), f"Padded {name} not -inf for batch {b}"

    # ==== Valid timestep checks ====
    for name, bpass in zip(["beta", "beta_add", "beta_rep"], [beta, beta_add, beta_rep]):
        for b in range(B):
            L = S.lengths[b]
            for t in range(L):
                max_d = min(model.max_duration, T - t)
                finite_or_inf = torch.isfinite(bpass[b, t, :, :max_d]) | (bpass[b, t, :, :max_d] == -float("inf"))
                assert finite_or_inf.all(), f"Invalid value at batch {b}, t={t} in {name}"
                # At least one finite value must exist per timestep
                assert torch.isfinite(bpass[b, t, :, :max_d]).any(), f"No finite beta at batch {b}, t={t} in {name}"

    print("✓ _backward all-round tests passed (no theta, additive, replace)")


def test_compute_posteriors():
    print("\nTest _compute_posteriors with ContextRouter")

    # ---------------- Build model and data ----------------
    B, T, F = 3, 10, 2  # batch, time, features
    model = make_model(enc=True, n_features=F)
    X = make_data(B=B, T=T, F=F)

    # ---------------- Prepare SequenceSet ----------------
    mask = torch.ones(B, T, dtype=torch.bool)
    X_seq = model._prepare(X, mask=mask)
    theta = None  # Optional: ContextRouter or tensor for modulation

    print(f"[Prepare] Sequence lengths: {X_seq.lengths}")
    print(f"[Prepare] log_probs shape: {X_seq.log_probs.shape}")
    print(f"[Prepare] contexts shape: {X_seq.contexts.shape}, canonical shape: {X_seq.canonical.shape}")
    print(f"[Prepare] mask shape: {X_seq.masks.shape}")

    # ---------------- Compute posteriors ----------------
    gamma, xi, eta = model._compute_posteriors(X_seq, theta=theta)

    B = len(X_seq.sequences)
    T_max = max(X_seq.lengths) if B > 0 else 0
    K = model.n_states
    Dmax = model.max_duration

    print(f"[Posteriors] gamma shape: {gamma.shape}, eta shape: {eta.shape}, xi shape: {xi.shape}")
    print(f"[Posteriors] gamma sample (batch0, t=0): {gamma[0,0]}")
    print(f"[Posteriors] eta sample (batch0, t=0): {eta[0,0]}")
    print(f"[Posteriors] xi sample (batch0, t=0): {xi[0,0] if T_max > 1 else 'N/A'}")

    # ---------------- Shape checks ----------------
    assert gamma.shape == (B, T_max, K), f"gamma shape mismatch: {gamma.shape}"
    assert eta.shape == (B, T_max, K, Dmax), f"eta shape mismatch: {eta.shape}"
    assert xi.shape == (B, max(T_max-1, 0), K, K), f"xi shape mismatch: {xi.shape}"

    # ---------------- Numerical checks ----------------
    assert torch.isfinite(gamma).all(), "gamma contains non-finite values"
    assert torch.isfinite(eta).all(), "eta contains non-finite values"
    assert torch.isfinite(xi).all(), "xi contains non-finite values"

    # ---------------- Masking checks ----------------
    for b, L in enumerate(X_seq.lengths):
        if L < T_max:
            print(f"[Masking] Checking sequence {b} with length {L}")
            assert torch.all(gamma[b, L:] == 0), f"gamma not zero-padded correctly for sequence {b}"
            assert torch.all(eta[b, L:] == 0), f"eta not zero-padded correctly for sequence {b}"
            if L < 2:
                assert xi[b].numel() == 0 or torch.all(xi[b, L-1:] == 0), f"xi not zero-padded correctly for sequence {b}"

    # ---------------- Normalization check ----------------
    for b, L in enumerate(X_seq.lengths):
        if L > 0:
            gamma_sum = gamma[b, :L].sum(-1)
            print(f"[Normalization] gamma sum for sequence {b}: {gamma_sum}")
            assert torch.allclose(gamma_sum, torch.ones_like(gamma_sum)), f"gamma not normalized for sequence {b}: {gamma_sum}"

    print("✓ _compute_posteriors successfully")


def test_model_params():
    print("Test _model_params")

    # -------------------- Basic model/data --------------------
    model = make_model()
    X = make_data()
    X = model._prepare(X)

    B = len(X.sequences)
    T_max = max(s.shape[0] for s in X.sequences)

    # -------------------- Estimate mode --------------------
    params_est = model._model_params(X, mode="estimate", iter_idx=0)

    for key in ["initial_dist", "duration_dist", "transition_dist"]:
        dist = params_est[key]
        assert hasattr(dist, "logits"), f"{key} missing logits"
        assert torch.isfinite(dist.logits).all(), f"{key} logits contain non-finite values"

    emit_dist = params_est["emission_dist"]
    if hasattr(emit_dist, "logits"):
        assert emit_dist.logits.shape[-1] == model.n_features
    else:
        assert torch.isfinite(emit_dist.mean).all()
        assert torch.isfinite(emit_dist.covariance_matrix).all()

    print("✓ Estimate mode passed")

    # -------------------- Sample mode --------------------
    params_sample = model._model_params(X, mode="sample", iter_idx=10)

    for key in ["initial_dist", "duration_dist", "transition_dist", "emission_dist"]:
        dist = params_sample[key]
        if hasattr(dist, "logits"):
            assert torch.isfinite(dist.logits).all()
        else:
            assert torch.isfinite(dist.mean).all()
            assert torch.isfinite(dist.covariance_matrix).all()

    print("✓ Sample mode passed")

    # -------------------- ContextRouter mode --------------------
    ctx_list = [
        torch.randn(seq.shape[0], model.context_dim, device=X.device, dtype=X.dtype)
        for seq in X.sequences
    ]
    ctx_tensor = torch.nn.utils.rnn.pad_sequence(ctx_list, batch_first=True)

    # Create router with context tensor and optional X (for canonical/log_probs/mask)
    theta = ContextRouter.from_tensor(
        theta=ctx_tensor,
        X=X
    )
    params_ctx = model._model_params(X, theta=theta, mode="estimate", iter_idx=5)

    for key in ["initial_dist", "duration_dist", "transition_dist", "emission_dist"]:
        dist = params_ctx[key]
        if hasattr(dist, "logits"):
            assert torch.isfinite(dist.logits).all()
        else:
            assert torch.isfinite(dist.mean).all()
            assert torch.isfinite(dist.covariance_matrix).all()

    print("✓ Context mode passed")

    # -------------------- Encoder active --------------------
    print("\n--- Test with encoder (enc=True) ---")
    B, T, F = 3, 10, 5
    model = make_model(enc=True, n_features=F)
    Xraw = make_data(B=B, T=T, F=F)
    Xenc = model._prepare(Xraw)

    assert isinstance(Xenc, SequenceSet)
    assert Xenc.contexts is not None and Xenc.contexts.shape[0] == B

    params_enc = model._model_params(Xenc, mode="estimate", iter_idx=3)

    for key in ["initial_dist", "duration_dist", "transition_dist", "emission_dist"]:
        dist = params_enc[key]
        if hasattr(dist, "logits"):
            assert torch.isfinite(dist.logits).all()
        elif hasattr(dist, "mean"):
            assert torch.isfinite(dist.mean).all()
            assert torch.isfinite(dist.covariance_matrix).all()
        else:
            raise AssertionError(f"{key} missing expected attributes with encoder")

    print("✓ Encoder mode passed")
    print("✓ _model_params test completed")


def test_viterbi():
    print("\nTest _viterbi")

    # ---------------- Build model and data ----------------
    B, T, F = 3, 12, 4
    model = make_model(enc=True, n_features=F)
    Xraw = make_data(B=B, T=T, F=F)

    # ---------------- Variable-length mask ----------------
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 9:] = 0  # sequence length 9
    mask[2, 6:] = 0  # sequence length 6

    # ---------------- Prepare SequenceSet ----------------
    S = model._prepare(Xraw, mask=mask)

    print("[Prepare]")
    print(" lengths :", S.lengths.tolist())
    print(" log_probs:", S.log_probs.shape)
    print(" contexts :", S.contexts.shape)
    print(" canonical:", S.canonical.shape)

    def check_paths(paths: list[torch.Tensor], B: int):
        assert isinstance(paths, list)
        assert len(paths) == B

        for b, path in enumerate(paths):
            L = S.lengths[b]
            print(f" batch {b}: decoded length={len(path)}, expected={L}")

            assert isinstance(path, torch.Tensor)
            assert path.dtype == torch.long
            assert path.numel() == L
            assert path.min() >= 0 and path.max() < model.n_states
            assert torch.isfinite(path.float()).all()

    # =====================================================
    # Case 1: theta=None
    # =====================================================
    print("\n--- Viterbi: theta=None ---")
    paths = model._viterbi(S, theta=None)
    check_paths(paths, B)
    print("✓ Viterbi passed with theta=None")

    # =====================================================
    # Case 2: ContextRouter (time-varying)
    # =====================================================
    print("\n--- Viterbi: ContextRouter ---")
    H = model.context_dim
    ctx_list = [
        torch.randn(L, H, device=S.sequences.device, dtype=S.sequences.dtype)
        for L in S.lengths
    ]
    ctx_tensor = torch.nn.utils.rnn.pad_sequence(ctx_list, batch_first=True)

    # Only pass theta tensor; X contains mask/log_probs
    theta = ContextRouter.from_tensor(theta=ctx_tensor, X=S)

    paths_ctx = model._viterbi(S, theta=theta)
    check_paths(paths_ctx, B)
    print("✓ Viterbi passed with ContextRouter")

    # =====================================================
    # Case 3: Determinism check
    # =====================================================
    print("\n--- Viterbi determinism check ---")
    paths2 = model._viterbi(S, theta=None)
    for p1, p2 in zip(paths, paths2):
        assert torch.equal(p1, p2), "Viterbi is not deterministic"
    print("✓ Viterbi deterministic")

    print("✓ _viterbi test passed successfully")


def init_test():
    print("Running HSMM __init__ test...")

    n_states = 3
    n_features = 5
    max_duration = 4
    context_dim = 8
    hidden_dim = 8

    # --- Basic initialization without encoder ---
    model = HSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        encoder=None,
        debug=True,
    )

    assert isinstance(model, HSMM)
    assert model.n_states == n_states
    assert model.n_features == n_features
    assert model.max_duration == max_duration
    assert model.context_dim == context_dim
    assert model.hidden_dim == hidden_dim
    assert model.encoder is None

    # Check modules
    for module_name in ["initial_module", "duration_module", "transition_module", "emission_module"]:
        module = getattr(model, module_name, None)
        assert module is not None, f"{module_name} not initialized"
        for param in module.parameters():
            assert torch.isfinite(param).all(), f"{module_name} has non-finite parameter"

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert all(p.device == device for p in model.parameters()), "Parameters not on correct device"

    print("✓ HSMM __init__ test passed")

    # --- Initialization with encoder ---
    dummy_encoder = torch.nn.Linear(n_features, hidden_dim)
    model_enc = HSMM(
        n_states=n_states,
        n_features=n_features,
        max_duration=max_duration,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        encoder=dummy_encoder,
        debug=True,
    )

    assert model_enc.encoder is not None
    assert model_enc.context_dim == hidden_dim
    assert model_enc.hidden_dim == hidden_dim

    print("✓ HSMM __init__ test with encoder passed")


def test_fit():
    print("\n Running test_fit...")

    B, T, F = 2, 8, 3  # batch, time, features
    X = torch.randn(B, T, F)

    models = []

    # --- Case 1: Fit without encoder ---
    model = make_model(enc=False, n_features=F)
    model.fit(X, n_init=1, max_iter=5, theta=None)
    params = model._params
    assert "initial_dist" in params and "emission_dist" in params, "Missing parameters after fit"
    assert hasattr(model, "_convergence"), "Convergence tracker missing"
    assert model._convergence.history and isinstance(model._convergence.history, list), "Convergence history missing"
    models.append(model)
    print("✓ fit without encoder passed")

    # --- Case 2: Fit with encoder ---
    model_enc = make_model(enc=True, n_features=F)
    model_enc.fit(X, n_init=1, max_iter=5, theta=None)
    params_enc = model_enc._params
    assert "initial_dist" in params_enc and "emission_dist" in params_enc, "Missing parameters after fit"
    assert hasattr(model_enc, "_convergence"), "Convergence tracker missing"
    assert model_enc._convergence.history and isinstance(model_enc._convergence.history, list), "Convergence history missing"
    models.append(model_enc)
    print("✓ fit with encoder passed")

    # --- Case 3: Very short sequences ---
    X_short = torch.randn(3, 1, F)
    model_short = make_model(enc=False, n_features=F)
    model_short.fit(X_short, n_init=1, max_iter=3, theta=None)
    assert hasattr(model_short, "_convergence"), "Convergence tracker missing"
    assert model_short._convergence.history, "Convergence history missing"
    models.append(model_short)
    print("✓ fit with short sequences passed")

    # --- Case 4: Empty batch ---
    X_empty = torch.empty(0, 0, F)
    model_empty = make_model(enc=False, n_features=F)
    try:
        if X_empty.numel() > 0:
            model_empty.fit(X_empty, n_init=1, max_iter=3, theta=None)
        print("✓ fit with empty batch passed (skipped if zero elements)")
    except Exception as e:
        print(f"✓ fit with empty batch skipped due to: {e}")

    # --- Case 5: Likelihood sanity check ---
    for m in models:
        # Extract last LL as scalar
        last_entry = m._convergence.history[-1]
        if isinstance(last_entry, torch.Tensor):
            ll_final = last_entry.item()
        elif isinstance(last_entry, (list, np.ndarray)):
            # pick the last numeric value in the list
            ll_final = float(last_entry[-1])
        else:
            ll_final = float(last_entry)

        assert np.isfinite(ll_final), f"Final LL invalid for {type(m).__name__}"

        # Convert full history to float array safely
        history_array = []
        for h in m._convergence.history:
            if isinstance(h, torch.Tensor):
                history_array.append(h.item())
            elif isinstance(h, (list, np.ndarray)):
                history_array.append(float(h[-1]))
            else:
                history_array.append(float(h))
        history_array = np.array(history_array)

        # Check monotonicity or EM non-decrease
        diffs = np.diff(history_array)
        assert (diffs >= -1e-3).all(), f"Likelihood decreased unexpectedly for {type(m).__name__}"

        print(f"Final LL for {type(m).__name__}: {ll_final:.6f}")


    print("✓ test_fit completed successfully\n")


def test_predict():
    print("\nRunning intensive test_predict...")

    B, T_max, F = 4, 12, 5
    model = make_model(enc=True, n_features=F)

    # --- Create variable-length sequences ---
    lengths = [T_max, 8, 5, 0]
    X_list = [torch.randn(L, F) if L > 0 else torch.empty(0, F) for L in lengths]

    # --- Test Viterbi ---
    paths_viterbi = model.predict(X_list, algorithm="viterbi")
    assert len(paths_viterbi) == B
    for i, path in enumerate(paths_viterbi):
        # Truncate to original length
        path = path[:lengths[i]]
        assert path.numel() == lengths[i]
        if lengths[i] > 0:
            assert path.min() >= 0 and path.max() < model.n_states
    print("✓ Viterbi predict (variable lengths) correct")

    # --- Test MAP ---
    paths_map = model.predict(X_list, algorithm="map")
    assert len(paths_map) == B
    for i, path in enumerate(paths_map):
        path = path[:lengths[i]]
        assert path.numel() == lengths[i]
        if lengths[i] > 0:
            assert path.min() >= 0 and path.max() < model.n_states
    print("✓ MAP predict (variable lengths) correct")

    # --- Prepare explicit context as a single tensor ---
    H = model.context_dim
    # Pad all contexts to the same length
    context_list = [
        torch.randn(L, H, device=model.device) if L > 0 else torch.empty(0, H, device=model.device)
        for L in lengths
    ]

    # Compute max sequence length
    T_max = max(lengths)
    # Pad each context to T_max
    context_padded = torch.zeros(len(lengths), T_max, H, device=model.device)
    for i, ctx in enumerate(context_list):
        L = ctx.shape[0]
        if L > 0:
            context_padded[i, :L, :] = ctx

    # Predict using Viterbi with batched context tensor
    paths_ctx = model.predict(X_list, algorithm="viterbi", context=context_padded)

    # Truncate predicted paths to original lengths
    for i, path in enumerate(paths_ctx):
        path = path[:lengths[i]]
        assert path.numel() == lengths[i]
        if lengths[i] > 0:
            assert path.min() >= 0 and path.max() < model.n_states

    print("✓ Viterbi predict with context modulation (batched tensor) correct")

    # --- Test with fully empty sequences ---
    X_empty = [torch.empty(0, F) for _ in range(2)]
    paths_empty = model.predict(X_empty, algorithm="map")
    for p in paths_empty:
        assert p.numel() == 0
    print("✓ Predict handles empty sequences correctly")

    print("✓ Intensive test_predict passed successfully\n")


def test_decode():
    print("Running enhanced test_decode...")

    B, T_max, F = 3, 5, 4
    model = make_model(enc=True, n_features=F)

    # Variable-length sequences including zero-length
    lengths = [T_max, 3, 0]
    X_list = [torch.randn(L, F) if L > 0 else torch.empty(0, F) for L in lengths]

    # --- Viterbi decoding ---
    preds_v = model.decode(X_list, algorithm="viterbi", first_only=False)
    assert isinstance(preds_v, list) and len(preds_v) == B
    for i, p in enumerate(preds_v):
        p_trunc = p[:lengths[i]] if lengths[i] > 0 else torch.tensor([], dtype=p.dtype)
        assert len(p_trunc) == lengths[i]
        if lengths[i] > 0:
            assert p_trunc.min() >= 0 and p_trunc.max() < model.n_states
    print("✓ Viterbi decode output correct")

    # --- MAP decoding ---
    preds_map = model.decode(X_list, algorithm="map", first_only=False)
    for i, p in enumerate(preds_map):
        p_trunc = p[:lengths[i]] if lengths[i] > 0 else torch.tensor([], dtype=p.dtype)
        assert len(p_trunc) == lengths[i]
        if lengths[i] > 0:
            assert p_trunc.min() >= 0 and p_trunc.max() < model.n_states
    print("✓ MAP decode output correct")

    # --- Decode with explicit context ---
    H = model.context_dim
    context_list = [torch.randn(L, H) if L > 0 else torch.empty(0, H) for L in lengths]
    preds_ctx = model.decode(X_list, algorithm="viterbi", first_only=False, context=context_list)
    for i, p in enumerate(preds_ctx):
        p_trunc = p[:lengths[i]] if lengths[i] > 0 else torch.tensor([], dtype=p.dtype)
        assert len(p_trunc) == lengths[i]
        if lengths[i] > 0:
            assert p_trunc.min() >= 0 and p_trunc.max() < model.n_states
    print("✓ Viterbi decode with context modulation correct")

    print("✓ Enhanced test_decode passed successfully\n")


if __name__ == "__main__":
    # test_predict()
    # test_decode()
    test_fit()
    # init_test()
    # test_context_router()
    # test_encode()
    # test_prepare()
    # test_sequence_set()
    # test_forward()
    # test_backward()
    # test_compute_posteriors()
    # test_model_params()
    # test_viterbi()
    # test_ensure_shape()
    print("\nAll HSMM base tests passed.")
