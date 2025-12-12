import torch
from nhsmm import utils
from nhsmm.models import HSMM
from nhsmm.context import CNN_LSTM_Encoder, ContextRouter, SequenceSet


def make_model(enc: bool = False, n_features: int = 5):
    """
    Build an HSMM instance with optional CNN+LSTM encoder.
    """
    context_dim = 32
    hidden_dim = max(context_dim, min(64, n_features * 2))  # → 32 if n_features=5

    encoder = CNN_LSTM_Encoder(
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

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 7:] = 0  # batch 1, last 3 steps padded
    mask[2, 5:] = 0  # batch 2, last 5 steps padded

    # --- no encoder ---
    m = make_model(enc=False, n_features=F)
    seq, ctx = m._encode(X, mask=mask)
    assert seq.shape == (B, T, m.context_dim)
    assert ctx.shape == (B, 1, m.context_dim)
    assert torch.all(seq == 0)
    assert torch.all(ctx == 0)
    print(f"✓ _encode (no encoder) → batch={B}, time={T}, context_dim={m.context_dim}")

    # --- with encoder ---
    m = make_model(enc=True, n_features=F)
    seq, ctx = m._encode(X, mask=mask)
    H = m.encoder.hidden_dim
    assert seq.shape == (B, T, H)
    assert ctx.shape == (B, 1, H)
    assert torch.isfinite(seq).all()
    assert torch.isfinite(ctx).all()

    # optional: check padded positions
    for b in range(B):
        pad_idx = (~mask[b]).nonzero(as_tuple=True)[0]
        if pad_idx.numel() > 0:
            assert torch.isfinite(seq[b, pad_idx]).all()

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

    print("✓ All SequenceSet tests passed")


def test_context_router():
    print("\nTest ContextRouter")

    B, T, H = 3, 5, 4  # batch, time, feature dims

    # ---------------- Fixtures ----------------
    lp1 = torch.randn(1, 10)       # 1-batch log_probs
    lpB = torch.randn(B, 10)       # B-batch log_probs
    lpBT = torch.randn(B, T)       # B,T log_probs

    mask1 = torch.ones(1, 1, 1, dtype=torch.bool)
    maskBT = torch.ones(B, T, 1, dtype=torch.bool)

    X_dummy = SequenceSet(
        sequences=torch.zeros(B, T, H),
        lengths=torch.full((B,), T, dtype=torch.long),
        masks=maskBT,
        contexts=torch.zeros(B, T, H),
        canonical=torch.zeros(B, 1, H),
    )

    # ---------------- Test 1: theta=None ----------------
    router = ContextRouter.from_tensor(None, X=X_dummy, log_probs=lpB, mask=maskBT)
    print("Test1 (theta=None, X=X_dummy):")
    assert router.canonical.shape == (B, 1, H)
    assert router.context.shape == (B, T, H)
    assert torch.all(router.mask == maskBT)
    assert router.log_probs is lpB

    # ---------------- Test 2: 1D theta ----------------
    theta1 = torch.arange(H, dtype=torch.float32)
    router = ContextRouter.from_tensor(theta1, X=X_dummy, log_probs=lpB, mask=maskBT)
    print("Test2 (theta 1D):")
    assert router.canonical.shape == (B, 1, H)
    assert router.context.shape == (B, T, H)
    assert torch.allclose(router.canonical[0], theta1.view(1, H))
    assert torch.all(router.mask == maskBT)
    assert router.log_probs is lpB

    # ---------------- Test 3: 2D theta, batch-dependent ----------------
    theta2 = torch.randn(B, H)
    router = ContextRouter.from_tensor(theta2, X=X_dummy, log_probs=lpB, mask=maskBT)
    print("Test3 (theta 2D batch-dependent):")
    assert torch.allclose(router.canonical[:, 0, :], theta2)
    assert torch.all(router.mask == maskBT)
    assert router.log_probs is lpB

    # ---------------- Test 4: 2D theta, time-dependent ----------------
    theta3 = torch.randn(T, H)
    router = ContextRouter.from_tensor(theta3, X=X_dummy, log_probs=lpB, mask=maskBT)
    print("Test4 (theta 2D time-dependent):")
    for b in range(B):
        assert torch.allclose(router.context[b], theta3)
    assert torch.all(router.mask == maskBT)
    assert router.log_probs is lpB

    # ---------------- Test 5: 3D theta ----------------
    theta4 = torch.randn(B, T, H)
    router = ContextRouter.from_tensor(theta4, X=X_dummy, log_probs=lpBT, mask=maskBT)
    print("Test5 (theta 3D):")
    assert torch.allclose(router.context, theta4)
    assert torch.all(router.mask == maskBT)
    assert router.log_probs is lpBT

    # ---------------- Test 6: Mixed canonical + time-varying ----------------
    canonical = torch.randn(B, 1, H)
    time_context = torch.randn(B, T, H)
    mixed_theta = time_context.clone()
    mixed_theta[:, 0, :] = canonical[:, 0, :]
    router = ContextRouter(canonical=canonical, context=mixed_theta,
                           log_probs=lpB, mask=maskBT)
    print("Test6 (mixed canonical + time-varying):")
    assert torch.allclose(router.context[:, 0, :], canonical[:, 0, :])
    assert torch.all(router.mask == maskBT)
    assert router.log_probs is lpB

    # ---------------- Test 7: Utilities ----------------
    router2 = router.clone()
    router3 = router.detach()
    router4 = router.to(device='cpu')
    print("Test7 (utilities):")
    assert torch.allclose(router2.context, router.context)
    assert router3.context.requires_grad is False
    assert router4.context.device.type == "cpu"
    assert torch.all(router2.mask == maskBT)
    assert torch.all(router3.mask == maskBT)
    assert torch.all(router4.mask.cpu() == maskBT)
    # log_probs propagation
    assert torch.allclose(router2.log_probs, lpB)
    assert torch.allclose(router3.log_probs, lpB)
    assert router4.log_probs.device.type == "cpu"

    print("✓ All ContextRouter tests passed with mask + log_prob validation")


def test_ensure_shape():
    m = make_model()
    T = 5
    for module in [m.initial_module, m.duration_module, m.transition_module]:
        out = m._ensure_shape(module, context=None, T=T)
        assert out.shape[0] == T
        assert torch.isfinite(out).all()
        print(f"✓ _ensure_shape {module.__class__.__name__}")


def test_forward():
    print("Test _forward")

    # Build model and data
    B, T, F = 3, 10, 2
    model = make_model(enc=True, n_features=F)
    X = make_data(B=B, T=T, F=F)

    # Variable-length mask
    mask = torch.ones(B, T, dtype=torch.bool)
    # mask[1, 7:] = 0    # batch 1 is length 7
    # mask[2, 5:] = 0    # batch 2 is length 5

    # Prepare SequenceSet
    S = model._prepare(X, mask=mask)

    print("Preparing sequences...")
    print(" S.log_probs:", S.log_probs.shape)
    print(" S.masks    :", S.masks.shape)
    print(" S.lengths  :", S.lengths)

    # Run forward
    print("Running _forward...")
    alpha = model._forward(S, theta=None)

    print(" alpha shape:", alpha.shape)
    print(" alpha dtype:", alpha.dtype)

    # ==== Basic structural checks ====
    assert isinstance(alpha, torch.Tensor)
    assert alpha.shape == (B, T, model.n_states, model.max_duration)

    # ==== Numerical validity ====
    # α is allowed to contain -inf, but never NaN
    assert not torch.isnan(alpha).any(), "Alpha contains NaNs"

    # ==== Padding checks ====
    # All padded timesteps must be exactly -inf for all states & durations
    for b in range(B):
        pad_idx = (~mask[b]).nonzero(as_tuple=True)[0]
        if pad_idx.numel() > 0:
            # Find the positions that are not -inf
            bad_pos = (alpha[b, pad_idx] != float("-inf")).nonzero(as_tuple=True)
            if bad_pos[0].numel() > 0:
                print(f"[DEBUG] Batch {b}, padded timesteps {pad_idx.tolist()} have non -inf values at positions {bad_pos}")
            assert (alpha[b, pad_idx] == float("-inf")).all(), f"Padded alpha not -inf for batch {b}"

    # ==== Valid timestep checks ====
    for b in range(B):
        L = S.lengths[b]
        for t in range(L):
            max_d = min(model.max_duration, t + 1)

            # d < max_d must be finite OR -inf only for impossible segments
            finite_or_inf = torch.isfinite(alpha[b, t, :, :max_d]) | (alpha[b, t, :, :max_d] == -float("inf"))
            assert finite_or_inf.all(), f"Invalid value at batch {b}, t={t}"

            # Must have at least one finite duration per timestep
            assert torch.isfinite(alpha[b, t, :, :max_d]).any(), f"No finite alpha at batch {b}, t={t}"

        # All durations beyond max_d must be -inf
        # if L < T:
            # last_t = L - 1
        # else:
            # last_t = T - 1
        # for t in range(L):
            # max_d = min(model.max_duration, t + 1)
            # tail = alpha[b, t, :, max_d:]
            # if tail.numel() > 0:
                # assert (tail == float("-inf")).all(), f"Durations > t+1 not -inf at batch {b}, t={t}"

    print("✓ _forward passed successfully")


def test_backward():
    print("\nTest _backward")

    # ---------------- Build model and data ----------------
    B, T, F = 3, 10, 2
    model = make_model(enc=True, n_features=F)
    X = make_data(B=B, T=T, F=F)

    # ---------------- Variable-length mask ----------------
    mask = torch.ones(B, T, dtype=torch.bool)
    # Example: uncomment to test variable-length sequences
    # mask[1, 7:] = 0  # batch 1 has length 7
    # mask[2, 5:] = 0  # batch 2 has length 5

    # ---------------- Prepare SequenceSet ----------------
    S = model._prepare(X, mask=mask)
    print(f"[Prepare] Mask shape: {S.masks.shape}")
    print(f"[Prepare] Encoded context: {S.contexts.shape}, canonical: {S.canonical.shape}")
    print(f"[Prepare] log_probs shape: {S.log_probs.shape}")

    # ---------------- Run backward pass ----------------
    beta = model._backward(S)  # [B, T, K, Dmax]
    
    # ---------------- Validate output ----------------
    assert isinstance(beta, torch.Tensor), f"Expected torch.Tensor, got {type(beta)}"
    B_out, T_out, K_out, Dmax_out = beta.shape

    expected_B = len(S.sequences)
    expected_T = max(seq.shape[0] for seq in S.sequences)
    expected_K = model.n_states
    expected_Dmax = model.max_duration

    assert B_out == expected_B, f"Batch size mismatch: {B_out} vs {expected_B}"
    assert T_out == expected_T, f"Sequence length mismatch: {T_out} vs {expected_T}"
    assert K_out == expected_K, f"Number of states mismatch: {K_out} vs {expected_K}"
    assert Dmax_out == expected_Dmax, f"Max duration mismatch: {Dmax_out} vs {expected_Dmax}"
    assert torch.isfinite(beta).all(), "Beta contains non-finite values"

    # ---------------- Optional debug ----------------
    print(f"[Backward] beta sample (batch0, timestep0): {beta[0,0]}")
    print("✓ _backward passed successfully")


def test_compute_posteriors():
    print("\nTest _compute_posteriors")

    B, T, F = 3, 10, 2  # batch, time, features
    model = make_model(enc=True, n_features=F)
    X = make_data(B=B, T=T, F=F)

    X = model._prepare(X)
    theta = None  # Optional context features

    gamma, xi, eta = model._compute_posteriors(X, theta=theta)

    B = len(X.sequences)
    T_max = max(X.lengths) if B > 0 else 0
    K = model.n_states
    Dmax = model.max_duration

    print(f"Batch size: {B}, Max sequence length: {T_max}, States: {K}, Max duration: {Dmax}")
    print(f"gamma shape: {gamma.shape}, eta shape: {eta.shape}, xi shape: {xi.shape}")

    # Check shapes
    assert gamma.shape == (B, T_max, K), f"gamma shape mismatch: {gamma.shape}"
    assert eta.shape == (B, T_max, K, Dmax), f"eta shape mismatch: {eta.shape}"
    assert xi.shape == (B, max(T_max-1,0), K, K), f"xi shape mismatch: {xi.shape}"

    # Check finite values
    assert torch.isfinite(gamma).all(), "gamma contains non-finite values"
    assert torch.isfinite(eta).all(), "eta contains non-finite values"
    assert torch.isfinite(xi).all(), "xi contains non-finite values"

    # Check masking: positions beyond actual sequence lengths should be zero
    for b, L in enumerate(X.lengths):
        if L < T_max:
            print(f"Checking padding for sequence {b}, length {L}")
            assert torch.all(gamma[b, L:] == 0), f"gamma not zero-padded correctly for sequence {b}"
            assert torch.all(eta[b, L:] == 0), f"eta not zero-padded correctly for sequence {b}"
            if L < 2:
                assert xi[b].numel() == 0 or torch.all(xi[b] == 0), f"xi not zero-padded correctly for sequence {b}"

    # Check normalization
    for b, L in enumerate(X.lengths):
        if L > 0:
            gamma_sum = gamma[b, :L].sum(-1)
            if not torch.allclose(gamma_sum, torch.ones_like(gamma_sum)):
                print(f"gamma not normalized for sequence {b}: {gamma_sum}")
            assert torch.allclose(gamma_sum, torch.ones_like(gamma_sum)), f"gamma not normalized for sequence {b}"

    print("✓ _compute_posteriors successfully")


def test_compute_posteriors_with_theta():
    print("Test _compute_posteriors with passed theta")
    
    model = make_model()
    X = make_data(B=2, T=5, F=model.n_features)
    S = model._prepare(X)
    
    # Context: list of per-sequence tensors
    theta = [torch.randn(L, model.context_dim) for L in S.lengths]
    
    gamma, xi, eta = model._compute_posteriors(S, theta=theta)
    
    # Shape and finiteness checks
    B, T_max, K, Dmax = len(S.sequences), max(S.lengths), model.n_states, model.max_duration
    assert gamma.shape == (B, T_max, K)
    assert eta.shape == (B, T_max, K, Dmax)
    assert xi.shape == (B, max(T_max-1,0), K, K)
    assert torch.isfinite(gamma).all()
    assert torch.isfinite(eta).all()
    assert torch.isfinite(xi).all()
    
    print("✓ _compute_posteriors with theta passed")


def test_model_params():
    print("Test _model_params")

    model = make_model()
    X = make_data()
    X = model._prepare(X)

    B = len(X.sequences)
    T_max = max([s.shape[0] for s in X.sequences])

    # -------------------- Estimate mode --------------------
    params_est = model._model_params(X, mode="estimate", iter_idx=0)

    for key in ["initial_dist", "duration_dist", "transition_dist"]:
        dist = params_est[key]
        assert hasattr(dist, "logits")
        assert torch.isfinite(dist.logits).all()

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

    # -------------------- Context mode --------------------
    ctx_list = [
        torch.randn(seq.shape[0], model.context_dim, device=X.device, dtype=X.dtype)
        for seq in X.sequences
    ]
    ctx_tensor = torch.nn.utils.rnn.pad_sequence(ctx_list, batch_first=True)

    theta = ContextRouter.from_tensor(
        theta=ctx_tensor,
        X=X,
        log_probs=X.log_probs,
        mask=X.masks,
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

    # -------------------- Test with encoder active --------------------
    print("\n--- Test with encoder (enc=True) ---")
    B, T, F = 3, 10, 5  # define batch, time, features
    model = make_model(enc=True, n_features=F)
    Xraw = make_data(B=B, T=T, F=F)

    # Prepare normally (this will call _encode internally)
    Xenc = model._prepare(Xraw)

    # Ensure shapes valid
    assert isinstance(Xenc, SequenceSet)
    assert Xenc.contexts is not None
    assert Xenc.contexts.shape[0] == B

    # Run params
    params_enc = model._model_params(Xenc, mode="estimate", iter_idx=3)

    # Validate same as before
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


def test_full_with_encoder():
    F = 5
    m = make_model(enc=True, n_features=F)
    X = make_data(F=F)
    S = m._prepare(X)
    alpha = m._forward(S)
    beta = m._backward(S)
    assert torch.isfinite(alpha[0]).all()
    assert torch.isfinite(beta[0]).all()
    print("✓ full HSMM pipeline (with encoder)")


def test_forward_with_encode_context():
    print("Test _forward with context from _encode")

    B, T, F = 3, 10, 5
    X = make_data(B, T, F)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 7:] = 0
    mask[2, 5:] = 0

    model = make_model(enc=True, n_features=F)
    S = model._prepare(X, mask=mask)

    # --- get context from _encode ---
    seq, ctx_canonical = model._encode(X, mask=mask)  # [B, T, H] or [B, 1, H]
    H = seq.shape[-1]

    # Make theta a list of per-sequence tensors, shape [T_i, H]
    theta = []
    for b, L in enumerate(S.lengths):
        if ctx_canonical.shape[1] == 1:
            # sequence-level context → repeat across time
            theta_b = ctx_canonical[b].repeat(L, 1)
        else:
            # time-level context → slice to sequence length
            theta_b = ctx_canonical[b, :L]
        theta.append(theta_b)

    # Forward pass
    alpha = model._forward(S, theta=theta)

    # Check shapes
    B_check = len(S.sequences)
    T_check = max(S.lengths)
    expected_shape = (B_check, T_check, model.n_states, model.max_duration)
    assert alpha.shape == expected_shape, f"Expected shape {expected_shape}, got {alpha.shape}"
    assert torch.isfinite(alpha).all(), "Alpha contains NaNs or infinities"

    print("✓ _forward with encoded context passed successfully")


if __name__ == "__main__":
    # test_context_router()
    # test_encode()
    # test_prepare()
    # test_sequence_set()
    # test_forward()
    # test_backward()
    # test_compute_posteriors()
    test_model_params()
    # test_forward_with_encode_context()
    # test_compute_posteriors_with_theta()
    # test_ensure_shape()
    # test_full_with_encoder()
    print("\nAll HSMM base tests passed.")
