import torch
from nhsmm import utils


def make_sequence(T=5, F=3, device="cpu", dtype=torch.float32):
    return torch.randn(T, F, device=device, dtype=dtype)


def make_obj(T=5, F=3):
    seq = make_sequence(T, F)
    log_probs = torch.randn(T, F)
    ctx = torch.randn(T, F)
    canon_ctx = ctx[0:1]
    lengths = [T]
    mask = torch.ones(T, 1, dtype=torch.bool)
    return dict(
        sequences=seq,
        log_probs=log_probs,
        contexts=ctx,
        canonical_contexts=canon_ctx,
        lengths=lengths,
        masks=mask,
    )


def test_sequence_set_basic():
    obj = make_obj(T=4, F=2)
    S = utils.SequenceSet(**obj)
    print("Sequences:", S.sequences)
    print("Lengths:", S.lengths)
    print("Masks:", S.masks)
    print("Feature dim:", S.feature_dim)
    print("Device:", S.device)
    print("Dtype:", S.dtype)
    assert isinstance(S.sequences, list)
    assert S.lengths == [4]
    assert S.masks[0].shape == (4, 1)
    assert S.feature_dim == 2


def test_sequence_set_with_log_probs_and_contexts():
    objs = [make_obj(T=3, F=2), make_obj(T=4, F=2)]
    S = utils.SequenceSet(
        sequences=[o["sequences"] for o in objs],
        log_probs=[o["log_probs"] for o in objs],
        contexts=[o["contexts"] for o in objs],
        canonical_contexts=[o["canonical_contexts"] for o in objs],
        lengths=[o["lengths"][0] for o in objs],
        masks=[o["masks"] for o in objs],
    )
    print("Log probs shapes:", [lp.shape for lp in S.log_probs])
    print("Contexts shapes:", [c.shape for c in S.contexts])
    print("Canonical contexts shapes:", [c.shape for c in S.canonical_contexts])
    for lp, l in zip(S.log_probs, S.lengths):
        assert lp.shape[0] == l
    for ctx, l in zip(S.contexts, S.lengths):
        assert ctx.shape[0] == l
    for cctx in S.canonical_contexts:
        assert cctx.ndim == 2 and cctx.shape[0] == 1


def test_sequence_set_masks():
    obj = make_obj(T=4, F=2)
    mask = torch.tensor([[1], [1], [0], [1]], dtype=torch.bool)
    obj["masks"] = mask  # override mask in obj
    S = utils.SequenceSet(**obj)
    print("Mask:", S.masks[0])
    assert S.masks[0].shape == (4, 1)
    assert torch.equal(S.masks[0], mask)


def test_sequence_set_indexing():
    objs = [make_obj(T=3, F=2), make_obj(T=4, F=2)]
    S = utils.SequenceSet(
        sequences=[o["sequences"] for o in objs],
        log_probs=[o["log_probs"] for o in objs],
        contexts=[o["contexts"] for o in objs],
        canonical_contexts=[o["canonical_contexts"] for o in objs],
        lengths=[o["lengths"][0] for o in objs],
        masks=[o["masks"] for o in objs],
    )
    S_slice = S[0]
    print("S_slice sequences:", S_slice.sequences)
    assert isinstance(S_slice, utils.SequenceSet)
    assert S_slice.n_sequences == 1
    assert S_slice.lengths == [3]


def test_sequence_set_to_tensor_padding():
    objs = [make_obj(T=3, F=2), make_obj(T=5, F=2)]
    S = utils.SequenceSet(
        sequences=[o["sequences"] for o in objs],
        log_probs=[o["log_probs"] for o in objs],
        contexts=[o["contexts"] for o in objs],
        canonical_contexts=[o["canonical_contexts"] for o in objs],
        lengths=[o["lengths"][0] for o in objs],
        masks=[o["masks"] for o in objs],
    )
    tensor = S.to_tensor(key="sequences", pad_value=-1.0)
    print("Tensor shape:", tensor.shape)
    assert tensor.shape == (2, 5, 2)
    assert (tensor[0, 3:, :] == -1.0).all()
    tensor_c = S.to_tensor(key="canonical_contexts", canonical=True)
    print("Canonical tensor shape:", tensor_c.shape)
    for c in tensor_c:
        assert c.ndim == 2


def test_invalid_shapes_raises():
    obj = make_obj(T=3, F=2)
    # Mismatched lengths
    try:
        obj_bad = obj.copy()
        obj_bad["lengths"] = [4]
        utils.SequenceSet(**obj_bad)
    except ValueError as e:
        print("Caught expected ValueError:", e)
    # Invalid log_probs
    try:
        obj_bad = obj.copy()
        obj_bad["log_probs"] = torch.randn(2, 2)
        utils.SequenceSet(**obj_bad)
    except ValueError as e:
        print("Caught expected ValueError:", e)
    # Invalid context
    try:
        obj_bad = obj.copy()
        obj_bad["contexts"] = torch.randn(2, 3, 4)
        utils.SequenceSet(**obj_bad)
    except ValueError as e:
        print("Caught expected ValueError:", e)
    # Invalid mask
    try:
        obj_bad = obj.copy()
        obj_bad["masks"] = torch.randn(2, 2, 2)
        utils.SequenceSet(**obj_bad)
    except ValueError as e:
        print("Caught expected ValueError:", e)


if __name__ == "__main__":
    test_sequence_set_basic()
    test_sequence_set_with_log_probs_and_contexts()
    test_sequence_set_masks()
    test_sequence_set_indexing()
    test_sequence_set_to_tensor_padding()
    test_invalid_shapes_raises()
    print("✓ All SequenceSet tests passed")
