import torch

from textedit_accel.dynamic_resolution import build_plan, pack_tokens, restore_tokens


def test_mixed_resolution_pack_and_restore():
    tokens = torch.arange(4 * 4, dtype=torch.float32).view(1, 4, 4, 1)
    edit = torch.zeros((4, 4), dtype=torch.bool)
    edit[0, 0] = True
    plan = build_plan(edit, coarse_factor=2, uniform_stride=0)

    packed = pack_tokens(tokens, plan)
    restored = restore_tokens(packed)

    # Selected 2x2 block remains lossless.
    assert torch.equal(restored[:, :2, :2], tokens[:, :2, :2])
    # Three unselected 2x2 blocks use one token each: 4 + 3 packed tokens.
    assert packed.tokens.shape == (1, 7, 1)
    assert restored[0, 2, 2, 0] == tokens[0, 2:4, 2:4, 0].mean()


def test_estimated_token_count_matches_packing():
    tokens = torch.randn(1, 5, 7, 3)
    edit = torch.zeros((5, 7), dtype=torch.bool)
    edit[4, 6] = True
    plan = build_plan(edit, coarse_factor=2, uniform_stride=0)

    assert plan.estimated_token_count == pack_tokens(tokens, plan).tokens.shape[1]
