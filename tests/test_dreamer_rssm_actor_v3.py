import torch

from scripts.train_dreamer_rssm_actor_v3 import blend_with_simlingo


def test_blend_with_simlingo_gate_zero_keeps_native_control():
    base = torch.tensor([[0.25, 0.40, 0.0]])
    proposal = torch.tensor([[-0.80, 0.0, 0.90, 0.0]])

    blended = blend_with_simlingo(base, proposal)

    assert torch.allclose(blended[0, :3], base[0])
    assert blended[0, 3].item() == 0.0


def test_blend_with_simlingo_gate_one_uses_signed_longitudinal_target():
    base = torch.tensor([[0.25, 0.40, 0.0]])
    proposal = torch.tensor([[-0.50, 0.0, 0.75, 1.0]])

    blended = blend_with_simlingo(base, proposal)

    assert torch.allclose(
        blended,
        torch.tensor([[-0.50, 0.0, 0.75, 1.0]]),
    )
    assert not bool((blended[:, 1] > 0.0).logical_and(blended[:, 2] > 0.0).any())


def test_blend_with_simlingo_partial_gate_never_commands_both_pedals():
    base = torch.tensor([[0.0, 0.60, 0.0], [0.0, 0.0, 0.70]])
    proposal = torch.tensor([
        [0.30, 0.0, 0.80, 0.5],
        [-0.30, 0.90, 0.0, 0.5],
    ])

    blended = blend_with_simlingo(base, proposal)

    contradictory = (blended[:, 1] > 0.0).logical_and(blended[:, 2] > 0.0)
    assert not bool(contradictory.any())
    assert torch.all(blended[:, 0].abs() <= 1.0)
    assert torch.all((blended[:, 3] >= 0.0) & (blended[:, 3] <= 1.0))
