import torch

from genterp.modeling import ContinuousTimeRoPE


def test_frequency_basis_matches_head_dim():
    rope = ContinuousTimeRoPE(head_dim=64)

    assert rope.freq.shape == (64 // 2,)
    assert torch.all(rope.freq > 0)
    assert torch.all(rope.freq[:-1] >= rope.freq[1:])


def test_static_tokens_have_zero_rotation():
    rope = ContinuousTimeRoPE(head_dim=16)
    age_days = torch.tensor([[0.0, 365.25, 3652.5]])
    is_static = torch.tensor([[True, False, False]])

    angles = rope.angles(age_days, is_static)

    assert angles.shape == (1, 3, 16 // 2)
    assert torch.equal(angles[:, 0], torch.zeros_like(angles[:, 0]))
    assert torch.isfinite(angles[:, 1:]).all()
