from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from genterp.modeling import CausalRoPEAttention, ContinuousTimeRoPE


def _dense_prefix_causal_reference(
    attn: CausalRoPEAttention,
    x: torch.Tensor,
    angles: torch.Tensor,
    event_pad: torch.Tensor,
    static_len: int,
) -> torch.Tensor:
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q = rearrange(q, "b s (h d) -> b h s d", h=attn.heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=attn.heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=attn.heads)
    q = attn.rope.rotate(q, angles)
    k = attn.rope.rotate(k, angles)

    B, _, S, _ = q.shape
    i = torch.arange(S, device=x.device).unsqueeze(1)
    j = torch.arange(S, device=x.device).unsqueeze(0)
    allow = ~(((j >= static_len) & (j > i)) | ((i < static_len) & (j >= static_len)))
    pad_full = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    pad_full[:, static_len:] = event_pad
    mask = allow.unsqueeze(0).unsqueeze(0) & ~pad_full[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn.proj(rearrange(out, "b h s d -> b s (h d)"))


def test_packed_prefix_causal_attention_matches_dense_mask_on_valid_tokens():
    torch.manual_seed(0)
    dim = 16
    heads = 4
    static_len = 3
    event_len = 6
    rope = ContinuousTimeRoPE(dim // heads)
    attn = CausalRoPEAttention(dim, heads, rope).eval()
    x = torch.randn(3, static_len + event_len, dim)
    event_pad = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [False, False, False, False, True, True],
        ]
    )
    event_ages = torch.arange(event_len, dtype=torch.float32).expand(3, event_len)
    is_static = torch.arange(static_len + event_len).lt(static_len).unsqueeze(0).expand(3, -1)
    angles = rope.angles(F.pad(event_ages, (static_len, 0), value=0.0), is_static)

    packed = attn(x, angles, event_pad, static_len)
    dense = _dense_prefix_causal_reference(attn, x, angles, event_pad, static_len)

    valid = torch.ones(3, static_len + event_len, dtype=torch.bool)
    valid[:, static_len:] = ~event_pad
    assert torch.allclose(packed[valid], dense[valid], atol=1e-5, rtol=1e-5)
    assert torch.equal(packed[~valid], torch.zeros_like(packed[~valid]))


def test_packed_attention_rejects_non_right_padded_event_mask():
    dim = 16
    heads = 4
    static_len = 3
    event_len = 4
    rope = ContinuousTimeRoPE(dim // heads)
    attn = CausalRoPEAttention(dim, heads, rope).eval()
    x = torch.randn(1, static_len + event_len, dim)
    event_pad = torch.tensor([[False, True, False, True]])
    event_ages = torch.arange(event_len, dtype=torch.float32).unsqueeze(0)
    is_static = torch.arange(static_len + event_len).lt(static_len).unsqueeze(0)
    angles = rope.angles(F.pad(event_ages, (static_len, 0), value=0.0), is_static)

    try:
        attn(x, angles, event_pad, static_len)
    except ValueError as exc:
        assert "right-padded" in str(exc)
    else:
        raise AssertionError("non-right-padded event mask should fail")
