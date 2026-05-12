from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


GM_ALPHA = 5e-5
GM_LAMBDA = 1e-3
GM_BETA_FEMALE = 0.080
GM_BETA_MALE = 0.090
H_SCALE = 1000.0
ROPE_BASE = 10000.0
PERIODIC_DAYS = (1.0, 7.0, 365.25)


def gompertz_makeham_H(age_years: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return (GM_ALPHA / beta) * torch.expm1(beta * age_years) + GM_LAMBDA * age_years


class GompertzRoPE(nn.Module):
    """Two-band continuous-time RoPE: H(age) on low band, raw days on high band with circadian/weekly/annual periods."""

    def __init__(self, head_dim: int):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        n_rot = head_dim // 2
        n_low = n_rot // 2
        n_high = n_rot - n_low

        low_freq = 1.0 / (ROPE_BASE ** (torch.arange(n_low, dtype=torch.float32) / max(n_low, 1)))
        periods = [PERIODIC_DAYS[i % len(PERIODIC_DAYS)] / (i // len(PERIODIC_DAYS) + 1) for i in range(n_high)]
        high_freq = 2 * math.pi / torch.tensor(periods, dtype=torch.float32)

        self.register_buffer("low_freq", low_freq, persistent=False)
        self.register_buffer("high_freq", high_freq, persistent=False)
        self.n_low = n_low
        self.n_high = n_high

    def angles(self, age_days: torch.Tensor, beta: torch.Tensor, is_static: torch.Tensor) -> torch.Tensor:
        t = age_days.float()
        b = beta.float().unsqueeze(-1) if beta.dim() == 1 else beta.float()
        H = gompertz_makeham_H(t / 365.25, b) * H_SCALE
        low = H.unsqueeze(-1) * self.low_freq.float()
        high = t.unsqueeze(-1) * self.high_freq.float()
        return torch.cat([low, high], dim=-1).masked_fill(is_static.unsqueeze(-1), 0.0)

    @staticmethod
    def apply(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        x_rot = rearrange(x, "b h s (d r) -> b h s d r", r=2)
        cos = torch.cos(angles.float()).to(x.dtype).unsqueeze(1)
        sin = torch.sin(angles.float()).to(x.dtype).unsqueeze(1)
        x0, x1 = x_rot.unbind(-1)
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return rearrange(out, "b h s d r -> b h s (d r)")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(mult * dim * 2 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class CausalRoPEAttention(nn.Module):
    def __init__(self, dim: int, heads: int, rope: GompertzRoPE, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = rope
        self.dropout = dropout

    def forward(self, x: torch.Tensor, angles: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.heads)
        q = self.rope.apply(q, angles)
        k = self.rope.apply(k, angles)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(rearrange(out, "b h s d -> b s (h d)"))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, rope: GompertzRoPE, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalRoPEAttention(dim, heads, rope, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(self, x: torch.Tensor, angles: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), angles, attn_mask)
        return x + self.mlp(self.norm2(x))


class _MAB(nn.Module):
    """Multihead attention block, set-style (no positional encoding)."""

    def __init__(self, dim: int, heads: int, mlp_mult: int = 4):
        super().__init__()
        self.heads = heads
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.norm_ff = RMSNorm(dim)
        self.ff = SwiGLU(dim, mlp_mult)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_pad: torch.Tensor | None) -> torch.Tensor:
        q = self.q_proj(self.norm_q(x))
        k, v = self.kv_proj(self.norm_kv(ctx)).chunk(2, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.heads)
        attn_mask = (~ctx_pad)[:, None, None, :] if ctx_pad is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.o_proj(rearrange(out, "b h s d -> b s (h d)"))
        return x + self.ff(self.norm_ff(x))


class SetTransformer(nn.Module):
    """SAB stack + PMA pooling to k_summary seeds. Permutation-invariant."""

    def __init__(self, dim: int, heads: int, n_blocks: int, k_summary: int, mlp_mult: int = 4):
        super().__init__()
        self.sab = nn.ModuleList([_MAB(dim, heads, mlp_mult) for _ in range(n_blocks)])
        self.seeds = nn.Parameter(torch.randn(1, k_summary, dim) * 0.02)
        self.pma = _MAB(dim, heads, mlp_mult)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        for blk in self.sab:
            x = blk(x, x, pad)
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        return self.pma(seeds, x, pad)


class AncestorEmbedding(nn.Module):
    """Sum-pooled bag embedding over OMOP concept_ancestor closures."""

    def __init__(self, n_atoms: int, dim: int, padding_idx: int = 0):
        super().__init__()
        self.bag = nn.EmbeddingBag(n_atoms, dim, mode="sum", padding_idx=padding_idx)
        nn.init.normal_(self.bag.weight, std=0.02)
        with torch.no_grad():
            self.bag.weight[padding_idx].zero_()

    def forward(self, atoms: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.bag(atoms, offsets)


@dataclass
class GenterpConfig:
    n_atoms: int = 65536
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    n_static_blocks: int = 2
    k_static_summary: int = 8
    mlp_mult: int = 4
    dropout: float = 0.0
    pad_atom_idx: int = 0


class Genterp(nn.Module):
    """Causal clinical FM: ancestor-bag tokens, Set-Transformer static prefix, Gompertz two-band RoPE."""

    def __init__(self, cfg: GenterpConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = AncestorEmbedding(cfg.n_atoms, cfg.dim, padding_idx=cfg.pad_atom_idx)
        self.static_encoder = SetTransformer(cfg.dim, cfg.n_heads, cfg.n_static_blocks, cfg.k_static_summary, cfg.mlp_mult)
        self.rope = GompertzRoPE(cfg.dim // cfg.n_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, self.rope, cfg.mlp_mult, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.n_atoms, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)
        self.register_buffer("_beta_by_sex", torch.tensor([GM_BETA_FEMALE, GM_BETA_MALE]), persistent=False)

    def _embed(self, atoms: torch.Tensor, offsets: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        return self.embed(atoms, offsets).view(*shape, self.cfg.dim)

    def _attn_mask(self, K: int, T: int, event_pad: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        S = K + T
        i = torch.arange(S, device=device).unsqueeze(1)
        j = torch.arange(S, device=device).unsqueeze(0)
        allow = ~(((j >= K) & (j > i)) | ((i < K) & (j >= K)))
        mask = allow.unsqueeze(0).unsqueeze(0)
        if event_pad is not None:
            B = event_pad.shape[0]
            pad_full = torch.zeros(B, S, dtype=torch.bool, device=device)
            pad_full[:, K:] = event_pad
            mask = mask & ~pad_full[:, None, None, :]
        return mask

    def forward(
        self,
        static_atoms: torch.Tensor,
        static_offsets: torch.Tensor,
        static_pad: torch.Tensor,
        static_shape: tuple[int, int],
        event_atoms: torch.Tensor,
        event_offsets: torch.Tensor,
        event_ages: torch.Tensor,
        event_pad: torch.Tensor,
        sex: torch.Tensor,
        return_hidden_states: bool = False,
    ):
        B, M = static_shape
        T = event_ages.shape[1]
        K = self.cfg.k_static_summary
        device = event_ages.device

        summary = self.static_encoder(self._embed(static_atoms, static_offsets, (B, M)), static_pad)
        events = self._embed(event_atoms, event_offsets, (B, T))
        x = torch.cat([summary, events], dim=1)

        beta = self._beta_by_sex[sex]
        is_static = (torch.arange(K + T, device=device) < K).expand(B, -1)
        ages_full = F.pad(event_ages, (K, 0), value=0.0)
        angles = self.rope.angles(ages_full, beta, is_static)
        mask = self._attn_mask(K, T, event_pad, device)

        hiddens: list[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x, angles, mask)
            if return_hidden_states:
                hiddens.append(x[:, K:])

        logits = self.head(self.norm(x[:, K:]))
        if return_hidden_states:
            return logits, torch.stack(hiddens, dim=1)
        return logits
