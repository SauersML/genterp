from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

ROPE_BASE = 10000.0


@lru_cache(maxsize=1)
def _flash_attn_varlen_func():
    """Return flash_attn_varlen_func if importable, else None.

    V100 (sm_70) is below flash-attn 2.x's sm_75 minimum, so packaging it as a
    hard dep would break on Workbench. Callers must handle None by falling back
    to the SDPA path.
    """
    try:
        from flash_attn import flash_attn_varlen_func
    except ModuleNotFoundError:
        return None
    return flash_attn_varlen_func


def _assert_right_padded_event_mask(event_pad: torch.Tensor) -> None:
    if event_pad.ndim != 2:
        raise ValueError("event_pad must have shape (batch, events)")
    if event_pad.numel() == 0:
        return
    invalid = event_pad[:, :-1] & ~event_pad[:, 1:]
    if invalid.any().item():
        raise ValueError("event_pad must be right-padded: valid events cannot appear after padding")


class ContinuousTimeRoPE(nn.Module):
    """Single-band continuous-time RoPE on age in days.

    Geometric ROPE frequency basis on raw age — wavelengths span from a day
    or two up to ~10⁴ days, covering same-encounter spacing through lifetime
    ordering. No hazard warping, no calendar periodicity baked in (calendar
    effects belong in tokens, not positional encoding). Static tokens get
    zero rotation.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        n_rot = head_dim // 2
        freq = 1.0 / (ROPE_BASE ** (torch.arange(n_rot, dtype=torch.float32) / max(n_rot, 1)))
        self.register_buffer("freq", freq, persistent=False)

    def angles(self, age_days: torch.Tensor, is_static: torch.Tensor) -> torch.Tensor:
        return (age_days.float().unsqueeze(-1) * self.freq.float()).masked_fill(is_static.unsqueeze(-1), 0.0)

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
    def __init__(self, dim: int, heads: int, rope: ContinuousTimeRoPE, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = rope
        self.dropout = dropout

    def _packed_prefix_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        static_len: int,
        event_pad: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H, D = q.shape
        _assert_right_padded_event_mask(event_pad)
        valid_events = (~event_pad).sum(dim=1)
        out = q.new_zeros(B, S, H, D)
        dropout_p = self.dropout if self.training else 0.0

        if static_len > 0:
            out[:, :static_len] = F.scaled_dot_product_attention(
                q[:, :static_len].transpose(1, 2),
                k[:, :static_len].transpose(1, 2),
                v[:, :static_len].transpose(1, 2),
                dropout_p=dropout_p,
            ).transpose(1, 2)

        if valid_events.max().item() == 0:
            return out

        flash_fn = _flash_attn_varlen_func() if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) else None
        if flash_fn is not None:
            event_keep = ~event_pad
            kv_keep = torch.ones(B, S, dtype=torch.bool, device=q.device)
            kv_keep[:, static_len:] = event_keep
            q_lens = valid_events.to(torch.int32)
            kv_lens = q_lens + static_len
            q_packed = q[:, static_len:][event_keep].contiguous()
            k_packed = k[kv_keep].contiguous()
            v_packed = v[kv_keep].contiguous()
            cu_q = F.pad(q_lens.cumsum(0), (1, 0))
            cu_kv = F.pad(kv_lens.cumsum(0), (1, 0))
            event_out = flash_fn(
                q_packed,
                k_packed,
                v_packed,
                cu_q,
                cu_kv,
                int(q_lens.max().item()),
                int(kv_lens.max().item()),
                dropout_p=dropout_p,
                causal=True,
            )
            out[:, static_len:][event_keep] = event_out
            return out

        for event_len_t in valid_events.unique(sorted=True).tolist():
            event_len = int(event_len_t)
            if event_len == 0:
                continue
            kv_len = static_len + event_len
            rows = valid_events == event_len
            qe = q[rows, static_len : static_len + event_len].transpose(1, 2)
            ke = k[rows, :kv_len].transpose(1, 2)
            ve = v[rows, :kv_len].transpose(1, 2)
            q_idx = torch.arange(static_len, kv_len, device=q.device).view(event_len, 1)
            kv_idx = torch.arange(kv_len, device=q.device).view(1, kv_len)
            mask = (kv_idx <= q_idx).view(1, 1, event_len, kv_len)
            out[rows, static_len:kv_len] = F.scaled_dot_product_attention(
                qe,
                ke,
                ve,
                attn_mask=mask,
                dropout_p=dropout_p,
            ).transpose(1, 2)
        return out

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        event_pad: torch.Tensor,
        static_len: int,
    ) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.heads)
        q = self.rope.apply(q.transpose(1, 2), angles).transpose(1, 2)
        k = self.rope.apply(k.transpose(1, 2), angles).transpose(1, 2)
        out = self._packed_prefix_causal_attention(q, k, v, static_len, event_pad)
        return self.proj(rearrange(out, "b s h d -> b s (h d)"))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, rope: ContinuousTimeRoPE, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalRoPEAttention(dim, heads, rope, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        event_pad: torch.Tensor,
        static_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (post_block_residual, pre_mlp_residual, mlp_additive_output)."""
        x = x + self.attn(self.norm1(x), angles, event_pad, static_len)
        mlp_out = self.mlp(self.norm2(x))
        return x + mlp_out, x, mlp_out


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


class AtomEmbedding(nn.Module):
    """Atom embedding for collapsed OMOP event codes."""

    def __init__(self, n_atoms: int, dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_atoms, dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        return self.embedding(atoms)


def _log1p_signed(z: torch.Tensor) -> torch.Tensor:
    return z.sign() * torch.log1p(z.abs())


def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    direct = torch.log((0.5 * torch.erfc(-x / math.sqrt(2.0))).clamp_min(torch.finfo(x.dtype).tiny))
    tail = -0.5 * x.pow(2) - torch.log((-x).clamp_min(1e-12)) - 0.5 * math.log(2 * math.pi)
    return torch.where(x > -10.0, direct, tail)


class ValueModulator(nn.Module):
    """Per-event multiplicative value modulation, conditioned on the concept itself.

      e_token = e_concept * (tanh(MLP([log1p_signed(z), e_concept])) if has_magnitude else 1)

    z = (value - μ[leaf_atom]) / σ[leaf_atom]; has_magnitude is the atom-level
    flag AND value-finiteness. The MLP is conditioned on the concept embedding,
    so the same z gets different modulation patterns for, e.g., HbA1c vs sodium.
    (μ, σ, atom_has_mag) come from the ETL value_stats.json. Last-layer bias = 2
    → tanh ≈ 0.96 at init so magnitude events start near identity multiplication.
    """

    def __init__(self, dim: int, n_atoms: int, hidden: int = 64):
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(1 + dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.normal_(self.value_mlp[-1].weight, std=0.02)
        nn.init.constant_(self.value_mlp[-1].bias, 2.0)
        self.register_buffer("value_mu", torch.zeros(n_atoms))
        self.register_buffer("value_sigma", torch.ones(n_atoms))
        self.register_buffer("atom_has_mag", torch.zeros(n_atoms, dtype=torch.bool))

    @torch.no_grad()
    def set_stats(self, value_mu: torch.Tensor, value_sigma: torch.Tensor, atom_has_mag: torch.Tensor) -> None:
        self.value_mu.copy_(value_mu.to(self.value_mu.dtype))
        self.value_sigma.copy_(value_sigma.to(self.value_sigma.dtype).clamp(min=1e-6))
        self.atom_has_mag.copy_(atom_has_mag.to(torch.bool))

    def event_has_magnitude(self, leaf_atom: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.atom_has_mag[leaf_atom] & torch.isfinite(value)

    def z_score(self, leaf_atom: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        mu = self.value_mu[leaf_atom]
        sigma = self.value_sigma[leaf_atom].clamp(min=1e-6)
        z = (value.float() - mu) / sigma
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return _log1p_signed(z)

    def forward(self, e_concept: torch.Tensor, leaf_atom: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        has_mag = self.event_has_magnitude(leaf_atom, value)
        z = self.z_score(leaf_atom, value).unsqueeze(-1)
        mlp_input = torch.cat([z, e_concept.float()], dim=-1)
        modulation = torch.tanh(self.value_mlp(mlp_input)).to(e_concept.dtype)
        return torch.where(has_mag.unsqueeze(-1), e_concept * modulation, e_concept)


class ValueHead(nn.Module):
    """Gaussian over log1p-signed z, conditioned on (hidden, predicted-leaf-atom embedding)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def params(self, hidden: torch.Tensor, concept_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.net(torch.cat([hidden, concept_emb], dim=-1)).chunk(2, dim=-1)
        return mu.squeeze(-1), log_sigma.squeeze(-1).clamp(-5.0, 5.0)

    def nll(self, hidden: torch.Tensor, concept_emb: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self.params(hidden, concept_emb)
        return 0.5 * ((z_target - mu) * (-log_sigma).exp()).pow(2) + log_sigma + 0.5 * math.log(2 * math.pi)

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor, concept_emb: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        mu, log_sigma = self.params(hidden, concept_emb)
        return mu + log_sigma.exp() * torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)


class MarkedTPPHead(nn.Module):
    """Marked temporal point process: log-normal mixture for Δt, conditional softmax mark|Δt.

    p(Δt | h) = Σ_k w_k(h) · LogNormal(Δt; μ_k(h), σ_k(h))
    p(m  | h, Δt) = softmax(E @ (W_h h + W_φ φ(Δt)))

    φ(Δt) is a fixed log-spaced sinusoidal embedding of Δt.
    """

    def __init__(
        self,
        dim: int,
        n_marks: int,
        mark_weight: nn.Parameter,
        n_mix: int = 8,
        time_dim: int = 32,
        sampled_mark_negatives: int = 4096,
    ):
        super().__init__()
        assert time_dim % 2 == 0
        if mark_weight.shape != (n_marks, dim):
            raise ValueError("mark_weight must have shape (n_marks, dim)")
        self.n_marks = n_marks
        self.n_mix = n_mix
        self.sampled_mark_negatives = sampled_mark_negatives
        self.time_proj = nn.Linear(dim, 3 * n_mix)
        self.mark_h_proj = nn.Linear(dim, dim, bias=False)
        self.mark_time_proj = nn.Linear(time_dim, dim, bias=False)
        self.mark_out = nn.Linear(dim, n_marks, bias=False)
        self.mark_out.weight = mark_weight
        nn.init.normal_(self.mark_h_proj.weight, std=0.02)
        nn.init.normal_(self.mark_time_proj.weight, std=0.02)
        freqs = torch.exp(torch.linspace(math.log(0.01), math.log(100.0), time_dim // 2))
        self.register_buffer("time_freqs", freqs, persistent=False)
        mark_noise_probs = torch.ones(n_marks, dtype=torch.float32)
        mark_noise_probs[0] = 0.0
        self.register_buffer("mark_noise_probs", mark_noise_probs / mark_noise_probs.sum().clamp(min=1.0))

    @torch.no_grad()
    def set_mark_noise_distribution(self, counts: torch.Tensor) -> None:
        if counts.shape != (self.n_marks,):
            raise ValueError(f"counts must have shape ({self.n_marks},)")
        probs = counts.to(device=self.mark_noise_probs.device, dtype=self.mark_noise_probs.dtype).clamp(min=0)
        probs[0] = 0.0
        total = probs.sum()
        if total <= 0:
            raise ValueError("mark noise distribution has no non-PAD mass")
        self.mark_noise_probs.copy_(probs / total)

    def _phi(self, delta_t: torch.Tensor) -> torch.Tensor:
        log_dt = (delta_t.clamp(min=1e-6)).log().unsqueeze(-1)
        phases = log_dt * self.time_freqs
        return torch.cat([phases.sin(), phases.cos()], dim=-1)

    def time_params(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_w_raw, mu, log_sigma = self.time_logits(hidden).chunk(3, dim=-1)
        return log_w_raw.log_softmax(dim=-1), mu, log_sigma.clamp(-5.0, 5.0)

    def time_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.time_proj(hidden)

    def mark_features(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        phi = self._phi(delta_t)
        return self.mark_h_proj(hidden) + self.mark_time_proj(phi)

    def mark_log_probs(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.mark_logits(hidden, delta_t).log_softmax(dim=-1)

    def mark_logits(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.mark_out(self.mark_features(hidden, delta_t))

    def sampled_mark_nll(self, hidden: torch.Tensor, delta_t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.numel() == 0:
            return hidden.sum() * 0.0
        features = self.mark_features(hidden, delta_t)
        k = min(self.sampled_mark_negatives, max(self.n_marks - 1, 1))
        negatives = torch.multinomial(self.mark_noise_probs, k, replacement=False)
        weight = self.mark_out.weight

        target = target.long()
        target_logits = (features * weight[target]).sum(dim=-1, keepdim=True)
        negative_logits = features @ weight[negatives].T
        negative_logits = negative_logits.masked_fill(negatives.unsqueeze(0) == target.unsqueeze(1), float("-inf"))

        q_neg = self.mark_noise_probs[negatives].clamp_min(torch.finfo(self.mark_noise_probs.dtype).tiny)
        negative_logits = negative_logits - (k * q_neg).log().to(negative_logits.dtype)
        logits = torch.cat([target_logits, negative_logits], dim=-1).float()
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels, reduction="sum")

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        log_w, mu, log_sigma = self.time_params(hidden)
        k = torch.distributions.Categorical(logits=log_w).sample()
        mu_k = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sigma_k = log_sigma.gather(-1, k.unsqueeze(-1)).squeeze(-1).exp()
        noise = torch.randn(mu_k.shape, generator=generator, device=mu_k.device, dtype=mu_k.dtype)
        delta_t = (mu_k + sigma_k * noise).exp()
        mark = torch.distributions.Categorical(logits=self.mark_log_probs(hidden, delta_t)).sample()
        return delta_t, mark


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
    n_time_mix: int = 8
    time_phi_dim: int = 32
    value_mlp_hidden: int = 64
    value_head_hidden: int = 64
    sampled_mark_negatives: int = 4096


class Genterp(nn.Module):
    """Clinical FM: collapsed atom tokens, value-modulated, Set-Transformer static prefix, continuous-time RoPE, marked-TPP head, gaussian value head."""

    def __init__(self, cfg: GenterpConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = AtomEmbedding(cfg.n_atoms, cfg.dim, padding_idx=cfg.pad_atom_idx)
        self.value_mod = ValueModulator(cfg.dim, cfg.n_atoms, cfg.value_mlp_hidden)
        self.static_encoder = SetTransformer(cfg.dim, cfg.n_heads, cfg.n_static_blocks, cfg.k_static_summary, cfg.mlp_mult)
        self.rope = ContinuousTimeRoPE(cfg.dim // cfg.n_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, self.rope, cfg.mlp_mult, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.dim)
        self.tpp = MarkedTPPHead(
            cfg.dim,
            cfg.n_atoms,
            self.embed.embedding.weight,
            cfg.n_time_mix,
            cfg.time_phi_dim,
            cfg.sampled_mark_negatives,
        )
        self.value_head = ValueHead(cfg.dim, cfg.value_head_hidden)
        self._static_mask_cache: dict[tuple[int, int, str, int | None], torch.Tensor] = {}
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = checkpoint

    def _embed(self, atoms: torch.Tensor) -> torch.Tensor:
        return self.embed(atoms)

    @staticmethod
    def _device_key(device: torch.device) -> tuple[str, int | None]:
        return device.type, device.index

    def _static_mask(self, B: int, K: int, T: int, device: torch.device) -> torch.Tensor:
        cache_key = (K, T, *self._device_key(device))
        cached = self._static_mask_cache.get(cache_key)
        if cached is None:
            cached = torch.arange(K + T, device=device).lt(K).unsqueeze(0)
            self._static_mask_cache[cache_key] = cached
        return cached.expand(B, -1)

    def forward(
        self,
        static_atoms: torch.Tensor,
        static_pad: torch.Tensor,
        event_atoms: torch.Tensor,
        event_ages: torch.Tensor,
        event_pad: torch.Tensor,
        target_atoms: torch.Tensor,
        event_values: torch.Tensor,
        return_transcoder_acts: bool = False,
        **loss_only_kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del loss_only_kwargs  # censor_age + sex + other loss-only or external fields ride the same batch dict
        B = static_atoms.shape[0]
        T = event_ages.shape[1]
        K = self.cfg.k_static_summary
        device = event_ages.device

        summary = self.static_encoder(self._embed(static_atoms), static_pad)
        events = self._embed(event_atoms)
        events = self.value_mod(events, target_atoms, event_values)
        x = torch.cat([summary, events], dim=1)

        is_static = self._static_mask(B, K, T, device)
        ages_full = F.pad(event_ages, (K, 0), value=0.0)
        angles = self.rope.angles(ages_full, is_static)

        pre_mlps: list[torch.Tensor] = []
        mlp_outs: list[torch.Tensor] = []
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, pre_mlp, mlp_out = self._gradient_checkpointing_func(blk, x, angles, event_pad, K)
            else:
                x, pre_mlp, mlp_out = blk(x, angles, event_pad, K)
            if return_transcoder_acts:
                pre_mlps.append(pre_mlp[:, K:])
                mlp_outs.append(mlp_out[:, K:])

        hidden = self.norm(x[:, K:])
        out: dict[str, torch.Tensor] = {"hidden": hidden}
        if return_transcoder_acts:
            out["pre_mlp"] = torch.stack(pre_mlps, dim=1)
            out["mlp_out"] = torch.stack(mlp_outs, dim=1)
        return out

    def loss(
        self,
        event_ages: torch.Tensor,
        target_atoms: torch.Tensor,
        event_pad: torch.Tensor,
        censor_age: torch.Tensor,
        event_values: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            event_ages=event_ages,
            event_pad=event_pad,
            target_atoms=target_atoms,
            event_values=event_values,
            **batch,
        )
        return marked_tpp_value_loss(
            self.tpp,
            self.value_mod,
            self.value_head,
            self.embed.weight,
            out["hidden"],
            event_ages,
            target_atoms,
            event_values,
            event_pad,
            censor_age,
        )


def marked_tpp_value_loss(
    tpp: MarkedTPPHead,
    value_mod: ValueModulator,
    value_head: ValueHead,
    atom_embedding: torch.Tensor,
    hidden: torch.Tensor,
    event_ages: torch.Tensor,
    target_atoms: torch.Tensor,
    event_values: torch.Tensor,
    event_pad: torch.Tensor,
    censor_age: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Joint NLL: marked-TPP (time + mark) + value (z-space gaussian) + right-censoring.

    Real next event at t:    -log p(Δt|h_t) - log p(m|h_t, Δt) - has_mag · log p(z|h_t, e_m)
    Censoring:               -log S(Δt_c|h_last)
    """
    h_pred = hidden[:, :-1]
    delta_real = event_ages[:, 1:] - event_ages[:, :-1]
    delta_censor = censor_age.unsqueeze(-1) - event_ages[:, :-1]
    target_real = target_atoms[:, 1:].clamp(min=0)
    value_real = event_values[:, 1:]
    real_mask = (~event_pad[:, :-1]) & (~event_pad[:, 1:])
    censor_mask = (~event_pad[:, :-1]) & event_pad[:, 1:]
    any_mask = real_mask | censor_mask
    n_real = real_mask.sum()
    n_censor = censor_mask.sum()

    delta_any = torch.where(real_mask, delta_real, delta_censor).clamp(min=1e-6)[any_mask]
    log_w, mu, log_sigma = tpp.time_params(h_pred[any_mask])
    log_dt = delta_any.log().unsqueeze(-1)
    inv_sigma = (-log_sigma).exp()
    log_pdf = -log_dt - log_sigma - 0.5 * math.log(2 * math.pi) - 0.5 * ((log_dt - mu) * inv_sigma).pow(2)
    time_lp = torch.logsumexp(log_w + log_pdf, dim=-1)
    z = (log_dt - mu) * inv_sigma
    time_ls = torch.logsumexp(log_w + _log_ndtr(-z), dim=-1)

    real_any = real_mask[any_mask]
    real_time_lp = time_lp[real_any]
    censor_time_ls = time_ls[~real_any]

    mag_mask = real_mask & value_mod.event_has_magnitude(target_real, value_real)
    n_mag = mag_mask.sum()

    delta_mark = delta_real.clamp(min=1e-6)[real_mask]
    if tpp.training:
        mark_loss = tpp.sampled_mark_nll(h_pred[real_mask], delta_mark, target_real[real_mask])
    else:
        mark_log_probs = tpp.mark_log_probs(h_pred[real_mask], delta_mark)
        mark_lp = torch.gather(mark_log_probs, -1, target_real[real_mask].unsqueeze(-1)).squeeze(-1)
        mark_loss = -mark_lp.sum()

    leaf_mag = target_real[mag_mask]
    z_target = value_mod.z_score(leaf_mag, value_real[mag_mask])
    value_nll_tokens = value_head.nll(h_pred[mag_mask], atom_embedding[leaf_mag], z_target)
    value_loss = value_nll_tokens.sum()

    time_loss = -real_time_lp.sum()
    censor_loss = -censor_time_ls.sum()
    time_nll = time_loss / n_real.clamp(min=1)
    mark_nll = mark_loss / n_real.clamp(min=1)
    value_nll = value_loss / n_mag.clamp(min=1)
    censor_nll = censor_loss / n_censor.clamp(min=1)
    # Each component is averaged over its own token population (n_real, n_real, n_mag, n_censor)
    # before summing, so each per-token NLL contributes equally to the gradient. Pooling the raw
    # sums over n_any would heavily underweight value (n_mag << n_real) and censor (one per subject).
    total = time_nll + mark_nll + value_nll + censor_nll

    return {
        "loss": total,
        "time_nll": time_nll,
        "mark_nll": mark_nll,
        "value_nll": value_nll,
        "censor_nll": censor_nll,
        "n_real": n_real,
        "n_censor": n_censor,
        "n_mag": n_mag,
    }
