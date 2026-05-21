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
    if event_pad.is_cuda:
        return
    invalid = event_pad[:, :-1] & ~event_pad[:, 1:]
    if invalid.any().item():
        raise ValueError("event_pad must be right-padded: valid events cannot appear after padding")


def _length_list(event_pad: torch.Tensor, event_lengths: torch.Tensor | list[int] | tuple[int, ...] | None) -> list[int]:
    if isinstance(event_lengths, (list, tuple)):
        return [int(length) for length in event_lengths]
    if event_lengths is not None:
        return [int(length) for length in event_lengths.detach().cpu().tolist()]
    return [int(length) for length in (~event_pad).sum(dim=1).detach().cpu().tolist()]


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
    def rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
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
        event_lengths: torch.Tensor | list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:
        B, S, H, D = q.shape
        _assert_right_padded_event_mask(event_pad)
        valid_event_lengths = _length_list(event_pad, event_lengths)
        out = q.new_zeros(B, S, H, D)
        dropout_p = self.dropout if self.training else 0.0

        if static_len > 0:
            out[:, :static_len] = F.scaled_dot_product_attention(
                q[:, :static_len].transpose(1, 2),
                k[:, :static_len].transpose(1, 2),
                v[:, :static_len].transpose(1, 2),
                dropout_p=dropout_p,
            ).transpose(1, 2)

        max_event_len = max(valid_event_lengths, default=0)
        if max_event_len == 0:
            return out

        flash_fn = None
        if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            flash_fn = _flash_attn_varlen_func()
        if flash_fn is not None:
            event_keep = ~event_pad
            kv_keep = torch.ones(B, S, dtype=torch.bool, device=q.device)
            kv_keep[:, static_len:] = event_keep
            q_lens = torch.tensor(valid_event_lengths, dtype=torch.int32, device=q.device)
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
                max_event_len,
                max_event_len + static_len,
                dropout_p=dropout_p,
                causal=True,
            )
            out[:, static_len:][event_keep] = event_out
            return out

        for event_len in sorted(set(valid_event_lengths)):
            if event_len == 0:
                continue
            kv_len = static_len + event_len
            rows = torch.tensor(
                [length == event_len for length in valid_event_lengths],
                device=q.device,
                dtype=torch.bool,
            )
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

    def _qkv_rope(self, x: torch.Tensor, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projection + rope. Returns ``(q, k, v)`` each of shape (B, S, H, D).

        Extracted so the decode pathway (genterp.decode) can reuse identical
        parameters and rope semantics without re-implementing the linear/rope
        stack. Training-mode forward calls this then runs packed varlen
        attention; decode-mode init calls this and stashes (k, v); decode-mode
        step calls this for the single new token and appends (k, v) to the
        KV cache.
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.heads)
        q = self.rope.rotate(q.transpose(1, 2), angles).transpose(1, 2)
        k = self.rope.rotate(k.transpose(1, 2), angles).transpose(1, 2)
        return q, k, v

    def _project_out(self, out: torch.Tensor) -> torch.Tensor:
        """Final output projection. ``out`` is (B, S, H, D)."""
        return self.proj(rearrange(out, "b s h d -> b s (h d)"))

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        event_pad: torch.Tensor,
        static_len: int,
        event_lengths: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        q, k, v = self._qkv_rope(x, angles)
        out = self._packed_prefix_causal_attention(q, k, v, static_len, event_pad, event_lengths)
        return self._project_out(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        rope: ContinuousTimeRoPE,
        n_layers: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        residual_scale: float | None = None,
    ):
        super().__init__()
        # residual_scale=None preserves legacy behavior (no scaling, ≡ 1.0) so
        # checkpoints trained before the depth-scaled residual landed continue
        # producing the same forward outputs after the upgrade. Set to a
        # number (e.g., (2*n_layers)**-0.5) on fresh runs to opt into the
        # depth-stable residual scheme.
        self.res_scale = (
            float(residual_scale)
            if residual_scale is not None
            else 1.0
        )
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
        event_lengths: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (post_block_residual, pre_mlp_residual, mlp_additive_output)."""
        x = x + self.res_scale * self.attn(self.norm1(x), angles, event_pad, static_len, event_lengths)
        mlp_out = self.mlp(self.norm2(x))
        return x + self.res_scale * mlp_out, x, mlp_out


class _MAB(nn.Module):
    """Multihead attention block, set-style (no positional encoding)."""

    def __init__(self, dim: int, heads: int, residual_blocks: int, mlp_mult: int = 4, residual_scale: float | None = None):
        super().__init__()
        self.heads = heads
        # Same legacy-default semantics as ``Block``: ``residual_scale=None``
        # means "no scaling" (≡ 1.0) so warm-started checkpoints continue
        # producing the same forward outputs as before the depth-scaled
        # residual scheme was added. Opt in by passing a number.
        self.res_scale = (
            float(residual_scale)
            if residual_scale is not None
            else 1.0
        )
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
        x = x + self.res_scale * self.o_proj(rearrange(out, "b h s d -> b s (h d)"))
        return x + self.res_scale * self.ff(self.norm_ff(x))


class SetTransformer(nn.Module):
    """SAB stack + PMA pooling to k_summary seeds. Permutation-invariant."""

    def __init__(self, dim: int, heads: int, n_blocks: int, k_summary: int, mlp_mult: int = 4):
        super().__init__()
        residual_blocks = n_blocks + 1
        self.sab = nn.ModuleList([_MAB(dim, heads, residual_blocks, mlp_mult) for _ in range(n_blocks)])
        self.seeds = nn.Parameter(torch.randn(1, k_summary, dim) * 0.02)
        self.pma = _MAB(dim, heads, residual_blocks, mlp_mult)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        for blk in self.sab:
            x = blk(x, x, pad)
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        return self.pma(seeds, x, pad)


class AtomEmbedding(nn.Module):
    """Atom embedding for OMOP event codes, optionally with ancestor-sum hierarchy.

    Flat mode (default, identical to old behavior): ``embed(atom) = self.embedding(atom)``.

    Hierarchical mode (after ``set_ancestors`` has been called with a non-empty
    table): ``embed(atom) = self.embedding(atom) + Σ_a∈ancestors(atom) self.ancestor_embedding(a)``.
    Rare leaf concepts share parameters through their SNOMED IS-A ancestors so a
    code with only ~50 patient hits still gets a meaningful representation via
    its broader ancestors, while common codes can override with their own leaf
    vector. The output projection in :class:`MarkedTPPHead` consumes
    :attr:`effective_weight` (computed on the fly), so input-side richness flows
    through to the mark logits without breaking weight tying.

    Warm-start: ancestor_embedding starts at zero, so on the very first call
    after ``set_ancestors`` the model is bit-identical to the flat-embedding
    checkpoint it loaded from — gradient pressure then learns the ancestor
    components. The original ``embedding`` Parameter path is preserved exactly
    (same module name, same shape) so existing state-dicts deserialize cleanly.
    """

    def __init__(
        self,
        n_atoms: int,
        dim: int,
        padding_idx: int = 0,
        n_ancestor_rows: int = 0,
        n_ancestor_cols: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(n_atoms, dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        # ancestor_embedding shape is fixed at construction so checkpoints
        # round-trip cleanly. n_ancestor_rows = 0 means "flat mode": a 1-row
        # zero placeholder that never gets indexed.
        rows = max(n_ancestor_rows + 1, 1)
        self.ancestor_embedding = nn.Embedding(rows, dim, padding_idx=0)
        nn.init.zeros_(self.ancestor_embedding.weight)
        # Persist the attached closure so resumed checkpoints keep the exact
        # hierarchy table that trained the ancestor embeddings.
        self.register_buffer(
            "ancestor_ids",
            torch.zeros(n_atoms, n_ancestor_cols, dtype=torch.long),
            persistent=True,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Reshape ``ancestor_ids`` to match the saved closure before the default copy.

        ``ancestor_ids`` is registered with width 0 at construction (the real
        width comes from the ETL ``ancestors.npz`` and varies per build). The
        default ``_load_from_state_dict`` does an in-place ``copy_``, which
        rejects any shape mismatch — so HF's ``from_pretrained`` would otherwise
        fail every time the saved closure had width > 0. Swap in a same-shape
        zero buffer here, then let the base class copy the saved tensor into
        it. ``ancestor_embedding`` is intentionally fixed-shape (Parameter) and
        is handled by the default path.
        """
        key = prefix + "ancestor_ids"
        saved = state_dict.get(key)
        if saved is not None and saved.shape != self.ancestor_ids.shape:
            self.ancestor_ids = torch.zeros(
                *saved.shape, dtype=self.ancestor_ids.dtype, device=self.ancestor_ids.device
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def set_ancestor_ids(self, ancestor_ids: torch.Tensor) -> None:
        """Attach the per-atom ancestor lookup table.

        ``ancestor_ids`` is a (n_atoms, max_anc) long tensor; row a holds the
        ancestor-node ids for atom a, right-padded with 0. Node id 0 is the
        padding slot in ancestor_embedding and contributes nothing to the sum.

        We do NOT resize ``ancestor_embedding`` here — that's a Parameter
        whose shape must be settled at construction time so checkpoint round-
        trips work. If the supplied ids reference rows past the embedding's
        size, we raise: the caller is expected to construct the model with
        ``n_ancestor_rows`` matching the ETL artifact.
        """
        device = self.embedding.weight.device
        ancestor_ids = ancestor_ids.to(device=device, dtype=torch.long)
        if ancestor_ids.shape[0] != self.embedding.num_embeddings:
            raise ValueError(
                f"ancestor_ids has {ancestor_ids.shape[0]} rows but embedding has "
                f"{self.embedding.num_embeddings} atoms"
            )
        max_node = int(ancestor_ids.max().item()) if ancestor_ids.numel() else 0
        if max_node >= self.ancestor_embedding.num_embeddings:
            raise ValueError(
                f"ancestor_ids references node id {max_node} but ancestor_embedding has only "
                f"{self.ancestor_embedding.num_embeddings} rows; reconstruct the model with "
                f"GenterpConfig(n_ancestor_rows={max_node})"
            )
        self.ancestor_ids = ancestor_ids

    def has_ancestors(self) -> bool:
        return self.ancestor_ids.numel() > 0 and self.ancestor_ids.shape[1] > 0

    def effective_weight(self) -> torch.Tensor:
        """Per-atom embedding (leaf + ancestor sum). Differentiable.

        In flat mode this is exactly ``self.embedding.weight``. In hierarchical
        mode the ancestor contribution is added via a single embedding-lookup +
        sum, materializing the full (n_atoms, dim) matrix. n_atoms is ~30k and
        dim is 512–1024, so the materialized tensor is ~30–60 MB — well within
        budget and recomputed each forward so gradients propagate back to both
        leaf and ancestor parameters.
        """
        leaf = self.embedding.weight
        if not self.has_ancestors():
            return leaf
        anc_emb = self.ancestor_embedding(self.ancestor_ids)
        anc_mask = (self.ancestor_ids != 0).to(anc_emb.dtype).unsqueeze(-1)
        anc_count = anc_mask.sum(dim=-2).clamp(min=1.0)
        anc = (anc_emb * anc_mask).sum(dim=-2) * anc_count.rsqrt()
        return leaf + anc

    @property
    def weight(self) -> torch.Tensor:
        return self.effective_weight()

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        if not self.has_ancestors():
            return self.embedding(atoms)
        leaf = self.embedding(atoms)
        anc_ids = self.ancestor_ids[atoms]
        anc_emb = self.ancestor_embedding(anc_ids)
        anc_mask = (anc_ids != 0).to(anc_emb.dtype).unsqueeze(-1)
        anc_count = anc_mask.sum(dim=-2).clamp(min=1.0)
        anc = (anc_emb * anc_mask).sum(dim=-2) * anc_count.rsqrt()
        return leaf + anc


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

    def __init__(self, dim: int, n_atoms: int, hidden: int = 64, z_clip: float = 12.0):
        super().__init__()
        if z_clip <= 0:
            raise ValueError("z_clip must be > 0")
        self.z_clip = float(z_clip)
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
        return _log1p_signed(z).clamp(min=-self.z_clip, max=self.z_clip)

    def forward(self, e_concept: torch.Tensor, leaf_atom: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        has_mag = self.event_has_magnitude(leaf_atom, value)
        selected_concepts = e_concept[has_mag]
        selected_z = self.z_score(leaf_atom[has_mag], value[has_mag]).unsqueeze(-1)
        mlp_input = torch.cat([selected_z, selected_concepts.float()], dim=-1)
        modulation = torch.tanh(self.value_mlp(mlp_input)).to(e_concept.dtype)
        out = e_concept.clone()
        out[has_mag] = selected_concepts * modulation
        return out


class ValueHead(nn.Module):
    """Student-t over log1p-signed z, conditioned on (hidden, predicted-leaf-atom embedding)."""

    def __init__(self, dim: int, hidden: int = 64, df: float = 4.0):
        super().__init__()
        if df <= 0:
            raise ValueError("df must be > 0")
        self.df = float(df)
        log_norm = (
            math.lgamma((self.df + 1.0) * 0.5)
            - math.lgamma(self.df * 0.5)
            - 0.5 * (math.log(self.df) + math.log(math.pi))
        )
        self.register_buffer("_log_norm", torch.tensor(log_norm, dtype=torch.float32), persistent=False)
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
        df = self.df
        y = ((z_target - mu).float() * (-log_sigma).exp().float()).pow(2) / df
        log_prob = self._log_norm - log_sigma.float() - 0.5 * (df + 1.0) * torch.log1p(y)
        return -log_prob.to(z_target.dtype)

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor, concept_emb: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        mu, log_sigma = self.params(hidden, concept_emb)
        gamma_shape = torch.full(mu.shape, self.df * 0.5, device=mu.device, dtype=torch.float32)
        chi2 = 2.0 * torch._standard_gamma(gamma_shape, generator=generator)
        noise = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        student_t = noise / torch.sqrt((chi2 / self.df).clamp_min(torch.finfo(chi2.dtype).tiny)).to(mu.dtype)
        return mu + log_sigma.exp() * student_t


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
        embed_module: AtomEmbedding,
        n_mix: int = 8,
        time_dim: int = 32,
        sampled_mark_negatives: int = 4096,
        exact_mark_loss_weight: float = 0.1,
        exact_mark_loss_max_tokens: int = 256,
        mark_z_loss_weight: float = 1e-4,
        mark_z_loss_max_tokens: int = 256,
    ):
        super().__init__()
        assert time_dim % 2 == 0
        mark_weight = embed_module.embedding.weight
        if mark_weight.shape != (n_marks, dim):
            raise ValueError("embed_module.embedding.weight must have shape (n_marks, dim)")
        self.n_marks = n_marks
        self.n_mix = n_mix
        self.sampled_mark_negatives = sampled_mark_negatives
        self.exact_mark_loss_weight = float(exact_mark_loss_weight)
        self.exact_mark_loss_max_tokens = int(exact_mark_loss_max_tokens)
        self.mark_z_loss_weight = float(mark_z_loss_weight)
        self.mark_z_loss_max_tokens = int(mark_z_loss_max_tokens)
        self.time_proj = nn.Linear(dim, 3 * n_mix)
        self.mark_h_proj = nn.Linear(dim, dim, bias=False)
        self.mark_time_proj = nn.Linear(time_dim, dim, bias=False)
        # mark_out remains a Linear so its Parameter ``weight`` stays tied to
        # the leaf atom embedding (preserves state-dict path used by HF's
        # save_pretrained dedup and old checkpoints' tied_weights_keys).
        # Effective output projection — including ancestor contributions when
        # hierarchical mode is active — is computed via ``_output_weight()``
        # below, which reads the embedding module dynamically.
        self.mark_out = nn.Linear(dim, n_marks, bias=False)
        self.mark_out.weight = mark_weight
        # Keep a non-registered reference to the embedding module so we can
        # resolve the effective output weight at forward time without making
        # the embedding a submodule of the TPP head (which would double-count
        # parameters in optimizer state).
        object.__setattr__(self, "_embed_module", embed_module)
        nn.init.normal_(self.mark_h_proj.weight, std=0.02)
        nn.init.normal_(self.mark_time_proj.weight, std=0.02)
        freqs = torch.exp(torch.linspace(math.log(0.01), math.log(100.0), time_dim // 2))
        self.register_buffer("time_freqs", freqs, persistent=False)
        mark_noise_probs = torch.ones(n_marks, dtype=torch.float32)
        mark_noise_probs[0] = 0.0
        self.register_buffer("mark_noise_probs", mark_noise_probs / mark_noise_probs.sum().clamp(min=1.0))
        self.register_buffer("_mark_negative_cache", torch.empty(0, dtype=torch.long), persistent=False)
        self._mark_negative_cache_offset = 0

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
        self._mark_negative_cache = self._mark_negative_cache.new_empty(0)
        self._mark_negative_cache_offset = 0

    def _sample_mark_negatives(self, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.mark_noise_probs.device)
        remaining = self._mark_negative_cache.numel() - self._mark_negative_cache_offset
        if remaining < k:
            draw_count = max(k * 256, k)
            self._mark_negative_cache = torch.multinomial(self.mark_noise_probs, draw_count, replacement=True)
            self._mark_negative_cache_offset = 0
        start = self._mark_negative_cache_offset
        stop = start + k
        self._mark_negative_cache_offset = stop
        return self._mark_negative_cache[start:stop]

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

    def _output_weight(self) -> torch.Tensor:
        """Effective mark-output projection weight (leaf + ancestor sums in
        hierarchical mode; identical to ``mark_out.weight`` in flat mode).

        We resolve this from the embedding module each call so hierarchical
        mode "just works" without re-tying parameters, and so a single
        warm-start checkpoint can flip on hierarchical embeddings at any
        training step without rebuilding the head.
        """
        embed = getattr(self, "_embed_module", None)
        if embed is None or not embed.has_ancestors():
            return self.mark_out.weight
        return embed.effective_weight()

    def mark_log_probs(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        logits = self.mark_logits(hidden, delta_t).float()
        logits[:, 0] = float("-inf")
        return logits.log_softmax(dim=-1)

    def mark_logits(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.mark_features(hidden, delta_t) @ self._output_weight().T

    @staticmethod
    def _sample_logits(logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        probs = logits.float().softmax(dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = probs.sum(dim=-1, keepdim=True)
        if not torch.all(total > 0):
            raise RuntimeError("cannot sample from logits with no finite probability mass")
        probs = probs / total
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    def sampled_mark_nll(
        self,
        hidden: torch.Tensor,
        delta_t: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if target.numel() == 0:
            zero = hidden.sum() * 0.0
            return zero if reduction != "none" else zero.expand(0)
        features = self.mark_features(hidden, delta_t)
        k = min(self.sampled_mark_negatives, max(self.n_marks - 1, 1))
        negatives = self._sample_mark_negatives(k)
        weight = self._output_weight()

        target = target.long()
        target_logits = (features * weight[target]).sum(dim=-1, keepdim=True)
        negative_logits = features @ weight[negatives].T
        negative_logits = negative_logits.masked_fill(negatives.unsqueeze(0) == target.unsqueeze(1), float("-inf"))

        q_neg = self.mark_noise_probs[negatives].clamp_min(torch.finfo(self.mark_noise_probs.dtype).tiny)
        negative_logits = negative_logits - (k * q_neg).log().to(negative_logits.dtype)
        logits = torch.cat([target_logits, negative_logits], dim=-1).float()
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels, reduction=reduction)

    def exact_mark_nll(
        self,
        hidden: torch.Tensor,
        delta_t: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if target.numel() == 0:
            zero = hidden.sum() * 0.0
            return zero if reduction != "none" else zero.expand(0)
        logits = self.mark_logits(hidden, delta_t).float()
        logits[:, 0] = float("-inf")
        return F.cross_entropy(logits, target.long(), reduction=reduction)

    def mark_z_loss(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if hidden.numel() == 0 or self.mark_z_loss_weight <= 0.0:
            return hidden.sum() * 0.0
        logits = self.mark_logits(hidden, delta_t).float()
        z = logits[:, 1:].logsumexp(dim=-1)
        return z.pow(2).mean()

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        log_w, mu, log_sigma = self.time_params(hidden)
        # NaN/Inf guards: a destabilized value head can emit NaN/Inf for
        # log_w/mu/log_sigma. Categorical(logits=NaN) is undefined and
        # (mu + sigma*noise).exp() with extreme mu overflows to +inf,
        # which then poisons _phi(delta_t) → mark_logits → NaN.
        # Without guards a single broken batch propagates NaN through the
        # entire rollout chain (max_steps × n_chains events of garbage).
        log_w = torch.nan_to_num(log_w, nan=-30.0, posinf=0.0, neginf=-30.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        # log_sigma is already clamped to [-5, 5] inside time_params; nan_to_num
        # belt-and-suspenders for the NaN case (clamp doesn't sanitize NaN).
        log_sigma = torch.nan_to_num(log_sigma, nan=0.0)
        k = self._sample_logits(log_w, generator)
        mu_k = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sigma_k = log_sigma.gather(-1, k.unsqueeze(-1)).squeeze(-1).exp()
        noise = torch.randn(mu_k.shape, generator=generator, device=mu_k.device, dtype=mu_k.dtype)
        # Clamp log-delta-t to ~[1ms, 1e9 days ≈ 2.7M years] before exp so a
        # rogue mu can't produce inf delta_t (which would then make _phi(inf)
        # → NaN downstream).
        log_delta_t = (mu_k + sigma_k * noise).clamp(min=-7.0, max=21.0)
        delta_t = log_delta_t.exp()
        # Training treats PAD as zero-probability mass (mark_noise_probs[0]=0),
        # so the model is never asked to emit PAD as a real next event.
        # At inference the unmasked categorical can still draw PAD whenever
        # the non-PAD logits are all small, which would waste a rollout step
        # on a non-event. Force PAD to -inf before sampling so chains only
        # ever extend with real atoms.
        mark_logits = self.mark_logits(hidden, delta_t).clone()
        mark_logits = torch.nan_to_num(mark_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        mark_logits[..., 0] = float("-inf")
        mark = self._sample_logits(mark_logits, generator)
        return delta_t, mark


@dataclass
class GenterpConfig:
    n_atoms: int = 65536
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    n_static_blocks: int = 2
    k_static_summary: int = 8
    n_sexes: int = 3
    mlp_mult: int = 4
    dropout: float = 0.0
    pad_atom_idx: int = 0
    n_time_mix: int = 8
    time_phi_dim: int = 32
    value_mlp_hidden: int = 64
    value_z_clip: float = 12.0
    value_head_hidden: int = 64
    value_head_df: float = 4.0
    sampled_mark_negatives: int = 4096
    exact_mark_loss_weight: float = 0.1
    exact_mark_loss_max_tokens: int = 256
    mark_z_loss_weight: float = 1e-4
    mark_z_loss_max_tokens: int = 256
    min_time_delta_days: float = 1e-4
    # Hierarchical embedding: number of distinct ancestor *nodes* across the
    # vocabulary (excluding the pad row at index 0). 0 keeps the flat path —
    # ancestor_embedding is a single zero row that's never indexed. Fixed at
    # construction time so checkpoints round-trip cleanly; flipping it on for
    # an existing run is a vocab-class change and forces optimizer rebuild.
    n_ancestor_rows: int = 0
    # Width of the per-atom ancestor closure (the ``max_anc`` axis of
    # ``ancestor_ids``). Carried in the config so the persistent buffer is
    # constructed at the saved width and HF ``from_pretrained`` shape-checks
    # pass. Refreshed from ``ancestors.npz`` at train-time before model
    # construction. 0 keeps the flat path; the buffer is then (n_atoms, 0).
    n_ancestor_cols: int = 0
    # Bag-NLL auxiliary loss over same-time atom sets. Same-timestamp atoms
    # in OMOP have no semantic ordering, but the AR mark loss for any
    # intra-group transition asks the model to predict an arbitrary
    # serialization of co-occurring labs, diagnoses, etc. This loss reuses the
    # existing mark projection weights to score the next time-group's atom set;
    # it does not add another prediction head.
    bag_loss_weight: float = 0.0
    bag_loss_negatives: int = 256
    # Per-subject loss normalization (off by default).
    # False = event-weighted: every emitted token contributes equally to the
    # loss, so high-utilizer subjects (many events in window) dominate the
    # gradient. True = subject-weighted: within each subject, token losses are
    # averaged first; then those per-subject means are averaged across the
    # batch. Heavy utilizers no longer dwarf typical patients in the gradient.
    # Applied uniformly to time, mark, value, and censor heads. Eval metrics
    # still report the event-weighted means under the same names so curves are
    # comparable to prior runs; the change is to ``loss`` only.
    per_subject_loss_norm: bool = False


class Genterp(nn.Module):
    """Clinical FM: collapsed atom tokens, value-modulated, Set-Transformer static prefix, continuous-time RoPE, marked-TPP head, Student-t value head."""

    def __init__(self, cfg: GenterpConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = AtomEmbedding(
            cfg.n_atoms,
            cfg.dim,
            padding_idx=cfg.pad_atom_idx,
            n_ancestor_rows=cfg.n_ancestor_rows,
            n_ancestor_cols=cfg.n_ancestor_cols,
        )
        self.sex_embedding = nn.Embedding(cfg.n_sexes, cfg.dim)
        nn.init.normal_(self.sex_embedding.weight, std=0.02)
        self.value_mod = ValueModulator(cfg.dim, cfg.n_atoms, cfg.value_mlp_hidden, cfg.value_z_clip)
        self.static_encoder = SetTransformer(cfg.dim, cfg.n_heads, cfg.n_static_blocks, cfg.k_static_summary, cfg.mlp_mult)
        self.rope = ContinuousTimeRoPE(cfg.dim // cfg.n_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, self.rope, cfg.n_layers, cfg.mlp_mult, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.dim)
        self.tpp = MarkedTPPHead(
            cfg.dim,
            cfg.n_atoms,
            self.embed,
            cfg.n_time_mix,
            cfg.time_phi_dim,
            cfg.sampled_mark_negatives,
            cfg.exact_mark_loss_weight,
            cfg.exact_mark_loss_max_tokens,
            cfg.mark_z_loss_weight,
            cfg.mark_z_loss_max_tokens,
        )
        self.value_head = ValueHead(cfg.dim, cfg.value_head_hidden, cfg.value_head_df)
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
        sex: torch.Tensor,
        event_atoms: torch.Tensor,
        event_ages: torch.Tensor,
        event_pad: torch.Tensor,
        target_atoms: torch.Tensor,
        event_values: torch.Tensor,
        length: torch.Tensor | None = None,
        return_transcoder_acts: bool = False,
        **loss_only_kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del loss_only_kwargs  # censor_age + other loss-only or external fields ride the same batch dict
        B = static_atoms.shape[0]
        T = event_ages.shape[1]
        K = self.cfg.k_static_summary
        device = event_ages.device

        # Per-layer NaN guards. When training destabilizes (e.g., a value-head
        # blow-up under the Student-t NLL × clipped z_target × extreme mu has
        # been observed pushing log_sigma to a clamp boundary and producing
        # gradient spikes large enough that Adam takes a catastrophic step,
        # leaving some weight rows in a degenerate state), specific input
        # combinations trigger NaN in one of the intermediate tensors. The
        # NaN then propagates through every downstream layer and the eval
        # comes out uniformly NaN → 0.5 C-index. Guards replace NaN/Inf with
        # 0.0 at each stage and, only when in eval mode, log the first layer
        # to fire so we know the source instead of guessing.
        nan_layers: list[str] = []
        def _guard(t: torch.Tensor, label: str) -> torch.Tensor:
            if torch.isnan(t).any() or torch.isinf(t).any():
                if not self.training:
                    nan_layers.append(label)
                return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            return t

        sex_context = _guard(self.sex_embedding(sex.long()).unsqueeze(1), "sex_embedding")
        static_emb = _guard(self._embed(static_atoms), "embed[static]")
        static_context = torch.cat([sex_context, static_emb], dim=1)
        static_context_pad = F.pad(static_pad, (1, 0), value=False)
        summary = _guard(self.static_encoder(static_context, static_context_pad), "static_encoder")
        events = _guard(self._embed(event_atoms), "embed[event]")
        events = _guard(self.value_mod(events, target_atoms, event_values), "value_mod")
        x = torch.cat([summary, events], dim=1)

        is_static = self._static_mask(B, K, T, device)
        ages_full = F.pad(event_ages, (K, 0), value=0.0)
        angles = self.rope.angles(ages_full, is_static)
        event_lengths = _length_list(event_pad, length) if length is not None else None

        pre_mlps: list[torch.Tensor] = []
        mlp_outs: list[torch.Tensor] = []
        for layer_idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x, pre_mlp, mlp_out = self._gradient_checkpointing_func(blk, x, angles, event_pad, K, event_lengths)
            else:
                x, pre_mlp, mlp_out = blk(x, angles, event_pad, K, event_lengths)
            x = _guard(x, f"block[{layer_idx}]")
            if return_transcoder_acts:
                pre_mlps.append(pre_mlp[:, K:])
                mlp_outs.append(mlp_out[:, K:])

        hidden = _guard(self.norm(x[:, K:]), "norm")
        if not self.training and nan_layers:
            # First layer to fire is the root cause; the rest are downstream
            # propagation (or the inputs already had NaN by the time the
            # block ran). Print once per forward.
            first = nan_layers[0]
            rest = len(nan_layers) - 1
            print(
                f"[genterp-forward] NaN detected — first_layer={first} "
                f"downstream_layers_also_hit={rest} B={B} T={T} "
                f"event_ages.max={float(event_ages.max().item()) if event_ages.numel() else 0.0:.3g} "
                f"event_values.absmax={float(event_values.abs().max().item()) if event_values.numel() else 0.0:.3g}"
            )
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
        # Pop bag-loss inputs before passing the rest to ``forward``; they
        # are only consumed by the loss path, not by the encoder.
        event_groups = batch.pop("event_groups", None)
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
            self.cfg.min_time_delta_days,
            event_groups=event_groups,
            bag_loss_weight=self.cfg.bag_loss_weight,
            bag_loss_negatives=self.cfg.bag_loss_negatives,
            pad_atom_idx=self.cfg.pad_atom_idx,
            per_subject_norm=self.cfg.per_subject_loss_norm,
        )


def bag_nll_same_time(
    hidden: torch.Tensor,
    event_atoms: torch.Tensor,
    event_groups: torch.Tensor,
    event_pad: torch.Tensor,
    output_weight: torch.Tensor,
    mark_noise_probs: torch.Tensor,
    n_negatives: int,
    pad_atom_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled multi-label BCE over same-time atom sets.

    For each (subject, group) transition g → g+1, the predictor is the hidden
    state at the last position of group g, and the bag target is the set of
    atoms in group g+1. Loss = -Σ_pos log σ(h·e_pos) - Σ_neg log(1 - σ(h·e_neg))
    with k unigram-sampled negatives shared across predictors (de-duplicated
    against actual positives so a sampled "negative" that's really in the bag
    isn't penalized).

    Returns ``(per-predictor loss, n_predictors)``. ``loss`` is 0 (with a
    differentiable zero) when no group transitions exist in the batch.
    """
    B, T = event_atoms.shape
    zero_loss = hidden.sum() * 0.0
    zero_count = hidden.new_tensor(0, dtype=torch.long)
    if T < 2 or hidden.numel() == 0:
        return zero_loss, zero_count
    real = (~event_pad) & (event_groups >= 0) & (event_atoms != pad_atom_idx)
    # Boundary at (b, s): real[s] & real[s+1] & groups[s] != groups[s+1].
    boundary = torch.zeros_like(real)
    boundary[:, :-1] = (
        real[:, :-1]
        & real[:, 1:]
        & (event_groups[:, 1:] != event_groups[:, :-1])
    )
    pred_b, pred_s = boundary.nonzero(as_tuple=True)
    P = pred_b.numel()
    if P == 0:
        return zero_loss, zero_count
    h_pred = hidden[pred_b, pred_s]  # (P, d)

    # Map each atom at (b, t) with group >= 1 to its predictor index.
    # Predictors are listed in (b ascending, s ascending) order — same as
    # the natural row-major flatten that ``nonzero`` returns. Within a row
    # the (g-1)-th boundary is the predictor for group g.
    valid_atom = real & (event_groups >= 1)
    atom_b, atom_t = valid_atom.nonzero(as_tuple=True)
    if atom_b.numel() == 0:
        return zero_loss, zero_count
    atom_g = event_groups[atom_b, atom_t]  # (A,)
    atom_ids = event_atoms[atom_b, atom_t].long()
    boundaries_per_row = boundary.long().sum(dim=1)  # (B,)
    row_offset = torch.zeros(B, dtype=torch.long, device=hidden.device)
    if B > 1:
        row_offset[1:] = boundaries_per_row[:-1].cumsum(0)
    pred_idx = row_offset[atom_b] + (atom_g.long() - 1)
    # Defensive clamp: any (b, t) whose predictor index escapes [0, P) would
    # indicate a group-numbering bug upstream. Drop those rather than write
    # past the end of h_pred.
    keep = (pred_idx >= 0) & (pred_idx < P)
    if not bool(keep.all()):
        atom_ids = atom_ids[keep]
        pred_idx = pred_idx[keep]
    if atom_ids.numel() == 0:
        return zero_loss, zero_count

    # Positive logits and BCE.
    pos_logits = (h_pred[pred_idx] * output_weight[atom_ids]).sum(dim=-1).float()
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="sum"
    )

    # Sampled negatives shared across predictors.
    V = output_weight.shape[0]
    k = max(1, min(n_negatives, V - 1))
    neg_atoms = torch.multinomial(mark_noise_probs, k, replacement=True)
    neg_logits = (h_pred @ output_weight[neg_atoms].T).float()  # (P, k)
    # Unmask negatives that coincide with this predictor's actual positives
    # so we don't penalize the model for predicting a true bag member.
    eq = atom_ids.unsqueeze(1) == neg_atoms.unsqueeze(0)  # (A, k)
    neg_mask = torch.ones_like(neg_logits, dtype=torch.bool)
    if eq.any():
        a_idx, j_idx = eq.nonzero(as_tuple=True)
        neg_mask[pred_idx[a_idx], j_idx] = False
    neg_bce = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="none"
    )
    neg_loss = (neg_bce * neg_mask.float()).sum()

    bag_loss = (pos_loss + neg_loss) / max(P, 1)
    return bag_loss, hidden.new_tensor(P, dtype=torch.long)


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
    min_time_delta_days: float,
    event_groups: torch.Tensor | None = None,
    bag_loss_weight: float = 0.0,
    bag_loss_negatives: int = 256,
    pad_atom_idx: int = 0,
    per_subject_norm: bool = False,
) -> dict[str, torch.Tensor]:
    """Joint NLL: marked-TPP (time + mark) + value (robust z-space density) + right-censoring.

    Real next event at t:    -log p(Δt|h_t) - log p(m|h_t, Δt) - has_mag · log p(z|h_t, e_m)
    Censoring:               -log S(Δt_c|h_last)
    """
    h_pred = hidden[:, :-1]
    delta_real = event_ages[:, 1:] - event_ages[:, :-1]
    delta_censor = censor_age.unsqueeze(-1) - event_ages[:, :-1]
    target_real = target_atoms[:, 1:].clamp(min=0)
    value_real = event_values[:, 1:]
    real_mask = (~event_pad[:, :-1]) & (~event_pad[:, 1:])
    time_real_mask = real_mask & (delta_real > float(min_time_delta_days))
    censor_mask = (~event_pad[:, :-1]) & event_pad[:, 1:]
    any_mask = time_real_mask | censor_mask
    n_real = real_mask.sum()
    n_time = time_real_mask.sum()
    n_censor = censor_mask.sum()

    delta_any = torch.where(real_mask, delta_real, delta_censor).clamp(min=1e-6)[any_mask]
    log_w, mu, log_sigma = tpp.time_params(h_pred[any_mask])
    log_dt = delta_any.log().unsqueeze(-1)
    inv_sigma = (-log_sigma).exp()
    log_pdf = -log_dt - log_sigma - 0.5 * math.log(2 * math.pi) - 0.5 * ((log_dt - mu) * inv_sigma).pow(2)
    time_lp = torch.logsumexp(log_w + log_pdf, dim=-1)
    z = (log_dt - mu) * inv_sigma
    time_ls = torch.logsumexp(log_w + _log_ndtr(-z), dim=-1)

    real_any = time_real_mask[any_mask]
    real_time_lp = time_lp[real_any]
    censor_time_ls = time_ls[~real_any]

    mag_mask = real_mask & value_mod.event_has_magnitude(target_real, value_real)
    n_mag = mag_mask.sum()

    delta_mark = delta_real.clamp(min=1e-6)[real_mask]
    mark_hidden = h_pred[real_mask]
    mark_target = target_real[real_mask]
    if tpp.training:
        sampled_mark_loss = tpp.sampled_mark_nll(mark_hidden, delta_mark, mark_target)
        rho = min(max(tpp.exact_mark_loss_weight, 0.0), 1.0)
        if rho > 0.0 and mark_target.numel() > 0:
            exact_tokens = min(max(tpp.exact_mark_loss_max_tokens, 1), mark_target.numel())
            exact_mark_loss = tpp.exact_mark_nll(
                mark_hidden[:exact_tokens],
                delta_mark[:exact_tokens],
                mark_target[:exact_tokens],
            ) * (mark_target.numel() / exact_tokens)
        else:
            exact_mark_loss = sampled_mark_loss.detach() * 0.0
        mark_loss = (1.0 - rho) * sampled_mark_loss + rho * exact_mark_loss
    else:
        sampled_mark_loss = h_pred.sum() * 0.0
        mark_log_probs = tpp.mark_log_probs(mark_hidden, delta_mark)
        mark_lp = torch.gather(mark_log_probs, -1, mark_target.unsqueeze(-1)).squeeze(-1)
        mark_loss = -mark_lp.sum()
        exact_mark_loss = mark_loss

    leaf_mag = target_real[mag_mask]
    z_target = value_mod.z_score(leaf_mag, value_real[mag_mask])
    value_nll_tokens = value_head.nll(h_pred[mag_mask], atom_embedding[leaf_mag], z_target)
    value_loss = value_nll_tokens.sum()
    value_nll_max = value_nll_tokens.detach().float().max() if value_nll_tokens.numel() else hidden.new_tensor(0.0)
    value_z_abs_max = z_target.detach().float().abs().max() if z_target.numel() else hidden.new_tensor(0.0)
    value_z_clipped = z_target.detach().float().abs().ge(value_mod.z_clip).float().sum()

    time_loss = -real_time_lp.sum()
    censor_loss = -censor_time_ls.sum()
    time_nll = time_loss / n_real.clamp(min=1)
    time_modeled_nll = time_loss / n_time.clamp(min=1)
    mark_nll = mark_loss / n_real.clamp(min=1)
    sampled_mark_nll = sampled_mark_loss / n_real.clamp(min=1)
    exact_mark_nll = exact_mark_loss / n_real.clamp(min=1)
    value_nll = value_loss / n_mag.clamp(min=1)
    censor_nll = censor_loss / n_censor.clamp(min=1)
    n_real_f = n_real.clamp(min=1).float()
    value_weight = (n_mag.float() / n_real_f).detach()
    censor_weight = (n_censor.float() / n_real_f).detach()
    if tpp.mark_z_loss_weight > 0.0 and mark_target.numel() > 0:
        z_tokens = min(max(tpp.mark_z_loss_max_tokens, 1), mark_target.numel())
        mark_z_loss = tpp.mark_z_loss(mark_hidden[:z_tokens], delta_mark[:z_tokens])
    else:
        mark_z_loss = hidden.sum() * 0.0
    weighted_mark_z_loss = tpp.mark_z_loss_weight * mark_z_loss

    # Same-time bag NLL (auxiliary, gated by config). When disabled (weight
    # 0.0 or no event_groups in the batch) it short-circuits to a
    # differentiable zero that contributes nothing to the gradient.
    if bag_loss_weight > 0.0 and event_groups is not None:
        bag_nll, n_bag_predictors = bag_nll_same_time(
            hidden=hidden,
            event_atoms=target_atoms,
            event_groups=event_groups,
            event_pad=event_pad,
            output_weight=tpp._output_weight(),
            mark_noise_probs=tpp.mark_noise_probs,
            n_negatives=bag_loss_negatives,
            pad_atom_idx=pad_atom_idx,
        )
    else:
        bag_nll = hidden.sum() * 0.0
        n_bag_predictors = hidden.new_tensor(0, dtype=torch.long)
    weighted_bag_nll = float(bag_loss_weight) * bag_nll

    # Emit per-subject loss + token counts during eval so the trainer can
    # stratify metrics by group (sex, age band, etc.) without a second
    # forward pass. Computed always (cheap: scatter_add over per-token NLLs)
    # but only logged when ``model.training`` is False.
    B = real_mask.shape[0]
    device = hidden.device
    b_grid_global = torch.arange(B, device=device).unsqueeze(1).expand_as(real_mask)
    b_idx_real_time_global = b_grid_global[any_mask][real_any]
    b_idx_mark_global = b_grid_global[real_mask]
    per_subject_time_sum = hidden.new_zeros(B, dtype=torch.float32).scatter_add_(
        0,
        b_idx_real_time_global,
        (-real_time_lp).detach().float(),
    )
    per_subject_real_count = hidden.new_zeros(B, dtype=torch.float32).scatter_add_(
        0,
        b_idx_mark_global,
        torch.ones(mark_target.numel(), device=device, dtype=torch.float32),
    )

    if per_subject_norm:
        # Subject-weighted total: avoid event-weighted aggregation that lets
        # heavy utilizers (4096-event windows) dominate the gradient. For each
        # subject in the batch we compute the mean per-token NLL of every head
        # over the tokens that came from that subject, then average those
        # subject-level means over subjects that emitted at least one token in
        # the head. Diagnostic per-token means above are kept event-weighted so
        # logged metrics remain directly comparable across reduction modes.
        B = real_mask.shape[0]
        device = hidden.device
        b_grid = torch.arange(B, device=device).unsqueeze(1).expand_as(real_mask)
        b_idx_any = b_grid[any_mask]
        b_idx_real_time = b_idx_any[real_any]
        b_idx_censor = b_idx_any[~real_any]
        b_idx_mark = b_grid[real_mask]
        b_idx_mag = b_grid[mag_mask]

        def _per_subject_mean(token_loss: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
            if token_loss.numel() == 0:
                return hidden.sum() * 0.0
            sums = hidden.new_zeros(B, dtype=torch.float32).scatter_add_(
                0, b_idx, token_loss.float()
            )
            counts = hidden.new_zeros(B, dtype=torch.float32).scatter_add_(
                0, b_idx, torch.ones_like(token_loss, dtype=torch.float32)
            )
            active = counts > 0
            if not bool(active.any()):
                return hidden.sum() * 0.0
            return (sums[active] / counts[active].clamp(min=1.0)).mean().to(hidden.dtype)

        time_total = _per_subject_mean(-real_time_lp, b_idx_real_time)
        censor_total = _per_subject_mean(-censor_time_ls, b_idx_censor)

        # Mark: rebuild per-token NLL. The exact-mark refinement is sampled
        # over only the first ``exact_tokens`` mark targets, which has no
        # well-defined per-subject decomposition (it's a bias correction on
        # the global sum). Under per-subject normalization we therefore
        # restrict the mark loss to the sampled-softmax NLL, which is the
        # dominant term anyway. The exact_mark log metric is still reported.
        if mark_target.numel() > 0:
            if tpp.training:
                mark_token_loss = tpp.sampled_mark_nll(mark_hidden, delta_mark, mark_target, reduction="none")
            else:
                mark_token_loss = -mark_lp
            mark_total = _per_subject_mean(mark_token_loss, b_idx_mark)
        else:
            mark_total = hidden.sum() * 0.0

        if n_mag.item() > 0:
            value_total = _per_subject_mean(value_nll_tokens, b_idx_mag)
        else:
            value_total = hidden.sum() * 0.0
        # Value/censor heads contribute proportionally to how often their
        # tokens appear per-subject. Re-use the event-weighted weights so the
        # multi-head balance does not change between reduction modes.
        total = (
            time_total
            + mark_total
            + value_weight * value_total
            + censor_weight * censor_total
            + weighted_mark_z_loss
            + weighted_bag_nll
        )
    else:
        total = time_nll + mark_nll + value_weight * value_nll + censor_weight * censor_nll + weighted_mark_z_loss + weighted_bag_nll

    return {
        "loss": total,
        "time_nll": time_nll,
        "time_modeled_nll": time_modeled_nll,
        "mark_nll": mark_nll,
        "sampled_mark_nll": sampled_mark_nll,
        "exact_mark_nll": exact_mark_nll,
        "mark_z_loss": mark_z_loss,
        "weighted_mark_z_loss": weighted_mark_z_loss,
        "value_nll": value_nll,
        "value_nll_max": value_nll_max,
        "value_z_abs_max": value_z_abs_max,
        "value_z_clipped": value_z_clipped,
        "censor_nll": censor_nll,
        "n_real": n_real,
        "n_time": n_time,
        "n_censor": n_censor,
        "n_mag": n_mag,
        "value_loss_weight": value_weight,
        "censor_loss_weight": censor_weight,
        "bag_nll": bag_nll,
        "weighted_bag_nll": weighted_bag_nll,
        "n_bag_predictors": n_bag_predictors,
        "n_subject": hidden.new_tensor(hidden.shape[0]),
        "per_subject_time_sum": per_subject_time_sum,
        "per_subject_real_count": per_subject_real_count,
    }
