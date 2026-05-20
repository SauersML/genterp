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
        event_lengths: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (post_block_residual, pre_mlp_residual, mlp_additive_output)."""
        x = x + self.attn(self.norm1(x), angles, event_pad, static_len, event_lengths)
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

    def __init__(self, n_atoms: int, dim: int, padding_idx: int = 0, n_ancestor_rows: int = 0):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(n_atoms, dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        # ancestor_embedding shape is fixed at construction so checkpoints
        # round-trip cleanly (loading happens before set_ancestor_ids; the
        # parameter is already the right shape). n_ancestor_rows = 0 means
        # "flat mode": a 1-row zero placeholder that never gets indexed (the
        # ancestor_ids buffer below has zero columns).
        rows = max(n_ancestor_rows + 1, 1)
        self.ancestor_embedding = nn.Embedding(rows, dim, padding_idx=0)
        nn.init.zeros_(self.ancestor_embedding.weight)
        self.register_buffer("ancestor_ids", torch.zeros(n_atoms, 0, dtype=torch.long), persistent=False)

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
        return leaf + (anc_emb * anc_mask).sum(dim=-2)

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
        return leaf + (anc_emb * anc_mask).sum(dim=-2)


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
        embed_module: "AtomEmbedding",
        n_mix: int = 8,
        time_dim: int = 32,
        sampled_mark_negatives: int = 4096,
    ):
        super().__init__()
        assert time_dim % 2 == 0
        mark_weight = embed_module.embedding.weight
        if mark_weight.shape != (n_marks, dim):
            raise ValueError("embed_module.embedding.weight must have shape (n_marks, dim)")
        self.n_marks = n_marks
        self.n_mix = n_mix
        self.sampled_mark_negatives = sampled_mark_negatives
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
        return self.mark_logits(hidden, delta_t).log_softmax(dim=-1)

    def mark_logits(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.mark_features(hidden, delta_t) @ self._output_weight().T

    def sampled_mark_nll(self, hidden: torch.Tensor, delta_t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.numel() == 0:
            return hidden.sum() * 0.0
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
        return F.cross_entropy(logits, labels, reduction="sum")

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        log_w, mu, log_sigma = self.time_params(hidden)
        k = torch.distributions.Categorical(logits=log_w).sample()
        mu_k = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sigma_k = log_sigma.gather(-1, k.unsqueeze(-1)).squeeze(-1).exp()
        noise = torch.randn(mu_k.shape, generator=generator, device=mu_k.device, dtype=mu_k.dtype)
        delta_t = (mu_k + sigma_k * noise).exp()
        # Training treats PAD as zero-probability mass (mark_noise_probs[0]=0),
        # so the model is never asked to emit PAD as a real next event.
        # At inference the unmasked categorical can still draw PAD whenever
        # the non-PAD logits are all small, which would waste a rollout step
        # on a non-event. Force PAD to -inf before sampling so chains only
        # ever extend with real atoms.
        mark_logits = self.mark_logits(hidden, delta_t).clone()
        mark_logits[..., 0] = float("-inf")
        mark = torch.distributions.Categorical(logits=mark_logits).sample()
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
    # Hierarchical embedding: number of distinct ancestor *nodes* across the
    # vocabulary (excluding the pad row at index 0). 0 keeps the flat path —
    # ancestor_embedding is a single zero row that's never indexed. Fixed at
    # construction time so checkpoints round-trip cleanly; flipping it on for
    # an existing run is a vocab-class change and forces optimizer rebuild.
    n_ancestor_rows: int = 0


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
        )
        self.sex_embedding = nn.Embedding(cfg.n_sexes, cfg.dim)
        nn.init.normal_(self.sex_embedding.weight, std=0.02)
        self.value_mod = ValueModulator(cfg.dim, cfg.n_atoms, cfg.value_mlp_hidden, cfg.value_z_clip)
        self.static_encoder = SetTransformer(cfg.dim, cfg.n_heads, cfg.n_static_blocks, cfg.k_static_summary, cfg.mlp_mult)
        self.rope = ContinuousTimeRoPE(cfg.dim // cfg.n_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, self.rope, cfg.mlp_mult, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.dim)
        self.tpp = MarkedTPPHead(
            cfg.dim,
            cfg.n_atoms,
            self.embed,
            cfg.n_time_mix,
            cfg.time_phi_dim,
            cfg.sampled_mark_negatives,
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

        sex_context = self.sex_embedding(sex.long()).unsqueeze(1)
        static_context = torch.cat([sex_context, self._embed(static_atoms)], dim=1)
        static_context_pad = F.pad(static_pad, (1, 0), value=False)
        summary = self.static_encoder(static_context, static_context_pad)
        events = self._embed(event_atoms)
        events = self.value_mod(events, target_atoms, event_values)
        x = torch.cat([summary, events], dim=1)

        is_static = self._static_mask(B, K, T, device)
        ages_full = F.pad(event_ages, (K, 0), value=0.0)
        angles = self.rope.angles(ages_full, is_static)
        event_lengths = _length_list(event_pad, length) if length is not None else None

        pre_mlps: list[torch.Tensor] = []
        mlp_outs: list[torch.Tensor] = []
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, pre_mlp, mlp_out = self._gradient_checkpointing_func(blk, x, angles, event_pad, K, event_lengths)
            else:
                x, pre_mlp, mlp_out = blk(x, angles, event_pad, K, event_lengths)
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
    value_nll_max = value_nll_tokens.detach().float().max() if value_nll_tokens.numel() else hidden.new_tensor(0.0)
    value_z_abs_max = z_target.detach().float().abs().max() if z_target.numel() else hidden.new_tensor(0.0)
    value_z_clipped = z_target.detach().float().abs().ge(value_mod.z_clip).float().sum()

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
        "value_nll_max": value_nll_max,
        "value_z_abs_max": value_z_abs_max,
        "value_z_clipped": value_z_clipped,
        "censor_nll": censor_nll,
        "n_real": n_real,
        "n_censor": n_censor,
        "n_mag": n_mag,
        "n_subject": hidden.new_tensor(hidden.shape[0]),
    }
