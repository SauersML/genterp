"""Inference-only KV-cached decode pathway for fast trajectory rollouts.

Why this lives outside ``modeling.py``
--------------------------------------
The training forward in :mod:`genterp.modeling` runs over the full extended
prefix at every call — O(N²) per token sampled. For Monte-Carlo rollouts
that's the wrong cost profile: with a 4K-event prefix and 50 sampled events
per chain, naive re-forward burns ~50× more compute than necessary.

Decode reuses every Parameter from the trained model — same qkv, same
rope, same proj, same SwiGLU, same RMSNorm — but reroutes the data flow:

  1. :func:`decode_init` runs **one** standard forward over the prefix and
     captures rotated K, V at every attention layer alongside the usual
     last-token hidden state.
  2. :func:`decode_step` extends each chain by one event, computes Q, K, V
     for the **single** new token, appends to the cache, and runs single-
     query SDPA over (cache + new). Per-step cost drops from O(N²) to
     O(N), and with the cache held across steps the rollout becomes
     ~T× cheaper overall.

Variable-length subjects in a batch are handled with a per-row valid-length
attention mask: positions past row ``r``'s ``valid_length[r]`` are masked
out so finished or short-prefix chains don't pollute alive ones. No padding
correction is needed afterward.

This module is import-only used by :mod:`genterp.eval_rollout` and any
future inference paths. The training forward is unchanged and is verified
bit-exact by a regression test in ``tests/test_decode.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from genterp.modeling import Block, Genterp


@dataclass
class DecodeCache:
    """Per-layer K, V plus per-row valid prefix length.

    Layouts:
      - ``k[layer]``, ``v[layer]`` : (B, n_heads, T_cached, head_dim)
      - ``valid_length`` : (B,) long  — index of the first PAD position
        within the cache for each row. Used to build attention masks so
        SDPA ignores positions past the row's valid prefix.
      - ``static_len`` : int  — number of static-prefix tokens at the front
        of each row (constant across rows).
    """

    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    valid_length: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    static_len: int = 0

    @property
    def cached_length(self) -> int:
        return int(self.k[0].shape[2]) if self.k else 0

    @property
    def batch_size(self) -> int:
        return int(self.k[0].shape[0]) if self.k else 0


def _attn_mask_for_cache(valid_length: torch.Tensor, cache_T: int) -> torch.Tensor:
    """Bool mask of shape (B, 1, 1, cache_T): True where SDPA must ignore.

    SDPA convention: a True entry in ``attn_mask`` *masks out* that position.
    For each row r, positions [valid_length[r], cache_T) are invalid prefix
    (PAD or never-written), so they get True; everything else (the row's
    real prefix plus the just-appended new K/V) gets False.
    """
    device = valid_length.device
    positions = torch.arange(cache_T, device=device).view(1, 1, 1, cache_T)
    return positions >= valid_length.view(-1, 1, 1, 1)


def _insert_new_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    insert_pos: torch.Tensor,
    advance_mask: torch.Tensor,
    *,
    grow_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if grow_cache:
        pad_shape = (*cache_k.shape[:2], 1, cache_k.shape[3])
        cache_k = torch.cat([cache_k, cache_k.new_zeros(pad_shape)], dim=2)
        cache_v = torch.cat([cache_v, cache_v.new_zeros(pad_shape)], dim=2)
    else:
        cache_k = cache_k.clone()
        cache_v = cache_v.clone()

    rows = torch.arange(cache_k.shape[0], device=cache_k.device)[advance_mask]
    if rows.numel() == 0:
        return cache_k, cache_v
    cols = insert_pos[advance_mask]
    cache_k[rows, :, cols, :] = k_new[advance_mask, :, 0, :]
    cache_v[rows, :, cols, :] = v_new[advance_mask, :, 0, :]
    return cache_k, cache_v


@torch.no_grad()
def decode_init(model: Genterp, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, DecodeCache]:
    """Run the full prefix through the model and build a KV cache.

    Mirrors ``Genterp.forward`` step-for-step but stashes per-layer rotated
    (K, V) into a :class:`DecodeCache`. Returns the last-real-token hidden
    state (shape (B, dim)) for kicking off TPP sampling, alongside the
    cache that subsequent ``decode_step`` calls extend.
    """
    static_atoms = batch["static_atoms"]
    static_pad = batch["static_pad"]
    sex = batch["sex"]
    event_atoms = batch["event_atoms"]
    event_ages = batch["event_ages"]
    event_pad = batch["event_pad"]
    target_atoms = batch["target_atoms"]
    event_values = batch["event_values"]
    # Synthetic test batches sometimes omit ``length`` and let attention infer
    # from event_pad; mirror that fallback so decode and forward stay in sync.
    length = batch.get("length")
    if length is None:
        length = (~event_pad).sum(dim=1).to(dtype=torch.long)

    B = static_atoms.shape[0]
    T = event_ages.shape[1]
    K = model.cfg.k_static_summary
    device = event_ages.device

    sex_context = model.sex_embedding(sex.long()).unsqueeze(1)
    static_context = torch.cat([sex_context, model._embed(static_atoms)], dim=1)
    static_context_pad = F.pad(static_pad, (1, 0), value=False)
    summary = model.static_encoder(static_context, static_context_pad)
    events = model._embed(event_atoms)
    events = model.value_mod(events, target_atoms, event_values)
    x = torch.cat([summary, events], dim=1)

    is_static = model._static_mask(B, K, T, device)
    ages_full = F.pad(event_ages, (K, 0), value=0.0)
    angles = model.rope.angles(ages_full, is_static)
    event_lengths_list = [int(v) for v in length.detach().cpu().tolist()]

    cache = DecodeCache(static_len=K)
    for blk in model.blocks:
        # Inline of Block.forward, splitting attn into qkv-rope (cacheable)
        # and packed-causal-attention (training-only impl path).
        normed = blk.norm1(x)
        q, k, v = blk.attn._qkv_rope(normed, angles)
        attn_out = blk.attn._packed_prefix_causal_attention(
            q, k, v, K, event_pad, event_lengths_list
        )
        attn_out = blk.attn._project_out(attn_out)
        x = x + blk.res_scale * attn_out
        mlp_out = blk.mlp(blk.norm2(x))
        x = x + blk.res_scale * mlp_out
        # Cache rotated K, V in SDPA-friendly (B, H, S, D) layout. The whole
        # static_len + event tensor goes in; per-row validity is tracked via
        # cache.valid_length so short-prefix rows don't pollute attention.
        cache.k.append(k.transpose(1, 2).contiguous())
        cache.v.append(v.transpose(1, 2).contiguous())

    hidden = model.norm(x[:, K:])
    safe_idx = (length.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    h_last = hidden.gather(1, safe_idx).squeeze(1)

    # static_len + actual event count per row: positions [0, K) are always
    # valid (static prefix); event positions are [K, K + length[r]).
    cache.valid_length = (length + K).to(device=device, dtype=torch.long)
    return h_last, cache


@torch.no_grad()
def _block_decode_step(
    blk: Block,
    x: torch.Tensor,
    angles: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    insert_pos: torch.Tensor,
    advance_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    grow_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-token block forward with KV cache. Returns (out, new_k, new_v).

    ``x`` is (B, 1, dim); ``angles`` is (B, 1, head_dim/2) for the new token
    only. ``cache_k/cache_v`` are (B, H, T_prev, D); the function writes the
    new token's rotated K/V into each row's first invalid slot, growing the
    cache only when at least one row has no free slot left.

    Attention is plain SDPA — no flash-attn varlen — because for a single
    query the kernel choice doesn't matter and SDPA's ``attn_mask`` cleanly
    handles per-row variable validity.
    """
    normed = blk.norm1(x)
    q, k, v = blk.attn._qkv_rope(normed, angles)
    # (B, 1, H, D) -> (B, H, 1, D)
    q_t = q.transpose(1, 2)
    k_new = k.transpose(1, 2)
    v_new = v.transpose(1, 2)
    new_k_cache, new_v_cache = _insert_new_kv(
        cache_k,
        cache_v,
        k_new,
        v_new,
        insert_pos,
        advance_mask,
        grow_cache=grow_cache,
    )
    attn = F.scaled_dot_product_attention(q_t, new_k_cache, new_v_cache, attn_mask=~attn_mask)
    # ~attn_mask: SDPA's bool attn_mask treats True as "keep"; our DecodeCache
    # mask is True for "drop". Invert here so the rest of the code reads in
    # "True == invalid" terms.
    # (B, H, 1, D) -> (B, 1, H, D)
    out = blk.attn._project_out(attn.transpose(1, 2))
    x = x + blk.res_scale * out
    mlp_out = blk.mlp(blk.norm2(x))
    return x + blk.res_scale * mlp_out, new_k_cache, new_v_cache


@torch.no_grad()
def decode_step(
    model: Genterp,
    cache: DecodeCache,
    new_atom: torch.Tensor,
    new_age_days: torch.Tensor,
    advance_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extend each chain by one event and return the new last-token hidden state.

    Parameters
    ----------
    model : the model holding the parameters (typically in eval mode).
    cache : DecodeCache to mutate in place — its K, V tensors and
        ``valid_length`` are updated for chains that ``advance_mask`` says are
        still alive.
    new_atom : (B,) long — sampled mark for each chain. Used to embed the new
        token; finished chains can pass anything (their cache update is masked
        out and the returned hidden state is unused).
    new_age_days : (B,) float — absolute age in days of the new token. Drives
        rope rotation for the new query/key.
    advance_mask : (B,) bool — True means "this chain just sampled a real new
        event; advance its valid_length and let the new K/V matter". False
        means "this chain is finished; keep cache K/V appended but don't
        increment valid_length, so the attention mask hides the new position
        on subsequent steps". Default None = all True.

    Returns
    -------
    h_new : (B, dim) hidden state of the new token (post final RMSNorm).
    """
    device = new_atom.device
    B = int(new_atom.shape[0])

    if advance_mask is None:
        advance_mask = torch.ones(B, dtype=torch.bool, device=device)

    # Embed the new token. ValueModulator only activates for atoms whose
    # value is finite; we pass NaN so it short-circuits — sampled events
    # don't have values from the value head in this rollout pathway.
    new_atom_2d = new_atom.unsqueeze(1)
    new_value_nan = torch.full((B, 1), float("nan"), device=device, dtype=torch.float32)
    e = model._embed(new_atom_2d)
    e = model.value_mod(e, new_atom_2d, new_value_nan)
    x = e  # (B, 1, dim)

    is_static_new = torch.zeros(B, 1, dtype=torch.bool, device=device)
    angles = model.rope.angles(new_age_days.unsqueeze(1), is_static_new)

    # New K/V land in each row's first invalid slot. Short rows reuse their
    # existing pad slots; the cache grows only when an alive row is already at
    # the global end.
    cached_T = cache.cached_length
    grow_cache = bool((advance_mask & (cache.valid_length >= cached_T)).any().item())
    post_append_T = cached_T + int(grow_cache)
    insert_pos = cache.valid_length.clamp(max=post_append_T - 1)
    post_valid_length = cache.valid_length + advance_mask.to(cache.valid_length.dtype)
    attn_mask = _attn_mask_for_cache(post_valid_length, post_append_T)

    for layer_idx, blk in enumerate(model.blocks):
        x, new_k, new_v = _block_decode_step(
            blk,
            x,
            angles,
            cache.k[layer_idx],
            cache.v[layer_idx],
            insert_pos,
            advance_mask,
            attn_mask,
            grow_cache,
        )
        cache.k[layer_idx] = new_k
        cache.v[layer_idx] = new_v

    cache.valid_length = post_valid_length

    h_new = model.norm(x[:, 0])  # (B, dim)
    return h_new
