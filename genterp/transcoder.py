"""Cross-Layer Transcoder (CLT) with JumpReLU activations + tanh sparsity penalty.

For each layer ℓ ∈ [0, L):
    preact[n, ℓ]   = W_enc[ℓ] @ pre_mlp[n, ℓ] + b_enc[ℓ]
    features[n, ℓ] = ReLU(preact) * (ReLU(preact) > θ[ℓ]) # JumpReLU, per-feature θ
For each output layer t ∈ [0, L):
    recon[n, t]    = W_diag[t] @ features[n, t]                       (same-layer write)
                   + Σ_{s < t} W_off[s, t] @ features[n, s] + b_dec[t] (cross-layer write)

Loss = Σ_t MSE((recon - mlp_out) / σ_t)
     + sparsity_coef · Σ tanh(features · ||W_diag[ℓ]||₂ / (σ_ℓ · scale))

Storage. The decoder is stored in two pieces:

  * ``diag_W`` (L, F, D) — same-layer dense decoders. Always present. The
    sparsity penalty uses these to scale per-feature contributions.
  * Off-diagonal blocks W_off[s, t] for t > s, packed over the strict upper
    triangle (no lower-triangle parameters exist at all — the old dec_mask is
    obsolete). Stored either dense or low-rank, controlled by
    ``CLTConfig.off_diagonal_rank``:

      None  →  ``off_blocks`` (N_off, F, D), dense.
      r:int →  W_off[s, t] = U[s, t] @ V[s], with shared V across destinations.
               ``off_U`` (N_off, F, r), ``off_V`` (L, r, D). The shared-V prior
               says "a feature at layer s writes into a fixed r-dim subspace
               of the residual stream; only its mix per destination varies".

N_off = L(L-1)/2. The dense path stores L(L+1)/2 blocks total (diagonal +
strict-upper-triangle), exactly half what a (L, L, F, D) tensor with a mask
would carry. The low-rank path further drops off-diagonal params by F·D /
(F+D)·r ≈ 28× at F=8K, D=1K, r=32.

Per-feature thresholds θ live in log space (always positive) and are trained
via STE backward through the JumpReLU forward. The tanh penalty is smooth
everywhere. θ is a model parameter, not derived from batch statistics, so the
gate is identical at training and inference — no calibration step.

References: Ameisen et al. "Circuit Tracing" (Anthropic, 2025); Rajamanoharan
et al. "JumpReLU SAEs" (GDM, 2024). BatchTopK (Bussmann et al. 2024) was
considered and rejected for clinical activation heterogeneity: top-K methods
select on raw magnitude across a pool, which suppresses subtle features
regardless of frequency. Per-feature θ decouples sparsity from magnitude so
rare-subtle features can survive on their own scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from einops import rearrange

from genterp.modeling import Genterp, MarkedTPPHead


@dataclass
class CLTConfig:
    n_layers: int
    dim: int
    n_features: int
    off_diagonal_rank: int | None = None
    init_log_threshold: float = -4.6   # ≈ log(0.01)
    jumprelu_bandwidth_frac: float = 0.1
    sparsity_coef: float = 1e-3
    sparsity_tanh_contribution_scale: float = 1.0
    activation_std_momentum: float = 0.99
    activation_std_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.jumprelu_bandwidth_frac <= 0:
            raise ValueError("jumprelu_bandwidth_frac must be > 0")
        if self.off_diagonal_rank is not None and self.off_diagonal_rank <= 0:
            raise ValueError("off_diagonal_rank must be a positive int or None")


@dataclass(frozen=True)
class FeatureActivationWindow:
    layer: int
    feature: int
    activation: float
    batch_index: int
    token_index: int
    window_start: int
    event_atoms: torch.Tensor
    target_atoms: torch.Tensor
    event_ages: torch.Tensor
    event_values: torch.Tensor


@dataclass(frozen=True)
class FeatureGraphEdge:
    source_layer: int
    source_feature: int
    target_layer: int
    target_feature: int
    weight: float


class _JumpReLU(torch.autograd.Function):
    """Forward: x * 1[x > theta]. Backward: pass-through on x, rectangular STE on theta."""

    @staticmethod
    def forward(ctx, x, theta, bandwidth):
        ctx.save_for_backward(x, theta, bandwidth)
        return x * (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, bandwidth = ctx.saved_tensors
        grad_x = grad_output * (x > theta).to(grad_output.dtype)
        in_band = ((x - theta).abs() < bandwidth / 2).to(grad_output.dtype)
        grad_theta = -grad_output * theta * in_band / bandwidth
        while grad_theta.dim() > theta.dim():
            grad_theta = grad_theta.sum(0)
        return grad_x, grad_theta, None


def _strict_upper_pairs(n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (source_idx, target_idx) for the strict upper triangle (t > s)."""
    pairs = [(s, t) for s in range(n_layers) for t in range(s + 1, n_layers)]
    if not pairs:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    s_list, t_list = zip(*pairs)
    return torch.tensor(s_list, dtype=torch.long), torch.tensor(t_list, dtype=torch.long)


class CrossLayerTranscoder(nn.Module):
    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        L, D, Fdim = cfg.n_layers, cfg.dim, cfg.n_features

        self.enc_weight = nn.Parameter(torch.empty(L, Fdim, D))
        self.enc_bias = nn.Parameter(torch.zeros(L, Fdim))
        nn.init.kaiming_uniform_(self.enc_weight, nonlinearity="relu")

        self.diag_W = nn.Parameter(torch.empty(L, Fdim, D))
        self.dec_bias = nn.Parameter(torch.zeros(L, D))
        with torch.no_grad():
            for s in range(L):
                nn.init.normal_(self.diag_W[s], std=0.02 / math.sqrt(max(L - s, 1)))

        s_idx, t_idx = _strict_upper_pairs(L)
        self.register_buffer("_off_s_idx", s_idx, persistent=False)
        self.register_buffer("_off_t_idx", t_idx, persistent=False)
        n_off = int(s_idx.numel())

        if cfg.off_diagonal_rank is None:
            self.off_blocks = nn.Parameter(torch.empty(n_off, Fdim, D)) if n_off > 0 else None
            if self.off_blocks is not None:
                with torch.no_grad():
                    for pair, src in enumerate(s_idx.tolist()):
                        nn.init.normal_(self.off_blocks[pair], std=0.02 / math.sqrt(max(L - src, 1)))
            self.off_U = None
            self.off_V = None
        else:
            r = cfg.off_diagonal_rank
            self.off_blocks = None
            self.off_U = nn.Parameter(torch.empty(n_off, Fdim, r)) if n_off > 0 else None
            self.off_V = nn.Parameter(torch.empty(L, r, D))
            if self.off_U is not None:
                with torch.no_grad():
                    for pair, src in enumerate(s_idx.tolist()):
                        nn.init.normal_(self.off_U[pair], std=0.02 / math.sqrt(max(L - src, 1)))
            with torch.no_grad():
                # E[UV]=0, Var[(UV)_ij] = r·σ_U²·σ_V² = σ_U² when σ_V = 1/√r.
                nn.init.normal_(self.off_V, std=1.0 / math.sqrt(r))

        self.register_buffer("per_layer_std", torch.ones(L))
        self.register_buffer("per_layer_std_initialized", torch.tensor(False))

        self.log_threshold = nn.Parameter(torch.full((L, Fdim), cfg.init_log_threshold))

    def _preact(self, pre_mlp: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nld,lfd->nlf", pre_mlp, self.enc_weight) + self.enc_bias

    def _theta_like(self, ref: torch.Tensor) -> torch.Tensor:
        return self.log_threshold.exp().to(ref.dtype)

    def _bandwidth_like(self, theta: torch.Tensor) -> torch.Tensor:
        return theta.detach() * self.cfg.jumprelu_bandwidth_frac

    def encode(self, pre_mlp: torch.Tensor) -> torch.Tensor:
        preact = self._preact(pre_mlp).relu()
        theta = self._theta_like(preact)
        return _JumpReLU.apply(preact, theta, self._bandwidth_like(theta))

    def _off_diagonal_contribution(self, features: torch.Tensor) -> torch.Tensor:
        """Return (n, L, D) — sum of cross-layer writes into each target layer."""
        L = self.cfg.n_layers
        n = features.shape[0]
        accum = features.new_zeros(n, L, self.cfg.dim)
        if self._off_s_idx.numel() == 0:
            return accum

        feats_at_s = features.index_select(1, self._off_s_idx)  # (n, N_off, F)
        if self.off_blocks is not None:
            # Dense path: (n, N_off, F) ⨯ (N_off, F, D) → (n, N_off, D)
            out_per_pair = torch.einsum("npf,pfd->npd", feats_at_s, self.off_blocks)
        else:
            assert self.off_U is not None and self.off_V is not None
            # Low-rank path: (feats @ U) @ V[s].
            tmp = torch.einsum("npf,pfr->npr", feats_at_s, self.off_U)         # (n, N_off, r)
            v_per_pair = self.off_V.index_select(0, self._off_s_idx)           # (N_off, r, D)
            out_per_pair = torch.einsum("npr,prd->npd", tmp, v_per_pair)        # (n, N_off, D)

        # Under autocast, the matmul ops above downcast to fp16/bf16 while
        # accum keeps the dtype of `features` (which is fp32 — the JumpReLU
        # custom autograd Function isn't autocast-aware and returns fp32).
        # index_add_ requires matching dtypes; cast the source to match accum.
        accum.index_add_(1, self._off_t_idx, out_per_pair.to(accum.dtype))
        return accum

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        diag = torch.einsum("nlf,lfd->nld", features, self.diag_W)
        return diag + self._off_diagonal_contribution(features) + self.dec_bias

    def _diagonal_decoder_norm(self) -> torch.Tensor:
        """L2 norm of each same-layer decoder vector, shape (L, F)."""
        return self.diag_W.float().norm(dim=-1)

    def _sparsity(self, features: torch.Tensor) -> torch.Tensor:
        per_layer_std = self.per_layer_std.view(1, self.cfg.n_layers, 1)
        contribution = features.float() * self._diagonal_decoder_norm().view(
            1, self.cfg.n_layers, self.cfg.n_features
        )
        normalized_contribution = contribution / per_layer_std
        return (
            torch.tanh(normalized_contribution / self.cfg.sparsity_tanh_contribution_scale)
            .sum(dim=(-2, -1))
            .mean()
        )

    def forward(self, pre_mlp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(pre_mlp)
        return self.decode(features), features

    @torch.no_grad()
    def _update_per_layer_std(self, mlp_out: torch.Tensor) -> None:
        batch_std = mlp_out.detach().float().std(dim=(0, 2), unbiased=False).clamp_min(self.cfg.activation_std_eps)
        if not bool(self.per_layer_std_initialized.item()):
            self.per_layer_std.copy_(batch_std)
            self.per_layer_std_initialized.fill_(True)
            return
        self.per_layer_std.mul_(self.cfg.activation_std_momentum).add_(batch_std, alpha=1.0 - self.cfg.activation_std_momentum)

    def loss(self, pre_mlp: torch.Tensor, mlp_out: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            self._update_per_layer_std(mlp_out)

        features = self.encode(pre_mlp)
        recon = self.decode(features)

        per_layer_std = self.per_layer_std.view(1, self.cfg.n_layers, 1)
        recon_err = ((recon.float() - mlp_out.float()) / per_layer_std).pow(2).sum(dim=-1).mean()
        sparsity = self._sparsity(features)
        total = recon_err + self.cfg.sparsity_coef * sparsity

        with torch.no_grad():
            n_active = (features > 0).float().sum(dim=-1).mean()
        return {"loss": total, "recon": recon_err, "sparsity": sparsity, "n_active": n_active}


def _event_window(batch: dict, batch_index: int, token_index: int, radius: int) -> tuple[int, int]:
    event_pad = batch["event_pad"][batch_index]
    valid = (~event_pad).nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return 0, 0
    first = int(valid[0].item())
    last_exclusive = int(valid[-1].item()) + 1
    return max(first, token_index - radius), min(last_exclusive, token_index + radius + 1)


@torch.no_grad()
def top_activating_examples(
    model: nn.Module,
    clt: CrossLayerTranscoder,
    batch: dict,
    *,
    k: int = 10,
    window_radius: int = 4,
    layer: int | None = None,
    feature: int | None = None,
) -> list[FeatureActivationWindow]:
    """Return event windows around the strongest CLT feature activations."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if window_radius < 0:
        raise ValueError("window_radius must be >= 0")
    if (layer is None) != (feature is None):
        raise ValueError("layer and feature must be provided together")

    base_model = unwrap_genterp_model(model)
    was_training = base_model.training
    base_model.eval()
    try:
        out = base_model(**batch, return_transcoder_acts=True)
        pre_mlp = rearrange(out["pre_mlp"], "b l t d -> (b t) l d")
        B, _, T, _ = out["pre_mlp"].shape
        features = rearrange(clt.encode(pre_mlp), "(b t) l f -> b t l f", b=B, t=T)
    finally:
        base_model.train(was_training)

    valid = ~batch["event_pad"].to(features.device)
    masked = features.masked_fill(~valid[:, :, None, None], float("-inf"))
    windows: list[FeatureActivationWindow] = []

    if layer is not None and feature is not None:
        if not (0 <= layer < clt.cfg.n_layers):
            raise ValueError("layer is out of range")
        if not (0 <= feature < clt.cfg.n_features):
            raise ValueError("feature is out of range")
        flat_scores = masked[:, :, layer, feature].reshape(-1)
        top_values, top_indices = torch.topk(flat_scores, min(k, flat_scores.numel()))
        layer_feature_indices = [(layer, feature, top_values, top_indices)]
    else:
        flat = rearrange(masked, "b t l f -> l f (b t)")
        top_values, top_indices = torch.topk(flat, min(k, flat.shape[-1]), dim=-1)
        layer_feature_indices = [
            (l, f, top_values[l, f], top_indices[l, f])
            for l in range(clt.cfg.n_layers)
            for f in range(clt.cfg.n_features)
        ]

    for l, f, values, indices in layer_feature_indices:
        for value, flat_index in zip(values.detach().cpu(), indices.detach().cpu(), strict=True):
            activation = float(value.item())
            if not math.isfinite(activation):
                continue
            batch_index = int(flat_index.item()) // T
            token_index = int(flat_index.item()) % T
            start, stop = _event_window(batch, batch_index, token_index, window_radius)
            windows.append(
                FeatureActivationWindow(
                    layer=int(l),
                    feature=int(f),
                    activation=activation,
                    batch_index=batch_index,
                    token_index=token_index,
                    window_start=start,
                    event_atoms=batch["event_atoms"][batch_index, start:stop].detach().cpu(),
                    target_atoms=batch["target_atoms"][batch_index, start:stop].detach().cpu(),
                    event_ages=batch["event_ages"][batch_index, start:stop].detach().cpu(),
                    event_values=batch["event_values"][batch_index, start:stop].detach().cpu(),
                )
            )
    return windows


def _as_index_tensor(indices: Iterable[int] | torch.Tensor | None, default: torch.Tensor, device: torch.device) -> torch.Tensor:
    if indices is None:
        return default.to(device=device, dtype=torch.long)
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.long)
    return torch.tensor(list(indices), device=device, dtype=torch.long)


def _feature_hidden_effects(clt: CrossLayerTranscoder, ref: torch.Tensor) -> torch.Tensor:
    effects = clt.diag_W.detach().to(device=ref.device, dtype=ref.dtype).clone()
    for source_layer, target_layer in zip(clt._off_s_idx.tolist(), clt._off_t_idx.tolist(), strict=True):
        effects[source_layer] = effects[source_layer] + _decoder_block(clt, source_layer, target_layer).detach().to(
            device=ref.device,
            dtype=ref.dtype,
        )
    return effects


@torch.no_grad()
def feature_to_output_attribution(
    clt: CrossLayerTranscoder,
    tpp: MarkedTPPHead,
    hidden: torch.Tensor,
    pre_mlp: torch.Tensor,
    delta_t: torch.Tensor,
    *,
    mark_indices: Iterable[int] | torch.Tensor | None = None,
    time_indices: Iterable[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Exact gradient and activation-gradient attribution from CLT features to mark/time logits."""
    features = clt.encode(pre_mlp.detach())
    hidden_ref = hidden.detach()
    mark_logits = tpp.mark_logits(hidden_ref, delta_t)
    time_logits = tpp.time_logits(hidden_ref)
    mark_idx = _as_index_tensor(mark_indices, mark_logits.argmax(dim=-1).unique(sorted=True), hidden.device)
    time_idx = _as_index_tensor(time_indices, torch.arange(time_logits.shape[-1], device=hidden.device), hidden.device)

    feature_effects = _feature_hidden_effects(clt, hidden_ref)
    mark_weight = tpp.mark_out.weight.index_select(0, mark_idx).to(device=hidden.device, dtype=hidden.dtype)
    mark_hidden_grad = mark_weight @ tpp.mark_h_proj.weight.to(device=hidden.device, dtype=hidden.dtype)
    time_hidden_grad = tpp.time_proj.weight.index_select(0, time_idx).to(device=hidden.device, dtype=hidden.dtype)
    mark_grad_one = torch.einsum("lfd,md->lfm", feature_effects, mark_hidden_grad)
    time_grad_one = torch.einsum("lfd,td->lft", feature_effects, time_hidden_grad)
    mark_grad = mark_grad_one.unsqueeze(0).expand(features.shape[0], -1, -1, -1)
    time_grad = time_grad_one.unsqueeze(0).expand(features.shape[0], -1, -1, -1)
    feature_values = features.detach()
    return {
        "features": feature_values,
        "mark_indices": mark_idx.detach(),
        "time_indices": time_idx.detach(),
        "mark_grad": mark_grad,
        "time_grad": time_grad,
        "mark_activation_attribution": mark_grad * feature_values.unsqueeze(-1),
        "time_activation_attribution": time_grad * feature_values.unsqueeze(-1),
    }


def _decoder_block(clt: CrossLayerTranscoder, source_layer: int, target_layer: int) -> torch.Tensor:
    if source_layer == target_layer:
        return clt.diag_W[source_layer]
    pair = ((clt._off_s_idx == source_layer) & (clt._off_t_idx == target_layer)).nonzero(as_tuple=False)
    if pair.numel() != 1:
        raise ValueError("target_layer must be >= source_layer")
    pair_idx = int(pair.item())
    if clt.off_blocks is not None:
        return clt.off_blocks[pair_idx]
    assert clt.off_U is not None and clt.off_V is not None
    return clt.off_U[pair_idx] @ clt.off_V[source_layer]


@torch.no_grad()
def feature_to_feature_attribution_graph(
    clt: CrossLayerTranscoder,
    *,
    top_k_per_layer_pair: int = 64,
    min_abs_weight: float = 0.0,
    include_same_layer: bool = False,
) -> list[FeatureGraphEdge]:
    """Decoder-mediated feature graph: earlier source features -> later target feature preactivations."""
    if top_k_per_layer_pair <= 0:
        raise ValueError("top_k_per_layer_pair must be > 0")
    edges: list[FeatureGraphEdge] = []
    enc = clt.enc_weight.detach().float()
    start_offset = 0 if include_same_layer else 1
    for source_layer in range(clt.cfg.n_layers):
        for target_layer in range(source_layer + start_offset, clt.cfg.n_layers):
            decoder = _decoder_block(clt, source_layer, target_layer).detach().float()
            scores = decoder @ enc[target_layer].T
            flat = scores.flatten()
            if min_abs_weight > 0:
                keep = flat.abs() >= min_abs_weight
                if not keep.any():
                    continue
                kept_indices = keep.nonzero(as_tuple=False).flatten()
                kept_values = flat[kept_indices]
                take = min(top_k_per_layer_pair, kept_values.numel())
                _, order = torch.topk(kept_values.abs(), take)
                top_indices = kept_indices[order]
            else:
                take = min(top_k_per_layer_pair, flat.numel())
                _, top_indices = torch.topk(flat.abs(), take)
            for flat_index in top_indices.tolist():
                source_feature = flat_index // clt.cfg.n_features
                target_feature = flat_index % clt.cfg.n_features
                edges.append(
                    FeatureGraphEdge(
                        source_layer=source_layer,
                        source_feature=source_feature,
                        target_layer=target_layer,
                        target_feature=target_feature,
                        weight=float(scores[source_feature, target_feature].item()),
                    )
                )
    edges.sort(key=lambda edge: abs(edge.weight), reverse=True)
    return edges


def unwrap_genterp_model(model: nn.Module) -> Genterp:
    """Return the inner :class:`Genterp` from a base model or training wrapper.

    Training checkpoints are saved as ``GenterpForCausalLM`` instances whose
    ``.model`` attribute is the actual ``Genterp``. CLT harvesting operates on
    that inner model because it needs ``return_transcoder_acts=True``, which is
    implemented by ``Genterp.forward`` rather than the Hugging Face wrapper.
    """
    if isinstance(model, Genterp):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, Genterp):
        return inner
    raise TypeError(f"expected Genterp or wrapper with .model: Genterp, got {type(model).__name__}")


@torch.no_grad()
def harvest_transcoder_acts(model: nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen Genterp and return CLT training activations.

    Returns ``(pre_mlp, mlp_out)``, each shaped ``(n_real_tokens, L, D)``.

    ``pre_mlp`` is the residual stream after the attention sublayer has been
    added back into the block residual, immediately before ``norm2`` and the
    SwiGLU MLP. This is the stream the CLT encoder reads, so a CLT "feature" is
    a direction in the pre-MLP residual stream, not in the normalized MLP input.

    ``mlp_out`` is the additive MLP output for the same block and token. The CLT
    decoder learns to reconstruct these per-layer MLP writes from the harvested
    pre-MLP residual features.
    """
    base_model = unwrap_genterp_model(model)
    was_training = base_model.training
    base_model.eval()
    try:
        out = base_model(**batch, return_transcoder_acts=True)
        keep = ~batch["event_pad"].reshape(-1)
        pre_mlp_flat = rearrange(out["pre_mlp"], "b l t d -> (b t) l d")[keep]
        mlp_out_flat = rearrange(out["mlp_out"], "b l t d -> (b t) l d")[keep]
        return pre_mlp_flat, mlp_out_flat
    finally:
        base_model.train(was_training)
