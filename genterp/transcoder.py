"""Cross-Layer Transcoder (CLT) with JumpReLU activations + tanh sparsity penalty.

For each layer ℓ ∈ [0, L):
    preact[n, ℓ]   = W_enc[ℓ] @ pre_mlp[n, ℓ] + b_enc[ℓ]
    features[n, ℓ] = ReLU(preact) * (ReLU(preact) > θ[ℓ]) # JumpReLU, per-feature θ
For each output layer t ∈ [0, L):
    recon[n, t]    = Σ_{ℓ ≤ t} W_dec[ℓ → t] @ features[n, ℓ] + b_dec[t]

Loss = Σ_t MSE((recon - mlp_out) / σ_t)
     + sparsity_coef · Σ tanh(features · ||W_dec[ℓ → ℓ]||₂ / (σ_ℓ · scale))

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

import torch
import torch.nn as nn
from einops import rearrange

from genterp.modeling import Genterp


@dataclass
class CLTConfig:
    n_layers: int
    dim: int
    n_features: int
    init_log_threshold: float = -4.6   # ≈ log(0.01)
    jumprelu_bandwidth_frac: float = 0.1
    sparsity_coef: float = 1e-3
    sparsity_tanh_contribution_scale: float = 1.0
    activation_std_momentum: float = 0.99
    activation_std_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.jumprelu_bandwidth_frac <= 0:
            raise ValueError("jumprelu_bandwidth_frac must be > 0")


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


class CrossLayerTranscoder(nn.Module):
    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        L, D, Fdim = cfg.n_layers, cfg.dim, cfg.n_features

        self.enc_weight = nn.Parameter(torch.empty(L, Fdim, D))
        self.enc_bias = nn.Parameter(torch.zeros(L, Fdim))
        nn.init.kaiming_uniform_(self.enc_weight, nonlinearity="relu")

        self.dec_weight = nn.Parameter(torch.zeros(L, L, Fdim, D))
        self.dec_bias = nn.Parameter(torch.zeros(L, D))
        with torch.no_grad():
            for s in range(L):
                for t in range(s, L):
                    nn.init.normal_(self.dec_weight[s, t], std=0.02 / math.sqrt(max(L - s, 1)))

        # dec_mask[s, t] = 1 if t >= s, else 0 — gradient flows only on (t >= s) entries.
        self.register_buffer("dec_mask", torch.triu(torch.ones(L, L))[..., None, None], persistent=False)
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

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        masked = self.dec_weight * self.dec_mask
        return torch.einsum("nsf,stfd->ntd", features, masked) + self.dec_bias

    def _diagonal_decoder_norm(self) -> torch.Tensor:
        layer_idx = torch.arange(self.cfg.n_layers, device=self.dec_weight.device)
        return self.dec_weight.float()[layer_idx, layer_idx].norm(dim=-1)

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
