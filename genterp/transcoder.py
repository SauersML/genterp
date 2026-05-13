"""Cross-Layer Transcoder (CLT) with JumpReLU activations + tanh sparsity penalty.

For each layer ℓ ∈ [0, L):
    preact[n, ℓ]   = W_enc[ℓ] @ pre_mlp[n, ℓ] + b_enc[ℓ]
    features[n, ℓ] = preact * (preact > θ[ℓ])             # JumpReLU, per-feature θ
For each output layer t ∈ [0, L):
    recon[n, t]    = Σ_{ℓ ≤ t} W_dec[ℓ → t] @ features[n, ℓ] + b_dec[t]

Loss = Σ_t MSE((recon - mlp_out) / σ_t)  +  sparsity_coef · Σ tanh(features / scale)

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
    jumprelu_bandwidth: float = 1e-3
    sparsity_coef: float = 1e-3
    sparsity_tanh_scale: float = 1.0
    activation_std_momentum: float = 0.99
    activation_std_eps: float = 1e-6


class _JumpReLU(torch.autograd.Function):
    """Forward: x · 𝟙[x > θ]. Backward: pass-through where active, rectangular STE on θ."""

    @staticmethod
    def forward(ctx, x, theta, bandwidth):
        ctx.save_for_backward(x, theta)
        ctx.bandwidth = bandwidth
        return x * (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        grad_x = grad_output * (x > theta).to(grad_output.dtype)
        in_band = ((x - theta).abs() < bw / 2).to(grad_output.dtype)
        grad_theta = -grad_output * x * in_band / bw
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

    def encode(self, pre_mlp: torch.Tensor) -> torch.Tensor:
        preact = self._preact(pre_mlp)
        return _JumpReLU.apply(preact, self._theta_like(preact), self.cfg.jumprelu_bandwidth)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        masked = self.dec_weight * self.dec_mask
        return torch.einsum("nsf,stfd->ntd", features, masked) + self.dec_bias

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
        self._update_per_layer_std(mlp_out)

        features = self.encode(pre_mlp)
        recon = self.decode(features)

        per_layer_std = self.per_layer_std.view(1, self.cfg.n_layers, 1)
        recon_err = ((recon.float() - mlp_out.float()) / per_layer_std).pow(2).sum(dim=-1).mean()
        sparsity = torch.tanh(features / self.cfg.sparsity_tanh_scale).sum(dim=(-2, -1)).mean()
        total = recon_err + self.cfg.sparsity_coef * sparsity

        with torch.no_grad():
            n_active = (features > 0).float().sum(dim=-1).mean()
        return {"loss": total, "recon": recon_err, "sparsity": sparsity, "n_active": n_active}


@torch.no_grad()
def harvest_transcoder_acts(model: Genterp, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run base model frozen, return (pre_mlp, mlp_out), each (n_real_tokens, L, D)."""
    was_training = model.training
    model.eval()
    try:
        out = model(**batch, return_transcoder_acts=True)
        keep = ~batch["event_pad"].reshape(-1)
        pre_mlp_flat = rearrange(out["pre_mlp"], "b l t d -> (b t) l d")[keep]
        mlp_out_flat = rearrange(out["mlp_out"], "b l t d -> (b t) l d")[keep]
        return pre_mlp_flat, mlp_out_flat
    finally:
        model.train(was_training)
