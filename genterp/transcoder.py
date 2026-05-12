"""Cross-Layer Transcoder (CLT) with JumpReLU activations.

For each layer ℓ ∈ [0, L):
    features[n, ℓ] = JumpReLU(W_enc[ℓ] @ pre_mlp[n, ℓ] + b_enc[ℓ];  θ[ℓ])
For each output layer t ∈ [0, L):
    mlp_out_recon[n, t] = Σ_{ℓ ≤ t} W_dec[ℓ → t] @ features[n, ℓ] + b_dec[t]

Loss = Σ_t MSE(recon[n, t], mlp_out[n, t])  +  l0_coef * L0_surrogate(features)

The L0 surrogate uses a Heaviside step function with a rectangular STE; threshold
parameters are stored in log space so they stay positive. References: Lindsey,
Pearce et al. "Circuit Tracing" (Anthropic, 2025); Rajamanoharan et al. "JumpReLU
SAEs" (GDM, 2024).
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
    l0_coef: float = 1e-3


class _JumpReLU(torch.autograd.Function):
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


class _HeavisideSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, bandwidth):
        ctx.save_for_backward(x, theta)
        ctx.bandwidth = bandwidth
        return (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        in_band = ((x - theta).abs() < bw / 2).to(grad_output.dtype)
        grad_x = grad_output * in_band / bw
        grad_theta = -grad_output * in_band / bw
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

    def loss(self, pre_mlp: torch.Tensor, mlp_out: torch.Tensor) -> dict[str, torch.Tensor]:
        preact = self._preact(pre_mlp)
        theta = self._theta_like(preact)
        features = _JumpReLU.apply(preact, theta, self.cfg.jumprelu_bandwidth)
        recon = self.decode(features)

        recon_err = (recon.float() - mlp_out.float()).pow(2).sum(dim=-1).mean()
        active = _HeavisideSTE.apply(preact, theta, self.cfg.jumprelu_bandwidth)
        l0 = active.sum(dim=(-2, -1)).mean()
        total = recon_err + self.cfg.l0_coef * l0

        with torch.no_grad():
            n_active_per_layer = active.sum(dim=-1).mean()
        return {"loss": total, "recon": recon_err, "l0": l0, "n_active": n_active_per_layer}


@torch.no_grad()
def harvest_transcoder_acts(model: Genterp, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run base model frozen, return (pre_mlp, mlp_out), each (n_real_tokens, L, D)."""
    was_training = model.training
    model.eval()
    try:
        _, pre_mlp, mlp_out = model(**batch, return_transcoder_acts=True)
        keep = ~batch["event_pad"].reshape(-1)
        pre_mlp_flat = rearrange(pre_mlp, "b l t d -> (b t) l d")[keep]
        mlp_out_flat = rearrange(mlp_out, "b l t d -> (b t) l d")[keep]
        return pre_mlp_flat, mlp_out_flat
    finally:
        model.train(was_training)
