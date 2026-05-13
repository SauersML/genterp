"""Cross-Layer Transcoder (CLT) with per-layer BatchTopK activations.

For each layer ℓ ∈ [0, L):
    features[n, ℓ] = BatchTopK_ℓ(ReLU(W_enc[ℓ] @ pre_mlp[n, ℓ] + b_enc[ℓ]))
For each output layer t ∈ [0, L):
    mlp_out_recon[n, t] = Σ_{ℓ ≤ t} W_dec[ℓ → t] @ features[n, ℓ] + b_dec[t]

Loss = Σ_t MSE((recon[n, t] - mlp_out[n, t]) / σ_t)

BatchTopK selects the top (k_per_token × n_tokens) preactivations per layer
across the whole batch. Average sparsity is exact at k_per_token features per
token per layer, with cross-token flexibility. After training,
fit_inference_threshold() records a per-layer threshold so encode() works on a
single sample at eval time. References: Bussmann et al. "BatchTopK SAEs"
(2024); Lindsey, Pearce et al. "Circuit Tracing" (Anthropic, 2025).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
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
    k_per_token: int = 32
    activation_std_momentum: float = 0.99
    activation_std_eps: float = 1e-6


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
        self.register_buffer("inference_threshold", torch.zeros(L))
        self.register_buffer("inference_threshold_initialized", torch.tensor(False))

    def _preact(self, pre_mlp: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nld,lfd->nlf", pre_mlp, self.enc_weight) + self.enc_bias

    def _batch_topk(self, relu: torch.Tensor) -> torch.Tensor:
        N, L, Fdim = relu.shape
        K = self.cfg.k_per_token
        if K <= 0 or N == 0:
            return torch.zeros_like(relu)
        k_total = min(K * N, N * Fdim)
        if k_total >= N * Fdim:
            return relu
        flat = relu.transpose(0, 1).reshape(L, N * Fdim)
        _, idx = flat.topk(k_total, dim=-1)
        mask = torch.zeros_like(flat).scatter_(-1, idx, 1.0)
        return (flat * mask).reshape(L, N, Fdim).transpose(0, 1).contiguous()

    def _threshold_gate(self, relu: torch.Tensor) -> torch.Tensor:
        theta = self.inference_threshold.view(1, -1, 1).to(relu.dtype)
        return relu * (relu > theta).to(relu.dtype)

    def encode(self, pre_mlp: torch.Tensor) -> torch.Tensor:
        relu = self._preact(pre_mlp).clamp_min(0)
        if self.training or not bool(self.inference_threshold_initialized.item()):
            return self._batch_topk(relu)
        return self._threshold_gate(relu)

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

        with torch.no_grad():
            n_active = (features > 0).float().sum(dim=-1).mean()
        return {"loss": recon_err, "recon": recon_err, "n_active": n_active}

    @torch.no_grad()
    def fit_inference_threshold(self, pre_mlp_batches: Iterable[torch.Tensor]) -> None:
        """Estimate per-layer threshold for batch-free inference. Records the mean
        (across supplied batches) of the k_total-th largest preactivation per layer."""
        K = self.cfg.k_per_token
        per_layer_kth: list[torch.Tensor] = []
        for pre_mlp in pre_mlp_batches:
            relu = self._preact(pre_mlp).clamp_min(0)
            N, L, Fdim = relu.shape
            if K <= 0 or N == 0:
                continue
            k_total = min(K * N, N * Fdim)
            if k_total >= N * Fdim:
                per_layer_kth.append(torch.zeros(L, dtype=torch.float32, device=relu.device))
                continue
            flat = relu.transpose(0, 1).reshape(L, N * Fdim)
            topk_vals, _ = flat.topk(k_total, dim=-1)
            per_layer_kth.append(topk_vals[:, -1].detach().float())
        if not per_layer_kth:
            return
        threshold = torch.stack(per_layer_kth).mean(dim=0)
        self.inference_threshold.copy_(threshold.to(self.inference_threshold.dtype))
        self.inference_threshold_initialized.fill_(True)


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
