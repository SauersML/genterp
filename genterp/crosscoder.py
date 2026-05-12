"""Multi-layer single sparse crosscoder (Lindsey et al. 2024)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from genterp.modeling import Genterp


@dataclass
class CrosscoderConfig:
    n_layers: int
    dim: int
    n_features: int
    l1_coef: float = 1.0
    init_dec_norm: float = 0.1


class MultiLayerCrosscoder(nn.Module):
    """Shared encoder over concat of L layer activations; per-layer decoder; L1 weighted by sum_l ||W_dec[l, f, :]||_2."""

    def __init__(self, cfg: CrosscoderConfig):
        super().__init__()
        self.cfg = cfg
        L, D, Fdim = cfg.n_layers, cfg.dim, cfg.n_features

        self.enc_weight = nn.Parameter(torch.empty(Fdim, L * D))
        self.enc_bias = nn.Parameter(torch.zeros(Fdim))
        nn.init.kaiming_uniform_(self.enc_weight, nonlinearity="relu")

        self.dec_weight = nn.Parameter(torch.empty(L, Fdim, D))
        self.dec_bias = nn.Parameter(torch.zeros(L, D))
        nn.init.kaiming_uniform_(self.dec_weight, nonlinearity="linear")
        with torch.no_grad():
            per_feat = self.dec_weight.pow(2).sum(dim=(0, 2)).sqrt().clamp(min=1e-6)
            self.dec_weight.mul_((cfg.init_dec_norm / per_feat)[None, :, None])

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        flat = rearrange(acts, "n l d -> n (l d)")
        return F.relu(F.linear(flat, self.enc_weight, self.enc_bias))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nf,lfd->nld", features, self.dec_weight) + self.dec_bias

    def forward(self, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(acts)
        return self.decode(features), features

    def loss(self, acts: torch.Tensor) -> dict[str, torch.Tensor]:
        recon, features = self.forward(acts)
        recon_err = (recon.float() - acts.float()).pow(2).sum(dim=-1).mean()
        feature_weight = self.dec_weight.float().norm(dim=-1).sum(dim=0)
        l1 = (features.float() * feature_weight).sum(dim=-1).mean()
        total = recon_err + self.cfg.l1_coef * l1
        with torch.no_grad():
            n_active = (features > 0).float().sum(dim=-1).mean()
        return {"loss": total, "recon": recon_err, "l1": l1, "n_active": n_active}


@torch.no_grad()
def harvest_activations(model: Genterp, batch: dict) -> torch.Tensor:
    """Run base model frozen, return (n_real_tokens, L, D) residual-stream activations."""
    was_training = model.training
    model.eval()
    try:
        _, hidden = model(**batch, return_hidden_states=True)
        keep = ~batch["event_pad"].reshape(-1)
        return rearrange(hidden, "b l t d -> (b t) l d")[keep]
    finally:
        model.train(was_training)
