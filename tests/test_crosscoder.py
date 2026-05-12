"""Harvest residuals from frozen Genterp, train multi-layer crosscoder, verify recon + sparsity."""

from __future__ import annotations

import torch

from genterp import CrosscoderConfig, Genterp, MultiLayerCrosscoder, harvest_activations
from tests._factories import make_batch, tiny_config


def test_crosscoder_training():
    torch.manual_seed(0)
    cfg = tiny_config(n_layers=3)
    base = Genterp(cfg)
    for p in base.parameters():
        p.requires_grad_(False)

    batch = make_batch(B=4, T=24, n_atoms=cfg.n_atoms)
    acts = harvest_activations(base, batch)
    n_tokens, n_layers, dim = acts.shape
    assert n_layers == cfg.n_layers and dim == cfg.dim
    assert n_tokens > 0

    cc = MultiLayerCrosscoder(CrosscoderConfig(n_layers=n_layers, dim=dim, n_features=128, l1_coef=1e-3))
    opt = torch.optim.Adam(cc.parameters(), lr=1e-2)

    init_recon = cc.loss(acts)["recon"].item()
    for _ in range(50):
        out = cc.loss(acts)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    final = cc.loss(acts)
    assert final["recon"].item() < init_recon
    assert final["n_active"].item() < cc.cfg.n_features
