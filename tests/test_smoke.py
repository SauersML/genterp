"""Forward + backward through Genterp with joint marked-TPP loss; dict return; transcoder acts."""

from __future__ import annotations

import torch

from genterp import Genterp, marked_tpp_loss
from tests._factories import make_batch, tiny_config


def test_forward_backward():
    cfg = tiny_config()
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)

    out = model(**batch)
    B, T = batch["event_ages"].shape
    assert out["hidden"].shape == (B, T, cfg.dim)

    ld = marked_tpp_loss(
        model.tpp, out["hidden"], batch["event_ages"], batch["target_atoms"], batch["event_pad"], batch["censor_age"]
    )
    assert torch.isfinite(ld["loss"])
    ld["loss"].backward()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 0


def test_transcoder_acts():
    cfg = tiny_config()
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)

    out = model(**batch, return_transcoder_acts=True)
    B, T = batch["event_ages"].shape
    assert out["hidden"].shape == (B, T, cfg.dim)
    assert out["pre_mlp"].shape == (B, cfg.n_layers, T, cfg.dim)
    assert out["mlp_out"].shape == (B, cfg.n_layers, T, cfg.dim)
    assert torch.isfinite(out["pre_mlp"]).all()
    assert torch.isfinite(out["mlp_out"]).all()


def test_tpp_sample():
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    batch = make_batch(n_atoms=cfg.n_atoms)
    with torch.no_grad():
        out = model(**batch)
        delta_t, mark = model.tpp.sample(out["hidden"][:, -1])
    assert delta_t.shape == (out["hidden"].shape[0],)
    assert mark.shape == (out["hidden"].shape[0],)
    assert (delta_t > 0).all()
    assert (mark >= 0).all() and (mark < cfg.n_atoms).all()
