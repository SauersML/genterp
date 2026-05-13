"""Forward + backward through Genterp with joint TPP + value loss; dict return; transcoder acts; sampling."""

from __future__ import annotations

import torch

from genterp import Genterp
from tests._factories import make_batch, tiny_config


def _mark_some_atoms_magnitude(model: Genterp, frac: float = 0.5) -> None:
    """Pretend a fraction of atoms are magnitude-bearing so the value pathway gets exercised."""
    n = model.cfg.n_atoms
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.randperm(n)[: int(frac * n)]] = True
    mask[0] = False
    model.value_mod.set_stats(
        value_mu=torch.zeros(n),
        value_sigma=torch.ones(n),
        atom_has_mag=mask,
    )


def test_forward_backward():
    cfg = tiny_config()
    model = Genterp(cfg)
    _mark_some_atoms_magnitude(model)
    batch = make_batch(n_atoms=cfg.n_atoms)

    out = model(**batch)
    B, T = batch["event_ages"].shape
    assert out["hidden"].shape == (B, T, cfg.dim)

    ld = model.loss(**batch)
    assert torch.isfinite(ld["loss"])
    assert ld["n_mag"].item() > 0
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


def test_value_head_sample():
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    _mark_some_atoms_magnitude(model)
    batch = make_batch(n_atoms=cfg.n_atoms)
    with torch.no_grad():
        out = model(**batch)
        leaf = batch["target_atoms"][:, -1].clamp(min=0)
        z = model.value_head.sample(out["hidden"][:, -1], model.embed.weight[leaf])
    assert z.shape == (out["hidden"].shape[0],)
    assert torch.isfinite(z).all()
