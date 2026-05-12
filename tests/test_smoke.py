"""Base model forward + backward + hidden states."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from genterp import Genterp
from tests._factories import make_batch, tiny_config


def test_forward_backward():
    cfg = tiny_config()
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)

    logits = model(**batch)
    B, T = batch["event_ages"].shape
    assert logits.shape == (B, T, cfg.n_atoms)

    target = torch.randint(0, cfg.n_atoms, (B, T))
    pad = batch["event_pad"][:, 1:]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.n_atoms), target[:, 1:].reshape(-1), reduction="none").view(pad.shape)
    loss = ce.masked_fill(pad, 0).sum() / (~pad).sum().clamp(min=1)
    assert torch.isfinite(loss)
    loss.backward()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 0


def test_hidden_states():
    cfg = tiny_config()
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)

    logits, hidden = model(**batch, return_hidden_states=True)
    B, T = batch["event_ages"].shape
    assert hidden.shape == (B, cfg.n_layers, T, cfg.dim)
    assert torch.isfinite(hidden).all()
    assert logits.shape == (B, T, cfg.n_atoms)
