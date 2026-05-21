"""Tests for the same-time bag-NLL auxiliary loss path.

Covers three things:

  1. ``_derive_event_groups`` produces dense per-subject group ids that
     agree with the contiguous-equal-timestamp partition.
  2. ``bag_nll_same_time`` produces a finite loss with non-zero gradient on
     a hand-built batch where group transitions exist, and short-circuits
     to a zero loss when no transitions are present.
  3. End-to-end: ``Genterp.loss`` returns identical totals (within float
     tolerance) for ``bag_loss_weight=0.0`` whether or not ``event_groups``
     is in the batch, i.e. enabling the feature is opt-in.
"""

from __future__ import annotations

import numpy as np
import torch

from genterp import Genterp, GenterpConfig
from genterp._synthetic import make_batch
from genterp.data import _derive_event_groups
from genterp.modeling import bag_nll_same_time


def test_derive_event_groups_dense():
    times = np.array([100, 100, 100, 200, 200, 300], dtype=np.int64)
    groups = _derive_event_groups(times)
    assert groups.tolist() == [0, 0, 0, 1, 1, 2]


def test_derive_event_groups_empty():
    out = _derive_event_groups(np.zeros(0, dtype=np.int64))
    assert out.shape == (0,)
    assert out.dtype == np.int32


def test_derive_event_groups_singleton():
    out = _derive_event_groups(np.array([42], dtype=np.int64))
    assert out.tolist() == [0]


def _bag_batch(B=2, T=8, dim=16, n_atoms=64, seed=0):
    """Build a tiny batch with two distinct time groups per subject."""
    torch.manual_seed(seed)
    hidden = torch.randn(B, T, dim, requires_grad=True)
    # Group layout: positions 0..3 are group 0, 4..7 are group 1.
    event_groups = torch.zeros(B, T, dtype=torch.long)
    event_groups[:, T // 2 :] = 1
    event_atoms = torch.randint(1, n_atoms, (B, T), dtype=torch.long)
    event_pad = torch.zeros(B, T, dtype=torch.bool)
    output_weight = torch.randn(n_atoms, dim) * 0.1
    noise = torch.ones(n_atoms)
    noise[0] = 0.0
    noise = noise / noise.sum()
    return hidden, event_atoms, event_groups, event_pad, output_weight, noise


def test_bag_nll_has_gradient_when_transitions_exist():
    hidden, atoms, groups, pad, w, noise = _bag_batch()
    loss, n_pred = bag_nll_same_time(
        hidden, atoms, groups, pad, w, noise, n_negatives=8
    )
    assert torch.isfinite(loss)
    assert int(n_pred.item()) == hidden.shape[0]  # one transition per subject
    loss.backward()
    # Some hidden positions must receive non-zero gradient (the ones acting
    # as predictors, i.e. last position of group 0 in each subject).
    grad_norms = hidden.grad.detach().norm(dim=-1)
    assert grad_norms.sum().item() > 0.0


def test_bag_nll_zero_when_no_transitions():
    hidden, atoms, _groups, pad, w, noise = _bag_batch()
    same_group = torch.zeros_like(_groups)  # one big group → no boundary
    loss, n_pred = bag_nll_same_time(
        hidden, atoms, same_group, pad, w, noise, n_negatives=8
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0
    assert int(n_pred.item()) == 0


def test_bag_loss_opt_in_does_not_change_baseline_total():
    """Without event_groups (or with weight 0.0) total loss must match the legacy path."""
    cfg = GenterpConfig(
        n_atoms=128,
        dim=64,
        n_heads=4,
        n_layers=2,
        n_static_blocks=1,
        k_static_summary=4,
        n_time_mix=4,
        time_phi_dim=16,
        bag_loss_weight=0.0,
    )
    model = Genterp(cfg).eval()
    batch = make_batch(B=2, M=3, T=10, n_atoms=cfg.n_atoms, seed=0)

    with torch.no_grad():
        no_groups = model.loss(**{k: v for k, v in batch.items() if k != "event_groups"})
        # Add groups but keep weight 0 → bag NLL must be a clean zero.
        groups = torch.zeros(2, 10, dtype=torch.long)
        groups[:, 5:] = 1
        with_groups = model.loss(event_groups=groups, **batch)

    assert torch.isfinite(no_groups["loss"])
    assert torch.isfinite(with_groups["loss"])
    assert torch.allclose(no_groups["loss"], with_groups["loss"], atol=1e-5)
    assert float(with_groups["bag_nll"].item()) == 0.0
    assert int(with_groups["n_bag_predictors"].item()) == 0


def test_bag_loss_adds_no_parameters():
    base = Genterp(GenterpConfig(n_atoms=16, dim=16, n_heads=2, n_layers=1))
    bag = Genterp(GenterpConfig(n_atoms=16, dim=16, n_heads=2, n_layers=1, bag_loss_weight=0.5))

    base_state = base.state_dict()
    bag_state = bag.state_dict()

    assert base_state.keys() == bag_state.keys()
    assert {k: tuple(v.shape) for k, v in base_state.items()} == {k: tuple(v.shape) for k, v in bag_state.items()}


def test_bag_loss_engages_when_weight_positive():
    cfg = GenterpConfig(
        n_atoms=128,
        dim=64,
        n_heads=4,
        n_layers=2,
        n_static_blocks=1,
        k_static_summary=4,
        n_time_mix=4,
        time_phi_dim=16,
        bag_loss_weight=0.5,
        bag_loss_negatives=8,
    )
    model = Genterp(cfg).train()
    batch = make_batch(B=2, M=3, T=10, n_atoms=cfg.n_atoms, seed=1)
    groups = torch.zeros(2, 10, dtype=torch.long)
    groups[:, 5:] = 1  # one transition per subject

    out = model.loss(event_groups=groups, **batch)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["bag_nll"])
    assert float(out["bag_nll"].item()) > 0.0
    assert int(out["n_bag_predictors"].item()) == 2  # B=2 predictors
    out["loss"].backward()
    grads_exist = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert grads_exist
