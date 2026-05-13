"""Collator contract: padded atom tensors, target_atoms, NaN-safe static pad, S>=1 invariant."""

from __future__ import annotations

import numpy as np
import torch

from genterp.data import AtomVocab, CodeAtomMap, _pad_atoms, collate


def test_code_atom_map_uses_single_collapsed_atom():
    vocab = AtomVocab({"A": 5, "B": 5, "P": 9})
    code_atoms = CodeAtomMap.from_vocab(vocab)

    assert code_atoms.atom("A") == 5
    assert code_atoms.atom("B") == 5
    assert code_atoms.atom("P") == 9
    assert code_atoms.atom("missing") == 0


def test_pad_atoms_always_emits_at_least_one_slot():
    atoms = _pad_atoms([[]])
    assert atoms.shape == (1, 1)
    assert atoms[0, 0].item() == 0  # PAD


def test_collate_shapes_and_targets():
    batch = [
        {
            "sex": 1,
            "static_atoms": [7, 13],
            "event_atoms": [21, 23, 24],
            "event_ages": np.array([100.0, 200.0, 365.25 * 20], dtype=np.float32),
            "event_values": np.array([1.5, np.nan, -2.0], dtype=np.float32),
            "censor_age_days": 30000.0,
        },
        {
            "sex": 0,
            "static_atoms": [31],
            "event_atoms": [41],
            "event_ages": np.array([50.0], dtype=np.float32),
            "event_values": np.array([0.25], dtype=np.float32),
            "censor_age_days": 25000.0,
        },
    ]
    out = collate(batch)

    assert out["static_atoms"].shape == (2, 2)
    assert out["event_ages"].shape == (2, 3)
    assert out["static_atoms"].tolist() == [[7, 13], [31, 0]]
    assert out["event_atoms"].tolist() == [[21, 23, 24], [41, 0, 0]]
    assert out["target_atoms"].tolist() == [[21, 23, 24], [41, 0, 0]]
    assert torch.allclose(out["event_values"][0], torch.tensor([1.5, torch.nan, -2.0]), equal_nan=True)
    assert out["event_values"][1, 0].item() == 0.25
    assert torch.isnan(out["event_values"][1, 1:]).all()
    assert out["event_pad"].tolist() == [[False, False, False], [False, True, True]]
    assert out["static_pad"][:, 0].tolist() == [False, False]
    assert out["sex"].tolist() == [1, 0]
    assert out["censor_age"].tolist() == [30000.0, 25000.0]


def test_collate_empty_static_is_nan_safe():
    batch = [
        {
            "sex": 0,
            "static_atoms": [],
            "event_atoms": [1],
            "event_ages": np.array([10.0], dtype=np.float32),
            "event_values": np.array([np.nan], dtype=np.float32),
            "censor_age_days": 100.0,
        }
    ]
    out = collate(batch)
    assert not out["static_pad"][0, 0].item(), "first static slot must be attendable to avoid all-masked softmax NaN"
    assert torch.isfinite(out["event_ages"]).all()
