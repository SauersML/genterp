"""Tiny synthetic batches and configs for tests."""

from __future__ import annotations

from genterp import GenterpConfig
from genterp._synthetic import make_batch as _make_batch


def make_batch(B: int = 2, M: int = 4, T: int = 12, n_atoms: int = 128, seed: int = 0) -> dict:
    return _make_batch(B=B, M=M, T=T, n_atoms=n_atoms, seed=seed)


def tiny_config(n_atoms: int = 128, dim: int = 64, n_heads: int = 4, n_layers: int = 2) -> GenterpConfig:
    return GenterpConfig(
        n_atoms=n_atoms,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        n_static_blocks=1,
        k_static_summary=4,
        n_time_mix=4,
        mark_rank=32,
        time_phi_dim=16,
    )
