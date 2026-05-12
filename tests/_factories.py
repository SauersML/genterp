"""Tiny synthetic batches and configs for tests."""

from __future__ import annotations

import torch

from genterp import GenterpConfig


def make_batch(B: int = 2, M: int = 4, T: int = 12, n_atoms: int = 128, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def rand_bags(b: int, s: int, bag_size_max: int = 4) -> list[list[list[int]]]:
        return [
            [
                torch.randint(1, n_atoms, (int(torch.randint(1, bag_size_max + 1, (1,), generator=g).item()),), generator=g).tolist()
                for _ in range(s)
            ]
            for _ in range(b)
        ]

    s_atoms, s_off = _flatten_bags([bag for row in rand_bags(B, M) for bag in row])
    e_atoms, e_off = _flatten_bags([bag for row in rand_bags(B, T) for bag in row])

    static_pad = torch.zeros(B, M, dtype=torch.bool)
    event_pad = torch.zeros(B, T, dtype=torch.bool)
    event_pad[B - 1, -3:] = True if T >= 3 else False

    event_ages = torch.zeros(B, T)
    for b in range(B):
        base = 20 * 365.25 + b * 10 * 365.25
        event_ages[b] = torch.linspace(base, base + 365.25 * 5, T)

    return {
        "static_atoms": s_atoms,
        "static_offsets": s_off,
        "static_pad": static_pad,
        "static_shape": (B, M),
        "event_atoms": e_atoms,
        "event_offsets": e_off,
        "event_pad": event_pad,
        "event_ages": event_ages,
        "sex": torch.tensor([i % 2 for i in range(B)], dtype=torch.long),
    }


def _flatten_bags(bags: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    flat: list[int] = []
    offs: list[int] = []
    pos = 0
    for bag in bags:
        offs.append(pos)
        flat.extend(bag)
        pos += len(bag)
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offs, dtype=torch.long)


def tiny_config(n_atoms: int = 128, dim: int = 64, n_heads: int = 4, n_layers: int = 2) -> GenterpConfig:
    return GenterpConfig(
        n_atoms=n_atoms,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        n_static_blocks=1,
        k_static_summary=4,
    )
