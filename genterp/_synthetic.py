"""Synthetic singleton-bag batches for demos and tests."""

from __future__ import annotations

import torch


def make_batch(B: int = 4, M: int = 4, T: int = 16, n_atoms: int = 256, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    s_atoms, e_atoms = [], []
    for _ in range(B):
        s_row = []
        for _ in range(M):
            s_row.append(int(torch.randint(1, n_atoms, (1,), generator=g).item()))
        s_atoms.append(s_row)
        e_row = []
        for _ in range(T):
            e_row.append(int(torch.randint(1, n_atoms, (1,), generator=g).item()))
        e_atoms.append(e_row)
    ages = torch.linspace(20 * 365.25, 70 * 365.25, T).unsqueeze(0).expand(B, T).contiguous()
    event_atoms = torch.tensor(e_atoms, dtype=torch.long)
    censor_age = ages[:, -1] + 30.0
    event_values = torch.full((B, T), float("nan"), dtype=torch.float32)
    has_val = torch.rand(B, T, generator=g) < 0.4
    event_values[has_val] = torch.randn(B, T, generator=g)[has_val] * 1.5
    return {
        "static_atoms": torch.tensor(s_atoms, dtype=torch.long),
        "static_pad": torch.zeros(B, M, dtype=torch.bool),
        "event_atoms": event_atoms,
        "event_pad": torch.zeros(B, T, dtype=torch.bool),
        "event_ages": ages,
        "event_values": event_values,
        "target_atoms": event_atoms.clone(),
        "censor_age": censor_age,
        "sex": torch.tensor([i % 2 for i in range(B)], dtype=torch.long),
    }
