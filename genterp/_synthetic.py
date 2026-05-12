"""Synthetic ancestor-bag batches for demos and tests."""

from __future__ import annotations

import torch


def make_batch(B: int = 4, M: int = 4, T: int = 16, n_atoms: int = 256, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    s_atoms, s_off, s_pos = [], [], 0
    e_atoms, e_off, e_pos = [], [], 0
    for _ in range(B):
        for _ in range(M):
            s_off.append(s_pos)
            k = int(torch.randint(1, 5, (1,), generator=g).item())
            bag = torch.randint(1, n_atoms, (k,), generator=g).tolist()
            s_atoms.extend(bag)
            s_pos += k
        for _ in range(T):
            e_off.append(e_pos)
            k = int(torch.randint(1, 5, (1,), generator=g).item())
            bag = torch.randint(1, n_atoms, (k,), generator=g).tolist()
            e_atoms.extend(bag)
            e_pos += k
    ages = torch.linspace(20 * 365.25, 70 * 365.25, T).unsqueeze(0).expand(B, T).contiguous()
    target_atoms = torch.tensor(e_atoms, dtype=torch.long)[torch.tensor(e_off, dtype=torch.long)].view(B, T)
    censor_age = ages[:, -1] + 30.0
    event_values = torch.full((B, T), float("nan"), dtype=torch.float32)
    has_val = torch.rand(B, T, generator=g) < 0.4
    event_values[has_val] = torch.randn(B, T, generator=g)[has_val] * 1.5
    return {
        "static_atoms": torch.tensor(s_atoms, dtype=torch.long),
        "static_offsets": torch.tensor(s_off, dtype=torch.long),
        "static_pad": torch.zeros(B, M, dtype=torch.bool),
        "static_shape": (B, M),
        "event_atoms": torch.tensor(e_atoms, dtype=torch.long),
        "event_offsets": torch.tensor(e_off, dtype=torch.long),
        "event_pad": torch.zeros(B, T, dtype=torch.bool),
        "event_ages": ages,
        "event_values": event_values,
        "target_atoms": target_atoms,
        "censor_age": censor_age,
        "sex": torch.tensor([i % 2 for i in range(B)], dtype=torch.long),
    }
