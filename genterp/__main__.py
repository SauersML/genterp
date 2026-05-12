"""End-to-end demo: build a small Genterp, train 100 steps on synthetic ancestor-bag batches, print the loss curve."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from genterp import Genterp, GenterpConfig
from genterp._synthetic import make_batch


def main() -> None:
    torch.manual_seed(0)
    n_atoms = 512
    cfg = GenterpConfig(n_atoms=n_atoms, dim=128, n_heads=4, n_layers=4, n_static_blocks=2, k_static_summary=8)
    model = Genterp(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    params = sum(p.numel() for p in model.parameters())
    print(f"genterp demo  device={('cuda' if torch.cuda.is_available() else 'cpu')}  params={params:,}")

    batch = make_batch(B=4, M=4, T=24, n_atoms=n_atoms, seed=0)
    targets = batch["event_atoms"][batch["event_offsets"]].view(4, 24)

    t0 = time.time()
    for step in range(100):
        logits = model(**batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, n_atoms), targets[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == 99:
            print(f"step {step:3d}  loss {loss.item():.4f}")
    print(f"ok  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
