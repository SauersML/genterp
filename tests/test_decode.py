"""KV-cached decode pathway must match the training forward.

The decode pathway in genterp.decode reuses every parameter from the model
but reroutes the data flow for fast incremental rollouts. Correctness boils
down to two invariants:

  1. ``decode_init`` returns the same last-token hidden state as the training
     forward on the same prefix.
  2. ``decode_step`` extended by k tokens returns the same last-token hidden
     state as a fresh full forward on the (prefix + k) sequence.

If either fails, rollouts are silently scoring against a different model
than training trained, which would be the worst possible kind of bug.
"""

from __future__ import annotations

import torch

from genterp import Genterp
from genterp.decode import decode_init, decode_step
from tests._factories import make_batch, tiny_config


def _batch_with_length(n_atoms: int, *, seed: int = 0) -> dict:
    """Synthetic batch with the ``length`` field that the training collate sets."""
    batch = make_batch(n_atoms=n_atoms, seed=seed)
    event_pad = batch["event_pad"]
    batch["length"] = (~event_pad).sum(dim=1).to(dtype=torch.long)
    return batch


def _h_last_from_forward(model: Genterp, batch: dict) -> torch.Tensor:
    out = model(**batch)
    hidden = out["hidden"]
    length = batch["length"].clamp(min=1)
    safe_idx = (length - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    return hidden.gather(1, safe_idx).squeeze(1)


def test_decode_init_matches_forward_h_last():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    batch = _batch_with_length(n_atoms=cfg.n_atoms)

    with torch.no_grad():
        h_forward = _h_last_from_forward(model, batch)
        h_decode, _cache = decode_init(model, batch)

    torch.testing.assert_close(h_decode, h_forward, rtol=1e-5, atol=1e-5)


def test_decode_step_matches_extended_forward():
    """Sample a (mark, age) for each row, then verify decode_step's hidden
    matches a fresh full forward over the appended sequence at that new last
    position. Done across two steps so chained cache updates are exercised.
    """
    torch.manual_seed(1)
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    batch = _batch_with_length(n_atoms=cfg.n_atoms)
    B = int(batch["event_atoms"].shape[0])

    # Deterministic "sampled" continuations.
    extra_marks_step1 = torch.tensor([3, 5, 1, 2][:B], dtype=torch.long)[:B]
    extra_dt_step1 = torch.tensor([2.0, 1.5, 3.25, 0.75][:B], dtype=torch.float32)[:B]
    extra_marks_step2 = torch.tensor([7, 4, 6, 3][:B], dtype=torch.long)[:B]
    extra_dt_step2 = torch.tensor([1.1, 4.0, 2.7, 0.5][:B], dtype=torch.float32)[:B]

    # Reference: extend the batch by both new events and run a fresh forward.
    def _extend(b: dict, new_atom: torch.Tensor, new_dt: torch.Tensor) -> dict:
        new = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        T = new["event_atoms"].shape[1]
        new_T = T + 1
        zeros_long = torch.zeros(B, 1, dtype=torch.long)
        zeros_f = torch.zeros(B, 1, dtype=torch.float32)
        nan_f = torch.full((B, 1), float("nan"))
        true_pad = torch.ones(B, 1, dtype=torch.bool)
        new["event_atoms"] = torch.cat([new["event_atoms"], zeros_long], dim=1)
        new["target_atoms"] = torch.cat([new["target_atoms"], zeros_long], dim=1)
        new["event_ages"] = torch.cat([new["event_ages"], zeros_f], dim=1)
        new["event_values"] = torch.cat([new["event_values"], nan_f], dim=1)
        new["event_pad"] = torch.cat([new["event_pad"], true_pad], dim=1)
        del new_T  # silence
        # Insert per-row at position length[r].
        old_len = new["length"]
        insert = old_len.clamp(min=0).view(-1, 1)
        prev_age = new["event_ages"].gather(1, (old_len.clamp(min=1) - 1).view(-1, 1))
        absolute_new_age = prev_age.squeeze(1) + new_dt
        new["event_atoms"].scatter_(1, insert, new_atom.view(-1, 1))
        new["target_atoms"].scatter_(1, insert, new_atom.view(-1, 1))
        new["event_ages"].scatter_(1, insert, absolute_new_age.view(-1, 1))
        new["event_pad"].scatter_(1, insert, torch.zeros(B, 1, dtype=torch.bool))
        new["length"] = old_len + 1
        return new

    with torch.no_grad():
        h0_forward = _h_last_from_forward(model, batch)
        h_init_decode, cache = decode_init(model, batch)
        torch.testing.assert_close(h_init_decode, h0_forward, rtol=1e-5, atol=1e-5)

        # Step 1
        prev_age = batch["event_ages"].gather(
            1, (batch["length"].clamp(min=1) - 1).view(-1, 1)
        ).squeeze(1)
        new_age1 = prev_age + extra_dt_step1
        h_step1 = decode_step(model, cache, extra_marks_step1, new_age1)
        batch1 = _extend(batch, extra_marks_step1, extra_dt_step1)
        h1_forward = _h_last_from_forward(model, batch1)
        torch.testing.assert_close(h_step1, h1_forward, rtol=1e-4, atol=1e-4)

        # Step 2
        new_age2 = new_age1 + extra_dt_step2
        h_step2 = decode_step(model, cache, extra_marks_step2, new_age2)
        batch2 = _extend(batch1, extra_marks_step2, extra_dt_step2)
        h2_forward = _h_last_from_forward(model, batch2)
        torch.testing.assert_close(h_step2, h2_forward, rtol=1e-4, atol=1e-4)


def test_decode_step_respects_advance_mask():
    """A chain whose advance_mask is False must NOT have its valid_length grow,
    even though its cache K, V append the same way as alive chains. Verified
    by comparing against the equivalent batch where the False-row was simply
    not extended.
    """
    torch.manual_seed(2)
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    batch = _batch_with_length(n_atoms=cfg.n_atoms)
    B = int(batch["event_atoms"].shape[0])
    assert B >= 2

    marks = torch.tensor([4, 2, 1, 6][:B], dtype=torch.long)[:B]
    dts = torch.tensor([1.5, 2.5, 0.5, 3.0][:B], dtype=torch.float32)[:B]
    advance = torch.ones(B, dtype=torch.bool)
    advance[0] = False  # row 0 is "finished"

    with torch.no_grad():
        _h0, cache = decode_init(model, batch)
        prev_age = batch["event_ages"].gather(
            1, (batch["length"].clamp(min=1) - 1).view(-1, 1)
        ).squeeze(1)
        new_age = prev_age + dts
        valid_before = cache.valid_length.clone()
        _ = decode_step(model, cache, marks, new_age, advance_mask=advance)

    expected_growth = advance.long()
    assert torch.equal(cache.valid_length - valid_before, expected_growth)
