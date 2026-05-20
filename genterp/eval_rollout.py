"""Monte-Carlo trajectory C-index — the model is asked to *simulate*, not integrated.

Why this exists
---------------
``eval_cindex.py`` extracts disease risk via a closed-form integral that
freezes ``h_last`` for the full 10-year horizon and convolves the marked-TPP
density with the disease's atom-set marginals. The stationarity assumption
on the hidden state — the patient's representation doesn't drift over a
decade — is the largest single source of slack in the eval and it has no
analogue in the training objective.

This module replaces that integral with the model's own generative dynamics:

  1. For every eligible subject, replicate the prefix to N parallel chains.
  2. At each step every chain forwards the model over its current event
     sequence to obtain a fresh ``h_last``, samples (Δt, mark) from the TPP
     head, and appends the new event in place.
  3. A chain is "finished" once its cumulative simulated Δt exceeds the
     horizon. Risk-per-disease is the fraction of chains that hit the
     disease's atom set within the horizon.

Because the hidden state evolves as the model imagines events, the rollout
naturally captures cascades (a sampled diagnosis early in the trajectory
reshapes everything after it), competing risks (whichever disease the
chain commits to first wins), and tail-of-time-density expressiveness
that the analytic survival convolution can't reach.

Cost
----
Each step does a full transformer forward over the extended prefix; no KV
cache (yet). Practical settings: ``n_chains=64`` × ``max_steps≈80`` ×
``n_subjects≈256`` runs in tens of minutes on a single GPU. Wider/longer
runs are cheap to dial in via the CLI flags.

Public API
----------
- ``run_rollout_cindex(model, cohort, ...)`` → ``dict`` with the same per-
  disease schema as :func:`genterp.eval_cindex.run_cindex` (drop-in for
  dashboards).
- ``python -m genterp.eval_rollout``         # full
- ``python -m genterp.eval_rollout --tiny``  # tiny
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from genterp.data import AtomVocab
from genterp.decode import decode_init, decode_step
from genterp.eval_cindex import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    HORIZON_DAYS,
    MIN_EVENTS_FOR_C_SUMMARY,
    CindexCohort,
    _bootstrap_c,
    _harrell_cindex,
    prepare_cindex_cohort,
)
from genterp.progress import ProgressLogger
from genterp.runtime import accelerator_label, configure_torch_runtime
from genterp.train import GenterpForCausalLM, final_model_path

DEFAULT_N_CHAINS = 64
DEFAULT_MAX_STEPS = 80
DEFAULT_SUBJECT_BATCH = 4   # subjects per forward (× n_chains = real batch)
DEFAULT_MAX_SUBJECTS = 0    # 0 = all eligible


def _expand_to_chains(batch: dict, n_chains: int, device: torch.device) -> dict:
    """Replicate each subject's batch row to ``n_chains`` parallel chains.

    No extra event-axis padding is needed here: the KV-cached decode pathway
    grows its cache one position at a time inside the model, so per-row event
    tensors can stay sized to the prefix only.
    """
    def rep(x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(n_chains, dim=0).to(device, non_blocking=device.type == "cuda")

    return {
        "static_atoms": rep(batch["static_atoms"]),
        "static_pad": rep(batch["static_pad"]),
        "sex": rep(batch["sex"]),
        "event_atoms": rep(batch["event_atoms"]),
        "target_atoms": rep(batch["target_atoms"]),
        "event_ages": rep(batch["event_ages"].float()),
        "event_values": rep(batch["event_values"].float()),
        "event_pad": rep(batch["event_pad"]),
        "censor_age": rep(batch["censor_age"].float()),
        "length": rep(batch["length"]).to(torch.long),
    }


def _build_atom_to_disease(cohort: CindexCohort, n_atoms_total: int, device: torch.device) -> torch.Tensor:
    """Boolean (n_atoms, n_diseases) — does each atom belong to each disease set."""
    n_d = len(cohort.disease_names)
    out = torch.zeros(n_atoms_total, n_d, dtype=torch.bool, device=device)
    for d_idx, atom_set in enumerate(cohort.atom_sets):
        if not atom_set:
            continue
        idx = torch.tensor(sorted(atom_set), dtype=torch.long, device=device)
        out[idx, d_idx] = True
    return out


@torch.no_grad()
def _rollout_subject_batch(
    model: torch.nn.Module,
    batch: dict,
    *,
    n_chains: int,
    max_steps: int,
    horizon_days: float,
    atom_to_disease: torch.Tensor,
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> torch.Tensor:
    """Run KV-cached rollouts for one subject mini-batch. Returns risk (B, n_diseases).

    One full prefix forward populates the KV cache; subsequent steps are
    single-query decodes that append one (k, v) per layer per step. Cost
    scales linearly in max_steps instead of quadratically.
    """
    inner = model.model
    B = int(batch["event_atoms"].shape[0])
    n_diseases = atom_to_disease.shape[1]

    expanded = _expand_to_chains(batch, n_chains, device)
    BC = B * n_chains

    landmark_age = expanded["event_ages"].gather(
        1, (expanded["length"].clamp(min=1) - 1).unsqueeze(1)
    ).squeeze(1)
    horizon_age = landmark_age + float(horizon_days)
    disease_hit = torch.zeros(BC, n_diseases, dtype=torch.bool, device=device)
    finished = torch.zeros(BC, dtype=torch.bool, device=device)
    current_age = landmark_age.clone()

    use_autocast = autocast_dtype is not None and device.type == "cuda"
    ac_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if use_autocast else contextlib.nullcontext()
    )

    with ac_ctx:
        h_last, cache = decode_init(inner, expanded)

    for _ in range(max_steps):
        if finished.all():
            break
        delta_t, mark = inner.tpp.sample(h_last.float())
        delta_t = delta_t.clamp(min=1e-3)
        new_age = current_age + delta_t

        alive = ~finished
        in_horizon = alive & (new_age <= horizon_age)

        # Credit disease hits for events landing strictly within the horizon.
        hit_rows = in_horizon.nonzero(as_tuple=True)[0]
        if hit_rows.numel() > 0:
            hits = atom_to_disease[mark[hit_rows]]
            disease_hit[hit_rows] = disease_hit[hit_rows] | hits

        # Advance only chains that placed the event inside the horizon. Past-
        # horizon chains stop drawing new events; their cache stops growing in
        # the "valid" sense even though we still append a row per step (the
        # appended K/V are masked out by attention via valid_length).
        with ac_ctx:
            h_last = decode_step(inner, cache, mark, new_age, advance_mask=in_horizon)
        # Only advance current_age for chains that just placed a real event.
        current_age = torch.where(in_horizon, new_age, current_age)
        finished = finished | (new_age > horizon_age)

    risk = disease_hit.view(B, n_chains, n_diseases).float().mean(dim=1)
    return risk


@torch.no_grad()
def score_rollout_risks(
    model: torch.nn.Module,
    cohort: CindexCohort,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    n_chains: int = DEFAULT_N_CHAINS,
    max_steps: int = DEFAULT_MAX_STEPS,
    subject_batch: int = DEFAULT_SUBJECT_BATCH,
    max_subjects: int = DEFAULT_MAX_SUBJECTS,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every eligible subject via rollouts.

    Returns ``(risks, used_mask)`` of shapes (n_subjects, n_diseases) and
    (n_subjects,). When ``max_subjects > 0`` only a longest-history subsample
    is scored; the unused entries get risk NaN and ``used_mask=False`` and
    are skipped downstream by run_rollout_cindex.
    """
    was_training = model.training
    model.eval()
    try:
        n_subjects_total = len(cohort.subjects)
        n_atoms_total = int(model.model.cfg.n_atoms)
        atom_to_disease = _build_atom_to_disease(cohort, n_atoms_total, device)

        risks = np.full((n_subjects_total, len(cohort.disease_names)), np.nan, dtype=np.float64)
        used = np.zeros(n_subjects_total, dtype=bool)

        if max_subjects > 0 and max_subjects < n_subjects_total:
            lengths = np.asarray([s.last_event_idx_local + 1 for s in cohort.subjects], dtype=np.int64)
            subject_order = np.argsort(-lengths)[:max_subjects]
        else:
            subject_order = np.arange(n_subjects_total)

        from torch.utils.data import DataLoader, Subset
        from genterp.data import collate
        scored_loader = DataLoader(
            Subset(cohort.dataset, subject_order.tolist()),
            batch_size=subject_batch,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

        cursor = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(scored_loader):
            batch_on_device = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            risk_b = _rollout_subject_batch(
                model, batch_on_device,
                n_chains=n_chains,
                max_steps=max_steps,
                horizon_days=HORIZON_DAYS,
                atom_to_disease=atom_to_disease,
                autocast_dtype=autocast_dtype,
                device=device,
            )
            B = risk_b.shape[0]
            indices = subject_order[cursor:cursor + B]
            risks[indices] = risk_b.cpu().numpy()
            used[indices] = True
            cursor += B
            if progress_every and (batch_idx + 1) % progress_every == 0:
                rate = cursor / max(time.time() - t0, 1e-6)
                print(f"  rollout {cursor:,}/{len(subject_order):,} subjects  ({rate:.2f} subj/s)")
    finally:
        if was_training:
            model.train()
    return risks, used


def run_rollout_cindex(
    model: torch.nn.Module,
    cohort: CindexCohort,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    n_chains: int = DEFAULT_N_CHAINS,
    max_steps: int = DEFAULT_MAX_STEPS,
    subject_batch: int = DEFAULT_SUBJECT_BATCH,
    max_subjects: int = DEFAULT_MAX_SUBJECTS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = 0,
    progress_every: int = 0,
) -> dict[str, dict[str, object]]:
    """Per-disease Harrell's C from rollout risks.

    Same schema as ``eval_cindex.run_cindex`` so callers can swap the two and
    compare. Subjects with NaN risk (unscored under ``max_subjects``) are
    excluded from the per-disease eligible cohort.
    """
    risks, used = score_rollout_risks(
        model, cohort,
        device=device, autocast_dtype=autocast_dtype,
        n_chains=n_chains, max_steps=max_steps,
        subject_batch=subject_batch, max_subjects=max_subjects,
        progress_every=progress_every,
    )

    rng = np.random.default_rng(rng_seed)
    results: dict[str, dict[str, object]] = {}
    for d_idx, name in enumerate(cohort.disease_names):
        eligible_mask = used & (~cohort.prior_case[:, d_idx]) & cohort.sex_eligible[:, d_idx]
        n_eligible = int(eligible_mask.sum())
        events_observed = int((cohort.observed[:, d_idx] & eligible_mask).sum())
        incidence = 100.0 * events_observed / n_eligible if n_eligible else 0.0
        r = risks[eligible_mask, d_idx]
        t = cohort.time_to_event[eligible_mask, d_idx]
        o = cohort.observed[eligible_mask, d_idx]
        c, n_pairs = _harrell_cindex(r, t, o)
        ci_band = _bootstrap_c(r, t, o, bootstrap_resamples, rng) if bootstrap_resamples > 0 else None
        results[name] = {
            "n_eligible": n_eligible,
            "prior_cases": int(cohort.prior_case[:, d_idx].sum()),
            "sex_excluded": int((~cohort.sex_eligible[:, d_idx]).sum()),
            "events": events_observed,
            "incidence_pct": incidence,
            "c_index": None if math.isnan(c) else float(c),
            "n_pairs": n_pairs,
            "c_index_lo": ci_band[0] if ci_band else None,
            "c_index_hi": ci_band[2] if ci_band else None,
            "bootstrap_resamples": bootstrap_resamples if ci_band else 0,
        }
    valid = [
        float(m["c_index"]) for m in results.values()  # type: ignore[arg-type]
        if m["c_index"] is not None and int(m["events"]) >= MIN_EVENTS_FOR_C_SUMMARY  # type: ignore[arg-type]
    ]
    if valid:
        results["__summary__"] = {
            "cindex_mean_well_powered": float(np.mean(valid)),
            "n_well_powered_diseases": len(valid),
            "min_events_threshold": MIN_EVENTS_FOR_C_SUMMARY,
            "n_chains": n_chains,
            "max_steps": max_steps,
        }
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo trajectory C-index on test cohort.")
    parser.add_argument("--tiny", action="store_true", help="Use runs-tiny/ instead of runs/.")
    parser.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS,
                        help="Parallel rollout chains per subject (default 64).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Max sampled events per chain (default 80; chains stop early when horizon reached).")
    parser.add_argument("--subject-batch", type=int, default=DEFAULT_SUBJECT_BATCH,
                        help="Subjects per forward pass (default 4 — real batch = subject_batch × n_chains).")
    parser.add_argument("--max-subjects", type=int, default=DEFAULT_MAX_SUBJECTS,
                        help="Cap subjects scored (0 = all; longest-history prefix when >0).")
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="Bootstrap resamples for per-disease 95%% CI (default 500).")
    args = parser.parse_args(argv)

    setup = ProgressLogger("eval_rollout", total_units=6)
    setup.start_unit("configure runtime", "select accelerator + precision")
    runtime = configure_torch_runtime()
    device = runtime.device
    autocast_dtype = torch.bfloat16 if runtime.bf16 else torch.float16 if runtime.fp16 else None
    setup.finish_unit("configure runtime", f"device={accelerator_label(runtime)}")

    runs_dir = Path.home() / "genterp" / ("runs-tiny" if args.tiny else "runs")
    etl_dir = Path.home() / "genterp" / "etl"
    setup.start_unit("locate final model", f"runs_dir={runs_dir}")
    final = final_model_path(runs_dir)
    if final is None:
        raise SystemExit(f"no final model under {runs_dir}; run genterp.train first")
    setup.finish_unit("locate final model", f"path={final}")

    setup.start_unit("load frozen model", f"path={final}")
    model = GenterpForCausalLM.from_pretrained(final).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    setup.finish_unit("load frozen model", f"params={sum(p.numel() for p in model.parameters()):,}")

    setup.start_unit("load vocab", f"vocab={etl_dir / 'vocab.json'}")
    vocab = AtomVocab(dict(json.loads((etl_dir / "vocab.json").read_text())))
    setup.finish_unit("load vocab", f"atoms={len(vocab):,}")

    setup.start_unit("build cindex cohort", "per-subject landmark + outcome table")
    cohort = prepare_cindex_cohort(etl_dir, vocab, pin_memory=runtime.dataloader_pin_memory)
    setup.finish_unit(
        "build cindex cohort",
        f"subjects={len(cohort.subjects):,}  diseases={len(cohort.disease_names)}",
    )

    setup.start_unit(
        "score rollouts",
        f"n_chains={args.n_chains}  max_steps={args.max_steps}  "
        f"subject_batch={args.subject_batch}  max_subjects={args.max_subjects or 'all'}",
    )
    t0 = time.time()
    results = run_rollout_cindex(
        model, cohort,
        device=device, autocast_dtype=autocast_dtype,
        n_chains=args.n_chains, max_steps=args.max_steps,
        subject_batch=args.subject_batch, max_subjects=args.max_subjects,
        bootstrap_resamples=args.bootstrap,
        progress_every=max(1, (len(cohort.subjects) // args.subject_batch) // 40),
    )
    setup.finish_unit("score rollouts", f"elapsed={time.time() - t0:.1f}s")

    name_width = max(len(n) for n in cohort.disease_names)
    header = (
        f"  {'disease':<{name_width}}  {'eligible':>8}  {'events':>6}  {'inc%':>6}  "
        f"{'C (95% CI)':<22}  {'pairs':>9}"
    )
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for name in cohort.disease_names:
        m = results[name]
        c = m["c_index"]
        if m["c_index_lo"] is not None and m["c_index_hi"] is not None:
            c_str = f"{c:.3f} [{m['c_index_lo']:.3f},{m['c_index_hi']:.3f}]"
        else:
            c_str = f"{c:.4f}" if c is not None else "  nan  "
        print(
            f"  {name:<{name_width}}  {m['n_eligible']:>8,}  {m['events']:>6,}  "
            f"{m['incidence_pct']:>5.2f}%  {c_str:<22}  {m['n_pairs']:>9,}"
        )
    print("═" * len(header))
    summary = results.get("__summary__")
    if summary:
        print(
            f"  cindex_mean (≥{MIN_EVENTS_FOR_C_SUMMARY} events): "
            f"{summary['cindex_mean_well_powered']:.4f}  "
            f"over {summary['n_well_powered_diseases']} well-powered diseases  "
            f"(n_chains={summary['n_chains']}, max_steps={summary['max_steps']})"
        )

    out_json = runs_dir / "cindex_rollout_results.json"
    out_json.write_text(json.dumps({
        "run_dir": str(runs_dir),
        "final_model": str(final),
        "horizon_years": HORIZON_DAYS / 365.25,
        "n_chains": args.n_chains,
        "max_steps": args.max_steps,
        "subject_batch": args.subject_batch,
        "max_subjects": args.max_subjects,
        "results": {k: v for k, v in results.items() if not k.startswith("_")},
        "summary": summary,
    }, indent=2))
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
