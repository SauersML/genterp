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
    DEFAULT_SWEEP_TOP_N,
    MIN_EVENTS_FOR_C_SUMMARY,
    CindexCohort,
    _bootstrap_c,
    _harrell_cindex,
    build_cohort_condition_phenotypes,
    prepare_cindex_cohort,
)
from genterp.progress import ProgressLogger
from genterp.runtime import accelerator_label, configure_torch_runtime
from genterp.train import GenterpForCausalLM, final_model_path

# Rollout cost knobs — set as module constants so the CLI stays flag-free.
# n_chains × subject_batch is the effective batch size for decode forwards;
# memory scales linearly with that product. Defaults tuned for a single
# 32-GB GPU at production prefix lengths.
N_CHAINS = 64
MAX_STEPS = 80
SUBJECT_BATCH = 4
MAX_SUBJECTS = 0       # 0 = all eligible
BOOTSTRAP_RESAMPLES = 500
# Multi-horizon snapshots. Each rollout chain runs to the longest horizon
# and records disease hits cumulatively for each shorter horizon along the
# way. Tells us how the model's discrimination scales with prediction
# window — flat across horizons means general model strength; degrading
# with horizon means the simulator drifts; improving with horizon means
# the early steps are noisy.
HORIZON_YEARS: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0)
# Legacy public names kept as aliases so external callers don't break.
DEFAULT_N_CHAINS = N_CHAINS
DEFAULT_MAX_STEPS = MAX_STEPS
DEFAULT_SUBJECT_BATCH = SUBJECT_BATCH
DEFAULT_MAX_SUBJECTS = MAX_SUBJECTS


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
    horizon_offsets_days: tuple[float, ...],
    atom_to_disease: torch.Tensor,
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> torch.Tensor:
    """Run KV-cached rollouts for one subject mini-batch.

    Returns risk shape ``(B, n_horizons, n_diseases)`` — risk per disease at
    each of the supplied prediction-window lengths. The chain runs to the
    longest horizon; intermediate snapshots are captured cumulatively along
    the way, so the per-step cost is the same as a single-horizon rollout.

    One full prefix forward populates the KV cache; subsequent steps are
    single-query decodes that append one (k, v) per layer per step. Cost
    scales linearly in max_steps instead of quadratically.
    """
    inner = model.model
    B = int(batch["event_atoms"].shape[0])
    n_diseases = atom_to_disease.shape[1]
    n_horizons = len(horizon_offsets_days)
    if n_horizons == 0:
        raise ValueError("horizon_offsets_days must contain at least one offset")

    expanded = _expand_to_chains(batch, n_chains, device)
    BC = B * n_chains

    landmark_age = expanded["event_ages"].gather(
        1, (expanded["length"].clamp(min=1) - 1).unsqueeze(1)
    ).squeeze(1)
    # Sort horizon offsets so the LAST one is the rollout limit and earlier
    # ones are intermediate snapshots. Use the original index order to
    # restore the caller's horizon order in the returned tensor.
    sorted_offsets = sorted(enumerate(horizon_offsets_days), key=lambda iv: iv[1])
    sort_idx = torch.tensor([i for i, _ in sorted_offsets], device=device, dtype=torch.long)
    offsets_sorted = torch.tensor(
        [v for _, v in sorted_offsets], device=device, dtype=landmark_age.dtype,
    )
    max_horizon_age = landmark_age + float(offsets_sorted[-1].item())
    # (n_horizons, BC): per-chain absolute age at each horizon boundary.
    horizon_ages = landmark_age.unsqueeze(0) + offsets_sorted.unsqueeze(1)

    disease_hit = torch.zeros(n_horizons, BC, n_diseases, dtype=torch.bool, device=device)
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
        # Mask for "in the longest-horizon window" — chains past this stop.
        in_max_horizon = alive & (new_age <= max_horizon_age)

        # Per-horizon "this event lands inside this snapshot's window."
        # (n_horizons, BC): bool. Vectorized — no Python loop over horizons.
        in_horizon_h = in_max_horizon.unsqueeze(0) & (new_age.unsqueeze(0) <= horizon_ages)

        # Per-event disease membership.
        # mark may carry junk values for dead chains but we mask them out via
        # advance_mask below; gather all rows uniformly.
        event_hits = atom_to_disease[mark]  # (BC, n_diseases) bool
        # Broadcast to (n_horizons, BC, n_diseases) and AND with horizon mask.
        gated_hits = event_hits.unsqueeze(0) & in_horizon_h.unsqueeze(-1)
        disease_hit = disease_hit | gated_hits

        with ac_ctx:
            h_last = decode_step(inner, cache, mark, new_age, advance_mask=in_max_horizon)
        current_age = torch.where(in_max_horizon, new_age, current_age)
        finished = finished | (new_age > max_horizon_age)

    # Mean over chains → risk per (subject, horizon, disease)
    risk_sorted = (
        disease_hit.view(n_horizons, B, n_chains, n_diseases).float().mean(dim=2)
    )  # (n_horizons, B, n_diseases) in SORTED horizon order
    risk_sorted = risk_sorted.transpose(0, 1)  # (B, n_horizons_sorted, n_diseases)
    # Restore caller's original horizon order.
    inverse = torch.empty_like(sort_idx)
    inverse[sort_idx] = torch.arange(n_horizons, device=device, dtype=torch.long)
    return risk_sorted.index_select(dim=1, index=inverse)


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
    horizon_years: tuple[float, ...] = HORIZON_YEARS,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every eligible subject via rollouts at each requested horizon.

    Returns ``(risks, used_mask)`` of shapes ``(n_subjects, n_horizons, n_diseases)``
    and ``(n_subjects,)``. When ``max_subjects > 0`` only a longest-history
    subsample is scored; the unused entries get risk NaN and
    ``used_mask=False`` and are skipped downstream by run_rollout_cindex.
    """
    horizon_offsets_days = tuple(float(y) * 365.25 for y in horizon_years)
    was_training = model.training
    model.eval()
    try:
        n_subjects_total = len(cohort.subjects)
        n_atoms_total = int(model.model.cfg.n_atoms)
        atom_to_disease = _build_atom_to_disease(cohort, n_atoms_total, device)

        risks = np.full(
            (n_subjects_total, len(horizon_offsets_days), len(cohort.disease_names)),
            np.nan, dtype=np.float64,
        )
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
                horizon_offsets_days=horizon_offsets_days,
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
    horizon_years: tuple[float, ...] = HORIZON_YEARS,
    rng_seed: int = 0,
    progress_every: int = 0,
) -> dict[str, dict[str, object]]:
    """Per-disease Harrell's C from rollout risks at each requested horizon.

    Returns a dict keyed by disease name. Each entry now carries a
    ``per_horizon`` list of `{horizon_years, c_index, c_index_lo, c_index_hi,
    n_pairs, events}` so the leaderboard can show e.g. 1y / 3y / 5y / 10y
    side-by-side. The top-level ``c_index`` / ``n_pairs`` fields still reflect
    the longest horizon for backwards compatibility with the existing JSON
    output and downstream tooling.
    """
    risks, used = score_rollout_risks(
        model, cohort,
        device=device, autocast_dtype=autocast_dtype,
        n_chains=n_chains, max_steps=max_steps,
        subject_batch=subject_batch, max_subjects=max_subjects,
        horizon_years=horizon_years,
        progress_every=progress_every,
    )

    rng = np.random.default_rng(rng_seed)
    horizon_days = np.asarray([y * 365.25 for y in horizon_years], dtype=np.float64)
    n_horizons = len(horizon_years)
    results: dict[str, dict[str, object]] = {}
    for d_idx, name in enumerate(cohort.disease_names):
        eligible_mask = used & (~cohort.prior_case[:, d_idx]) & cohort.sex_eligible[:, d_idx]
        n_eligible = int(eligible_mask.sum())
        # Outcomes per horizon: an event is "observed within H years" iff the
        # actual time-to-event is within that window (and observed at all).
        t_full = cohort.time_to_event[eligible_mask, d_idx]
        o_full = cohort.observed[eligible_mask, d_idx]

        per_horizon: list[dict[str, object]] = []
        for h_idx, hy in enumerate(horizon_years):
            r_h = risks[eligible_mask, h_idx, d_idx]
            hdays = float(horizon_days[h_idx])
            # Observed within this horizon: original event + within window.
            o_h = o_full & (t_full <= hdays)
            # Time-to-event clipped at horizon for C-index discordance pairing.
            t_h = np.minimum(t_full, hdays)
            events_in_h = int(o_h.sum())
            c_h, n_pairs_h = _harrell_cindex(r_h, t_h, o_h)
            ci_band = _bootstrap_c(r_h, t_h, o_h, bootstrap_resamples, rng) if bootstrap_resamples > 0 else None
            per_horizon.append({
                "horizon_years": float(hy),
                "events": events_in_h,
                "c_index": None if math.isnan(c_h) else float(c_h),
                "n_pairs": int(n_pairs_h),
                "c_index_lo": ci_band[0] if ci_band else None,
                "c_index_hi": ci_band[2] if ci_band else None,
            })

        # Backward-compat top-level fields use the longest horizon.
        top = per_horizon[-1] if per_horizon else {}
        events_observed = int(top.get("events", 0))  # type: ignore[arg-type]
        incidence = 100.0 * events_observed / n_eligible if n_eligible else 0.0
        results[name] = {
            "n_eligible": n_eligible,
            "prior_cases": int(cohort.prior_case[:, d_idx].sum()),
            "sex_excluded": int((~cohort.sex_eligible[:, d_idx]).sum()),
            "events": events_observed,
            "incidence_pct": incidence,
            "c_index": top.get("c_index"),
            "n_pairs": top.get("n_pairs", 0),
            "c_index_lo": top.get("c_index_lo"),
            "c_index_hi": top.get("c_index_hi"),
            "bootstrap_resamples": bootstrap_resamples,
            "per_horizon": per_horizon,
        }
    valid = [
        float(m["c_index"]) for m in results.values()  # type: ignore[arg-type]
        if m["c_index"] is not None and int(m["events"]) >= MIN_EVENTS_FOR_C_SUMMARY  # type: ignore[arg-type]
    ]
    _ = n_horizons  # silence unused-name lint; kept for clarity above
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

    setup.start_unit(
        "build cindex cohort",
        f"OHDSI domain==Condition + top-{DEFAULT_SWEEP_TOP_N} by cohort coverage",
    )
    sweep_phenotypes = build_cohort_condition_phenotypes(etl_dir, top_n=DEFAULT_SWEEP_TOP_N)
    if not sweep_phenotypes:
        raise SystemExit(
            "OHDSI sweep returned no phenotypes — ohdsi_disease_phenotypes.json "
            "missing from ETL cache. Re-run scripts/aou_etl.py to build the "
            "OHDSI PhenotypeLibrary canonical disease list."
        )
    cohort_mode = f"sweep (top-{DEFAULT_SWEEP_TOP_N})"
    cohort = prepare_cindex_cohort(
        etl_dir, vocab,
        pin_memory=runtime.dataloader_pin_memory,
        phenotypes=sweep_phenotypes,
    )
    setup.finish_unit(
        "build cindex cohort",
        f"mode={cohort_mode}  subjects={len(cohort.subjects):,}  "
        f"diseases={len(cohort.disease_names)}",
    )

    setup.start_unit(
        "score rollouts",
        f"n_chains={N_CHAINS}  max_steps={MAX_STEPS}  "
        f"subject_batch={SUBJECT_BATCH}  max_subjects={MAX_SUBJECTS or 'all'}",
    )
    t0 = time.time()
    results = run_rollout_cindex(
        model, cohort,
        device=device, autocast_dtype=autocast_dtype,
        n_chains=N_CHAINS, max_steps=MAX_STEPS,
        subject_batch=SUBJECT_BATCH, max_subjects=MAX_SUBJECTS,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        progress_every=max(1, (len(cohort.subjects) // SUBJECT_BATCH) // 40),
    )
    setup.finish_unit("score rollouts", f"elapsed={time.time() - t0:.1f}s")

    # Leaderboard sort: best-predicted first. NaN C-index (too few events for
    # a meaningful estimate) falls to the bottom.
    def _c_sort_key(name: str) -> float:
        c = results[name].get("c_index")
        return float(c) if isinstance(c, (int, float)) else float("-inf")

    ordered_names = sorted(cohort.disease_names, key=_c_sort_key, reverse=True)
    display_width = min(60, max(len(n) for n in cohort.disease_names))
    # One column per horizon, sorted in HORIZON_YEARS order.
    horizon_cols = [f"C@{int(y) if float(y).is_integer() else y}y" for y in HORIZON_YEARS]
    horizon_col_width = 7  # "0.5xx" or " nan "
    header = (
        f"  {'disease':<{display_width}}  {'eligible':>8}  {'events':>6}  {'inc%':>6}  "
        + "  ".join(f"{col:>{horizon_col_width}}" for col in horizon_cols)
    )
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))

    def _fmt_c(value: object) -> str:
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return f"{float(value):.3f}"
        return "  nan "

    for name in ordered_names:
        m = results[name]
        horizon_cells = []
        per_horizon = m.get("per_horizon", []) or []
        horizon_lookup = {float(h["horizon_years"]): h for h in per_horizon}  # type: ignore[index]
        for y in HORIZON_YEARS:
            cell = horizon_lookup.get(float(y), {})
            horizon_cells.append(f"{_fmt_c(cell.get('c_index')):>{horizon_col_width}}")
        truncated = name if len(name) <= display_width else name[: display_width - 1] + "…"
        print(
            f"  {truncated:<{display_width}}  {m['n_eligible']:>8,}  {m['events']:>6,}  "
            f"{m['incidence_pct']:>5.2f}%  "
            + "  ".join(horizon_cells)
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
        "horizon_years": list(HORIZON_YEARS),
        "n_chains": N_CHAINS,
        "max_steps": MAX_STEPS,
        "subject_batch": SUBJECT_BATCH,
        "max_subjects": MAX_SUBJECTS,
        "results": {k: v for k, v in results.items() if not k.startswith("_")},
        "summary": summary,
    }, indent=2))
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
