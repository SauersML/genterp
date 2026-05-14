"""10-year C-index evaluation on the held-out test split.

For each subject in the test cohort:
  1. Find a landmark timestamp = latest event before `LANDMARK_AGE_DAYS`.
     Subjects whose record doesn't reach that age are skipped.
  2. Truncate event history to events strictly before the landmark.
  3. Run the frozen final-model forward on the history; take the hidden state
     at the last valid position.
  4. For each target disease atom, compute the 10-year risk via the
     **first-event approximation**:

         risk_X = ∫_0^T  p_time(Δt | h_last) · p_mark(X | h_last, Δt)  dΔt

     evaluated by trapezoidal quadrature on a log-spaced day grid. This is a
     monotone proxy for true cumulative risk (doesn't include "X happens
     after other events"), which is fine for ranking-only metrics like
     Harrell's C-index.
  5. Ground truth = was atom X recorded for that subject in
     `[landmark, landmark + 10y]`, observed from the raw events parquet.
     Censoring = subject's observation_period_end - landmark.

Harrell's C-index with right-censoring is reported per disease, plus the
fraction of test subjects retained and the disease prevalence.

Run as:
    python -m genterp.eval_cindex            # full run, runs/
    python -m genterp.eval_cindex --tiny     # tiny run, runs-tiny/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from genterp.data import AtomVocab, EventStore, PAD_ATOM, collate
from genterp.progress import ProgressLogger
from genterp.runtime import configure_torch_runtime, accelerator_label
from genterp.train import GenterpForCausalLM, final_model_path


# Default disease panel. Codes are OMOP `vocabulary_id/concept_code` strings
# (matching the format vocab.json uses). Pulled from common cardiometabolic /
# oncology / neurology atlases. Any code not present in the collapsed vocab
# is skipped at runtime with a notice.
DEFAULT_DISEASES: dict[str, str] = {
    "Type 2 diabetes":          "SNOMED/44054006",
    "Essential hypertension":   "SNOMED/59621000",
    "Acute MI":                 "SNOMED/22298006",
    "Heart failure":            "SNOMED/84114007",
    "Atrial fibrillation":      "SNOMED/49436004",
    "Stroke (CVA)":             "SNOMED/230690007",
    "Chronic kidney disease":   "SNOMED/709044004",
    "COPD":                     "SNOMED/13645005",
    "Alzheimer's disease":      "SNOMED/26929004",
    "Breast cancer":            "SNOMED/254837009",
    "Colorectal cancer":        "SNOMED/93761005",
}

LANDMARK_AGE_DAYS = 50.0 * 365.25      # only subjects whose record reaches age 50
HORIZON_DAYS = 10.0 * 365.25           # 10-year window for outcome assessment
MIN_FOLLOWUP_DAYS = 365.25             # require ≥ 1 yr post-landmark observation
N_QUAD_POINTS = 24                     # log-spaced Δt grid for risk integral
EVAL_BATCH_SIZE = 16
DATALOADER_WORKERS = 2


@dataclass
class SubjectIndex:
    """All the per-subject scalars we need from subjects.parquet + events store."""
    subject_id: int
    start: int
    end: int
    birth_seconds: float
    censor_seconds: float
    sex: int
    landmark_idx_local: int            # index within the subject's event window
    landmark_age_days: float
    censor_age_days: float


def _build_subject_index(etl_dir: Path, events: EventStore) -> list[SubjectIndex]:
    """Walk the test cohort, locate landmark per subject, drop ineligibles."""
    subjects = pq.read_table(etl_dir / "subjects.parquet").to_pylist()
    test = [s for s in subjects if s.get("split") == "test"]
    eligible: list[SubjectIndex] = []
    skipped_no_history = 0
    skipped_short_followup = 0
    for s in test:
        start, end = int(s["start"]), int(s["end"])
        birth_seconds = float(s["birth_seconds"])
        censor_seconds = float(s["censor_seconds"])
        n_rows = end - start + 1
        if n_rows <= 1:
            skipped_no_history += 1
            continue
        time_seconds = events.time_seconds.slice(start, n_rows).to_numpy(zero_copy_only=False)
        ages_days = (time_seconds - birth_seconds) / 86400.0
        before = ages_days < LANDMARK_AGE_DAYS
        if not before.any():
            skipped_no_history += 1
            continue
        landmark_idx_local = int(before.sum()) - 1
        landmark_age = float(ages_days[landmark_idx_local])
        censor_age = (censor_seconds - birth_seconds) / 86400.0
        if censor_age - landmark_age < MIN_FOLLOWUP_DAYS:
            skipped_short_followup += 1
            continue
        eligible.append(SubjectIndex(
            subject_id=int(s["subject_id"]),
            start=start,
            end=end,
            birth_seconds=birth_seconds,
            censor_seconds=censor_seconds,
            sex=int(s.get("sex", 0) or 0),
            landmark_idx_local=landmark_idx_local,
            landmark_age_days=landmark_age,
            censor_age_days=censor_age,
        ))
    print(
        f"[eval_cindex] cohort: test_total={len(test):,}  eligible={len(eligible):,}  "
        f"skipped_no_history={skipped_no_history:,}  skipped_short_followup={skipped_short_followup:,}"
    )
    return eligible


class LandmarkDataset(Dataset):
    """Yields one batch entry per eligible test subject — history-only events
    up to (and excluding) the landmark index. Same key/value layout as the
    training collate so we can reuse `genterp.data.collate`.
    """

    def __init__(self, etl_dir: Path, events: EventStore, subjects: list[SubjectIndex], max_events: int):
        self.events = events
        self.subjects = subjects
        self.max_events = max_events

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> dict:
        s = self.subjects[idx]
        n_rows = s.end - s.start + 1
        atoms = self.events.atom.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        times = self.events.time_seconds.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        values = self.events.value.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        delta_days = (times - s.birth_seconds) / 86400.0
        real_atom = atoms != PAD_ATOM

        static_mask = (delta_days <= 0.5) & real_atom
        event_mask = (delta_days > 0.5) & real_atom & (delta_days < LANDMARK_AGE_DAYS)
        event_idx_local = np.where(event_mask)[0]
        # Keep the most recent max_events events before the landmark so
        # long-lived subjects don't blow the sequence cap.
        event_idx_local = event_idx_local[-self.max_events:]

        return {
            "sex": s.sex,
            "static_atoms": atoms[static_mask].astype(np.int64).tolist(),
            "event_atoms": atoms[event_idx_local].astype(np.int64).tolist(),
            "event_ages": delta_days[event_idx_local].astype(np.float32),
            "event_values": values[event_idx_local].astype(np.float32),
            "censor_age_days": float(s.censor_age_days),
            "length": int(event_idx_local.size),
        }


def _build_outcome_table(
    events: EventStore,
    subjects: list[SubjectIndex],
    disease_atoms: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (subject, disease) compute:
       - time_to_event_days  : days from landmark to first occurrence of the
                                disease atom, or to censor if it never occurs
                                within the observation window
       - observed            : True if a disease atom occurred before
                                min(landmark + HORIZON, censor)
       - censored_at_horizon : True if the subject was right-censored at the
                                horizon without observing the event

    Returns three (n_subjects, n_diseases) arrays. Subjects who reach the
    horizon without an event are treated as right-censored at HORIZON_DAYS;
    subjects whose observation ends earlier are censored at that earlier time.
    """
    n_s = len(subjects)
    atom_ids = list(disease_atoms.values())
    n_d = len(atom_ids)
    time_to_event = np.full((n_s, n_d), np.nan, dtype=np.float64)
    observed = np.zeros((n_s, n_d), dtype=bool)
    for i, s in enumerate(subjects):
        n_rows = s.end - s.start + 1
        atoms = events.atom.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        times = events.time_seconds.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        ages_days = (times - s.birth_seconds) / 86400.0
        post = ages_days >= s.landmark_age_days
        post_atoms = atoms[post]
        post_ages = ages_days[post]
        # cap observation horizon at min(censor, landmark + horizon)
        horizon_age = min(s.censor_age_days, s.landmark_age_days + HORIZON_DAYS)
        for d_idx, aid in enumerate(atom_ids):
            hits = (post_atoms == aid) & (post_ages <= horizon_age)
            if hits.any():
                first_age = float(post_ages[hits].min())
                time_to_event[i, d_idx] = first_age - s.landmark_age_days
                observed[i, d_idx] = True
            else:
                time_to_event[i, d_idx] = horizon_age - s.landmark_age_days
                observed[i, d_idx] = False
    return time_to_event, observed, atom_ids


def _compute_disease_risks(
    model: torch.nn.Module,
    h_last: torch.Tensor,
    disease_atoms_t: torch.Tensor,
    horizon_days: float,
    n_grid: int,
) -> torch.Tensor:
    """First-event approximation cumulative risk per (subject, disease).

    h_last: (B, D) tensor — hidden state at each subject's landmark.
    disease_atoms_t: (n_diseases,) long tensor of atom IDs.
    Returns (B, n_diseases) risk scores.
    """
    device = h_last.device
    B, D = h_last.shape
    tpp = model.model.tpp

    grid = torch.logspace(0.0, math.log10(horizon_days), n_grid, device=device, dtype=torch.float32)
    # Time mixture at the landmark hidden state — (B, K_mix) each.
    log_w, mu, log_sigma = tpp.time_params(h_last.float())
    inv_sigma = (-log_sigma).exp()                                              # (B, K)
    log_dt = grid.log().view(n_grid, 1, 1)                                       # (G, 1, 1)
    mu_b = mu.unsqueeze(0)                                                       # (1, B, K)
    inv_sigma_b = inv_sigma.unsqueeze(0)                                         # (1, B, K)
    log_sigma_b = log_sigma.unsqueeze(0)                                         # (1, B, K)
    log_w_b = log_w.unsqueeze(0)                                                 # (1, B, K)
    log_pdf = (
        -log_dt - log_sigma_b - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((log_dt - mu_b) * inv_sigma_b).pow(2)
    )                                                                            # (G, B, K)
    time_density = torch.logsumexp(log_w_b + log_pdf, dim=-1).exp()              # (G, B)

    # Mark probabilities over the grid. We can batch over (G, B) by expanding.
    h_expanded = h_last.unsqueeze(0).expand(n_grid, B, D).reshape(n_grid * B, D)  # (G*B, D)
    dt_expanded = grid.unsqueeze(1).expand(n_grid, B).reshape(n_grid * B)         # (G*B,)
    mark_lp = tpp.mark_log_probs(h_expanded.float(), dt_expanded)                 # (G*B, n_atoms)
    mark_p_disease = mark_lp.index_select(-1, disease_atoms_t).exp()              # (G*B, n_disease)
    mark_p_disease = mark_p_disease.view(n_grid, B, -1)                            # (G, B, n_disease)

    integrand = time_density.unsqueeze(-1) * mark_p_disease                       # (G, B, n_disease)
    dgrid = torch.diff(grid).view(-1, 1, 1)                                       # (G-1, 1, 1)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    risk = (avg * dgrid).sum(dim=0)                                                # (B, n_disease)
    return risk


def _harrell_cindex(risks: np.ndarray, time_to_event: np.ndarray, observed: np.ndarray) -> float:
    """Harrell's C with right-censoring. Vectorized via pair enumeration.

    risks            : (n,)
    time_to_event    : (n,) days from landmark to event or censor
    observed         : (n,) bool — True if event observed in [landmark, horizon]
    """
    n = len(risks)
    if n < 2:
        return float("nan")
    # Permissible pair (i, j): one had an event at the earlier time.
    risks = np.asarray(risks, dtype=np.float64)
    t = np.asarray(time_to_event, dtype=np.float64)
    e = np.asarray(observed, dtype=bool)

    # Pair matrix is O(n²). For 12k subjects = 144M pairs ~ 1.1GB float64 — too big.
    # Use bucketing by event-time order instead: iterate over event subjects.
    event_indices = np.flatnonzero(e)
    concordant = 0.0
    permissible = 0.0
    for i in event_indices:
        ti = t[i]
        ri = risks[i]
        # All j with t[j] > ti contribute (regardless of e[j]); ties on time skip.
        partners = t > ti
        if not partners.any():
            continue
        rj = risks[partners]
        permissible += partners.sum()
        concordant += float(np.sum(ri > rj)) + 0.5 * float(np.sum(ri == rj))
    return concordant / permissible if permissible > 0 else float("nan")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="10-year C-index eval on test cohort.")
    parser.add_argument("--tiny", action="store_true", help="Use runs-tiny/ instead of runs/.")
    parser.add_argument("--landmark-age", type=float, default=LANDMARK_AGE_DAYS / 365.25,
                        help="Landmark age in years (default: 50).")
    parser.add_argument("--horizon-years", type=float, default=HORIZON_DAYS / 365.25,
                        help="Outcome horizon in years (default: 10).")
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="If >0, cap eligible cohort for a fast smoke test.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON path to dump per-disease results.")
    args = parser.parse_args(argv)

    global LANDMARK_AGE_DAYS, HORIZON_DAYS
    LANDMARK_AGE_DAYS = float(args.landmark_age) * 365.25
    HORIZON_DAYS = float(args.horizon_years) * 365.25

    setup = ProgressLogger("eval_cindex", total_units=8)
    setup.start_unit("configure runtime", "select accelerator + precision")
    runtime = configure_torch_runtime()
    device = runtime.device
    autocast_dtype = (
        torch.bfloat16 if runtime.bf16 else torch.float16 if runtime.fp16 else None
    )
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

    setup.start_unit("load vocab + resolve disease atoms", f"vocab={etl_dir / 'vocab.json'}")
    vocab = AtomVocab(dict(json.loads((etl_dir / "vocab.json").read_text())))
    disease_atoms: dict[str, int] = {}
    for name, code in DEFAULT_DISEASES.items():
        aid = vocab.encode(code)
        if aid == PAD_ATOM:
            print(f"  skip {name} ({code}) — not in vocab")
            continue
        disease_atoms[name] = aid
    if not disease_atoms:
        raise SystemExit("none of the default disease codes resolved to atoms; vocab too narrow")
    setup.finish_unit("load vocab + resolve disease atoms", f"vocab={len(vocab):,} diseases={len(disease_atoms)}")

    setup.start_unit("load shared event store", f"events={etl_dir / 'events.parquet'}")
    events = EventStore.from_parquet(etl_dir / "events.parquet")
    setup.finish_unit("load shared event store", f"rows={events.num_rows:,}")

    setup.start_unit("build subject index", f"split=test landmark_age={LANDMARK_AGE_DAYS/365.25:.1f}y")
    subjects = _build_subject_index(etl_dir, events)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
        print(f"  --max-subjects → truncating to {len(subjects):,}")
    if not subjects:
        raise SystemExit("no eligible test subjects")
    setup.finish_unit("build subject index", f"eligible={len(subjects):,}")

    setup.start_unit("score risks", f"batch_size={EVAL_BATCH_SIZE} grid={N_QUAD_POINTS}")
    dataset = LandmarkDataset(etl_dir, events, subjects, max_events=4096)
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=runtime.dataloader_pin_memory,
        collate_fn=collate,
    )
    disease_atoms_t = torch.tensor(list(disease_atoms.values()), dtype=torch.long, device=device)
    all_risks = np.zeros((len(subjects), len(disease_atoms)), dtype=np.float64)
    cursor = 0
    progress_tick = max(1, len(loader) // 40)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_on_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ac_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None and device.type == "cuda"
                else torch.cuda.amp.autocast(enabled=False) if device.type == "cuda"
                else torch.autocast(device_type=device.type, enabled=False)
            )
            with ac_ctx:
                out = model.model.forward(
                    static_atoms=batch_on_device["static_atoms"],
                    static_pad=batch_on_device["static_pad"],
                    sex=batch_on_device["sex"],
                    event_atoms=batch_on_device["event_atoms"],
                    event_ages=batch_on_device["event_ages"],
                    event_pad=batch_on_device["event_pad"],
                    target_atoms=batch_on_device["target_atoms"],
                    event_values=batch_on_device["event_values"],
                    length=batch_on_device.get("length"),
                )
            hidden = out["hidden"]                                  # (B, T, D)
            lengths = batch_on_device["length"].clamp(min=1)
            last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
            h_last = hidden.gather(1, last_idx).squeeze(1)          # (B, D)
            risk = _compute_disease_risks(model, h_last, disease_atoms_t, HORIZON_DAYS, N_QUAD_POINTS)
            n = risk.shape[0]
            all_risks[cursor : cursor + n] = risk.float().cpu().numpy()
            cursor += n
            if batch_idx % progress_tick == 0:
                print(f"  scored {cursor:,}/{len(subjects):,} subjects")
    setup.finish_unit("score risks", f"subjects_scored={cursor:,}")

    setup.start_unit("build outcome table", f"horizon={HORIZON_DAYS/365.25:.1f}y")
    time_to_event, observed, _ = _build_outcome_table(events, subjects, disease_atoms)
    setup.finish_unit("build outcome table", f"matrix=({len(subjects):,}, {len(disease_atoms)})")

    setup.start_unit("compute Harrell C-index per disease", "")
    results: dict[str, dict[str, float]] = {}
    name_width = max(len(n) for n in disease_atoms)
    print("\n" + "═" * (name_width + 60))
    print(f"  {'disease':<{name_width}}  {'atom':>6}  {'events':>8}  {'prev%':>6}  {'C-index':>9}  {'n':>6}")
    print("─" * (name_width + 60))
    for d_idx, name in enumerate(disease_atoms):
        events_observed = int(observed[:, d_idx].sum())
        n_total = len(subjects)
        prevalence = 100.0 * events_observed / n_total if n_total else 0.0
        c = _harrell_cindex(all_risks[:, d_idx], time_to_event[:, d_idx], observed[:, d_idx])
        results[name] = {
            "atom": int(disease_atoms[name]),
            "events": events_observed,
            "n": n_total,
            "prevalence_pct": prevalence,
            "c_index": c,
        }
        c_str = f"{c:.4f}" if not math.isnan(c) else "  nan  "
        print(f"  {name:<{name_width}}  {disease_atoms[name]:>6}  {events_observed:>8,}  {prevalence:>5.2f}%  {c_str:>9}  {n_total:>6,}")
    print("═" * (name_width + 60))
    setup.finish_unit("compute Harrell C-index per disease", f"diseases={len(results)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "run_dir": str(runs_dir),
            "final_model": str(final),
            "landmark_age_years": LANDMARK_AGE_DAYS / 365.25,
            "horizon_years": HORIZON_DAYS / 365.25,
            "n_subjects": len(subjects),
            "results": results,
        }, indent=2))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
