"""10-year incident-disease C-index on the held-out test cohort.

This is "landmark survival analysis": fix a calendar age (default 50 years),
predict for each subject whether disease X first occurs in the next 10 years,
and measure Harrell's C-index against right-censored ground truth.

Per subject:

  1. Landmark = a FIXED age (50 y by default; configurable via --landmark-age).
     Subjects whose record doesn't reach the landmark age are skipped.
  2. History = all events strictly before the landmark age.
  3. Last event = the most recent event in history. gap_to_landmark = landmark
     age - last event age. Subjects with gap > --max-gap-years are skipped
     (default 5 y — beyond that the hidden state is too stale to score from).
  4. Run the frozen final model over the history → hidden state at the last
     valid position becomes h_last.
  5. For each target disease atom, compute the 10-year cumulative incidence
     under the model's hazard, evaluated on a per-subject log-spaced Δt grid
     from gap_to_landmark to gap_to_landmark + 10y (NOT from 0):

         λ_X(Δt | h)   = p_time(Δt | h) · p_mark(X | h, Δt) / S(Δt | h)
         risk_X        = 1 - exp( -∫_{gap}^{gap+10y} λ_X(Δt | h) dΔt )

     where S(Δt | h) is the survival function (no event yet by Δt). This
     is the standard cumulative-incidence form for a marked TPP — bounded
     in [0, 1] and correctly accounts for multi-event futures under
     stationarity (i.e. the assumption that h doesn't drift much over the
     horizon). The naive first-event integral ∫ p_time · p_mark systematically
     underestimates for chronically-ill subjects because mass is spent on
     the first event being something else; the hazard fix above removes
     that bias.

  6. Ground truth = disease atom in [landmark, min(landmark+10y, censor)].
     Subjects with PRIOR occurrences (before landmark) are excluded from that
     disease's eval — this is incident prediction, not point-prevalence.

Output: per disease, the eligible-cohort size, post-landmark event count,
prevalence in eligible cohort, Harrell's C with right-censoring, and the
atom's source-code equivalence class for transparency about what's being
scored.

Run:
    python -m genterp.eval_cindex          # full cohort under ~/genterp/runs/
    python -m genterp.eval_cindex --tiny   # tiny cohort under ~/genterp/runs-tiny/

Landmark age, horizon, and staleness gap are constants at the top of this
module — edit the file if you want different ones.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from genterp.data import AtomVocab, EventStore, PAD_ATOM, collate
from genterp.modeling import _log_ndtr
from genterp.progress import ProgressLogger
from genterp.runtime import configure_torch_runtime, accelerator_label
from genterp.train import GenterpForCausalLM, final_model_path


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

LANDMARK_AGE_DAYS = 50.0 * 365.25     # FIXED for all subjects, not "last event before age 50"
HORIZON_DAYS = 10.0 * 365.25
MIN_FOLLOWUP_DAYS = 365.25            # require ≥ 1 yr post-landmark observation
MAX_GAP_DAYS = 5.0 * 365.25           # reject subjects whose last pre-landmark event is too stale
N_QUAD_POINTS = 24
EVAL_BATCH_SIZE = 16
DATALOADER_WORKERS = 2


@dataclass
class SubjectIndex:
    """Per-subject scalars needed to score risk and look up ground truth."""
    subject_id: int
    start: int                     # row offset of this subject's first event in events.parquet
    end: int                       # row offset of last event (inclusive)
    birth_seconds: float
    censor_seconds: float
    sex: int
    last_event_idx_local: int      # index *within this subject's window* of last pre-landmark event
    last_event_age_days: float     # age (days) at that last pre-landmark event
    gap_to_landmark_days: float    # LANDMARK_AGE_DAYS - last_event_age_days, ≥ 0
    censor_age_days: float


def _build_subject_index(events: EventStore, etl_dir: Path) -> list[SubjectIndex]:
    """Walk the test cohort, find the last pre-landmark event per subject,
    drop subjects with no pre-landmark history, short follow-up, or stale gap."""
    rows = pq.read_table(etl_dir / "subjects.parquet").to_pylist()
    test = [r for r in rows if r.get("split") == "test"]
    eligible: list[SubjectIndex] = []
    skipped_no_history = 0
    skipped_short_followup = 0
    skipped_stale_gap = 0
    for r in test:
        start, end = int(r["start"]), int(r["end"])
        birth_seconds = float(r["birth_seconds"])
        censor_seconds = float(r["censor_seconds"])
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
        last_event_idx_local = int(before.sum()) - 1
        last_event_age = float(ages_days[last_event_idx_local])
        censor_age = (censor_seconds - birth_seconds) / 86400.0
        if censor_age - LANDMARK_AGE_DAYS < MIN_FOLLOWUP_DAYS:
            skipped_short_followup += 1
            continue
        gap = LANDMARK_AGE_DAYS - last_event_age
        if gap > MAX_GAP_DAYS:
            skipped_stale_gap += 1
            continue
        eligible.append(SubjectIndex(
            subject_id=int(r["subject_id"]),
            start=start,
            end=end,
            birth_seconds=birth_seconds,
            censor_seconds=censor_seconds,
            sex=int(r.get("sex", 0) or 0),
            last_event_idx_local=last_event_idx_local,
            last_event_age_days=last_event_age,
            gap_to_landmark_days=float(gap),
            censor_age_days=float(censor_age),
        ))
    print(
        f"[eval_cindex] cohort: test_total={len(test):,}  eligible={len(eligible):,}  "
        f"skipped_no_history={skipped_no_history:,}  "
        f"skipped_short_followup<{MIN_FOLLOWUP_DAYS/365.25:.1f}y={skipped_short_followup:,}  "
        f"skipped_stale_gap>{MAX_GAP_DAYS/365.25:.1f}y={skipped_stale_gap:,}"
    )
    return eligible


class LandmarkDataset(Dataset):
    """Yields one batch entry per eligible test subject — events strictly
    before the FIXED landmark age. Reuses the training collate.
    """

    def __init__(self, events: EventStore, subjects: list[SubjectIndex], max_events: int):
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
        event_idx_local = np.where(event_mask)[0][-self.max_events:]
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
    disease_atom_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (subject, disease) return:
       time_to_event : days from FIXED LANDMARK AGE to first occurrence in
                        [landmark, min(landmark+horizon, censor)], or to censor
                        if no event in that window.
       observed     : bool — True if a disease atom occurred before
                        min(landmark+horizon, censor).
       prior_case   : bool — True if the subject had any occurrence of this
                        disease atom BEFORE the landmark (prevalent at baseline,
                        excluded from incident-disease eval).

    All three arrays are (n_subjects, n_diseases).
    """
    n_s = len(subjects)
    n_d = len(disease_atom_ids)
    time_to_event = np.full((n_s, n_d), np.nan, dtype=np.float64)
    observed = np.zeros((n_s, n_d), dtype=bool)
    prior_case = np.zeros((n_s, n_d), dtype=bool)
    atom_arr = np.asarray(disease_atom_ids, dtype=np.int64)
    for i, s in enumerate(subjects):
        n_rows = s.end - s.start + 1
        atoms = events.atom.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        times = events.time_seconds.slice(s.start, n_rows).to_numpy(zero_copy_only=False)
        ages_days = (times - s.birth_seconds) / 86400.0
        pre_mask = ages_days < LANDMARK_AGE_DAYS
        post_mask = ages_days >= LANDMARK_AGE_DAYS
        pre_atoms = atoms[pre_mask]
        post_atoms = atoms[post_mask]
        post_ages = ages_days[post_mask]
        horizon_age = min(s.censor_age_days, LANDMARK_AGE_DAYS + HORIZON_DAYS)
        for d_idx, aid in enumerate(atom_arr.tolist()):
            if (pre_atoms == aid).any():
                prior_case[i, d_idx] = True
                # Still fill time_to_event with the censor distance so arrays
                # are non-NaN; downstream Harrell's C will mask by ~prior_case.
                time_to_event[i, d_idx] = horizon_age - LANDMARK_AGE_DAYS
                observed[i, d_idx] = False
                continue
            hits = (post_atoms == aid) & (post_ages <= horizon_age)
            if hits.any():
                first_age = float(post_ages[hits].min())
                time_to_event[i, d_idx] = first_age - LANDMARK_AGE_DAYS
                observed[i, d_idx] = True
            else:
                time_to_event[i, d_idx] = horizon_age - LANDMARK_AGE_DAYS
                observed[i, d_idx] = False
    return time_to_event, observed, prior_case


def _compute_disease_risks(
    model: torch.nn.Module,
    h_last: torch.Tensor,
    gap_to_landmark_days: torch.Tensor,
    disease_atoms_t: torch.Tensor,
    horizon_days: float,
    n_grid: int,
) -> torch.Tensor:
    """Cumulative-incidence risk per (subject, disease), via mark-specific hazard.

    Survival-analysis derivation. Given the model's first-event density
    p_time(Δt | h) and mark distribution p_mark(X | h, Δt), the mark-specific
    *hazard rate* at time Δt is

        λ_X(Δt | h) = p_time(Δt | h) · p_mark(X | h, Δt) / S(Δt | h)

    where S(Δt | h) is the survival function (probability the next event
    hasn't occurred yet by Δt). Under stationarity of the model's predictive
    distribution over the horizon, the cumulative incidence is

        P(≥1 event of mark X in [a, b]) = 1 - exp(-∫_a^b λ_X(Δt | h) dΔt).

    This is the standard cumulative-incidence form for a marked TPP and is
    bounded in [0, 1]. The naive first-event integral ∫ p_time · p_mark
    only counts the case where the FIRST event is X and falls in window —
    which underestimates risk for high-utilization subjects (whose mass is
    near the origin) because lots of mass is "spent" on the first event
    being something else. The hazard form fixes that.

    Per-subject integration window is [gap_to_landmark, gap_to_landmark +
    horizon_days] because we want events in [landmark, landmark + horizon]
    expressed as a Δt-from-last-event window.

    h_last:               (B, D)
    gap_to_landmark_days: (B,) — landmark age minus last-event age
    disease_atoms_t:      (n_diseases,)
    Returns: (B, n_diseases) risk scores in [0, 1].
    """
    device = h_last.device
    B, D = h_last.shape
    tpp = model.model.tpp

    grid_low = gap_to_landmark_days.clamp(min=1.0)
    grid_high = grid_low + float(horizon_days)
    log_lo = grid_low.log().unsqueeze(-1)
    log_hi = grid_high.log().unsqueeze(-1)
    alpha = torch.linspace(0.0, 1.0, n_grid, device=device).unsqueeze(0)
    log_grid = log_lo + alpha * (log_hi - log_lo)
    grid = log_grid.exp()                                                       # (B, G)

    log_w, mu, log_sigma = tpp.time_params(h_last.float())                       # each (B, K)
    inv_sigma = (-log_sigma).exp()

    log_dt = log_grid.unsqueeze(-1)                                              # (B, G, 1)
    log_w_b = log_w.unsqueeze(1)                                                 # (B, 1, K)
    mu_b = mu.unsqueeze(1)
    log_sigma_b = log_sigma.unsqueeze(1)
    inv_sigma_b = inv_sigma.unsqueeze(1)
    z = (log_dt - mu_b) * inv_sigma_b                                            # (B, G, K)
    log_pdf = -log_dt - log_sigma_b - 0.5 * math.log(2 * math.pi) - 0.5 * z.pow(2)
    log_surv_per_mix = _log_ndtr(-z)                                             # (B, G, K)
    log_p_time = torch.logsumexp(log_w_b + log_pdf, dim=-1)                      # (B, G)
    log_surv = torch.logsumexp(log_w_b + log_surv_per_mix, dim=-1)               # (B, G)
    # Clamp survival from below to avoid div-by-zero hazards if the model is
    # over-confident near the tail; 1e-12 is well below anything trainable.
    log_hazard_total = log_p_time - log_surv.clamp(min=math.log(1e-12))           # (B, G)

    G = grid.shape[1]
    h_expanded = h_last.unsqueeze(1).expand(B, G, D).reshape(B * G, D)
    dt_expanded = grid.reshape(B * G)
    mark_lp = tpp.mark_log_probs(h_expanded.float(), dt_expanded)                # (B*G, n_atoms)
    log_mark_X = mark_lp.index_select(-1, disease_atoms_t).view(B, G, -1)        # (B, G, n_disease)

    log_lambda_X = log_hazard_total.unsqueeze(-1) + log_mark_X                   # (B, G, n_disease)
    lambda_X = log_lambda_X.exp()
    dgrid = torch.diff(grid, dim=-1).unsqueeze(-1)                               # (B, G-1, 1)
    avg = 0.5 * (lambda_X[:, :-1] + lambda_X[:, 1:])
    cum_hazard_X = (avg * dgrid).sum(dim=1).clamp(min=0.0)                       # (B, n_disease)
    risk = 1.0 - (-cum_hazard_X).exp()
    return risk


def _harrell_cindex(risks: np.ndarray, time_to_event: np.ndarray, observed: np.ndarray) -> tuple[float, int, int]:
    """Harrell's C with right-censoring. Returns (C, n_pairs, n_concordant_eq)."""
    if len(risks) < 2:
        return float("nan"), 0, 0
    risks = np.asarray(risks, dtype=np.float64)
    t = np.asarray(time_to_event, dtype=np.float64)
    e = np.asarray(observed, dtype=bool)
    event_indices = np.flatnonzero(e)
    concordant = 0.0
    permissible = 0
    for i in event_indices:
        ti = t[i]
        ri = risks[i]
        # A pair (i, j) is permissible iff t[i] < t[j] AND e[i]=True
        # (we enumerate the event subject as i). Ties on time are skipped.
        partners = t > ti
        if not partners.any():
            continue
        rj = risks[partners]
        permissible += int(partners.sum())
        concordant += float(np.sum(ri > rj)) + 0.5 * float(np.sum(ri == rj))
    if permissible == 0:
        return float("nan"), 0, 0
    return concordant / permissible, permissible, int(concordant)


def _resolve_disease_atoms(vocab: AtomVocab) -> tuple[dict[str, int], dict[int, list[str]]]:
    """Resolve each disease code to its post-collapse atom, and return the full
    equivalence class (source codes mapping to that atom) for transparency.
    """
    atom_to_codes: dict[int, list[str]] = defaultdict(list)
    for code, aid in vocab.code_to_atom.items():
        atom_to_codes[int(aid)].append(code)
    for aid in atom_to_codes:
        atom_to_codes[aid].sort()

    resolved: dict[str, int] = {}
    classes: dict[int, list[str]] = {}
    for name, code in DEFAULT_DISEASES.items():
        aid = vocab.encode(code)
        if aid == PAD_ATOM:
            print(f"  [skip] {name} ({code}) — not in vocab (rolled out at collapse threshold)")
            continue
        resolved[name] = aid
        classes[aid] = atom_to_codes.get(aid, [code])
    return resolved, classes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="10-year incident-disease C-index on test cohort.")
    parser.add_argument("--tiny", action="store_true", help="Use runs-tiny/ instead of runs/.")
    args = parser.parse_args(argv)

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
    disease_atoms, eq_classes = _resolve_disease_atoms(vocab)
    if not disease_atoms:
        raise SystemExit("no disease atoms resolved")
    print("  resolved disease atoms (with collapse equivalence class):")
    for name, aid in disease_atoms.items():
        codes = eq_classes[aid]
        head = " | ".join(codes[:3]) + (f"  (+{len(codes)-3} more)" if len(codes) > 3 else "")
        print(f"    {name:24s} atom={aid:>6}  class[{len(codes):>3}]: {head}")
    setup.finish_unit("load vocab + resolve disease atoms", f"diseases_resolved={len(disease_atoms)}")

    setup.start_unit("load shared event store", f"events={etl_dir / 'events.parquet'}")
    events = EventStore.from_parquet(etl_dir / "events.parquet")
    setup.finish_unit("load shared event store", f"rows={events.num_rows:,}")

    setup.start_unit(
        "build subject index",
        f"split=test landmark_age={LANDMARK_AGE_DAYS/365.25:.1f}y horizon={HORIZON_DAYS/365.25:.1f}y "
        f"max_gap={MAX_GAP_DAYS/365.25:.1f}y",
    )
    subjects = _build_subject_index(events, etl_dir)
    if not subjects:
        raise SystemExit("no eligible test subjects after landmark + follow-up + gap filters")
    setup.finish_unit("build subject index", f"eligible={len(subjects):,}")

    setup.start_unit("score cumulative incidence (mark-specific hazard)", f"batch_size={EVAL_BATCH_SIZE} grid={N_QUAD_POINTS}")
    dataset = LandmarkDataset(events, subjects, max_events=4096)
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=runtime.dataloader_pin_memory,
        collate_fn=collate,
    )
    disease_atom_ids = list(disease_atoms.values())
    disease_atoms_t = torch.tensor(disease_atom_ids, dtype=torch.long, device=device)
    all_risks = np.zeros((len(subjects), len(disease_atoms)), dtype=np.float64)
    cursor = 0
    progress_tick = max(1, len(loader) // 40)
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_on_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ac_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else contextlib.nullcontext()
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
            hidden = out["hidden"]                                            # (B, T, D)
            lengths = batch_on_device["length"].clamp(min=1)
            last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
            h_last = hidden.gather(1, last_idx).squeeze(1)                    # (B, D)
            # Gather per-subject gaps for this batch.
            n_b = h_last.shape[0]
            gap_b = torch.tensor(
                [subjects[cursor + j].gap_to_landmark_days for j in range(n_b)],
                dtype=torch.float32, device=device,
            )
            risk = _compute_disease_risks(model, h_last, gap_b, disease_atoms_t, HORIZON_DAYS, N_QUAD_POINTS)
            all_risks[cursor : cursor + n_b] = risk.float().cpu().numpy()
            cursor += n_b
            if batch_idx % progress_tick == 0:
                print(f"  scored {cursor:,}/{len(subjects):,} subjects")
    setup.finish_unit("score cumulative incidence (mark-specific hazard)", f"subjects_scored={cursor:,}")

    setup.start_unit(
        "build outcome table",
        f"window=[landmark, min(landmark+horizon, censor)]  excluding prevalent cases at landmark",
    )
    time_to_event, observed, prior_case = _build_outcome_table(events, subjects, disease_atom_ids)
    setup.finish_unit("build outcome table", f"shape=({len(subjects):,}, {len(disease_atoms)})")

    setup.start_unit("compute Harrell C-index per disease", "incident cases only (prevalents masked out)")
    name_width = max(len(n) for n in disease_atoms)
    header = f"  {'disease':<{name_width}}  {'atom':>6}  {'class':>5}  {'prior':>7}  {'eligible':>8}  {'events':>7}  {'inc%':>6}  {'C-index':>9}  {'pairs':>9}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for d_idx, name in enumerate(disease_atoms):
        eligible_mask = ~prior_case[:, d_idx]
        n_eligible = int(eligible_mask.sum())
        n_prior = int(prior_case[:, d_idx].sum())
        events_observed = int((observed[:, d_idx] & eligible_mask).sum())
        incidence = 100.0 * events_observed / n_eligible if n_eligible else 0.0
        c, n_pairs, _ = _harrell_cindex(
            all_risks[eligible_mask, d_idx],
            time_to_event[eligible_mask, d_idx],
            observed[eligible_mask, d_idx],
        )
        aid = disease_atoms[name]
        class_size = len(eq_classes.get(aid, []))
        c_str = f"{c:.4f}" if not math.isnan(c) else "  nan  "
        print(
            f"  {name:<{name_width}}  {aid:>6}  {class_size:>5}  {n_prior:>7,}  "
            f"{n_eligible:>8,}  {events_observed:>7,}  {incidence:>5.2f}%  {c_str:>9}  {n_pairs:>9,}"
        )
    print("═" * len(header))
    print("  Columns: class = # source codes collapsed into this atom; prior = subjects")
    print("           excluded for prevalent disease at landmark; eligible = n_total - prior;")
    print("           events = incident cases in [landmark, +10y]; inc% = events / eligible;")
    print("           pairs = permissible pairs used in Harrell's C.")
    setup.finish_unit("compute Harrell C-index per disease", f"diseases={len(disease_atoms)}")


if __name__ == "__main__":
    main()
