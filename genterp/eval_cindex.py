"""10-year incident-disease C-index on the held-out test cohort.

Landmark survival analysis. For each subject in the test split:

  1. Landmark = a FIXED calendar age (50 y; constant at top of module).
  2. History = events strictly before landmark.
  3. Last event must be within `MAX_GAP_DAYS` of landmark (≤ 5 y) or the
     subject is skipped — predicting from stale state is uninformative.
  4. Post-landmark observation must extend ≥ 1 y or the subject is skipped.

Phenotype definitions are SNOMED-root + IS-A descendants, NOT a single atom.
For each disease we list the SNOMED parent concept (e.g. "Type 2 diabetes
mellitus" = SNOMED/44054006). All cohort concepts that are descendants of
that parent (via OMOP `concept_ancestor`) are resolved to their post-collapse
atom IDs and unioned. A subject is a case if ≥ 1 event in
[landmark, landmark+10y] hits any atom in the disease set; prevalent if any
pre-landmark event hits the set. This matches the OHDSI concept-set pattern
and is the standard clinically-valid phenotype.

Model risk for the set is the cumulative incidence under the marked TPP:

    λ_set(Δt | h)   = p_time(Δt | h) · ( Σ_{a∈set} p_mark(a | h, Δt) ) / S(Δt | h)
    risk_set        = 1 - exp( -∫_{gap}^{gap+10y} λ_set(Δt | h) dΔt )

This is mathematically equivalent to summing per-atom cumulative hazards and
exponentiating once. Stationarity assumption (h doesn't drift over horizon)
is the only approximation; cumulative-incidence is bounded in [0, 1] and
correctly handles multi-event futures.

Public API (used by genterp.train for periodic in-loop eval):
  - prepare_cindex_cohort(etl_dir, vocab, *, max_events) → cohort dict
  - run_cindex(model, cohort, *, device, autocast_dtype) → per-disease metrics

CLI:
  python -m genterp.eval_cindex            # full
  python -m genterp.eval_cindex --tiny     # tiny
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


# Disease phenotypes — each value is the SNOMED root concept code. The case
# set is the union of all cohort concepts that are IS-A descendants of the
# root (resolved via OMOP concept_ancestor, cached in coverage_and_ancestors).
DEFAULT_DISEASES: dict[str, str] = {
    "Type 2 diabetes mellitus": "SNOMED/44054006",
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

LANDMARK_AGE_DAYS = 50.0 * 365.25
HORIZON_DAYS = 10.0 * 365.25
MIN_FOLLOWUP_DAYS = 365.25
MAX_GAP_DAYS = 5.0 * 365.25
N_QUAD_POINTS = 24
EVAL_BATCH_SIZE = 16
DATALOADER_WORKERS = 2
MAX_EVENTS = 4096


@dataclass
class SubjectIndex:
    subject_id: int
    start: int
    end: int
    birth_seconds: float
    censor_seconds: float
    sex: int
    last_event_idx_local: int
    last_event_age_days: float
    gap_to_landmark_days: float
    censor_age_days: float


def _build_subject_index(events: EventStore, etl_dir: Path) -> list[SubjectIndex]:
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
        time_seconds = events.time_seconds.slice(start, n_rows).to_numpy()
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
    """One batch row per eligible test subject — events strictly before the
    FIXED landmark age. Reuses the training collate.
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
        atoms = self.events.atom.slice(s.start, n_rows).to_numpy()
        times = self.events.time_seconds.slice(s.start, n_rows).to_numpy()
        values = self.events.value.slice(s.start, n_rows).to_numpy()
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


def _find_etl_cache(etl_dir: Path) -> Path | None:
    cache_root = etl_dir / "cache"
    if not cache_root.is_dir():
        return None
    candidates = [d for d in cache_root.iterdir() if d.is_dir() and (d / "concept_codes.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _resolve_disease_atom_sets(
    vocab: AtomVocab, etl_dir: Path
) -> tuple[dict[str, set[int]], dict[str, dict[str, object]]]:
    """For each disease in DEFAULT_DISEASES, return:
      - atom_sets[name]    = set of post-collapse atom IDs covering the SNOMED
                             descendant tree of that disease's root in the cohort
      - phenotype_info[name] = {root_code, root_cid, n_descendant_cids,
                                n_atoms, sample_codes (first few)}
    Walks the ETL's cached coverage_and_ancestors-*.json to find IS-A
    descendants of each root WITHIN the cohort vocab. Descendants that
    didn't survive the 500-subject collapse are silently dropped.
    """
    cache_dir = _find_etl_cache(etl_dir)
    if cache_dir is None:
        raise SystemExit(f"no ETL cache dir under {etl_dir}/cache — run aou_etl.py first")

    cc_pairs = json.loads((cache_dir / "concept_codes.json").read_text())
    cid_to_code = {int(cid): str(code) for cid, code in cc_pairs}
    code_to_cid = {code: cid for cid, code in cid_to_code.items()}

    ca_files = sorted(cache_dir.glob("coverage_and_ancestors-*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not ca_files:
        raise SystemExit(f"no coverage_and_ancestors-*.json under {cache_dir}")
    cov_anc = json.loads(ca_files[0].read_text())

    # Invert: descendants_of[ancestor_cid] = {cohort descendant cids}.
    descendants_of: dict[int, set[int]] = defaultdict(set)
    for entry in cov_anc.get("ancestors", []):
        desc_cid, ancestors_list = int(entry[0]), entry[1]
        descendants_of[desc_cid].add(desc_cid)  # self-descendant
        for anc_pair in ancestors_list:
            anc_cid = int(anc_pair[0])
            descendants_of[anc_cid].add(desc_cid)

    atom_sets: dict[str, set[int]] = {}
    phenotype_info: dict[str, dict[str, object]] = {}
    for name, root_code in DEFAULT_DISEASES.items():
        root_cid = code_to_cid.get(root_code)
        if root_cid is None:
            print(f"  [skip] {name} — root concept {root_code} not in cohort vocabulary (no descendant in AoU)")
            continue
        desc_cids = descendants_of.get(root_cid, set())
        if not desc_cids:
            print(f"  [skip] {name} — root {root_code} (cid={root_cid}) has zero cohort descendants")
            continue
        atoms: set[int] = set()
        sample_codes: list[str] = []
        for cid in desc_cids:
            code = cid_to_code.get(cid)
            if code is None:
                continue
            aid = vocab.encode(code)
            if aid != PAD_ATOM:
                atoms.add(aid)
                if len(sample_codes) < 4:
                    sample_codes.append(code)
        if not atoms:
            print(f"  [skip] {name} — {len(desc_cids)} descendants but none survived vocab collapse")
            continue
        atom_sets[name] = atoms
        phenotype_info[name] = {
            "root_code": root_code,
            "root_cid": int(root_cid),
            "n_descendant_cids": len(desc_cids),
            "n_atoms": len(atoms),
            "sample_codes": sample_codes,
        }
    return atom_sets, phenotype_info


def _build_outcome_table(
    events: EventStore,
    subjects: list[SubjectIndex],
    atom_sets: dict[str, set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (subject, disease_set) return time_to_event, observed, prior_case
    where membership is checked as "atom ∈ disease set" (any descendant counts).
    """
    n_s = len(subjects)
    n_d = len(atom_sets)
    time_to_event = np.full((n_s, n_d), np.nan, dtype=np.float64)
    observed = np.zeros((n_s, n_d), dtype=bool)
    prior_case = np.zeros((n_s, n_d), dtype=bool)
    set_arrays = [np.asarray(sorted(s), dtype=np.int64) for s in atom_sets.values()]
    for i, s in enumerate(subjects):
        n_rows = s.end - s.start + 1
        atoms = events.atom.slice(s.start, n_rows).to_numpy()
        times = events.time_seconds.slice(s.start, n_rows).to_numpy()
        ages_days = (times - s.birth_seconds) / 86400.0
        pre_mask = ages_days < LANDMARK_AGE_DAYS
        post_mask = ages_days >= LANDMARK_AGE_DAYS
        pre_atoms = atoms[pre_mask]
        post_atoms = atoms[post_mask]
        post_ages = ages_days[post_mask]
        horizon_age = min(s.censor_age_days, LANDMARK_AGE_DAYS + HORIZON_DAYS)
        for d_idx, atom_arr in enumerate(set_arrays):
            in_set_pre = np.isin(pre_atoms, atom_arr)
            if in_set_pre.any():
                prior_case[i, d_idx] = True
                time_to_event[i, d_idx] = horizon_age - LANDMARK_AGE_DAYS
                observed[i, d_idx] = False
                continue
            in_set_post = np.isin(post_atoms, atom_arr)
            hits = in_set_post & (post_ages <= horizon_age)
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
    flat_atoms_t: torch.Tensor,
    set_membership: torch.Tensor,  # (n_diseases, n_flat_atoms) bool
    horizon_days: float,
    n_grid: int,
) -> torch.Tensor:
    """Cumulative incidence per disease SET. Sums per-atom cumulative hazards
    across each disease's atom set before exponentiating, which is mathematically
    equivalent to integrating the set-level mark hazard.

    h_last:               (B, D)
    gap_to_landmark_days: (B,)
    flat_atoms_t:         (n_flat_atoms,) — union of atoms across all diseases
    set_membership:       (n_diseases, n_flat_atoms) — one-hot per disease set
    Returns: (B, n_diseases) cumulative incidence in [0, 1].
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

    log_w, mu, log_sigma = tpp.time_params(h_last.float())
    inv_sigma = (-log_sigma).exp()
    log_dt = log_grid.unsqueeze(-1)
    log_w_b = log_w.unsqueeze(1)
    mu_b = mu.unsqueeze(1)
    log_sigma_b = log_sigma.unsqueeze(1)
    inv_sigma_b = inv_sigma.unsqueeze(1)
    z = (log_dt - mu_b) * inv_sigma_b
    log_pdf = -log_dt - log_sigma_b - 0.5 * math.log(2 * math.pi) - 0.5 * z.pow(2)
    log_surv_per_mix = _log_ndtr(-z)
    log_p_time = torch.logsumexp(log_w_b + log_pdf, dim=-1)
    log_surv = torch.logsumexp(log_w_b + log_surv_per_mix, dim=-1)
    log_hazard_total = log_p_time - log_surv.clamp(min=math.log(1e-12))

    G = grid.shape[1]
    h_expanded = h_last.unsqueeze(1).expand(B, G, D).reshape(B * G, D)
    dt_expanded = grid.reshape(B * G)
    mark_lp = tpp.mark_log_probs(h_expanded.float(), dt_expanded)               # (B*G, n_atoms)
    log_mark_flat = mark_lp.index_select(-1, flat_atoms_t).view(B, G, -1)        # (B, G, n_flat)

    # Per-atom mark-specific log-hazard, then cumulative hazard per atom in window.
    log_lambda_per_atom = log_hazard_total.unsqueeze(-1) + log_mark_flat         # (B, G, n_flat)
    lambda_per_atom = log_lambda_per_atom.exp()
    dgrid = torch.diff(grid, dim=-1).unsqueeze(-1)                               # (B, G-1, 1)
    avg = 0.5 * (lambda_per_atom[:, :-1] + lambda_per_atom[:, 1:])
    cum_hazard_per_atom = (avg * dgrid).sum(dim=1).clamp(min=0.0)                # (B, n_flat)

    # Sum cumulative hazards across each disease's atom set (one matmul).
    cum_hazard_set = cum_hazard_per_atom @ set_membership.t().float()             # (B, n_diseases)
    return 1.0 - (-cum_hazard_set).exp()


def _harrell_cindex(risks: np.ndarray, time_to_event: np.ndarray, observed: np.ndarray) -> tuple[float, int]:
    if len(risks) < 2:
        return float("nan"), 0
    risks = np.asarray(risks, dtype=np.float64)
    t = np.asarray(time_to_event, dtype=np.float64)
    e = np.asarray(observed, dtype=bool)
    event_indices = np.flatnonzero(e)
    concordant = 0.0
    permissible = 0
    for i in event_indices:
        partners = t > t[i]
        if not partners.any():
            continue
        rj = risks[partners]
        ri = risks[i]
        permissible += int(partners.sum())
        concordant += float(np.sum(ri > rj)) + 0.5 * float(np.sum(ri == rj))
    if permissible == 0:
        return float("nan"), 0
    return concordant / permissible, permissible


# ───────────────────── Public API for training-loop integration ─────────────────────


@dataclass
class CindexCohort:
    """One-time-built artifact for cheap per-eval C-index scoring.

    The subject index, dataloader, atom sets, outcome table, and the union-of-atoms
    tensor + set membership matrix are all cohort-level (don't depend on the
    model weights). Computed once at training startup and reused on every eval
    cycle so the per-eval cost is just the model forward pass + a small matmul.
    """
    subjects: list[SubjectIndex]
    dataset: LandmarkDataset
    loader: DataLoader
    disease_names: list[str]
    atom_sets: dict[str, set[int]]
    phenotype_info: dict[str, dict[str, object]]
    time_to_event: np.ndarray        # (n_subjects, n_diseases)
    observed: np.ndarray             # (n_subjects, n_diseases)
    prior_case: np.ndarray           # (n_subjects, n_diseases)
    gaps_days: np.ndarray            # (n_subjects,)
    flat_atoms: np.ndarray           # (n_flat,)
    set_membership: np.ndarray       # (n_diseases, n_flat) one-hot float


def prepare_cindex_cohort(
    etl_dir: Path,
    vocab: AtomVocab,
    *,
    events: EventStore | None = None,
    max_events: int = MAX_EVENTS,
    batch_size: int = EVAL_BATCH_SIZE,
    num_workers: int = DATALOADER_WORKERS,
    pin_memory: bool = False,
) -> CindexCohort:
    """Build all cohort-level structures the per-eval scorer needs."""
    if events is None:
        events = EventStore.from_parquet(etl_dir / "events.parquet")
    subjects = _build_subject_index(events, etl_dir)
    atom_sets, phenotype_info = _resolve_disease_atom_sets(vocab, etl_dir)
    if not subjects:
        raise SystemExit("no eligible test subjects after filters")
    if not atom_sets:
        raise SystemExit("no disease atom sets resolved (no SNOMED descendants in cohort)")

    print("  resolved disease phenotypes (SNOMED root → cohort-descendant atom set):")
    for name, info in phenotype_info.items():
        head = " | ".join(info["sample_codes"][:3])
        print(
            f"    {name:30s} root={info['root_code']:18s} "
            f"descendant_cids={info['n_descendant_cids']:>4}  atoms={info['n_atoms']:>4}  "
            f"e.g. {head}"
        )

    time_to_event, observed, prior_case = _build_outcome_table(events, subjects, atom_sets)
    gaps_days = np.asarray([s.gap_to_landmark_days for s in subjects], dtype=np.float64)

    disease_names = list(atom_sets)
    all_atoms = sorted(set.union(*atom_sets.values()))
    flat_atoms = np.asarray(all_atoms, dtype=np.int64)
    atom_pos = {a: i for i, a in enumerate(all_atoms)}
    set_membership = np.zeros((len(disease_names), len(all_atoms)), dtype=np.float32)
    for d_idx, name in enumerate(disease_names):
        for a in atom_sets[name]:
            set_membership[d_idx, atom_pos[a]] = 1.0

    dataset = LandmarkDataset(events, subjects, max_events=max_events)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return CindexCohort(
        subjects=subjects,
        dataset=dataset,
        loader=loader,
        disease_names=disease_names,
        atom_sets=atom_sets,
        phenotype_info=phenotype_info,
        time_to_event=time_to_event,
        observed=observed,
        prior_case=prior_case,
        gaps_days=gaps_days,
        flat_atoms=flat_atoms,
        set_membership=set_membership,
    )


def run_cindex(
    model: torch.nn.Module,
    cohort: CindexCohort,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    progress_every: int = 0,
) -> dict[str, dict[str, object]]:
    """Score risks under the current model + compute Harrell's C per disease.

    Returns a dict keyed by disease name. Per-disease entry:
      {n_eligible, prior_cases, events, incidence_pct, c_index, n_pairs}
    """
    flat_atoms_t = torch.from_numpy(cohort.flat_atoms).to(device)
    set_membership_t = torch.from_numpy(cohort.set_membership).to(device)
    all_risks = np.zeros((len(cohort.subjects), len(cohort.disease_names)), dtype=np.float64)
    cursor = 0
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(cohort.loader):
                batch_on_device = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                }
                ac_ctx = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if use_autocast
                    else contextlib.nullcontext()
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
                hidden = out["hidden"]
                lengths = batch_on_device["length"].clamp(min=1)
                last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
                h_last = hidden.gather(1, last_idx).squeeze(1)
                n_b = h_last.shape[0]
                gap_b = torch.from_numpy(cohort.gaps_days[cursor : cursor + n_b].astype(np.float32)).to(device)
                risk = _compute_disease_risks(
                    model, h_last, gap_b, flat_atoms_t, set_membership_t, HORIZON_DAYS, N_QUAD_POINTS
                )
                all_risks[cursor : cursor + n_b] = risk.float().cpu().numpy()
                cursor += n_b
                if progress_every and batch_idx % progress_every == 0:
                    print(f"  scored {cursor:,}/{len(cohort.subjects):,} subjects")
    finally:
        if was_training:
            model.train()

    results: dict[str, dict[str, object]] = {}
    for d_idx, name in enumerate(cohort.disease_names):
        eligible_mask = ~cohort.prior_case[:, d_idx]
        n_eligible = int(eligible_mask.sum())
        n_prior = int(cohort.prior_case[:, d_idx].sum())
        events_observed = int((cohort.observed[:, d_idx] & eligible_mask).sum())
        incidence = 100.0 * events_observed / n_eligible if n_eligible else 0.0
        c, n_pairs = _harrell_cindex(
            all_risks[eligible_mask, d_idx],
            cohort.time_to_event[eligible_mask, d_idx],
            cohort.observed[eligible_mask, d_idx],
        )
        results[name] = {
            "n_eligible": n_eligible,
            "prior_cases": n_prior,
            "events": events_observed,
            "incidence_pct": incidence,
            "c_index": None if math.isnan(c) else float(c),
            "n_pairs": n_pairs,
        }
    return results


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

    setup.start_unit("load vocab", f"vocab={etl_dir / 'vocab.json'}")
    vocab = AtomVocab(dict(json.loads((etl_dir / "vocab.json").read_text())))
    setup.finish_unit("load vocab", f"atoms={len(vocab):,}")

    setup.start_unit("build cindex cohort", "subjects + atom sets + outcome table")
    cohort = prepare_cindex_cohort(etl_dir, vocab, pin_memory=runtime.dataloader_pin_memory)
    setup.finish_unit("build cindex cohort", f"subjects={len(cohort.subjects):,} diseases={len(cohort.disease_names)}")

    setup.start_unit("score cumulative incidence (mark-specific hazard)", f"batch_size={EVAL_BATCH_SIZE} grid={N_QUAD_POINTS}")
    results = run_cindex(
        model, cohort,
        device=device, autocast_dtype=autocast_dtype,
        progress_every=max(1, len(cohort.loader) // 40),
    )
    setup.finish_unit("score cumulative incidence (mark-specific hazard)", f"subjects_scored={len(cohort.subjects):,}")

    name_width = max(len(n) for n in cohort.disease_names)
    header = f"  {'disease':<{name_width}}  {'root':>14}  {'set':>5}  {'prior':>7}  {'eligible':>8}  {'events':>7}  {'inc%':>6}  {'C-index':>9}  {'pairs':>9}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for name in cohort.disease_names:
        m = results[name]
        info = cohort.phenotype_info[name]
        c = m["c_index"]
        c_str = f"{c:.4f}" if c is not None else "  nan  "
        print(
            f"  {name:<{name_width}}  {info['root_code']:>14}  {info['n_atoms']:>5}  "
            f"{m['prior_cases']:>7,}  {m['n_eligible']:>8,}  {m['events']:>7,}  "
            f"{m['incidence_pct']:>5.2f}%  {c_str:>9}  {m['n_pairs']:>9,}"
        )
    print("═" * len(header))
    print("  Columns: root = SNOMED parent concept defining the phenotype;")
    print("           set  = # cohort atoms (descendants of root) in the disease set;")
    print("           prior = subjects with any set-atom event before landmark (excluded);")
    print("           eligible = total - prior; events = incident set-atom hits in 10-y window;")
    print("           inc% = events / eligible; pairs = permissible pairs in Harrell's C.")

    # Always dump for downstream plotting.
    out_json = runs_dir / "cindex_results.json"
    out_json.write_text(json.dumps({
        "run_dir": str(runs_dir),
        "final_model": str(final),
        "landmark_age_years": LANDMARK_AGE_DAYS / 365.25,
        "horizon_years": HORIZON_DAYS / 365.25,
        "max_gap_years": MAX_GAP_DAYS / 365.25,
        "n_subjects_eligible": len(cohort.subjects),
        "phenotype_info": cohort.phenotype_info,
        "results": results,
    }, indent=2))
    out_npz = runs_dir / "cindex_arrays.npz"
    # Re-run risks-only to also save the underlying matrix for bootstrap CIs.
    flat_atoms_t = torch.from_numpy(cohort.flat_atoms).to(device)
    set_membership_t = torch.from_numpy(cohort.set_membership).to(device)
    all_risks = np.zeros((len(cohort.subjects), len(cohort.disease_names)), dtype=np.float64)
    cursor = 0
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    with torch.no_grad():
        for batch in cohort.loader:
            batch_on_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ac_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if use_autocast else contextlib.nullcontext()
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
            hidden = out["hidden"]
            lengths = batch_on_device["length"].clamp(min=1)
            last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
            h_last = hidden.gather(1, last_idx).squeeze(1)
            n_b = h_last.shape[0]
            gap_b = torch.from_numpy(cohort.gaps_days[cursor : cursor + n_b].astype(np.float32)).to(device)
            risk = _compute_disease_risks(
                model, h_last, gap_b, flat_atoms_t, set_membership_t, HORIZON_DAYS, N_QUAD_POINTS
            )
            all_risks[cursor : cursor + n_b] = risk.float().cpu().numpy()
            cursor += n_b
    np.savez_compressed(
        out_npz,
        disease_names=np.asarray(cohort.disease_names),
        risks=all_risks,
        time_to_event=cohort.time_to_event,
        observed=cohort.observed,
        prior_case=cohort.prior_case,
    )
    print(f"  wrote {out_json}")
    print(f"  wrote {out_npz}")


if __name__ == "__main__":
    main()
