"""Incident-disease C-index on the held-out test cohort.

Landmark survival analysis. For each subject in the test split:

  1. Landmark = `clip(observation_period_end - HORIZON, [40, 85] years)`.
     i.e. the latest age that still leaves a full HORIZON of follow-up,
     bounded to the adult-chronic-disease window. This naturally distributes
     the cohort across ages 40–85 (depending on each subject's birth year and
     loss-to-follow-up timing) instead of pinning everyone at age 50 — no
     extra eval cost.
  2. History = events strictly before the per-subject landmark.
  3. Eligibility filters:
       - ≥ MIN_PRE_LANDMARK_HISTORY_DAYS (6 mo) of observation before landmark
       - last pre-landmark event within MAX_GAP_DAYS (5 y) of landmark
       - ≥ MIN_FOLLOWUP_DAYS (1 y) of observation after landmark
  4. Phenotype = SNOMED root + IS-A descendants (resolved via OMOP
     `concept_ancestor`, cached in coverage_and_ancestors). A subject is a
     CASE if there are ≥ `min_occurrences` (default 2) post-landmark events
     hitting the disease's atom set, with consecutive qualifying events
     ≥ `min_gap_days` (default 30 d) apart — OHDSI/PheKB phecode rule. They
     are PREVALENT (excluded for that disease) if any pre-landmark event hits
     the set. Sex-restricted phenotypes only apply to the matching sex.

Model risk for the set is the cumulative incidence under the marked TPP:

    λ_set(Δt | h)   = p_time(Δt | h) · ( Σ_{a∈set} p_mark(a | h, Δt) ) / S(Δt | h)
    risk_set        = 1 - exp( -∫_{gap}^{gap+HORIZON} λ_set(Δt | h) dΔt )

Where gap = landmark_age - last_event_age. Equivalent to summing per-atom
cumulative hazards and exponentiating once. Stationarity assumption (h
doesn't drift over horizon) is the only approximation; cumulative-incidence
is bounded in [0, 1] and correctly handles multi-event futures.

Public API (used by genterp.train for periodic in-loop eval):
  - prepare_cindex_cohort(etl_dir, vocab, *, events, ...) → CindexCohort
  - run_cindex(model, cohort, *, device, autocast_dtype, bootstrap_resamples)
        → per-disease metrics (plus optional 95% CIs)

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
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class DiseasePhenotype:
    """Clinical phenotype: SNOMED root + a phecode-style occurrence rule.

    Case = ≥`min_occurrences` post-landmark events with atoms in the disease's
    descendant set, with consecutive qualifying events ≥`min_gap_days` apart.

    Prevalent = ≥1 pre-landmark event in the set. Conservative: a single
    prior occurrence excludes the subject from this disease's analysis.

    Sex-restricted phenotypes apply the sex filter to eligibility only
    (e.g. breast cancer eligible cohort = females only).
    """
    name: str
    root_code: str              # "VOCAB/concept_code" matching vocab.json keys
    sex: str | None = None      # None | "M" | "F"
    min_occurrences: int = 2
    min_gap_days: float = 30.0

    @property
    def safe_key(self) -> str:
        s = self.name.lower()
        for ch in " ()'.,":
            s = s.replace(ch, "_")
        return s.strip("_")


DEFAULT_DISEASES: list[DiseasePhenotype] = [
    DiseasePhenotype("Type 2 diabetes mellitus", "SNOMED/44054006"),
    DiseasePhenotype("Essential hypertension",   "SNOMED/59621000"),
    DiseasePhenotype("Acute MI",                 "SNOMED/22298006"),
    DiseasePhenotype("Heart failure",            "SNOMED/84114007"),
    DiseasePhenotype("Atrial fibrillation",      "SNOMED/49436004"),
    DiseasePhenotype("Stroke (CVA)",             "SNOMED/230690007"),
    DiseasePhenotype("Chronic kidney disease",   "SNOMED/709044004"),
    DiseasePhenotype("COPD",                     "SNOMED/13645005"),
    DiseasePhenotype("Alzheimer's disease",      "SNOMED/26929004"),
    DiseasePhenotype("Breast cancer",            "SNOMED/254837009", sex="F"),
    DiseasePhenotype("Prostate cancer",          "SNOMED/399068003", sex="M"),
    DiseasePhenotype("Colorectal cancer",        "SNOMED/93761005"),
]

LANDMARK_AGE_MIN_DAYS = 40.0 * 365.25
LANDMARK_AGE_MAX_DAYS = 85.0 * 365.25
HORIZON_DAYS = 10.0 * 365.25
MIN_FOLLOWUP_DAYS = 365.25
MAX_GAP_DAYS = 5.0 * 365.25
MIN_PRE_LANDMARK_HISTORY_DAYS = 180.0    # ≥6 mo of pre-landmark observation
MIN_EVENTS_FOR_C_SUMMARY = 30            # mean-C ignores diseases below this
DEFAULT_BOOTSTRAP_RESAMPLES = 0          # 0 in-loop; CLI bumps to 500
N_QUAD_POINTS = 24
EVAL_BATCH_SIZE = 16
DATALOADER_WORKERS = 2
MAX_EVENTS = 4096

# AoU sex encoding: aou_etl.py uses IF(gender_concept_id = 8507, 1, 0).
SEX_MALE_CODE = 1
SEX_FEMALE_CODE = 0


def _compute_landmark_age(censor_age_days: float) -> float:
    """Latest landmark that still leaves HORIZON of follow-up, clamped to the
    adult window [LANDMARK_AGE_MIN, LANDMARK_AGE_MAX].
    """
    natural = censor_age_days - HORIZON_DAYS
    return float(np.clip(natural, LANDMARK_AGE_MIN_DAYS, LANDMARK_AGE_MAX_DAYS))


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
    first_event_age_days: float
    landmark_age_days: float             # per-subject landmark — not global
    gap_to_landmark_days: float          # landmark - last_event_age
    censor_age_days: float


def _build_subject_index(events: EventStore, etl_dir: Path) -> list[SubjectIndex]:
    rows = pq.read_table(etl_dir / "subjects.parquet").to_pylist()
    test = [r for r in rows if r.get("split") == "test"]
    eligible: list[SubjectIndex] = []
    skipped_no_history = 0
    skipped_short_history = 0
    skipped_short_followup = 0
    skipped_stale_gap = 0
    skipped_no_window = 0
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
        ages_days = ages_days[ages_days >= 0]
        if ages_days.size == 0:
            skipped_no_history += 1
            continue
        censor_age = (censor_seconds - birth_seconds) / 86400.0
        landmark_age = _compute_landmark_age(censor_age)
        if censor_age - landmark_age < MIN_FOLLOWUP_DAYS:
            skipped_short_followup += 1
            continue
        before = ages_days < landmark_age
        if not before.any():
            skipped_no_history += 1
            continue
        first_event_age = float(ages_days[0])
        last_event_age = float(ages_days[before][-1])
        if landmark_age - first_event_age < MIN_PRE_LANDMARK_HISTORY_DAYS:
            skipped_short_history += 1
            continue
        gap = landmark_age - last_event_age
        if gap > MAX_GAP_DAYS:
            skipped_stale_gap += 1
            continue
        # Edge: landmark could clip to LANDMARK_AGE_MIN even when censor is
        # earlier (rare). Skip if the resulting window is degenerate.
        if landmark_age + HORIZON_DAYS <= landmark_age + 1:
            skipped_no_window += 1
            continue
        # last_event_idx is the position of the last event with age < landmark
        # within the FULL subject window (used by LandmarkDataset).
        full_ages = (events.time_seconds.slice(start, n_rows).to_numpy() - birth_seconds) / 86400.0
        last_event_idx_local = int((full_ages < landmark_age).sum()) - 1
        eligible.append(SubjectIndex(
            subject_id=int(r["subject_id"]),
            start=start,
            end=end,
            birth_seconds=birth_seconds,
            censor_seconds=censor_seconds,
            sex=int(r.get("sex", 0) or 0),
            last_event_idx_local=last_event_idx_local,
            last_event_age_days=last_event_age,
            first_event_age_days=first_event_age,
            landmark_age_days=float(landmark_age),
            gap_to_landmark_days=float(gap),
            censor_age_days=float(censor_age),
        ))
    print(
        f"[eval_cindex] cohort: test_total={len(test):,}  eligible={len(eligible):,}  "
        f"skipped_no_history={skipped_no_history:,}  "
        f"skipped_short_history<{MIN_PRE_LANDMARK_HISTORY_DAYS/30:.0f}mo={skipped_short_history:,}  "
        f"skipped_short_followup<{MIN_FOLLOWUP_DAYS/365.25:.1f}y={skipped_short_followup:,}  "
        f"skipped_stale_gap>{MAX_GAP_DAYS/365.25:.1f}y={skipped_stale_gap:,}"
    )
    if eligible:
        lms = np.asarray([s.landmark_age_days / 365.25 for s in eligible])
        print(
            f"[eval_cindex] landmark ages (years):  "
            f"mean={lms.mean():.1f}  median={np.median(lms):.1f}  "
            f"p10={np.percentile(lms, 10):.1f}  p90={np.percentile(lms, 90):.1f}  "
            f"min={lms.min():.1f}  max={lms.max():.1f}"
        )
    return eligible


class LandmarkDataset(Dataset):
    """One batch row per eligible test subject — events strictly before THAT
    subject's landmark age. Reuses the training collate.
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
        event_mask = (delta_days > 0.5) & real_atom & (delta_days < s.landmark_age_days)
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
) -> tuple[list[DiseasePhenotype], list[set[int]], dict[str, dict[str, object]]]:
    """Resolve each `DiseasePhenotype` to its cohort-descendant atom set.

    Walks the ETL's cached coverage_and_ancestors-*.json to find IS-A
    descendants of each root concept WITHIN the cohort. Descendants that
    didn't survive vocab collapse are silently dropped. Returns parallel
    lists of (phenotypes, atom_sets) restricted to resolvable diseases,
    plus an info dict with the source-code class for transparency.
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

    descendants_of: dict[int, set[int]] = defaultdict(set)
    for entry in cov_anc.get("ancestors", []):
        desc_cid, ancestors_list = int(entry[0]), entry[1]
        descendants_of[desc_cid].add(desc_cid)
        for anc_pair in ancestors_list:
            descendants_of[int(anc_pair[0])].add(desc_cid)

    resolved_phenotypes: list[DiseasePhenotype] = []
    atom_sets: list[set[int]] = []
    info: dict[str, dict[str, object]] = {}
    for pheno in DEFAULT_DISEASES:
        root_cid = code_to_cid.get(pheno.root_code)
        if root_cid is None:
            print(f"  [skip] {pheno.name} — root {pheno.root_code} not in cohort vocab")
            continue
        desc_cids = descendants_of.get(root_cid, set())
        if not desc_cids:
            print(f"  [skip] {pheno.name} — root {pheno.root_code} has no cohort descendants")
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
            print(f"  [skip] {pheno.name} — {len(desc_cids)} descendants but none survived vocab collapse")
            continue
        resolved_phenotypes.append(pheno)
        atom_sets.append(atoms)
        info[pheno.name] = {
            "root_code": pheno.root_code,
            "root_cid": int(root_cid),
            "sex_restriction": pheno.sex,
            "min_occurrences": pheno.min_occurrences,
            "min_gap_days": pheno.min_gap_days,
            "n_descendant_cids": len(desc_cids),
            "n_atoms": len(atoms),
            "sample_codes": sample_codes,
        }
    return resolved_phenotypes, atom_sets, info


def _first_qualifying_age(
    hit_ages: np.ndarray, min_occurrences: int, min_gap_days: float
) -> float | None:
    """Walk sorted hit ages and find the age at which the `min_occurrences`-th
    qualifying event (≥ `min_gap_days` apart) occurs. Returns None if the
    rule isn't satisfied.
    """
    if hit_ages.size < min_occurrences:
        return None
    qual = [float(hit_ages[0])]
    for age in hit_ages[1:]:
        age = float(age)
        if age - qual[-1] >= min_gap_days:
            qual.append(age)
            if len(qual) >= min_occurrences:
                return qual[-1]
    return None


def _build_outcome_table(
    events: EventStore,
    subjects: list[SubjectIndex],
    phenotypes: list[DiseasePhenotype],
    atom_sets: list[set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each (subject, disease) return:
       time_to_event : days from this subject's landmark to first qualifying
                        post-landmark event (in [landmark, landmark+HORIZON]),
                        or to censor if no event qualifies.
       observed     : True if a ≥min_occurrences set-atom phenotype was
                        established post-landmark.
       prior_case   : True if any pre-landmark event hits the set (excluded).
       sex_eligible : True unless this disease is sex-restricted and the
                        subject's sex doesn't match.
    """
    n_s = len(subjects)
    n_d = len(phenotypes)
    time_to_event = np.full((n_s, n_d), np.nan, dtype=np.float64)
    observed = np.zeros((n_s, n_d), dtype=bool)
    prior_case = np.zeros((n_s, n_d), dtype=bool)
    sex_eligible = np.ones((n_s, n_d), dtype=bool)
    set_arrays = [np.asarray(sorted(s), dtype=np.int64) for s in atom_sets]
    for i, s in enumerate(subjects):
        # Per-disease sex eligibility — independent of any event data.
        for d_idx, pheno in enumerate(phenotypes):
            if pheno.sex == "M" and s.sex != SEX_MALE_CODE:
                sex_eligible[i, d_idx] = False
            elif pheno.sex == "F" and s.sex != SEX_FEMALE_CODE:
                sex_eligible[i, d_idx] = False
        n_rows = s.end - s.start + 1
        atoms = events.atom.slice(s.start, n_rows).to_numpy()
        times = events.time_seconds.slice(s.start, n_rows).to_numpy()
        ages_days = (times - s.birth_seconds) / 86400.0
        landmark = s.landmark_age_days
        pre_mask = ages_days < landmark
        post_mask = ages_days >= landmark
        pre_atoms = atoms[pre_mask]
        post_atoms = atoms[post_mask]
        post_ages = ages_days[post_mask]
        horizon_age = min(s.censor_age_days, landmark + HORIZON_DAYS)
        for d_idx, atom_arr in enumerate(set_arrays):
            pheno = phenotypes[d_idx]
            if np.isin(pre_atoms, atom_arr).any():
                prior_case[i, d_idx] = True
                time_to_event[i, d_idx] = horizon_age - landmark
                observed[i, d_idx] = False
                continue
            in_set_post = np.isin(post_atoms, atom_arr) & (post_ages <= horizon_age)
            if not in_set_post.any():
                observed[i, d_idx] = False
                time_to_event[i, d_idx] = horizon_age - landmark
                continue
            hit_ages = np.sort(post_ages[in_set_post])
            case_age = _first_qualifying_age(hit_ages, pheno.min_occurrences, pheno.min_gap_days)
            if case_age is not None:
                observed[i, d_idx] = True
                time_to_event[i, d_idx] = case_age - landmark
            else:
                observed[i, d_idx] = False
                time_to_event[i, d_idx] = horizon_age - landmark
    return time_to_event, observed, prior_case, sex_eligible


def _compute_disease_risks(
    model: torch.nn.Module,
    h_last: torch.Tensor,
    gap_to_landmark_days: torch.Tensor,
    flat_atoms_t: torch.Tensor,
    set_membership: torch.Tensor,
    horizon_days: float,
    n_grid: int,
) -> torch.Tensor:
    """Cumulative incidence per disease SET via mark-specific hazard.

    Math: sum per-atom cumulative hazards across each disease's atom set and
    exponentiate once (1 - exp(-Σ_a Λ_a)). Same answer as integrating the
    set-level mark hazard and bounded in [0, 1]. Implemented as one matmul
    so cost is independent of disease count.
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
    grid = log_grid.exp()

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
    mark_lp = tpp.mark_log_probs(h_expanded.float(), dt_expanded)
    log_mark_flat = mark_lp.index_select(-1, flat_atoms_t).view(B, G, -1)

    log_lambda_per_atom = log_hazard_total.unsqueeze(-1) + log_mark_flat
    lambda_per_atom = log_lambda_per_atom.exp()
    dgrid = torch.diff(grid, dim=-1).unsqueeze(-1)
    avg = 0.5 * (lambda_per_atom[:, :-1] + lambda_per_atom[:, 1:])
    cum_hazard_per_atom = (avg * dgrid).sum(dim=1).clamp(min=0.0)

    cum_hazard_set = cum_hazard_per_atom @ set_membership.t().float()
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


def _bootstrap_c(
    risks: np.ndarray, time_to_event: np.ndarray, observed: np.ndarray,
    n_resamples: int, rng: np.random.Generator,
) -> tuple[float, float, float] | None:
    """Subject-resampled bootstrap of Harrell's C. Returns (lo2.5, med, hi97.5)
    or None if degenerate.
    """
    n = len(risks)
    if n < 2 or n_resamples <= 0:
        return None
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        c, _ = _harrell_cindex(risks[idx], time_to_event[idx], observed[idx])
        if not math.isnan(c):
            samples.append(c)
    if not samples:
        return None
    lo, med, hi = np.percentile(samples, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


# ───────────────────── Public API for training-loop integration ─────────────────────


@dataclass
class CindexCohort:
    subjects: list[SubjectIndex]
    dataset: LandmarkDataset
    loader: DataLoader
    phenotypes: list[DiseasePhenotype]
    disease_names: list[str]
    atom_sets: list[set[int]]
    phenotype_info: dict[str, dict[str, object]]
    time_to_event: np.ndarray
    observed: np.ndarray
    prior_case: np.ndarray
    sex_eligible: np.ndarray
    gaps_days: np.ndarray
    flat_atoms: np.ndarray
    set_membership: np.ndarray
    landmark_summary: dict[str, float] = field(default_factory=dict)


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
    if events is None:
        events = EventStore.from_parquet(etl_dir / "events.parquet")
    subjects = _build_subject_index(events, etl_dir)
    phenotypes, atom_sets, phenotype_info = _resolve_disease_atom_sets(vocab, etl_dir)
    if not subjects:
        raise SystemExit("no eligible test subjects after filters")
    if not phenotypes:
        raise SystemExit("no disease phenotypes resolved (no SNOMED descendants in cohort)")

    print("  resolved disease phenotypes (SNOMED root + cohort-descendant atoms; ≥occ rule):")
    for pheno in phenotypes:
        info = phenotype_info[pheno.name]
        head = " | ".join(info["sample_codes"][:3])
        sex_tag = f" sex={pheno.sex}" if pheno.sex else ""
        rule = f"≥{pheno.min_occurrences} hits ≥{int(pheno.min_gap_days)}d apart"
        print(
            f"    {pheno.name:30s} root={pheno.root_code:18s} "
            f"descendants={info['n_descendant_cids']:>4}  atoms={info['n_atoms']:>4}  "
            f"{rule:24s}{sex_tag}  e.g. {head}"
        )

    time_to_event, observed, prior_case, sex_eligible = _build_outcome_table(
        events, subjects, phenotypes, atom_sets
    )
    gaps_days = np.asarray([s.gap_to_landmark_days for s in subjects], dtype=np.float64)
    landmarks = np.asarray([s.landmark_age_days / 365.25 for s in subjects], dtype=np.float64)
    landmark_summary = {
        "mean_years": float(landmarks.mean()),
        "median_years": float(np.median(landmarks)),
        "min_years": float(landmarks.min()),
        "max_years": float(landmarks.max()),
        "p10_years": float(np.percentile(landmarks, 10)),
        "p90_years": float(np.percentile(landmarks, 90)),
    }

    disease_names = [p.name for p in phenotypes]
    all_atoms = sorted(set.union(*atom_sets))
    flat_atoms = np.asarray(all_atoms, dtype=np.int64)
    atom_pos = {a: i for i, a in enumerate(all_atoms)}
    set_membership = np.zeros((len(disease_names), len(all_atoms)), dtype=np.float32)
    for d_idx, s in enumerate(atom_sets):
        for a in s:
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
        phenotypes=phenotypes,
        disease_names=disease_names,
        atom_sets=atom_sets,
        phenotype_info=phenotype_info,
        time_to_event=time_to_event,
        observed=observed,
        prior_case=prior_case,
        sex_eligible=sex_eligible,
        gaps_days=gaps_days,
        flat_atoms=flat_atoms,
        set_membership=set_membership,
        landmark_summary=landmark_summary,
    )


def _score_risks(
    model: torch.nn.Module,
    cohort: CindexCohort,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    progress_every: int = 0,
) -> np.ndarray:
    flat_atoms_t = torch.from_numpy(cohort.flat_atoms).to(device)
    set_membership_t = torch.from_numpy(cohort.set_membership).to(device)
    all_risks = np.zeros((len(cohort.subjects), len(cohort.disease_names)), dtype=np.float64)
    cursor = 0
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    with torch.no_grad():
        for batch_idx, batch in enumerate(cohort.loader):
            batch_on_device = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
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
            if progress_every and batch_idx % progress_every == 0:
                print(f"  scored {cursor:,}/{len(cohort.subjects):,} subjects")
    return all_risks


def run_cindex(
    model: torch.nn.Module,
    cohort: CindexCohort,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = 0,
    progress_every: int = 0,
) -> dict[str, dict[str, object]]:
    """Score risks under the current model + compute Harrell's C per disease.

    Eligibility per disease = ~prior_case AND sex_eligible. Bootstrap CIs are
    computed only when `bootstrap_resamples > 0` (off by default; CLI sets
    it to 500, in-loop training keeps it at 0 to bound per-eval cost).
    """
    was_training = model.training
    model.eval()
    try:
        all_risks = _score_risks(model, cohort, device, autocast_dtype, progress_every)
    finally:
        if was_training:
            model.train()

    rng = np.random.default_rng(rng_seed)
    results: dict[str, dict[str, object]] = {}
    for d_idx, name in enumerate(cohort.disease_names):
        eligible_mask = (~cohort.prior_case[:, d_idx]) & cohort.sex_eligible[:, d_idx]
        n_eligible = int(eligible_mask.sum())
        n_prior = int(cohort.prior_case[:, d_idx].sum())
        n_sex_excl = int((~cohort.sex_eligible[:, d_idx]).sum())
        events_observed = int((cohort.observed[:, d_idx] & eligible_mask).sum())
        incidence = 100.0 * events_observed / n_eligible if n_eligible else 0.0
        r = all_risks[eligible_mask, d_idx]
        t = cohort.time_to_event[eligible_mask, d_idx]
        o = cohort.observed[eligible_mask, d_idx]
        c, n_pairs = _harrell_cindex(r, t, o)
        ci_band = _bootstrap_c(r, t, o, bootstrap_resamples, rng) if bootstrap_resamples > 0 else None
        results[name] = {
            "n_eligible": n_eligible,
            "prior_cases": n_prior,
            "sex_excluded": n_sex_excl,
            "events": events_observed,
            "incidence_pct": incidence,
            "c_index": None if math.isnan(c) else float(c),
            "n_pairs": n_pairs,
            "c_index_lo": ci_band[0] if ci_band else None,
            "c_index_hi": ci_band[2] if ci_band else None,
            "bootstrap_resamples": bootstrap_resamples if ci_band else 0,
        }
    # Summary: mean C across diseases with ≥MIN_EVENTS_FOR_C_SUMMARY events.
    valid = [
        m["c_index"] for m in results.values()
        if m["c_index"] is not None and m["events"] >= MIN_EVENTS_FOR_C_SUMMARY
    ]
    if valid:
        results["__summary__"] = {  # leading underscore so it's distinguishable
            "cindex_mean_well_powered": float(np.mean(valid)),
            "n_well_powered_diseases": len(valid),
            "min_events_threshold": MIN_EVENTS_FOR_C_SUMMARY,
        }
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Incident-disease C-index on test cohort.")
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

    setup.start_unit("build cindex cohort", "per-subject landmark + outcome table")
    cohort = prepare_cindex_cohort(etl_dir, vocab, pin_memory=runtime.dataloader_pin_memory)
    setup.finish_unit(
        "build cindex cohort",
        f"subjects={len(cohort.subjects):,} diseases={len(cohort.disease_names)} "
        f"landmark_mean={cohort.landmark_summary['mean_years']:.1f}y",
    )

    setup.start_unit("score cumulative incidence (mark-specific hazard)", f"batch_size={EVAL_BATCH_SIZE} grid={N_QUAD_POINTS}")
    results = run_cindex(
        model, cohort,
        device=device, autocast_dtype=autocast_dtype,
        bootstrap_resamples=500,
        progress_every=max(1, len(cohort.loader) // 40),
    )
    setup.finish_unit("score cumulative incidence (mark-specific hazard)", f"subjects_scored={len(cohort.subjects):,}")

    name_width = max(len(n) for n in cohort.disease_names)
    header = (
        f"  {'disease':<{name_width}}  {'root':>14}  {'sex':>3}  {'set':>5}  "
        f"{'prior':>6}  {'sex_ex':>6}  {'eligible':>8}  {'events':>6}  {'inc%':>6}  "
        f"{'C (95% CI)':<22}  {'pairs':>9}"
    )
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for name in cohort.disease_names:
        m = results[name]
        info = cohort.phenotype_info[name]
        c = m["c_index"]
        c_str = f"{c:.4f}" if c is not None else "  nan  "
        if m["c_index_lo"] is not None and m["c_index_hi"] is not None:
            c_str = f"{c:.3f} [{m['c_index_lo']:.3f},{m['c_index_hi']:.3f}]"
        sex_tag = info["sex_restriction"] or "-"
        print(
            f"  {name:<{name_width}}  {info['root_code']:>14}  {sex_tag:>3}  {info['n_atoms']:>5}  "
            f"{m['prior_cases']:>6,}  {m['sex_excluded']:>6,}  {m['n_eligible']:>8,}  "
            f"{m['events']:>6,}  {m['incidence_pct']:>5.2f}%  {c_str:<22}  {m['n_pairs']:>9,}"
        )
    print("═" * len(header))
    summary = results.get("__summary__")
    if summary:
        print(
            f"  cindex_mean (≥{MIN_EVENTS_FOR_C_SUMMARY} events): "
            f"{summary['cindex_mean_well_powered']:.4f}  "
            f"over {summary['n_well_powered_diseases']} well-powered diseases"
        )
    print(
        f"  landmark ages (y):  mean={cohort.landmark_summary['mean_years']:.1f}  "
        f"median={cohort.landmark_summary['median_years']:.1f}  "
        f"p10={cohort.landmark_summary['p10_years']:.1f}  "
        f"p90={cohort.landmark_summary['p90_years']:.1f}"
    )
    print("  Columns: root = SNOMED parent concept; sex = phenotype sex restriction;")
    print("           set = # cohort atoms (descendants); prior = pre-landmark hits;")
    print("           sex_ex = excluded by sex; eligible = total - prior - sex_ex;")
    print("           events = ≥2-occurrence post-landmark hits in 10-y window;")
    print("           pairs = permissible pairs in Harrell's C.")

    out_json = runs_dir / "cindex_results.json"
    out_json.write_text(json.dumps({
        "run_dir": str(runs_dir),
        "final_model": str(final),
        "landmark_age_min_years": LANDMARK_AGE_MIN_DAYS / 365.25,
        "landmark_age_max_years": LANDMARK_AGE_MAX_DAYS / 365.25,
        "horizon_years": HORIZON_DAYS / 365.25,
        "max_gap_years": MAX_GAP_DAYS / 365.25,
        "min_pre_landmark_history_days": MIN_PRE_LANDMARK_HISTORY_DAYS,
        "min_events_for_summary": MIN_EVENTS_FOR_C_SUMMARY,
        "landmark_summary": cohort.landmark_summary,
        "n_subjects_eligible": len(cohort.subjects),
        "phenotype_info": cohort.phenotype_info,
        "results": {k: v for k, v in results.items() if not k.startswith("_")},
        "summary": summary,
    }, indent=2))
    out_npz = runs_dir / "cindex_arrays.npz"
    all_risks = _score_risks(model, cohort, device, autocast_dtype, progress_every=0)
    np.savez_compressed(
        out_npz,
        disease_names=np.asarray(cohort.disease_names),
        risks=all_risks,
        time_to_event=cohort.time_to_event,
        observed=cohort.observed,
        prior_case=cohort.prior_case,
        sex_eligible=cohort.sex_eligible,
        landmark_age_days=np.asarray([s.landmark_age_days for s in cohort.subjects], dtype=np.float64),
    )
    print(f"  wrote {out_json}")
    print(f"  wrote {out_npz}")


if __name__ == "__main__":
    main()
