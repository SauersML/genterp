"""--tiny toggles a 10× data subsample + smaller cache namespace.

Tiny mode is wired in two layers:
  - scripts/aou_etl.py: main() parses --tiny, sets module-level TINY, each
    per-domain SQL injects MOD(person_id, N)=0 inside its WHERE, and
    _cache_key gains a _tiny{N}x suffix so tiny artifacts don't collide.
  - genterp/train.py: main() parses --tiny and builds a smaller GenterpConfig.

Tests here only cover (a). The training-side switch is a tiny if/else and the model
shape is exercised by tests/test_smoke.py with custom configs.
"""

from __future__ import annotations

from scripts import aou_etl


def _with_tiny(monkeypatch, enabled: bool):
    monkeypatch.setattr(aou_etl, "TINY", enabled)


def test_tiny_predicate_empty_when_disabled(monkeypatch):
    _with_tiny(monkeypatch, False)
    assert aou_etl._tiny_predicate("person_id") == ""
    assert aou_etl._tiny_predicate("m.person_id") == ""


def test_tiny_predicate_injects_mod_filter_when_enabled(monkeypatch):
    _with_tiny(monkeypatch, True)
    n = aou_etl.TINY_PERSON_MOD
    assert aou_etl._tiny_predicate("person_id") == f" AND MOD(person_id, {n}) = 0"
    assert aou_etl._tiny_predicate("de.person_id") == f" AND MOD(de.person_id, {n}) = 0"


def test_cache_key_namespaces_tiny_separately(monkeypatch):
    _with_tiny(monkeypatch, False)
    full = aou_etl._cache_key("proj.ds")
    _with_tiny(monkeypatch, True)
    tiny = aou_etl._cache_key("proj.ds")

    assert "tiny" not in full
    assert tiny.endswith(f"_tiny{aou_etl.TINY_PERSON_MOD}x")
    assert tiny != full


def test_tiny_filter_lands_in_per_domain_sql(monkeypatch):
    _with_tiny(monkeypatch, True)
    n = aou_etl.TINY_PERSON_MOD
    cdr = "proj.ds"

    assert f"MOD(de.person_id, {n}) = 0" in aou_etl._drug_events_sql(cdr)
    assert f"MOD(m.person_id, {n}) = 0" in aou_etl._measurement_events_sql(cdr)
    assert f"MOD(person_id, {n}) = 0" in aou_etl._non_drug_events_cte(cdr, with_time=True)


def test_main_parses_tiny_flag_and_rejects_unknown(monkeypatch):
    """main() exposes --tiny through argparse so `run.sh --tiny` propagates."""
    import argparse
    import pytest

    # Quick-exit main() before it tries to talk to BigQuery: swap WORKSPACE_CDR check.
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)

    with pytest.raises(SystemExit):
        aou_etl.main(["--tiny"])  # exits on missing WORKSPACE_CDR — but argparse accepted --tiny
    assert aou_etl.TINY is True

    with pytest.raises((SystemExit, argparse.ArgumentError)):
        aou_etl.main(["--no-such-flag"])
