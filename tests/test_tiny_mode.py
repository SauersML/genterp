"""GENTERP_TINY=1 toggles a 10× data subsample + smaller cache namespace.

Tiny mode is wired in two layers:
  - scripts/aou_etl.py: each per-domain SQL gets a MOD(person_id, N)=0 filter, and
    _cache_key gains a _tiny{N}x suffix so tiny artifacts don't collide with full.
  - genterp/train.py: the GenterpConfig in main() shrinks from dim=512/8/8 to
    dim=32/2/2 when GENTERP_TINY=1.

Tests here only cover (a). The training-side switch is a tiny if/else and the model
shape is already exercised by tests/test_smoke.py with custom configs.
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
