from __future__ import annotations

from scripts.aou_etl import (
    EVENT_CACHE_SCHEMA,
    _cache_key,
    _coverage_sql,
    _drug_events_sql,
    _observation_string_value_counts_sql,
    _own_counts_sql,
)


def test_drug_event_counts_and_coverage_use_ingredient_concepts():
    cdr = "project.dataset"

    for sql in (_drug_events_sql(cdr), _coverage_sql(cdr), _own_counts_sql(cdr)):
        assert "drug_strength" in sql
        assert "ingredient_concept_id AS cid" in sql
        assert "drug_concept_id AS cid" not in sql


def test_observation_string_counts_require_event_timestamp():
    sql = _observation_string_value_counts_sql("project.dataset")

    assert "o.observation_datetime IS NOT NULL" in sql
    assert "LOWER(o.value_as_string)" in sql
    assert "LENGTH(n.normalized_value) <= 64" in sql
    assert "REGEXP_CONTAINS(n.normalized_value, r'@')" in sql
    assert "REGEXP_CONTAINS(n.normalized_value, r'\\d{7,}')" in sql


def test_event_cache_key_includes_role_schema_version():
    key = _cache_key("project.dataset")

    assert EVENT_CACHE_SCHEMA in key
    assert "values-v5" not in key
