"""
Structural / integrity tests for the per-source enrichment pattern tables
(NOAA / market / SEC).

These are deliberately *shallow but broad*: pure-Python, no SparkSession. They
pin the shape of the hand-maintained pattern dictionaries and the pure helper
functions — nothing that touches the distributed enrichment code paths.

Rationale (ROI vs. maintenance debt):
  - The pattern tables are hand-edited data. The real bug class is a malformed
    entry (missing key, empty keyword list, typo'd/duplicated relation) that
    *silently drops enrichment* on the cluster with no error. A structural
    guard catches that class at ~zero runtime cost and only fails when someone
    genuinely changes the data — which is exactly when a human should re-check.
  - We assert *invariants* (every entry has a relationship URI, keyword lists
    are non-empty, ranks are distinct), not the full contents. That keeps the
    tests from breaking every time a legitimate new event type / sector is
    added, so they carry very little maintenance debt.
  - The end-to-end linkers (which need Spark) are intentionally out of scope
    here; see test_bls_enrichment.py for that heavier, more brittle tier.
"""
import pytest

from spark_jobs.enrichment.intra_source.noaa import patterns as noaa
from spark_jobs.enrichment.intra_source.market import patterns as market
from spark_jobs.enrichment.intra_source.market import measurements as market_meas
from spark_jobs.enrichment.intra_source.market import correlations as market_corr
from spark_jobs.enrichment.intra_source.sec import patterns as sec
from spark_jobs.enrichment.intra_source.sec import correlations as sec_corr


def _is_uri(value):
    """Accept both str and rdflib.URIRef; require an http(s) IRI."""
    return str(value).startswith(("http://", "https://"))


def _distinct_positive_ranks(hierarchy):
    """A rank map is valid if values are ints and (for a strict hierarchy) unique."""
    values = list(hierarchy.values())
    assert all(isinstance(v, int) for v in values)
    assert len(values) >= 2
    return values


# ======================================================================
# NOAA
# ======================================================================

def test_noaa_event_patterns_well_formed():
    assert noaa.NOAA_EVENT_PATTERNS, "no NOAA event patterns defined"
    for key, entry in noaa.NOAA_EVENT_PATTERNS.items():
        assert {"description", "event_types", "cap_category", "relationship"} <= set(entry), key
        assert entry["event_types"], f"{key} has no event_types"
        assert all(isinstance(e, str) and e for e in entry["event_types"]), key
        assert _is_uri(entry["relationship"]), f"{key} relationship not a URI"
        assert isinstance(entry["cap_category"], str) and entry["cap_category"], key


def test_noaa_event_types_unique_across_categories():
    """A given warning string must map to exactly one category, else enrichment
    would double-classify or race depending on dict order."""
    seen = {}
    for key, entry in noaa.NOAA_EVENT_PATTERNS.items():
        for event in entry["event_types"]:
            assert event not in seen, f"'{event}' in both {seen.get(event)} and {key}"
            seen[event] = key


def test_noaa_relationships_unique_per_category():
    rels = [str(e["relationship"]) for e in noaa.NOAA_EVENT_PATTERNS.values()]
    assert len(rels) == len(set(rels)), "duplicate relationship URIs across categories"


@pytest.mark.parametrize("hierarchy", [
    noaa.SEVERITY_HIERARCHY,
    noaa.URGENCY_HIERARCHY,
    noaa.CERTAINTY_HIERARCHY,
])
def test_noaa_strict_hierarchies_have_distinct_ranks(hierarchy):
    values = _distinct_positive_ranks(hierarchy)
    assert len(values) == len(set(values)), "strict CAP hierarchy must have unique ranks"
    assert all(_is_uri(k) for k in hierarchy), "hierarchy keys must be CAP URIs"


def test_noaa_response_type_severity_monotonic_range():
    # RESPONSE_TYPE_SEVERITY intentionally reuses ranks (None/AllClear both 0),
    # so only assert the range/typing, not uniqueness.
    values = _distinct_positive_ranks(noaa.RESPONSE_TYPE_SEVERITY)
    assert min(values) >= 0
    assert all(_is_uri(k) for k in noaa.RESPONSE_TYPE_SEVERITY)


# ======================================================================
# market
# ======================================================================

@pytest.mark.parametrize("gics,key,pascal", [
    ("Information Technology", "information_technology_sector", "InformationTechnology"),
    ("Health Care", "health_care_sector", "HealthCare"),
])
def test_market_gics_sector_helpers(gics, key, pascal):
    assert market._gics_sector_to_key(gics) == key
    assert market._gics_sector_to_pascal(gics) == pascal


def test_market_default_sector_keys_follow_convention():
    for key in market.DEFAULT_MARKET_SECTOR_PATTERNS:
        assert key.endswith("_sector"), f"sector key {key!r} missing _sector suffix"


def test_market_option_strategy_patterns_well_formed():
    assert market.MARKET_OPTION_STRATEGY_PATTERNS
    for key, entry in market.MARKET_OPTION_STRATEGY_PATTERNS.items():
        assert _is_uri(entry["relationship"]), key
        assert _is_uri(entry["pattern_uri"]), key
        assert entry["description"], key


def test_market_measurement_types_have_class_and_properties():
    assert market_meas.MEASUREMENT_TYPES
    for group, classes in market_meas.MEASUREMENT_TYPES.items():
        assert classes, f"measurement group {group} is empty"
        for cls_name, props in classes.items():
            assert _is_uri(props["class"]), f"{group}/{cls_name} class not a URI"
            assert any(_is_uri(v) for v in props.values()), f"{group}/{cls_name} has no URI props"


def test_market_known_correlations_well_formed():
    corr = market_corr.KNOWN_CORRELATIONS
    assert corr
    names = [c["name"] for c in corr]
    assert len(names) == len(set(names)), "duplicate correlation names"
    for c in corr:
        assert {"source_type", "target_type", "relationship", "strength"} <= set(c), c["name"]
        assert _is_uri(c["relationship"]), c["name"]
        assert c["strength"] in {"weak", "medium", "strong"}, c["name"]


# ======================================================================
# SEC
# ======================================================================

@pytest.mark.parametrize("table,uri_key", [
    ("SEC_SECTOR_PATTERNS", "sector_uri"),
    ("SEC_VIOLATION_PATTERNS", "violation_uri"),
    ("SEC_COMPANY_STATUS_PATTERNS", "status_uri"),
])
def test_sec_pattern_tables_well_formed(table, uri_key):
    patterns = getattr(sec, table)
    assert patterns, f"{table} is empty"
    for key, entry in patterns.items():
        assert _is_uri(entry[uri_key]), f"{table}/{key} {uri_key} not a URI"
        assert _is_uri(entry["relationship"]), f"{table}/{key} relationship not a URI"
        assert entry["keywords"], f"{table}/{key} has no keyword groups"
        # A per-dataset keyword list may be empty by design ("don't match this
        # sector via this dataset"), but the entry must be matchable somewhere.
        assert any(entry["keywords"].values()), f"{table}/{key} matchable via no dataset"
        for group, kws in entry["keywords"].items():
            assert all(isinstance(k, str) and k for k in kws), f"{table}/{key}/{group}"


def test_sec_known_correlations_well_formed():
    corr = sec_corr.KNOWN_CORRELATIONS
    assert corr
    names = [c["name"] for c in corr]
    assert len(names) == len(set(names)), "duplicate correlation names"
    for c in corr:
        assert {"source_dataset", "target_dataset", "match_strategy",
                "relationship", "strength"} <= set(c), c["name"]
        assert _is_uri(c["relationship"]), c["name"]
        assert c["strength"] in {"weak", "medium", "strong"}, c["name"]
