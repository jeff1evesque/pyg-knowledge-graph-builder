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
from spark_jobs.enrichment.intra_source.market import correlations as market_corr
from spark_jobs.enrichment.intra_source.sec import patterns as sec


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


# market/measurements.py and sec/measurements.py were deleted, and the test that
# stood here went with the market one. It asserted that a config no production
# code imported was well-formed -- the only thing keeping either file alive, and
# a check that could never fail in a way that mattered. Both files described the
# classes, date properties and grouping keys their linkers already hardcode, so
# they were a second description of the code sitting beside it, consulted by
# nothing while being maintained as though they were wired up.
#
# bls/measurements.py is the one that stays, because bls/base_enricher.py really
# does loop over it: adding an entry there changes what the pipeline emits.
#
# The SEC config also described two entries nothing implements -- transaction-
# level temporal sequencing over NonDerivativeTransaction (18.16% of filings)
# and DerivativeTransaction (7.05%). That is real missing graph structure rather
# than dead config, so it was raised as its own issue instead of being deleted
# silently.


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



# ======================================================================
# Edge origin classification (rdf_utils.classify_edge_origin)
# ======================================================================

def test_classify_edge_origin_by_predicate_namespace():
    """A minted predicate means the pipeline inferred the link."""
    from spark_jobs.utils.rdf_utils import (
        classify_edge_origin, BLS_ENRICHMENT, SEC_ENRICHMENT,
        NOAA_ENRICHMENT, MARKET_ENRICHMENT, CPI,
    )

    for ns in (BLS_ENRICHMENT, SEC_ENRICHMENT, NOAA_ENRICHMENT,
               MARKET_ENRICHMENT):
        assert classify_edge_origin(f"{ns}linkedTo", "a_X", "b_Y") == (
            "enrichment"
        ), ns

    assert classify_edge_origin(f"{CPI}hasValue", "cpi_S", "cpi_I") == "raw"
    # Missing predicate and plain endpoints must not raise.
    assert classify_edge_origin("", "cpi_S", "cpi_I") == "raw"


def test_unification_is_detected_via_endpoints_not_predicate():
    """The regression that made this function take endpoints at all.

    Unification links carry a MINTED NODE but a STANDARD predicate
    (unified:November owl:sameAs cpi:November). Classifying on the predicate
    alone reported them as "raw" -- an inferred link presented as an observed
    fact, which is exactly the mislabelling edge origin exists to prevent. On
    the e2e fixtures that was 16 of 19 supposedly-raw edge types.
    """
    from spark_jobs.utils.rdf_utils import classify_edge_origin, OWL_SAME_AS

    # owl:sameAs is NOT in any pipeline namespace ...
    assert classify_edge_origin(OWL_SAME_AS, "cpi_Month", "ppi_Month") == "raw"
    # ... so only the minted endpoint reveals the edge as pipeline-made.
    assert classify_edge_origin(
        OWL_SAME_AS, "bls_enrichment_UnifiedMonth", "cpi_PercentChange"
    ) == "unification"
    assert classify_edge_origin(
        OWL_SAME_AS, "cpi_PercentChange", "bls_enrichment_UnifiedMonth"
    ) == "unification"


def test_an_inferred_bls_link_is_still_classified_as_enrichment():
    """The behaviour this used to guard, kept after the namespaces moved.

    It previously asserted that BLS_ENRICHMENT sits UNDER https://jefflevesque.com/ontology/bls-common/
    and is therefore at risk of being swallowed by the shorter source
    namespace. That arrangement was the defect: a URI under bls.gov claims BLS
    defined a term we invented. The minted vocabularies now live under a
    domain this project controls, which dissolves the shadowing hazard here
    rather than managing it -- the general ordering invariant is asserted over
    the whole table in tests/test_namespaces.py.

    What must not change is the verdict: an inferred BLS link is enrichment,
    not raw.
    """
    from spark_jobs.utils.rdf_utils import (
        classify_edge_origin, BLS_ENRICHMENT, ONTOLOGY_BASE,
    )

    assert str(BLS_ENRICHMENT).startswith(ONTOLOGY_BASE)
    assert "bls.gov" not in str(BLS_ENRICHMENT)
    assert classify_edge_origin(
        f"{BLS_ENRICHMENT}apparelSectorCorrelation", "a_X", "b_Y"
    ) == "enrichment"


def test_pipeline_node_type_prefixes_derive_from_the_namespace_registry():
    """Adding an enrichment namespace must not need a second list updated."""
    from spark_jobs.utils.rdf_utils import PIPELINE_NODE_TYPE_PREFIXES

    assert "bls_enrichment_" in PIPELINE_NODE_TYPE_PREFIXES
    assert "unified_" in PIPELINE_NODE_TYPE_PREFIXES
    # Source namespaces must NOT be treated as pipeline-minted.
    assert "cpi_" not in PIPELINE_NODE_TYPE_PREFIXES
    assert "cap_" not in PIPELINE_NODE_TYPE_PREFIXES
