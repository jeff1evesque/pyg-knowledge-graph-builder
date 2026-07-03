"""
Tier-2 end-to-end smoke tests for the NOAA / market / SEC intra-source linkers.

Mirrors the shape of test_bls_enrichment.py: drive each linker's `enrich()`
over a tiny in-memory (subject, predicate, object) DataFrame on a local
SparkSession and assert on the *output* triples.

Scope is deliberately one representative pass per source, NOT exhaustive:
  - one happy path that proves a real enrichment relationship is produced, and
  - one negative path proving foreign / empty input short-circuits to zero new
    triples (the has_<source> gate at the top of every enrich()).

This is the intentional Tier-2 ceiling. It couples to each linker's output
schema and a single predicate choice — enough to catch a broken code path on
the cluster without the maintenance debt of pinning every relationship/window.
The exhaustive per-relationship tier (Tier 3) is intentionally not built.

Uses the session-scoped `spark` and `make_triples` fixtures from conftest.py.
"""
from spark_jobs.utils.rdf_utils import NOAA_ENRICHMENT

from spark_jobs.enrichment.intra_source.noaa_linker import (
    NOAAIntraSourceLinker,
    NOAA_EVENT_PATTERNS,
    RDF_TYPE as NOAA_RDF_TYPE,
    WEATHER_ALERT_TYPE,
    HAS_INFO,
    HAS_EVENT,
    HAS_EVENT_CLASSIFICATION,
)
from spark_jobs.enrichment.intra_source.market_linker import (
    MarketIntraSourceLinker,
    RDF_TYPE as MARKET_RDF_TYPE,
    EQUITY_SNAPSHOT_TYPE,
    SYMBOL_PRED,
    CAPTURE_TIME_PRED,
    PRECEDES_PRED,
)
from spark_jobs.enrichment.intra_source.sec_linker import (
    SECIntraSourceLinker,
    SEC_VIOLATION_PATTERNS,
    _RDF_TYPE as SEC_RDF_TYPE,
    _SEC_ADMIN_NS,
    _ADMIN_PROCEEDING_TYPE,
    _HAS_VIOLATION_TYPE,
)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# NOAA
# ======================================================================

def test_noaa_enrich_classifies_event_category(spark, make_triples):
    """
    A single WeatherAlert whose Info carries cap:hasEvent = "Tornado Warning"
    is classified into the severe_weather category via NOAA_EVENT_PATTERNS.
    """
    alert = "http://example.org/alert/1"
    info = "http://example.org/alert/1/info"
    rows = [
        (alert, NOAA_RDF_TYPE, WEATHER_ALERT_TYPE),
        (alert, HAS_INFO, info),
        (info, HAS_EVENT, "Tornado Warning"),
    ]

    result = NOAAIntraSourceLinker(spark).enrich(make_triples(rows))
    triples = _triple_set(result)

    category_uri = str(NOAA_ENRICHMENT["severe_weather"])
    relationship = str(NOAA_EVENT_PATTERNS["severe_weather"]["relationship"])
    # Generic classification triple + the category-specific relationship triple.
    assert (alert, HAS_EVENT_CLASSIFICATION, category_uri) in triples
    assert (alert, relationship, category_uri) in triples


def test_noaa_enrich_empty_for_non_noaa_data(spark, make_triples):
    rows = [("http://example.org/x", NOAA_RDF_TYPE, "http://example.org/SomethingElse")]
    result = NOAAIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# market
# ======================================================================

def test_market_enrich_sequences_snapshots_by_time(spark, make_triples):
    """
    Two EquitySnapshots for the same symbol link earlier -> later via
    market_enrichment:precedes, regardless of input row order.
    """
    early = "http://example.org/snap/AAPL_1000"
    late = "http://example.org/snap/AAPL_1100"
    rows = [
        (late, MARKET_RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (late, SYMBOL_PRED, "AAPL"),
        (late, CAPTURE_TIME_PRED, "2024-11-15T11:00:00"),
        (early, MARKET_RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (early, SYMBOL_PRED, "AAPL"),
        (early, CAPTURE_TIME_PRED, "2024-11-15T10:00:00"),
    ]

    result = MarketIntraSourceLinker(spark).enrich(make_triples(rows))
    triples = _triple_set(result)

    assert (early, PRECEDES_PRED, late) in triples
    # precedes is directional: no backward edge.
    assert (late, PRECEDES_PRED, early) not in triples


def test_market_enrich_empty_for_non_market_data(spark, make_triples):
    rows = [("http://example.org/x", MARKET_RDF_TYPE, "http://example.org/NotASnapshot")]
    result = MarketIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# SEC
# ======================================================================

def test_sec_enrich_tags_violation_type_by_uri_keyword(spark, make_triples):
    """
    An administrative-proceeding entity whose URI contains the normalized
    "Securities Fraud" keyword is tagged with hasViolationType ->
    SecuritiesFraudViolation.
    """
    entity = f"{_SEC_ADMIN_NS}SecuritiesFraud_Proceeding_001"
    rows = [
        (entity, SEC_RDF_TYPE, _ADMIN_PROCEEDING_TYPE),
    ]

    result = SECIntraSourceLinker(spark).enrich(make_triples(rows))
    triples = _triple_set(result)

    violation_uri = str(SEC_VIOLATION_PATTERNS["securities_fraud"]["violation_uri"])
    assert (entity, _HAS_VIOLATION_TYPE, violation_uri) in triples


def test_sec_enrich_empty_for_non_sec_data(spark, make_triples):
    rows = [(f"{EQUITY_SNAPSHOT_TYPE}", MARKET_RDF_TYPE, EQUITY_SNAPSHOT_TYPE)]
    result = SECIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0
