"""
Tests for the NOAA intra-source enrichment linker (NOAAIntraSourceLinker).

Two tiers, mirroring test_bls_linker.py:
  - Tier 2 smokes: enrich() classifies a known event, and foreign input
    short-circuits to zero triples.
  - Targeted deep test: severity escalation (`_link_severity_escalations`) —
    the window-ordered, rank-comparison logic that is easy to regress. Covers
    the positive case, the non-escalation case (severity drops), and the
    secondary urgency-tiebreak signal.

Drives enrich() over tiny in-memory triples on the shared local SparkSession
(`spark` / `make_triples` fixtures from conftest.py).
"""

from spark_jobs.utils.rdf_utils import NOAA_ENRICHMENT

from spark_jobs.enrichment.intra_source.noaa_linker import (
    NOAAIntraSourceLinker,
    NOAA_EVENT_PATTERNS,
    SEVERITY_HIERARCHY,
    URGENCY_HIERARCHY,
    RDF_TYPE,
    WEATHER_ALERT_TYPE,
    HAS_INFO,
    HAS_EVENT,
    HAS_EVENT_CLASSIFICATION,
    HAS_SENT_TIME,
    HAS_SEVERITY,
    HAS_URGENCY,
    HAS_AREA,
    HAS_AREA_DESC,
    ESCALATES_TO_PRED,
)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


def _severity_uri(level):
    """The CAP severity URI at a given rank (Minor=1 .. Extreme=4)."""
    return next(uri for uri, lvl in SEVERITY_HIERARCHY.items() if lvl == level)


def _urgency_uri(level):
    """The CAP urgency URI at a given rank (Past=1 .. Immediate=4)."""
    return next(uri for uri, lvl in URGENCY_HIERARCHY.items() if lvl == level)


def _alert(rows, alert, info, area_desc, sent_time, severity_uri, urgency_uri):
    """Append the triples for one fully-specified alert to `rows`."""
    area = f"{alert}/area"
    rows.extend([
        (alert, RDF_TYPE, WEATHER_ALERT_TYPE),
        (alert, HAS_INFO, info),
        (info, HAS_SENT_TIME, sent_time),
        (info, HAS_SEVERITY, severity_uri),
        (info, HAS_URGENCY, urgency_uri),
        (info, HAS_AREA, area),
        (area, HAS_AREA_DESC, area_desc),
    ])


# ======================================================================
# Tier 2 smokes
# ======================================================================

def test_noaa_enrich_classifies_event_category(spark, make_triples):
    """WeatherAlert with cap:hasEvent "Tornado Warning" -> severe_weather."""
    alert = "http://example.org/alert/1"
    info = "http://example.org/alert/1/info"
    rows = [
        (alert, RDF_TYPE, WEATHER_ALERT_TYPE),
        (alert, HAS_INFO, info),
        (info, HAS_EVENT, "Tornado Warning"),
    ]

    triples = _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))

    category_uri = str(NOAA_ENRICHMENT["severe_weather"])
    relationship = str(NOAA_EVENT_PATTERNS["severe_weather"]["relationship"])
    assert (alert, HAS_EVENT_CLASSIFICATION, category_uri) in triples
    assert (alert, relationship, category_uri) in triples


def test_noaa_enrich_empty_for_non_noaa_data(spark, make_triples):
    rows = [("http://example.org/x", RDF_TYPE, "http://example.org/SomethingElse")]
    result = NOAAIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# Deep: severity escalation
# ======================================================================

def test_severity_escalation_links_increasing_severity(spark, make_triples):
    """Same area, Moderate -> Severe over time yields a directional escalatesTo."""
    a1, a2 = "http://example.org/alert/a1", "http://example.org/alert/a2"
    rows = []
    _alert(rows, a1, a1 + "/i", "Test County",
           "2024-11-15T10:00:00", _severity_uri(2), _urgency_uri(3))
    _alert(rows, a2, a2 + "/i", "Test County",
           "2024-11-15T11:00:00", _severity_uri(3), _urgency_uri(3))

    triples = _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert (a1, ESCALATES_TO_PRED, a2) in triples
    # Directional: the higher-severity alert does not escalate back down.
    assert (a2, ESCALATES_TO_PRED, a1) not in triples


def test_no_escalation_when_severity_decreases(spark, make_triples):
    """Severe -> Minor is a de-escalation: no escalatesTo edge."""
    a1, a2 = "http://example.org/alert/b1", "http://example.org/alert/b2"
    rows = []
    _alert(rows, a1, a1 + "/i", "Calm County",
           "2024-11-15T10:00:00", _severity_uri(3), _urgency_uri(3))
    _alert(rows, a2, a2 + "/i", "Calm County",
           "2024-11-15T11:00:00", _severity_uri(1), _urgency_uri(3))

    triples = _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert not any(p == ESCALATES_TO_PRED for _, p, _ in triples)


def test_escalation_via_urgency_when_severity_equal(spark, make_triples):
    """Same severity but rising urgency (Expected -> Immediate) still escalates."""
    a1, a2 = "http://example.org/alert/c1", "http://example.org/alert/c2"
    rows = []
    _alert(rows, a1, a1 + "/i", "Urgent County",
           "2024-11-15T10:00:00", _severity_uri(3), _urgency_uri(3))
    _alert(rows, a2, a2 + "/i", "Urgent County",
           "2024-11-15T11:00:00", _severity_uri(3), _urgency_uri(4))

    triples = _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert (a1, ESCALATES_TO_PRED, a2) in triples
