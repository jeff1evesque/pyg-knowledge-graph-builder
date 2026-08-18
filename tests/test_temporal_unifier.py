"""
Tests for the temporal entity unifier (TemporalUnifier).

Mirrors the linker test files (test_{bls,noaa,market,sec,cross_source}_linker.py):
  - Same-period observations from *different* sources (a BLS month URI and a
    market timestamp literal) collapse onto a single unified month entity,
    asserted at the value level (shared unified URI + owl:sameAs links).
  - Different-period observations are NOT unified — two distinct entities,
    no cross-period sameAs.
  - The market period path (market-quotes:captureTime) is pinned. It replaced
    an expirationDate/observedAt pair from a feeds vocabulary that no longer
    exists; that pair is also where a reference to a non-existent
    MARKET_OPTIONS symbol once broke build_graph in all modes (Bug B).
  - Non-temporal input short-circuits to zero triples.

Drives enrich() over tiny in-memory triples on the shared local SparkSession
(`spark` / `make_triples` fixtures from conftest.py).
"""
from spark_jobs.enrichment.temporal_unifier import (
    TemporalUnifier,
    RDF_TYPE,
    RDFS_LABEL,
    OWL_SAME_AS,
    UNIFIED_BASE,
    UNIFIED_MONTH_TYPE,
    SOURCE_MONTH_TYPE,
    SOURCE_YEAR_TYPE,
    MARKET_CAPTURE_TIME,
    OBSERVED_IN_PERIOD_PRED,
)

from spark_jobs.utils.rdf_utils import (
    CPI as _CPI_NS,
    MARKET_QUOTES,
    SEC_FILINGS,
    SYNTHETIC_TEMPORAL_IDS,
    identifier_namespace,
)

# Read from the constants rather than restating them. A literal here goes stale
# silently the moment a namespace moves — which is exactly what happened: this
# file kept asserting the pre-migration home of the synthetic temporals while
# the unifier had already moved them, and the four failures read as a code
# regression rather than a stale test.
#
# Split deliberately: CPI names the PREDICATES (hasMonth, hasYear), CPI_ID names
# the THINGS (the observation, the period). Sharing one constant for both is the
# conflation that hid the collection bug, since a test whose periods live under
# ontology/ cannot notice that no real period ever does.
CPI = str(_CPI_NS)
CPI_ID = identifier_namespace(CPI)
SEC_TEMPORAL = SYNTHETIC_TEMPORAL_IDS["sec"]
MARKET_TEMPORAL = SYNTHETIC_TEMPORAL_IDS["market-quotes"]
QUOTES_TEMPORAL = MARKET_TEMPORAL


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


def test_same_period_across_sources_unifies_to_one_entity(spark, make_triples):
    """A BLS month URI and a market timestamp in the same month both link
    from the SAME unified month entity via owl:sameAs."""
    cpi_month = CPI_ID + "November"
    rows = [
        (CPI_ID + "obs/1", CPI + "hasMonth", cpi_month),
        ("https://jefflevesque.com/id/market-quotes/snap/1", MARKET_CAPTURE_TIME,
         "2024-11-15T10:00:00"),
    ]

    triples = _triple_set(TemporalUnifier(spark).enrich(make_triples(rows)))

    unified = UNIFIED_BASE + "November"
    assert (unified, OWL_SAME_AS, cpi_month) in triples
    assert (unified, OWL_SAME_AS, MARKET_TEMPORAL + "November") in triples
    assert (unified, RDF_TYPE, UNIFIED_MONTH_TYPE) in triples
    assert (unified, RDFS_LABEL, "November") in triples

    # Exactly ONE unified month entity — both sources collapsed onto it
    month_entities = {s for s, p, o in triples
                      if p == RDF_TYPE and o == UNIFIED_MONTH_TYPE}
    assert month_entities == {unified}


def test_different_periods_are_not_unified(spark, make_triples):
    """A BLS November and a market March produce two distinct unified
    entities with no cross-period sameAs links."""
    cpi_month = CPI_ID + "November"
    rows = [
        (CPI_ID + "obs/1", CPI + "hasMonth", cpi_month),
        ("https://jefflevesque.com/id/market-quotes/snap/1", MARKET_CAPTURE_TIME,
         "2024-03-10T10:00:00"),
    ]

    triples = _triple_set(TemporalUnifier(spark).enrich(make_triples(rows)))

    november = UNIFIED_BASE + "November"
    march = UNIFIED_BASE + "March"
    market_march = MARKET_TEMPORAL + "March"

    # Each source links only to its own period's entity
    assert (november, OWL_SAME_AS, cpi_month) in triples
    assert (march, OWL_SAME_AS, market_march) in triples
    assert (november, OWL_SAME_AS, market_march) not in triples
    assert (march, OWL_SAME_AS, cpi_month) not in triples

    month_entities = {s for s, p, o in triples
                      if p == RDF_TYPE and o == UNIFIED_MONTH_TYPE}
    assert month_entities == {november, march}


def test_source_temporal_uris_are_typed(spark, make_triples):
    """The source-side temporal URIs get an rdf:type of their own.

    They arrive from the sources bare — `cpi:February` with no type anywhere —
    and node_mapper only creates nodes for typed URIs. Untyped, they were not
    nodes, so every measurement->period triple pointing at them AND the unifier's
    own sameAs links to them were dropped during edge resolution: the graph had
    no temporal dimension and the Unified* nodes were isolated. Typing them is
    what makes both hops of the cross-source bridge into real edges.
    """
    cpi_month, cpi_year = CPI_ID + "November", CPI_ID + "2024"
    rows = [
        (CPI_ID + "obs/1", CPI + "hasMonth", cpi_month),
        (CPI_ID + "obs/1", CPI + "hasYear", cpi_year),
        ("https://jefflevesque.com/id/market-quotes/snap/1", MARKET_CAPTURE_TIME,
         "2024-11-15T10:00:00"),
    ]

    triples = _triple_set(TemporalUnifier(spark).enrich(make_triples(rows)))

    assert (cpi_month, RDF_TYPE, SOURCE_MONTH_TYPE) in triples
    assert (cpi_month, RDFS_LABEL, "November") in triples
    assert (cpi_year, RDF_TYPE, SOURCE_YEAR_TYPE) in triples

    # The synthetic URIs minted from date literals are equally untyped at the
    # source, so they need typing too.
    assert (MARKET_TEMPORAL + "November", RDF_TYPE, SOURCE_MONTH_TYPE) in triples

    # Source and unified temporal entities stay distinguishable: the sameAs
    # target must not carry the type of the canonical entity pointing at it.
    assert (cpi_month, RDF_TYPE, UNIFIED_MONTH_TYPE) not in triples
    assert (UNIFIED_BASE + "November", RDF_TYPE, SOURCE_MONTH_TYPE) not in triples


def test_dated_entities_reach_the_period_they_state(spark, make_triples):
    """A source that states a date must end up ATTACHED to the period.

    The date-literal sources (SEC, NOAA, market-feeds) have no equivalent of the
    BLS `obs hasMonth cpi:February` triple — they state a date and nothing else.
    The collector read the date value and threw away the subject, so the unifier
    minted temporal/sec/August, linked it sameAs unified:August, and stopped:
    nothing pointed at the period. That is a bridge with no on-ramp, and it
    still produced Unified* nodes and sameAs edges, so the graph read as
    connected while every SEC filing sat outside the temporal spine.

    Asserted through to the unified entity, because reaching temporal/sec/August
    is only useful if that is the same August the other sources reach.
    """
    filing = "https://jefflevesque.com/id/sec-filings/0000842657-26-000011_Filing"
    rows = [
        (filing, RDF_TYPE, str(SEC_FILINGS.SECFiling)),
        # Stated as a URI rather than a literal, which is the live shape.
        (filing, str(SEC_FILINGS.hasFilingDate),
         "https://jefflevesque.com/id/sec-filings/Date_2026-08-14"),
    ]

    triples = _triple_set(TemporalUnifier(spark).enrich(make_triples(rows)))

    sec_august = SEC_TEMPORAL + "August"
    assert (filing, OBSERVED_IN_PERIOD_PRED, sec_august) in triples, (
        "the filing does not reach the period it dates — the synthetic period "
        "node is an island"
    )
    assert (filing, OBSERVED_IN_PERIOD_PRED, SEC_TEMPORAL + "2026") in triples
    assert (UNIFIED_BASE + "August", OWL_SAME_AS, sec_august) in triples

    # The on-ramp is something this pipeline inferred, not a fact SEC reported.
    from spark_jobs.utils.rdf_utils import classify_edge_origin

    assert classify_edge_origin(
        OBSERVED_IN_PERIOD_PRED, "filings_SECFiling", "temporal_SourceMonth"
    ) == "enrichment"


def test_quote_snapshots_reach_the_spine_via_capture_time(spark, make_triples):
    """The quotes vocabulary gets its own period, off captureTime.

    Market was the one source contributing no temporal entity at all: the market
    collector keys on the FEEDS predicates (observedAt, expirationDate) and a
    quote snapshot has neither, so it matched nothing. A snapshot's only other
    route into the graph is its ticker, which reaches nothing when no filing
    names the same company — so market could be entirely disconnected while the
    unifier reported success.

    quoteTime and tradeTime are deliberately NOT collected: they are epoch
    milliseconds, and the date parser reads `(\\d{4})` as a year, so
    "1783009237630" would mint a period called 1783.
    """
    snap = "https://jefflevesque.com/id/market-quotes/snapshot/A/2026-07-02"
    rows = [
        (snap, RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (snap, str(MARKET_QUOTES.captureTime), "2026-07-02T16:40:40.167Z"),
        (snap, str(MARKET_QUOTES.quoteTime), "1783009237630"),
    ]

    triples = _triple_set(TemporalUnifier(spark).enrich(make_triples(rows)))

    quotes_july = QUOTES_TEMPORAL + "July"
    assert (snap, OBSERVED_IN_PERIOD_PRED, quotes_july) in triples
    assert (UNIFIED_BASE + "July", OWL_SAME_AS, quotes_july) in triples
    assert (snap, OBSERVED_IN_PERIOD_PRED, QUOTES_TEMPORAL + "2026") in triples

    # The epoch-millis predicate must not have minted a period of its own.
    assert not [t for t in triples if t[2].endswith("/1783")], (
        "an epoch-millisecond timestamp was parsed as a calendar year"
    )


def test_source_temporal_types_are_not_pipeline_minted_for_origin(spark):
    """A measurement->period edge stays `raw`, and only the sameAs is unification.

    `cpi:obs hasMonth cpi:February` is an observed source fact; only its TYPE
    comes from this pipeline. Putting SourceMonth under an enrichment namespace
    would flip 1,205 reported facts to `enrichment` in graph_schema.json — the
    exact mislabelling classify_edge_origin exists to prevent.
    """
    from spark_jobs.utils.rdf_utils import classify_edge_origin

    assert classify_edge_origin(
        CPI + "hasMonth", "cpi_Index", "temporal_SourceMonth"
    ) == "raw"
    assert classify_edge_origin(
        OWL_SAME_AS, "bls_enrichment_UnifiedMonth", "temporal_SourceMonth"
    ) == "unification"


def test_non_temporal_input_short_circuits(spark, make_triples):
    rows = [("http://example.org/x", RDF_TYPE, "http://example.org/Widget")]
    result = TemporalUnifier(spark).enrich(make_triples(rows))
    assert result.count() == 0
