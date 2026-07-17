"""
Tests for the temporal entity unifier (TemporalUnifier).

Mirrors the linker test files (test_{bls,noaa,market,sec,cross_source}_linker.py):
  - Same-period observations from *different* sources (a BLS month URI and a
    market timestamp literal) collapse onto a single unified month entity,
    asserted at the value level (shared unified URI + owl:sameAs links).
  - Different-period observations are NOT unified — two distinct entities,
    no cross-period sameAs.
  - The market option expiration-date path (market:expirationDate) is pinned:
    this module previously shipped a reference to a non-existent
    MARKET_OPTIONS symbol that broke build_graph in all modes (Bug B), so the
    period derivation through that predicate gets a direct value-level test.
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
    UNIFIED_YEAR_TYPE,
    MARKET_OBSERVED_AT,
    MARKET_EXPIRATION_DATE,
    MARKET_OPTION_CONTRACT_TYPE,
)

CPI = "https://www.bls.gov/cpi/"
MARKET_TEMPORAL = "https://financial-data.org/temporal/"


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


def test_same_period_across_sources_unifies_to_one_entity(spark, make_triples):
    """A BLS month URI and a market timestamp in the same month both link
    from the SAME unified month entity via owl:sameAs."""
    cpi_month = CPI + "November"
    rows = [
        (CPI + "obs/1", CPI + "hasMonth", cpi_month),
        ("https://financial-data.org/snap/1", MARKET_OBSERVED_AT,
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
    cpi_month = CPI + "November"
    rows = [
        (CPI + "obs/1", CPI + "hasMonth", cpi_month),
        ("https://financial-data.org/snap/1", MARKET_OBSERVED_AT,
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


def test_market_expiration_date_derives_period(spark, make_triples):
    """Option expiration dates (market:expirationDate) derive month + year
    temporals — the path whose broken symbol reference (Bug B) once took
    down build_graph in all modes."""
    opt = "https://financial-data.org/opt/1"
    rows = [
        (opt, RDF_TYPE, MARKET_OPTION_CONTRACT_TYPE),
        (opt, MARKET_EXPIRATION_DATE, "2025-01-17"),
    ]

    triples = _triple_set(TemporalUnifier(spark).enrich(make_triples(rows)))

    assert (UNIFIED_BASE + "January", OWL_SAME_AS,
            MARKET_TEMPORAL + "January") in triples
    assert (UNIFIED_BASE + "January", RDF_TYPE, UNIFIED_MONTH_TYPE) in triples
    assert (UNIFIED_BASE + "Year2025", OWL_SAME_AS,
            MARKET_TEMPORAL + "2025") in triples
    assert (UNIFIED_BASE + "Year2025", RDF_TYPE, UNIFIED_YEAR_TYPE) in triples


def test_non_temporal_input_short_circuits(spark, make_triples):
    rows = [("http://example.org/x", RDF_TYPE, "http://example.org/Widget")]
    result = TemporalUnifier(spark).enrich(make_triples(rows))
    assert result.count() == 0
