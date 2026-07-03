"""
Tests for the market intra-source enrichment linker (MarketIntraSourceLinker).

Two tiers, mirroring test_bls_linker.py:
  - Tier 2 smokes: enrich() sequences snapshots by time, and foreign input
    short-circuits to zero triples.
  - Targeted deep test: moneyness classification (`_compute_moneyness`) — the
    2%-band ATM threshold plus call/put ITM/OTM arithmetic, which is real math
    and easy to regress. Parametrized over all five branches.

Drives enrich() over tiny in-memory triples on the shared local SparkSession
(`spark` / `make_triples` fixtures from conftest.py).
"""
import pytest

from spark_jobs.enrichment.intra_source.market_linker import (
    MarketIntraSourceLinker,
    RDF_TYPE,
    EQUITY_SNAPSHOT_TYPE,
    OPTION_SNAPSHOT_TYPE,
    SYMBOL_PRED,
    CAPTURE_TIME_PRED,
    STRIKE_PRICE_PRED,
    UNDERLYING_PRICE_PRED,
    CONTRACT_TYPE_PRED,
    PRECEDES_PRED,
    HAS_MONEYNESS_PRED,
    ATM_URI,
    ITM_URI,
    OTM_URI,
)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# Tier 2 smokes
# ======================================================================

def test_market_enrich_sequences_snapshots_by_time(spark, make_triples):
    """Two EquitySnapshots for one symbol link earlier -> later via precedes."""
    early = "http://example.org/snap/AAPL_1000"
    late = "http://example.org/snap/AAPL_1100"
    rows = [
        (late, RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (late, SYMBOL_PRED, "AAPL"),
        (late, CAPTURE_TIME_PRED, "2024-11-15T11:00:00"),
        (early, RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (early, SYMBOL_PRED, "AAPL"),
        (early, CAPTURE_TIME_PRED, "2024-11-15T10:00:00"),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert (early, PRECEDES_PRED, late) in triples
    assert (late, PRECEDES_PRED, early) not in triples


def test_market_enrich_empty_for_non_market_data(spark, make_triples):
    rows = [("http://example.org/x", RDF_TYPE, "http://example.org/NotASnapshot")]
    result = MarketIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# Deep: moneyness classification
# ======================================================================

@pytest.mark.parametrize("contract,strike,underlying,expected", [
    ("CALL", 90.0, 100.0, "ITM"),    # call, strike below spot -> in the money
    ("CALL", 110.0, 100.0, "OTM"),   # call, strike above spot -> out of the money
    ("PUT", 110.0, 100.0, "ITM"),    # put, strike above spot -> in the money
    ("PUT", 90.0, 100.0, "OTM"),     # put, strike below spot -> out of the money
    ("CALL", 101.0, 100.0, "ATM"),   # within 2% band -> at the money (overrides side)
    ("PUT", 99.0, 100.0, "ATM"),     # within 2% band from the other side
])
def test_market_moneyness_classification(
    spark, make_triples, contract, strike, underlying, expected
):
    uri = {"ATM": ATM_URI, "ITM": ITM_URI, "OTM": OTM_URI}[expected]
    opt = "http://example.org/opt/1"
    rows = [
        (opt, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (opt, STRIKE_PRICE_PRED, str(strike)),
        (opt, UNDERLYING_PRICE_PRED, str(underlying)),
        (opt, CONTRACT_TYPE_PRED, contract),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert (opt, HAS_MONEYNESS_PRED, uri) in triples
