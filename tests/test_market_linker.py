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
    HAS_UNDERLYING_EQUITY_PRED,
    STRADDLE_WITH_PRED,
    CALL_SPREAD_WITH_PRED,
    UNDERLYING_SYMBOL_PRED,
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


# ======================================================================
# Deep: OCC symbol as the only source of contract identity
# ======================================================================

def test_option_links_to_underlying_equity_from_its_occ_symbol(spark, make_triples):
    """An option reaches its underlying equity with no underlyingSymbol stated.

    Quote snapshots emit neither underlyingSymbol nor expirationDate -- those
    belong to the FEEDS vocabulary -- so the inner joins in this step dropped
    every option row and it logged "no option snapshots found" on data made
    entirely of option snapshots. Everything needed is in the OCC symbol.
    """
    opt = "http://example.org/opt/A_260717C00065000"
    eq = "http://example.org/snap/A"
    rows = [
        (opt, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (opt, SYMBOL_PRED, "A     260717C00065000"),
        (opt, CAPTURE_TIME_PRED, "2026-07-02T16:40:40.167Z"),
        (eq, RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (eq, SYMBOL_PRED, "A"),
        (eq, CAPTURE_TIME_PRED, "2026-07-02T16:40:40.167Z"),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert (opt, HAS_UNDERLYING_EQUITY_PRED, eq) in triples


def test_straddle_detected_from_occ_symbols_alone(spark, make_triples):
    """A call and a put on the same underlying/expiry/strike pair as a straddle.

    Strategy detection needs underlying AND expiration, and quotes state
    neither. Both come out of the OCC symbol, so this is the whole step
    working from the identifiers alone.
    """
    call = "http://example.org/opt/A_260717C00065000"
    put = "http://example.org/opt/A_260717P00065000"
    when = "2026-07-02T16:40:40.167Z"
    rows = [
        (call, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (call, SYMBOL_PRED, "A     260717C00065000"),
        (call, STRIKE_PRICE_PRED, "65.0"),
        (call, CONTRACT_TYPE_PRED, "C"),
        (call, CAPTURE_TIME_PRED, when),
        (put, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (put, SYMBOL_PRED, "A     260717P00065000"),
        (put, STRIKE_PRICE_PRED, "65.0"),
        (put, CONTRACT_TYPE_PRED, "P"),
        (put, CAPTURE_TIME_PRED, when),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))

    straddles = {t for t in triples if t[1] == STRADDLE_WITH_PRED}
    assert straddles, "no straddle detected from a matching call/put pair"
    assert any({s, o} == {call, put} for s, _p, o in straddles), (
        f"straddle links do not connect the call to the put: {sorted(straddles)}"
    )


def test_a_stated_underlying_wins_over_the_derived_one(spark, make_triples):
    """What the data says beats what the symbol implies.

    The fallback exists because quotes state nothing; it must not override a
    vocabulary that does. A feeds-style graph keeps its own answer.
    """
    opt = "http://example.org/opt/1"
    eq = "http://example.org/snap/ZZZ"
    rows = [
        (opt, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (opt, SYMBOL_PRED, "A     260717C00065000"),
        (opt, UNDERLYING_SYMBOL_PRED, "ZZZ"),
        (opt, CAPTURE_TIME_PRED, "2026-07-02T16:40:40.167Z"),
        (eq, RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (eq, SYMBOL_PRED, "ZZZ"),
        (eq, CAPTURE_TIME_PRED, "2026-07-02T16:40:40.167Z"),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert (opt, HAS_UNDERLYING_EQUITY_PRED, eq) in triples


# ======================================================================
# Deep: no step may make a node its own object (#360)
#
# Market reaches every property it sequences on through an inner join, so a
# snapshot stating one of them twice arrives at the window twice. The copies
# sorted next to each other and lead() handed the snapshot its own URI back.
# ======================================================================

def test_snapshot_does_not_precede_itself(spark, make_triples):
    """A snapshot stating two capture times must not precede its own copy."""
    dup = "http://example.org/snap/AAPL_1000"
    late = "http://example.org/snap/AAPL_1100"
    rows = [
        (dup, RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (dup, SYMBOL_PRED, "AAPL"),
        (dup, CAPTURE_TIME_PRED, "2024-11-15T10:00:00"),
        (dup, CAPTURE_TIME_PRED, "2024-11-15T10:30:00"),
        (late, RDF_TYPE, EQUITY_SNAPSHOT_TYPE),
        (late, SYMBOL_PRED, "AAPL"),
        (late, CAPTURE_TIME_PRED, "2024-11-15T11:00:00"),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))
    precedes = {(s, o) for s, p, o in triples if p == PRECEDES_PRED}

    assert (dup, dup) not in precedes
    assert (dup, late) in precedes


def test_vertical_spread_does_not_pair_an_option_with_itself(spark, make_triples):
    """An option stating two strikes must not spread with its own copy.

    The strike join is what fans the option out, and strike is also what orders
    the chain, so the copies land next to each other in one (underlying,
    expiration, type, capture_time) partition.
    """
    dup = "http://example.org/opt/A_260717C00065000"
    higher = "http://example.org/opt/A_260717C00070000"
    when = "2026-07-02T16:40:40.167Z"
    rows = [
        (dup, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (dup, SYMBOL_PRED, "A     260717C00065000"),
        (dup, STRIKE_PRICE_PRED, "65.0"),
        (dup, STRIKE_PRICE_PRED, "66.0"),
        (dup, CONTRACT_TYPE_PRED, "C"),
        (dup, CAPTURE_TIME_PRED, when),
        (higher, RDF_TYPE, OPTION_SNAPSHOT_TYPE),
        (higher, SYMBOL_PRED, "A     260717C00070000"),
        (higher, STRIKE_PRICE_PRED, "70.0"),
        (higher, CONTRACT_TYPE_PRED, "C"),
        (higher, CAPTURE_TIME_PRED, when),
    ]

    triples = _triple_set(MarketIntraSourceLinker(spark).enrich(make_triples(rows)))
    spreads = {(s, o) for s, p, o in triples if p == CALL_SPREAD_WITH_PRED}

    assert (dup, dup) not in spreads
    assert (dup, higher) in spreads
