"""
Tests for the cross-source enrichment linker (CrossSourceLinker).

Mirrors the per-source linker files (test_{bls,noaa,market,sec}_linker.py):
  - Tier 2 smokes: enrich() links entities from *different* source families
    through shared keys (ticker symbol, unified month), a single detected
    source short-circuits to zero triples, and two sources with nothing
    linkable produce no edges touching the input entities.
  - Targeted deep test: the NOAA geographic FIPS chain in
    `_link_by_geography` — the four-hop Alert -> Info -> Area -> Geocode
    traversal joined against the state-FIPS lookup, which is the module's
    trickiest computation and would regress silently (alerts simply lose
    their affectsRegion edges). Parametrized over mapped and unmapped codes.

Drives CrossSourceLinker over tiny in-memory triples on the shared local
SparkSession (`spark` / `make_triples` fixtures from conftest.py).
"""
import pytest

from rdflib.namespace import OWL, RDF, RDFS

from spark_jobs.enrichment.cross_source_linker import CrossSourceLinker
from spark_jobs.utils.rdf_utils import (
    ALERT, BLS_ENRICHMENT, CAP, CPI, MARKET_QUOTES, WEATHER,
    SEC_FILINGS, UNIFIED,
)

RDF_TYPE = str(RDF.type)
OWL_SAME_AS = str(OWL.sameAs)
RDFS_LABEL = str(RDFS.label)

REFERS_TO_COMPANY = str(BLS_ENRICHMENT.refersToCompany)
UNIFIED_COMPANY_TYPE = str(BLS_ENRICHMENT.UnifiedCompany)
UNIFIED_MONTH_TYPE = str(BLS_ENRICHMENT.UnifiedMonth)
AFFECTS_REGION = str(BLS_ENRICHMENT.affectsRegion)
LEADS_TO = str(BLS_ENRICHMENT.leadsTo)
TICKER_PRED = str(BLS_ENRICHMENT.ticker)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# Tier 2 smokes
# ======================================================================

def test_cross_source_company_linking_shares_unified_company(spark, make_triples):
    """A market quote and an SEC issuer with the same symbol both link to
    the same unified Company entity (the cross-source hub).

    Both sides of this join were dead. The market side asked for
    market-feeds:StockTicker, a class nothing declares -- the model is
    EquitySnapshot / OptionSnapshot -- so it matched nothing. The SEC
    side asked for hasIssuerTicker, which upstream renamed to
    hasIssuerTradingSymbol and moved off the filing onto the filings:Issuer
    node. The previous version of this test asserted the dead shape was correct,
    using a fixture built to match it — which is why the suite stayed green
    while SEC and market built as disconnected islands.

    The SEC subject here is the ISSUER, not the filing: a ticker identifies a
    company, and a filing reaches that company through its hasIssuer edge.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0000320193"
    rows = [
        (snapshot, RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (snapshot, str(MARKET_QUOTES.symbol), "AAPL"),
        (issuer, RDF_TYPE, str(SEC_FILINGS.Issuer)),
        (issuer, str(SEC_FILINGS.hasIssuerTradingSymbol), "AAPL"),
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_AAPL"
    assert (snapshot, REFERS_TO_COMPANY, company) in triples
    assert (issuer, REFERS_TO_COMPANY, company) in triples
    assert (company, RDF_TYPE, UNIFIED_COMPANY_TYPE) in triples
    assert (company, TICKER_PRED, "AAPL") in triples


def test_cross_source_company_linking_resolves_occ_option_symbols(
    spark, make_triples
):
    """An option snapshot joins on the equity its contract is written against.

    The quote API writes option symbols in OCC form — the ticker left-justified
    in a six-character field, then YYMMDD, then C/P, then the strike. Matched
    literally, "A     260717C00065000" equals no issuer's ticker and every
    OptionSnapshot stays isolated, which is what the fixtures showed: two option
    nodes with zero incident edges.
    """
    option = str(MARKET_QUOTES) + "snapshot/A_260717C00065000/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0001090872"
    rows = [
        (option, RDF_TYPE, str(MARKET_QUOTES.OptionSnapshot)),
        (option, str(MARKET_QUOTES.symbol), "A     260717C00065000"),
        (issuer, RDF_TYPE, str(SEC_FILINGS.Issuer)),
        (issuer, str(SEC_FILINGS.hasIssuerTradingSymbol), "A"),
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_A"
    assert (option, REFERS_TO_COMPANY, company) in triples, (
        "the option did not resolve to its underlying equity's company"
    )
    assert (issuer, REFERS_TO_COMPANY, company) in triples


def test_cross_source_detects_market_from_the_quotes_namespace(spark, make_triples):
    """Quote snapshots must register as a market source.

    _detect_sources gates the company/ticker step behind 'market' being in
    available_sources, and it tested subjects against the FEEDS namespace alone.
    Every market node in the e2e fixtures is a quote snapshot, so the gate was
    always shut and the bridge never ran — fixing the join itself would have
    changed nothing while this held.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    rows = [
        (snapshot, RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (snapshot, str(MARKET_QUOTES.symbol), "AAPL"),
    ]

    linker = CrossSourceLinker(spark, make_triples(rows))

    assert 'market' in linker.available_sources


def test_cross_source_does_not_unify_temporal_entities(spark, make_triples):
    """Temporal alignment belongs to TemporalUnifier, not here.

    This module used to run its own temporal alignment as step 1, matching
    subjects with ``(January|...|December)$`` -- anchored only at the END, with
    no separator required. That matched any URI merely ENDING with a month name
    and asserted a measurement IS a month:

        unified:September owl:sameAs cpi:...PercentChange_..._2025_September

    owl:sameAs is RDF's strongest claim (the two URIs denote one individual), so
    those became real graph edges and licensed a reasoner to merge them. On the
    e2e fixtures 263 subjects end with a month name and only 13 are actual month
    URIs -- the other 250 are measurements.

    It was also redundant: TemporalUnifier runs as enrichment PHASE 2, before
    this module, with its output merged into triples_df. Measured on the
    fixtures, this step produced 273 sameAs triples -- 259 false, 14 vacuous
    self-links (unified:April sameAs unified:April, from matching the node
    Phase 2 had just created) and 0 correct links TemporalUnifier had not
    already found.

    The previous test here asserted the buggy behaviour was correct, using a
    fixture (``cpi/month/2024-January``) chosen to match the regex. No such URI
    exists in real data, which contains only bare month URIs and measurements.
    """
    measurement = str(CPI) + "SeasonallyAdjustedPercentChange_Food_2025_September"
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL_1"
    rows = [
        (measurement, RDFS_LABEL, "Food, Sep 2025"),
        (snapshot, RDFS_LABEL, "AAPL snapshot"),  # second source, no month
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    assert not [
        t for t in triples if t[1] == OWL_SAME_AS and t[2] == measurement
    ], "a measurement was asserted to BE a month"
    assert not [
        t for t in triples if t[0] == str(UNIFIED) + "September"
    ], "cross-source linker must not mint unified month entities"


def test_cross_source_does_not_unify_year_suffixed_uris(spark, make_triples):
    """Same guard for the four-digit year path.

    The removed step matched years with ``(\\d{4})$`` -- also anchored only at
    the end -- so any URI ending in four digits was treated as a year entity and
    given an owl:sameAs from unified:Year{YYYY}. A measurement whose name ends
    in a year is not the year.
    """
    measurement = str(CPI) + "UnadjustedIndex_Apparel_2024"
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL_1"
    rows = [
        (measurement, RDFS_LABEL, "Apparel index 2024"),
        (snapshot, RDFS_LABEL, "AAPL snapshot"),  # second source
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    assert not [
        t for t in triples if t[1] == OWL_SAME_AS and t[2] == measurement
    ], "a measurement was asserted to BE a year"
    assert not [
        t for t in triples if t[0] == str(UNIFIED) + "Year2024"
    ], "cross-source linker must not mint unified year entities"


def test_cross_source_single_source_short_circuits(spark, make_triples):
    """Only one source family present -> no cross-source enrichment at all."""
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    rows = [
        (snapshot, RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (snapshot, str(MARKET_QUOTES.symbol), "AAPL"),
    ]

    result = CrossSourceLinker(spark, make_triples(rows)).enrich()

    assert result.count() == 0


def test_cross_source_unlinkable_sources_produce_no_entity_edges(spark, make_triples):
    """Two sources with no shared key (no tickers, months, sectors, regions)
    produce no edges touching the input entities and no linking predicates
    (unified-region scaffolding triples are allowed)."""
    market_e = str(MARKET_QUOTES) + "zzq-a"
    sec_e = str(SEC_FILINGS) + "zzq-b"
    rows = [
        (market_e, RDFS_LABEL, "zzq a"),
        (sec_e, RDFS_LABEL, "zzq b"),
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    inputs = {market_e, sec_e}
    assert all(s not in inputs and o not in inputs for s, _, o in triples)
    assert all(
        p not in (REFERS_TO_COMPANY, OWL_SAME_AS, LEADS_TO, AFFECTS_REGION)
        for _, p, _ in triples
    )


# ======================================================================
# Deep: NOAA geographic FIPS chain
# ======================================================================

@pytest.mark.parametrize("fips,region_key", [
    ("048", "Texas"),         # SAME-code head, the shape upstream emits
    ("040", "Oklahoma"),      # ditto, from the Tornado Warning fixture
    ("48", "Texas"),          # bare 2-digit, tolerated
    ("099", None),            # unmapped code -> no region link
])
def test_cross_source_geography_fips_chain(spark, make_triples, fips, region_key):
    """State-FIPS linking traverses Alert -> Info -> Area -> Geocode and maps
    the code through the state lookup. No area-description triples are present,
    so this also pins that the FIPS strategy works standalone.

    Upstream emits nws:hasStateFIPS as the 3-character head of a SAME code --
    value[:3], so "048" not "48" -- and this pinned only the 2-digit form, which
    is the shape that never arrives. Both are parametrised now: the 3-digit case
    is what production sees, the 2-digit case pins that normalising it did not
    break the plain form."""
    alert = str(ALERT) + "urn:oid:2.49.0.1.840.0.test"
    info = alert + "#info"
    area = alert + "#area"
    geocode = alert + "#geocode-" + fips
    rows = [
        # nws:WeatherAlert, not cap:Alert -- upstream types alert nodes with the
        # former and emits the latter nowhere, so a fixture built on cap:Alert
        # exercised a shape production never sees.
        (alert, RDF_TYPE, str(WEATHER.WeatherAlert)),
        (alert, str(CAP.hasInfo), info),
        (info, str(CAP.hasArea), area),
        (area, str(CAP.hasGeocode), geocode),
        (geocode, str(WEATHER.hasStateFIPS), fips),
    ]

    linker = CrossSourceLinker(spark, make_triples(rows))
    triples = _triple_set(linker._link_by_geography())

    region_links = {t for t in triples if t[1] == AFFECTS_REGION}
    if region_key is None:
        assert region_links == set()
    else:
        expected = str(UNIFIED) + region_key + "Region"
        assert region_links == {(alert, AFFECTS_REGION, expected)}
