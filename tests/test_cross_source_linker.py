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
from spark_jobs.enrichment.intra_source.market.patterns import (
    _gics_sector_to_pascal,
)
from spark_jobs.enrichment.sector_crosswalk import (
    EQUITY_SECTOR_TYPE, RELATED_TO_ECONOMIC_SECTOR,
)
from spark_jobs.utils.rdf_utils import (
    ALERT, BLS_ENRICHMENT, CAP, CPI, JOLTS, LAUS, MARKET_ENRICHMENT,
    MARKET_QUOTES, SEC_ENRICHMENT, WEATHER, SEC_FILINGS, UNIFIED,
    identifier_namespace,
)

# The regional series are INDIVIDUALS, so they live under id/, not ontology/.
# Keying a fixture on the term namespace builds entities no source detector
# recognises, and the whole geographic step then reports "no bls source".
JOLTS_ID = identifier_namespace(str(JOLTS))
LAUS_ID = identifier_namespace(str(LAUS))

RDF_TYPE = str(RDF.type)
OWL_SAME_AS = str(OWL.sameAs)
RDFS_LABEL = str(RDFS.label)

REFERS_TO_COMPANY = str(BLS_ENRICHMENT.refersToCompany)

# Spelled from SEC_ENRICHMENT, not BLS_ENRICHMENT: the unified company node is
# the one sec_linker mints, and the two modules used to disagree about its type
# as well as about its key.
UNIFIED_COMPANY_TYPE = str(SEC_ENRICHMENT.UnifiedCompany)
HAS_CIK_PRED = str(SEC_ENRICHMENT.hasCik)
SHARES_SUB_INDUSTRY = str(MARKET_ENRICHMENT.sharesSubIndustryWith)
BELONGS_TO_SECTOR = str(BLS_ENRICHMENT.belongsToSector)
UNIFIED_MONTH_TYPE = str(BLS_ENRICHMENT.UnifiedMonth)
AFFECTS_REGION = str(BLS_ENRICHMENT.affectsRegion)
HAS_REGION = str(BLS_ENRICHMENT.hasRegion)
WITHIN_CENSUS_REGION = str(BLS_ENRICHMENT.withinCensusRegion)
LEADS_TO = str(BLS_ENRICHMENT.leadsTo)
TICKER_PRED = str(BLS_ENRICHMENT.ticker)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# Tier 2 smokes
# ======================================================================

def _issuer_rows(issuer, cik, symbol=None):
    """An SEC issuer as upstream states it: a CIK always, a ticker sometimes."""
    rows = [
        (issuer, RDF_TYPE, str(SEC_FILINGS.Issuer)),
        (issuer, str(SEC_FILINGS.hasIssuerCik), cik),
    ]
    if symbol is not None:
        rows.append((issuer, str(SEC_FILINGS.hasIssuerTradingSymbol), symbol))
    return rows


def _snapshot_rows(snapshot, symbol, snapshot_type=None):
    return [
        (snapshot, RDF_TYPE, str(snapshot_type or MARKET_QUOTES.EquitySnapshot)),
        (snapshot, str(MARKET_QUOTES.symbol), symbol),
    ]


def test_cross_source_company_linking_shares_unified_company(spark, make_triples):
    """A market quote and an SEC issuer for one company land on ONE node.

    Both sides of this join were once dead. The market side asked for
    market-feeds:StockTicker, a class nothing declares -- the model is
    EquitySnapshot / OptionSnapshot -- so it matched nothing. The SEC
    side asked for hasIssuerTicker, which upstream renamed to
    hasIssuerTradingSymbol and moved off the filing onto the filings:Issuer
    node.

    Both are now keyed on the CIK instead, so this asserts the CIK-keyed node
    sec_linker also mints. The SEC subject is the ISSUER, not the filing: a
    filing is a document and reaches its company through hasIssuer.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0000320193"
    rows = (
        _snapshot_rows(snapshot, "AAPL")
        + _issuer_rows(issuer, "0000320193", "AAPL")
    )

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_0000320193"
    assert (snapshot, REFERS_TO_COMPANY, company) in triples
    assert (issuer, REFERS_TO_COMPANY, company) in triples
    assert (company, RDF_TYPE, UNIFIED_COMPANY_TYPE) in triples
    assert (company, HAS_CIK_PRED, "0000320193") in triples
    assert (company, TICKER_PRED, "AAPL") in triples


def test_a_ticker_and_its_filing_resolve_to_one_company_node(spark, make_triples):
    """The split this work exists to close: ONE company, not two nodes.

    The market side used to mint unified:Company_{SYMBOL} while sec_linker
    minted unified:Company_{CIK} -- same URI prefix, different key, never
    reconciled -- so the quote and the filing described the same company
    through two nodes an extra hop apart, each carrying half its
    neighbourhood.

    Asserting the absence is the point. A regression here re-adds a node
    rather than removing one, so a test that only checked the CIK-keyed node
    was present would pass while the split came back.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0000320193"
    rows = (
        _snapshot_rows(snapshot, "AAPL")
        + _issuer_rows(issuer, "0000320193", "AAPL")
    )

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    companies = {
        s for s, _p, _o in triples
        if s.startswith(str(UNIFIED) + "Company_")
    }
    assert companies == {str(UNIFIED) + "Company_0000320193"}, (
        f"expected exactly one company node keyed on the CIK, got {companies}"
    )

    assert str(UNIFIED) + "Company_AAPL" not in companies
    assert not any(
        o == str(UNIFIED) + "Company_AAPL" for _s, _p, o in triples
    ), "a symbol-keyed company node is still being referenced"


def test_the_company_bridge_reaches_a_filing_with_no_trading_symbol(
    spark, make_triples
):
    """The coverage win: hasIssuerCik is on every filing, the ticker is not.

    hasIssuerTradingSymbol has one upstream emitter -- the ownership-form
    document parser -- so it caps the bridge at the Form 3/4/5 share of
    filings, a measured 24.57%. An issuer stating only its CIK is the other
    75.43%, and it used to reach no company node at all.
    """
    issuer = str(SEC_FILINGS) + "Issuer_0000084112"
    snapshot = str(MARKET_QUOTES) + "snapshot/CBOE/2026-07-02"
    rows = (
        _snapshot_rows(snapshot, "CBOE")
        + _issuer_rows(issuer, "0000084112")
    )

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_0000084112"
    assert (issuer, REFERS_TO_COMPANY, company) in triples, (
        "an issuer stating no ticker did not reach its company node"
    )
    assert (company, RDF_TYPE, UNIFIED_COMPANY_TYPE) in triples


def test_the_constituents_map_widens_the_bridge_beyond_ingested_filings(
    spark, make_triples
):
    """The CSV path: a quote reaches a company whose filings are not here.

    The in-graph ticker -> CIK derivation only covers issuers whose ownership
    filings were ingested. The index-constituents CSV pairs the symbol with
    the CIK for every index member, which is the column market/patterns.py
    used to read past.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0000084112"
    rows = (
        _snapshot_rows(snapshot, "AAPL")
        + _issuer_rows(issuer, "0000084112")
    )

    triples = _triple_set(
        CrossSourceLinker(
            spark, make_triples(rows), ticker_cik_map={"AAPL": "0000320193"}
        ).enrich()
    )

    company = str(UNIFIED) + "Company_0000320193"
    assert (snapshot, REFERS_TO_COMPANY, company) in triples
    assert (company, TICKER_PRED, "AAPL") in triples


def test_the_filings_win_when_the_constituents_csv_disagrees(spark, make_triples):
    """One CIK per symbol, and the graph's own pairing outranks the file.

    A ticker is reassigned when a company delists and another takes the
    symbol, so the CSV can carry a pairing that was true at publication and is
    not true of the filing in front of us. Emitting both would rebuild the
    split on the market side -- one snapshot pointing at two companies.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0000320193"
    rows = (
        _snapshot_rows(snapshot, "AAPL")
        + _issuer_rows(issuer, "0000320193", "AAPL")
    )

    triples = _triple_set(
        CrossSourceLinker(
            spark, make_triples(rows), ticker_cik_map={"AAPL": "0000999999"}
        ).enrich()
    )

    linked = {o for s, p, o in triples if s == snapshot and p == REFERS_TO_COMPANY}
    assert linked == {str(UNIFIED) + "Company_0000320193"}, (
        f"the snapshot should reach exactly the filed CIK, got {linked}"
    )


def test_an_unpadded_constituent_cik_fails_loudly(spark, make_triples):
    """An unpadded key must raise, not yield an empty bridge.

    The CSV states the CIK unpadded ("320193") and the filings state it padded
    to ten ("0000320193"). Joined as strings those match nothing, emit no
    edge, and raise nothing -- so the failure is indistinguishable from a
    fixture with no overlap and survives review. The guard turns the silent
    version of this bug into a loud one.
    """
    rows = _issuer_rows(str(SEC_FILINGS) + "Issuer_0000320193", "0000320193")

    with pytest.raises(ValueError, match="zero-padded"):
        CrossSourceLinker(
            spark, make_triples(rows), ticker_cik_map={"AAPL": "320193"}
        )


def test_sub_industry_peers_link_and_a_shared_sector_alone_does_not(
    spark, make_triples
):
    """Sub-industry is the signal; sector is too coarse to be one.

    NVDA and AMD are both GICS "Semiconductors"; MSFT is "Application
    Software". All three sit in the same GICS SECTOR (Information Technology),
    which is the point -- the largest sector holds 73 of 503 constituents, so a
    peer edge built on sector would link all three and say nothing. The
    sub-industry buckets are 3-5 constituents wide, where the claim is sharp.
    """
    snapshots = {
        symbol: str(MARKET_QUOTES) + f"snapshot/{symbol}/2026-07-02"
        for symbol in ("NVDA", "AMD", "MSFT")
    }
    ciks = {"NVDA": "0001045810", "AMD": "0000002488", "MSFT": "0000789019"}

    rows = [(str(SEC_FILINGS) + "Issuer_0000320193", RDF_TYPE, str(SEC_FILINGS.Issuer))]
    for symbol, snapshot in snapshots.items():
        rows += _snapshot_rows(snapshot, symbol)
        rows += _issuer_rows(
            str(SEC_FILINGS) + f"Issuer_{ciks[symbol]}", ciks[symbol]
        )

    triples = _triple_set(
        CrossSourceLinker(
            spark,
            make_triples(rows),
            ticker_cik_map=ciks,
            sub_industries=[
                ("NVDA", "Semiconductors"),
                ("AMD", "Semiconductors"),
                ("MSFT", "Application Software"),
            ],
        ).enrich()
    )

    peers = {
        (s, o) for s, p, o in triples if p == SHARES_SUB_INDUSTRY
    }

    def company(symbol):
        return str(UNIFIED) + "Company_" + ciks[symbol]

    assert peers == {(company("AMD"), company("NVDA"))}, (
        "expected exactly one unordered peer pair, the two semiconductor "
        f"constituents, got {peers}"
    )

    assert not any(
        company("MSFT") in pair for pair in peers
    ), "a constituent sharing only the broader sector was linked as a peer"


def test_sub_industry_peers_are_not_minted_for_absent_companies(
    spark, make_triples
):
    """The CSV lists every index member; the graph carries a few of them.

    Minting a company node per constituent would bolt hundreds of vertices
    onto the graph whose only edges are to each other -- a disconnected clique
    lattice describing companies no source in this build mentions.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/NVDA/2026-07-02"
    issuer = str(SEC_FILINGS) + "Issuer_0001045810"
    rows = _snapshot_rows(snapshot, "NVDA") + _issuer_rows(issuer, "0001045810")

    triples = _triple_set(
        CrossSourceLinker(
            spark,
            make_triples(rows),
            ticker_cik_map={"NVDA": "0001045810", "AMD": "0000002488"},
            sub_industries=[
                ("NVDA", "Semiconductors"),
                ("AMD", "Semiconductors"),
            ],
        ).enrich()
    )

    assert not [t for t in triples if t[1] == SHARES_SUB_INDUSTRY], (
        "a peer edge was minted to a company the graph does not carry"
    )
    assert str(UNIFIED) + "Company_0000002488" not in {s for s, _p, _o in triples}


# ======================================================================
# Equity sector -> economic sector (the curated crosswalk)
# ======================================================================

def _equity_sector(gics):
    return str(MARKET_ENRICHMENT) + _gics_sector_to_pascal(gics) + "Sector"


def _sector_rows(*gics_names):
    """A market snapshot plus the equity sector nodes market_linker types."""
    snapshot = str(MARKET_QUOTES) + "snapshot/XOM/2026-07-02"
    rows = _snapshot_rows(snapshot, "XOM")
    rows += _issuer_rows(str(SEC_FILINGS) + "Issuer_0000034088", "0000034088")
    for gics in gics_names:
        rows.append((_equity_sector(gics), RDF_TYPE, EQUITY_SECTOR_TYPE))
    return rows


def test_a_mapped_equity_sector_reaches_the_economic_hub(spark, make_triples):
    """bls:EconomicSector is the best-connected hub and the market side had
    zero edges to it. This is the edge that changes that."""
    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(_sector_rows("Energy"))).enrich()
    )

    assert (
        _equity_sector("Energy"),
        RELATED_TO_ECONOMIC_SECTOR,
        str(BLS_ENRICHMENT.EnergySector),
    ) in triples

    # The destination must be a node, or node_mapper drops the edge.
    assert (
        str(BLS_ENRICHMENT.EnergySector), RDF_TYPE,
        str(BLS_ENRICHMENT.EconomicSector),
    ) in triples


def test_one_equity_sector_may_reach_several_economic_sectors(spark, make_triples):
    """Many-to-many by design: Consumer Staples covers three consumption
    categories and covers essentially all of each."""
    triples = _triple_set(
        CrossSourceLinker(
            spark, make_triples(_sector_rows("Consumer Staples"))
        ).enrich()
    )

    reached = {
        o for s, p, o in triples
        if p == RELATED_TO_ECONOMIC_SECTOR and s == _equity_sector("Consumer Staples")
    }
    assert reached == {
        str(BLS_ENRICHMENT.FoodSector),
        str(BLS_ENRICHMENT.TobaccoSector),
        str(BLS_ENRICHMENT.PersonalCareSector),
    }


@pytest.mark.parametrize(
    "gics", ["Information Technology", "Communication Services", "Utilities"]
)
def test_a_deliberately_unmapped_equity_sector_emits_no_edge(
    spark, make_triples, gics
):
    """The gaps are the decision, asserted end to end rather than on the table.

    A contributor could leave sector_crosswalk.py alone and re-add the mapping
    in the linker — a name match, a fallback, a "reasonable default" — and the
    table test would still pass. This one would not.
    """
    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(_sector_rows(gics))).enrich()
    )

    emitted = [
        t for t in triples
        if t[1] == RELATED_TO_ECONOMIC_SECTOR and t[0] == _equity_sector(gics)
    ]
    assert not emitted, (
        f"{gics} is deliberately unmapped and emitted {emitted}. The economic "
        "`Information` sector is publishing and telecom, not semiconductors; "
        "`Utilities` has no economic counterpart at all."
    )


def test_an_equity_sector_absent_from_the_build_mints_nothing(spark, make_triples):
    """The table covers eleven sectors; a build classifying into two should not
    carry nine sector nodes with one edge each and no constituents."""
    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(_sector_rows("Energy"))).enrich()
    )

    assert not [
        t for t in triples
        if t[1] == RELATED_TO_ECONOMIC_SECTOR
        and t[0] != _equity_sector("Energy")
    ]


# ======================================================================
# Filings -> economic sector, via SIC
# ======================================================================

def _filing_with_sic(accession, cik, sic):
    filing = str(SEC_FILINGS) + f"{accession}_Filing"
    issuer = str(SEC_FILINGS) + f"Issuer_{cik}"
    return filing, [
        (filing, RDF_TYPE, str(SEC_FILINGS.SECFiling)),
        (filing, str(SEC_FILINGS.hasSic), sic),
        (filing, str(SEC_FILINGS.hasIssuer), issuer),
    ] + _issuer_rows(issuer, cik)


def test_a_filings_sic_gives_its_company_a_sector(spark, make_triples):
    """hasSic is stated on every filing and this repo read it nowhere.

    The sector lands on the COMPANY, so every filing by that issuer inherits
    it through the node Part 1 unified — which is why this could not land
    before the company bridge was CIK-keyed.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/PFE/2026-07-02"
    _filing, rows = _filing_with_sic("0001178913-26-003947", "0000078003", "2836")
    rows += _snapshot_rows(snapshot, "PFE")

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_0000078003"
    assert (
        company, BELONGS_TO_SECTOR, str(BLS_ENRICHMENT.ManufacturingSector)
    ) in triples, "SIC 2836 (biological products) is SIC division D"


def test_a_filing_in_a_reserved_sic_gap_gets_no_sector(spark, make_triples):
    """SIC leaves 18-19, 68-69 and 90 unassigned. Snapping to the nearest
    division would emit a confident, wrong membership claim."""
    snapshot = str(MARKET_QUOTES) + "snapshot/PFE/2026-07-02"
    _filing, rows = _filing_with_sic("0001178913-26-003947", "0000078003", "1850")
    rows += _snapshot_rows(snapshot, "PFE")

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_0000078003"
    assert not [
        t for t in triples if t[0] == company and t[1] == BELONGS_TO_SECTOR
    ]


def test_the_keyword_classifier_no_longer_claims_sec_entities(spark, make_triples):
    """A filing's local name is an accession number, not a description.

    The keyword classifier matched BLS sector fragments against it, so any
    sector keyword appearing inside one produced a meaningless membership
    claim on the same predicate the real classification uses.
    """
    snapshot = str(MARKET_QUOTES) + "snapshot/PFE/2026-07-02"
    # A local name deliberately containing a live BLS sector keyword.
    filing = str(SEC_FILINGS) + "0001178913-26-energy_Filing"
    rows = _snapshot_rows(snapshot, "PFE") + [
        (filing, RDF_TYPE, str(SEC_FILINGS.SECFiling)),
    ] + _issuer_rows(str(SEC_FILINGS) + "Issuer_0000078003", "0000078003")

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    assert not [
        t for t in triples
        if t[0].startswith(str(SEC_FILINGS)) and t[1] == BELONGS_TO_SECTOR
    ], "the keyword classifier still assigns sectors to SEC entities"


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
    rows = (
        _snapshot_rows(
            option, "A     260717C00065000", MARKET_QUOTES.OptionSnapshot
        )
        + _issuer_rows(issuer, "0001090872", "A")
    )

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    company = str(UNIFIED) + "Company_0001090872"
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
    ("048", "Texas"),         # SAME-code head, the shape upstream emits TODAY
    ("040", "Oklahoma"),      # ditto, from the Tornado Warning fixture
    ("48", "Texas"),          # bare 2-digit, the shape upstream is MOVING TO
    ("20", "Kansas"),         # ditto
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
    break the plain form.

    BOTH FORMS MUST KEEP PASSING. Upstream is correcting hasStateFIPS to drop
    the leading pad digit, so the 2-digit rows are not hypothetical -- they are
    the shape after the fix. Reading the last two digits is correct either way,
    which is what makes that fix non-breaking and needs no backfill, and it is
    why this reader stays tolerant permanently rather than being tightened once
    the fix lands. Narrowing it to one form breaks the join in one direction or
    the other."""
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


# ======================================================================
# Deep: region unification (the weather -> economic path)
# ======================================================================

def _alert_rows(fips):
    alert = str(ALERT) + "urn:oid:2.49.0.1.840.0.test"
    info, area = alert + "#info", alert + "#area"
    geocode = alert + "#geocode-" + fips
    return alert, [
        (alert, RDF_TYPE, str(WEATHER.WeatherAlert)),
        (alert, str(CAP.hasInfo), info),
        (info, str(CAP.hasArea), area),
        (area, str(CAP.hasGeocode), geocode),
        (geocode, str(WEATHER.hasStateFIPS), fips),
    ]


def _jolts_region_rows(name):
    """A regional job-openings series, as the live fixture shapes it."""
    region = str(JOLTS_ID) + f"{name}_Region"
    series = str(JOLTS_ID) + f"{name}_June2026_HiresLevel"
    return region, [
        (region, RDF_TYPE, str(JOLTS.Region)),
        (region, RDFS_LABEL, f"{name} region"),
        (series, RDF_TYPE, str(JOLTS.HiresLevel)),
        (series, str(JOLTS.hasRegion), region),
    ]


def test_a_weather_alert_and_a_regional_series_share_a_region_node(
    spark, make_triples
):
    """The path Part 4 exists to build.

    Two region vocabularies existed -- bls_enrichment_GeographicRegion, where
    the weather feed attaches, and jolts_Region, where the regional economic
    series attach -- and NOTHING joined them, so
    `WeatherAlert -> Region -> regional indicator` did not exist even though
    both halves were built.

    The join is geographic and only geographic. A sector is not located in a
    region, so weather reaches economic activity through place and reaches an
    industry only transitively.
    """
    alert, rows = _alert_rows("020")            # Kansas
    region, jolts_rows = _jolts_region_rows("Midwest")
    rows = rows + jolts_rows

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    kansas = str(UNIFIED) + "KansasRegion"
    midwest = str(UNIFIED) + "MidwestCensusRegion"

    assert (alert, AFFECTS_REGION, kansas) in triples
    assert (kansas, WITHIN_CENSUS_REGION, midwest) in triples, (
        "Kansas did not reach its census region — the 51-row table is the only "
        "thing that can carry a state to the job-openings grain"
    )
    assert (midwest, OWL_SAME_AS, region) in triples, (
        "the census region and the jolts region are still two nodes"
    )
    assert (midwest, RDF_TYPE, str(BLS_ENRICHMENT.CensusRegion)) in triples


def test_a_state_reaches_only_its_own_census_region(spark, make_triples):
    """Kansas is Midwest. A table row that also put it in the South would give
    the weather feed a path to series it has nothing to do with."""
    alert, rows = _alert_rows("020")
    for name in ("Midwest", "South", "Northeast", "West"):
        rows += _jolts_region_rows(name)[1]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    kansas = str(UNIFIED) + "KansasRegion"
    reached = {o for s, p, o in triples if s == kansas and p == WITHIN_CENSUS_REGION}
    assert reached == {str(UNIFIED) + "MidwestCensusRegion"}


def test_region_unification_does_not_merge_a_shared_label_prefix(
    spark, make_triples
):
    """West Virginia is not Virginia.

    The LAUS matcher asked `subject.contains(state_key)`, so every West
    Virginia series linked to VirginiaRegion as well as to its own -- and
    Virginia is the ONLY state name that sits inside another, so this pair is
    the whole population rather than a sample of it. Merging two genuinely
    distinct regions is worse than missing one: the weather feed then reaches
    the wrong state's unemployment series with full confidence.

    WHAT THIS DOES NOT FIX, stated so nobody reads more into the anchor than is
    there: a metro area whose NAME BEGINS with a state name still claims that
    state. "Kansas_City_MO" reads as "_Kansas_City_..." and starts with
    "_Kansas_", so it lands on KansasRegion, though the metro straddles Kansas
    and Missouri. Separating that needs the federal delineation file that puts
    metro resolution out of scope for this work, so the anchor is deliberately
    the state-vs-state fix and not the metro one.
    """
    west_virginia = str(LAUS_ID) + "West_Virginia_June2026_UnemploymentRate"
    virginia = str(LAUS_ID) + "Virginia_June2026_UnemploymentRate"
    alert, rows = _alert_rows("020")
    rows += [
        (west_virginia, RDF_TYPE, str(LAUS.UnemploymentRate)),
        (virginia, RDF_TYPE, str(LAUS.UnemploymentRate)),
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    def regions_of(entity):
        return {o for s, p, o in triples if s == entity and p == HAS_REGION}

    assert regions_of(west_virginia) == {str(UNIFIED) + "WestVirginiaRegion"}, (
        "the West Virginia series reached a region that is not West Virginia"
    )
    assert regions_of(virginia) == {str(UNIFIED) + "VirginiaRegion"}, (
        "the Virginia series lost its own region to the anchoring rule"
    )


def test_the_regional_state_fips_reader_is_written_against_the_agreed_shape(
    spark, make_triples
):
    """laus:hasStateFIPS is NOT emitted yet — this drives it synthetically.

    The regional series currently state a name and a slug and no code at all.
    The reader is written and tested ahead of the term because the failure it
    guards is silent: a numeric 1 never matches a string "01", and the join
    comes back empty with no error, looking exactly like data that does not
    overlap.
    """
    series = str(LAUS_ID) + "SomeArea_June2026_UnemploymentRate"
    alert, rows = _alert_rows("020")
    rows += [
        (series, RDF_TYPE, str(LAUS.UnemploymentRate)),
        (series, str(LAUS.hasStateFIPS), "20"),
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    kansas = str(UNIFIED) + "KansasRegion"
    assert (series, HAS_REGION, kansas) in triples
    assert (alert, AFFECTS_REGION, kansas) in triples, (
        "both sides must land on ONE region node, or the path does not exist"
    )


def test_a_census_region_code_keys_the_bridge_when_upstream_states_one(
    spark, make_triples
):
    """jolts:hasCensusRegionCode is the term upstream is adding.

    Zero-padded string, per the agreed shape. The name path already works, so
    the code merely confirms the bridge rather than enabling it — but a code
    that failed to key it would be a silent regression the day it lands.
    """
    region = str(JOLTS_ID) + "Region_2"
    series = str(JOLTS_ID) + "Region_2_June2026_HiresLevel"
    alert, rows = _alert_rows("020")
    rows += [
        (region, RDF_TYPE, str(JOLTS.Region)),
        (region, str(JOLTS.hasCensusRegionCode), "02"),
        (series, str(JOLTS.hasRegion), region),
    ]

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    midwest = str(UNIFIED) + "MidwestCensusRegion"
    assert (midwest, OWL_SAME_AS, region) in triples


def test_no_census_region_is_minted_when_nothing_reaches_one(spark, make_triples):
    """Four typed nodes with no incident edge is four rows of feature tensor
    describing nothing."""
    market_e = str(MARKET_QUOTES) + "snapshot/AAPL/2026-07-02"
    rows = _snapshot_rows(market_e, "AAPL") + _issuer_rows(
        str(SEC_FILINGS) + "Issuer_0000320193", "0000320193"
    )

    triples = _triple_set(
        CrossSourceLinker(spark, make_triples(rows)).enrich()
    )

    assert not [t for t in triples if t[1] == WITHIN_CENSUS_REGION]
    assert not [t for t in triples if t[2] == str(BLS_ENRICHMENT.CensusRegion)]
