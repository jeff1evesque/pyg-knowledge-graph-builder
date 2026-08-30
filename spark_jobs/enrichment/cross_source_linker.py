"""
Cross-Source Enrichment Linker — PySpark Implementation

Links entities across different data source families (BLS, SEC, Market, NOAA).
Runs entirely on Spark executors — no rdflib, no driver data operations.

All methods:
1. Read from self.triples_df (the enriched triples from all intra-source enrichers)
2. Return new triples DataFrames
3. Never call .collect(), .toPandas(), or .toLocalIterator()
"""
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from rdflib.namespace import RDF, RDFS, OWL

from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, UNIFIED, CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER,
    WKYENG, SEC_ENRICHMENT, SEC_FILINGS, CAP, WEATHER,
    MARKET_ENRICHMENT, MARKET_QUOTES,
    ALERT,
    identifier_namespace,
)
from spark_jobs.enrichment.settle import settle
from spark_jobs.enrichment.intra_source.bls.patterns import (
    BLS_SECTOR_PATTERNS,
)
from spark_jobs.enrichment.intra_source.market.patterns import (
    assert_ciks_are_padded, _gics_sector_to_pascal,
)
from spark_jobs.enrichment.intra_source.market.symbols import equity_symbol
from spark_jobs.enrichment.region_crosswalk import (
    CENSUS_REGION_CODES,
    STATE_ABBREVIATIONS,
    STATE_FIPS_TO_NAME,
    STATE_TO_CENSUS_REGION,
    US_STATES,
    normalized_state_fips,
    state_key,
)
from spark_jobs.enrichment.sector_crosswalk import (
    EQUITY_SECTOR_TYPE as _EQUITY_SECTOR_TYPE,
    EQUITY_TO_ECONOMIC_SECTORS,
    RELATED_TO_ECONOMIC_SECTOR,
    RELATION_CONFIDENCE,
    SIC_DIVISIONS,
)
from spark_jobs.utils.spark_rdf_utils import (
    extract_entities_by_type, extract_property, deduplicate_against_existing,
)
from typing import Dict, List, Optional, Sequence, Set, Tuple
import logging

logger = logging.getLogger(__name__)


def _delimited_local_name(subject: Column) -> Column:
    """The URI's local name with every run of non-letters as one "_", wrapped.

    "…/id/laus/Arkansas_June2026_HiresLevel" -> "_Arkansas_June_HiresLevel_"

    Wrapping in the separator is what turns a substring test into a WORD test:
    "_Arkansas_June_" does not contain "_Kansas_", while the raw URI does
    contain "Kansas". Collapsing digit runs too, so a year between two names
    still reads as one boundary rather than gluing them together.
    """
    local = F.regexp_extract(subject, r"[/#]([^/#]+)$", 1)
    return F.concat(
        F.lit("_"), F.regexp_replace(local, r"[^A-Za-z]+", "_"), F.lit("_")
    )


def _entity_prefixes(*namespaces) -> List[str]:
    """Both halves of the term/individual split, for matching entity URIs.

    Every prefix test in this module asks "is this ENTITY from source X" —
    against `subject`, or against an `entity` column resolved from one. Those
    are individuals and live under id/, so the term namespace alone matches
    nothing. Both are returned because these are candidate-selection gates
    rather than assertions: a graph carrying term-level triples (rdf:type
    objects, ontology axioms) should not fall out of source detection, and
    every downstream step re-filters on specific predicates anyway.
    """
    out: List[str] = []
    for ns in namespaces:
        term = str(ns)
        out.append(term)
        out.append(identifier_namespace(term))
    return out


_BLS_ENTITY_PREFIXES = _entity_prefixes(
    CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER, WKYENG
)
_SEC_ENTITY_PREFIXES = _entity_prefixes(SEC_FILINGS)

# What _detect_sources tests subjects against, which decides whether 'market' is
# in available_sources at all -- and the company/ticker step below is gated on
# that. This named only the feeds vocabulary, so a graph carrying quote
# snapshots (id/market-quotes/snapshot/...) reported NO market source and the
# SEC-to-market bridge was skipped without running.
_MARKET_ENTITY_PREFIXES = _entity_prefixes(MARKET_QUOTES)

# The symbol-bearing classes, as (type, predicate). Equity and option snapshots
# arrive in the same feed and state their ticker the same way.
_MARKET_SYMBOL_TYPES = (
    (str(MARKET_QUOTES.EquitySnapshot), str(MARKET_QUOTES.symbol)),
    (str(MARKET_QUOTES.OptionSnapshot), str(MARKET_QUOTES.symbol)),
)

# The state name, abbreviation, FIPS and census-region tables now live in
# enrichment/region_crosswalk.py, imported above. They moved because three
# different paths key on them -- the weather geocodes, the local-area economic
# series, and the census-region bridge -- and a table copied into each is a
# table that drifts between them.

# URI constants
_RDF_TYPE = str(RDF.type)
_OWL_SAME_AS = str(OWL.sameAs)
_RDFS_LABEL = str(RDFS.label)

# The unified company node, as sec_linker._unify_company_entities already mints
# it: keyed on the padded CIK, typed sec:UnifiedCompany, stating sec:hasCik.
#
# BOTH halves are spelled from that module's constants rather than from this
# one's, and that is the entire fix for Part 1 of the split-company defect. This
# module used to mint unified:Company_{SYMBOL} typed bls:UnifiedCompany, which
# collided with NOTHING sec_linker produced -- same URI prefix, different key,
# different type -- so one company became two nodes carrying half its
# neighbourhood each, and the graph stayed connected only because the issuer
# happened to point at both. Keying and typing identically is what makes them
# one node.
_UNIFIED_COMPANY_PREFIX = f"{str(UNIFIED)}Company_"
_UNIFIED_COMPANY_TYPE = str(SEC_ENRICHMENT.UnifiedCompany)
_HAS_CIK = str(SEC_ENRICHMENT.hasCik)

_FILINGS_HAS_ISSUER_CIK = str(SEC_FILINGS.hasIssuerCik)
_FILINGS_HAS_ISSUER_TRADING_SYMBOL = str(SEC_FILINGS.hasIssuerTradingSymbol)
_FILINGS_HAS_ISSUER = str(SEC_FILINGS.hasIssuer)
_FILINGS_HAS_SIC = str(SEC_FILINGS.hasSic)

# Peer relation between two companies in one GICS sub-industry. In the MARKET
# enrichment namespace because the sub-industry is GICS vocabulary, and the
# subject and object are both companies the market feed classifies.
_SHARES_SUB_INDUSTRY = str(MARKET_ENRICHMENT.sharesSubIndustryWith)

# NOAA type URIs — aligned with updated RML mapper
_NWS_WEATHER_ALERT_TYPE = str(WEATHER.WeatherAlert)

# NOAA area description — on Area subject (alert:{id}#area)
_CAP_HAS_AREA_DESC = str(CAP.hasAreaDescription)

# NOAA geocode properties for geographic linking.
#
# nws:hasFIPSCode is deliberately NOT bound here. It was, and was never
# referenced -- the FIPS strategy below reads hasStateFIPS only. Worse than
# unused: the name reads as "the county FIPS is available", when it is not.
# Verified against the mapper -- hasFIPSCode and hasSAMECode are both mapped
# from the same fips_code column and render byte-identical ("024031" for a
# geocode whose county FIPS is "24031"). Anyone reaching for the dead constant
# would have joined a 6-character SAME code against a 5-character county-FIPS
# lookup. Nothing on any branch changes this, so a county-level join here would
# have to derive the county FIPS itself rather than wait for a corrected term.
_NWS_HAS_STATE_FIPS = str(WEATHER.hasStateFIPS)

# The economic side of the region join is keyed on NAMES, not codes.
#
# An earlier version of this file also read laus:hasStateFIPS,
# metro:hasStateFIPS and jolts:hasCensusRegionCode, on the strength of an issue
# saying those terms were coming. They were checked against the mapper and none
# of the three exists, is emitted, or is planned on any branch -- the same issue
# was wrong three times about filings:hasSic, so its claims are not a source.
# The readers were deleted rather than left matching zero rows and looking like
# working code.
#
# Nothing is lost by that. Every one of those surveys already exposes its
# geography as a NAMED node -- laus:hasState -> id/laus/Alabama,
# jolts:hasRegion -> id/jolts/Midwest_Region -- and the name is what this module
# joins on. The full path weather alert -> state region -> census region ->
# jolts region is live on the fixtures today.
#
# Two further notes for anyone tempted to add a code reader later:
#
#   * The JOLTS region identifier upstream is NOT numeric. Its catalog carries
#     "state_code": "MW" for "Midwest region", so a reader expecting census
#     regions 1-4 would match nothing even if the term shipped.
#   * The metro catalog holds no FIPS column at all, so that term would need
#     the codes sourced first, not merely mapped.
_JOLTS_REGION_TYPE = str(JOLTS.Region)

# state region -> census region. In BLS_ENRICHMENT because this pipeline infers
# it from a table it maintains; see region_crosswalk.
_WITHIN_CENSUS_REGION = str(BLS_ENRICHMENT.withinCensusRegion)

# NOAA structural properties for traversal
_CAP_HAS_INFO = str(CAP.hasInfo)
_CAP_HAS_AREA = str(CAP.hasArea)
_CAP_HAS_GEOCODE = str(CAP.hasGeocode)


class CrossSourceLinker:
    """
    Links entities across data source families using PySpark.

    Strategies:
    1. Temporal Alignment — Unify temporal entities across all sources
    2. Sector-Based Linking — Link entities in same economic sectors
    3. Company/Ticker Linking — Link entities referencing same companies
    4. Geographic Linking — Link entities by geographic region
    5. Causal Relationships — Discover potential causal links
    6. Measurement Type Alignment — Link similar measurement types
    """

    def __init__(
        self,
        spark: SparkSession,
        triples_df: DataFrame,
        ticker_cik_map: Optional[Dict[str, str]] = None,
        sub_industries: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        self.spark = spark
        self.triples_df = triples_df

        # Index-constituent reference data, read from the constituents CSV on
        # the driver before the job starts. Both are small enough to broadcast
        # (~500 rows) and both are OPTIONAL: with neither, the company bridge
        # still resolves through the CIKs the filings state, and the peer step
        # simply produces nothing.
        self._ticker_cik_map = dict(ticker_cik_map or {})
        self._sub_industries = list(sub_industries or [])

        # Validate before the first join rather than after the last one. An
        # unpadded CIK here joins against no filing and emits no edge, which is
        # indistinguishable from a fixture with no overlap -- so it has to stop
        # the build instead of being discovered in a graph_schema.json diff.
        assert_ciks_are_padded(self._ticker_cik_map)

        self.available_sources = self._detect_sources()
        logger.info(f"Detected data sources: {', '.join(self.available_sources)}")

    def _detect_sources(self) -> Set[str]:
        """Which sources are present. Reads every row, not a sample.

        This used to read the first 100,000 subjects and decide from those.
        Whether a source turned up in that window depended on how the rows
        happened to be laid out, not on whether the source was there. The same
        shape in bls_linker skipped the whole BLS leg on every four-source run
        (#350); here it would skip the steps gated on a source instead.

        Naming the source in one column and taking the distinct values reads
        the data once and collects at most one row per source.
        """
        by_source = (
            ('bls', _BLS_ENTITY_PREFIXES),
            ('sec', _SEC_ENTITY_PREFIXES),
            ('market', _MARKET_ENTITY_PREFIXES),
            # NOAA alert instances use the ALERT namespace
            # (https://api.weather.gov/alerts/) — a real publisher namespace, so
            # it has no id/ counterpart and is matched as-is. CAP/WEATHER are
            # checked too, for type triples.
            ('noaa', [str(ALERT), *_entity_prefixes(CAP, WEATHER)]),
        )

        def _starts_with_any(prefixes: List[str]) -> Column:
            test = F.col("subject").startswith(prefixes[0])
            for p in prefixes[1:]:
                test = test | F.col("subject").startswith(p)
            return test

        # First match wins, so the branch order is the old if/elif order.
        source = F.when(_starts_with_any(by_source[0][1]), by_source[0][0])
        for name, prefixes in by_source[1:]:
            source = source.when(_starts_with_any(prefixes), name)

        rows = (
            self.triples_df
            .select(source.alias("source"))
            .distinct()
            .collect()
        )
        return {row.source for row in rows if row.source is not None}

    def enrich(self) -> DataFrame:
        """Run all cross-source enrichment. Returns new triples only."""
        if len(self.available_sources) < 2:
            logger.info("Less than 2 data sources detected, skipping cross-source enrichment")
            return self.spark.createDataFrame([], schema=self.triples_df.schema)

        logger.info("=" * 60)
        logger.info("Starting Cross-Source Enrichment (PySpark)")
        logger.info("=" * 60)

        new_dfs: List[DataFrame] = []

        # NOTE: temporal alignment used to be step 1 here and has been removed.
        # TemporalUnifier (enrichment PHASE 2) already does it correctly, and
        # runs BEFORE this one with its output merged into triples_df -- so this
        # step duplicated the job on a frame that already contained the unified
        # nodes. Its matcher was also anchored only at the end
        # (``(January|...)$`` against the whole subject), so it matched any URI
        # merely ENDING with a month name and asserted
        # ``unified:September owl:sameAs cpi:...PercentChange_..._September`` --
        # i.e. that a measurement IS the month. owl:sameAs is RDF's strongest
        # claim, so those became real graph edges and licensed a reasoner to
        # merge the two. On the e2e fixtures it produced 273 sameAs triples: 259
        # false, 14 vacuous self-links (unified:April sameAs unified:April, from
        # matching the node Phase 2 had just created), and 0 that TemporalUnifier
        # did not already find correctly. Deleted rather than repaired: fixing
        # the regex would still leave two implementations minting the same
        # unified:{Month} URIs.
        logger.info("\n[Step 1/8] Creating sector-based links...")
        self._append(new_dfs, self._link_by_sector(), "[Step 1/8]")

        logger.info("\n[Step 2/8] Relating equity sectors to economic sectors...")
        self._append(
            new_dfs, self._link_equity_to_economic_sectors(), "[Step 2/8]"
        )

        logger.info("\n[Step 3/8] Linking by company/ticker...")
        if 'sec' in self.available_sources and 'market' in self.available_sources:
            self._append(new_dfs, self._link_by_company(), "[Step 3/8]")

            logger.info("\n[Step 4/8] Linking constituents by sub-industry...")
            self._append(new_dfs, self._link_by_sub_industry(), "[Step 4/8]")

            logger.info("\n[Step 5/8] Linking filings to sectors by SIC code...")
            self._append(new_dfs, self._link_filings_by_sic(), "[Step 5/8]")
        else:
            # Steps 4 and 5 print their headers inside the branch, so without
            # this the log jumps from step 3 to step 6 and three link families
            # go missing with nothing said.
            logger.warning(
                "  [Steps 3-5/8] skipped: both sec and market are needed, "
                f"detected {', '.join(sorted(self.available_sources))}"
            )

        logger.info("\n[Step 6/8] Linking by geographic region...")
        self._append(new_dfs, self._link_by_geography(), "[Step 6/8]")

        logger.info("\n[Step 7/8] Creating causal relationships...")
        self._append(new_dfs, self._create_causal_links(), "[Step 7/8]")

        logger.info("\n[Step 8/8] Aligning measurement types...")
        self._append(new_dfs, self._align_measurement_types(), "[Step 8/8]")

        if not new_dfs:
            return self.spark.createDataFrame([], schema=self.triples_df.schema)

        all_new = new_dfs[0]
        for df in new_dfs[1:]:
            all_new = all_new.unionByName(df)

        # Truncate the plan before the dedup pass (see spark_jobs.enrichment.settle).
        # Each step above is a join over triples_df, so this union carries five of
        # those subtrees; feeding it straight into dropDuplicates + the anti-join
        # against triples_df makes Catalyst's constraint inference blow the driver
        # heap while *planning* (OutOfMemoryError inside .cache(), before any task
        # runs). Materializing here replaces the six subtrees with one scan.
        all_new = settle(all_new)

        all_new = all_new.dropDuplicates(["subject", "predicate", "object"])
        all_new = deduplicate_against_existing(all_new, self.triples_df)

        all_new = all_new.cache()
        total = all_new.count()

        logger.info(f"\nCross-source enrichment produced {total} new triples")
        return all_new

    @staticmethod
    def _append(lst: list, item: Optional[DataFrame], step: str):
        """Keep a step's triples, and say so when it made none.

        A step that returns None used to log only its opening line. The graph
        then came out missing a whole link family and read exactly like a
        healthy one -- that is how the causal step went unnoticed (#350).
        """
        if item is not None:
            lst.append(item)
            return
        logger.warning(f"  {step} produced no triples")

    # ================================================================
    # Step 2: Sector Linking
    # ================================================================

    def _link_by_sector(self) -> Optional[DataFrame]:
        """
        Link entities across sources to unified sectors.

        Builds a keyword lookup from BLS_SECTOR_PATTERNS, broadcasts it,
        and joins against entity URIs to find sector matches.
        """
        # Build sector keyword lookup: (keyword_normalized, sector_uri)
        sector_rows = []
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            sector_uri = str(pattern['sector_uri'])
            for dataset_name, keywords in pattern['keywords'].items():
                for keyword in keywords:
                    normalized = keyword.replace(' ', '_').replace("'", '').replace(',', '').replace('-', '_')
                    sector_rows.append((normalized.lower(), sector_uri))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows,
            schema=["keyword", "sector_uri"]
        ).dropDuplicates(["keyword", "sector_uri"])

        # Extract the local name from each subject URI for keyword matching
        entities = (
            self.triples_df
            .select("subject")
            .distinct()
            .withColumn(
                "local_name",
                F.lower(F.regexp_extract("subject", r"[/#]([^/#]+)$", 1))
            )
            .filter(F.length(F.col("local_name")) > 0)
        )

        # SEC entities are excluded from the keyword classifier.
        #
        # They now reach a sector through filings:hasSic -- the issuer's
        # registered industry classification, stated by upstream on every
        # filing -- which is a fact about the company rather than a substring
        # of its URI. See _link_filings_by_sic.
        #
        # What this classifier was doing to them was matching BLS keyword
        # fragments against an accession number: a filing's local name is
        # "0001178913-26-003947_Filing", and any sector keyword that happens to
        # appear inside one produces a confident, meaningless membership claim.
        # Keeping both paths would leave the real classification competing with
        # the accidental one on the same predicate.
        sec_match = F.lit(False)
        for prefix in _SEC_ENTITY_PREFIXES:
            sec_match = sec_match | F.col("subject").startswith(prefix)
        entities = entities.filter(~sec_match)

        # Join: entity local name contains sector keyword
        # Use broadcast for the small sector lookup
        matched = entities.join(
            F.broadcast(sector_df),
            entities.local_name.contains(sector_df.keyword),
            how="inner"
        )

        # Sector type triples
        sector_type_triples = (
            matched.select("sector_uri").distinct()
            .select(
                F.col("sector_uri").alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(str(BLS_ENRICHMENT.EconomicSector)).alias("object")
            )
        )

        # belongsToSector triples
        belongs_triples = matched.select(
            F.col("subject").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.belongsToSector)).alias("predicate"),
            F.col("sector_uri").alias("object")
        )

        # NO CORRELATION TWIN. This used to also emit
        # bls:hasSectorCorrelation over the identical (subject, object) pairs,
        # built from the same `matched` frame in the same breath -- so the two
        # relations could not disagree, and the second carried no information
        # the first did not. On a fixture build that was 1,718 edges, 16.8% of
        # the graph, and for a GNN it is two parallel edge types over identical
        # node pairs: double the message passing, nothing learned.
        #
        # belongsToSector is the one that survives because it is the one
        # anything reads -- _create_causal_links joins on it, and
        # edge_feature_extractor lists it among the relations whose type alone
        # carries the signal. hasSectorCorrelation was read by nothing.
        #
        # The name promised something real -- a correlation DERIVED between a
        # series and a sector -- and no path ever computed one. If that is
        # wanted later it is new work, and it should not reuse a predicate that
        # spent this long meaning "copy of the line above".
        result = sector_type_triples.unionByName(belongs_triples)

        logger.info("  Sector linking triples prepared (lazy)")
        return result

    # ================================================================
    # Step 3: Company/Ticker Linking
    # ================================================================

    def _market_symbol_bearers(self) -> DataFrame:
        """(entity, symbol) for every market entity that names a ticker.

        Keyed on the classes each market vocabulary actually declares, per
        _MARKET_SYMBOL_TYPES. The previous version asked for
        market-feeds:StockTicker, which belongs to NEITHER model -- feeds is
        PriceObservation / OptionContract, quotes is EquitySnapshot /
        OptionSnapshot -- so it matched nothing no matter which namespace it
        was pointed at, and the SEC-to-market bridge had no market side.
        """
        frames = [
            extract_entities_by_type(self.triples_df, type_uri).join(
                extract_property(self.triples_df, symbol_pred, "raw_symbol"),
                on=F.col("entity") == F.col("subject"),
            ).select("entity", "raw_symbol")
            for type_uri, symbol_pred in _MARKET_SYMBOL_TYPES
        ]

        union = frames[0]
        for frame in frames[1:]:
            union = union.unionByName(frame)

        return union.select(
            F.col("entity"),
            equity_symbol(F.col("raw_symbol")).alias("symbol"),
        ).filter(F.length(F.col("symbol")) > 0).dropDuplicates()

    def _issuer_ciks(self) -> DataFrame:
        """(issuer_uri, cik) for every SEC issuer that states a CIK.

        This is the SEC half of the company bridge, and it replaces a half
        keyed on ``hasIssuerTradingSymbol``. The ticker was never the issuer's
        identifier here -- it is emitted by exactly one upstream parser, the
        ownership-form document walker, so it requires a filing that (a) had
        its document fetched and stored, (b) stored it as XML, and (c) is a
        Form 3/4/5, the only forms whose schema carries <issuerTradingSymbol>.
        The EDGAR search API never returns a ticker at all.

        MEASURED, from a census of 23,983 filing rows across 2026-08-07,
        2026-08-06, 2026-08-04 and 2025-10-02 (bin/census_sec_terms.py):

            rows stating hasIssuerTradingSymbol   5,893   24.57% of all filings
              of which root_form 4                5,229   100% of the 5,229
              of which root_form 3                  655   100% of the 655
              of which root_form 5                    9   100% of the 9
            rows on any other form                    0

        So the OLD bridge's ceiling was exactly the ownership-form share --
        about a quarter of filings -- and no amount of market-side work could
        lift it. hasIssuerCik has no such ceiling: the issuer states it on
        every filing, from both the metadata path and the document path, which
        is what moves this from a quarter to all of them.

        The CIK arrives already zero-padded to ten, because
        utils/canonicalization.canonicalize_sec_identifiers runs in the LOADER,
        before any enricher sees the frame. Nothing is re-padded here on
        purpose: if that ever stops being true the join under-covers visibly
        rather than being silently patched in two places that can disagree.
        """
        return extract_property(
            self.triples_df, _FILINGS_HAS_ISSUER_CIK, "raw_cik"
        ).select(
            F.col("subject").alias("issuer_uri"),
            F.trim(F.col("raw_cik")).alias("cik"),
        ).filter(F.length(F.col("cik")) > 0).dropDuplicates()

    def _ticker_to_cik(self) -> Optional[DataFrame]:
        """(symbol, cik) — how the market side reaches the regulator's key.

        Two sources, in priority order:

        0. THE FILINGS THEMSELVES. An ownership-form issuer states both its
           ticker and its CIK, so the graph already carries the mapping for
           every issuer whose filings were ingested. It needs no external file
           and it cannot disagree with the SEC half of the bridge, because it
           IS the SEC half read a second way.
        1. THE INDEX CONSTITUENTS CSV, which pairs the symbol with the CIK for
           index members whose filings may not have been ingested at all. This
           is the widening path, and the reason market/patterns.py now reads a
           column it used to discard.

        The priority matters when the two disagree: a ticker is reassigned when
        a company delists and another takes the symbol, so the CSV can carry a
        pairing that was true at publication and is not true of the filing in
        front of us. Ranking keeps ONE CIK per symbol, so a snapshot links to
        one company rather than to two -- which would rebuild, on the market
        side, exactly the split this work removes.
        """
        graph_pairs = extract_property(
            self.triples_df, _FILINGS_HAS_ISSUER_TRADING_SYMBOL, "raw_symbol"
        ).select(
            F.col("subject").alias("issuer_uri"),
            equity_symbol(F.col("raw_symbol")).alias("symbol"),
        ).filter(F.length(F.col("symbol")) > 0).join(
            self._issuer_ciks(), "issuer_uri", "inner"
        ).select(
            F.col("symbol"),
            F.col("cik"),
            F.lit(0).alias("priority"),
        )

        pairs = graph_pairs
        if self._ticker_cik_map:
            csv_pairs = self.spark.createDataFrame(
                sorted(self._ticker_cik_map.items()), schema=["symbol", "cik"]
            ).select(
                F.col("symbol"),
                F.col("cik"),
                F.lit(1).alias("priority"),
            )
            pairs = pairs.unionByName(csv_pairs)

        pairs = pairs.dropDuplicates(["symbol", "cik", "priority"])

        # One CIK per symbol. Ordered by priority then by the CIK itself, so a
        # tie inside one source resolves the same way on every run rather than
        # on whichever row a shuffle happens to deliver first.
        ranked = pairs.withColumn(
            "_rank",
            F.row_number().over(
                Window.partitionBy("symbol").orderBy(
                    F.col("priority").asc(), F.col("cik").asc()
                )
            ),
        )

        return ranked.filter(F.col("_rank") == 1).select("symbol", "cik")

    def _market_companies(self) -> DataFrame:
        """(entity, symbol, cik) for every market entity that reaches a company.

        An INNER join against the ticker -> CIK table, deliberately: a snapshot
        whose ticker resolves to no CIK contributes nothing. It does NOT fall
        back to a symbol-keyed company node, which is exactly what used to mint
        the second half of the split entity.
        """
        return self._market_symbol_bearers().join(
            self._ticker_to_cik(), "symbol", "inner"
        ).select("entity", "symbol", "cik")

    def _link_by_company(self) -> Optional[DataFrame]:
        """
        Link SEC issuers and market quotes to ONE unified company node.

        Both sides now key on the regulator's CIK, so a filing and a quote for
        the same company land on the same ``unified:Company_{CIK}`` node that
        sec_linker._unify_company_entities already mints:

            Filing -> Issuer -> Company <- Snapshot

        WHAT THIS REPLACES. Two code paths used to mint company entities into
        the same URI namespace under different keys -- sec_linker on the CIK,
        this module on the ticker -- and nothing reconciled them. The CIK-keyed
        node carried the filing structure, the symbol-keyed node carried the
        market link, and neither carried both. The graph still looked connected,
        because the issuer pointed at both, so a quote reached a filing as
        ``Snapshot -> Company_<SYMBOL> <- Issuer -> Company_<CIK>``: a split
        entity doing a bridge's job. For a GNN that is strictly worse than one
        node -- the neighbourhood describing one company is halved, and the two
        halves sit an extra hop apart.

        The type is spelled from sec_linker's constant for the same reason the
        key is: the two paths used to disagree about that as well
        (bls:UnifiedCompany here, sec:UnifiedCompany there), which sharded one
        concept across two node types in the PyG schema even where the URIs
        did match.

        The market side needs a ticker -> CIK step to get there; see
        _ticker_to_cik. The SEC side needs nothing, which is the coverage win:
        it reads hasIssuerCik, stated by every filing, instead of
        hasIssuerTradingSymbol, stated by a measured 24.57% of them.
        """
        market_companies = self._market_companies()
        issuer_companies = self._issuer_ciks()

        unified_uri = F.concat(F.lit(_UNIFIED_COMPANY_PREFIX), F.col("cik"))

        # Scaffolding for exactly the companies something points at, matching
        # how _link_by_geography scopes its region nodes. A typed node with no
        # incident edge is a row of feature tensor describing nothing.
        all_ciks = (
            market_companies.select("cik")
            .unionByName(issuer_companies.select("cik"))
            .distinct()
        )

        type_triples = all_ciks.select(
            unified_uri.alias("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.lit(_UNIFIED_COMPANY_TYPE).alias("object"),
        )

        cik_triples = all_ciks.select(
            unified_uri.alias("subject"),
            F.lit(_HAS_CIK).alias("predicate"),
            F.col("cik").alias("object"),
        )

        # The ticker survives as a PROPERTY of the unified company rather than
        # as its key. It is still the symbol a human recognises the company by,
        # and it is still what the market feed states -- it is simply no longer
        # what decides node identity.
        ticker_label_triples = market_companies.select("cik", "symbol").distinct().select(
            unified_uri.alias("subject"),
            F.lit(str(BLS_ENRICHMENT.ticker)).alias("predicate"),
            F.col("symbol").alias("object"),
        )

        market_links = market_companies.select(
            F.col("entity").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.refersToCompany)).alias("predicate"),
            unified_uri.alias("object"),
        )

        sec_links = issuer_companies.select(
            F.col("issuer_uri").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.refersToCompany)).alias("predicate"),
            unified_uri.alias("object"),
        )

        result = (
            type_triples
            .unionByName(cik_triples)
            .unionByName(ticker_label_triples)
            .unionByName(market_links)
            .unionByName(sec_links)
        )

        logger.info("  Company linking triples prepared (lazy)")
        return result

    # ================================================================
    # Step 1b: Equity sector -> economic sector
    # ================================================================

    def _link_equity_to_economic_sectors(self) -> Optional[DataFrame]:
        """Join the two sector vocabularies, by hand, with the gaps kept.

        bls:EconomicSector is the best-connected hub in the graph -- every
        economic-indicator type attaches to it -- and the market side had zero
        edges to it, because the market enricher classifies into its own GICS
        namespace and nothing reconciled the two. This is the Part 1 split one
        level up: the same concept represented twice, in parallel, never
        joined.

        The join is a CURATED table, not a name match, and it is deliberately
        incomplete: three GICS sectors map to nothing. See
        enrichment/sector_crosswalk.py for each row's reasoning and for why
        "completing" the table would make the graph worse.

        Emitted under relatedToEconomicSector, never belongsToSector. A
        constituent BELONGS TO its GICS sector; that GICS sector is merely
        RELATED TO an economic one. Collapsing membership and similarity into
        one relation would deny a GNN the ability to weight them differently.
        """
        rows = [
            (
                str(MARKET_ENRICHMENT[f"{_gics_sector_to_pascal(gics)}Sector"]),
                sector_uri,
                confidence,
            )
            for gics, mapped in EQUITY_TO_ECONOMIC_SECTORS.items()
            for sector_uri, confidence in mapped
        ]

        if not rows:
            return None

        crosswalk = self.spark.createDataFrame(
            sorted(rows), schema=["equity_sector", "economic_sector", "confidence"]
        )

        # Only the equity sectors this build actually classified something
        # into. The table covers eleven GICS sectors; a build holding quotes
        # for two of them should not carry nine sector nodes with one edge
        # each and no constituents.
        present = (
            self.triples_df
            .filter(F.col("predicate") == _RDF_TYPE)
            .filter(F.col("object") == _EQUITY_SECTOR_TYPE)
            .select(F.col("subject").alias("equity_sector"))
            .distinct()
        )

        matched = crosswalk.join(F.broadcast(present), "equity_sector", "inner")

        # Type the economic sector too. _link_by_sector types the ones its
        # keywords hit, but a build with market data and no BLS feed would
        # otherwise point these edges at an untyped URI, and node_mapper drops
        # an edge whose destination is not a node.
        economic_types = matched.select("economic_sector").distinct().select(
            F.col("economic_sector").alias("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.lit(str(BLS_ENRICHMENT.EconomicSector)).alias("object"),
        )

        related = matched.select(
            F.col("equity_sector").alias("subject"),
            F.lit(RELATED_TO_ECONOMIC_SECTOR).alias("predicate"),
            F.col("economic_sector").alias("object"),
        )

        # Confidence rides as a property of the equity sector node rather than
        # splitting the relation into strong/moderate variants, which would
        # double the edge types for something a scalar states better.
        confidence = matched.select(
            F.col("equity_sector").alias("subject"),
            F.lit(RELATION_CONFIDENCE).alias("predicate"),
            F.col("confidence").alias("object"),
        ).distinct()

        logger.info("  Equity-to-economic sector triples prepared (lazy)")
        return economic_types.unionByName(related).unionByName(confidence)

    # ================================================================
    # Step 1c: Filings -> economic sector, via the SIC code
    # ================================================================

    def _link_filings_by_sic(self) -> Optional[DataFrame]:
        """The filings' own industry code, instead of a keyword match.

        filings:hasSic is emitted by the mapper and this repo read it nowhere,
        so the filings side reached the economic sector hub only through
        _link_by_sector's keyword classifier, which matches substrings of a
        URI's local name and for a filing is matching an accession number.

        IT IS NOT ON EVERY FILING, and do not read a partial result here as a
        defect. The term comes from EDGAR's ``_source.sics`` array, which is
        routinely empty, and the mapper omits a triple rather than asserting a
        blank literal -- so a filing EDGAR left unclassified says nothing about
        its SIC. Measured on the e2e fixtures: 15 of 40 filings, 38%, giving
        14 of 39 companies a sector.

        That ceiling is EDGAR's, not this join's. Widening it means finding a
        second source for the classification, not fixing anything here.

        The sector lands on the COMPANY, not on the filing, and that is what
        makes it worth doing here rather than as a SEC intra-source step: a
        filing is a document, its industry is a fact about the issuer, and
        every filing by that issuer then inherits it through the company node
        Part 1 unified. The chain is

            Filing -hasSic-> code
            Filing -hasIssuer-> Issuer -hasIssuerCik-> CIK -> Company

        so this depends on the CIK-keyed company node existing, which is why
        it lands with Part 1 rather than before it.

        belongsToSector, not relatedToEconomicSector: this IS membership. The
        issuer's registered industry classification is a claim about what the
        company does, unlike the GICS-to-economic mapping above, which is a
        claim about two taxonomies resembling each other.
        """
        sic = extract_property(
            self.triples_df, _FILINGS_HAS_SIC, "raw_sic"
        ).select(
            F.col("subject").alias("filing"),
            F.trim(F.col("raw_sic")).alias("sic"),
        ).filter(F.length(F.col("sic")) > 0)

        if sic.head(1) == []:
            return None

        filing_issuers = self.triples_df.filter(
            F.col("predicate") == _FILINGS_HAS_ISSUER
        ).select(
            F.col("subject").alias("filing"),
            F.col("object").alias("issuer_uri"),
        )

        # SIC major group -> economic sector, expanded to one row per group so
        # the join is an equality rather than a range. Ninety-nine rows at
        # most, and it keeps the division boundaries stated once, in the
        # crosswalk module.
        division_rows = [
            (f"{group:02d}", division, sector)
            for low, high, division, sector in SIC_DIVISIONS
            for group in range(low, high + 1)
        ]
        divisions = self.spark.createDataFrame(
            division_rows, schema=["major_group", "division", "economic_sector"]
        )

        companies = (
            sic
            .join(filing_issuers, "filing", "inner")
            .join(self._issuer_ciks(), "issuer_uri", "inner")
            # Normalise to the 4-digit form before slicing. A SIC code typed
            # numeric upstream loses its leading zero, and "100" sliced at two
            # is "10" -- right only by accident -- while "755" sliced at two is
            # "75", which is a different division entirely.
            .withColumn(
                "major_group",
                F.substring(F.lpad(F.col("sic"), 4, "0"), 1, 2),
            )
            .filter(F.length(F.col("sic")) <= 4)
            .join(F.broadcast(divisions), "major_group", "inner")
            .select("cik", "economic_sector")
            .distinct()
        )

        company_uri = F.concat(F.lit(_UNIFIED_COMPANY_PREFIX), F.col("cik"))

        belongs = companies.select(
            company_uri.alias("subject"),
            F.lit(str(BLS_ENRICHMENT.belongsToSector)).alias("predicate"),
            F.col("economic_sector").alias("object"),
        )

        sector_types = companies.select("economic_sector").distinct().select(
            F.col("economic_sector").alias("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.lit(str(BLS_ENRICHMENT.EconomicSector)).alias("object"),
        )

        logger.info("  SIC sector triples prepared (lazy)")
        return belongs.unionByName(sector_types)

    # ================================================================
    # Step 3b: Sub-industry peer edges
    # ================================================================

    def _link_by_sub_industry(self) -> Optional[DataFrame]:
        """Peer edges between companies sharing one GICS sub-industry.

        WHY SUB-INDUSTRY AND NOT SECTOR. The constituents CSV carries both, and
        only the coarse one was ever read. Sector is too coarse to be a
        similarity signal: eleven buckets over 503 constituents, the largest
        holding 73 of them, so "same sector" separates almost nothing and a
        peer edge built on it is noise with a high degree. Sub-industry is
        roughly 160 buckets of 3-5 constituents, where "same sub-industry" is a
        sharp claim -- these are companies competing in one market.

        WHY PEER EDGES AND NOT A HUB NODE. A shared sub-industry node would be
        another high-degree hub that aggregates its whole bucket into one
        representation, which is the failure Part 2 of this work is unpicking
        for the period nodes. A bucket of 3-5 is small enough that the pairs
        themselves are cheap -- about ten edges per bucket -- and a pair says
        what a hub cannot: which specific company this one competes with.

        Restricted to companies the graph actually carries. The CSV lists every
        index member, and minting a node for each would add hundreds of
        vertices whose only edges are to each other -- a disconnected clique
        lattice bolted onto the graph, describing companies no source in this
        build mentions.

        No cross-taxonomy judgment is involved: one vocabulary, one column,
        already downloaded. That is why this half of the sector work lands
        first and the equity-to-economic mapping (which needs judgment, and
        deliberate gaps) is separate.
        """
        if not self._sub_industries:
            return None

        constituents = self.spark.createDataFrame(
            sorted(set(self._sub_industries)), schema=["symbol", "sub_industry"]
        )

        # Symbol -> CIK -> the company node, then keep only the companies some
        # source in this build actually points at.
        present = (
            self._market_companies().select("cik")
            .unionByName(self._issuer_ciks().select("cik"))
            .distinct()
        )

        members = (
            constituents.join(F.broadcast(self._ticker_to_cik()), "symbol", "inner")
            .join(present, "cik", "left_semi")
            .select("cik", "sub_industry")
            .distinct()
        )

        left = members.select(
            F.col("cik").alias("cik_a"), F.col("sub_industry")
        )
        right = members.select(
            F.col("cik").alias("cik_b"), F.col("sub_industry")
        )

        # One edge per unordered pair. `<` rather than `!=` because the relation
        # is symmetric and emitting both directions would double every peer
        # edge -- the same doubling the duplicated sector relation already costs
        # this graph elsewhere.
        pairs = left.join(right, "sub_industry", "inner").filter(
            F.col("cik_a") < F.col("cik_b")
        )

        result = pairs.select(
            F.concat(F.lit(_UNIFIED_COMPANY_PREFIX), F.col("cik_a")).alias("subject"),
            F.lit(_SHARES_SUB_INDUSTRY).alias("predicate"),
            F.concat(F.lit(_UNIFIED_COMPANY_PREFIX), F.col("cik_b")).alias("object"),
        )

        logger.info("  Sub-industry peer triples prepared (lazy)")
        return result

    # ================================================================
    # Step 4: Geographic Linking
    # ================================================================

    def _link_by_geography(self) -> Optional[DataFrame]:
        """
        Link entities by US state. Matches state names in subject URIs,
        NOAA alert area descriptions, and NOAA state FIPS codes.

        For NOAA: The updated RML mapper produces:
          - Area descriptions on cap:Area subjects (alert:{id}#area)
            via cap:hasAreaDescription
          - State FIPS codes on cap:Geocode subjects (alert:{id}#geocode-{fips})
            via nws:hasStateFIPS

        To link NOAA alerts to geographic regions, we:
          1. Match area descriptions containing state names
          2. Match state FIPS codes to state names
        Both approaches trace back to the Alert subject via the
        Alert → Info → Area → Geocode chain.
        """
        us_states = US_STATES
        state_fips_to_name = STATE_FIPS_TO_NAME

        # state_slug is the state name with spaces replaced by the separator
        # the source URIs use, wrapped in it. See _delimited_local_name.
        state_rows = [
            (
                name,
                state_key(name),
                STATE_ABBREVIATIONS[name],
                "_" + name.replace(' ', '_') + "_",
                "_" + state_key(name) + "_",
            )
            for name in us_states
        ]
        states_df = self.spark.createDataFrame(
            state_rows,
            schema=[
                "state_name", "state_key", "state_abbr",
                "state_slug", "state_slug_nospace",
            ],
        )

        # Create unified region entities
        region_uri = F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region"))

        type_triples = states_df.select(
            region_uri.alias("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.lit(str(BLS_ENRICHMENT.GeographicRegion)).alias("object")
        )

        label_triples = states_df.select(
            region_uri.alias("subject"),
            F.lit(_RDFS_LABEL).alias("predicate"),
            F.col("state_name").alias("object")
        )

        # Match LAUS entities containing state names
        laus_links = None
        if 'bls' in self.available_sources:
            laus_prefixes = _entity_prefixes(LAUS)
            laus_match = F.col("subject").startswith(laus_prefixes[0])
            for p in laus_prefixes[1:]:
                laus_match = laus_match | F.col("subject").startswith(p)
            laus_entities = (
                self.triples_df
                .select("subject")
                .distinct()
                .filter(laus_match)
            )

            # The state name must be the DELIMITED HEAD of the local name, not
            # a substring of the whole URI anywhere.
            #
            # `subject.contains(state_key)` linked every West Virginia series to
            # VirginiaRegion as well as to its own, because "WestVirginia"
            # contains "Virginia". That is the only such pair among the 51
            # names -- see test_exactly_one_state_name_contains_another, which
            # also records that Arkansas/Kansas is NOT one of them, the
            # comparison being case-sensitive and the 'k' lowercase. One pair is
            # enough: merging two genuinely distinct regions is worse than
            # missing one, since the weather feed then reaches the wrong state's
            # unemployment series with full confidence.
            #
            # It does NOT separate a metro area whose name BEGINS with a state
            # name: "Kansas_City_MO" still lands on KansasRegion, though the
            # metro straddles two states. That needs the federal delineation
            # file which puts metro resolution out of scope, so this is the
            # state-vs-state fix and not the metro one.
            #
            # Anchored at the HEAD because these URIs put the area first
            # ("Midwest_June2026_HiresLevel"). An unanchored delimited match
            # does not separate the Virginia pair either -- "_West_Virginia_"
            # contains "_Virginia_" as a substring -- so the anchor is doing
            # real work rather than being a convenience.
            #
            # Both separator spellings are accepted because the two are minted
            # differently across sources: "New_York" from a label slug,
            # "NewYork" from a URI-safe key.
            laus_matched = laus_entities.crossJoin(F.broadcast(states_df)).filter(
                _delimited_local_name(F.col("subject")).startswith(
                    F.col("state_slug")
                )
                | _delimited_local_name(F.col("subject")).startswith(
                    F.col("state_slug_nospace")
                )
            )

            laus_links = laus_matched.select(
                F.col("subject"),
                F.lit(str(BLS_ENRICHMENT.hasRegion)).alias("predicate"),
                F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
            )

        # Match NOAA alerts by area description containing state name
        # AND by state FIPS code
        noaa_links = None
        if 'noaa' in self.available_sources:
            noaa_link_dfs: List[DataFrame] = []

            # Structural chain shared by both strategies (lazy — defining
            # these costs nothing until joined):
            #   Alert → hasInfo → Info → hasArea → Area
            info_areas = self.triples_df.filter(
                F.col("predicate") == _CAP_HAS_AREA
            ).select(
                F.col("subject").alias("info"),
                F.col("object").alias("area"),
            )

            alert_infos = self.triples_df.filter(
                F.col("predicate") == _CAP_HAS_INFO
            ).select(
                F.col("subject").alias("alert"),
                F.col("object").alias("info"),
            )

            # Strategy 1: Area description contains state name
            # Area descriptions are on cap:Area subjects, so join through the
            # forward chain: Alert → Info → Area → hasAreaDescription
            area_descs = extract_property(
                self.triples_df, _CAP_HAS_AREA_DESC, "area_desc"
            )

            if area_descs.head(1):
                # area_descs has (subject=area_uri, area_desc)
                # Join: area_desc → area → info → alert
                area_to_alert = (
                    area_descs
                    .join(info_areas, area_descs.subject == info_areas.area, "inner")
                    .join(alert_infos, "info", "inner")
                    .select("alert", "area_desc")
                )

                # Match the ABBREVIATION as well as the full name.
                #
                # NWS writes area descriptions as "Lincoln, KS; Russell, KS" --
                # county plus two-letter state -- and the full name never
                # appears. Matching only `contains("Kansas")` therefore matched
                # nothing on every alert ever ingested, and this strategy
                # produced no link at all.
                #
                # The abbreviation is matched with its ", " separator rather
                # than bare, because two letters appear inside ordinary words:
                # a bare "contains('OR')" hits "ORANGE" and a bare "IN" hits
                # almost everything. Anchoring on the comma is how NWS actually
                # writes it and keeps the match to the state field.
                noaa_area_matched = area_to_alert.crossJoin(F.broadcast(states_df)).filter(
                    F.col("area_desc").contains(F.col("state_name"))
                    | F.col("area_desc").contains(
                        F.concat(F.lit(", "), F.col("state_abbr"))
                    )
                )

                noaa_area_links = noaa_area_matched.select(
                    F.col("alert").alias("subject"),
                    F.lit(str(BLS_ENRICHMENT.affectsRegion)).alias("predicate"),
                    F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
                )
                noaa_link_dfs.append(noaa_area_links)

            # Strategy 2: State FIPS code matching
            #
            # Geocode subjects carry nws:hasStateFIPS as the 3-character head of
            # a SAME code ("040", "024"), never a 2-digit FIPS -- upstream slices
            # it as value[:3], so the leading part-digit is structural and always
            # present. The lookup below is keyed 2-digit; see the normalisation.
            #
            # Note also that nws:hasFIPSCode carries the SAME value as
            # nws:hasSAMECode -- both read upstream's fips_code column, which is
            # populated from properties.geocode.SAME. hasFIPSCode is a misnomer
            # for the SAME code and is not a county FIPS.
            state_fips_rows = [
                (fips, name, name.replace(' ', ''))
                for fips, name in state_fips_to_name.items()
            ]
            state_fips_df = self.spark.createDataFrame(
                state_fips_rows, ["fips_code", "state_name", "state_key"]
            )

            # Normalize the code to the two digits the lookup above is keyed on.
            #
            # The lookup holds 2-digit state FIPS ("20" Kansas, "42"
            # Pennsylvania) while the mapper emits the 3-digit head of a SAME
            # code -- "020", "042" -- which is a leading part-digit followed by
            # the state. Compared as strings those never match, so this whole
            # strategy linked nothing on every alert ever ingested.
            #
            # normalized_state_fips reads the LAST two digits, which is correct
            # for the current 3-character form AND for the 2-character form
            # upstream is moving to -- see the note on that function for why it
            # must stay tolerant of both permanently, and for the lpad trap it
            # exists to avoid.
            state_fips_triples = self.triples_df.filter(
                F.col("predicate") == _NWS_HAS_STATE_FIPS
            ).select(
                F.col("subject").alias("geocode"),
                normalized_state_fips(F.col("object")).alias("state_fips"),
            )

            if state_fips_triples.head(1):
                # Trace geocode → area → info → alert
                area_geocodes = self.triples_df.filter(
                    F.col("predicate") == _CAP_HAS_GEOCODE
                ).select(
                    F.col("subject").alias("area"),
                    F.col("object").alias("geocode"),
                )

                geocode_to_alert = (
                    state_fips_triples
                    .join(area_geocodes, "geocode", "inner")
                    .join(info_areas, "area", "inner")
                    .join(alert_infos, "info", "inner")
                    .select("alert", "state_fips")
                    .dropDuplicates()
                )

                fips_matched = geocode_to_alert.join(
                    F.broadcast(state_fips_df),
                    geocode_to_alert.state_fips == state_fips_df.fips_code,
                    "inner",
                )

                noaa_fips_links = fips_matched.select(
                    F.col("alert").alias("subject"),
                    F.lit(str(BLS_ENRICHMENT.affectsRegion)).alias("predicate"),
                    F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
                )
                noaa_link_dfs.append(noaa_fips_links)

            if noaa_link_dfs:
                noaa_links = noaa_link_dfs[0]
                for df in noaa_link_dfs[1:]:
                    noaa_links = noaa_links.unionByName(df)
                noaa_links = noaa_links.dropDuplicates()

        # Only the regions something actually points at.
        #
        # All 50 states used to be typed and labelled unconditionally, while the
        # links below are conditional on a LAUS or NOAA entity naming one. With
        # no LAUS feed present that is 50 typed nodes with zero incident edges —
        # a whole node type of isolated vertices a GNN can propagate nothing
        # through, and 50 rows of feature tensor describing nothing.
        #
        # Scoping the scaffolding to the linked set makes the region nodes a
        # consequence of the data rather than of the state list. Nothing else
        # changes: a region that IS linked gets exactly the type and label it
        # got before.
        links = [df for df in (laus_links, noaa_links) if df is not None]
        if not links:
            logger.info("  No geographic links matched — no region entities minted")
            return None

        all_links = links[0]
        for df in links[1:]:
            all_links = all_links.unionByName(df)
        all_links = all_links.dropDuplicates()

        linked_regions = all_links.select(F.col("object").alias("subject")).distinct()

        result = (
            type_triples.join(linked_regions, "subject", "left_semi")
            .unionByName(label_triples.join(linked_regions, "subject", "left_semi"))
            .unionByName(all_links)
        )

        census = self._link_census_regions(linked_regions)
        if census is not None:
            result = result.unionByName(census)

        logger.info("  Geographic linking triples prepared (lazy)")
        return result

    def _link_census_regions(
        self, linked_state_regions: DataFrame
    ) -> Optional[DataFrame]:
        """The rung that gets the weather feed to the job-openings series.

        The job-openings regions are NOT states -- they are the four census
        regions -- so the state-level path cannot reach them and they were the
        second, unjoined region vocabulary (`jolts_Region`). This adds the
        census region as a shared node and hangs the state regions off it:

            WeatherAlert -> KansasRegion -> MidwestCensusRegion
                                              ^
                            jolts:Midwest_Region (owl:sameAs)
                                              ^
                            jolts HiresLevel -hasRegion-

        so the path a weather alert has to a regional economic series exists
        through geography, and through nothing else. Sector is a different axis
        and must not be used for this: a sector is not located in a region.

        owl:sameAs to the source region entity, matching how TemporalUnifier
        joins the same period stated in two source vocabularies. It is the right
        claim here -- the jolts Midwest region and the census Midwest region are
        one region, not two that resemble each other.

        Keyed on the census-region CODE where upstream states it, and on the
        region NAME otherwise. The code is what upstream is adding
        (jolts:hasCensusRegionCode); the name path works today, so the bridge
        does not have to wait for a term that will merely confirm it.
        """
        census_rows = [
            (name, code, state_key(name))
            for name, code in CENSUS_REGION_CODES.items()
        ]
        census_df = self.spark.createDataFrame(
            census_rows, schema=["census_name", "census_code", "census_key"]
        )

        census_uri = F.concat(
            F.lit(str(UNIFIED)), F.col("census_key"), F.lit("CensusRegion")
        )

        # state region -> census region, from the 51-row table. Restricted to
        # the state regions something already points at, so the census node is
        # a consequence of the data rather than of the table.
        membership_rows = [
            (state_key(state), region, state)
            for state, region in STATE_TO_CENSUS_REGION.items()
        ]
        membership_df = self.spark.createDataFrame(
            membership_rows, schema=["member_key", "census_name", "state_name"]
        ).withColumn(
            "region_uri",
            F.concat(F.lit(str(UNIFIED)), F.col("member_key"), F.lit("Region")),
        )

        within = (
            membership_df.join(
                linked_state_regions.withColumnRenamed("subject", "region_uri"),
                "region_uri",
                "left_semi",
            )
            .join(F.broadcast(census_df), "census_name", "inner")
            .select(
                F.col("region_uri").alias("subject"),
                F.lit(_WITHIN_CENSUS_REGION).alias("predicate"),
                census_uri.alias("object"),
            )
        )

        # The source-side region entities, matched by NAME.
        #
        # This is the only path, and it is sufficient: the job-openings survey
        # states its region as a named node (id/jolts/Midwest_Region) and never
        # as a code. A code-keyed path was written here first, on the strength
        # of an issue claiming jolts:hasCensusRegionCode was coming; it is not,
        # and the upstream region identifier is "MW" rather than a census
        # number anyway, so a numeric reader would have matched nothing even
        # if it had shipped.
        source_regions = self.triples_df.filter(
            (F.col("predicate") == _RDF_TYPE)
            & (F.col("object") == _JOLTS_REGION_TYPE)
        ).select(F.col("subject").alias("entity")).crossJoin(
            F.broadcast(census_df)
        ).filter(
            # The same delimited-head rule the state match uses. "Midwest_Region"
            # reads as "_Midwest_Region_", which starts with "_Midwest_"; a
            # region merely CONTAINING a census name does not match.
            _delimited_local_name(F.col("entity")).startswith(
                F.concat(F.lit("_"), F.col("census_name"), F.lit("_"))
            )
        ).select("entity", "census_name", "census_key").dropDuplicates()

        same_as = source_regions.select(
            F.concat(
                F.lit(str(UNIFIED)), F.col("census_key"), F.lit("CensusRegion")
            ).alias("subject"),
            F.lit(_OWL_SAME_AS).alias("predicate"),
            F.col("entity").alias("object"),
        )

        # Scaffolding for exactly the census regions something reaches, from
        # either side.
        reached = (
            within.select(F.col("object").alias("subject"))
            .unionByName(same_as.select("subject"))
            .distinct()
        )

        if reached.head(1) == []:
            return None

        type_triples = reached.select(
            F.col("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.lit(str(BLS_ENRICHMENT.CensusRegion)).alias("object"),
        )

        label_triples = census_df.select(
            census_uri.alias("subject"),
            F.lit(_RDFS_LABEL).alias("predicate"),
            F.col("census_name").alias("object"),
        ).join(reached, "subject", "left_semi")

        return (
            type_triples
            .unionByName(label_triples)
            .unionByName(within)
            .unionByName(same_as)
        )

    # ================================================================
    # Step 5: Causal Links
    # ================================================================

    def _create_causal_links(self) -> Optional[DataFrame]:
        """
        Link a BLS indicator to a market entity in the same economic sector.

        WHY THIS NEVER FIRED. Both sides were selected by filtering for
        bls:belongsToSector -- but only the BLS side emits that. The market
        enricher writes market:belongsToSector, a DIFFERENT predicate, because
        its sector is a GICS sector and lives in its own vocabulary. So the
        market half of the filter matched nothing on every graph ever built,
        the inner join had one empty side, and the step returned None while
        logging that it ran. The e2e suite recorded the absence and blamed the
        missing constituents CSV; that was only half the reason, and the join
        would have stayed dead with the CSV configured.

        The fix is NOT to make market emit the BLS predicate. A GICS sector and
        an economic sector are different classifications of different things --
        that is the whole premise of enrichment/sector_crosswalk.py -- and
        collapsing them onto one predicate would assert the equivalence that
        module exists to deny.

        Instead the market side is carried to the economic sector THROUGH the
        crosswalk, which is what it is for:

            BLS series  -belongsToSector->  EconomicSector
                                                  ^
                                    -relatedToEconomicSector-
                                                  |
            snapshot -market:belongsToSector-> GICS sector

        so both sides end up keyed on the same economic sector and the join has
        something to match. Note this makes the link only as good as the
        curated table -- a GICS sector deliberately mapped to nothing (see the
        three gaps) contributes no causal edge, which is correct.
        """
        bls_sector = (
            self.triples_df
            .filter(F.col("predicate") == str(BLS_ENRICHMENT.belongsToSector))
            .select(
                F.col("subject").alias("entity"),
                F.col("object").alias("sector"),
            )
        )

        # The market side, resolved to an ECONOMIC sector via the crosswalk.
        market_gics = (
            self.triples_df
            .filter(F.col("predicate") == str(MARKET_ENRICHMENT.belongsToSector))
            .select(
                F.col("subject").alias("entity"),
                F.col("object").alias("equity_sector"),
            )
        )

        # Built from the TABLE, not from the relatedToEconomicSector triples.
        #
        # Those triples are minted by _link_equity_to_economic_sectors in this
        # same enrich() call and are not unioned into triples_df until every
        # step has run -- so reading them here finds nothing, and the join dies
        # exactly the way the predicate mismatch used to. Every step in this
        # module reads the INPUT frame; none can see another's output.
        crosswalk = self.spark.createDataFrame(
            sorted({
                (
                    str(MARKET_ENRICHMENT[f"{_gics_sector_to_pascal(gics)}Sector"]),
                    sector_uri,
                )
                for gics, mapped in EQUITY_TO_ECONOMIC_SECTORS.items()
                for sector_uri, _confidence in mapped
            }),
            schema=["equity_sector", "sector"],
        )

        market_resolved = market_gics.join(
            crosswalk, "equity_sector", "inner"
        ).select("entity", "sector")

        bls_prefixes = _BLS_ENTITY_PREFIXES
        market_prefixes = _MARKET_ENTITY_PREFIXES

        # BLS entities in sectors
        bls_filter = F.col("entity").startswith(bls_prefixes[0])
        for p in bls_prefixes[1:]:
            bls_filter = bls_filter | F.col("entity").startswith(p)

        bls_sector = bls_sector.filter(bls_filter).select(
            F.col("entity").alias("bls_entity"), F.col("sector").alias("bls_sector")
        )

        # Market entities in sectors
        market_filter = F.col("entity").startswith(market_prefixes[0])
        for p in market_prefixes[1:]:
            market_filter = market_filter | F.col("entity").startswith(p)

        market_sector = market_resolved.filter(market_filter).select(
            F.col("entity").alias("market_entity"), F.col("sector").alias("market_sector")
        )

        if bls_sector.head(1) and market_sector.head(1):
            # Join on sector — BLS indicator leadsTo Market ticker in same sector
            causal = bls_sector.join(
                market_sector,
                bls_sector.bls_sector == market_sector.market_sector,
                how="inner"
            )

            result = causal.select(
                F.col("bls_entity").alias("subject"),
                F.lit(str(BLS_ENRICHMENT.leadsTo)).alias("predicate"),
                F.col("market_entity").alias("object")
            )

            logger.info("  Causal link triples prepared (lazy)")
            return result

        return None

    # ================================================================
    # Step 6: Measurement Type Alignment
    # ================================================================

    def _align_measurement_types(self) -> Optional[DataFrame]:
        """
        Add unified measurement type classifications.
        Maps source-specific types to unified types via broadcast join.
        """
        type_mappings = [
            (str(CPI.Index), str(BLS_ENRICHMENT.PriceIndex)),
            (str(PPI.IndexValue), str(BLS_ENRICHMENT.PriceIndex)),
            (str(JOLTS.JobOpeningsRate), str(BLS_ENRICHMENT.RateMeasurement)),
            (str(JOLTS.HiresRate), str(BLS_ENRICHMENT.RateMeasurement)),
            (str(JOLTS.QuitsRate), str(BLS_ENRICHMENT.RateMeasurement)),
            (str(LAUS.UnemploymentRate), str(BLS_ENRICHMENT.RateMeasurement)),
        ]

        mapping_df = self.spark.createDataFrame(
            type_mappings, schema=["source_type", "unified_type"]
        )

        # Find entities with these source types
        typed_entities = (
            self.triples_df
            .filter(F.col("predicate") == _RDF_TYPE)
            .select(F.col("subject").alias("entity"), F.col("object").alias("source_type"))
        )

        matched = typed_entities.join(F.broadcast(mapping_df), on="source_type", how="inner")

        result = matched.select(
            F.col("entity").alias("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.col("unified_type").alias("object")
        )

        logger.info("  Measurement alignment triples prepared (lazy)")
        return result


def enrich_cross_source(
    spark: SparkSession,
    triples_df: DataFrame,
    ticker_cik_map: Optional[Dict[str, str]] = None,
    sub_industries: Optional[Sequence[Tuple[str, str]]] = None,
) -> DataFrame:
    """Main entry point for cross-source enrichment.

    The two reference tables are read from the index-constituents CSV by the
    caller (see EnrichmentPipeline), not here, so this module stays free of S3
    and stays testable with a literal dict.
    """
    linker = CrossSourceLinker(spark, triples_df, ticker_cik_map, sub_industries)
    return linker.enrich()