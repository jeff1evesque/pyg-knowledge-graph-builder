"""
Cross-Source Enrichment Linker — PySpark Implementation

Links entities across different data source families (BLS, SEC, Market, NOAA).
Runs entirely on Spark executors — no rdflib, no driver data operations.

All methods:
1. Read from self.triples_df (the enriched triples from all intra-source enrichers)
2. Return new triples DataFrames
3. Never call .collect(), .toPandas(), or .toLocalIterator()
"""
from pyspark.sql import DataFrame, SparkSession
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
from spark_jobs.enrichment.intra_source.bls.patterns import (
    BLS_SECTOR_CORRELATION, BLS_SECTOR_PATTERNS,
)
from spark_jobs.enrichment.intra_source.market.patterns import (
    assert_ciks_are_padded,
)
from spark_jobs.enrichment.intra_source.market.symbols import equity_symbol
from spark_jobs.utils.spark_rdf_utils import (
    extract_entities_by_type, extract_property, deduplicate_against_existing,
)
from typing import Dict, List, Optional, Sequence, Set, Tuple
import logging

logger = logging.getLogger(__name__)


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

# Two-letter postal abbreviations, for matching NWS area descriptions.
#
# NWS writes them as "Lincoln, KS; Russell, KS" and never spells the state out,
# so the full-name list alone matched nothing. Kept beside the state list the
# geographic step already carries rather than derived, because the postal code
# is not a function of the name.
_STATE_ABBREVIATIONS = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
    'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
}

# URI constants
_RDF_TYPE = str(RDF.type)
_OWL_SAME_AS = str(OWL.sameAs)
_RDFS_LABEL = str(RDFS.label)
_SECTOR_CORRELATION = str(BLS_SECTOR_CORRELATION)

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
# unused: the name reads as "the county FIPS is available", when upstream
# populates that column from properties.geocode.SAME and it carries the SAME
# code unchanged. Anyone reaching for the dead constant would have joined a
# SAME code against a county-FIPS lookup. Upstream is correcting the term; if a
# county-level join is ever wanted here, bind it then, against the fixed shape.
_NWS_HAS_STATE_FIPS = str(WEATHER.hasStateFIPS)

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
        """Detect which data sources are present (collects at most ~10 rows)."""
        sources = set()

        bls_prefixes = _BLS_ENTITY_PREFIXES
        sec_prefixes = _SEC_ENTITY_PREFIXES
        market_prefixes = _MARKET_ENTITY_PREFIXES
        # NOAA alert instances use the ALERT namespace
        # (https://api.weather.gov/alerts/) — a real publisher namespace, so it
        # has no id/ counterpart and is matched as-is. CAP/WEATHER are checked
        # too, for type triples.
        noaa_prefixes = [str(ALERT), *_entity_prefixes(CAP, WEATHER)]

        # Sample subjects to detect sources — bounded, fast
        sample = (
            self.triples_df
            .select("subject")
            .limit(100000)
            .distinct()
            .collect()
        )

        for row in sample:
            s = row.subject
            if any(s.startswith(p) for p in bls_prefixes):
                sources.add('bls')
            elif any(s.startswith(p) for p in sec_prefixes):
                sources.add('sec')
            elif any(s.startswith(p) for p in market_prefixes):
                sources.add('market')
            elif any(s.startswith(p) for p in noaa_prefixes):
                sources.add('noaa')

            if len(sources) == 4:
                break

        return sources

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
        logger.info("\n[Step 1/6] Creating sector-based links...")
        self._append(new_dfs, self._link_by_sector())

        logger.info("\n[Step 2/6] Linking by company/ticker...")
        if 'sec' in self.available_sources and 'market' in self.available_sources:
            self._append(new_dfs, self._link_by_company())

            logger.info("\n[Step 3/6] Linking constituents by sub-industry...")
            self._append(new_dfs, self._link_by_sub_industry())

        logger.info("\n[Step 4/6] Linking by geographic region...")
        self._append(new_dfs, self._link_by_geography())

        logger.info("\n[Step 5/6] Creating causal relationships...")
        self._append(new_dfs, self._create_causal_links())

        logger.info("\n[Step 6/6] Aligning measurement types...")
        self._append(new_dfs, self._align_measurement_types())

        if not new_dfs:
            return self.spark.createDataFrame([], schema=self.triples_df.schema)

        all_new = new_dfs[0]
        for df in new_dfs[1:]:
            all_new = all_new.unionByName(df)

        # Truncate the plan before the dedup pass (see EnrichmentPipeline._settle).
        # Each step above is a join over triples_df, so this union carries five of
        # those subtrees; feeding it straight into dropDuplicates + the anti-join
        # against triples_df makes Catalyst's constraint inference blow the driver
        # heap while *planning* (OutOfMemoryError inside .cache(), before any task
        # runs). Materializing here replaces the six subtrees with one scan.
        all_new = all_new.localCheckpoint(eager=True)

        all_new = all_new.dropDuplicates(["subject", "predicate", "object"])
        all_new = deduplicate_against_existing(all_new, self.triples_df)

        all_new = all_new.cache()
        total = all_new.count()

        logger.info(f"\nCross-source enrichment produced {total} new triples")
        return all_new

    @staticmethod
    def _append(lst: list, item: Optional[DataFrame]):
        if item is not None:
            lst.append(item)

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

        # Sector correlation triples — one predicate for every sector, the
        # sector itself carried by the object (see BLS_SECTOR_CORRELATION).
        correlation_triples = matched.select(
            F.col("subject").alias("subject"),
            F.lit(_SECTOR_CORRELATION).alias("predicate"),
            F.col("sector_uri").alias("object")
        )

        result = sector_type_triples.unionByName(belongs_triples).unionByName(correlation_triples)

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
        us_states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
            'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
            'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
            'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
            'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
            'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
            'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
            'West Virginia', 'Wisconsin', 'Wyoming'
        ]

        # State FIPS code → state name mapping (2-digit codes)
        state_fips_to_name = {
            '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
            '06': 'California', '08': 'Colorado', '09': 'Connecticut',
            '10': 'Delaware', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
            '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
            '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
            '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan',
            '27': 'Minnesota', '28': 'Mississippi', '29': 'Missouri',
            '30': 'Montana', '31': 'Nebraska', '32': 'Nevada',
            '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
            '36': 'New York', '37': 'North Carolina', '38': 'North Dakota',
            '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon',
            '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina',
            '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
            '49': 'Utah', '50': 'Vermont', '51': 'Virginia',
            '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin',
            '56': 'Wyoming',
        }

        state_rows = [
            (s, s.replace(' ', ''), _STATE_ABBREVIATIONS[s]) for s in us_states
        ]
        states_df = self.spark.createDataFrame(
            state_rows, schema=["state_name", "state_key", "state_abbr"]
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

            laus_matched = laus_entities.crossJoin(F.broadcast(states_df)).filter(
                F.col("subject").contains(F.col("state_key"))
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
            # Taking the LAST two digits rather than stripping a leading zero:
            # the part-digit is not always 0 (it marks a partial-county code),
            # so trimming zeros would silently mangle those.
            #
            # substring(-2, 2) alone, NOT lpad(...,2,"0") first. Spark's lpad
            # truncates when the input is LONGER than the target width, and it
            # truncates from the right -- lpad("040",2,"0") is "04", not "40".
            # That destroyed the state digit and silently produced the WRONG
            # state: every "0NN" head resolved to state "0N", so Oklahoma (040)
            # and Texas (048) both linked to Arizona (04), and the unmapped 099
            # linked to Connecticut (09) instead of dropping. This failed
            # louder than a dead join -- it emitted confident, wrong edges.
            #
            # substring with a negative start already reads from the end and is
            # safe on short input, so no padding is needed at all.
            state_fips_triples = self.triples_df.filter(
                F.col("predicate") == _NWS_HAS_STATE_FIPS
            ).select(
                F.col("subject").alias("geocode"),
                F.substring(F.trim(F.col("object")), -2, 2).alias("state_fips"),
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

        logger.info("  Geographic linking triples prepared (lazy)")
        return result

    # ================================================================
    # Step 5: Causal Links
    # ================================================================

    def _create_causal_links(self) -> Optional[DataFrame]:
        """
        Create causal relationships by joining entities that share
        the same sector across different source families.
        """
        belongs_to_sector = (
            self.triples_df
            .filter(F.col("predicate") == str(BLS_ENRICHMENT.belongsToSector))
            .select(
                F.col("subject").alias("entity"),
                F.col("object").alias("sector")
            )
        )

        bls_prefixes = _BLS_ENTITY_PREFIXES
        market_prefixes = _MARKET_ENTITY_PREFIXES

        # BLS entities in sectors
        bls_filter = F.col("entity").startswith(bls_prefixes[0])
        for p in bls_prefixes[1:]:
            bls_filter = bls_filter | F.col("entity").startswith(p)

        bls_sector = belongs_to_sector.filter(bls_filter).select(
            F.col("entity").alias("bls_entity"), F.col("sector").alias("bls_sector")
        )

        # Market entities in sectors
        market_filter = F.col("entity").startswith(market_prefixes[0])
        for p in market_prefixes[1:]:
            market_filter = market_filter | F.col("entity").startswith(p)

        market_sector = belongs_to_sector.filter(market_filter).select(
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