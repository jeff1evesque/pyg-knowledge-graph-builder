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
from rdflib.namespace import RDF, RDFS, OWL, XSD

from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT, NOAA_ENRICHMENT,
    UNIFIED, CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER,
    WKYENG, SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP, MARKET, CAP, NWS,
    ALERT,
)
from spark_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from spark_jobs.utils.spark_rdf_utils import (
    extract_entities_by_type, extract_property, build_entity_properties,
    deduplicate_against_existing,
)
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

# URI constants
_RDF_TYPE = str(RDF.type)
_OWL_SAME_AS = str(OWL.sameAs)
_RDFS_LABEL = str(RDFS.label)

# NOAA type URIs — aligned with updated RML mapper
_NWS_WEATHER_ALERT_TYPE = str(NWS.WeatherAlert)
_CAP_ALERT_TYPE = str(CAP.Alert)

# NOAA area description — on Area subject (alert:{id}#area)
_CAP_HAS_AREA_DESC = str(CAP.hasAreaDescription)

# NOAA geocode properties for geographic linking
_NWS_HAS_STATE_FIPS = str(NWS.hasStateFIPS)
_NWS_HAS_FIPS_CODE = str(NWS.hasFIPSCode)

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

    def __init__(self, spark: SparkSession, triples_df: DataFrame):
        self.spark = spark
        self.triples_df = triples_df
        self.available_sources = self._detect_sources()
        logger.info(f"Detected data sources: {', '.join(self.available_sources)}")

    def _detect_sources(self) -> Set[str]:
        """Detect which data sources are present (collects at most ~10 rows)."""
        sources = set()

        bls_prefixes = [str(ns) for ns in [CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER, WKYENG]]
        sec_prefixes = [str(ns) for ns in [SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP]]
        market_prefix = str(MARKET)
        # NOAA alert instances use the ALERT namespace (https://api.weather.gov/alerts/)
        # Also check CAP namespace for type triples
        noaa_prefixes = [str(ALERT), str(CAP), str(NWS)]

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
            elif s.startswith(market_prefix):
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
        logger.info("\n[Step 1/5] Creating sector-based links...")
        self._append(new_dfs, self._link_by_sector())

        logger.info("\n[Step 2/5] Linking by company/ticker...")
        if 'sec' in self.available_sources and 'market' in self.available_sources:
            self._append(new_dfs, self._link_by_company())

        logger.info("\n[Step 3/5] Linking by geographic region...")
        self._append(new_dfs, self._link_by_geography())

        logger.info("\n[Step 4/5] Creating causal relationships...")
        self._append(new_dfs, self._create_causal_links())

        logger.info("\n[Step 5/5] Aligning measurement types...")
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
        # Build sector keyword lookup: (keyword_normalized, sector_uri, relationship)
        sector_rows = []
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            sector_uri = str(pattern['sector_uri'])
            relationship = str(pattern['relationship'])
            for dataset_name, keywords in pattern['keywords'].items():
                for keyword in keywords:
                    normalized = keyword.replace(' ', '_').replace("'", '').replace(',', '').replace('-', '_')
                    sector_rows.append((normalized.lower(), sector_uri, relationship))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows,
            schema=["keyword", "sector_uri", "relationship"]
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

        # Sector correlation triples
        correlation_triples = matched.select(
            F.col("subject").alias("subject"),
            F.col("relationship").alias("predicate"),
            F.col("sector_uri").alias("object")
        )

        result = sector_type_triples.unionByName(belongs_triples).unionByName(correlation_triples)

        logger.info("  Sector linking triples prepared (lazy)")
        return result

    # ================================================================
    # Step 3: Company/Ticker Linking
    # ================================================================

    def _link_by_company(self) -> Optional[DataFrame]:
        """
        Link SEC filings to Market tickers via ticker symbol.

        Creates unified:Company_<SYMBOL> entities.
        """
        # Get market ticker symbols
        market_tickers = (
            extract_entities_by_type(self.triples_df, str(MARKET.StockTicker))
            .join(
                extract_property(self.triples_df, str(MARKET.symbol), "symbol"),
                on=F.col("entity") == F.col("subject")
            )
            .select(F.col("entity").alias("ticker_uri"), "symbol")
            .drop("subject")
        )

        # Get SEC filing ticker symbols
        sec_tickers = extract_property(
            self.triples_df, str(SEC_FILINGS.hasIssuerTicker), "symbol"
        ).select(
            F.col("subject").alias("filing_uri"),
            F.col("symbol")
        )

        # Create unified company entities
        unified_uri = F.concat(F.lit(str(UNIFIED)), F.lit("Company_"), F.col("symbol"))

        # Type triples
        all_symbols = market_tickers.select("symbol").union(sec_tickers.select("symbol")).distinct()
        type_triples = all_symbols.select(
            unified_uri.alias("subject"),
            F.lit(_RDF_TYPE).alias("predicate"),
            F.lit(str(BLS_ENRICHMENT.UnifiedCompany)).alias("object")
        )

        # Ticker label triples
        ticker_label_triples = all_symbols.select(
            unified_uri.alias("subject"),
            F.lit(str(BLS_ENRICHMENT.ticker)).alias("predicate"),
            F.col("symbol").alias("object")
        )

        # Market ticker → company
        market_links = market_tickers.select(
            F.col("ticker_uri").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.refersToCompany)).alias("predicate"),
            F.concat(F.lit(str(UNIFIED)), F.lit("Company_"), F.col("symbol")).alias("object")
        )

        # SEC filing → company
        sec_links = sec_tickers.select(
            F.col("filing_uri").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.refersToCompany)).alias("predicate"),
            F.concat(F.lit(str(UNIFIED)), F.lit("Company_"), F.col("symbol")).alias("object")
        )

        result = (
            type_triples
            .unionByName(ticker_label_triples)
            .unionByName(market_links)
            .unionByName(sec_links)
        )

        logger.info("  Company linking triples prepared (lazy)")
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

        state_rows = [(s, s.replace(' ', '')) for s in us_states]
        states_df = self.spark.createDataFrame(state_rows, schema=["state_name", "state_key"])

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
            laus_prefix = str(LAUS)
            laus_entities = (
                self.triples_df
                .select("subject")
                .distinct()
                .filter(F.col("subject").startswith(laus_prefix))
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

                noaa_area_matched = area_to_alert.crossJoin(F.broadcast(states_df)).filter(
                    F.col("area_desc").contains(F.col("state_name"))
                )

                noaa_area_links = noaa_area_matched.select(
                    F.col("alert").alias("subject"),
                    F.lit(str(BLS_ENRICHMENT.affectsRegion)).alias("predicate"),
                    F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
                )
                noaa_link_dfs.append(noaa_area_links)

            # Strategy 2: State FIPS code matching
            # Geocode subjects have nws:hasStateFIPS with 2-digit state codes
            state_fips_rows = [
                (fips, name, name.replace(' ', ''))
                for fips, name in state_fips_to_name.items()
            ]
            state_fips_df = self.spark.createDataFrame(
                state_fips_rows, ["fips_code", "state_name", "state_key"]
            )

            state_fips_triples = self.triples_df.filter(
                F.col("predicate") == _NWS_HAS_STATE_FIPS
            ).select(
                F.col("subject").alias("geocode"),
                F.col("object").alias("state_fips"),
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

        result = type_triples.unionByName(label_triples)
        if laus_links is not None:
            result = result.unionByName(laus_links)
        if noaa_links is not None:
            result = result.unionByName(noaa_links)

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

        bls_prefixes = [str(ns) for ns in [CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER, WKYENG]]
        market_prefix = str(MARKET)

        # BLS entities in sectors
        bls_filter = F.col("entity").startswith(bls_prefixes[0])
        for p in bls_prefixes[1:]:
            bls_filter = bls_filter | F.col("entity").startswith(p)

        bls_sector = belongs_to_sector.filter(bls_filter).select(
            F.col("entity").alias("bls_entity"), F.col("sector").alias("bls_sector")
        )

        # Market entities in sectors
        market_sector = belongs_to_sector.filter(
            F.col("entity").startswith(market_prefix)
        ).select(
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


def enrich_cross_source(spark: SparkSession, triples_df: DataFrame) -> DataFrame:
    """Main entry point for cross-source enrichment."""
    linker = CrossSourceLinker(spark, triples_df)
    return linker.enrich()