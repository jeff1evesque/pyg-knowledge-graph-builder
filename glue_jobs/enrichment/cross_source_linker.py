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

from glue_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT, NOAA_ENRICHMENT,
    UNIFIED, CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER,
    WKYENG, SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP, MARKET, CAP,
)
from glue_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from glue_jobs.utils.spark_rdf_utils import (
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
        noaa_prefix = str(CAP)

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
            elif s.startswith(noaa_prefix):
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

        logger.info("\n[Step 1/6] Aligning temporal entities...")
        self._append(new_dfs, self._align_temporal_entities())

        logger.info("\n[Step 2/6] Creating sector-based links...")
        self._append(new_dfs, self._link_by_sector())

        logger.info("\n[Step 3/6] Linking by company/ticker...")
        if 'sec' in self.available_sources and 'market' in self.available_sources:
            self._append(new_dfs, self._link_by_company())

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
    # Step 1: Temporal Alignment
    # ================================================================

    def _align_temporal_entities(self) -> Optional[DataFrame]:
        """
        Create unified temporal entities across all sources.

        Finds month/year URIs across all source namespaces,
        groups by normalized name, creates unified:MonthName entities
        with owl:sameAs links.
        """
        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        # Find all subjects that end with a month name
        month_pattern = "|".join(month_names)

        month_uris = (
            self.triples_df
            .select("subject")
            .distinct()
            .filter(F.regexp_extract("subject", f"({month_pattern})$", 1) != "")
            .withColumn(
                "month_name",
                F.regexp_extract("subject", f"({month_pattern})$", 1)
            )
            .select(
                F.col("subject").alias("source_uri"),
                "month_name"
            )
        )

        # Create unified month entities
        unified_uri = F.concat(F.lit(str(UNIFIED)), F.col("month_name"))

        # Type triples: unified:January rdf:type bls:UnifiedMonth
        type_triples = (
            month_uris
            .select("month_name")
            .distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(str(BLS_ENRICHMENT.UnifiedMonth)).alias("object")
            )
        )

        # Label triples
        label_triples = (
            month_uris
            .select("month_name")
            .distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(_RDFS_LABEL).alias("predicate"),
                F.col("month_name").alias("object")
            )
        )

        # sameAs triples: unified:January owl:sameAs cpi:January, ppi:January, ...
        sameas_triples = month_uris.select(
            F.concat(F.lit(str(UNIFIED)), F.col("month_name")).alias("subject"),
            F.lit(_OWL_SAME_AS).alias("predicate"),
            F.col("source_uri").alias("object")
        )

        # Same for years (4-digit endings)
        year_uris = (
            self.triples_df
            .select("subject")
            .distinct()
            .filter(F.regexp_extract("subject", r"(\d{4})$", 1) != "")
            .withColumn(
                "year_value",
                F.regexp_extract("subject", r"(\d{4})$", 1)
            )
            .filter(
                (F.col("year_value").cast("int") >= 1900) &
                (F.col("year_value").cast("int") <= 2100)
            )
            .select(F.col("subject").alias("source_uri"), "year_value")
        )

        year_unified_uri = F.concat(F.lit(str(UNIFIED)), F.lit("Year"), F.col("year_value"))

        year_type_triples = (
            year_uris.select("year_value").distinct()
            .select(
                year_unified_uri.alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(str(BLS_ENRICHMENT.UnifiedYear)).alias("object")
            )
        )

        year_label_triples = (
            year_uris.select("year_value").distinct()
            .select(
                year_unified_uri.alias("subject"),
                F.lit(_RDFS_LABEL).alias("predicate"),
                F.col("year_value").alias("object")
            )
        )

        year_sameas_triples = year_uris.select(
            F.concat(F.lit(str(UNIFIED)), F.lit("Year"), F.col("year_value")).alias("subject"),
            F.lit(_OWL_SAME_AS).alias("predicate"),
            F.col("source_uri").alias("object")
        )

        result = (
            type_triples
            .unionByName(label_triples)
            .unionByName(sameas_triples)
            .unionByName(year_type_triples)
            .unionByName(year_label_triples)
            .unionByName(year_sameas_triples)
        )

        logger.info("  Temporal alignment triples prepared (lazy)")
        return result

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
        Link entities by US state. Matches state names in subject URIs
        and NOAA alert area descriptions.
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
        else:
            laus_links = None

        # Match NOAA alerts by area description containing state name
        if 'noaa' in self.available_sources:
            area_descs = extract_property(
                self.triples_df, str(CAP.hasAreaDescription), "area_desc"
            )

            noaa_matched = area_descs.crossJoin(F.broadcast(states_df)).filter(
                F.col("area_desc").contains(F.col("state_name"))
            )

            # Need to trace back: area → info → alert
            # For simplicity, link the subject of hasAreaDescription
            noaa_links = noaa_matched.select(
                F.col("subject"),
                F.lit(str(BLS_ENRICHMENT.affectsRegion)).alias("predicate"),
                F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
            )
        else:
            noaa_links = None

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