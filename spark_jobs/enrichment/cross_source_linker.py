"""
Cross-Source Enrichment Linker — PySpark Implementation

Links entities across the data sources a run picked. What runs here is generic:
which sources are present, the sector keyword step, the company and region
hubs, the steps that pair named sources, and the measurement types. Each
source's side of them is declared in its spec (spark_jobs/sources/) and lives
in its own intra_source/<source>/cross_source.py.

Runs entirely on Spark executors — no rdflib, no driver data operations.

All methods:
1. Read from self.triples_df (the enriched triples from all intra-source enrichers)
2. Return new triples DataFrames
3. Never call .collect(), .toPandas(), or .toLocalIterator()
"""
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from rdflib.namespace import RDF, RDFS, OWL

from spark_jobs import sources
from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, UNIFIED, SEC_ENRICHMENT, MARKET_ENRICHMENT,
)
from spark_jobs.settle import settle
from spark_jobs.enrichment.intra_source.bls.patterns import (
    BLS_SECTOR_PATTERNS,
)
from spark_jobs.enrichment.intra_source.market.patterns import (
    assert_ciks_are_padded,
)
from spark_jobs.enrichment.region_crosswalk import (
    CENSUS_REGION_CODES,
    STATE_ABBREVIATIONS,
    STATE_TO_CENSUS_REGION,
    US_STATES,
    state_key,
)
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.spark_rdf_utils import deduplicate_against_existing
from typing import Dict, List, Optional, Sequence, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# The state name, abbreviation, FIPS and census-region tables live in
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
UNIFIED_COMPANY_PREFIX = f"{str(UNIFIED)}Company_"
_UNIFIED_COMPANY_TYPE = str(SEC_ENRICHMENT.UnifiedCompany)
_HAS_CIK = str(SEC_ENRICHMENT.hasCik)

# state region -> census region. In BLS_ENRICHMENT because this pipeline infers
# it from a table it maintains; see region_crosswalk.
_WITHIN_CENSUS_REGION = str(BLS_ENRICHMENT.withinCensusRegion)


@dataclass
class CrossSourceContext:
    """What a source's cross-source functions read.

    specs holds the picked sources whose data is present. states is filled in
    before the region hub asks for keys, and symbol_ciks and company_ciks once
    the company hub has run.
    """

    spark: SparkSession
    triples_df: DataFrame
    specs: Tuple[SourceSpec, ...]
    ticker_cik_map: Dict[str, str]
    sub_industries: List[Tuple[str, str]]
    states: Optional[DataFrame] = None
    symbol_ciks: Optional[DataFrame] = None
    company_ciks: Optional[DataFrame] = None

    def entity_prefixes(self, name: str) -> List[str]:
        """The URI prefixes of one present source's entities."""
        spec = next(spec for spec in self.specs if spec.name == name)
        return sources.entity_prefixes(spec.entity_namespaces)


def _union(frames: Sequence[DataFrame]) -> DataFrame:
    result = frames[0]
    for frame in frames[1:]:
        result = result.unionByName(frame)
    return result


class CrossSourceLinker:
    """
    Links entities across the data sources a run picked, using PySpark.

    Strategies:
    1. Sector Linking — keywords, plus each source's own sector links
    2. Company Hub — one node per CIK, from each source's company keys
    3. Region Hub — one node per state and census region, from each source's
       region keys
    4. Steps that pair named sources — each runs when its sources are present
    5. Measurement Type Alignment — from each source's measurement types
    """

    def __init__(
        self,
        spark: SparkSession,
        triples_df: DataFrame,
        ticker_cik_map: Optional[Dict[str, str]] = None,
        sub_industries: Optional[Sequence[Tuple[str, str]]] = None,
        specs: Optional[Sequence[SourceSpec]] = None,
    ):
        self.spark = spark
        self.triples_df = triples_df
        # The sources the run's paths picked. Every registered source when the
        # caller does not say.
        self.specs = sources.REGISTERED if specs is None else tuple(specs)

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

        # The picked sources with data in the frame, in registration order.
        # Only these give keys and run steps.
        self.detected_specs = tuple(
            spec for spec in self.specs if spec.name in self.available_sources
        )
        self._context = CrossSourceContext(
            spark=spark,
            triples_df=triples_df,
            specs=self.detected_specs,
            ticker_cik_map=self._ticker_cik_map,
            sub_industries=self._sub_industries,
        )

    def _detect_sources(self) -> Set[str]:
        """Which picked sources have data in the frame. Reads every row, not a
        sample.

        This used to read the first 100,000 subjects and decide from those.
        Whether a source turned up in that window depended on how the rows
        happened to be laid out, not on whether the source was there. The same
        shape in bls_linker skipped the whole BLS leg on every four-source run
        (#350); here it would skip the steps gated on a source instead.

        A source is present when some subject sits under one of its spec's
        entity_namespaces. Naming the source in one column and taking the
        distinct values reads the data once and collects at most one row per
        source.
        """
        by_source = [
            (spec.name, sources.entity_prefixes(spec.entity_namespaces))
            for spec in self.specs
            if spec.entity_namespaces
        ]
        if not by_source:
            return set()

        def _starts_with_any(prefixes: List[str]) -> Column:
            test = F.col("subject").startswith(prefixes[0])
            for p in prefixes[1:]:
                test = test | F.col("subject").startswith(p)
            return test

        # First match wins, in registration order.
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
        logger.info("\n[Step 1/5] Creating sector links...")
        self._append(new_dfs, self._link_by_sector(), "[Step 1/5] keywords")
        for spec in self.detected_specs:
            if spec.sector_keys is not None:
                self._append(
                    new_dfs, spec.sector_keys(self._context),
                    f"[Step 1/5] {spec.label} sectors",
                )

        logger.info("\n[Step 2/5] Linking by company...")
        self._append(new_dfs, self._link_by_company(), "[Step 2/5]")

        logger.info("\n[Step 3/5] Linking by geographic region...")
        self._append(new_dfs, self._link_by_geography(), "[Step 3/5]")

        # A step that pairs sources names the ones it needs, and runs only when
        # every one of them is present. Nothing else here is gated on a source.
        logger.info("\n[Step 4/5] Linking named pairs of sources...")
        for spec in self.detected_specs:
            for title, needs, step in spec.cross_source_steps:
                missing = [name for name in needs if name not in self.available_sources]
                if missing:
                    logger.warning(
                        f"  [Step 4/5] {title}: skipped, needs "
                        f"{', '.join(missing)}; detected "
                        f"{', '.join(sorted(self.available_sources))}"
                    )
                    continue
                logger.info(f"  {title}...")
                self._append(new_dfs, step(self._context), f"[Step 4/5] {title}")

        logger.info("\n[Step 5/5] Aligning measurement types...")
        self._append(new_dfs, self._align_measurement_types(), "[Step 5/5]")

        if not new_dfs:
            return self.spark.createDataFrame([], schema=self.triples_df.schema)

        all_new = _union(new_dfs)

        # Truncate the plan before the dedup pass (see spark_jobs.settle).
        # Each step above is a join over triples_df, so this union carries many of
        # those subtrees; feeding it straight into dropDuplicates + the anti-join
        # against triples_df makes Catalyst's constraint inference blow the driver
        # heap while *planning* (OutOfMemoryError inside .cache(), before any task
        # runs). Materializing here replaces the subtrees with one scan.
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
    # Step 1: Sector Linking
    # ================================================================

    def _link_by_sector(self) -> Optional[DataFrame]:
        """
        Link entities to economic sectors by the keywords in their URIs.

        Builds a keyword lookup from BLS_SECTOR_PATTERNS, broadcasts it,
        and joins against entity URIs to find sector matches.

        Only the sources whose spec sets sector_keywords are classified this
        way: BLS, market and NOAA. A keyword inside a longer name makes a false
        claim -- Birmingham_AL contains "ham", a food keyword -- so a new
        source stays out unless it opts in.
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

        # The entities of a source that has not opted in are left out.
        #
        # SEC is the one today. Its companies reach a sector through
        # filings:hasSic -- the issuer's registered industry classification --
        # which is a fact about the company rather than a substring of its URI
        # (see SEC's sector_keys). What this classifier was doing to SEC
        # entities was matching BLS keyword fragments against an accession
        # number: a filing's local name is "0001178913-26-003947_Filing", and
        # any sector keyword that happens to appear inside one produces a
        # confident, meaningless membership claim. Keeping both paths would
        # leave the real classification competing with the accidental one on
        # the same predicate.
        opted_out = [
            prefix
            for spec in self.specs
            if not spec.sector_keywords
            for prefix in sources.entity_prefixes(spec.entity_namespaces)
        ]
        if opted_out:
            opted_out_match = F.lit(False)
            for prefix in opted_out:
                opted_out_match = opted_out_match | F.col("subject").startswith(prefix)
            entities = entities.filter(~opted_out_match)

        # The vocabulary this pipeline mints is excluded too, for the same
        # reason and with a worse symptom.
        #
        # This classifier reads a URI's last path segment. The sector nodes are
        # NAMED after the keywords, so they matched themselves: bls:EnergySector
        # has the local name "energysector", which contains "energy", so the
        # graph carried bls:EnergySector belongsToSector bls:EnergySector. On
        # the 2026-08-30 run that was 11 self-loops, plus junk between unrelated
        # economic sectors (bls:ApparelSector -> ManufacturingSector).
        #
        # It also reached the GICS sectors and undid the crosswalk. All three
        # sectors enrichment/sector_crosswalk.py deliberately maps to NOTHING
        # were mapped anyway, under the STRONGER predicate:
        #
        #     market:InformationTechnologySector -> bls:InformationSector
        #     market:CommunicationServicesSector -> bls:InformationSector
        #     market:UtilitiesSector             -> bls:EnergySector
        #
        # which is the chip-maker-to-telephone-price link that module names as
        # the reason those rows are blank, asserted as membership rather than
        # the similarity the curated table states. A GICS sector reaches an
        # economic sector through that table or not at all.
        #
        # Scoped to the two ontology namespaces, not to a "Sector" suffix:
        # id/eci/State_and_local_government_workers_WorkerSector is real source
        # data and must still be classified.
        vocabulary_match = F.lit(False)
        for namespace in (str(BLS_ENRICHMENT), str(MARKET_ENRICHMENT)):
            vocabulary_match = (
                vocabulary_match | F.col("subject").startswith(namespace)
            )
        entities = entities.filter(~vocabulary_match)

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
        # anything reads -- BLS's causal step joins on it, and
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
    # Step 2: Company Hub
    # ================================================================

    def _link_by_company(self) -> Optional[DataFrame]:
        """
        Link every source's company references to ONE unified company node.

        Each source keys a company on the regulator's CIK, directly or through
        a ticker, so a filing and a quote for the same company land on the same
        ``unified:Company_{CIK}`` node that sec_linker._unify_company_entities
        already mints:

            Filing -> Issuer -> Company <- Snapshot

        A source gives its side through its spec's company_keys: entities that
        state a CIK, entities that name a ticker, and the ticker -> CIK pairings
        it knows. A ticker resolves to ONE CIK, the pairing with the lowest
        priority, so a snapshot links to one company rather than to two -- which
        would rebuild, on the ticker side, exactly the split this removes. The
        hub uses whatever keys the present sources give, so a run keeps every
        company link its own sources can make.

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
        """
        entities, symbols, pairings = [], [], []
        for spec in self.detected_specs:
            if spec.company_keys is None:
                continue
            keys = spec.company_keys(self._context)
            if keys.entities is not None:
                entities.append(keys.entities)
            if keys.symbols is not None:
                symbols.append(keys.symbols)
            if keys.symbol_ciks is not None:
                pairings.append(keys.symbol_ciks)

        symbol_ciks = None
        if pairings:
            pairs = _union(pairings).dropDuplicates(["symbol", "cik", "priority"])

            # One CIK per symbol. Ordered by priority then by the CIK itself, so
            # a tie inside one source resolves the same way on every run rather
            # than on whichever row a shuffle happens to deliver first.
            ranked = pairs.withColumn(
                "_rank",
                F.row_number().over(
                    Window.partitionBy("symbol").orderBy(
                        F.col("priority").asc(), F.col("cik").asc()
                    )
                ),
            )
            symbol_ciks = ranked.filter(F.col("_rank") == 1).select("symbol", "cik")
        self._context.symbol_ciks = symbol_ciks

        # (entity, cik, symbol). An INNER join on the ticker, deliberately: an
        # entity whose ticker resolves to no CIK contributes nothing. It does NOT
        # fall back to a symbol-keyed company node, which is exactly what used to
        # mint the second half of the split entity.
        references = []
        if symbols and symbol_ciks is not None:
            references.append(
                _union(symbols)
                .join(symbol_ciks, "symbol", "inner")
                .select("entity", "cik", "symbol")
            )
        if entities:
            references.append(
                _union(entities).select(
                    "entity", "cik", F.lit(None).cast("string").alias("symbol")
                )
            )
        if not references:
            self._context.company_ciks = None
            return None
        companies = _union(references)

        unified_uri = F.concat(F.lit(UNIFIED_COMPANY_PREFIX), F.col("cik"))

        # Scaffolding for exactly the companies something points at, matching
        # how _link_by_geography scopes its region nodes. A typed node with no
        # incident edge is a row of feature tensor describing nothing.
        all_ciks = companies.select("cik").distinct()
        self._context.company_ciks = all_ciks

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
        ticker_label_triples = (
            companies.filter(F.col("symbol").isNotNull())
            .select("cik", "symbol")
            .distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(str(BLS_ENRICHMENT.ticker)).alias("predicate"),
                F.col("symbol").alias("object"),
            )
        )

        links = companies.select(
            F.col("entity").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.refersToCompany)).alias("predicate"),
            unified_uri.alias("object"),
        )

        result = (
            type_triples
            .unionByName(cik_triples)
            .unionByName(ticker_label_triples)
            .unionByName(links)
        )

        logger.info("  Company linking triples prepared (lazy)")
        return result

    # ================================================================
    # Step 3: Region Hub
    # ================================================================

    def _link_by_geography(self) -> Optional[DataFrame]:
        """
        Link entities to unified US state regions, and the states to census
        regions.

        Each present source gives its links through its spec's region_keys:
        BLS's local-area series by the state they name, NOAA's alerts by their
        area descriptions and state FIPS codes. This builds the region nodes
        those links point at, and only those.
        """
        # state_slug is the state name with spaces replaced by the separator
        # the source URIs use, wrapped in it. See bls/cross_source.py,
        # delimited_local_name.
        state_rows = [
            (
                name,
                state_key(name),
                STATE_ABBREVIATIONS[name],
                "_" + name.replace(' ', '_') + "_",
                "_" + state_key(name) + "_",
            )
            for name in US_STATES
        ]
        states_df = self.spark.createDataFrame(
            state_rows,
            schema=[
                "state_name", "state_key", "state_abbr",
                "state_slug", "state_slug_nospace",
            ],
        )
        self._context.states = states_df

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

        links: List[DataFrame] = []
        census_regions: List[DataFrame] = []
        for spec in self.detected_specs:
            if spec.region_keys is None:
                continue
            keys = spec.region_keys(self._context)
            if keys.state_links is not None:
                links.append(keys.state_links)
            if keys.census_regions is not None:
                census_regions.append(keys.census_regions)

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
        if not links:
            logger.info("  No geographic links matched — no region entities minted")
            return None

        all_links = _union(links).dropDuplicates()

        linked_regions = all_links.select(F.col("object").alias("subject")).distinct()

        result = (
            type_triples.join(linked_regions, "subject", "left_semi")
            .unionByName(label_triples.join(linked_regions, "subject", "left_semi"))
            .unionByName(all_links)
        )

        census = self._link_census_regions(linked_regions, census_regions)
        if census is not None:
            result = result.unionByName(census)

        logger.info("  Geographic linking triples prepared (lazy)")
        return result

    def _link_census_regions(
        self,
        linked_state_regions: DataFrame,
        census_regions: Sequence[DataFrame],
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
        one region, not two that resemble each other. The source region entities
        are each source's RegionKeys.census_regions, matched by name.
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

        same_as = None
        if census_regions:
            same_as = _union(census_regions).select(
                F.concat(
                    F.lit(str(UNIFIED)), F.col("census_key"), F.lit("CensusRegion")
                ).alias("subject"),
                F.lit(_OWL_SAME_AS).alias("predicate"),
                F.col("entity").alias("object"),
            )

        # Scaffolding for exactly the census regions something reaches, from
        # either side.
        reached = within.select(F.col("object").alias("subject"))
        if same_as is not None:
            reached = reached.unionByName(same_as.select("subject"))
        reached = reached.distinct()

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

        result = (
            type_triples
            .unionByName(label_triples)
            .unionByName(within)
        )
        if same_as is not None:
            result = result.unionByName(same_as)
        return result

    # ================================================================
    # Step 5: Measurement Type Alignment
    # ================================================================

    def _align_measurement_types(self) -> Optional[DataFrame]:
        """
        Add unified measurement type classifications.
        Maps source-specific types to unified types via broadcast join.

        The rows are each present source's measurement_types, each typed as its
        class_mappings target, so a row is stated once, in the spec.
        """
        type_mappings = [
            (source_type, spec.class_mappings[source_type])
            for spec in self.detected_specs
            for source_type in spec.measurement_types
        ]

        if not type_mappings:
            return None

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
    specs: Optional[Sequence[SourceSpec]] = None,
) -> DataFrame:
    """Main entry point for cross-source enrichment.

    The two reference tables are read from the index-constituents CSV by the
    caller (see EnrichmentPipeline), not here, so this module stays free of S3
    and stays testable with a literal dict. ``specs`` is the sources the run's
    paths picked; every registered source when not given.
    """
    linker = CrossSourceLinker(
        spark, triples_df, ticker_cik_map, sub_industries, specs=specs
    )
    return linker.enrich()
