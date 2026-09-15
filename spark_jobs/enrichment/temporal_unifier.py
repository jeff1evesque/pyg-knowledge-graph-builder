"""
Temporal Entity Unifier (PySpark)

Unifies temporal entities (months, years, quarters) across all data sources.
Creates unified temporal entities and links source-specific temporal entities
to them using owl:sameAs.

Also types the SOURCE temporal URIs it links to (temporal:Source{Month,Year,
Quarter}). Those arrive untyped from the sources, and node_mapper only makes
nodes out of typed URIs — so without this neither the measurement->period edges
nor the sameAs links resolve. See _create_source_temporal_types.

All operations run as distributed PySpark DataFrame transformations.

Example output triples:
    cpi:November      rdf:type      temporal:SourceMonth
    cpi:November      rdfs:label    "November"

    unified:November  rdf:type      bls:UnifiedMonth
    unified:November  rdfs:label    "November"
    unified:November  owl:sameAs    cpi:November
    unified:November  owl:sameAs    ppi:November
    unified:November  owl:sameAs    https://jefflevesque.com/id/temporal/market-quotes/November

    unified:Year2024  rdf:type      bls:UnifiedYear
    unified:Year2024  rdfs:label    "2024"
    unified:Year2024  owl:sameAs    cpi:2024
    unified:Year2024  owl:sameAs    https://jefflevesque.com/id/temporal/sec/2024

    unified:Q1        rdf:type      bls:UnifiedQuarter
    unified:Q1        rdfs:label    "Q1"
    unified:Q1        owl:sameAs    wkyeng:Q1
    unified:Q1        bls:coversMonth  unified:January
    unified:Q1        bls:coversMonth  unified:February
    unified:Q1        bls:coversMonth  unified:March

    unified:Day2026-02-14  rdf:type       bls:UnifiedDay
    unified:Day2026-02-14  rdfs:label     "2026-02-14"
    unified:Day2026-02-14  owl:sameAs     https://jefflevesque.com/id/temporal/sec/2026-02-14
    unified:February       bls:coversDay  unified:Day2026-02-14

THE PERIOD LADDER, AND WHY IT HAS A DAY RUNG
--------------------------------------------
A dated entity attaches to ONE period -- the finest grain its own date supports
-- and the period nodes chain upward: day -> month -> quarter -> year. Nothing
attaches to two grains at once.

That rule is the fix for a hub. observedInPeriod used to link every dated entity
to its month AND its year, so both nodes accumulated an edge per entity in the
whole build; the month became the graph's largest hub by a wide margin (3,867 of
the cross-source edges over two node types, 97% from one source family), and a
hub at that degree oversquashes -- aggregating over it mixes every source in the
period into one representation.

Monthly is the correct grain for the BLS feeds, which publish monthly and arrive
already stating `obs hasMonth cpi:February`. It was the wrong grain for the
three sources that are intraday or event-timed -- market quotes, SEC filings and
weather alerts -- which are exactly the sources that reach
_collect_date_based_temporals, so the routing needs no source list.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from functools import reduce
from typing import List, Optional, Sequence

from spark_jobs import sources
from spark_jobs.settle import settle
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT, UNIFIED, SOURCE_TEMPORAL

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"

UNIFIED_MONTH_TYPE = str(BLS_ENRICHMENT.UnifiedMonth)
UNIFIED_YEAR_TYPE = str(BLS_ENRICHMENT.UnifiedYear)
UNIFIED_QUARTER_TYPE = str(BLS_ENRICHMENT.UnifiedQuarter)
UNIFIED_DAY_TYPE = str(BLS_ENRICHMENT.UnifiedDay)
COVERS_MONTH_PRED = str(BLS_ENRICHMENT.coversMonth)

# Month -> day, the rung this work adds to the period ladder.
#
# WHY A DAY GRAIN EXISTS AT ALL. observedInPeriod used to attach every dated
# entity from every source directly to a month node, and the month nodes were
# the graph's largest hub by a wide margin -- 3,867 of the cross-source edges on
# two node types, 97% of them from one source family. A hub at that degree
# OVERSQUASHES: aggregating over it mixes every source in the period into one
# representation, and "same month" is not a relationship a model can learn
# anything from.
#
# Monthly is the RIGHT grain for the economic indicator feeds, whose source data
# is published monthly and which arrive already stating `obs hasMonth
# cpi:February`. It is the wrong grain for the three sources that are intraday
# or event-timed -- market quotes, SEC filings, weather alerts -- which were
# being flattened into it. All three already carry day-resolution timestamps, so
# this needs nothing upstream.
#
# The day hangs off the month rather than replacing it, so nothing loses
# reachability: a quote still reaches February, now as
# Snapshot -> Day -> February rather than Snapshot -> February. What changes is
# the month's in-degree, which drops by roughly the number of days in it.
COVERS_DAY_PRED = str(BLS_ENRICHMENT.coversDay)

# The on-ramp from a dated entity to the period it falls in.
#
# The date-literal sources (SEC, NOAA, market quotes) state a date and nothing
# else -- unlike the BLS datasets, whose source data already says
# `obs hasMonth cpi:February` and so arrives on the spine under its own power.
# For those three the unifier minted temporal/{source}/February and linked it
# sameAs unified:February, and stopped: nothing pointed AT the period, because
# the collector read the date value and discarded the subject it belonged to.
# The result was a two-node island per period per source -- a bridge with no
# on-ramp -- while the graph still reported unified temporal entities and
# looked connected.
#
# In BLS_ENRICHMENT rather than SOURCE_TEMPORAL because classify_edge_origin
# reads pipeline provenance off the predicate's namespace, and SOURCE_TEMPORAL
# is not one of ENRICHMENT_NAMESPACES -- minting it there would report these
# inferred links as raw source facts. coversMonth above is in the same
# namespace for the same reason and is likewise emitted for every source.
OBSERVED_IN_PERIOD_PRED = str(BLS_ENRICHMENT.observedInPeriod)

# Types for the source-side temporal URIs the unifier links TO — see the
# SOURCE_TEMPORAL note in rdf_utils for why these are their own namespace
# rather than reusing the Unified* types.
SOURCE_MONTH_TYPE = str(SOURCE_TEMPORAL.SourceMonth)
SOURCE_YEAR_TYPE = str(SOURCE_TEMPORAL.SourceYear)
SOURCE_QUARTER_TYPE = str(SOURCE_TEMPORAL.SourceQuarter)
SOURCE_DAY_TYPE = str(SOURCE_TEMPORAL.SourceDay)

SOURCE_TEMPORAL_TYPE_BY_KIND = {
    "month": SOURCE_MONTH_TYPE,
    "year": SOURCE_YEAR_TYPE,
    "quarter": SOURCE_QUARTER_TYPE,
    "day": SOURCE_DAY_TYPE,
}

# The grains an entity may be attached to, finest first.
#
# observedInPeriod links each dated entity to the FINEST grain that entity's own
# date supports -- see enrich(). Attaching to every grain is what made the month
# a hub: an entity that already reaches February through a day does not need a
# second edge saying February, and the second edge is the expensive one.
PERIOD_GRAINS_FINEST_FIRST = ("day", "month", "year")

UNIFIED_BASE = str(UNIFIED)

# Valid month names for regex matching
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Month number → name mapping for date parsing
MONTH_NUM_TO_NAME = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}

# Quarter → months mapping
QUARTER_MONTH_MAP = {
    "Q1": ["January", "February", "March"],
    "Q2": ["April", "May", "June"],
    "Q3": ["July", "August", "September"],
    "Q4": ["October", "November", "December"],
}


class TemporalUnifier:
    """
    Unifies temporal entities across the picked data sources using PySpark.

    Strategies:
    1. Collect the periods a source states as URIs, with its spec's
       temporal_collector (BLS: explicit month/year/quarter entities)
    2. Extract day/month/year from the date literals a source states under
       its spec's date_predicates → synthetic temporal URIs under its
       temporal_prefix (SEC, market, NOAA)
    3. Group all temporal URIs by normalized name
    4. Produce unified entities with owl:sameAs links
    """

    def __init__(
        self,
        spark: SparkSession,
        specs: Optional[Sequence[SourceSpec]] = None,
    ):
        self.spark = spark
        # The sources the run's paths picked. Every registered source when the
        # caller does not say; a source finds nothing in data it does not have.
        self.specs = sources.REGISTERED if specs is None else tuple(specs)

    def enrich(self, triples_df: DataFrame) -> DataFrame:
        """
        Run temporal unification across the picked sources.

        Args:
            triples_df: DataFrame with columns (subject, predicate, object)

        Returns:
            DataFrame of NEW triples (unified temporal entities + sameAs links)
        """
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        logger.info("=" * 60)
        logger.info("Starting Temporal Unification (PySpark)")
        logger.info("=" * 60)

        triples_df.cache()

        # Collect (temporal_uri, normalized_name, kind) from the picked sources.
        # kind is "day", "month", "year", or "quarter".
        temporal_dfs: List[DataFrame] = []

        # The date-literal sources, kept as (subject, temporal_uri, ...) frames.
        # The subject is dropped for the union below -- which only wants the
        # distinct periods -- but survives here so enrich() can also emit the
        # link from each dated entity to its period.
        dated_dfs: List[DataFrame] = []

        for spec in self.specs:
            if spec.temporal_collector is None and not spec.date_predicates:
                continue
            logger.info(f"  Collecting {spec.label} temporal entities...")
            if spec.temporal_collector is not None:
                temporal_dfs.extend(spec.temporal_collector(triples_df))
            if spec.date_predicates:
                df = self._collect_date_based_temporals(
                    triples_df, list(spec.date_predicates), spec.temporal_prefix
                )
                if df is not None:
                    dated_dfs.append(df)

        # Settle the date-literal frames once. Each is a scan of triples_df and
        # each feeds BOTH the period union below and the period links further
        # down, so without this their subtrees are planned twice over -- the
        # same Catalyst blow-up the all_temporals checkpoint exists to avoid.
        dated: Optional[DataFrame] = None
        if dated_dfs:
            dated = settle(
                reduce(DataFrame.unionAll, dated_dfs).dropDuplicates(
                    ["subject", "temporal_uri", "normalized_name", "kind"]
                )
            )
            temporal_dfs.append(
                dated.select("temporal_uri", "normalized_name", "kind")
            )

        if not temporal_dfs:
            logger.info("No temporal entities found in any source")
            return empty

        # Union all temporal entities: (temporal_uri, normalized_name, kind)
        #
        # Truncate the plan here (see spark_jobs.settle). all_temporals is a
        # union of every collector above -- each one a scan of triples_df -- and the
        # three _create_unified_* branches below each build on it, so without this its
        # whole subtree is re-planned into all three and Catalyst's constraint
        # inference blows the driver heap while *planning* (OutOfMemoryError inside
        # .cache(), before a single task runs). It survives a 1-source run and OOMs a
        # 7-source one. Eager, so it is materialized once and shared by the branches.
        all_temporals = settle(
            reduce(DataFrame.unionAll, temporal_dfs).dropDuplicates(
                ["temporal_uri", "normalized_name", "kind"]
            )
        )

        # Produce unified triples
        new_dfs: List[DataFrame] = []

        new_dfs.append(self._create_source_temporal_types(all_temporals))

        if dated is not None:
            # The on-ramp: each entity that stated a date -> the period it
            # names, at the FINEST grain that entity's date supports.
            #
            # This used to emit every grain the collectors produced, so one
            # filing attached to both its month and its year, and every dated
            # entity in the build piled onto the same two nodes. That made the
            # month the graph's largest hub -- 3,867 cross-source edges across
            # two node types -- and a hub at that degree oversquashes: a GNN
            # aggregating over it mixes every source in the period into one
            # representation, which is why "same month" taught a model nothing.
            #
            # Coarser periods are NOT lost, they are reached through the finer
            # one: Snapshot -> Day -> February -> 2026, via the coversDay and
            # coversMonth links below. Same reachability, one more hop, roughly
            # a 30x smaller fan-in on the month.
            new_dfs.append(self._finest_grain_links(dated))

        df = self._create_unified_months(all_temporals)
        if df is not None:
            new_dfs.append(df)

        df = self._create_unified_years(all_temporals)
        if df is not None:
            new_dfs.append(df)

        df = self._create_unified_quarters(all_temporals)
        if df is not None:
            new_dfs.append(df)

        df = self._create_unified_days(all_temporals)
        if df is not None:
            new_dfs.append(df)

        if not new_dfs:
            logger.info("No unified temporal triples produced")
            return empty

        result = reduce(DataFrame.unionAll, new_dfs).cache()
        count = result.count()

        logger.info("=" * 60)
        logger.info(f"Temporal Unification Complete: {count} triples")
        logger.info("=" * 60)

        return result

    def _finest_grain_links(self, dated: DataFrame) -> DataFrame:
        """observedInPeriod, one grain per entity: the finest it supports.

        `dated` carries a row per (subject, grain), so a filing appears three
        times -- day, month, year. Emitting all three attaches one entity to
        three period nodes and inflates the coarse ones by the number of
        entities rather than by the number of finer periods.

        Ranked per subject rather than filtered to "day", because the rule has
        to hold for an entity whose date has no day in it: a value of "2024-11"
        yields no day row and must still reach its month. A hardcoded grain
        would silently orphan those.
        """
        rank = F.lit(len(PERIOD_GRAINS_FINEST_FIRST))
        for position, grain in enumerate(PERIOD_GRAINS_FINEST_FIRST):
            rank = F.when(F.col("kind") == grain, F.lit(position)).otherwise(rank)

        ranked = dated.withColumn("_grain_rank", rank)

        finest = ranked.groupBy("subject").agg(
            F.min("_grain_rank").alias("_grain_rank")
        )

        return (
            ranked.join(finest, ["subject", "_grain_rank"], "inner")
            .select(
                F.col("subject"),
                F.lit(OBSERVED_IN_PERIOD_PRED).alias("predicate"),
                F.col("temporal_uri").alias("object"),
            )
            .dropDuplicates()
        )

    # ================================================================
    # Date literals → synthetic temporal URIs
    # ================================================================

    def _collect_date_based_temporals(
        self,
        triples_df: DataFrame,
        date_predicates: List[str],
        synthetic_prefix: str,
    ) -> Optional[DataFrame]:
        """
        Extract month/year from date literal values and create synthetic
        temporal URIs.

        Works for any source that stores dates as literal values
        (SEC filing dates, NOAA alert timestamps, etc.)

        For NOAA: In the updated RML mapper, cap:hasSentTime and other
        date properties are on the Info subject (alert:{id}#info), not
        the Alert subject. This method keys only on the predicate, so it works
        regardless of which subject the date property appears on -- the entity
        it links to the period is whichever subject stated the date.

        The SUBJECT is carried through, not just the date value. It is what
        OBSERVED_IN_PERIOD_PRED links to the period node in enrich(); without it
        the synthetic period is an island. See that constant for the history.

        Some sources state the date as a URI rather than a literal -- SEC points
        hasFilingDate at sec-common:Date_2026-08-14 -- and the regexes below
        read the date out of either form, so both shapes land on the spine. The
        entity linked to the period is the one that STATED the date (the
        filing), not the date node, so the period sits one hop from the filing
        rather than two.

        Args:
            triples_df: The triples DataFrame
            date_predicates: List of predicate URIs that hold date values
            synthetic_prefix: URI prefix for synthetic temporal entities
                              e.g., "https://jefflevesque.com/id/temporal/sec/"

        Produces (subject, temporal_uri, normalized_name, kind) rows like:
            (filing, ".../temporal/sec/November", "November", "month")
            (filing, ".../temporal/sec/2024", "2024", "year")
        """
        # Filter to triples with the relevant date predicates
        date_triples = triples_df.filter(
            F.col("predicate").isin(date_predicates)
        ).select(
            F.col("subject"),
            F.col("object").alias("date_value"),
        )

        if date_triples.head(1) == []:
            return None

        date_triples = date_triples.dropDuplicates()

        # Extract day, month number and year from date strings.
        # Handles ISO formats: "2024-11-15", "2024-11-15T10:30:00", etc.
        # Uses substring extraction — no UDF needed.
        parsed = date_triples.withColumn(
            "year_str", F.regexp_extract(F.col("date_value"), r"(\d{4})", 1)
        ).withColumn(
            "month_str", F.regexp_extract(F.col("date_value"), r"\d{4}-(\d{2})", 1)
        ).withColumn(
            # The day is OPTIONAL, unlike the other two -- a GUARD rather than
            # an observation. Every source reaching this method states a full
            # YYYY-MM-DD today, in every committed fixture. But nothing
            # upstream PREVENTS a partial date (hasPeriodOfReport is the
            # plausible one: a quarterly filing reports on a period, not a
            # day), and if the day grain were mandatory such a value would
            # yield no day row, take the month row with it in _finest_grain_
            # links, and drop the entity off the temporal spine entirely with
            # nothing raised. Two characters of regex keep that from being
            # silent.
            "day_str",
            F.regexp_extract(F.col("date_value"), r"(\d{4}-\d{2}-\d{2})", 1),
        ).filter(
            (F.col("year_str") != "") & (F.col("month_str") != "")
        )

        if parsed.head(1) == []:
            return None

        # Map month number to month name using a broadcast join
        month_mapping = self.spark.createDataFrame(
            list(MONTH_NUM_TO_NAME.items()),
            ["month_str", "month_name"]
        )

        parsed = parsed.join(F.broadcast(month_mapping), "month_str", "inner")

        # Produce month rows
        months = parsed.select(
            F.col("subject"),
            F.concat(F.lit(synthetic_prefix), F.col("month_name")).alias("temporal_uri"),
            F.col("month_name").alias("normalized_name"),
            F.lit("month").alias("kind"),
        ).dropDuplicates()

        # Produce year rows
        years = parsed.select(
            F.col("subject"),
            F.concat(F.lit(synthetic_prefix), F.col("year_str")).alias("temporal_uri"),
            F.col("year_str").alias("normalized_name"),
            F.lit("year").alias("kind"),
        ).dropDuplicates()

        # Produce day rows.
        #
        # Only the sources that reach THIS method get them, and that is the
        # whole routing rule -- no source list to keep in sync. The three
        # date-literal sources (SEC filings, market quotes, NOAA alerts) are
        # exactly the sub-monthly ones, because a source that publishes monthly
        # states its period as a URI and arrives through its spec's
        # temporal_collector instead (BLS's collect_bls_months_years), which
        # emits no day at all. The BLS feeds therefore stay on the month grain
        # by construction rather than by exclusion.
        days = parsed.filter(F.col("day_str") != "").select(
            F.col("subject"),
            F.concat(F.lit(synthetic_prefix), F.col("day_str")).alias("temporal_uri"),
            F.col("day_str").alias("normalized_name"),
            F.lit("day").alias("kind"),
        ).dropDuplicates()

        return months.unionAll(years).unionAll(days)

    # ================================================================
    # Produce unified entities
    # ================================================================

    def _create_source_temporal_types(
        self, all_temporals: DataFrame
    ) -> DataFrame:
        """
        Type the SOURCE temporal URIs so they can become graph nodes.

        Source data references periods as bare URIs — cpi:February, eci:2024,
        jolts:August — and none of them carries an rdf:type. node_mapper only
        creates nodes for typed URIs, so they were never nodes, and every triple
        pointing at them (hasMonth / hasYear / hasStartMonth / hasEndMonth /
        hasStartYear / hasEndYear — ~1,205 on the e2e fixtures) was silently
        dropped during edge resolution. The graph had no temporal dimension: no
        measurement was connected to when it happened.

        The unifier's own owl:sameAs links pointed at the same untyped URIs, so
        they were dropped too, leaving the Unified{Month,Year} nodes as isolated
        vertices a GNN can propagate nothing through.

        This method emits the missing rdf:type (plus a label) for exactly the
        set of temporal URIs the unifier already collects — the same set it
        links to — which completes the intended bridge:

            cpi obs -> cpi:February -> unified:February <- eci:February <- eci obs

        Both hops are edges only once BOTH endpoints are typed.

        Produces, per source temporal URI:
            cpi:February  rdf:type    temporal:SourceMonth
            cpi:February  rdfs:label  "February"

        Note the synthetic URIs minted by the date-literal collectors
        (sec/noaa/market temporal/{Month}) are typed here too — they are equally
        untyped otherwise, and their only edges are the sameAs links.
        """
        distinct = all_temporals.select(
            "temporal_uri", "normalized_name", "kind"
        ).dropDuplicates()

        type_expr = None
        for kind, type_uri in SOURCE_TEMPORAL_TYPE_BY_KIND.items():
            cond = F.col("kind") == kind
            type_expr = (
                F.when(cond, F.lit(type_uri)) if type_expr is None
                else type_expr.when(cond, F.lit(type_uri))
            )

        typed = distinct.withColumn("type_uri", type_expr.otherwise(F.lit(None)))
        # An unrecognized kind would otherwise emit `<uri> rdf:type NULL`, which
        # node_mapper would turn into a garbage node type rather than failing.
        typed = typed.filter(F.col("type_uri").isNotNull())

        type_triples = typed.select(
            F.col("temporal_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.col("type_uri").alias("object"),
        )

        label_triples = typed.select(
            F.col("temporal_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        return type_triples.unionAll(label_triples)

    def _create_unified_months(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified month entities and owl:sameAs links.

        For each unique month name, produces:
          unified:{MonthName}  rdf:type    bls:UnifiedMonth
          unified:{MonthName}  rdfs:label  "{MonthName}"
          unified:{MonthName}  owl:sameAs  <source_temporal_uri_1>
          unified:{MonthName}  owl:sameAs  <source_temporal_uri_2>
          ...
        """
        months = all_temporals.filter(F.col("kind") == "month")

        if months.head(1) == []:
            return None

        # Build unified URI
        months = months.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.col("normalized_name"))
        )

        # Type triples (one per unique month name)
        distinct_months = months.select("unified_uri", "normalized_name").dropDuplicates()

        type_triples = distinct_months.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_MONTH_TYPE).alias("object"),
        )

        label_triples = distinct_months.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        # sameAs triples (one per source temporal URI)
        same_as_triples = months.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        return type_triples.unionAll(label_triples).unionAll(same_as_triples)

    def _create_unified_years(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified year entities and owl:sameAs links.

        For each unique year value, produces:
          unified:Year{YYYY}  rdf:type    bls:UnifiedYear
          unified:Year{YYYY}  rdfs:label  "{YYYY}"
          unified:Year{YYYY}  owl:sameAs  <source_temporal_uri_1>
          ...
        """
        years = all_temporals.filter(F.col("kind") == "year")

        if years.head(1) == []:
            return None

        years = years.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.lit("Year"), F.col("normalized_name"))
        )

        distinct_years = years.select("unified_uri", "normalized_name").dropDuplicates()

        type_triples = distinct_years.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_YEAR_TYPE).alias("object"),
        )

        label_triples = distinct_years.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        same_as_triples = years.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        return type_triples.unionAll(label_triples).unionAll(same_as_triples)

    def _create_unified_quarters(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified quarter entities, owl:sameAs links, and
        coversMonth links to unified months.

        For each unique quarter label, produces:
          unified:{Q1}  rdf:type           bls:UnifiedQuarter
          unified:{Q1}  rdfs:label         "Q1"
          unified:{Q1}  owl:sameAs         <source_quarter_uri>
          unified:{Q1}  bls:coversMonth    unified:January
          unified:{Q1}  bls:coversMonth    unified:February
          unified:{Q1}  bls:coversMonth    unified:March
        """
        quarters = all_temporals.filter(F.col("kind") == "quarter")

        if quarters.head(1) == []:
            return None

        quarters = quarters.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.col("normalized_name"))
        )

        distinct_quarters = quarters.select(
            "unified_uri", "normalized_name"
        ).dropDuplicates()

        type_triples = distinct_quarters.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_QUARTER_TYPE).alias("object"),
        )

        label_triples = distinct_quarters.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        same_as_triples = quarters.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        # coversMonth links: quarter → constituent months
        # Build a small DataFrame of (quarter_label, month_name) pairs
        covers_rows = []
        for q_label, month_names in QUARTER_MONTH_MAP.items():
            for m_name in month_names:
                covers_rows.append((q_label, m_name))

        covers_df = self.spark.createDataFrame(
            covers_rows, ["quarter_label", "month_name"]
        )

        # Join to get (unified_quarter_uri, unified_month_uri)
        covers_joined = distinct_quarters.join(
            F.broadcast(covers_df),
            distinct_quarters.normalized_name == covers_df.quarter_label,
            "inner",
        )

        covers_triples = covers_joined.select(
            F.col("unified_uri").alias("subject"),
            F.lit(COVERS_MONTH_PRED).alias("predicate"),
            F.concat(F.lit(UNIFIED_BASE), F.col("month_name")).alias("object"),
        )

        return (
            type_triples
            .unionAll(label_triples)
            .unionAll(same_as_triples)
            .unionAll(covers_triples)
        )


    def _create_unified_days(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified day entities, owl:sameAs links, and the coversDay link
        from the unified month each day falls in.

        For each distinct date, produces:
          unified:Day2026-02-14  rdf:type         bls:UnifiedDay
          unified:Day2026-02-14  rdfs:label       "2026-02-14"
          unified:Day2026-02-14  owl:sameAs       <source_day_uri>
          unified:February       bls:coversDay    unified:Day2026-02-14

        The coversDay link is what keeps the month reachable. A quote no longer
        points at February directly -- it points at the day, and the day sits
        under February -- so the month's in-degree falls from "every dated
        entity in the build" to "the days that occurred", while every path that
        existed before still exists one hop longer.

        The day is genuinely cross-source, which is the point: 2026-02-14 is
        the same day for a filing, a quote and a weather alert, so the unified
        day is a bridge in its own right at roughly a thirtieth of the fan-in.
        Unlike the month, it is also a period over which those three sources
        can plausibly be RELATED -- a filing and the market's reaction to it
        share a day, and share a month only incidentally.
        """
        days = all_temporals.filter(F.col("kind") == "day")

        if days.head(1) == []:
            return None

        # unified:Day{YYYY-MM-DD}. Prefixed with "Day" for the same reason
        # years are prefixed with "Year": a bare local name that starts with a
        # digit is a fragile URI, and the prefix keeps the three unified period
        # vocabularies visibly distinct.
        days = days.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.lit("Day"), F.col("normalized_name")),
        )

        distinct_days = days.select("unified_uri", "normalized_name").dropDuplicates()

        type_triples = distinct_days.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_DAY_TYPE).alias("object"),
        )

        label_triples = distinct_days.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        same_as_triples = days.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        # month -> day, via the month NUMBER carried in the date itself. No
        # join back to the collected months is needed and none is wanted: the
        # unified month URI is a pure function of the date, and deriving it
        # here keeps this correct for a month that only a day ever mentioned.
        month_mapping = self.spark.createDataFrame(
            list(MONTH_NUM_TO_NAME.items()), ["month_str", "month_name"]
        )

        covers_triples = (
            distinct_days
            .withColumn(
                "month_str",
                F.regexp_extract(F.col("normalized_name"), r"^\d{4}-(\d{2})", 1),
            )
            .join(F.broadcast(month_mapping), "month_str", "inner")
            .select(
                F.concat(F.lit(UNIFIED_BASE), F.col("month_name")).alias("subject"),
                F.lit(COVERS_DAY_PRED).alias("predicate"),
                F.col("unified_uri").alias("object"),
            )
        )

        return (
            type_triples
            .unionAll(label_triples)
            .unionAll(same_as_triples)
            .unionAll(covers_triples)
        )


def unify_temporal_entities(
    spark: SparkSession, triples_df: DataFrame
) -> DataFrame:
    """
    Convenience function for temporal unification.

    Args:
        spark: SparkSession
        triples_df: DataFrame with (subject, predicate, object)

    Returns:
        DataFrame of new temporal unification triples
    """
    unifier = TemporalUnifier(spark)
    return unifier.enrich(triples_df)
