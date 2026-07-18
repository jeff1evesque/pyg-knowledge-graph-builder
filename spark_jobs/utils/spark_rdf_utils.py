"""
PySpark utilities for RDF triple DataFrames.

Canonical schema: (subject: string, predicate: string, object: string)

Bridge strategy:
- rdflib-based enrichers (BLS, SEC) mutate the rdflib graph on the driver
- After they finish, we diff the graph to find new triples
- Serialize the diff to N-Triples and parallelize into Spark
- This is a ONE-TIME bounded transfer, not a streaming bridge
- Cross-source enrichment runs entirely on PySpark executors
- As BLS/SEC migrate to PySpark, the diff shrinks to zero
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from rdflib import Graph
from rdflib.namespace import RDF
from typing import Optional, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)

TRIPLES_SCHEMA = StructType([
    StructField("subject", StringType(), nullable=False),
    StructField("predicate", StringType(), nullable=False),
    StructField("object", StringType(), nullable=False),
])


def collect_sorted(df: DataFrame) -> List:
    """``collect()`` with a deterministic row order.

    Spark returns collected rows in task-completion order, which varies between
    runs. Anything built from a bare ``collect()`` inherits that: metadata list
    entries drift run-to-run, so outputs are never byte-reproducible.

    Worse than ordering, callers that fold rows into a dict keeping the FIRST
    value per key turn this into a *content* non-determinism — a different value
    can win on each run. Both patterns caused real reproducibility bugs (see the
    metadata builders in feature_extractor.py and the node_type -> type_uri map
    in node_mapper.py).

    Sorting by every column stringified is deterministic regardless of schema,
    and callers use this on small driver-side frames (hundreds to a few thousand
    rows), not in hot paths.
    """
    return sorted(df.collect(), key=lambda row: tuple(str(v) for v in row))


# ============================================
# LOADING: S3 → Spark DataFrame (fully distributed)
# ============================================

def load_ntriples_to_triples_df(spark: SparkSession, s3_paths: List[str]) -> DataFrame:
    """
    Load N-Triples files directly into Spark. Fully distributed —
    each executor reads and parses its partition independently.
    """
    logger.info(f"Loading {len(s3_paths)} N-Triples files into Spark DataFrame...")

    raw = spark.read.text(s3_paths)

    lines = (
        raw
        .filter(F.col("value").isNotNull())
        .filter(~F.col("value").startswith("#"))
        .filter(F.length(F.trim(F.col("value"))) > 0)
    )

    parsed = lines.select(
        F.regexp_extract("value", r"^<([^>]+)>", 1).alias("subject"),
        F.regexp_extract("value", r"^<[^>]+>\s+<([^>]+)>", 1).alias("predicate"),
        F.regexp_extract("value", r"^<[^>]+>\s+<[^>]+>\s+<([^>]+)>", 1).alias("object_uri"),
        F.regexp_extract("value", r'^<[^>]+>\s+<[^>]+>\s+"([^"]*)"', 1).alias("object_literal"),
    )

    triples_df = (
        parsed
        .withColumn(
            "object",
            F.when(F.length(F.col("object_uri")) > 0, F.col("object_uri"))
             .otherwise(F.col("object_literal"))
        )
        .filter(
            (F.length(F.col("subject")) > 0) &
            (F.length(F.col("predicate")) > 0) &
            (F.length(F.col("object")) > 0)
        )
        .select("subject", "predicate", "object")
    )

    logger.info("N-Triples loaded into distributed DataFrame")
    return triples_df


def load_turtle_to_triples_df(spark: SparkSession, rdflib_graph: Graph) -> DataFrame:
    """
    Fallback for non-N-Triples formats. Serializes rdflib graph to NT
    on driver, then parallelizes across executors.
    """
    logger.info(f"Converting rdflib graph ({len(rdflib_graph)} triples) to Spark DataFrame...")

    nt_data = rdflib_graph.serialize(format="nt")
    lines = [line for line in nt_data.strip().split("\n") if line.strip() and not line.startswith("#")]

    num_partitions = max(1, len(lines) // 10000)
    rdd = spark.sparkContext.parallelize(lines, numSlices=num_partitions)
    raw = rdd.toDF(schema=StructType([StructField("value", StringType(), nullable=False)]))

    parsed = raw.select(
        F.regexp_extract("value", r"^<([^>]+)>", 1).alias("subject"),
        F.regexp_extract("value", r"^<[^>]+>\s+<([^>]+)>", 1).alias("predicate"),
        F.regexp_extract("value", r"^<[^>]+>\s+<[^>]+>\s+<([^>]+)>", 1).alias("object_uri"),
        F.regexp_extract("value", r'^<[^>]+>\s+<[^>]+>\s+"([^"]*)"', 1).alias("object_literal"),
    )

    triples_df = (
        parsed
        .withColumn(
            "object",
            F.when(F.length(F.col("object_uri")) > 0, F.col("object_uri"))
             .otherwise(F.col("object_literal"))
        )
        .filter(
            (F.length(F.col("subject")) > 0) &
            (F.length(F.col("predicate")) > 0) &
            (F.length(F.col("object")) > 0)
        )
        .select("subject", "predicate", "object")
    )

    logger.info("Converted rdflib graph to distributed DataFrame")
    return triples_df


# ============================================
# BRIDGE: rdflib enrichment diff → Spark DataFrame
# (One-time bounded transfer after rdflib enrichers finish)
# ============================================

def snapshot_rdflib_graph(graph: Graph) -> Set[Tuple[str, str, str]]:
    """
    Take a snapshot of all triples in an rdflib graph as a set of
    (subject, predicate, object) string tuples.

    Called BEFORE rdflib enrichers run to establish a baseline.
    The set is held in driver memory — bounded by the raw graph size
    which is already in driver memory via rdflib anyway.

    Args:
        graph: rdflib Graph

    Returns:
        Set of (s, p, o) string tuples
    """
    return {(str(s), str(p), str(o)) for s, p, o in graph}


def rdflib_diff_to_triples_df(
    spark: SparkSession,
    graph: Graph,
    before_snapshot: Set[Tuple[str, str, str]]
) -> Optional[DataFrame]:
    """
    Compute the diff between current rdflib graph and a previous snapshot,
    and return the new triples as a Spark DataFrame.

    This is how rdflib-based enricher output gets into the Spark world:
    1. Snapshot graph before enrichers run
    2. Run rdflib enrichers (they mutate the graph)
    3. Diff to find new triples
    4. Serialize diff to NT lines
    5. Parallelize into Spark DataFrame

    The diff is bounded by the number of triples added by rdflib enrichers
    (typically 50K-500K), NOT the full graph size.

    Args:
        spark: SparkSession
        graph: rdflib Graph (after enrichment)
        before_snapshot: Snapshot taken before enrichment

    Returns:
        DataFrame of new triples, or None if no new triples
    """
    current = {(str(s), str(p), str(o)) for s, p, o in graph}
    new_triples = current - before_snapshot

    if not new_triples:
        logger.info("  No new rdflib triples to transfer to Spark")
        return None

    logger.info(f"  Transferring {len(new_triples)} new rdflib triples to Spark DataFrame...")

    # Convert to NT lines for parsing consistency
    nt_lines = []
    for s, p, o in new_triples:
        if o.startswith("http://") or o.startswith("https://"):
            nt_lines.append(f"<{s}> <{p}> <{o}> .")
        else:
            # Escape quotes in literals
            escaped = o.replace('\\', '\\\\').replace('"', '\\"')
            nt_lines.append(f'<{s}> <{p}> "{escaped}" .')

    num_partitions = max(1, len(nt_lines) // 10000)
    rdd = spark.sparkContext.parallelize(nt_lines, numSlices=num_partitions)
    raw = rdd.toDF(schema=StructType([StructField("value", StringType(), nullable=False)]))

    parsed = raw.select(
        F.regexp_extract("value", r"^<([^>]+)>", 1).alias("subject"),
        F.regexp_extract("value", r"^<[^>]+>\s+<([^>]+)>", 1).alias("predicate"),
        F.regexp_extract("value", r"^<[^>]+>\s+<[^>]+>\s+<([^>]+)>", 1).alias("object_uri"),
        F.regexp_extract("value", r'^<[^>]+>\s+<[^>]+>\s+"([^"]*)"', 1).alias("object_literal"),
    )

    triples_df = (
        parsed
        .withColumn(
            "object",
            F.when(F.length(F.col("object_uri")) > 0, F.col("object_uri"))
             .otherwise(F.col("object_literal"))
        )
        .filter(
            (F.length(F.col("subject")) > 0) &
            (F.length(F.col("predicate")) > 0) &
            (F.length(F.col("object")) > 0)
        )
        .select("subject", "predicate", "object")
    )

    logger.info(f"  Transferred {len(new_triples)} rdflib triples to Spark")
    return triples_df


# ============================================
# ENRICHMENT HELPERS: DataFrame operations
# ============================================

def extract_entities_by_type(triples_df: DataFrame, rdf_type: str) -> DataFrame:
    """Extract all subjects that have rdf:type = rdf_type."""
    return (
        triples_df
        .filter(
            (F.col("predicate") == str(RDF.type)) &
            (F.col("object") == rdf_type)
        )
        .select(F.col("subject").alias("entity"))
        .distinct()
    )


def extract_property(triples_df: DataFrame, predicate: str, alias: str = "value") -> DataFrame:
    """Extract (subject, object) pairs for a given predicate."""
    return (
        triples_df
        .filter(F.col("predicate") == predicate)
        .select(F.col("subject"), F.col("object").alias(alias))
    )


def build_entity_properties(
    triples_df: DataFrame,
    entity_type: str,
    properties: dict
) -> DataFrame:
    """
    Build a pivoted entity DataFrame by joining multiple properties.
    Replaces SPARQL SELECT with multiple property patterns.
    """
    entities = extract_entities_by_type(triples_df, entity_type)
    result = entities
    for prop_uri, alias in properties.items():
        prop_df = (
            triples_df
            .filter(F.col("predicate") == prop_uri)
            .select(F.col("subject").alias("entity"), F.col("object").alias(alias))
        )
        result = result.join(prop_df, "entity", "left")
    return result


def deduplicate_against_existing(new_triples: DataFrame, existing_triples: DataFrame) -> DataFrame:
    """Remove triples from new that already exist in existing. Fully distributed."""
    return new_triples.join(
        existing_triples,
        on=["subject", "predicate", "object"],
        how="left_anti"
    )