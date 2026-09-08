"""
PyTorch Geometric Knowledge Graph Builder - Spark job entry point

Apache Spark job (runs on a standalone Spark cluster with the RAPIDS
Accelerator for Apache Spark) that orchestrates the complete pipeline:
  Raw RDF → PySpark triples_df → Enrich → Build PyG HeteroData → Save

Storage model (local-first):
  - Interim enriched Parquet is written to a shared local working
    directory only (regenerable; avoids remote-object-store write cost).
  - Final artifacts (the .pt HeteroData and the six metadata JSON files)
    are written locally AND, when an S3 archive is configured, mirrored
    to S3 as a durable, reusable catalog.

Source data may be read from local paths or from S3 via the s3a://
scheme (both supported; local is the default).

Supports two source formats:
  - ntriples:       One triple per line in .nt files (default)
  - turtle_parquet: Self-contained Turtle blobs in a Parquet column

Supports three execution modes:
  - full:            End-to-end RDF enrichment + PyG graph construction
  - enrichment_only: Create reusable enriched Parquet artifacts (local)
  - pyg_only:        Build PyG graph from existing enriched Parquet artifacts

Launch with spark-submit (see bin/submit_spark_job.sh). Parameters:
    --mode:                    full | enrichment_only | pyg_only
    --source_paths:            comma-separated source path(s)/URI(s);
                               local dirs or s3a://... (full, enrichment_only)
    --input_mode:              s3 | local (default: s3). "local" reads a
                               staged mirror of the s3a:// sources from
                               --local_source_root instead of object storage;
                               see bin/stage_sources.sh
    --local_source_root:       root of the staged mirror, required when
                               --input_mode local
    --local_work_dir:          base dir on the shared filesystem for
                               enriched Parquet + local final artifacts
    --s3_archive_bucket:       optional S3 bucket to mirror final artifacts
    --s3_pyg_key:              optional S3 key for the .pt (metadata prefix
                               is derived from it)
    --enable_ontology_mapping: true | false (default: true)
    --allow_overwrite: true | false (default: false). Off, the job refuses to
        start when the artifacts it would write are already present.
    --time_period:             label (e.g. "2024-12") for output naming
    --pyg_config:              optional JSON string with PyG config
    --parquet_partitions:      number of Parquet output partitions (default 200)
    --source_format:           ntriples | turtle_parquet (default ntriples)
    --turtle_column:           Turtle column name when
                               source_format=turtle_parquet (default: empty,
                               auto-detected per source)

Example (N-Triples, local source, local + S3 archive):
    spark_jobs/build_graph.py \\
        --mode full \\
        --source_paths /data/rdf/monthly/2024-12/ \\
        --local_work_dir /data \\
        --s3_archive_bucket my-archive \\
        --s3_pyg_key pyg/year=2024/month=12/hetero_data.pt \\
        --enable_ontology_mapping true \\
        --time_period 2024-12

Example (Turtle Parquet, s3a source, local only):
    spark_jobs/build_graph.py \\
        --mode enrichment_only \\
        --source_paths s3a://my-data-lake/raw/sec/filings/2024-12/ \\
        --local_work_dir /data \\
        --source_format turtle_parquet \\
        --turtle_column triples \\
        --time_period 2024-12

Example (same sources, read from a staged mirror instead of object storage):
    bin/stage_sources.sh --dest /srv/pyg-source --nodes worker-a,worker-b \\
        --source s3a://my-data-lake/raw/sec/filings/2024-12/

    spark_jobs/build_graph.py \\
        --mode enrichment_only \\
        --source_paths s3a://my-data-lake/raw/sec/filings/2024-12/ \\
        --input_mode local \\
        --local_source_root /srv/pyg-source \\
        --local_work_dir /data \\
        --source_format turtle_parquet \\
        --time_period 2024-12
"""
import sys
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import boto3

# ============================================
# Project imports
# ============================================
from spark_jobs.enrichment.pipeline import EnrichmentPipeline

from spark_jobs.utils.spark_rdf_utils import literal_datatype_observations
from spark_jobs.utils.canonicalization import canonicalize_source_triples

# The Turtle parse path. It lives in a package rather than here because
# cloudpickle would otherwise ship it to executors by value; the module's own
# docstring says what that constrains. Imported by the two names this file
# uses, NOT re-exported: a caller in that module resolves its collaborators
# from that module's globals, so a test patching them here would patch a name
# nothing reads.
from spark_jobs.graph.turtle import (
    PARSE_BATCH_ROWS,
    turtle_batches_to_arrow,
)

from spark_jobs.pyg_builder.metadata_writer import (
    write_metadata_to_s3,
    write_metadata_to_local,
    write_latest_alias,
    derive_metadata_prefix,
    derive_node_index_prefix,
)

# What the job was asked to do. The PyG availability flag comes from here too:
# JobConfig has to reject a pyg-requiring mode before the job starts, so the
# probe lives beside the validation that reads it.
from spark_jobs.graph.config import (
    PYG_BUILDER_AVAILABLE,
    JobConfig,
    parse_args,
)

if PYG_BUILDER_AVAILABLE:
    from spark_jobs.pyg_builder.constructor import build_hetero_data

# ============================================
# Logging setup
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_graph")


def _utcnow() -> datetime:
    """Naive UTC now — drop-in for the deprecated datetime.utcnow().

    Returns a *naive* datetime (no tzinfo) so isoformat()/strftime()
    output stays byte-identical to the historical utcnow() call sites
    (a tz-aware isoformat() would insert "+00:00" before the manual "Z").
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ============================================
# Constants
# ============================================
# The shape every loader below produces, whatever it read. object_datatype is
# not here: the Turtle path carries it only long enough for load_source_triples
# to build the marker triples, then drops it.
TRIPLES_SCHEMA = StructType([
    StructField("subject", StringType(), nullable=False),
    StructField("predicate", StringType(), nullable=False),
    StructField("object", StringType(), nullable=False),
])


# ============================================
# N-Triples parsing (raw RDF → triples DataFrame)
# ============================================
def load_ntriples_to_dataframe(
    spark: SparkSession, source_path: str
) -> DataFrame:
    """
    Load N-Triples files into a PySpark triples DataFrame.

    Reads from a local path or an s3a:// URI. Runs entirely on Spark
    executors — no driver-side parsing.

    N-Triples format (one triple per line):
        <subject> <predicate> <object> .

    Object can be a URI (<...>) or a literal ("..."^^<type> or "..."@lang).

    Args:
        spark: Active SparkSession
        source_path: S3 path containing .nt files

    Returns:
        DataFrame with columns (subject: string, predicate: string, object: string)
    """
    logger.info(f"Loading N-Triples from {source_path}")

    raw_lines = spark.read.text(source_path)

    # Filter blank lines and comments (executor-side)
    raw_lines = raw_lines.filter(
        (F.trim(F.col("value")) != "")
        & (~F.col("value").startswith("#"))
    )

    # Parse N-Triples using regex on executors.
    #
    # Subject and predicate are always URIs: <http://...>
    # Object is either a URI or a literal.
    #
    # Regex strategy:
    #   - Match subject: first <...>
    #   - Match predicate: second <...>
    #   - Match raw_object: everything after predicate up to trailing " ."
    #
    # The raw_object regex uses a non-greedy match with an anchor on
    # the final " ." to avoid truncating literals that contain dots.
    triples_df = raw_lines.select(
        F.regexp_extract("value", r"<([^>]+)>", 1).alias("subject"),
        F.regexp_extract("value", r"<[^>]+>\s+<([^>]+)>", 1).alias(
            "predicate"
        ),
        # Non-greedy: capture everything between predicate and final " ."
        # The (?s) flag is not needed since we read line-by-line.
        F.regexp_extract(
            "value", r"<[^>]+>\s+<[^>]+>\s+(.+?)\s*\.\s*$", 1
        ).alias("raw_object"),
    )

    # Clean object field (executor-side):
    #   URI:            <http://...>           → http://...
    #   Typed literal:  "value"^^<datatype>    → value
    #   Lang literal:   "value"@en             → value
    #   Plain literal:  "value"                → value
    triples_df = triples_df.withColumn(
        "object",
        F.when(
            F.col("raw_object").startswith("<"),
            F.regexp_extract("raw_object", r"<([^>]+)>", 1),
        )
        .when(
            F.col("raw_object").contains("^^"),
            F.regexp_extract("raw_object", r'"((?:[^"\\]|\\.)*)"', 1),
        )
        .when(
            F.col("raw_object").rlike(r'"[^"]*"@'),
            F.regexp_extract("raw_object", r'"((?:[^"\\]|\\.)*)"', 1),
        )
        .otherwise(
            F.regexp_extract("raw_object", r'"((?:[^"\\]|\\.)*)"', 1)
        ),
    ).drop("raw_object")

    # Drop rows where parsing failed (executor-side)
    triples_df = triples_df.filter(
        (F.col("subject") != "") & (F.col("predicate") != "")
    )

    return triples_df


# ============================================
# Turtle Parquet parsing (source Parquet → triples DataFrame)
# ============================================
# Column names a Turtle Parquet source may use for its blob, in preference
# order. Sources written by different scrapers disagree — 'triples' for most,
# 'rdf_turtle' for others — and --source_paths takes many prefixes in a single
# run, so the column is resolved per source rather than once for the whole job.
# One submission can therefore span sources that do not agree on the name: each
# is parsed into the same (subject, predicate, object) schema before the union,
# so what the column was called on disk never reaches downstream steps.
TURTLE_COLUMN_CANDIDATES = ("triples", "rdf_turtle")


def resolve_turtle_column(columns, turtle_column: str = "") -> str:
    """
    Pick the column holding the Turtle blob for one source.

    An explicit --turtle_column is honored verbatim and must exist, so a source
    with a third name — or one carrying two candidates where only one is meant —
    stays forceable. Otherwise the first candidate present wins.

    Args:
        columns: Column names in the source Parquet schema.
        turtle_column: Explicit override; empty means auto-detect.

    Returns:
        Name of the column to read Turtle strings from.

    Raises:
        ValueError: If the override is absent, or no candidate is present.
    """
    if turtle_column:
        if turtle_column not in columns:
            raise ValueError(
                f"Column '{turtle_column}' not found in Parquet schema. "
                f"Available columns: {list(columns)}. "
                f"Set --turtle_column to the correct column name."
            )
        return turtle_column

    for candidate in TURTLE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"No Turtle column found. Tried {list(TURTLE_COLUMN_CANDIDATES)}; "
        f"available columns: {list(columns)}. Set --turtle_column to the "
        f"correct column name."
    )


def load_turtle_parquet_to_dataframe(
    spark: SparkSession,
    source_path: str,
    turtle_column: str = "",
) -> DataFrame:
    """
    Load RDF Turtle strings from a Parquet file column into a PySpark
    triples DataFrame.

    Each row in the source Parquet contains a self-contained Turtle
    blob in `turtle_column`. Each blob may contain multiple subjects,
    predicate lists (;), object lists (,), prefix declarations (@prefix),
    and XSD-typed literals. A Python UDF calls rdflib to parse each
    blob correctly — this is the one place a UDF is appropriate because
    the input is a small string per row, not a per-triple operation.
    All downstream enrichment and PyG construction steps remain pure
    Spark expressions.

    The UDF emits fully-expanded (subject, predicate, object) triples
    with all prefixed names resolved to absolute URIs. The output
    DataFrame has the same schema as load_ntriples_to_dataframe() and
    is compatible with all downstream pipeline steps.

    Object values are normalized to match the pipeline's existing
    convention:
      - URI objects:     full URI string (angle brackets stripped)
      - Typed literals:  the lexical value only (^^datatype stripped)
      - Lang literals:   the string value only (@lang stripped)
      - Plain literals:  the string value as-is
      - Blank nodes:     kept as "_:identifier" strings

    Requires rdflib installed in the Spark Python environment on all
    workers (see requirements.txt).

    Args:
        spark: Active SparkSession
        source_path: Local path or s3a:// URI to source Parquet files
        turtle_column: Name of the column containing Turtle strings. Empty
                       (the default) auto-detects from
                       TURTLE_COLUMN_CANDIDATES; set it via --turtle_column
                       to force one name.

    Returns:
        DataFrame with columns (subject: string, predicate: string,
        object: string, object_datatype: string), one row per triple.

        ``object_datatype`` is the fourth column ON PURPOSE and is not part of
        the pipeline's canonical schema -- ``load_source_triples`` turns it into
        marker triples and drops it, the same way it does with SOURCE_COLUMN.
        This function used to build the markers itself and union them in, which
        read the parse a second time; see #375 and the note at the return.

    Raises:
        ValueError: If no usable Turtle column is found in the Parquet schema.
    """
    raw_df = spark.read.parquet(source_path)
    turtle_column = resolve_turtle_column(raw_df.columns, turtle_column)

    logger.info(
        f"Loading Turtle Parquet from {source_path}, "
        f"column='{turtle_column}'"
    )

    # Select only the turtle column — all other metadata columns
    # (scraped content, timestamps, etc.) are not needed for the
    # triples DataFrame
    turtle_df = raw_df.select(F.col(turtle_column)).filter(
        F.col(turtle_column).isNotNull()
        & (F.trim(F.col(turtle_column)) != "")
    )

    # ============================================
    # Parse: one Turtle blob -> many triple rows, streamed
    # ============================================
    # A real parser is used here because Turtle has prefix resolution,
    # predicate lists (;), object lists (,), and typed literals that cannot be
    # correctly parsed with regex.
    #
    # mapInArrow, NOT a UDF returning an array. A UDF must hand every triple
    # from a blob back as one value, and one SEC blob in the 2026-09 data is
    # 5.6 MB -> 57,350 triples -> 14.8 MB in a single piece, which deadlocks the
    # executor against its Python worker. turtle_batches_to_arrow carries the
    # whole account; it is the function to read before changing any of this.
    #
    # The parsing itself is turtle_to_rows(); read the note above it for why it
    # is pyoxigraph rather than rdflib, and for what rdflib still decides.
    #
    # Object normalization matches the pipeline convention established in
    # load_ntriples_to_dataframe():
    #   - URI     → the URI
    #   - Literal → str(literal.toPython()) for numerics,
    #               the lexical form for strings (no ^^datatype)
    #   - BNode   → f"_:{content_hash}" (see deterministic_bnode_labels;
    #               a parser's own labels are per-parse, which made the
    #               whole pipeline non-reproducible)
    #
    # Malformed blobs contribute no rows so that parse errors skip the blob
    # rather than failing the job. The zero-triple contribution is visible in
    # the final triple count logged after loading.

    triple_schema = StructType([
        StructField("subject", StringType(), nullable=False),
        StructField("predicate", StringType(), nullable=False),
        StructField("object", StringType(), nullable=False),
        # The literal's declared ^^<datatype>, "" for URIs/bnodes/plain
        # literals. Dropped by load_source_triples once the observation markers
        # are built -- the frame that reaches the pipeline is the canonical
        # 3-column one.
        StructField("object_datatype", StringType(), nullable=False),
    ])

    # Spark's own batch bound, so one knob governs how much crosses the boundary
    # at a time rather than this path inventing a second one.
    max_rows = int(
        spark.conf.get(
            "spark.sql.execution.arrow.maxRecordsPerBatch", str(PARSE_BATCH_ROWS)
        )
    )

    def parse_turtle_batches(batches):
        return turtle_batches_to_arrow(batches, max_rows)

    parsed_df = turtle_df.mapInArrow(parse_turtle_batches, triple_schema).filter(
        (F.col("subject") != "")
        & (F.col("predicate") != "")
    )

    # Returned with object_datatype still on it, and WITHOUT the datatype
    # markers unioned in. Building them here read parsed_df a second time, and
    # nothing had cached it yet, so Spark ran the rdflib UDF over every source
    # twice: one pass for the three-column projection, another for the marker
    # branch's distinct(). Measured on the 2026-09-05 run, those two passes were
    # 9.17 and 9.29 of the load phase's 20.57 task-hours.
    #
    # load_source_triples derives the markers once, off the cached frame, and
    # drops this column -- the same shape it already uses for SOURCE_COLUMN. #375
    return parsed_df


# ============================================
# Per-source accounting
# ============================================
#
# The triples frame is (subject, predicate, object) and nothing else, so two
# identical triples are indistinguishable in every respect -- there is no fourth
# column in which they COULD differ. That is why the manifest could report how
# many rows were deduplicated but never which source contributed them, and why
# the alternative was to guess: attribute a shared fact to whichever source was
# listed later, or to neither, or to both. Every one of those is a convention
# rather than an observation, and the first is not even stable, since it changes
# when the source paths are passed in a different order.
#
# Stamping the source at load time removes the question instead of answering it.
#
# THE COLUMN DOES NOT SURVIVE THIS FUNCTION. It is attached per path, used to
# compute the statistics below, and dropped before the frame is returned, so the
# ~25 modules that construct or read the three-column shape are untouched and the
# extra width is never carried through an enrichment shuffle.
#
# CARRYING IT FURTHER IS A REAL OPTION, DELIBERATELY NOT TAKEN.
# If the source travelled with each row through enrichment, dropDuplicates would
# become an aggregation that collects sources rather than discarding rows, and the
# graph itself would know that N independent sources asserted a given fact --
# available to a model as a corroboration signal rather than merely as a
# statistic. The reason it is not done here is empirical, not aesthetic: measured
# on real data, a three-source run removed 177,828 duplicate rows from 19.6M, and
# roughly 107,619 of those were one source repeating ITSELF. Cross-source
# agreement is therefore at most ~0.36% of rows and in reality lower. A feature
# that reads "1 source" for better than 99.7% of edges has almost no variance for
# a GNN to learn from, and the `origin` split (raw / enrichment / unification)
# already separates reported facts from derived ones with far more of it.
#
# Revisit if genuinely overlapping sources are added -- a second market vendor, or
# a provider restating company facts the filings already carry. Then agreement
# becomes a real signal and the schema change earns its cost.

SOURCE_COLUMN = "_source"

# Path fragment -> short label. Keyed on fragments rather than whole paths so a
# label survives a bucket or prefix change, which keeps counts comparable across
# runs that read the same data from different locations.
_SOURCE_LABEL_PATTERNS = (
    ("source=sec", "sec"),
    ("source=bls", "bls"),
    ("/noaa/", "noaa"),
    ("quotes", "market"),
)

# Recognised source names, matched against whole path SEGMENTS. Production paths
# carry the partition fragments above; the committed fixtures are plain files
# (ntriples/sec.nt), and without this every e2e run would label its sources
# positionally -- correct but useless for reading a report.
_SOURCE_NAMES = frozenset({"sec", "bls", "noaa", "market"})


def source_label(source_path: str, index: int) -> str:
    """A short, stable label for a source path.

    Deliberately NOT the path itself. The manifest is copied to object storage
    and pasted into issues, and a bucket-qualified URI used as a map key spreads
    deployment detail through a structure people quote casually --
    ``config.source_paths`` already records the paths once, in a field a reader
    knows to treat as sensitive.

    Falls back to a positional label rather than to any part of the path, so an
    unrecognised source cannot leak one either.
    """
    lowered = source_path.lower()
    for fragment, label in _SOURCE_LABEL_PATTERNS:
        if fragment in lowered:
            return label

    # Whole segments only, never substrings: a bucket called "secure-data" or a
    # directory named "marketing" contains a source name but is not one, and a
    # mislabelled source is worse than an unlabelled one because it reads as
    # authoritative.
    for segment in lowered.replace("\\", "/").split("/"):
        stem = segment.split(".", 1)[0]
        if stem in _SOURCE_NAMES:
            return stem

    return f"source_{index}"


def per_source_triple_stats(stamped_df: DataFrame) -> dict:
    """Rows each source contributed, and how the duplicates among them arose.

    Two kinds of duplicate are reported separately because they mean opposite
    things. A source repeating ITSELF is a hygiene signal worth raising upstream.
    Two sources independently stating the same fact is corroboration -- arguably
    the point of a knowledge graph -- and reporting it against a source would make
    a good outcome look like a defect.

    Computed AFTER canonicalization, so the counts match what the enrichment
    pipeline's dropDuplicates will actually collapse rather than what the raw
    files happened to spell.
    """
    triple = ["subject", "predicate", "object"]

    contributed = {
        row[SOURCE_COLUMN]: row["n"]
        for row in stamped_df.groupBy(SOURCE_COLUMN).count()
        .withColumnRenamed("count", "n").collect()
    }

    # Rows a source would lose to dedup on its own, ignoring every other source.
    distinct_within = {
        row[SOURCE_COLUMN]: row["n"]
        for row in stamped_df.select(*triple, SOURCE_COLUMN).distinct()
        .groupBy(SOURCE_COLUMN).count()
        .withColumnRenamed("count", "n").collect()
    }

    # Facts more than one source states. Counted per (triple, source) pair so a
    # source repeating itself does not inflate the number of sources agreeing.
    by_triple = (
        stamped_df.select(*triple, SOURCE_COLUMN).distinct()
        .groupBy(*triple).agg(F.countDistinct(SOURCE_COLUMN).alias("sources"))
    )
    shared = by_triple.filter(F.col("sources") > 1)

    return {
        "sources": {
            label: {
                "rows_contributed": contributed.get(label, 0),
                "duplicates_within_source": (
                    contributed.get(label, 0) - distinct_within.get(label, 0)
                ),
            }
            for label in sorted(contributed)
        },
        "facts_stated_by_multiple_sources": shared.count(),
    }


# ============================================
# Source loader dispatcher
# ============================================
def load_source_triples(
    spark: SparkSession,
    config: "JobConfig",
) -> Tuple[DataFrame, int, dict]:
    """
    Load raw source data into a triples DataFrame.

    Dispatches to the correct loader based on config.source_format:
      - "ntriples":       load_ntriples_to_dataframe()
      - "turtle_parquet": load_turtle_parquet_to_dataframe()

    Supports multiple source paths (local directories or s3a:// URIs).
    When more than one path is provided, each is loaded independently
    and the results are unioned into a single triples DataFrame before
    caching. Duplicates are not deduplicated here — neither those across
    paths nor those a single source already contains.

    What removes them is EnrichmentPipeline's per-phase
    ``dropDuplicates(["subject", "predicate", "object"])`` over the whole
    frame, NOT the left_anti pattern this docstring used to name.
    ``deduplicate_against_existing()`` is a real helper and the linkers do use
    it, but it only drops NEW triples that already exist — the source is its
    "existing" side, so it can never remove a duplicate the source arrived
    with. The distinction matters to anyone reading the triple counts: a real
    source carries duplicates (one SEC day: ~8.7%), and they are collapsed
    during enrichment rather than at load, which is why ``initial_triples`` is
    a pre-dedup count while ``final_triples`` is post-dedup.

    Caches the resulting DataFrame and forces materialization so that
    downstream steps do not re-trigger the parse UDF.

    Args:
        spark: Active SparkSession
        config: Parsed job configuration

    Returns:
        Tuple of (triples_df cached on executors, triple_count)

    Raises:
        FileNotFoundError: If no triples are parsed from any source path.
        ValueError: If turtle_column is not found (turtle_parquet only).
    """
    source_paths = list(config.source_paths)
    read_paths = list(config.read_paths)

    logger.info(
        f"Source format: {config.source_format}, "
        f"input mode: {config.input_mode}, "
        f"{len(read_paths)} path(s)"
    )
    for path in read_paths:
        logger.info(f"  {path}")

    loaded: List[DataFrame] = []

    for index, source_path in enumerate(read_paths):
        if config.source_format == "ntriples":
            df = load_ntriples_to_dataframe(spark, source_path)
        else:
            df = load_turtle_parquet_to_dataframe(
                spark,
                source_path,
                turtle_column=config.turtle_column,
            )
        # Canonicalize per path rather than once after the union, so the source
        # stamp survives. canonicalize_sec_identifiers ends in an explicit
        # three-column select, which would drop any column added before it.
        # Applying it here is equivalent: every rule inside is a row-wise column
        # expression, so it does not matter whether rows from different paths are
        # in the same frame yet.
        df = canonicalize_source_triples(df)
        # Labelled from the DECLARED path, not the one just read. A staged
        # mirror keeps the key layout so both spell the source the same way,
        # but the staging root is chosen by whoever ran the sync -- keying the
        # label off it would let a root named /srv/sec-mirror relabel every
        # source in the run, and per-source statistics have to mean the same
        # thing in both input modes to be comparable at all.
        df = df.withColumn(
            SOURCE_COLUMN, F.lit(source_label(source_paths[index], index))
        )
        loaded.append(df)
        logger.info(f"  Parsed: {source_path}")

    # Union all prefix DataFrames into one.
    # unionAll is a lazy transformation — no data moves until cache().
    if len(loaded) == 1:
        triples_df = loaded[0]
    else:
        triples_df = loaded[0]
        for df in loaded[1:]:
            triples_df = triples_df.unionAll(df)

    # Identifier canonicalization already ran per path above — collapsing the two
    # spellings upstream uses for one entity onto a single URI, before anything
    # reads the frame. See canonicalization.py: it adds no triple and drops none.
    # It happens inside this function rather than in a pipeline phase because the
    # enriched Parquet is written from this frame, so a later repair would leave
    # `pyg_only` runs reading the unrepaired shape.

    # Cache and materialize once — all downstream steps read from cache.
    # For turtle_parquet, this also ensures the rdflib UDF runs exactly
    # once across all prefixes rather than being re-triggered by each
    # downstream action.
    #
    # The stamped frame is what gets cached, so the per-source statistics below
    # read from cache rather than re-triggering the parse. That ordering is the
    # whole reason this costs an aggregation instead of a second parse: on real
    # input the parse is ~176s and the aggregation is seconds.
    # DISK_ONLY, not .cache(). MEMORY_AND_DISK unrolls this frame through
    # MemoryStore.putIteratorAsValues, and that unroll is what deadlocks the
    # parse. The count below is stage 13, which wedged twice on 2026-09-06
    # (issue-380-dump, issue-380-validate) with one task left of 169: the task
    # thread sat in SocketInputStream.read inside putIteratorAsValues, holding
    # the GpuArrowReader open, while "stdout writer for python" sat in
    # SocketOutputStream.write holding the stream monitors. The reader stops
    # draining Python's output while it waits for unroll memory, so Python
    # blocks writing output, so it stops reading input, so the writer blocks
    # too -- both socket directions full and nothing left to break the tie.
    # DISK_ONLY writes the iterator straight to the disk store, never runs that
    # unroll, and the stream keeps draining. Same reasoning as the assembly
    # leg's persist below, which was moved off .cache() for the sibling
    # deadlock.
    triples_df = triples_df.persist(StorageLevel.DISK_ONLY)
    # Materialized with its own action, BEFORE the marker frame forks off it.
    # Assembling the union first and counting once would leave both branches
    # racing a cache nothing had populated yet, which is exactly the shape #375
    # was filed about.
    count = triples_df.count()

    # Datatype markers, built once from the cached parse instead of by reading
    # the source again. Only the turtle loader hands `object_datatype` up here;
    # load_ntriples_to_dataframe returns the three canonical columns and reads
    # no datatype at all, so the column's presence is what picks the path.
    #
    # SOURCE_COLUMN rides along so per-source accounting charges each marker to
    # the source that declared it -- which came for free while the loader built
    # markers per source, and has to be asked for now that the parse is shared.
    #
    # Cached and counted here rather than left lazy: the distinct() underneath
    # is a shuffle over the whole frame, and every downstream action -- the
    # statistics below, the enrichment phases, the Parquet write -- would
    # otherwise run it again. One parse and one shuffle, against the two parses
    # and one shuffle this replaces.
    if "object_datatype" in triples_df.columns:
        markers = literal_datatype_observations(
            triples_df, carry=[SOURCE_COLUMN]
        ).cache()
        count += markers.count()
        triples_df = triples_df.drop("object_datatype").unionByName(markers)

    # Checked before the statistics below, which would otherwise aggregate an
    # empty frame on the way to raising anyway.
    if count == 0:
        raise FileNotFoundError(
            f"No triples parsed from {config.source_format} source(s): "
            f"{source_paths}."
            + (
                f" Check that column '{config.turtle_column}' contains "
                f"valid Turtle strings."
                if config.source_format == "turtle_parquet"
                else " Check that .nt files exist at the source paths."
            )
        )

    source_stats = per_source_triple_stats(triples_df)
    for label, stats in source_stats["sources"].items():
        logger.info(
            f"  {label}: {stats['rows_contributed']:,} rows"
            f" ({stats['duplicates_within_source']:,} repeated within the source)"
        )
    logger.info(
        f"  facts stated by more than one source: "
        f"{source_stats['facts_stated_by_multiple_sources']:,}"
    )

    # Drop the stamp: everything downstream expects (subject, predicate, object).
    # A projection off a cached frame does not re-read the source.
    triples_df = triples_df.drop(SOURCE_COLUMN)

    logger.info(
        f"Loaded {count:,} triples total across "
        f"{len(source_paths)} path(s)"
    )
    return triples_df, count, source_stats


# ============================================
# Parquet I/O for enriched triples
# ============================================
def save_enriched_parquet(
    triples_df: DataFrame, output_path: str, num_partitions: int
) -> None:
    """
    Save enriched triples DataFrame as Parquet to the shared local
    working directory.

    Data flows directly from executors to the target path — nothing
    passes through the driver. The path must be on a filesystem visible
    to every worker (e.g. an NFS mount) so the partitioned output can be
    read back distributed in a later pyg_only run. Repartitions to control
    output file count and size for efficient downstream reads.

    Args:
        triples_df: Enriched triples DataFrame (subject, predicate, object)
        output_path: Local (shared) path, e.g.
            "/data/enriched/year=2024/month=12/triples"
        num_partitions: Number of output Parquet partitions
    """
    (
        triples_df
        .repartition(num_partitions)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    logger.info(
        f"Saved enriched triples ({num_partitions} partitions) "
        f"to {output_path}"
    )


DATASET_DESCRIPTOR_NAME = "dataset.json"


def dataset_descriptor_path(enriched_parquet_path: str) -> str:
    """Where the dataset descriptor sits: beside the Parquet dir, not inside it.

    Inside would put a non-Parquet file in a directory Spark reads as one. A
    leading underscore would make Spark's reader skip it, but Hadoop's hidden
    file filter then also hides it from an explicit read -- it could be written
    and never read back.
    """
    parent = enriched_parquet_path.rstrip("/").rsplit("/", 1)[0]
    return f"{parent}/{DATASET_DESCRIPTOR_NAME}"


def save_dataset_descriptor(config: "JobConfig", spark: SparkSession) -> None:
    """Record which sources produced this enriched output, beside the output.

    The enriched Parquet is subject/predicate/object only -- SOURCE_COLUMN is
    dropped at the write -- so a later pyg_only run cannot tell what its graph
    was built from. Without this, that fact lives solely in the enrichment
    manifest: a different file, under a different prefix, that a reader has to
    know to go looking for. It is lost outright the moment a graph is copied
    anywhere else.

    Labels, not paths. source_label() exists so a source can be named without
    naming a bucket, and what is written here reaches the published graph schema.
    """
    labels = sorted({
        source_label(path, index)
        for index, path in enumerate(config.source_paths)
    })
    body = json.dumps({
        "dataset": config.dataset,
        "sources": labels,
        "time_period": config.time_period,
        "written": _utcnow().isoformat(),
    }, indent=2).encode("utf-8")

    path = dataset_descriptor_path(config.enriched_parquet_path)
    try:
        _write_manifest_bytes(spark, path, body)
        logger.info(f"Saved dataset descriptor to {path}")
        logger.info(f"  dataset: {config.dataset or '(unnamed)'}")
        logger.info(f"  sources: {', '.join(labels) or '(none)'}")
    except Exception as e:
        # Not fatal: the enriched output is already written and correct, and a
        # graph built without this simply records no sources.
        logger.warning(f"Failed to save dataset descriptor: {e}")


def load_dataset_descriptor(
    spark: SparkSession, enriched_parquet_path: str
) -> Dict[str, Any]:
    """Read the descriptor beside the enriched output, or ``{}`` if absent.

    Absent is normal rather than an error: every enriched directory written
    before this existed has none, and a graph schema that records nothing is
    honest where one that guessed would not be.
    """
    path = dataset_descriptor_path(enriched_parquet_path)
    try:
        # One row per line, not one row per file: the `wholetext` option is
        # silently ignored here, so a pretty-printed descriptor arrives split
        # and rows[0] is just "{". Rejoining is safe because the file is well
        # under a block -- Spark reads it as a single partition and collect()
        # preserves order within one.
        rows = spark.read.text(path).collect()
        if rows:
            return json.loads("\n".join(row[0] for row in rows))
    except Exception as e:
        logger.info(
            f"No readable dataset descriptor at {path} ({type(e).__name__}: "
            f"{e}); the graph schema will not name its sources."
        )
    return {}


def load_enriched_parquet(
    spark: SparkSession, input_path: str
) -> DataFrame:
    """
    Load enriched triples from Parquet in the shared local working
    directory.

    Reads directly into a distributed DataFrame on executors.
    This loads Parquet that this pipeline previously wrote
    (schema: subject, predicate, object) — not source Parquet.

    Args:
        spark: Active SparkSession
        input_path: Local (shared) path to enriched Parquet

    Returns:
        DataFrame with columns (subject, predicate, object)
    """
    logger.info(f"Loading enriched Parquet from {input_path}")
    triples_df = spark.read.parquet(input_path)
    return triples_df


# ============================================
# PyG output (driver → S3)
# ============================================
def save_pyg_to_s3(s3_client, hetero_data, bucket: str, key: str):
    """
    Save PyTorch Geometric HeteroData to S3.

    Serializes to a temp file, then hands boto3 the path so its multipart
    upload streams from disk.

    Staging through disk rather than a BytesIO for the same reason as
    save_pyg_local below: a day of four sources is ~53 GiB of tensors, and
    buffering the serialized copy alongside them doubles a driver footprint
    that already does not fit. Disk costs one .pt for the length of the upload.

    Args:
        s3_client: Boto3 S3 client
        hetero_data: PyG HeteroData object
        bucket: S3 bucket name
        key: S3 key
    """
    import os
    import tempfile

    import torch

    handle, staged = tempfile.mkstemp(prefix="hetero_data.", suffix=".pt")
    os.close(handle)
    try:
        torch.save(hetero_data, staged)
        size_mb = os.path.getsize(staged) / (1024 * 1024)
        s3_client.upload_file(
            Filename=staged,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )
    finally:
        os.unlink(staged)

    logger.info(
        f"Saved PyG HeteroData ({size_mb:.2f} MB) to s3://{bucket}/{key}"
    )


def save_pyg_local(hetero_data, local_path: str, spark: SparkSession = None) -> None:
    """
    Save PyTorch Geometric HeteroData to the job's work dir.

    Runs on the driver — only compact tensors are in memory.

    Despite the name (kept for its callers), the destination is not necessarily
    local: ``local_work_dir`` is an ``s3a://`` URI on the cluster, and
    ``torch.save`` to such a path writes a junk ``./s3a:/...`` tree on the
    driver's disk and reports success.

    Both destinations stream through a file handle, and neither ever holds the
    serialized graph in memory. That matters more than it sounds: a day of four
    sources builds ~53 GiB of tensors, and the earlier version serialized a URI
    destination into a BytesIO first. That is a second full copy in Python, and
    handing it to Hadoop across Py4J makes a third in the JVM — on a driver heap
    of 8g. "Only compact tensors are in memory" stopped being true somewhere
    between the fixtures this was written against and a real day of data.

    A local path is written in place. A URI is staged into a temp file beside
    the destination's local spill area, then moved across by Hadoop, which
    copies in blocks. Peak memory is the HeteroData itself either way; the URI
    case costs disk equal to one .pt while the move runs.

    Args:
        hetero_data: PyG HeteroData object
        local_path: Destination for the .pt file — bare path or URI
        spark: Active SparkSession; required when local_path is a non-local URI
    """
    import os
    import tempfile

    import torch

    from spark_jobs.utils.fs_utils import (
        is_local_path,
        local_filesystem_path,
        write_file,
    )

    if is_local_path(local_path):
        path = local_filesystem_path(local_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(hetero_data, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Saved PyG HeteroData ({size_mb:.2f} MB) to {path}")
        return

    handle, staged = tempfile.mkstemp(prefix="hetero_data.", suffix=".pt")
    os.close(handle)
    try:
        torch.save(hetero_data, staged)
        size_mb = os.path.getsize(staged) / (1024 * 1024)
        write_file(staged, local_path, spark=spark)
    finally:
        # write_file consumes the staged file on success; this is the failure path.
        if os.path.exists(staged):
            os.unlink(staged)

    logger.info(
        f"Saved PyG HeteroData ({size_mb:.2f} MB) to {local_path} (via Hadoop FS)"
    )


# ============================================
# Pipeline phases
# ============================================
def run_enrichment(
    spark: SparkSession,
    triples_df: DataFrame,
    enable_ontology_mapping: bool = True,
    market_sector_definitions_bucket: str = "",
    market_sector_definitions_key: str = "",
    class_mappings: Dict[str, Any] = None,
) -> tuple:
    """
    Run the enrichment pipeline on a triples DataFrame.

    All enrichment runs as PySpark DataFrame operations on executors.

    Args:
        spark: Active SparkSession
        triples_df: Raw triples DataFrame (subject, predicate, object)
        enable_ontology_mapping: Whether to run ontology mapping
        class_mappings: Per-run additions/overrides to the built-in
            CLASS_MAPPINGS table (see ontology_mapper)

    Returns:
        Tuple of (enriched_triples_df, enrichment_stats_dict)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: RDF ENRICHMENT")
    logger.info("=" * 80)
    logger.info(
        f"Ontology mapping: "
        f"{'enabled' if enable_ontology_mapping else 'disabled'}"
    )
    logger.info("")

    start_time = time.time()

    pipeline = EnrichmentPipeline(
        spark,
        triples_df,
        sector_definitions_bucket=market_sector_definitions_bucket,
        sector_definitions_key=market_sector_definitions_key,
    )
    stats = pipeline.run(
        enable_ontology_mapping=enable_ontology_mapping,
        class_mappings=class_mappings,
    )
    enriched_df = pipeline.get_enriched_triples_df()

    elapsed = time.time() - start_time
    logger.info(f"Enrichment completed in {elapsed:.1f}s")

    return enriched_df, stats


def run_pyg_construction(
    spark: SparkSession,
    triples_df: DataFrame,
    pyg_config: Dict[str, Any] = None,
    time_period: str = "",
    dataset: str = "",
    sources: Optional[List[str]] = None,
) -> Tuple:
    """
    Run PyG HeteroData construction from enriched triples DataFrame.

    Heavy computation (node ID assignment, edge resolution, feature
    extraction) runs on Spark executors. Only compact tensors cross
    to the driver for final HeteroData assembly.

    Also collects metadata artifacts during construction for the six
    metadata JSON files. All metadata collect() calls target small
    aggregated DataFrames — no driver memory concern.

    Args:
        spark: Active SparkSession
        triples_df: Enriched triples DataFrame
        pyg_config: Optional PyG construction configuration
        time_period: Time period label for metadata

    Returns:
        Tuple of (HeteroData, MetadataCollector)
    """
    if not PYG_BUILDER_AVAILABLE:
        raise ImportError(
            "PyG builder not available. Install PyTorch Geometric."
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: PyG CONSTRUCTION")
    logger.info("=" * 80)
    if pyg_config:
        logger.info(f"PyG config: {json.dumps(pyg_config, indent=2)}")
    logger.info("")

    start_time = time.time()

    config = pyg_config or {}
    hetero_data, metadata, node_index_df = build_hetero_data(
        spark, triples_df, config, time_period=time_period,
        dataset=dataset, sources=sources,
    )

    elapsed = time.time() - start_time

    logger.info(f"PyG construction completed in {elapsed:.1f}s")
    _log_hetero_data_summary(hetero_data)

    return hetero_data, metadata, node_index_df


def _log_hetero_data_summary(hetero_data):
    """Log HeteroData node/edge type summary."""
    logger.info("HeteroData summary:")
    logger.info(f"  Node types: {hetero_data.node_types}")
    logger.info(f"  Edge types: {hetero_data.edge_types}")
    for node_type in hetero_data.node_types:
        store = hetero_data[node_type]
        num_nodes = getattr(store, "num_nodes", "unknown")
        num_features = (
            store.x.shape[1]
            if hasattr(store, "x") and store.x is not None
            else 0
        )
        logger.info(
            f"  [{node_type}] nodes={num_nodes}, features={num_features}"
        )
    for edge_type in hetero_data.edge_types:
        store = hetero_data[edge_type]
        num_edges = (
            store.edge_index.shape[1]
            if hasattr(store, "edge_index")
            else "unknown"
        )
        logger.info(f"  [{edge_type}] edges={num_edges}")


# ============================================
# Spark session initialization
# ============================================
def get_spark_session() -> SparkSession:
    """
    Get or create a SparkSession.

    The master and the bulk of the RAPIDS Accelerator / GPU-resource
    configuration are supplied by spark-submit (see bin/submit_spark_job.sh)
    and/or the cluster's spark-defaults. As an idempotent safety net we set
    ``spark.rapids.sql.enabled=true`` here so GPU acceleration is on even for
    a bare submit; if the RAPIDS plugin jar is not on the classpath this flag
    is simply ignored.

    ``PYG_RAPIDS_SQL_ENABLED=false`` turns it off. Setting the key here rather
    than reading what spark-submit passed is what made it unreachable: builder
    options are applied over the submitted conf, so ``--conf
    spark.rapids.sql.enabled=false`` was accepted, logged, and then overwritten.
    A benchmark arm for issue #342 ran a whole leg believing RAPIDS was off when
    the event log says it was on. The parse is a Python UDF that RAPIDS cannot
    accelerate, so being able to take it out of the query path is the thing #342
    needs to measure.
    """
    import os

    rapids_enabled = os.environ.get("PYG_RAPIDS_SQL_ENABLED", "true").strip().lower()
    if rapids_enabled not in ("true", "false"):
        raise ValueError(
            f"PYG_RAPIDS_SQL_ENABLED must be 'true' or 'false', got "
            f"{os.environ['PYG_RAPIDS_SQL_ENABLED']!r}. Anything else would be "
            f"read as false by Spark and quietly disable GPU acceleration."
        )
    logger.info(f"RAPIDS SQL acceleration requested: {rapids_enabled}")
    return (
        SparkSession.builder.appName("PyG-Knowledge-Graph-Builder")
        .config("spark.rapids.sql.enabled", rapids_enabled)
        # A settled frame's checkpoint is dead as soon as the frame that
        # replaces it settles, but Spark keeps checkpoint files until the job
        # ends unless told otherwise. A month's run settles six times over
        # 13.8 GB in enrichment alone, and assembly settles five more frames,
        # so without this the work dir carries tens of GB nothing reads.
        .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
        .getOrCreate()
    )


# ============================================
# Final-artifact persistence (local + optional S3 mirror)
# ============================================
def save_node_index(node_index_df: DataFrame, output_path: str) -> None:
    """Write the (node_type, node_id) -> uri identity map beside the .pt.

    hetero_data.pt stores only feature tensors and counts, so without this the
    graph is anonymous: nothing says which real-world entity each row is, which
    blocks joining training labels and attributing predictions.

    Parquet rather than a seventh JSON: production is ~30-50M triples/month, so
    this can reach millions of rows. A single JSON would be hundreds of MB and
    must be parsed whole to resolve one entity, where Parquet supports
    predicate pushdown and matches how the rest of the pipeline stores bulk
    data.

    coalesce(1) keeps the output a single deterministic file. The row content
    is already ordered by _node_index(); leaving it partitioned would spread it
    across part-files whose count and boundaries depend on cluster shape, which
    would make the artifact vary between environments for no benefit. These are
    identity rows (three short strings), not a compute-heavy dataset.
    """
    (
        node_index_df
        .coalesce(1)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    logger.info(f"Saved node index to {output_path}")


def save_final_artifacts(
    config: JobConfig, s3_client, hetero_data, metadata, node_index_df=None,
    spark: SparkSession = None,
) -> Dict[str, Any]:
    """
    Persist the final .pt HeteroData, the six metadata JSON files, and the
    node index.

    Always writes under config.local_work_dir. When an S3 archive
    is configured (config.archive_to_s3), the .pt and metadata are also
    mirrored to S3 via boto3 as a durable, reusable catalog.

    The .pt and the metadata JSONs are driver-side blobs, so they are written
    through the scheme-aware writer: local_work_dir is a bare POSIX path on a
    developer machine but an s3a:// URI on the cluster, and plain open() would
    put the cluster's artifacts on the driver's local disk under a junk
    ./s3a:/... tree while reporting success. `spark` is what makes the URI case
    possible (it carries the Hadoop config) and is required whenever
    local_work_dir is not a plain path. This is the same defect #197 fixed for
    the job manifest; all three writers now share one implementation.

    The node index is written by Spark straight from the executors, so it
    lands wherever the output path points -- a local dir, or object storage
    when local_work_dir is an s3a:// URI (how the cluster runs). It is NOT
    part of the boto3 mirror: it is a distributed Parquet dataset, not a
    single driver-side blob.

    Returns a dict of output locations for the job result/manifest.
    """
    metadata_files = metadata.to_metadata_files()

    # --- Job work dir (always) ---
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA + METADATA")
    logger.info("=" * 80)
    save_pyg_local(hetero_data, config.pyg_output_path, spark=spark)
    local_metadata_dir = derive_metadata_prefix(config.pyg_output_path)
    write_metadata_to_local(metadata_files, local_metadata_dir, spark=spark)

    # Overwritten by every build, so the fixed key always resolves to the most
    # recent one. Written unconditionally rather than only when archiving: it
    # points at the work dir, which is where the artifacts actually are.
    #
    # `--time_period latest` is a legal non-monthly label, and it renders to
    # this very directory -- at which point the period copy already IS the
    # alias and re-writing it would only log the same bytes twice.
    latest_metadata_dir = derive_metadata_prefix(config.latest_pyg_path)
    if latest_metadata_dir != local_metadata_dir:
        write_latest_alias(metadata_files, latest_metadata_dir, spark=spark)

    node_index_dir = derive_node_index_prefix(config.pyg_output_path)
    if node_index_df is not None:
        save_node_index(node_index_df, node_index_dir)

    locations: Dict[str, Any] = {
        "node_index_location": node_index_dir,
        "pyg_location": config.pyg_output_path,
        "metadata_location": local_metadata_dir,
        "latest_metadata_location": latest_metadata_dir,
    }

    # --- S3 mirror (optional) ---
    if config.archive_to_s3:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE: ARCHIVING FINAL ARTIFACTS TO S3")
        logger.info("=" * 80)
        s3_metadata_prefix = derive_metadata_prefix(config.s3_pyg_key)
        save_pyg_to_s3(
            s3_client, hetero_data,
            config.s3_archive_bucket, config.s3_pyg_key,
        )
        write_metadata_to_s3(
            s3_client, metadata_files,
            config.s3_archive_bucket, s3_metadata_prefix,
        )
        locations["pyg_s3_location"] = (
            f"s3://{config.s3_archive_bucket}/{config.s3_pyg_key}"
        )
        locations["metadata_s3_location"] = (
            f"s3://{config.s3_archive_bucket}/{s3_metadata_prefix}"
        )

    return locations


# ============================================
# Execution modes
# ============================================
def execute_full_pipeline(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: full
    Raw source → triples_df → Enrich → Save enriched Parquet (local)
    → Build PyG → Save .pt + metadata (local + optional S3)

    Supports both N-Triples (.nt files) and Turtle Parquet sources
    via config.source_format.
    """
    logger.info("Executing FULL PIPELINE mode")
    logger.info("")

    # Step 1: Load source → distributed triples DataFrame
    logger.info("=" * 80)
    logger.info("PHASE: LOADING SOURCE TRIPLES")
    logger.info("=" * 80)
    start_time = time.time()

    triples_df, initial_count, source_stats = load_source_triples(spark, config)

    load_elapsed = time.time() - start_time
    logger.info(f"Loaded {initial_count:,} triples in {load_elapsed:.1f}s")
    logger.info("")

    # Step 2: Enrich (all on executors)
    enriched_df, enrichment_stats = run_enrichment(
        spark,
        triples_df,
        config.enable_ontology_mapping,
        market_sector_definitions_bucket=config.market_sector_definitions_bucket,
        market_sector_definitions_key=config.market_sector_definitions_key,
        class_mappings=config.class_mappings,
    )

    # Unpersist raw triples — enriched_df is independently cached
    triples_df.unpersist()

    # Step 3: Save enriched Parquet (executors → shared local dir, no driver)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED TRIPLES (Parquet, local)")
    logger.info("=" * 80)
    save_enriched_parquet(
        enriched_df, config.enriched_parquet_path, config.parquet_partitions
    )
    save_dataset_descriptor(config, spark)

    # Step 4: Build PyG (executors → compact tensors → driver)
    hetero_data, metadata, node_index_df = run_pyg_construction(
        spark, enriched_df, config.pyg_config,
        time_period=config.time_period,
        dataset=config.dataset,
        sources=sorted({
            source_label(path, index)
            for index, path in enumerate(config.source_paths)
        }),
    )

    # Step 5: Save final artifacts (local + optional S3)
    locations = save_final_artifacts(
        config, s3_client, hetero_data, metadata, node_index_df, spark=spark
    )

    return {
        "mode": "full",
        "source_format": config.source_format,
        "initial_triples": initial_count,
        # Per-source breakdown of what went IN, alongside the enrichment
        # block's account of what happened during the run. Kept separate
        # because they answer different questions: this one is about the
        # inputs, that one about the pipeline.
        "sources": source_stats,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": config.enriched_parquet_path,
        **locations,
    }


def execute_enrichment_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: enrichment_only
    Raw source → triples_df → Enrich → Save enriched Parquet (local)

    Creates a reusable Parquet artifact for multiple PyG experiments.
    Supports both N-Triples (.nt files) and Turtle Parquet sources
    via config.source_format.
    """
    logger.info("Executing ENRICHMENT ONLY mode")
    logger.info("")

    # Step 1: Load source → distributed triples DataFrame
    logger.info("=" * 80)
    logger.info("PHASE: LOADING SOURCE TRIPLES")
    logger.info("=" * 80)
    start_time = time.time()

    triples_df, initial_count, source_stats = load_source_triples(spark, config)

    load_elapsed = time.time() - start_time
    logger.info(f"Loaded {initial_count:,} triples in {load_elapsed:.1f}s")
    logger.info("")

    # Step 2: Enrich (all on executors)
    enriched_df, enrichment_stats = run_enrichment(
        spark,
        triples_df,
        config.enable_ontology_mapping,
        market_sector_definitions_bucket=config.market_sector_definitions_bucket,
        market_sector_definitions_key=config.market_sector_definitions_key,
        class_mappings=config.class_mappings,
    )

    # Unpersist raw triples
    triples_df.unpersist()

    # Step 3: Save enriched Parquet (executors → shared local dir)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED TRIPLES (Parquet, local)")
    logger.info("=" * 80)
    save_enriched_parquet(
        enriched_df, config.enriched_parquet_path, config.parquet_partitions
    )
    save_dataset_descriptor(config, spark)

    return {
        "mode": "enrichment_only",
        "source_format": config.source_format,
        "initial_triples": initial_count,
        # Per-source breakdown of what went IN, alongside the enrichment
        # block's account of what happened during the run. Kept separate
        # because they answer different questions: this one is about the
        # inputs, that one about the pipeline.
        "sources": source_stats,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": config.enriched_parquet_path,
    }


def execute_parse_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: parse_only
    Raw source → triples_df → count → stop.

    A DIAGNOSTIC MODE. It builds no graph and writes no Parquet, and nothing in
    the ordinary pipeline calls it.

    It exists because the stage that materialises the parse deadlocks between
    the executor JVM and its Python worker, at 168 of 169 tasks, and never
    clears (issue #388). The stall is a race: on 2026-09-07 the same job, from
    the same mirror, with byte-identical Spark properties, stalled on one
    attempt and cleared the same stage in 68.7s on the next. There is no diff
    to read, so the only instrument that says anything is repetition -- and a
    full run costs about three hours to deliver exactly ONE parse attempt.

    This mode is that attempt on its own. The count below is the first action
    the seed leg takes; everything before it is Parquet metadata listing. So a
    trial here reaches the verdict in under two minutes, against the three
    hours a full run needs to reach the same point once.

    THE PLAN MUST STAY IDENTICAL to what a real seed leg submits. That is the
    whole basis for reading a result here as a statement about a real run, and
    it is why this calls ``load_source_triples`` with the same config rather
    than assembling a cheaper frame of its own: same sources, same union, same
    canonicalization, same cache, same count. Anything added between the
    session and the count changes what is being measured.
    """
    logger.info("Executing PARSE ONLY mode (diagnostic -- writes nothing)")
    logger.info("")

    logger.info("=" * 80)
    logger.info("PHASE: LOADING SOURCE TRIPLES")
    logger.info("=" * 80)
    start_time = time.time()

    triples_df, initial_count, source_stats = load_source_triples(spark, config)

    load_elapsed = time.time() - start_time
    logger.info(f"Loaded {initial_count:,} triples in {load_elapsed:.1f}s")
    logger.info("")

    # Released here rather than left to session teardown so a trial that is
    # timed by the loop is not also paying for the cache it leaves behind.
    triples_df.unpersist()

    return {
        "mode": "parse_only",
        "source_format": config.source_format,
        "initial_triples": initial_count,
        # The number the loop compares across trials. Recorded in the result,
        # and so in the manifest, rather than left for something to scrape back
        # out of a log whose format is not a contract.
        "parse_seconds": round(load_elapsed, 1),
        "sources": source_stats,
    }


def execute_pyg_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: pyg_only
    Enriched Parquet (local) → triples_df → Build PyG
    → Save .pt + metadata (local + optional S3)

    Rapid experimentation: loads existing enriched Parquet from the shared
    local working directory, constructs PyG with different configurations.
    Typically 5-10 minutes.

    Note: pyg_only always reads from enriched Parquet previously
    written by this pipeline (schema: subject, predicate, object).
    source_format and turtle_column do not apply in this mode.
    """
    logger.info("Executing PyG ONLY mode")
    logger.info("")

    # Step 1: Load enriched Parquet → distributed DataFrame
    logger.info("=" * 80)
    logger.info("PHASE: LOADING ENRICHED TRIPLES (Parquet, local)")
    logger.info("=" * 80)
    start_time = time.time()

    # The input path, not the derived one: this run may be reading a previous
    # run's published output and writing its graph somewhere else entirely.
    enriched_path = config.enriched_input_path
    triples_df = load_enriched_parquet(spark, enriched_path)
    # DISK_ONLY, not .cache(). The assembly leg runs on a 16g executor, which
    # leaves about 5 GB of storage pool for a frame of 421M rows, so
    # MEMORY_AND_DISK never holds it and the blocks live on disk regardless.
    # What that level does add is a read path: a block fetched from disk gets
    # promoted back into memory, and to make room Spark takes the
    # UnifiedMemoryManager monitor and evicts. On 2026-09-06 it picked a GPU
    # broadcast to evict, and writing one of those out needs the RAPIDS GPU
    # semaphore -- held by the tasks queued behind that same monitor. The
    # executor sat for nine minutes with no GC, no reads and no GPU, until the
    # driver dropped it on a heartbeat timeout. DISK_ONLY skips the promotion,
    # so that eviction never runs here.
    triples_df = triples_df.persist(StorageLevel.DISK_ONLY)
    triple_count = triples_df.count()

    load_elapsed = time.time() - start_time
    logger.info(
        f"Loaded {triple_count:,} enriched triples in {load_elapsed:.1f}s"
    )
    logger.info("")

    # Step 2: Build PyG (executors → compact tensors → driver)
    # What this enriched output was built from. pyg_only never sees
    # source_paths -- it reads Parquet a previous run wrote -- so the descriptor
    # beside that Parquet is the only way the graph can name its own sources.
    descriptor = load_dataset_descriptor(spark, config.enriched_input_path)

    hetero_data, metadata, node_index_df = run_pyg_construction(
        spark, triples_df, config.pyg_config,
        time_period=config.time_period,
        dataset=config.dataset or descriptor.get("dataset", ""),
        sources=descriptor.get("sources"),
    )

    # Step 3: Save final artifacts (local + optional S3)
    locations = save_final_artifacts(
        config, s3_client, hetero_data, metadata, node_index_df, spark=spark
    )

    return {
        "mode": "pyg_only",
        "enriched_parquet_source": enriched_path,
        "enriched_triple_count": triple_count,
        **locations,
    }


# ============================================
# Job manifest and summary
# ============================================
def _write_manifest_bytes(spark: SparkSession, path: str, body: bytes):
    """Write ``body`` to ``path``, honoring the path's URI scheme.

    ``local_work_dir`` may be a bare POSIX path (``/data``) or a URI on shared
    storage (``s3a://bucket/prefix``). A plain ``open()`` treats the latter
    literally and creates a junk ``./s3a:/...`` tree on the driver's local disk
    instead of writing to the object store. Route any non-local URI through the
    Hadoop FileSystem API so it lands on the same filesystem (with the same S3A/IAM
    config) that Spark writes every other artifact to; keep the plain-local path on
    the direct filesystem call.

    Thin wrapper kept for its call site and for the #197 history; the logic now
    lives in spark_jobs.utils.fs_utils.write_bytes, shared with the .pt and
    metadata writers, which turned out to have the identical defect.
    """
    from spark_jobs.utils.fs_utils import write_bytes

    write_bytes(path, body, spark=spark)


def save_job_manifest(
    spark: SparkSession, s3_client, config: JobConfig, result: Dict, elapsed: float
):
    """
    Save a JSON manifest with job metadata to the job's work dir (local path or
    shared-storage URI), and mirror it to a dedicated S3 archive when configured.
    """
    manifest = {
        "job_timestamp": _utcnow().isoformat() + "Z",
        "time_period": config.time_period,
        "mode": config.mode,
        "config": {
            "source_paths": config.source_paths,
            # Recorded because it changes nothing about the graph and
            # everything about how a run's timings should be read: an s3 leg
            # is bounded by the site uplink and a local leg is not, so the two
            # are not comparable without knowing which this was. The root
            # itself is deployment detail and stays out, like the paths above.
            "input_mode": config.input_mode,
            "source_format": config.source_format,
            "turtle_column": config.turtle_column,
            "local_work_dir": config.local_work_dir,
            "enriched_parquet_path": config.enriched_parquet_path,
            "pyg_output_path": config.pyg_output_path,
            "s3_archive_bucket": config.s3_archive_bucket,
            "s3_pyg_key": config.s3_pyg_key,
            # pyg_only and parse_only never reach the enrichment phase, so this
            # job's flag describes something they never ran. Recorded verbatim it
            # reads as a statement about the enriched Parquet being consumed --
            # which it is not, and which is how the 2026-07-29 build's empty
            # class hierarchy got attributed to a flag that had nothing to do
            # with it. null means "not applicable to this mode"; the answer that
            # IS true of the data is in ontology_schema.json
            # (ontology_mapping_enabled), derived from the triples.
            "enable_ontology_mapping": (
                None if config.mode in ("pyg_only", "parse_only")
                else config.enable_ontology_mapping
            ),
            "pyg_config": config.pyg_config,
            "parquet_partitions": config.parquet_partitions,
        },
        "result": _make_json_serializable(result),
        "elapsed_seconds": round(elapsed, 2),
    }
    body = json.dumps(manifest, indent=2, default=str).encode("utf-8")

    filename = (
        f"{config.mode}_{_utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    rel_key = f"manifests/{config.period_partition}/{filename}"

    # Work dir (always) — local path or shared-storage URI (s3a://, hdfs://, ...).
    try:
        work_path = f"{config.local_work_dir.rstrip('/')}/{rel_key}"
        _write_manifest_bytes(spark, work_path, body)
        logger.info(f"Saved job manifest to {work_path}")
    except Exception as e:
        logger.warning(f"Failed to save job manifest to work dir: {e}")

    # S3 mirror (optional)
    if config.archive_to_s3:
        try:
            s3_client.put_object(
                Bucket=config.s3_archive_bucket,
                Key=rel_key,
                Body=body,
                ContentType="application/json",
            )
            logger.info(
                f"Saved job manifest to "
                f"s3://{config.s3_archive_bucket}/{rel_key}"
            )
        except Exception as e:
            logger.warning(f"Failed to save S3 job manifest: {e}")


def _make_json_serializable(obj):
    """Recursively convert non-serializable types for JSON output."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return sorted(list(obj))
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def print_final_banner(config: JobConfig, result: Dict, elapsed: float):
    """Print final job summary banner."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("JOB COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Mode:          {config.mode}")
    logger.info(f"  Source format: {config.source_format}")
    logger.info(f"  Time period:   {config.time_period}")
    logger.info(f"  Duration:      {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    logger.info("")

    if "enriched_parquet_location" in result:
        logger.info(
            f"  Enriched Parquet: {result['enriched_parquet_location']}"
        )
    if "pyg_location" in result:
        logger.info(f"  PyG output:       {result['pyg_location']}")
    if "metadata_location" in result:
        logger.info(f"  Metadata:         {result['metadata_location']}")
    if "pyg_s3_location" in result:
        logger.info(f"  PyG (S3):         {result['pyg_s3_location']}")
    if "metadata_s3_location" in result:
        logger.info(f"  Metadata (S3):    {result['metadata_s3_location']}")

    if "enrichment" in result and isinstance(result["enrichment"], dict):
        stats = result["enrichment"]
        logger.info("")
        logger.info(
            f"  Initial triples:  "
            f"{stats.get('initial_triples', 'N/A'):>12,}"
        )
        # Added and removed are reported separately for the reason given on
        # EnrichmentPipeline.stats: their sum is dominated by source duplicate
        # removal, so the net alone says nothing about what enrichment did.
        # .get() with a default keeps a manifest written before these existed
        # readable rather than raising here.
        logger.info(
            f"  Enrichment added: "
            f"{stats.get('enrichment_added', 'N/A'):>+12,}"
        )
        logger.info(
            f"  Duplicates gone:  "
            f"{-stats.get('duplicates_removed', 0):>+12,}"
        )
        logger.info(
            f"  Final triples:    "
            f"{stats.get('final_triples', 'N/A'):>12,}"
        )
        logger.info(
            f"  Net change:       "
            f"{stats.get('total_enrichment', 'N/A'):>+12,}"
        )

    logger.info("")
    logger.info("=" * 80)


# ============================================
# Work dir occupancy preflight
# ============================================
def check_work_dir_occupancy(config: JobConfig, spark: SparkSession) -> None:
    """Refuse to start when the artifacts this run would write already exist.

    Two partition schemes decide where a run writes and neither carries run
    identity: the outer path is whatever ``--local_work_dir`` was given, the
    inner one is derived from ``--time_period``, which describes the data. Two
    runs sharing that pair write byte-identical paths. What keeps runs apart
    today is a convention -- the launcher mints a fresh timestamped work dir --
    that is load-bearing and unenforced.

    The artifacts do not even fail the same way when it lapses. Spark overwrites
    the enriched Parquet, the ``.pt`` and its metadata are replaced at fixed
    keys, and the manifests accumulate -- so the run that survives is a mixture
    and the provenance beside it describes both.

    Checked per mode, against what that mode WRITES. ``pyg_only`` reads enriched
    Parquet and is meant to find it -- several ``pyg_only`` legs over one seed is
    the ordinary experiment loop -- so its input is never occupancy.

    Args:
        config: Parsed job configuration.
        spark: Active SparkSession, needed to resolve non-local URIs.

    Raises:
        FileExistsError: An artifact this mode writes is already present and
            ``--allow_overwrite`` was not given.
    """
    from spark_jobs.utils.fs_utils import join_path, path_exists

    occupied: List[str] = []

    if config.mode in ("full", "enrichment_only"):
        # _SUCCESS, not the directory: Spark writes the marker last, so its
        # presence means a previous run finished. A directory that exists
        # without it is the debris of a run that died mid-write, which is not
        # something worth refusing to overwrite.
        if path_exists(
            join_path(config.enriched_parquet_path, "_SUCCESS"), spark=spark
        ):
            occupied.append(
                f"enriched Parquet at {config.enriched_parquet_path}"
            )

    if config.mode in ("full", "pyg_only"):
        # The period copy only. latest_pyg_path is a fixed-key alias whose whole
        # job is to name the newest build, so it is replaced by design and is
        # never occupancy. The .pt stands in for the metadata and node index
        # beside it: they are written after it, so if it is there they are too.
        if path_exists(config.pyg_output_path, spark=spark):
            occupied.append(f"PyG graph at {config.pyg_output_path}")

    if not occupied:
        return

    if config.allow_overwrite:
        for item in occupied:
            logger.warning(f"--allow_overwrite: replacing {item}")
        logger.warning("")
        return

    raise FileExistsError(
        f"--local_work_dir {config.local_work_dir} already holds a finished "
        f"run for --time_period {config.time_period}: "
        + "; ".join(occupied)
        + ". Writing here replaces those artifacts in place and leaves a "
        "manifest history describing both runs. Point --local_work_dir at a "
        "fresh directory, or pass --allow_overwrite true to replace them "
        "deliberately."
    )


# ============================================
# Main entry point
# ============================================
def main():
    """
    Main entry point for the Spark job.

    Parses arguments, initializes Spark (and an S3 client when needed),
    dispatches to the appropriate execution mode.
    """
    job_start_time = time.time()

    logger.info("=" * 80)
    logger.info("PyTorch Geometric Knowledge Graph Builder")
    logger.info("=" * 80)
    logger.info(f"Start time: {_utcnow().isoformat()}Z")
    logger.info("")

    # Parse and validate arguments
    try:
        config = parse_args()
    except (ValueError, ImportError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    logger.info(f"Configuration: {config}")
    logger.info("")

    # Initialize Spark. Create an S3 client only when the job needs S3
    # (archiving final artifacts or reading external definitions from S3).
    spark = get_spark_session()
    # Where settle() puts its copies, in enrichment and in assembly alike. Under
    # the work dir so it shares the run's storage and lifetime -- the shared
    # mount on a cluster run, the same object store when the work dir is an
    # s3a:// URI. Set here, before the mode dispatch, so every mode gets one:
    # without a dir set settle() falls back to localCheckpoint, which is what
    # let one evicted executor abort the 2026-08-29 run at 47 minutes.
    spark.sparkContext.setCheckpointDir(f"{config.local_work_dir}/checkpoints")
    needs_s3 = bool(
        config.archive_to_s3 or config.market_sector_definitions_bucket
    )
    s3_client = boto3.client("s3") if needs_s3 else None

    logger.info(
        "Initialized SparkSession"
        + (" and S3 client" if s3_client is not None else "")
    )
    logger.info("")

    # Execute pipeline
    try:
        # Before the handler, so a run that would overwrite a finished one
        # costs seconds rather than discovering it hours in -- or never.
        check_work_dir_occupancy(config, spark)

        mode_handlers = {
            "full": execute_full_pipeline,
            "enrichment_only": execute_enrichment_only,
            "parse_only": execute_parse_only,
            "pyg_only": execute_pyg_only,
        }

        handler = mode_handlers[config.mode]
        result = handler(config, spark, s3_client)

    except FileExistsError as e:
        # No traceback: this is a refusal, not a crash, and the message is
        # the whole of what the operator needs.
        logger.error(str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    # Save manifest and print summary
    elapsed = time.time() - job_start_time
    save_job_manifest(spark, s3_client, config, result, elapsed)
    print_final_banner(config, result, elapsed)

    logger.info("Done.")
    return result


if __name__ == "__main__":
    main()