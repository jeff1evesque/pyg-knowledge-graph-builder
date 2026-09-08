"""
Every way a triple gets into the job, and what the job records about where it
came from.

Three loaders behind one dispatcher: ``load_ntriples_to_dataframe`` for ``.nt``
files, ``load_turtle_parquet_to_dataframe`` for the Turtle blobs the archive
actually holds, and ``load_source_triples`` which picks between them per source
path, stamps each row with the source it came from, and unions the result.

The per-source stamp is not bookkeeping. ``source_label`` reads the source out
of path fragments, so the same run reports the same per-source triple counts
whether it read the bucket directly or a staged local mirror of it -- which is
what makes the two input modes comparable at all.

The Turtle parsing itself is NOT here: it runs on executors and lives in
``turtle.py``, which is constrained in what it may import. This module is
driver-side Spark and has no such limit.
"""
import logging
from typing import List, Tuple

from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

from spark_jobs.utils.spark_rdf_utils import literal_datatype_observations
from spark_jobs.utils.canonicalization import canonicalize_source_triples
from spark_jobs.graph.turtle import PARSE_BATCH_ROWS, turtle_batches_to_arrow
from spark_jobs.graph.config import JobConfig

# The job's logger, not this module's -- see the note in config.py.
logger = logging.getLogger("build_graph")


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
    config: JobConfig,
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
