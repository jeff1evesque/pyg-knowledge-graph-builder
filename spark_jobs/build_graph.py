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
    --local_work_dir:          base dir on the shared filesystem for
                               enriched Parquet + local final artifacts
    --s3_archive_bucket:       optional S3 bucket to mirror final artifacts
    --s3_pyg_key:              optional S3 key for the .pt (metadata prefix
                               is derived from it)
    --enable_ontology_mapping: true | false (default: true)
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
"""
import sys
import json
import logging
import re
import time
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List

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

from spark_jobs.pyg_builder.metadata_writer import (
    write_metadata_to_s3,
    write_metadata_to_local,
    derive_metadata_prefix,
    derive_node_index_prefix,
)

# PyG builder — imported conditionally since enrichment_only mode doesn't need it
_pyg_builder_available = False
try:
    from spark_jobs.pyg_builder.constructor import build_hetero_data

    _pyg_builder_available = True
except ImportError:
    pass

# ============================================
# Logging setup
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_graph")

_PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})$")


def period_partition(time_period: str) -> str:
    """Render a ``YYYY-MM`` period label as Hive-style partition directories.

        "2024-12" -> "year=2024/month=12"

    Spark's partition discovery turns ``key=value`` directory names into real
    columns, so reading a parent directory across several periods yields
    ``year``/``month`` to filter and prune on. An opaque ``2024-12`` directory
    gives it nothing -- the period is only recoverable by parsing the path.

    ``time_period`` is a free-form label (nothing validates its shape, and
    ``--time_period`` accepts any string), so a value that is not ``YYYY-MM``
    is passed through as a single segment rather than forced into a partition
    shape that would misdescribe what it holds. Only ``day=`` is deliberately
    absent: ``time_period`` is monthly, so a day level would carry one value
    per month -- path depth with no pruning benefit.
    """
    match = _PERIOD_RE.match(time_period)
    if not match:
        logger.warning(
            f"time_period '{time_period}' is not YYYY-MM; writing it as a "
            f"single path segment with no year=/month= partitioning"
        )
        return time_period

    year, month = match.groups()
    return f"year={year}/month={month}"


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
VALID_MODES = {"full", "enrichment_only", "pyg_only"}
VALID_SOURCE_FORMATS = {"ntriples", "turtle_parquet"}

TRIPLES_SCHEMA = StructType([
    StructField("subject", StringType(), nullable=False),
    StructField("predicate", StringType(), nullable=False),
    StructField("object", StringType(), nullable=False),
])

# Default number of Parquet output partitions.
# Targets ~128 MB per partition for 30-50M triples (~2-4 GB total Parquet).
DEFAULT_PARQUET_PARTITIONS = 200

# ============================================
# Source coverage: which SEC feed this job handles
# ============================================
# The archive holds eight SEC feeds under raw/source=sec/. This pipeline
# handles exactly one of them, and the restriction has until now been
# incidental — a property of whichever prefix the caller happened to pass —
# rather than stated anywhere. Measured against the archive:
#
#   feed=filings             218 objects,   2.4 GB   RDF (rdf_turtle column)
#   feed=filings_documents   712,351 objects, 150 GB  raw filing documents
#   feed=filing-detail       364 objects            crawler telemetry columns
#   feed=litigation            1 object              only, no Turtle column
#   feed=press-release         2 objects             (parsed / parse-start /
#   feed=speeches              2 objects              failures / user-agent)
#   feed=statements            2 objects
#   feed=testimony             1 object
#
# filings_documents is not Parquet at all and not one format either: sampled
# over 400,000 keys it is 233,759 .xml, 160,196 .zip (upstream now packages
# each filing's documents together with its XBRL members), plus .txt, .htm,
# .pdf and images. Nothing there is RDF.
#
# So a run pointed at the SEC source root does not under-cover quietly — the
# six telemetry feeds have no Turtle column at all and resolve_turtle_column
# raises, while filings_documents is not something the Parquet reader can open.
# It fails, but it fails deep in the loader with a column-name error that says
# nothing about feeds. This turns that into a statement of scope at the point
# the job is configured.
#
# The one thing NOT guarded here, because storage says it is already resolved:
# the retired crawler also wrote into feed=filings itself, on a 10-column
# schema whose RDF column was named `triples` rather than `rdf_turtle`. All 218
# objects now carry the same 29-column scraper schema, so the migration ran to
# completion and no mixed-schema read is possible. Worth re-checking if that
# object count ever jumps backwards.
#
# Keyed on the partition name rather than on "sec" anywhere in the path: a
# local fixture directory called /data/sec/ is not the archive convention and
# is none of this check's business.
SEC_SOURCE_PARTITION = "source=sec"
SEC_HANDLED_FEED = "feed=filings"
# filings_documents starts with the handled feed's name, so a plain substring
# test would accept it. It is a different feed and carries no RDF.
SEC_UNHANDLED_FEEDS = (
    "feed=filings_documents", "feed=filing-detail", "feed=litigation",
    "feed=press-release", "feed=speeches", "feed=statements",
    "feed=testimony",
)


def assert_sec_paths_name_the_handled_feed(source_paths: List[str]) -> None:
    """Every SEC source path must name feed=filings explicitly.

    Raises:
        ValueError: if a path under the SEC source partition names a feed this
            pipeline does not handle, or names no feed at all.
    """
    for path in source_paths:
        if SEC_SOURCE_PARTITION not in path:
            continue
        unhandled = [f for f in SEC_UNHANDLED_FEEDS if f in path]
        if unhandled:
            raise ValueError(
                f"source path names an unhandled SEC feed {unhandled[0]!r}: "
                f"{path}. Only {SEC_HANDLED_FEED!r} carries RDF; the others "
                f"hold crawler telemetry or raw XML and no pipeline step reads "
                f"them."
            )
        if SEC_HANDLED_FEED not in path:
            raise ValueError(
                f"SEC source path does not name a feed: {path}. This pipeline "
                f"handles {SEC_HANDLED_FEED!r} only, and the SEC source "
                f"partition holds seven other feeds it cannot read. Point at "
                f"raw/{SEC_SOURCE_PARTITION}/{SEC_HANDLED_FEED}/... instead."
            )


# ============================================
# Parameter parsing
# ============================================
DEFAULT_PYG_FILENAME = "hetero_data.pt"


class JobConfig:
    """Parsed and validated job configuration."""

    def __init__(self, args: Dict[str, str]):
        self.mode = args.get("mode", "full").lower().strip()

        # Shared local working directory (must be on a filesystem visible
        # to every Spark worker, e.g. an NFS mount). Holds the interim
        # enriched Parquet and the local copy of the final artifacts.
        self.local_work_dir = (args.get("local_work_dir") or "").rstrip("/")

        # Optional S3 archive for the FINAL artifacts (.pt + metadata).
        # When s3_archive_bucket is set, the final artifacts are mirrored
        # to S3 via boto3 in addition to being written locally.
        self.s3_archive_bucket = args.get("s3_archive_bucket", "") or ""
        self.s3_pyg_key = args.get("s3_pyg_key", "") or ""

        # Optional external definitions (small CSV read from S3 via boto3)
        self.market_sector_definitions_bucket = args.get(
            "market_sector_definitions_bucket", ""
        )
        self.market_sector_definitions_key = args.get(
            "market_sector_definitions_key", ""
        )

        # Source path(s)/URI(s): a single value or a comma-separated list.
        # Each entry is a local directory or an s3a:// URI. Whitespace
        # around each entry is stripped. Required for full/enrichment_only.
        # Examples:
        #   local:    "/data/rdf/monthly/2024-12/"
        #   s3a:      "s3a://my-data-lake/rdf/monthly/2024-12/"
        #   multiple: "/data/sec/filings/,/data/sec/litigations/"
        raw_sources = args.get("source_paths", "") or ""
        self.source_paths: List[str] = [
            p.strip() for p in raw_sources.split(",") if p.strip()
        ]

        # Source format:
        #   "ntriples":       one triple per line in .nt files
        #   "turtle_parquet": self-contained Turtle blobs in a Parquet column
        self.source_format = (
            args.get("source_format", "ntriples").lower().strip()
        )

        # Column name containing Turtle strings when
        # source_format=turtle_parquet. Empty means auto-detect per source
        # (see resolve_turtle_column), so one run can span sources whose
        # scrapers named the column differently; set it to force one name.
        self.turtle_column = args.get("turtle_column", "")

        # Optional
        self.enable_ontology_mapping = (
            args.get("enable_ontology_mapping", "true").lower() == "true"
        )
        self.time_period = args.get("time_period") or datetime.now().strftime(
            "%Y-%m"
        )
        self.pyg_config = self._parse_pyg_config(args.get("pyg_config", ""))
        self.class_mappings = self._parse_json_arg(
            args.get("class_mappings", ""), "class_mappings"
        )
        self.parquet_partitions = int(
            args.get("parquet_partitions", str(DEFAULT_PARQUET_PARTITIONS))
        )

        # Local filename for the .pt (override for experiment variants,
        # e.g. "hetero_data_512d.pt"). Metadata directory is derived from it.
        self.pyg_filename = (
            args.get("pyg_filename") or DEFAULT_PYG_FILENAME
        )

        # Hive-style partition directories for the period, so a read across
        # several periods gets year/month as real columns to prune on.
        #
        # Everything a period produces stays under one partition directory,
        # so a build can be copied, archived or deleted as a unit.
        self.period_partition = period_partition(self.time_period)

        # Derived local paths (under local_work_dir)
        self.enriched_parquet_path = (
            f"{self.local_work_dir}/enriched/"
            f"{self.period_partition}/triples"
        )
        self.pyg_output_path = (
            f"{self.local_work_dir}/pyg/"
            f"{self.period_partition}/{self.pyg_filename}"
        )

        # Derived S3 key for the archived .pt (only used when archiving)
        if self.s3_archive_bucket and not self.s3_pyg_key:
            self.s3_pyg_key = (
                f"pyg/{self.period_partition}/{self.pyg_filename}"
            )

        self._validate()

    @property
    def archive_to_s3(self) -> bool:
        """Whether final artifacts should also be mirrored to S3."""
        return bool(self.s3_archive_bucket)

    def _parse_json_arg(self, raw: str, name: str) -> Dict[str, Any]:
        """Parse a JSON CLI argument, or a path to a JSON file."""
        raw = (raw or "").strip()
        if not raw:
            return {}
        if not raw.startswith("{"):
            try:
                raw = Path(raw).read_text()
            except OSError as e:
                logger.warning(f"Could not read {name} file: {e}")
                return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse {name} JSON, ignoring: {e}")
            return {}

    def _parse_pyg_config(self, config_str: str) -> Dict[str, Any]:
        if not config_str or config_str.strip() == "":
            return {}
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Could not parse pyg_config JSON, using defaults: {e}"
            )
            return {}

    def _validate(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                f"Must be one of: {', '.join(VALID_MODES)}"
            )
        if not self.local_work_dir:
            raise ValueError("local_work_dir is required")

        if self.source_format not in VALID_SOURCE_FORMATS:
            raise ValueError(
                f"Invalid source_format '{self.source_format}'. "
                f"Must be one of: {', '.join(VALID_SOURCE_FORMATS)}"
            )

        if self.mode in ("full", "enrichment_only"):
            if not self.source_paths:
                raise ValueError(
                    f"source_paths is required for mode '{self.mode}'"
                )
            assert_sec_paths_name_the_handled_feed(self.source_paths)
        if self.mode in ("full", "pyg_only"):
            if not _pyg_builder_available:
                raise ImportError(
                    f"PyG builder not available. Mode '{self.mode}' requires "
                    "spark_jobs.pyg_builder.constructor."
                )
        if self.parquet_partitions < 1:
            raise ValueError("parquet_partitions must be >= 1")

    def __repr__(self):
        return (
            f"JobConfig(mode={self.mode}, "
            f"source_paths={self.source_paths}, "
            f"source_format={self.source_format}, "
            f"turtle_column={self.turtle_column}, "
            f"local_work_dir={self.local_work_dir}, "
            f"enriched_parquet_path={self.enriched_parquet_path}, "
            f"pyg_output_path={self.pyg_output_path}, "
            f"s3_archive_bucket={self.s3_archive_bucket or '(none)'}, "
            f"s3_pyg_key={self.s3_pyg_key or '(none)'}, "
            f"ontology_mapping={self.enable_ontology_mapping}, "
            f"parquet_partitions={self.parquet_partitions})"
        )


def parse_args() -> JobConfig:
    """Parse job arguments from the command line (spark-submit)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Geometric Knowledge Graph Builder"
    )
    parser.add_argument("--mode", choices=list(VALID_MODES), default="full")
    parser.add_argument(
        "--source_paths",
        default="",
        help="Comma-separated source path(s)/URI(s): local dirs or s3a://...",
    )
    parser.add_argument(
        "--local_work_dir",
        required=True,
        help="Shared working directory (visible to all workers) for enriched "
        "Parquet and local final artifacts",
    )
    parser.add_argument(
        "--s3_archive_bucket",
        default="",
        help="Optional S3 bucket to mirror final artifacts (.pt + metadata)",
    )
    parser.add_argument(
        "--s3_pyg_key",
        default="",
        help="Optional S3 key for the archived .pt (metadata prefix derived "
        "from it); defaults to pyg/year=YYYY/month=MM/<pyg_filename>",
    )
    parser.add_argument("--pyg_filename", default=DEFAULT_PYG_FILENAME)
    parser.add_argument("--enable_ontology_mapping", default="true")
    parser.add_argument("--time_period", default="")
    parser.add_argument("--pyg_config", default="")
    parser.add_argument("--class_mappings", default="")
    parser.add_argument(
        "--parquet_partitions", default=str(DEFAULT_PARQUET_PARTITIONS)
    )
    parser.add_argument(
        "--source_format",
        choices=list(VALID_SOURCE_FORMATS),
        default="ntriples",
    )
    parser.add_argument("--turtle_column", default="")
    parser.add_argument("--market_sector_definitions_bucket", default="")
    parser.add_argument("--market_sector_definitions_key", default="")

    parsed = parser.parse_args()
    return JobConfig(vars(parsed))


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


def deterministic_bnode_labels(graph, rounds: int = 3) -> dict:
    """Map every blank node in an rdflib graph to a content-derived label.

    rdflib mints a FRESH RANDOM identifier for each blank node on every parse
    (``_:nd0dc1ff...`` one run, ``_:n54d9d7...`` the next). Those labels flow
    into the triples as subject/object values and are later hashed into feature
    vector slots, so an identical input produced a different graph on every run
    — the ``jolts_*`` non-determinism (JOLTS models measurements as blank nodes).
    Replacing the random label with one derived from the blank node's *content*
    makes the parse reproducible.

    A blank node's signature is the sorted set of its outgoing
    ``(predicate, object)`` and incoming ``(subject, predicate)`` edges. Nested
    blank-node references would be circular, so they contribute the neighbour's
    signature from the previous round and the whole thing is refined ``rounds``
    times (standard colour-refinement). Three rounds distinguishes any nesting
    depth this data actually uses; more only matters for pathologically
    symmetric graphs.

    Structurally identical blank nodes hash alike, and collapsing them would
    silently merge two distinct measurements, so a group of colliding nodes gets
    a positional suffix. Which member takes which index depends on set iteration
    order and is therefore NOT stable — but it does not need to be: identical
    signatures mean the nodes are interchangeable, so any assignment yields the
    same triple multiset.
    """
    import hashlib

    from rdflib import BNode

    bnodes = {
        term
        for triple in graph
        for term in (triple[0], triple[2])
        if isinstance(term, BNode)
    }
    if not bnodes:
        return {}

    def _ref(term, labels):
        return labels[term] if isinstance(term, BNode) else str(term)

    labels = {b: "" for b in bnodes}
    for _ in range(rounds):
        labels = {
            b: hashlib.sha1(
                "\n".join(
                    sorted(
                        f">|{p}|{_ref(o, labels)}"
                        for p, o in graph.predicate_objects(b)
                    )
                    + sorted(
                        f"<|{p}|{_ref(s, labels)}"
                        for s, p in graph.subject_predicates(b)
                    )
                ).encode("utf-8")
            ).hexdigest()
            for b in bnodes
        }

    by_signature: dict = {}
    for b in bnodes:
        by_signature.setdefault(labels[b], []).append(b)

    return {
        b: (f"{sig[:32]}_{i}" if len(group) > 1 else sig[:32])
        for sig, group in by_signature.items()
        for i, b in enumerate(group)
    }


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
        object: string), one row per triple. Compatible with all
        downstream pipeline steps.

    Raises:
        ValueError: If no usable Turtle column is found in the Parquet schema.
    """
    from pyspark.sql.types import ArrayType

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
    # UDF: parse one Turtle blob → list of (s, p, o) structs
    # ============================================
    # rdflib is used here because Turtle has prefix resolution,
    # predicate lists (;), object lists (,), and typed literals that
    # cannot be correctly parsed with regex. The UDF runs per-row on
    # executors — each blob is small (~60 triples), so the per-row
    # overhead is acceptable. All downstream operations remain
    # pure Spark expressions.
    #
    # Object normalization matches the pipeline convention established
    # in load_ntriples_to_dataframe():
    #   - URIRef  → str(uri)
    #   - Literal → str(literal.toPython()) for numerics,
    #               str(literal) for strings (lexical form, no ^^datatype)
    #   - BNode   → f"_:{content_hash}" (see deterministic_bnode_labels;
    #               rdflib's own labels are random per parse, which made the
    #               whole pipeline non-reproducible)
    #
    # Malformed blobs return an empty list so that parse errors skip
    # the row rather than failing the job. The zero-triple contribution
    # is visible in the final triple count logged after loading.

    triple_schema = ArrayType(
        StructType([
            StructField("subject", StringType(), nullable=False),
            StructField("predicate", StringType(), nullable=False),
            StructField("object", StringType(), nullable=False),
            # The literal's declared ^^<datatype>, "" for URIs/bnodes/plain
            # literals. Internal to this UDF and dropped below once the
            # observation markers are built -- the frame this loader returns
            # is the canonical 3-column one.
            StructField("object_datatype", StringType(), nullable=False),
        ])
    )

    @F.udf(returnType=triple_schema)
    def parse_turtle_blob(turtle_str):
        """
        Parse a self-contained Turtle string into a list of
        (subject, predicate, object) dicts with fully-expanded URIs.
        """
        if not turtle_str or not turtle_str.strip():
            return []

        try:
            from rdflib import Graph, URIRef, Literal, BNode

            g = Graph()
            g.parse(data=turtle_str, format="turtle")

            # rdflib's blank-node labels are random per parse, so emitting them
            # verbatim made the whole pipeline non-reproducible (they end up
            # hashed into feature slots). Relabel by content instead.
            bnode_labels = deterministic_bnode_labels(g)

            triples = []
            for s, p, o in g:
                # Subject
                if isinstance(s, URIRef):
                    subj = str(s)
                elif isinstance(s, BNode):
                    subj = f"_:{bnode_labels[s]}"
                else:
                    # Skip unexpected subject types (should not occur
                    # in well-formed Turtle)
                    continue

                # Predicate — always a URIRef in valid RDF
                pred = str(p)

                # Object — normalize to pipeline convention
                datatype = ""
                if isinstance(o, URIRef):
                    obj = str(o)
                elif isinstance(o, BNode):
                    obj = f"_:{bnode_labels[o]}"
                elif isinstance(o, Literal):
                    # toPython() returns a Python native type for
                    # numeric/date XSD types; str() of that matches
                    # what the pipeline expects for numeric literals.
                    # For plain strings, str(literal) returns the
                    # lexical form without ^^datatype or @lang.
                    py_val = o.toPython()
                    obj = str(py_val)
                    # ...which is exactly why the datatype has to be taken
                    # off the Literal here: toPython() is where the source's
                    # ^^<xsd:...> declaration stops existing, and it is the
                    # only evidence of a literal property's rdfs:range.
                    if o.datatype is not None:
                        datatype = str(o.datatype)
                else:
                    obj = str(o)

                triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "object_datatype": datatype,
                })

            return triples

        except Exception:
            # Malformed Turtle — skip this row silently.
            # The zero-triple contribution will be visible in the
            # final triple count logged after loading.
            return []

    # ============================================
    # Apply UDF and explode into one row per triple
    # ============================================
    parsed_df = turtle_df.withColumn(
        "_triples", parse_turtle_blob(F.col(turtle_column))
    )

    exploded_df = parsed_df.select(
        F.explode(F.col("_triples")).alias("_triple")
    )

    parsed_df = exploded_df.select(
        F.col("_triple.subject").alias("subject"),
        F.col("_triple.predicate").alias("predicate"),
        F.col("_triple.object").alias("object"),
        F.col("_triple.object_datatype").alias("object_datatype"),
    ).filter(
        (F.col("subject") != "")
        & (F.col("predicate") != "")
    )

    triples_df = parsed_df.select(
        "subject", "predicate", "object"
    ).unionByName(literal_datatype_observations(parsed_df))

    return triples_df


# ============================================
# Source loader dispatcher
# ============================================
def load_source_triples(
    spark: SparkSession,
    config: "JobConfig",
) -> Tuple[DataFrame, int]:
    """
    Load raw source data into a triples DataFrame.

    Dispatches to the correct loader based on config.source_format:
      - "ntriples":       load_ntriples_to_dataframe()
      - "turtle_parquet": load_turtle_parquet_to_dataframe()

    Supports multiple source paths (local directories or s3a:// URIs).
    When more than one path is provided, each is loaded independently
    and the results are unioned into a single triples DataFrame before
    caching. Duplicates across paths are not deduplicated here —
    the enrichment pipeline's left_anti join pattern handles any
    duplicate triples naturally.

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

    logger.info(
        f"Source format: {config.source_format}, "
        f"{len(source_paths)} path(s)"
    )
    for path in source_paths:
        logger.info(f"  {path}")

    loaded: List[DataFrame] = []

    for source_path in source_paths:
        if config.source_format == "ntriples":
            df = load_ntriples_to_dataframe(spark, source_path)
        else:
            df = load_turtle_parquet_to_dataframe(
                spark,
                source_path,
                turtle_column=config.turtle_column,
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

    # Collapse identifiers upstream spells two ways onto one, before anything
    # reads the frame. See canonicalization.py: this adds no triple and drops
    # none, it only makes the two spellings of one entity the same URI. Done
    # here rather than in a pipeline phase because the enriched Parquet is
    # written from this frame, so a later repair would leave `pyg_only` runs
    # reading the unrepaired shape.
    triples_df = canonicalize_source_triples(triples_df)

    # Cache and materialize once — all downstream steps read from cache.
    # For turtle_parquet, this also ensures the rdflib UDF runs exactly
    # once across all prefixes rather than being re-triggered by each
    # downstream action.
    triples_df = triples_df.cache()
    count = triples_df.count()

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

    logger.info(
        f"Loaded {count:,} triples total across "
        f"{len(source_paths)} path(s)"
    )
    return triples_df, count


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

    Serializes using torch.save to a BytesIO buffer, then uploads
    via S3 multipart upload to avoid holding multiple copies in memory.

    This runs on the driver — only compact tensors are in memory.

    Args:
        s3_client: Boto3 S3 client
        hetero_data: PyG HeteroData object
        bucket: S3 bucket name
        key: S3 key
    """
    import torch

    # Serialize to buffer — single copy in memory alongside HeteroData
    buffer = io.BytesIO()
    torch.save(hetero_data, buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 * 1024)
    buffer.seek(0)

    # Upload using the buffer directly (no .getvalue() copy)
    s3_client.upload_fileobj(
        Fileobj=buffer,
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

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

    The two destinations are handled differently on purpose:

    * A local path keeps the direct ``torch.save(obj, path)`` stream. torch
      writes incrementally to the file handle, so peak memory stays at the
      HeteroData itself.
    * A URI has no file handle to stream to, so the graph is serialized to a
      buffer first and the bytes handed to the Hadoop writer — the same shape as
      save_pyg_to_s3 above. This costs a second in-memory copy of the (compact)
      tensors, which is simply what writing to object storage costs; it is the
      same trade the S3 archive path already makes.

    Buffering unconditionally would be simpler, but it would impose that second
    copy on every local run — including the e2e suite and any single-machine
    build — to serve a case those runs never hit.

    Args:
        hetero_data: PyG HeteroData object
        local_path: Destination for the .pt file — bare path or URI
        spark: Active SparkSession; required when local_path is a non-local URI
    """
    import os

    import torch

    from spark_jobs.utils.fs_utils import (
        is_local_path,
        local_filesystem_path,
        write_bytes,
    )

    if is_local_path(local_path):
        path = local_filesystem_path(local_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(hetero_data, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Saved PyG HeteroData ({size_mb:.2f} MB) to {path}")
        return

    buffer = io.BytesIO()
    torch.save(hetero_data, buffer)
    size_mb = buffer.tell() / (1024 * 1024)

    # getbuffer() rather than getvalue(): a memoryview onto the buffer instead of
    # yet another full copy of a graph that may be gigabytes.
    write_bytes(local_path, buffer.getbuffer(), spark=spark)

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
    if not _pyg_builder_available:
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
        spark, triples_df, config, time_period=time_period
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
    """
    return (
        SparkSession.builder.appName("PyG-Knowledge-Graph-Builder")
        .config("spark.rapids.sql.enabled", "true")
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

    node_index_dir = derive_node_index_prefix(config.pyg_output_path)
    if node_index_df is not None:
        save_node_index(node_index_df, node_index_dir)

    locations: Dict[str, Any] = {
        "node_index_location": node_index_dir,
        "pyg_location": config.pyg_output_path,
        "metadata_location": local_metadata_dir,
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

    triples_df, initial_count = load_source_triples(spark, config)

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

    # Step 4: Build PyG (executors → compact tensors → driver)
    hetero_data, metadata, node_index_df = run_pyg_construction(
        spark, enriched_df, config.pyg_config,
        time_period=config.time_period,
    )

    # Step 5: Save final artifacts (local + optional S3)
    locations = save_final_artifacts(
        config, s3_client, hetero_data, metadata, node_index_df, spark=spark
    )

    return {
        "mode": "full",
        "source_format": config.source_format,
        "initial_triples": initial_count,
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

    triples_df, initial_count = load_source_triples(spark, config)

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

    return {
        "mode": "enrichment_only",
        "source_format": config.source_format,
        "initial_triples": initial_count,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": config.enriched_parquet_path,
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

    enriched_path = config.enriched_parquet_path
    triples_df = load_enriched_parquet(spark, enriched_path)
    triples_df = triples_df.cache()
    triple_count = triples_df.count()

    load_elapsed = time.time() - start_time
    logger.info(
        f"Loaded {triple_count:,} enriched triples in {load_elapsed:.1f}s"
    )
    logger.info("")

    # Step 2: Build PyG (executors → compact tensors → driver)
    hetero_data, metadata, node_index_df = run_pyg_construction(
        spark, triples_df, config.pyg_config,
        time_period=config.time_period,
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
            "source_format": config.source_format,
            "turtle_column": config.turtle_column,
            "local_work_dir": config.local_work_dir,
            "enriched_parquet_path": config.enriched_parquet_path,
            "pyg_output_path": config.pyg_output_path,
            "s3_archive_bucket": config.s3_archive_bucket,
            "s3_pyg_key": config.s3_pyg_key,
            # pyg_only never reaches the enrichment phase, so this job's flag
            # describes something it never ran. Recorded verbatim it reads as
            # a statement about the enriched Parquet being consumed -- which
            # it is not, and which is how the 2026-07-29 build's empty class
            # hierarchy got attributed to a flag that had nothing to do with
            # it. null means "not applicable to this mode"; the answer that
            # IS true of the data is in ontology_schema.json
            # (ontology_mapping_enabled), derived from the triples.
            "enable_ontology_mapping": (
                None if config.mode == "pyg_only"
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
        logger.info(
            f"  Final triples:    "
            f"{stats.get('final_triples', 'N/A'):>12,}"
        )
        logger.info(
            f"  Total enrichment: "
            f"{stats.get('total_enrichment', 'N/A'):>12,}"
        )

    logger.info("")
    logger.info("=" * 80)


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
        mode_handlers = {
            "full": execute_full_pipeline,
            "enrichment_only": execute_enrichment_only,
            "pyg_only": execute_pyg_only,
        }

        handler = mode_handlers[config.mode]
        result = handler(config, spark, s3_client)

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