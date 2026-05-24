"""
PyTorch Geometric Knowledge Graph Builder - Main Glue Job Entry Point

AWS Glue job that orchestrates the complete pipeline:
  Raw RDF (S3) → PySpark triples_df → Enrich → Build PyG HeteroData → Save to S3

Supports two source formats:
  - ntriples:       One triple per line in .nt files (default)
  - turtle_parquet: Self-contained Turtle blobs in a Parquet column

Supports three execution modes:
  - full:            End-to-end RDF enrichment + PyG graph construction
  - enrichment_only: Create reusable enriched Parquet artifacts (save to S3)
  - pyg_only:        Build PyG graph from existing enriched Parquet artifacts

Usage (AWS Glue):
    Parameters:
        --mode:                    full | enrichment_only | pyg_only
        --source_bucket:           S3 bucket containing raw RDF files
        --source_prefix:           S3 prefix for raw RDF files
        --output_bucket:           S3 bucket for outputs
        --enriched_parquet_prefix: S3 prefix for enriched Parquet artifact
        --pyg_output_key:          S3 key for PyG HeteroData output (.pt)
        --enable_ontology_mapping: true | false (default: false)
        --time_period:             Time period label (e.g., "2024-12") for output naming
        --pyg_config:              Optional JSON string with PyG construction config
        --parquet_partitions:      Number of Parquet output partitions (default: 200)
        --source_format:           ntriples | turtle_parquet (default: ntriples)
        --turtle_column:           Column name containing Turtle strings when
                                   source_format=turtle_parquet (default: triples)

Example Glue job parameters (N-Triples source):
    {
        "--mode": "full",
        "--source_bucket": "my-data-lake",
        "--source_prefix": "rdf/monthly/2024-12/",
        "--output_bucket": "my-data-lake",
        "--enriched_parquet_prefix": "enriched/2024-12/triples/",
        "--pyg_output_key": "pyg/2024-12/hetero_data.pt",
        "--enable_ontology_mapping": "true",
        "--time_period": "2024-12",
        "--parquet_partitions": "200"
    }

Example Glue job parameters (Turtle Parquet source):
    {
        "--mode": "enrichment_only",
        "--source_bucket": "my-data-lake",
        "--source_prefix": "raw/sec/filings/2024-12/",
        "--output_bucket": "my-data-lake",
        "--enriched_parquet_prefix": "enriched/2024-12/triples/",
        "--source_format": "turtle_parquet",
        "--turtle_column": "triples",
        "--time_period": "2024-12",
        "--parquet_partitions": "200"
    }
"""
import sys
import json
import logging
import time
import io
from datetime import datetime
from typing import Dict, Any, Tuple, List

# ============================================
# AWS Glue imports (graceful fallback for local)
# ============================================
try:
    from awsglue.utils import getResolvedOptions
    from awsglue.context import GlueContext
    from pyspark.context import SparkContext

    RUNNING_IN_GLUE = True
except ImportError:
    RUNNING_IN_GLUE = False

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import boto3

# ============================================
# Project imports
# ============================================
from glue_jobs.enrichment.pipeline import EnrichmentPipeline

from glue_jobs.pyg_builder.metadata_writer import (
    write_metadata_to_s3,
    derive_metadata_prefix,
)

# PyG builder — imported conditionally since enrichment_only mode doesn't need it
_pyg_builder_available = False
try:
    from glue_jobs.pyg_builder.constructor import build_hetero_data

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
# Parameter parsing
# ============================================
class JobConfig:
    """Parsed and validated job configuration."""

    def __init__(self, args: Dict[str, str]):
        self.mode = args.get("mode", "full").lower().strip()
        self.output_bucket = args.get("output_bucket", "")

        self.market_sector_definitions_bucket = args.get(
            "market_sector_definitions_bucket", ""
        )
        self.market_sector_definitions_key = args.get(
            "market_sector_definitions_key", ""
        )

        # Source parameters (required for full and enrichment_only)
        self.source_bucket = args.get("source_bucket", "")

        # source_prefix accepts a single prefix or a comma-separated list
        # of prefixes within the same source_bucket.
        # Examples:
        #   single:   "rdf/monthly/2024-12/"
        #   multiple: "raw/sec/filings/2024-12/,raw/sec/proceedings/2024-12/,raw/sec/suspensions/2024-12/,raw/sec/litigations/2024-12/"
        # Leading/trailing whitespace around each prefix is stripped.
        raw_prefix = args.get("source_prefix", "")
        self.source_prefixes: List[str] = [
            p.strip() for p in raw_prefix.split(",") if p.strip()
        ]

        # Source format:
        #   "ntriples":       one triple per line in .nt files
        #   "turtle_parquet": self-contained Turtle blobs in a Parquet column
        self.source_format = (
            args.get("source_format", "ntriples").lower().strip()
        )

        # Column name containing Turtle strings when
        # source_format=turtle_parquet. Configurable so that different
        # source Parquet schemas can be handled without code changes.
        self.turtle_column = args.get("turtle_column", "triples")

        # Output keys / prefixes
        self.enriched_parquet_prefix = args.get(
            "enriched_parquet_prefix", ""
        )
        self.pyg_output_key = args.get("pyg_output_key", "")

        # Optional
        self.enable_ontology_mapping = (
            args.get("enable_ontology_mapping", "false").lower() == "true"
        )
        self.time_period = args.get(
            "time_period", datetime.now().strftime("%Y-%m")
        )
        self.pyg_config = self._parse_pyg_config(args.get("pyg_config", ""))
        self.parquet_partitions = int(
            args.get("parquet_partitions", str(DEFAULT_PARQUET_PARTITIONS))
        )

        # Derived defaults
        if not self.enriched_parquet_prefix:
            self.enriched_parquet_prefix = (
                f"enriched/{self.time_period}/triples/"
            )
        if not self.pyg_output_key:
            self.pyg_output_key = f"pyg/{self.time_period}/hetero_data.pt"

        self._validate()

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
        if not self.output_bucket:
            raise ValueError("output_bucket is required")

        if self.source_format not in VALID_SOURCE_FORMATS:
            raise ValueError(
                f"Invalid source_format '{self.source_format}'. "
                f"Must be one of: {', '.join(VALID_SOURCE_FORMATS)}"
            )

        if self.mode in ("full", "enrichment_only"):
            if not self.source_bucket:
                raise ValueError(
                    f"source_bucket is required for mode '{self.mode}'"
                )
            if not self.source_prefixes:
                raise ValueError(
                    f"source_prefix is required for mode '{self.mode}'"
                )
        if self.mode in ("full", "pyg_only"):
            if not _pyg_builder_available:
                raise ImportError(
                    f"PyG builder not available. Mode '{self.mode}' requires "
                    "glue_jobs.pyg_builder.constructor."
                )
        if self.mode == "pyg_only":
            if not self.enriched_parquet_prefix:
                raise ValueError(
                    "enriched_parquet_prefix is required for mode 'pyg_only'"
                )
        if self.parquet_partitions < 1:
            raise ValueError("parquet_partitions must be >= 1")

    def __repr__(self):
        return (
            f"JobConfig(mode={self.mode}, "
            f"source_bucket={self.source_bucket}, "
            f"source_prefixes={self.source_prefixes}, "
            f"source_format={self.source_format}, "
            f"turtle_column={self.turtle_column}, "
            f"output={self.output_bucket}, "
            f"enriched_parquet={self.enriched_parquet_prefix}, "
            f"pyg_output={self.pyg_output_key}, "
            f"ontology_mapping={self.enable_ontology_mapping}, "
            f"parquet_partitions={self.parquet_partitions})"
        )


def parse_args() -> JobConfig:
    """Parse job arguments from Glue or CLI."""
    if RUNNING_IN_GLUE:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "mode",
                "source_bucket",
                "source_prefix",
                "output_bucket",
                "enriched_parquet_prefix",
                "pyg_output_key",
                "enable_ontology_mapping",
                "time_period",
                "pyg_config",
                "parquet_partitions",
                "source_format",
                "turtle_column",
                "market_sector_definitions_bucket",
                "market_sector_definitions_key",
            ],
        )
    else:
        import argparse

        parser = argparse.ArgumentParser(
            description="PyTorch Geometric Knowledge Graph Builder"
        )
        parser.add_argument(
            "--mode", choices=list(VALID_MODES), default="full"
        )
        parser.add_argument("--source_bucket", default="")
        parser.add_argument("--source_prefix", default="")
        parser.add_argument("--output_bucket", required=True)
        parser.add_argument("--enriched_parquet_prefix", default="")
        parser.add_argument("--pyg_output_key", default="")
        parser.add_argument("--enable_ontology_mapping", default="false")
        parser.add_argument("--time_period", default="")
        parser.add_argument("--pyg_config", default="")
        parser.add_argument(
            "--parquet_partitions",
            default=str(DEFAULT_PARQUET_PARTITIONS),
        )
        parser.add_argument(
            "--source_format",
            choices=list(VALID_SOURCE_FORMATS),
            default="ntriples",
        )
        parser.add_argument("--turtle_column", default="triples")
        parser.add_argument("--market_sector_definitions_bucket", default="")
        parser.add_argument("--market_sector_definitions_key", default="")

        parsed = parser.parse_args()
        args = vars(parsed)

    return JobConfig(args)


# ============================================
# N-Triples parsing (raw RDF → triples DataFrame)
# ============================================
def load_ntriples_to_dataframe(
    spark: SparkSession, source_path: str
) -> DataFrame:
    """
    Load N-Triples files from S3 into a PySpark triples DataFrame.

    Runs entirely on Spark executors — no driver-side parsing.

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
def load_turtle_parquet_to_dataframe(
    spark: SparkSession,
    source_path: str,
    turtle_column: str = "triples",
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

    Requires rdflib on Glue workers. Add to the Glue job configuration:
      --additional-python-modules rdflib==6.3.2

    Args:
        spark: Active SparkSession
        source_path: S3 path to source Parquet files
        turtle_column: Name of the column containing Turtle strings.
                       Default "triples". Configurable via
                       --turtle_column Glue job parameter.

    Returns:
        DataFrame with columns (subject: string, predicate: string,
        object: string), one row per triple. Compatible with all
        downstream pipeline steps.

    Raises:
        ValueError: If turtle_column is not found in the Parquet schema.
    """
    from pyspark.sql.types import ArrayType

    logger.info(
        f"Loading Turtle Parquet from {source_path}, "
        f"column='{turtle_column}'"
    )

    raw_df = spark.read.parquet(source_path)

    if turtle_column not in raw_df.columns:
        raise ValueError(
            f"Column '{turtle_column}' not found in Parquet schema. "
            f"Available columns: {raw_df.columns}. "
            f"Set --turtle_column to the correct column name."
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
    #   - BNode   → f"_:{bnode}" (kept as blank node identifier)
    #
    # Malformed blobs return an empty list so that parse errors skip
    # the row rather than failing the job. The zero-triple contribution
    # is visible in the final triple count logged after loading.

    triple_schema = ArrayType(
        StructType([
            StructField("subject", StringType(), nullable=False),
            StructField("predicate", StringType(), nullable=False),
            StructField("object", StringType(), nullable=False),
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

            triples = []
            for s, p, o in g:
                # Subject
                if isinstance(s, URIRef):
                    subj = str(s)
                elif isinstance(s, BNode):
                    subj = f"_:{str(s)}"
                else:
                    # Skip unexpected subject types (should not occur
                    # in well-formed Turtle)
                    continue

                # Predicate — always a URIRef in valid RDF
                pred = str(p)

                # Object — normalize to pipeline convention
                if isinstance(o, URIRef):
                    obj = str(o)
                elif isinstance(o, BNode):
                    obj = f"_:{str(o)}"
                elif isinstance(o, Literal):
                    # toPython() returns a Python native type for
                    # numeric/date XSD types; str() of that matches
                    # what the pipeline expects for numeric literals.
                    # For plain strings, str(literal) returns the
                    # lexical form without ^^datatype or @lang.
                    py_val = o.toPython()
                    obj = str(py_val)
                else:
                    obj = str(o)

                triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
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

    triples_df = exploded_df.select(
        F.col("_triple.subject").alias("subject"),
        F.col("_triple.predicate").alias("predicate"),
        F.col("_triple.object").alias("object"),
    ).filter(
        (F.col("subject") != "")
        & (F.col("predicate") != "")
    )

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

    Supports multiple source prefixes within the same source_bucket.
    When more than one prefix is provided, each is loaded independently
    and the results are unioned into a single triples DataFrame before
    caching. Duplicates across prefixes are not deduplicated here —
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
        FileNotFoundError: If no triples are parsed from any source prefix.
        ValueError: If turtle_column is not found (turtle_parquet only).
    """
    source_paths = [
        f"s3://{config.source_bucket}/{prefix}"
        for prefix in config.source_prefixes
    ]

    logger.info(
        f"Source format: {config.source_format}, "
        f"{len(source_paths)} prefix(es)"
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
                else " Check that .nt files exist at the source prefixes."
            )
        )

    logger.info(
        f"Loaded {count:,} triples total across "
        f"{len(source_paths)} prefix(es)"
    )
    return triples_df, count


# ============================================
# Parquet I/O for enriched triples
# ============================================
def save_enriched_parquet(
    triples_df: DataFrame, output_path: str, num_partitions: int
) -> None:
    """
    Save enriched triples DataFrame to S3 as Parquet.

    Data flows directly from executors to S3 — nothing passes
    through the driver. Repartitions to control output file count
    and size for efficient downstream reads.

    Args:
        triples_df: Enriched triples DataFrame (subject, predicate, object)
        output_path: S3 path (e.g., "s3://bucket/enriched/2024-12/triples/")
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
    Load enriched triples from Parquet on S3.

    Reads directly into a distributed DataFrame on executors.
    This loads Parquet that this pipeline previously wrote
    (schema: subject, predicate, object) — not source Parquet.

    Args:
        spark: Active SparkSession
        input_path: S3 path to enriched Parquet

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


# ============================================
# Pipeline phases
# ============================================
def run_enrichment(
    spark: SparkSession,
    triples_df: DataFrame,
    enable_ontology_mapping: bool = False,
    market_sector_definitions_bucket: str = "",
    market_sector_definitions_key: str = "",
) -> tuple:
    """
    Run the enrichment pipeline on a triples DataFrame.

    All enrichment runs as PySpark DataFrame operations on executors.

    Args:
        spark: Active SparkSession
        triples_df: Raw triples DataFrame (subject, predicate, object)
        enable_ontology_mapping: Whether to run ontology mapping

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
    stats = pipeline.run(enable_ontology_mapping=enable_ontology_mapping)
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
    hetero_data, metadata = build_hetero_data(
        spark, triples_df, config, time_period=time_period
    )

    elapsed = time.time() - start_time

    logger.info(f"PyG construction completed in {elapsed:.1f}s")
    _log_hetero_data_summary(hetero_data)

    return hetero_data, metadata


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

    In AWS Glue, created from GlueContext.
    In local mode, creates a local SparkSession.
    """
    if RUNNING_IN_GLUE:
        sc = SparkContext.getOrCreate()
        glue_context = GlueContext(sc)
        return glue_context.spark_session
    else:
        return (
            SparkSession.builder.appName(
                "PyG-Knowledge-Graph-Builder"
            ).getOrCreate()
        )


# ============================================
# Execution modes
# ============================================
def execute_full_pipeline(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: full
    Raw source → triples_df → Enrich → Save Parquet → Build PyG → Save .pt → Save metadata

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
    )

    # Unpersist raw triples — enriched_df is independently cached
    triples_df.unpersist()

    # Step 3: Save enriched Parquet (executors → S3, no driver)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED TRIPLES (Parquet)")
    logger.info("=" * 80)
    enriched_path = (
        f"s3://{config.output_bucket}/{config.enriched_parquet_prefix}"
    )
    save_enriched_parquet(
        enriched_df, enriched_path, config.parquet_partitions
    )

    # Step 4: Build PyG (executors → compact tensors → driver)
    hetero_data, metadata = run_pyg_construction(
        spark, enriched_df, config.pyg_config,
        time_period=config.time_period,
    )

    # Step 5: Save PyG (driver → S3)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA")
    logger.info("=" * 80)
    save_pyg_to_s3(
        s3_client, hetero_data, config.output_bucket, config.pyg_output_key
    )

    # Step 6: Save metadata (small JSON files → S3)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING METADATA")
    logger.info("=" * 80)
    metadata_prefix = derive_metadata_prefix(config.pyg_output_key)
    metadata_files = metadata.to_metadata_files()
    write_metadata_to_s3(
        s3_client, metadata_files,
        config.output_bucket, metadata_prefix,
    )

    return {
        "mode": "full",
        "source_format": config.source_format,
        "initial_triples": initial_count,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": enriched_path,
        "pyg_location": (
            f"s3://{config.output_bucket}/{config.pyg_output_key}"
        ),
        "metadata_location": (
            f"s3://{config.output_bucket}/{metadata_prefix}"
        ),
    }


def execute_enrichment_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: enrichment_only
    Raw source → triples_df → Enrich → Save enriched Parquet to S3

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
        spark, triples_df, config.enable_ontology_mapping
    )

    # Unpersist raw triples
    triples_df.unpersist()

    # Step 3: Save enriched Parquet (executors → S3 directly)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED TRIPLES (Parquet)")
    logger.info("=" * 80)
    enriched_path = (
        f"s3://{config.output_bucket}/{config.enriched_parquet_prefix}"
    )
    save_enriched_parquet(
        enriched_df, enriched_path, config.parquet_partitions
    )

    return {
        "mode": "enrichment_only",
        "source_format": config.source_format,
        "initial_triples": initial_count,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": enriched_path,
    }


def execute_pyg_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: pyg_only
    Enriched Parquet (S3) → triples_df → Build PyG → Save .pt → Save metadata

    Rapid experimentation: loads existing enriched Parquet,
    constructs PyG with different configurations.
    Typically 5-10 minutes.

    Note: pyg_only always reads from enriched Parquet previously
    written by this pipeline (schema: subject, predicate, object).
    source_format and turtle_column do not apply in this mode.
    """
    logger.info("Executing PyG ONLY mode")
    logger.info("")

    # Step 1: Load enriched Parquet → distributed DataFrame
    logger.info("=" * 80)
    logger.info("PHASE: LOADING ENRICHED TRIPLES (Parquet)")
    logger.info("=" * 80)
    start_time = time.time()

    enriched_path = (
        f"s3://{config.output_bucket}/{config.enriched_parquet_prefix}"
    )
    triples_df = load_enriched_parquet(spark, enriched_path)
    triples_df = triples_df.cache()
    triple_count = triples_df.count()

    load_elapsed = time.time() - start_time
    logger.info(
        f"Loaded {triple_count:,} enriched triples in {load_elapsed:.1f}s"
    )
    logger.info("")

    # Step 2: Build PyG (executors → compact tensors → driver)
    hetero_data, metadata = run_pyg_construction(
        spark, triples_df, config.pyg_config,
        time_period=config.time_period,
    )

    # Step 3: Save PyG (driver → S3)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA")
    logger.info("=" * 80)
    save_pyg_to_s3(
        s3_client, hetero_data, config.output_bucket, config.pyg_output_key
    )

    # Step 4: Save metadata (small JSON files → S3)
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING METADATA")
    logger.info("=" * 80)
    metadata_prefix = derive_metadata_prefix(config.pyg_output_key)
    metadata_files = metadata.to_metadata_files()
    write_metadata_to_s3(
        s3_client, metadata_files,
        config.output_bucket, metadata_prefix,
    )

    return {
        "mode": "pyg_only",
        "enriched_parquet_source": enriched_path,
        "enriched_triple_count": triple_count,
        "pyg_location": (
            f"s3://{config.output_bucket}/{config.pyg_output_key}"
        ),
        "metadata_location": (
            f"s3://{config.output_bucket}/{metadata_prefix}"
        ),
    }


# ============================================
# Job manifest and summary
# ============================================
def save_job_manifest(
    s3_client, config: JobConfig, result: Dict, elapsed: float
):
    """Save a JSON manifest with job metadata to S3."""
    manifest = {
        "job_timestamp": datetime.utcnow().isoformat() + "Z",
        "time_period": config.time_period,
        "mode": config.mode,
        "config": {
            "source_bucket": config.source_bucket,
            "source_prefix": config.source_prefix,
            "source_format": config.source_format,
            "turtle_column": config.turtle_column,
            "output_bucket": config.output_bucket,
            "enriched_parquet_prefix": config.enriched_parquet_prefix,
            "pyg_output_key": config.pyg_output_key,
            "enable_ontology_mapping": config.enable_ontology_mapping,
            "pyg_config": config.pyg_config,
            "parquet_partitions": config.parquet_partitions,
        },
        "result": _make_json_serializable(result),
        "elapsed_seconds": round(elapsed, 2),
    }

    manifest_key = (
        f"manifests/{config.time_period}/"
        f"{config.mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )

    try:
        s3_client.put_object(
            Bucket=config.output_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            f"Saved job manifest to "
            f"s3://{config.output_bucket}/{manifest_key}"
        )
    except Exception as e:
        logger.warning(f"Failed to save job manifest: {e}")


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
    Main entry point for the Glue job.

    Parses arguments, initializes Spark and AWS clients,
    dispatches to the appropriate execution mode.
    """
    job_start_time = time.time()

    logger.info("=" * 80)
    logger.info("PyTorch Geometric Knowledge Graph Builder")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.utcnow().isoformat()}Z")
    logger.info(f"Running in: {'AWS Glue' if RUNNING_IN_GLUE else 'Local'}")
    logger.info("")

    # Parse and validate arguments
    try:
        config = parse_args()
    except (ValueError, ImportError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    logger.info(f"Configuration: {config}")
    logger.info("")

    # Initialize Spark and AWS clients
    spark = get_spark_session()
    s3_client = boto3.client("s3")

    logger.info(
        f"Initialized SparkSession "
        f"({'Glue' if RUNNING_IN_GLUE else 'local'}) and S3 client"
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
    save_job_manifest(s3_client, config, result, elapsed)
    print_final_banner(config, result, elapsed)

    logger.info("Done.")
    return result


if __name__ == "__main__":
    main()