"""
PyTorch Geometric Knowledge Graph Builder - Main Glue Job Entry Point

AWS Glue job that orchestrates the complete pipeline:
  Raw RDF (N-Triples, S3) → PySpark triples_df → Enrich → Build PyG HeteroData → Save to S3

Supports three execution modes:
  - full:            End-to-end RDF enrichment + PyG graph construction
  - enrichment_only: Create reusable enriched Parquet artifacts (save to S3)
  - pyg_only:        Build PyG graph from existing enriched Parquet artifacts

Usage (AWS Glue):
    Parameters:
        --mode:                    full | enrichment_only | pyg_only
        --source_bucket:           S3 bucket containing raw RDF files
        --source_prefix:           S3 prefix for raw RDF files (e.g., "rdf/2024/12/")
        --output_bucket:           S3 bucket for outputs
        --enriched_parquet_prefix: S3 prefix for enriched Parquet artifact
        --pyg_output_key:          S3 key for PyG HeteroData output (.pt)
        --enable_ontology_mapping: true | false (default: false)
        --time_period:             Time period label (e.g., "2024-12") for output naming
        --pyg_config:              Optional JSON string with PyG construction config

Usage (Local / Testing):
    python build_graph.py \
        --mode full \
        --source_bucket my-bucket \
        --source_prefix rdf/2024/12/ \
        --output_bucket my-bucket \
        --enriched_parquet_prefix enriched/2024-12/triples/ \
        --pyg_output_key pyg/2024-12/hetero_data.pt \
        --time_period 2024-12

Example Glue job parameters:
    {
        "--mode": "full",
        "--source_bucket": "my-data-lake",
        "--source_prefix": "rdf/monthly/2024-12/",
        "--output_bucket": "my-data-lake",
        "--enriched_parquet_prefix": "enriched/2024-12/triples/",
        "--pyg_output_key": "pyg/2024-12/hetero_data.pt",
        "--enable_ontology_mapping": "true",
        "--time_period": "2024-12"
    }
"""
import sys
import json
import logging
import time
import io
from datetime import datetime
from typing import Dict, Optional, Any

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
from glue_jobs.enrichment.pipeline import enrich_graph

# PyG builder - imported conditionally since enrichment_only mode doesn't need it
# and PyG/torch may not be available in all environments
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

TRIPLES_SCHEMA = StructType([
    StructField("subject", StringType(), nullable=False),
    StructField("predicate", StringType(), nullable=False),
    StructField("object", StringType(), nullable=False),
])


# ============================================
# Parameter parsing
# ============================================
class JobConfig:
    """
    Parsed and validated job configuration.

    Handles both AWS Glue resolved options and local CLI arguments.
    """

    def __init__(self, args: Dict[str, str]):
        # Required
        self.mode = args.get("mode", "full").lower().strip()
        self.output_bucket = args.get("output_bucket", "")

        # Source parameters (required for full and enrichment_only)
        self.source_bucket = args.get("source_bucket", "")
        self.source_prefix = args.get("source_prefix", "")

        # Output keys / prefixes
        self.enriched_parquet_prefix = args.get("enriched_parquet_prefix", "")
        self.pyg_output_key = args.get("pyg_output_key", "")

        # Optional
        self.enable_ontology_mapping = (
            args.get("enable_ontology_mapping", "false").lower() == "true"
        )
        self.time_period = args.get("time_period", datetime.now().strftime("%Y-%m"))
        self.pyg_config = self._parse_pyg_config(args.get("pyg_config", ""))

        # Derived defaults
        if not self.enriched_parquet_prefix:
            self.enriched_parquet_prefix = f"enriched/{self.time_period}/triples/"

        if not self.pyg_output_key:
            self.pyg_output_key = f"pyg/{self.time_period}/hetero_data.pt"

        # Validate
        self._validate()

    def _parse_pyg_config(self, config_str: str) -> Dict[str, Any]:
        """Parse optional PyG construction configuration from JSON string."""
        if not config_str or config_str.strip() == "":
            return {}
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse pyg_config JSON, using defaults: {e}")
            return {}

    def _validate(self):
        """Validate configuration."""
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {', '.join(VALID_MODES)}"
            )

        if not self.output_bucket:
            raise ValueError("output_bucket is required")

        # Mode-specific validation
        if self.mode in ("full", "enrichment_only"):
            if not self.source_bucket:
                raise ValueError(f"source_bucket is required for mode '{self.mode}'")
            if not self.source_prefix:
                raise ValueError(f"source_prefix is required for mode '{self.mode}'")

        if self.mode in ("full", "pyg_only"):
            if not _pyg_builder_available:
                raise ImportError(
                    f"PyG builder not available. Mode '{self.mode}' requires "
                    "glue_jobs.pyg_builder.constructor. Install PyTorch Geometric "
                    "or use mode 'enrichment_only'."
                )

        if self.mode == "pyg_only":
            if not self.enriched_parquet_prefix:
                raise ValueError(
                    "enriched_parquet_prefix is required for mode 'pyg_only'"
                )

    def __repr__(self):
        return (
            f"JobConfig(mode={self.mode}, "
            f"source={self.source_bucket}/{self.source_prefix}, "
            f"output={self.output_bucket}, "
            f"enriched_parquet={self.enriched_parquet_prefix}, "
            f"pyg_output={self.pyg_output_key}, "
            f"ontology_mapping={self.enable_ontology_mapping})"
        )


def parse_args() -> JobConfig:
    """
    Parse job arguments from Glue or CLI.

    Returns:
        Validated JobConfig instance
    """
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
            ],
        )
    else:
        import argparse

        parser = argparse.ArgumentParser(
            description="PyTorch Geometric Knowledge Graph Builder"
        )
        parser.add_argument(
            "--mode",
            choices=list(VALID_MODES),
            default="full",
            help="Execution mode",
        )
        parser.add_argument(
            "--source_bucket", default="", help="S3 bucket for raw RDF"
        )
        parser.add_argument(
            "--source_prefix", default="", help="S3 prefix for raw RDF files"
        )
        parser.add_argument(
            "--output_bucket", required=True, help="S3 bucket for outputs"
        )
        parser.add_argument(
            "--enriched_parquet_prefix",
            default="",
            help="S3 prefix for enriched Parquet triples",
        )
        parser.add_argument(
            "--pyg_output_key", default="", help="S3 key for PyG output"
        )
        parser.add_argument(
            "--enable_ontology_mapping",
            default="false",
            help="Enable ontology mapping",
        )
        parser.add_argument(
            "--time_period", default="", help="Time period label"
        )
        parser.add_argument(
            "--pyg_config", default="", help="PyG config JSON"
        )

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

    N-Triples format: each line is a single triple:
        <subject_uri> <predicate_uri> <object_uri_or_literal> .

    This reads the raw text lines, parses subject/predicate/object
    using regex, and returns a DataFrame with schema:
        (subject: string, predicate: string, object: string)

    Runs entirely on Spark executors — no driver-side parsing.

    Args:
        spark: Active SparkSession
        source_path: S3 path (e.g., "s3://bucket/prefix/") containing .nt files

    Returns:
        DataFrame with columns (subject, predicate, object)
    """
    logger.info(f"Loading N-Triples from {source_path}")

    # Read raw text lines across all .nt files under the path
    raw_lines = spark.read.text(source_path)

    # Filter out blank lines and comments
    raw_lines = raw_lines.filter(
        (F.trim(F.col("value")) != "") & (~F.col("value").startswith("#"))
    )

    # Parse N-Triples lines using regex
    # N-Triples format:
    #   URI:     <http://example.org/subject>
    #   Literal: "value"^^<datatype> or "value"@lang or "value"
    #
    # Pattern: <subject> <predicate> <object_or_literal> .
    #
    # We extract:
    #   subject:   content between first < >
    #   predicate: content between second < >
    #   object:    everything after predicate up to the final " ."
    #
    # For URIs as objects: <http://...>
    # For literals: "value"^^<type> or "value"
    triples_df = raw_lines.select(
        # Subject: first <...> on the line
        F.regexp_extract("value", r"<([^>]+)>", 1).alias("subject"),
        # Predicate: second <...> on the line
        F.regexp_extract("value", r"<[^>]+>\s+<([^>]+)>", 1).alias("predicate"),
        # Object: everything between the predicate and the trailing " ."
        # This captures both URI objects (<...>) and literal objects ("...")
        F.regexp_extract(
            "value", r"<[^>]+>\s+<[^>]+>\s+(.*)\s+\.\s*$", 1
        ).alias("raw_object"),
    )

    # Clean up the object field:
    # - URI objects: strip angle brackets  <http://...> → http://...
    # - Literal objects: strip quotes and optional datatype/lang
    #   "295.8"^^<xsd:decimal> → 295.8
    #   "Food"@en → Food
    #   "plain string" → plain string
    triples_df = triples_df.withColumn(
        "object",
        F.when(
            # URI object: starts with <
            F.col("raw_object").startswith("<"),
            F.regexp_extract("raw_object", r"<([^>]+)>", 1),
        ).when(
            # Typed literal: "value"^^<datatype>
            F.col("raw_object").contains("^^"),
            F.regexp_extract("raw_object", r'"([^"]*)"', 1),
        ).when(
            # Language-tagged literal: "value"@lang
            F.col("raw_object").rlike(r'"[^"]*"@'),
            F.regexp_extract("raw_object", r'"([^"]*)"', 1),
        ).otherwise(
            # Plain literal: "value"
            F.regexp_extract("raw_object", r'"([^"]*)"', 1)
        ),
    ).drop("raw_object")

    # Filter out any rows where parsing failed (empty subject or predicate)
    triples_df = triples_df.filter(
        (F.col("subject") != "") & (F.col("predicate") != "")
    )

    count = triples_df.count()
    logger.info(f"Parsed {count:,} triples from N-Triples files")

    return triples_df


# ============================================
# Parquet I/O for enriched triples
# ============================================
def save_enriched_parquet(
    triples_df: DataFrame, output_path: str
) -> int:
    """
    Save enriched triples DataFrame to S3 as Parquet.

    Uses Spark's distributed write — data flows directly from
    executors to S3 without passing through the driver.

    Args:
        triples_df: Enriched triples DataFrame (subject, predicate, object)
        output_path: S3 path (e.g., "s3://bucket/enriched/2024-12/triples/")

    Returns:
        Number of triples saved
    """
    count = triples_df.count()

    triples_df.write.mode("overwrite").parquet(output_path)

    logger.info(
        f"Saved enriched triples ({count:,} triples) to {output_path}"
    )
    return count


def load_enriched_parquet(
    spark: SparkSession, input_path: str
) -> DataFrame:
    """
    Load enriched triples from Parquet on S3.

    Reads directly into a distributed DataFrame on executors.

    Args:
        spark: Active SparkSession
        input_path: S3 path to enriched Parquet
                    (e.g., "s3://bucket/enriched/2024-12/triples/")

    Returns:
        DataFrame with columns (subject, predicate, object)
    """
    triples_df = spark.read.parquet(input_path)

    count = triples_df.count()
    logger.info(
        f"Loaded enriched triples ({count:,} triples) from {input_path}"
    )
    return triples_df


# ============================================
# PyG output (driver → S3)
# ============================================
def save_pyg_to_s3(s3_client, hetero_data, bucket: str, key: str):
    """
    Save PyTorch Geometric HeteroData to S3.

    Serializes using torch.save to a BytesIO buffer, then uploads to S3.
    This runs on the driver — only compact tensors are in memory at this point.

    Args:
        s3_client: Boto3 S3 client
        hetero_data: PyG HeteroData object
        bucket: S3 bucket name
        key: S3 key
    """
    import torch

    buffer = io.BytesIO()
    torch.save(hetero_data, buffer)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    size_mb = len(buffer.getvalue()) / (1024 * 1024)
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
) -> tuple:
    """
    Run the enrichment pipeline on a triples DataFrame.

    All enrichment runs as PySpark DataFrame operations on executors.

    Args:
        spark: Active SparkSession
        triples_df: Raw triples DataFrame (subject, predicate, object)
        enable_ontology_mapping: Whether to run ontology mapping

    Returns:
        Tuple of (enriched_triples_df, enrichment_stats)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: RDF ENRICHMENT")
    logger.info("=" * 80)

    initial_count = triples_df.count()
    logger.info(f"Input triples: {initial_count:,}")
    logger.info(
        f"Ontology mapping: "
        f"{'enabled' if enable_ontology_mapping else 'disabled'}"
    )
    logger.info("")

    start_time = time.time()

    # enrich_graph returns stats; the pipeline mutates triples_df internally
    # and we retrieve the final enriched DataFrame from the pipeline object
    from glue_jobs.enrichment.pipeline import EnrichmentPipeline

    pipeline = EnrichmentPipeline(spark, triples_df)
    stats = pipeline.run(enable_ontology_mapping=enable_ontology_mapping)
    enriched_df = pipeline.get_enriched_triples_df()

    elapsed = time.time() - start_time
    logger.info(f"Enrichment completed in {elapsed:.1f}s")

    return enriched_df, stats


def run_pyg_construction(
    spark: SparkSession,
    triples_df: DataFrame,
    pyg_config: Dict[str, Any] = None,
) -> Any:
    """
    Run PyG HeteroData construction from enriched triples DataFrame.

    Heavy computation (node ID assignment, edge resolution, feature extraction)
    runs on Spark executors. Only compact tensors cross to the driver for
    final HeteroData assembly.

    Args:
        spark: Active SparkSession
        triples_df: Enriched triples DataFrame (subject, predicate, object)
        pyg_config: Optional configuration for PyG construction

    Returns:
        PyG HeteroData object
    """
    if not _pyg_builder_available:
        raise ImportError(
            "PyG builder not available. Install PyTorch Geometric and ensure "
            "glue_jobs.pyg_builder.constructor is importable."
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: PyG CONSTRUCTION")
    logger.info("=" * 80)

    triple_count = triples_df.count()
    logger.info(f"Input triples: {triple_count:,}")
    if pyg_config:
        logger.info(f"PyG config: {json.dumps(pyg_config, indent=2)}")
    logger.info("")

    start_time = time.time()

    config = pyg_config or {}
    hetero_data = build_hetero_data(spark, triples_df, config)

    elapsed = time.time() - start_time

    # Log HeteroData summary
    logger.info(f"PyG construction completed in {elapsed:.1f}s")
    logger.info("HeteroData summary:")
    logger.info(f"  Node types: {hetero_data.node_types}")
    logger.info(f"  Edge types: {hetero_data.edge_types}")
    for node_type in hetero_data.node_types:
        store = hetero_data[node_type]
        num_nodes = (
            store.num_nodes if hasattr(store, "num_nodes") else "unknown"
        )
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

    return hetero_data


# ============================================
# Spark session initialization
# ============================================
def get_spark_session() -> SparkSession:
    """
    Get or create a SparkSession.

    In AWS Glue, the SparkSession is created from the GlueContext.
    In local mode, a local SparkSession is created.

    Returns:
        Active SparkSession
    """
    if RUNNING_IN_GLUE:
        sc = SparkContext.getOrCreate()
        glue_context = GlueContext(sc)
        return glue_context.spark_session
    else:
        return (
            SparkSession.builder
            .appName("PyG-Knowledge-Graph-Builder")
            .getOrCreate()
        )


# ============================================
# Main execution modes
# ============================================
def execute_full_pipeline(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: full
    Raw RDF (N-Triples) → triples_df → Enrich → Save Parquet → Build PyG → Save .pt

    This is the default end-to-end mode.
    """
    logger.info("Executing FULL PIPELINE mode")
    logger.info("")

    # Step 1: Load raw N-Triples from S3 into triples DataFrame
    logger.info("=" * 80)
    logger.info("PHASE: LOADING RAW RDF (N-Triples → DataFrame)")
    logger.info("=" * 80)
    start_time = time.time()

    source_path = f"s3://{config.source_bucket}/{config.source_prefix}"
    triples_df = load_ntriples_to_dataframe(spark, source_path)
    triples_df = triples_df.cache()
    initial_count = triples_df.count()

    load_elapsed = time.time() - start_time
    logger.info(f"Loading completed in {load_elapsed:.1f}s")
    logger.info("")

    if initial_count == 0:
        raise FileNotFoundError(
            f"No triples parsed from N-Triples files at {source_path}"
        )

    # Step 2: Enrich (all on executors)
    enriched_df, enrichment_stats = run_enrichment(
        spark, triples_df, config.enable_ontology_mapping
    )

    # Step 3: Save enriched triples as Parquet
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED TRIPLES (Parquet)")
    logger.info("=" * 80)
    enriched_path = (
        f"s3://{config.output_bucket}/{config.enriched_parquet_prefix}"
    )
    save_enriched_parquet(enriched_df, enriched_path)

    # Step 4: Build PyG (heavy work on executors, tensors to driver)
    hetero_data = run_pyg_construction(spark, enriched_df, config.pyg_config)

    # Step 5: Save PyG HeteroData
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA")
    logger.info("=" * 80)
    save_pyg_to_s3(
        s3_client, hetero_data, config.output_bucket, config.pyg_output_key
    )

    return {
        "mode": "full",
        "initial_triples": initial_count,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": enriched_path,
        "pyg_location": (
            f"s3://{config.output_bucket}/{config.pyg_output_key}"
        ),
    }


def execute_enrichment_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: enrichment_only
    Raw RDF (N-Triples) → triples_df → Enrich → Save enriched Parquet to S3

    Creates a reusable enriched Parquet artifact that can be used
    for multiple PyG experiments without re-enrichment.
    """
    logger.info("Executing ENRICHMENT ONLY mode")
    logger.info("")

    # Step 1: Load raw N-Triples from S3 into triples DataFrame
    logger.info("=" * 80)
    logger.info("PHASE: LOADING RAW RDF (N-Triples → DataFrame)")
    logger.info("=" * 80)
    start_time = time.time()

    source_path = f"s3://{config.source_bucket}/{config.source_prefix}"
    triples_df = load_ntriples_to_dataframe(spark, source_path)
    triples_df = triples_df.cache()
    initial_count = triples_df.count()

    load_elapsed = time.time() - start_time
    logger.info(f"Loading completed in {load_elapsed:.1f}s")
    logger.info("")

    if initial_count == 0:
        raise FileNotFoundError(
            f"No triples parsed from N-Triples files at {source_path}"
        )

    # Step 2: Enrich (all on executors)
    enriched_df, enrichment_stats = run_enrichment(
        spark, triples_df, config.enable_ontology_mapping
    )

    # Step 3: Save enriched triples as Parquet
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED TRIPLES (Parquet)")
    logger.info("=" * 80)
    enriched_path = (
        f"s3://{config.output_bucket}/{config.enriched_parquet_prefix}"
    )
    saved_count = save_enriched_parquet(enriched_df, enriched_path)

    return {
        "mode": "enrichment_only",
        "initial_triples": initial_count,
        "enrichment": enrichment_stats,
        "enriched_parquet_location": enriched_path,
        "enriched_triple_count": saved_count,
    }


def execute_pyg_only(
    config: JobConfig, spark: SparkSession, s3_client
):
    """
    Mode: pyg_only
    Enriched Parquet (S3) → triples_df → Build PyG → Save PyG to S3

    Rapid experimentation mode: loads existing enriched Parquet
    and constructs PyG with different configurations.
    Typically completes in 5-10 minutes.
    """
    logger.info("Executing PyG ONLY mode")
    logger.info("")

    # Step 1: Load enriched triples from Parquet
    logger.info("=" * 80)
    logger.info("PHASE: LOADING ENRICHED TRIPLES (Parquet)")
    logger.info("=" * 80)
    start_time = time.time()

    enriched_path = (
        f"s3://{config.output_bucket}/{config.enriched_parquet_prefix}"
    )
    triples_df = load_enriched_parquet(spark, enriched_path)
    triples_df = triples_df.cache()
    triples_df.count()  # materialize cache

    load_elapsed = time.time() - start_time
    logger.info(f"Loading completed in {load_elapsed:.1f}s")
    logger.info("")

    # Step 2: Build PyG (heavy work on executors, tensors to driver)
    hetero_data = run_pyg_construction(spark, triples_df, config.pyg_config)

    # Step 3: Save PyG HeteroData
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA")
    logger.info("=" * 80)
    save_pyg_to_s3(
        s3_client, hetero_data, config.output_bucket, config.pyg_output_key
    )

    return {
        "mode": "pyg_only",
        "enriched_parquet_source": enriched_path,
        "pyg_location": (
            f"s3://{config.output_bucket}/{config.pyg_output_key}"
        ),
    }


# ============================================
# Job manifest and summary
# ============================================
def save_job_manifest(
    s3_client, config: JobConfig, result: Dict, elapsed: float
):
    """
    Save a JSON manifest with job metadata to S3.

    Useful for tracking experiment lineage and reproducing results.
    """
    manifest = {
        "job_timestamp": datetime.utcnow().isoformat() + "Z",
        "time_period": config.time_period,
        "mode": config.mode,
        "config": {
            "source_bucket": config.source_bucket,
            "source_prefix": config.source_prefix,
            "output_bucket": config.output_bucket,
            "enriched_parquet_prefix": config.enriched_parquet_prefix,
            "pyg_output_key": config.pyg_output_key,
            "enable_ontology_mapping": config.enable_ontology_mapping,
            "pyg_config": config.pyg_config,
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
    logger.info(f"  Time period:   {config.time_period}")
    logger.info(f"  Duration:      {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    logger.info("")

    if "enriched_parquet_location" in result:
        logger.info(
            f"  Enriched Parquet: {result['enriched_parquet_location']}"
        )
    if "pyg_location" in result:
        logger.info(f"  PyG output:       {result['pyg_location']}")

    if "enrichment" in result and isinstance(result["enrichment"], dict):
        stats = result["enrichment"]
        logger.info("")
        logger.info(
            f"  Initial triples:  "
            f"{stats.get('initial_triples', 'N/A'):>10}"
        )
        logger.info(
            f"  Final triples:    "
            f"{stats.get('final_triples', 'N/A'):>10}"
        )
        logger.info(
            f"  Total enrichment: "
            f"{stats.get('total_enrichment', 'N/A'):>10}"
        )

    logger.info("")
    logger.info("=" * 80)


# ============================================
# Main entry point
# ============================================
def main():
    """
    Main entry point for the Glue job.

    Parses arguments, initializes Spark and AWS clients, and dispatches
    to the appropriate execution mode.
    """
    job_start_time = time.time()

    logger.info("=" * 80)
    logger.info("PyTorch Geometric Knowledge Graph Builder")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.utcnow().isoformat()}Z")
    logger.info(f"Running in: {'AWS Glue' if RUNNING_IN_GLUE else 'Local'}")
    logger.info("")

    # ----------------------------------------
    # Parse and validate arguments
    # ----------------------------------------
    try:
        config = parse_args()
    except (ValueError, ImportError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    logger.info(f"Configuration: {config}")
    logger.info("")

    # ----------------------------------------
    # Initialize Spark and AWS clients
    # ----------------------------------------
    spark = get_spark_session()
    s3_client = boto3.client("s3")

    logger.info(
        f"Initialized SparkSession "
        f"({'Glue' if RUNNING_IN_GLUE else 'local'}) and S3 client"
    )
    logger.info("")

    # ----------------------------------------
    # Execute pipeline
    # ----------------------------------------
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
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    # ----------------------------------------
    # Save manifest and print summary
    # ----------------------------------------
    elapsed = time.time() - job_start_time

    save_job_manifest(s3_client, config, result, elapsed)
    print_final_banner(config, result, elapsed)

    # ----------------------------------------
    # Spark cleanup
    # ----------------------------------------
    logger.info("Done.")
    return result


if __name__ == "__main__":
    main()