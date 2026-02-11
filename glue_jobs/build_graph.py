"""
PyTorch Geometric Knowledge Graph Builder - Main Glue Job Entry Point

AWS Glue job that orchestrates the complete pipeline:
  Raw RDF (S3) → Enrich RDF → Build PyG HeteroData → Save to S3

Supports three execution modes:
  - full:            End-to-end RDF enrichment + PyG graph construction
  - enrichment_only: Create reusable enriched RDF artifacts (save to S3)
  - pyg_only:        Build PyG graph from existing enriched RDF artifacts

Usage (AWS Glue):
    Parameters:
        --mode:                  full | enrichment_only | pyg_only
        --source_bucket:         S3 bucket containing raw RDF files
        --source_prefix:         S3 prefix for raw RDF files (e.g., "rdf/2024/12/")
        --output_bucket:         S3 bucket for outputs
        --enriched_rdf_key:      S3 key for enriched RDF artifact (.ttl)
        --pyg_output_key:        S3 key for PyG HeteroData output (.pt)
        --enable_ontology_mapping: true | false (default: false)
        --time_period:           Time period label (e.g., "2024-12") for output naming
        --rdf_format:            RDF serialization format (default: turtle)
        --pyg_config:            Optional JSON string with PyG construction config

Usage (Local / Testing):
    python build_graph.py \
        --mode full \
        --source_bucket my-bucket \
        --source_prefix rdf/2024/12/ \
        --output_bucket my-bucket \
        --enriched_rdf_key enriched/2024-12/graph.ttl \
        --pyg_output_key pyg/2024-12/hetero_data.pt \
        --time_period 2024-12

Example Glue job parameters:
    {
        "--mode": "full",
        "--source_bucket": "my-data-lake",
        "--source_prefix": "rdf/monthly/2024-12/",
        "--output_bucket": "my-data-lake",
        "--enriched_rdf_key": "enriched/2024-12/knowledge_graph.ttl",
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

import boto3
from rdflib import Graph

# ============================================
# Project imports
# ============================================
from glue_jobs.utils.rdf_utils import RDFGraphLoader, RDFGraphWriter
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
VALID_RDF_FORMATS = {"turtle", "xml", "n3", "nt", "json-ld"}
RDF_CONTENT_TYPES = {
    "turtle": "text/turtle",
    "xml": "application/rdf+xml",
    "n3": "text/n3",
    "nt": "application/n-triples",
    "json-ld": "application/ld+json",
}
RDF_FILE_EXTENSIONS = {
    "turtle": ".ttl",
    "xml": ".rdf",
    "n3": ".n3",
    "nt": ".nt",
    "json-ld": ".jsonld",
}


# ============================================
# Parameter parsing
# ============================================
class JobConfig:
    """
    Parsed and validated job configuration

    Handles both AWS Glue resolved options and local CLI arguments.
    """

    def __init__(self, args: Dict[str, str]):
        # Required
        self.mode = args.get("mode", "full").lower().strip()
        self.output_bucket = args.get("output_bucket", "")

        # Source parameters (required for full and enrichment_only)
        self.source_bucket = args.get("source_bucket", "")
        self.source_prefix = args.get("source_prefix", "")

        # Output keys
        self.enriched_rdf_key = args.get("enriched_rdf_key", "")
        self.pyg_output_key = args.get("pyg_output_key", "")

        # Optional
        self.enable_ontology_mapping = args.get("enable_ontology_mapping", "false").lower() == "true"
        self.time_period = args.get("time_period", datetime.now().strftime("%Y-%m"))
        self.rdf_format = args.get("rdf_format", "turtle").lower().strip()
        self.pyg_config = self._parse_pyg_config(args.get("pyg_config", ""))

        # Derived defaults
        if not self.enriched_rdf_key:
            ext = RDF_FILE_EXTENSIONS.get(self.rdf_format, ".ttl")
            self.enriched_rdf_key = f"enriched/{self.time_period}/knowledge_graph{ext}"

        if not self.pyg_output_key:
            self.pyg_output_key = f"pyg/{self.time_period}/hetero_data.pt"

        # Validate
        self._validate()

    def _parse_pyg_config(self, config_str: str) -> Dict[str, Any]:
        """Parse optional PyG construction configuration from JSON string"""
        if not config_str or config_str.strip() == "":
            return {}
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse pyg_config JSON, using defaults: {e}")
            return {}

    def _validate(self):
        """Validate configuration"""
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {', '.join(VALID_MODES)}"
            )

        if self.rdf_format not in VALID_RDF_FORMATS:
            raise ValueError(
                f"Invalid rdf_format '{self.rdf_format}'. Must be one of: {', '.join(VALID_RDF_FORMATS)}"
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
            if not self.enriched_rdf_key:
                raise ValueError("enriched_rdf_key is required for mode 'pyg_only'")

    def __repr__(self):
        return (
            f"JobConfig(mode={self.mode}, source={self.source_bucket}/{self.source_prefix}, "
            f"output={self.output_bucket}, enriched_rdf={self.enriched_rdf_key}, "
            f"pyg_output={self.pyg_output_key}, ontology_mapping={self.enable_ontology_mapping})"
        )


def parse_args() -> JobConfig:
    """
    Parse job arguments from Glue or CLI

    Returns:
        Validated JobConfig instance
    """
    if RUNNING_IN_GLUE:
        # AWS Glue parameter resolution
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "mode",
                "source_bucket",
                "source_prefix",
                "output_bucket",
                "enriched_rdf_key",
                "pyg_output_key",
                "enable_ontology_mapping",
                "time_period",
                "rdf_format",
                "pyg_config",
            ],
        )
    else:
        # Local CLI argument parsing
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
        parser.add_argument("--source_bucket", default="", help="S3 bucket for raw RDF")
        parser.add_argument("--source_prefix", default="", help="S3 prefix for raw RDF files")
        parser.add_argument("--output_bucket", required=True, help="S3 bucket for outputs")
        parser.add_argument("--enriched_rdf_key", default="", help="S3 key for enriched RDF")
        parser.add_argument("--pyg_output_key", default="", help="S3 key for PyG output")
        parser.add_argument("--enable_ontology_mapping", default="false", help="Enable ontology mapping")
        parser.add_argument("--time_period", default="", help="Time period label")
        parser.add_argument("--rdf_format", default="turtle", help="RDF format")
        parser.add_argument("--pyg_config", default="", help="PyG config JSON")

        parsed = parser.parse_args()
        args = vars(parsed)

    return JobConfig(args)


# ============================================
# S3 operations
# ============================================
def list_rdf_files(s3_client, bucket: str, prefix: str, rdf_format: str = "turtle") -> list:
    """
    List all RDF files under an S3 prefix

    Supports pagination for large datasets (100+ ontology files).

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 key prefix
        rdf_format: Expected RDF format to filter by extension

    Returns:
        List of S3 keys
    """
    valid_extensions = {
        "turtle": [".ttl", ".turtle"],
        "xml": [".rdf", ".xml", ".owl"],
        "n3": [".n3"],
        "nt": [".nt", ".ntriples"],
        "json-ld": [".jsonld", ".json"],
    }

    extensions = valid_extensions.get(rdf_format, [".ttl"])
    rdf_keys = []

    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(key.lower().endswith(ext) for ext in extensions):
                rdf_keys.append(key)

    logger.info(f"Found {len(rdf_keys)} RDF files under s3://{bucket}/{prefix}")
    return rdf_keys


def load_rdf_from_s3(s3_client, bucket: str, keys: list, rdf_format: str = "turtle") -> Graph:
    """
    Load multiple RDF files from S3 into a single graph

    Uses RDFGraphLoader for proper namespace binding.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        keys: List of S3 keys to load
        rdf_format: RDF serialization format

    Returns:
        Merged RDFLib Graph with all triples and bound namespaces
    """
    loader = RDFGraphLoader(s3_client=s3_client)

    for i, key in enumerate(keys):
        logger.info(f"  Loading file {i + 1}/{len(keys)}: {key}")
        try:
            loader.load_from_s3(bucket, key, format=rdf_format)
        except Exception as e:
            logger.error(f"  Failed to load {key}: {e}")
            # Continue loading remaining files — partial data is better than none
            continue

    logger.info(f"Total graph size after loading: {len(loader.graph)} triples")
    return loader.graph


def save_enriched_rdf_to_s3(
    s3_client, graph: Graph, bucket: str, key: str, rdf_format: str = "turtle"
):
    """
    Save enriched RDF graph to S3

    Args:
        s3_client: Boto3 S3 client
        graph: Enriched RDFLib Graph
        bucket: S3 bucket name
        key: S3 key
        rdf_format: Serialization format
    """
    writer = RDFGraphWriter(s3_client=s3_client)
    writer.save_to_s3(graph, bucket, key, format=rdf_format)
    logger.info(f"Saved enriched RDF ({len(graph)} triples) to s3://{bucket}/{key}")


def save_pyg_to_s3(s3_client, hetero_data, bucket: str, key: str):
    """
    Save PyTorch Geometric HeteroData to S3

    Serializes using torch.save to a BytesIO buffer, then uploads to S3.

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

    size_mb = buffer.tell() / (1024 * 1024)
    logger.info(f"Saved PyG HeteroData ({size_mb:.2f} MB) to s3://{bucket}/{key}")


def load_enriched_rdf_from_s3(
    s3_client, bucket: str, key: str, rdf_format: str = "turtle"
) -> Graph:
    """
    Load enriched RDF from S3 (for pyg_only mode)

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 key for enriched RDF
        rdf_format: RDF serialization format

    Returns:
        RDFLib Graph with enriched triples
    """
    loader = RDFGraphLoader(s3_client=s3_client)
    graph = loader.load_from_s3(bucket, key, format=rdf_format)
    logger.info(f"Loaded enriched RDF ({len(graph)} triples) from s3://{bucket}/{key}")
    return graph


# ============================================
# Pipeline execution
# ============================================
def run_enrichment(
    graph: Graph, enable_ontology_mapping: bool = False
) -> Dict[str, Any]:
    """
    Run the enrichment pipeline on a loaded graph

    Args:
        graph: RDFLib Graph with raw RDF triples
        enable_ontology_mapping: Whether to run ontology mapping

    Returns:
        Enrichment statistics dictionary
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: RDF ENRICHMENT")
    logger.info("=" * 80)
    logger.info(f"Input graph size: {len(graph)} triples")
    logger.info(f"Ontology mapping: {'enabled' if enable_ontology_mapping else 'disabled'}")
    logger.info("")

    start_time = time.time()

    stats = enrich_graph(graph, enable_ontology_mapping=enable_ontology_mapping)

    elapsed = time.time() - start_time
    logger.info(f"Enrichment completed in {elapsed:.1f}s")

    return stats


def run_pyg_construction(
    graph: Graph, pyg_config: Dict[str, Any] = None
) -> Any:
    """
    Run PyG HeteroData construction from enriched RDF graph

    Args:
        graph: Enriched RDFLib Graph
        pyg_config: Optional configuration for PyG construction
            Example config:
            {
                "node_types": ["CPI_Index", "PPI_MonthlyChange", "JOLTS_Level", ...],
                "edge_types": ["precedes", "correlatesWith", "belongsToSector", ...],
                "feature_config": {
                    "numeric_properties": ["indexValue", "changeValue", "rate"],
                    "categorical_properties": ["hasCategory", "hasIndustry"],
                    "normalize": true
                },
                "include_temporal_nodes": true,
                "include_sector_nodes": true
            }

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
    logger.info(f"Input graph size: {len(graph)} triples")
    if pyg_config:
        logger.info(f"PyG config: {json.dumps(pyg_config, indent=2)}")
    logger.info("")

    start_time = time.time()

    # build_hetero_data is the expected interface from pyg_builder/constructor.py
    # Signature: build_hetero_data(graph: Graph, config: Dict) -> HeteroData
    config = pyg_config or {}
    hetero_data = build_hetero_data(graph, config)

    elapsed = time.time() - start_time

    # Log HeteroData summary
    logger.info(f"PyG construction completed in {elapsed:.1f}s")
    logger.info(f"HeteroData summary:")
    logger.info(f"  Node types: {hetero_data.node_types}")
    logger.info(f"  Edge types: {hetero_data.edge_types}")
    for node_type in hetero_data.node_types:
        store = hetero_data[node_type]
        num_nodes = store.num_nodes if hasattr(store, "num_nodes") else "unknown"
        num_features = store.x.shape[1] if hasattr(store, "x") and store.x is not None else 0
        logger.info(f"  [{node_type}] nodes={num_nodes}, features={num_features}")
    for edge_type in hetero_data.edge_types:
        store = hetero_data[edge_type]
        num_edges = store.edge_index.shape[1] if hasattr(store, "edge_index") else "unknown"
        logger.info(f"  [{edge_type}] edges={num_edges}")

    return hetero_data


# ============================================
# Main execution modes
# ============================================
def execute_full_pipeline(config: JobConfig, s3_client):
    """
    Mode: full
    Raw RDF → Enrich → Build PyG → Save both artifacts

    This is the default end-to-end mode.
    """
    logger.info("Executing FULL PIPELINE mode")
    logger.info("")

    # Step 1: Load raw RDF from S3
    logger.info("=" * 80)
    logger.info("PHASE: LOADING RAW RDF")
    logger.info("=" * 80)
    start_time = time.time()

    rdf_keys = list_rdf_files(
        s3_client, config.source_bucket, config.source_prefix, config.rdf_format
    )

    if not rdf_keys:
        raise FileNotFoundError(
            f"No RDF files found at s3://{config.source_bucket}/{config.source_prefix} "
            f"with format '{config.rdf_format}'"
        )

    graph = load_rdf_from_s3(s3_client, config.source_bucket, rdf_keys, config.rdf_format)

    load_elapsed = time.time() - start_time
    logger.info(f"Loading completed in {load_elapsed:.1f}s")
    logger.info("")

    # Step 2: Enrich
    enrichment_stats = run_enrichment(graph, config.enable_ontology_mapping)

    # Step 3: Save enriched RDF
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED RDF")
    logger.info("=" * 80)
    save_enriched_rdf_to_s3(
        s3_client, graph, config.output_bucket, config.enriched_rdf_key, config.rdf_format
    )

    # Step 4: Build PyG
    hetero_data = run_pyg_construction(graph, config.pyg_config)

    # Step 5: Save PyG
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA")
    logger.info("=" * 80)
    save_pyg_to_s3(s3_client, hetero_data, config.output_bucket, config.pyg_output_key)

    return {
        "mode": "full",
        "rdf_files_loaded": len(rdf_keys),
        "enrichment": enrichment_stats,
        "enriched_rdf_location": f"s3://{config.output_bucket}/{config.enriched_rdf_key}",
        "pyg_location": f"s3://{config.output_bucket}/{config.pyg_output_key}",
    }


def execute_enrichment_only(config: JobConfig, s3_client):
    """
    Mode: enrichment_only
    Raw RDF → Enrich → Save enriched RDF to S3

    Creates a reusable enriched RDF artifact that can be used
    for multiple PyG experiments without re-enrichment.
    """
    logger.info("Executing ENRICHMENT ONLY mode")
    logger.info("")

    # Step 1: Load raw RDF from S3
    logger.info("=" * 80)
    logger.info("PHASE: LOADING RAW RDF")
    logger.info("=" * 80)
    start_time = time.time()

    rdf_keys = list_rdf_files(
        s3_client, config.source_bucket, config.source_prefix, config.rdf_format
    )

    if not rdf_keys:
        raise FileNotFoundError(
            f"No RDF files found at s3://{config.source_bucket}/{config.source_prefix} "
            f"with format '{config.rdf_format}'"
        )

    graph = load_rdf_from_s3(s3_client, config.source_bucket, rdf_keys, config.rdf_format)

    load_elapsed = time.time() - start_time
    logger.info(f"Loading completed in {load_elapsed:.1f}s")
    logger.info("")

    # Step 2: Enrich
    enrichment_stats = run_enrichment(graph, config.enable_ontology_mapping)

    # Step 3: Save enriched RDF
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING ENRICHED RDF")
    logger.info("=" * 80)
    save_enriched_rdf_to_s3(
        s3_client, graph, config.output_bucket, config.enriched_rdf_key, config.rdf_format
    )

    return {
        "mode": "enrichment_only",
        "rdf_files_loaded": len(rdf_keys),
        "enrichment": enrichment_stats,
        "enriched_rdf_location": f"s3://{config.output_bucket}/{config.enriched_rdf_key}",
    }


def execute_pyg_only(config: JobConfig, s3_client):
    """
    Mode: pyg_only
    Enriched RDF (S3) → Build PyG → Save PyG to S3

    Rapid experimentation mode: loads existing enriched RDF
    and constructs PyG with different configurations.
    Typically completes in 5-10 minutes.
    """
    logger.info("Executing PyG ONLY mode")
    logger.info("")

    # Step 1: Load enriched RDF from S3
    logger.info("=" * 80)
    logger.info("PHASE: LOADING ENRICHED RDF")
    logger.info("=" * 80)
    start_time = time.time()

    graph = load_enriched_rdf_from_s3(
        s3_client, config.output_bucket, config.enriched_rdf_key, config.rdf_format
    )

    load_elapsed = time.time() - start_time
    logger.info(f"Loading completed in {load_elapsed:.1f}s")
    logger.info("")

    # Step 2: Build PyG
    hetero_data = run_pyg_construction(graph, config.pyg_config)

    # Step 3: Save PyG
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA")
    logger.info("=" * 80)
    save_pyg_to_s3(s3_client, hetero_data, config.output_bucket, config.pyg_output_key)

    return {
        "mode": "pyg_only",
        "enriched_rdf_source": f"s3://{config.output_bucket}/{config.enriched_rdf_key}",
        "pyg_location": f"s3://{config.output_bucket}/{config.pyg_output_key}",
    }


# ============================================
# Job summary
# ============================================
def save_job_manifest(s3_client, config: JobConfig, result: Dict, elapsed: float):
    """
    Save a JSON manifest with job metadata to S3

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
            "enriched_rdf_key": config.enriched_rdf_key,
            "pyg_output_key": config.pyg_output_key,
            "rdf_format": config.rdf_format,
            "enable_ontology_mapping": config.enable_ontology_mapping,
            "pyg_config": config.pyg_config,
        },
        "result": _make_json_serializable(result),
        "elapsed_seconds": round(elapsed, 2),
    }

    manifest_key = f"manifests/{config.time_period}/{config.mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        s3_client.put_object(
            Bucket=config.output_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved job manifest to s3://{config.output_bucket}/{manifest_key}")
    except Exception as e:
        logger.warning(f"Failed to save job manifest: {e}")


def _make_json_serializable(obj):
    """Recursively convert non-serializable types for JSON output"""
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
    """Print final job summary banner"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("JOB COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Mode:          {config.mode}")
    logger.info(f"  Time period:   {config.time_period}")
    logger.info(f"  Duration:      {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    logger.info("")

    if "enriched_rdf_location" in result:
        logger.info(f"  Enriched RDF:  {result['enriched_rdf_location']}")
    if "pyg_location" in result:
        logger.info(f"  PyG output:    {result['pyg_location']}")

    if "enrichment" in result and isinstance(result["enrichment"], dict):
        stats = result["enrichment"]
        logger.info("")
        logger.info(f"  Initial triples:  {stats.get('initial_triples', 'N/A'):>10}")
        logger.info(f"  Final triples:    {stats.get('final_triples', 'N/A'):>10}")
        logger.info(f"  Total enrichment: {stats.get('total_enrichment', 'N/A'):>10}")

    logger.info("")
    logger.info("=" * 80)


# ============================================
# Main entry point
# ============================================
def main():
    """
    Main entry point for the Glue job

    Parses arguments, initializes AWS clients, and dispatches
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
    # Initialize AWS clients
    # ----------------------------------------
    if RUNNING_IN_GLUE:
        sc = SparkContext.getOrCreate()
        glue_context = GlueContext(sc)
        s3_client = boto3.client("s3")
        logger.info("Initialized Glue context and S3 client")
    else:
        s3_client = boto3.client("s3")
        logger.info("Initialized S3 client (local mode)")

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
        result = handler(config, s3_client)

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
    # Glue cleanup
    # ----------------------------------------
    if RUNNING_IN_GLUE:
        logger.info("Committing Glue job...")
        # Note: If using Glue bookmarks or dynamic frames, commit here
        # glue_context.commit()

    logger.info("Done.")
    return result


if __name__ == "__main__":
    main()