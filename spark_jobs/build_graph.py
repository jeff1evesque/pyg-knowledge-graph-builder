"""
PyTorch Geometric Knowledge Graph Builder - Spark job entry point

Apache Spark job (runs on a standalone Spark cluster with the RAPIDS
Accelerator for Apache Spark) that orchestrates the complete pipeline:
  Raw RDF → PySpark triples_df → Enrich → Build PyG HeteroData → Save

Storage model (local-first):
  - Interim enriched Parquet is written to a shared local working
    directory only (regenerable; avoids remote-object-store write cost).
  - Final artifacts (the .pt HeteroData and the seven metadata JSON
    files) are written locally AND, when an S3 archive is configured,
    mirrored to S3 as a durable, reusable catalog. Each is digested as it
    is written and checksums.json goes in last, so a consumer can check
    the bytes it fetched before torch.load unpickles them.

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
from typing import Dict, Any, Optional, Tuple, List

from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
import boto3

# ============================================
# Project imports
# ============================================
from spark_jobs.enrichment.pipeline import EnrichmentPipeline

# Getting triples in, and the per-source accounting that goes with it. The
# Turtle parsing under it runs on executors and lives in graph/turtle.py, whose
# imports are constrained; nothing here is.
from spark_jobs.graph.loading import load_source_triples, source_label

# Reading and writing the job's artifacts -- the interim enriched Parquet and
# its descriptor, the final .pt and metadata, the manifest.
from spark_jobs.graph.persistence import (
    load_dataset_descriptor,
    load_enriched_parquet,
    save_dataset_descriptor,
    save_enriched_parquet,
    save_final_artifacts,
    save_job_manifest,
    utcnow,
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
# Final summary
# ============================================
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
    logger.info(f"Start time: {utcnow().isoformat()}Z")
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