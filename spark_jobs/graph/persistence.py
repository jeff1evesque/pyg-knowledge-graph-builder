"""
Everything the job writes down, and everything it reads back.

Two kinds of artifact live here, and they are not the same kind of thing:

  * the INTERIM enriched Parquet plus its dataset.json descriptor, which exist
    so that `enrichment_only` and `pyg_only` can be two submissions instead of
    one -- the descriptor is how the second run learns what the first read
  * the FINAL artifacts -- the .pt, the six metadata JSON files, the node index
    and the job manifest -- written locally and, when an archive bucket is
    configured, mirrored to S3. Each driver-side one is digested as it is
    written and the digests land in checksums.json beside them, so a consumer
    can check the bytes it fetched before it loads them

Both go through the filesystem helpers rather than plain ``open()``, because
``local_work_dir`` may be a bare POSIX path or an ``s3a://`` URI and the job
must not care which.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from pyspark.sql import SparkSession, DataFrame

from spark_jobs.graph.config import JobConfig
from spark_jobs.graph.loading import source_label
from spark_jobs.pyg_builder.metadata_writer import (
    write_metadata_to_s3,
    write_metadata_to_local,
    write_latest_alias,
    write_checksums_to_local,
    write_checksums_to_s3,
    derive_metadata_prefix,
    derive_node_index_prefix,
    metadata_dir_name,
)

# The job's logger, not this module's -- see the note in config.py.
logger = logging.getLogger("build_graph")


# ============================================
# Timestamps
# ============================================
def utcnow() -> datetime:
    """Naive UTC now — drop-in for the deprecated datetime.utcnow().

    Returns a *naive* datetime (no tzinfo) so isoformat()/strftime()
    output stays byte-identical to the historical utcnow() call sites
    (a tz-aware isoformat() would insert "+00:00" before the manual "Z").
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


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


def save_dataset_descriptor(config: JobConfig, spark: SparkSession) -> None:
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
        "written": utcnow().isoformat(),
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
class _HashingWriter:
    """A file object that digests every byte on its way to the real one.

    ``torch.save`` takes any object with ``write`` and ``flush``, so the .pt's
    SHA-256 is computed during the write that already happens: no second read
    of a file that reaches tens of gigabytes, and no second copy in memory --
    the serializer hands over one memoryview per tensor storage and both the
    digest and the file read it in place. What it costs is CPU: SHA-256 runs at
    2.5 GB/s here against a write at ~2.0 GB/s, so about 0.4s per GB.

    ``write`` and ``flush`` are the whole surface torch uses for a normal save.
    Nothing else is implemented on purpose: a serializer that started seeking
    past bytes would get an AttributeError here rather than a digest quietly
    describing a file it does not match.

    Handing torch a file object rather than a path has a second effect worth
    knowing. Torch names the zip archive inside the .pt after the destination's
    basename, so the same graph written as ``hetero_data.pt`` and written
    through the ``mkstemp`` name the S3 and URI paths stage into came out as
    different bytes. A file object has no name -- torch calls the archive
    "archive" every time -- so every destination now gets identical bytes and
    one digest describes them all.
    """

    def __init__(self, handle):
        self._handle = handle
        self._digest = hashlib.sha256()

    def write(self, chunk):
        self._digest.update(chunk)
        return self._handle.write(chunk)

    def flush(self):
        return self._handle.flush()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _save_hetero_data(hetero_data, path: str) -> Dict[str, Any]:
    """Serialize to ``path``; return its byte count and SHA-256.

    The size comes from the finished file rather than a counter in the writer,
    so it is what a verifier will actually stat.
    """
    import os

    import torch

    with open(path, "wb") as handle:
        writer = _HashingWriter(handle)
        torch.save(hetero_data, writer)

    return {"bytes": os.path.getsize(path), "sha256": writer.hexdigest()}


def _artifact_records(
    pyg_key: str,
    pt_record: Dict[str, Any],
    metadata_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Name each digested artifact relative to the period directory.

    ``pyg/year=2024/month=12/hetero_data.pt`` becomes ``hetero_data.pt``, and
    each metadata file becomes ``metadata/<name>`` (or the variant's derived
    directory). The period directory is the unit that gets copied, archived or
    deleted, so names relative to it keep verifying after a copy.
    """
    records = {pyg_key.rsplit("/", 1)[-1]: pt_record}
    directory = metadata_dir_name(pyg_key)
    records.update({
        f"{directory}/{name}": record
        for name, record in metadata_records.items()
    })
    return records


def save_pyg_to_s3(s3_client, hetero_data, bucket: str, key: str) -> Dict[str, Any]:
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

    Returns:
        {"bytes", "sha256"} over the object uploaded.
    """
    import os
    import tempfile

    handle, staged = tempfile.mkstemp(prefix="hetero_data.", suffix=".pt")
    os.close(handle)
    try:
        record = _save_hetero_data(hetero_data, staged)
        size_mb = record["bytes"] / (1024 * 1024)
        s3_client.upload_file(
            Filename=staged,
            Bucket=bucket,
            Key=key,
            ExtraArgs={
                "ContentType": "application/octet-stream",
                # S3 verifies the transfer against this and returns it from
                # HeadObject, so a consumer can ask what an object is without
                # downloading 42 GB of it. At this size the upload is
                # multipart, which makes the stored value a checksum OF the
                # part checksums with a "-N" suffix -- NOT the file's SHA-256.
                # The whole-file digest is the one in checksums.json. (It is
                # the same reason ETag is not a content hash.)
                "ChecksumAlgorithm": "SHA256",
            },
        )
    finally:
        os.unlink(staged)

    logger.info(
        f"Saved PyG HeteroData ({size_mb:.2f} MB, sha256 {record['sha256']}) "
        f"to s3://{bucket}/{key}"
    )
    return record


def save_pyg_local(
    hetero_data, local_path: str, spark: SparkSession = None
) -> Dict[str, Any]:
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

    Returns:
        {"bytes", "sha256"} over the file written. Hadoop copies the staged
        file across byte for byte, so the digest describes the destination in
        both branches.
    """
    import os
    import tempfile

    from spark_jobs.utils.fs_utils import (
        is_local_path,
        local_filesystem_path,
        write_file,
    )

    if is_local_path(local_path):
        path = local_filesystem_path(local_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        record = _save_hetero_data(hetero_data, path)
        size_mb = record["bytes"] / (1024 * 1024)
        logger.info(
            f"Saved PyG HeteroData ({size_mb:.2f} MB, "
            f"sha256 {record['sha256']}) to {path}"
        )
        return record

    handle, staged = tempfile.mkstemp(prefix="hetero_data.", suffix=".pt")
    os.close(handle)
    try:
        record = _save_hetero_data(hetero_data, staged)
        size_mb = record["bytes"] / (1024 * 1024)
        write_file(staged, local_path, spark=spark)
    finally:
        # write_file consumes the staged file on success; this is the failure path.
        if os.path.exists(staged):
            os.unlink(staged)

    logger.info(
        f"Saved PyG HeteroData ({size_mb:.2f} MB, sha256 {record['sha256']}) "
        f"to {local_path} (via Hadoop FS)"
    )
    return record


# ============================================
# Final-artifact persistence (local + optional S3 mirror)
# ============================================
def save_node_index(node_index_df: DataFrame, output_path: str) -> None:
    """Write the (node_type, node_id) -> uri identity map beside the .pt.

    hetero_data.pt stores only feature tensors and counts, so without this the
    graph is anonymous: nothing says which real-world entity each row is, which
    blocks joining training labels and attributing predictions.

    Parquet rather than a seventh JSON: production is 322.7M triples for a
    single four-source day, so this can reach millions of rows. A single JSON
    would be hundreds of MB and must be parsed whole to resolve one entity,
    where Parquet supports predicate pushdown and matches how the rest of the
    pipeline stores bulk data.

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

    Every driver-side artifact is digested as it is written, and checksums.json
    goes in last -- so whoever fetches the .pt can tell it is the file this job
    wrote before torch.load() unpickles it. The node index is not covered: it
    is written on executors, so digesting it would mean the driver re-reading
    every part file.

    Returns a dict of output locations for the job result/manifest, plus the
    `artifacts` block those digests make up.
    """
    metadata_files = metadata.to_metadata_files()

    # --- Job work dir (always) ---
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE: SAVING PyG HETERODATA + METADATA")
    logger.info("=" * 80)
    pt_record = save_pyg_local(hetero_data, config.pyg_output_path, spark=spark)
    local_metadata_dir = derive_metadata_prefix(config.pyg_output_path)
    metadata_records = write_metadata_to_local(
        metadata_files, local_metadata_dir, spark=spark
    )

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

    # Last, so it describes a directory that is already complete.
    artifacts = _artifact_records(
        config.pyg_output_path, pt_record, metadata_records
    )
    write_checksums_to_local(artifacts, local_metadata_dir, spark=spark)

    locations: Dict[str, Any] = {
        "node_index_location": node_index_dir,
        "pyg_location": config.pyg_output_path,
        "metadata_location": local_metadata_dir,
        "latest_metadata_location": latest_metadata_dir,
        # Also in checksums.json beside the artifacts. Both, deliberately: the
        # manifest answers "what did run X produce", and checksums.json travels
        # with the period directory when it is copied somewhere the manifest
        # is not -- and manifests accumulate one per run, so someone holding a
        # .pt cannot tell which of them describes it.
        "artifacts": artifacts,
    }

    # --- S3 mirror (optional) ---
    if config.archive_to_s3:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE: ARCHIVING FINAL ARTIFACTS TO S3")
        logger.info("=" * 80)
        s3_metadata_prefix = derive_metadata_prefix(config.s3_pyg_key)
        s3_pt_record = save_pyg_to_s3(
            s3_client, hetero_data,
            config.s3_archive_bucket, config.s3_pyg_key,
        )
        s3_metadata_records = write_metadata_to_s3(
            s3_client, metadata_files,
            config.s3_archive_bucket, s3_metadata_prefix,
        )
        write_checksums_to_s3(
            s3_client,
            _artifact_records(
                config.s3_pyg_key, s3_pt_record, s3_metadata_records
            ),
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
# Job manifest
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
        "job_timestamp": utcnow().isoformat() + "Z",
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
        f"{config.mode}_{utcnow().strftime('%Y%m%d_%H%M%S')}.json"
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
                # Single-part at this size, so S3 stores the body's own
                # SHA-256 and verifies it on receipt.
                ChecksumAlgorithm="SHA256",
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
