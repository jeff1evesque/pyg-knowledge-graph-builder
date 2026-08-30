"""
Filesystem access that honors the path's URI scheme.

``local_work_dir`` may be a bare POSIX path (``/data``) or a URI on shared
storage (``s3a://bucket/prefix``). Every driver-side artifact — the job
manifest, the ``.pt`` HeteroData, the six metadata JSONs — is written with
plain Python I/O, and plain Python I/O treats ``s3a://bucket/x`` as a *relative
path*: it creates a junk ``./s3a:/bucket/x`` tree on the driver's local disk and
reports success. Nothing raises, so the job exits 0 having written its most
important outputs nowhere anyone will look.

This module is the single place that decides local-vs-Hadoop, so a new
driver-side writer cannot reintroduce the bug by forgetting to check.

Distributed writes (Spark's own ``.write.parquet(...)``) already resolve the
scheme through Hadoop and must NOT come through here.
"""

import logging
import os
import shutil
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_local_path(path: str) -> bool:
    """Whether ``path`` addresses the driver's own filesystem.

    Bare POSIX paths and explicit ``file://`` URIs are local; anything carrying
    another scheme (``s3a://``, ``hdfs://``, ``gs://``) is not. Windows drive
    letters are not considered: this runs on the Spark driver, on Linux.
    """
    scheme = urlparse(path).scheme
    return not scheme or scheme == "file"


def local_filesystem_path(path: str) -> str:
    """The on-disk path for a local URI or bare path (strips ``file://``)."""
    return urlparse(path).path if urlparse(path).scheme == "file" else path


def write_bytes(path: str, body, spark=None) -> None:
    """Write ``body`` to ``path``, routing by the path's URI scheme.

    ``body`` is any bytes-like object.

    FOR SMALL PAYLOADS ONLY — manifests, metadata, anything measured in
    megabytes. The non-local branch has to hand the bytes across Py4J, which
    copies them into a JVM array, so the whole payload has to fit the driver
    heap on top of the copy Python is already holding. Use ``write_file`` for
    anything large: it streams from disk and never materializes the payload in
    either heap.

    Local paths take the direct filesystem call (creating parent directories).
    Non-local URIs go through the Hadoop FileSystem API, so they land on the
    same filesystem — with the same S3A endpoint and IAM credentials — that
    Spark writes every other artifact to. Hadoop creates parent "directories"
    itself, and on object storage there are none to create.

    Args:
        path: Destination, bare path or URI.
        spark: Active SparkSession. Required for non-local URIs — it is the only
            handle to the JVM's Hadoop configuration. Passing None for a
            non-local URI raises rather than silently falling back to local I/O,
            which is the failure this module exists to prevent.

    Raises:
        ValueError: ``path`` is a non-local URI and no SparkSession was given.
    """
    if is_local_path(path):
        local_path = local_filesystem_path(path)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as handle:
            handle.write(body)
        return

    if spark is None:
        raise ValueError(
            f"cannot write to {path!r}: it is a non-local URI and no "
            "SparkSession was supplied to resolve it. Writing it with plain "
            "local I/O would create a junk './"
            f"{urlparse(path).scheme}:/...' tree on the driver's disk instead "
            "of reaching shared storage."
        )

    jvm = spark._jvm
    hadoop_conf = spark._jsc.hadoopConfiguration()
    juri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(juri, hadoop_conf)
    jpath = jvm.org.apache.hadoop.fs.Path(path)
    stream = fs.create(jpath, True)  # overwrite
    try:
        stream.write(bytearray(body))
    finally:
        stream.close()


def write_file(src_path: str, path: str, spark=None) -> None:
    """Move the file at ``src_path`` to ``path``, routing by URI scheme.

    The streaming counterpart to ``write_bytes``, for payloads too big to hold
    in memory. Hadoop copies the file across in blocks, so peak memory is a
    block rather than the whole file — which is what makes a graph measured in
    tens of gigabytes writable at all.

    ``src_path`` is consumed: on success it no longer exists, whichever branch
    ran. Callers stage into a temp file and hand it over.

    Args:
        src_path: Local file to move.
        path: Destination, bare path or URI.
        spark: Active SparkSession. Required for non-local URIs, for the same
            reason as ``write_bytes``.

    Raises:
        ValueError: ``path`` is a non-local URI and no SparkSession was given.
    """
    if is_local_path(path):
        local_path = local_filesystem_path(path)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.move(src_path, local_path)
        return

    if spark is None:
        raise ValueError(
            f"cannot write to {path!r}: it is a non-local URI and no "
            "SparkSession was supplied to resolve it. Writing it with plain "
            "local I/O would create a junk './"
            f"{urlparse(path).scheme}:/...' tree on the driver's disk instead "
            "of reaching shared storage."
        )

    jvm = spark._jvm
    hadoop_conf = spark._jsc.hadoopConfiguration()
    juri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(juri, hadoop_conf)
    fs.copyFromLocalFile(
        True,   # delSrc — the staged file has no purpose once it is across
        True,   # overwrite
        jvm.org.apache.hadoop.fs.Path("file://" + os.path.abspath(src_path)),
        jvm.org.apache.hadoop.fs.Path(path),
    )


def path_exists(path: str, spark=None) -> bool:
    """Whether ``path`` exists, routing by the path's URI scheme.

    The read-side counterpart to ``write_bytes``. It matters for the same
    reason: ``os.path.exists("s3a://bucket/key")`` asks the driver's local disk
    about a path that was never going to be there and answers False. A caller
    using that to decide whether a destination is free would conclude every
    object-store destination is free, every time.

    Args:
        path: Bare path or URI.
        spark: Active SparkSession. Required for non-local URIs — it is the only
            handle to the JVM's Hadoop configuration. Passing None for a
            non-local URI raises rather than answering False, because False is
            indistinguishable from a real answer and is wrong in the direction
            that loses data.

    Raises:
        ValueError: ``path`` is a non-local URI and no SparkSession was given.
    """
    if is_local_path(path):
        return os.path.exists(local_filesystem_path(path))

    if spark is None:
        raise ValueError(
            f"cannot test {path!r} for existence: it is a non-local URI and no "
            "SparkSession was supplied to resolve it. Answering from the "
            "driver's local disk would report every object-store path as "
            "missing."
        )

    jvm = spark._jvm
    hadoop_conf = spark._jsc.hadoopConfiguration()
    juri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(juri, hadoop_conf)
    return bool(fs.exists(jvm.org.apache.hadoop.fs.Path(path)))


def join_path(prefix: str, name: str) -> str:
    """Join a path segment onto a path or URI.

    ``os.path.join`` is wrong for URIs on principle even though it happens to
    work for the forward-slash case; this also normalizes a missing or doubled
    separator, which matters because the metadata prefix is derived by string
    manipulation and its trailing slash is not guaranteed.
    """
    return f"{prefix.rstrip('/')}/{name.lstrip('/')}"
