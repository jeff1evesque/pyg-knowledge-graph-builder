"""Unit tests for the scheme-aware driver-side writer.

The bug these guard against (#197, and its recurrence in the .pt and metadata
writers) is not an exception — it is a *silent success*: plain ``open()`` on
``s3a://bucket/key`` writes a junk ``./s3a:/bucket/key`` tree under the driver's
current directory and returns normally. So the assertions here are about where
bytes land and about refusing to write at all when the destination cannot be
reached, not about happy-path round-trips.

No SparkSession is built: the Hadoop branch needs a live JVM and is covered by
the cluster suite (tests/e2e/test_cluster_submit.py). What is testable without
one — the routing decision, and the guard that turns an unreachable URI into a
loud failure instead of a quiet local write — is exactly what regressed twice.
"""

import json
import os

import pytest

from spark_jobs.utils.fs_utils import (
    is_local_path,
    join_path,
    local_filesystem_path,
    write_bytes,
)


# --------------------------------------------------------------------------- #
# scheme classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", [
    "/data/pyg/hetero_data.pt",
    "relative/path.json",
    "hetero_data.pt",
    "file:///data/pyg/hetero_data.pt",
])
def test_local_paths_are_local(path):
    assert is_local_path(path)


@pytest.mark.parametrize("path", [
    "s3a://bucket/pyg/hetero_data.pt",
    "s3://bucket/pyg/hetero_data.pt",
    "hdfs://namenode:8020/pyg/hetero_data.pt",
    "gs://bucket/pyg/hetero_data.pt",
])
def test_uri_paths_are_not_local(path):
    assert not is_local_path(path)


def test_file_uri_strips_its_scheme():
    assert local_filesystem_path("file:///data/x.pt") == "/data/x.pt"
    assert local_filesystem_path("/data/x.pt") == "/data/x.pt"


# --------------------------------------------------------------------------- #
# local writes
# --------------------------------------------------------------------------- #

def test_write_bytes_creates_parent_directories(tmp_path):
    dest = tmp_path / "pyg" / "year=2099" / "month=01" / "graph_schema.json"
    write_bytes(str(dest), b'{"version": "1.0"}')

    assert dest.is_file()
    assert json.loads(dest.read_text()) == {"version": "1.0"}


def test_write_bytes_overwrites(tmp_path):
    dest = tmp_path / "x.json"
    write_bytes(str(dest), b"first")
    write_bytes(str(dest), b"second")

    assert dest.read_bytes() == b"second"


def test_write_bytes_accepts_file_uri(tmp_path):
    dest = tmp_path / "sub" / "x.pt"
    write_bytes(f"file://{dest}", b"payload")

    assert dest.read_bytes() == b"payload"


# --------------------------------------------------------------------------- #
# the actual regression guard
# --------------------------------------------------------------------------- #

def test_uri_without_spark_raises_instead_of_writing_locally(tmp_path, monkeypatch):
    """A non-local URI with no SparkSession must fail loudly, not write junk.

    This is the whole point of the module. The pre-fix behavior was to succeed,
    log "Saved ... to s3a://...", and leave the bytes in ./s3a:/... on the
    driver. A caller that cannot supply a SparkSession cannot reach object
    storage, and pretending otherwise is what cost us the artifacts.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        write_bytes("s3a://bucket/pyg/hetero_data.pt", b"payload", spark=None)

    # The message must name the path -- a bare "SparkSession required" does not
    # tell you which of the several driver-side writers went wrong.
    assert "s3a://bucket/pyg/hetero_data.pt" in str(excinfo.value)

    # And nothing may have been written under the driver's cwd.
    assert not (tmp_path / "s3a:").exists()
    assert list(tmp_path.iterdir()) == []


def test_uri_with_spark_goes_to_hadoop_not_the_local_disk(tmp_path, monkeypatch):
    """The URI branch must call Hadoop's FileSystem, never open().

    Uses a fake JVM handle rather than a real SparkSession: the assertion is
    about which code path is taken and with what path/bytes, which needs no JVM.
    A real end-to-end s3a:// write is asserted by the cluster suite.
    """
    monkeypatch.chdir(tmp_path)

    written = {}

    class _Stream:
        def write(self, data):
            written["body"] = bytes(data)

        def close(self):
            written["closed"] = True

    class _FileSystem:
        @staticmethod
        def get(uri, conf):
            written["uri"] = uri
            return _FileSystem()

        def create(self, path, overwrite):
            written["path"] = path
            written["overwrite"] = overwrite
            return _Stream()

    class _JVM:
        class java:
            class net:
                URI = staticmethod(lambda s: f"URI({s})")

        class org:
            class apache:
                class hadoop:
                    class fs:
                        FileSystem = _FileSystem
                        Path = staticmethod(lambda s: f"Path({s})")

    class _FakeSpark:
        _jvm = _JVM

        class _jsc:
            @staticmethod
            def hadoopConfiguration():
                return "hadoop-conf"

    write_bytes("s3a://bucket/pyg/hetero_data.pt", b"payload", spark=_FakeSpark)

    assert written["path"] == "Path(s3a://bucket/pyg/hetero_data.pt)"
    assert written["uri"] == "URI(s3a://bucket/pyg/hetero_data.pt)"
    assert written["overwrite"] is True
    assert written["body"] == b"payload"
    assert written["closed"] is True

    # Nothing touched the driver's local disk.
    assert not (tmp_path / "s3a:").exists()
    assert list(tmp_path.iterdir()) == []


def test_stream_is_closed_even_when_the_write_fails(tmp_path):
    """A failed write must not leak the Hadoop output stream.

    On S3A the stream buffers and uploads on close(); leaking it on an error
    path holds the buffer and can leave a multipart upload dangling.
    """
    closed = []

    class _Stream:
        def write(self, data):
            raise IOError("connection reset")

        def close(self):
            closed.append(True)

    class _FileSystem:
        @staticmethod
        def get(uri, conf):
            return _FileSystem()

        def create(self, path, overwrite):
            return _Stream()

    class _JVM:
        class java:
            class net:
                URI = staticmethod(lambda s: s)

        class org:
            class apache:
                class hadoop:
                    class fs:
                        FileSystem = _FileSystem
                        Path = staticmethod(lambda s: s)

    class _FakeSpark:
        _jvm = _JVM

        class _jsc:
            @staticmethod
            def hadoopConfiguration():
                return "hadoop-conf"

    with pytest.raises(IOError):
        write_bytes("s3a://bucket/x.pt", b"payload", spark=_FakeSpark)

    assert closed == [True]


# --------------------------------------------------------------------------- #
# path joining
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("prefix,expected", [
    ("s3a://bucket/pyg/metadata", "s3a://bucket/pyg/metadata/graph_schema.json"),
    ("s3a://bucket/pyg/metadata/", "s3a://bucket/pyg/metadata/graph_schema.json"),
    ("/data/pyg/metadata/", "/data/pyg/metadata/graph_schema.json"),
])
def test_join_path_normalizes_the_separator(prefix, expected):
    """derive_metadata_prefix builds its result by string manipulation, so the
    trailing slash is not guaranteed; both forms must produce one separator."""
    assert join_path(prefix, "graph_schema.json") == expected


# --------------------------------------------------------------------------- #
# the two writers that regressed
# --------------------------------------------------------------------------- #

def test_metadata_writer_refuses_a_uri_without_spark(tmp_path, monkeypatch):
    """write_metadata_to_local must not silently localize an s3a:// prefix.

    Before the fix this wrote all six JSONs to ./s3a:/... and logged success.
    """
    from spark_jobs.pyg_builder.metadata_writer import write_metadata_to_local

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        write_metadata_to_local(
            {"graph_schema.json": {"version": "1.0"}},
            "s3a://bucket/pyg/year=2099/metadata/",
        )

    assert not (tmp_path / "s3a:").exists()


def test_save_pyg_refuses_a_uri_without_spark(tmp_path, monkeypatch):
    """save_pyg_local must not silently localize an s3a:// destination.

    Before the fix this torch.save()d the graph to ./s3a:/... and logged
    "Saved PyG HeteroData ... to s3a://...", so the job exited 0 with its single
    most important artifact stranded on the driver.
    """
    import torch
    from torch_geometric.data import HeteroData

    from spark_jobs.build_graph import save_pyg_local

    monkeypatch.chdir(tmp_path)

    data = HeteroData()
    data["thing"].x = torch.zeros(2, 3)

    with pytest.raises(ValueError):
        save_pyg_local(data, "s3a://bucket/pyg/year=2099/hetero_data.pt")

    assert not (tmp_path / "s3a:").exists()


def test_save_pyg_still_writes_plain_local_paths(tmp_path):
    """No regression for local runs: a bare POSIX path round-trips as before."""
    import torch
    from torch_geometric.data import HeteroData

    from spark_jobs.build_graph import save_pyg_local

    data = HeteroData()
    data["thing"].x = torch.zeros(2, 3)

    dest = tmp_path / "pyg" / "year=2099" / "hetero_data.pt"
    save_pyg_local(data, str(dest))

    assert dest.is_file()
    reloaded = torch.load(str(dest), weights_only=False)
    assert reloaded["thing"].x.shape == (2, 3)
