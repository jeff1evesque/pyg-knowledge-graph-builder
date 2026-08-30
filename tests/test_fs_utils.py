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

import pytest

from spark_jobs.utils.fs_utils import (
    is_local_path,
    join_path,
    local_filesystem_path,
    path_exists,
    write_bytes,
    write_file,
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
# existence, routed the same way as the writes
# --------------------------------------------------------------------------- #

def test_path_exists_answers_for_local_paths(tmp_path):
    present = tmp_path / "thing.pt"
    present.write_bytes(b"x")

    assert path_exists(str(present))
    assert not path_exists(str(tmp_path / "absent.pt"))


def test_path_exists_sees_directories(tmp_path):
    """The enriched Parquet is a directory, and its _SUCCESS marker a file
    inside it — the occupancy preflight tests both shapes."""
    directory = tmp_path / "triples"
    directory.mkdir()
    assert path_exists(str(directory))


def test_path_exists_accepts_a_file_uri(tmp_path):
    present = tmp_path / "thing.pt"
    present.write_bytes(b"x")
    assert path_exists(f"file://{present}")


def test_path_exists_on_a_uri_without_spark_raises(tmp_path, monkeypatch):
    """False would be indistinguishable from a real answer, and wrong in the
    direction that overwrites someone's finished run."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="non-local URI"):
        path_exists("s3a://bucket/pyg/hetero_data.pt")


def _fake_spark(answer, record):
    """A JVM handle that records what it was asked and returns ``answer``.

    Same shape as the write test above, and for the same reason: the assertion
    is about which code path runs and with what arguments, which needs no JVM.
    """
    class _FileSystem:
        @staticmethod
        def get(uri, conf):
            record["uri"] = uri
            record["conf"] = conf
            return _FileSystem()

        def exists(self, path):
            record["path"] = path
            return answer

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

    return _FakeSpark


@pytest.mark.parametrize("answer", [True, False])
def test_path_exists_on_a_uri_asks_hadoop_not_the_local_disk(
    tmp_path, monkeypatch, answer
):
    """The URI branch must go through Hadoop's FileSystem, both ways.

    This is the branch that decides whether an object-store work dir looks
    occupied. Answering it from the driver's local disk would report every
    s3a:// destination as free and overwrite a finished run.
    """
    monkeypatch.chdir(tmp_path)
    record = {}

    result = path_exists(
        "s3a://bucket/pyg/hetero_data.pt",
        spark=_fake_spark(answer, record),
    )

    assert result is answer
    assert record["uri"] == "URI(s3a://bucket/pyg/hetero_data.pt)"
    assert record["path"] == "Path(s3a://bucket/pyg/hetero_data.pt)"
    assert record["conf"] == "hadoop-conf"

    # Nothing consulted, or created on, the driver's own filesystem.
    assert list(tmp_path.iterdir()) == []


def test_path_exists_returns_a_real_bool_not_a_java_object(tmp_path, monkeypatch):
    """``fs.exists`` comes back through Py4J; the caller branches on it, so it
    has to be a Python bool rather than something merely truthy."""
    monkeypatch.chdir(tmp_path)

    class _Truthy:
        def __bool__(self):
            return True

    result = path_exists(
        "s3a://bucket/pyg/hetero_data.pt", spark=_fake_spark(_Truthy(), {})
    )
    assert result is True


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
# write_file — the streaming path for payloads too big to buffer
# --------------------------------------------------------------------------- #

def test_write_file_moves_a_local_file_and_creates_parents(tmp_path):
    src = tmp_path / "staged.pt"
    src.write_bytes(b"payload")
    dest = tmp_path / "nested" / "deeper" / "hetero_data.pt"

    write_file(str(src), str(dest))

    assert dest.read_bytes() == b"payload"
    assert not src.exists(), "the staged file must not be left behind"


def test_write_file_accepts_file_uri(tmp_path):
    src = tmp_path / "staged.pt"
    src.write_bytes(b"payload")
    dest = tmp_path / "hetero_data.pt"

    write_file(str(src), f"file://{dest}")

    assert dest.read_bytes() == b"payload"


def test_write_file_refuses_a_uri_without_spark(tmp_path):
    """Same guard as write_bytes: a non-local URI with no session must raise.

    Falling back to local I/O here would move the graph into a junk './s3a:/...'
    tree and report success -- the exact silent failure this module exists for.
    """
    src = tmp_path / "staged.pt"
    src.write_bytes(b"payload")
    monkey_cwd = tmp_path / "cwd"
    monkey_cwd.mkdir()

    with pytest.raises(ValueError) as excinfo:
        write_file(str(src), "s3a://bucket/pyg/hetero_data.pt")

    assert "s3a://bucket/pyg/hetero_data.pt" in str(excinfo.value)
    assert not (monkey_cwd / "s3a:").exists()
    assert src.exists(), "a refused write must leave the staged file for the caller"


def test_write_file_uri_streams_through_hadoop_copy(tmp_path, monkeypatch):
    """The URI branch must hand Hadoop the FILE, not the file's bytes.

    copyFromLocalFile streams in blocks. The buffered alternative -- read it all,
    ship it over Py4J -- needs the whole payload resident in the Python heap AND
    again in the JVM's, which a graph measured in tens of gigabytes cannot do.
    """
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "staged.pt"
    src.write_bytes(b"payload")

    called = {}

    class _FileSystem:
        @staticmethod
        def get(uri, conf):
            called["uri"] = uri
            return _FileSystem()

        def copyFromLocalFile(self, del_src, overwrite, src_path, dst_path):
            called["del_src"] = del_src
            called["overwrite"] = overwrite
            called["src"] = src_path
            called["dst"] = dst_path

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

    write_file(str(src), "s3a://bucket/pyg/hetero_data.pt", spark=_FakeSpark)

    assert called["uri"] == "URI(s3a://bucket/pyg/hetero_data.pt)"
    assert called["dst"] == "Path(s3a://bucket/pyg/hetero_data.pt)"
    assert called["src"] == f"Path(file://{src})"
    assert called["overwrite"] is True
    assert called["del_src"] is True, "Hadoop must clean up the staged file"

    # Nothing landed on the driver's local disk under the URI's scheme.
    assert not (tmp_path / "s3a:").exists()


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
