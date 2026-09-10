"""Integrity digests for what a build writes down (issue #400).

Nothing a build produced could be checked after the fact. That matters more
than a corrupted download usually would, because ``hetero_data.pt`` is a pickle
and every reader loads it with ``weights_only=False`` — a swapped artifact is
arbitrary code execution on whoever trains against it, not merely a wrong
graph. ``checksums.json`` is the control that makes it checkable *before* the
load.

Two properties are worth pinning here and nowhere else:

  * the digest is taken during the write that already happens, so it describes
    the bytes that actually landed rather than a re-read that could differ
  * the record is period-root-relative, so a copied period directory still
    verifies from files inside it — which is the whole point of writing it
    beside the artifacts instead of only into the job manifest

Content-level assertions about the six metadata files stay in
``test_metadata_writer.py``; the manifest's ``artifacts`` block is pinned in
``test_job_manifest.py``.
"""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_jobs.graph.persistence import (
    _HashingWriter,
    _save_hetero_data,
    save_final_artifacts,
    save_pyg_local,
)
from spark_jobs.pyg_builder.metadata_writer import (
    CHECKSUMS_FILE,
    derive_metadata_prefix,
)


METADATA_FILES = {
    "graph_schema.json": {"version": "1.2", "node_types": {"thing": {"count": 3}}},
    "feature_spec.json": {"version": "1.0"},
    "normalization.json": {"version": "1.0"},
    "encoding_config.json": {"version": "1.0"},
    "ontology_schema.json": {"version": "1.0"},
    "slot_mapping.json": {"version": "1.0"},
}


def _hetero(rows=3, cols=4, fill=0.0):
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["thing"].x = torch.full((rows, cols), fill)
    return data


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(work_dir, **overrides):
    """The fields save_final_artifacts() reads, and nothing else."""
    period = f"{work_dir}/pyg/year=2099/month=01"
    config = SimpleNamespace(
        pyg_output_path=f"{period}/hetero_data.pt",
        latest_pyg_path=f"{work_dir}/pyg/latest/hetero_data.pt",
        archive_to_s3=False,
        s3_archive_bucket="",
        s3_pyg_key="",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _Metadata:
    """Stands in for MetadataCollector: save_final_artifacts only asks for this."""

    def to_metadata_files(self):
        return dict(METADATA_FILES)


class _FakeS3:
    """Records what boto3 was asked to send, including the checksum arguments."""

    def __init__(self):
        self.uploads = []
        self.objects = {}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.uploads.append(
            {"key": Key, "extra": ExtraArgs or {},
             "body": Path(Filename).read_bytes()}
        )

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = {"body": Body, **kwargs}


# ======================================================================
# The hashing writer — digest during the write, not from a second read
# ======================================================================

def test_hashed_save_digests_exactly_the_bytes_on_disk(tmp_path):
    """The whole feature rests on this: hash the write, not a re-read.

    A second pass over a 42 GB .pt is what the wrapper exists to avoid, so the
    digest has to be provably the same one that pass would have produced.
    """
    path = tmp_path / "hetero_data.pt"

    record = _save_hetero_data(_hetero(), str(path))

    assert record["sha256"] == _sha256(path)
    assert record["bytes"] == path.stat().st_size


def test_hashed_save_round_trips_through_torch_load(tmp_path):
    """Digesting must not disturb the stream it is digesting.

    torch.save takes a file OBJECT here rather than a path, which is a
    different serializer entry point; the artifact still has to load.
    """
    import torch

    path = tmp_path / "hetero_data.pt"
    _save_hetero_data(_hetero(rows=5, cols=6, fill=1.5), str(path))

    reloaded = torch.load(str(path), weights_only=False)
    assert reloaded["thing"].x.shape == (5, 6)
    assert float(reloaded["thing"].x[0][0]) == 1.5


def test_the_same_graph_digests_the_same_from_every_destination(tmp_path):
    """One digest has to describe the work-dir copy and the S3 copy alike.

    It did not before. torch bakes the destination's basename into the zip
    archive it writes, and the S3 and URI paths stage through ``mkstemp``, so
    the same graph serialized as ``hetero_data.pt`` and as
    ``hetero_data.4f2a.pt`` came out as different bytes. Handing torch a file
    object drops the name from the archive, which is what makes the digests
    agree — assert it with the two names that actually occur.
    """
    named = _save_hetero_data(_hetero(), str(tmp_path / "hetero_data.pt"))
    staged = _save_hetero_data(_hetero(), str(tmp_path / "hetero_data.4f2a.pt"))

    assert named["sha256"] == staged["sha256"]
    assert named["bytes"] == staged["bytes"]


def test_digest_changes_when_the_graph_changes(tmp_path):
    """Guards the degenerate pass: a constant digest satisfies every test above."""
    # Same filename in two directories, so content is the only thing differing.
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    one = _save_hetero_data(_hetero(fill=0.0), str(tmp_path / "a" / "hetero_data.pt"))
    other = _save_hetero_data(_hetero(fill=1.0), str(tmp_path / "b" / "hetero_data.pt"))

    assert one["sha256"] != other["sha256"]


def test_hashing_writer_passes_every_chunk_through_untouched(tmp_path):
    """The wrapper is a passthrough; anything it drops is silent data loss."""
    path = tmp_path / "raw.bin"
    chunks = [b"", b"one", memoryview(b"two"), bytearray(b"three")]

    with open(path, "wb") as handle:
        writer = _HashingWriter(handle)
        for chunk in chunks:
            writer.write(chunk)
        writer.flush()

    assert path.read_bytes() == b"onetwothree"
    assert writer.hexdigest() == _sha256(path)


def test_save_pyg_local_reports_the_digest_of_the_file_it_wrote(tmp_path):
    """The in-place branch: what the caller gets back must describe the file."""
    dest = tmp_path / "pyg" / "year=2099" / "hetero_data.pt"

    record = save_pyg_local(_hetero(), str(dest))

    assert record == {"bytes": dest.stat().st_size, "sha256": _sha256(dest)}


def test_save_pyg_uri_digests_the_bytes_that_land(spark, tmp_path):
    """The staged branch: Hadoop copies the temp file across, so the digest holds.

    A ``file://`` URI takes the same non-local code path as ``s3a://`` — staged
    into a temp file, then moved by the Hadoop FileSystem — but resolves
    locally, so the destination can be read back and compared here.
    """
    dest = tmp_path / "pyg" / "year=2099" / "hetero_data.pt"

    record = save_pyg_local(_hetero(), f"file://{dest}", spark=spark)

    assert dest.is_file()
    assert record["sha256"] == _sha256(dest)
    assert record["bytes"] == dest.stat().st_size


# ======================================================================
# checksums.json — a period directory verifies from inside itself
# ======================================================================

def _verify(period_dir: Path):
    """Do what a consumer does: read checksums.json, re-hash what it names."""
    checksums = json.loads(
        (period_dir / "metadata" / CHECKSUMS_FILE).read_text()
    )
    assert checksums["algorithm"] == "sha256"

    verified = {}
    for relative, record in checksums["artifacts"].items():
        target = period_dir / relative
        assert target.is_file(), f"checksums.json names a missing {relative}"
        body = target.read_bytes()
        verified[relative] = (
            len(body) == record["bytes"]
            and hashlib.sha256(body).hexdigest() == record["sha256"]
        )
    return verified


def test_period_directory_verifies_from_files_inside_it(tmp_path):
    """The acceptance criterion, end to end over the real write path."""
    config = _config(str(tmp_path))

    save_final_artifacts(config, None, _hetero(), _Metadata())

    period = Path(config.pyg_output_path).parent
    verified = _verify(period)
    assert set(verified) == {"hetero_data.pt"} | {
        f"metadata/{name}" for name in METADATA_FILES
    }
    assert all(verified.values()), f"digest mismatch: {verified}"


def test_the_record_survives_a_copy_of_the_period_directory(tmp_path):
    """Why the paths are period-root-relative rather than absolute.

    The period directory is the unit that gets copied, archived or deleted, so
    the artifact that says what is in it has to keep resolving somewhere else.
    """
    config = _config(str(tmp_path))
    save_final_artifacts(config, None, _hetero(), _Metadata())

    elsewhere = tmp_path / "downloaded"
    shutil.copytree(Path(config.pyg_output_path).parent, elsewhere)

    assert all(_verify(elsewhere).values())


def test_a_tampered_pt_fails_the_check(tmp_path):
    """The case the feature exists for: swapped bytes must not verify.

    A replaced .pt is arbitrary code execution on the next reader, since every
    consumer loads it with weights_only=False.
    """
    config = _config(str(tmp_path))
    save_final_artifacts(config, None, _hetero(), _Metadata())

    period = Path(config.pyg_output_path).parent
    swapped = _save_hetero_data(_hetero(fill=9.0), str(period / "hetero_data.pt"))

    verified = _verify(period)
    assert verified["hetero_data.pt"] is False
    assert all(
        ok for name, ok in verified.items() if name != "hetero_data.pt"
    ), "only the tampered artifact should fail"
    assert swapped["sha256"] == _sha256(period / "hetero_data.pt")


def test_a_uri_work_dir_gets_the_same_record_through_hadoop(
    spark, tmp_path, monkeypatch
):
    """The cluster's configuration, where the work dir is an object-store URI.

    Every driver-side artifact then goes through the Hadoop FileSystem instead
    of open(), and a writer that forgets that routing writes a junk ``./file:``
    tree on the driver's local disk and logs success — the defect fs_utils
    exists to prevent, reintroduced one file at a time. The negative test in
    test_metadata_writer.py proves the record REFUSES a URI with no session;
    this proves it reaches one when it has it.

    ``file://`` takes the same non-local branch as ``s3a://`` and resolves
    somewhere readable, so the digests can be checked against real files.
    """
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "cluster"
    config = _config(f"file://{work}")

    locations = save_final_artifacts(
        config, None, _hetero(), _Metadata(), spark=spark
    )

    assert not (tmp_path / "file:").exists(), "wrote to a literal ./file: dir"

    period = work / "pyg" / "year=2099" / "month=01"
    verified = _verify(period)
    assert set(verified) == {"hetero_data.pt"} | {
        f"metadata/{name}" for name in METADATA_FILES
    }
    assert all(verified.values()), f"digest mismatch: {verified}"
    assert locations["artifacts"] == json.loads(
        (period / "metadata" / CHECKSUMS_FILE).read_text()
    )["artifacts"]


def test_checksums_are_written_into_the_variant_directory(tmp_path):
    """Two experiment variants in one period must not overwrite each other.

    The metadata directory is already derived per variant, which is why the
    record lives there rather than at the period root — a root-level file would
    need a naming convention of its own.
    """
    period = f"{tmp_path}/pyg/year=2099/month=01"
    config = _config(str(tmp_path), pyg_output_path=f"{period}/hetero_data_512d.pt")

    save_final_artifacts(config, None, _hetero(), _Metadata())

    variant = Path(period) / "hetero_data_512d_metadata" / CHECKSUMS_FILE
    assert variant.is_file()
    assert not (Path(period) / "metadata").exists()

    named = set(json.loads(variant.read_text())["artifacts"])
    assert "hetero_data_512d.pt" in named
    assert "hetero_data_512d_metadata/graph_schema.json" in named


def test_the_manifest_block_is_what_checksums_json_records(tmp_path):
    """save_final_artifacts hands the same digests to the job manifest.

    Both homes, deliberately: manifests accumulate one per run under a
    different prefix, so someone holding a .pt cannot tell which describes it.
    """
    config = _config(str(tmp_path))

    locations = save_final_artifacts(config, None, _hetero(), _Metadata())

    written = json.loads(
        (Path(derive_metadata_prefix(config.pyg_output_path)) / CHECKSUMS_FILE)
        .read_text()
    )
    assert locations["artifacts"] == written["artifacts"]


# ======================================================================
# The S3 mirror
# ======================================================================

@pytest.fixture
def archived(tmp_path):
    """A build with the S3 mirror on, and the fake client it wrote through."""
    config = _config(
        str(tmp_path),
        archive_to_s3=True,
        s3_archive_bucket="test-bucket",
        s3_pyg_key="pyg/year=2099/month=01/hetero_data.pt",
    )
    s3 = _FakeS3()
    save_final_artifacts(config, s3, _hetero(), _Metadata())
    return config, s3


def test_the_mirror_carries_its_own_checksums(archived):
    """A consumer holding only the S3 prefix has to be able to verify it."""
    _, s3 = archived

    key = "pyg/year=2099/month=01/metadata/checksums.json"
    assert key in s3.objects
    checksums = json.loads(s3.objects[key]["body"])

    assert set(checksums["artifacts"]) == {"hetero_data.pt"} | {
        f"metadata/{name}" for name in METADATA_FILES
    }

    uploaded = s3.uploads[0]["body"]
    recorded = checksums["artifacts"]["hetero_data.pt"]
    assert recorded["sha256"] == hashlib.sha256(uploaded).hexdigest()
    assert recorded["bytes"] == len(uploaded)

    for name in METADATA_FILES:
        body = s3.objects[f"pyg/year=2099/month=01/metadata/{name}"]["body"]
        recorded = checksums["artifacts"][f"metadata/{name}"]
        assert recorded["sha256"] == hashlib.sha256(body).hexdigest()


def test_the_mirror_and_the_work_dir_agree_on_the_pt(archived, tmp_path):
    """Both copies are serialized separately; they must still be the same bytes.

    Two torch.save calls, one to the work dir and one to the upload's staging
    file. Handing torch a file object is what keeps them identical — see
    test_the_same_graph_digests_the_same_from_every_destination.
    """
    config, s3 = archived

    local = Path(config.pyg_output_path)
    assert _sha256(local) == hashlib.sha256(s3.uploads[0]["body"]).hexdigest()


def test_s3_is_asked_to_verify_the_transfer(archived):
    """S3 checks the checksum on receipt and returns it from HeadObject.

    Without it a truncated multipart upload is stored as a valid object, and a
    consumer has to download 42 GB to learn what it is holding.
    """
    _, s3 = archived

    assert s3.uploads[0]["extra"]["ChecksumAlgorithm"] == "SHA256"
    assert all(
        obj.get("ChecksumAlgorithm") == "SHA256" for obj in s3.objects.values()
    ), "a metadata put was sent without a checksum"
