"""Reading a staged mirror must be indistinguishable from reading the bucket.

The parse phase moved off object storage because opening it from 144 tasks at
once took the site network down (issue #342). That is only a safe trade if the
two input modes are interchangeable, which is what these tests pin:

  - the same key layout, so source_label() still recognises every source
  - the same per-source labels, whatever the mirror root is called
  - a scheme-qualified path, so fs.defaultFS cannot send a "local" read back
    over the wire
  - a mirror that is missing or empty fails before the job starts, rather than
    parsing to zero triples several minutes in
"""

import os
from pathlib import Path

import pytest

from spark_jobs.build_graph import JobConfig, source_label, staged_local_path


BASE_ARGS = {
    "mode": "enrichment_only",
    "local_work_dir": "/work",
    "source_format": "turtle_parquet",
}


def config(**overrides) -> JobConfig:
    return JobConfig({**BASE_ARGS, **overrides})


# ---------------------------------------------------------------------------
# path mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["s3a", "s3n", "s3"])
def test_object_storage_uri_maps_under_the_root(scheme):
    assert staged_local_path(
        f"{scheme}://my-bucket/raw/source=sec/feed=filings/", "/srv/mirror"
    ) == "file:///srv/mirror/my-bucket/raw/source=sec/feed=filings/"


def test_bucket_is_kept_so_two_buckets_cannot_collide():
    """Two buckets can share a key prefix; the root must keep them apart."""
    one = staged_local_path("s3a://my-bucket/raw/2026/", "/srv/mirror")
    two = staged_local_path("s3a://some-bucket/raw/2026/", "/srv/mirror")
    assert one != two


def test_already_local_paths_pass_through():
    """A run may mix a staged bucket with a directory that was never in S3."""
    assert staged_local_path("/data/rdf/2024-12/", "/srv/mirror") == (
        "/data/rdf/2024-12/"
    )
    assert staged_local_path("file:///data/rdf/", "/srv/mirror") == (
        "file:///data/rdf/"
    )


def test_result_names_its_scheme():
    """A bare path resolves against fs.defaultFS.

    A site that points fs.defaultFS at object storage would otherwise send a
    read the operator explicitly asked to keep local straight back over the
    uplink -- the exact traffic staging exists to remove.
    """
    assert staged_local_path("s3a://b/k/", "/srv/mirror").startswith("file:///")


def test_relative_root_is_refused():
    with pytest.raises(ValueError, match="absolute path"):
        staged_local_path("s3a://b/k/", "relative/mirror")


# ---------------------------------------------------------------------------
# labels: the reason the mapping keeps the key layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3a://my-data-lake/raw/source=sec/feed=filings/2024-12/", "sec"),
        ("s3a://my-data-lake/raw/source=bls/cpi/2024-12/", "bls"),
        ("s3a://my-data-lake/noaa/daily/2024-12/", "noaa"),
        ("s3a://my-data-lake/vendor/intraday/quotes/day=21/", "market"),
    ],
)
def test_staging_preserves_the_source_label(uri, expected):
    """Same label from the bucket and from the mirror.

    per_source_triple_stats() keys on this, so a mirror that changed it would
    make the two modes' reports incomparable -- and the whole point of staging
    is that the graph does not change.
    """
    assert source_label(uri, 0) == expected
    staged = staged_local_path(uri, "/srv/mirror")
    assert source_label(staged, 0) == expected


def test_label_comes_from_the_declared_path_not_the_root(tmp_path):
    """A root that happens to name a source must not relabel the run.

    read_paths is what gets opened; source_paths is what gets labelled. This is
    the case that separates them: both sources below come back "sec" if the
    label is taken from the resolved path, because the root says so.
    """
    root = tmp_path / "source=sec-mirror"
    for key in ("my-data-lake/raw/source=bls/cpi", "my-data-lake/noaa/daily"):
        staged = root / key
        staged.mkdir(parents=True)
        (staged / "part-0.parquet").write_bytes(b"x")

    cfg = config(
        source_paths="s3a://my-data-lake/raw/source=bls/cpi/,s3a://my-data-lake/noaa/daily/",
        input_mode="local",
        local_source_root=str(root),
    )
    declared = [source_label(p, i) for i, p in enumerate(cfg.source_paths)]
    resolved = [source_label(p, i) for i, p in enumerate(cfg.read_paths)]

    assert declared == ["bls", "noaa"]
    assert resolved == ["sec", "sec"], (
        "the root wins when the resolved path is labelled -- which is why "
        "load_source_triples labels from source_paths"
    )


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------
def test_s3_is_the_default_and_reads_the_uris_unchanged():
    """Nothing changes for a deployment that has not staged anything."""
    cfg = config(source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/")
    assert cfg.input_mode == "s3"
    assert cfg.read_paths == cfg.source_paths


def test_local_mode_requires_a_root():
    with pytest.raises(ValueError, match="local_source_root is required"):
        config(
            source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/",
            input_mode="local",
        )


def test_unknown_input_mode_is_refused():
    with pytest.raises(ValueError, match="Invalid input_mode"):
        config(source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/", input_mode="nfs")


def test_missing_mirror_fails_before_the_job_starts(tmp_path):
    """The cheap half of the check, and the one that catches a forgotten sync.

    The driver can only see its own host; the staging script is what compares
    the other workers. Without this, a root that was never staged surfaces as
    an empty parse well into the run.
    """
    with pytest.raises(FileNotFoundError, match="stage_sources.sh"):
        config(
            source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/",
            input_mode="local",
            local_source_root=str(tmp_path / "never-staged"),
        )


def test_empty_mirror_is_treated_as_missing(tmp_path):
    """`aws s3 sync` of a wrong prefix leaves a directory and no files."""
    mirror = tmp_path / "mirror" / "my-data-lake" / "raw" / "source=sec" / "feed=filings"
    mirror.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        config(
            source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/",
            input_mode="local",
            local_source_root=str(tmp_path / "mirror"),
        )


def test_staged_mirror_passes_validation(tmp_path):
    mirror = tmp_path / "mirror" / "my-data-lake" / "raw" / "source=sec" / "feed=filings"
    mirror.mkdir(parents=True)
    (mirror / "part-0.parquet").write_bytes(b"not really parquet")

    cfg = config(
        source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/",
        input_mode="local",
        local_source_root=str(tmp_path / "mirror"),
    )
    assert cfg.read_paths == [
        f"file://{tmp_path}/mirror/my-data-lake/raw/source=sec/feed=filings/"
    ]


def test_repr_states_which_mode_the_run_used(tmp_path):
    """The driver log is where a run's timings get read back.

    An s3 leg is bounded by the site uplink and a local leg is not, so the two
    are not comparable without knowing which this was.
    """
    mirror = tmp_path / "mirror" / "my-data-lake" / "raw" / "source=sec" / "feed=filings"
    mirror.mkdir(parents=True)
    (mirror / "part-0.parquet").write_bytes(b"x")

    cfg = config(
        source_paths="s3a://my-data-lake/raw/source=sec/feed=filings/",
        input_mode="local",
        local_source_root=str(tmp_path / "mirror"),
    )
    assert "input_mode=local" in repr(cfg)


# ---------------------------------------------------------------------------
# the coupling between the two halves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "uri",
    [
        "s3a://my-bucket/raw/source=sec/feed=filings/2024-12/",
        "s3a://my-bucket/raw/source=bls/cpi/",
        "s3://test-bucket/vendor/intraday/quotes/year=2026/month=08/day=21/",
        "s3a://b/k",
    ],
)
def test_staging_script_and_job_agree_on_where_a_source_lands(uri):
    """bin/stage_sources.sh writes it; build_graph.py reads it.

    Nothing else in either file would notice if the two mappings drifted apart:
    the sync would succeed, the job would start, and the parse would find an
    empty directory. The script exposes its mapping through PRINT_LOCAL_PATH
    so this can be checked without a bucket or a cluster.
    """
    import subprocess

    script = Path(__file__).resolve().parent.parent / "bin" / "stage_sources.sh"
    written = subprocess.run(
        [str(script), "--dest", "/srv/mirror"],
        env={**os.environ, "PRINT_LOCAL_PATH": uri},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    read = staged_local_path(uri, "/srv/mirror")
    assert read.startswith("file://")
    assert read[len("file://"):].rstrip("/") == written.rstrip("/")
