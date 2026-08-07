"""Regression tests for job-manifest writing (spark_jobs.build_graph.save_job_manifest).

The manifest is written to ``config.local_work_dir``, which on a real cluster is a
shared-storage URI (``s3a://bucket/prefix``), not a POSIX path. The original
implementation wrote it with a bare ``open()``, so a URI work dir produced a junk
``./s3a:/...`` directory tree on the driver's local disk and the manifest never
reached shared storage -- while every cluster test still passed (none asserted on the
manifest). These tests pin the routing: a URI must resolve through the Hadoop
FileSystem to its real location, never to a literal ``<scheme>:`` directory.

A ``file://`` URI exercises the exact same non-local code path as ``s3a://`` (any
scheme that isn't a bare local path) but resolves through the local Spark session's
LocalFileSystem, so the assertion runs in the fast CPU suite with no cluster or S3.
"""

import glob
import json
import os
from types import SimpleNamespace


from spark_jobs.build_graph import save_job_manifest


def _config(work_dir: str, mode: str = "enrichment_only") -> SimpleNamespace:
    """Minimal stand-in for JobConfig: save_job_manifest only reads these fields."""
    return SimpleNamespace(
        mode=mode,
        time_period="2099-01",
        source_paths=["s3a://bucket/raw/sec"],
        source_format="turtle_parquet",
        turtle_column="triples",
        local_work_dir=work_dir.rstrip("/"),
        period_partition="year=2099/month=01",
        enriched_parquet_path=(
            f"{work_dir}/enriched/year=2099/month=01/triples"
        ),
        pyg_output_path=(
            f"{work_dir}/pyg/year=2099/month=01/hetero_data.pt"
        ),
        s3_archive_bucket="",
        s3_pyg_key="",
        enable_ontology_mapping=False,
        pyg_config={},
        parquet_partitions=2,
        archive_to_s3=False,
    )


def _find_manifest(base_dir) -> str:
    matches = glob.glob(
        os.path.join(
            str(base_dir), "manifests", "year=2099", "month=01", "*.json"
        )
    )
    assert matches, (
        f"no manifest written under "
        f"{base_dir}/manifests/year=2099/month=01/"
    )
    return matches[0]


def test_manifest_uri_workdir_does_not_create_literal_scheme_dir(spark, tmp_path, monkeypatch):
    """A URI work dir must resolve to its real path, not a ``./<scheme>:`` dir in CWD."""
    # Run from tmp_path so any stray literal directory is easy to detect and can't
    # pollute the repo checkout (the original bug wrote ./s3a:/... into the driver CWD).
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work_uri = f"file://{work}"

    save_job_manifest(spark, None, _config(work_uri), {"mode": "enrichment_only"}, 1.23)

    # The manifest landed at the resolved path...
    manifest = _find_manifest(work)
    payload = json.loads(open(manifest, "rb").read())
    assert payload["mode"] == "enrichment_only"
    assert payload["config"]["local_work_dir"] == work_uri
    # ...and NOT under a literal directory named after the scheme.
    assert not os.path.exists(tmp_path / "file:"), "wrote to a literal ./file: dir"


def test_manifest_plain_local_workdir_still_writes(spark, tmp_path):
    """A bare POSIX work dir keeps writing directly to that path (no regression)."""
    work = tmp_path / "local_work"
    save_job_manifest(spark, None, _config(str(work)), {"mode": "enrichment_only"}, 0.5)

    manifest = _find_manifest(work)
    assert json.loads(open(manifest, "rb").read())["time_period"] == "2099-01"


def test_manifest_records_the_ontology_flag_where_it_applies(spark, tmp_path):
    """An enriching mode's manifest states the flag it actually honored."""
    work = tmp_path / "enriching"
    config = _config(str(work), mode="enrichment_only")
    config.enable_ontology_mapping = True
    save_job_manifest(spark, None, config, {"mode": "enrichment_only"}, 0.5)

    payload = json.loads(open(_find_manifest(work), "rb").read())
    assert payload["config"]["enable_ontology_mapping"] is True


def test_manifest_does_not_claim_an_ontology_flag_pyg_only_never_used(
    spark, tmp_path
):
    """pyg_only must not report a flag for a phase it never reaches.

    That job enriches nothing -- it reads a Parquet some earlier job wrote --
    so its own --enable_ontology_mapping is inert. Written verbatim it reads
    as a claim about that Parquet, and the 2026-07-29 build's empty class
    hierarchy was diagnosed off exactly that misreading. null says "not
    applicable"; ontology_schema.json's ontology_mapping_enabled carries the
    verdict that is actually derived from the triples.
    """
    work = tmp_path / "pyg_only"
    config = _config(str(work), mode="pyg_only")
    config.enable_ontology_mapping = False
    save_job_manifest(spark, None, config, {"mode": "pyg_only"}, 0.5)

    payload = json.loads(open(_find_manifest(work), "rb").read())
    assert payload["config"]["enable_ontology_mapping"] is None, (
        "pyg_only manifest states an enrichment flag it never honored"
    )
