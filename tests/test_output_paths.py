"""Unit tests for artifact output-path construction.

Every artifact path derives from one ``time_period`` label. These pin the
Hive-style ``year=YYYY/month=MM`` layout: Spark's partition discovery reads
``key=value`` directory names into real columns, so a read spanning several
periods gets ``year``/``month`` to filter and prune on. An opaque ``2024-12``
directory gives it nothing.

Pure Python: runs under ``pytest -m "not e2e"`` with no Spark fixture.
"""
import pytest

from spark_jobs.build_graph import JobConfig, period_partition


# ======================================================================
# period_partition — the YYYY-MM -> year=/month= conversion
# ======================================================================

@pytest.mark.parametrize("period,expected", [
    ("2024-12", "year=2024/month=12"),
    ("2099-01", "year=2099/month=01"),
    ("2000-09", "year=2000/month=09"),
])
def test_period_partition_renders_hive_directories(period, expected):
    assert period_partition(period) == expected


def test_period_partition_keeps_zero_padding():
    """month=01, not month=1 — zero-padding keeps lexical and numeric order
    identical, so a plain path sort is chronological."""
    assert period_partition("2024-01").endswith("month=01")


@pytest.mark.parametrize("period", [
    "latest",          # free-form label
    "2024",            # year only
    "2024-12-25",      # a date, not a month
    "24-12",           # two-digit year
    "",                # unset
])
def test_period_partition_passes_through_non_monthly_labels(period):
    """``time_period`` is free-form — nothing validates its shape and
    ``--time_period`` accepts any string. A label that is not YYYY-MM is
    written as a single segment rather than forced into a partition shape
    that would misdescribe what the directory holds.
    """
    assert period_partition(period) == period


# ======================================================================
# JobConfig — every derived path carries the partition
# ======================================================================

def _config(**overrides):
    args = {
        "mode": "enrichment_only",
        "source_paths": "/data/raw/sec",
        "source_format": "turtle_parquet",
        "local_work_dir": "/work",
        "time_period": "2024-12",
    }
    args.update(overrides)
    return JobConfig(args)


def test_enriched_and_pyg_paths_are_partitioned():
    c = _config()
    assert c.enriched_parquet_path == (
        "/work/enriched/year=2024/month=12/triples"
    )
    assert c.pyg_output_path == (
        "/work/pyg/year=2024/month=12/hetero_data.pt"
    )


def test_s3_pyg_key_default_is_partitioned():
    c = _config(s3_archive_bucket="my-archive")
    assert c.s3_pyg_key == "pyg/year=2024/month=12/hetero_data.pt"


def test_explicit_s3_pyg_key_is_not_overridden():
    """The default only fills in when the caller supplied nothing."""
    c = _config(s3_archive_bucket="my-archive", s3_pyg_key="custom/key.pt")
    assert c.s3_pyg_key == "custom/key.pt"


def test_period_partition_is_exposed_on_the_config():
    """save_job_manifest builds the manifest key from this field, so it has to
    be part of the config surface rather than recomputed at each call site."""
    assert _config().period_partition == "year=2024/month=12"


def test_paths_share_one_partition_rendering():
    """All artifacts for a period must land under the same partition path —
    if these ever diverge, a period's outputs scatter across two directories.
    """
    c = _config()
    part = c.period_partition
    assert part in c.enriched_parquet_path
    assert part in c.pyg_output_path


def test_non_monthly_period_still_yields_usable_paths():
    c = _config(time_period="latest")
    assert c.enriched_parquet_path == "/work/enriched/latest/triples"
    assert c.pyg_output_path == "/work/pyg/latest/hetero_data.pt"
