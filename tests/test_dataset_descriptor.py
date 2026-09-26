"""The dataset descriptor: what an enriched output says about its own inputs.

The enriched Parquet is subject/predicate/object only -- SOURCE_COLUMN is dropped
at the write -- so a pyg_only run reading it back has no way to know which sources
produced it. Before the descriptor, that fact lived solely in the enrichment
manifest: a different file, under a different prefix, that nothing linked to the
graph except a shared parent directory.
"""
import json

from spark_jobs.graph.config import JobConfig
from spark_jobs.graph.persistence import (
    DATASET_DESCRIPTOR_NAME,
    dataset_descriptor_path,
    load_dataset_descriptor,
    save_dataset_descriptor,
)


def test_descriptor_sits_beside_the_parquet_dir_not_inside_it():
    """Inside the Parquet directory it would be read as Parquet.

    A leading underscore would make Spark's reader skip it, but Hadoop's hidden
    file filter then hides it from an explicit read too -- writable, never
    readable. A sibling avoids the whole question.
    """
    path = dataset_descriptor_path(
        "/work/run/enriched/year=2026/month=08/triples"
    )
    assert path == f"/work/run/enriched/year=2026/month=08/{DATASET_DESCRIPTOR_NAME}"
    assert "/triples/" not in path


def test_descriptor_path_ignores_a_trailing_slash():
    with_slash = dataset_descriptor_path("/work/enriched/2026/triples/")
    without = dataset_descriptor_path("/work/enriched/2026/triples")
    assert with_slash == without


def test_descriptor_path_handles_a_uri_scheme():
    """Work dirs are POSIX paths on one deployment and object-store URIs on
    another; the descriptor has to land beside the data either way."""
    path = dataset_descriptor_path("s3a://bucket/prefix/enriched/2026/triples")
    assert path == f"s3a://bucket/prefix/enriched/2026/{DATASET_DESCRIPTOR_NAME}"


def test_round_trip(spark, tmp_path):
    enriched = tmp_path / "enriched" / "year=2026" / "month=08" / "triples"
    enriched.mkdir(parents=True)
    body = {
        "dataset": "all-sources",
        "sources": ["bls", "market", "noaa", "sec"],
        "time_period": "2026-08",
        "day": "2026-08-14",
    }
    (tmp_path / "enriched" / "year=2026" / "month=08"
     / DATASET_DESCRIPTOR_NAME).write_text(json.dumps(body, indent=2))

    assert load_dataset_descriptor(spark, str(enriched)) == body


def test_a_descriptor_written_before_the_day_reads_it_as_empty(spark, tmp_path):
    """Every descriptor written before 1.5 has no day key.

    It reads as "", the value a run with no day writes, so a pyg_only leg over
    it stamps "" rather than failing or guessing. The rest reads as it did.
    """
    enriched = tmp_path / "enriched" / "year=2026" / "month=08" / "triples"
    enriched.mkdir(parents=True)
    body = {
        "dataset": "all-sources",
        "sources": ["bls", "market", "noaa", "sec"],
        "time_period": "2026-08",
    }
    (enriched.parent / DATASET_DESCRIPTOR_NAME).write_text(json.dumps(body))

    assert load_dataset_descriptor(spark, str(enriched)) == {**body, "day": ""}


def test_the_writer_records_the_day_the_sources_were_cut_from(spark, tmp_path):
    """The enriching leg writes its day beside the Parquet it cut from that day,
    which is how a pyg_only leg reading that Parquet learns it."""
    config = JobConfig({
        "mode": "enrichment_only",
        "source_paths": (
            "s3a://b/raw/source=sec/feed=filings/year=2026/month=09/"
            "24.snappy.parquet"
        ),
        "source_format": "turtle_parquet",
        "local_work_dir": str(tmp_path),
        "time_period": "2026-09",
    })

    save_dataset_descriptor(config, spark)

    descriptor = load_dataset_descriptor(spark, config.enriched_parquet_path)
    assert descriptor["day"] == "2026-09-24"
    assert descriptor["sources"] == ["sec"]


def test_missing_descriptor_reads_as_empty_rather_than_raising(spark, tmp_path):
    """Absent is the normal case, not a failure.

    Every enriched directory written before this existed has no descriptor, and
    a pyg_only run over one of them must still build a graph -- it just records
    no sources.
    """
    enriched = tmp_path / "enriched" / "year=2026" / "month=08" / "triples"
    enriched.mkdir(parents=True)
    assert load_dataset_descriptor(spark, str(enriched)) == {}
