"""The dataset descriptor: what an enriched output says about its own inputs.

The enriched Parquet is subject/predicate/object only -- SOURCE_COLUMN is dropped
at the write -- so a pyg_only run reading it back has no way to know which sources
produced it. Before the descriptor, that fact lived solely in the enrichment
manifest: a different file, under a different prefix, that nothing linked to the
graph except a shared parent directory.
"""
import json

from spark_jobs.build_graph import (
    DATASET_DESCRIPTOR_NAME,
    dataset_descriptor_path,
    load_dataset_descriptor,
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
    }
    (tmp_path / "enriched" / "year=2026" / "month=08"
     / DATASET_DESCRIPTOR_NAME).write_text(json.dumps(body, indent=2))

    assert load_dataset_descriptor(spark, str(enriched)) == body


def test_missing_descriptor_reads_as_empty_rather_than_raising(spark, tmp_path):
    """Absent is the normal case, not a failure.

    Every enriched directory written before this existed has no descriptor, and
    a pyg_only run over one of them must still build a graph -- it just records
    no sources.
    """
    enriched = tmp_path / "enriched" / "year=2026" / "month=08" / "triples"
    enriched.mkdir(parents=True)
    assert load_dataset_descriptor(spark, str(enriched)) == {}
