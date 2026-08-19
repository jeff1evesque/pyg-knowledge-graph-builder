"""Unit tests for the SEC source-feed restriction (build_graph).

The archive holds eight SEC feeds and this pipeline reads one. That was true
before these tests and stated nowhere: the restriction lived in whichever
prefix the caller happened to type. A run pointed one level up does fail, but
it fails inside the loader with "No Turtle column found" — a message about a
column, from a job that was actually misconfigured about a feed.

Paths below are written as plain mounts rather than object-store URIs. The
check keys on the ``source=``/``feed=`` partition names and never on the
scheme, so the scheme is not part of what is under test.

Pure Python: runs under ``pytest -m "not e2e"`` with no Spark fixture.
"""
import pytest

from spark_jobs.build_graph import (
    SEC_HANDLED_FEED,
    SEC_UNHANDLED_FEEDS,
    JobConfig,
    assert_sec_paths_name_the_handled_feed,
)

_HANDLED = f"/mnt/archive/raw/source=sec/{SEC_HANDLED_FEED}/year=2026/month=08/"


def _config(**overrides):
    args = {
        "mode": "enrichment_only",
        "source_paths": _HANDLED,
        "source_format": "turtle_parquet",
        "local_work_dir": "/work",
        "time_period": "2026-08",
    }
    args.update(overrides)
    return JobConfig(args)


def test_handled_feed_is_accepted():
    assert _config().source_paths == [_HANDLED]


def test_sec_source_root_without_a_feed_is_rejected():
    """The case the restriction exists for: one level up from the feed."""
    with pytest.raises(ValueError, match="does not name a feed"):
        _config(source_paths="/mnt/archive/raw/source=sec/")


@pytest.mark.parametrize("feed", SEC_UNHANDLED_FEEDS)
def test_each_unhandled_sec_feed_is_rejected(feed):
    with pytest.raises(ValueError, match="unhandled SEC feed"):
        _config(source_paths=f"/mnt/archive/raw/source=sec/{feed}/")


def test_filings_documents_is_not_mistaken_for_filings():
    """``feed=filings_documents`` contains ``feed=filings`` as a substring.

    It is 712,351 objects of raw filing documents with no RDF in them, so a
    plain substring test would wave through the largest unreadable feed in the
    archive. The unhandled list is checked first for exactly this reason.
    """
    with pytest.raises(ValueError, match="filings_documents"):
        assert_sec_paths_name_the_handled_feed(
            ["/mnt/archive/raw/source=sec/feed=filings_documents/"]
        )


def test_one_bad_path_among_several_is_rejected():
    """--source_paths takes a list, and every entry has to be readable."""
    with pytest.raises(ValueError, match="litigation"):
        _config(source_paths=(
            f"{_HANDLED},"
            "/mnt/archive/raw/source=bls/feed=cpi/,"
            "/mnt/archive/raw/source=sec/feed=litigation/"
        ))


@pytest.mark.parametrize("path", [
    "/data/sec/filings/",              # local directory, not the archive layout
    "/mnt/archive/raw/source=bls/",   # another source entirely
    "/mnt/archive/raw/noaa/",         # NOAA does not use source= at all
    "/work/enriched/year=2026/month=08/triples",
])
def test_non_archive_paths_are_left_alone(path):
    """Keyed on the ``source=sec`` partition, not on the letters "sec".

    A fixture directory named /data/sec/ is not the archive convention, and a
    validator that guessed at it would reject working local runs.
    """
    assert_sec_paths_name_the_handled_feed([path])


def test_pyg_only_mode_does_not_read_sources():
    """pyg_only reads the enriched Parquet, so it takes no source path."""
    config = JobConfig({
        "mode": "pyg_only",
        "local_work_dir": "/work",
        "time_period": "2026-08",
    })
    assert config.source_paths == []
