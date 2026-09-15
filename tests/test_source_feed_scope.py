"""Unit tests for matching source paths to sources, and for the SEC feed
restriction (JobConfig).

Every source path belongs to exactly one registered source. JobConfig matches
the paths before Spark starts, rejects a path that matches none or two, and
hands each picked source the paths matched to it for its own check.

The archive holds eight SEC feeds and this pipeline reads one. That was true
before these tests and stated nowhere: the restriction lived in whichever
prefix the caller happened to type. A run pointed one level up does fail, but
it fails inside the loader with "No Turtle column found" — a message about a
column, from a job that was actually misconfigured about a feed.

Paths below are written as plain mounts rather than object-store URIs. The
checks key on path fragments such as ``source=`` and ``feed=``, never on the
scheme, so the scheme is not part of what is under test.

Pure Python: runs under ``pytest -m "not e2e"`` with no Spark fixture.
"""
import pytest

from spark_jobs import sources
from spark_jobs.graph.config import JobConfig
from spark_jobs.sources.sec import (
    SEC_HANDLED_FEED,
    SEC_UNHANDLED_FEEDS,
    assert_sec_paths_name_the_handled_feed,
)
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import ONTOLOGY_BASE

_HANDLED = f"/mnt/archive/raw/source=sec/{SEC_HANDLED_FEED}/year=2026/month=08/"
_BLS = "/mnt/archive/raw/source=bls/feed=cpi/year=2026/month=08/"
_NOAA = "/mnt/archive/raw/noaa/year=2026/month=08/"

_ARGS = {
    "mode": "enrichment_only",
    "source_paths": _HANDLED,
    "source_format": "turtle_parquet",
    "local_work_dir": "/work",
    "time_period": "2026-08",
}


def _config(**overrides):
    return JobConfig({**_ARGS, **overrides})


def test_handled_feed_is_accepted():
    assert _config().source_paths == [_HANDLED]


# --------------------------------------------------------------------------- #
# which source each path belongs to
# --------------------------------------------------------------------------- #

def test_a_path_that_names_no_source_is_rejected():
    with pytest.raises(ValueError, match="matches no registered source"):
        _config(source_paths="/mnt/archive/raw/year=2026/month=08/")


def test_a_path_that_names_two_sources_is_rejected():
    with pytest.raises(ValueError, match="more than one source"):
        _config(source_paths="/mnt/archive/raw/source=bls/noaa/")


def test_a_run_without_market_picks_the_other_three():
    config = _config(source_paths=f"{_NOAA},{_HANDLED},{_BLS}")

    assert [spec.name for spec in config.path_specs] == ["noaa", "sec", "bls"]
    assert [spec.name for spec in config.source_specs] == ["bls", "sec", "noaa"]


def test_each_source_checks_only_the_paths_matched_to_it():
    namespace = f"{ONTOLOGY_BASE}toy/"
    seen = []
    toy = SourceSpec(
        name="toy",
        path_fragments=("source=toy",),
        namespaces=((namespace, "toy"),),
        enrichment_namespace=namespace,
        check_paths=seen.extend,
    )
    toy_path = "/mnt/archive/raw/source=toy/year=2026/month=08/"

    JobConfig(
        {**_ARGS, "source_paths": f"{_BLS},{toy_path}"},
        specs=(*sources.REGISTERED, toy),
    )

    assert seen == [toy_path]


# --------------------------------------------------------------------------- #
# the SEC feed
# --------------------------------------------------------------------------- #

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
    """pyg_only reads the enriched Parquet, so it takes no source path and
    picks no source."""
    config = JobConfig({
        "mode": "pyg_only",
        "local_work_dir": "/work",
        "time_period": "2026-08",
    })
    assert config.source_paths == []
    assert config.source_specs == ()
