"""Per-source triple accounting: labels, counts, and the two kinds of duplicate.

WHY THIS EXISTS

The triples frame is (subject, predicate, object) and nothing else, so two
identical rows are indistinguishable and the manifest could never say which
source contributed them. The alternative to recording it was to guess -- charge a
shared fact to whichever source was listed later, or to neither, or to both --
and the first of those is not even stable, since it changes when the paths are
passed in a different order.

Stamping the source at load time replaces the guess with an observation. These
tests pin the two things that makes possible, and the one thing it must NOT do.

THE TWO KINDS OF DUPLICATE ARE NOT THE SAME THING

A source repeating ITSELF is a hygiene signal -- measured at ~8.7% for one real
source, worth raising upstream. Two sources independently stating the same fact
is corroboration, arguably the point of a knowledge graph, and charging it
against a source would report a good outcome as a defect. Conflating them is the
whole defect being fixed, so they are asserted separately here.

THE STAMP MUST NOT ESCAPE

Roughly 25 modules construct or read the three-column shape. If the extra column
survived the loader, every one of them would receive a frame it does not expect
-- and the failure would not be loud, because a union or a select of named
columns simply carries it along. test_stamp_is_dropped_before_return is the guard
for that, and it is the most important test in this file.
"""

from pathlib import Path

import pytest

from spark_jobs.build_graph import (
    SOURCE_COLUMN,
    JobConfig,
    load_source_triples,
    per_source_triple_stats,
    source_label,
)

pytestmark = pytest.mark.usefixtures("spark")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "e2e" / "ntriples"


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "path, expected",
    [
        ("s3a://b/raw/source=sec/feed=filings/year=2026/month=08/21.parquet", "sec"),
        ("s3a://b/raw/source=bls/feed=cpi/2026.parquet", "bls"),
        ("s3a://b/raw/noaa/nws/alerts/year=2026/month=08/21.parquet", "noaa"),
        ("s3a://b/market/intraday/quotes/year=2026/day=21/snap.parquet", "market"),
        ("/local/dir/source=sec/feed=filings/", "sec"),
    ],
)
def test_source_label_recognises_known_sources(path, expected):
    assert source_label(path, 0) == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        # The committed fixtures are plain files with no partition fragments.
        ("tests/fixtures/e2e/ntriples/sec.nt", "sec"),
        ("tests/fixtures/e2e/ntriples/market.nt", "market"),
        ("/abs/path/noaa.nt", "noaa"),
    ],
)
def test_source_label_recognises_a_bare_filename(path, expected):
    assert source_label(path, 0) == expected


@pytest.mark.parametrize(
    "path",
    [
        # Contains a source name as a SUBSTRING but is not that source. A wrong
        # label is worse than a positional one, because it reads as authoritative.
        "s3a://secure-data.example.com/prefix/x.parquet",
        "/data/marketing/reports/x.parquet",
        "/data/bls-archive-mirror/x.parquet",
    ],
)
def test_substring_matches_do_not_produce_a_label(path):
    assert source_label(path, 7) == "source_7"


def test_unrecognised_source_falls_back_positionally_not_to_the_path():
    """The fallback must not echo any part of the path.

    The manifest is copied to object storage and pasted into issues. A label
    derived from an unrecognised path would put whatever that path contains --
    bucket, prefix, host -- into a map key, which is exactly the structure people
    quote casually. config.source_paths already records the path once, in a field
    a reader knows to treat as deployment detail.
    """
    path = "s3a://some-private-bucket.example.com/internal/prefix/x.parquet"
    label = source_label(path, 3)

    assert label == "source_3"
    for fragment in ("private", "bucket", "example", "internal", "prefix", "s3a"):
        assert fragment not in label


def test_labels_are_stable_across_locations():
    """Same data from a different bucket must land under the same key.

    Otherwise counts cannot be compared between runs that read the same source
    from different locations, which is the situation every environment change
    produces.
    """
    # Short placeholder hosts on purpose: a longer, hostname-shaped stand-in
    # trips the repository's own hardcoded-bucket rule, which cannot tell a
    # plausible fake from a real one and should not have to.
    a = "s3a://b1/raw/source=sec/feed=filings/2026/08/21.parquet"
    b = "s3a://b2/archive/raw/source=sec/feed=filings/2026/08/21.parquet"
    assert source_label(a, 0) == source_label(b, 1) == "sec"


# --------------------------------------------------------------------------- #
# counts
# --------------------------------------------------------------------------- #

def _stamped(spark, rows):
    """(subject, predicate, object, _source) frame from tuples."""
    return spark.createDataFrame(
        rows, ["subject", "predicate", "object", SOURCE_COLUMN]
    )


def test_rows_contributed_is_per_source(spark):
    df = _stamped(spark, [
        ("s1", "p", "o", "sec"),
        ("s2", "p", "o", "sec"),
        ("s3", "p", "o", "market"),
    ])
    stats = per_source_triple_stats(df)

    assert stats["sources"]["sec"]["rows_contributed"] == 2
    assert stats["sources"]["market"]["rows_contributed"] == 1


def test_a_source_repeating_itself_is_charged_to_that_source(spark):
    """The hygiene signal. sec states one fact twice; market states it once."""
    df = _stamped(spark, [
        ("tsco", "isNamed", "Tractor Supply", "sec"),
        ("tsco", "isNamed", "Tractor Supply", "sec"),
        ("cik", "is", "916365", "sec"),
        ("tsco", "closed", "52.10", "market"),
    ])
    stats = per_source_triple_stats(df)

    assert stats["sources"]["sec"]["duplicates_within_source"] == 1
    assert stats["sources"]["market"]["duplicates_within_source"] == 0


def test_two_sources_agreeing_is_not_charged_to_either(spark):
    """Corroboration is not a defect, and must not be reported as one."""
    df = _stamped(spark, [
        ("tsco", "isNamed", "Tractor Supply", "sec"),
        ("tsco", "isNamed", "Tractor Supply", "market"),
    ])
    stats = per_source_triple_stats(df)

    assert stats["sources"]["sec"]["duplicates_within_source"] == 0
    assert stats["sources"]["market"]["duplicates_within_source"] == 0
    assert stats["facts_stated_by_multiple_sources"] == 1


def test_self_repetition_does_not_inflate_the_agreement_count(spark):
    """One source stating a fact ten times is still ONE source stating it.

    The implementation takes distinct (triple, source) pairs before counting
    distinct sources. Without that step, repetition inside a single source would
    look like several sources agreeing -- turning a hygiene problem into a false
    corroboration signal, which is the worst of both readings.
    """
    df = _stamped(spark, [
        ("a", "p", "o", "sec"),
        ("a", "p", "o", "sec"),
        ("a", "p", "o", "sec"),
    ])
    stats = per_source_triple_stats(df)

    assert stats["facts_stated_by_multiple_sources"] == 0
    assert stats["sources"]["sec"]["duplicates_within_source"] == 2


def test_both_kinds_of_duplicate_at_once(spark):
    """The case the old conventions could not express.

    sec repeats itself once AND agrees with market on the same fact. Under
    "blame whoever came second" the answer would change with argument order;
    here both facts are reported and neither depends on ordering.
    """
    df = _stamped(spark, [
        ("a", "p", "o", "sec"),
        ("a", "p", "o", "sec"),
        ("a", "p", "o", "market"),
        ("b", "p", "o", "market"),
    ])
    stats = per_source_triple_stats(df)

    assert stats["sources"]["sec"]["rows_contributed"] == 2
    assert stats["sources"]["sec"]["duplicates_within_source"] == 1
    assert stats["sources"]["market"]["rows_contributed"] == 2
    assert stats["sources"]["market"]["duplicates_within_source"] == 0
    assert stats["facts_stated_by_multiple_sources"] == 1


def test_counts_do_not_depend_on_source_order(spark):
    """The property the discarded convention could not provide."""
    rows = [
        ("a", "p", "o", "sec"),
        ("a", "p", "o", "market"),
        ("b", "p", "o", "sec"),
        ("b", "p", "o", "sec"),
    ]
    forward = per_source_triple_stats(_stamped(spark, rows))
    reversed_ = per_source_triple_stats(_stamped(spark, list(reversed(rows))))

    assert forward == reversed_


# --------------------------------------------------------------------------- #
# the stamp must not escape the loader
# --------------------------------------------------------------------------- #

def test_stamp_is_dropped_before_return(spark, tmp_path):
    """The loader must hand back three columns, exactly as it always did.

    The source column exists only between the parse and the statistics. If it
    survived, every stage after this one would receive a frame one column wider
    than it expects. That would not fail loudly: a union of named columns or a
    select carries an unexpected column along quite happily, so the first sign
    would be a wrong result somewhere much later.
    """
    config = JobConfig({
        "mode": "enrichment_only",
        "source_paths": f"{FIXTURES / 'sec.nt'},{FIXTURES / 'market.nt'}",
        "source_format": "ntriples",
        "local_work_dir": str(tmp_path),
        "time_period": "2026-08",
    })

    triples_df, count, stats = load_source_triples(spark, config)

    assert triples_df.columns == ["subject", "predicate", "object"], (
        f"loader returned {triples_df.columns}; the source stamp escaped and "
        "every downstream stage now receives a frame it does not expect"
    )
    assert count > 0
    assert set(stats["sources"]) == {"sec", "market"}
    assert sum(s["rows_contributed"] for s in stats["sources"].values()) == count
