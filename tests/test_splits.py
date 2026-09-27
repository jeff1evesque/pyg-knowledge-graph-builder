"""Unit tests for the stock split reader (spark_jobs.graph.splits).

The split feed is not a source. A run reads it on the driver, keeps the rows
dated the day it is building, and hands them to the splits/ table. What these
pin most is the difference between a day with no splits ([]) and a day the
feed was not read for (None), because a reader of the table can only tell the
two apart by it.

Tier 1: pure, no Spark. The S3 client is a stub (tests/split_feed.py).
"""
import logging
from datetime import date, datetime, timezone

import pytest

from spark_jobs.graph import splits
from split_feed import feed_client, month_object

DAY = "2026-09-02"
KEY = "year=2026/09.snappy.parquet"
# 00:02 Eastern on the day, when the feed writes: 04:02 UTC in September.
FRESH = datetime(2026, 9, 2, 4, 2, tzinfo=timezone.utc)
# 23:59 Eastern the day before. After midnight UTC, so a check made in UTC
# would call it fresh.
STALE = datetime(2026, 9, 2, 3, 59, tzinfo=timezone.utc)

# The month so far, as the feed holds it: the day before, the day, and the day
# after, which the object already carries by the time the nightly run reads it.
MONTH = [
    ("amdd", "1:10", "09/01/2026"),
    ("cycn", "1:7", "09/02/2026"),
    ("aph", "2:1", "09/02/2026"),
    ("ucar", "1:20", "09/03/2026"),
]


def _read(rows=MONTH, written=FRESH):
    client = feed_client({KEY: (month_object(rows), written)})
    return splits.read_splits("b", "", DAY, client)


def _warnings(caplog):
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]


def test_only_the_days_rows_are_kept():
    assert [row["ticker"] for row in _read()] == ["APH", "CYCN"]


def test_a_row_carries_the_five_columns_of_the_table():
    row = _read()[0]
    assert row == {
        "ticker": "APH",
        "ratio": "2:1",
        "shares_after": 2.0,
        "shares_before": 1.0,
        "split_date": date(2026, 9, 2),
    }


def test_tickers_are_upper_cased():
    """The feed writes them in lower case. The quotes' symbols are upper case,
    and a ticker is what a split joins to them by."""
    assert _read([("aph", "2:1", "09/02/2026")])[0]["ticker"] == "APH"


@pytest.mark.parametrize("ratio, after, before", [
    ("2:1", 2.0, 1.0),
    ("4.00:1.00", 4.0, 1.0),
    ("1:20", 1.0, 20.0),
    (":", None, None),
    ("2.19050002", None, None),
    ("0:1", None, None),
])
def test_a_ratio_parses_to_shares_after_and_before(ratio, after, before):
    assert splits.parse_ratio(ratio) == (after, before)


def test_a_ratio_that_does_not_parse_keeps_its_text():
    row = _read([("xyz.ks", ":", "09/02/2026")])[0]
    assert row["ratio"] == ":"
    assert row["shares_after"] is None
    assert row["shares_before"] is None


def test_the_feeds_date_format_parses():
    assert splits.parse_split_date("09/02/2026") == date(2026, 9, 2)
    assert splits.parse_split_date("2026-09-02") is None


def test_an_exact_duplicate_appears_once():
    rows = [("nby", "1:35", "09/02/2026"), ("nby", "1:35", "09/02/2026")]
    assert len(_read(rows)) == 1


def test_a_day_the_feed_listed_nothing_is_empty_not_missing():
    """The object is fresh and holds no row dated the day: the feed ran and
    listed no splits. That is [] and gets an empty partition, not None."""
    assert _read([("amdd", "1:10", "09/01/2026")]) == []


def test_the_month_object_is_named_after_the_days_month():
    assert splits.splits_key("", DAY) == KEY
    assert splits.splits_key("feed/", DAY) == f"feed/{KEY}"
    assert splits.splits_key("", "2026-10-01") == "year=2026/10.snappy.parquet"


def test_a_missing_object_is_a_warning_naming_it(caplog):
    client = feed_client({})
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert splits.read_splits("b", "", DAY, client) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert f"s3://b/{KEY}" in warnings[0]


def test_an_object_last_written_before_the_day_began_is_not_read(caplog):
    """The feed rewrites the month's object just after midnight Eastern on each
    weekday it runs. An object from before the day began was not written by
    that day's run, and read, it would report the day as having no splits."""
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert _read(written=STALE) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "2026-09-01 23:59 EDT" in warnings[0]
    assert f"before {DAY} began" in warnings[0]


def test_an_object_written_after_midnight_eastern_on_the_day_is_read():
    assert _read(written=FRESH)


def test_a_dropped_connection_is_a_warning_not_a_failed_run(caplog):
    """Not a ClientError, so it needs its own catch. Uncaught, it would stop
    the tables phase over a table nothing else depends on."""
    from botocore.exceptions import EndpointConnectionError

    class _Unreachable:
        @staticmethod
        def get_object(Bucket, Key):
            raise EndpointConnectionError(endpoint_url="https://s3.example.com")

    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert splits.read_splits("b", "", DAY, _Unreachable) is None
    assert len(_warnings(caplog)) == 1


def test_an_object_that_is_not_the_feeds_parquet_is_a_warning(caplog):
    client = feed_client({KEY: (b"not parquet", FRESH)})
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert splits.read_splits("b", "", DAY, client) is None
    assert len(_warnings(caplog)) == 1


def test_no_bucket_reads_nothing():
    client = feed_client({KEY: (month_object(MONTH), FRESH)})
    assert splits.read_splits("", "", DAY, client) is None
    assert client.requested == []


def test_the_loaded_line_names_the_object_and_the_count(caplog):
    with caplog.at_level(logging.INFO, logger="build_graph"):
        _read()
    loaded = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("Loaded ")
    ]
    assert loaded == [f"Loaded 2 stock splits for {DAY} from s3://b/{KEY}"]
