"""Unit tests for the one-minute quote reader (spark_jobs.graph.minute_quotes).

The minute feed is not a source. A run lists the day's folder on the driver,
reads each minute's file, and hands the rows to the minute_quotes/ table. What
these pin: a day with no files is None, so it gets no partition; the table's
columns, with the feed's epoch milliseconds turned into UTC times; and one row
per stock per minute, sorted by ticker, then minute.

Tier 1: pure, no Spark. The S3 client is a stub (tests/minute_quote_feed.py).
"""
import logging
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from minute_quote_feed import FEED_SCHEMA, folder_client, minute_file, quote
from spark_jobs.graph import minute_quotes

DAY = "2026-10-28"
FOLDER = "year=2026/month=10/day=28/"


def _read(files, prefix="", page_size=2):
    """Read DAY from a folder holding ``files``, {file name: body}."""
    folder = minute_quotes.day_folder(prefix, DAY)
    client = folder_client(
        {f"{folder}{name}": body for name, body in files.items()}, page_size
    )
    return minute_quotes.read_minute_quotes("b", prefix, DAY, client)


def _table(rows):
    """Rows as {column: value} dicts, to compare a column at a time."""
    return [dict(zip(minute_quotes.COLUMNS, row)) for row in rows]


def _feed(column):
    """The feed's name for one of the table's columns."""
    return minute_quotes.FEED_NAMES.get(column, column)


def _schema_without(column):
    """The feed's schema less one of the table's columns, as if upstream had
    renamed it."""
    return FEED_SCHEMA.remove(FEED_SCHEMA.get_field_index(_feed(column)))


def _warnings(caplog):
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]


def test_a_days_files_are_read_into_rows():
    rows = _read({
        "13-31.parquet": minute_file([quote("AAPL"), quote("MSFT")]),
        "13-32.parquet": minute_file([
            quote("AAPL", "2026-10-28T13:32:00Z"),
            quote("MSFT", "2026-10-28T13:32:00Z"),
        ]),
    })
    assert [(row["ticker"], row["capture_time"]) for row in _table(rows)] == [
        ("AAPL", "2026-10-28T13:31:00.000Z"),
        ("AAPL", "2026-10-28T13:32:00.000Z"),
        ("MSFT", "2026-10-28T13:31:00.000Z"),
        ("MSFT", "2026-10-28T13:32:00.000Z"),
    ]


def test_the_rows_are_sorted_by_ticker_then_minute():
    """Whatever order the files and their rows come in. A ticker read prunes
    row groups only when each ticker's minutes sit together."""
    rows = _read({
        "a.parquet": minute_file([
            quote("MSFT", "2026-10-28T13:32:00Z"), quote("AAPL", "2026-10-28T13:32:00Z"),
        ]),
        "b.parquet": minute_file([
            quote("MSFT", "2026-10-28T13:31:00Z"), quote("AAPL", "2026-10-28T13:31:00Z"),
        ]),
    })
    pairs = [(row["ticker"], row["capture_time"]) for row in _table(rows)]
    assert pairs == sorted(pairs)
    assert [ticker for ticker, _ in pairs] == ["AAPL", "AAPL", "MSFT", "MSFT"]


def test_a_row_carries_the_tables_columns_and_not_the_raw_json():
    (row,) = _read({"13-31.parquet": minute_file([quote()])})
    assert _table([row]) == [{
        "ticker": "AAPL",
        "capture_time": "2026-10-28T13:31:00.000Z",
        "quote_time": "2026-10-28T13:30:59.123Z",
        "trade_time": "2026-10-28T13:30:58.004Z",
        "bid": 262.5,
        "ask": 262.55,
        "bid_size": 300,
        "ask_size": 200,
        "last": 262.52,
        "mark": 262.525,
        "open": 260.1,
        "high": 263.0,
        "low": 259.8,
        "prev_close": 259.58,
        "volume": 1234567,
        "net_change": 2.94,
        "net_percent_change": 1.1326,
    }]
    assert len(row) == len(minute_quotes.COLUMNS)


def test_the_feeds_epoch_milliseconds_become_utc_times():
    """The same form snapshots/ gives its capture times, so the three time
    columns compare as text."""
    assert minute_quotes.epoch_ms_time(0) == "1970-01-01T00:00:00.000Z"
    assert minute_quotes.epoch_ms_time(1793194259123) == "2026-10-28T13:30:59.123Z"
    assert minute_quotes.epoch_ms_time(None) is None


def test_a_quote_with_no_trade_time_keeps_its_row():
    (row,) = _table(_read({"13-31.parquet": minute_file([quote(trade_time=None)])}))
    assert row["trade_time"] is None
    assert row["quote_time"] == "2026-10-28T13:30:59.123Z"


@pytest.mark.parametrize("written, read", [
    ("2026-10-28T13:31:00Z", "2026-10-28T13:31:00.000Z"),
    ("2026-10-28T13:31:00+00:00", "2026-10-28T13:31:00.000Z"),
    ("2026-10-28T09:31:00-04:00", "2026-10-28T13:31:00.000Z"),
    ("2026-10-28T13:31:00", "2026-10-28T13:31:00.000Z"),
    ("not a time", None),
])
def test_a_capture_time_reads_as_one_utc_form(written, read):
    assert minute_quotes.iso_time(written) == read


def test_a_capture_time_written_as_a_timestamp_reads_the_same():
    schema = FEED_SCHEMA.set(
        FEED_SCHEMA.get_field_index(_feed("capture_time")),
        pa.field(_feed("capture_time"), pa.timestamp("ms", tz="UTC")),
    )
    body = minute_file(
        [quote(minute=datetime(2026, 10, 28, 13, 31, tzinfo=timezone.utc))], schema
    )
    (row,) = _table(_read({"13-31.parquet": body}))
    assert row["capture_time"] == "2026-10-28T13:31:00.000Z"


def test_tickers_are_trimmed_and_upper_cased():
    """snapshots/ spells a symbol in upper case, and the two join on it."""
    rows = _read({"13-31.parquet": minute_file([quote(" brk/b ")])})
    assert _table(rows)[0]["ticker"] == "BRK/B"


def test_a_stock_quoted_twice_in_a_minute_appears_once(caplog):
    """The table is one row per stock per minute. The first file in key order
    wins, and the repeat is logged."""
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = _read({
            "13-31-a.parquet": minute_file([quote(bid=1.0)]),
            "13-31-b.parquet": minute_file([quote(bid=2.0)]),
        })
    assert [row["bid"] for row in _table(rows)] == [1.0]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith("1 minute quote rows")


def test_a_row_with_no_ticker_or_no_minute_is_skipped(caplog):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = _read({"13-31.parquet": minute_file([
            quote(ticker=None), quote(ticker=" "), quote(minute=None), quote(),
        ])})
    assert len(rows) == 1
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith("3 minute quote rows")


def test_a_day_with_no_files_returns_none_and_says_why(caplog):
    """Only the day's folder is listed. A file the day before is not the day."""
    client = folder_client({
        "year=2026/month=10/day=27/13-31.parquet": minute_file([quote()]),
    })
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert minute_quotes.read_minute_quotes("b", "", DAY, client) is None

    assert client.listed == [FOLDER]
    assert client.requested == []
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert f"s3://b/{FOLDER}" in warnings[0]


def test_only_the_parquet_files_directly_in_the_folder_are_read():
    rows = _read({
        "13-31.parquet": minute_file([quote()]),
        "_SUCCESS": b"",
        "13-31.json": b"{}",
        "retries/13-31.parquet": minute_file([quote("MSFT")]),
    })
    assert [row["ticker"] for row in _table(rows)] == ["AAPL"]


def test_the_listing_follows_every_page():
    files = {
        f"13-3{minute}.parquet": minute_file([quote(minute=f"2026-10-28T13:3{minute}:00Z")])
        for minute in range(1, 6)
    }
    assert len(_read(files, page_size=2)) == 5


@pytest.mark.parametrize("body", [
    b"not parquet",
    minute_file([quote()], _schema_without("volume")),
    minute_file([quote(bid_size="lots")], FEED_SCHEMA.set(
        FEED_SCHEMA.get_field_index(_feed("bid_size")),
        pa.field(_feed("bid_size"), pa.string()),
    )),
], ids=["not parquet", "a column missing", "a column of text"])
def test_a_file_that_cannot_be_read_is_left_out_and_the_rest_kept(caplog, body):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = _read({
            "13-31.parquet": minute_file([quote()]),
            "13-32.parquet": body,
        })
    assert len(rows) == 1
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "Could not read 1 of the 2 minute quote files" in warnings[0]
    assert f"{FOLDER}13-32.parquet" in warnings[0]


def test_a_day_with_no_file_that_can_be_read_returns_none(caplog):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert _read({"13-31.parquet": b"not parquet"}) is None
    assert f"No minute_quotes table for {DAY}" in _warnings(caplog)


@pytest.mark.parametrize("error", ["dropped connection", "access denied"])
def test_a_folder_that_cannot_be_listed_is_a_warning_not_a_failed_run(caplog, error):
    """Either costs the table and never the run."""
    from botocore.exceptions import ClientError, EndpointConnectionError

    raised = {
        "dropped connection": EndpointConnectionError(
            endpoint_url="https://s3.example.com"
        ),
        "access denied": ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "ListObjectsV2",
        ),
    }[error]

    class _Unlistable:
        @staticmethod
        def list_objects_v2(**request):
            raise raised

    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert minute_quotes.read_minute_quotes("b", "", DAY, _Unlistable) is None
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith(f"Could not list minute quotes under s3://b/{FOLDER}")


def test_no_bucket_reads_nothing():
    client = folder_client({f"{FOLDER}13-31.parquet": minute_file([quote()])})
    assert minute_quotes.read_minute_quotes("", "", DAY, client) is None
    assert client.listed == []


def test_the_days_folder_is_named_from_the_prefix_and_the_day():
    assert minute_quotes.day_folder("", DAY) == FOLDER
    assert minute_quotes.day_folder("feed/", DAY) == f"feed/{FOLDER}"
    assert minute_quotes.day_folder("/a/b/", "2026-01-05") == (
        "a/b/year=2026/month=01/day=05/"
    )


def test_the_loaded_line_counts_the_rows_stocks_minutes_and_files(caplog):
    with caplog.at_level(logging.INFO, logger="build_graph"):
        _read({
            "13-31.parquet": minute_file([quote("AAPL"), quote("MSFT")]),
            "13-32.parquet": minute_file([quote("AAPL", "2026-10-28T13:32:00Z")]),
        }, prefix="feed")
    loaded = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("Loaded ")
    ]
    assert loaded == [
        f"Loaded 3 minute quotes for {DAY}, 2 stocks over 2 minutes, from 2 "
        f"files under s3://b/feed/{FOLDER}"
    ]
