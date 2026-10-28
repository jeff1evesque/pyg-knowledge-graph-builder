"""Unit tests for the one-minute quote reader (spark_jobs.graph.minute_quotes).

The minute feed is not a source. A run lists the day's folder on the driver,
reads each minute's file, and hands the rows to the minute_quotes/ table. What
these pin: a day with no files is None, so it gets no partition; the table's
columns, read from the feed's names, with each row's minute taken from its
file's name and the feed's epoch milliseconds turned into UTC times; and one row
per stock per minute, sorted by ticker, then minute.

Tier 1: pure, no Spark. The S3 client is a stub (tests/minute_quote_feed.py).
"""
import logging

import pyarrow as pa
import pytest

from minute_quote_feed import FEED_SCHEMA, file_name, folder_client, minute_file, quote
from spark_jobs.graph import minute_quotes

DAY = "2026-10-28"
FOLDER = "year=2026/month=10/day=28/"


def _name(hh_mm):
    """The name of DAY's file for the minute ``hh_mm``, UTC."""
    return file_name(f"{DAY}T{hh_mm}:00Z")


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


def _schema_without(name):
    """The feed's schema less one column, as if upstream had renamed it."""
    return FEED_SCHEMA.remove(FEED_SCHEMA.get_field_index(name))


def _warnings(caplog):
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]


def test_a_days_files_are_read_into_rows():
    rows = _read({
        _name("13:31"): minute_file([quote("AAPL"), quote("MSFT")]),
        _name("13:32"): minute_file([quote("AAPL"), quote("MSFT")]),
    })
    assert [(row["ticker"], row["capture_time"]) for row in _table(rows)] == [
        ("AAPL", "2026-10-28T13:31:00.000Z"),
        ("AAPL", "2026-10-28T13:32:00.000Z"),
        ("MSFT", "2026-10-28T13:31:00.000Z"),
        ("MSFT", "2026-10-28T13:32:00.000Z"),
    ]


def test_the_rows_are_sorted_by_ticker_then_minute():
    """Whatever order a file's rows come in. A ticker read prunes row groups
    only when each ticker's minutes sit together."""
    rows = _read({
        _name("13:32"): minute_file([quote("MSFT"), quote("AAPL")]),
        _name("13:31"): minute_file([quote("MSFT"), quote("AAPL")]),
    })
    pairs = [(row["ticker"], row["capture_time"]) for row in _table(rows)]
    assert pairs == sorted(pairs)
    assert [ticker for ticker, _ in pairs] == ["AAPL", "AAPL", "MSFT", "MSFT"]


def test_a_row_carries_the_tables_columns_and_not_the_raw_json():
    """Each read from the feed's name for it."""
    (row,) = _read({_name("13:31"): minute_file([quote()])})
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
    (row,) = _table(_read({_name("13:31"): minute_file([quote(trade_time=None)])}))
    assert row["trade_time"] is None
    assert row["quote_time"] == "2026-10-28T13:30:59.123Z"


@pytest.mark.parametrize("key, minute", [
    ("year=2026/month=10/day=28/2026-10-28T13-31-00Z.snappy.parquet",
     "2026-10-28T13:31:00.000Z"),
    ("2026-10-28T20-00-00Z.parquet", "2026-10-28T20:00:00.000Z"),
    ("13-31.parquet", None),
    ("2026-10-28T13:31:00Z.parquet", None),
    ("2026-10-28T24-31-00Z.snappy.parquet", None),
])
def test_a_files_name_is_its_minute(key, minute):
    assert minute_quotes.file_minute(key) == minute


def test_a_rows_minute_is_its_files_name_not_when_its_call_was_answered():
    """No column holds the minute a poll was due. A late or retried call is
    answered in the next minute, and its rows still belong to its file's."""
    (row,) = _table(_read({
        _name("13:31"): minute_file([quote(capture_timestamp="2026-10-28T13:32:00.412Z")]),
    }))
    assert row["capture_time"] == "2026-10-28T13:31:00.000Z"


def test_a_file_not_named_for_a_minute_is_left_out_unread(caplog):
    """Its rows would have no minute."""
    client = folder_client({
        f"{FOLDER}{_name('13:31')}": minute_file([quote()]),
        f"{FOLDER}latest.parquet": minute_file([quote("MSFT")]),
    })
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = minute_quotes.read_minute_quotes("b", "", DAY, client)

    assert [row["ticker"] for row in _table(rows)] == ["AAPL"]
    assert client.requested == [f"{FOLDER}{_name('13:31')}"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "Could not read 1 of the 2 minute quote files" in warnings[0]
    assert f"{FOLDER}latest.parquet: not named for a minute" in warnings[0]


def test_tickers_are_trimmed_and_upper_cased():
    """snapshots/ spells a symbol in upper case, and the two join on it."""
    rows = _read({_name("13:31"): minute_file([quote(" brk/b ")])})
    assert _table(rows)[0]["ticker"] == "BRK/B"


def test_a_stock_quoted_twice_in_a_minute_appears_once(caplog):
    """The table is one row per stock per minute. The first read wins, and the
    repeat is logged."""
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = _read({
            _name("13:31"): minute_file([quote(bid_price=1.0), quote(bid_price=2.0)]),
        })
    assert [row["bid"] for row in _table(rows)] == [1.0]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith("1 minute quote rows")


def test_a_row_with_no_ticker_is_skipped(caplog):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = _read({_name("13:31"): minute_file([
            quote(symbol=None), quote(symbol=" "), quote(),
        ])})
    assert len(rows) == 1
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith("2 minute quote rows")


def test_a_day_with_no_files_returns_none_and_says_why(caplog):
    """Only the day's folder is listed. A file the day before is not the day."""
    client = folder_client({
        f"year=2026/month=10/day=27/{file_name('2026-10-27T13:31:00Z')}":
            minute_file([quote()]),
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
        _name("13:31"): minute_file([quote()]),
        "_SUCCESS": b"",
        "2026-10-28T13-31-00Z.json": b"{}",
        f"retries/{_name('13:31')}": minute_file([quote("MSFT")]),
    })
    assert [row["ticker"] for row in _table(rows)] == ["AAPL"]


def test_the_listing_follows_every_page():
    files = {_name(f"13:3{minute}"): minute_file([quote()]) for minute in range(1, 6)}
    assert len(_read(files, page_size=2)) == 5


@pytest.mark.parametrize("body", [
    b"not parquet",
    minute_file([quote()], _schema_without("total_volume")),
    minute_file([quote(bid_size="lots")], FEED_SCHEMA.set(
        FEED_SCHEMA.get_field_index("bid_size"), pa.field("bid_size", pa.string()),
    )),
], ids=["not parquet", "a column missing", "a column of text"])
def test_a_file_that_cannot_be_read_is_left_out_and_the_rest_kept(caplog, body):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        rows = _read({
            _name("13:31"): minute_file([quote()]),
            _name("13:32"): body,
        })
    assert len(rows) == 1
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "Could not read 1 of the 2 minute quote files" in warnings[0]
    assert f"{FOLDER}{_name('13:32')}" in warnings[0]


def test_a_day_with_no_file_that_can_be_read_returns_none(caplog):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert _read({_name("13:31"): b"not parquet"}) is None
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
    client = folder_client({f"{FOLDER}{_name('13:31')}": minute_file([quote()])})
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
            _name("13:31"): minute_file([quote("AAPL"), quote("MSFT")]),
            _name("13:32"): minute_file([quote("AAPL")]),
        }, prefix="feed")
    loaded = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("Loaded ")
    ]
    assert loaded == [
        f"Loaded 3 minute quotes for {DAY}, 2 stocks over 2 minutes, from 2 "
        f"files under s3://b/feed/{FOLDER}"
    ]
