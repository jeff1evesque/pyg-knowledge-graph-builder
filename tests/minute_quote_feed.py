"""A stub S3 client serving the one-minute quote feed.

Shared by tests/test_minute_quotes.py and tests/test_query_tables.py. The feed is
one Parquet file per minute in a day's folder, named for its minute, in the
feed's own column names, which the reader maps to the table's. The reader lists
the folder, then reads each file, so the stub serves both, two keys a page so a
listing takes more than one.
"""
import io
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

# A minute's file as upstream writes it, less columns no test needs: the quote
# in the feed's names with its text as large strings, and beside it when each
# call was answered, a column of nulls, and the raw JSON, none of which the
# table keeps. Written out rather than built from the reader's name map, so a
# wrong name in the map fails the tests.
FEED_SCHEMA = pa.schema([
    ("capture_timestamp", pa.large_string()),
    ("symbol", pa.large_string()),
    ("ask_price", pa.float64()),
    ("ask_size", pa.int64()),
    ("bid_price", pa.float64()),
    ("bid_size", pa.int64()),
    ("close_price", pa.float64()),
    ("exchange_id", pa.null()),
    ("high_price", pa.float64()),
    ("last_price", pa.float64()),
    ("low_price", pa.float64()),
    ("mark", pa.float64()),
    ("net_change", pa.float64()),
    ("net_percent_change", pa.float64()),
    ("open_price", pa.float64()),
    ("quote_time", pa.int64()),
    ("total_volume", pa.int64()),
    ("trade_time", pa.int64()),
    ("raw_json", pa.large_string()),
])

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def epoch_ms(text: str) -> int:
    """A UTC ISO-8601 time as the feed's epoch milliseconds."""
    return (datetime.fromisoformat(text) - _EPOCH) // timedelta(milliseconds=1)


def file_name(minute="2026-10-28T13:31:00Z"):
    """A minute's file name, as upstream names it: the minute the poll was due,
    in UTC, with dashes for the colons."""
    return f"{minute.replace(':', '-')}.snappy.parquet"


def quote(symbol="AAPL", **overrides):
    """One stock's quote, a row of a minute's file in the feed's names, unless
    overridden."""
    row = {
        "capture_timestamp": "2026-10-28T13:31:01.234Z",
        "symbol": symbol,
        "ask_price": 262.55,
        "ask_size": 200,
        "bid_price": 262.5,
        "bid_size": 300,
        "close_price": 259.58,
        "high_price": 263.0,
        "last_price": 262.52,
        "low_price": 259.8,
        "mark": 262.525,
        "net_change": 2.94,
        "net_percent_change": 1.1326,
        "open_price": 260.1,
        "quote_time": epoch_ms("2026-10-28T13:30:59.123Z"),
        "total_volume": 1234567,
        "trade_time": epoch_ms("2026-10-28T13:30:58.004Z"),
        "raw_json": '{"quote": "the bulk of the file"}',
    }
    row.update(overrides)
    return row


def minute_file(rows, schema=FEED_SCHEMA):
    """Parquet bytes for these rows, as upstream writes one minute's file."""
    table = pa.Table.from_pylist(
        [{name: row.get(name) for name in schema.names} for row in rows],
        schema=schema,
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()


def folder_client(objects, page_size=2):
    """A stub client serving ``objects``, {key: body}.

    Lists like S3 with a delimiter: the keys directly under the prefix, in key
    order, and none below a further slash. Records each listed prefix and each
    key fetched, in order.
    """
    class _Client:
        listed = []
        requested = []

        @classmethod
        def list_objects_v2(cls, Bucket, Prefix, Delimiter=None,
                            ContinuationToken=None):
            cls.listed.append(Prefix)
            keys = sorted(
                key for key in objects
                if key.startswith(Prefix)
                and not (Delimiter and Delimiter in key[len(Prefix):])
            )
            start = int(ContinuationToken or 0)
            page = {"Contents": [{"Key": key} for key in keys[start:start + page_size]]}
            if start + page_size < len(keys):
                page["IsTruncated"] = True
                page["NextContinuationToken"] = str(start + page_size)
            return page

        @classmethod
        def get_object(cls, Bucket, Key):
            cls.requested.append(Key)
            if Key not in objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            body = objects[Key]

            class _Body:
                @staticmethod
                def read():
                    return body

            return {"Body": _Body}

    return _Client
