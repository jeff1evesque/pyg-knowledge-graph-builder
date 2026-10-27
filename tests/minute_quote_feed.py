"""A stub S3 client serving the one-minute quote feed.

Shared by tests/test_minute_quotes.py and tests/test_query_tables.py. The feed is
one Parquet file per minute in a day's folder, with the raw JSON column the table
leaves out. The reader lists the folder, then reads each file, so the stub
serves both, two keys a page so a listing takes more than one.
"""
import io
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from spark_jobs.graph import minute_quotes

_TYPES = {
    "ticker": pa.string(),
    "capture_time": pa.string(),
    "quote_time": pa.int64(),
    "trade_time": pa.int64(),
    "bid_size": pa.int64(),
    "ask_size": pa.int64(),
    "volume": pa.int64(),
}
# In the feed's names, which the reader maps to the table's.
FEED_SCHEMA = pa.schema(
    [
        (minute_quotes.FEED_NAMES.get(column, column), _TYPES.get(column, pa.float64()))
        for column in minute_quotes.COLUMNS
    ]
    + [("raw_json", pa.string())]
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def epoch_ms(text: str) -> int:
    """A UTC ISO-8601 time as the feed's epoch milliseconds."""
    return (datetime.fromisoformat(text) - _EPOCH) // timedelta(milliseconds=1)


def quote(ticker="AAPL", minute="2026-10-28T13:31:00Z", **overrides):
    """One stock's quote for one minute, in the table's names, unless overridden."""
    row = {
        "ticker": ticker,
        "capture_time": minute,
        "quote_time": epoch_ms("2026-10-28T13:30:59.123Z"),
        "trade_time": epoch_ms("2026-10-28T13:30:58.004Z"),
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
        "raw_json": '{"quote": "the bulk of the file"}',
    }
    row.update(overrides)
    return row


def minute_file(rows, schema=FEED_SCHEMA):
    """Parquet bytes for these rows, as upstream writes one minute's file."""
    named = [
        {minute_quotes.FEED_NAMES.get(name, name): value for name, value in row.items()}
        for row in rows
    ]
    table = pa.Table.from_pylist(
        [{name: row.get(name) for name in schema.names} for row in named],
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
