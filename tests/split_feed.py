"""A stub S3 client serving the stock split feed.

Shared by tests/test_splits.py and tests/test_query_tables.py. The feed is one
Parquet object per month with three string columns, and besides the rows the
reader checks the object's write time, so the stub serves both.
"""
import io

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

FEED_SCHEMA = pa.schema([
    ("ticker", pa.string()),
    ("split_ratio", pa.string()),
    ("split_date", pa.string()),
])


def month_object(rows):
    """Parquet bytes for (ticker, split_ratio, split_date) rows, as the feed writes them."""
    table = pa.Table.from_pylist(
        [dict(zip(FEED_SCHEMA.names, row)) for row in rows], schema=FEED_SCHEMA
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()


def feed_client(objects):
    """A stub client serving ``objects``, {key: (body, last_modified)}.

    Records every key requested, in order, so a test can assert what was
    fetched and what was not.
    """
    class _Client:
        requested = []

        @classmethod
        def get_object(cls, Bucket, Key):
            cls.requested.append(Key)
            if Key not in objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            body, written = objects[Key]

            class _Body:
                @staticmethod
                def read():
                    return body

            return {"Body": _Body, "LastModified": written}

    return _Client
