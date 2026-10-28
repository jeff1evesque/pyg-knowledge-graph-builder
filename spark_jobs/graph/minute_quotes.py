"""
One-minute quotes: the index constituents' quotes, one file per minute of the
trading day, read from the upstream minute feed into the minute_quotes/ query
table.

The feed is not RDF and not a source. Each minute of the trading day upstream
polls the constituents' quotes and writes one Parquet file into the day's
folder, ``<prefix>/year=YYYY/month=MM/day=DD/``: 391 files on a full day, about
503 rows each. It is read on the driver, as the stock split feed is, and never
becomes triples, so it changes nothing in the graph or the other tables.

A file is named for the minute its poll was due, in UTC with dashes for the
colons: ``2026-10-28T13-31-00Z.snappy.parquet``. No column holds that minute.
The feed's capture_timestamp is when each call was answered, and a late or
retried call is answered in the next minute, so a row's capture_time is its
file's name.

The folders are dated in UTC. The trading day, 9:30 to 16:00 Eastern, falls
inside one UTC date all year, so a day's folder holds the whole trading day.

The feed's raw JSON column is not read. It is the bulk of each file and answers
nothing a lookup asks.
"""
import io
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# The job's logger, not this module's -- see the note in graph/config.py.
logger = logging.getLogger("build_graph")

# The feed's name. A day's marker lists it among the day's sources when the day
# has a minute_quotes/ partition (bin/publish_run.py).
FEED_NAME = "minute-quotes"

# The table's columns, in order.
COLUMNS = (
    "ticker", "capture_time", "quote_time", "trade_time", "bid", "ask",
    "bid_size", "ask_size", "last", "mark", "open", "high", "low",
    "prev_close", "volume", "net_change", "net_percent_change",
)
# The feed's name for a column, where it is not the column's own.
FEED_NAMES: Dict[str, str] = {
    "ticker": "symbol",
    "bid": "bid_price",
    "ask": "ask_price",
    "last": "last_price",
    "open": "open_price",
    "high": "high_price",
    "low": "low_price",
    "prev_close": "close_price",
    "volume": "total_volume",
}
# The columns read from each file: all but capture_time, the file's name.
FILE_COLUMNS = tuple(column for column in COLUMNS if column != "capture_time")
# Read as numbers: prices and changes as doubles, sizes and the day's running
# volume as whole numbers.
_DOUBLES = (
    "bid", "ask", "last", "mark", "open", "high", "low", "prev_close",
    "net_change", "net_percent_change",
)
_LONGS = ("bid_size", "ask_size", "volume")
# Epoch milliseconds in the feed, UTC ISO-8601 in the table.
_EPOCH_MS = ("quote_time", "trade_time")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# The minute at the start of a file's name.
_FILE_MINUTE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})Z\.")


def day_folder(prefix: str, day: str) -> str:
    """The folder holding ``day``'s files, ending in a slash."""
    year, month, dd = day.split("-")
    name = f"year={year}/month={month}/day={dd}/"
    prefix = (prefix or "").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


def iso_time(value: datetime) -> str:
    """A UTC time as ``YYYY-MM-DDTHH:MM:SS.mmmZ``, the way snapshots/ writes its
    capture times."""
    return f"{value:%Y-%m-%dT%H:%M:%S}.{value.microsecond // 1000:03d}Z"


def file_minute(key: str) -> Optional[str]:
    """The minute a file is named for, as ``iso_time`` writes a time, or None
    when its name is not a minute."""
    match = _FILE_MINUTE.match(key.rsplit("/", 1)[-1])
    if not match:
        return None
    try:
        return iso_time(datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S"))
    except ValueError:
        return None


def epoch_ms_time(value: Optional[int]) -> Optional[str]:
    """Epoch milliseconds as ``iso_time`` writes a time, or None when there is
    none. Added as whole milliseconds, so none is lost to rounding."""
    if value is None:
        return None
    try:
        return iso_time(_EPOCH + timedelta(milliseconds=value))
    except OverflowError:
        return None


def _day_files(client, bucket: str, folder: str) -> List[str]:
    """The Parquet files directly in ``folder``, in key order."""
    keys: List[str] = []
    request = {"Bucket": bucket, "Prefix": folder, "Delimiter": "/"}
    while True:
        page = client.list_objects_v2(**request)
        keys += [
            item["Key"] for item in page.get("Contents", [])
            if item["Key"].endswith(".parquet")
        ]
        if not page.get("IsTruncated"):
            return sorted(keys)
        request["ContinuationToken"] = page["NextContinuationToken"]


def read_minute_quotes(
    bucket: str,
    prefix: str,
    day: str,
    s3_client=None,
) -> Optional[List[Tuple]]:
    """The day's quotes, one row per stock per minute, as tuples in COLUMNS
    order, sorted by ticker, then minute.

    None when the day was not read: no location set, no files in the day's
    folder, a folder that could not be listed, or no file that could be read.
    Each None is logged. A file that cannot be read, or is not named for a
    minute, is left out and logged, and the rest of the day is kept.
    """
    # Imported here, not at the top, so bin/publish_run.py can import this
    # module on the system python with the standard library alone.
    import boto3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from botocore.exceptions import BotoCoreError, ClientError

    if not bucket:
        logger.info(
            "No minute quotes location (--minute_quotes_bucket), so no "
            "minute_quotes table"
        )
        return None

    folder = day_folder(prefix, day)
    where = f"s3://{bucket}/{folder}"

    # BotoCoreError as well as ClientError: a dropped connection or missing
    # credentials must cost the minute_quotes table, never the run.
    try:
        client = s3_client or boto3.client("s3")
        keys = _day_files(client, bucket, folder)
    except (ClientError, BotoCoreError) as e:
        logger.warning(
            f"Could not list minute quotes under {where}: {e}. No "
            f"minute_quotes table for {day}"
        )
        return None
    if not keys:
        logger.warning(
            f"No minute quote files under {where}, so no minute_quotes table "
            f"for {day}"
        )
        return None

    # The type each column is read as. The feed writes its text as large
    # strings.
    types = {
        "ticker": pa.string(),
        **{column: pa.float64() for column in _DOUBLES},
        **{column: pa.int64() for column in _LONGS + _EPOCH_MS},
    }
    quotes: Dict[Tuple[str, str], Tuple] = {}
    unread: List[str] = []
    skipped = duplicates = 0
    for key in keys:
        minute = file_minute(key)
        if minute is None:
            unread.append(f"{key}: not named for a minute")
            continue
        # Everything about one file is inside the try, so a file upstream wrote
        # in another shape costs that minute and never the run.
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            read = pq.read_table(
                io.BytesIO(body),
                columns=[FEED_NAMES.get(column, column) for column in FILE_COLUMNS],
            )
            table = pa.table(
                [
                    read[FEED_NAMES.get(column, column)].cast(types[column])
                    for column in FILE_COLUMNS
                ],
                names=list(FILE_COLUMNS),
            )
            records = table.to_pylist()
        except (ClientError, BotoCoreError, pa.ArrowException, ValueError,
                TypeError, KeyError, OSError) as e:
            unread.append(f"{key}: {e}")
            continue

        for record in records:
            ticker = (record["ticker"] or "").strip().upper()
            if not ticker:
                skipped += 1
                continue
            if (ticker, minute) in quotes:
                duplicates += 1
                continue
            record["ticker"], record["capture_time"] = ticker, minute
            for column in _EPOCH_MS:
                record[column] = epoch_ms_time(record[column])
            quotes[(ticker, minute)] = tuple(record[column] for column in COLUMNS)

    if unread:
        logger.warning(
            f"Could not read {len(unread)} of the {len(keys)} minute quote "
            f"files under {where}, and left them out. The first: {unread[0]}"
        )
    if len(unread) == len(keys):
        logger.warning(f"No minute_quotes table for {day}")
        return None
    if skipped:
        logger.warning(
            f"{skipped} minute quote rows under {where} had no ticker, and "
            f"were skipped"
        )
    if duplicates:
        logger.warning(
            f"{duplicates} minute quote rows under {where} repeated a stock "
            f"and minute already read, and were skipped"
        )

    rows = [quotes[pair] for pair in sorted(quotes)]
    stocks = len({ticker for ticker, _ in quotes})
    minutes = len({minute for _, minute in quotes})
    logger.info(
        f"Loaded {len(rows)} minute quotes for {day}, {stocks} stocks over "
        f"{minutes} minutes, from {len(keys) - len(unread)} files under {where}"
    )
    return rows
