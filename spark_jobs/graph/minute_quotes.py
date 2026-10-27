"""
One-minute quotes: the index constituents' quotes, one file per minute of the
trading day, read from the upstream minute feed into the minute_quotes/ query
table.

The feed is not RDF and not a source. Each minute of the trading day upstream
polls the constituents' quotes and writes one Parquet file into the day's
folder, ``<prefix>/year=YYYY/month=MM/day=DD/``: 391 files on a full day, about
503 rows each. It is read on the driver, as the stock split feed is, and never
becomes triples, so it changes nothing in the graph or the other tables.

The folders are dated in UTC. The trading day, 9:30 to 16:00 Eastern, falls
inside one UTC date all year, so a day's folder holds the whole trading day.

The feed's raw JSON column is not read. It is the bulk of each file and answers
nothing a lookup asks.
"""
import io
import logging
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
FEED_NAMES: Dict[str, str] = {}
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


def day_folder(prefix: str, day: str) -> str:
    """The folder holding ``day``'s files, ending in a slash."""
    year, month, dd = day.split("-")
    name = f"year={year}/month={month}/day={dd}/"
    prefix = (prefix or "").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


def iso_time(value) -> Optional[str]:
    """A time as ``YYYY-MM-DDTHH:MM:SS.mmmZ`` in UTC, the way snapshots/ writes
    its capture times, or None when it is not one. A time with no zone is
    taken as UTC."""
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.strip())
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None
    return f"{value:%Y-%m-%dT%H:%M:%S}.{value.microsecond // 1000:03d}Z"


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
    Each None is logged. A file that cannot be read is left out and logged, and
    the rest of the day is kept.
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

    # The type each column is read as. capture_time keeps the feed's, a string
    # or a timestamp, and iso_time reads either.
    types = {
        "ticker": pa.string(),
        **{column: pa.float64() for column in _DOUBLES},
        **{column: pa.int64() for column in _LONGS + _EPOCH_MS},
    }
    quotes: Dict[Tuple[str, str], Tuple] = {}
    unread: List[str] = []
    skipped = duplicates = 0
    for key in keys:
        # Everything about one file is inside the try, so a file upstream wrote
        # in another shape costs that minute and never the run.
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            read = pq.read_table(
                io.BytesIO(body),
                columns=[FEED_NAMES.get(column, column) for column in COLUMNS],
            )
            table = pa.table(
                [
                    read[FEED_NAMES.get(column, column)].cast(types[column])
                    if column in types
                    else read[FEED_NAMES.get(column, column)]
                    for column in COLUMNS
                ],
                names=list(COLUMNS),
            )
            records = table.to_pylist()
        except (ClientError, BotoCoreError, pa.ArrowException, ValueError,
                TypeError, KeyError, OSError) as e:
            unread.append(f"{key}: {e}")
            continue

        for record in records:
            ticker = (record["ticker"] or "").strip().upper()
            minute = iso_time(record["capture_time"])
            if not ticker or minute is None:
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
            f"{skipped} minute quote rows under {where} had no ticker or no "
            f"capture_time, and were skipped"
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
