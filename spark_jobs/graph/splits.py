"""
Stock splits: the day's splits, read from the upstream split feed.

The feed is not RDF and not a source. It is one small Parquet object per month,
``<prefix>/year=YYYY/MM.snappy.parquet``, with three string columns: ``ticker``
(lower case), ``split_ratio`` (new shares to old, ``4:1``) and ``split_date``
(``MM/DD/YYYY``). It is read on the driver, as the index-constituents CSV is, and
never becomes triples, so it changes nothing in the graph or the other tables.

``split_date`` is the last trading day at the old price. Checked against the
quotes for KLAC, DD, CRWD, MNST and APH in 2026: each changed scale on the next
trading day.

The feed rewrites the month's object just after midnight Eastern on each weekday
it runs, with the month so far. So by the nightly run the object can already hold
the next day's splits, and an object last written before the day began was not
written by that day's run. The first is why rows are picked by date, and the
second is why such an object counts as missing: read, it would report a day the
feed never covered as a day with no splits.
"""
import io
import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# The job's logger, not this module's -- see the note in graph/config.py.
logger = logging.getLogger("build_graph")

# The feed's name. A day's marker lists it among the day's sources when the day
# has a splits/ partition (bin/publish_run.py).
FEED_NAME = "stock-split"

# The feed runs just after midnight Eastern, so a day begins at Eastern midnight.
FEED_TIMEZONE = ZoneInfo("America/New_York")

FEED_COLUMNS = ("ticker", "split_ratio", "split_date")


def splits_key(prefix: str, day: str) -> str:
    """The month object holding ``day``'s splits."""
    year, month, _ = day.split("-")
    name = f"year={year}/{month}.snappy.parquet"
    prefix = (prefix or "").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


def parse_ratio(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """``(shares_after, shares_before)`` from ``4:1`` or ``4.00:1.00``.

    Both None unless the text is two positive numbers. The feed also writes a
    bare ``:``.
    """
    after, colon, before = (text or "").partition(":")
    if not colon:
        return None, None
    try:
        numbers = (float(after), float(before))
    except ValueError:
        return None, None
    if not all(math.isfinite(n) and n > 0 for n in numbers):
        return None, None
    return numbers


def parse_split_date(text: Optional[str]) -> Optional[date]:
    """The date in ``MM/DD/YYYY``, or None when it is not one."""
    try:
        return datetime.strptime((text or "").strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def read_splits(
    bucket: str,
    prefix: str,
    day: str,
    s3_client=None,
) -> Optional[List[Dict[str, Any]]]:
    """The splits dated ``day``, sorted by ticker.

    ``[]`` when the feed ran for the day and listed no splits. None when it was
    not read for the day: no location set, no object, an object last written
    before the day began, or one that could not be read. Each None is logged.
    """
    # Imported here, not at the top, so bin/publish_run.py can import this
    # module on the system python with the standard library alone.
    import boto3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from botocore.exceptions import BotoCoreError, ClientError

    if not bucket:
        logger.info(
            "No stock splits location (--stock_splits_bucket), so no splits "
            "table"
        )
        return None

    key = splits_key(prefix, day)
    where = f"s3://{bucket}/{key}"

    # BotoCoreError as well as ClientError: a dropped connection or missing
    # credentials must cost the splits table, never the run.
    try:
        client = s3_client or boto3.client("s3")
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.warning(
                f"No stock splits file at {where}, so no splits table for {day}"
            )
        else:
            logger.warning(
                f"Could not read stock splits from {where}: {e}. No splits "
                f"table for {day}"
            )
        return None
    except BotoCoreError as e:
        logger.warning(
            f"Could not read stock splits from {where}: {e}. No splits table "
            f"for {day}"
        )
        return None

    written = response.get("LastModified")
    began = datetime.fromisoformat(day).replace(tzinfo=FEED_TIMEZONE)
    if written is None or written < began:
        when = (
            written.astimezone(FEED_TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")
            if written is not None else "at an unknown time"
        )
        logger.warning(
            f"Stock splits file {where} was last written {when}, before {day} "
            f"began, so the feed did not run for {day}. No splits table"
        )
        return None

    try:
        table = pq.read_table(io.BytesIO(body), columns=list(FEED_COLUMNS))
    except (pa.ArrowException, ValueError, OSError) as e:
        logger.warning(
            f"Could not read stock splits from {where}: {e}. No splits table "
            f"for {day}"
        )
        return None

    wanted = date.fromisoformat(day)
    splits: Dict[Tuple[str, str], Dict[str, Any]] = {}
    undated = 0
    for row in table.to_pylist():
        split_date = parse_split_date(row["split_date"])
        if split_date is None:
            undated += 1
            continue
        if split_date != wanted:
            continue
        ticker = (row["ticker"] or "").strip().upper()
        ratio = row["split_ratio"] or ""
        shares_after, shares_before = parse_ratio(ratio)
        # Keyed on what is published, so the feed's duplicate rows appear once.
        splits[(ticker, ratio)] = {
            "ticker": ticker,
            "ratio": ratio,
            "shares_after": shares_after,
            "shares_before": shares_before,
            "split_date": split_date,
        }

    if undated:
        logger.warning(
            f"{undated} rows in {where} have a split_date that is not "
            f"MM/DD/YYYY, and were skipped"
        )
    rows = [splits[pair] for pair in sorted(splits)]
    logger.info(f"Loaded {len(rows)} stock splits for {day} from {where}")
    return rows
