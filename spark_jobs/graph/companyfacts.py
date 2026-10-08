"""
SEC company facts history: the numbers filed on a day, read from the upstream
history into the companyfacts/ query table.

The history is not read as a source. It is one Parquet object per year,
``<prefix>/YYYY.snappy.parquet``, holding the year's filings: one row for each
number a filing reported, under the year it was filed. It carries no RDF. It is
read on the driver, as the stock split feed is, and never becomes triples, so it
changes nothing in the graph or the other tables.

A filing reports earlier periods again beside the new one, so a number appears
once for every filing that reported it. Upstream marks a number a filing gave
again, unchanged, as ``repeated``, and those rows are left out. What remains is
each number once, on the day it was first filed, and each later change on the
day it was filed.

EDGAR takes filings until 10 p.m. Eastern. An object last written before then on
the day can still lack some of the day's filings, so it counts as not written for
that day. Read, it would publish part of a day as the whole of it, and no later
day's table puts the rest back.
"""
import io
import logging
from datetime import date, datetime, time
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

# The job's logger, not this module's -- see the note in graph/config.py.
logger = logging.getLogger("build_graph")

# The feed's name, the same as the snapshot's Feed in sources/sec.py: a day's
# marker lists it among the day's sources when the day has a companyfacts/
# partition (bin/publish_run.py).
FEED_NAME = "sec-companyfacts"

FEED_TIMEZONE = ZoneInfo("America/New_York")
# When EDGAR stops taking filings for the day, in FEED_TIMEZONE.
EDGAR_CLOSES = time(22, 0)

# The status upstream gives a number a filing gave again, unchanged.
REPEATED = "repeated"

# The columns kept, in the history's own order. The rest of the history is the
# filed year again, an empty Turtle column and upstream's record of its own fetch.
COLUMNS = (
    "cik", "entity_name", "taxonomy", "concept", "unit", "period_start",
    "period_end", "value", "fy", "fp", "form", "accn", "filed", "status",
    "first_filed", "prior_value", "property", "length",
)
_DATE_COLUMNS = ("period_start", "period_end", "filed", "first_filed")


def history_key(prefix: str, day: str) -> str:
    """The year object holding ``day``'s filings."""
    name = f"{day[:4]}.snappy.parquet"
    prefix = (prefix or "").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


def _date(value) -> Optional[date]:
    """``YYYY-MM-DD`` as a date, or None when it is not one."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _year(value) -> Optional[int]:
    """A fiscal year such as ``2026`` as a number, or None when it is not one."""
    text = "" if value is None else str(value)
    return int(text) if text.isdigit() else None


def _row(record: dict) -> Tuple:
    """One history row as the table holds it: dates as dates, ``fy`` a number."""
    for column in _DATE_COLUMNS:
        record[column] = _date(record[column])
    record["fy"] = _year(record["fy"])
    return tuple(record[column] for column in COLUMNS)


def read_companyfacts(
    bucket: str,
    prefix: str,
    day: str,
    s3_client=None,
) -> Optional[List[Tuple]]:
    """The history rows filed on ``day``, without the repeated ones, as tuples
    in COLUMNS order, sorted by company, concept and period.

    ``[]`` when the history was read and nothing was filed that day. None when
    it was not read for the day: no location set, no object, an object last
    written before EDGAR closed on the day, or one that could not be read. Each
    None is logged.
    """
    # Imported here, not at the top, so bin/publish_run.py can import this
    # module on the system python with the standard library alone.
    import boto3
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from botocore.exceptions import BotoCoreError, ClientError

    if not bucket:
        logger.info(
            "No company facts history location (--companyfacts_history_bucket), "
            "so no companyfacts table"
        )
        return None

    key = history_key(prefix, day)
    where = f"s3://{bucket}/{key}"

    # BotoCoreError as well as ClientError: a dropped connection or missing
    # credentials must cost the companyfacts table, never the run.
    try:
        client = s3_client or boto3.client("s3")
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.warning(
                f"No company facts history at {where}, so no companyfacts "
                f"table for {day}"
            )
        else:
            logger.warning(
                f"Could not read the company facts history from {where}: {e}. "
                f"No companyfacts table for {day}"
            )
        return None
    except BotoCoreError as e:
        logger.warning(
            f"Could not read the company facts history from {where}: {e}. No "
            f"companyfacts table for {day}"
        )
        return None

    written = response.get("LastModified")
    closed = datetime.combine(
        date.fromisoformat(day), EDGAR_CLOSES, tzinfo=FEED_TIMEZONE
    )
    if written is None or written < closed:
        when = (
            written.astimezone(FEED_TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")
            if written is not None else "at an unknown time"
        )
        logger.warning(
            f"Company facts history {where} was last written {when}, before "
            f"EDGAR closed on {day}, so it may lack some of that day's filings. "
            f"No companyfacts table"
        )
        return None

    # Everything from here is inside the try, so an object upstream changed the
    # shape of costs the table and never the run.
    try:
        table = pq.read_table(io.BytesIO(body), columns=list(COLUMNS))
        filed = table.filter(pc.equal(table["filed"], day))
        # A row with no status is kept: only a row upstream calls a repeat goes.
        kept = filed.filter(
            pc.fill_null(pc.not_equal(filed["status"], REPEATED), True)
        )
        records = sorted(
            kept.to_pylist(),
            key=lambda r: tuple(
                r[column] or "" for column in (
                    "cik", "taxonomy", "concept", "unit", "period_end",
                    "period_start", "accn",
                )
            ),
        )
        rows = [_row(record) for record in records]
    except (pa.ArrowException, ValueError, TypeError, KeyError, OSError) as e:
        logger.warning(
            f"Could not read the company facts history from {where}: {e}. No "
            f"companyfacts table for {day}"
        )
        return None

    logger.info(
        f"Loaded {len(rows)} company facts filed on {day} from {where}, "
        f"leaving out {filed.num_rows - len(rows)} repeated"
    )
    return rows
