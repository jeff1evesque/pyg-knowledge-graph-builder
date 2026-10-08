"""Unit tests for the SEC company facts history reader
(spark_jobs.graph.companyfacts).

The history is not a source. A run reads the year's object on the driver, keeps
the rows filed on the day it is building, leaves out the numbers a filing only
repeated, and hands the rest to the companyfacts/ table. As with the splits, a
day nothing was filed ([]) and a day the history was not read for (None) must
stay apart, because a reader of the table can only tell them apart by it.

Tier 1: pure, no Spark. The S3 client is a stub (tests/split_feed.py), serving
an object built by tests/companyfacts_feed.py.
"""
import logging
from datetime import date, datetime, timezone

from companyfacts_feed import HISTORY_SCHEMA, fact, history_object
from spark_jobs.graph import companyfacts
from spark_jobs.sources import sec
from split_feed import feed_client

DAY = "2026-10-06"
PREFIX = "raw/source=sec/feed=companyfacts"
KEY = f"{PREFIX}/2026.snappy.parquet"
# 23:15 Eastern on the day, when upstream writes: 03:15 UTC the next morning.
FRESH = datetime(2026, 10, 7, 3, 15, tzinfo=timezone.utc)
# 23:15 Eastern on the day before.
EARLIER = datetime(2026, 10, 6, 3, 15, tzinfo=timezone.utc)

# A day of the history: the day's own filings, a number a 10-Q only repeated,
# and filings from the days either side.
REVENUE = fact(
    cik="0000789019", entity_name="MICROSOFT CORPORATION", filed=DAY,
    first_filed=DAY, accn="0000950170-26-000101", period_start="2026-04-01",
    period_end="2026-06-30", value=76441000000.0, fp="FY", fy="2026",
    form="10-K",
)
RESTATED = fact(
    cik="0000789019", entity_name="MICROSOFT CORPORATION", filed=DAY,
    first_filed="2025-07-30", accn="0000950170-26-000101",
    concept="NetIncomeLoss", property="netIncome", period_start="2024-07-01",
    period_end="2025-06-30", value=88136000000.0, prior_value=88100000000.0,
    status="changed", fp="FY", fy="2026", form="10-K", length="year",
)
REPEATED = fact(
    cik="0000789019", entity_name="MICROSOFT CORPORATION", filed=DAY,
    first_filed="2025-07-30", accn="0000950170-26-000101",
    period_start="2024-04-01", period_end="2024-06-30", value=64727000000.0,
    status="repeated", fp="FY", fy="2026", form="10-K",
)
# An instant has no start, and a fee table no fiscal year or property.
FEE = fact(
    cik="0000019617", entity_name="JPMORGAN CHASE & CO", taxonomy="ffd",
    concept="NrrtvMaxAggtOfferingPric", period_start=None, period_end=DAY,
    value=7605840.0, fy=None, fp=None, form="424B2",
    accn="0001213900-26-107088", filed=DAY, first_filed=DAY, property=None,
    length=None,
)
DAY_BEFORE = fact(filed="2026-10-05", first_filed="2026-10-05")
DAY_AFTER = fact(filed="2026-10-07", first_filed="2026-10-07")
YEAR = [DAY_AFTER, REPEATED, RESTATED, DAY_BEFORE, REVENUE, FEE]


def _read(rows=YEAR, written=FRESH, day=DAY, prefix=PREFIX):
    client = feed_client({KEY: (history_object(rows), written)})
    return companyfacts.read_companyfacts("b", prefix, day, client)


def _as_dicts(rows):
    return [dict(zip(companyfacts.COLUMNS, row)) for row in rows]


def _warnings(caplog):
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]


def test_only_the_days_filings_are_kept_without_the_repeats():
    """A filing reports earlier periods again beside the new one. Upstream marks
    the unchanged ones repeated, and leaving them out keeps each number once,
    on the day it was first filed."""
    rows = _as_dicts(_read())
    assert [(row["cik"], row["concept"], row["status"]) for row in rows] == [
        ("0000019617", "NrrtvMaxAggtOfferingPric", "new"),
        ("0000789019", "NetIncomeLoss", "changed"),
        ("0000789019", "RevenueFromContractWithCustomerExcludingAssessedTax", "new"),
    ]
    assert {row["filed"] for row in rows} == {date(2026, 10, 6)}


def test_a_row_carries_the_tables_columns_and_types():
    row = _as_dicts(_read())[1]
    assert row == {
        "cik": "0000789019",
        "entity_name": "MICROSOFT CORPORATION",
        "taxonomy": "us-gaap",
        "concept": "NetIncomeLoss",
        "unit": "USD",
        "period_start": date(2024, 7, 1),
        "period_end": date(2025, 6, 30),
        "value": 88136000000.0,
        "fy": 2026,
        "fp": "FY",
        "form": "10-K",
        "accn": "0000950170-26-000101",
        "filed": date(2026, 10, 6),
        "status": "changed",
        "first_filed": date(2025, 7, 30),
        "prior_value": 88100000000.0,
        "property": "netIncome",
        "length": "year",
    }


def test_the_columns_that_describe_no_filing_are_left_out():
    """Besides the filing, the history holds the filed year again, an empty
    Turtle column, and upstream's record of its own fetch: its URL, its user
    agent and its timings."""
    left_out = set(HISTORY_SCHEMA.names) - set(companyfacts.COLUMNS)
    assert left_out == {
        "filed_year", "fetch_start_time", "fetch_end_time", "url",
        "user_agent", "rdf_turtle",
    }
    assert all(len(row) == len(companyfacts.COLUMNS) for row in _read())


def test_an_instant_and_a_fee_table_keep_their_empty_fields():
    fee = _as_dicts(_read())[0]
    assert fee["period_start"] is None
    assert fee["period_end"] == date(2026, 10, 6)
    assert (fee["fy"], fee["fp"], fee["property"], fee["length"]) == (
        None, None, None, None,
    )


def test_a_row_with_no_status_is_kept():
    """Only a row upstream calls a repeat is left out."""
    rows = _read([fact(filed=DAY, status=None)])
    assert len(rows) == 1


def test_a_day_nothing_was_filed_is_empty_not_missing():
    """The object is fresh and holds no row filed on the day. That is [] and
    gets an empty partition, not None."""
    assert _read([DAY_BEFORE, DAY_AFTER]) == []


def test_a_day_of_repeats_only_is_empty():
    assert _read([REPEATED]) == []


def test_the_year_object_is_named_after_the_days_year():
    assert companyfacts.history_key(PREFIX, DAY) == KEY
    assert companyfacts.history_key(f"/{PREFIX}/", DAY) == KEY
    assert companyfacts.history_key("", "2027-01-04") == "2027.snappy.parquet"


def test_a_missing_object_is_a_warning_naming_it(caplog):
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert companyfacts.read_companyfacts(
            "b", PREFIX, DAY, feed_client({})
        ) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert f"s3://b/{KEY}" in warnings[0]


def test_an_object_last_written_on_an_earlier_day_is_still_read(caplog):
    """Upstream rewrites the object only when something new was filed, so on a
    day nothing was it keeps an earlier day's write time. Rows are picked by
    filed alone, and such a day is empty rather than missing."""
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert _read([DAY_BEFORE], written=EARLIER) == []
        assert _read(written=EARLIER) == _read()
    assert _warnings(caplog) == []


def test_a_dropped_connection_is_a_warning_not_a_failed_run(caplog):
    from botocore.exceptions import EndpointConnectionError

    class _Unreachable:
        @staticmethod
        def get_object(Bucket, Key):
            raise EndpointConnectionError(endpoint_url="https://s3.example.com")

    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert companyfacts.read_companyfacts(
            "b", PREFIX, DAY, _Unreachable
        ) is None
    assert len(_warnings(caplog)) == 1


def test_an_object_that_is_not_parquet_is_a_warning(caplog):
    client = feed_client({KEY: (b"not parquet", FRESH)})
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert companyfacts.read_companyfacts("b", PREFIX, DAY, client) is None
    assert len(_warnings(caplog)) == 1


def test_an_object_missing_a_column_is_a_warning(caplog):
    """A column upstream drops costs the table, never the run."""
    import pyarrow as pa

    schema = pa.schema([f for f in HISTORY_SCHEMA if f.name != "length"])
    client = feed_client({KEY: (history_object(YEAR, schema), FRESH)})
    with caplog.at_level(logging.WARNING, logger="build_graph"):
        assert companyfacts.read_companyfacts("b", PREFIX, DAY, client) is None
    assert len(_warnings(caplog)) == 1


def test_no_bucket_reads_nothing():
    client = feed_client({KEY: (history_object(YEAR), FRESH)})
    assert companyfacts.read_companyfacts("", PREFIX, DAY, client) is None
    assert client.requested == []


def test_the_loaded_line_names_the_object_and_the_counts(caplog):
    with caplog.at_level(logging.INFO, logger="build_graph"):
        _read()
    loaded = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("Loaded ")
    ]
    assert loaded == [
        f"Loaded 3 company facts filed on {DAY} from s3://b/{KEY}, "
        f"leaving out 1 repeated"
    ]


def test_the_feed_is_named_as_the_snapshots_feed_is():
    """One name for both: a day's marker lists it for the table, and a run's
    sources list it for the snapshot. bin/publish_run.py cannot import
    sources/sec.py on the system python, so the name is written twice."""
    names = {feed.name for feed in sec.SPEC.feeds}
    assert companyfacts.FEED_NAME in names
