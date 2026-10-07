"""A stub SEC company facts history object.

Shared by tests/test_companyfacts.py and tests/test_query_tables.py. The history
is one Parquet object per year, in upstream's columns and types, beside columns
like the ones the table leaves out: upstream's record of its own fetch and an
empty Turtle column. split_feed.feed_client serves it, with a write time.
"""
import io

import pyarrow as pa
import pyarrow.parquet as pq

HISTORY_SCHEMA = pa.schema([
    ("cik", pa.large_string()),
    ("entity_name", pa.large_string()),
    ("taxonomy", pa.large_string()),
    ("concept", pa.large_string()),
    ("unit", pa.large_string()),
    ("period_start", pa.large_string()),
    ("period_end", pa.large_string()),
    ("value", pa.float64()),
    ("fy", pa.large_string()),
    ("fp", pa.large_string()),
    ("form", pa.large_string()),
    ("accn", pa.large_string()),
    ("filed", pa.large_string()),
    ("status", pa.large_string()),
    ("first_filed", pa.large_string()),
    ("prior_value", pa.float64()),
    ("filed_year", pa.int64()),
    ("fetch_start_time", pa.float64()),
    ("fetch_end_time", pa.float64()),
    ("url", pa.large_string()),
    ("user_agent", pa.large_string()),
    ("rdf_turtle", pa.null()),
    ("property", pa.large_string()),
    ("length", pa.large_string()),
])


def fact(**overrides):
    """One history row: Apple's quarterly revenue, first filed 2026-07-31,
    unless overridden."""
    row = {
        "cik": "0000320193",
        "entity_name": "Apple Inc.",
        "taxonomy": "us-gaap",
        "concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "unit": "USD",
        "period_start": "2026-03-29",
        "period_end": "2026-06-27",
        "value": 94930000000.0,
        "fy": "2026",
        "fp": "Q3",
        "form": "10-Q",
        "accn": "0000320193-26-000071",
        "filed": "2026-07-31",
        "status": "new",
        "first_filed": "2026-07-31",
        "prior_value": None,
        "filed_year": 2026,
        "fetch_start_time": 1.0,
        "fetch_end_time": 2.0,
        "url": "https://example.com/companyfacts.json",
        "user_agent": "example-agent/1.0",
        "rdf_turtle": None,
        "property": "revenue",
        "length": "quarter",
    }
    row.update(overrides)
    return row


def history_object(rows, schema=HISTORY_SCHEMA):
    """Parquet bytes for these rows, as upstream writes the year's object."""
    table = pa.Table.from_pylist(
        [{name: row.get(name) for name in schema.names} for row in rows],
        schema=schema,
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()
