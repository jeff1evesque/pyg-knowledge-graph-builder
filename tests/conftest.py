"""
Pytest configuration for the PySpark enrichment tests.

These tests exercise the current PySpark DataFrame API (BLSIntraSourceLinker,
BLSDatasetEnricher) against a *local* SparkSession — no standalone cluster and
no RAPIDS Accelerator are required. The same application code runs unchanged on
a GPU cluster; RAPIDS is a drop-in SQL plugin, so plain local Spark is a valid
way to unit-test the enrichment logic.
"""
import logging

import pytest

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Spark's own logging is very chatty at INFO; keep test output readable.
logging.getLogger("py4j").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    """
    Session-scoped local SparkSession.

    Uses local[2] so window functions and joins exercise real partitioning
    while staying fast. Shuffle partitions are capped to keep small-data tests
    from spawning 200 empty tasks per shuffle.
    """
    from pyspark.sql import SparkSession

    warehouse = tmp_path_factory.mktemp("spark_warehouse")

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("pyg-kg-builder-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


@pytest.fixture
def make_triples(spark):
    """
    Factory fixture: build a (subject, predicate, object) triples DataFrame
    from a list of 3-tuples of URI/literal strings.
    """
    def _make(rows):
        return spark.createDataFrame(
            rows, schema="subject STRING, predicate STRING, object STRING"
        )

    return _make
