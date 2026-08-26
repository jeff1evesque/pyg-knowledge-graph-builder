"""
Pytest configuration for the PySpark enrichment tests.

These tests exercise the current PySpark DataFrame API (BLSIntraSourceLinker,
BLSDatasetEnricher) against a *local* SparkSession — no standalone cluster and
no RAPIDS Accelerator are required. The same application code runs unchanged on
a GPU cluster; RAPIDS is a drop-in SQL plugin, so plain local Spark is a valid
way to unit-test the enrichment logic.
"""
import logging
import os
import sys

import pytest

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Spark's own logging is very chatty at INFO; keep test output readable.
logging.getLogger("py4j").setLevel(logging.ERROR)

# Driver heap, set at IMPORT time rather than inside the spark fixture.
#
# spark.driver.memory is only read when the JVM launches, and in local mode the
# builder is already too late -- it has to arrive through PYSPARK_SUBMIT_ARGS
# before the first getOrCreate() anywhere in the process.
#
# "Anywhere" is the part that matters. There are two session fixtures: this one
# and tests/e2e/conftest.py's, and the e2e directory sorts first, so a plain
# `pytest tests/` builds the e2e session first. Setting this inside the fixture
# below meant it ran only when a UNIT test first asked for a session -- long
# after the e2e fixture had launched the JVM on Spark's 1g default. The whole
# combined run then shared a 1g heap and died partway through with ~150
# OOM-cascade failures that look like unrelated test breakage.
#
# The root conftest is imported before any child conftest, so setting it here
# covers both fixtures. setdefault, so an explicit value still wins -- the e2e
# workflow passes its own (.github/workflows/e2e.yml).
os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 4g pyspark-shell")

# Executors are separate Python processes and default to whatever `python3`
# PATH resolves to -- not this venv. Any test whose query carries a Python UDF
# (the rdflib Turtle parser) then dies with ModuleNotFoundError on the executor
# while the driver imports the module fine, which reads as a code failure
# rather than an environment one.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    """
    Session-scoped local SparkSession.

    Uses local[2] so window functions and joins exercise real partitioning
    while staying fast. Shuffle partitions are capped to keep small-data tests
    from spawning 200 empty tasks per shuffle.

    Retention: this session is reused across the whole suite, including the
    feature-encoding tests, each of which runs a large query (long unionAll
    chains of hash expressions). Spark's SQLAppStatusListener retains one plan
    per execution, so without bounding it the driver heap grows with every
    query and eventually GC-thrashes. Retained executions are capped low below;
    the heap itself is set at module scope, since it has to beat every fixture
    in the process to the first getOrCreate().
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
        # Bound driver-side retention so a long session doesn't leak plan
        # metadata for every executed query.
        .config("spark.sql.ui.retainedExecutions", "5")
        .config("spark.ui.retainedJobs", "20")
        .config("spark.ui.retainedStages", "40")
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
