"""Spark session tuned for the end-to-end pipeline smoke test.

Overrides the shared session-scoped ``spark`` fixture (tests/conftest.py), which
is sized for the tiny unit tests (default ~1 GB heap). The full ``build_graph``
pipeline fans out into hundreds of Spark jobs/stages even on the small fixtures,
and the driver's status listeners retain per-job/stage/task metadata by default
(1000 jobs / 1000 stages / 100000 tasks) — that unbounded accumulation OOMs the
driver heap (observed: OutOfMemoryError in SQLAppStatusListener.onJobStart).

This fixture caps that retention so the status store stays small, and is
function-scoped so each e2e test gets a fresh SparkContext (no cross-test
accumulation).

The driver *heap* itself is raised in the CI e2e job via PYSPARK_SUBMIT_ARGS
(``spark.driver.memory`` is only honored at JVM launch, before pyspark starts).
"""

import pytest


@pytest.fixture(scope="function")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    warehouse = tmp_path_factory.mktemp("spark_warehouse_e2e")

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("pyg-kg-builder-e2e")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.driver.host", "127.0.0.1")
        # Bound the driver status store — the pipeline emits hundreds of
        # jobs/stages on tiny data; default retention OOMs the heap.
        .config("spark.ui.retainedJobs", "40")
        .config("spark.ui.retainedStages", "80")
        .config("spark.ui.retainedTasks", "1000")
        .config("spark.sql.ui.retainedExecutions", "40")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()
