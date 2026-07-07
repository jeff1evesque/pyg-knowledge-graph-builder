"""Spark session tuned for the end-to-end pipeline smoke test.

Overrides the shared session-scoped ``spark`` fixture (tests/conftest.py), which
is sized for the tiny unit tests. The full ``build_graph`` pipeline fans out into
~1,300 Spark jobs/stages even on small fixtures, and the driver's status
listeners retain per-job/stage/task metadata by default — that accumulation OOMs
the driver heap (OutOfMemoryError in SQLAppStatusListener.onJobStart). This
fixture caps that retention. Driver *heap* is raised out-of-band via
PYSPARK_SUBMIT_ARGS (``spark.driver.memory`` is only honored at JVM launch).

This fixture is **session-scoped**: the SparkContext is built once and reused
across all e2e tests, so SparkContext startup is paid once rather than per test.
It was previously function-scoped to bound a driver OOM from accumulating the
tests' *combined* fan-out on the heap; that OOM is now fixed by localCheckpoint
at the enrichment phase boundaries (#186), which truncates the logical plan so
retention no longer grows unbounded. The status-store retention caps below still
bound the listener bus regardless of scope. The tests are mutually isolated —
each writes into its own function-scoped ``tmp_path`` — so sharing the session is
safe.

Parallelism is set to ``local[8]`` with 8 shuffle partitions to use more of the
available cores; the suite is stage-count bound (thousands of tiny per-type/
per-chunk stages), so more parallelism cuts wall-clock materially. Do not raise
the driver heap here — this is not memory bound.

GPU acceleration is opt-in: set ``SPARK_RAPIDS=1`` to run the same tests through
the RAPIDS Accelerator (drop-in SQL plugin). CPU vs GPU produces the same logical
results, so the assertions are identical either way.
"""

import os
import sys

import pytest


def _truthy(val) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    warehouse = tmp_path_factory.mktemp("spark_warehouse_e2e")

    # Pin the executor/worker Python to the *same* interpreter running the
    # driver (this venv). Otherwise PySpark launches workers with whatever
    # `python3` is first on PATH — e.g. the system interpreter, which lacks
    # rdflib — and the turtle_parquet parse UDF silently returns [] for every
    # row (0 triples). Running `.venv/bin/python -m pytest` without activating
    # the venv is exactly that situation, so set this explicitly rather than
    # relying on PATH/PYSPARK_PYTHON being configured by the caller.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    builder = (
        SparkSession.builder
        .master("local[8]")
        .appName("pyg-kg-builder-e2e")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.driver.host", "127.0.0.1")
        # Bound the driver status store — the pipeline emits hundreds of
        # jobs/stages on tiny data; default retention OOMs the heap.
        .config("spark.ui.retainedJobs", "40")
        .config("spark.ui.retainedStages", "80")
        .config("spark.ui.retainedTasks", "1000")
        .config("spark.sql.ui.retainedExecutions", "40")
    )

    # Opt-in GPU acceleration (SPARK_RAPIDS=1). Mirrors
    # conf/spark-rapids.conf.template, but only the settings that apply in
    # local[*] mode — GPU *resource scheduling* (executor.resource.gpu.*,
    # discoveryScript) is a standalone/YARN concern and is omitted here.
    # The RAPIDS Accelerator jar must be on the classpath: set RAPIDS_JAR, or if
    # the plugin fails to load pass it at launch via
    #   PYSPARK_SUBMIT_ARGS="--jars <rapids.jar> ... pyspark-shell"
    if _truthy(os.environ.get("SPARK_RAPIDS")):
        builder = (
            builder
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "true")
            .config("spark.rapids.sql.concurrentGpuTasks",
                    os.environ.get("RAPIDS_CONCURRENT_GPU_TASKS", "2"))
            .config("spark.rapids.memory.pinnedPool.size",
                    os.environ.get("RAPIDS_PINNED_POOL", "2G"))
            # Bound the RAPIDS GPU memory pool. On unified-memory hosts the GPU
            # shares system RAM, and RMM's default (~1.0 of device memory) grows
            # to consume the whole pool and starves the JVM driver (OOM). Cap it
            # so the heap + Python workers keep headroom; override upward on
            # discrete-GPU hosts.
            .config("spark.rapids.memory.gpu.allocFraction",
                    os.environ.get("RAPIDS_GPU_ALLOC_FRACTION", "0.3"))
            .config("spark.rapids.memory.gpu.pool", "ASYNC")
            .config("spark.rapids.sql.format.parquet.reader.type", "MULTITHREADED")
            .config("spark.rapids.sql.explain",
                    os.environ.get("RAPIDS_EXPLAIN", "NONE"))
        )
        rapids_jar = os.environ.get("RAPIDS_JAR")
        if rapids_jar:
            builder = builder.config("spark.jars", rapids_jar)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()
