"""Assert that a RAPIDS run actually executes on the GPU.

Without this, the GPU e2e suite is green in both worlds: if the RAPIDS plugin fails
to load, or silently declines to accelerate the plan, the pipeline still produces
correct output on the CPU and every other test passes. The suite would report
"GPU: 3 passed" while the GPU sat idle.

Reading the plan is the subtle part. Under adaptive query execution the plan you
get back renders as CPU (`AdaptiveSparkPlan isFinalPlan=false`) even when the GPU
runs the query, because AQE substitutes the GPU operators at execution time. So we
disable AQE for this one assertion, where the executed plan is the literal truth.
"""

import os

import pytest
from pyspark.sql import functions as F

pytestmark = pytest.mark.e2e


def _truthy(val) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


requires_rapids = pytest.mark.skipif(
    not _truthy(os.environ.get("SPARK_RAPIDS")),
    reason="SPARK_RAPIDS is not set: this is a CPU run, so there is no GPU to assert on",
)


@requires_rapids
def test_rapids_plugin_is_loaded(spark):
    """The plugin can be configured yet fail to load (e.g. jar missing from the
    driver classpath), in which case Spark carries on silently on the CPU."""
    assert "com.nvidia.spark.SQLPlugin" in spark.conf.get("spark.plugins", "")
    assert _truthy(spark.conf.get("spark.rapids.sql.enabled", "false"))


@requires_rapids
def test_query_actually_executes_on_the_gpu(spark):
    """A GPU-eligible query must produce GPU operators in the executed plan."""
    previous_aqe = spark.conf.get("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    try:
        df = spark.range(0, 10_000).withColumn("k", F.col("id") % 7)
        agg = df.groupBy("k").agg(F.sum("id").alias("total"))

        assert agg.count() == 7  # force execution

        plan = agg._jdf.queryExecution().executedPlan().toString()
        gpu_operators = [
            line.strip()
            for line in plan.splitlines()
            if "Gpu" in line
        ]
        assert gpu_operators, (
            "RAPIDS is enabled but no GPU operator appears in the executed plan -- "
            "the query silently ran on the CPU.\nExecuted plan:\n" + plan
        )
    finally:
        spark.conf.set("spark.sql.adaptive.enabled", previous_aqe)
