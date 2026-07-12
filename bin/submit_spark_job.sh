#!/usr/bin/env bash
#
# Submit the PyTorch Geometric Knowledge Graph Builder to an Apache Spark
# standalone cluster with the RAPIDS Accelerator for Apache Spark (GPU).
#
# All cluster-specific values are supplied via environment variables so
# nothing is hardcoded. At minimum, set SPARK_MASTER_URL.
#
# Usage:
#   SPARK_MASTER_URL=spark://<host>:7077 \
#     bin/submit_spark_job.sh \
#       --mode full \
#       --source_paths /data/rdf/monthly/2024-12/ \
#       --local_work_dir /data/pyg \
#       --time_period 2024-12
#
# Environment variables:
#   SPARK_MASTER_URL            (required) e.g. spark://<host>:7077
#   RAPIDS_JAR                  (optional) path to the RAPIDS Accelerator jar,
#                               if it is not already on the cluster classpath
#   GPU_DISCOVERY_SCRIPT        (optional) GPU resource discovery script;
#                               defaults to the script shipped with Spark
#   GPU_PER_EXECUTOR            (optional, default 1)
#   GPU_PER_TASK                (optional, default 0.125 -> up to 8 tasks/GPU)
#   RAPIDS_CONCURRENT_GPU_TASKS (optional, default 2)
#   RAPIDS_PINNED_POOL          (optional, default 2G)
#   RAPIDS_EXPLAIN              (optional, default NONE; set ALL to log which
#                               operators run on GPU vs fall back to CPU)
#   SPARK_EXTRA_CONF            (optional) extra "--conf k=v" flags, space-separated
#
set -euo pipefail

: "${SPARK_MASTER_URL:?set SPARK_MASTER_URL, e.g. spark://<host>:7077}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Package the application code for the executors.
PKG_ZIP="$(mktemp -t spark_jobs.XXXXXX.zip)"
trap 'rm -f "$PKG_ZIP"' EXIT
rm -f "$PKG_ZIP"
zip -q -r "$PKG_ZIP" spark_jobs

GPU_DISCOVERY_SCRIPT="${GPU_DISCOVERY_SCRIPT:-${SPARK_HOME:-/opt/spark}/examples/src/main/scripts/getGpusResources.sh}"

spark-submit \
  --master "$SPARK_MASTER_URL" \
  ${RAPIDS_JAR:+--jars "$RAPIDS_JAR"} \
  --py-files "$PKG_ZIP" \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.rapids.sql.concurrentGpuTasks="${RAPIDS_CONCURRENT_GPU_TASKS:-2}" \
  --conf spark.executor.resource.gpu.amount="${GPU_PER_EXECUTOR:-1}" \
  --conf spark.task.resource.gpu.amount="${GPU_PER_TASK:-0.125}" \
  --conf spark.executor.resource.gpu.discoveryScript="$GPU_DISCOVERY_SCRIPT" \
  --conf spark.rapids.memory.pinnedPool.size="${RAPIDS_PINNED_POOL:-2G}" \
  --conf spark.rapids.memory.gpu.allocFraction="${RAPIDS_GPU_ALLOC_FRACTION:-0.25}" \
  --conf spark.rapids.sql.format.parquet.reader.type=MULTITHREADED \
  --conf spark.rapids.sql.explain="${RAPIDS_EXPLAIN:-NONE}" \
  --conf spark.hadoop.fs.s3a.path.style.access="${S3A_PATH_STYLE_ACCESS:-true}" \
  --conf spark.hadoop.fs.s3a.connection.establish.timeout=5000 \
  --conf spark.hadoop.fs.s3a.connection.timeout=10000 \
  --conf spark.hadoop.fs.s3a.attempts.maximum=3 \
  --conf spark.hadoop.fs.s3a.retry.limit=3 \
  ${SPARK_EXTRA_CONF:-} \
  spark_jobs/build_graph.py "$@"
