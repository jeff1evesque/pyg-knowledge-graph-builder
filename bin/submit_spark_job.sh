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

# Prefer $SPARK_HOME/bin/spark-submit over whatever is on PATH: a non-interactive
# shell (cron, CI, a test harness) frequently has no Spark on PATH at all, and the
# failure -- "spark-submit: command not found" -- looks nothing like a Spark problem.
SPARK_SUBMIT="spark-submit"
if [[ -x "${SPARK_HOME:-}/bin/spark-submit" ]]; then
  SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"
elif ! command -v spark-submit >/dev/null 2>&1; then
  echo "spark-submit not found on PATH; set SPARK_HOME (e.g. SPARK_HOME=/opt/spark)" >&2
  exit 1
fi

# Ship the Python environment to the executors.
#
# --py-files carries our modules but NOT their dependencies: on a cluster the
# executors run whatever Python spark-submit finds (usually the system one, with no
# rdflib/torch), and the job dies with ModuleNotFoundError on a worker. Packing the
# venv (bin/package_venv.sh) and unpacking it on each executor keeps this repo's
# dependencies this repo's problem, rather than something every node must be
# provisioned with.
#
# Skipped automatically for local[*] runs, which already use the current interpreter.
VENV_ARCHIVE="${VENV_ARCHIVE:-dist/pyspark_venv.tar.gz}"
VENV="${VENV:-.venv}"
venv_args=()
if [[ "$SPARK_MASTER_URL" != local* ]]; then
  if [[ -f "$VENV_ARCHIVE" ]]; then
    # The archive is unpacked in each EXECUTOR's working directory, so only they can
    # see ./environment. In client mode the driver runs here, with no archive unpacked
    # -- point it at the local venv, or it dies with
    # "Cannot run program ./environment/bin/python: No such file or directory".
    venv_args=(
      --archives "${VENV_ARCHIVE}#environment"
      --conf spark.pyspark.python=./environment/bin/python
      --conf "spark.pyspark.driver.python=${VENV}/bin/python"
    )
  else
    echo "WARNING: ${VENV_ARCHIVE} not found. Executors will use the system Python," >&2
    echo "         which almost certainly lacks this job's dependencies. Build it with:" >&2
    echo "           bin/package_venv.sh" >&2
  fi
fi

"$SPARK_SUBMIT" \
  "${venv_args[@]}" \
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
  --conf spark.rapids.memory.gpu.minAllocFraction="${RAPIDS_GPU_MIN_ALLOC_FRACTION:-0.05}" \
  --conf spark.rapids.sql.format.parquet.reader.type=MULTITHREADED \
  --conf spark.rapids.sql.explain="${RAPIDS_EXPLAIN:-NONE}" \
  --conf spark.hadoop.fs.s3a.path.style.access="${S3A_PATH_STYLE_ACCESS:-true}" \
  --conf spark.hadoop.fs.s3a.connection.establish.timeout=5000 \
  --conf spark.hadoop.fs.s3a.connection.timeout=10000 \
  --conf spark.hadoop.fs.s3a.attempts.maximum=3 \
  --conf spark.hadoop.fs.s3a.retry.limit=3 \
  ${SPARK_EXTRA_CONF:-} \
  spark_jobs/build_graph.py "$@"
