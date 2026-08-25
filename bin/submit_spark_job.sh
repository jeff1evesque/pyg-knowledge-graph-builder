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
#       --local_work_dir /data \
#       --time_period 2024-12
#
# Environment variables:
#   SPARK_MASTER_URL            (required) e.g. spark://<host>:7077
#   SPARK_DRIVER_HOST           (optional) address the EXECUTORS dial back on; see note
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
#   DRIVER_MEMORY               (optional, default 4g) driver heap. The pipeline fans
#                               out into ~1,300 stages and OOMs Spark's 1g default.
#   EXECUTOR_MEMORY             (optional, default 4g; cluster masters only) heap per
#                               executor. The default is a FLOOR chosen to be safe on
#                               any worker, not a size for real input -- Spark waits
#                               forever on a request larger than a worker can offer,
#                               so this errs small deliberately. Set it to what the
#                               cluster actually has, e.g. 64g.
#   EXECUTOR_CORES              (optional, unset; cluster masters only) cores per
#                               executor. Unset means each executor takes every core
#                               on its worker, which is usually what this job wants.
#   RAPIDS_GPU_ALLOC_FRACTION   (optional, default 0.25) fraction of FREE GPU memory
#                               the RMM pool takes. Must not exceed the max below;
#                               the script refuses to submit if it does. The default is
#                               a floor for unified-memory GPUs; see the note below.
#   RAPIDS_GPU_MAX_ALLOC_FRACTION
#                               (optional, default 0.4) hard cap on the pool as a
#                               fraction of TOTAL GPU memory. Raising the alloc
#                               fraction alone cannot grow the pool past this.
#   NETWORK_TIMEOUT             (optional, default 120s) how long the driver waits on an
#                               unresponsive executor before evicting it. See note below.
#   EXECUTOR_HEARTBEAT_INTERVAL (optional, default 10s) must stay well under the above.
#   MAX_PLAN_STRING_LENGTH      (optional, default 16k) see note below
#   PARQUET_NANOS_AS_LONG       (optional, default true) see note below
#   SPARK_EXTRA_CONF            (optional) extra "--conf k=v" flags, space-separated
#
# For a large run, source a profile before setting the cluster's own variables:
#   . bin/profiles/large-run.env
# It sets sizing only -- no addresses, buckets or paths.
#
# SPARK_DRIVER_HOST: in client mode the driver picks its own advertised address from
# the host's first non-loopback interface, and every executor dials back on it. On a
# node with more than one network -- e.g. a management LAN plus a dedicated cluster
# fabric -- that guess can land on the wrong one. The failure is quiet and expensive:
# executors on OTHER hosts cannot reach the driver, time out after 120s, exit 1, and
# are relaunched forever, while the executor that happens to be co-located with the
# driver connects over loopback and runs the whole job. The cluster looks healthy, the
# job succeeds, and every task lands on one node. Set this to the driver's address on
# the same network as SPARK_MASTER_URL.
#
# nanosAsLong: source Parquet written by pandas/pyarrow carries nanosecond timestamps
# (INT64 TIMESTAMP(NANOS)), which Spark's Parquet reader rejects outright --
# "Illegal Parquet type: INT64 (TIMESTAMP(NANOS,true))" -- and it rejects them during
# schema conversion, so the job dies on the very first read even though it goes on to
# select nothing but the Turtle blob column. Reading those columns as raw longs costs
# this pipeline nothing (it never looks at them) and is the difference between a source
# being readable and not. Set PARQUET_NANOS_AS_LONG=false to restore Spark's default.
#
# maxPlanStringLength: Spark renders the physical plan to a STRING for the SQL UI on
# every execution (SQLExecution.withNewExecutionId -> explainString), and that string
# defaults to spark.sql.maxPlanStringLength = 2147483632b (~2GB) -- effectively
# unbounded. The enrichment plan over several unioned sources is big enough that the
# driver OOMs inside StringBuilder.<init> while *describing* the query, before running
# it, and no amount of --driver-memory helps. A 7-source run OOMs without this cap and
# passes with it (at the stock 4g heap). Cost: plans shown in the SQL UI and by
# explain() are truncated past this length.
#
# NETWORK_TIMEOUT: Spark evicts an executor that misses heartbeats for 120s, and a
# localCheckpoint'd RDD has no replica -- so evicting one executor destroys the
# checkpoint blocks it held, and any job still reading them aborts with "Checkpoint
# block rdd_N not found". A busy executor can go quiet for longer than 120s without
# being dead: measured, one stalled for 173s inside block eviction while spilling to
# disk under GPU and heap pressure, was evicted, and took a 2.5-hour four-source run
# with it. The JVM was alive the whole time and logged its own SIGTERM. Raising this
# lets a large run ride out a stall that a small one never produces. It is a real
# trade: a genuinely dead executor also goes undetected for that much longer, which is
# why the default stays at Spark's own value and the large-run profile raises it.
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

# GPU RESOURCE SCHEDULING IS A CLUSTER CONCERN, and requesting it in local mode hangs
# the job with no error.
#
# These three tell the MASTER which workers own a GPU so it can place executors on
# them. local[*] has no master and no separate executor: the driver is the executor,
# and it advertises no custom resources ("No custom resources configured for
# spark.driver"). spark.task.resource.gpu.amount then makes every task require a GPU
# that nothing offers, so the task set is submitted and never scheduled -- no error, no
# timeout, no progress. Measured: a local run sat on task set 0.0 for 10 minutes and
# would have sat there forever; with these three omitted the same run finished in 59s.
#
# Dropping them costs no GPU acceleration. RAPIDS talks to CUDA directly and is
# unaffected by Spark's resource scheduler -- the same local run still placed 63,002
# operators on the GPU.
#
# Local mode OVERRIDES them to 0 rather than omitting them, and the difference matters:
# a site spark-defaults.conf commonly sets spark.task.resource.gpu.amount for the
# cluster's benefit, and spark-submit reads that file whether or not this script names
# the same key. Omitting the flag therefore leaves the requirement in force and the job
# still hangs -- measured on a host whose spark-defaults.conf sets it to 1.
gpu_args=(
  --conf spark.executor.resource.gpu.amount=0
  --conf spark.task.resource.gpu.amount=0
)
if [[ "$SPARK_MASTER_URL" != local* ]]; then
  gpu_args=(
    --conf spark.executor.resource.gpu.amount="${GPU_PER_EXECUTOR:-1}"
    --conf spark.task.resource.gpu.amount="${GPU_PER_TASK:-0.125}"
    --conf spark.executor.resource.gpu.discoveryScript="$GPU_DISCOVERY_SCRIPT"
  )
fi

# EXECUTOR SIZING. Nothing set spark.executor.memory -- not this script, not the site
# spark-defaults.conf -- so every executor took Spark's 1g default while the workers
# advertised two orders of magnitude more. That survived because the cluster suite runs
# on committed fixtures measured in kilobytes, where 1g is ample. Real input is not:
# one day across four sources parses to ~19.8M triples and the market linker alone adds
# ~2.4M. The failure mode is the expensive kind -- not a crash, but constant spilling
# and a job that looks like it works while running many times slower than it should.
#
# The default is deliberately modest rather than sized to the workers. Spark treats an
# executor-memory request larger than a worker can offer as UNSATISFIABLE, and its
# response to an unsatisfiable request is to wait, not to fail -- the same hang class
# this script already guards against for GPU resources. A default that assumed large
# workers would turn a slow job into a silent one on a smaller cluster, which is worse.
# So: 4g floor, and a warning that names the variable, on the principle that the person
# running a real job is better placed to size it than a constant in here.
#
# Local mode has no executor process (the driver is the executor), so --driver-memory
# already covers it and the flag is omitted rather than set to something misleading.
executor_args=()
if [[ "$SPARK_MASTER_URL" != local* ]]; then
  executor_args=(--executor-memory "${EXECUTOR_MEMORY:-4g}")
  if [[ -z "${EXECUTOR_MEMORY:-}" ]]; then
    echo "WARNING: EXECUTOR_MEMORY is unset; defaulting each executor to 4g." >&2
    echo "         That is a floor, not a size for real input. Set it to what the" >&2
    echo "         cluster's workers can actually offer, e.g. EXECUTOR_MEMORY=64g." >&2
  fi
  # Left unset by default: standalone mode gives an executor every core on its worker,
  # which is what this job wants. Named here so choosing otherwise is a decision rather
  # than an accident, and because it interacts with GPU_PER_TASK -- task slots are
  # min(cores/task.cpus, gpu/task.gpu), so cores beyond that ratio sit idle.
  if [[ -n "${EXECUTOR_CORES:-}" ]]; then
    executor_args+=(--executor-cores "$EXECUTOR_CORES")
  fi
fi

# THE GPU POOL FRACTIONS MUST BE ORDERED, and RAPIDS enforces it after the JVM is up:
# it computes pool = (gpu.free - reserve) * allocFraction, compares against
# cap = gpu.total * maxAllocFraction, and refuses to start when pool exceeds cap. That
# rejection arrives as an executor-plugin shutdown inside a py4j stack trace, after the
# phase banner and before any progress line, so from the job's own log it reads as a
# startup hang rather than a bad setting. Measured: ~25 minutes lost to allocFraction=0.5
# against a site maxAllocFraction of 0.4.
#
# Checking it here costs nothing and fails in one line before any work is scheduled.
#
# The comparison is allocFraction > maxAllocFraction, which is STRICTER than what RAPIDS
# itself rejects: because allocFraction scales gpu.FREE while the cap is a fraction of
# gpu.TOTAL, a larger allocFraction is legal whenever enough memory is already in use.
# That is not a property to depend on -- it means the same config starts on a busy GPU
# and fails on an idle one. Requiring the ordering makes the setting deterministic; to
# genuinely want a bigger pool, raise both.
#
# THE 0.25/0.4 DEFAULT IS A HARDWARE FLOOR, NOT AN UNTUNED VALUE -- see README section 3
# and conf/spark-rapids.conf.template. On an integrated / unified-memory GPU the "GPU
# memory" IS the host's RAM, so a large pool starves the OS and the JVM and drives the
# machine into swap. Do not raise it globally to fix one big run.
#
# It IS too small for a large run, and the symptom is worth recognising: task slots per
# executor are min(cores/task.cpus, gpu/task.gpu), which at the GPU_PER_TASK default of
# 0.125 is 8 tasks sharing one RMM pool whatever the core count. Measured on a
# four-source day whose market source alone parses to 345M triples: 215,144
# "[RMM] [error] maximum pool size exceeded" lines from a single executor, rising hour
# over hour (1.9k -> 72k -> 122k) until it stalled and was evicted, which destroyed the
# localCheckpoint blocks it held and aborted the job. Raise both fractions for that run
# via bin/profiles/large-run.env -- opted into per run, where the host-RAM trade can be
# weighed against what else is resident -- rather than changing the floor here.
RAPIDS_GPU_ALLOC_FRACTION="${RAPIDS_GPU_ALLOC_FRACTION:-0.25}"
RAPIDS_GPU_MAX_ALLOC_FRACTION="${RAPIDS_GPU_MAX_ALLOC_FRACTION:-0.4}"
if awk "BEGIN{exit !($RAPIDS_GPU_ALLOC_FRACTION > $RAPIDS_GPU_MAX_ALLOC_FRACTION)}"; then
  echo "ERROR: RAPIDS_GPU_ALLOC_FRACTION=${RAPIDS_GPU_ALLOC_FRACTION} exceeds" >&2
  echo "       RAPIDS_GPU_MAX_ALLOC_FRACTION=${RAPIDS_GPU_MAX_ALLOC_FRACTION}." >&2
  echo "       RAPIDS caps the pool at gpu.total * maxAllocFraction, so this either" >&2
  echo "       fails at startup or succeeds only while the GPU is already partly in" >&2
  echo "       use. Lower the alloc fraction, or raise both to widen the cap." >&2
  exit 2
fi

"$SPARK_SUBMIT" \
  "${executor_args[@]}" \
  "${venv_args[@]}" \
  "${gpu_args[@]}" \
  --master "$SPARK_MASTER_URL" \
  --driver-memory "${DRIVER_MEMORY:-4g}" \
  ${SPARK_DRIVER_HOST:+--conf spark.driver.host="$SPARK_DRIVER_HOST"} \
  ${RAPIDS_JAR:+--jars "$RAPIDS_JAR"} \
  --py-files "$PKG_ZIP" \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.rapids.sql.concurrentGpuTasks="${RAPIDS_CONCURRENT_GPU_TASKS:-2}" \
  --conf spark.rapids.memory.pinnedPool.size="${RAPIDS_PINNED_POOL:-2G}" \
  --conf spark.rapids.memory.gpu.allocFraction="${RAPIDS_GPU_ALLOC_FRACTION}" \
  --conf spark.rapids.memory.gpu.minAllocFraction="${RAPIDS_GPU_MIN_ALLOC_FRACTION:-0}" \
  --conf spark.rapids.memory.gpu.maxAllocFraction="${RAPIDS_GPU_MAX_ALLOC_FRACTION}" \
  --conf spark.rapids.sql.format.parquet.reader.type=MULTITHREADED \
  --conf spark.rapids.sql.explain="${RAPIDS_EXPLAIN:-NONE}" \
  --conf spark.network.timeout="${NETWORK_TIMEOUT:-120s}" \
  --conf spark.executor.heartbeatInterval="${EXECUTOR_HEARTBEAT_INTERVAL:-10s}" \
  --conf spark.sql.maxPlanStringLength="${MAX_PLAN_STRING_LENGTH:-16k}" \
  --conf spark.sql.legacy.parquet.nanosAsLong="${PARQUET_NANOS_AS_LONG:-true}" \
  --conf spark.hadoop.fs.s3a.path.style.access="${S3A_PATH_STYLE_ACCESS:-true}" \
  --conf spark.hadoop.fs.s3a.connection.establish.timeout=5000 \
  --conf spark.hadoop.fs.s3a.connection.timeout=10000 \
  --conf spark.hadoop.fs.s3a.attempts.maximum=3 \
  --conf spark.hadoop.fs.s3a.retry.limit=3 \
  ${SPARK_EXTRA_CONF:-} \
  spark_jobs/build_graph.py "$@"
