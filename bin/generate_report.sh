#!/usr/bin/env bash
#
# Run the test suites LOCALLY and (over)write reports/report.{html,json}.
#
# This is a local-only convenience: the end-to-end smoke suite is too heavy for
# the GitHub Actions runners (see README "Testing"), so its results — and the
# combined report — are produced on a developer machine, never in CI. The fast
# unit suite IS covered by CI (the tests badge); it is included here only so the
# report is a complete local picture.
#
# The report always overwrites the previous pair at a fixed path, so `reports/`
# holds exactly one HTML + one JSON. JUnit XML is written to a temp dir and
# discarded.
#
# Suites included:
#   1. Fast unit suite      (marker: not e2e)          — CPU
#   2. E2E pipeline smoke    (marker: e2e)              — CPU
#   3. E2E pipeline smoke    (marker: e2e, RAPIDS)      — GPU, only if a GPU and
#                                                         RAPIDS jar are present
#   4. Cluster submit        (marker: cluster)          — submits the REAL job to a
#                                                         standalone cluster; only if
#                                                         SPARK_MASTER_URL et al. are set
#
# Suites 1-3 all run in local[*], so none of them can see the cluster: not GPU
# resource scheduling, not the packaged venv the executors need, not object storage.
# Suite 4 is the only one that exercises the job the way production runs it.
#
# Usage:
#   bin/generate_report.sh              # fast + e2e(CPU) [+ e2e(GPU), + cluster if configured]
#   bin/generate_report.sh --no-gpu     # never attempt the GPU run
#
# Environment variables:
#   VENV_PYTHON    python to use; default .venv/bin/python
#   DRIVER_MEMORY  driver heap for the e2e runs; default 4g
#   PYTEST_WORKERS xdist workers for the FAST suite only; unset = sequential.
#                  Each worker builds its own SparkSession, so this trades memory
#                  for wall-clock -- a local-machine optimization, off by default
#                  so CI is unaffected. Try 4; do not exceed physical cores.
#   TWIN_RUNS_SEQUENTIAL=1
#                  Run the e2e reproducibility twin pipelines one-at-a-time
#                  instead of overlapped. Halves their peak memory (one 2g driver
#                  JVM instead of two) at roughly double their wall-clock; for
#                  memory-constrained hosts.
#   RAPIDS_JAR     explicit RAPIDS jar (else newest under RAPIDS_JAR_DIR)
#   RAPIDS_JAR_DIR dir to search for the jar; default /opt/spark/jars
#
#   SPARK_MASTER_URL           standalone master, e.g. spark://<host>:7077
#   CLUSTER_SMOKE_SOURCE_PATH  source data readable by EVERY node (executors are
#                              spread across machines, so a driver-local path fails)
#   CLUSTER_SMOKE_OUTPUT_PATH  shared output base, e.g. s3a://<bucket>/<prefix>
#                              (the run is written to a date-partitioned subpath)
#
# Exit status is non-zero if any suite failed (the report is still written).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
DRIVER_MEMORY="${DRIVER_MEMORY:-4g}"
REPORT_DIR="$REPO_ROOT/reports/tests"
WANT_GPU=1
[[ "${1:-}" == "--no-gpu" ]] && WANT_GPU=0

JUNIT_DIR="$(mktemp -d)"
trap 'rm -rf "$JUNIT_DIR"' EXIT

export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export PYSPARK_SUBMIT_ARGS="--driver-memory ${DRIVER_MEMORY} pyspark-shell"

# Isolate the local[*] suites from any cluster configuration.
#
# If SPARK_HOME points at a cluster's Spark install, the local suites read its
# spark-defaults.conf and inherit spark.master (pointing at the cluster) AND its GPU
# task requirement. Local mode has no worker to advertise a GPU, so the request can
# never be satisfied and the suite HANGS FOREVER with no error -- it does not fail,
# it just sits there. The cluster suite re-supplies SPARK_HOME for itself below,
# since spark-submit genuinely needs it.
CLUSTER_SPARK_HOME="${SPARK_HOME:-/opt/spark}"
unset SPARK_HOME

status=0
declare -a SPECS

echo "==> Fast unit suite (CPU)${PYTEST_WORKERS:+ [${PYTEST_WORKERS} workers]}"
# Parallelism is opt-in and local-only. Each xdist worker is its own process and
# therefore builds its OWN SparkSession (tests/conftest.py's `spark` fixture is
# session-scoped), so N workers means N local-mode JVMs. That trades memory for
# wall-clock: fine on a developer machine, not on a small CI runner -- which is
# why CI (.github/workflows/tests.yml) calls pytest directly and stays
# sequential. Deliberately NOT `-n auto`: that keys off core count, so the same
# command would behave completely differently across machines.
"$VENV_PYTHON" -m pytest tests/ -m "not e2e" \
  ${PYTEST_WORKERS:+-n "$PYTEST_WORKERS"} \
  --junitxml="$JUNIT_DIR/fast.xml" -q || status=1
SPECS+=("Fast (unit) suite=CPU=$JUNIT_DIR/fast.xml")

echo "==> End-to-end pipeline smoke (CPU)"
# "not cluster": the cluster suite is also marked e2e, but it submits to a real
# cluster and must not run inside the local[*] suites.
"$VENV_PYTHON" -m pytest tests/ -m "e2e and not cluster" \
  --junitxml="$JUNIT_DIR/e2e_cpu.xml" -q || status=1
SPECS+=("End-to-end pipeline smoke=CPU=$JUNIT_DIR/e2e_cpu.xml")

# Optional GPU run — only if a GPU and a RAPIDS jar are both present.
if [[ "$WANT_GPU" == "1" ]]; then
  if [[ -z "${RAPIDS_JAR:-}" ]]; then
    RAPIDS_JAR_DIR="${RAPIDS_JAR_DIR:-/opt/spark/jars}"
    RAPIDS_JAR="$(ls -t "$RAPIDS_JAR_DIR"/rapids-4-spark_*.jar 2>/dev/null | head -n1 || true)"
  fi
  if command -v nvidia-smi >/dev/null 2>&1 \
     && [[ -n "${RAPIDS_JAR:-}" && -f "${RAPIDS_JAR:-}" ]]; then
    echo "==> End-to-end pipeline smoke (GPU via RAPIDS: $RAPIDS_JAR)"
    SPARK_RAPIDS=1 RAPIDS_JAR="$RAPIDS_JAR" \
      "$VENV_PYTHON" -m pytest tests/ -m "e2e and not cluster" \
      --junitxml="$JUNIT_DIR/e2e_gpu.xml" -q || status=1
    SPECS+=("End-to-end pipeline smoke=GPU (RAPIDS Accelerator)=$JUNIT_DIR/e2e_gpu.xml")
  else
    echo "==> Skipping GPU run (no GPU and/or RAPIDS jar found)"
  fi
fi

# Optional cluster run — only if a standalone cluster and an output location are
# configured. This is the only suite that submits the job the way production does
# (spark-submit to a master, code + venv shipped to executors, GPU scheduling
# negotiated with the workers); everything above runs in local[*] and cannot see
# any of that.
# Gate matches the cluster test's own requirements (tests/e2e/test_cluster_submit.py):
# it needs a master and a shared-storage output location; the source is OPTIONAL --
# when CLUSTER_SMOKE_SOURCE_PATH is unset the test auto-stages the repo's fixtures to
# an s3a:// output base. Requiring it here would wrongly skip the suite for the exact
# self-contained run the test supports.
if [[ -n "${SPARK_MASTER_URL:-}" && -n "${CLUSTER_SMOKE_OUTPUT_PATH:-}" ]]; then
  echo "==> Cluster submit (real job -> ${SPARK_MASTER_URL})"
  # This is a smoke test, not a benchmark. The pipeline fans out into ~1,300 stages
  # regardless of data size, and Spark's default of 200 shuffle partitions means each
  # of those stages schedules 200 tasks over a few KB of fixtures -- pure scheduling
  # overhead. Shrinking the partition count is what makes this finish in minutes;
  # shrinking the *data* would not, because the stage count does not depend on it.
  SPARK_HOME="$CLUSTER_SPARK_HOME" \
  SPARK_EXTRA_CONF="${SPARK_EXTRA_CONF:-} --conf spark.sql.shuffle.partitions=${CLUSTER_SMOKE_SHUFFLE_PARTITIONS:-8}" \
    "$VENV_PYTHON" -m pytest tests/ -m cluster \
    --junitxml="$JUNIT_DIR/cluster.xml" -q || status=1
  SPECS+=("Cluster submit=Standalone cluster (GPU)=$JUNIT_DIR/cluster.xml")
else
  echo "==> Skipping cluster run (set SPARK_MASTER_URL and CLUSTER_SMOKE_OUTPUT_PATH"
  echo "    to submit the real job to a standalone cluster; CLUSTER_SMOKE_SOURCE_PATH"
  echo "    is optional -- fixtures auto-stage to the output base when it is unset)"
fi

echo "==> Rendering report -> $REPORT_DIR/report.{html,json}"
"$VENV_PYTHON" bin/generate_test_report.py "$REPORT_DIR" "${SPECS[@]}"

exit "$status"
