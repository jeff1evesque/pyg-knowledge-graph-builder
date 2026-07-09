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
#
# Usage:
#   bin/generate_report.sh              # fast + e2e(CPU) [+ e2e(GPU) if available]
#   bin/generate_report.sh --no-gpu     # never attempt the GPU run
#
# Environment variables:
#   VENV_PYTHON    python to use; default .venv/bin/python
#   DRIVER_MEMORY  driver heap for the e2e runs; default 4g
#   RAPIDS_JAR     explicit RAPIDS jar (else newest under RAPIDS_JAR_DIR)
#   RAPIDS_JAR_DIR dir to search for the jar; default /opt/spark/jars
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

status=0
declare -a SPECS

echo "==> Fast unit suite (CPU)"
"$VENV_PYTHON" -m pytest tests/ -m "not e2e" \
  --junitxml="$JUNIT_DIR/fast.xml" -q || status=1
SPECS+=("Fast (unit) suite=CPU=$JUNIT_DIR/fast.xml")

echo "==> End-to-end pipeline smoke (CPU)"
"$VENV_PYTHON" -m pytest tests/ -m e2e \
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
      "$VENV_PYTHON" -m pytest tests/ -m e2e \
      --junitxml="$JUNIT_DIR/e2e_gpu.xml" -q || status=1
    SPECS+=("End-to-end pipeline smoke=GPU (RAPIDS Accelerator)=$JUNIT_DIR/e2e_gpu.xml")
  else
    echo "==> Skipping GPU run (no GPU and/or RAPIDS jar found)"
  fi
fi

echo "==> Rendering report -> $REPORT_DIR/report.{html,json}"
"$VENV_PYTHON" bin/generate_test_report.py "$REPORT_DIR" "${SPECS[@]}"

exit "$status"
