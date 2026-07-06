#!/usr/bin/env bash
#
# Run the end-to-end pipeline smoke test (the `e2e` marker) against a local,
# in-process SparkSession. No Spark cluster is used — the test spins up its own
# `local[*]` Spark, so the standalone cluster does NOT need to be running.
#
# Usage:
#   bin/run_e2e_tests.sh            # CPU (default)
#   bin/run_e2e_tests.sh gpu        # GPU via the RAPIDS Accelerator
#
# Environment variables:
#   VENV_PYTHON   (optional) python to use; default .venv/bin/python
#   RAPIDS_JAR    (optional) path to the RAPIDS Accelerator jar. If unset in
#                 gpu mode, the newest jar under $RAPIDS_JAR_DIR is used.
#   RAPIDS_JAR_DIR(optional) dir to search for the jar; default /opt/spark/jars
#   DRIVER_MEMORY (optional) driver heap; default 4g (the full pipeline fans out
#                 into ~1,300 stages and OOMs the default heap).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-cpu}"
VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
DRIVER_MEMORY="${DRIVER_MEMORY:-4g}"

export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export PYSPARK_SUBMIT_ARGS="--driver-memory ${DRIVER_MEMORY} pyspark-shell"

case "$MODE" in
  cpu)
    ;;
  gpu)
    if [[ -z "${RAPIDS_JAR:-}" ]]; then
      RAPIDS_JAR_DIR="${RAPIDS_JAR_DIR:-/opt/spark/jars}"
      # Newest matching jar, so a version bump on the node needs no edit here.
      RAPIDS_JAR="$(ls -t "$RAPIDS_JAR_DIR"/rapids-4-spark_*.jar 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "${RAPIDS_JAR:-}" || ! -f "$RAPIDS_JAR" ]]; then
      echo "error: RAPIDS jar not found (looked in ${RAPIDS_JAR_DIR:-/opt/spark/jars})." >&2
      echo "       Set RAPIDS_JAR=/path/to/rapids-4-spark_*.jar and retry." >&2
      exit 1
    fi
    export SPARK_RAPIDS=1
    export RAPIDS_JAR
    echo "GPU mode — RAPIDS_JAR=$RAPIDS_JAR"
    ;;
  *)
    echo "usage: bin/run_e2e_tests.sh [cpu|gpu]" >&2
    exit 2
    ;;
esac

exec "$VENV_PYTHON" -m pytest tests/ -m e2e -s
