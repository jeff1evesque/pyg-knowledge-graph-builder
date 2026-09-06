#!/usr/bin/env bash
#
# Run the fast unit suite — everything except the `e2e` marker, in parallel,
# against local in-process SparkSessions. No Spark cluster is used or needed.
#
# This is the sibling of bin/run_e2e_tests.sh, which runs the heavy `e2e` half.
# It exists because the correct invocation is not `pytest tests/`, and the two
# differences each cost real time:
#
#   -m "not e2e"   what CI runs (.github/workflows/tests.yml). The e2e tests are
#                  the full-pipeline smoke and belong to run_e2e_tests.sh.
#                  Including them turns ~90 seconds into 10+ minutes.
#   -n 8           pytest-xdist, declared in requirements-test.txt. Measured on
#                  this 20-core box: ~248s serial -> ~90s at 8 workers. Returns
#                  flatten quickly because each worker builds its own
#                  SparkSession, which is a fixed cost -- so this is a number,
#                  deliberately, and NOT `-n auto`, which silently scales with
#                  whatever machine it lands on.
#
# It also unsets SPARK_HOME and points SPARK_CONF_DIR at an empty directory. A
# shell that has sourced a cluster env otherwise leaks the standalone conf into
# local mode, which asks for a per-task GPU that local mode cannot satisfy: the
# session comes up, reports 0 active tasks, and hangs with no error.
#
# Usage:
#   bin/run_tests.sh                      # the whole fast suite
#   bin/run_tests.sh tests/test_foo.py    # any pytest args are passed through
#
# Environment variables:
#   VENV_PYTHON     (optional) python to use; default .venv/bin/python
#   PYTEST_WORKERS  (optional) xdist workers; default 8
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
PYTEST_WORKERS="${PYTEST_WORKERS:-8}"

# Reused rather than mktemp'd so repeated runs do not leave a trail of empty
# directories behind them.
EMPTY_CONF="${TMPDIR:-/tmp}/pyg-empty-spark-conf"
mkdir -p "$EMPTY_CONF"

export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export SPARK_CONF_DIR="$EMPTY_CONF"
unset SPARK_HOME

# Default to the whole suite; anything on the command line replaces that, so a
# single file or a -k expression works without repeating the flags above.
if [[ $# -gt 0 ]]; then
  exec "$VENV_PYTHON" -m pytest -m "not e2e" -n "$PYTEST_WORKERS" "$@"
fi

exec "$VENV_PYTHON" -m pytest tests/ -q -m "not e2e" -n "$PYTEST_WORKERS"
