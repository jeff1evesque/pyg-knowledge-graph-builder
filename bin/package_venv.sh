#!/usr/bin/env bash
#
# Package the virtualenv so Spark executors can run this job's Python code.
#
# Why this is needed: `--py-files` ships *our* modules, not their third-party
# dependencies. On a cluster, spark-submit launches the driver and executors with
# whatever Python it finds -- typically the system one, which has no rdflib, no
# torch, and no pyspark extras. The job then dies with `ModuleNotFoundError` on a
# worker, far from anything that looks like the cause.
#
# Provisioning those packages onto every node would make the cluster carry this
# application's dependency set, and every version bump becomes a re-provision. So
# instead we ship the environment with the job: pack it here, and
# bin/submit_spark_job.sh passes it via --archives and points the executors'
# interpreter at it.
#
# The archive is cached in dist/ and only rebuilt when the requirements change.
#
# Usage:
#   bin/package_venv.sh                 # build dist/pyspark_venv.tar.gz if stale
#   bin/package_venv.sh --force         # always rebuild
#
# Environment:
#   VENV          virtualenv to pack (default: .venv)
#   VENV_ARCHIVE  output path (default: dist/pyspark_venv.tar.gz)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
VENV_ARCHIVE="${VENV_ARCHIVE:-dist/pyspark_venv.tar.gz}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

[[ -x "${VENV}/bin/python" ]] || {
  echo "no virtualenv at ${VENV} -- create it first (python -m venv ${VENV})" >&2
  exit 1
}

mkdir -p "$(dirname "$VENV_ARCHIVE")"

# Rebuild only if the archive is older than any requirements file.
if [[ "$FORCE" -eq 0 && -f "$VENV_ARCHIVE" ]]; then
  stale=0
  for req in requirements.txt requirements-test.txt; do
    [[ -f "$req" && "$req" -nt "$VENV_ARCHIVE" ]] && stale=1
  done
  if [[ "$stale" -eq 0 ]]; then
    echo "==> ${VENV_ARCHIVE} is up to date (use --force to rebuild)"
    exit 0
  fi
fi

if ! "${VENV}/bin/python" -c "import venv_pack" 2>/dev/null; then
  echo "==> installing venv-pack into ${VENV}"
  "${VENV}/bin/pip" install --quiet venv-pack
fi

echo "==> packing ${VENV} -> ${VENV_ARCHIVE} (this is large; torch dominates)"
rm -f "$VENV_ARCHIVE"
# --prefix is required: without it venv-pack tries to pack the *running* interpreter's
# environment, which fails ("Current environment is not a virtual environment").
"${VENV}/bin/python" -m venv_pack --prefix "$VENV" --output "$VENV_ARCHIVE" --force

echo "==> done: $(du -h "$VENV_ARCHIVE" | cut -f1) at ${VENV_ARCHIVE}"
