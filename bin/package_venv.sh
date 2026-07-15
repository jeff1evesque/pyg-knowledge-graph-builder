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

REQUIREMENTS="${EXECUTOR_REQUIREMENTS:-requirements-executor.txt}"
BUILD_VENV="${BUILD_VENV:-dist/executor_venv}"
VENV_ARCHIVE="${VENV_ARCHIVE:-dist/pyspark_venv.tar.gz}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

[[ -f "$REQUIREMENTS" ]] || { echo "missing ${REQUIREMENTS}" >&2; exit 1; }

mkdir -p "$(dirname "$VENV_ARCHIVE")"

if [[ "$FORCE" -eq 0 && -f "$VENV_ARCHIVE" && ! "$REQUIREMENTS" -nt "$VENV_ARCHIVE" ]]; then
  echo "==> ${VENV_ARCHIVE} is up to date (use --force to rebuild)"
  exit 0
fi

# Build a dedicated EXECUTOR environment rather than packing the developer venv.
#
# The dev venv is ~3.2 GB, almost all of it driver-only (nvidia 2.9 G, torch 0.9 G,
# triton 0.65 G). Shipping that to every executor is not merely wasteful: on a
# unified-memory GPU, unpacking it floods the page cache, host free memory collapses,
# and RAPIDS then computes a GPU pool below its own floor and kills the executor --
# which Spark relaunches, which unpacks it again. The job spirals instead of running.
echo "==> building executor environment from ${REQUIREMENTS}"
rm -rf "$BUILD_VENV"
python3 -m venv "$BUILD_VENV"
"${BUILD_VENV}/bin/pip" install --quiet --upgrade pip
"${BUILD_VENV}/bin/pip" install --quiet -r "$REQUIREMENTS"
"${BUILD_VENV}/bin/pip" install --quiet venv-pack

echo "==> packing ${BUILD_VENV} -> ${VENV_ARCHIVE}"
rm -f "$VENV_ARCHIVE"
# --prefix is required: without it venv-pack tries to pack the *running* interpreter's
# environment, which fails ("Current environment is not a virtual environment").
"${BUILD_VENV}/bin/python" -m venv_pack --prefix "$BUILD_VENV" --output "$VENV_ARCHIVE" --force

echo "==> done: $(du -h "$VENV_ARCHIVE" | cut -f1) at ${VENV_ARCHIVE}"
