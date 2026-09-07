#!/usr/bin/env bash
# Poll the Spark master and the live driver UI, appending one JSON object per sample
# to <run-dir>/samples.jsonl. Runs detached alongside a run and exits when told to.
#
#   bin/cluster_sampler.sh <run-dir> [stop-file]
#
# WHY THIS EXISTS: the driver UI belongs to the driver process and vanishes the
# instant it exits, including on the failure path. Event logs cover what Spark
# records, but not "how busy was the cluster at 23:14", and not the host's own memory
# pressure -- which on a unified-memory box is what sizes the RMM pool and what has
# killed runs outright.
#
# Environment:
#   PYG_SPARK_MASTER_UI   master web UI. Default: the SPARK_MASTER_URL host on :8080.
#   PYG_SPARK_DRIVER_UI   driver web UI. Default: the SPARK_DRIVER_HOST on :4040.
#   PYG_PYTHON            interpreter. Default: the repo venv, else python3.
#   PYG_SAMPLE_SECONDS    seconds between samples (default 30).
#
# Nothing here names a host: the two addresses come from the same variables the job
# itself is configured with.
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <run-dir> [stop-file]" >&2
  exit 2
fi

RD="${1%/}"
# The stop file is per-run for the same reason netsample.py takes one: a shared flag
# in the run directory let a killed attempt's cleanup stop the live run's samplers.
STOP="${2:-$RD/STOP}"
OUT="$RD/samples.jsonl"
INTERVAL="${PYG_SAMPLE_SECONDS:-30}"

_host() {                      # spark://host:7077 -> host
  local u="${1#*://}"
  echo "${u%%:*}"
}

MASTER_UI="${PYG_SPARK_MASTER_UI:-}"
if [[ -z "$MASTER_UI" && -n "${SPARK_MASTER_URL:-}" ]]; then
  MASTER_UI="http://$(_host "$SPARK_MASTER_URL"):8080"
fi
DRIVER_UI="${PYG_SPARK_DRIVER_UI:-}"
if [[ -z "$DRIVER_UI" && -n "${SPARK_DRIVER_HOST:-}" ]]; then
  DRIVER_UI="http://${SPARK_DRIVER_HOST}:4040"
fi
if [[ -z "$MASTER_UI" ]]; then
  echo "no master UI: set PYG_SPARK_MASTER_UI or SPARK_MASTER_URL" >&2
  exit 2
fi

PY="${PYG_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -n "${PYG_REPO_ROOT:-}" && -x "$PYG_REPO_ROOT/.venv/bin/python" ]]; then
    PY="$PYG_REPO_ROOT/.venv/bin/python"
  else
    PY=python3
  fi
fi

while [[ ! -f "$STOP" ]]; do
    curl -s --max-time 8 "$MASTER_UI/json/" 2>/dev/null \
      | "$PY" -c '
import json, sys, time, urllib.request

UI = sys.argv[1] if len(sys.argv) > 1 else ""

def ui(path):
    if not UI:
        return None
    try:
        with urllib.request.urlopen(UI + "/api/v1" + path, timeout=8) as r:
            return json.load(r)
    except Exception:
        return None

try:
    m = json.load(sys.stdin)
except Exception:
    m = {}

apps = m.get("activeapps", []) or []
rec = {
    "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "cores": m.get("cores"), "coresused": m.get("coresused"),
    "workers": len(m.get("workers", []) or []),
    "activeapps": len(apps),
    "app": apps[0].get("id") if apps else None,
    "appname": apps[0].get("name") if apps else None,
}

# Host memory: on a unified-memory GPU the pool is host RAM, so this is the number
# that predicts an executor being evicted.
try:
    mem = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        mem[k] = int(v.split()[0])
    rec["mem_total_gb"] = round(mem["MemTotal"] / 1e6, 1)
    rec["mem_avail_gb"] = round(mem["MemAvailable"] / 1e6, 1)
    rec["swap_used_gb"] = round((mem["SwapTotal"] - mem["SwapFree"]) / 1e6, 1)
except Exception:
    pass

if rec["app"]:
    execs = ui("/applications/%s/executors" % rec["app"])
    if execs:
        alive = [e for e in execs if e.get("isActive")]
        rec["executors_alive"] = len(alive)
        rec["tasks_active"] = sum(e.get("activeTasks", 0) for e in alive)
        rec["tasks_failed"] = sum(e.get("failedTasks", 0) for e in alive)
        rec["gc_ms"] = sum(e.get("totalGCTime", 0) for e in alive)
        rec["task_ms"] = sum(e.get("totalDuration", 0) for e in alive)
        rec["mem_spill_gb"] = round(
            sum(e.get("memoryMetrics", {}).get("usedOnHeapStorageMemory", 0)
                for e in alive) / 1e9, 2)
    stages = ui("/applications/%s/stages" % rec["app"])
    if stages:
        rec["stages_done"] = sum(1 for s in stages if s.get("status") == "COMPLETE")
        rec["stages_active"] = sum(1 for s in stages if s.get("status") == "ACTIVE")
        rec["stages_failed"] = sum(1 for s in stages if s.get("status") == "FAILED")
        rec["input_gb"] = round(sum(s.get("inputBytes", 0) for s in stages) / 1e9, 2)
        rec["shuffle_write_gb"] = round(
            sum(s.get("shuffleWriteBytes", 0) for s in stages) / 1e9, 2)
        rec["disk_spill_gb"] = round(
            sum(s.get("diskBytesSpilled", 0) for s in stages) / 1e9, 2)

print(json.dumps(rec))
' "$DRIVER_UI" >> "$OUT" 2>/dev/null
    sleep "$INTERVAL"
done
