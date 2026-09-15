#!/usr/bin/env bash
# Run one day of a schedule: build the graph from that day's sources, publish it, and
# only then prune older runs that are already published.
#
#   bin/daily_run.sh [--data-date YYYY-MM-DD] <schedule-dir>
#
# <schedule-dir> holds an untracked env.sh, the same contract as a run directory's
# (see bin/profiles/run-env.example.sh) plus the settings below. Each day gets a run
# directory of its own, <schedule-dir>/runs/<RUN_ID>/, holding a copy of env.sh that
# names the sources the day resolved to. Every step is logged to
# <schedule-dir>/daily.log.
#
# The day is --data-date when given. Otherwise it is today on this host's clock, less
# PYG_SCHEDULE_DATA_LAG_DAYS; PYG_SCHEDULE_TODAY stands in for today, for tests.
#
# IN ORDER
#   refuse a checkout with uncommitted changes, and log the commit that runs
#   check that every source is there; a missing one skips the day
#   wait for an idle cluster
#   stage the sources, when the run reads a local mirror
#   bring MemFree up to PYG_MEMFREE_GATE_GB on every node, when that is set
#   bin/run_cluster_notebook.sh --data-date <day> <run-dir>
#   bin/publish_run.py <run-dir> --upload
#   prune, and only after that publish went through: the newest
#   PYG_SCHEDULE_RETAIN_RUNS runs keep their work directory, and an older run loses
#   its work directory only if bin/publish_run.py --published finds it listed
#
# SETTINGS, from env.sh
#   PYG_SCHEDULE_DATA_LAG_DAYS      days from the data day to today; not read with
#                                   --data-date
#   PYG_SCHEDULE_RETAIN_RUNS        how many of the newest runs keep a work directory
#   PYG_YEARLY_SOURCE_PREFIXES      optional, comma separated. Each prefix holds one
#                                   YYYY.* file per year, and a run reads the newest
#                                   one whose year is not after the data day's.
#   PYG_MEMFREE_GATE_GB             optional. MemFree every node must reach first.
#   PYG_EXPECTED_WORKERS            optional. ALIVE workers on an idle cluster;
#                                   default, the number of PYG_STAGE_NODES.
#   PYG_SCHEDULE_IDLE_WAIT_SECONDS  optional. How long to wait for idle; default 600.
#   PYG_SCHEDULE_RETRY_SECONDS      optional. Pause between tries; default 60.
#
# EXIT CODES
#   0  published, then pruned
#   1  the staging, the run or the publish failed; the run stays, nothing is pruned
#   2  refused before the run started: a setting, a dirty checkout, sources that
#      could not be listed, a busy cluster, or memory that would not come free
#   3  skipped: a source is not there, because upstream is late or the day has
#      none, as on a market holiday. Kept apart from 1 so that a late upstream
#      write does not make the failure signal mean nothing.
#
# Nothing in this file names a host, a bucket or a path.
set -uo pipefail

usage() {
  echo "usage: $(basename "$0") [--data-date YYYY-MM-DD] <schedule-dir>" >&2
  exit 2
}

SD=""
DATA_DATE=""
DATE_GIVEN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-date)
      [[ $# -ge 2 ]] || usage
      DATA_DATE="$2"
      DATE_GIVEN=1
      shift 2
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      ;;
    *)
      [[ -z "$SD" ]] || usage
      SD="${1%/}"
      shift
      ;;
  esac
done
[[ -n "$SD" ]] || usage
if [[ ! -f "$SD/env.sh" ]]; then
  echo "no $SD/env.sh -- a schedule directory holds the env.sh its runs are built from" >&2
  exit 2
fi
SD="$(cd "$SD" && pwd)"
ENV_FILE="$SD/env.sh"

log() { printf '%s %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$SD/daily.log"; }
refuse() { log "REFUSED: $*"; exit 2; }
fail() { log "FAILED: $*"; exit 1; }
skip() { log "SKIPPED: $*"; exit 3; }

# One day at a time per schedule. systemd does not start a unit that is still
# running, but a copy started by hand beside the timer's would share the cluster.
exec 8>"$SD/.daily.lock"
if ! flock -n 8; then
  echo "another bin/daily_run.sh is using $SD; not starting a second" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# The checkout and the day
# ---------------------------------------------------------------------------
# A setting that does not depend on the day. env.sh may build paths from RUN_ID and
# the day, and fail on purpose while they are unset, so it is read with stand-ins.
setting() {
  RUN_ID=00000000T000000Z PYG_RUN_DIR=/nonexistent \
    PYG_DATA_YEAR=0000 PYG_DATA_MONTH=01 PYG_DATA_DAY=01 \
    bash -c '. "$1" >/dev/null 2>&1; printf "%s" "${!2-}"' _ "$ENV_FILE" "$1"
}

REPO="$(setting PYG_REPO_ROOT)"
[[ -n "$REPO" && -x "$REPO/bin/run_cluster_notebook.sh" ]] \
  || refuse "PYG_REPO_ROOT in env.sh is not a checkout of this repository: '$REPO'"

# The run builds from this checkout, and the assembly leg packages it over an hour
# in. An uncommitted edit would publish a graph that no commit reproduces.
if ! dirty="$(git -C "$REPO" status --porcelain 2>&1)"; then
  refuse "cannot read the checkout at $REPO: $dirty"
fi
[[ -z "$dirty" ]] || refuse "the checkout at $REPO has uncommitted changes"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

valid_date() {
  [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "$(date -d "$1" +%F 2>/dev/null)" == "$1" ]]
}
if [[ -z "$DATE_GIVEN" ]]; then
  LAG="$(setting PYG_SCHEDULE_DATA_LAG_DAYS)"
  [[ "$LAG" =~ ^[0-9]+$ ]] \
    || refuse "PYG_SCHEDULE_DATA_LAG_DAYS must be a whole number of days, got '$LAG'"
  TODAY="${PYG_SCHEDULE_TODAY:-$(date +%F)}"
  valid_date "$TODAY" || refuse "PYG_SCHEDULE_TODAY must be a date written YYYY-MM-DD, got '$TODAY'"
  # From noon, so a daylight-saving change cannot move the answer across midnight.
  DATA_DATE="$(date -d "$TODAY 12:00 $LAG days ago" +%F)"
fi
valid_date "$DATA_DATE" \
  || refuse "--data-date must be a real date written YYYY-MM-DD, got '$DATA_DATE'"

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RD="$SD/runs/$RUN_ID"
[[ ! -e "$RD" ]] || refuse "$RD already exists"
export PYG_RUN_DIR="$RD"
export PYG_DATA_YEAR="${DATA_DATE:0:4}" PYG_DATA_MONTH="${DATA_DATE:5:2}" PYG_DATA_DAY="${DATA_DATE:8:2}"

# Loaded in a subshell first: a ${VAR:?} that fails would end this script where it
# stands, with no line in the log to say why.
( . "$ENV_FILE" ) >/dev/null 2>&1 || refuse "env.sh does not load for the day $DATA_DATE"
# shellcheck source=/dev/null
. "$ENV_FILE"
log "day $DATA_DATE, run $RUN_ID, commit ${COMMIT:-unknown}"

RETAIN="${PYG_SCHEDULE_RETAIN_RUNS:-}"
[[ "$RETAIN" =~ ^[1-9][0-9]*$ ]] \
  || refuse "PYG_SCHEDULE_RETAIN_RUNS must be 1 or more, got '$RETAIN'"
[[ "${PYG_WORK_DIR:-}" == */"$RUN_ID" ]] \
  || refuse "PYG_WORK_DIR must end in the run id, got '${PYG_WORK_DIR:-}'"
[[ -n "${PYG_PUBLISH_ROOT:-}" ]] \
  || refuse "PYG_PUBLISH_ROOT is not set; a schedule publishes, and its prune depends on that"
RETRY="${PYG_SCHEDULE_RETRY_SECONDS:-60}"
IDLE_WAIT="${PYG_SCHEDULE_IDLE_WAIT_SECONDS:-600}"
[[ "$RETRY" =~ ^[0-9]+$ && "$IDLE_WAIT" =~ ^[0-9]+$ ]] \
  || refuse "PYG_SCHEDULE_RETRY_SECONDS and PYG_SCHEDULE_IDLE_WAIT_SECONDS must be whole seconds"

# A new run directory has no runner-venv of its own, and the kernel the notebook runs
# in is found by name, so it can live inside an old run directory that gets tidied away.
[[ -n "${PYG_RUNNER_PYTHON:-}" && -x "$PYG_RUNNER_PYTHON" ]] \
  || refuse "PYG_RUNNER_PYTHON must name the notebook runner's python, got '${PYG_RUNNER_PYTHON:-}'"
KERNEL="${PYG_NOTEBOOK_KERNEL:-pyg-notebook-runner}"
if ! kernel="$("$PYG_RUNNER_PYTHON" - "$KERNEL" 2>&1 <<'PY'
import shutil
import sys

from jupyter_client.kernelspec import KernelSpecManager

spec = KernelSpecManager().get_kernel_spec(sys.argv[1])
if not shutil.which(spec.argv[0]):
    sys.exit(f"the notebook kernel {sys.argv[1]} runs {spec.argv[0]}, which is not there")
PY
)"; then
  refuse "${kernel##*$'\n'}"
fi

# ---------------------------------------------------------------------------
# The sources: every one there, or the day is skipped
# ---------------------------------------------------------------------------
SOURCES=()
listed="$(python3 - "$DATA_DATE" "${PYG_SOURCE_PATHS:-}" "${PYG_YEARLY_SOURCE_PREFIXES:-}" <<'PY'
import json
import re
import subprocess
import sys

day, dated, yearly = sys.argv[1], sys.argv[2], sys.argv[3]
year = int(day[:4])


def split(uri):
    scheme, _, rest = uri.partition("://")
    bucket, _, key = rest.partition("/")
    return scheme, bucket, key


def keys(bucket, prefix):
    r = subprocess.run(["aws", "s3api", "list-objects-v2", "--bucket", bucket,
                        "--prefix", prefix, "--output", "json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"cannot list s3://{bucket}/{prefix}: {r.stderr.strip()}")
        sys.exit(2)
    body = json.loads(r.stdout) if r.stdout.strip() else {}
    return [item["Key"] for item in body.get("Contents") or []]


missing = 0
for uri in filter(None, (p.strip() for p in dated.split(","))):
    scheme, bucket, key = split(uri)
    found = keys(bucket, key)
    # A path ending in / is a directory and needs something under it. Any other path
    # names one object, and a key that only starts with it, such as a .tmp upload,
    # is not that object.
    there = bool(found) if uri.endswith("/") else key in found
    if there:
        print(f"SOURCE {uri}")
    else:
        print(f"missing: {uri}")
        missing += 1

for prefix in filter(None, (p.strip() for p in yearly.split(","))):
    scheme, bucket, key = split(prefix.rstrip("/") + "/")
    years = {}
    for found in keys(bucket, key):
        name = re.fullmatch(r"(\d{4})\.[^/]+", found[len(key):])
        if name and int(name.group(1)) <= year:
            years[int(name.group(1))] = found
    if years:
        newest = years[max(years)]
        print(f"SOURCE {scheme}://{bucket}/{newest}")
        print(f"yearly: {scheme}://{bucket}/{key} -> {newest[len(key):]}")
    else:
        print(f"missing: {scheme}://{bucket}/{key} has no YYYY file for {year} or before")
        missing += 1

sys.exit(3 if missing else 0)
PY
)"
listed_rc=$?
while IFS= read -r line; do
  case "$line" in
    "SOURCE "*) SOURCES+=("${line#SOURCE }") ;;
    ?*) log "  $line" ;;
  esac
done <<< "$listed"
case "$listed_rc" in
  0) ;;
  3) skip "day $DATA_DATE: a source is not there" ;;
  *) refuse "the sources for $DATA_DATE could not be listed" ;;
esac
(( ${#SOURCES[@]} > 0 )) \
  || refuse "env.sh names no sources; set PYG_SOURCE_PATHS or PYG_YEARLY_SOURCE_PREFIXES"
log "all ${#SOURCES[@]} sources are there"

# ---------------------------------------------------------------------------
# An idle cluster. Two drivers on one standalone cluster starve each other.
# ---------------------------------------------------------------------------
IFS=',' read -ra NODES <<< "${PYG_STAGE_NODES:-}"
WANT="${PYG_EXPECTED_WORKERS:-${#NODES[@]}}"
[[ "$WANT" =~ ^[1-9][0-9]*$ ]] \
  || refuse "set PYG_STAGE_NODES or PYG_EXPECTED_WORKERS, so an idle cluster can be recognised"
MASTER_UI="${PYG_SPARK_MASTER_UI:-}"
if [[ -z "$MASTER_UI" && -n "${SPARK_MASTER_URL:-}" ]]; then
  master="${SPARK_MASTER_URL#*://}"
  MASTER_UI="http://${master%%:*}:8080"
fi
[[ -n "$MASTER_UI" ]] || refuse "no master UI: set PYG_SPARK_MASTER_UI or SPARK_MASTER_URL"

deadline=$((SECONDS + IDLE_WAIT))
while :; do
  state="$(curl -s --max-time 10 "$MASTER_UI/json/" 2>/dev/null | python3 -c '
import json, sys
try:
    m = json.load(sys.stdin)
except ValueError:
    sys.exit()
print(sum(w.get("state") == "ALIVE" for w in m.get("workers") or []), len(m.get("activeapps") or []))
' 2>/dev/null)"
  [[ "$state" == "$WANT 0" ]] && break
  (( SECONDS < deadline )) \
    || refuse "the cluster is not idle: ALIVE workers and running applications read '${state:-no answer}', not '$WANT 0'"
  sleep "$RETRY"
done
log "the cluster is idle: $WANT ALIVE workers, no running application"

# ---------------------------------------------------------------------------
# The run directory: env.sh as it stands today, then the sources the day resolved to
# ---------------------------------------------------------------------------
mkdir -p "$RD" || fail "cannot create $RD"
cp "$ENV_FILE" "$RD/env.sh" || fail "cannot copy env.sh into $RD"
if [[ -f "$SD/extra-checks.sh" ]]; then
  cp "$SD/extra-checks.sh" "$RD/extra-checks.sh" || fail "cannot copy extra-checks.sh into $RD"
fi
joined="$(IFS=,; printf '%s' "${SOURCES[*]}")"
{
  printf '\n# Added by bin/daily_run.sh: the sources run %s read for %s.\n' "$RUN_ID" "$DATA_DATE"
  printf 'export PYG_SOURCE_PATHS=%q\n' "$joined"
} >> "$RD/env.sh"
export PYG_SOURCE_PATHS="$joined"

if [[ "${PYG_INPUT_MODE:-s3}" == "local" ]]; then
  log "staging the sources; see $RD/stage.log"
  "$REPO/bin/stage_sources.sh" --sources "$PYG_SOURCE_PATHS" > "$RD/stage.log" 2>&1 \
    || fail "staging failed; see $RD/stage.log"
fi

# ---------------------------------------------------------------------------
# Free memory, when env.sh asks for it. On a unified-memory host RAPIDS sizes its
# pool from MemFree, which a finished run's file cache holds near zero for hours.
# ---------------------------------------------------------------------------
LOCAL_HOST="$(hostname -s)"
is_local() {
  local addrs
  [[ "$1" == "$LOCAL_HOST" ]] && return 0
  # Captured, not piped into grep -q: grep would close the pipe early, and pipefail
  # would then report this host's address as another host's.
  addrs="$(ip -4 -o addr show 2>/dev/null)"
  grep -q " $1/" <<< "$addrs"
}
if [[ -n "${PYG_MEMFREE_GATE_GB:-}" ]]; then
  for node in "${NODES[@]}"; do
    passed=""
    for try in 1 2 3; do
      if is_local "$node"; then
        python3 "$REPO/bin/mem_reclaim.py" "$PYG_MEMFREE_GATE_GB" >> "$RD/memgate.log" 2>&1 \
          && passed=1
      else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" python3 - "$PYG_MEMFREE_GATE_GB" \
          < "$REPO/bin/mem_reclaim.py" >> "$RD/memgate.log" 2>&1 && passed=1
      fi
      [[ -n "$passed" ]] && break
      (( try < 3 )) && sleep "$RETRY"
    done
    [[ -n "$passed" ]] \
      || refuse "$node did not reach ${PYG_MEMFREE_GATE_GB} G of MemFree in 3 tries; see $RD/memgate.log"
    log "the memory gate passed on $node"
  done
fi

# ---------------------------------------------------------------------------
# The run, then the publish. Either failing leaves everything where it is.
# ---------------------------------------------------------------------------
log "running; see $RD/run.log"
"$REPO/bin/run_cluster_notebook.sh" --data-date "$DATA_DATE" "$RD" > "$RD/launcher.log" 2>&1
launcher_rc=$?
run_rc="$(cat "$RD/run.done" 2>/dev/null || echo missing)"
[[ "$run_rc" == "0" ]] \
  || fail "the run did not finish cleanly (launcher rc=$launcher_rc, run.done=$run_rc), so it is not published. See $RD/outcome.txt"
log "the run finished"

log "publishing; see $RD/publish.log"
python3 "$REPO/bin/publish_run.py" "$RD" --upload > "$RD/publish.out" 2>&1
publish_rc=$?
[[ "$publish_rc" == "0" ]] \
  || fail "the publish exited $publish_rc; the run stays on the mount and nothing is pruned. See $RD/publish.log"
log "published"

# ---------------------------------------------------------------------------
# The prune. The newest runs keep their work directory whatever state they are in.
# An older run's work directory goes only when it is that run's own directory in
# this schedule's work root, and the destination lists the run as published. That
# is asked of the destination rather than read from publish.done: a publish run
# again on a finished run is refused, and publish.done then holds the refusal.
# ---------------------------------------------------------------------------
ROOT="${PYG_WORK_DIR%/*}"
REAL_ROOT="$(cd "$ROOT" 2>/dev/null && pwd -P)"
seen=0
pruned=0
for id in $(for dir in "$SD"/runs/*/; do dir="${dir%/}"; echo "${dir##*/}"; done \
              | grep -E '^[0-9]{8}T[0-9]{6}Z$' | sort -r); do
  [[ -f "$SD/runs/$id/run-config.txt" ]] || continue        # never started: no work dir
  seen=$((seen + 1))
  (( seen <= RETAIN )) && continue
  work="$(sed -n 's/^work dir *: *//p' "$SD/runs/$id/run-config.txt" | head -1)"
  [[ -e "$work" ]] || continue
  if [[ "$work" != "$ROOT/$id" || -L "$work" || -z "$REAL_ROOT" \
        || "$(cd "$work/.." 2>/dev/null && pwd -P)" != "$REAL_ROOT" ]]; then
    log "  kept $id: its work dir $work is not $ROOT/$id"
    continue
  fi
  if ! python3 "$REPO/bin/publish_run.py" "$SD/runs/$id" --published >> "$RD/prune.log" 2>&1; then
    log "  kept $id: the destination does not list it as published"
    continue
  fi
  rm -rf -- "$work" && pruned=$((pruned + 1)) && log "  removed the work dir of $id"
done
log "done: $DATA_DATE is published as run $RUN_ID; removed $pruned older work dir(s)"
exit 0
