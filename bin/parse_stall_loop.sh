#!/usr/bin/env bash
# Run the parse on its own, over and over, until it deadlocks -- and record the rate.
#
#   bin/parse_stall_loop.sh <run-dir> [trials]
#
# <run-dir> is the same shape every cluster run uses: an untracked env.sh with the
# identity and intent (see bin/profiles/run-env.example.sh). Afterwards it holds
# trials.jsonl, one line per trial, and stalls/trial-NNN/ for any capture.
#
# WHY THIS EXISTS
# ---------------
# The stage that materialises the parse deadlocks between the executor JVM and its
# Python worker at 168 of 169 tasks and never clears (issue #388). It is a race, and
# the evidence for that is not an inference: on 2026-09-07 the same job read the same
# mirror twice within half an hour, with byte-identical Spark properties -- same
# PythonMapInArrowExec=false, same task GPU fraction, same batch size -- and stalled
# on the first attempt while clearing the same stage in 68.7s on the second. There is
# no diff to read.
#
# A full notebook run costs about three hours and delivers exactly ONE parse attempt.
# At the observed rate -- three of the last seven -- that is a coin flip you wait a
# morning for, and it is why three separate fixes have each been declared verified on
# a single clean run and each come back.
#
# The parse is the FIRST action a seed leg takes. Everything before it is Parquet
# metadata listing, so a trial reaches its verdict in under two minutes: measured at
# 1m23s and 1m54s, against a 68.7s count. This turns one attempt per morning into
# twenty per hour, which is the difference between an anecdote and a rate.
#
# WHAT IT DOES NOT DO
# -------------------
# It does not kill a job it caught. The whole value of a live stall is in state that
# dies with the process: the socket queues that show ~4 MB stuck in both directions
# at once, and the Python worker stack, which needs `sudo py-spy dump --pid <worker>`
# and therefore a worker still there to dump. So a capture stops the LOOP and leaves
# the job running, with instructions.
#
# It does not start a stall watchdog either. bin/submit_spark_job.sh starts one per
# submit, so every trial is covered without this file having to remember.
#
# It reproduces the parse deadlock and nothing else. The other stall on record --
# a GPU shuffle exchange in an assembly leg -- self-cleared and let its leg finish,
# where this one never clears. That is the failure that kills runs, so it is the one
# worth twenty trials an hour.
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <run-dir> [trials]" >&2
  exit 2
fi

RD="${1%/}"
TRIALS="${2:-20}"

if [[ ! -d "$RD" ]]; then
  echo "no such run directory: $RD" >&2
  exit 2
fi
if [[ ! -f "$RD/env.sh" ]]; then
  echo "no $RD/env.sh -- copy bin/profiles/run-env.example.sh and fill it in" >&2
  exit 2
fi

# One loop per run directory, for the reason run_cluster_notebook.sh takes the same
# lock: two of them interleaving trials would write one trials.jsonl describing two
# different sets of conditions, and the rate computed off it would be of nothing.
exec 9>"$RD/.lock"
if ! flock -n 9; then
  echo "another run is already using $RD (see $RD/loop.log); stop it first" >&2
  exit 2
fi

# Sourced once here for the static values. env.sh interpolates RUN_ID into
# PYG_WORK_DIR, so it is re-sourced per trial below with that trial's id -- which is
# what gives each trial its own work dir, and so its own manifest.
RUN_ID=placeholder
export RUN_ID
# shellcheck source=/dev/null
. "$RD/env.sh"

# The seed leg's sizing, sourced AFTER env.sh because env.sh is what names it.
#
# This is load-bearing, not tidiness. Left out, a trial submits at the defaults --
# GPU_PER_TASK=0.125 and a 4g executor, so eight parse slots against the seed leg's
# sixty-four, on a fraction of the memory. Concurrency across that boundary is the
# most likely thing the race turns on, so a trial run at the wrong sizing is not a
# cheap version of the seed leg's parse; it is a different experiment that happens to
# use the same code.
seed_profile() {
  [[ -n "${PYG_SEED_PROFILE:-}" && -f "${PYG_SEED_PROFILE}" ]] || return 0
  # shellcheck source=/dev/null
  . "$PYG_SEED_PROFILE"
}
seed_profile

: "${PYG_REPO_ROOT:?env.sh must set PYG_REPO_ROOT}"
: "${PYG_SOURCE_PATHS:?env.sh must set PYG_SOURCE_PATHS}"
: "${PYG_TIME_PERIOD:?env.sh must set PYG_TIME_PERIOD}"

REPO="$PYG_REPO_ROOT"
TRIAL_LOG="$RD/trials.jsonl"
WANT_FREE_GB="${PYG_WANT_FREE_GB:-90}"
TRIAL_TIMEOUT="${PYG_TRIAL_TIMEOUT:-1800}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RD/loop.log"; }

LOCAL_HOST="$(hostname -s)"
NODE_LIST="${PYG_STAGE_NODES:-}"
REMOTE_NODES=()
for h in ${NODE_LIST//,/ }; do
  ip -4 -o addr show 2>/dev/null | grep -q " $h/" && continue
  [[ "$h" == "$LOCAL_HOST" ]] && continue
  REMOTE_NODES+=("$h")
done
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5)

mkdir -p "$RD/stalls"

# ---------------------------------------------------------------------------
# Reclaim the file cache before every trial.
#
# Not housekeeping. Each executor opens its RMM pool as a fraction of MemFree AT
# EXECUTOR START, so MemFree is an input to the trial, not a background condition:
# runs on 2026-09-01 and 2026-09-03 opened pools of 989 MB and 337 MB against a
# healthy 18-19 GB and died of it. A trial that trips over that is not a trial of
# the parse, and averaging it into a rate makes the rate mean nothing.
#
# It gates rather than aborts. chain.sh aborts the whole run when a node cannot
# reach the target, which cost an arm on 2026-09-05 at 88.2G against a 90G ask. Here
# a short node SKIPS one trial and the loop carries on -- the reading stays clean and
# the instrument stays up. Skipped trials are recorded, so they cannot be mistaken
# for clean ones.
# ---------------------------------------------------------------------------
reclaim() {
  local ok=1 out rc
  out="$(timeout 900 python3 "$REPO/bin/mem_reclaim.py" "$WANT_FREE_GB" 2>&1)"
  rc=$?
  (( rc != 0 )) && ok=0
  log "  local: $out"
  for h in "${REMOTE_NODES[@]}"; do
    # Piped over stdin, so the far side needs no checkout of this repo.
    out="$("${SSH[@]}" "$h" "timeout 900 python3 - $WANT_FREE_GB" \
           <"$REPO/bin/mem_reclaim.py" 2>&1)"
    rc=$?
    (( rc != 0 )) && ok=0
    log "  $h: $out"
  done
  (( ok == 1 ))
}

# Spark's scratch is dead weight the moment a driver exits -- it is keyed to that
# SparkContext's own UUID, so nothing can reattach to it. Twenty trials leave twenty
# of them, and on 2026-09-05 one node had let 101 GB accumulate this way.
sweep_scratch() {
  local dir="${SPARK_LOCAL_DIRS:-/var/lib/spark/local}"
  rm -rf "${dir:?}"/spark-* 2>/dev/null
  for h in "${REMOTE_NODES[@]}"; do
    "${SSH[@]}" "$h" "rm -rf '${dir:?}'/spark-* 2>/dev/null" 2>/dev/null
  done
}

# The count and the seconds come out of the manifest the job wrote, not out of a log
# line. A log format is not a contract; parse_seconds is in the result dict for this.
read_manifest() {
  local work_dir="$1"
  python3 - "$work_dir" <<'PY' 2>/dev/null
import glob, json, sys
hits = sorted(glob.glob(f"{sys.argv[1]}/manifests/*/*/parse_only_*.json"))
if not hits:
    print("null null")
    raise SystemExit
payload = json.load(open(hits[-1]))
result = payload.get("result", {})
print(result.get("parse_seconds", "null"), result.get("initial_triples", "null"))
PY
}

record() {
  python3 - "$TRIAL_LOG" "$@" <<'PY'
import json, sys
path, keys = sys.argv[1], sys.argv[2:]
row = dict(zip(keys[::2], keys[1::2]))
for numeric in ("trial", "seconds", "parse_seconds", "triples", "submit_rc"):
    raw = row.get(numeric)
    if raw is None:
        continue
    # A missing manifest reads back as "null"/"" -- store it as JSON null rather
    # than as the four-character string, which would sort and sum as data.
    if raw in ("", "null", "None"):
        row[numeric] = None
        continue
    try:
        row[numeric] = float(raw) if "." in raw else int(raw)
    except ValueError:
        pass
with open(path, "a") as handle:
    handle.write(json.dumps(row) + "\n")
PY
}

log "parse stall loop: up to $TRIALS trial(s) into $RD"
log "want MemFree >= ${WANT_FREE_GB}G on $((${#REMOTE_NODES[@]} + 1)) node(s) before each"
log "commit $(cd "$REPO" && git rev-parse --short HEAD) on $(cd "$REPO" && git rev-parse --abbrev-ref HEAD)"

CAUGHT=""
for (( trial = 1; trial <= TRIALS; trial++ )); do
  # The trial number is part of the id, not decoration. date is second-resolution and
  # a skipped or fast-failing trial takes less than that, so two trials can be stamped
  # the same second -- and since PYG_WORK_DIR is built from RUN_ID, that would silently
  # put them in one work directory and have the second read the first's manifest.
  TRIAL_ID="$(date -u +%Y%m%dT%H%M%SZ)-t$(printf '%03d' "$trial")"
  STALL_DIR="$RD/stalls/trial-$TRIAL_ID"

  log ""
  log "trial $trial/$TRIALS  id=$TRIAL_ID"

  if ! reclaim; then
    log "  SKIPPED: a node could not reach ${WANT_FREE_GB}G MemFree"
    record trial "$trial" run_id "$TRIAL_ID" verdict skipped
    continue
  fi

  # Each trial re-sources env.sh under its own RUN_ID, which is what moves
  # PYG_WORK_DIR to a fresh directory. Two trials sharing one would have the second
  # reading the first's manifest and reporting its parse time as its own.
  RUN_ID="$TRIAL_ID"
  export RUN_ID
  # shellcheck source=/dev/null
  . "$RD/env.sh"
  seed_profile

  # A per-trial dump directory, so a capture is unambiguously THIS trial's. Sharing
  # one directory is how run_cluster_notebook.sh used to end a run before it began,
  # on a capture an earlier attempt had left behind (#386).
  mkdir -p "$STALL_DIR" "$PYG_WORK_DIR"
  export PYG_STALL_DUMP_DIR="$STALL_DIR"

  # The mirror is already on both nodes and staging defaults to on, which would
  # re-pull every source from object storage on every trial and bill the egress once
  # per node. The loop reads what is there; it does not stage.
  export PYG_STAGE_ENABLED=false

  # setsid, so a caught job outlives this loop. The loop exits the moment it has a
  # capture, and the evidence worth having -- the socket queues, a worker py-spy can
  # still reach -- exists only while that job is up. A child in this session would
  # take the loop's SIGHUP with it.
  started=$(date +%s)
  setsid bash -c 'exec "$@"' _ \
    "$REPO/bin/submit_spark_job.sh" \
    --mode parse_only \
    --local_work_dir "$PYG_WORK_DIR" \
    --time_period "$PYG_TIME_PERIOD" \
    --source_paths "$PYG_SOURCE_PATHS" \
    --source_format "${PYG_SOURCE_FORMAT:-turtle_parquet}" \
    --parquet_partitions "${PYG_PARQUET_PARTITIONS:-200}" \
    >"$RD/trial-$TRIAL_ID.log" 2>&1 &
  submit_pid=$!
  # The whole group, because the timeout path below has to take the spark-submit JVM
  # with it. Signalling the wrapper alone leaves that JVM holding every core on the
  # cluster, and the next trial then measures a parse competing with an orphan --
  # which is not a trial of anything.
  submit_pgid="$(ps -o pgid= -p "$submit_pid" 2>/dev/null | tr -d ' ')"

  # Wait for whichever comes first: the submit finishing, or a capture appearing. A
  # stalled parse does not exit -- that is the entire defect -- so waiting on the
  # submit alone would sit here until the trial cap.
  stalled=""
  while kill -0 "$submit_pid" 2>/dev/null; do
    if compgen -G "$STALL_DIR/stall-*" > /dev/null 2>&1; then
      stalled=1
      break
    fi
    if (( $(date +%s) - started > TRIAL_TIMEOUT )); then
      break
    fi
    command sleep 5
  done
  elapsed=$(( $(date +%s) - started ))

  if [[ -n "$stalled" ]]; then
    capture="$(ls -d "$STALL_DIR"/stall-* 2>/dev/null | tail -1)"
    log "  STALL CAUGHT after ${elapsed}s -> $capture"
    record trial "$trial" run_id "$TRIAL_ID" verdict stalled \
           seconds "$elapsed" capture "$capture" work_dir "$PYG_WORK_DIR"
    CAUGHT="$capture"
    break
  fi

  if kill -0 "$submit_pid" 2>/dev/null; then
    log "  TIMEOUT after ${elapsed}s with no capture -- killing the submit"
    if [[ -n "$submit_pgid" ]]; then
      kill -TERM -"$submit_pgid" 2>/dev/null
      command sleep 5
      kill -KILL -"$submit_pgid" 2>/dev/null
    else
      kill -TERM "$submit_pid" 2>/dev/null
      command sleep 5
      kill -KILL "$submit_pid" 2>/dev/null
    fi
    record trial "$trial" run_id "$TRIAL_ID" verdict timeout seconds "$elapsed"
  else
    wait "$submit_pid"; rc=$?
    read -r parse_seconds triples <<<"$(read_manifest "$PYG_WORK_DIR")"
    if (( rc == 0 )); then
      log "  clean: parse ${parse_seconds}s, ${triples} triples (${elapsed}s wall)"
      record trial "$trial" run_id "$TRIAL_ID" verdict clean seconds "$elapsed" \
             parse_seconds "$parse_seconds" triples "$triples" submit_rc "$rc"
    else
      log "  ERROR: submit exited rc=$rc after ${elapsed}s -- see trial-$TRIAL_ID.log"
      record trial "$trial" run_id "$TRIAL_ID" verdict error seconds "$elapsed" \
             submit_rc "$rc"
    fi
  fi

  # Only ever on a trial that finished. A caught stall breaks out above with its job
  # still running, and sweeping under a live executor would take its scratch away.
  sweep_scratch
  rm -rf "${PYG_WORK_DIR:?}/checkpoints" 2>/dev/null
done

log ""
log "===== summary ====="
python3 - "$TRIAL_LOG" <<'PY' | tee -a "$RD/loop.log"
import collections, json, os, sys
if not os.path.exists(sys.argv[1]):
    print("  no trials recorded")
    raise SystemExit
rows = [json.loads(line) for line in open(sys.argv[1])]
counts = collections.Counter(row["verdict"] for row in rows)
for verdict, n in counts.most_common():
    print(f"  {verdict:8s} {n}")
clean = [r["parse_seconds"] for r in rows
         if r["verdict"] == "clean" and isinstance(r.get("parse_seconds"), (int, float))]
if clean:
    print(f"  parse seconds over {len(clean)} clean trial(s): "
          f"min {min(clean):.1f} median {sorted(clean)[len(clean) // 2]:.1f} "
          f"max {max(clean):.1f}")
attempts = counts["clean"] + counts["stalled"]
if attempts:
    print(f"  stall rate: {counts['stalled']}/{attempts}")
PY

if [[ -n "$CAUGHT" ]]; then
  log ""
  log "THE JOB IS STILL RUNNING and deliberately not killed. While it sits:"
  log "  sudo py-spy dump --pid \$(pgrep -f '[p]yspark.daemon|[p]yspark.worker' | head -1)"
  log "  ss -tn | awk 'NR==1 || (\$2+0)>0 || (\$3+0)>0'"
  log "then stop it with:"
  log "  pkill -f '[b]uild_graph.py'"
  exit 99
fi

log "no stall in $TRIALS trial(s)"
exit 0
