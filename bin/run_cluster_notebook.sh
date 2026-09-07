#!/usr/bin/env bash
# Drive one real cluster run of the experiment notebook and leave a report behind
# whatever happens.
#
#   bin/run_cluster_notebook.sh <run-dir>
#
# <run-dir> holds everything about this run and nothing about the code: an
# untracked env.sh with the identity and intent (see bin/profiles/run-env.example.sh),
# and afterwards the executed notebook, run log, traces, event log and outcome.txt.
#
# WHAT THIS DOES
#   start a network trace on every node and a cluster sampler locally
#   run the notebook headlessly, teeing every cell as it arrives
#   kill the run if the local network starts collapsing
#   stop early if the stall watchdog captures a stalled stage
#   record the outcome either way, then sweep this run's checkpoints
#
# WHAT IT DELIBERATELY DOES NOT DO
# Acceptance checks for whatever the run is meant to prove go in <run-dir>/extra-checks.sh,
# which bin/record_run_outcome.sh sources. They used to live here and grew to about a
# third of the file, all of it checks for issues that had since merged. See
# bin/profiles/extra-checks.example.sh.
#
# WHY THIS IS TRACKED
# It used to exist only as a copy inside each run directory, copied forward from
# whichever run came before. Three separate silent failures on run 20260906T184507Z
# came from that: a stop flag cleared on one node but set on two, a checkpoint sweep
# keyed on a hardcoded run name so every copy swept nothing, and nothing recorded at
# all when the harness gave up. Fixing those in one copy fixed them for no other run.
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <run-dir>" >&2
  exit 2
fi

RD="${1%/}"
if [[ ! -d "$RD" ]]; then
  echo "no such run directory: $RD" >&2
  exit 2
fi
if [[ ! -f "$RD/env.sh" ]]; then
  echo "no $RD/env.sh -- copy bin/profiles/run-env.example.sh and fill it in" >&2
  exit 2
fi

# One supervisor per run directory. Two of them sharing one is not a hypothetical:
# an attempt that was killed and relaunched left the first supervisor alive, and it
# went on to append to the run log and to overwrite outcome.txt with a report of a
# different run. Refusing to start is the honest answer -- kill the old one first.
exec 9>"$RD/.lock"
if ! flock -n 9; then
  echo "another run is already using $RD (see $RD/run.log); stop it first" >&2
  exit 2
fi

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
# shellcheck source=/dev/null
. "$RD/env.sh"

: "${PYG_REPO_ROOT:?env.sh must set PYG_REPO_ROOT}"
: "${PYG_WORK_DIR:?env.sh must set PYG_WORK_DIR}"

REPO="$PYG_REPO_ROOT"
NOTEBOOK="${PYG_NOTEBOOK:-$REPO/notebook/multi_experiment.ipynb}"
KERNEL="${PYG_NOTEBOOK_KERNEL:-pyg-notebook-runner}"
RUNNER="${PYG_RUNNER_PYTHON:-$RD/runner-venv/bin/python}"   # needs nbformat + nbclient
STALL_DIR="${PYG_STALL_DUMP_DIR:-$RD/stalls}"
RTT_LIMIT_MS="${PYG_GATEWAY_RTT_LIMIT_MS:-60}"
RTT_STRIKES="${PYG_GATEWAY_STRIKES:-6}"

# Everything this run owns carries its run id, so nothing it writes can be mistaken
# for another run's and nothing another run does can stop it.
STOP="$RD/STOP-$RUN_ID"
DONE="$RD/run.done"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RD/run.log"; }

# The nodes to trace, minus this box. Addresses come from env.sh; none are named here.
LOCAL_HOST="$(hostname -s)"
NODE_LIST="${PYG_STAGE_NODES:-}"
REMOTE_NODES=()
for h in ${NODE_LIST//,/ }; do
  ip -4 -o addr show 2>/dev/null | grep -q " $h/" && continue
  [[ "$h" == "$LOCAL_HOST" ]] && continue
  REMOTE_NODES+=("$h")
done
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5)

: > "$RD/run.log"
rm -f "$DONE" "$RD/killed-by"

# Spark refuses to start when the event log directory is missing, which is how one
# run died 50 seconds in: every earlier run dir inherited the directory from a copy,
# and the first one assembled file by file did not have it.
EVENTLOG="$(sed -n 's/.*spark\.eventLog\.dir=\([^ ]*\).*/\1/p' <<<"${SPARK_EXTRA_CONF:-}")"
mkdir -p "$RD/eventlog" "$STALL_DIR"
[[ -n "$EVENTLOG" ]] && mkdir -p "${EVENTLOG#file://}"

log "RUN_ID=$RUN_ID"
log "work dir : $PYG_WORK_DIR"
log "input    : ${PYG_INPUT_MODE:-remote}${PYG_LOCAL_SOURCE_ROOT:+ mirror at $PYG_LOCAL_SOURCE_ROOT}"
[[ -n "${PYG_SEED_PROFILE:-}" ]] && log "seed     : $(grep -E '^export GPU_PER_TASK' "$PYG_SEED_PROFILE")"

# ---------------------------------------------------------------------------
# Network trace on every node, 1 Hz, fsynced per line.
#
# Both the trace file and the stop flag carry the run id. Before, a killed attempt's
# cleanup would set a shared stop flag and the live run's sampler on the other node
# would exit at once; the end-of-run copy then brought the dead attempt's trace back
# and the report described a different run. Sweeping stray samplers first covers the
# other half of it -- a supervisor that was SIGKILLed never sets any flag at all.
# ---------------------------------------------------------------------------
NETSAMPLE="$REPO/bin/netsample.py"
# The bracket keeps the pattern from matching the shell that carries it -- over ssh
# the remote command line contains this very string, and a plain pattern would have
# the sweep kill itself before it reached the mkdir.
SWEEP="pkill -f '[n]etsample\.py'"
eval "$SWEEP" 2>/dev/null
setsid nohup python3 "$NETSAMPLE" "$RD/net-$LOCAL_HOST-$RUN_ID.tsv" "$STOP" \
  </dev/null >/dev/null 2>&1 &
for h in "${REMOTE_NODES[@]}"; do
  "${SSH[@]}" "$h" "$SWEEP; mkdir -p '$RD'" 2>/dev/null
  scp -o BatchMode=yes -q "$NETSAMPLE" "$h:$RD/netsample.py" 2>/dev/null
  "${SSH[@]}" -f "$h" "cd '$RD' && PYG_NET_WAN_IFACE='${PYG_NET_WAN_IFACE:-}' \
     PYG_NET_FABRIC_IFACES='${PYG_NET_FABRIC_IFACES:-}' \
     setsid nohup python3 '$RD/netsample.py' '$RD/net-$h-$RUN_ID.tsv' '$STOP' \
     </dev/null >/dev/null 2>&1 &" 2>/dev/null
done
log "network trace started on $((${#REMOTE_NODES[@]} + 1)) node(s)"

# Cluster sampler: cores, executors, host memory and swap. The driver UI dies with the
# driver, so on the failure path this is the only record of how busy the cluster was.
setsid nohup bash "$REPO/bin/cluster_sampler.sh" "$RD" "$STOP" >/dev/null 2>&1 &
echo $! > "$RD/sampler-$RUN_ID.pid"
log "cluster sampler started -> $RD/samples.jsonl"

{
  echo "run id      : $RUN_ID"
  echo "started     : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
  echo "work dir    : $PYG_WORK_DIR"
  echo "input mode  : ${PYG_INPUT_MODE:-remote}${PYG_LOCAL_SOURCE_ROOT:+ mirror at $PYG_LOCAL_SOURCE_ROOT}"
  echo "seed profile: ${PYG_SEED_PROFILE:-none}"
  echo "timeouts    : seed ${PYG_SEED_TIMEOUT:-unset}s  experiment ${PYG_EXPERIMENT_TIMEOUT:-unset}s"
} > "$RD/run-config.txt"

# ---------------------------------------------------------------------------
# THE NETWORK GUARD. This is why the run is allowed to happen at all.
#
# The 2026-08-28 collapse announced itself before it arrived: latency to the gateway's
# own LAN interface went 1ms -> 82ms over five seconds, and the route dropped ~60s
# later. A LAN round trip is sub-millisecond when the forwarding plane is healthy.
# Losing a run costs hours; losing the house's network has twice cost a power-cycle of
# both machines.
# ---------------------------------------------------------------------------
GATEWAY="$(ip route | awk '/^default/ && !seen {print $3; seen=1}')"
watchdog() {
  local strikes=0 rtt
  while [[ ! -f "$DONE" ]]; do
    sleep 5
    rtt="$(ping -n -c1 -W1 "$GATEWAY" 2>/dev/null \
           | awk -F'[=/ ]' '/time=/ && !s {print int($(NF-1)); s=1}')"
    if [[ -z "$rtt" ]] || (( rtt > RTT_LIMIT_MS )); then
      strikes=$((strikes + 1))
      log "WATCHDOG: gateway rtt ${rtt:-unreachable}ms > ${RTT_LIMIT_MS}ms, strike ${strikes}/${RTT_STRIKES}"
      if (( strikes >= RTT_STRIKES )); then
        log "WATCHDOG: KILLING THE RUN to protect the network"
        echo "network" > "$RD/killed-by"
        pkill -TERM -g "$NB_PGID" 2>/dev/null
        sleep 5
        pkill -KILL -g "$NB_PGID" 2>/dev/null
        return
      fi
    else
      strikes=0
    fi
  done
}

# ---------------------------------------------------------------------------
# Execute the notebook headlessly. Its own submit() streams each leg's output and
# enforces per-leg timeouts; the caps in env.sh bound the legs, not this script.
#
# bin/execute_notebook.py, NOT nbconvert. nbconvert captures each cell's output into
# the .ipynb and nowhere else, so the run log held three lines of its own chatter and
# nothing from the job -- which is how a run whose seed died at 47 minutes, and whose
# three experiments then failed outright, was recorded as "rc=0, errors: none".
# ---------------------------------------------------------------------------
log "launching notebook (PYG_SEED_ONLY=${PYG_SEED_ONLY:-unset})"
setsid bash -c 'exec "$1" "$2" "$3" "$4" "$5"' _ \
  "$RUNNER" "$REPO/bin/execute_notebook.py" \
  "$NOTEBOOK" "$RD/executed-$RUN_ID.ipynb" "$KERNEL" \
  > "$RD/notebook.log" 2>&1 &
NB_PID=$!
NB_PGID="$(ps -o pgid= -p "$NB_PID" | tr -d ' ')"
log "notebook pid $NB_PID (pgid $NB_PGID) -> $RD/notebook.log"

watchdog &
WD_PID=$!

# The stall watchdog is NOT started here. bin/submit_spark_job.sh starts one per
# cluster submit, so every leg is covered without this file having to remember.
# env.sh points PYG_STALL_DUMP_DIR at the directory the loop below watches.
#
# Wait for whichever comes first: the notebook finishing, or a captured stall. A
# stalled leg does not exit -- it sits until its cap -- so waiting on the notebook
# alone would burn hours to learn the run was already dead.
STALLED=""
while kill -0 "$NB_PID" 2>/dev/null; do
  if compgen -G "$STALL_DIR/stall-*" > /dev/null 2>&1; then
    STALLED=1
    break
  fi
  command sleep 10
done

record() { bash "$REPO/bin/record_run_outcome.sh" "$RD" "$PYG_WORK_DIR" > /dev/null 2>&1; }

if [[ -n "$STALLED" ]]; then
  rc=99
  echo "$rc" > "$DONE"
  log "STALL CAPTURED: $(ls -d "$STALL_DIR"/stall-* 2>/dev/null | tail -1)"
  log "the job is STILL RUNNING and deliberately not killed -- inspect it, then:"
  log "  pkill -f execute_notebook.py && touch $STOP"
  # Record what the run had done by this point. A capture that turned out to be
  # false once exited here an hour before the job finished everything, and because
  # nothing was recorded on the way out, a successful run produced no report at all.
  # A capture is a reason to look, not proof the run is dead.
  record
  log "outcome recorded so far -> $RD/outcome.txt (the job may still be running)"
  # The checkpoint sweep at the bottom is NOT reached from here, and must not be:
  # the job is still running and may still read these.
  if [[ -d "$PYG_WORK_DIR/checkpoints" ]]; then
    log "checkpoints NOT swept ($(du -sh "$PYG_WORK_DIR/checkpoints" 2>/dev/null | cut -f1)) -- the job is still live."
    log "  once it has exited: rm -rf '$PYG_WORK_DIR/checkpoints'"
  fi
  exit "$rc"
fi

wait "$NB_PID"; rc=$?
echo "$rc" > "$DONE"
touch "$STOP"
for h in "${REMOTE_NODES[@]}"; do
  "${SSH[@]}" "$h" "touch '$STOP'" 2>/dev/null
  # Copy through .part: if the run directory ever turns out to be shared between
  # nodes, source and destination are the same file and a direct copy truncates it.
  if scp -o BatchMode=yes -o ConnectTimeout=10 -q \
       "$h:$RD/net-$h-$RUN_ID.tsv" "$RD/net-$h-$RUN_ID.tsv.part" 2>/dev/null; then
    mv -f "$RD/net-$h-$RUN_ID.tsv.part" "$RD/net-$h-$RUN_ID.tsv"
  fi
done
kill "$WD_PID" 2>/dev/null

log "notebook exited rc=$rc$( [[ -f "$RD/killed-by" ]] && echo ' (KILLED BY WATCHDOG)' )"
log "artifacts: $RD/executed-$RUN_ID.ipynb, $RD/notebook.log, $RD/samples.jsonl, $RD/net-*-$RUN_ID.tsv"
record
log "outcome recorded -> $RD/outcome.txt"

# Spark checkpoints are dead weight the moment the driver exits: they are keyed to the
# SparkContext's own UUID and RDD ids, so nothing later can reattach to them -- they
# cannot resume a failed run, which is the only reason anyone would keep them. This
# runs AFTER the recording so the report still sees what the run wrote.
#
# The guard keys on RUN_ID, which this invocation generated. It used to be a literal
# path, and every copy of the harness carried the PREVIOUS run's name in it, so the
# sweep silently did nothing until somebody remembered to edit the line -- one run
# left 462 GB behind that way. Matching RUN_ID cannot delete outside this run's own
# directory.
if [[ "$PYG_WORK_DIR" == */"$RUN_ID" && -d "$PYG_WORK_DIR/checkpoints" ]]; then
  log "removing $(du -sh "$PYG_WORK_DIR/checkpoints" 2>/dev/null | cut -f1) of checkpoints"
  rm -rf "${PYG_WORK_DIR:?}/checkpoints"
fi
