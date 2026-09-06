#!/usr/bin/env bash
# Summarise a cluster notebook run into <run-dir>/outcome.txt.
#
# WHY THIS IS TRACKED
# -------------------
# This used to exist only as a copy inside each run directory, alongside the rest
# of the run harness. On run 20260906T184507Z the harness aborted on a false stall
# just under an hour before the job actually finished, and because the recording
# lived on the far side of that abort, a run that succeeded produced no report at
# all. The run whose harness gave up is exactly the run whose report is worth
# having, so this has to be runnable on its own, from a clean checkout, after the
# fact.
#
# It only ever READS.
#
#   bin/record_run_outcome.sh <run-dir> [work-dir]
#
# <run-dir>   the run's artifacts: executed notebook, run.log, eventlog/, traces
# <work-dir>  where the job wrote enriched Parquet and .pt files; optional
#
# Environment:
#   PYG_RUN_NODES     comma-separated hosts to scan for executor logs. Defaults
#                     to PYG_STAGE_NODES from <run-dir>/env.sh when that exists.
#   SPARK_WORK_DIR    executor log root on each node (default /opt/spark/work)
#
# Per-run acceptance checks belong in <run-dir>/extra-checks.sh, which is sourced
# at the end if present. Keeping them out here is deliberate: the issue-specific
# blocks that accumulated in the previous version grew to about 40% of it and
# were all checks for issues that had since merged.
#
# The job's own output is in the executed notebook, NOT in the run log. The
# notebook runner writes only its own progress lines there, so anything that
# greps the run log sees a clean run no matter what happened -- that is how a
# 2026-08-29 run which failed four times out of four was recorded as "rc=0,
# errors: none". Cell outputs are flattened into notebook-cells.txt first and
# every section below reads that.
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <run-dir> [work-dir]" >&2
  exit 2
fi

RD="${1%/}"
WORK_DIR="${2:-}"
OUT="$RD/outcome.txt"
CELLS="$RD/notebook-cells.txt"
SPARK_WORK_DIR="${SPARK_WORK_DIR:-/opt/spark/work}"

if [[ ! -d "$RD" ]]; then
  echo "no such run directory: $RD" >&2
  exit 2
fi

# The newest executed notebook. The runner rewrites it after every cell, so
# there is something to read even mid-run, and everything to read once it ends.
NB="$(ls -t "$RD"/executed-*.ipynb 2>/dev/null | head -1)"
if [[ -n "$NB" ]]; then
  python3 - "$NB" "$CELLS" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as fh:
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                fh.write(f"{o.get('ename','')}: {o.get('evalue','')}\n")
                for line in o.get("traceback", []):
                    fh.write(line + "\n")
                continue
            t = o.get("text") or o.get("data", {}).get("text/plain") or ""
            if isinstance(t, list):
                t = "".join(t)
            fh.write(t if t.endswith("\n") or not t else t + "\n")
PY
fi

# Each submission prints "=== <name>" and then, when it ends, a status line
# reading "ok in N min | ..." or "FAILED (exit N) in N min | ...".
nbdigest() {   # $1 = state | table
  if [[ ! -s "$CELLS" ]]; then
    if [[ "$1" == state ]]; then
      echo "NOT KNOWN -- no executed notebook"
    else
      echo "  no executed notebook. The runner writes it as it goes, so this"
      echo "  means the run never reached its first cell."
    fi
    return
  fi
  python3 - "$CELLS" "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
mode = sys.argv[2]
# No \b after the alternation: "FAILED (exit 1)" ends in ")", and ") " is not a
# word boundary, so a \b there matches nothing.
status = re.compile(r"^\s{0,6}(ok|FAILED \(exit -?\d+\)|TIMEOUT|TIMED OUT|KILLED|SKIPPED)\s.*\bin [\d.]+ min\b")
subs = []
for line in lines:
    m = re.match(r"^===\s+(\S.*?)\s*$", line)
    if m:
        subs.append([m.group(1), None])
    elif subs and subs[-1][1] is None and status.match(line):
        subs[-1][1] = line.strip()

if mode == "state":
    if not subs:
        print("NOT KNOWN -- notebook ran no submissions")
    else:
        ok = sum(1 for _, s in subs if s and s.startswith("ok"))
        unfinished = sum(1 for _, s in subs if s is None)
        failed = len(subs) - ok - unfinished
        if ok == len(subs):
            bits = [f"all {ok} submissions ok"]
        elif ok == 0:
            bits = [f"NO SUBMISSION SUCCEEDED (0 of {len(subs)})"]
        else:
            bits = [f"{ok} of {len(subs)} submissions ok"]
        if failed:
            bits.append(f"{failed} FAILED")
        if unfinished:
            bits.append(f"{unfinished} never reported a result")
        print(", ".join(bits))
else:
    if not subs:
        print("  the notebook ran no submissions")
    for name, s in subs:
        print(f"  {name}")
        print(f"      {s if s else 'no status line -- this leg never finished'}")
PY
}

{
echo "======== RUN OUTCOME ========"
echo "written  : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "run dir  : $(basename "$RD")"
echo "run id   : $(grep -m1 'RUN_ID=' "$RD/run.log" 2>/dev/null | sed 's/.*RUN_ID=//')"
# The harness rc is the notebook runner's, never the job's, and is labelled that
# way. A harness that aborted says nothing about whether the job succeeded.
echo "harness  : $([ -f "$RD/run.done" ] && echo "runner rc=$(cat "$RD/run.done")" || echo 'no rc recorded')"
echo "state    : $(nbdigest state)"
echo "read from: $([ -n "$NB" ] && basename "$NB" || echo '(no executed notebook)')"
[ -f "$RD/killed-by" ] && echo "KILLED BY: $(cat "$RD/killed-by")"

echo
echo "--- per submission ---"
nbdigest table

echo
echo "--- phases the job reported ---"
if [[ -s "$CELLS" ]]; then
  grep -aE "PHASE:|Loaded [0-9,]+ triples|Enriched Parquet:|Initial triples|Final triples|[0-9,]+ rows \(|facts stated by" \
    "$CELLS" | sed -E 's/^[[:space:]]*/  /' | tail -40
else
  echo "  no executed notebook"
fi

echo
echo "--- task concurrency per stage ---"
echo "  (how many tasks each stage actually ran at once, read from the event log)"
python3 - "$RD/eventlog" <<'PY'
import glob, json, os, subprocess, sys, collections
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*")))
if not files:
    print("  no event log"); raise SystemExit
for f in files:
    if os.path.isdir(f):
        continue
    # Spark compresses the event log with zstd here, which gzip cannot read, and
    # there is no zstandard module on the host. Shell out.
    starts, ends, stage = [], [], {}
    try:
        if ".zstd" in f:
            proc = subprocess.Popen(["zstdcat", f], stdout=subprocess.PIPE, text=True,
                                    errors="replace", stderr=subprocess.DEVNULL)
            fh = proc.stdout
        else:
            fh = open(f, "rt", errors="replace")
        with fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                t = e.get("Event", "")
                if t == "SparkListenerTaskStart":
                    starts.append((e["Task Info"]["Launch Time"], e["Stage ID"]))
                elif t == "SparkListenerTaskEnd":
                    ends.append((e["Task Info"]["Finish Time"], e["Stage ID"]))
                elif t == "SparkListenerStageCompleted":
                    si = e["Stage Info"]
                    stage[si["Stage ID"]] = (si.get("Stage Name", "")[:40],
                                             si.get("Number of Tasks", 0),
                                             (si.get("Completion Time", 0) - si.get("Submission Time", 0)) / 1000.0)
    except Exception as ex:
        print(f"  {os.path.basename(f)}: unreadable ({ex})"); continue

    per = collections.defaultdict(list)
    for ts, sid in starts:
        per[sid].append((ts, 1))
    for ts, sid in ends:
        per[sid].append((ts, -1))
    print(f"  {os.path.basename(f)}")
    rows = []
    for sid, evs in per.items():
        evs.sort()
        cur = peak = 0
        conc = []
        for _, delta in evs:
            cur += delta
            peak = max(peak, cur)
            if delta == 1:
                conc.append(cur)
        conc.sort()
        med = conc[len(conc)//2] if conc else 0
        name, ntasks, secs = stage.get(sid, ("?", len(conc), 0))
        rows.append((secs, sid, name, ntasks, med, peak))
    for secs, sid, name, ntasks, med, peak in sorted(rows, reverse=True)[:12]:
        print(f"    stage {sid:>4}  {secs:8.1f}s  {ntasks:>7} tasks  median concurrency {med:>4}  peak {peak:>4}  {name}")
PY

echo
echo "--- network during the run (kernel byte counters, not Spark's) ---"
python3 - "$RD" <<'PY'
import csv, glob, os, re, sys, datetime
d = sys.argv[1]

# The run window, taken from the event log file names (app-<yyyyMMddHHmmss>-<n>),
# because those are written by Spark and need no cooperation from the harness.
#
# A trace whose samples fall outside this window is a leftover from an earlier
# attempt, not this run. That has happened repeatedly: a stop flag is cleared on
# one node but set on both, so the second node's sampler exits at once and the
# end-of-run copy brings the previous attempt's file back. Summarising it prints
# a confident set of numbers about a different run.
starts = []
for p in glob.glob(os.path.join(d, "eventlog", "app-*")):
    m = re.search(r"app-(\d{14})-", os.path.basename(p))
    if m:
        starts.append(datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").timestamp())
window_start = min(starts) if starts else None

for name in sorted(glob.glob(os.path.join(d, "net-*.tsv"))):
    label = os.path.basename(name)
    rows = list(csv.DictReader(open(name), delimiter="\t"))
    if len(rows) < 2:
        print(f"  {label}: too few samples")
        continue
    first, last = float(rows[0]["ts"]), float(rows[-1]["ts"])

    def fmt(t):
        return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")

    if window_start is not None and last < window_start:
        print(f"  {label}: STALE -- {len(rows)} samples covering {fmt(first)}-{fmt(last)}, "
              f"all before this run's first application at {fmt(window_start)}. "
              f"Left over from an earlier attempt; not summarised.")
        continue
    prev, mbit = None, []
    for r in rows:
        t, rx = float(r["ts"]), int(r["wan_rx"])
        if prev and t > prev[0]:
            mbit.append((rx - prev[1]) * 8 / 1e6 / (t - prev[0]))
        prev = (t, rx)
    total = (int(rows[-1]["wan_rx"]) - int(rows[0]["wan_rx"])) / 1e9
    print(f"  {label}: {len(rows)} samples {fmt(first)}-{fmt(last)} | internet in {total:.2f} GB"
          f" | peak {max(mbit):.0f} Mbit/s"
          f" | peak sockets-to-443 {max(int(r['est443']) for r in rows)}"
          f" | peak tracked-flows {max(int(r['ct']) for r in rows)}")
if not glob.glob(os.path.join(d, "net-*.tsv")):
    print("  no network traces in this run directory")
PY

echo
echo "--- errors ---"
if [[ -s "$CELLS" ]]; then
  python3 - "$CELLS" <<'PY'
import collections, re, sys
pat = re.compile(r"\[ERROR\]|Pipeline failed|AnalysisException|PATH_NOT_FOUND|Mkdirs failed"
                 r"|Failed to rename|UnknownHost|Checkpoint block .* not found|ExecutorLostFailure"
                 r"|heartbeat timed out|OutOfMemory|maximum pool size exceeded", re.I)
# One dead executor produces hundreds of near-identical "Lost task" lines. Sorting
# by count alone puts that chatter on top and pushes the one line that says why the
# run died off the end, so those lines are ranked last.
noise = re.compile(r"Lost task \d|TaskKilled|Killing all running tasks|ShuffleMapStage \d+ ")
counts, first = collections.Counter(), {}
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not pat.search(line) or "SecurityManager" in line:
        continue
    key = re.sub(r"\d+", "#", line)[:150]
    counts[key] += 1
    first.setdefault(key, line)
if not counts:
    print("  none")
ranked = sorted(counts.items(), key=lambda kv: (bool(noise.search(first[kv[0]])), -kv[1]))
for key, n in ranked[:14]:
    print(f"  x{n:<6} {first[key][:170]}")
PY
else
  echo "  not known -- no executed notebook to read"
fi

echo
echo "--- executors: GPU pool pressure and stalls ---"
echo "  (the memory-pool lines exist only in executor stderr; the driver never sees them)"
# A wedged executor logs nothing for minutes and the last thing it wrote is a
# burst of pool errors. "last work" far below "shutdown" is that wedge.
APP="$(ls -S "$RD"/eventlog/app-*.zstd 2>/dev/null | head -1 | xargs -r basename | sed 's/\.zstd.*//')"
NODES="${PYG_RUN_NODES:-$(sed -n 's/^export PYG_STAGE_NODES=//p' "$RD/env.sh" 2>/dev/null | tr -d "\"'")}"
if [[ -z "$APP" ]]; then
  echo "  no event log, so no app id to look up"
else
  echo "  app $APP"
  SCAN='for f in '"$SPARK_WORK_DIR"'/'"$APP"'/*/stderr; do
    [ -f "$f" ] || continue
    n=$(grep -ac "maximum pool size exceeded" "$f")
    last=$(grep -aE "^[0-9]{2}/[0-9]{2}/[0-9]{2} " "$f" | grep -avE "commanded a shutdown|SIGNAL TERM" | tail -1 | cut -c1-17)
    rmm=$(grep -aoE "^\[[0-9-]+ [0-9:]{8}" "$f" | tail -1 | tr -d "[")
    end=$(grep -aE "commanded a shutdown|SIGNAL TERM" "$f" | tail -1 | cut -c1-17)
    printf "    exec %-3s %8s pool-exceeded | last spark line %s | last pool line %s | shutdown %s\n" \
      "$(basename "$(dirname "$f")")" "$n" "${last:-none}" "${rmm:-none}" "${end:-none}"
  done'
  echo "  $(hostname -s) (local)"
  bash -c "$SCAN" 2>/dev/null || echo "    unreadable"
  for h in $(echo "$NODES" | tr ',' ' '); do
    ip -4 -o addr show 2>/dev/null | grep -q " $h/" && continue   # that is this box
    echo "  $h"
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "$SCAN" 2>/dev/null \
      || echo "    unreachable, or the worker has already rotated the logs away"
  done
fi

if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
  echo
  echo "--- what landed in the work directory ---"
  du -sh "$WORK_DIR" 2>/dev/null | sed 's/^/  /'
  # The parentheses matter: without them -not binds only to the parquet branch
  # and half-written .pt files under _temporary get counted as artifacts.
  find "$WORK_DIR" \( -name '*.pt' -o -name '*.parquet' \) \
    -not -path '*_temporary*' 2>/dev/null | wc -l | sed 's/^/  artifact files: /'
  find "$WORK_DIR" -name '*.pt' -not -path '*_temporary*' \
    -printf '  %f  %s bytes\n' 2>/dev/null | sort
  echo "  reliable checkpoint dirs left behind: $(find "$WORK_DIR" -type d -name 'rdd-*' 2>/dev/null | wc -l)"
fi

# Per-run acceptance checks. Whatever this run was meant to prove goes here, so
# that this file stays the same from one run to the next.
if [[ -f "$RD/extra-checks.sh" ]]; then
  echo
  # shellcheck source=/dev/null
  source "$RD/extra-checks.sh"
fi
echo "=================================================================="
} > "$OUT" 2>&1

cat "$OUT"
