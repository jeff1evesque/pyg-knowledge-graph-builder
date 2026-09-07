# Per-run acceptance checks. Copy to <run-dir>/extra-checks.sh, keep only what this
# run is meant to prove, and leave it untracked:
#
#     cp bin/profiles/extra-checks.example.sh ~/pyg-runs/issue-NNN/extra-checks.sh
#
# bin/record_run_outcome.sh sources this at the end of the report, so plain echo
# lands in outcome.txt. In scope here:
#
#     RD         the run directory
#     WORK_DIR   where the job wrote its output, or empty
#     CELLS      the notebook's cell output, flattened to one text file
#
# WHY THIS IS SEPARATE. These blocks used to live in the launcher, where they grew to
# about a third of it -- all of them checks for issues that had since merged, copied
# forward into every later run because the launcher was copied forward. A check for
# this run belongs to this run. The launcher and the recorder stay the same.
#
# Set this to the checkout the run used.
REPO="${PYG_REPO_ROOT:-$HOME/path/to/pyg-knowledge-graph-builder}"

# --------------------------------------------------------------------------- #
# Example: does anything precede itself? (issue #360)
#
# The unit tests prove the helper on fixtures; only this says the call sites are
# wired to real data correctly. It reads the enriched Parquet, which the checkpoint
# sweep does not touch, so it still works long after the run.
# --------------------------------------------------------------------------- #
echo
echo "--- #360: does anything still precede itself? ---"
echo "  (before the fix, run 20260901T195728Z: 6,219 -- 5,988 bls + 231 sec)"
ENRICHED="$(find "${WORK_DIR:-/nonexistent}/enriched" -mindepth 2 -maxdepth 3 \
              -type d -name triples 2>/dev/null | head -1)"
if [[ -n "$ENRICHED" ]]; then
  # Local Spark, because the cluster is idle by now and a 14 GB scan does not need
  # it. SPARK_HOME must be hidden: the site config asks for a GPU per task, local
  # mode has no worker to advertise one, and the request is never met -- 0 tasks
  # forever, no error. The test suite hides it the same way, in tests/spark_env.py.
  # This cost 30 minutes on the 2026-09-02 run.
  env -u SPARK_HOME -u SPARK_CONF_DIR SPARK_LOCAL_IP=127.0.0.1 \
      "$REPO/.venv/bin/python" "$REPO/bin/selfloops.py" \
      "$ENRICHED" "$RD/selfloops.json" > "$RD/selfloops.log" 2>&1
  grep -v -E "^[0-9]{2}/[0-9]{2}|WARN |Setting default|To adjust|^\[Stage" "$RD/selfloops.log" \
    | sed '/^$/d' | sed 's/^/  /'
else
  echo "  NOT MEASURED: no enriched Parquet under ${WORK_DIR:-(no work dir)}"
fi

# --------------------------------------------------------------------------- #
# Example: two counts of the same frame should land seconds apart (issue #364)
#
# The shape to copy for anything measured from the job's own log lines: name the
# baseline, print both sides, print the verdict. A check that prints a number
# without the number it is being compared against is not a check.
# --------------------------------------------------------------------------- #
echo
echo "--- #364: how far apart are the two market counts? ---"
echo "  (before the fix, run 20260902T203507Z: 17:54:05 -> 18:13:25 = 19m 20s)"
first=$(grep -a 'Market Intra-Source Enrichment Complete' "$CELLS" | head -1)
second=$(grep -a 'Market enrichment produced' "$CELLS" | head -1)
if [[ -n "$first" && -n "$second" ]]; then
  a=$(sed 's/^ *//' <<<"$first" | cut -c1-19)
  b=$(sed 's/^ *//' <<<"$second" | cut -c1-19)
  gap=$(( $(date -d "$b" +%s) - $(date -d "$a" +%s) ))
  echo "  first  count : $(sed 's/^ *//' <<<"$first")"
  echo "  second count : $(sed 's/^ *//' <<<"$second")"
  echo "  gap          : ${gap}s ($((gap / 60))m $((gap % 60))s)"
  if (( gap <= 120 )); then
    echo "  VERDICT: PASS -- the second count read materialized rows"
  else
    echo "  VERDICT: FAIL -- ${gap}s apart, so it is still recomputing"
  fi
else
  echo "  NOT MEASURED: could not find both market count lines"
fi

# --------------------------------------------------------------------------- #
# Example: how long did each assembly leg's construction phase take? (issue #358)
#
# Phase timings come in pairs of log lines. Walk the marks rather than grepping for
# one of them, so a leg that started and never finished is reported as such instead
# of silently dropping out of the table.
# --------------------------------------------------------------------------- #
echo
echo "--- #358: how long did each assembly leg's PyG CONSTRUCTION take? ---"
echo "  (baseline 52.2 min 2026-08-31, 50.8 min 2026-09-01)"
mapfile -t _marks < <(grep -aoE \
  '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[INFO\] build_graph - PHASE: (PyG CONSTRUCTION|SAVING PyG HETERODATA)' \
  "$CELLS")
_leg=0; _start=""
for _m in "${_marks[@]}"; do
  _ts="${_m:0:19}"
  if [[ "$_m" == *"PyG CONSTRUCTION"* ]]; then
    _start="$_ts"
  elif [[ -n "$_start" ]]; then
    _leg=$((_leg + 1))
    _secs=$(( $(date -d "$_ts" +%s) - $(date -d "$_start" +%s) ))
    printf '  leg %d: %s -> %s = %dm %02ds\n' \
      "$_leg" "$_start" "$_ts" "$((_secs / 60))" "$((_secs % 60))"
    _start=""
  fi
done
if (( _leg == 0 )); then
  echo "  NOT MEASURED: no completed PyG CONSTRUCTION phase"
  [[ -n "$_start" ]] && echo "  (a leg STARTED at $_start and never reached the save)"
fi
