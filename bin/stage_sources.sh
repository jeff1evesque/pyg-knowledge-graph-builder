#!/usr/bin/env bash
#
# Mirror object-storage source prefixes onto every worker's local disk, so the
# parse phase reads from disk instead of the site uplink.
#
# WHY THIS EXISTS
#
# The Turtle parse is a CPU-only Python UDF, and the job used to hold it to
# eight concurrent tasks because Spark schedules every stage under one resource
# profile and that profile reserves a quarter GPU per task. Removing the
# reservation is correct -- the parse never touches the GPU -- but it takes the
# phase from 8 to 144 concurrent object-storage readers. On a deployment whose
# workers reach the bucket over a site uplink rather than an in-region link,
# that is enough to take the network down: measured twice, the readers filled
# the gateway's flow table (~330 -> ~1,900 tracked flows in 60s) and every node
# lost its default route within a second of the others.
#
# Established sockets were not the problem -- they stayed flat around 155.
# CHURN was: reads slower than fs.s3a.connection.timeout are aborted and
# redialled, which under congestion creates connections faster the more
# congested the link gets.
#
# Staging removes the uplink from the job's critical path entirely. It also
# makes the parse faster than either arm ever measured, because both nodes read
# local NVMe in parallel instead of contending for one pipe.
#
# WHY IT IS NOT A SPARK JOB
#
# A distributed copy is the same failure with a different label. This is one
# sequential transfer per node, with a bounded number of connections, and a
# watchdog on the gateway that kills the transfer if latency starts climbing.
#
# WHAT IT DOES NOT DO
#
# It does not share one copy between nodes. Each worker gets its own, because a
# single shared device would serialise reads that are currently parallel per
# node. The mirrored data is read-only for the duration of a run, so there is
# nothing to keep in sync and no coordination problem.
#
# USAGE
#
#   bin/stage_sources.sh \
#       --dest /srv/pyg-source \
#       --nodes worker-a,worker-b \
#       --source s3a://bucket/prefix/one/ \
#       --source s3a://bucket/prefix/two/
#
#   spark_jobs/build_graph.py ... \
#       --source_paths s3a://bucket/prefix/one/,s3a://bucket/prefix/two/ \
#       --input_mode local \
#       --local_source_root /srv/pyg-source
#
# The job keeps naming the s3a:// URIs. This script and build_graph.py derive
# the same local path from them (<dest>/<bucket>/<key>), so the two input modes
# read the same bytes under the same layout and report the same per-source
# statistics.
#
# Re-running is cheap and safe. A source already on disk is not downloaded again:
# the preflight LIST knows the prefix's object count and total size, and one local
# walk per node decides whether there is anything to fetch. Ten runs against the
# same day cost one transfer and nine listings. Pass --force for a prefix that gets
# rewritten in place rather than added to.
#
# Environment fallbacks for every flag, so a run profile can carry them:
#   PYG_LOCAL_SOURCE_ROOT, PYG_STAGE_NODES, PYG_STAGE_CONCURRENCY,
#   PYG_STAGE_RTT_LIMIT_MS, PYG_STAGE_VERIFY, PYG_STAGE_FORCE
#
# Nothing in this file names a host, a bucket, an address or a path. Keep it
# that way -- it is tracked in a public repository.
set -euo pipefail

DEST="${PYG_LOCAL_SOURCE_ROOT:-}"
NODES="${PYG_STAGE_NODES:-}"
CONCURRENCY="${PYG_STAGE_CONCURRENCY:-4}"
RTT_LIMIT_MS="${PYG_STAGE_RTT_LIMIT_MS:-40}"
VERIFY="${PYG_STAGE_VERIFY:-true}"
FORCE="${PYG_STAGE_FORCE:-false}"
DRY_RUN=false
SOURCES=()
staged_skipped=0

# Reads the USAGE block out of the header rather than repeating it, and finds it
# by content so the two cannot drift apart as this file is edited.
usage() {
  awk '/^# USAGE$/{p=1} /^set -euo/{exit} p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)         DEST="$2"; shift 2 ;;
    --source)       SOURCES+=("$2"); shift 2 ;;
    --sources)      IFS=',' read -ra _s <<< "$2"; SOURCES+=("${_s[@]}"); shift 2 ;;
    --nodes)        NODES="$2"; shift 2 ;;
    --concurrency)  CONCURRENCY="$2"; shift 2 ;;
    --rtt-limit-ms) RTT_LIMIT_MS="$2"; shift 2 ;;
    --verify)       VERIFY=true; shift ;;
    --no-verify)    VERIFY=false; shift ;;
    --force)        FORCE=true; shift ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)      usage 0 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Argument checks. Every one of these fails before a byte moves.
# ---------------------------------------------------------------------------
[[ -n "$DEST" ]] || { echo "ERROR: --dest is required" >&2; exit 2; }
[[ "$DEST" = /* ]] || {
  echo "ERROR: --dest must be an absolute path, got '$DEST'." >&2
  echo "       Every worker opens this path itself, so a relative one would" >&2
  echo "       resolve against whatever directory each executor started in." >&2
  exit 2
}
DEST="${DEST%/}"
if [[ -z "${PRINT_LOCAL_PATH:-}" ]]; then
  [[ ${#SOURCES[@]} -gt 0 ]] || { echo "ERROR: at least one --source is required" >&2; exit 2; }
  command -v aws >/dev/null || { echo "ERROR: the aws CLI is not on PATH" >&2; exit 2; }

  # The mirror is written by whoever runs this and read by whoever the executors
  # run as -- usually not the same account. Both halves fail late and confusingly
  # if they are wrong: an unwritable root fails partway through the first sync,
  # and an unreadable one fails per task once the job is already under way.
  if [[ ! -d "$DEST" ]]; then
    parent="$(dirname "$DEST")"
    if [[ ! -w "$parent" ]]; then
      echo "ERROR: cannot create $DEST -- $parent is not writable by $(id -un)." >&2
      echo "       Create it once on every worker, owned by the account that" >&2
      echo "       runs this script, and readable by the account the Spark" >&2
      echo "       executors run as:" >&2
      echo "         sudo install -d -o $(id -un) -g $(id -gn) -m 755 '$DEST'" >&2
      exit 2
    fi
    mkdir -p "$DEST"
  fi
  [[ -w "$DEST" ]] || { echo "ERROR: $DEST is not writable by $(id -un)" >&2; exit 2; }
fi

if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || (( CONCURRENCY < 1 || CONCURRENCY > 16 )); then
  echo "ERROR: --concurrency must be 1-16, got '$CONCURRENCY'." >&2
  echo "       The ceiling is deliberate. This transfer exists because" >&2
  echo "       unbounded read concurrency took the network down; a knob that" >&2
  echo "       can be set back to unbounded is not a fix." >&2
  exit 2
fi

# Node list defaults to this host alone, which is right for a single-worker
# cluster and obviously wrong for any other -- so it is printed, not assumed.
if [[ -z "$NODES" ]]; then
  NODES="$(hostname)"
  echo "NOTE: --nodes not given, staging only $NODES." >&2
  echo "      Every worker needs its own copy at $DEST. A worker without one" >&2
  echo "      fails its tasks with FileNotFoundError once the job starts." >&2
fi
IFS=',' read -ra NODE_LIST <<< "$NODES"

# ---------------------------------------------------------------------------
# Bounded transfer settings.
#
# The AWS CLI takes max_concurrent_requests from a config file only -- there is
# no flag and no environment variable for it -- so this generates one rather
# than editing the caller's. Credentials are unaffected: they come from the
# environment (an instance-metadata endpoint or a profile), not from here.
# Point PYG_STAGE_AWS_CONFIG at your own file to take this over.
# ---------------------------------------------------------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
AWS_CFG="${PYG_STAGE_AWS_CONFIG:-$WORK/aws-config}"
if [[ -z "${PYG_STAGE_AWS_CONFIG:-}" ]]; then
  cat > "$AWS_CFG" <<EOF
[default]
s3 =
    max_concurrent_requests = ${CONCURRENCY}
    max_queue_size = 128
    multipart_threshold = 128MB
    multipart_chunksize = 32MB
    addressing_style = path
EOF
fi

# Forwarded to remote nodes explicitly: an ssh command runs a non-login shell
# that inherits none of the submitting shell's environment, and credentials
# resolved on this host say nothing about the next one.
FORWARD_ENV=()
for var in AWS_EC2_METADATA_SERVICE_ENDPOINT AWS_DEFAULT_REGION AWS_REGION \
           AWS_PROFILE AWS_ENDPOINT_URL AWS_CA_BUNDLE; do
  if [[ -n "${!var:-}" ]]; then
    FORWARD_ENV+=("$var=${!var}")
  fi
done

# ---------------------------------------------------------------------------
# Gateway watchdog.
#
# The failure this guards against announced itself before it arrived: latency
# to the gateway's own LAN interface went from 1ms to 82ms over five seconds,
# and the route dropped ~60s later. A local-network round trip is sub-millisecond
# when the forwarding plane is healthy, so tens of milliseconds means it is not,
# whatever the cause. Killing the transfer there costs a restart; not killing it
# has twice cost the whole network.
# ---------------------------------------------------------------------------
# No `exit` in these awk programs, and no `grep -q` on a pipe anywhere in this
# file. A reader that stops early closes the pipe while the writer is still going,
# the writer takes SIGPIPE, and `set -o pipefail` turns that into status 141 --
# which `set -e` then treats as a fatal error. It is a race, so it fails
# intermittently: this line killed the script roughly one run in four before the
# test suite caught it. Reading the whole (tiny) output costs nothing.
GATEWAY="$(ip route 2>/dev/null | awk '/^default/ && !seen {print $3; seen=1}')"
if ! command -v ping >/dev/null; then
  # Distinguished from a failing probe on purpose. "No ping binary" must not
  # read as "the network is dying" -- that would abort every transfer on a host
  # without it, which is the opposite of useful.
  echo "WARNING: ping is not on PATH, so the gateway watchdog is disabled." >&2
  echo "         The transfer is still bounded to $CONCURRENCY connections." >&2
  GATEWAY=""
fi

watchdog() {
  local pid="$1" strikes=0
  [[ -n "$GATEWAY" ]] || return 0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    local rtt
    # Same SIGPIPE trap as above, and worse here: a truncated read leaves rtt
    # empty, which this function counts as a strike -- so the race would abort
    # healthy transfers rather than just failing loudly.
    rtt="$(ping -n -c1 -W1 "$GATEWAY" 2>/dev/null \
           | awk -F'[=/ ]' '/time=/ && !seen {print int($(NF-1)); seen=1}')"
    if [[ -z "$rtt" ]] || (( rtt > RTT_LIMIT_MS )); then
      strikes=$((strikes + 1))
      echo "  WATCHDOG: gateway rtt ${rtt:-unreachable}ms (limit ${RTT_LIMIT_MS}ms), strike ${strikes}/3"
      if (( strikes >= 3 )); then
        echo "  WATCHDOG: aborting the transfer to let the network recover." >&2
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
        return 1
      fi
    else
      strikes=0
    fi
  done
  return 0
}

# ---------------------------------------------------------------------------
# s3a://bucket/key -> <dest>/bucket/key, the same mapping staged_local_path()
# makes in spark_jobs/build_graph.py. Both must agree or the job reads nothing.
# ---------------------------------------------------------------------------
local_path_for() {
  local uri="$1" rest="${1#*://}"
  case "$uri" in
    s3a://*|s3n://*|s3://*) echo "$DEST/${rest%/}" ;;
    *) echo "ERROR: --source must be an s3a:// / s3:// URI, got '$uri'" >&2; return 1 ;;
  esac
}

# Print where one URI would be staged, and stop. Exists so the mapping above can
# be tested against staged_local_path() in build_graph.py without a bucket, a
# cluster or a byte of traffic -- the two are only useful if they agree, and
# nothing else in either file would notice if they stopped agreeing.
if [[ -n "${PRINT_LOCAL_PATH:-}" ]]; then
  local_path_for "$PRINT_LOCAL_PATH"
  exit 0
fi

is_local_node() {
  local n="$1" addrs
  [[ "$n" == "$(hostname)" || "$n" == "$(hostname -s)" || "$n" == localhost ]] && return 0
  # Captured first rather than piped into `grep -q`: grep stops at the first
  # match, `ip` takes SIGPIPE, and pipefail makes the function report "remote"
  # for the host it is running on -- which would send it looking for an ssh
  # route to itself.
  addrs="$(ip -o addr show 2>/dev/null)"
  grep -qw "$n" <<< "$addrs"
}

# ---------------------------------------------------------------------------
# Transfer. Nodes run one at a time on purpose: two nodes syncing at once puts
# twice the load on the one link this is trying to protect.
# ---------------------------------------------------------------------------
echo "staging ${#SOURCES[@]} source(s) to ${#NODE_LIST[@]} node(s)"
echo "  destination root : $DEST"
echo "  concurrency      : $CONCURRENCY request(s) per transfer, one node at a time"
echo "  gateway watchdog : ${GATEWAY:-(disabled)} at ${RTT_LIMIT_MS}ms"
echo

# Preflight: one LIST per source, before any transfer. It costs a single
# request, prints how much is about to move so a long sync does not look hung,
# and fails here rather than mid-copy when credentials or the prefix are wrong.
gb() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1000000000}'; }
total_bytes=0
REMOTE_OBJECTS=()
REMOTE_BYTES=()
IS_OBJECT=()
for uri in "${SOURCES[@]}"; do
  s3_uri="s3://${uri#*://}"

  # A source path may name one object rather than a prefix -- four of the five
  # sources in a full run are single .parquet files. The distinction has to be
  # made here because `aws s3 sync` on an exact object key downloads NOTHING and
  # still exits 0: it treats the key as a prefix, computes an empty relative
  # path for the one object it finds, and copies nothing. Staging would report
  # success having transferred 13% of the data.
  rest="${uri#*://}"
  if aws s3api head-object --bucket "${rest%%/*}" --key "${rest#*/}" \
       >/dev/null 2>&1; then
    IS_OBJECT+=(true)
  else
    IS_OBJECT+=(false)
  fi
  summary="$(AWS_CONFIG_FILE="$AWS_CFG" aws s3 ls --recursive --summarize "$s3_uri" 2>&1 | tail -3)" || {
    echo "ERROR: cannot list $s3_uri" >&2
    echo "$summary" >&2
    exit 1
  }
  objects="$(awk '/Total Objects:/{print $3}' <<< "$summary")"
  bytes="$(awk '/Total Size:/{print $3}' <<< "$summary")"
  if [[ -z "${bytes:-}" ]]; then
    echo "ERROR: $s3_uri matched nothing. Check the prefix." >&2
    exit 1
  fi
  REMOTE_OBJECTS+=("$objects")
  REMOTE_BYTES+=("$bytes")
  total_bytes=$((total_bytes + bytes))
  if ${IS_OBJECT[-1]}; then kind="single object"; else kind="prefix"; fi
  printf '  %s\n    %s objects, %s GB (%s)\n' "$s3_uri" "$objects" "$(gb "$bytes")" "$kind"
done
printf '  total per node: %s GB (x%d nodes, transferred one node at a time)\n\n' \
  "$(gb "$total_bytes")" "${#NODE_LIST[@]}"

# ---------------------------------------------------------------------------
# Already staged?
#
# `aws s3 sync` is itself incremental -- it compares size and modification time
# per object and transfers only what differs -- so re-running this never
# re-downloads a day that is already on disk. What it still does is walk the
# whole prefix on both sides to work that out, per node, on every submission.
#
# This skips even that: the preflight LIST already knows how many objects the
# prefix holds and how many bytes they are, so one local walk decides whether
# there is anything to do at all. Matching count AND total size means the mirror
# is complete; the case that gets past it -- same file count, same total bytes,
# different contents -- is what --verify's digest is for.
#
# Immutable snapshots are the assumption. Pass --force (or PYG_STAGE_FORCE=true)
# for a prefix that gets rewritten in place, which puts the per-object
# comparison back in charge.
# ---------------------------------------------------------------------------
local_tally_cmd() {
  printf 'find %q -type f -printf "%%s\\n" 2>/dev/null | awk "{n++; b+=\\$1} END{print n+0, b+0}"' "$1"
}

already_staged() {
  local node="$1" target="$2" want_objects="$3" want_bytes="$4" tally
  $FORCE && return 1
  if is_local_node "$node"; then
    tally="$(bash -c "$(local_tally_cmd "$target")")"
  else
    tally="$(ssh -o BatchMode=yes "$node" "bash -c $(printf '%q' "$(local_tally_cmd "$target")")" 2>/dev/null || echo "0 0")"
  fi
  [[ "$tally" == "$want_objects $want_bytes" ]]
}

for node in "${NODE_LIST[@]}"; do
  echo "== $node =="
  remote_cfg=""
  if ! is_local_node "$node"; then
    remote_cfg="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" 'mktemp')"
    scp -o BatchMode=yes -q "$AWS_CFG" "$node:$remote_cfg"
  fi

  for i in "${!SOURCES[@]}"; do
    uri="${SOURCES[$i]}"
    s3_uri="s3://${uri#*://}"
    target="$(local_path_for "$uri")"
    echo "  ${s3_uri}"
    echo "    -> ${target}"

    if already_staged "$node" "$target" "${REMOTE_OBJECTS[$i]}" "${REMOTE_BYTES[$i]}"; then
      echo "    already staged (${REMOTE_OBJECTS[$i]} objects, $(gb "${REMOTE_BYTES[$i]}") GB) -- nothing to download"
      staged_skipped=$((staged_skipped + 1))
      continue
    fi

    if $DRY_RUN; then
      echo "    (dry run, nothing transferred)"
      continue
    fi

    # `cp` for one object, `sync` for a prefix. `sync` cannot do the first (see
    # the preflight note) and `cp` would not be incremental for the second.
    if ${IS_OBJECT[$i]}; then
      transfer_cmd=(aws s3 cp "$s3_uri" "$target" --only-show-errors)
      make_dir="$(dirname "$target")"
    else
      transfer_cmd=(aws s3 sync "$s3_uri" "$target" --only-show-errors)
      make_dir="$target"
    fi

    if is_local_node "$node"; then
      mkdir -p "$make_dir"
      AWS_CONFIG_FILE="$AWS_CFG" "${transfer_cmd[@]}" &
    else
      # shellcheck disable=SC2029  # the remote command is built here on purpose
      ssh -o BatchMode=yes "$node" \
        "mkdir -p $(printf '%q' "$make_dir") && env AWS_CONFIG_FILE='$remote_cfg' ${FORWARD_ENV[*]} \
         ${transfer_cmd[*]@Q}" &
    fi
    sync_pid=$!

    if ! watchdog "$sync_pid"; then
      wait "$sync_pid" 2>/dev/null || true
      echo >&2
      echo "ABORTED: the gateway degraded during the transfer." >&2
      echo "  Lower --concurrency (currently $CONCURRENCY) and run again." >&2
      echo "  aws s3 sync is incremental, so a retry resumes rather than restarts." >&2
      exit 1
    fi
    wait "$sync_pid"
  done

  if [[ -n "$remote_cfg" ]]; then
    ssh -o BatchMode=yes "$node" "rm -f '$remote_cfg'" || true
  fi
  echo
done

if $DRY_RUN; then
  echo "dry run complete."
  exit 0
fi

# ---------------------------------------------------------------------------
# Verification.
#
# Spark lists the input on the driver and then every task opens the path on
# whichever node it landed on. Divergent copies therefore do not fail loudly --
# they produce a graph built from whatever each node happened to hold. A digest
# over names and contents is the cheap way to know that did not happen; it runs
# against local disk, so it costs no network at all.
# ---------------------------------------------------------------------------
if $VERIFY; then
  echo "== verifying =="
  digest_cmd="cd '$DEST' && find . -type f | LC_ALL=C sort \
    | xargs -r -d '\n' sha256sum | sha256sum | cut -c1-16"
  first_digest=""
  first_node=""
  for node in "${NODE_LIST[@]}"; do
    if is_local_node "$node"; then
      d="$(bash -c "$digest_cmd")"
    else
      d="$(ssh -o BatchMode=yes "$node" "bash -c \"$digest_cmd\"")"
    fi
    echo "  $node : $d"
    if [[ -z "$first_digest" ]]; then
      first_digest="$d"; first_node="$node"
    elif [[ "$d" != "$first_digest" ]]; then
      echo >&2
      echo "ERROR: $node does not hold the same data as $first_node." >&2
      echo "  Spark assigns splits from the driver's listing and each task" >&2
      echo "  opens the path locally, so a run against divergent copies does" >&2
      echo "  not fail -- it silently builds a graph from a mixture." >&2
      echo "  Re-run staging for $node before submitting." >&2
      exit 1
    fi
  done
  echo "  all nodes match"
  echo
fi

expected=$(( ${#SOURCES[@]} * ${#NODE_LIST[@]} ))
if (( staged_skipped == expected )); then
  echo "nothing to download -- every source was already on every node."
else
  echo "staged. $((expected - staged_skipped)) of $expected node/source pairs transferred."
fi
echo "pass these to the job:"
echo "  --input_mode local --local_source_root $DEST"
