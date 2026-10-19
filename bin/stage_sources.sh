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
# A LOCAL STORE OF EACH DAY'S FILES
#
#   bin/stage_sources.sh ... --store /mnt/drive/pyg-sources --day 2026-10-16 --save
#
# The store is a folder, on a drive of the host that runs this, that keeps each day's
# files in <store>/day=YYYY-MM-DD/<bucket>/<key>. Beside them, _sources.json lists each
# source with its file count and total size. It is written after the files, so a day
# folder without it is incomplete.
#
# A source that list names, whose files still match it, comes from the store: this
# host copies it, and the other nodes copy it from this host over ssh. S3 is not
# asked. Any other source is downloaded from S3 as before. With --save, a source this
# host downloaded, or already held, is copied into the store and added to the list,
# and the other nodes copy it from this host instead of downloading it again. When a
# store is set, this host goes first.
#
# A new day comes from S3, so it gets the newest yearly files. A day staged again
# later gets the files it got the first time, even a yearly file S3 has overwritten
# since. --no-store reads nothing from the store, for a day upstream fixed after it
# was saved; with --save, the fresh download replaces the saved copy. With no --day,
# or no store folder (a drive that is not mounted), staging prints a note, downloads
# from S3 and saves nothing.
#
# Environment fallbacks for the flags, so a run profile can carry them:
#   PYG_LOCAL_SOURCE_ROOT, PYG_STAGE_NODES, PYG_STAGE_CONCURRENCY,
#   PYG_STAGE_RTT_LIMIT_MS, PYG_STAGE_VERIFY, PYG_STAGE_FORCE,
#   PYG_SOURCE_STORE_ROOT (--store), PYG_SOURCE_STORE_SAVE=true (--save),
#   PYG_SOURCE_STORE_SKIP=true (--no-store)
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
STORE="${PYG_SOURCE_STORE_ROOT:-}"
STORE_SAVE="${PYG_SOURCE_STORE_SAVE:-false}"
STORE_SKIP="${PYG_SOURCE_STORE_SKIP:-false}"
DAY=""
DRY_RUN=false
SOURCES=()
staged_skipped=0
from_store=0
saved_sources=0

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
    --store)        STORE="$2"; shift 2 ;;
    --day)          DAY="$2"; shift 2 ;;
    --save)         STORE_SAVE=true; shift ;;
    --no-store)     STORE_SKIP=true; shift ;;
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

for setting in "PYG_SOURCE_STORE_SAVE=$STORE_SAVE" "PYG_SOURCE_STORE_SKIP=$STORE_SKIP"; do
  if [[ "${setting#*=}" != true && "${setting#*=}" != false ]]; then
    echo "ERROR: ${setting%%=*} must be true or false, got '${setting#*=}'." >&2
    exit 2
  fi
done
if [[ -n "$STORE" && "$STORE" != /?* ]]; then
  echo "ERROR: --store must be an absolute path below /, got '$STORE'." >&2
  exit 2
fi
STORE="${STORE%/}"
if [[ -n "$DAY" ]] && ! { [[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
                          && [[ "$(date -d "$DAY" +%F 2>/dev/null)" == "$DAY" ]]; }; then
  echo "ERROR: --day must be a real date written YYYY-MM-DD, got '$DAY'." >&2
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

# One walk of a path: its file count and total size. That pair is how a copy is
# judged complete, in the store and on every node alike.
local_tally_cmd() {
  printf 'find %q -type f -printf "%%s\\n" 2>/dev/null | awk "{n++; b+=\\$1} END{print n+0, b+0}"' "$1"
}

node_tally() {   # node_tally <node> <path>
  if is_local_node "$1"; then
    bash -c "$(local_tally_cmd "$2")"
  else
    ssh -o BatchMode=yes "$1" "bash -c $(printf '%q' "$(local_tally_cmd "$2")")" 2>/dev/null || echo "0 0"
  fi
}

# ---------------------------------------------------------------------------
# The store: each day's files, kept on this host and read before S3. See the
# USAGE notes at the top.
# ---------------------------------------------------------------------------
DAY_DIR=""          # <store>/day=<day>, when the store is in use
USE_STORE=false     # copy what the day's list names instead of downloading it
SAVE=false          # save what this host downloads or already holds
save_failed=""
LOCAL_NODE=""
for node in "${NODE_LIST[@]}"; do
  if is_local_node "$node"; then LOCAL_NODE="$node"; break; fi
done
if [[ -n "$STORE" && -z "$DAY" ]]; then
  echo "NOTE: no --day given, so the store at $STORE is not used." >&2
  echo "      Every source comes from S3, and nothing is saved." >&2
elif [[ -n "$STORE" && ! -d "$STORE" ]]; then
  echo "NOTE: the store $STORE is not there. Is its drive mounted?" >&2
  echo "      Every source comes from S3, and nothing is saved." >&2
elif [[ -n "$STORE" ]]; then
  command -v python3 >/dev/null || { echo "ERROR: the store needs python3 on PATH" >&2; exit 2; }
  DAY_DIR="$STORE/day=$DAY"
  $STORE_SKIP || USE_STORE=true
  if $STORE_SAVE && [[ -z "$LOCAL_NODE" ]]; then
    echo "NOTE: this host is not one of the nodes, so nothing is saved to the store." >&2
  elif $STORE_SAVE && [[ ! -w "$STORE" ]]; then
    echo "NOTE: $STORE is not writable by $(id -un), so nothing is saved to it." >&2
  else
    SAVE="$STORE_SAVE"
  fi

  # A copy from the store replaces a node's copy whole, so a path that could point
  # outside its own folder is refused before anything is removed.
  for uri in "${SOURCES[@]}"; do
    key="$(local_path_for "$uri")"
    IFS='/' read -ra parts <<< "${key#"$DEST"/}"
    for part in "${parts[@]}"; do
      if [[ -z "$part" || "$part" == . || "$part" == .. ]]; then
        echo "ERROR: $uri has an empty, '.' or '..' part, so the store cannot hold it." >&2
        exit 2
      fi
    done
  done

  # This host first, so the other nodes can copy from it what it saves.
  if [[ -n "$LOCAL_NODE" ]]; then
    ordered=("$LOCAL_NODE")
    for node in "${NODE_LIST[@]}"; do
      [[ "$node" == "$LOCAL_NODE" ]] || ordered+=("$node")
    done
    NODE_LIST=("${ordered[@]}")
  fi
fi

# The day's list: "<files> <bytes>" for each <bucket>/<key> it names.
declare -A LISTED=()
if $USE_STORE && [[ -f "$DAY_DIR/_sources.json" ]]; then
  listed="$(python3 - "$DAY_DIR/_sources.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as fh:
        sources = json.load(fh)["sources"]
    lines = [f"{uri.split('://', 1)[-1].rstrip('/')}\t{int(entry['files'])} {int(entry['bytes'])}"
             for uri, entry in sources.items()]
except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
    sys.exit(f"NOTE: {sys.argv[1]} cannot be read ({exc!r}), so nothing comes from it")
print("\n".join(lines))
PY
)" || true
  while IFS=$'\t' read -r key counts; do
    if [[ -n "$key" ]]; then LISTED["$key"]="$counts"; fi
  done <<< "$listed"
fi

# Names source $1 in the day's list with its file count $2 and total size $3, or with
# no counts takes it out. The list is replaced whole, so it is never seen half written.
list_source() {
  python3 - "$DAY" "$DAY_DIR/_sources.json" "$@" <<'PY'
import json
import os
import sys
import tempfile

day, path, uri, *counts = sys.argv[1:]


def where(u):
    return u.split("://", 1)[-1].rstrip("/")


try:
    with open(path) as fh:
        sources = dict(json.load(fh)["sources"])
except FileNotFoundError:
    sources = {}
except (OSError, ValueError, KeyError, TypeError):
    # Unreadable: start a new list. What it named is saved again when next downloaded.
    sources = {}
if not counts and not any(where(k) == where(uri) for k in sources):
    sys.exit()
sources = {k: v for k, v in sources.items() if where(k) != where(uri)}
if counts:
    sources[uri] = {"files": int(counts[0]), "bytes": int(counts[1])}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".sources-", suffix=".json")
with os.fdopen(fd, "w") as fh:
    json.dump({"day": day, "sources": dict(sorted(sources.items()))}, fh, indent=2)
    fh.write("\n")
    fh.flush()
    os.fsync(fh.fileno())
os.chmod(tmp, 0o644)
os.replace(tmp, path)
PY
}

# Saves source $1 from this host's copy. The list drops the source before its files
# change and names it again only once they are all in place, so a save that stops
# partway leaves it out. A save that fails stops the saving, not the staging: the
# other nodes then download the source themselves.
save_to_store() {
  local i="$1" target saved
  $SAVE && ! ${FROM_STORE[$i]} && ! $DRY_RUN || return 0
  target="$(local_path_for "${SOURCES[$i]}")"
  saved="$DAY_DIR/${target#"$DEST"/}"
  if list_source "${SOURCES[$i]}" \
      && rm -rf -- "$saved" \
      && mkdir -p -- "$(dirname -- "$saved")" \
      && cp -R --preserve=timestamps -- "$target" "$saved" \
      && [[ "$(bash -c "$(local_tally_cmd "$saved")")" == "${REMOTE_OBJECTS[$i]} ${REMOTE_BYTES[$i]}" ]] \
      && sync -f -- "$saved" \
      && list_source "${SOURCES[$i]}" "${REMOTE_OBJECTS[$i]}" "${REMOTE_BYTES[$i]}"; then
    FROM_STORE[i]=true
    saved_sources=$((saved_sources + 1))
    echo "    saved to the store"
  else
    echo "WARNING: could not save ${SOURCES[$i]} to $DAY_DIR, so nothing more is saved." >&2
    echo "         The other nodes download it from S3." >&2
    rm -rf -- "$saved" 2>/dev/null || true
    SAVE=false
    save_failed="${SOURCES[$i]}"
  fi
}

# Copies source $2 from the store onto node $1, replacing whatever copy the node
# held, then checks the copy against the day's list.
copy_from_store() {
  local node="$1" i="$2" target key
  target="$(local_path_for "${SOURCES[$i]}")"
  key="${target#"$DEST"/}"
  echo "    copying from the store"
  if is_local_node "$node"; then
    rm -rf -- "$target" && mkdir -p -- "$(dirname -- "$target")" \
      && cp -R --preserve=timestamps -- "$DAY_DIR/$key" "$target"
  else
    # shellcheck disable=SC2029  # the remote command is built here on purpose
    tar -C "$DAY_DIR" -cf - "$key" | ssh -o BatchMode=yes "$node" \
      "rm -rf -- $(printf '%q' "$target") && mkdir -p -- $(printf '%q' "$DEST") && tar -C $(printf '%q' "$DEST") -xf -"
  fi || { echo "ERROR: copying $key from the store to $node failed." >&2; exit 1; }
  if [[ "$(node_tally "$node" "$target")" != "${REMOTE_OBJECTS[$i]} ${REMOTE_BYTES[$i]}" ]]; then
    echo "ERROR: the copy of $key on $node does not match $DAY_DIR/_sources.json." >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Transfer. Nodes run one at a time on purpose: two nodes syncing at once puts
# twice the load on the one link this is trying to protect.
# ---------------------------------------------------------------------------
echo "staging ${#SOURCES[@]} source(s) to ${#NODE_LIST[@]} node(s)"
echo "  destination root : $DEST"
echo "  concurrency      : $CONCURRENCY request(s) per transfer, one node at a time"
echo "  gateway watchdog : ${GATEWAY:-(disabled)} at ${RTT_LIMIT_MS}ms"
if [[ -n "$DAY_DIR" ]]; then
  if $USE_STORE; then reads="read first"; else reads="not read (--no-store)"; fi
  if $SAVE; then saves="downloads saved"; else saves="nothing saved"; fi
  echo "  store            : $DAY_DIR, $reads, $saves"
fi
echo

# Preflight: one LIST per source, before any transfer. It costs a single
# request, prints how much is about to move so a long sync does not look hung,
# and fails here rather than mid-copy when credentials or the prefix are wrong.
gb() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1000000000}'; }
total_bytes=0
in_store=0
REMOTE_OBJECTS=()
REMOTE_BYTES=()
IS_OBJECT=()
FROM_STORE=()
for i in "${!SOURCES[@]}"; do
  uri="${SOURCES[$i]}"
  s3_uri="s3://${uri#*://}"
  FROM_STORE[i]=false

  # In the store when the day's list names it and its files there still match. The
  # count and size then come from the list, and S3 is not asked.
  if $USE_STORE; then
    key="$(local_path_for "$uri")"
    key="${key#"$DEST"/}"
    if [[ -n "${LISTED[$key]:-}" ]] \
        && [[ "$(bash -c "$(local_tally_cmd "$DAY_DIR/$key")")" == "${LISTED[$key]}" ]]; then
      FROM_STORE[i]=true
      IS_OBJECT[i]=false
      REMOTE_OBJECTS[i]="${LISTED[$key]% *}"
      REMOTE_BYTES[i]="${LISTED[$key]#* }"
      in_store=$((in_store + 1))
      total_bytes=$((total_bytes + REMOTE_BYTES[i]))
      printf '  %s\n    %s objects, %s GB (in the store)\n' \
        "$s3_uri" "${REMOTE_OBJECTS[$i]}" "$(gb "${REMOTE_BYTES[$i]}")"
      continue
    fi
  fi

  # A source path may name one object rather than a prefix -- four of the five
  # sources in a full run are single .parquet files. The distinction has to be
  # made here because `aws s3 sync` on an exact object key downloads NOTHING and
  # still exits 0: it treats the key as a prefix, computes an empty relative
  # path for the one object it finds, and copies nothing. Staging would report
  # success having transferred 13% of the data.
  rest="${uri#*://}"
  if aws s3api head-object --bucket "${rest%%/*}" --key "${rest#*/}" \
       >/dev/null 2>&1; then
    IS_OBJECT[i]=true
  else
    IS_OBJECT[i]=false
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
  REMOTE_OBJECTS[i]="$objects"
  REMOTE_BYTES[i]="$bytes"
  total_bytes=$((total_bytes + bytes))
  if ${IS_OBJECT[$i]}; then kind="single object"; else kind="prefix"; fi
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
already_staged() {
  local node="$1" target="$2" want_objects="$3" want_bytes="$4"
  $FORCE && return 1
  [[ "$(node_tally "$node" "$target")" == "$want_objects $want_bytes" ]]
}

for node in "${NODE_LIST[@]}"; do
  echo "== $node =="
  remote_cfg=""
  on_this_host=false
  if is_local_node "$node"; then on_this_host=true; fi

  for i in "${!SOURCES[@]}"; do
    uri="${SOURCES[$i]}"
    s3_uri="s3://${uri#*://}"
    target="$(local_path_for "$uri")"
    echo "  ${s3_uri}"
    echo "    -> ${target}"

    if already_staged "$node" "$target" "${REMOTE_OBJECTS[$i]}" "${REMOTE_BYTES[$i]}"; then
      echo "    already staged (${REMOTE_OBJECTS[$i]} objects, $(gb "${REMOTE_BYTES[$i]}") GB) -- nothing to download"
      staged_skipped=$((staged_skipped + 1))
      # Matches S3's count and size, so it is saved as a download would be.
      if $on_this_host; then save_to_store "$i"; fi
      continue
    fi

    if $DRY_RUN; then
      echo "    (dry run, nothing transferred)"
      continue
    fi

    if ${FROM_STORE[$i]}; then
      copy_from_store "$node" "$i"
      from_store=$((from_store + 1))
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

    if $on_this_host; then
      mkdir -p "$make_dir"
      AWS_CONFIG_FILE="$AWS_CFG" "${transfer_cmd[@]}" &
    else
      # The transfer settings, sent once to a node that downloads anything at all.
      if [[ -z "$remote_cfg" ]]; then
        remote_cfg="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" 'mktemp')"
        scp -o BatchMode=yes -q "$AWS_CFG" "$node:$remote_cfg"
      fi
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
    if $on_this_host; then save_to_store "$i"; fi
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
elif (( from_store > 0 )); then
  echo "staged. $((expected - staged_skipped)) of $expected node/source pairs transferred, $from_store of them from the store."
else
  echo "staged. $((expected - staged_skipped)) of $expected node/source pairs transferred."
fi
# One line on the store, which bin/daily_run.sh copies into its log.
if [[ -n "$DAY_DIR" ]]; then
  if $USE_STORE; then
    line="store: $DAY_DIR held $in_store of the ${#SOURCES[@]} sources"
  else
    line="store: $DAY_DIR was not read (--no-store)"
  fi
  if [[ -n "$save_failed" ]]; then
    line+="; saved $saved_sources, then could not save $save_failed"
  elif $STORE_SAVE; then
    line+="; saved $saved_sources"
  fi
  echo "$line"
elif [[ -n "$STORE" && -z "$DAY" ]]; then
  echo "store: not used, no --day given"
elif [[ -n "$STORE" ]]; then
  echo "store: $STORE is not there, so nothing was read from it or saved"
fi
echo "pass these to the job:"
echo "  --input_mode local --local_source_root $DEST"
