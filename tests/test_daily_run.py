"""Pin what `bin/daily_run.sh` does with a day: which day it builds, when it skips the
day, and what it may delete afterwards.

WHY THIS IS WORTH A TEST
------------------------
Nobody watches this script run, and two of its possible mistakes would pass unseen:
a prune that removes a run whose upload failed deletes the only copy of that run, and
a late upstream write reported as a failure teaches everyone to ignore the failures.
It also removes files from places runs started by hand use too, so what it may remove
is pinned as closely as what it builds.

HOW
---
The script runs for real against a fake checkout. The launcher, staging, memory and
publish scripts in it are stubs that record what they were asked, and aws, curl, ssh,
git, ip and hostname are stub binaries. The fake S3 is a folder, as in
tests/test_publish_run.py, and the mirror is a folder standing for this node's disk,
so nothing here touches a cluster, a bucket or the network.
"""

import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY = REPO_ROOT / "bin" / "daily_run.sh"

RUN_ID = "20261001T043000Z"
LOCAL = "node-a"
REMOTE = "node-b"
IDLE = json.dumps({"workers": [{"state": "ALIVE"}, {"state": "ALIVE"}], "activeapps": []})

pytestmark = [
    pytest.mark.launcher,
    pytest.mark.skipif(shutil.which("flock") is None,
                       reason="the script takes a per-schedule lock with flock"),
]

STUB_AWS = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.path.join(os.environ["FAKE_CALLS"], "aws.txt"), "a") as fh:
    fh.write(" ".join(args) + "\\n")
if args[:2] != ["s3api", "list-objects-v2"]:
    sys.exit(f"stub aws: no case for {args}")
bucket = Path(os.environ["FAKE_S3"]) / args[args.index("--bucket") + 1]
prefix = args[args.index("--prefix") + 1]
keys = sorted(p.relative_to(bucket).as_posix() for p in bucket.rglob("*")
              if p.is_file()) if bucket.exists() else []
print(json.dumps({"Contents": [{"Key": k} for k in keys if k.startswith(prefix)]}))
'''

# Reads the run directory's env.sh the way bin/run_cluster_notebook.sh does, and leaves
# the files that it and the recorder leave.
STUB_LAUNCHER = '''#!/usr/bin/env bash
printf "%s\\n" "$*" >> "$FAKE_CALLS/launcher.txt"
rd="${@: -1}"
day="$2"
export PYG_RUN_DIR="$rd" PYG_DATA_YEAR="${day:0:4}" PYG_DATA_MONTH="${day:5:2}" PYG_DATA_DAY="${day:8:2}"
. "$rd/env.sh"
printf "%s\\n" "$PYG_SOURCE_PATHS" > "$FAKE_CALLS/launcher-sources.txt"
mkdir -p "$PYG_WORK_DIR"
printf "run id      : %s\\nwork dir    : %s\\n" "$RUN_ID" "$PYG_WORK_DIR" > "$rd/run-config.txt"
echo "${FAKE_RUN_RC:-0}" > "$rd/run.done"
'''

# Stands in for bin/stage_sources.sh on this node: each source it is given appears in
# the mirror, as a download would leave it.
STUB_STAGE = '''#!/usr/bin/env bash
printf "%s\\n" "$*" >> "$FAKE_CALLS/stage.txt"
[ "${FAKE_STAGE_RC:-0}" = 0 ] || exit "$FAKE_STAGE_RC"
IFS=, read -ra uris <<< "$2"
for uri in "${uris[@]}"; do
  rest="${uri#*://}"
  path="$PYG_LOCAL_SOURCE_ROOT/${rest%/}"
  if [[ "$uri" == */ ]]; then
    mkdir -p "$path" && touch "$path/snapshot-01.parquet"
  else
    mkdir -p "${path%/*}" && touch "$path"
  fi
done
'''

# A run counts as listed at the destination once an upload of it has succeeded.
STUB_PUBLISH = '''import os, sys
from pathlib import Path

rd, mode = Path(sys.argv[1]), sys.argv[2]
with open(os.path.join(os.environ["FAKE_CALLS"], "publish.txt"), "a") as fh:
    fh.write(f"{rd.name} {mode}\\n")
if mode == "--upload":
    rc = int(os.environ.get("FAKE_PUBLISH_RC", "0"))
    if rc == 0:
        (rd / "listed").touch()
    sys.exit(rc)
sys.exit(0 if mode == "--published" and (rd / "listed").exists() else 1)
'''

STUB_MEM = '''import os, sys

with open(os.path.join(os.environ["FAKE_CALLS"], "mem.txt"), "a") as fh:
    fh.write(" ".join(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ.get("FAKE_MEM_RC", "0")))
'''


def _script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _stub(path: Path, body: str) -> Path:
    return _script(path, "#!/usr/bin/env bash\n" + body)


class Schedule:
    """A schedule directory, a fake checkout, a work root, a mirror, a fake S3 and stubs."""

    def __init__(self, tmp_path: Path, lag: int = 1, retain: int = 2):
        self.tmp = tmp_path
        self.sd = tmp_path / "schedule"
        self.sd.mkdir()
        self.repo = tmp_path / "repo"
        self.work_root = tmp_path / "work"
        self.work_root.mkdir()
        self.mirror = tmp_path / "mirror"
        self.s3 = tmp_path / "s3"
        self.s3.mkdir()
        self.calls = tmp_path / "calls"
        self.calls.mkdir()
        self.bin = tmp_path / "stubbin"

        _script(self.repo / "bin" / "run_cluster_notebook.sh", STUB_LAUNCHER)
        _script(self.repo / "bin" / "stage_sources.sh", STUB_STAGE)
        _script(self.repo / "bin" / "publish_run.py", STUB_PUBLISH)
        _script(self.repo / "bin" / "mem_reclaim.py", STUB_MEM)
        _stub(self.repo / "runner", "exit 0\n")

        _script(self.bin / "aws", STUB_AWS)
        _stub(self.bin / "curl", 'printf "%s" "${FAKE_MASTER_JSON:-}"\n')
        # The other node: what it is asked to run is recorded, and nothing is run.
        _stub(self.bin / "ssh", 'printf "%s\\n" "$*" >> "$FAKE_CALLS/ssh.txt"\n'
                               '{ cat; echo "--- end of input"; } >> "$FAKE_CALLS/ssh-stdin.txt"\n'
                               'case " $* " in *" python3 "*) exit "${FAKE_MEM_RC:-0}" ;; esac\n')
        _stub(self.bin / "git", 'case " $* " in\n'
                                '  *" status "*) printf "%s" "${FAKE_GIT_STATUS:-}" ;;\n'
                                '  *" rev-parse "*) echo abc1234 ;;\n'
                                '  *) exit 1 ;;\n'
                                'esac\n')
        _stub(self.bin / "ip", f'echo "1: eth0    inet {LOCAL}/24 scope global eth0"\n')
        _stub(self.bin / "hostname", f"echo {LOCAL}\n")

        day = "year=${PYG_DATA_YEAR:?}/month=${PYG_DATA_MONTH}"
        (self.sd / "env.sh").write_text("\n".join([
            f'export PYG_REPO_ROOT="{self.repo}"',
            f'export PYG_WORK_DIR="{self.work_root}/${{RUN_ID:?set RUN_ID first}}"',
            "export SPARK_MASTER_URL=spark://MASTER-HOST:7077",
            f"export PYG_STAGE_NODES={LOCAL},{REMOTE}",
            f'export PYG_RUNNER_PYTHON="{self.repo}/runner"',
            "export PYG_INPUT_MODE=local",
            f'export PYG_LOCAL_SOURCE_ROOT="{self.mirror}"',
            "export PYG_SOURCE_PATHS="
            f'"s3a://bucket/raw/source=a/{day}/${{PYG_DATA_DAY}}.snappy.parquet,'
            f's3a://bucket/quotes/{day}/day=${{PYG_DATA_DAY}}/"',
            'export PYG_YEARLY_SOURCE_PREFIXES="s3a://bucket/raw/source=b/feed=x/"',
            'export PYG_TIME_PERIOD="${PYG_DATA_YEAR}-${PYG_DATA_MONTH}"',
            "export PYG_PUBLISH_ROOT=s3://bucket/runs",
            "export PYG_MEMFREE_GATE_GB=90",
            f"export PYG_SCHEDULE_DATA_LAG_DAYS={lag}",
            f"export PYG_SCHEDULE_RETAIN_RUNS={retain}",
            "export PYG_SCHEDULE_IDLE_WAIT_SECONDS=0",
            "export PYG_SCHEDULE_RETRY_SECONDS=0",
        ]) + "\n")

    def put(self, key: str) -> None:
        path = self.s3 / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    def add_sources(self, day: str, years=(2025,)) -> None:
        """Everything env.sh names for the day: a day file, a day directory, and the
        yearly files."""
        y, m, d = day.split("-")
        self.put(f"bucket/raw/source=a/year={y}/month={m}/{d}.snappy.parquet")
        self.put(f"bucket/quotes/year={y}/month={m}/day={d}/snapshot-01.parquet")
        for year in years:
            self.put(f"bucket/raw/source=b/feed=x/{year}.snappy.parquet")

    def add_old_run(self, run_id: str, published: bool = True, work_dir=None) -> Path:
        """An earlier day's run, as this script and the launcher leave it."""
        rd = self.sd / "runs" / run_id
        rd.mkdir(parents=True)
        work = Path(work_dir) if work_dir is not None else self.work_root / run_id
        work.mkdir(parents=True, exist_ok=True)
        (rd / "run-config.txt").write_text(f"run id      : {run_id}\nwork dir    : {work}\n")
        if published:
            (rd / "listed").touch()
        return work

    def run(self, *args: str, **env_changes):
        env = dict(os.environ)
        env.update({
            "PATH": f"{self.bin}{os.pathsep}{env['PATH']}",
            "HOME": str(self.tmp),
            "RUN_ID": RUN_ID,
            "FAKE_S3": str(self.s3),
            "FAKE_CALLS": str(self.calls),
            "FAKE_MASTER_JSON": IDLE,
            "PYG_SCHEDULE_TODAY": "2026-10-01",
        })
        for key, value in env_changes.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run(["bash", str(DAILY), *args, str(self.sd)],
                              env=env, capture_output=True, text=True, timeout=60)

    def recorded(self, name: str) -> str:
        path = self.calls / f"{name}.txt"
        return path.read_text() if path.exists() else ""

    def sent_to_the_other_node(self) -> list:
        """Each script the other node was sent, one entry per ssh call."""
        return [chunk for chunk in self.recorded("ssh-stdin").split("--- end of input\n")
                if chunk]

    def log(self) -> str:
        path = self.sd / "daily.log"
        return path.read_text() if path.exists() else ""


@pytest.fixture
def schedule(tmp_path):
    def build(**kwargs):
        return Schedule(tmp_path, **kwargs)
    return build


# --------------------------------------------------------------------------- #
# The day
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("today, lag, day", [
    ("2026-10-01", 1, "2026-09-30"),
    ("2026-03-01", 1, "2026-02-28"),
    ("2027-01-01", 3, "2026-12-29"),
], ids=["a-month", "a-short-month", "a-year"])
def test_the_day_is_today_less_the_lag(schedule, today, lag, day):
    s = schedule(lag=lag)
    s.add_sources(day)

    r = s.run(PYG_SCHEDULE_TODAY=today)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"--data-date {day} " in s.recorded("launcher")
    y, m, d = day.split("-")
    assert f"s3a://bucket/quotes/year={y}/month={m}/day={d}/" in s.recorded("launcher-sources")


def test_the_flag_names_the_day_instead(schedule):
    s = schedule()
    s.add_sources("2026-09-09")

    r = s.run("--data-date", "2026-09-09")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--data-date 2026-09-09 " in s.recorded("launcher")


# --------------------------------------------------------------------------- #
# A missing source skips the day
# --------------------------------------------------------------------------- #

def test_a_missing_source_skips_the_day(schedule):
    """Upstream is late, or the market was shut. A failed unit on every holiday would
    leave the failure signal meaning nothing."""
    s = schedule()
    s.put("bucket/raw/source=a/year=2026/month=09/30.snappy.parquet")
    s.put("bucket/raw/source=b/feed=x/2026.snappy.parquet")

    r = s.run()
    assert r.returncode == 3
    assert "SKIPPED" in s.log()
    assert "missing: s3a://bucket/quotes/year=2026/month=09/day=30/" in s.log()
    assert s.recorded("stage") == ""
    assert s.recorded("launcher") == ""
    assert not (s.sd / "runs").exists()


def test_a_key_that_only_starts_with_a_source_file_is_not_that_file(schedule):
    """An upload still in progress, for instance."""
    s = schedule()
    s.add_sources("2026-09-30")
    day_file = s.s3 / "bucket/raw/source=a/year=2026/month=09/30.snappy.parquet"
    day_file.rename(day_file.with_name("30.snappy.parquet.tmp"))

    assert s.run().returncode == 3


def test_a_yearly_prefix_reads_the_newest_year_not_after_the_day(schedule):
    """A feed can stay on last year's file well into the new year, and every feed does
    in January, so the day's own year cannot simply be written into the path."""
    s = schedule()
    s.add_sources("2026-09-30", years=(2024, 2025, 2027))

    r = s.run()
    assert r.returncode == 0, r.stdout + r.stderr
    sources = s.recorded("launcher-sources")
    assert "s3a://bucket/raw/source=b/feed=x/2025.snappy.parquet" in sources
    assert "2027" not in sources
    assert "2025.snappy.parquet" in s.recorded("stage")
    assert "-> 2025.snappy.parquet" in s.log()
    # The run directory's own env.sh names what the run read.
    assert "2025.snappy.parquet" in (s.sd / "runs" / RUN_ID / "env.sh").read_text()


def test_a_yearly_prefix_with_nothing_old_enough_skips_the_day(schedule):
    s = schedule()
    s.add_sources("2026-09-30", years=(2027,))

    assert s.run().returncode == 3
    assert s.recorded("launcher") == ""


# --------------------------------------------------------------------------- #
# The prune: only what this schedule made
# --------------------------------------------------------------------------- #

def test_the_newest_runs_are_kept_and_only_published_older_ones_removed(schedule):
    s = schedule(retain=2)
    s.add_sources("2026-09-30")
    runs = s.sd / "runs"
    newest_older = s.add_old_run("20260930T043000Z")
    unpublished = s.add_old_run("20260928T043000Z", published=False)
    removed = ("20260929T043000Z", "20260927T043000Z")
    for run_id in removed:
        s.add_old_run(run_id)
    never_started = runs / "20260926T043000Z"
    never_started.mkdir()
    (never_started / "stage.log").write_text("staging failed\n")

    r = s.run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert (s.work_root / RUN_ID).is_dir()
    assert newest_older.is_dir()
    assert (runs / "20260930T043000Z").is_dir()
    # Not listed as published, so kept whole, for inspection.
    assert unpublished.is_dir()
    assert (runs / "20260928T043000Z").is_dir()
    for run_id in removed:
        assert not (s.work_root / run_id).exists()
        assert not (runs / run_id).exists()
    assert not never_started.exists()


def test_the_prune_never_removes_anything_outside_the_work_root(schedule, tmp_path):
    s = schedule(retain=1)
    s.add_sources("2026-09-30")
    elsewhere = s.add_old_run("20260929T043000Z",
                              work_dir=tmp_path / "elsewhere" / "20260929T043000Z")
    misnamed = s.add_old_run("20260928T043000Z", work_dir=s.work_root / "20250101T000000Z")
    precious = tmp_path / "precious"
    precious.mkdir()
    linked = s.work_root / "20260927T043000Z"
    linked.symlink_to(precious)
    s.add_old_run("20260927T043000Z", work_dir=linked)

    r = s.run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert elsewhere.is_dir()
    assert misnamed.is_dir()
    assert linked.is_symlink()
    assert precious.is_dir()
    for run_id in ("20260929T043000Z", "20260928T043000Z", "20260927T043000Z"):
        assert (s.sd / "runs" / run_id).is_dir()


def test_a_failing_publish_blocks_the_prune(schedule):
    """The run that could not be uploaded is the only copy of it, and older runs wait
    for a day whose publish went through."""
    s = schedule(retain=1)
    s.add_sources("2026-09-30")
    older = s.add_old_run("20260929T043000Z")

    r = s.run(FAKE_PUBLISH_RC="1")
    assert r.returncode == 1
    assert "FAILED" in s.log()
    assert (s.work_root / RUN_ID).is_dir()
    assert older.is_dir()
    assert (s.sd / "runs" / "20260929T043000Z").is_dir()
    assert "--published" not in s.recorded("publish")


def test_a_run_that_did_not_finish_is_not_published(schedule):
    s = schedule()
    s.add_sources("2026-09-30")

    assert s.run(FAKE_RUN_RC="1").returncode == 1
    assert s.recorded("publish") == ""


def test_only_the_local_copies_this_schedule_downloaded_are_removed(schedule):
    """The mirror is shared with runs started by hand. A copy a node already held when
    the schedule staged is not the schedule's to remove, even once no day reads it."""
    s = schedule()
    s.add_sources("2026-09-30")
    s.add_sources("2026-10-01")
    by_hand = s.mirror / "bucket/raw/source=a/year=2026/month=09/30.snappy.parquet"
    by_hand.parent.mkdir(parents=True)
    by_hand.write_bytes(b"staged by a run started by hand")

    first = s.run(PYG_SCHEDULE_TODAY="2026-10-01", RUN_ID="20261001T043000Z")
    assert first.returncode == 0, first.stdout + first.stderr
    downloaded = s.mirror / "bucket/quotes/year=2026/month=09/day=30"
    assert downloaded.is_dir()
    assert f"{LOCAL}\t{downloaded}" in (s.sd / "mirror-downloads.tsv").read_text()

    second = s.run(PYG_SCHEDULE_TODAY="2026-10-02", RUN_ID="20261002T043000Z")
    assert second.returncode == 0, second.stdout + second.stderr
    assert not downloaded.exists()
    assert not downloaded.parent.exists()          # left empty, so removed too
    assert by_hand.read_bytes() == b"staged by a run started by hand"
    # Read by both days, so kept.
    assert (s.mirror / "bucket/raw/source=b/feed=x/2025.snappy.parquet").exists()
    # The other node is sent the same removal.
    removals = [chunk for chunk in s.sent_to_the_other_node() if "rm -rf" in chunk]
    assert len(removals) == 1
    assert "day=30" in removals[0]


# --------------------------------------------------------------------------- #
# Refused before the run starts
# --------------------------------------------------------------------------- #

def test_a_checkout_with_uncommitted_changes_is_refused(schedule):
    """The assembly leg packages the checkout over an hour into the run, so an edit
    in it would publish a graph that no commit reproduces."""
    s = schedule()
    s.add_sources("2026-09-30")

    r = s.run(FAKE_GIT_STATUS=" M spark_jobs/build_graph.py\n")
    assert r.returncode == 2
    assert "uncommitted" in s.log()
    assert s.recorded("aws") == ""
    assert s.recorded("launcher") == ""


def test_a_notebook_kernel_that_is_gone_is_refused(schedule):
    s = schedule()
    s.add_sources("2026-09-30")
    _stub(s.repo / "runner",
          'echo "the notebook kernel pyg-notebook-runner runs /gone/python, '
          'which is not there" >&2\nexit 1\n')

    r = s.run()
    assert r.returncode == 2
    assert "/gone/python" in s.log()


def test_a_busy_cluster_is_refused(schedule):
    s = schedule()
    s.add_sources("2026-09-30")
    busy = json.dumps({"workers": [{"state": "ALIVE"}, {"state": "ALIVE"}],
                       "activeapps": [{"id": "app-1"}]})

    r = s.run(FAKE_MASTER_JSON=busy)
    assert r.returncode == 2
    assert "not idle" in s.log()
    assert s.recorded("stage") == ""
    assert s.recorded("launcher") == ""


def test_the_memory_gate_runs_on_every_node(schedule):
    s = schedule()
    s.add_sources("2026-09-30")

    assert s.run().returncode == 0
    assert s.recorded("mem").splitlines() == ["90"]
    assert f"{REMOTE} python3" in s.recorded("ssh")


def test_memory_that_will_not_come_free_is_refused_after_three_tries(schedule):
    s = schedule()
    s.add_sources("2026-09-30")

    assert s.run(FAKE_MEM_RC="1").returncode == 2
    assert len(s.recorded("mem").splitlines()) == 3
    assert s.recorded("launcher") == ""


def test_a_second_copy_refuses_to_share_the_schedule(schedule):
    s = schedule()
    lock = open(s.sd / ".daily.lock", "w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        r = s.run()
    finally:
        lock.close()
    assert r.returncode == 2
    assert "not starting a second" in r.stderr


# --------------------------------------------------------------------------- #
# --check: how a new schedule directory is tried before a timer runs it
# --------------------------------------------------------------------------- #

def test_check_names_what_the_day_reads_and_does_nothing_else(schedule):
    s = schedule(retain=1)
    s.add_sources("2026-09-30")
    older = s.add_old_run("20260101T043000Z")

    r = s.run("--check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reads s3a://bucket/quotes/year=2026/month=09/day=30/" in s.log()
    assert "reads s3a://bucket/raw/source=b/feed=x/2025.snappy.parquet" in s.log()
    for name in ("stage", "launcher", "publish", "mem", "ssh"):
        assert s.recorded(name) == "", name
    assert not (s.sd / "runs" / RUN_ID).exists()
    assert not (s.sd / "mirror-downloads.tsv").exists()
    assert older.is_dir()


def test_check_says_when_a_source_is_not_there(schedule):
    s = schedule()
    s.put("bucket/raw/source=a/year=2026/month=09/30.snappy.parquet")
    s.put("bucket/raw/source=b/feed=x/2025.snappy.parquet")

    assert s.run("--check").returncode == 3
    assert "missing: s3a://bucket/quotes/year=2026/month=09/day=30/" in s.log()
