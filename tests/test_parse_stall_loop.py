"""Pin the properties that make `bin/parse_stall_loop.sh` an instrument rather than a script.

WHY THESE AND NOT OTHERS
------------------------
The loop exists to answer one question -- how often does the parse deadlock, and does
a candidate fix move that number (#388). Everything it does is in service of a rate
being meaningful, and each way that rate can quietly stop being meaningful is a bug
that does not announce itself:

1. A capture that outlives its trial. `run_cluster_notebook.sh` had exactly this: a
   stall directory shared across attempts, so a capture an earlier attempt left behind
   ended the next run before it began. Here it would be worse -- every trial after the
   first would read as stalled, and the rate would come out 100%.

2. A work directory shared between trials, so trial 2 reads trial 1's manifest and
   reports its parse time as its own.

3. Killing the job it caught. The evidence only exists while the job is up: ~4 MB
   queued in both directions of the loopback socket, and a Python worker py-spy can
   still be pointed at. A loop that tidies up after itself destroys the thing it was
   built to collect.

4. Aborting the whole loop when one node is short of memory. chain.sh does abort, and
   on 2026-09-05 that ended a run at 88.2G against a 90G ask. Twenty trials in, that
   is the instrument switching itself off overnight.

5. Re-staging the sources every trial, which would pull every source from object
   storage once per node, twenty times over.

HOW
---
The loop is run for real against stubs: a fake submit_spark_job.sh that records its
arguments and can pretend to stall, a fake mem_reclaim.py, and stubbed ssh/pkill/ip.
Nothing here starts Spark, reclaims memory, or touches a cluster.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOP = REPO_ROOT / "bin" / "parse_stall_loop.sh"

LOCAL = "node-a"
REMOTE = "node-b"

pytestmark = [
    pytest.mark.launcher,
    pytest.mark.skipif(
        shutil.which("flock") is None,
        reason="the loop takes a per-run-directory lock with flock",
    ),
]


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


class Loop:
    """A run directory, a fake checkout, and stubs for everything that leaves the box."""

    def __init__(self, tmp_path: Path, submit_body: str, reclaim_rc: int = 0):
        self.tmp = tmp_path
        self.rd = tmp_path / "run"
        self.rd.mkdir()
        self.repo = tmp_path / "repo"
        self.calls = tmp_path / "calls"
        self.calls.mkdir()
        self.scratch = tmp_path / "scratch"
        self.scratch.mkdir()

        _stub(self.repo / "bin" / "submit_spark_job.sh", submit_body)
        # Run with python3 on both nodes, so it has to be Python.
        reclaim = self.repo / "bin" / "mem_reclaim.py"
        reclaim.write_text(
            "import sys\n"
            f'open(r"{self.calls}/reclaim.txt", "a").write(" ".join(sys.argv[1:]) + "\\n")\n'
            'print("MemFree 91.0G -> 91.0G, target %sG" % sys.argv[1])\n'
            f"sys.exit({reclaim_rc})\n"
        )

        (self.rd / "env.sh").write_text(
            f'export PYG_REPO_ROOT="{self.repo}"\n'
            f'export PYG_WORK_DIR="{tmp_path}/work/${{RUN_ID:?}}"\n'
            f'export PYG_STAGE_NODES="{LOCAL},{REMOTE}"\n'
            'export PYG_SOURCE_PATHS="s3a://bucket/raw/source=bls/feed=cpi/2026.snappy.parquet"\n'
            'export PYG_TIME_PERIOD=2026-09\n'
            'export PYG_SOURCE_FORMAT=turtle_parquet\n'
            'export PYG_PARQUET_PARTITIONS=200\n'
            # Never let the sweep reach a real Spark scratch directory.
            f'export SPARK_LOCAL_DIRS="{self.scratch}"\n'
        )

        self.bin = tmp_path / "stubbin"
        rec = f'printf "%s\\n" "$*" >> "{self.calls}"'
        _stub(self.bin / "ssh", f'{rec}/ssh.txt\ntrue\n')
        _stub(self.bin / "pkill", f'{rec}/pkill.txt\ntrue\n')
        _stub(self.bin / "ip", f'echo "1: eth0    inet {LOCAL}/24 scope global eth0"\n')
        _stub(self.bin / "hostname", f'echo {LOCAL}\n')
        _stub(self.bin / "git", 'echo stub\n')

    def run(self, trials: int = 1, timeout: int = 120, args=None):
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        env["HOME"] = str(self.tmp)
        env["PYG_TRIAL_TIMEOUT"] = "20"
        argv = args if args is not None else [str(self.rd), str(trials)]
        return subprocess.run(
            ["bash", str(LOOP), *argv],
            env=env, capture_output=True, text=True, timeout=timeout,
        )

    def recorded(self, name: str) -> str:
        path = self.calls / f"{name}.txt"
        return path.read_text() if path.exists() else ""

    def trials(self) -> list:
        path = self.rd / "trials.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]


# A submit that writes the manifest the loop reads its numbers out of, then exits 0.
CLEAN_SUBMIT = """
printf "%s\\n" "$*" >> "{calls}/submit.txt"
echo "PYG_STAGE_ENABLED=${{PYG_STAGE_ENABLED:-unset}}" >> "{calls}/submit_env.txt"
echo "PYG_STALL_DUMP_DIR=${{PYG_STALL_DUMP_DIR:-unset}}" >> "{calls}/submit_env.txt"
echo "GPU_PER_TASK=${{GPU_PER_TASK:-unset}}" >> "{calls}/submit_env.txt"
mkdir -p "$PYG_WORK_DIR/manifests/year=2026/month=09"
cat > "$PYG_WORK_DIR/manifests/year=2026/month=09/parse_only_1.json" <<'JSON'
{{"result": {{"parse_seconds": 68.7, "initial_triples": 19600000}}}}
JSON
exit 0
"""

# A submit that stalls: it drops a capture where the watchdog would have, then keeps
# running -- which is what a deadlocked parse actually does.
STALL_SUBMIT = """
printf "%s\\n" "$*" >> "{calls}/submit.txt"
mkdir -p "$PYG_STALL_DUMP_DIR/stall-20260101T000001Z-stage13"
sleep 60
"""

# A submit that returns before the clock ticks, so several trials share one second.
INSTANT_SUBMIT = """
printf "%s\\n" "$*" >> "{calls}/submit.txt"
exit 0
"""


@pytest.fixture
def loop(tmp_path):
    def build(submit=CLEAN_SUBMIT, reclaim_rc=0):
        calls = tmp_path / "calls"
        return Loop(tmp_path, submit.format(calls=calls), reclaim_rc=reclaim_rc)
    return build


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def test_refuses_without_a_run_directory(loop):
    assert loop().run(args=[]).returncode == 2


def test_refuses_a_run_directory_without_an_env_file(loop):
    h = loop()
    (h.rd / "env.sh").unlink()
    result = h.run()
    assert result.returncode == 2
    assert "env.sh" in result.stderr


# --------------------------------------------------------------------------- #
# What it submits
# --------------------------------------------------------------------------- #

def test_submits_parse_only(loop):
    """Any other mode would run enrichment too, and cost forty minutes a trial."""
    h = loop()
    h.run(trials=1)
    assert "--mode parse_only" in h.recorded("submit")


def test_does_not_restage_the_sources(loop):
    """Staging defaults to on. Left alone it re-pulls every source, once per node, per trial."""
    h = loop()
    h.run(trials=1)
    assert "PYG_STAGE_ENABLED=false" in h.recorded("submit_env")


def test_submits_at_the_seed_legs_sizing(loop):
    """Without the seed profile a trial runs at the defaults, which is a different experiment.

    GPU_PER_TASK decides how many parse tasks share the GPU -- 64 slots at the seed
    leg's 0.03125 against 8 at the 0.125 default. Concurrency across the JVM/Python
    boundary is the most likely thing the race turns on, so a trial at the wrong
    sizing measures something else and reports it as a parse trial.
    """
    h = loop()
    (h.rd / "seed-profile.env").write_text("export GPU_PER_TASK=0.03125\n")
    with open(h.rd / "env.sh", "a") as env:
        env.write(f'export PYG_SEED_PROFILE="{h.rd}/seed-profile.env"\n')

    h.run(trials=1)

    assert "GPU_PER_TASK=0.03125" in h.recorded("submit_env")


def test_reclaims_memory_on_every_node_before_each_trial(loop):
    """MemFree at executor start sizes the RMM pool, so it is an input to the trial."""
    h = loop()
    h.run(trials=2)
    assert h.recorded("reclaim").count("90") == 2, "local reclaim did not run once per trial"
    assert h.recorded("ssh").count("mem_reclaim") == 0, "remote reclaim is piped, not pathed"
    assert h.recorded("ssh").count("python3 -") >= 2, "no remote reclaim"


# --------------------------------------------------------------------------- #
# Failure 1: a capture must not outlive its trial
# --------------------------------------------------------------------------- #

def test_each_trial_gets_its_own_stall_directory(loop):
    """Shared, one capture would read as a stall on every later trial and the rate would be 100%."""
    h = loop()
    h.run(trials=3)
    dirs = sorted(p.name for p in (h.rd / "stalls").glob("trial-*"))
    assert len(dirs) == 3, f"expected one dump dir per trial, got {dirs}"
    assert len(set(dirs)) == 3


def test_a_capture_left_by_an_earlier_trial_does_not_end_a_later_one(loop):
    """The bug this class of harness has actually had (#386)."""
    h = loop()
    stale = h.rd / "stalls" / "trial-000-old" / "stall-20250101T000000Z-stage13"
    stale.mkdir(parents=True)

    result = h.run(trials=2)

    assert result.returncode == 0, "a stale capture ended the loop"
    assert [row["verdict"] for row in h.trials()] == ["clean", "clean"]


# --------------------------------------------------------------------------- #
# Failure 2: trials must not read each other's numbers
# --------------------------------------------------------------------------- #

def test_each_trial_gets_its_own_work_dir(loop):
    h = loop()
    h.run(trials=2)
    work_dirs = {
        line.split("--local_work_dir ")[1].split()[0]
        for line in h.recorded("submit").splitlines()
    }
    assert len(work_dirs) == 2, f"trials shared a work dir: {work_dirs}"


def test_trials_inside_one_second_still_get_their_own_work_dir(loop):
    """The trial id is second-resolution, and trials can be faster than that.

    A submit that returns immediately is not contrived -- it is what a misconfigured
    run does, and what every skipped trial does. With a timestamp-only id these land
    in the same work directory and the second reads the first's manifest.
    """
    h = loop(submit=INSTANT_SUBMIT)
    h.run(trials=3)
    work_dirs = {
        line.split("--local_work_dir ")[1].split()[0]
        for line in h.recorded("submit").splitlines()
    }
    assert len(work_dirs) == 3, f"fast trials collided: {work_dirs}"


def test_records_the_parse_seconds_from_the_manifest(loop):
    """Out of the manifest, not scraped from a log line whose format is not a contract."""
    h = loop()
    h.run(trials=1)
    row = h.trials()[0]
    assert row["verdict"] == "clean"
    assert row["parse_seconds"] == 68.7
    assert row["triples"] == 19600000


# --------------------------------------------------------------------------- #
# Failure 3: a caught job must be left alive
# --------------------------------------------------------------------------- #

def test_a_caught_stall_stops_the_loop_without_killing_the_job(loop):
    """The socket queues and the worker stack die with the process. Leave it up."""
    h = loop(submit=STALL_SUBMIT)

    result = h.run(trials=5, timeout=120)

    assert result.returncode == 99
    verdicts = [row["verdict"] for row in h.trials()]
    assert verdicts == ["stalled"], f"loop kept going after a capture: {verdicts}"
    assert h.recorded("pkill") == "", "the loop killed the job it caught"


def test_a_caught_stall_does_not_sweep_scratch(loop):
    """Sweeping under a live executor takes its scratch out from under it."""
    h = loop(submit=STALL_SUBMIT)
    marker = h.scratch / "spark-live"
    marker.mkdir()

    h.run(trials=2, timeout=120)

    assert marker.exists(), "swept scratch while the caught job was still running"


def test_a_clean_trial_does_sweep_scratch(loop):
    """Twenty trials leave twenty dead scratch dirs; one node had let 101 GB build up."""
    h = loop()
    marker = h.scratch / "spark-dead"
    marker.mkdir()

    h.run(trials=1)

    assert not marker.exists(), "dead scratch was never swept"


# --------------------------------------------------------------------------- #
# Failure 4: a short node skips a trial, it does not end the loop
# --------------------------------------------------------------------------- #

def test_a_node_short_of_memory_skips_the_trial_and_keeps_going(loop):
    """chain.sh aborts here, and that ended a real run at 88.2G against a 90G ask."""
    h = loop(reclaim_rc=1)

    result = h.run(trials=3)

    assert result.returncode == 0
    verdicts = [row["verdict"] for row in h.trials()]
    assert verdicts == ["skipped", "skipped", "skipped"]
    assert h.recorded("submit") == "", "ran a trial at an unknown MemFree"


def test_skipped_trials_are_not_counted_as_clean(loop):
    """A skipped trial that read as clean would make any fix look better than it is."""
    h = loop(reclaim_rc=1)
    h.run(trials=2)
    assert all(row["verdict"] == "skipped" for row in h.trials())


# --------------------------------------------------------------------------- #
# The record itself
# --------------------------------------------------------------------------- #

def test_one_line_per_trial_each_with_a_verdict(loop):
    h = loop()
    h.run(trials=3)
    rows = h.trials()
    assert len(rows) == 3
    assert [row["trial"] for row in rows] == [1, 2, 3]
    assert all(row["verdict"] for row in rows)


def test_summary_reports_the_stall_rate(loop):
    h = loop()
    result = h.run(trials=2)
    assert "stall rate: 0/2" in result.stdout
