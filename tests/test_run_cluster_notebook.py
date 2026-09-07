"""Pin the three silent failures that put `bin/run_cluster_notebook.sh` under version control.

WHY THIS IS WORTH A TEST
------------------------
This launcher used to exist only as a copy inside each run directory, copied forward
from whichever run came before. On one run -- 20260906T184507Z -- that produced three
separate failures, none of which announced itself:

1. The stop flag was cleared on one node but set on two, so every run after the first
   began with the flag still set on the second machine. That machine's trace ended
   before the run began, and the end-of-run copy brought it back as this run's. The
   report then printed a confident set of network numbers describing a different run,
   and had been doing so for two days.

2. The checkpoint sweep was keyed on a literal work path carrying one run's name.
   Every copy of the harness carried the PREVIOUS run's name there, so the sweep
   matched nothing and deleted nothing until somebody edited the line. One run left
   462 GB behind that way.

3. Nothing was recorded when the harness gave up. The recorder ran only on the
   success path, so a run that aborted on a false stall -- 69 minutes before the job
   actually finished, successfully -- produced no report at all.

Each of those is one line's worth of code and none of them fails loudly, which is
exactly the shape a test catches and a reader does not.

HOW
---
The launcher is run for real against stub binaries: ssh, scp, ping and pkill are shell
scripts that record their arguments, and the notebook runner is a stub that exits with
whatever code the case needs. PYG_REPO_ROOT points at a fake checkout holding stubs for
the other harness pieces, so nothing here starts Spark, touches a cluster, or signals a
process on the machine running the tests.

The lock case is here for the same reason as the rest: two supervisors sharing one run
directory is what let a killed attempt's cleanup overwrite a live run's report.
"""

import fcntl
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "bin" / "run_cluster_notebook.sh"

RUN_ID = "20260101T000000Z"
LOCAL = "node-a"
REMOTE = "node-b"

pytestmark = [
    pytest.mark.launcher,
    pytest.mark.skipif(shutil.which("flock") is None,
                       reason="the launcher takes a per-run-directory lock with flock"),
]


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


class Harness:
    """A run directory, a fake checkout of the harness, and stubs for everything else."""

    def __init__(self, tmp_path: Path, work_dir: str, runner_body: str):
        self.tmp = tmp_path
        self.rd = tmp_path / "run"
        self.rd.mkdir()
        self.repo = tmp_path / "repo"
        self.calls = tmp_path / "calls"
        self.calls.mkdir()
        self.work_dir = work_dir

        # The other tracked harness pieces, stubbed: each records that it ran and with
        # what, so the assertions are about what the launcher asked for.
        _stub(self.repo / "bin" / "record_run_outcome.sh",
              f'printf "%s\\n" "$@" >> "{self.calls}/record.txt"\n')
        _stub(self.repo / "bin" / "cluster_sampler.sh",
              f'printf "%s\\n" "$@" >> "{self.calls}/sampler.txt"\n')
        # Launched with python3, so it has to be Python -- the launcher runs it the
        # same way on every node.
        netsample = self.repo / "bin" / "netsample.py"
        netsample.write_text(
            "import sys\n"
            f'open(r"{self.calls}/netsample.txt", "a").write(" ".join(sys.argv[1:]) + "\\n")\n'
            'open(sys.argv[1], "a").close()\n'
        )
        _stub(self.repo / "bin" / "execute_notebook.py", "true\n")
        _stub(self.repo / "runner", runner_body)

        (self.rd / "env.sh").write_text(
            f'export PYG_REPO_ROOT="{self.repo}"\n'
            f'export PYG_WORK_DIR="{work_dir}"\n'
            f'export PYG_STAGE_NODES="{LOCAL},{REMOTE}"\n'
            f'export PYG_RUNNER_PYTHON="{self.repo}/runner"\n'
            f'export PYG_NOTEBOOK="{self.repo}/notebook.ipynb"\n'
        )

        self.bin = tmp_path / "stubbin"
        rec = f'printf "%s\\n" "$*" >> "{self.calls}"'
        _stub(self.bin / "ssh", f'{rec}/ssh.txt\ntrue\n')
        # A copy that lands, so the launcher's move off the .part file is exercised.
        _stub(self.bin / "scp", f'{rec}/scp.txt\ntouch "${{@: -1}}"\n')
        # Never signal anything on the machine running the tests.
        _stub(self.bin / "pkill", f'{rec}/pkill.txt\ntrue\n')
        # A healthy gateway, so the network guard never fires. No address is printed:
        # the launcher only passes it to ping, which is stubbed too.
        _stub(self.bin / "ping", 'echo "64 bytes: icmp_seq=1 ttl=64 time=0.2 ms"\n')
        _stub(self.bin / "ip", 'case "$1" in\n'
                               '  route) echo "default via GATEWAY dev eth0" ;;\n'
                               f'  *) echo "1: eth0    inet {LOCAL}/24 scope global eth0" ;;\n'
                               'esac\n')
        _stub(self.bin / "hostname", f'echo {LOCAL}\n')

    def run(self, run_id: str = RUN_ID, args=None, timeout: int = 90):
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        env["RUN_ID"] = run_id
        env["HOME"] = str(self.tmp)
        return subprocess.run(
            ["bash", str(LAUNCHER), *(args if args is not None else [str(self.rd)])],
            env=env, capture_output=True, text=True, timeout=timeout,
        )

    def recorded(self, name: str) -> str:
        path = self.calls / f"{name}.txt"
        return path.read_text() if path.exists() else ""


@pytest.fixture
def harness(tmp_path):
    def build(work_dir=None, runner_body="exit 0\n"):
        wd = work_dir if work_dir is not None else str(tmp_path / "work" / RUN_ID)
        h = Harness(tmp_path, wd, runner_body)
        (Path(wd) / "checkpoints").mkdir(parents=True)
        return h
    return build


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def test_refuses_without_a_run_directory(harness):
    h = harness()
    assert h.run(args=[]).returncode == 2


def test_refuses_a_run_directory_without_an_env_file(harness):
    h = harness()
    (h.rd / "env.sh").unlink()
    r = h.run()
    assert r.returncode == 2
    assert "env.sh" in r.stderr


# --------------------------------------------------------------------------- #
# Failure 1: a stop flag one run can set on another
# --------------------------------------------------------------------------- #

def test_stop_flag_and_traces_carry_the_run_id(harness):
    """Nothing this run writes may be named the same as another run's.

    The shared `STOP` file is what let a killed attempt's cleanup stop the live run's
    samplers, and the shared `net-node2.tsv` is what let the dead attempt's trace be
    copied back and summarised as this run's.
    """
    h = harness()
    assert h.run().returncode == 0

    assert (h.rd / f"STOP-{RUN_ID}").exists()
    assert not (h.rd / "STOP").exists()
    assert (h.rd / f"net-{LOCAL}-{RUN_ID}.tsv").exists()

    # The remote sampler is told the same per-run names, and the same flag is what
    # stops it at the end.
    ssh = h.recorded("ssh")
    assert f"net-{REMOTE}-{RUN_ID}.tsv" in ssh
    assert f"touch '{h.rd}/STOP-{RUN_ID}'" in ssh


def test_stray_samplers_are_swept_on_every_node_before_the_run(harness):
    """A supervisor that was SIGKILLed sets no flag at all, so clearing one is not enough.

    The sweep pattern must not match the shell that carries it: over ssh the remote
    command line contains the pattern itself, and an unguarded one kills that shell
    before it reaches the rest of the command.
    """
    h = harness()
    assert h.run().returncode == 0

    assert "[n]etsample" in h.recorded("pkill")
    assert "[n]etsample" in h.recorded("ssh")


def test_the_local_node_is_not_treated_as_remote(harness):
    """This box is in PYG_STAGE_NODES, and must not be ssh'd into to trace itself."""
    h = harness()
    assert h.run().returncode == 0

    assert f" {LOCAL} " not in h.recorded("ssh")
    assert f" {REMOTE} " in h.recorded("ssh")
    assert f"net-{LOCAL}-{RUN_ID}.tsv" in h.recorded("netsample")


# --------------------------------------------------------------------------- #
# Failure 2: a checkpoint sweep keyed on a name that goes stale on copy
# --------------------------------------------------------------------------- #

def test_checkpoints_are_swept_when_the_work_dir_is_this_run_s(harness):
    h = harness()
    checkpoints = Path(h.work_dir) / "checkpoints"
    assert h.run().returncode == 0
    assert not checkpoints.exists()


def test_checkpoints_survive_a_work_dir_that_is_not_this_run_s(tmp_path, harness):
    """The guard is a path match, not trust in the variable.

    If the work directory does not end in the run id this invocation generated, it
    belongs to some other run and must not be deleted -- which is also what stops a
    copied harness from sweeping the directory named in the copy.
    """
    h = harness(work_dir=str(tmp_path / "work" / "20250101T000000Z"))
    checkpoints = Path(h.work_dir) / "checkpoints"
    assert h.run().returncode == 0
    assert checkpoints.exists()


# --------------------------------------------------------------------------- #
# Failure 3: nothing recorded when the harness gives up
# --------------------------------------------------------------------------- #

def test_outcome_is_recorded_on_the_success_path(harness):
    h = harness()
    assert h.run().returncode == 0
    assert h.recorded("record").splitlines() == [str(h.rd), h.work_dir]
    assert (h.rd / "run.done").read_text().strip() == "0"


def test_outcome_is_recorded_when_a_stall_is_captured(harness):
    """The run whose harness gave up is the run whose report is worth having."""
    h = harness(runner_body="sleep 30\n")
    (h.rd / "stalls").mkdir(exist_ok=True)
    (h.rd / "stalls" / "stall-20260101T000001Z-stage9").mkdir()

    r = h.run()
    assert r.returncode == 99
    assert (h.rd / "run.done").read_text().strip() == "99"
    assert h.recorded("record").splitlines() == [str(h.rd), h.work_dir]


def test_a_captured_stall_leaves_the_checkpoints_alone(harness):
    """The job is still running on this path and may still read them."""
    h = harness(runner_body="sleep 30\n")
    (h.rd / "stalls").mkdir(exist_ok=True)
    (h.rd / "stalls" / "stall-20260101T000001Z-stage9").mkdir()
    checkpoints = Path(h.work_dir) / "checkpoints"

    assert h.run().returncode == 99
    assert checkpoints.exists()


def test_a_failing_notebook_still_records_its_outcome(harness):
    h = harness(runner_body="exit 1\n")
    assert h.run().returncode == 0        # the launcher reports, it does not fail
    assert (h.rd / "run.done").read_text().strip() == "1"
    assert h.recorded("record").splitlines() == [str(h.rd), h.work_dir]


# --------------------------------------------------------------------------- #
# One supervisor per run directory
# --------------------------------------------------------------------------- #

def test_a_second_supervisor_refuses_to_share_a_run_directory(harness):
    """Two of them sharing one is what overwrote a live run's report with another's."""
    h = harness()
    lock = open(h.rd / ".lock", "w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        r = h.run()
    finally:
        lock.close()
    assert r.returncode == 2
    assert "already using" in r.stderr
