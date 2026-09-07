"""Pin how `bin/cluster_sampler.sh` finds the cluster, and when it stops.

WHY THIS IS WORTH A TEST
------------------------
The sampler is the only record of how busy the cluster was on the failure path: the
driver UI belongs to the driver process and goes away with it, taking the executor and
host-memory numbers with it. It has already gone missing twice by being a per-run copy
that nobody remembered to bring across, and the launcher logs "cluster sampler started"
either way -- so a run reports telemetry it does not have.

Now that it is tracked, the two things that can still make it useless are pinned here:
the addresses it polls, which are derived rather than written down, and the stop file,
which has to be this run's and not a name shared with every other run in the directory.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLER = REPO_ROOT / "bin" / "cluster_sampler.sh"

pytestmark = pytest.mark.launcher


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture
def sampler(tmp_path):
    """Run the sampler for exactly one pass and report what it polled."""
    rd = tmp_path / "run"
    rd.mkdir()
    calls = tmp_path / "calls"
    calls.mkdir()
    stop = rd / "STOP-20260101T000000Z"

    # One pass, then stop: the stub records the URL it was given and sets the flag.
    _stub(tmp_path / "bin" / "curl",
          f'printf "%s\\n" "$*" >> "{calls}/curl.txt"\n'
          f'touch "{stop}"\n'
          'echo "{}"\n')
    # Invoked as `python -c <code> <driver-ui>`; only the last argument is ours.
    _stub(tmp_path / "bin" / "python-stub",
          f'printf "%s\\n" "${{@: -1}}" >> "{calls}/python.txt"\n'
          'cat > /dev/null\n')

    def run(stop_file=None, **env_overrides):
        env = dict(os.environ)
        env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
        env["PYG_PYTHON"] = str(tmp_path / "bin" / "python-stub")
        env["PYG_SAMPLE_SECONDS"] = "0"
        for key in ("PYG_SPARK_MASTER_UI", "PYG_SPARK_DRIVER_UI",
                    "SPARK_MASTER_URL", "SPARK_DRIVER_HOST"):
            env.pop(key, None)
        env.update({k: v for k, v in env_overrides.items() if v is not None})
        args = [str(rd)] + ([stop_file] if stop_file else [])
        return subprocess.run(["bash", str(SAMPLER), *args],
                              env=env, capture_output=True, text=True, timeout=60)

    return type("S", (), {"run": staticmethod(run), "rd": rd, "stop": stop,
                          "polled": staticmethod(
                              lambda name: (calls / f"{name}.txt").read_text()
                              if (calls / f"{name}.txt").exists() else "")})


# --------------------------------------------------------------------------- #
# Where it polls
# --------------------------------------------------------------------------- #

def test_both_uis_are_derived_from_the_job_s_own_settings(sampler):
    """Nothing here names a host: the addresses come from the variables the job is
    already configured with, which is what lets this file be tracked at all."""
    r = sampler.run(str(sampler.stop),
                    SPARK_MASTER_URL="spark://master-host:7077",
                    SPARK_DRIVER_HOST="driver-host")
    assert r.returncode == 0, r.stderr
    assert "http://master-host:8080/json/" in sampler.polled("curl")
    assert sampler.polled("python").strip() == "http://driver-host:4040"


def test_an_explicit_ui_wins_over_the_derived_one(sampler):
    """The UI does not always sit on the default port beside the master."""
    r = sampler.run(str(sampler.stop),
                    SPARK_MASTER_URL="spark://master-host:7077",
                    PYG_SPARK_MASTER_UI="http://elsewhere:9999")
    assert r.returncode == 0, r.stderr
    assert "http://elsewhere:9999/json/" in sampler.polled("curl")


def test_it_refuses_to_start_with_no_master_to_poll(sampler):
    """Silently sampling nothing for three hours is the failure this replaces."""
    r = sampler.run(str(sampler.stop))
    assert r.returncode == 2
    assert "master" in r.stderr


def test_it_refuses_without_a_run_directory(sampler):
    r = subprocess.run(["bash", str(SAMPLER)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "usage:" in r.stderr


# --------------------------------------------------------------------------- #
# When it stops
# --------------------------------------------------------------------------- #

def test_the_stop_file_argument_wins_over_the_old_shared_name(sampler):
    """A shared STOP in the run directory must not stop a sampler given its own.

    That is what let a killed attempt's cleanup end the live run's telemetry.
    """
    (sampler.rd / "STOP").touch()
    r = sampler.run(str(sampler.stop), SPARK_MASTER_URL="spark://master-host:7077")
    assert r.returncode == 0, r.stderr
    assert sampler.polled("curl"), "the shared STOP stopped a sampler given its own"


def test_a_flag_that_is_already_set_stops_it_before_the_first_poll(sampler):
    sampler.stop.touch()
    r = sampler.run(str(sampler.stop), SPARK_MASTER_URL="spark://master-host:7077")
    assert r.returncode == 0, r.stderr
    assert sampler.polled("curl") == ""


def test_it_falls_back_to_the_shared_name_when_given_no_flag(sampler):
    """Older run directories, and anyone running this by hand, still work."""
    (sampler.rd / "STOP").touch()
    r = sampler.run(None, SPARK_MASTER_URL="spark://master-host:7077")
    assert r.returncode == 0, r.stderr
    assert sampler.polled("curl") == ""
