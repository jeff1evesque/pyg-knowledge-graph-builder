"""Pin what `bin/schedule_run.sh` installs, and when it installs nothing.

WHY THIS IS WORTH A TEST
------------------------
A timer installed wrong does not fail where anyone sees it: it never fires, or runs
something other than the schedule it was meant for. And the rendered units are written
outside the repository, where a value that should have stayed in an untracked env.sh
would be easy to miss.

HOW
---
The script runs for real with HOME in a temporary directory, so the units land under
it, and systemctl, loginctl and systemd-analyze are stubs that record what they were
asked. Nothing here touches the systemd manager of the machine running the tests.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "bin" / "schedule_run.sh"
CALENDAR = "Tue..Sat *-*-* 00:30:00"

pytestmark = pytest.mark.launcher


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


class Scheduler:
    """A schedule directory, a fake checkout, a home directory and stub systemd tools."""

    def __init__(self, tmp_path: Path, name: str = "schedule"):
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.sd = tmp_path / name
        self.sd.mkdir()
        self.repo = tmp_path / "repo"
        _stub(self.repo / "bin" / "daily_run.sh", "exit 0\n")
        self.calls = tmp_path / "calls"
        self.calls.mkdir()
        self.bin = tmp_path / "stubbin"
        _stub(self.bin / "systemctl", f'printf "%s\\n" "$*" >> "{self.calls}/systemctl.txt"\n')
        _stub(self.bin / "loginctl", 'echo "${FAKE_LINGER:-yes}"\n')
        _stub(self.bin / "systemd-analyze",
              f'printf "%s\\n" "$*" >> "{self.calls}/systemd-analyze.txt"\n'
              '[ -z "${FAKE_BAD_CALENDAR:-}" ] || '
              '{ echo "Failed to parse calendar specification" >&2; exit 1; }\n')
        self.write_env(f'export PYG_SCHEDULE_ONCALENDAR="{CALENDAR}"\n')

    def write_env(self, schedule_block: str) -> None:
        (self.sd / "env.sh").write_text(
            f'export PYG_REPO_ROOT="{self.repo}"\n'
            'export PYG_WORK_DIR="/srv/work/${RUN_ID:?set RUN_ID first}"\n'
            + schedule_block)

    def run(self, *args: str, **env_changes):
        env = dict(os.environ)
        env.pop("XDG_CONFIG_HOME", None)
        env.update({"PATH": f"{self.bin}{os.pathsep}{env['PATH']}", "HOME": str(self.home)})
        env.update(env_changes)
        return subprocess.run(["bash", str(SCHEDULER), *args, str(self.sd)],
                              env=env, capture_output=True, text=True, timeout=30)

    @property
    def units(self) -> Path:
        return self.home / ".config" / "systemd" / "user"

    def recorded(self, name: str) -> str:
        path = self.calls / f"{name}.txt"
        return path.read_text() if path.exists() else ""


def test_a_dry_run_renders_the_calendar_it_was_given_and_installs_nothing(tmp_path):
    s = Scheduler(tmp_path)

    r = s.run("--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"OnCalendar={CALENDAR}" in r.stdout
    assert "Persistent=true" in r.stdout
    assert f'ExecStart="{s.repo}/bin/daily_run.sh" "{s.sd}"' in r.stdout
    assert "SuccessExitStatus=3" in r.stdout
    assert not s.units.exists()
    assert s.recorded("systemctl") == ""


def test_without_a_calendar_nothing_is_scheduled(tmp_path):
    """Scheduling is opt-in per directory: an env.sh made for a run started by hand,
    with no schedule in it, must not turn into a timer."""
    s = Scheduler(tmp_path)
    s.write_env("")

    for args in ((), ("--dry-run",)):
        r = s.run(*args)
        assert r.returncode == 2
        assert "PYG_SCHEDULE_ONCALENDAR" in r.stderr
    assert not s.units.exists()
    assert s.recorded("systemctl") == ""


def test_the_units_hold_no_value_the_schedule_directory_did_not_give(tmp_path):
    """Every path in them comes from env.sh or from where the units are written, and
    nothing else from env.sh is copied in."""
    s = Scheduler(tmp_path)
    s.write_env(f'export PYG_SCHEDULE_ONCALENDAR="{CALENDAR}"\n'
                "export PYG_SCHEDULE_UNIT=graph-nightly\n")

    r = s.run("--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not re.search(r"@[A-Z_]+@", r.stdout)
    assert "Unit=graph-nightly.service" in r.stdout
    absolute = re.findall(r'(?:^|(?<=[\s"=]))(/[^\s"]+)', r.stdout, flags=re.M)
    assert absolute
    assert all(path.startswith(str(tmp_path)) for path in absolute), absolute
    assert "/srv/work" not in r.stdout
    assert str(REPO_ROOT) not in r.stdout


def test_installing_writes_both_units_and_enables_the_timer(tmp_path):
    s = Scheduler(tmp_path)

    r = s.run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert f'ExecStart="{s.repo}/bin/daily_run.sh"' in (s.units / "pyg-daily.service").read_text()
    assert f"OnCalendar={CALENDAR}" in (s.units / "pyg-daily.timer").read_text()
    calls = s.recorded("systemctl").splitlines()
    assert calls[:2] == ["--user daemon-reload", "--user enable --now pyg-daily.timer"]
    assert "linger" not in r.stdout


def test_a_calendar_systemd_cannot_read_is_refused_before_anything_is_written(tmp_path):
    s = Scheduler(tmp_path)

    r = s.run(FAKE_BAD_CALENDAR="1")
    assert r.returncode == 2
    assert "does not accept" in r.stderr
    assert not s.units.exists()
    assert s.recorded("systemctl") == ""


def test_it_says_so_when_linger_is_off(tmp_path):
    """Without linger a user timer fires only while that user is logged in, which on a
    machine nobody sits at means hardly ever."""
    s = Scheduler(tmp_path)

    r = s.run(FAKE_LINGER="no")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "loginctl enable-linger" in r.stdout


def test_a_unit_name_that_is_not_plain_is_refused(tmp_path):
    s = Scheduler(tmp_path)
    s.write_env(f'export PYG_SCHEDULE_ONCALENDAR="{CALENDAR}"\n'
                'export PYG_SCHEDULE_UNIT="../elsewhere"\n')

    assert s.run("--dry-run").returncode == 2


def test_a_percent_sign_in_a_path_reaches_the_unit_doubled(tmp_path):
    """systemd reads % as the start of a specifier, so a single one would be replaced."""
    s = Scheduler(tmp_path, name="100%-daily")

    r = s.run("--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert '100%%-daily"' in r.stdout
