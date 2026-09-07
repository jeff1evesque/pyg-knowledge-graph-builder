"""Pin the reading of /proc/meminfo in `bin/mem_reclaim.py`.

WHY THIS IS WORTH A TEST
------------------------
RAPIDS sizes its pool from MemFree, not MemAvailable, and on a unified-memory host
those are different numbers by more than 100 GB: a finished run leaves its file cache
behind, the kernel holds it until something asks, and MemFree sits near zero for hours.
A run launched into that gap on 2026-09-03 came up with a 337 MB pool instead of 24 GB
and died.

This tool exists to close the gap before the next run starts, and a launcher gates on
its exit status. Reading the wrong key, or the wrong unit, gives a gate that passes
when it should not -- which looks exactly like no gate at all. The unit is the easy one
to get wrong: /proc/meminfo is in kB and every decision here is in GB.

The allocation loop itself is not exercised. Asking the kernel for a hundred gigabytes
in a test worker is not a test, and the loop's own stopping conditions are visible in
the printed line the launcher reads.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "bin" / "mem_reclaim.py"

PROC_MEMINFO = """\
MemTotal:       121000000 kB
MemFree:          1300000 kB
MemAvailable:   115000000 kB
Buffers:            50000 kB
Cached:         100000000 kB
SwapTotal:        8000000 kB
SwapFree:         7000000 kB
"""


def _load_tool():
    """Import by path -- bin/ is scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("mem_reclaim_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def test_meminfo_is_read_in_gigabytes(tool, tmp_path):
    """kB in the file, GB in every comparison. This is the 1000x the gate rests on."""
    path = tmp_path / "meminfo"
    path.write_text(PROC_MEMINFO)
    mem = tool.meminfo(str(path))

    assert mem["MemFree"] == pytest.approx(1.3)
    assert mem["MemAvailable"] == pytest.approx(115.0)
    assert mem["MemTotal"] == pytest.approx(121.0)


def test_free_and_available_are_kept_apart(tool, tmp_path):
    """The whole tool exists because they are not the same number. Reading
    MemAvailable where MemFree is meant makes it report success and do nothing."""
    path = tmp_path / "meminfo"
    path.write_text(PROC_MEMINFO)
    mem = tool.meminfo(str(path))
    assert mem["MemFree"] < mem["MemAvailable"]


def test_swap_used_is_derivable(tool, tmp_path):
    """The sampler reports swap from the same two keys; they have to parse."""
    path = tmp_path / "meminfo"
    path.write_text(PROC_MEMINFO)
    mem = tool.meminfo(str(path))
    assert mem["SwapTotal"] - mem["SwapFree"] == pytest.approx(1.0)


def test_a_target_already_met_allocates_nothing_and_succeeds(tmp_path):
    """Exit 0 without touching a page, so a launcher can gate on this unconditionally."""
    r = subprocess.run([sys.executable, str(TOOL), "0"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert "nothing to do" in r.stdout


def test_usage_is_rejected_rather_than_guessed(tmp_path):
    r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2
    assert "usage:" in r.stderr
