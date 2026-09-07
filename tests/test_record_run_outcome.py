"""Pin which network traces `bin/record_run_outcome.sh` will summarise.

WHY THIS IS WORTH A TEST
------------------------
The report's network numbers were wrong for two days and said nothing about it. A
stop flag was cleared on one node but set on both, so the second node's sampler
exited as the run began, and the end-of-run copy brought the previous attempt's
file back under the name this run expected. The report summarised it: a confident
count of samples, bytes and peak sockets, describing a different run.

There are two defences and they cover different cases. A trace written by the
current launcher carries the run id in its name, which is decisive. Older traces
carry no run id, so they are judged against the run's window, taken from the Spark
event log file names. The window alone is NOT enough: a run directory reused for a
second run still holds the first run's event logs, so the earliest application in
it need not belong to this run.

The other half is that a trace which IS this run's must survive both checks. It
starts before Spark does -- the launcher starts tracing, then submits -- so a rule
that wants every sample inside the window would throw away every good trace.
"""

import datetime
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER = REPO_ROOT / "bin" / "record_run_outcome.sh"

RUN_ID = "20260101T120000Z"
OTHER_RUN = "20251231T235500Z"

# Spark's own name for the application. The recorder reads the run's window from
# it, so the trace timestamps below are derived from the same value rather than
# written down twice -- and stay correct in any timezone.
APP = "app-20260101120500-0001.zstd"
WINDOW = datetime.datetime.strptime("20260101120500", "%Y%m%d%H%M%S").timestamp()
DURING = WINDOW + 60      # while the job was running
BEFORE = WINDOW - 3600    # an hour before it started

pytestmark = pytest.mark.launcher

COLUMNS = ("ts\tiso\twan_rx\twan_tx\twan_rxp\twan_txp\tfab_rx\tfab_tx"
           "\test443\tsyn_sent\ttimewait\ttcp_all\tct")


def _trace(path: Path, start_epoch: float, samples: int = 4):
    lines = [COLUMNS]
    for i in range(samples):
        ts = start_epoch + i
        lines.append(f"{ts:.3f}\t00:00:0{i}\t{1000 + i * 10}\t{2000 + i * 10}"
                     f"\t1\t1\t0\t0\t3\t0\t0\t9\t42")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with an event log naming the run's window, and a run log."""
    rd = tmp_path / "run"
    (rd / "eventlog").mkdir(parents=True)
    (rd / "eventlog" / APP).write_bytes(b"")
    (rd / "run.log").write_text(f"[12:00:00] RUN_ID={RUN_ID}\n")
    return rd


def _report(rd: Path) -> str:
    env = dict(os.environ)
    env["PYG_RUN_NODES"] = ""          # no executor logs to scan
    r = subprocess.run(["bash", str(RECORDER), str(rd)],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return (rd / "outcome.txt").read_text()


def _network_section(text: str) -> str:
    body = text.split("--- network during the run", 1)[1]
    return body.split("--- errors ---", 1)[0]


# --------------------------------------------------------------------------- #
# The run id in the name is decisive
# --------------------------------------------------------------------------- #

def test_a_trace_named_for_another_run_is_not_summarised(run_dir):
    """The exact failure: a leftover file, summarised as this run's."""
    # Sitting inside this run's window, so only the name gives it away.
    _trace(run_dir / f"net-node-b-{OTHER_RUN}.tsv", DURING)
    section = _network_section(_report(run_dir))

    assert f"from run {OTHER_RUN}, not this one" in section
    assert "internet in" not in section


def test_this_run_s_trace_is_summarised_even_though_it_predates_spark(run_dir):
    """Tracing starts before the job is submitted, so its first samples are always
    outside the event log's window. The name settles it; the window must not."""
    _trace(run_dir / f"net-node-a-{RUN_ID}.tsv", BEFORE)   # tracing starts first
    section = _network_section(_report(run_dir))

    assert "internet in" in section
    assert "STALE" not in section


def test_both_kinds_of_trace_are_reported_separately(run_dir):
    """A reused run directory holds both. One line each, and only one summarised."""
    _trace(run_dir / f"net-node-a-{RUN_ID}.tsv", DURING)
    _trace(run_dir / f"net-node-b-{OTHER_RUN}.tsv", DURING)
    section = _network_section(_report(run_dir))

    assert section.count("internet in") == 1
    assert f"net-node-b-{OTHER_RUN}.tsv" in section


# --------------------------------------------------------------------------- #
# Traces with no run id in the name fall back to the window
# --------------------------------------------------------------------------- #

def test_a_legacy_trace_ending_before_the_run_began_is_called_stale(run_dir):
    """The old names carry nothing to match on, so the window is all there is."""
    _trace(run_dir / "net-node2.tsv", BEFORE)             # ends before the app
    section = _network_section(_report(run_dir))

    assert "STALE" in section
    assert "internet in" not in section


def test_a_legacy_trace_covering_the_run_is_still_summarised(run_dir):
    _trace(run_dir / "net-node1.tsv", DURING)
    section = _network_section(_report(run_dir))

    assert "internet in" in section
    assert "STALE" not in section


def test_without_a_recorded_run_id_the_window_still_applies(run_dir):
    """Running the recorder by hand on a directory whose run log is gone."""
    (run_dir / "run.log").unlink()
    _trace(run_dir / f"net-node-b-{OTHER_RUN}.tsv", BEFORE)
    section = _network_section(_report(run_dir))

    assert "STALE" in section
    assert "internet in" not in section
