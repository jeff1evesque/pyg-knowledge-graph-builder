"""
Pin the stuck-versus-slow decision in `bin/stall_watchdog.py`.

WHY THIS IS WORTH A TEST
------------------------
The watchdog exists to capture a thread dump while a stall is live, because on
2026-09-06 seven runs stalled and that dump was never taken -- the window closed
before anyone decided it was worth taking. Its expensive failure is therefore a
false NEGATIVE: if the detector does not fire, the run is killed, the JVM goes
away and the evidence is gone for good. A false positive costs a few hundred KB
of dumps nobody reads.

That asymmetry is why the "slow task must not trip it" case is here too. A
detector that fires on every draining stage gets turned off, and then it is not
watching when it matters.

The numbers below are the real ones from that day, so these are regression
guards against specific misreadings that actually happened:

  - a task running 738s in a stage whose longest completed task took 70s WAS
    stuck, and was first read as a dead run
  - a task running 165s where the median was 47s and the max 76s WAS stuck, and
    was first written up as "interrupted, not stalled"
  - the "25-30 seconds per task" figure that misreading rested on was wrong;
    the median is 47s

WHAT THIS DOES NOT COVER
------------------------
The polling loop and the shape of Spark's REST responses. Mocking the JSON would
only assert that our guess about Spark matches our guess about Spark, which is
false confidence about the one part that is genuinely unverified -- the driver
UI was already down when the watchdog was written, so no live response has been
seen.

Validate that on the next run instead: point it at a healthy job and confirm it
resolves the application, walks the active stages, and reports "Slow, not stuck"
rather than raising. That is a five-minute check and it is the real test of the
API assumptions.

No Spark session, no network: these are pure functions.
"""
import importlib.util
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "bin" / "stall_watchdog.py"


def _load_tool():
    """Import by path — bin/ is scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("_stall_watchdog", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wd():
    return _load_tool()


def _stage(finished_seconds, running_ages_seconds):
    """Build a task list: some finished, some still running for a given age."""
    now_ms = time.time() * 1000.0
    tasks = [
        {"status": "SUCCESS", "duration": s * 1000.0, "taskId": i, "index": i}
        for i, s in enumerate(finished_seconds)
    ]
    for n, age in enumerate(running_ages_seconds):
        tasks.append(
            {
                "status": "RUNNING",
                "launchTime": now_ms - age * 1000.0,
                "taskId": 900 + n,
                "index": 900 + n,
                "executorId": str(n),
            }
        )
    return tasks


# --- launchTime parsing -----------------------------------------------------
# Spark serialises this as a date string over REST and as epoch ms in the event
# log. The watchdog reads the REST form; the event-log form is accepted so the
# same helper can be pointed at a saved log.


def test_launch_time_accepts_the_rest_date_form(wd):
    assert wd.parse_launch_time("2026-09-06T10:00:47.244GMT") == pytest.approx(
        1788688847244.0
    )


def test_launch_time_accepts_epoch_milliseconds(wd):
    assert wd.parse_launch_time(1788703247244) == pytest.approx(1788703247244.0)


def test_launch_time_returns_none_rather_than_raising(wd):
    """A shape we did not anticipate must not kill the poll loop."""
    assert wd.parse_launch_time("nonsense") is None
    assert wd.parse_launch_time(None) is None


# --- the stuck-versus-slow decision ----------------------------------------


def test_the_0959_stall_is_detected(wd):
    """738s against a 70s stage maximum. First read as a dead run; it was stuck."""
    tasks = _stage([47, 48, 70], [738])
    _running, finished, stuck = wd.classify(tasks, 1.5)
    assert len(stuck) == 1
    assert max(finished) == 70_000
    assert stuck[0][1] / 1000.0 == pytest.approx(738, abs=2)


def test_the_0925_stall_is_detected(wd):
    """165s where the median was 47s and the max 76s. Written up as 'interrupted'."""
    tasks = _stage([40, 47, 55, 76], [165])
    _running, _finished, stuck = wd.classify(tasks, 1.5)
    assert len(stuck) == 1


def test_a_slow_task_does_not_trip_it(wd):
    """80s against a 76s max is inside the noise. Firing here gets it switched off."""
    tasks = _stage([40, 47, 55, 76], [80])
    _running, _finished, stuck = wd.classify(tasks, 1.5)
    assert stuck == []


def test_the_factor_is_what_moves_the_line(wd):
    """Same task, two thresholds — so the knob does what the flag says."""
    tasks = _stage([50, 76], [100])
    assert len(wd.classify(tasks, 1.2)[2]) == 1      # 100s > 76 * 1.2
    assert wd.classify(tasks, 2.0)[2] == []          # 100s < 76 * 2.0


def test_no_completed_task_means_capture_anyway(wd):
    """
    A stage where nothing has finished gives no baseline to compare against.
    Capture rather than stay quiet: a spurious dump is cheap, a missed stall is
    the failure this tool exists to prevent.
    """
    tasks = _stage([], [600])
    _running, finished, stuck = wd.classify(tasks, 1.5)
    assert finished == []
    assert len(stuck) == 1


def test_a_stage_with_nothing_running_is_never_stuck(wd):
    tasks = _stage([47, 48, 70], [])
    assert wd.classify(tasks, 1.5)[2] == []


def test_running_tasks_with_unparseable_launch_times_are_skipped(wd):
    """Better to under-report one task than to crash the watchdog mid-run."""
    tasks = [
        {"status": "SUCCESS", "duration": 47_000},
        {"status": "RUNNING", "launchTime": "not-a-date", "taskId": 1, "index": 1},
    ]
    running, _finished, stuck = wd.classify(tasks, 1.5)
    assert running == []
    assert stuck == []


# --- dump rendering ---------------------------------------------------------


def test_thread_dump_html_becomes_one_frame_per_line(wd):
    """The dump arrives as a UI page; what gets read is the stack."""
    page = (
        "<td>java.base@17.0.19/jdk.internal.misc.Unsafe.park(Native Method)<br/>"
        "com.nvidia.spark.rapids.PrioritySemaphore.acquire(PrioritySemaphore.scala:83)<br/>"
        "com.nvidia.spark.rapids.SemaphoreTaskInfo.blockUntilReady(GpuSemaphore.scala:261)"
        "</td>"
    )
    lines = [line.strip() for line in wd.dump_to_text(page).splitlines() if line.strip()]
    assert lines == [
        "java.base@17.0.19/jdk.internal.misc.Unsafe.park(Native Method)",
        "com.nvidia.spark.rapids.PrioritySemaphore.acquire(PrioritySemaphore.scala:83)",
        "com.nvidia.spark.rapids.SemaphoreTaskInfo.blockUntilReady(GpuSemaphore.scala:261)",
    ]


def test_dump_text_unescapes_entities(wd):
    """Frames carry &quot; and &lt;; a reader should not have to decode them."""
    assert "\"main\" <no lock>" in wd.dump_to_text("<td>&quot;main&quot; &lt;no lock&gt;</td>")


def test_dump_text_handles_an_empty_response(wd):
    assert wd.dump_to_text(None) == ""
    assert wd.dump_to_text("") == ""
