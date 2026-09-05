"""Pin the invariants that hold a sizing profile together.

A profile is a list of exports with no structure, so nothing stops two settings
that only work as a pair from drifting apart. Every assertion here is a pairing
that cost a failed run to discover, where the broken combination submits
happily and dies later.
"""
import re
from pathlib import Path

import pytest

PROFILES = Path(__file__).resolve().parent.parent / "bin" / "profiles"


def _exports(name: str) -> dict[str, str]:
    """Parse `export K=V` out of a profile without sourcing it."""
    text = (PROFILES / name).read_text()
    return {
        m.group("key"): m.group("value").strip('"').strip("'")
        for m in re.finditer(
            r"^export\s+(?P<key>\w+)=(?P<value>\S*)", text, re.MULTILINE
        )
    }


@pytest.fixture(scope="module")
def large_run() -> dict[str, str]:
    return _exports("large-run.env")


@pytest.fixture(scope="module")
def pyg_assembly() -> dict[str, str]:
    return _exports("pyg-assembly.env")


@pytest.fixture(scope="module", params=["large-run.env", "pyg-assembly.env"])
def any_profile(request) -> dict[str, str]:
    """Both profiles, for the invariants that hold of either."""
    return _exports(request.param)


def test_alloc_fraction_does_not_exceed_its_cap(large_run):
    """The launcher refuses to submit when it does, so this fails fast instead.

    Catching it here costs a second; catching it from the launcher costs
    whatever the run was queued behind.
    """
    alloc = float(large_run["RAPIDS_GPU_ALLOC_FRACTION"])
    cap = float(large_run["RAPIDS_GPU_MAX_ALLOC_FRACTION"])
    assert 0 < alloc <= cap < 1.0


def test_uncapped_slots_require_no_declared_gpu(large_run):
    """GPU_PER_TASK=0 only reaches the core count if no GPU resource is declared.

    Task slots are min(cores/task.cpus, gpu/task.gpu). Setting task.gpu to 0
    while the executor still declares a GPU leaves the slot count capped by the
    resource, so the profile reads as uncapped and behaves as capped -- the
    quietest possible way to not get the concurrency you asked for.
    """
    if large_run.get("GPU_PER_TASK") == "0":
        assert large_run.get("GPU_PER_EXECUTOR") == "0", (
            "GPU_PER_TASK=0 needs GPU_PER_EXECUTOR=0, or slots stay capped"
        )


def test_uncapped_slots_require_a_lowered_batch_size(large_run):
    """The pairing that two failed runs paid for.

    RAPIDS_BATCH_SIZE_BYTES is per concurrent task, and on a unified-memory host
    the RMM pool is system RAM -- so at the 1g default, 144 concurrent tasks ask
    for roughly 170 GB on a 121 GiB host. Both settings must move together:
    raising concurrency without lowering the batch is the combination that
    wedged both executors on 2026-08-29 and drained the host on the retry.

    Asserted as an implication rather than an equality so the profile can be
    retuned; what must not happen is uncapping the slots and leaving the batch
    size at its default.
    """
    if large_run.get("GPU_PER_TASK") == "0":
        batch = large_run.get("RAPIDS_BATCH_SIZE_BYTES")
        assert batch is not None, (
            "GPU_PER_TASK=0 without RAPIDS_BATCH_SIZE_BYTES leaves the batch at "
            "RAPIDS' 1g default, which does not fit 144 concurrent tasks"
        )
        assert batch != "1g", "the default is the value that does not fit"


def test_the_pool_floor_is_reachable(any_profile):
    """A floor above the alloc fraction refuses every executor.

    The two are not on the same base -- the floor is a fraction of TOTAL GPU
    memory and the alloc fraction is of FREE -- so the ordering is what keeps the
    floor meetable on a host that has anything else resident. The launcher
    refuses this combination; failing here costs a second instead of a queue slot.
    """
    floor = float(any_profile["RAPIDS_GPU_MIN_ALLOC_FRACTION"])
    alloc = float(any_profile["RAPIDS_GPU_ALLOC_FRACTION"])
    assert 0 < floor <= alloc


def test_the_pool_floor_leaves_room_under_a_healthy_pool(any_profile):
    """The floor catches an unusable executor, not a busy one.

    A replacement executor sizes its pool against memory free at startup, so a
    floor set near a healthy pool refuses executors that would have worked and
    burns spark.deploy.maxExecutorRetries doing it. Half the alloc fraction is
    already generous: the pool that motivated this was 67.8 MB against a healthy
    8.8 GiB, which is three orders of magnitude down, not a near miss.
    """
    floor = float(any_profile["RAPIDS_GPU_MIN_ALLOC_FRACTION"])
    alloc = float(any_profile["RAPIDS_GPU_ALLOC_FRACTION"])
    assert floor <= alloc / 2, (
        "a floor this close to the alloc fraction refuses executors that a "
        "partly-used GPU would otherwise have started"
    )


def test_the_network_timeout_is_not_raised_past_ten_minutes(any_profile):
    """600s is the ceiling, not a dial to turn when an executor stalls.

    It was already raised once, from Spark's 120s, to tolerate a 173s stall
    inside block eviction. The trade is stated in both profiles: a genuinely
    dead executor also goes undetected for the whole window. On 2026-08-26 a
    stall outlasted even this and the timeout fired anyway -- so raising it
    further buys nothing and costs detection. The stall is the thing to
    remove; #346 removes a cause of it rather than widening the window.
    """
    timeout = any_profile["NETWORK_TIMEOUT"]
    assert timeout.endswith("s"), f"expected seconds, got {timeout!r}"
    assert int(timeout[:-1]) <= 600, (
        "raising the network timeout hides a dead executor for longer without "
        "fixing what stalled it"
    )

    heartbeat = any_profile["EXECUTOR_HEARTBEAT_INTERVAL"]
    assert int(heartbeat[:-1]) * 2 <= int(timeout[:-1]), (
        "the heartbeat has to fit inside the timeout several times over, or a "
        "live executor is evicted for missing one"
    )


def test_assembly_leaves_the_host_to_the_driver(pyg_assembly):
    """The assembly leg's whole reason for existing as a separate profile.

    The HeteroData is built in the driver's own memory -- numpy, outside any
    -Xmx -- so what sizes it is how much of the host nothing else has claimed.
    An executor heap the size of large-run.env's would take that room back and
    reproduce the 2026-08-25 host OOM.
    """
    assert int(pyg_assembly["EXECUTOR_MEMORY"].rstrip("gG")) <= 16
    assert float(pyg_assembly["RAPIDS_GPU_ALLOC_FRACTION"]) <= 0.12


def test_assembly_slots_leave_each_one_a_usable_pool(pyg_assembly):
    """GPU_PER_TASK and the alloc fraction only work as a pair.

    The slots divide one pool, so halving GPU_PER_TASK halves what each task
    gets without touching a line that mentions memory. Nothing fails at submit;
    the leg runs and pays in retries. Going from four slots to eight on
    2026-09-05 took "maximum pool size exceeded" from 34,571 to 100,024 on one
    leg and 45,912 to 132,419 on the other, and returned 17% of the wall for
    it. Eight is where that trade was still worth making.

    A floor rather than an equality -- fewer slots is always safe for the pool.
    What this refuses is another halving on its own, with the alloc fraction
    left where it is.
    """
    assert float(pyg_assembly["GPU_PER_TASK"]) >= 0.125, (
        "more than eight slots per executor divides the pool below the ~1.1 GiB "
        "a slot that was measured; RAPIDS_GPU_ALLOC_FRACTION has to rise with "
        "it, at the driver's expense, or the slot count stays put"
    )


def test_large_run_is_not_for_assembly(large_run):
    """The assembly leg needs the host for the driver, not the executors.

    This profile gives the executor side most of the box, which is right for
    enrichment and wrong for building HeteroData in driver memory. Pinned
    because the two profiles differ by numbers alone, so the wrong one is easy
    to pass and produces an OOM rather than an error.
    """
    assert int(large_run["EXECUTOR_MEMORY"].rstrip("gG")) >= 32
    assert "pyg-assembly" in (PROFILES / "large-run.env").read_text(), (
        "large-run.env must point readers at the assembly profile"
    )
