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
