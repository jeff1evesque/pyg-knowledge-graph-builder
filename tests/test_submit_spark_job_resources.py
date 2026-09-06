"""The launcher must size executor memory, and refuse an unsatisfiable GPU pool.

WHY THIS IS A SEPARATE FILE

test_submit_spark_job_conf.py asserts by regex over the launcher's *text*, which
cannot see either defect here: one is about a flag that was entirely absent, the
other about a comparison between two values rather than the presence of either.
test_submit_spark_job_local_mode.py runs the launcher, but its subject is what
local mode must NOT request; these are about what a CLUSTER master must, and its
helper returns only `--conf` pairs while `--executor-memory` is a bare flag.

So this file runs the launcher against a stub spark-submit and asserts over the
full argv.

THE TWO DEFECTS

1. Nothing set `spark.executor.memory` -- not the launcher, not the site
   spark-defaults.conf -- so every executor on a cluster run took Spark's 1g
   default while the workers advertised far more. This never surfaced because the
   cluster suite runs on committed fixtures measured in kilobytes. Real input is
   not that: one day across four sources parses to ~19.8M triples. The failure
   mode is not a crash but constant spilling, so the job still succeeds and
   nothing reports that it ran many times slower than it should have.

2. RAPIDS computes its pool as (gpu.free - reserve) * allocFraction and caps it
   at gpu.total * maxAllocFraction, refusing to start when the pool exceeds the
   cap. That rejection arrives as an executor-plugin shutdown inside a py4j stack
   trace, after the phase banner and before any progress line, so from the job's
   own log it reads as a startup hang. Measured: ~25 minutes lost to
   allocFraction=0.5 against a site maxAllocFraction of 0.4.

WHY THE GUARD IS STRICTER THAN RAPIDS

allocFraction scales gpu.FREE while the cap is a fraction of gpu.TOTAL, so an
allocFraction above maxAllocFraction is legal whenever enough memory is already
in use. That is not a property to rely on: the same configuration starts on a
busy GPU and fails on an idle one. Requiring alloc <= max makes it deterministic.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "bin" / "submit_spark_job.sh"

CLUSTER_MASTER = "spark://stub-master:7077"

pytestmark = [
    pytest.mark.launcher,
    pytest.mark.skipif(
        shutil.which("zip") is None,
        reason="the launcher packages spark_jobs with zip before invoking spark-submit",
    ),
]


def _run(tmp_path: Path, master: str, **env_overrides):
    """Run the launcher against a stub spark-submit.

    Returns (completed_process, argv, submitted) where argv is what the stub was
    handed and `submitted` says whether it was invoked at all -- the distinction
    matters for the refusal cases, where the point is that no work is scheduled.
    """
    spark_home = tmp_path / "spark"
    # exist_ok: a test may call this twice on one tmp_path to compare two configs.
    (spark_home / "bin").mkdir(parents=True, exist_ok=True)
    recorded = tmp_path / "argv.txt"
    marker = tmp_path / "submitted"

    stub = spark_home / "bin" / "spark-submit"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{recorded}"\n'
        f'touch "{marker}"\n'
    )
    stub.chmod(0o755)

    env = {
        **os.environ,
        "SPARK_HOME": str(spark_home),
        "SPARK_MASTER_URL": master,
        # Whether a venv archive has been built is irrelevant here and only
        # changes whether --archives appears.
        "VENV_ARCHIVE": str(tmp_path / "absent.tar.gz"),
        **{k: str(v) for k, v in env_overrides.items()},
    }

    proc = subprocess.run(
        [str(LAUNCHER), "--mode", "pyg_only",
         "--local_work_dir", str(tmp_path / "work")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    argv = recorded.read_text().splitlines() if recorded.exists() else []
    return proc, argv, marker.exists()


def _flag_value(argv: list[str], flag: str) -> str | None:
    """The value following `flag`, or None when the flag is absent."""
    for name, value in zip(argv, argv[1:]):
        if name == flag:
            return value
    return None


def _conf(argv: list[str]) -> dict[str, str]:
    out = {}
    for flag, pair in zip(argv, argv[1:]):
        if flag == "--conf":
            key, _, value = pair.partition("=")
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# executor sizing
# --------------------------------------------------------------------------- #

def test_cluster_master_sets_executor_memory(tmp_path):
    """Without this the executors silently take Spark's 1g default."""
    proc, argv, submitted = _run(tmp_path, CLUSTER_MASTER)

    assert proc.returncode == 0, proc.stderr
    assert submitted
    assert _flag_value(argv, "--executor-memory") is not None, (
        "a cluster submission emitted no --executor-memory, so every executor "
        "falls back to Spark's 1g default. That does not fail -- it spills, and "
        "the job succeeds while running many times slower than it should."
    )


def test_executor_memory_default_is_a_conservative_floor(tmp_path):
    """The default must not assume large workers.

    Spark treats an executor-memory request bigger than any worker can offer as
    unsatisfiable, and waits on it rather than failing -- the same hang class the
    GPU settings already guard against. A default sized for this cluster would
    turn a slow job into a silent one on a smaller cluster, so it errs small and
    warns instead.
    """
    proc, argv, _ = _run(tmp_path, CLUSTER_MASTER)

    assert _flag_value(argv, "--executor-memory") == "4g"
    assert "EXECUTOR_MEMORY" in proc.stderr, (
        "the launcher defaulted executor memory without telling anyone. The "
        "default is a floor, so the warning is what makes it a decision rather "
        "than a surprise."
    )


def test_executor_memory_is_configurable(tmp_path):
    proc, argv, _ = _run(tmp_path, CLUSTER_MASTER, EXECUTOR_MEMORY="64g")

    assert _flag_value(argv, "--executor-memory") == "64g"
    assert "EXECUTOR_MEMORY is unset" not in proc.stderr, (
        "warned about an unset variable that was set"
    )


def test_local_master_omits_executor_memory(tmp_path):
    """Local mode has no executor process -- the driver is the executor.

    Emitting the flag there would be misleading rather than harmful, but the
    honest signal is its absence: --driver-memory is what sizes a local run.
    """
    _, argv, _ = _run(tmp_path, "local[*]")

    assert _flag_value(argv, "--executor-memory") is None
    assert _flag_value(argv, "--driver-memory") is not None


def test_executor_cores_only_when_asked(tmp_path):
    """Unset means each executor takes every core on its worker, which is wanted."""
    _, argv, _ = _run(tmp_path, CLUSTER_MASTER)
    assert _flag_value(argv, "--executor-cores") is None

    _, argv, _ = _run(tmp_path, CLUSTER_MASTER, EXECUTOR_CORES="16")
    assert _flag_value(argv, "--executor-cores") == "16"


# --------------------------------------------------------------------------- #
# GPU pool fractions
# --------------------------------------------------------------------------- #

def test_alloc_fraction_above_max_is_refused_before_submitting(tmp_path):
    """The whole point is that nothing is scheduled.

    RAPIDS would reject this itself, but only after the JVM is up and only as a
    plugin shutdown inside a stack trace. Catching it here turns ~25 minutes of
    confusion into one line.
    """
    proc, _, submitted = _run(
        tmp_path, CLUSTER_MASTER,
        RAPIDS_GPU_ALLOC_FRACTION="0.5", RAPIDS_GPU_MAX_ALLOC_FRACTION="0.4",
    )

    assert proc.returncode != 0, "an unsatisfiable GPU pool was submitted anyway"
    assert not submitted, (
        "spark-submit was invoked with a pool configuration RAPIDS will reject"
    )
    assert "RAPIDS_GPU_ALLOC_FRACTION" in proc.stderr
    # Both numbers must appear, or the message cannot be acted on.
    assert "0.5" in proc.stderr and "0.4" in proc.stderr


def test_alloc_fraction_equal_to_max_is_allowed(tmp_path):
    """The bound is <=, not <. Equal is the largest deterministic setting."""
    proc, argv, submitted = _run(
        tmp_path, CLUSTER_MASTER,
        RAPIDS_GPU_ALLOC_FRACTION="0.4", RAPIDS_GPU_MAX_ALLOC_FRACTION="0.4",
    )

    assert proc.returncode == 0, proc.stderr
    assert submitted
    assert _conf(argv)["spark.rapids.memory.gpu.allocFraction"] == "0.4"


def test_max_alloc_fraction_is_emitted(tmp_path):
    """It must be passed, not left to the site defaults.

    The guard compares against a value this script chose; if the cap actually in
    force came from spark-defaults.conf instead, the two could disagree and the
    check would be validating a number the job never used.
    """
    _, argv, _ = _run(tmp_path, CLUSTER_MASTER)
    conf = _conf(argv)

    assert conf.get("spark.rapids.memory.gpu.maxAllocFraction") == "0.4"
    assert conf.get("spark.rapids.memory.gpu.allocFraction") == "0.25"


def test_min_alloc_fraction_defaults_to_no_floor(tmp_path):
    """Unset means no floor, which is RAPIDS' own default.

    Pinned because the floor refuses executors: turning it on for callers who did
    not ask would fail runs that currently start.
    """
    _, argv, _ = _run(tmp_path, CLUSTER_MASTER)

    assert _conf(argv).get("spark.rapids.memory.gpu.minAllocFraction") == "0"


def test_min_alloc_fraction_is_passed_through(tmp_path):
    """The floor is what turns a 67.8 MB pool into a refusal instead of a
    std::bad_alloc on the executor's first allocation."""
    _, argv, _ = _run(
        tmp_path, CLUSTER_MASTER, RAPIDS_GPU_MIN_ALLOC_FRACTION="0.02",
    )

    assert _conf(argv).get("spark.rapids.memory.gpu.minAllocFraction") == "0.02"


def test_min_alloc_fraction_above_alloc_is_refused_before_submitting(tmp_path):
    """A floor above the alloc fraction can never be met.

    The floor is a fraction of TOTAL memory and the alloc fraction is of FREE, so
    once the floor exceeds it every executor is refused as soon as anything else
    is resident -- which on a unified-memory host is always. That reads as a
    cluster that will not start rather than as a bad setting.
    """
    proc, _, submitted = _run(
        tmp_path, CLUSTER_MASTER,
        RAPIDS_GPU_ALLOC_FRACTION="0.08", RAPIDS_GPU_MIN_ALLOC_FRACTION="0.5",
    )

    assert proc.returncode != 0, "an unmeetable pool floor was submitted anyway"
    assert not submitted
    assert "RAPIDS_GPU_MIN_ALLOC_FRACTION" in proc.stderr
    assert "0.5" in proc.stderr and "0.08" in proc.stderr


def test_min_alloc_fraction_equal_to_alloc_is_allowed(tmp_path):
    """The bound is <=, matching the alloc/max guard above."""
    proc, argv, submitted = _run(
        tmp_path, CLUSTER_MASTER,
        RAPIDS_GPU_ALLOC_FRACTION="0.08", RAPIDS_GPU_MIN_ALLOC_FRACTION="0.08",
    )

    assert proc.returncode == 0, proc.stderr
    assert submitted
    assert _conf(argv)["spark.rapids.memory.gpu.minAllocFraction"] == "0.08"


# ======================================================================
# RAPIDS batch size — what actually decides how many tasks fit in host RAM
# ======================================================================

def test_batch_size_defaults_to_the_rapids_default(tmp_path):
    """Unset, the launcher must not change behaviour for existing callers.

    1g is RAPIDS' own default. Stating it explicitly rather than omitting the
    conf makes the value visible in the submitted command, which is where
    anyone debugging a memory failure will look for it.
    """
    _, argv, _ = _run(tmp_path, CLUSTER_MASTER)
    assert _conf(argv)["spark.rapids.sql.batchSizeBytes"] == "1g"


def test_batch_size_is_configurable(tmp_path):
    """The setting that makes high task concurrency survivable.

    It is per CONCURRENT TASK, so on a unified-memory host -- where the RMM pool
    comes out of system RAM -- it, not the slot count, decides how many tasks
    fit. Measured 2026-08-29: ~1.0 GB of host RAM per task at the 1g default, so
    144 concurrent tasks want ~170 GB on a 121 GiB host and the run dies. At
    256m all 144 held with 44 GB free -- and the parse took 4,833.8s against
    1,408.3s at 64 tasks and the 1g default, so this is a survival lever rather
    than a speed one.
    """
    _, argv, _ = _run(tmp_path, CLUSTER_MASTER, RAPIDS_BATCH_SIZE_BYTES="256m")
    assert _conf(argv)["spark.rapids.sql.batchSizeBytes"] == "256m"


def test_extra_conf_can_still_override_the_batch_size(tmp_path):
    """SPARK_EXTRA_CONF is emitted last, so it wins on duplicate keys.

    Pinned because a run in flight sets the batch size that way, and a launcher
    change that moved SPARK_EXTRA_CONF earlier would silently revert it to the
    default mid-run -- the job would keep going and die of memory an hour later.
    """
    _, argv, _ = _run(
        tmp_path, CLUSTER_MASTER,
        SPARK_EXTRA_CONF="--conf spark.rapids.sql.batchSizeBytes=128m",
    )
    pairs = [p for f, p in zip(argv, argv[1:])
             if f == "--conf" and p.startswith("spark.rapids.sql.batchSizeBytes=")]
    assert pairs[-1] == "spark.rapids.sql.batchSizeBytes=128m", (
        "SPARK_EXTRA_CONF must be emitted after the launcher's own defaults"
    )
