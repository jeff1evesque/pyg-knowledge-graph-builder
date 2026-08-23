"""The launcher must not request a GPU per task when the master is local.

WHY THIS IS A SEPARATE FILE FROM test_submit_spark_job_conf.py

That file pins settings whose absence hangs the job, which is this defect's category
exactly -- but it asserts by regex over the launcher's *text*, and that approach cannot
see this bug. The GPU settings were present in the text the whole time. The defect was
that they were present when the master is `local[*]`, where nothing can satisfy them.

So these tests run the launcher instead of reading it. A stub `spark-submit` records the
arguments it was handed, and the assertions are about what was actually emitted for a
given master. No Spark, no GPU, no cluster: the stub is a shell script that writes its
argv to a file.

THE DEFECT

`spark.task.resource.gpu.amount` makes every task require a GPU. It is a CLUSTER
scheduling setting -- it tells the master which workers own a GPU so it can place
executors on them. In local mode there is no master and no separate executor process:
the driver is the executor, and it advertises no custom resources ("No custom resources
configured for spark.driver"). Every task then requires a GPU that nothing offers, and
Spark's response to an unsatisfiable resource request is to WAIT, not to fail. Measured:
a local run sat on task set 0.0 for 10 minutes having started zero tasks, and would have
sat there indefinitely. With the requirement lifted the same run finished in 53 seconds.

WHY THE LOCAL BRANCH OVERRIDES TO 0 RATHER THAN OMITTING

Omitting the keys does not work, and test_local_master_overrides_rather_than_omitting is
the guard for it. `spark-submit` reads the site `spark-defaults.conf`, which commonly
sets `spark.task.resource.gpu.amount` for the cluster's benefit. A key this script does
not name is therefore not unset -- it takes the site value, the requirement stays in
force, and the job still hangs. This was measured on a host whose defaults file sets it
to 1: the first fix omitted the flags and changed nothing.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "bin" / "submit_spark_job.sh"

GPU_EXECUTOR = "spark.executor.resource.gpu.amount"
GPU_TASK = "spark.task.resource.gpu.amount"
GPU_DISCOVERY = "spark.executor.resource.gpu.discoveryScript"

pytestmark = [
    pytest.mark.launcher,
    pytest.mark.skipif(
        shutil.which("zip") is None,
        reason="the launcher packages spark_jobs with zip before invoking spark-submit",
    ),
]


def _emitted_conf(tmp_path: Path, master: str) -> dict[str, str]:
    """Run the launcher against a stub spark-submit; return the --conf map it emitted.

    The stub is preferred over whatever is on PATH the same way a real Spark install
    would be: the launcher checks $SPARK_HOME/bin/spark-submit first.
    """
    spark_home = tmp_path / "spark"
    (spark_home / "bin").mkdir(parents=True)
    recorded = tmp_path / "argv.txt"

    stub = spark_home / "bin" / "spark-submit"
    # One argument per line: values are never split, so a conf carrying a space
    # survives intact.
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{recorded}"\n'
    )
    stub.chmod(0o755)

    env = {
        **os.environ,
        "SPARK_HOME": str(spark_home),
        "SPARK_MASTER_URL": master,
        # Keep the venv archive out of it: whether one has been built is irrelevant
        # here and only changes whether --archives appears.
        "VENV_ARCHIVE": str(tmp_path / "absent.tar.gz"),
    }

    proc = subprocess.run(
        [str(LAUNCHER), "--mode", "pyg_only", "--local_work_dir", str(tmp_path / "work")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"launcher failed for master {master!r} (exit {proc.returncode})\n{proc.stderr}"
    )

    argv = recorded.read_text().splitlines()
    conf: dict[str, str] = {}
    for flag, pair in zip(argv, argv[1:]):
        if flag == "--conf":
            key, _, value = pair.partition("=")
            conf[key] = value
    return conf


def test_local_master_requests_no_gpu_per_task(tmp_path):
    """A local master must not leave a per-task GPU requirement in force.

    This is the hang. Anything other than 0 here and the job waits forever.
    """
    conf = _emitted_conf(tmp_path, "local[*]")

    assert conf.get(GPU_TASK) == "0", (
        f"local mode emitted {GPU_TASK}={conf.get(GPU_TASK)!r}. Any nonzero value makes "
        "every task require a GPU the local driver does not advertise, and Spark waits "
        "on the unsatisfiable request instead of failing -- the job hangs with no error."
    )
    assert conf.get(GPU_EXECUTOR) == "0"


def test_local_master_overrides_rather_than_omitting(tmp_path):
    """The keys must be PRESENT and zero, not absent.

    Absent is not unset: spark-submit falls back to the site spark-defaults.conf, which
    on a GPU host sets these for the cluster. Omitting them leaves the requirement in
    force and the hang unfixed.
    """
    conf = _emitted_conf(tmp_path, "local[*]")

    for key in (GPU_TASK, GPU_EXECUTOR):
        assert key in conf, (
            f"{key} was omitted for a local master rather than overridden to 0. A key "
            "this launcher does not name takes its value from the site "
            "spark-defaults.conf, so omission does not lift the requirement."
        )


def test_local_master_still_enables_rapids(tmp_path):
    """Lifting the resource request must not cost GPU acceleration.

    RAPIDS talks to CUDA directly and is unaffected by Spark's resource scheduler. If
    this ever fails, the local branch has gone too far: it would have turned the GPU
    off rather than stopped asking the scheduler for one.
    """
    conf = _emitted_conf(tmp_path, "local[*]")

    assert conf.get("spark.plugins") == "com.nvidia.spark.SQLPlugin"
    assert conf.get("spark.rapids.sql.enabled") == "true"


def test_cluster_master_still_requests_a_gpu(tmp_path):
    """The cluster path must be untouched by the local branch.

    Without these the standalone master has no way to know which workers own a GPU, and
    the job that was hanging locally starts hanging on the cluster instead.
    """
    conf = _emitted_conf(tmp_path, "spark://master.invalid:7077")

    assert conf.get(GPU_TASK) not in (None, "0")
    assert conf.get(GPU_EXECUTOR) not in (None, "0")
    assert conf.get(GPU_DISCOVERY), (
        "the cluster path must keep the discovery script; without it a worker cannot "
        "report the GPU addresses it owns"
    )


def test_discovery_script_is_not_sent_for_a_local_master(tmp_path):
    """Nothing discovers GPUs for a driver-as-executor, so the script is cluster-only."""
    conf = _emitted_conf(tmp_path, "local[*]")

    assert GPU_DISCOVERY not in conf
