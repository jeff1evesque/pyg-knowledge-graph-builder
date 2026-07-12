"""Submit the real job to a real standalone Spark cluster.

Every other test in this repo runs against a local ``SparkSession`` (``local[*]``)
with the venv's own PySpark. That is hermetic and CI-portable, but it means one path
is never exercised: **the job as actually submitted to a cluster**. Local mode:

  * reads none of the cluster's ``spark-defaults.conf``, so cluster-only settings
    (GPU resource scheduling, S3A) never apply;
  * has no master and no workers, so executor/worker resource negotiation -- the
    thing that decides whether a GPU job is ever *scheduled* -- is untested;
  * imports the job in-process instead of zipping it and shipping it to executors.

A cluster whose workers advertise no GPU accepts the job and then **never schedules
it**: it hangs forever with no error, while every local test stays green. So this
test bounds the submission with a timeout -- a hang must fail, not wait.

Opt-in. Skipped unless SPARK_MASTER_URL and CLUSTER_SMOKE_SOURCE_PATH are set, so CI
and ordinary local runs are unaffected. Nothing about any particular deployment is
hardcoded; the cluster's address, its data, and the RAPIDS jar are all inputs.

    SPARK_MASTER_URL=spark://<host>:7077 \
    CLUSTER_SMOKE_SOURCE_PATH=s3a://<bucket>/<prefix>/ \
      python -m pytest tests/e2e/test_cluster_submit.py -m cluster -s

The source path must be readable by **every** node: executors run across the cluster,
so a directory that exists only on the driver's disk will not do (hence an s3a:// URI
in practice).
"""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.cluster]

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "bin" / "submit_spark_job.sh"

MASTER_URL = os.environ.get("SPARK_MASTER_URL", "")
SOURCE_PATH = os.environ.get("CLUSTER_SMOKE_SOURCE_PATH", "")
# Where the job writes. On a cluster this MUST be storage every node can reach: the
# executors are spread across machines, so a file:// path resolves to a different
# local disk on each one and the commit protocol cannot assemble the output ("Mkdirs
# failed to create file:/..."). In practice that means an s3a:// URI.
OUTPUT_BASE = os.environ.get("CLUSTER_SMOKE_OUTPUT_PATH", "")
TIMEOUT_S = int(os.environ.get("CLUSTER_SMOKE_TIMEOUT", "900"))

requires_cluster = pytest.mark.skipif(
    not (MASTER_URL and SOURCE_PATH and OUTPUT_BASE),
    reason=(
        "set SPARK_MASTER_URL, CLUSTER_SMOKE_SOURCE_PATH and CLUSTER_SMOKE_OUTPUT_PATH "
        "to run against a real standalone cluster (skipped by default: CI has none)"
    ),
)


def _run_output_path() -> str:
    """Date-partitioned, per-run output prefix under CLUSTER_SMOKE_OUTPUT_PATH.

    Every run writes a fresh prefix rather than overwriting the last one, so a failed
    run's output is still there to inspect. Expiring it is the storage layer's job (a
    lifecycle rule), not the test's.
    """
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    return (
        f"{OUTPUT_BASE.rstrip('/')}"
        f"/year={now:%Y}/month={now:%m}/day={now:%d}/{stamp}"
    )


def _truthy(val) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="module")
def submission():
    """Submit the job once through its own launcher; hand the result to the tests."""
    work_dir = _run_output_path()

    env = {
        **os.environ,
        "SPARK_MASTER_URL": MASTER_URL,
        # Make RAPIDS report, per operator, whether it ran on the GPU. Without this
        # a CPU fallback is invisible: the job still succeeds and still produces a
        # correct graph, just on the CPU.
        "RAPIDS_EXPLAIN": "ALL",
    }

    cmd = [
        str(LAUNCHER),
        "--mode", "enrichment_only",
        "--source_paths", SOURCE_PATH,
        "--source_format", "turtle_parquet",
        "--turtle_column", "triples",
        "--local_work_dir", work_dir,
        "--time_period", "2099-01",
        "--parquet_partitions", "2",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the job HUNG for {TIMEOUT_S}s and was killed.\n\n"
            "The usual cause is that the cluster's workers advertise no GPU while "
            "spark-defaults.conf makes every task request one: the request can never "
            "be satisfied, so the job is never scheduled and Spark waits in silence.\n\n"
            f"stdout tail:\n{(exc.stdout or b'')[-2000:]!r}"
        )

    return proc, work_dir


@requires_cluster
def test_job_completes_on_the_cluster(submission):
    proc, _ = submission
    assert proc.returncode == 0, (
        f"submission failed (exit {proc.returncode}).\n"
        f"stderr tail:\n{proc.stderr[-3000:]}"
    )


@requires_cluster
def test_job_writes_its_artifacts(submission):
    """The artifacts must land in shared storage, not on some executor's local disk."""
    _, work_dir = submission

    if not work_dir.startswith("s3a://"):
        files = [p for p in Path(work_dir).rglob("*") if p.is_file()]
        assert files, f"the job produced no artifacts under {work_dir}"
        return

    import boto3

    without_scheme = work_dir[len("s3a://"):]
    bucket, _, prefix = without_scheme.partition("/")
    listing = boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [obj["Key"] for obj in listing.get("Contents", [])]

    assert keys, f"the job wrote nothing under s3a://{bucket}/{prefix}"
    # _temporary left behind means the commit protocol did not finish, even if the
    # job reported success.
    leftovers = [k for k in keys if "_temporary" in k]
    assert not leftovers, f"commit did not complete; _temporary files remain: {leftovers[:3]}"


@requires_cluster
def test_job_actually_ran_on_the_gpu(submission):
    """A successful run is not evidence of GPU execution.

    If the RAPIDS plugin fails to load -- a missing or mismatched jar, a worker that
    advertises no GPU -- Spark does not raise. It completes the job on the CPU and
    produces exactly the same graph. Assert on RAPIDS' own per-operator decisions.
    """
    proc, _ = submission
    output = proc.stdout + proc.stderr

    if not _truthy(os.environ.get("CLUSTER_SMOKE_EXPECT_GPU", "1")):
        pytest.skip("CLUSTER_SMOKE_EXPECT_GPU=0: not asserting GPU placement")

    on_gpu = output.count("will run on GPU")
    assert on_gpu > 0, (
        "the job completed but RAPIDS placed NO operator on the GPU -- it silently "
        "ran on the CPU.\n"
        "Check that the workers advertise a GPU (spark.worker.resource.gpu.amount) "
        "and that the RAPIDS jar is on the cluster classpath.\n"
        f"RAPIDS output tail:\n{output[-2000:]}"
    )
