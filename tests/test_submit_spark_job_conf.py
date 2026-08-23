"""Pin the spark-submit settings whose absence hangs the job instead of failing it.

Every setting asserted here shares one property: if it is dropped, the job does
not crash -- it waits forever with no error. That makes these regressions very
expensive to diagnose by hand and very cheap to catch here.
"""

import re
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parent.parent / "bin" / "submit_spark_job.sh"


@pytest.fixture(scope="module")
def conf() -> dict[str, str]:
    """Map every `--conf k=v` the launcher passes to spark-submit."""
    text = LAUNCHER.read_text()
    return {
        m.group("key"): m.group("value").strip('"')
        for m in re.finditer(
            r"--conf\s+(?P<key>[\w.]+)=(?P<value>\S+)",
            text,
        )
    }


def test_s3a_path_style_access_enabled(conf):
    """Buckets with dots in the name are unreachable without path-style access.

    S3A's default virtual-host addressing builds <bucket>.s3.<region>.amazonaws.com,
    which the *.s3.<region>.amazonaws.com wildcard cert cannot match when the bucket
    name contains dots -- TLS fails before any S3 call. boto3 and the AWS CLI switch
    to path-style automatically, so a working `aws s3 ls` does NOT prove Spark works.
    """
    value = conf.get("spark.hadoop.fs.s3a.path.style.access")
    assert value is not None, "launcher must set spark.hadoop.fs.s3a.path.style.access"
    assert "true" in value, f"path-style access must default to true, got {value!r}"


def test_s3a_fails_fast(conf):
    """S3A defaults (20 attempts, 200s socket timeout) turn a misconfig into a hang."""
    for key in (
        "spark.hadoop.fs.s3a.connection.establish.timeout",
        "spark.hadoop.fs.s3a.connection.timeout",
        "spark.hadoop.fs.s3a.attempts.maximum",
        "spark.hadoop.fs.s3a.retry.limit",
    ):
        assert key in conf, f"{key} must be set so S3A errors instead of hanging"

    assert int(conf["spark.hadoop.fs.s3a.attempts.maximum"]) <= 5
    assert int(conf["spark.hadoop.fs.s3a.retry.limit"]) <= 5
    assert int(conf["spark.hadoop.fs.s3a.connection.timeout"]) <= 60_000


def test_rapids_gpu_pool_is_capped(conf):
    """RAPIDS defaults to pooling ~all GPU memory.

    On a unified-memory / integrated GPU that memory is the host's RAM, so the
    default starves the OS and the JVM and the node thrashes to a standstill.

    The default is asserted from the launcher's text rather than from the --conf
    line, because it is no longer expanded inline there: allocFraction is now
    resolved into a variable first so it can be compared against
    maxAllocFraction before submitting. What matters is that a safe default
    exists, not where it is written. The emitted VALUE is asserted separately in
    test_submit_spark_job_resources.py, which runs the launcher rather than
    reading it.
    """
    raw = conf.get("spark.rapids.memory.gpu.allocFraction")
    assert raw is not None, "launcher must cap spark.rapids.memory.gpu.allocFraction"

    text = LAUNCHER.read_text()
    default = re.search(r"RAPIDS_GPU_ALLOC_FRACTION:-([0-9.]+)\}", text)
    assert default, "allocFraction must carry a safe default somewhere in the launcher"
    assert 0 < float(default.group(1)) < 1.0, "allocFraction must be capped below 1.0"


def test_rapids_gpu_pool_has_a_hard_cap(conf):
    """maxAllocFraction must be emitted, not left to the site defaults.

    The launcher refuses to submit when allocFraction exceeds it, and that check
    is only meaningful if the cap it compares against is the one the job will
    actually run under. Inheriting the cap from spark-defaults.conf would let the
    two disagree silently.
    """
    raw = conf.get("spark.rapids.memory.gpu.maxAllocFraction")
    assert raw is not None, (
        "launcher must set spark.rapids.memory.gpu.maxAllocFraction; without it "
        "the pool ceiling comes from the site config and the pre-submit check "
        "validates a number the job never used"
    )

    text = LAUNCHER.read_text()
    default = re.search(r"RAPIDS_GPU_MAX_ALLOC_FRACTION:-([0-9.]+)\}", text)
    assert default, "maxAllocFraction must carry a default"
    assert 0 < float(default.group(1)) <= 1.0


def test_executor_requests_a_gpu(conf):
    """The executor asks for a GPU; the cluster's workers must advertise one.

    Documented in the README under 'Cluster prerequisites for GPU runs' -- a worker
    that does not advertise a GPU can never satisfy this request, and the job waits
    forever without scheduling a task.
    """
    assert "spark.executor.resource.gpu.amount" in conf
    assert "spark.executor.resource.gpu.discoveryScript" in conf
