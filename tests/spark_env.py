"""Keep the local test sessions away from a cluster's Spark configuration.

When SPARK_HOME points at a cluster install, PySpark reads that install's
spark-defaults.conf as the JVM launches. Ours asks for a GPU per task. Local
mode has no worker to advertise one, so the request can never be satisfied and
the suite HANGS FOREVER with no error -- it does not fail, it just sits there.
SPARK_CONF_DIR names such a directory on its own, so it does the same thing.

bin/generate_report.sh already unsets SPARK_HOME for the suites it runs, but a
bare `pytest` never goes through it. The README's cluster smoke section tells
you to export SPARK_HOME, so the next command typed in that same shell hangs.

Both variables are put back on the way out. tests/e2e/test_cluster_submit.py
submits through bin/submit_spark_job.sh, which needs SPARK_HOME to find
spark-submit.
"""
import os
from contextlib import contextmanager

# Both are read while the JVM launches, so hiding them around getOrCreate() is
# enough -- the session that comes back is already clear of them.
CLUSTER_ENV_VARS = ("SPARK_HOME", "SPARK_CONF_DIR")


@contextmanager
def ignore_cluster_conf():
    """Hide the cluster's Spark configuration for the duration of the block."""
    saved = {name: os.environ.pop(name, None) for name in CLUSTER_ENV_VARS}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
