"""Guard on tests/spark_env.py, which keeps the local sessions off a cluster.

Note what these can and cannot cover. The failure they exist for is a hang
inside the session fixture, so no test in this suite can catch it directly --
the run would stop at the fixture, before any assertion. What is testable is
the guard itself: that it hides both variables, and that it puts them back, so
tests/e2e/test_cluster_submit.py still finds spark-submit afterwards.
"""
import os

import pytest

from spark_env import CLUSTER_ENV_VARS, ignore_cluster_conf


def test_cluster_vars_are_hidden_inside_the_block(monkeypatch):
    for name in CLUSTER_ENV_VARS:
        monkeypatch.setenv(name, "/opt/spark")

    with ignore_cluster_conf():
        for name in CLUSTER_ENV_VARS:
            assert name not in os.environ

    for name in CLUSTER_ENV_VARS:
        assert os.environ[name] == "/opt/spark"


def test_vars_that_were_unset_stay_unset(monkeypatch):
    for name in CLUSTER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with ignore_cluster_conf():
        pass

    for name in CLUSTER_ENV_VARS:
        assert name not in os.environ


def test_vars_come_back_when_the_block_raises(monkeypatch):
    monkeypatch.setenv("SPARK_HOME", "/opt/spark")

    with pytest.raises(RuntimeError):
        with ignore_cluster_conf():
            raise RuntimeError("session failed to build")

    assert os.environ["SPARK_HOME"] == "/opt/spark"


def test_session_asks_for_no_gpu(spark):
    """
    The request that cannot be met in local mode is the per-task GPU, so pin
    its absence on the live session. The e2e fixture's opt-in RAPIDS settings
    are deliberately GPU-scheduling free, so this holds for that session too.
    """
    conf = spark.sparkContext.getConf()

    assert conf.get("spark.master", "").startswith("local")
    assert conf.get("spark.task.resource.gpu.amount", None) is None
    assert conf.get("spark.executor.resource.gpu.amount", None) is None
