"""
Import smoke test — the cheapest, highest-ROI guard in the suite.

Walks every module under `spark_jobs` and imports it. This catches import-time
breakage (missing symbols, forward-reference NameErrors, bad relative imports,
mis-indented methods) across all data sources at once — the class of bug that
otherwise only surfaces on the cluster, mid-spark-submit.

No SparkSession is started; importing a module only defines its classes and
functions. Runs in well under a second.
"""
import importlib
import pkgutil

import pytest

import spark_jobs


def _all_module_names():
    names = []
    for info in pkgutil.walk_packages(spark_jobs.__path__, prefix="spark_jobs."):
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _all_module_names())
def test_module_imports(module_name):
    importlib.import_module(module_name)
