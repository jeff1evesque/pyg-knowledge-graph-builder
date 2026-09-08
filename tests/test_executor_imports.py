"""What the parse module may import, given who imports it.

``spark_jobs/graph/turtle.py`` is the one module in this repo that Python
workers import for themselves. build_graph.py is submitted by path, so on the
driver it is ``__main__`` and cloudpickle ships anything defined there by value
-- the code object travels and no executor ever imports it. The moment a
function moves into a package, cloudpickle pickles it by reference instead, and
every task runs a real ``import spark_jobs.graph.turtle`` inside the executor
venv.

That venv is packaged from requirements-executor.txt, not from this checkout.
So an import this module can satisfy on the driver is not evidence of anything:
if the transitive imports reach past what executors carry, every parse task
raises ModuleNotFoundError while the whole unit suite stays green. That already
cost a run on 2026-09-06, over a missing setuptools.

The walk is static rather than an import, because importing proves only what
the DRIVER has.
"""
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_REQUIREMENTS = REPO_ROOT / "requirements-executor.txt"

# The module under guard, plus the package __init__ files that importing it
# runs. An empty __init__.py is the reason this module is safe to import at
# all, so it is part of what the guard covers.
GUARDED = (
    REPO_ROOT / "spark_jobs" / "__init__.py",
    REPO_ROOT / "spark_jobs" / "graph" / "__init__.py",
    REPO_ROOT / "spark_jobs" / "graph" / "turtle.py",
)

# Present on an executor without being requested by name:
#   pyspark    -- the worker runtime itself, which is what runs this code
#   numpy      -- pandas and pyarrow both pull it in
# Everything else must be spelled out in requirements-executor.txt, so that
# adding a dependency to the executors and allowing it here are one action.
IMPLICIT = frozenset({"pyspark", "numpy"})


def _declared_requirements():
    """Top-level import names from requirements-executor.txt."""
    names = set()
    for line in EXECUTOR_REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # "pyoxigraph>=0.4.0,<0.6.0" -> "pyoxigraph"
        for separator in ("<", ">", "=", "!", "~", "[", ";", " "):
            line = line.split(separator, 1)[0]
        if line:
            names.add(line.replace("-", "_"))
    return names


def _root_modules(source):
    """Every module root imported anywhere in a file, nested imports included.

    ``ast.walk`` rather than a scan of the top level, because this module puts
    its heavy imports INSIDE the function bodies on purpose -- a checker that
    read only module-level imports would see nothing but ``typing`` and pass
    whatever was hidden one indent deeper.
    """
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # A relative import stays inside spark_jobs, which the caller
            # already walks; there is no root package name to check.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_the_parse_module_imports_nothing_executors_lack():
    allowed = _declared_requirements() | IMPLICIT | set(sys.stdlib_module_names)

    offenders = {}
    for path in GUARDED:
        outside = {
            root
            for root in _root_modules(path.read_text())
            if root not in allowed and root != "spark_jobs"
        }
        if outside:
            offenders[path.name] = sorted(outside)

    assert not offenders, (
        f"{offenders} are imported by the parse path but are not in the "
        f"executor venv (requirements-executor.txt plus {sorted(IMPLICIT)}). "
        f"Executors import spark_jobs.graph.turtle for real, so every parse "
        f"task would raise ModuleNotFoundError -- and this suite would still "
        f"pass, because the driver venv has them. Either move the import back "
        f"behind the driver or add the package to requirements-executor.txt."
    )


def test_the_parse_module_reaches_no_other_first_party_module():
    """Nothing under spark_jobs/ may be pulled in besides the guarded files.

    The check above walks a fixed list. A new ``from spark_jobs.x import y``
    would drag x's own imports onto the executors without appearing in it, so
    the list is only sound while the module stays a leaf.
    """
    reached = set()
    for path in GUARDED:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or ""). \
                    startswith("spark_jobs"):
                reached.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("spark_jobs"):
                        reached.add(alias.name)

    assert not reached, (
        f"spark_jobs.graph.turtle now imports {sorted(reached)}. Executors "
        f"import it for real, so those modules and everything THEY import "
        f"ship to the workers too. Add them to GUARDED here once you have "
        f"confirmed their imports are executor-safe."
    )


def test_the_checker_catches_an_import_executors_do_not_have():
    """Prove the walk sees a nested import before trusting it to guard one.

    The defect this exists to catch is exactly an import tucked inside a
    function, which is where every heavy import in the parse module already
    lives.
    """
    hidden = "def f():\n    import boto3\n    return boto3\n"
    assert "boto3" in _root_modules(hidden)

    top_level = "import boto3\n"
    assert "boto3" in _root_modules(top_level)

    aliased = "def f():\n    from botocore.session import Session\n"
    assert "botocore" in _root_modules(aliased)


def test_the_declared_requirements_are_actually_read():
    """A parser that silently returned an empty set would allow nothing and
    fail loudly; one that returned everything would allow anything and never
    fire. Pin the names the file really carries."""
    declared = _declared_requirements()

    assert {"pyoxigraph", "rdflib", "pandas", "pyarrow"} <= declared
    assert "boto3" not in declared
