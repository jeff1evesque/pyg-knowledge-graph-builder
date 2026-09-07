"""Guard the diagnostic mode that reproduces the parse stall (#388).

WHAT THE MODE IS FOR
--------------------
The stage that materialises the parse deadlocks between the executor JVM and its
Python worker at 168 of 169 tasks and never clears. It is a race, not a
misconfiguration: on 2026-09-07 the same job read the same mirror twice within
half an hour, with byte-identical Spark properties, and stalled on the first
attempt while clearing the same stage in 68.7s on the second. Nothing can be
concluded from a single trial, and a full run costs about three hours to deliver
exactly one.

``parse_only`` is that trial on its own -- under two minutes to a verdict --
so ``bin/parse_stall_loop.sh`` can take enough of them to measure a rate.

WHAT THESE TESTS PROTECT
------------------------
A trial is only evidence about a real run if it submits the SAME PLAN a real
seed leg submits. That is the property at risk here, and it is not the kind that
announces itself when it breaks: someone trims a source, adds a ``.limit()`` to
make the loop turn faster, or reaches for a cheaper frame than
``load_source_triples`` builds, and every test still passes while every number
the loop produces quietly stops describing the thing being hunted. A stall hunt
that measures the wrong plan is worse than no stall hunt, because it produces
confident answers.

So the guard below is an ALLOWLIST of what the handler may call, not a list of
things it may not. A denylist only catches the shortcuts someone thought of.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from spark_jobs import build_graph
from spark_jobs.build_graph import (
    VALID_MODES,
    JobConfig,
    check_work_dir_occupancy,
    execute_parse_only,
)


BASE_ARGS = {
    "mode": "parse_only",
    "local_work_dir": "/work",
    "source_format": "turtle_parquet",
    "source_paths": "s3a://bucket/raw/source=bls/feed=cpi/2026.snappy.parquet",
}


def config(**overrides) -> JobConfig:
    return JobConfig({**BASE_ARGS, **overrides})


# ---------------------------------------------------------------------------
# the mode is reachable, and dispatches
# ---------------------------------------------------------------------------
def test_parse_only_is_a_valid_mode():
    assert "parse_only" in VALID_MODES


def test_every_valid_mode_has_a_handler():
    """A mode in VALID_MODES with no handler is a KeyError hours into a run.

    Written against the dict literal rather than by calling main(), which would
    need a SparkSession. Deliberately covers every mode, not just parse_only:
    the next one added gets this for free.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(build_graph.main)))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "mode_handlers"
            for t in node.targets
        )
    ]
    assert len(handlers) == 1, "mode_handlers is not a single dict literal"

    keys = {
        key.value
        for key in handlers[0].value.keys
        if isinstance(key, ast.Constant)
    }
    assert keys == VALID_MODES, (
        f"mode_handlers keys {sorted(keys)} do not match "
        f"VALID_MODES {sorted(VALID_MODES)}"
    )


# ---------------------------------------------------------------------------
# it is validated like the modes that parse for real
# ---------------------------------------------------------------------------
def test_parse_only_requires_source_paths():
    with pytest.raises(ValueError, match="source_paths is required"):
        config(source_paths="")


def test_parse_only_applies_the_sec_feed_guard():
    """The trial reads the same paths, so it gets the same refusal.

    A mode that accepted a decommissioned SEC feed would parse a shape no real
    run parses, which is the one thing a reproduction cannot afford.
    """
    with pytest.raises(ValueError):
        config(
            source_paths=(
                "s3a://bucket/raw/source=sec/feed=litigation/2026.snappy.parquet"
            )
        )


# ---------------------------------------------------------------------------
# the plan invariant
# ---------------------------------------------------------------------------
# Everything the handler is allowed to call. Adding to this set is a decision
# about whether the trial still reproduces a real seed leg -- make it
# deliberately, and say why in the handler.
ALLOWED_CALLS = frozenset(
    {
        "load_source_triples",  # the whole point: the real loader, real config
        "unpersist",            # release the cache the count populated
        "info",                 # logger
        "time",                 # time.time, for the number the loop compares
        "round",
    }
)


def _calls_in(func) -> set:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_parse_only_calls_nothing_that_could_change_the_plan():
    """The submitted plan must match a real seed leg's, or the trial proves nothing."""
    unexpected = _calls_in(execute_parse_only) - ALLOWED_CALLS
    assert not unexpected, (
        f"execute_parse_only calls {sorted(unexpected)}, which is not in the "
        "allowlist. If the trial no longer submits the same plan a real seed "
        "leg submits, its results stop describing real runs."
    )


def test_parse_only_uses_the_shared_loader():
    """Not a cheaper frame of its own -- same sources, union, cache and count."""
    assert "load_source_triples" in _calls_in(execute_parse_only)


def test_parse_only_writes_nothing():
    """No Parquet, no artifacts of its own. It is a measurement.

    Asserted through the same call allowlist rather than by scanning the source
    text: a substring scan matches the handler's own prose about writing
    nothing, and a test that its docstring can fail is not a test of the code.
    """
    writers = {"parquet", "save", "write", "saveAsTable", "toPandas", "collect"}
    assert not (_calls_in(execute_parse_only) & writers)


# ---------------------------------------------------------------------------
# what it hands back
# ---------------------------------------------------------------------------
class _FakeFrame:
    def __init__(self):
        self.unpersisted = False

    def unpersist(self):
        self.unpersisted = True


def test_parse_only_reports_the_count_and_the_seconds(monkeypatch):
    """parse_seconds is the number trials are compared on, so it is in the result.

    Recorded here rather than scraped back out of a log line, whose format is
    not a contract and would silently stop matching.
    """
    frame = _FakeFrame()
    monkeypatch.setattr(
        build_graph,
        "load_source_triples",
        lambda spark, config: (frame, 19_600_000, {"bls": 1}),
    )

    result = execute_parse_only(config(), spark=None, s3_client=None)

    assert result["mode"] == "parse_only"
    assert result["initial_triples"] == 19_600_000
    assert isinstance(result["parse_seconds"], float)
    assert result["sources"] == {"bls": 1}


def test_parse_only_releases_the_cache(monkeypatch):
    """A trial that left its cache behind would charge the next trial for it."""
    frame = _FakeFrame()
    monkeypatch.setattr(
        build_graph,
        "load_source_triples",
        lambda spark, config: (frame, 1, {}),
    )

    execute_parse_only(config(), spark=None, s3_client=None)

    assert frame.unpersisted, "execute_parse_only left the parse cached"


# ---------------------------------------------------------------------------
# the loop runs many trials into one work dir
# ---------------------------------------------------------------------------
def test_parse_only_never_refuses_an_occupied_work_dir():
    """Thirty trials share a work dir; occupancy is about artifacts, and it writes none.

    The other modes refuse to overwrite a finished run. If that applied here the
    loop would stop on its second trial.
    """
    occupied = SimpleNamespace(mode="parse_only")
    check_work_dir_occupancy(occupied, spark=None)
