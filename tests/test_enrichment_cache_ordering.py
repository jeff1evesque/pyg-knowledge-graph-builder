"""
Pin the cache-then-count ordering in the BLS and SEC intra-source linkers.

WHAT THIS PROTECTS
------------------
Both linkers cache their own source's subset, build four steps of lazy frames on
top of it, then count the union. The count is the FIRST thing that reads those
frames -- the steps themselves compute nothing. Dropping the cache above the
count therefore does not save memory, it just means the count rebuilds the subset
from the full source frame instead of reading what was cached moments earlier.

That is the shape #380 named as "a defect on its own terms", and it was fixed by
moving the unpersist below the count in both files. Nothing tested the ordering,
so the next refactor could quietly put it back.

WHY AN AST CHECK AND NOT A BEHAVIOUR TEST
-----------------------------------------
The defect is invisible in output: both orderings return exactly the same
triples. The only difference is how much work Spark does to produce them, and
asserting on that from a unit test would mean reaching into the query execution
listener and measuring scans -- a slow, flaky test of Spark's internals rather
than of our code. The ordering itself is the thing worth pinning, so it is
checked directly.

The checker only looks at TOP-LEVEL statements of `enrich`. Both linkers also
unpersist inside an early-return branch, before anything has been counted, and
that one is correct -- descending into it would flag working code.

Following tests/test_parse_batching.py: the checker is proven against the defect
it exists to catch before it is trusted, because a guard that passes the broken
code is worse than no guard.
"""
import ast
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

LINKERS = [
    ("bls", "spark_jobs/enrichment/intra_source/bls_linker.py", "BLSIntraSourceLinker"),
    ("sec", "spark_jobs/enrichment/intra_source/sec_linker.py", "SECIntraSourceLinker"),
]


def _enrich_of(source, class_name):
    """The `enrich` FunctionDef of the named class."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "enrich":
                    return item
    return None


def _calls(node, method):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    )


def _top_level_unpersist(fn):
    """Line of the first bare `<something>.unpersist()` statement in fn's body.

    Bare statement only: the early-return branch's unpersist lives inside an If,
    and that one runs before any count, so it is not what this guards.
    """
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and _calls(stmt.value, "unpersist"):
            return stmt.lineno
    return None


def _top_level_count(fn):
    """Line of the first top-level assignment whose value calls .count()."""
    for stmt in fn.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            if any(_calls(node, "count") for node in ast.walk(stmt.value)):
                return stmt.lineno
    return None


@pytest.mark.parametrize("name,path,class_name", LINKERS, ids=[row[0] for row in LINKERS])
def test_the_cache_is_dropped_after_the_count_that_reads_it(name, path, class_name):
    fn = _enrich_of((REPO_ROOT / path).read_text(), class_name)
    assert fn is not None, f"{class_name}.enrich not found -- did the class move?"

    count_line = _top_level_count(fn)
    unpersist_line = _top_level_unpersist(fn)

    assert count_line is not None, (
        f"{class_name}.enrich no longer counts at the top level. If the count "
        "moved, this guard needs rewriting rather than deleting."
    )
    assert unpersist_line is not None, (
        f"{class_name}.enrich no longer unpersists its cached subset at the top "
        "level. If caching went away entirely that is fine, but check it."
    )
    assert unpersist_line > count_line, (
        f"{class_name}.enrich drops its cached subset at line {unpersist_line}, "
        f"above the count at line {count_line}. The count is the first thing "
        "that reads the cached frames, so it will rebuild the subset from the "
        "full source frame instead of reading the cache. #380"
    )


def test_the_checker_catches_the_ordering_it_exists_to_catch():
    """The pre-#380 shape, so a passing guard means something."""
    broken = textwrap.dedent(
        """
        class BLSIntraSourceLinker:
            def enrich(self, triples_df):
                if not new_dfs:
                    self._bls_triples.unpersist()
                    return empty
                all_new = all_new.cache()
                self._bls_triples.unpersist()
                total = all_new.count()
                return all_new
        """
    )
    fn = _enrich_of(broken, "BLSIntraSourceLinker")
    assert _top_level_unpersist(fn) < _top_level_count(fn), (
        "the checker failed to see the defective ordering, so it proves nothing "
        "about the real files"
    )


def test_the_checker_ignores_the_early_return_unpersist():
    """
    A file with ONLY the early-return unpersist and a later count must not be
    read as the defect. That branch drops the cache before anything is counted,
    which is correct, and an earlier version of this check that walked into If
    bodies would have flagged it.
    """
    fine = textwrap.dedent(
        """
        class BLSIntraSourceLinker:
            def enrich(self, triples_df):
                if not new_dfs:
                    self._bls_triples.unpersist()
                    return empty
                total = all_new.count()
                self._bls_triples.unpersist()
                return all_new
        """
    )
    fn = _enrich_of(fine, "BLSIntraSourceLinker")
    assert _top_level_unpersist(fn) > _top_level_count(fn)
