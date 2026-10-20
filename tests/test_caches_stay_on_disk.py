"""
Keep every frame the Spark jobs cache on disk, not in memory (#466).

When an executor stores a cached block in memory and its memory for cached
blocks is full, Spark pushes other blocks out to make room. If the block it
picks is a GPU broadcast, writing it out needs the GPU. The storing task then
holds the memory lock and waits for the GPU, while the tasks holding the GPU
wait for the memory lock, and the executor freezes until the driver drops it.
The graph job froze this way on 2026-09-06 (#380), and the enrichment job on
2026-10-18. A frame persisted with StorageLevel.DISK_ONLY never asks for room
in memory, so it cannot start this.

So any `.cache()`, or any `.persist()` without DISK_ONLY, in the code under
spark_jobs/ fails here. Everything there runs inside one of the two jobs. The
only exceptions are two schema-sized frames in feature_extractor.py, which #381
kept in memory on purpose.

The scan parses the code, so a comment or docstring that mentions `.cache()`
does not count. As in tests/test_enrichment_cache_ordering.py, the checker is
shown catching each shape before it is trusted.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Thousands of rows each, not graph-sized.
ALLOWED_IN_MEMORY = {
    "spark_jobs/pyg_builder/feature_extractor.py": {
        "all_hierarchy.cache()",
        "schema_df.cache()",
    },
}


def _disk_only(call):
    """Whether a .persist() call passes StorageLevel.DISK_ONLY."""
    levels = list(call.args) + [keyword.value for keyword in call.keywords]
    return any(
        isinstance(level, ast.Attribute) and level.attr == "DISK_ONLY"
        for level in levels
    )


def _kept_in_memory(source):
    """(line, code) of each .cache(), and each .persist() without DISK_ONLY."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method = node.func.attr
        if method == "cache" or (method == "persist" and not _disk_only(node)):
            found.append((node.lineno, ast.unparse(node)))
    return sorted(found)


def _spark_job_files():
    return sorted((REPO_ROOT / "spark_jobs").rglob("*.py"))


def test_no_spark_job_keeps_a_cached_frame_in_memory():
    offenders = []
    for path in _spark_job_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        allowed = ALLOWED_IN_MEMORY.get(relative, set())
        for line, code in _kept_in_memory(path.read_text()):
            if code not in allowed:
                offenders.append(f"{relative}:{line}: {code}")
    assert not offenders, (
        "These keep a cached frame in executor memory:\n  "
        + "\n  ".join(offenders)
        + "\nUse .persist(StorageLevel.DISK_ONLY) instead. A block stored in "
        "memory can freeze an executor when Spark pushes a GPU broadcast out to "
        "make room for it. #466"
    )


def test_the_scan_reads_both_jobs():
    """
    The scan finds the two frames it allows, so it is reading the real files.
    If one of them goes away, take it out of ALLOWED_IN_MEMORY too.
    """
    files = {path.relative_to(REPO_ROOT).as_posix() for path in _spark_job_files()}
    assert "spark_jobs/enrichment/temporal_unifier.py" in files
    for relative, allowed in ALLOWED_IN_MEMORY.items():
        source = (REPO_ROOT / relative).read_text()
        found = {code for _, code in _kept_in_memory(source)}
        assert allowed <= found, (
            f"{relative} no longer has {sorted(allowed - found)}; "
            "take it out of ALLOWED_IN_MEMORY"
        )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("df = df.cache()", ["df.cache()"]),
        ("df.cache()", ["df.cache()"]),
        ("df = df.persist()", ["df.persist()"]),
        (
            "df = df.persist(StorageLevel.MEMORY_AND_DISK)",
            ["df.persist(StorageLevel.MEMORY_AND_DISK)"],
        ),
        ("df = df.persist(StorageLevel.DISK_ONLY)", []),
        ("df = df.persist(\n    StorageLevel.DISK_ONLY\n)", []),
        ("df = df.persist(storageLevel=StorageLevel.DISK_ONLY)", []),
        ('# DISK_ONLY, not .cache()\n"""Nor .cache() in here."""', []),
    ],
    ids=[
        "cache",
        "cache-as-a-statement",
        "persist-with-the-default-level",
        "persist-in-memory",
        "disk-only",
        "disk-only-over-two-lines",
        "disk-only-by-keyword",
        "comment-and-docstring",
    ],
)
def test_the_checker_sees_each_shape(source, expected):
    assert [code for _, code in _kept_in_memory(source)] == expected
