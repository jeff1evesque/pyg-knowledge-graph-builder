"""
Guard the bound on what one parse call hands back across the Python boundary.

THE DEFECT THIS EXISTS TO CATCH
-------------------------------
The Turtle parse was a ``@F.udf(returnType=ArrayType(...))``, which must return
every triple from a blob as ONE value. Nothing bounded it. Almost every blob is
tiny -- bls, market and noaa are 1-4 KB and a few dozen triples -- so it looked
fine for months and the code said so in a comment: "each blob is small (~60
triples)".

One SEC filing in the 2026-09 data is 5.6 MB. It parses into 57,350 triples,
which is 14.8 MB pickled through the socket in a single piece. When it lands the
executor's writer thread blocks pushing input into a full socket while the task
thread blocks waiting for output that cannot fit, and neither ever moves.

What that looks like from outside is the part worth remembering: no exception,
no lost executor, no OOM, no failed task. The stage sits one task short of done
and both nodes go to ~99% idle until the job's cap kills it hours later. It cost
seven cluster runs and a day on 2026-09-06 before a thread dump showed the two
blocked threads. See #380.

So these tests do not check that parsing is correct -- test_turtle_parser.py and
test_literal_datatypes.py already do. They check the thing that was never
checked: that no single hand-back is unbounded, whatever the input does.

WHY A SIZE TEST AND NOT A DEADLOCK TEST
---------------------------------------
The deadlock itself needs two hosts, a full socket buffer and an unlucky
interleaving; it is not reproducible in a unit test and was not reproducible on
demand on the cluster either -- it took a different partition every run. The
precondition IS testable and is what the fix removes, so that is what is pinned
here. A test that can only fail when the stars align is not a regression guard.

No Spark session: turtle_batches_to_arrow takes and returns pyarrow directly.
"""
import ast
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")

from spark_jobs.build_graph import (  # noqa: E402
    PARSE_BATCH_ROWS,
    TRIPLE_FIELD_NAMES,
    turtle_batches_to_arrow,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_GRAPH = REPO_ROOT / "spark_jobs" / "build_graph.py"


def _turtle(n_triples, prefix="ex"):
    """A well-formed Turtle document stating exactly n_triples distinct triples."""
    lines = [f"@prefix {prefix}: <http://example.org/{prefix}#> ."]
    lines += [f"{prefix}:s{i} {prefix}:p {prefix}:o{i} ." for i in range(n_triples)]
    return "\n".join(lines)


def _batch(blobs):
    """What mapInArrow hands the parser: one column of Turtle strings."""
    return pa.RecordBatch.from_arrays(
        [pa.array(blobs, type=pa.string())], names=["triples"]
    )


def _rows(batches):
    """Flatten yielded batches back to a list of tuples, in order."""
    out = []
    for b in batches:
        cols = [b.column(i).to_pylist() for i in range(b.num_columns)]
        out.extend(zip(*cols))
    return out


# --- the regression guard ---------------------------------------------------


def test_a_huge_blob_is_split_into_bounded_batches():
    """THE test. One blob, many triples, and no single hand-back is unbounded.

    Before the fix this was one value of 57,350 triples. If this assertion ever
    fails again, the cluster will deadlock and give no error while doing it.
    """
    batches = list(turtle_batches_to_arrow([_batch([_turtle(25_000)])], max_rows=1_000))

    assert len(batches) == 25, "25,000 triples at 1,000 per batch must be 25 batches"
    assert all(b.num_rows <= 1_000 for b in batches), (
        "a batch exceeded max_rows -- the bound that prevents the #380 deadlock "
        "is not being applied"
    )


def test_the_bound_holds_across_many_blobs_in_one_partition():
    """The counter must not reset per blob, or a partition can still hand back
    everything at once."""
    blobs = [_turtle(400) for _ in range(50)]  # 20,000 triples over 50 rows
    batches = list(turtle_batches_to_arrow([_batch(blobs)], max_rows=1_000))

    assert sum(b.num_rows for b in batches) == 20_000
    assert all(b.num_rows <= 1_000 for b in batches)


def test_the_bound_holds_across_multiple_input_batches():
    """mapInArrow delivers an iterator of batches, not one."""
    inputs = [_batch([_turtle(3_000)]), _batch([_turtle(3_000, prefix="two")])]
    batches = list(turtle_batches_to_arrow(inputs, max_rows=500))

    assert sum(b.num_rows for b in batches) == 6_000
    assert all(b.num_rows <= 500 for b in batches)


# --- nothing is lost or reordered at a boundary ------------------------------


def test_no_triple_is_lost_or_duplicated_at_a_batch_boundary():
    rows = _rows(turtle_batches_to_arrow([_batch([_turtle(1_000)])], max_rows=7))
    subjects = [r[0] for r in rows]

    assert len(rows) == 1_000
    assert len(set(subjects)) == 1_000, "a subject was dropped or repeated"


def test_order_is_preserved_across_batches():
    flat = _rows(turtle_batches_to_arrow([_batch([_turtle(500)])], max_rows=1_000))
    split = _rows(turtle_batches_to_arrow([_batch([_turtle(500)])], max_rows=13))

    assert flat == split, "batching changed the row order"


def test_a_boundary_landing_exactly_on_max_rows_emits_no_empty_batch():
    batches = list(turtle_batches_to_arrow([_batch([_turtle(20)])], max_rows=10))

    assert [b.num_rows for b in batches] == [10, 10], "trailing empty batch emitted"


# --- shape and schema --------------------------------------------------------


def test_batches_carry_the_four_triple_columns_as_strings():
    batch = next(iter(turtle_batches_to_arrow([_batch([_turtle(5)])])))

    assert batch.schema.names == list(TRIPLE_FIELD_NAMES)
    assert all(f.type == pa.string() for f in batch.schema)


def test_small_input_is_one_batch():
    batches = list(turtle_batches_to_arrow([_batch([_turtle(5)])]))
    assert len(batches) == 1 and batches[0].num_rows == 5


def test_default_bound_is_applied_when_none_is_given():
    batches = list(turtle_batches_to_arrow([_batch([_turtle(PARSE_BATCH_ROWS + 50)])]))
    assert all(b.num_rows <= PARSE_BATCH_ROWS for b in batches)


# --- the error policy the UDF used to own ------------------------------------


def test_an_empty_partition_yields_nothing():
    assert list(turtle_batches_to_arrow([])) == []


def test_a_partition_of_blobs_with_no_triples_yields_nothing():
    """No empty batch, which Spark would otherwise carry through the plan."""
    assert list(turtle_batches_to_arrow([_batch(["", "   ", None])])) == []


def test_a_malformed_blob_is_skipped_and_the_good_ones_survive():
    """A parse error must cost its own blob and nothing else -- the run continues
    and the missing triples show up in the count logged after loading."""
    rows = _rows(
        turtle_batches_to_arrow([_batch([_turtle(3), "@prefix bad", _turtle(2, "b")])])
    )
    assert len(rows) == 5


def test_a_missing_parser_still_fails_the_job(monkeypatch):
    """ImportError is NOT a malformed blob. Under a blanket except, an executor
    venv without pyoxigraph answers "no triples" for every row and the job
    succeeds with an empty graph."""
    import spark_jobs.build_graph as bg

    def no_parser(_):
        raise ImportError("No module named 'pyoxigraph'")

    monkeypatch.setattr(bg, "turtle_rows_or_skip", no_parser)
    with pytest.raises(ImportError):
        list(bg.turtle_batches_to_arrow([_batch([_turtle(3)])]))


# --- the structural guard ----------------------------------------------------


def _array_returning_udfs(source):
    """Names of functions decorated as a Python UDF whose returnType is an array.

    Resolves the schema through a variable, because that is how the defect was
    actually written -- ``triple_schema = ArrayType(...)`` on one line and
    ``@F.udf(returnType=triple_schema)`` on another. A checker that only matches
    ``returnType=ArrayType(...)`` inline passes the very code it exists to catch,
    which is what the first version of this did.
    """
    tree = ast.parse(source)

    array_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        if not any(
            isinstance(sub, ast.Name) and sub.id == "ArrayType"
            for sub in ast.walk(value)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                array_names.add(target.id)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if getattr(dec.func, "attr", getattr(dec.func, "id", "")) != "udf":
                continue
            for kw in dec.keywords:
                if kw.arg != "returnType":
                    continue
                referenced = {
                    sub.id for sub in ast.walk(kw.value) if isinstance(sub, ast.Name)
                }
                if "ArrayType" in referenced or referenced & array_names:
                    offenders.append(node.name)
    return offenders


def test_the_checker_catches_the_shape_that_caused_the_outage():
    """Prove the guard fires before trusting it to guard anything.

    A structural check that cannot flag the original defect is worse than none:
    it reports clean forever and reads like coverage.
    """
    offending = """
from pyspark.sql.types import ArrayType, StructType
triple_schema = ArrayType(StructType([]))

@F.udf(returnType=triple_schema)
def parse_turtle_blob(turtle_str):
    return turtle_rows_or_skip(turtle_str)
"""
    assert _array_returning_udfs(offending) == ["parse_turtle_blob"]

    inline = """
@F.udf(returnType=ArrayType(StructType([])))
def other(x):
    return []
"""
    assert _array_returning_udfs(inline) == ["other"]

    fine = """
@F.udf(returnType=StringType())
def scalar(x):
    return ""
"""
    assert _array_returning_udfs(fine) == []


def test_the_parse_path_does_not_return_an_unbounded_array():
    """Fail if anyone puts the array-returning UDF back.

    The size tests above pin the behaviour of the current function. This pins
    the SHAPE, because the defect was not a wrong number -- it was a design in
    which no number could be wrong. A Python UDF declared with an ArrayType
    return has no bound available to it at all.
    """
    offenders = _array_returning_udfs(BUILD_GRAPH.read_text())

    assert not offenders, (
        f"{offenders} declare a Python UDF returning ArrayType. That hands every "
        f"row's output back as one unbounded value and is what deadlocked the "
        f"cluster in #380. Stream bounded batches instead -- see "
        f"turtle_batches_to_arrow."
    )
