"""Structural guards against plan-truncation regressions.

Three times now, a frame reused across a loop has been ``.cache()``d without
truncating its logical plan. ``.cache()`` materializes rows; it does NOT replace
the plan. Every downstream query therefore re-analyzes the entire upstream
lineage on the driver, and Catalyst's constraint inference makes that cost grow
super-linearly:

  * ``56b6a23`` — ``EnrichmentPipeline._settle``: 6 -> 436 -> 5,854 plan nodes
    over three phases, then OutOfMemoryError on the driver heap.
  * ``c716d44`` — plan-string rendering exhausted the heap
    (``spark.sql.maxPlanStringLength``).
  * The edge-feature work — the same pattern on the frames shared across the
    per-edge-type loop: 138.9s of a 161s pipeline was driver-side re-analysis,
    which ``localCheckpoint(eager=True)`` took to 14.2s.

Each was found by a human noticing slowness or a crash. Nothing asserted the
invariant, so nothing stopped it coming back.

WHY THESE TESTS MEASURE A DIFFERENTIAL, NOT A THRESHOLD
-------------------------------------------------------
The obvious guard -- "the phase must finish in under N seconds" -- is a flaky
test on a shared machine, and a flaky performance test is one people learn to
ignore. An absolute plan-node ceiling is not much better: the legitimate plan
size changes whenever the encoder changes, so the ceiling needs maintenance and
drifts upward until it asserts nothing.

So these tests assert the *invariant* instead: **truncation makes downstream
plans independent of upstream lineage depth.** The same pipeline is run twice
over inputs that differ only in how deep their lineage is, and the resulting
plan sizes must be identical. That is time-independent, immune to machine load,
and needs no maintenance as the encoder evolves -- the number it compares
against is the other run, not a constant.

Measured on the fixtures below when written:

    current code    shallow 305   deep 305    delta 0
    .cache() only   shallow 3307  deep 7472   delta 4165

Deliberately in the FAST suite (no ``e2e`` marker): these guard a defect class
that has already cost three incidents, so they must run on every push, not only
in the manual e2e job.
"""

import ast
import logging
from pathlib import Path

from pyspark import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_jobs.settle import settle
from spark_jobs.pyg_builder.edge_feature_extractor import EdgeFeatureExtractor
from spark_jobs.pyg_builder.edge_mapper import EdgeMapper
from spark_jobs.pyg_builder.node_mapper import NodeMapper

CPI_INDEX = "https://jefflevesque.com/ontology/cpi/Index"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PRECEDES = "https://jefflevesque.com/ontology/bls-common/enrichment/precedes"
HAS_MONTH = "https://example.org/hasMonth"
HAS_YEAR = "https://example.org/hasYear"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

TRIPLE_SCHEMA = "subject STRING, predicate STRING, object STRING"


def _plan_nodes(df) -> int:
    """Node count of a DataFrame's analyzed logical plan.

    ``numberedTreeString`` renders one line per node, which is the cheapest
    handle on plan size that does not depend on Spark internals beyond the
    public QueryExecution.
    """
    tree = df._jdf.queryExecution().logical().numberedTreeString()
    return len([line for line in tree.split("\n") if line.strip()])


def _temporal_rows(n=6):
    """A minimal chain of typed, dated observations linked by a temporal edge."""
    rows = []
    for i in range(n):
        uri = f"https://jefflevesque.com/ontology/cpi/obs{i}"
        rows += [
            (uri, RDF_TYPE, CPI_INDEX),
            (uri, HAS_MONTH, str((i % 12) + 1)),
            (uri, HAS_YEAR, "2020"),
        ]
    for i in range(n - 1):
        rows.append((
            f"https://jefflevesque.com/ontology/cpi/obs{i}",
            PRECEDES,
            f"https://jefflevesque.com/ontology/cpi/obs{i + 1}",
        ))
    return rows


def _labelled_rows(n=6):
    """``_temporal_rows`` plus an ``rdfs:label`` on every node.

    The temporal fixture carries no labels, and ``_extract_node_labels``
    filters to exactly that predicate -- run against it the extractor returns
    an empty frame, and a guard on an empty frame asserts nothing.
    """
    rows = _temporal_rows(n)
    rows += [
        (
            f"https://jefflevesque.com/ontology/cpi/obs{i}",
            RDFS_LABEL,
            f"CPI observation {i}",
        )
        for i in range(n)
    ]
    return rows


def _node_table(spark, triples):
    """The node id table both endpoint extractors join against."""
    node_id_df, _ = NodeMapper(spark, {}).build_node_id_table(triples)
    return node_id_df


def _deepen(df, rounds):
    """Grow a frame's logical plan without changing the rows it holds.

    ``union`` + ``dropDuplicates`` is what the enrichment pipeline actually does
    to ``triples_df`` phase by phase, so this reproduces the real shape of the
    lineage -- and because the union is with itself, the row set is unchanged.
    Anything downstream that truncates its plan is unaffected by this; anything
    that merely caches carries every added node into each query it feeds.
    """
    for _ in range(rounds):
        df = df.union(df).dropDuplicates()
    return df


def _max_consumed_plan(spark, triples, monkeypatch):
    """Run the real edge-feature path; return the largest plan handed to Spark.

    Spies on the driver-side consumption points rather than on any particular
    intermediate frame: what matters is the plan of whatever the per-edge-type
    loop finally executes, whichever call executes it.
    """
    cfg = {}
    node_id_df, counts = NodeMapper(spark, cfg).build_node_id_table(triples)
    edge_indices, edges_final = EdgeMapper(spark, cfg).build_edge_indices(
        triples, node_id_df, counts
    )

    seen = []
    original_collect = DataFrame.collect
    original_iter = DataFrame.toLocalIterator

    def spy_collect(self):
        seen.append(_plan_nodes(self))
        return original_collect(self)

    def spy_iter(self, prefetchPartitions=False):
        seen.append(_plan_nodes(self))
        return original_iter(self, prefetchPartitions)

    # monkeypatch restores these even if the extractor raises, so a failure here
    # cannot leak a patched DataFrame into the rest of the session-scoped suite.
    monkeypatch.setattr(DataFrame, "collect", spy_collect)
    monkeypatch.setattr(DataFrame, "toLocalIterator", spy_iter)

    features = EdgeFeatureExtractor(spark, cfg).build_edge_features(
        triples, node_id_df, edges_final, edge_indices
    )

    monkeypatch.undo()

    assert features, (
        "the fixture produced no edge features -- this guard would pass "
        "vacuously; fix the fixture rather than the assertion"
    )
    # A refactor that stopped collecting entirely would leave `seen` empty and
    # make the comparison below trivially true. Require real evidence.
    assert seen, "no driver-side collect observed; the guard measured nothing"

    return max(seen)


# --------------------------------------------------------------------------- #
# the edge-feature loop
# --------------------------------------------------------------------------- #

def test_edge_feature_plans_are_independent_of_upstream_lineage(
    spark, monkeypatch
):
    """Deepening the input's lineage must not change the per-edge-type plans.

    This is the guard for the frames shared across the per-edge-type loop
    (``edges_with_idx``, endpoint numerics, endpoint labels). They are
    ``localCheckpoint(eager=True)``ed precisely so that upstream lineage stops
    at them; if any is downgraded to ``.cache()``, the enrichment lineage flows
    into every per-type query and this delta becomes large.

    Both runs execute the identical row set -- ``_deepen`` unions a frame with
    itself and dedupes -- so any difference in plan size is lineage leakage and
    nothing else.
    """
    rows = _temporal_rows()

    shallow_df = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    deep_df = _deepen(
        spark.createDataFrame(
            rows, schema="subject STRING, predicate STRING, object STRING"
        ),
        rounds=4,
    )

    # Confirm the setup actually created the condition under test. Without this
    # a change that made _deepen a no-op would leave the test green and blind.
    assert _plan_nodes(deep_df) > _plan_nodes(shallow_df), (
        "_deepen did not deepen the plan; the test cannot detect leakage"
    )

    shallow = _max_consumed_plan(spark, shallow_df, monkeypatch)
    deep = _max_consumed_plan(spark, deep_df, monkeypatch)

    assert deep == shallow, (
        f"per-edge-type plan grew with upstream lineage depth "
        f"({shallow} -> {deep} nodes). A frame shared across the edge-type "
        f"loop is materialized but not TRUNCATED -- almost certainly a "
        f".cache() where localCheckpoint(eager=True) is required. See "
        f"build_edge_features in edge_feature_extractor.py; .cache() keeps the "
        f"rows but leaves the plan, so every per-type query re-analyzes the "
        f"whole enrichment lineage on the driver."
    )


# --------------------------------------------------------------------------- #
# the enrichment pipeline — the site that already regressed once
# --------------------------------------------------------------------------- #

def test_settle_truncates_the_plan_to_a_leaf(spark):
    """``_settle`` must replace the plan, not merely materialize the rows.

    ``.cache()`` would satisfy any row-level assertion while leaving the plan
    fully intact, which is exactly how ``56b6a23`` happened.
    """
    from spark_jobs.enrichment.pipeline import EnrichmentPipeline

    df = _deepen(
        spark.createDataFrame(
            _temporal_rows(),
            schema="subject STRING, predicate STRING, object STRING",
        ),
        rounds=4,
    )
    assert _plan_nodes(df) > 1, "fixture is not deep enough to test truncation"

    settled = EnrichmentPipeline._settle(None, df)

    assert _plan_nodes(settled) == 1, (
        f"_settle left a {_plan_nodes(settled)}-node plan; it must collapse to "
        f"a single scan. .cache() materializes rows but keeps the plan, and "
        f"Catalyst then re-analyzes the whole tree every phase (measured "
        f"6 -> 436 -> 5,854 nodes, then driver OOM)."
    )
    assert settled.count() == df.count(), "_settle changed the row set"


def test_settle_keeps_plan_flat_across_successive_phases(spark):
    """Phase-by-phase growth must stay flat, which is the actual failure shape.

    The pipeline unions each phase's output into ``triples_df``. Settling once
    is not enough -- the guard is that node count does not compound across
    phases. Without truncation this sequence is the measured 6 -> 436 -> 5,854.
    """
    from spark_jobs.enrichment.pipeline import EnrichmentPipeline

    df = spark.createDataFrame(
        _temporal_rows(),
        schema="subject STRING, predicate STRING, object STRING",
    )

    sizes = []
    for phase in range(3):
        addition = df.withColumn(
            "subject", F.concat(F.col("subject"), F.lit(f"_p{phase}"))
        )
        df = EnrichmentPipeline._settle(None, df.union(addition).dropDuplicates())
        sizes.append(_plan_nodes(df))

    assert sizes == [1, 1, 1], (
        f"plan node count compounded across phases: {sizes}. Each phase must "
        f"start from a truncated plan; growth here is the super-linear "
        f"constraint-inference blowup that OOMed the driver at 4g and 8g."
    )


# --------------------------------------------------------------------------- #
# settle() is the ONLY place that decides how a frame is materialized
# --------------------------------------------------------------------------- #

_SPARK_JOBS_DIR = Path(__file__).resolve().parent.parent / "spark_jobs"
_PYG_BUILDER_DIR = _SPARK_JOBS_DIR / "pyg_builder"
_SETTLE_MODULE = _SPARK_JOBS_DIR / "settle.py"


def _local_checkpoint_calls(path: Path):
    """Line numbers of real ``.localCheckpoint(...)`` calls in a source file.

    Parsed, not grepped, so the prose in settle()'s own docstring -- which has
    to name the method it is replacing -- cannot trip the guard.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "localCheckpoint"
    )


def test_only_settle_chooses_the_checkpoint_kind():
    """No module under ``spark_jobs`` may call ``localCheckpoint`` for itself.

    This is the guard for the defect that actually happened, and a behavioural
    test could not have caught it. ``_settle`` moved to reliable
    ``checkpoint()`` so a lost executor could not destroy unrecoverable work --
    but two other copies of the same idea, in ``temporal_unifier`` (twice) and
    ``cross_source_linker``, kept calling ``localCheckpoint`` and nobody
    noticed. Nothing in the unit suite reaches those lines; they only run on a
    real multi-source job.

    So it stayed green until 2026-08-30, when a four-source run lost two
    executors inside temporal unification's ``dated`` frame and died on
    ``Checkpoint block rdd_23348_23 not found`` at 157 minutes, taking both
    downstream legs with it. ``localCheckpoint`` truncates the plan, so once
    the blocks are gone there is nothing left to recompute from.

    Scoped to the whole of ``spark_jobs`` rather than to ``enrichment``,
    because narrowing it to the package where the incident happened is what
    let assembly keep five sites of its own. A guard aimed at the last
    incident only ever catches the last incident.

    The invariant is structural, so assert it structurally: exactly one module
    picks the storage level, and every caller goes through it.
    """
    offenders = {
        path.relative_to(_SPARK_JOBS_DIR.parent): lines
        for path in sorted(_SPARK_JOBS_DIR.rglob("*.py"))
        if path != _SETTLE_MODULE
        for lines in [_local_checkpoint_calls(path)]
        if lines
    }

    assert not offenders, (
        f"localCheckpoint called outside settle(): "
        f"{ {str(p): ls for p, ls in offenders.items()} }. Route it through "
        f"spark_jobs.settle instead. A memory-only checkpoint is "
        f"destroyed with the executor holding it, and because it truncates the "
        f"plan there is nothing left to recompute -- the task fails four times "
        f"and the whole job aborts."
    )


def test_settle_writes_a_reliable_checkpoint_when_a_dir_is_set(
    spark, monkeypatch
):
    """With a checkpoint dir, settle() must not choose the memory-only copy.

    ``build_graph`` sets the dir to the run's work dir, so this is the path
    every cluster run takes. Patched rather than driven against a real dir: the
    ``spark`` fixture is session-scoped and PySpark offers no way to unset a
    checkpoint dir, so setting one for real would leak into every test after.
    """
    df = spark.createDataFrame(
        _temporal_rows(), schema="subject STRING, predicate STRING, object STRING"
    )
    calls = []
    monkeypatch.setattr(
        SparkContext, "getCheckpointDir", lambda self: "/tmp/checkpoints"
    )
    monkeypatch.setattr(
        DataFrame, "checkpoint", lambda self, eager=True: calls.append("reliable") or self
    )
    monkeypatch.setattr(
        DataFrame, "localCheckpoint", lambda self, eager=True: calls.append("local") or self
    )

    settle(df)

    assert calls == ["reliable"], (
        f"settle() made {calls or 'no checkpoint'} with a checkpoint dir set; "
        f"it must call checkpoint(eager=True). localCheckpoint here is what "
        f"turns one evicted executor into an aborted run."
    )


def test_settle_falls_back_to_local_and_says_so_without_a_dir(
    spark, monkeypatch, caplog
):
    """No dir means no reliable option, but it must not pass silently.

    The unit suite sets no checkpoint dir, so this is the path the tests
    themselves take -- the fallback exists to keep them running unchanged. The
    warning is the part worth asserting: it is the only signal that a cluster
    run has lost its safety net.
    """
    df = spark.createDataFrame(
        _temporal_rows(), schema="subject STRING, predicate STRING, object STRING"
    )
    calls = []
    monkeypatch.setattr(SparkContext, "getCheckpointDir", lambda self: None)
    monkeypatch.setattr(
        DataFrame, "checkpoint", lambda self, eager=True: calls.append("reliable") or self
    )
    monkeypatch.setattr(
        DataFrame, "localCheckpoint", lambda self, eager=True: calls.append("local") or self
    )

    with caplog.at_level(logging.WARNING, logger="spark_jobs.settle"):
        settle(df)

    assert calls == ["local"], f"settle() made {calls} with no checkpoint dir"
    assert "does not survive losing an executor" in caplog.text, (
        "the fallback must warn; without a checkpoint dir a cluster run loses "
        "every settled phase to a single evicted executor, and the log is the "
        "only place that is visible"
    )


# --------------------------------------------------------------------------- #
# the assembly path — the same sites, one package over
# --------------------------------------------------------------------------- #

def _assembly_inputs(spark, rows=None):
    """Node ids and edges for a build_edge_features call, built unpatched.

    Kept out of the patched window on purpose: only what the extractor chooses
    should be recorded, not what the mappers do to get there.
    """
    triples = spark.createDataFrame(rows or _labelled_rows(), schema=TRIPLE_SCHEMA)
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    edge_indices, edges_final = EdgeMapper(spark, {}).build_edge_indices(
        triples, node_id_df, counts
    )
    return triples, node_id_df, edges_final, edge_indices


def _record_checkpoint_kinds(monkeypatch, checkpoint_dir):
    """Record which checkpoint kind each settled frame chose.

    Both methods return ``self``, so the frames stay lazy and the fixture
    stays cheap -- these tests assert the choice, not the truncation. The
    truncation is what the lineage tests above assert.
    """
    calls = []
    monkeypatch.setattr(
        SparkContext, "getCheckpointDir", lambda self: checkpoint_dir
    )
    monkeypatch.setattr(
        DataFrame,
        "checkpoint",
        lambda self, eager=True: calls.append("reliable") or self,
    )
    monkeypatch.setattr(
        DataFrame,
        "localCheckpoint",
        lambda self, eager=True: calls.append("local") or self,
    )
    return calls


def test_numeric_property_extraction_truncates_independently_of_lineage(spark):
    """Endpoint numerics must leave a leaf plan however deep the input is.

    This frame is read by every edge type's segment-1/2 encoding, so leakage
    here is paid once per type, not once. On the 2026-08 graph it is
    305,127,393 rows before narrowing -- deep lineage on it is what put ~380
    stages into a single plan.
    """
    shallow = spark.createDataFrame(_temporal_rows(), schema=TRIPLE_SCHEMA)
    deep = _deepen(
        spark.createDataFrame(_temporal_rows(), schema=TRIPLE_SCHEMA), rounds=4
    )
    assert _plan_nodes(deep) > _plan_nodes(shallow), (
        "_deepen did not deepen the plan; the test cannot detect leakage"
    )

    extractor = EdgeFeatureExtractor(spark, {})
    shallow_out = extractor._extract_node_numeric_properties(
        shallow, _node_table(spark, shallow)
    )
    deep_out = extractor._extract_node_numeric_properties(
        deep, _node_table(spark, deep)
    )

    assert shallow_out.count() > 0, (
        "the fixture produced no numeric endpoint properties -- this guard "
        "would pass vacuously; fix the fixture rather than the assertion"
    )
    assert _plan_nodes(shallow_out) == _plan_nodes(deep_out) == 1, (
        f"_extract_node_numeric_properties left plans of "
        f"{_plan_nodes(shallow_out)} and {_plan_nodes(deep_out)} nodes; both "
        f"must collapse to a single scan. A .cache() here keeps the rows and "
        f"leaves the lineage, and every per-edge-type query re-analyzes it."
    )


def test_label_extraction_truncates_independently_of_lineage(spark):
    """Endpoint labels must leave a leaf plan however deep the input is.

    Same shape as the numeric frame and read the same way -- once per edge
    type -- so it carries the same leakage cost and needs the same guard.
    """
    shallow = spark.createDataFrame(_labelled_rows(), schema=TRIPLE_SCHEMA)
    deep = _deepen(
        spark.createDataFrame(_labelled_rows(), schema=TRIPLE_SCHEMA), rounds=4
    )
    assert _plan_nodes(deep) > _plan_nodes(shallow), (
        "_deepen did not deepen the plan; the test cannot detect leakage"
    )

    extractor = EdgeFeatureExtractor(spark, {})
    shallow_out = extractor._extract_node_labels(
        shallow, _node_table(spark, shallow)
    )
    deep_out = extractor._extract_node_labels(deep, _node_table(spark, deep))

    assert shallow_out.count() > 0, (
        "the fixture produced no endpoint labels -- this guard would pass "
        "vacuously; fix the fixture rather than the assertion"
    )
    assert _plan_nodes(shallow_out) == _plan_nodes(deep_out) == 1, (
        f"_extract_node_labels left plans of {_plan_nodes(shallow_out)} and "
        f"{_plan_nodes(deep_out)} nodes; both must collapse to a single scan."
    )


def test_assembly_settles_reliably_when_a_checkpoint_dir_is_set(
    spark, monkeypatch
):
    """With a checkpoint dir, no assembly frame may take the memory-only copy.

    ``build_graph`` sets the dir before it dispatches on mode, so this is the
    path every cluster run takes in ``pyg_only`` as well as ``full``. The
    structural guard proves each site calls ``settle``; this proves what
    ``settle`` then does for the assembly phase, which is the half that
    actually keeps the work when an executor is lost.
    """
    triples, node_id_df, edges_final, edge_indices = _assembly_inputs(spark)

    calls = _record_checkpoint_kinds(monkeypatch, "/tmp/checkpoints")
    EdgeFeatureExtractor(spark, {}).build_edge_features(
        triples, node_id_df, edges_final, edge_indices
    )
    monkeypatch.undo()

    assert calls, (
        "assembly settled nothing; the guard measured nothing. Either the "
        "fixture stopped reaching the settled frames or the calls were removed"
    )
    assert set(calls) == {"reliable"}, (
        f"assembly made {calls} with a checkpoint dir set; every frame must "
        f"take checkpoint(eager=True). These frames hold the whole phase -- "
        f"one evicted executor with a memory-only copy aborts the job, and "
        f"because settling truncates the plan there is nothing to recompute."
    )


def test_assembly_falls_back_to_local_without_a_checkpoint_dir(
    spark, monkeypatch, caplog
):
    """No dir means no reliable option, but assembly must not pass silently.

    The unit suite sets no checkpoint dir, so this is the path these tests
    take themselves. The warning is the part worth asserting: it is the only
    signal that a run has lost its safety net.
    """
    triples, node_id_df, edges_final, edge_indices = _assembly_inputs(spark)

    calls = _record_checkpoint_kinds(monkeypatch, None)
    with caplog.at_level(logging.WARNING, logger="spark_jobs.settle"):
        EdgeFeatureExtractor(spark, {}).build_edge_features(
            triples, node_id_df, edges_final, edge_indices
        )
    monkeypatch.undo()

    assert set(calls) == {"local"}, (
        f"assembly made {calls} with no checkpoint dir; the fallback is "
        f"localCheckpoint"
    )
    assert "does not survive losing an executor" in caplog.text, (
        "the fallback must warn for assembly too; the log is the only place a "
        "missing checkpoint dir is visible"
    )


# --------------------------------------------------------------------------- #
# assembly must not reach into enrichment to get settle()
# --------------------------------------------------------------------------- #

def _imported_modules(path: Path):
    """Module names a source file imports, as written.

    Relative imports keep their leading dots, so ``from ..enrichment import x``
    stays visible even though its absolute name is never spelled in the file.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return names


def test_pyg_builder_does_not_import_enrichment():
    """Assembly may not import from the enrichment package.

    ``settle`` sits at the top of ``spark_jobs`` rather than under
    ``enrichment`` for this reason. The two run in separate modes -- a
    ``pyg_only`` job never builds an EnrichmentPipeline -- and an import the
    other way would make assembly load a package it does not use, for one
    six-line function. It would also invite the next shared helper to be
    reached for in place, which is how there came to be two divergent copies
    of settle in the first place.
    """
    offenders = {
        str(path.relative_to(_SPARK_JOBS_DIR.parent)): names
        for path in sorted(_PYG_BUILDER_DIR.rglob("*.py"))
        for names in [[
            name
            for name in _imported_modules(path)
            if "enrichment" in name.lstrip(".").split(".")
        ]]
        if names
    }

    assert not offenders, (
        f"pyg_builder imports from enrichment: {offenders}. Shared helpers "
        f"belong at the top of spark_jobs -- see spark_jobs/settle.py -- not "
        f"in whichever package happened to need them first."
    )
