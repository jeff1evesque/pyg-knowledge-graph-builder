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

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_jobs.pyg_builder.edge_feature_extractor import EdgeFeatureExtractor
from spark_jobs.pyg_builder.edge_mapper import EdgeMapper
from spark_jobs.pyg_builder.node_mapper import NodeMapper

CPI_INDEX = "https://jefflevesque.com/ontology/cpi/Index"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PRECEDES = "https://jefflevesque.com/ontology/bls-common/enrichment/precedes"
HAS_MONTH = "https://example.org/hasMonth"
HAS_YEAR = "https://example.org/hasYear"


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
