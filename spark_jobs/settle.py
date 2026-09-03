"""Materialize a frame and truncate its logical plan.

At the top of ``spark_jobs`` because both halves of the job settle frames: the
enrichment pipeline and the temporal unifier, and the assembly path in
``pyg_builder``. It is not a method on EnrichmentPipeline because pipeline.py
already imports temporal_unifier.py, so importing back the other way is a
cycle. It is not under ``enrichment`` because assembly runs in its own mode and
should not have to load the enrichment package to settle a frame.
"""
from pyspark.sql import DataFrame
import logging

logger = logging.getLogger(__name__)


def settle(df: DataFrame) -> DataFrame:
    """Materialize *and truncate* the logical plan.

    Callers grow a frame by unioning each phase's output into it, so its
    logical plan deepens every phase. ``.cache()`` materializes the rows but
    does NOT truncate the plan -- Catalyst keeps re-analyzing the ever-deeper
    union/dropDuplicates/regexp tree, and constraint inference makes the plan
    blow up super-linearly (measured 6 -> 436 -> 5,854 plan nodes over three
    phases on a 184-triple fixture, then OutOfMemoryError with ~1.5M constraint
    BitSets on the driver heap). Checkpointing eagerly writes the rows out and
    replaces the plan with a single-node scan, so each phase starts planning
    fresh and the node count stays flat.

    This used ``localCheckpoint(eager=True)``, on the reasoning that an
    executor-local copy is fine for a batch job because it recomputes from
    source on failure. It does not recompute: settling truncates the plan, so
    once the blocks are gone there is nothing left to recompute from, the task
    fails four times and the job aborts.

    That is not a hypothetical. Executors stall under GPU pool pressure on this
    hardware often enough that a long run should expect to lose one, and runs
    survive it -- 2026-08-27 lost two and still exited 0 -- as long as the work
    is reliably checkpointed. The 2026-08-30 run lost two executors inside the
    FIRST of the three ``localCheckpoint`` sites the enrichment path still had
    (temporal unification's ``dated`` frame) and died on ``Checkpoint block
    rdd_23348_23 not found`` after 157 minutes, taking both downstream legs
    with it. It never reached the other two, in temporal unification and in
    the cross-source union; all three now settle through here.

    Assembly had five more of the same sites, in ``edge_feature_extractor``,
    and they carry the same risk for longer: they hold the frames every
    per-edge-type query reads, so losing one is losing the phase. They settle
    through here now too.

    ``checkpoint(eager=True)`` writes the same truncating copy to the session's
    checkpoint dir, which build_graph points at the run's work dir. Callers
    that set no checkpoint dir keep the old behaviour, so the unit suite runs
    unchanged; the session is read off the DataFrame so this works with no
    instance to hand.
    """
    if df.sparkSession.sparkContext.getCheckpointDir():
        return df.checkpoint(eager=True)
    logger.warning(
        "No checkpoint dir is set, so this phase settles with "
        "localCheckpoint, which does not survive losing an executor."
    )
    return df.localCheckpoint(eager=True)
