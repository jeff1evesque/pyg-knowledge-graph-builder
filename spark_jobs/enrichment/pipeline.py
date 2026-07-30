"""
Main Enrichment Pipeline Orchestrator

Architecture:
- Intra-source enrichers run independently on PySpark
- After intra-source: all new triples merged into triples_df on executors
- Temporal unification runs on PySpark (cross-source temporal alignment)
- Cross-source enrichment runs on PySpark reading from triples_df
- Ontology mapping runs on PySpark (optional)
"""
from pyspark.sql import SparkSession, DataFrame
from spark_jobs.enrichment.intra_source_linker import enrich_intra_source
from spark_jobs.enrichment.temporal_unifier import TemporalUnifier
from spark_jobs.enrichment.cross_source_linker import enrich_cross_source
from spark_jobs.enrichment.ontology_mapper import OntologyMapper
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """
    Orchestrates enrichment on PySpark executors.

    triples_df (executors): source of truth, grows with each phase
    """

    def __init__(
        self,
        spark: SparkSession,
        triples_df: DataFrame,
        sector_definitions_bucket: str = "",
        sector_definitions_key: str = "",
    ):
        self.spark = spark
        self.triples_df = triples_df
        self._sector_definitions_bucket = sector_definitions_bucket
        self._sector_definitions_key = sector_definitions_key
        self.stats: Dict[str, any] = {
            'initial_triples': 0,
            'intra_source': {},
            'temporal_triples': 0,
            'cross_source_triples': 0,
            'ontology_mapping_triples': 0,
            'final_triples': 0,
            'total_enrichment': 0
        }

    def _settle(self, df: DataFrame) -> DataFrame:
        """Materialize *and truncate* the logical plan.

        The pipeline grows ``triples_df`` by unioning each phase's output, so
        its logical plan deepens on every phase. ``.cache()`` materializes the
        rows but does NOT truncate the logical plan — Catalyst keeps
        re-analyzing the ever-deeper union/dropDuplicates/regexp tree, and
        constraint inference makes the plan blow up super-linearly (measured
        6 -> 436 -> 5,854 plan nodes over three phases on a 184-triple fixture,
        then OutOfMemoryError with ~1.5M constraint BitSets on the driver
        heap). ``localCheckpoint(eager=True)`` writes the rows to the block
        manager and replaces the plan with a single-node scan, so each phase
        starts planning fresh and the node count stays flat.

        localCheckpoint is executor-local (not fault-tolerant); that is fine
        for a batch job, which simply recomputes from source on failure, and
        avoids the reliable-checkpoint requirement of an HDFS/S3 checkpoint dir.
        """
        return df.localCheckpoint(eager=True)

    def run(
        self,
        enable_ontology_mapping: bool = True,
        enable_skos: bool = False,
    ) -> Dict[str, any]:
        logger.info("=" * 80)
        logger.info("STARTING ENRICHMENT PIPELINE")
        logger.info("=" * 80)

        self.stats['initial_triples'] = self.triples_df.count()
        logger.info(f"Initial triples_df size: {self.stats['initial_triples']} triples")

        # Truncate the source lineage before any phase builds on it. For the
        # turtle_parquet path this also detaches the per-row rdflib parse UDF
        # so it is not re-planned into every downstream phase.
        self.triples_df = self._settle(self.triples_df)

        # ============================================
        # PHASE 1: INTRA-SOURCE ENRICHMENT
        # ============================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: INTRA-SOURCE ENRICHMENT")
        logger.info("=" * 80)

        # Run all intra-source enrichers
        intra_result = enrich_intra_source(
            self.spark,
            self.triples_df,
            sector_definitions_bucket=self._sector_definitions_bucket,
            sector_definitions_key=self._sector_definitions_key,
        )

        self.stats['intra_source'] = intra_result.get('stats', {})

        # Merge all new triples into triples_df on executors
        # PySpark enricher output (already on executors)
        spark_new = intra_result.get('spark_new_triples')
        if spark_new is not None:
            old_df = self.triples_df
            self.triples_df = self._settle(
                self.triples_df
                .unionByName(spark_new)
                .dropDuplicates(["subject", "predicate", "object"])
            )
            old_df.unpersist()

            logger.info("Merged intra-source enrichment into triples_df on executors")

        # ============================================
        # PHASE 2: TEMPORAL UNIFICATION (PySpark)
        # ============================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: TEMPORAL UNIFICATION")
        logger.info("=" * 80)

        temporal_unifier = TemporalUnifier(self.spark)
        temporal_new = temporal_unifier.enrich(self.triples_df)

        temporal_count = temporal_new.count()
        self.stats['temporal_triples'] = temporal_count

        if temporal_count > 0:
            old_df = self.triples_df
            self.triples_df = self._settle(
                self.triples_df
                .unionByName(temporal_new)
                .dropDuplicates(["subject", "predicate", "object"])
            )
            old_df.unpersist()
            temporal_new.unpersist()

        logger.info(f"Temporal unification added {temporal_count} triples")

        # ============================================
        # PHASE 3: CROSS-SOURCE ENRICHMENT (PySpark)
        # ============================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: CROSS-SOURCE ENRICHMENT")
        logger.info("=" * 80)

        cross_new = enrich_cross_source(self.spark, self.triples_df)

        cross_count = cross_new.count()
        self.stats['cross_source_triples'] = cross_count

        if cross_count > 0:
            old_df = self.triples_df
            self.triples_df = self._settle(
                self.triples_df
                .unionByName(cross_new)
                .dropDuplicates(["subject", "predicate", "object"])
            )
            old_df.unpersist()
            cross_new.unpersist()

        logger.info(f"Cross-source enrichment added {cross_count} triples")

        # ============================================
        # PHASE 4: ONTOLOGY MAPPING (PySpark, optional)
        # ============================================
        if enable_ontology_mapping:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 4: ONTOLOGY MAPPING")
            logger.info("=" * 80)

            ontology_mapper = OntologyMapper(self.spark)
            ontology_new = ontology_mapper.enrich(
                self.triples_df, enable_skos=enable_skos
            )

            ontology_count = ontology_new.count()
            self.stats['ontology_mapping_triples'] = ontology_count

            if ontology_count > 0:
                old_df = self.triples_df
                self.triples_df = self._settle(
                    self.triples_df
                    .unionByName(ontology_new)
                    .dropDuplicates(["subject", "predicate", "object"])
                )
                old_df.unpersist()
                ontology_new.unpersist()

            logger.info(f"Ontology mapping added {ontology_count} triples")

        # Final stats
        self.stats['final_triples'] = self.triples_df.count()

        self.stats['total_enrichment'] = (
            self.stats['final_triples'] - self.stats['initial_triples']
        )

        self._print_summary()
        return self.stats

    def get_enriched_triples_df(self) -> Optional[DataFrame]:
        """Get the final enriched triples DataFrame for downstream use."""
        return self.triples_df

    def _print_summary(self):
        logger.info("\n" + "=" * 80)
        logger.info("ENRICHMENT PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Initial triples:         {self.stats['initial_triples']:>10,}")
        logger.info(f"  Temporal triples:        {self.stats['temporal_triples']:>10,}")
        logger.info(f"  Cross-source triples:    {self.stats['cross_source_triples']:>10,}")
        logger.info(f"  Ontology mapping triples:{self.stats['ontology_mapping_triples']:>10,}")
        logger.info(f"  Final triples:           {self.stats['final_triples']:>10,}")
        logger.info(f"  Total enrichment:        {self.stats['total_enrichment']:>10,}")
        logger.info("=" * 80)


def enrich_graph(
    spark: SparkSession,
    triples_df: DataFrame,
    enable_ontology_mapping: bool = True,
    enable_skos: bool = False,
) -> Dict[str, any]:
    """Main entry point for graph enrichment."""
    pipeline = EnrichmentPipeline(spark, triples_df)
    return pipeline.run(
        enable_ontology_mapping=enable_ontology_mapping,
        enable_skos=enable_skos,
    )