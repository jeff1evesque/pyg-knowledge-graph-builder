"""
Main Enrichment Pipeline Orchestrator

Architecture:
- Intra-source enrichers run independently (some rdflib, some PySpark)
- After intra-source: all new triples merged into triples_df on executors
- Cross-source enrichment runs on PySpark reading from triples_df
- rdflib graph only needed for rdflib-based intra-source enrichers
- As enrichers migrate to PySpark, rdflib usage shrinks to zero
"""
from rdflib import Graph
from pyspark.sql import SparkSession, DataFrame
from glue_jobs.enrichment.intra_source_linker import enrich_intra_source
from glue_jobs.enrichment.cross_source_linker import enrich_cross_source
from glue_jobs.utils.spark_rdf_utils import (
    snapshot_rdflib_graph, rdflib_diff_to_triples_df, deduplicate_against_existing
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """
    Orchestrates enrichment across two execution engines.

    triples_df (executors): source of truth, grows with each phase
    rdflib graph (driver): used only by rdflib-based enrichers, shrinks over time
    """

    def __init__(
        self,
        graph: Graph,
        spark: Optional[SparkSession] = None,
        triples_df: Optional[DataFrame] = None
    ):
        self.graph = graph
        self.spark = spark
        self.triples_df = triples_df
        self.stats = {
            'initial_triples': len(graph),
            'intra_source': {},
            'cross_source_triples': 0,
            'final_triples': 0,
            'total_enrichment': 0
        }

    def run(self, enable_ontology_mapping: bool = False) -> Dict[str, any]:
        logger.info("=" * 80)
        logger.info("STARTING ENRICHMENT PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Initial graph size: {self.stats['initial_triples']} triples")
        logger.info(f"PySpark available: {self.spark is not None}")

        # ============================================
        # PHASE 1: INTRA-SOURCE ENRICHMENT
        # ============================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: INTRA-SOURCE ENRICHMENT")
        logger.info("=" * 80)

        # Snapshot rdflib graph BEFORE rdflib enrichers run
        rdflib_snapshot = snapshot_rdflib_graph(self.graph) if self.spark else None

        # Run all intra-source enrichers
        intra_result = enrich_intra_source(
            self.graph,
            spark=self.spark,
            triples_df=self.triples_df
        )

        self.stats['intra_source'] = intra_result.get('stats', {})

        # Merge all new triples into triples_df on executors
        if self.spark is not None and self.triples_df is not None:
            new_dfs = []

            # PySpark enricher output (already on executors)
            spark_new = intra_result.get('spark_new_triples')
            if spark_new is not None:
                new_dfs.append(spark_new)

            # rdflib enricher output (diff → parallelize → executors)
            if rdflib_snapshot is not None:
                rdflib_new = rdflib_diff_to_triples_df(
                    self.spark, self.graph, rdflib_snapshot
                )
                if rdflib_new is not None:
                    new_dfs.append(rdflib_new)

            # Union into triples_df
            if new_dfs:
                combined = new_dfs[0]
                for df in new_dfs[1:]:
                    combined = combined.unionByName(df)

                old_df = self.triples_df
                self.triples_df = (
                    self.triples_df
                    .unionByName(combined)
                    .dropDuplicates(["subject", "predicate", "object"])
                    .cache()
                )
                # Force materialization so old DF can be unpersisted
                self.triples_df.count()
                old_df.unpersist()

                logger.info("Merged intra-source enrichment into triples_df on executors")

        # ============================================
        # PHASE 2: CROSS-SOURCE ENRICHMENT (PySpark)
        # ============================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: CROSS-SOURCE ENRICHMENT")
        logger.info("=" * 80)

        if self.spark is not None and self.triples_df is not None:
            cross_new = enrich_cross_source(self.spark, self.triples_df)

            cross_count = cross_new.count()
            self.stats['cross_source_triples'] = cross_count

            if cross_count > 0:
                old_df = self.triples_df
                self.triples_df = (
                    self.triples_df
                    .unionByName(cross_new)
                    .dropDuplicates(["subject", "predicate", "object"])
                    .cache()
                )
                self.triples_df.count()
                old_df.unpersist()
                cross_new.unpersist()

            logger.info(f"Cross-source enrichment added {cross_count} triples")
        else:
            logger.info("Skipping cross-source enrichment (no Spark session)")

        # ============================================
        # PHASE 3: ONTOLOGY MAPPING (optional)
        # ============================================
        # TODO: Migrate to PySpark. For now, skip or run on rdflib.
        if enable_ontology_mapping:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 3: ONTOLOGY MAPPING")
            logger.info("=" * 80)
            logger.info("  (Ontology mapping not yet migrated to PySpark — skipping)")

        # Final stats
        if self.triples_df is not None:
            self.stats['final_triples'] = self.triples_df.count()
        else:
            self.stats['final_triples'] = len(self.graph)

        self.stats['total_enrichment'] = self.stats['final_triples'] - self.stats['initial_triples']

        self._print_summary()
        return self.stats

    def get_enriched_triples_df(self) -> Optional[DataFrame]:
        """Get the final enriched triples DataFrame for downstream use."""
        return self.triples_df

    def _print_summary(self):
        logger.info("\n" + "=" * 80)
        logger.info("ENRICHMENT PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Initial triples:       {self.stats['initial_triples']:>10,}")
        logger.info(f"  Cross-source triples:  {self.stats['cross_source_triples']:>10,}")
        logger.info(f"  Final triples:         {self.stats['final_triples']:>10,}")
        logger.info(f"  Total enrichment:      {self.stats['total_enrichment']:>10,}")
        logger.info("=" * 80)


def enrich_graph(
    graph: Graph,
    enable_ontology_mapping: bool = False,
    spark: Optional[SparkSession] = None,
    triples_df: Optional[DataFrame] = None
) -> Dict[str, any]:
    """Main entry point for graph enrichment."""
    pipeline = EnrichmentPipeline(graph, spark=spark, triples_df=triples_df)
    return pipeline.run(enable_ontology_mapping=enable_ontology_mapping)