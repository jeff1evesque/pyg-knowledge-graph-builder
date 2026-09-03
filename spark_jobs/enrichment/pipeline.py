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
from spark_jobs.enrichment.intra_source.market.patterns import (
    get_sub_industries, get_ticker_cik_map,
)
from spark_jobs.enrichment.ontology_mapper import OntologyMapper
from spark_jobs.settle import settle
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
            # Two independent movements, kept apart because their sum is not
            # interpretable. Every phase unions its output and dropDuplicates
            # over the WHOLE frame, so duplicates already present in the source
            # are collapsed alongside any the phases produced. On the committed
            # fixtures that is zero -- the samples are small and clean -- but a
            # real source carries them (one SEC day: ~8.7%), and folding the two
            # into one net made a healthy run report NEGATIVE enrichment, which
            # reads as data loss and is not.
            'enrichment_added': 0,
            'duplicates_removed': 0,
            # Retained: the true net change, and the key the job manifest has
            # always carried. It is the sum of the two above, not a measure of
            # enrichment on its own.
            'total_enrichment': 0
        }

    def _settle(self, df: DataFrame) -> DataFrame:
        """Materialize and truncate the plan between phases.

        The body lives in ``spark_jobs.settle`` because the temporal unifier
        settles its own frames the same way and used to do it with a second,
        divergent copy -- that copy still called ``localCheckpoint`` after this
        one moved to reliable checkpoints, and it aborted the 2026-08-30 run.
        Read the reasoning there.

        Kept as a method so the plan-truncation tests can call it unbound.
        """
        return settle(df)

    def run(
        self,
        enable_ontology_mapping: bool = True,
        enable_skos: bool = False,
        class_mappings: Optional[Dict[str, Optional[str]]] = None,
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

        # The constituents CSV is read HERE, on the driver, once — not inside
        # the linker. Two reasons: the linker runs against a DataFrame and
        # should not also own an S3 client, and both tables are small enough
        # (~500 rows) that reading them before the phase starts costs one GET
        # and keeps the executors out of it entirely.
        cross_new = enrich_cross_source(
            self.spark,
            self.triples_df,
            ticker_cik_map=get_ticker_cik_map(
                bucket=self._sector_definitions_bucket,
                key=self._sector_definitions_key,
            ),
            sub_industries=get_sub_industries(
                bucket=self._sector_definitions_bucket,
                key=self._sector_definitions_key,
            ),
        )

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

            ontology_mapper = OntologyMapper(self.spark, class_mappings)
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

        # What the phases actually produced, summed from what each reported
        # rather than derived from the total -- the whole point is that the
        # total cannot tell these apart.
        self.stats['enrichment_added'] = (
            sum(
                source.get('total_triples_added', 0)
                for source in self.stats['intra_source'].values()
            )
            + self.stats['temporal_triples']
            + self.stats['cross_source_triples']
            + self.stats['ontology_mapping_triples']
        )

        # The remainder is what dropDuplicates collapsed. Reported as a positive
        # count of rows removed, so the arithmetic reads
        # initial + added - removed = final.
        self.stats['duplicates_removed'] = (
            self.stats['initial_triples']
            + self.stats['enrichment_added']
            - self.stats['final_triples']
        )

        self._print_summary()
        return self.stats

    def get_enriched_triples_df(self) -> Optional[DataFrame]:
        """Get the final enriched triples DataFrame for downstream use."""
        return self.triples_df

    def _print_summary(self):
        """The triple accounting, with the two movements kept apart.

        Reads as arithmetic that closes: initial + added - removed = final.
        The previous form printed only the net, which on any real source is
        dominated by duplicate removal and therefore goes negative -- a healthy
        run reporting what looks like a six-figure loss. It also omitted the
        intra-source phase entirely, so the lines did not sum to the total they
        sat above.
        """
        intra = sum(
            source.get('total_triples_added', 0)
            for source in self.stats['intra_source'].values()
        )
        logger.info("\n" + "=" * 80)
        logger.info("ENRICHMENT PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Initial triples:         {self.stats['initial_triples']:>10,}")
        logger.info("")
        logger.info(f"    Intra-source triples:  {intra:>10,}")
        logger.info(f"    Temporal triples:      {self.stats['temporal_triples']:>10,}")
        logger.info(f"    Cross-source triples:  {self.stats['cross_source_triples']:>10,}")
        logger.info(f"    Ontology mapping:      {self.stats['ontology_mapping_triples']:>10,}")
        logger.info(f"  Enrichment added:        {self.stats['enrichment_added']:>+10,}")
        logger.info("")
        # Source duplicates, not anything enrichment did -- see the stats dict.
        logger.info(f"  Duplicates removed:      {-self.stats['duplicates_removed']:>+10,}")
        logger.info("")
        logger.info(f"  Final triples:           {self.stats['final_triples']:>10,}")
        logger.info(f"  Net change:              {self.stats['total_enrichment']:>+10,}")
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