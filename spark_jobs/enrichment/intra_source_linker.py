"""
Intra-Source Enrichment Entry Point

Runs the intra-source linker of each source a run picked and returns their
combined new triples. Which linker a source uses is declared in its spec
(spark_jobs/sources/), so nothing here names a source.

All sources: PySpark (returns DataFrame on executors)
"""
from pyspark.sql import SparkSession, DataFrame
from typing import Dict, List, Optional, Sequence
import logging

from spark_jobs import sources
from spark_jobs.sources.spec import RunOptions, SourceSpec

logger = logging.getLogger(__name__)


def enrich_intra_source(
    spark: SparkSession,
    triples_df: DataFrame,
    sector_definitions_bucket: str = "",
    sector_definitions_key: str = "",
    source_data_day: str = "",
    specs: Optional[Sequence[SourceSpec]] = None,
) -> Dict:
    """
    Run intra-source enrichment for each picked source.

    Args:
        spark: Active SparkSession
        triples_df: DataFrame with columns [subject, predicate, object]
        specs: The sources the run's paths picked. Defaults to every
            registered source; a linker finds nothing in data it does not have.

    Returns:
        Dict with:
        - 'stats': per-source enrichment statistics, keyed by source name
        - 'spark_new_triples': DataFrame of all enrichment output
          (stays on executors, never collected here)
    """
    specs = sources.REGISTERED if specs is None else specs
    options = RunOptions(
        sector_definitions_bucket=sector_definitions_bucket,
        sector_definitions_key=sector_definitions_key,
        source_data_day=source_data_day,
    )
    stats: Dict[str, Dict[str, any]] = {}
    spark_new_dfs: List[DataFrame] = []

    for spec in specs:
        if spec.linker is None:
            continue
        # Run tooling reads these lines ("Market enrichment produced ..."), so
        # their wording is part of what a run reports.
        logger.info(f"Checking for {spec.label} data...")
        try:
            new_df = spec.linker(spark, options).enrich(triples_df)
            spark_new_dfs.append(new_df)
            count = new_df.count()
            stats[spec.name] = {'total_triples_added': count}
            logger.info(f"{spec.label} enrichment produced {count} new triples")
        except Exception as e:
            logger.error(f"{spec.label} enrichment failed: {e}", exc_info=True)
            stats[spec.name] = {'total_triples_added': 0, 'error': str(e)}

    # ----------------------------------------
    # Combine PySpark outputs (stays on executors)
    # ----------------------------------------
    spark_new_triples = None
    if spark_new_dfs:
        spark_new_triples = spark_new_dfs[0]
        for df in spark_new_dfs[1:]:
            spark_new_triples = spark_new_triples.unionByName(df)

    return {
        'stats': stats,
        'spark_new_triples': spark_new_triples
    }
