"""
Intra-Source Enrichment Entry Point

Routes enrichment to the appropriate enricher per source family.
Returns both rdflib stats and PySpark new triples DataFrame.

The bridge to rdflib is NOT done here — it's handled by pipeline.py
via the snapshot/diff mechanism. This keeps the flow clean:
- rdflib enrichers mutate the graph (driver)
- PySpark enrichers return DataFrames (executors)
- pipeline.py merges both into triples_df
"""
from rdflib import Graph
from pyspark.sql import SparkSession, DataFrame
from glue_jobs.enrichment.intra_source.bls_linker import BLSIntraSourceLinker
from glue_jobs.enrichment.intra_source.sec_linker import SECIntraSourceLinker
from glue_jobs.enrichment.intra_source.market_linker import MarketIntraSourceLinker
from glue_jobs.enrichment.intra_source.noaa_linker import NOAAIntraSourceLinker
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def enrich_intra_source(
    graph: Graph,
    spark: Optional[SparkSession] = None,
    triples_df: Optional[DataFrame] = None
) -> Dict:
    """
    Run intra-source enrichment for all detected data sources.

    Returns:
        Dict with:
        - 'stats': per-source enrichment statistics
        - 'spark_new_triples': DataFrame of PySpark enrichment output
          (stays on executors, never collected here)
    """
    stats: Dict[str, Dict[str, any]] = {}
    spark_new_dfs: List[DataFrame] = []

    # ----------------------------------------
    # BLS Enrichment (rdflib — mutates graph)
    # ----------------------------------------
    logger.info("Checking for BLS data...")
    try:
        bls_linker = BLSIntraSourceLinker(graph)
        if bls_linker.available_datasets:
            stats['bls'] = bls_linker.enrich()
        else:
            logger.info("No BLS data detected")
    except Exception as e:
        logger.error(f"BLS enrichment failed: {e}", exc_info=True)
        stats['bls'] = {'total_triples_added': 0, 'error': str(e)}

    # ----------------------------------------
    # SEC Enrichment (rdflib — mutates graph)
    # ----------------------------------------
    logger.info("Checking for SEC data...")
    try:
        sec_linker = SECIntraSourceLinker(graph)
        if sec_linker.available_datasets:
            stats['sec'] = sec_linker.enrich()
        else:
            logger.info("No SEC data detected")
    except Exception as e:
        logger.error(f"SEC enrichment failed: {e}", exc_info=True)
        stats['sec'] = {'total_triples_added': 0, 'error': str(e)}

    # ----------------------------------------
    # Market Enrichment (PySpark — returns DataFrame)
    # ----------------------------------------
    logger.info("Checking for Market data...")
    if spark is not None and triples_df is not None:
        try:
            market_linker = MarketIntraSourceLinker(spark)
            market_new_df = market_linker.enrich(triples_df)
            spark_new_dfs.append(market_new_df)
            market_count = market_new_df.count()
            stats['market'] = {'total_triples_added': market_count}
        except Exception as e:
            logger.error(f"Market enrichment failed: {e}", exc_info=True)
            stats['market'] = {'total_triples_added': 0, 'error': str(e)}
    else:
        logger.info("Spark not available, skipping Market enrichment")

    # ----------------------------------------
    # NOAA Enrichment (PySpark — returns DataFrame)
    # ----------------------------------------
    logger.info("Checking for NOAA data...")
    if spark is not None and triples_df is not None:
        try:
            noaa_linker = NOAAIntraSourceLinker(spark)
            noaa_new_df = noaa_linker.enrich(triples_df)
            spark_new_dfs.append(noaa_new_df)
            noaa_count = noaa_new_df.count()
            stats['noaa'] = {'total_triples_added': noaa_count}
        except Exception as e:
            logger.error(f"NOAA enrichment failed: {e}", exc_info=True)
            stats['noaa'] = {'total_triples_added': 0, 'error': str(e)}
    else:
        logger.info("Spark not available, skipping NOAA enrichment")

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