"""
Intra-Source Enrichment Entry Point

Routes enrichment to the appropriate enricher per source family.
All enrichers now use PySpark — returns combined new triples DataFrame.

Migration status:
- BLS:    PySpark (returns DataFrame on executors)
- SEC:    PySpark (returns DataFrame on executors)
- Market: PySpark (returns DataFrame on executors)
- NOAA:   PySpark (returns DataFrame on executors)
"""
from pyspark.sql import SparkSession, DataFrame
from glue_jobs.enrichment.intra_source.bls_linker import BLSIntraSourceLinker
from glue_jobs.enrichment.intra_source.sec_linker import SECIntraSourceLinker
from glue_jobs.enrichment.intra_source.market_linker import MarketIntraSourceLinker
from glue_jobs.enrichment.intra_source.noaa_linker import NOAAIntraSourceLinker
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def enrich_intra_source(
    spark: SparkSession,
    triples_df: DataFrame,
) -> Dict:
    """
    Run intra-source enrichment for all detected data sources.

    Args:
        spark: SparkSession
        triples_df: DataFrame with (subject, predicate, object)

    Returns:
        Dict with:
        - 'stats': per-source enrichment statistics
        - 'spark_new_triples': DataFrame of all enrichment output
          (stays on executors, never collected here)
    """
    stats: Dict[str, Dict] = {}
    spark_new_dfs: List[DataFrame] = []

    # ----------------------------------------
    # BLS Enrichment (PySpark — returns DataFrame)
    # ----------------------------------------
    logger.info("Checking for BLS data...")
    try:
        bls_linker = BLSIntraSourceLinker(spark)
        bls_new_df = bls_linker.enrich(triples_df)
        if bls_new_df is not None:
            spark_new_dfs.append(bls_new_df)
            bls_count = bls_new_df.count()
            stats['bls'] = {'total_triples_added': bls_count}
            logger.info(f"BLS enrichment produced {bls_count} new triples")
        else:
            stats['bls'] = {'total_triples_added': 0}
            logger.info("No BLS data detected")
    except Exception as e:
        logger.error(f"BLS enrichment failed: {e}", exc_info=True)
        stats['bls'] = {'total_triples_added': 0, 'error': str(e)}

    # ----------------------------------------
    # SEC Enrichment (PySpark — returns DataFrame)
    # ----------------------------------------
    logger.info("Checking for SEC data...")
    try:
        sec_linker = SECIntraSourceLinker(spark)
        sec_new_df = sec_linker.enrich(triples_df)
        if sec_new_df is not None:
            spark_new_dfs.append(sec_new_df)
            sec_count = sec_new_df.count()
            stats['sec'] = {'total_triples_added': sec_count}
            logger.info(f"SEC enrichment produced {sec_count} new triples")
        else:
            stats['sec'] = {'total_triples_added': 0}
            logger.info("No SEC data detected")
    except Exception as e:
        logger.error(f"SEC enrichment failed: {e}", exc_info=True)
        stats['sec'] = {'total_triples_added': 0, 'error': str(e)}

    # ----------------------------------------
    # Market Enrichment (PySpark — returns DataFrame)
    # ----------------------------------------
    logger.info("Checking for Market data...")
    try:
        market_linker = MarketIntraSourceLinker(spark)
        market_new_df = market_linker.enrich(triples_df)
        if market_new_df is not None:
            spark_new_dfs.append(market_new_df)
            market_count = market_new_df.count()
            stats['market'] = {'total_triples_added': market_count}
            logger.info(f"Market enrichment produced {market_count} new triples")
        else:
            stats['market'] = {'total_triples_added': 0}
            logger.info("No Market data detected")
    except Exception as e:
        logger.error(f"Market enrichment failed: {e}", exc_info=True)
        stats['market'] = {'total_triples_added': 0, 'error': str(e)}

    # ----------------------------------------
    # NOAA Enrichment (PySpark — returns DataFrame)
    # ----------------------------------------
    logger.info("Checking for NOAA data...")
    try:
        noaa_linker = NOAAIntraSourceLinker(spark)
        noaa_new_df = noaa_linker.enrich(triples_df)
        if noaa_new_df is not None:
            spark_new_dfs.append(noaa_new_df)
            noaa_count = noaa_new_df.count()
            stats['noaa'] = {'total_triples_added': noaa_count}
            logger.info(f"NOAA enrichment produced {noaa_count} new triples")
        else:
            stats['noaa'] = {'total_triples_added': 0}
            logger.info("No NOAA data detected")
    except Exception as e:
        logger.error(f"NOAA enrichment failed: {e}", exc_info=True)
        stats['noaa'] = {'total_triples_added': 0, 'error': str(e)}

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
        'spark_new_triples': spark_new_triples,
    }