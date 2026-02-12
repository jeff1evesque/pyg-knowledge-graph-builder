"""
Main Intra-Source Enrichment Entry Point
Routes to appropriate source-specific linkers
"""
from rdflib import Graph
from glue_jobs.enrichment.intra_source.bls_linker import BLSIntraSourceLinker
from glue_jobs.enrichment.intra_source.sec_linker import SECIntraSourceLinker
from glue_jobs.enrichment.intra_source.market_linker import MarketIntraSourceLinker
from glue_jobs.enrichment.intra_source.noaa_linker import NOAAIntraSourceLinker
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def enrich_intra_source(graph: Graph) -> Dict[str, int]:
    """
    Main entry point for intra-source enrichment

    Detects data sources and applies appropriate enrichment strategies.
    Each linker internally detects whether its data is present and
    returns early with total_triples_added=0 if not found.
    """
    stats = {
        'bls': {},
        'sec': {},
        'market': {},
        'noaa': {}
    }

    # BLS enrichment
    logger.info("Starting BLS intra-source enrichment...")
    bls_linker = BLSIntraSourceLinker(graph)
    stats['bls'] = bls_linker.enrich()

    # SEC enrichment
    logger.info("\nStarting SEC intra-source enrichment...")
    sec_linker = SECIntraSourceLinker(graph)
    stats['sec'] = sec_linker.enrich()

    # Market enrichment
    logger.info("\nStarting Market intra-source enrichment...")
    market_linker = MarketIntraSourceLinker(graph)
    stats['market'] = market_linker.enrich()

    # NOAA enrichment
    logger.info("\nStarting NOAA intra-source enrichment...")
    noaa_linker = NOAAIntraSourceLinker(graph)
    stats['noaa'] = noaa_linker.enrich()

    return stats