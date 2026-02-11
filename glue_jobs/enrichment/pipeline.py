"""
Main Enrichment Pipeline Orchestrator

Coordinates the complete RDF enrichment process:
1. Intra-source enrichment (within BLS, SEC, Market, NOAA)
2. Cross-source enrichment (across BLS ↔ SEC ↔ Market ↔ NOAA)
3. Optional: Ontology mapping for standardization

This is the main entry point called by build_graph.py
"""
from rdflib import Graph
from glue_jobs.enrichment.intra_source_linker import enrich_intra_source
from glue_jobs.enrichment.cross_source_linker import enrich_cross_source
from glue_jobs.enrichment.ontology_mapper import map_ontologies
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """
    Main enrichment pipeline orchestrator

    Execution flow:
    1. Intra-source enrichment (BLS, SEC, Market, NOAA independently)
    2. Cross-source enrichment (linking across sources)
    3. Optional: Ontology mapping (standardization)

    Example:
        pipeline = EnrichmentPipeline(graph)
        stats = pipeline.run(enable_ontology_mapping=True)
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.stats = {
            'initial_triples': len(graph),
            'intra_source': {},
            'cross_source': {},
            'ontology_mapping': {},
            'final_triples': 0,
            'total_enrichment': 0
        }

    def run(self, enable_ontology_mapping: bool = False) -> Dict[str, any]:
        """
        Run the complete enrichment pipeline

        Args:
            enable_ontology_mapping: Whether to run ontology mapping step

        Returns:
            Dictionary with enrichment statistics
        """
        logger.info("=" * 80)
        logger.info("STARTING ENRICHMENT PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Initial graph size: {self.stats['initial_triples']} triples")
        logger.info("")

        # Step 1: Intra-source enrichment
        logger.info("=" * 80)
        logger.info("PHASE 1: INTRA-SOURCE ENRICHMENT")
        logger.info("=" * 80)
        logger.info("Enriching within each data source family (BLS, SEC, Market, NOAA)")
        logger.info("")

        self.stats['intra_source'] = enrich_intra_source(self.graph)

        intra_source_triples = len(self.graph) - self.stats['initial_triples']
        logger.info("")
        logger.info(f"Intra-source enrichment added: {intra_source_triples} triples")
        logger.info("")

        # Step 2: Cross-source enrichment
        logger.info("=" * 80)
        logger.info("PHASE 2: CROSS-SOURCE ENRICHMENT")
        logger.info("=" * 80)
        logger.info("Linking entities across data source families")
        logger.info("")

        cross_source_start = len(self.graph)
        self.stats['cross_source'] = enrich_cross_source(self.graph)

        cross_source_triples = len(self.graph) - cross_source_start
        logger.info("")
        logger.info(f"Cross-source enrichment added: {cross_source_triples} triples")
        logger.info("")

        # Step 3: Optional ontology mapping
        if enable_ontology_mapping:
            logger.info("=" * 80)
            logger.info("PHASE 3: ONTOLOGY MAPPING")
            logger.info("=" * 80)
            logger.info("Standardizing vocabularies and creating equivalence mappings")
            logger.info("")

            ontology_start = len(self.graph)
            self.stats['ontology_mapping'] = self._run_ontology_mapping()

            ontology_triples = len(self.graph) - ontology_start
            logger.info("")
            logger.info(f"Ontology mapping added: {ontology_triples} triples")
            logger.info("")

        # Final statistics
        self.stats['final_triples'] = len(self.graph)
        self.stats['total_enrichment'] = self.stats['final_triples'] - self.stats['initial_triples']

        self._print_final_summary()

        return self.stats

    def _run_ontology_mapping(self) -> Dict[str, int]:
        """
        Run ontology mapping step

        Creates owl:equivalentProperty and owl:equivalentClass mappings
        to standardize vocabularies across sources
        """
        logger.info("Running ontology mapping...")

        stats = map_ontologies(
            self.graph,
            enable_skos=False,  # Set to True if publishing as Linked Open Data
            enable_dublin_core=False  # Set to True if you want metadata timestamps
        )

        logger.info(f"  Property equivalences: {stats['property_equivalences']}")
        logger.info(f"  Class equivalences: {stats['class_equivalences']}")
        logger.info(f"  Normalized labels: {stats['normalized_labels']}")

        return stats

    def _print_final_summary(self):
        """Print final enrichment summary"""
        logger.info("=" * 80)
        logger.info("ENRICHMENT PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("SUMMARY:")
        logger.info(f"  Initial triples:        {self.stats['initial_triples']:>10,}")
        logger.info(f"  Final triples:          {self.stats['final_triples']:>10,}")
        logger.info(f"  Total enrichment:       {self.stats['total_enrichment']:>10,}")
        logger.info("")

        # Intra-source breakdown
        logger.info("INTRA-SOURCE ENRICHMENT:")
        for source, source_stats in self.stats['intra_source'].items():
            if isinstance(source_stats, dict) and 'total_triples_added' in source_stats:
                logger.info(f"  {source.upper():>6}: {source_stats['total_triples_added']:>10,} triples")
        logger.info("")

        # Cross-source breakdown
        logger.info("CROSS-SOURCE ENRICHMENT:")
        if isinstance(self.stats['cross_source'], dict):
            for key, value in self.stats['cross_source'].items():
                if key != 'available_sources' and isinstance(value, int):
                    logger.info(f"  {key.replace('_', ' ').title():>30}: {value:>10,}")
        logger.info("")

        # Ontology mapping breakdown (if enabled)
        if self.stats['ontology_mapping']:
            logger.info("ONTOLOGY MAPPING:")
            for key, value in self.stats['ontology_mapping'].items():
                logger.info(f"  {key.replace('_', ' ').title():>30}: {value:>10,}")
            logger.info("")

        logger.info("=" * 80)


def enrich_graph(graph: Graph, enable_ontology_mapping: bool = False) -> Dict[str, any]:
    """
    Main entry point for graph enrichment

    This is the function called by build_graph.py

    Args:
        graph: RDFLib graph to enrich
        enable_ontology_mapping: Whether to run ontology mapping

    Returns:
        Dictionary with enrichment statistics

    Example:
        from glue_jobs.enrichment.pipeline import enrich_graph

        stats = enrich_graph(graph, enable_ontology_mapping=True)
    """
    pipeline = EnrichmentPipeline(graph)
    return pipeline.run(enable_ontology_mapping=enable_ontology_mapping)
