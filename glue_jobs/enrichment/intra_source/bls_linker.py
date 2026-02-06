"""
BLS Intra-Source Enrichment Orchestrator
Coordinates enrichment across all BLS datasets
"""
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from glue_jobs.utils.rdf_utils import (
    CPI, PPI, JOLTS, EMPSIT, ECI,
    BLS_ENRICHMENT, get_month_name, get_year_value
)
from glue_jobs.enrichment.intra_source.base import IntraSourceEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.cpi_enricher import CPIEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.ppi_enricher import PPIEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.eci_enricher import ECIEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.jolts_enricher import JOLTSEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.empsit_enricher import EMPSITEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.ximpim_enricher import XIMPIMEnricher
from typing import Dict, Set, Optional, List
import logging

logger = logging.getLogger(__name__)


class BLSIntraSourceLinker(IntraSourceEnricher):
    """
    Orchestrates BLS intra-source enrichment across all datasets
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.available_datasets = self.detect_datasets()
        self.enrichers = self._initialize_enrichers()
        logger.info(f"Detected datasets: {', '.join(self.available_datasets)}")

    def detect_datasets(self) -> Set[str]:
        """Detect which BLS datasets are present in the graph"""
        datasets = set()
        dataset_checks = {
            'cpi': CPI,
            'ppi': PPI,
            'eci': ECI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'ximpim': XIMPIM,
        }

        for dataset_name, namespace in dataset_checks.items():
            query = f"""
            ASK {{
                ?s ?p ?o .
                FILTER(STRSTARTS(STR(?s), "{namespace}"))
            }}
            """
            if self.graph.query(query).askAnswer:
                datasets.add(dataset_name)

        return datasets

    def _initialize_enrichers(self) -> Dict:
        """Initialize dataset-specific enrichers"""
        enrichers = {}

        if 'cpi' in self.available_datasets:
            enrichers['cpi'] = CPIEnricher(self.graph)
        if 'ppi' in self.available_datasets:
            enrichers['ppi'] = PPIEnricher(self.graph)
        if 'eci' in self.available_datasets:
            enrichers['eci'] = ECIEnricher(self.graph)
        if 'jolts' in self.available_datasets:
            enrichers['jolts'] = JOLTSEnricher(self.graph)
        if 'empsit' in self.available_datasets:
            enrichers['empsit'] = EMPSITEnricher(self.graph)
        if 'ximpim' in self.available_datasets:
            enrichers['ximpim'] = XIMPIMEnricher(self.graph)

        return enrichers

    def enrich(self) -> Dict[str, int]:
        """Run all BLS intra-source enrichment steps"""
        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting BLS Intra-Source Enrichment")
        logger.info("=" * 60)

        # Step 1: Unify temporal entities
        logger.info("\n[Step 1/5] Unifying temporal entities...")
        self.unify_temporal_entities()

        # Step 2: Link temporal sequences (delegated to enrichers)
        logger.info("\n[Step 2/5] Linking temporal sequences...")
        self.link_temporal_sequences()

        # Step 3: Apply sector patterns
        logger.info("\n[Step 3/5] Applying sector patterns...")
        self.apply_sector_patterns()

        # Step 4: Apply correlations
        logger.info("\n[Step 4/5] Applying correlations...")
        self.apply_known_correlations()

        # Step 5: Link hierarchies
        logger.info("\n[Step 5/5] Linking hierarchies...")
        self.link_cross_dataset_hierarchies()

        final_count = len(self.graph)
        enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("BLS Intra-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {enrichment_count}")
        logger.info(f"  - Temporal unification: {self.stats['temporal_unified']}")
        logger.info(f"  - Temporal sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Correlation links: {self.stats['correlation_links']}")
        logger.info(f"  - Hierarchy links: {self.stats['hierarchy_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': enrichment_count,
            'available_datasets': list(self.available_datasets),
            **self.stats
        }

    def unify_temporal_entities(self):
        """Unify temporal entities across all BLS datasets"""
        # (Keep existing implementation from original file)
        pass

    def link_temporal_sequences(self):
        """Delegate temporal sequence linking to dataset enrichers"""
        for dataset_name, enricher in self.enrichers.items():
            logger.info(f"  Processing {dataset_name}...")
            links = enricher.link_temporal_sequences()
            self.stats['temporal_sequences'] += links

    def apply_sector_patterns(self):
        """Apply sector-based linking patterns"""
        # (Keep existing implementation from original file)
        pass

    def apply_known_correlations(self):
        """Apply known correlations from domain expertise"""
        # (Keep existing implementation from original file)
        pass

    def link_cross_dataset_hierarchies(self):
        """Link hierarchies across datasets"""
        # (Keep existing implementation from original file)
        pass