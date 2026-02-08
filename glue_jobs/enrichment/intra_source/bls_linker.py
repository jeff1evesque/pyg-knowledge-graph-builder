"""
BLS Intra-Source Enrichment Orchestrator
Coordinates enrichment across all BLS datasets
"""
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, OWL
from glue_jobs.utils.rdf_utils import (
    CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS,
    BLS_ENRICHMENT, UNIFIED, get_month_name, get_year_value
)
from glue_jobs.enrichment.intra_source.base import IntraSourceEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.cpi_enricher import CPIEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.ppi_enricher import PPIEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.eci_enricher import ECIEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.jolts_enricher import JOLTSEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.empsit_enricher import EMPSITEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.ximpim_enricher import XIMPIMEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.laus_enricher import LAUSEnricher
from glue_jobs.enrichment.intra_source.bls.enrichers.metro_enricher import METROEnricher
from glue_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from glue_jobs.enrichment.intra_source.bls.correlations import KNOWN_CORRELATIONS
from typing import Dict, Set, Optional, List
import logging

logger = logging.getLogger(__name__)


def normalize_keyword_for_uri_matching(keyword: str) -> str:
    """
    Normalize a keyword for matching against BLS URIs

    BLS URIs use underscores and remove special characters:
    - Spaces become underscores
    - Apostrophes are removed
    - Commas are removed
    - Hyphens may become underscores

    Examples:
        "Food at home" → "Food_at_home"
        "Owners' equivalent rent" → "Owners_equivalent_rent"
        "Men's and boys' apparel" → "Mens_and_boys_apparel"
        "All items less food, shelter, and energy" → "All_items_less_food_shelter_and_energy"

    Args:
        keyword: Human-readable keyword from BLS_SECTOR_PATTERNS

    Returns:
        Normalized keyword for URI substring matching
    """
    return (keyword
            .replace(' ', '_')
            .replace("'", '')
            .replace(',', '')
            .replace('-', '_')
            .replace('(', '')
            .replace(')', ''))


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
            'laus': LAUS,
            'metro': METRO,
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
        if 'laus' in self.available_datasets:
            enrichers['laus'] = LAUSEnricher(self.graph)
        if 'metro' in self.available_datasets:
            enrichers['metro'] = METROEnricher(self.graph)

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

    def _get_namespace(self, dataset_name: str):
        """Get namespace for a dataset"""
        namespace_map = {
            'cpi': CPI,
            'ppi': PPI,
            'eci': ECI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'ximpim': XIMPIM,
            'laus': LAUS,
        }
        return namespace_map.get(dataset_name)

    def unify_temporal_entities(self):
        """
        Unify temporal entities across all BLS datasets

        Creates unified Month and Year entities and links dataset-specific
        temporal entities to them using owl:sameAs

        Example:
            cpi:November, ppi:November, jolts:November, laus:November → unified:November
            cpi:2024, ppi:2024, jolts:2024, laus:2024 → unified:Year2024
        """
        # Get all unique months and years across all datasets
        months_by_name = {}
        years_by_value = {}

        # Collect months from all datasets
        for dataset_name in self.available_datasets:
            namespace = self._get_namespace(dataset_name)
            if namespace:
                # Query for months
                month_query = f"""
                SELECT DISTINCT ?month WHERE {{
                    ?s ?p ?month .
                    FILTER(STRSTARTS(STR(?month), "{namespace}"))
                    FILTER(REGEX(STR(?month), "(January|February|March|April|May|June|July|August|September|October|November|December)$"))
                }}
                """
                for row in self.graph.query(month_query):
                    month_name = get_month_name(row.month)
                    if month_name not in months_by_name:
                        months_by_name[month_name] = []
                    months_by_name[month_name].append(row.month)

                # Query for years
                year_query = f"""
                SELECT DISTINCT ?year WHERE {{
                    ?s ?p ?year .
                    FILTER(STRSTARTS(STR(?year), "{namespace}"))
                    FILTER(REGEX(STR(?year), "[0-9]{{4}}$"))
                }}
                """
                for row in self.graph.query(year_query):
                    year_value = get_year_value(row.year)
                    if year_value not in years_by_value:
                        years_by_value[year_value] = []
                    years_by_value[year_value].append(row.year)

        # Create unified temporal entities and link with owl:sameAs
        for month_name, month_uris in months_by_name.items():
            unified_month = UNIFIED[month_name]
            self.graph.add((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))
            self.graph.add((unified_month, RDFS.label, Literal(month_name)))

            for month_uri in month_uris:
                self.graph.add((unified_month, OWL.sameAs, month_uri))
                self.stats['temporal_unified'] += 1

        for year_value, year_uris in years_by_value.items():
            unified_year = UNIFIED[f"Year{year_value}"]
            self.graph.add((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))
            self.graph.add((unified_year, RDFS.label, Literal(year_value)))

            for year_uri in year_uris:
                self.graph.add((unified_year, OWL.sameAs, year_uri))
                self.stats['temporal_unified'] += 1

        logger.info(f"  Unified {len(months_by_name)} months and {len(years_by_value)} years")
        logger.info(f"  Created {self.stats['temporal_unified']} temporal unification links")

    def link_temporal_sequences(self):
        """
        Delegate temporal sequence linking to dataset enrichers

        Each dataset enricher handles its own temporal sequence linking
        based on its specific measurement types and temporal properties.

        Example:
            CPI Index for Food in Nov 2024 → precedes → CPI Index for Food in Dec 2024
        """
        for dataset_name, enricher in self.enrichers.items():
            logger.info(f"  Processing {dataset_name}...")
            links = enricher.link_temporal_sequences()
            self.stats['temporal_sequences'] += links

    def apply_sector_patterns(self):
        """
        Apply sector-based linking patterns

        Links category entities to economic sectors based on keyword matching.
        Uses normalized keyword matching to handle URI format (underscores, no special chars).

        Example:
            cpi:All_items_Food_Food_at_home_Entity → belongsToSector → bls:FoodSector
            cpi:All_items_Energy_Entity → belongsToSector → bls:EnergySector
        """
        logger.info("  Applying sector patterns...")

        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            sector_uri = pattern['sector_uri']
            relationship = pattern['relationship']

            # Process each dataset that has keywords for this sector
            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                if not keywords:
                    continue

                namespace = self._get_namespace(dataset_name)
                if not namespace:
                    continue

                # Find category entities for this dataset
                # Categories typically end with "_Entity"
                category_query = f"""
                SELECT DISTINCT ?category WHERE {{
                    ?measurement ?prop ?category .
                    FILTER(STRSTARTS(STR(?category), "{namespace}"))
                    FILTER(REGEX(STR(?category), "_Entity$"))
                }}
                """

                categories = list(self.graph.query(category_query))

                # Match categories against keywords
                for row in categories:
                    category_uri = row.category
                    category_str = str(category_uri)

                    # Check if any keyword matches this category
                    matched = False
                    for keyword in keywords:
                        normalized_keyword = normalize_keyword_for_uri_matching(keyword)

                        if normalized_keyword in category_str:
                            # Link category to sector
                            self.graph.add((
                                category_uri,
                                BLS_ENRICHMENT.belongsToSector,
                                sector_uri
                            ))

                            # Add sector correlation relationship
                            self.graph.add((
                                category_uri,
                                relationship,
                                sector_uri
                            ))

                            self.stats['sector_links'] += 2
                            matched = True
                            break  # Only link once per category

        logger.info(f"  Added {self.stats['sector_links']} sector links")

    def apply_known_correlations(self):
        """
        Apply known correlations from domain expertise

        Links entities across datasets based on economic relationships.
        Uses pattern matching on category URIs.

        Example:
            ppi:Food_Manufacturing_Entity → producerConsumerLink → cpi:Food_Entity
            eci:Wages_and_salaries_Entity → wageInflationLink → cpi:All_items_Entity
        """
        logger.info("  Applying known correlations...")

        for correlation in KNOWN_CORRELATIONS:
            source_dataset = correlation['source_dataset']
            target_dataset = correlation['target_dataset']
            source_pattern = correlation['source_pattern']
            target_pattern = correlation['target_pattern']
            relationship = correlation['relationship']

            # Skip if datasets not available
            if source_dataset not in self.available_datasets or target_dataset not in self.available_datasets:
                continue

            # Normalize patterns for URI matching
            source_pattern_normalized = normalize_keyword_for_uri_matching(source_pattern)
            target_pattern_normalized = normalize_keyword_for_uri_matching(target_pattern)

            # Get namespaces
            source_namespace = self._get_namespace(source_dataset)
            target_namespace = self._get_namespace(target_dataset)

            if not source_namespace or not target_namespace:
                continue

            # Find source categories matching pattern
            source_query = f"""
            SELECT DISTINCT ?category WHERE {{
                ?measurement ?prop ?category .
                FILTER(STRSTARTS(STR(?category), "{source_namespace}"))
                FILTER(REGEX(STR(?category), "_Entity$"))
                FILTER(CONTAINS(STR(?category), "{source_pattern_normalized}"))
            }}
            """

            # Find target categories matching pattern
            target_query = f"""
            SELECT DISTINCT ?category WHERE {{
                ?measurement ?prop ?category .
                FILTER(STRSTARTS(STR(?category), "{target_namespace}"))
                FILTER(REGEX(STR(?category), "_Entity$"))
                FILTER(CONTAINS(STR(?category), "{target_pattern_normalized}"))
            }}
            """

            source_categories = list(self.graph.query(source_query))
            target_categories = list(self.graph.query(target_query))

            # Create correlation links
            links_added = 0
            for source_row in source_categories:
                for target_row in target_categories:
                    self.graph.add((
                        source_row.category,
                        relationship,
                        target_row.category
                    ))
                    links_added += 1
                    self.stats['correlation_links'] += 1

            if links_added > 0:
                logger.info(f"    {correlation['name']}: {links_added} links")

        logger.info(f"  Total correlation links added: {self.stats['correlation_links']}")

    def link_cross_dataset_hierarchies(self):
        """
        Link hierarchies across datasets

        Identifies parent-child relationships within category URIs based on
        hierarchical path structure (e.g., All_items_Food_Food_at_home).

        Example:
            cpi:All_items_Food_Food_at_home_Entity → hasParent → cpi:All_items_Food_Entity
            cpi:All_items_Food_Entity → hasParent → cpi:All_items_Entity
        """
        logger.info("  Linking cross-dataset hierarchies...")

        for dataset_name in self.available_datasets:
            namespace = self._get_namespace(dataset_name)
            if not namespace:
                continue

            # Get all category entities
            category_query = f"""
            SELECT DISTINCT ?category WHERE {{
                ?measurement ?prop ?category .
                FILTER(STRSTARTS(STR(?category), "{namespace}"))
                FILTER(REGEX(STR(?category), "_Entity$"))
            }}
            """

            categories = list(self.graph.query(category_query))

            # Build hierarchy based on URI path structure
            # e.g., All_items_Food_Food_at_home_Entity has parent All_items_Food_Entity
            dataset_links = 0
            for row in categories:
                category_uri = row.category
                category_str = str(category_uri)

                # Extract the path part (remove namespace and _Entity suffix)
                if '_Entity' in category_str:
                    # Get the local name (everything after the last /)
                    local_name = category_str.split('/')[-1]
                    path = local_name.replace('_Entity', '')
                    path_parts = path.split('_')

                    # If path has multiple parts, try to find parent
                    if len(path_parts) > 1:
                        # Try progressively shorter paths to find parent
                        for i in range(len(path_parts) - 1, 0, -1):
                            parent_path = '_'.join(path_parts[:i])
                            parent_uri = URIRef(f"{namespace}{parent_path}_Entity")

                            # Check if parent exists in graph
                            parent_exists = False
                            for _ in self.graph.triples((parent_uri, None, None)):
                                parent_exists = True
                                break

                            if parent_exists:
                                self.graph.add((
                                    category_uri,
                                    BLS_ENRICHMENT.hasParent,
                                    parent_uri
                                ))
                                self.stats['hierarchy_links'] += 1
                                dataset_links += 1
                                break  # Only link to immediate parent

            if dataset_links > 0:
                logger.info(f"    {dataset_name}: {dataset_links} hierarchy links")

        logger.info(f"  Total hierarchy links added: {self.stats['hierarchy_links']}")