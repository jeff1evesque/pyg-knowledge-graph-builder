"""
BLS Intra-Source Enrichment
Links entities within the BLS family (CPI, PPI, ECI, JOLTS, etc.)

This module encodes domain knowledge about BLS data relationships.
Patterns are only defined for datasets whose ontologies have been analyzed.

Currently supported datasets: CPI, PPI
TODO: Add JOLTS, EMPSIT, ECI, LAUS, METRO, REALER, WKYENG, XIMPIM as ontologies are provided
"""

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
from glue_jobs.utils.rdf_utils import (
    CPI, PPI, JOLTS, EMPSIT, ECI, LAUS, METRO, REALER, WKYENG, XIMPIM,
    BLS_ENRICHMENT, UNIFIED,
    get_month_name, get_year_value, get_namespace_from_uri
)
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================
# DOMAIN KNOWLEDGE: BLS LINKING PATTERNS
# ============================================
# IMPORTANT: Only include datasets whose ontologies have been analyzed!
# Currently includes: CPI, PPI
# TODO: Add JOLTS, EMPSIT, etc. as their ontologies are provided

BLS_SECTOR_PATTERNS = {
    'food_sector': {
        'description': 'Food production, distribution, and consumption chain',
        'sector_uri': BLS_ENRICHMENT.FoodSector,
        'keywords': {
            # CPI: Consumer food prices (from CPI ontology analysis)
            # Categories: Food, Food at home, Food away from home, etc.
            'cpi': ['Food', 'Grocery', 'Restaurant', 'Dining', 'Meals', 'Beverages'],

            # PPI: Producer food costs (from PPI ontology analysis)
            # Categories: Food manufacturing, Foodstuffs, Finished consumer foods, etc.
            'ppi': ['Food', 'Manufacturing', 'Foodstuffs', 'Feeds', 'Finished Consumer Foods'],

            # JOLTS: TODO - Add when JOLTS ontology is provided
            # Expected: Food services, restaurants, etc.
            # 'jolts': [],

            # EMPSIT: TODO - Add when EMPSIT ontology is provided
            # Expected: Food services employment, leisure and hospitality, etc.
            # 'empsit': [],
        },
        'relationship': BLS_ENRICHMENT.foodSectorCorrelation
    },
    'energy_sector': {
        'description': 'Energy production, distribution, and consumption',
        'sector_uri': BLS_ENRICHMENT.EnergySector,
        'keywords': {
            # CPI: Consumer energy prices
            # Categories: Energy, Gasoline, Fuel oil, Electricity, Natural gas, etc.
            'cpi': ['Energy', 'Gasoline', 'Fuel', 'Electricity', 'Gas', 'Utility'],

            # PPI: Producer energy costs
            # Categories: Energy goods, Petroleum, etc.
            'ppi': ['Energy', 'Goods', 'Materials', 'Fuel', 'Petroleum'],

            # JOLTS: TODO
            # Expected: Mining, oil and gas extraction, utilities, etc.
            # 'jolts': [],

            # EMPSIT: TODO
            # Expected: Mining, utilities employment, etc.
            # 'empsit': [],
        },
        'relationship': BLS_ENRICHMENT.energySectorCorrelation
    },
    'housing_sector': {
        'description': 'Housing, construction, and shelter',
        'sector_uri': BLS_ENRICHMENT.HousingSector,
        'keywords': {
            # CPI: Consumer housing costs
            # Categories: Housing, Shelter, Rent, Owners' equivalent rent, etc.
            'cpi': ['Housing', 'Shelter', 'Rent', 'Owners', 'Lodging'],

            # PPI: Construction and building materials
            # Categories: Construction, Building materials, Residential construction, etc.
            'ppi': ['Construction', 'Building', 'Materials', 'Residential'],

            # JOLTS: TODO
            # Expected: Construction industry, etc.
            # 'jolts': [],

            # EMPSIT: TODO
            # Expected: Construction employment, etc.
            # 'empsit': [],
        },
        'relationship': BLS_ENRICHMENT.housingSectorCorrelation
    },
    'transportation_sector': {
        'description': 'Transportation and warehousing',
        'sector_uri': BLS_ENRICHMENT.TransportationSector,
        'keywords': {
            # CPI: Consumer transportation costs
            # Categories: Transportation, Motor vehicles, Gasoline, Public transportation, etc.
            'cpi': ['Transportation', 'Vehicle', 'Automobile', 'Transit', 'Airfare'],

            # PPI: Transportation services and trade
            # Categories: Transportation and warehousing, Trade services, etc.
            'ppi': ['Transportation', 'Warehousing', 'Trade', 'Shipping'],

            # JOLTS: TODO
            # Expected: Transportation and warehousing industry, etc.
            # 'jolts': [],

            # EMPSIT: TODO
            # Expected: Transportation employment, etc.
            # 'empsit': [],
        },
        'relationship': BLS_ENRICHMENT.transportationSectorCorrelation
    },
    'goods_services': {
        'description': 'Goods vs Services distinction across datasets',
        'sector_uri': BLS_ENRICHMENT.GoodsServicesSector,
        'keywords': {
            # CPI: Goods and services categories
            # Categories: Commodities, Services, Durables, Nondurables, etc.
            'cpi': ['Goods', 'Services', 'Commodities'],

            # PPI: Final/Intermediate demand split
            # Categories: Final demand goods, Final demand services, Processed goods, etc.
            'ppi': ['Goods', 'Services', 'Final Demand', 'Intermediate Demand',
                    'Processed', 'Unprocessed'],

            # JOLTS: TODO
            # Expected: Goods-producing, Service-providing industries, etc.
            # 'jolts': [],

            # EMPSIT: TODO
            # Expected: Goods-producing, Service-providing employment, etc.
            # 'empsit': [],
        },
        'relationship': BLS_ENRICHMENT.goodsServicesRelation
    }
}

# Specific known correlations based on economic theory
# IMPORTANT: Only include correlations between datasets we've analyzed!
KNOWN_CORRELATIONS = [
    {
        'name': 'cpi_ppi_food_chain',
        'source_dataset': 'ppi',
        'target_dataset': 'cpi',
        'source_pattern': 'Food Manufacturing',
        'target_pattern': 'Food',
        'description': 'Producer food costs lead to consumer food prices',
        'relationship': BLS_ENRICHMENT.producerConsumerLink,
        'lag_months': 1,
        'strength': 'strong'
    },
    {
        'name': 'cpi_ppi_energy_chain',
        'source_dataset': 'ppi',
        'target_dataset': 'cpi',
        'source_pattern': 'Energy Goods',
        'target_pattern': 'Energy',
        'description': 'Producer energy costs lead to consumer energy prices',
        'relationship': BLS_ENRICHMENT.producerConsumerLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'ppi_supply_chain',
        'source_dataset': 'ppi',
        'target_dataset': 'ppi',
        'source_pattern': 'Intermediate Demand',
        'target_pattern': 'Final Demand',
        'description': 'Intermediate goods costs flow to final demand prices',
        'relationship': BLS_ENRICHMENT.supplyChainLink,
        'direction': 'upstream',
        'strength': 'medium'
    },
    {
        'name': 'ppi_processed_unprocessed',
        'source_dataset': 'ppi',
        'target_dataset': 'ppi',
        'source_pattern': 'Unprocessed',
        'target_pattern': 'Processed',
        'description': 'Unprocessed goods costs affect processed goods prices',
        'relationship': BLS_ENRICHMENT.processingChainLink,
        'direction': 'upstream',
        'strength': 'medium'
    },
    # TODO: Add CPI-JOLTS correlations when JOLTS ontology is provided
    # Example: Employment in food services (JOLTS) affects food away from home prices (CPI)

    # TODO: Add PPI-JOLTS correlations when JOLTS ontology is provided
    # Example: Job openings in manufacturing (JOLTS) correlate with production costs (PPI)

    # TODO: Add JOLTS-EMPSIT correlations when both ontologies are provided
    # Example: Job openings (JOLTS) lead employment changes (EMPSIT)
]

# Measurement type mappings for temporal linking
# IMPORTANT: Only include datasets we've analyzed!
MEASUREMENT_TYPES = {
    'cpi': {
        'Index': {
            'class': CPI.Index,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },
        'PercentChange': {
            'class': CPI.PercentChange,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },
        'RelativeImportance': {
            'class': CPI.RelativeImportance,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        }
    },
    'ppi': {
        'MonthlyChange': {
            'class': PPI.MonthlyChange,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasEndMonth,
            'year_property': PPI.hasEndYear
        },
        'TwelveMonthChange': {
            'class': PPI.TwelveMonthChange,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasEndMonth,
            'year_property': PPI.hasEndYear
        },
        'IndexValue': {
            'class': PPI.IndexValue,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        },
        'RelativeImportance': {
            'class': PPI.RelativeImportance,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        }
    },
    # TODO: Add JOLTS measurement types when ontology is provided
    # Expected measurement types: JobOpenings, Hires, Separations, Quits, Layoffs, etc.
    # 'jolts': {
    #     'JobOpenings': {
    #         'class': JOLTS.JobOpenings,
    #         'category_property': JOLTS.hasIndustry,
    #         'month_property': JOLTS.hasMonth,
    #         'year_property': JOLTS.hasYear
    #     },
    #     'Hires': {...},
    #     'Separations': {...},
    #     etc.
    # },

    # TODO: Add EMPSIT measurement types when ontology is provided
    # Expected measurement types: Employment, Unemployment, LaborForce, etc.
    # 'empsit': {
    #     'Employment': {
    #         'class': EMPSIT.Employment,
    #         'category_property': EMPSIT.hasIndustry,
    #         'month_property': EMPSIT.hasMonth,
    #         'year_property': EMPSIT.hasYear
    #     },
    #     'Unemployment': {...},
    #     etc.
    # }
}


class BLSIntraSourceLinker:
    """
    Enriches BLS data with intra-source relationships across all BLS datasets.

    Uses encoded domain knowledge to create links. Patterns are only defined
    for datasets whose ontologies have been analyzed (currently: CPI, PPI).

    When new datasets are added (e.g., JOLTS), update:
    1. BLS_SECTOR_PATTERNS - Add keywords for the new dataset
    2. KNOWN_CORRELATIONS - Add correlations involving the new dataset
    3. MEASUREMENT_TYPES - Add measurement type mappings

    The linker automatically detects which datasets are present in the graph
    and only applies patterns for available datasets.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.enrichment_count = 0
        self.stats = {
            'temporal_unified': 0,
            'temporal_sequences': 0,
            'sector_links': 0,
            'correlation_links': 0,
            'hierarchy_links': 0
        }

        # Month ordering for temporal sequences
        self.month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        # Detect which datasets are present in the graph
        self.available_datasets = self._detect_available_datasets()
        logger.info(f"Detected datasets in graph: {', '.join(self.available_datasets)}")

    def _detect_available_datasets(self) -> Set[str]:
        """
        Detect which BLS datasets are present in the graph.

        Returns:
            Set of dataset names (e.g., {'cpi', 'ppi'})
        """
        datasets = set()

        # Check for each dataset by looking for its namespace
        dataset_checks = {
            'cpi': CPI,
            'ppi': PPI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'eci': ECI,
            'laus': LAUS,
            'metro': METRO,
            'realer': REALER,
            'wkyeng': WKYENG,
            'ximpim': XIMPIM
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

    def enrich(self) -> Dict[str, int]:
        """
        Run all BLS intra-source enrichment steps.

        Only applies enrichment patterns for datasets that are present in the graph
        and have been configured in the pattern dictionaries.

        Returns:
            Dict with enrichment statistics
        """
        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting BLS Intra-Source Enrichment")
        logger.info(f"Available datasets: {', '.join(sorted(self.available_datasets))}")
        logger.info("=" * 60)

        # Step 1: Unify temporal entities across all BLS datasets
        logger.info("\n[Step 1/5] Unifying temporal entities...")
        self.unify_temporal_entities()

        # Step 2: Link temporal sequences
        logger.info("\n[Step 2/5] Linking temporal sequences...")
        self.link_temporal_sequences()

        # Step 3: Apply sector-based linking patterns
        logger.info("\n[Step 3/5] Applying sector-based patterns...")
        self.apply_sector_patterns()

        # Step 4: Apply known correlations
        logger.info("\n[Step 4/5] Applying known correlations...")
        self.apply_known_correlations()

        # Step 5: Link hierarchies across datasets
        logger.info("\n[Step 5/5] Linking cross-dataset hierarchies...")
        self.link_cross_dataset_hierarchies()

        final_count = len(self.graph)
        self.enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("BLS Intra-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {self.enrichment_count}")
        logger.info(f"  - Temporal unification: {self.stats['temporal_unified']}")
        logger.info(f"  - Temporal sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Correlation links: {self.stats['correlation_links']}")
        logger.info(f"  - Hierarchy links: {self.stats['hierarchy_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': self.enrichment_count,
            'available_datasets': list(self.available_datasets),
            **self.stats
        }

    def unify_temporal_entities(self):
        """
        Unify temporal entities across all BLS datasets.

        Creates unified Month and Year entities that link to dataset-specific temporal entities.
        Example: cpi:January, ppi:January → bls:January

        This works for any BLS dataset that uses Month and Year classes.
        """
        # Find all Month entities across all BLS datasets
        month_query = """
        SELECT DISTINCT ?month ?monthClass WHERE {
            ?month a ?monthClass .
            FILTER(
                ?monthClass = <https://www.bls.gov/cpi/Month> ||
                ?monthClass = <https://www.bls.gov/ppi/Month> ||
                ?monthClass = <https://www.bls.gov/jolts/Month> ||
                ?monthClass = <https://www.bls.gov/empsit/Month> ||
                ?monthClass = <https://www.bls.gov/eci/Month> ||
                ?monthClass = <https://www.bls.gov/laus/Month> ||
                ?monthClass = <https://www.bls.gov/metro/Month> ||
                ?monthClass = <https://www.bls.gov/realer/Month> ||
                ?monthClass = <https://www.bls.gov/wkyeng/Month> ||
                ?monthClass = <https://www.bls.gov/ximpim/Month>
            )
        }
        """

        # Find all Year entities
        year_query = """
        SELECT DISTINCT ?year ?yearClass WHERE {
            ?year a ?yearClass .
            FILTER(
                ?yearClass = <https://www.bls.gov/cpi/Year> ||
                ?yearClass = <https://www.bls.gov/ppi/Year> ||
                ?yearClass = <https://www.bls.gov/jolts/Year> ||
                ?yearClass = <https://www.bls.gov/empsit/Year> ||
                ?yearClass = <https://www.bls.gov/eci/Year> ||
                ?yearClass = <https://www.bls.gov/laus/Year> ||
                ?yearClass = <https://www.bls.gov/metro/Year> ||
                ?yearClass = <https://www.bls.gov/realer/Year> ||
                ?yearClass = <https://www.bls.gov/wkyeng/Year> ||
                ?yearClass = <https://www.bls.gov/ximpim/Year>
            )
        }
        """

        months = list(self.graph.query(month_query))
        years = list(self.graph.query(year_query))

        logger.info(f"Found {len(months)} month entities, {len(years)} year entities")

        # Group months by name
        month_groups = {}
        for row in months:
            month_name = get_month_name(row.month)
            if month_name not in month_groups:
                month_groups[month_name] = []
            month_groups[month_name].append(row.month)

        # Group years by value
        year_groups = {}
        for row in years:
            year_value = get_year_value(row.year)
            if year_value not in year_groups:
                year_groups[year_value] = []
            year_groups[year_value].append(row.year)

        # Create unified temporal entities
        for month_name, month_uris in month_groups.items():
            if len(month_uris) > 1:
                unified_month = BLS_ENRICHMENT[month_name]
                self.graph.add((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))
                self.graph.add((unified_month, RDFS.label, Literal(month_name, datatype=XSD.string)))

                for month_uri in month_uris:
                    self.graph.add((month_uri, BLS_ENRICHMENT.unifiedAs, unified_month))
                    self.stats['temporal_unified'] += 1

        for year_value, year_uris in year_groups.items():
            if len(year_uris) > 1:
                unified_year = BLS_ENRICHMENT[year_value]
                self.graph.add((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))
                self.graph.add((unified_year, RDFS.label, Literal(year_value, datatype=XSD.string)))

                for year_uri in year_uris:
                    self.graph.add((year_uri, BLS_ENRICHMENT.unifiedAs, unified_year))
                    self.stats['temporal_unified'] += 1

        logger.info(f"Created {len(month_groups)} unified months, {len(year_groups)} unified years")
        logger.info(f"Added {self.stats['temporal_unified']} unification links")

    def link_temporal_sequences(self):
        """
        Add temporal ordering for measurements across all datasets.

        Links consecutive time periods using bls:precedes relationship.
        Only processes measurement types that are configured in MEASUREMENT_TYPES.
        """
        # Link sequences for each measurement type in each dataset
        for dataset, measurement_types in MEASUREMENT_TYPES.items():
            if dataset not in self.available_datasets:
                logger.info(f"  Skipping {dataset} (not in graph)")
                continue

            for measurement_name, config in measurement_types.items():
                count = self._link_sequences_for_measurement_type(
                    measurement_type=config['class'],
                    category_property=config['category_property'],
                    month_property=config['month_property'],
                    year_property=config['year_property'],
                    dataset_name=dataset,
                    measurement_name=measurement_name
                )
                self.stats['temporal_sequences'] += count

    def _link_sequences_for_measurement_type(self,
                                             measurement_type: URIRef,
                                             category_property: URIRef,
                                             month_property: URIRef,
                                             year_property: URIRef,
                                             dataset_name: str,
                                             measurement_name: str) -> int:
        """Helper to link temporal sequences for a specific measurement type"""

        query = f"""
        SELECT ?measurement ?category ?month ?year WHERE {{
            ?measurement a <{measurement_type}> ;
                        <{category_property}> ?category ;
                        <{month_property}> ?month ;
                        <{year_property}> ?year .
        }}
        """

        results = list(self.graph.query(query))

        if not results:
            return 0

        # Group by category
        by_category = {}
        for row in results:
            category = row.category
            if category not in by_category:
                by_category[category] = []

            by_category[category].append({
                'measurement': row.measurement,
                'month': get_month_name(row.month),
                'year': get_year_value(row.year)
            })

        # For each category, sort by time and add precedes links
        links_added = 0
        for category, measurements in by_category.items():
            measurements.sort(key=lambda x: (
                int(x['year']),
                self.month_order.index(x['month'])
            ))

            for i in range(len(measurements) - 1):
                current = measurements[i]['measurement']
                next_measurement = measurements[i + 1]['measurement']

                self.graph.add((
                    current,
                    BLS_ENRICHMENT.precedes,
                    next_measurement
                ))
                links_added += 1

        logger.info(
            f"  {dataset_name}.{measurement_name}: {links_added} sequence links across {len(by_category)} categories")
        return links_added

    def apply_sector_patterns(self):
        """
        Apply sector-based linking patterns from configuration.

        Creates economic sector entities and links related entities across datasets.
        Only applies patterns for datasets that are:
        1. Present in the graph (self.available_datasets)
        2. Configured in BLS_SECTOR_PATTERNS
        """
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            logger.info(f"  Processing sector: {sector_name}")

            # Create sector entity
            sector_uri = pattern['sector_uri']
            self.graph.add((sector_uri, RDF.type, BLS_ENRICHMENT.EconomicSector))
            self.graph.add((sector_uri, RDFS.label, Literal(pattern['description'], datatype=XSD.string)))

            # Find entities in each dataset matching keywords
            dataset_entities = {}
            total_entities = 0

            for dataset, keywords in pattern['keywords'].items():
                # Only process if dataset is available AND configured
                if dataset not in self.available_datasets:
                    logger.info(f"    Skipping {dataset} (not in graph)")
                    continue

                if not keywords:
                    logger.info(f"    Skipping {dataset} (no keywords configured)")
                    continue

                namespace = self._get_namespace_for_dataset(dataset)
                if namespace:
                    entities = self._find_entities_by_keywords(namespace, keywords)
                    dataset_entities[dataset] = entities
                    total_entities += len(entities)

                    # Link entities to sector
                    for entity in entities:
                        self.graph.add((entity, BLS_ENRICHMENT.belongsToSector, sector_uri))
                        self.stats['sector_links'] += 1

            # Create cross-dataset correlations within sector
            datasets = list(dataset_entities.keys())
            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    dataset1 = datasets[i]
                    dataset2 = datasets[j]

                    for entity1 in dataset_entities[dataset1]:
                        for entity2 in dataset_entities[dataset2]:
                            self.graph.add((entity1, pattern['relationship'], entity2))
                            self.stats['sector_links'] += 1

            logger.info(f"    Found {total_entities} entities across {len(dataset_entities)} datasets")

    def apply_known_correlations(self):
        """
        Apply specific known correlations from domain expertise.

        Creates directed relationships based on economic theory.
        Only applies correlations where both source and target datasets are available.
        """
        for correlation in KNOWN_CORRELATIONS:
            source_dataset = correlation['source_dataset']
            target_dataset = correlation['target_dataset']

            # Check if both datasets are available
            if source_dataset not in self.available_datasets:
                logger.info(f"  Skipping {correlation['name']}: {source_dataset} not available")
                continue

            if target_dataset not in self.available_datasets:
                logger.info(f"  Skipping {correlation['name']}: {target_dataset} not available")
                continue

            logger.info(f"  Applying: {correlation['name']}")

            source_ns = self._get_namespace_for_dataset(source_dataset)
            target_ns = self._get_namespace_for_dataset(target_dataset)

            if not source_ns or not target_ns:
                continue

            source_entities = self._find_entities_by_pattern(source_ns, correlation['source_pattern'])
            target_entities = self._find_entities_by_pattern(target_ns, correlation['target_pattern'])

            links_added = 0
            for source_entity in source_entities:
                for target_entity in target_entities:
                    self.graph.add((source_entity, correlation['relationship'], target_entity))
                    links_added += 1

                    # Add metadata about the correlation
                    if 'lag_months' in correlation:
                        correlation_node = BLS_ENRICHMENT[f"Correlation_{correlation['name']}"]
                        self.graph.add((correlation_node, RDF.type, BLS_ENRICHMENT.EconomicCorrelation))
                        self.graph.add((correlation_node, BLS_ENRICHMENT.lagMonths,
                                        Literal(correlation['lag_months'], datatype=XSD.integer)))
                        self.graph.add((correlation_node, BLS_ENRICHMENT.strength,
                                        Literal(correlation['strength'], datatype=XSD.string)))

            self.stats['correlation_links'] += links_added
            logger.info(f"    Created {links_added} correlation links")

    def link_cross_dataset_hierarchies(self):
        """
        Link hierarchies across datasets based on similar paths.

        Currently only implemented for CPI and PPI (the datasets we've analyzed).
        This method can be extended as more datasets with hierarchies are added.
        """
        if 'cpi' not in self.available_datasets or 'ppi' not in self.available_datasets:
            logger.info("  Skipping: Requires both CPI and PPI")
            return

        # Find CPI categories with hierarchies
        cpi_hierarchy_query = """
        SELECT ?category ?parent ?fullPath ?label WHERE {
            ?category <https://www.bls.gov/cpi/hasParent> ?parent ;
                     <https://www.bls.gov/cpi/hasFullPath> ?fullPath .
            OPTIONAL { ?category rdfs:label ?label }
        }
        """

        # Find PPI groupings with hierarchies
        ppi_hierarchy_query = """
        SELECT ?grouping ?parent ?fullPath ?label WHERE {
            ?grouping <https://www.bls.gov/ppi/hasParent> ?parent ;
                     <https://www.bls.gov/ppi/hasFullPath> ?fullPath .
            OPTIONAL { ?grouping rdfs:label ?label }
        }
        """

        cpi_hierarchies = list(self.graph.query(cpi_hierarchy_query))
        ppi_hierarchies = list(self.graph.query(ppi_hierarchy_query))

        logger.info(f"  Analyzing {len(cpi_hierarchies)} CPI and {len(ppi_hierarchies)} PPI hierarchies")

        links_added = 0
        for cpi_row in cpi_hierarchies:
            cpi_path = str(cpi_row.fullPath).lower()
            cpi_label = str(cpi_row.label).lower() if cpi_row.label else ""

            for ppi_row in ppi_hierarchies:
                ppi_path = str(ppi_row.fullPath).lower()
                ppi_label = str(ppi_row.label).lower() if ppi_row.label else ""

                # Check for keyword overlap in paths or labels
                if self._paths_are_related(cpi_path, ppi_path) or \
                        self._labels_are_related(cpi_label, ppi_label):
                    self.graph.add((
                        cpi_row.category,
                        BLS_ENRICHMENT.relatedHierarchy,
                        ppi_row.grouping
                    ))
                    links_added += 1

        self.stats['hierarchy_links'] = links_added
        logger.info(f"  Created {links_added} cross-hierarchy links")

    # ============================================
    # HELPER METHODS
    # ============================================

    def _get_namespace_for_dataset(self, dataset: str) -> Optional[Namespace]:
        """Get namespace for a dataset name"""
        mapping = {
            'cpi': CPI,
            'ppi': PPI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'eci': ECI,
            'laus': LAUS,
            'metro': METRO,
            'realer': REALER,
            'wkyeng': WKYENG,
            'ximpim': XIMPIM
        }
        return mapping.get(dataset.lower())

    def _find_entities_by_keywords(self, namespace: Namespace, keywords: List[str]) -> List[URIRef]:
        """Find entities in a namespace containing any of the keywords"""
        if not keywords:
            return []

        keyword_filters = " || ".join([
            f'CONTAINS(LCASE(?label), "{kw.lower()}")'
            for kw in keywords
        ])

        query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity rdfs:label ?label .
            FILTER(STRSTARTS(STR(?entity), "{namespace}"))
            FILTER({keyword_filters})
        }}
        """

        results = self.graph.query(query)
        return [row.entity for row in results]

    def _find_entities_by_pattern(self, namespace: Namespace, pattern: str) -> List[URIRef]:
        """Find entities matching a specific pattern"""
        query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity rdfs:label ?label .
            FILTER(STRSTARTS(STR(?entity), "{namespace}"))
            FILTER(CONTAINS(LCASE(?label), "{pattern.lower()}"))
        }}
        """

        results = self.graph.query(query)
        return [row.entity for row in results]

    def _paths_are_related(self, path1: str, path2: str) -> bool:
        """Check if two hierarchical paths are related"""
        # Keywords that indicate related concepts
        keywords = ['food', 'energy', 'housing', 'transportation', 'goods', 'services',
                    'manufacturing', 'construction', 'trade', 'warehousing']

        for keyword in keywords:
            if keyword in path1 and keyword in path2:
                return True

        return False

    def _labels_are_related(self, label1: str, label2: str) -> bool:
        """Check if two labels are related"""
        if not label1 or not label2:
            return False

        # Simple word overlap check
        words1 = set(label1.split())
        words2 = set(label2.split())

        # Remove common words
        common_words = {'the', 'and', 'or', 'for', 'of', 'in', 'to', 'a', 'an'}
        words1 -= common_words
        words2 -= common_words

        # Check for overlap
        overlap = words1 & words2
        return len(overlap) >= 1