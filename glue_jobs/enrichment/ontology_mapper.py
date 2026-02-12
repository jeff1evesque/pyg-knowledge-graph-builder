"""
Ontology Mapping Utilities

Maps between different ontology vocabularies and creates standardized mappings.
This is OPTIONAL - the pipeline works without it, but it improves interoperability.

Use cases:
1. Standardize property names across BLS datasets (hasMonth → unified:hasMonth)
2. Create owl:equivalentProperty and owl:equivalentClass mappings
3. Add SKOS metadata for better semantic interoperability
4. Align classification systems (NAICS, SIC, GICS, etc.)

Example:
    from glue_jobs.enrichment.ontology_mapper import OntologyMapper

    mapper = OntologyMapper(graph)
    mapper.create_equivalence_mappings()
    mapper.normalize_labels()
"""
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
from glue_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT, NOAA_ENRICHMENT,
    UNIFIED, CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER,
    SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP, MARKET, CAP
)
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# SKOS namespace for semantic metadata
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DCTERMS = Namespace("http://purl.org/dc/terms/")


class OntologyMapper:
    """
    Maps between different ontology vocabularies

    Creates standardized mappings to improve interoperability and
    make PyG construction easier with consistent property names.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.stats = {
            'property_equivalences': 0,
            'class_equivalences': 0,
            'normalized_labels': 0,
            'skos_concepts': 0,
            'classification_alignments': 0
        }

        # Bind additional namespaces
        self.graph.bind("skos", SKOS)
        self.graph.bind("dcterms", DCTERMS)

        # Load mapping configurations
        self.property_mappings = self._load_property_mappings()
        self.class_mappings = self._load_class_mappings()

    def _load_property_mappings(self) -> Dict[str, str]:
        """
        Load property equivalence mappings

        Maps dataset-specific properties to unified properties
        """
        return {
            # ============================================
            # TEMPORAL PROPERTY MAPPINGS
            # ============================================
            # All month properties → unified:hasMonth
            str(CPI.hasMonth): str(UNIFIED.hasMonth),
            str(PPI.hasStartMonth): str(UNIFIED.hasMonth),
            str(PPI.hasEndMonth): str(UNIFIED.hasMonth),
            str(ECI.hasMonth): str(UNIFIED.hasMonth),
            str(JOLTS.hasMonth): str(UNIFIED.hasMonth),
            str(EMPSIT.hasMonth): str(UNIFIED.hasMonth),
            str(XIMPIM.hasMonth): str(UNIFIED.hasMonth),
            str(LAUS.hasMonth): str(UNIFIED.hasMonth),
            str(METRO.hasMonth): str(UNIFIED.hasMonth),
            str(REALER.hasMonth): str(UNIFIED.hasMonth),

            # All year properties → unified:hasYear
            str(CPI.hasYear): str(UNIFIED.hasYear),
            str(PPI.hasStartYear): str(UNIFIED.hasYear),
            str(PPI.hasEndYear): str(UNIFIED.hasYear),
            str(ECI.hasYear): str(UNIFIED.hasYear),
            str(JOLTS.hasYear): str(UNIFIED.hasYear),
            str(EMPSIT.hasYear): str(UNIFIED.hasYear),
            str(XIMPIM.hasYear): str(UNIFIED.hasYear),
            str(LAUS.hasYear): str(UNIFIED.hasYear),
            str(METRO.hasYear): str(UNIFIED.hasYear),
            str(REALER.hasYear): str(UNIFIED.hasYear),

            # ============================================
            # MEASUREMENT VALUE MAPPINGS
            # ============================================
            # All measurement values → unified:measurementValue
            str(CPI.indexValue): str(UNIFIED.measurementValue),
            str(PPI.changeValue): str(UNIFIED.measurementValue),
            str(PPI.indexValue): str(UNIFIED.measurementValue),
            str(JOLTS.level): str(UNIFIED.measurementValue),
            str(JOLTS.rate): str(UNIFIED.measurementValue),
            str(EMPSIT.value): str(UNIFIED.measurementValue),
            str(ECI.indexValue): str(UNIFIED.measurementValue),
            str(MARKET.observedPrice): str(UNIFIED.measurementValue),
            str(MARKET.lastPrice): str(UNIFIED.measurementValue),

            # ============================================
            # CATEGORY/ENTITY MAPPINGS
            # ============================================
            # All category properties → unified:hasCategory
            str(CPI.hasCategory): str(UNIFIED.hasCategory),
            str(PPI.hasCommodityGrouping): str(UNIFIED.hasCategory),
            str(ECI.hasOccupationalGroup): str(UNIFIED.hasCategory),
            str(JOLTS.hasIndustry): str(UNIFIED.hasCategory),
            str(EMPSIT.hasIndustry): str(UNIFIED.hasCategory),
            str(EMPSIT.hasCategory): str(UNIFIED.hasCategory),

            # ============================================
            # COMPANY/TICKER MAPPINGS
            # ============================================
            str(MARKET.symbol): str(UNIFIED.ticker),
            # SEC CIK mappings handled separately

            # ============================================
            # GEOGRAPHIC MAPPINGS
            # ============================================
            str(LAUS.hasState): str(UNIFIED.hasRegion),
            str(METRO.hasMetropolitanArea): str(UNIFIED.hasRegion),
        }

    def _load_class_mappings(self) -> Dict[str, str]:
        """
        Load class equivalence mappings

        Maps dataset-specific classes to unified classes
        """
        return {
            # ============================================
            # MEASUREMENT TYPE MAPPINGS
            # ============================================
            # Price indices
            str(CPI.Index): str(BLS_ENRICHMENT.PriceIndex),
            str(PPI.IndexValue): str(BLS_ENRICHMENT.PriceIndex),

            # Rate measurements
            str(JOLTS.JobOpeningsRate): str(BLS_ENRICHMENT.RateMeasurement),
            str(JOLTS.HiresRate): str(BLS_ENRICHMENT.RateMeasurement),
            str(JOLTS.QuitsRate): str(BLS_ENRICHMENT.RateMeasurement),
            str(LAUS.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),
            str(METRO.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),

            # Change measurements
            str(CPI.PercentChange): str(BLS_ENRICHMENT.ChangeMeasurement),
            str(PPI.MonthlyChange): str(BLS_ENRICHMENT.ChangeMeasurement),
            str(PPI.TwelveMonthChange): str(BLS_ENRICHMENT.ChangeMeasurement),
            str(ECI.PercentChangeData): str(BLS_ENRICHMENT.ChangeMeasurement),

            # Level measurements
            str(JOLTS.JobOpeningsLevel): str(BLS_ENRICHMENT.LevelMeasurement),
            str(JOLTS.HiresLevel): str(BLS_ENRICHMENT.LevelMeasurement),
            str(EMPSIT.EmployeeCount): str(BLS_ENRICHMENT.LevelMeasurement),
            str(LAUS.LaborForceData): str(BLS_ENRICHMENT.LevelMeasurement),

            # ============================================
            # ENTITY TYPE MAPPINGS
            # ============================================
            # Economic indicators
            str(CPI.Category): str(BLS_ENRICHMENT.EconomicIndicator),
            str(PPI.CommodityGrouping): str(BLS_ENRICHMENT.EconomicIndicator),

            # Industry classifications
            str(JOLTS.Industry): str(BLS_ENRICHMENT.IndustryClassification),
            str(EMPSIT.Industry): str(BLS_ENRICHMENT.IndustryClassification),
            str(ECI.Industry): str(BLS_ENRICHMENT.IndustryClassification),

            # Occupational classifications
            str(ECI.OccupationalGroup): str(BLS_ENRICHMENT.OccupationalClassification),
            str(EMPSIT.Occupation): str(BLS_ENRICHMENT.OccupationalClassification),
        }

    def create_equivalence_mappings(self):
        """
        Create owl:equivalentProperty and owl:equivalentClass mappings

        This makes it easier to query the graph with standardized properties
        and enables reasoning engines to infer relationships.
        """
        logger.info("Creating ontology equivalence mappings...")

        # Property equivalences
        for source_prop, target_prop in self.property_mappings.items():
            source_uri = URIRef(source_prop)
            target_uri = URIRef(target_prop)

            # Add equivalence if not already exists
            if (source_uri, OWL.equivalentProperty, target_uri) not in self.graph:
                self.graph.add((source_uri, OWL.equivalentProperty, target_uri))
                self.stats['property_equivalences'] += 1

        # Class equivalences
        for source_class, target_class in self.class_mappings.items():
            source_uri = URIRef(source_class)
            target_uri = URIRef(target_class)

            # Add equivalence if not already exists
            if (source_uri, OWL.equivalentClass, target_uri) not in self.graph:
                self.graph.add((source_uri, OWL.equivalentClass, target_uri))
                self.stats['class_equivalences'] += 1

        logger.info(f"  Created {self.stats['property_equivalences']} property equivalences")
        logger.info(f"  Created {self.stats['class_equivalences']} class equivalences")

    def normalize_labels(self):
        """
        Normalize labels across ontologies

        Adds skos:prefLabel for entities that only have rdfs:label
        This improves semantic interoperability with other knowledge graphs.
        """
        logger.info("Normalizing labels...")

        # Add skos:prefLabel for entities with rdfs:label but no skos:prefLabel
        query = """
        SELECT ?entity ?label WHERE {
            ?entity rdfs:label ?label .
            FILTER NOT EXISTS { ?entity skos:prefLabel ?prefLabel }
        }
        """

        for row in self.graph.query(query):
            self.graph.add((row.entity, SKOS.prefLabel, row.label))
            self.stats['normalized_labels'] += 1

        logger.info(f"  Added {self.stats['normalized_labels']} skos:prefLabel triples")

    def create_skos_concepts(self):
        """
        Create SKOS concept schemes for major classifications

        Useful for publishing the knowledge graph as Linked Open Data
        """
        logger.info("Creating SKOS concept schemes...")

        # Create concept schemes for major classifications
        concept_schemes = {
            'bls_sectors': {
                'uri': BLS_ENRICHMENT.SectorConceptScheme,
                'label': 'BLS Economic Sectors',
                'description': 'Economic sector classifications used in BLS data'
            },
            'bls_industries': {
                'uri': BLS_ENRICHMENT.IndustryConceptScheme,
                'label': 'BLS Industry Classifications',
                'description': 'Industry classifications across BLS datasets'
            },
            'bls_occupations': {
                'uri': BLS_ENRICHMENT.OccupationConceptScheme,
                'label': 'BLS Occupational Classifications',
                'description': 'Occupational group classifications in ECI and EMPSIT'
            }
        }

        for scheme_name, scheme_config in concept_schemes.items():
            scheme_uri = scheme_config['uri']

            # Add concept scheme
            self.graph.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
            self.graph.add((scheme_uri, RDFS.label, Literal(scheme_config['label'])))
            self.graph.add((scheme_uri, DCTERMS.description, Literal(scheme_config['description'])))

            self.stats['skos_concepts'] += 1

        logger.info(f"  Created {self.stats['skos_concepts']} SKOS concept schemes")

    def align_classification_systems(self):
        """
        Align different classification systems

        Examples:
        - NAICS codes (used in XIMPIM, JOLTS, EMPSIT)
        - SIC codes (legacy)
        - GICS sectors (market data)
        - Harmonized System codes (XIMPIM)

        This is ADVANCED and optional - only needed if you want to
        map between different industry classification systems.
        """
        logger.info("Aligning classification systems...")

        # Example: Map NAICS to unified industry classifications
        # This would require external NAICS ontology or mapping tables

        # For now, just log that this is available for future enhancement
        logger.info("  Classification alignment available for future enhancement")
        logger.info("  (Requires external NAICS/SIC/GICS mapping tables)")

    def add_dublin_core_metadata(self):
        """
        Add Dublin Core metadata to major entities

        Useful for publishing the knowledge graph
        """
        logger.info("Adding Dublin Core metadata...")

        # Add dcterms:created for unified temporal entities
        query = f"""
        SELECT ?entity ?label WHERE {{
            ?entity a <{BLS_ENRICHMENT.UnifiedMonth}> ;
                    rdfs:label ?label .
            FILTER NOT EXISTS {{ ?entity <{DCTERMS.created}> ?created }}
        }}
        """

        for row in self.graph.query(query):
            # Add creation timestamp
            self.graph.add((
                row.entity,
                DCTERMS.created,
                Literal(datetime.now().isoformat(), datatype=XSD.dateTime)
            ))

        logger.info("  Added Dublin Core metadata")

    def get_stats(self) -> Dict[str, int]:
        """Return mapping statistics"""
        return self.stats.copy()


def map_ontologies(graph: Graph,
                   enable_skos: bool = False,
                   enable_dublin_core: bool = False) -> Dict[str, int]:
    """
    Convenience function for ontology mapping

    Args:
        graph: RDFLib graph to enrich
        enable_skos: Whether to create SKOS concept schemes
        enable_dublin_core: Whether to add Dublin Core metadata

    Returns:
        Dictionary with mapping statistics

    Example:
        from glue_jobs.enrichment.ontology_mapper import map_ontologies

        stats = map_ontologies(graph, enable_skos=True)
    """
    mapper = OntologyMapper(graph)

    # Always create equivalence mappings and normalize labels
    mapper.create_equivalence_mappings()
    mapper.normalize_labels()

    # Optional enhancements
    if enable_skos:
        mapper.create_skos_concepts()

    if enable_dublin_core:
        mapper.add_dublin_core_metadata()

    return mapper.get_stats()