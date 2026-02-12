"""
Utilities for loading and manipulating RDF graphs
Foundation for CPI data processing
"""

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)

# ============================================
# BLS DATA SOURCE NAMESPACES
# ============================================
# Based on actual RML mapper outputs

CPI = Namespace("https://www.bls.gov/cpi/")
PPI = Namespace("https://www.bls.gov/ppi/")
ECI = Namespace("https://www.bls.gov/eci/")
EMPSIT = Namespace("https://www.bls.gov/empsit/")
JOLTS = Namespace("https://www.bls.gov/jolts/")
LAUS = Namespace("https://www.bls.gov/laus/")
METRO = Namespace("https://www.bls.gov/metro/")
REALER = Namespace("https://www.bls.gov/realer/")
WKYENG = Namespace("https://www.bls.gov/wkyeng/")
XIMPIM = Namespace("https://www.bls.gov/ximpim/")

# ============================================
# SEC data namespace
# ============================================

SEC_ADMIN = Namespace("https://www.sec.gov/ontology/administrative-proceedings#")
SEC_LIT = Namespace("https://www.sec.gov/ontology/litigation#")
SEC_SUSP = Namespace("https://www.sec.gov/ontology/trading-suspensions#")
SEC_FILINGS = Namespace("http://www.sec.gov/filings#")

# ============================================
# Market data namespace (TODO: Update with actual namespace)
# ============================================

MARKET = Namespace("https://financial-data.org/")
MARKET_OPTIONS = Namespace("https://financial-data.org/options/")

# ============================================
# NOAA WEATHER DATA NAMESPACES
# ============================================

# NOAA base namespace
NOAA = Namespace("https://www.noaa.gov/")

# CAP (Common Alerting Protocol) namespace
CAP = Namespace("http://www.oasis-open.org/committees/emergency/cap/1.2/")

# NWS (National Weather Service) extensions
NWS = Namespace("http://api.weather.gov/ontology/")

# Alert instances
ALERT = Namespace("http://api.weather.gov/alerts/")

# Atom feed namespace
ATOM = Namespace("http://www.w3.org/2005/Atom/")

# ============================================
# ENRICHMENT NAMESPACES
# ============================================
# Created by this pipeline for cross-source linking

BLS_ENRICHMENT = Namespace("https://www.bls.gov/enrichment/")
SEC_ENRICHMENT = Namespace("https://www.sec.gov/enrichment/")
NOAA_ENRICHMENT = Namespace("https://www.noaa.gov/enrichment/")
MARKET_ENRICHMENT = Namespace("https://financial-data.org/enrichment/")
UNIFIED = Namespace("https://example.org/unified/")


class RDFGraphLoader:
    """Load and parse RDF data from S3"""

    def __init__(self, s3_client=None):
        self.s3_client = s3_client
        self.graph = Graph()
        self._bind_namespaces()

    def _bind_namespaces(self):
        """Bind all known namespaces to the graph for cleaner serialization"""
        # Standard
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

        # BLS data sources
        self.graph.bind("cpi", CPI)
        self.graph.bind("ppi", PPI)
        self.graph.bind("eci", ECI)
        self.graph.bind("empsit", EMPSIT)
        self.graph.bind("jolts", JOLTS)
        self.graph.bind("laus", LAUS)
        self.graph.bind("metro", METRO)
        self.graph.bind("realer", REALER)
        self.graph.bind("wkyeng", WKYENG)
        self.graph.bind("ximpim", XIMPIM)

        # NOAA weather data
        self.graph.bind("noaa", NOAA)
        self.graph.bind("cap", CAP)
        self.graph.bind("nws", NWS)
        self.graph.bind("alert", ALERT)
        self.graph.bind("atom", ATOM)

        # SEC data sources
        self.graph.bind("sec", SEC_ADMIN)  # Administrative proceedings
        self.graph.bind("seclit", SEC_LIT)  # Litigation
        self.graph.bind("secsusp", SEC_SUSP)  # Trading suspensions
        self.graph.bind("filings", SEC_FILINGS)  # Filings

        # Other data sources
        self.graph.bind("market", MARKET)

        # Enrichment
        self.graph.bind("bls", BLS_ENRICHMENT)
        self.graph.bind("sec_enrichment", SEC_ENRICHMENT)
        self.graph.bind("noaa_enrichment", NOAA_ENRICHMENT)
        self.graph.bind("unified", UNIFIED)

        # Standard
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

    def load_from_s3(self, bucket: str, key: str, format: str = "turtle"):
        """Load RDF file from S3 into graph"""
        logger.info(f"Loading RDF from s3://{bucket}/{key}")

        if not self.s3_client:
            raise ValueError("S3 client not initialized")

        # Download from S3
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        rdf_data = obj['Body'].read().decode('utf-8')

        # Parse into graph
        self.graph.parse(data=rdf_data, format=format)
        logger.info(f"Loaded {len(self.graph)} triples")

        return self.graph

    def load_from_string(self, rdf_data: str, format: str = "turtle"):
        """Load RDF from string (useful for testing)"""
        logger.info(f"Loading RDF from string")
        self.graph.parse(data=rdf_data, format=format)
        logger.info(f"Loaded {len(self.graph)} triples")
        return self.graph

    def get_entities_by_type(self, entity_type: URIRef) -> List[URIRef]:
        """Get all entities of a specific type"""
        query = f"""
        SELECT ?entity WHERE {{
            ?entity a <{entity_type}> .
        }}
        """
        results = self.graph.query(query)
        return [row.entity for row in results]

    def get_temporal_entities(self,
                              entity_type: URIRef,
                              month_property: URIRef,
                              year_property: URIRef) -> Dict[tuple, List[URIRef]]:
        """
        Get entities grouped by time period

        Returns:
            Dict mapping (year, month) tuples to lists of entity URIs
        """
        query = f"""
        SELECT ?entity ?month ?year WHERE {{
            ?entity a <{entity_type}> ;
                    <{month_property}> ?month ;
                    <{year_property}> ?year .
        }}
        """
        results = self.graph.query(query)

        # Group by (year, month)
        temporal_groups = {}
        for row in results:
            # Extract just the local name from month/year URIs
            month_name = str(row.month).split('/')[-1]
            year_name = str(row.year).split('/')[-1]

            key = (year_name, month_name)
            if key not in temporal_groups:
                temporal_groups[key] = []
            temporal_groups[key].append(row.entity)

        return temporal_groups

    def get_all_months(self) -> Set[URIRef]:
        """Get all Month entities in the graph"""
        return set(self.get_entities_by_type(CPI.Month))

    def get_all_years(self) -> Set[URIRef]:
        """Get all Year entities in the graph"""
        return set(self.get_entities_by_type(CPI.Year))

    def get_hierarchical_entities(self,
                                  entity_type: URIRef,
                                  parent_property: URIRef) -> Dict[URIRef, List[URIRef]]:
        """
        Get entities organized by parent-child relationships

        Returns:
            Dict mapping parent URIs to lists of child URIs
        """
        query = f"""
        SELECT ?child ?parent WHERE {{
            ?child a <{entity_type}> ;
                   <{parent_property}> ?parent .
        }}
        """
        results = self.graph.query(query)

        hierarchy = {}
        for row in results:
            parent = row.parent
            child = row.child

            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(child)

        return hierarchy

    def get_root_entities(self, entity_type: URIRef, parent_property: URIRef) -> List[URIRef]:
        """Get entities that have no parent (root of hierarchy)"""
        query = f"""
        SELECT ?entity WHERE {{
            ?entity a <{entity_type}> .
            FILTER NOT EXISTS {{ ?entity <{parent_property}> ?parent }}
        }}
        """
        results = self.graph.query(query)
        return [row.entity for row in results]


class RDFGraphWriter:
    """Write enriched RDF data to S3"""

    def __init__(self, s3_client=None):
        self.s3_client = s3_client

    def save_to_s3(self, graph: Graph, bucket: str, key: str, format: str = "turtle"):
        """Save RDF graph to S3"""
        logger.info(f"Saving {len(graph)} triples to s3://{bucket}/{key}")

        if not self.s3_client:
            raise ValueError("S3 client not initialized")

        # Serialize graph
        rdf_data = graph.serialize(format=format)

        # Upload to S3
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=rdf_data.encode('utf-8'),
            ContentType='text/turtle' if format == 'turtle' else 'application/rdf+xml'
        )

        logger.info(f"Successfully saved to S3")

    def save_to_string(self, graph: Graph, format: str = "turtle") -> str:
        """Save RDF graph to string (useful for testing)"""
        return graph.serialize(format=format)


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_month_name(month_uri: URIRef) -> str:
    """Extract month name from URI (e.g., cpi:November -> 'November')"""
    return str(month_uri).split('/')[-1]


def get_year_value(year_uri: URIRef) -> str:
    """Extract year value from URI (e.g., cpi:2024 -> '2024')"""
    return str(year_uri).split('/')[-1]


def create_unified_temporal_uri(month: str, year: str) -> URIRef:
    """Create unified temporal entity URI"""
    return UNIFIED[f"{month}{year}"]