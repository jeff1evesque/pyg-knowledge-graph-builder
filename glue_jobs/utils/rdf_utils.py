# glue_jobs/utils/rdf_utils.py
"""
Utilities for loading and manipulating RDF graphs
Start here to establish basic RDF operations
"""

from rdflib import Graph, Namespace, URIRef, Literal
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Define BLS data source namespaces (matching actual RML mapper outputs)
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

# SEC data namespaces
SEC = Namespace("https://www.sec.gov/")

# Market data namespace
MARKET = Namespace("https://example.org/market/")  # TODO: Update with actual namespace

# NOAA weather data namespace
NOAA = Namespace("https://www.weather.gov/")  # TODO: Update with actual namespace

# Enrichment/unified namespace (created by this pipeline)
BLS = Namespace("https://www.bls.gov/ontology/")  # Common BLS enrichment vocabulary
UNIFIED = Namespace("https://example.org/unified/")  # Cross-source unified entities

# Standard namespaces
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


class RDFGraphLoader:
    """Load and parse RDF data from S3"""

    def __init__(self, s3_client):
        self.s3_client = s3_client
        self.graph = Graph()

        # Bind namespaces to graph for cleaner serialization
        self._bind_namespaces()

    def _bind_namespaces(self):
        """Bind all known namespaces to the graph"""
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
        self.graph.bind("sec", SEC)
        self.graph.bind("market", MARKET)
        self.graph.bind("noaa", NOAA)
        self.graph.bind("bls", BLS)
        self.graph.bind("unified", UNIFIED)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

    def load_from_s3(self, bucket: str, key: str, format: str = "turtle"):
        """Load RDF file from S3 into graph"""
        logger.info(f"Loading RDF from s3://{bucket}/{key}")

        # Download from S3
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        rdf_data = obj['Body'].read().decode('utf-8')

        # Parse into graph
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

    def get_temporal_entities(self, entity_type: URIRef,
                              month_property: URIRef,
                              year_property: URIRef) -> Dict:
        """Get entities grouped by time period"""
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
            key = (str(row.year), str(row.month))
            if key not in temporal_groups:
                temporal_groups[key] = []
            temporal_groups[key].append(row.entity)

        return temporal_groups
