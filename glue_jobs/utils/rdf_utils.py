"""
Utilities for loading and manipulating RDF graphs
Foundation for CPI data processing
"""

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
from typing import List, Tuple
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
# Market data namespace
# ============================================
# Source-agnostic namespace for equity/option quote snapshots.
# The upstream scraper (currently Schwab) writes flat snapshot rows;
# this namespace models them generically so switching data sources
# only requires changing the scraper, not the enrichment pipeline.

MARKET = Namespace("https://financial-data.org/")

# ============================================
# NOAA WEATHER DATA NAMESPACES
# ============================================

# NOAA base namespace
NOAA = Namespace("https://www.noaa.gov/")

# CAP (Common Alerting Protocol) namespace
CAP = Namespace("http://www.oasis-open.org/committees/emergency/cap/1.2/")

# NWS (National Weather Service) extensions
NWS = Namespace("https://api.weather.gov/ontology#")

# Alert instances
ALERT = Namespace("https://api.weather.gov/alerts/")

# GeoSPARQL namespace (used by CAP Area alignment)
GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")

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

# ============================================
# CANONICAL NAMESPACE → PREFIX MAPPING
# ============================================
# Single source of truth for all PyG builder modules (node_mapper,
# edge_mapper, feature_extractor). Each entry is (namespace_uri, prefix).
#
# Order matters for node_mapper and edge_mapper: longer/more-specific
# namespaces must appear before shorter ones so that startsWith matching
# picks the most specific prefix (e.g., "https://financial-data.org/enrichment/"
# before "https://financial-data.org/").
#
# The index position in this list is used by feature_extractor for
# ontology source membership encoding.

NAMESPACE_PREFIXES: List[Tuple[str, str]] = [
    (str(CPI), "cpi"),
    (str(PPI), "ppi"),
    (str(ECI), "eci"),
    (str(EMPSIT), "empsit"),
    (str(JOLTS), "jolts"),
    (str(LAUS), "laus"),
    (str(METRO), "metro"),
    (str(REALER), "realer"),
    (str(WKYENG), "wkyeng"),
    (str(XIMPIM), "ximpim"),
    (str(BLS_ENRICHMENT), "bls_enrichment"),
    (str(SEC_FILINGS), "filings"),
    (str(SEC_ADMIN), "sec_admin"),
    (str(SEC_LIT), "sec_lit"),
    (str(SEC_SUSP), "sec_susp"),
    (str(SEC_ENRICHMENT), "sec_enrichment"),
    (str(MARKET_ENRICHMENT), "market_enrichment"),
    (str(MARKET), "market"),
    (str(CAP), "cap"),
    (str(NWS), "nws"),
    (str(ALERT), "alert"),
    (str(NOAA_ENRICHMENT), "noaa_enrichment"),
    (str(NOAA), "noaa"),
    (str(GEOSPARQL), "geosparql"),
    (str(UNIFIED), "unified"),
    (str(OWL), "owl"),
    (str(RDFS), "rdfs"),
]

# Derived: namespace → integer index for feature_extractor ontology
# source membership encoding. Built automatically from NAMESPACE_PREFIXES.
ONTOLOGY_NAMESPACE_INDICES: List[Tuple[str, int]] = [
    (ns, idx) for idx, (ns, _prefix) in enumerate(NAMESPACE_PREFIXES)
]


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