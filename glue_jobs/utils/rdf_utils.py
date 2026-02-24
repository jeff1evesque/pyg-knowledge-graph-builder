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