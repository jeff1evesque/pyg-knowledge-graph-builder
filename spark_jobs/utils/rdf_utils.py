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
# The upstream scraper writes flat snapshot rows;
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
# EDGE ORIGIN CLASSIFICATION
# ============================================
# Namespaces this pipeline mints itself, as opposed to reading from a source.
ENRICHMENT_NAMESPACES: Tuple[str, ...] = (
    str(BLS_ENRICHMENT),
    str(SEC_ENRICHMENT),
    str(NOAA_ENRICHMENT),
    str(MARKET_ENRICHMENT),
)

ORIGIN_RAW = "raw"
ORIGIN_ENRICHMENT = "enrichment"
ORIGIN_UNIFICATION = "unification"

# Node-type prefixes that only this pipeline mints. Derived from
# NAMESPACE_PREFIXES so a new enrichment namespace is picked up automatically
# rather than needing a second list kept in sync.
_PIPELINE_NAMESPACES = ENRICHMENT_NAMESPACES + (str(UNIFIED),)
PIPELINE_NODE_TYPE_PREFIXES: Tuple[str, ...] = tuple(
    f"{prefix}_"
    for ns, prefix in NAMESPACE_PREFIXES
    if ns in _PIPELINE_NAMESPACES
)

# Identity linking uses the standard OWL vocabulary, not a pipeline namespace.
OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"


def classify_edge_origin(
    predicate_uri: str,
    src_type: str = "",
    dst_type: str = "",
) -> str:
    """Whether an edge was observed in the source data or created by this pipeline.

    Enrichment adds ~91,000 triples on top of the raw data, so a graph edge may
    be a reported fact or something the pipeline inferred. Those deserve very
    different trust from a downstream model, and nothing recorded the difference
    — graph_schema.json reported "unknown" for every edge type.

    Both the predicate AND the endpoints are consulted, because the pipeline
    marks its output in two different places:

      * inferred links carry a minted PREDICATE
        (``bls.gov/enrichment/apparelSectorCorrelation``)
      * unification links carry a minted NODE but a standard predicate
        (``unified:November owl:sameAs cpi:November``)

    Classifying on the predicate alone therefore reports every unification edge
    as ``raw`` — an inferred link presented as an observed fact, the exact
    mislabelling this exists to prevent. On the e2e fixtures that was 16 of 19
    supposedly-raw edge types.

    Anything whose predicate and both endpoints are outside the pipeline's own
    namespaces came from a source scraper.

    Three values, not the four the metadata docstring once listed (raw /
    intra-enrichment / cross-enrichment / unification): intra- and cross-source
    enrichment share a namespace, so nothing here can separate them, and
    promising a distinction that is never emitted is the same defect as the
    "unknown" it replaces.

    Says *that* an edge was derived, not *why* — which source facts justified
    it. That is triple-level lineage, a much larger change, worth building only
    once a model's predictions need explaining.
    """
    minted = (
        predicate_uri.startswith(_PIPELINE_NAMESPACES)
        if predicate_uri
        else False
    ) or any(
        t.startswith(PIPELINE_NODE_TYPE_PREFIXES)
        for t in (src_type or "", dst_type or "")
    )

    if not minted:
        return ORIGIN_RAW
    if predicate_uri == OWL_SAME_AS or predicate_uri.endswith("#sameAs"):
        return ORIGIN_UNIFICATION
    return ORIGIN_ENRICHMENT


    if predicate_uri.startswith(str(UNIFIED)):
        return ORIGIN_UNIFICATION
    if predicate_uri.startswith(ENRICHMENT_NAMESPACES):
        return ORIGIN_ENRICHMENT
    return ORIGIN_RAW


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