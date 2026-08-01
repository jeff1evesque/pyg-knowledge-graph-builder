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

# Types for the SOURCE-side temporal entities (cpi:February, eci:2024, ...).
# Those URIs are referenced by measurements but carry no rdf:type of their own,
# so node_mapper never made them nodes and every hasMonth/hasYear/hasStart*/
# hasEnd* triple pointing at them was dropped during edge resolution. The
# TemporalUnifier types them (see _create_source_temporal_types).
#
# Deliberately NOT under a *_ENRICHMENT namespace: an enrichment prefix would
# make classify_edge_origin() read those measurement->month edges as pipeline-
# inferred, when they are observed source facts — only the TYPE is ours. And
# deliberately distinct from UNIFIED: collapsing both onto UnifiedMonth would
# make `unified:February sameAs cpi:February` a link between two nodes of the
# same type, erasing which one is canonical.
SOURCE_TEMPORAL = Namespace("https://example.org/temporal/")

# Statements ABOUT the pipeline's own derivations rather than about the data.
#
# Two kinds live here, both consumed by the metadata collector and ignored by
# every encoder:
#
#   * observations the loader can make but the mapper cannot -- the XSD
#     datatype a source declared on a literal, which survives only at parse
#     time (``prov:observedLiteralDatatype``);
#   * how each derived axiom set was arrived at -- declared, curated, or
#     observed (``prov:derivedBy``), so ontology_schema.json can report
#     provenance instead of presenting an inference as a declaration;
#   * which route produced each INDIVIDUAL axiom, restated edge-for-edge
#     under a route predicate. A set-level description cannot answer "is
#     THIS edge one a person wrote or one a spelling rule guessed", and the
#     two do not deserve the same trust: the naming rules produced
#     AllItemsLessShelter -> Shelter and three more inversions before the
#     negating-qualifier guard caught them, while a curated edge cannot
#     misread a name because a person chose it.
#
# Deliberately NOT a *_ENRICHMENT namespace: these are not claims about
# entities, so classify_edge_origin() must never see them as inferred links.
# Also deliberately absent from NAMESPACE_PREFIXES below: nothing in this
# namespace is ever an rdf:type, so it can never name a node type, and adding
# it would move ONTOLOGY_NAMESPACE_INDICES and change the encoding contract
# digest -- invalidating every trained model to describe URIs no encoder sees.
PROVENANCE = Namespace("https://example.org/provenance/")

# Subject of the derivedBy statements: the axiom SET being described, not any
# individual axiom. One statement per set keeps the marker count constant.
PROV_DERIVED_BY = str(PROVENANCE.derivedBy)
PROV_OBSERVED_LITERAL_DATATYPE = str(PROVENANCE.observedLiteralDatatype)

PROV_CLASS_HIERARCHY = str(PROVENANCE.classHierarchy)
PROV_PROPERTY_DOMAIN = str(PROVENANCE.propertyDomain)
PROV_PROPERTY_RANGE = str(PROVENANCE.propertyRange)
PROV_PROPERTY_HIERARCHY = str(PROVENANCE.propertyHierarchy)

# Per-axiom route markers: the same (subject, object) pair as the axiom, under
# a predicate naming how it was arrived at. One extra triple per derived
# axiom, so the count tracks the derivation rather than the data.
#
# An axiom carrying no marker was not ours -- it came out of the source -- and
# that is how the collector counts declared vs derived. Which is the only way
# to answer "does any source declare rdfs:subClassOf?" from ENRICHED triples,
# where the raw count is 34 and the true answer is still no.
PROV_ROUTE_CURATED_SUBCLASS = str(PROVENANCE.subClassOfCurated)
PROV_ROUTE_NAMED_SUBCLASS = str(PROVENANCE.subClassOfInferredFromNaming)
PROV_ROUTE_OBSERVED_DOMAIN = str(PROVENANCE.domainObservedFromSubjectTypes)
PROV_ROUTE_OBSERVED_RANGE = str(PROVENANCE.rangeObservedFromObjectTypes)
PROV_ROUTE_DATATYPE_RANGE = str(PROVENANCE.rangeFromDeclaredDatatype)
PROV_ROUTE_CURATED_SUBPROPERTY = str(PROVENANCE.subPropertyOfCurated)

# route predicate -> the label ontology_schema.json publishes for it.
PROV_ROUTE_LABELS = {
    PROV_ROUTE_CURATED_SUBCLASS: "curated class mappings",
    PROV_ROUTE_NAMED_SUBCLASS: "class naming",
    PROV_ROUTE_CURATED_SUBPROPERTY: "curated property mappings",
    PROV_ROUTE_OBSERVED_DOMAIN: "observed subject types",
    PROV_ROUTE_OBSERVED_RANGE: "observed object types",
    PROV_ROUTE_DATATYPE_RANGE: "declared literal datatype",
}

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
    (str(SOURCE_TEMPORAL), "temporal"),
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