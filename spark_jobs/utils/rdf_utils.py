"""
Utilities for loading and manipulating RDF graphs
Foundation for CPI data processing
"""

from rdflib import Namespace, URIRef
from rdflib.namespace import RDFS, OWL
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================
# SOURCE VOCABULARIES — terms the SCRAPERS mint
# ============================================
# These used to sit on the upstream providers' own domains. None of those
# organizations defined these terms: the upstreams publish tabular and
# document formats, not RDF, and every class and property below was invented
# by the upstream scrapers.
#
# They now resolve to a domain we control, split so that a reader can tell a
# term from a thing by looking at the URI:
#
#   https://jefflevesque.com/ontology/{source}/   classes and properties
#   https://jefflevesque.com/id/{source}/         individuals
#
# Node types come from rdf:type objects and edges from predicates, both of
# which are terms, so the constants below are what the mappers need. Code that
# asks which source an ENTITY belongs to needs the id/ side instead — see
# SOURCE_IDENTIFIERS and identifier_namespace() further down.
SOURCE_BASE = "https://jefflevesque.com/ontology/"

CPI = Namespace(f"{SOURCE_BASE}cpi/")
PPI = Namespace(f"{SOURCE_BASE}ppi/")
ECI = Namespace(f"{SOURCE_BASE}eci/")
EMPSIT = Namespace(f"{SOURCE_BASE}empsit/")
JOLTS = Namespace(f"{SOURCE_BASE}jolts/")
LAUS = Namespace(f"{SOURCE_BASE}laus/")
METRO = Namespace(f"{SOURCE_BASE}metro/")
REALER = Namespace(f"{SOURCE_BASE}realer/")
WKYENG = Namespace(f"{SOURCE_BASE}wkyeng/")
XIMPIM = Namespace(f"{SOURCE_BASE}ximpim/")

# Shared BLS classes (Month, Year, Industry, Region, ...) that the hand-
# authored table schemas declare once for every category. 'bls-common' rather
# than 'bls' because BLS_ENRICHMENT below already owns .../ontology/bls/ —
# merging the two would make observed BLS facts read as pipeline-inferred.
BLS_COMMON = Namespace(f"{SOURCE_BASE}bls-common/")

# ============================================
# SEC data namespaces
# ============================================
# 'sec-*' rather than 'sec' for the same reason: SEC_ENRICHMENT owns
# .../ontology/sec/.

SEC_COMMON = Namespace(f"{SOURCE_BASE}sec-common/")
SEC_ADMIN = Namespace(f"{SOURCE_BASE}sec-administrative-proceedings/")
SEC_LIT = Namespace(f"{SOURCE_BASE}sec-litigation/")
SEC_SUSP = Namespace(f"{SOURCE_BASE}sec-trading-suspensions/")
SEC_FILINGS = Namespace(f"{SOURCE_BASE}sec-filings/")

# ============================================
# Market data namespaces
# ============================================
# There are TWO market vocabularies, and one constant was previously asked to
# name both — which is why market enrichment silently produced nothing:
#
#   MARKET_QUOTES  the quote-snapshot API scraper (upstream). Flat
#                  snapshots: EquitySnapshot / OptionSnapshot, with
#                  captureTime, askPrice, delta.
#
#   MARKET_FEEDS   the HTML feed scrapers (upstream). A
#                  different model: PriceObservation / OptionContract, with
#                  observedAt and expirationDate.
#
# MARKET used to point at the feeds namespace while measurements.py keyed on
# the quotes local names, so it matched nothing on real data and nothing
# raised. temporal_unifier.py, which keys on observedAt / PriceObservation,
# was correct all along — for the feeds vocabulary.
#
# They are deliberately NOT unified here. Aligning EquitySnapshot with
# PriceObservation is an ontology decision, not a rename, and pretending one
# name covers both is what produced the original defect.
MARKET_QUOTES = Namespace(f"{SOURCE_BASE}market-quotes/")
MARKET_FEEDS = Namespace(f"{SOURCE_BASE}market-feeds/")
MARKET_FEEDS_OPTIONS = Namespace(f"{SOURCE_BASE}market-feeds-options/")

# ============================================
# NOAA WEATHER DATA NAMESPACES
# ============================================
# Verified against 275 live alerts from api.weather.gov: NWS publishes
# wx:Alert plus ~30 lowercase properties (affectedZones, areaDesc, geocode).
# Not one of the ~29 terms the scraper emits is among them, and their live
# JSON-LD context declares api.weather.gov/ontology# as @vocab — so minting
# our terms there put our inventions inside a live publisher's vocabulary.
#
# Likewise CAP: OASIS registered urn:oasis:names:tc:emergency:cap:1.2 as an
# XML namespace naming lowercase ELEMENTS. There is no OASIS CAP RDF
# vocabulary, so cap:hasAreaDescription and cap:AlertMessage are our model OF
# CAP rather than CAP itself.

WEATHER = Namespace(f"{SOURCE_BASE}weather/")
CAP = Namespace(f"{SOURCE_BASE}cap-model/")

# Alert instances — real identifiers for real NWS alerts. Genuinely theirs,
# unlike the vocabulary that used to sit alongside them, so left alone.
ALERT = Namespace("https://api.weather.gov/alerts/")

# GeoSPARQL namespace (used by CAP Area alignment). Really OGC's.
GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")

# Atom feed namespace
ATOM = Namespace("http://www.w3.org/2005/Atom/")

# Every source vocabulary above, for the invariant tests. These are minted by
# the SCRAPERS rather than by this pipeline, which is why they are not in
# ENRICHMENT_NAMESPACES -- an edge using one of them is an observed source
# fact, not something the pipeline inferred -- but they are still ours, so
# they are held to the same rule about which domain they may live on.
SOURCE_VOCABULARIES: Tuple[str, ...] = (
    str(CPI), str(PPI), str(ECI), str(EMPSIT), str(JOLTS),
    str(LAUS), str(METRO), str(REALER), str(WKYENG), str(XIMPIM),
    str(BLS_COMMON),
    str(SEC_COMMON), str(SEC_ADMIN), str(SEC_LIT), str(SEC_SUSP),
    str(SEC_FILINGS),
    str(MARKET_QUOTES), str(MARKET_FEEDS), str(MARKET_FEEDS_OPTIONS),
    str(WEATHER), str(CAP),
)

# Vocabularies their publishers really did define, and which we therefore
# reuse at their real URIs. Reusing a real term is correct RDF and is what
# makes federation work; minting a NEW term into one is the defect the
# namespace tests exist to catch.
#
# This list is short, and deliberately so. It was checked rather than assumed:
# api.weather.gov/ontology# looked like it belonged here, but NWS publishes
# wx:Alert plus ~30 lowercase properties there and not one of the ~29 terms
# the scraper emitted was among them.
PUBLISHER_VOCABULARIES: Tuple[str, ...] = (
    str(ALERT),
    str(GEOSPARQL),
    str(ATOM),
)

# ============================================
# SOURCE IDENTIFIERS — individuals the SCRAPERS mint
# ============================================
# The namespaces above name classes and properties. These name the things
# themselves: id/cpi/February is the month, ontology/cpi/Month is its class.
#
# The header of this module used to say only the TERM namespaces were needed
# here, because "the id/ namespaces appear only as subjects and objects". That
# is true of node typing and edge naming, and false of everything that asks
# "which source is this ENTITY from" — intra-source detection, per-source
# filtering, temporal collection. Those must match an entity URI against the
# id/ namespace; matching the ontology/ one silently matches nothing, since no
# individual lives there. Before the term/individual split a single namespace
# covered both and the distinction did not exist, so every such call site was
# written against the term namespace and kept working. They do not any more.
#
# Derived from the term namespace rather than written out, so the two cannot
# drift apart: a new source vocabulary gets its identifier namespace for free.
IDENTIFIER_BASE = "https://jefflevesque.com/id/"


def identifier_namespace(term_namespace: str) -> str:
    """The id/ namespace matching a term namespace.

    ``https://jefflevesque.com/ontology/cpi/`` -> ``https://jefflevesque.com/id/cpi/``

    Raises rather than guessing for anything outside SOURCE_BASE: a publisher
    vocabulary (api.weather.gov/alerts/) has no id/ counterpart, and quietly
    returning something plausible would reintroduce the silent-no-match bug
    this function exists to prevent.
    """
    if not term_namespace.startswith(SOURCE_BASE):
        raise ValueError(
            f"{term_namespace!r} is not a source vocabulary under {SOURCE_BASE!r}; "
            "it has no identifier namespace"
        )
    return f"{IDENTIFIER_BASE}{term_namespace[len(SOURCE_BASE):]}"


# Every source vocabulary's identifier counterpart, same order as
# SOURCE_VOCABULARIES. Pinned by test_namespaces.py.
SOURCE_IDENTIFIERS: Tuple[str, ...] = tuple(
    identifier_namespace(ns) for ns in SOURCE_VOCABULARIES
)

# ============================================
# MINTED NAMESPACES — terms this project invents
# ============================================
# Everything below is a term WE define, so every one of them lives under a
# domain we control. The namespaces above are the publishers' own vocabularies
# and stay exactly where they are: those really are their terms.
#
# They used to sit under the publishers' domains (bls.gov/enrichment/,
# sec.gov/enrichment/, noaa.gov/enrichment/, financial-data.org/enrichment/)
# and under example.org. Both were wrong, in different ways:
#
#   * a URI under bls.gov claims BLS is the authority for that term. Nobody at
#     BLS defined bls_enrichment:RateMeasurement -- we did. Anyone consuming
#     this graph, or federating it with real BLS-published RDF, is entitled to
#     believe the URI and would be wrong. It would also collide outright if
#     BLS ever published under that path.
#   * example.org is reserved by RFC 2606 for documentation. It cannot lie
#     about ownership, but it is nobody's, so another project using it (which
#     is exactly what it is for) collides with us, and a reviewer cannot tell
#     a deliberate choice from a leftover placeholder.
#
# ONE base, sub-pathed by concern. Changing it is a one-line edit here --
# nothing downstream hardcodes a namespace string -- but it is not free: these
# URIs are hashed into feature slots, so moving them moves every slot and
# changes encoding_config.json's contract digest. That is the intended signal
# (graphs built either side are not comparable), not a side effect.
ONTOLOGY_BASE = "https://jefflevesque.com/ontology/"

BLS_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}bls/")
SEC_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}sec/")
NOAA_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}noaa/")
MARKET_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}market/")
UNIFIED = Namespace(f"{ONTOLOGY_BASE}unified/")

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
#
# Under the shared base like the rest, but NOT in ENRICHMENT_NAMESPACES --
# that list, not the base URI, is what classify_edge_origin() reads. Sharing a
# base with the enrichment namespaces must not start classifying these edges
# as pipeline-inferred; see the origin test that pins it.
SOURCE_TEMPORAL = Namespace(f"{ONTOLOGY_BASE}temporal/")

# Where the synthetic temporal INDIVIDUALS live.
#
# SOURCE_TEMPORAL above holds the TYPES (SourceMonth, SourceYear,
# SourceQuarter). temporal_unifier mints one individual per period per source
# -- November, 2024 -- and those are things, not terms, so they belong under
# the identifier base rather than alongside their own types.
#
# They were minted at https://www.sec.gov/temporal/ and
# https://www.noaa.gov/temporal/, which is the same false-provenance defect as
# every source vocabulary: nobody at those organizations minted
# sec.gov/temporal/November. This pipeline did, on every build.
#
# Keyed by source so two sources naming the same period stay distinct until
# the unifier deliberately links them -- collapsing them here would pre-empt
# the sameAs step and hide which source observed what.
#
# IDENTIFIER_BASE is defined with the source identifiers above -- these are
# individuals like any other, minted by this pipeline rather than a scraper.
# The two market vocabularies get separate keys for the same reason any two
# sources do. They are not one source with two spellings -- see the
# MARKET_QUOTES / MARKET_FEEDS note above -- and only market-feeds had a key
# here, so quote snapshots had nowhere to mint a period and never reached the
# spine at all.
SYNTHETIC_TEMPORAL_IDS: Dict[str, str] = {
    "sec": f"{IDENTIFIER_BASE}temporal/sec/",
    "noaa": f"{IDENTIFIER_BASE}temporal/noaa/",
    "market-feeds": f"{IDENTIFIER_BASE}temporal/market-feeds/",
    "market-quotes": f"{IDENTIFIER_BASE}temporal/market-quotes/",
}

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
PROVENANCE = Namespace(f"{ONTOLOGY_BASE}provenance/")

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
# Order matters for node_mapper and edge_mapper: where one namespace is a
# string prefix of another, the longer must appear first, or startsWith
# matching claims the URI for the shorter one and names the node type after
# the wrong vocabulary. Asserted over every pair in tests/test_namespaces.py
# rather than maintained by eye.
#
# Note what moving the minted terms under one base did NOT do: siblings like
# .../ontology/bls/ and .../ontology/sec/ are not prefixes of EACH OTHER --
# sharing a parent path is not the same relation. It actually removed two real
# prefix pairs that did exist (financial-data.org/ vs its /enrichment/ child,
# and noaa.gov/ vs its /enrichment/ child), so the ordering hazard here is
# smaller now, not larger. The invariant test stands either way.
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
    (str(BLS_COMMON), "bls_common"),
    (str(BLS_ENRICHMENT), "bls_enrichment"),
    (str(SEC_FILINGS), "filings"),
    (str(SEC_ADMIN), "sec_admin"),
    (str(SEC_LIT), "sec_lit"),
    (str(SEC_SUSP), "sec_susp"),
    (str(SEC_COMMON), "sec_common"),
    (str(SEC_ENRICHMENT), "sec_enrichment"),
    (str(MARKET_ENRICHMENT), "market_enrichment"),
    # market-feeds-options/ before market-feeds/: the latter is a string
    # prefix of the former, and node_mapper matches by startswith down this
    # list, so the shorter one would claim every options URI and name the
    # node type after the wrong vocabulary. Pinned by
    # test_longer_namespaces_precede_the_shorter_ones_they_extend.
    (str(MARKET_FEEDS_OPTIONS), "market_feeds_options"),
    (str(MARKET_FEEDS), "market_feeds"),
    (str(MARKET_QUOTES), "market_quotes"),
    (str(CAP), "cap"),
    (str(WEATHER), "weather"),
    (str(ALERT), "alert"),
    (str(NOAA_ENRICHMENT), "noaa_enrichment"),
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