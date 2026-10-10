"""Namespace constants: every vocabulary the pipeline reads or mints.

Nothing here imports from spark_jobs. That is what lets the source specs in
spark_jobs/sources/ use these constants while rdf_utils builds its namespace
tables from those specs. rdf_utils re-exports every name here, so an import
from rdf_utils keeps working.
"""

from rdflib import Namespace

# ============================================
# SOURCE VOCABULARIES — terms the upstream mappers mint
# ============================================
# These used to sit on the upstream providers' own domains. None of those
# organizations defined these terms: the upstreams publish tabular and
# document formats, not RDF, and every class and property below was invented
# by the upstream RML mappers.
#
# They now resolve to a domain we control, split two ways. A reader can tell a
# term from a thing, and a term from which source, by looking at the URI:
#
#   https://jefflevesque.com/ontology/{source}/{group}/   classes and properties
#   https://jefflevesque.com/id/{source}/{group}/         individuals
#
# Two segments under the base, not one. Flat, the ten BLS dataset vocabularies
# sat at ontology/cpi/ … ontology/ximpim/ and claimed generic names in a space
# meant to hold every source -- plenty of agencies publish a CPI -- while
# nothing in the URI recorded which source a term came from.
#
# Node types come from rdf:type objects and edges from predicates, both of
# which are terms, so the constants below are what node_mapper and edge_mapper
# need. Code that asks which source an ENTITY belongs to needs the id/ side
# instead — see identifier_namespace() further down, and SOURCE_IDENTIFIERS in
# rdf_utils.
SOURCE_BASE = "https://jefflevesque.com/ontology/"

CPI = Namespace(f"{SOURCE_BASE}bls/cpi/")
PPI = Namespace(f"{SOURCE_BASE}bls/ppi/")
ECI = Namespace(f"{SOURCE_BASE}bls/eci/")
EMPSIT = Namespace(f"{SOURCE_BASE}bls/empsit/")
JOLTS = Namespace(f"{SOURCE_BASE}bls/jolts/")
LAUS = Namespace(f"{SOURCE_BASE}bls/laus/")
METRO = Namespace(f"{SOURCE_BASE}bls/metro/")
REALER = Namespace(f"{SOURCE_BASE}bls/realer/")
WKYENG = Namespace(f"{SOURCE_BASE}bls/wkyeng/")
XIMPIM = Namespace(f"{SOURCE_BASE}bls/ximpim/")

# Shared BLS classes (Month, Year, Industry, Region, ...) that the hand-
# authored table schemas declare once for every category.
BLS_COMMON = Namespace(f"{SOURCE_BASE}bls/common/")

# ============================================
# SEC data namespaces
# ============================================

# Only the filings feed and the companyfacts snapshot are collected.
# sec-administrative-proceedings, sec-litigation and sec-trading-suspensions were removed with the linker paths
# that keyed on them: upstream publishes no administrative-proceedings or
# trading-suspensions feed at all, and feed=litigation was last written 787 days
# ago. Code keyed on a source nobody collects cannot be distinguished from
# working code by any test, because both produce nothing — which is how two of
# the defects on this branch stayed hidden.
SEC_COMMON = Namespace(f"{SOURCE_BASE}sec/common/")
SEC_FILINGS = Namespace(f"{SOURCE_BASE}sec/filings/")
# The companyfacts snapshot: each company's latest XBRL numbers.
SEC_COMPANYFACTS = Namespace(f"{SOURCE_BASE}sec/companyfacts/")

# ============================================
# Market data namespace
# ============================================
# ONE market vocabulary. Equity and option quotes arrive together in the
# upstream intraday snapshots -- EquitySnapshot / OptionSnapshot, flat, with
# captureTime, askPrice, strikePrice, delta -- and that is the only market RDF
# published anywhere.
#
# There used to be a second, MARKET_FEEDS (PriceObservation / OptionContract,
# observedAt / expirationDate) from an HTML feed, plus a MARKET_FEEDS_OPTIONS
# beside it. No such data exists in either bucket, so both are gone along with
# the code that read them. Their presence was expensive: one constant asked to
# name both models is what made market enrichment silently produce nothing in
# the first place, and every market change since had to reason about which of
# two vocabularies it meant.
MARKET_QUOTES = Namespace(f"{SOURCE_BASE}market/quotes/")

# ============================================
# NOAA WEATHER DATA NAMESPACES
# ============================================
# Verified against 275 live alerts from api.weather.gov: NWS publishes
# wx:Alert plus ~30 lowercase properties (affectedZones, areaDesc, geocode).
# Not one of the ~29 terms the upstream mapper emits is among them, and their
# live JSON-LD context declares api.weather.gov/ontology# as @vocab — so minting
# our terms there put our inventions inside a live publisher's vocabulary.
#
# Likewise CAP: OASIS registered urn:oasis:names:tc:emergency:cap:1.2 as an
# XML namespace naming lowercase ELEMENTS. There is no OASIS CAP RDF
# vocabulary, so cap:hasAreaDescription and cap:AlertMessage are our model OF
# CAP rather than CAP itself.

WEATHER = Namespace(f"{SOURCE_BASE}noaa/weather/")
CAP = Namespace(f"{SOURCE_BASE}noaa/cap-model/")

# Alert instances — real identifiers for real NWS alerts. Genuinely theirs,
# unlike the vocabulary that used to sit alongside them, so left alone.
ALERT = Namespace("https://api.weather.gov/alerts/")

# GeoSPARQL namespace (used by CAP Area alignment). Really OGC's.
GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")

# Atom feed namespace
ATOM = Namespace("http://www.w3.org/2005/Atom/")

# ============================================
# SOURCE IDENTIFIERS — individuals the upstream mappers mint
# ============================================
# The namespaces above name classes and properties. These name the things
# themselves: id/bls/cpi/February is the month, ontology/bls/cpi/Month is its
# class.
#
# The header of rdf_utils used to say only the TERM namespaces were needed
# there, because "the id/ namespaces appear only as subjects and objects". That
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

    ``https://jefflevesque.com/ontology/bls/cpi/`` ->
    ``https://jefflevesque.com/id/bls/cpi/``

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

# One level down, at <source>/enrichment/, rather than at <source>/.
#
# The upstream mappers nest each source vocabulary under its source, so the
# terms they emit are ontology/bls/cpi/, ontology/sec/filings/ and so on. Held
# at ontology/bls/, an enrichment namespace is a string prefix of every one of
# its source's vocabularies, and classify_edge_origin() decides observed vs
# inferred with startswith -- so every fact a source reported would be labelled
# something this pipeline made up. The graph still builds and the tests that do
# not touch namespaces stay green; graph_schema.json just lies about the origin
# of every edge.
#
# Moving enrichment down leaves ontology/<source>/ holding nothing but its own
# children, which removes the prefix relationship rather than ordering around
# it. test_no_enrichment_namespace_is_a_prefix_of_a_source_vocabulary pins it.
BLS_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}bls/enrichment/")
SEC_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}sec/enrichment/")
NOAA_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}noaa/enrichment/")
MARKET_ENRICHMENT = Namespace(f"{ONTOLOGY_BASE}market/enrichment/")
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
