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
# They now resolve to a domain we control, split so that a reader can tell a
# term from a thing by looking at the URI:
#
#   https://jefflevesque.com/ontology/{source}/   classes and properties
#   https://jefflevesque.com/id/{source}/         individuals
#
# Node types come from rdf:type objects and edges from predicates, both of
# which are terms, so the constants below are what node_mapper and edge_mapper
# need. Code that asks which source an ENTITY belongs to needs the id/ side
# instead — see identifier_namespace() further down, and SOURCE_IDENTIFIERS in
# rdf_utils.
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

# Only the filings feed is collected. sec-administrative-proceedings,
# sec-litigation and sec-trading-suspensions were removed with the linker paths
# that keyed on them: upstream publishes no administrative-proceedings or
# trading-suspensions feed at all, and feed=litigation was last written 787 days
# ago. Code keyed on a source nobody collects cannot be distinguished from
# working code by any test, because both produce nothing — which is how two of
# the defects on this branch stayed hidden.
SEC_COMMON = Namespace(f"{SOURCE_BASE}sec-common/")
SEC_FILINGS = Namespace(f"{SOURCE_BASE}sec-filings/")

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
MARKET_QUOTES = Namespace(f"{SOURCE_BASE}market-quotes/")

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

WEATHER = Namespace(f"{SOURCE_BASE}weather/")
CAP = Namespace(f"{SOURCE_BASE}cap-model/")

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
# themselves: id/cpi/February is the month, ontology/cpi/Month is its class.
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
