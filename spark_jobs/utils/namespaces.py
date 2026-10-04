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
#
# The archive still holds the flat forms; see LEGACY_VOCABULARIES below.
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
#
# It was 'bls-common', hyphenated to stay clear of ontology/bls/, which
# BLS_ENRICHMENT held. Enrichment has moved to ontology/bls/enrichment/, so the
# slash is free and the hyphen no longer buys anything.
BLS_COMMON = Namespace(f"{SOURCE_BASE}bls/common/")

# ============================================
# SEC data namespaces
# ============================================
# Were 'sec-common' and 'sec-filings', hyphenated for the same reason BLS_COMMON
# was, and nested now for the same reason it is.

# Only the filings feed and the companyfacts snapshot are collected.
# sec-administrative-proceedings, sec-litigation and sec-trading-suspensions were removed with the linker paths
# that keyed on them: upstream publishes no administrative-proceedings or
# trading-suspensions feed at all, and feed=litigation was last written 787 days
# ago. Code keyed on a source nobody collects cannot be distinguished from
# working code by any test, because both produce nothing — which is how two of
# the defects on this branch stayed hidden.
SEC_COMMON = Namespace(f"{SOURCE_BASE}sec/common/")
SEC_FILINGS = Namespace(f"{SOURCE_BASE}sec/filings/")
# The companyfacts snapshot: each company's latest XBRL numbers. Never flat, so
# it has no legacy form.
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
# LEGACY VOCABULARIES — the flat forms still in the archive
# ============================================
# What the mappers emitted before they nested each vocabulary under its source.
# Nothing rewrites the archive, so every object written before that deploy still
# carries these, and they have to stay readable for as long as those objects do
# -- which is forever: published runs cannot be backfilled, so a past run is
# only reproducible while its inputs still parse.
#
# BLS is why this is a table rather than a cutover date. Its feeds write one
# object per observation year and append by streaming stored row groups through
# untouched, so a deploy does not rewrite the rows already in the file. A single
# 2026 object ends up holding flat rows and nested rows interleaved, split
# unevenly by which series happened to be restated. No date separates them.
#
# Each legacy namespace registers beside its current form at the SAME prefix and
# the SAME ontology-source slot, so a node reads identically whichever form it
# arrived in -- cpi:Index is cpi_Index either way. That is what lets one run
# read flat, nested and mixed input without knowing which it has.
#
# Read-only history, not a second live vocabulary: after the deploy nothing
# mints these again. They can be dropped if the archive is ever migrated, and
# doing so costs one contract digest.
LEGACY_VOCABULARIES = {
    str(CPI): f"{SOURCE_BASE}cpi/",
    str(PPI): f"{SOURCE_BASE}ppi/",
    str(ECI): f"{SOURCE_BASE}eci/",
    str(EMPSIT): f"{SOURCE_BASE}empsit/",
    str(JOLTS): f"{SOURCE_BASE}jolts/",
    str(LAUS): f"{SOURCE_BASE}laus/",
    str(METRO): f"{SOURCE_BASE}metro/",
    str(REALER): f"{SOURCE_BASE}realer/",
    str(WKYENG): f"{SOURCE_BASE}wkyeng/",
    str(XIMPIM): f"{SOURCE_BASE}ximpim/",
    str(BLS_COMMON): f"{SOURCE_BASE}bls-common/",
    str(SEC_COMMON): f"{SOURCE_BASE}sec-common/",
    str(SEC_FILINGS): f"{SOURCE_BASE}sec-filings/",
    str(MARKET_QUOTES): f"{SOURCE_BASE}market-quotes/",
    str(WEATHER): f"{SOURCE_BASE}weather/",
    str(CAP): f"{SOURCE_BASE}cap-model/",
}


def with_legacy(
    pairs: "tuple[tuple[str, str], ...]",
) -> "tuple[tuple[str, str], ...]":
    """Each (namespace, prefix), followed by the legacy form where there is one.

    The legacy entry takes the SAME prefix, which is the point: both forms name
    one vocabulary, so both must produce one node type name. Neither form is a
    string prefix of the other, so their relative order does not matter.
    """
    expanded = []
    for namespace, prefix in pairs:
        expanded.append((namespace, prefix))
        legacy = LEGACY_VOCABULARIES.get(namespace)
        if legacy:
            expanded.append((legacy, prefix))
    return tuple(expanded)

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


# Where a legacy vocabulary's INDIVIDUALS sat, where that is not
# identifier_namespace() of its legacy term namespace.
#
# BLS is the only case, and it is an inconsistency in the flat layout rather
# than an oversight here: the shared terms were at ontology/bls-common/ while
# the individuals they type went under a bare id/bls/ -- <id/bls/June> a
# laus:Month. identifier_namespace() derives id/bls-common/, which nothing has
# ever minted into.
#
# TWO prefixes, because that shared space is not flat. Months and years sit
# directly under id/bls/, and the states sit a segment deeper at
# id/bls/state/ -- 2,572 individuals in the 2026 metro feed, including the
# subjects of its own measurements, with hasParentRegion alone pointing at them
# 19,392 times. Both moved under id/bls/common/ when the shared space did, the
# state segment riding along as a suffix.
#
# Ordered narrowest first, and the pair is what the ordering is FOR:
#
#   id/bls/state/  must be tried before id/bls/, or a bare id/bls/ rewrite
#                  would claim it and produce id/bls/common/state/... only by
#                  accident of the anchor refusing it -- see below.
#   id/bls/        is a string prefix of all ten id/bls/<dataset>/, so it is
#                  ANCHORED to one trailing segment. Unanchored it would turn
#                  id/bls/cpi/Index into id/bls/common/cpi/Index.
#
# id/bls/state/ needs no anchor: no BLS dataset is named 'state' (cpi, ppi,
# eci, empsit, jolts, laus, metro, realer, wkyeng, ximpim), so a plain prefix
# match cannot reach an already-nested URI. That is also why the anchor alone
# could never have covered it -- id/bls/state/AL and id/bls/cpi/February are
# both two segments, and only the vocabulary knows which one is a dataset.
LEGACY_IDENTIFIER_OVERRIDES = {
    f"{SOURCE_BASE}bls-common/": (
        (f"{IDENTIFIER_BASE}bls/state/", f"{IDENTIFIER_BASE}bls/common/state/"),
        (f"{IDENTIFIER_BASE}bls/", f"{IDENTIFIER_BASE}bls/common/"),
    ),
}


def legacy_identifier_pairs(legacy_term_namespace, current_term_namespace):
    """(legacy id prefix, current id prefix) for one vocabulary's individuals.

    One pair for a vocabulary whose individuals sit where identifier_namespace()
    says they do, and whatever LEGACY_IDENTIFIER_OVERRIDES declares for one that
    does not.
    """
    override = LEGACY_IDENTIFIER_OVERRIDES.get(legacy_term_namespace)
    if override:
        return override
    return (
        (
            identifier_namespace(legacy_term_namespace),
            identifier_namespace(current_term_namespace),
        ),
    )


def legacy_rewrites(namespaces=None):
    """(legacy prefix, current prefix, anchored) for these vocabularies.

    The terms, then the individuals, because the mappers moved both.
    ``namespaces`` defaults to every vocabulary with a legacy form; the loader
    passes one source's, since it canonicalises per path.

    ``anchored`` marks a rewrite whose target sits INSIDE its source, where a
    plain prefix match would also claim URIs that are already correct --
    id/bls/ -> id/bls/common/ is the one. It is derived rather than declared,
    so a prefix that grows to contain its own target cannot quietly stop being
    anchored.

    Longest prefix first, the same rule _owned_namespaces uses. The anchor and
    the ordering are independent guards and both are load bearing:
    id/bls/state/ is refused by the anchor and so needs the ordering, and
    id/bls/cpi/ is refused by the ordering being longest-first and would still
    need the anchor if a shorter legacy prefix were ever added above it.
    """
    selected = LEGACY_VOCABULARIES if namespaces is None else namespaces
    rewrites = []
    for namespace in selected:
        legacy = LEGACY_VOCABULARIES.get(namespace)
        if not legacy:
            continue
        pairs = ((legacy, namespace),) + tuple(
            legacy_identifier_pairs(legacy, namespace)
        )
        for old, new in pairs:
            rewrites.append((old, new, new.startswith(old)))
    return sorted(rewrites, key=lambda rewrite: -len(rewrite[0]))


def canonical_uri(uri: str) -> str:
    """A URI under a legacy vocabulary, in the spelling the mappers emit now.

    The driver-side twin of canonicalization._legacy_rewrite, for the tools that
    read source RDF without going through the loader.
    bin/check_vocabulary_drift.py is the one that needs it: it parses the
    fixtures directly, and comparing their flat URIs against nested constants
    would report every vocabulary as uncovered and quietly switch the guard off.
    """
    for old, new, anchored in legacy_rewrites():
        if not uri.startswith(old):
            continue
        if anchored and "/" in uri[len(old):]:
            continue
        return f"{new}{uri[len(old):]}"
    return uri


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
#
# Nothing reads these from storage -- enrichment terms are minted fresh each
# run -- so unlike the source vocabularies they need no legacy form.
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
