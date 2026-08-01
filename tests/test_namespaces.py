"""Invariants for the namespace table (spark_jobs/utils/rdf_utils.py).

Pure Python, no SparkSession — these are structural facts about a table, and
they are the kind that fail silently. Nothing in the pipeline raises when a
namespace moves: node names just stop matching, prefixes just collide, URIs
just claim the wrong authority. Every assertion here covers something that
would otherwise be found by reading a graph and noticing it looks wrong.

The specific defect that prompted the file: six namespaces holding terms this
project invents were published under other organizations' domains
(bls.gov/enrichment/, sec.gov/enrichment/, noaa.gov/enrichment/,
financial-data.org/enrichment/) and under example.org, which RFC 2606 reserves
for documentation. A URI under bls.gov asserts BLS defined that term. None of
them did.
"""
import pytest

from spark_jobs.utils import rdf_utils
from spark_jobs.utils.rdf_utils import (
    ENRICHMENT_NAMESPACES,
    NAMESPACE_PREFIXES,
    ONTOLOGY_BASE,
    PIPELINE_NODE_TYPE_PREFIXES,
    PROVENANCE,
    SOURCE_TEMPORAL,
    UNIFIED,
    BLS_ENRICHMENT,
    MARKET_ENRICHMENT,
    NOAA_ENRICHMENT,
    SEC_ENRICHMENT,
)

# Every namespace whose terms this project invents. Publishers' vocabularies
# (cpi:, jolts:, cap:, sec.gov/filings#, ...) are deliberately absent: those
# really are their terms and belong on their domains.
MINTED = {
    "BLS_ENRICHMENT": str(BLS_ENRICHMENT),
    "SEC_ENRICHMENT": str(SEC_ENRICHMENT),
    "NOAA_ENRICHMENT": str(NOAA_ENRICHMENT),
    "MARKET_ENRICHMENT": str(MARKET_ENRICHMENT),
    "UNIFIED": str(UNIFIED),
    "SOURCE_TEMPORAL": str(SOURCE_TEMPORAL),
    "PROVENANCE": str(PROVENANCE),
}

# Domains belonging to the data publishers. A term we mint under any of these
# claims an authority we do not have.
PUBLISHER_DOMAINS = (
    "bls.gov",
    "sec.gov",
    "noaa.gov",
    "weather.gov",
    "financial-data.org",
    "market.example",
    "oasis-open.org",
)

# RFC 2606 / RFC 6761 reserved names. Reserved precisely so nobody owns them,
# which is what makes them unusable for terms that need a stable authority.
RESERVED_DOMAINS = (
    "example.org",
    "example.com",
    "example.net",
    "example.edu",
    "localhost",
    "invalid",
)


# ======================================================================
# The defect itself — where minted terms are allowed to live
# ======================================================================

@pytest.mark.parametrize("name,namespace", sorted(MINTED.items()))
def test_minted_namespaces_live_under_our_base(name, namespace):
    assert namespace.startswith(ONTOLOGY_BASE), (
        f"{name} is {namespace!r}, outside {ONTOLOGY_BASE!r}. Terms this "
        f"project defines must resolve to a domain this project controls."
    )


@pytest.mark.parametrize("name,namespace", sorted(MINTED.items()))
def test_no_minted_term_claims_a_publishers_domain(name, namespace):
    """The actual bug, as a test rather than a promise.

    bls_enrichment:RateMeasurement under bls.gov states that BLS defined it.
    Consumers are entitled to believe a URI, and federating this graph with
    real BLS-published RDF would merge our invention into their vocabulary.
    """
    for domain in PUBLISHER_DOMAINS:
        assert domain not in namespace, (
            f"{name} is {namespace!r}, which claims {domain} as the authority "
            f"for a term we invented"
        )


@pytest.mark.parametrize("name,namespace", sorted(MINTED.items()))
def test_no_minted_term_uses_a_reserved_domain(name, namespace):
    for domain in RESERVED_DOMAINS:
        assert domain not in namespace, (
            f"{name} is {namespace!r}; {domain} is reserved for documentation, "
            f"so it is nobody's and any other project using it collides with us"
        )


def test_the_base_is_a_single_point_of_change():
    """One constant, so re-homing the vocabulary stays a one-line edit."""
    assert ONTOLOGY_BASE.startswith("https://")
    assert ONTOLOGY_BASE.endswith("/")
    for namespace in MINTED.values():
        assert namespace.endswith("/"), namespace


def test_publisher_vocabularies_were_left_alone():
    """Out of scope, and must stay out: those are genuinely their terms."""
    publisher_namespaces = [
        ns for ns, _prefix in NAMESPACE_PREFIXES
        if ns not in set(MINTED.values())
        and not ns.startswith("http://www.w3.org/")
    ]
    assert publisher_namespaces, "no publisher namespaces left in the table"
    for ns in publisher_namespaces:
        assert not ns.startswith(ONTOLOGY_BASE), (
            f"{ns} was moved under our base, but it is a publisher's own "
            f"vocabulary and we are not its authority either"
        )


# ======================================================================
# Table invariants — silent-failure territory
# ======================================================================

def test_longer_namespaces_precede_the_shorter_ones_they_extend():
    """Ordering as an invariant over every pair, not a spot check.

    node_mapper matches by startsWith down this list in order, so if A is a
    string prefix of B and A comes first, every B URI is claimed by A and the
    node type is named after the wrong vocabulary. Nothing raises; the graph
    just quietly merges two vocabularies.
    """
    order = {ns: i for i, (ns, _prefix) in enumerate(NAMESPACE_PREFIXES)}
    violations = []
    for shorter, _ in NAMESPACE_PREFIXES:
        for longer, _ in NAMESPACE_PREFIXES:
            if longer == shorter or not longer.startswith(shorter):
                continue
            if order[shorter] < order[longer]:
                violations.append((shorter, longer))

    assert not violations, (
        "namespace(s) ordered so a shorter prefix shadows a longer one:\n"
        + "\n".join(
            f"  {s!r} (index {order[s]}) precedes {ln!r} (index {order[ln]})"
            for s, ln in violations
        )
    )


def test_prefixes_are_unique():
    """Two namespaces sharing a prefix silently merge unrelated node types."""
    seen = {}
    for namespace, prefix in NAMESPACE_PREFIXES:
        assert prefix not in seen, (
            f"prefix {prefix!r} is claimed by both {seen.get(prefix)!r} and "
            f"{namespace!r}; their node types would collapse into one name"
        )
        seen[prefix] = namespace


def test_namespaces_are_unique():
    namespaces = [ns for ns, _prefix in NAMESPACE_PREFIXES]
    assert len(namespaces) == len(set(namespaces))


def test_every_minted_namespace_is_registered_except_provenance():
    """Provenance is deliberately absent — and that has to stay deliberate.

    Nothing in it is ever an rdf:type, so it can never name a node type, and
    registering it would shift ONTOLOGY_NAMESPACE_INDICES and change the
    encoding contract digest to describe URIs no encoder sees.
    """
    registered = {ns for ns, _prefix in NAMESPACE_PREFIXES}
    for name, namespace in MINTED.items():
        if name == "PROVENANCE":
            assert namespace not in registered
        else:
            assert namespace in registered, f"{name} is not in the table"


# ======================================================================
# Hardcoded type names must be derivable from the table
# ======================================================================
#
# _CANONICAL_TYPE_PRIORITY holds literal strings ("temporal_SourceMonth").
# They implement the pinning that stops one month sharding across every
# namespace that names it — and if a namespace prefix is ever renamed they
# match nothing, silently, and the sharding comes back. Nothing else fails.

def _naming_rule(namespace: str, local: str) -> str:
    """Reproduce node_mapper's URI -> PyG name rule for a registered namespace."""
    for ns, prefix in NAMESPACE_PREFIXES:
        if namespace == ns:
            return f"{prefix}_{local}"
    raise AssertionError(f"{namespace} is not registered")


def test_canonical_type_names_are_producible_by_the_naming_rule():
    from spark_jobs.pyg_builder.node_mapper import _CANONICAL_TYPE_PRIORITY

    producible = {
        f"{prefix}_" for _ns, prefix in NAMESPACE_PREFIXES
    }
    for name in _CANONICAL_TYPE_PRIORITY:
        assert any(name.startswith(p) for p in producible), (
            f"{name!r} starts with no registered prefix, so the naming rule "
            f"can never produce it and the canonical-type pinning it "
            f"implements is dead code"
        )


def test_the_temporal_type_names_still_match_their_namespace():
    """The pinned names and SOURCE_TEMPORAL must not drift apart."""
    from spark_jobs.pyg_builder.node_mapper import _CANONICAL_TYPE_PRIORITY

    expected = {
        _naming_rule(str(SOURCE_TEMPORAL), local)
        for local in ("SourceMonth", "SourceYear", "SourceQuarter")
    }
    assert set(_CANONICAL_TYPE_PRIORITY) == expected


def test_sector_fragments_match_a_real_node_type_name():
    from spark_jobs.pyg_builder.node_mapper import _SECTOR_TYPE_FRAGMENTS

    # The fragments are matched against node type NAMES, so each must be a
    # substring of something the naming rule can produce. EconomicSector is
    # minted by the BLS enrichment vocabulary.
    candidates = {
        _naming_rule(str(BLS_ENRICHMENT), "EconomicSector"),
        _naming_rule(str(BLS_ENRICHMENT), "Sector"),
    }
    for fragment in _SECTOR_TYPE_FRAGMENTS:
        assert any(fragment in name for name in candidates), (
            f"sector fragment {fragment!r} matches no producible node type "
            f"name; the sector filter it drives would silently do nothing"
        )


# ======================================================================
# Edge-origin classification must not be widened by the shared base
# ======================================================================

def test_source_temporal_is_not_an_enrichment_namespace():
    """Sharing a base must not reclassify observed facts as inferred.

    The temporal TYPES are ours, but the measurement->month edges are observed
    source facts. classify_edge_origin() reads ENRICHMENT_NAMESPACES, not the
    base URI — so moving SOURCE_TEMPORAL under the same base as the enrichment
    vocabularies must not sweep it in.
    """
    assert str(SOURCE_TEMPORAL) not in ENRICHMENT_NAMESPACES
    assert str(UNIFIED) not in ENRICHMENT_NAMESPACES
    assert all(ns.startswith(ONTOLOGY_BASE) for ns in ENRICHMENT_NAMESPACES)


def test_pipeline_node_type_prefixes_still_resolve():
    """Derived from the table, so a rename must not empty it out."""
    assert PIPELINE_NODE_TYPE_PREFIXES
    prefixes = {p for _ns, p in NAMESPACE_PREFIXES}
    for entry in PIPELINE_NODE_TYPE_PREFIXES:
        assert entry.endswith("_")
        assert entry[:-1] in prefixes


def test_classify_edge_origin_still_separates_the_three_origins():
    """All three verdicts still reachable after the namespaces moved.

    Unification is keyed on the owl:sameAs PREDICATE with a minted endpoint,
    not on the unified namespace — a unified predicate that is not sameAs is
    an ordinary enrichment edge.
    """
    classify = rdf_utils.classify_edge_origin
    assert classify(f"{BLS_ENRICHMENT}correlatesWith") == (
        rdf_utils.ORIGIN_ENRICHMENT
    )
    assert classify(f"{UNIFIED}hasMonth") == rdf_utils.ORIGIN_ENRICHMENT
    assert classify(
        rdf_utils.OWL_SAME_AS,
        src_type="unified_UnifiedMonth",
        dst_type="cpi_Month",
    ) == rdf_utils.ORIGIN_UNIFICATION
    assert classify("https://www.bls.gov/cpi/hasMonth") == rdf_utils.ORIGIN_RAW


def test_a_source_temporal_endpoint_is_not_called_pipeline_inferred():
    """The hazard the shared base actually creates.

    The temporal TYPES are ours, but a measurement->month edge is an observed
    source fact. Classification reads PIPELINE_NODE_TYPE_PREFIXES, which is
    built from ENRICHMENT_NAMESPACES + UNIFIED — temporal is in neither, and
    sharing a base URI with them must not change that.
    """
    assert rdf_utils.classify_edge_origin(
        "https://www.bls.gov/cpi/hasMonth",
        src_type="cpi_Index",
        dst_type="temporal_SourceMonth",
    ) == rdf_utils.ORIGIN_RAW
