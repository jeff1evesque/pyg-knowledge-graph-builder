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
    SOURCE_VOCABULARIES,
    PUBLISHER_VOCABULARIES,
    SYNTHETIC_TEMPORAL_IDS,
    IDENTIFIER_BASE,
)

# Every namespace whose terms this project invents, in either repo.
#
# The source vocabularies (cpi:, jolts:, filings:, cap:, ...) used to be
# excluded here on the grounds that they were the publishers' own terms. That
# was wrong, and checking it is what corrected it: BLS publishes CSV and the
# SEC publishes XML, so nobody there defined cpi:Index or filings:Form4. The
# scrapers did. NWS was the sharpest case -- its live JSON-LD context really
# does declare https://api.weather.gov/ontology# as @vocab, but it holds
# wx:Alert and ~30 lowercase properties, and not one of the ~29 terms the
# scraper emitted was among them.
#
# What survived the check is in PUBLISHER_VOCABULARIES: alert: identifiers,
# GeoSPARQL, Atom.
MINTED = {
    "BLS_ENRICHMENT": str(BLS_ENRICHMENT),
    "SEC_ENRICHMENT": str(SEC_ENRICHMENT),
    "NOAA_ENRICHMENT": str(NOAA_ENRICHMENT),
    "MARKET_ENRICHMENT": str(MARKET_ENRICHMENT),
    "UNIFIED": str(UNIFIED),
    "SOURCE_TEMPORAL": str(SOURCE_TEMPORAL),
    "PROVENANCE": str(PROVENANCE),
    **{
        f"SOURCE[{namespace.rsplit('/', 2)[-2]}]": namespace
        for namespace in SOURCE_VOCABULARIES
    },
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
    """The ones that really are theirs must stay where they are.

    Re-homing our own inventions is the fix; re-homing someone else's real
    vocabulary would be the same mistake pointing the other way, and it would
    break federation -- the whole reason to reuse a published URI is that it
    denotes the same thing everywhere.
    """
    assert PUBLISHER_VOCABULARIES, "no publisher vocabularies left at all"
    for namespace in PUBLISHER_VOCABULARIES:
        assert not namespace.startswith(ONTOLOGY_BASE), (
            f"{namespace} was moved under our base, but its publisher really "
            f"did define it and we are not its authority"
        )


def test_every_registered_namespace_is_ours_or_a_real_publisher_vocabulary():
    """No third category. Anything else in the table is unaccounted for.

    A namespace that is neither minted by us nor a vocabulary someone else
    genuinely published is exactly the state this migration removed, and the
    only way it comes back is by being added without anyone deciding which
    kind it is.
    """
    accounted = set(MINTED.values()) | set(PUBLISHER_VOCABULARIES)
    unaccounted = [
        ns for ns, _prefix in NAMESPACE_PREFIXES
        if ns not in accounted and not ns.startswith("http://www.w3.org/")
    ]
    assert not unaccounted, (
        f"{unaccounted} are registered but classified as neither ours nor a "
        f"publisher's; decide which and add them to the matching list"
    )


@pytest.mark.parametrize("source,namespace", sorted(SYNTHETIC_TEMPORAL_IDS.items()))
def test_synthetic_temporal_individuals_are_not_minted_on_a_publishers_domain(
        source, namespace):
    """The defect that survived the first pass, pinned so it cannot return.

    temporal_unifier mints one individual per period per source -- November,
    2024 -- on every build. Two of the three were still being minted at
    https://www.sec.gov/temporal/ and https://www.noaa.gov/temporal/ after the
    source vocabularies had moved: nobody at those organizations minted
    sec.gov/temporal/November, this pipeline did.

    They are easy to miss because they are string literals passed as an
    argument rather than namespace constants, so nothing that scans the
    namespace table sees them. Hence this test reads the mapping the unifier
    actually uses.
    """
    for domain in PUBLISHER_DOMAINS:
        assert domain not in namespace, (
            f"the {source} synthetic temporal individuals are minted at "
            f"{namespace!r}, which claims {domain} as the authority for a URI "
            f"this pipeline invents on every build"
        )


@pytest.mark.parametrize("source,namespace", sorted(SYNTHETIC_TEMPORAL_IDS.items()))
def test_synthetic_temporal_individuals_live_under_the_identifier_base(
        source, namespace):
    """They are things, not terms.

    Their TYPES (SourceMonth, SourceYear, SourceQuarter) stay under
    SOURCE_TEMPORAL. Putting the individuals alongside their own types would
    re-conflate exactly what the /ontology/ and /id/ split exists to separate.
    """
    assert namespace.startswith(IDENTIFIER_BASE), namespace
    assert namespace.endswith("/"), namespace
    assert not namespace.startswith(str(SOURCE_TEMPORAL)), (
        f"{namespace} sits under the TYPE namespace; individuals belong under "
        f"{IDENTIFIER_BASE}"
    )


def test_every_synthetic_temporal_source_is_distinct():
    """Two sources naming the same period must not pre-emptively collapse.

    Merging them here would do the unifier's job before it runs and erase
    which source observed which period.
    """
    assert len(set(SYNTHETIC_TEMPORAL_IDS.values())) == len(SYNTHETIC_TEMPORAL_IDS)


def test_no_module_hardcodes_a_publisher_namespace():
    """No URI literal anywhere in spark_jobs/ may name a publisher's domain.

    Every namespace this project uses has a constant in rdf_utils. A literal
    spelled out at the point of use is a copy that does not move when the
    constant does -- and because a stale namespace matches nothing rather than
    raising, the failure is silent: enrichment produces zero triples and the
    suite stays green.

    Four separate places were found this way, all of them already broken by
    the migration before this test existed:

      * temporal_unifier minted synthetic temporal individuals at
        sec.gov/temporal/ and noaa.gov/temporal/ on every build;
      * its SEC_DATE_PREDS carried hasattr()/else fallbacks spelling the old
        sec.gov predicates -- dead code that would have reintroduced them;
      * sec_linker and sec/measurements re-declared the shared SEC namespace
        as "http://www.sec.gov#";
      * noaa/patterns keyed its severity, urgency and certainty ordinals on 21
        CAP value URIs, so escalation detection silently stopped ranking.

    Publishers' real vocabularies are exempt: reusing those at their real URIs
    is correct, and they are listed in PUBLISHER_VOCABULARIES.
    """
    import re
    from pathlib import Path

    import spark_jobs

    allowed = set(PUBLISHER_VOCABULARIES)
    package = Path(spark_jobs.__file__).parent

    offenders = {}
    for module in sorted(package.rglob("*.py")):
        literals = re.findall(r'"(https?://[^"]+)"', module.read_text())
        bad = [
            literal for literal in literals
            if any(domain in literal for domain in PUBLISHER_DOMAINS)
            and not any(literal.startswith(ok) for ok in allowed)
        ]
        if bad:
            offenders[str(module.relative_to(package))] = sorted(set(bad))

    assert not offenders, (
        "hardcoded publisher-domain URIs found; use the rdf_utils constant so "
        f"they move with the vocabulary:\n{offenders}"
    )


def test_no_source_vocabulary_sits_on_a_publishers_domain():
    """The defect this migration fixed, stated over the source terms.

    cpi:Index under bls.gov says BLS defined it. BLS publishes CSV.
    """
    for namespace in SOURCE_VOCABULARIES:
        assert namespace.startswith(ONTOLOGY_BASE), namespace
        for domain in PUBLISHER_DOMAINS:
            assert domain not in namespace, (
                f"{namespace} claims {domain} as the authority for a term the "
                f"scrapers invented"
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
        for local in ("SourceMonth", "SourceYear", "SourceQuarter", "SourceDay")
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
