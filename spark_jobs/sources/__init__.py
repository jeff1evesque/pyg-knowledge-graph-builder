"""The registered data sources, the per-source tables built from them, and
which sources a run's paths pick.

Each source is declared once, as a SourceSpec in its own module here. Tables
that used to be written out source by source are built from every registered
spec: the namespace table and its source and enrichment subsets, the synthetic
period prefixes, the ontology mapping rows and the edge relation fragments.
Every input path belongs to exactly one registered source, and the sources a
run's paths name are the ones whose functions the job calls.

This package imports neither pyspark nor rdf_utils. rdf_utils builds its
namespace tables from it, and every module that imports rdf_utils would
otherwise load pyspark too.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from rdflib.namespace import OWL, RDFS

from spark_jobs.sources import bls, market, noaa, sec
from spark_jobs.sources.spec import RELATION_CATEGORIES, SourceSpec
from spark_jobs.utils.namespaces import (
    GEOSPARQL,
    SOURCE_BASE,
    SOURCE_TEMPORAL,
    UNIFIED,
    identifier_namespace,
)

# Registration order is table order: each source's namespaces follow the
# previous source's in NAMESPACE_PREFIXES, and its relation fragments follow the
# previous source's in the lists the edge encoding config records, so a new
# source goes at the end. The order places no ontology-source slot; see
# rdf_utils.ONTOLOGY_NAMESPACE_INDICES.
REGISTERED: Tuple[SourceSpec, ...] = (bls.SPEC, sec.SPEC, market.SPEC, noaa.SPEC)

# Namespaces no single source owns, after every source's own: GeoSPARQL, which
# any source may reuse, the pipeline's unified and temporal vocabularies, and
# OWL and RDFS.
SHARED_NAMESPACES: Tuple[Tuple[str, str], ...] = (
    (str(GEOSPARQL), "geosparql"),
    (str(UNIFIED), "unified"),
    (str(SOURCE_TEMPORAL), "temporal"),
    (str(OWL), "owl"),
    (str(RDFS), "rdfs"),
)

# Relation fragments no single source owns. A category's fragments are these,
# then each source's in registration order.
SHARED_RELATION_FRAGMENTS: Dict[str, Tuple[str, ...]] = {
    "temporal": (
        "precedes", "follows", "hasNext", "hasPrevious", "temporallyRelated",
    ),
    # "Correlation" is a suffix: the cross-source linker names one relation per
    # sector (energySectorCorrelation, ...).
    "correlation": ("correlatesWith", "relatedTo", "Correlation"),
    "causal": ("leadsTo", "impacts", "causes", "affects"),
    "skip": (
        "belongsToSector", "sameAs", "hasParent", "hasChild",
        "equivalentClass", "equivalentProperty", "imports",
        "refersToCompany", "hasRegion", "affectsRegion",
    ),
}


# ======================================================================
# Which source a path belongs to
# ======================================================================

def specs_for_path(
    path: str, specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[SourceSpec, ...]:
    """The sources a path could belong to.

    First by path fragment (source=sec, quotes, /noaa/). Fragments rather than
    whole paths, so a path keeps its source across a bucket or prefix change.
    When no fragment matches, by a whole path segment named after the source,
    which is how the committed fixtures (ntriples/sec.nt, turtle_parquet/bls/cpi)
    are recognised. Whole segments only, never substrings: a bucket called
    secure-data or a directory named marketing holds a source's name but is not
    that source.
    """
    lowered = path.lower()
    by_fragment = tuple(
        spec for spec in specs
        if any(fragment in lowered for fragment in spec.path_fragments)
    )
    if by_fragment:
        return by_fragment

    stems = {
        segment.split(".", 1)[0]
        for segment in lowered.replace("\\", "/").split("/")
    }
    return tuple(spec for spec in specs if spec.name in stems)


def match_path(path: str, specs: Sequence[SourceSpec] = REGISTERED) -> SourceSpec:
    """The one source a path belongs to.

    Raises:
        ValueError: when the path matches no source, or more than one.
    """
    found = specs_for_path(path, specs)
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(
            f"source path matches no registered source: {path}. A path has to "
            f"carry one source's path fragment, or a folder or file named after "
            f"it. Registered sources: {', '.join(spec.name for spec in specs)}."
        )
    raise ValueError(
        f"source path matches more than one source "
        f"({', '.join(spec.name for spec in found)}): {path}"
    )


def pick(
    paths: Sequence[str], specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[SourceSpec, ...]:
    """The sources a run's paths name, once each, in registration order."""
    matched = {match_path(path, specs) for path in paths}
    return tuple(spec for spec in specs if spec in matched)


def entity_prefixes(namespaces: Sequence[str]) -> List[str]:
    """The URI prefixes an entity under these namespaces starts with.

    Each namespace, and its id/ namespace where it has one. Entities are
    individuals, which live under id/, so the term namespace alone matches none
    of them. A publisher's own vocabulary, such as NOAA's alert identifiers, has
    no id/ namespace and is matched as it is.
    """
    prefixes: List[str] = []
    for namespace in namespaces:
        prefixes.append(namespace)
        if namespace.startswith(SOURCE_BASE):
            prefixes.append(identifier_namespace(namespace))
    return prefixes


# ======================================================================
# Which source a term belongs to
# ======================================================================

def _owned_namespaces(
    specs: Sequence[SourceSpec],
) -> Tuple[Tuple[str, str], ...]:
    """(namespace, source name) for every namespace a source owns, longest first.

    Longest first so a namespace nested inside another resolves to its own
    source rather than to the one it sits under. A spec's enrichment namespace
    is one of its own (tests/test_source_registry.py), so it needs no entry of
    its own. SHARED_NAMESPACES are left out: no source owns them.
    """
    table = [
        (namespace, spec.name)
        for spec in specs
        for namespace, _prefix in spec.namespaces
    ]
    return tuple(sorted(table, key=lambda pair: (-len(pair[0]), pair[0])))


def _source_of(uri: str, table: Sequence[Tuple[str, str]]) -> Optional[str]:
    for namespace, name in table:
        if uri.startswith(namespace):
            return name
    return None


def source_of_type_uri(
    uri: str, specs: Sequence[SourceSpec] = REGISTERED,
) -> Optional[str]:
    """The source whose vocabulary a term comes from, or None.

    None for the shared vocabularies — temporal, unified, OWL, RDFS,
    GeoSPARQL — which belong to no source, and for a term under no registered
    namespace at all. Both are answers rather than failures, and neither may be
    attributed to a source by guessing.
    """
    return _source_of(uri, _owned_namespaces(specs))


def sources_in_type_uris(
    uris: Sequence[str], specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[str, ...]:
    """The sources these terms come from, once each, in registration order."""
    table = _owned_namespaces(specs)
    found = {name for name in (_source_of(uri, table) for uri in uris) if name}
    return tuple(spec.name for spec in specs if spec.name in found)


# ======================================================================
# Tables built from every registered source, whatever a run picks
# ======================================================================

def namespace_prefixes(
    specs: Sequence[SourceSpec] = REGISTERED,
) -> List[Tuple[str, str]]:
    """NAMESPACE_PREFIXES: each source's namespaces in turn, then the shared ones."""
    return [pair for spec in specs for pair in spec.namespaces] + list(
        SHARED_NAMESPACES
    )


def source_vocabularies(
    specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[str, ...]:
    """The namespaces holding each source's own terms.

    Those under SOURCE_BASE other than the source's enrichment namespace. A
    publisher's real vocabulary, such as NOAA's alert identifiers, is not under
    that base.
    """
    return tuple(
        namespace
        for spec in specs
        for namespace, _prefix in spec.namespaces
        if namespace.startswith(SOURCE_BASE)
        and namespace != spec.enrichment_namespace
    )


def enrichment_namespaces(
    specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[str, ...]:
    """Where this pipeline mints enrichment terms, one namespace per source."""
    return tuple(spec.enrichment_namespace for spec in specs)


def synthetic_temporal_ids(
    specs: Sequence[SourceSpec] = REGISTERED,
) -> Dict[str, str]:
    """Where each date-bearing source's period individuals are minted.

    Keyed by the prefix's last segment (sec, noaa, market-quotes), which is the
    name the temporal unifier's tests look a prefix up by.
    """
    return {
        spec.temporal_prefix.rstrip("/").rsplit("/", 1)[-1]: spec.temporal_prefix
        for spec in specs
        if spec.temporal_prefix
    }


def _merged_rows(specs: Sequence[SourceSpec], field_name: str) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for spec in specs:
        for term, target in getattr(spec, field_name).items():
            if term in merged:
                raise ValueError(
                    f"{term} is mapped by two sources, the second is {spec.name!r}"
                )
            merged[term] = target
    return merged


def property_mappings(specs: Sequence[SourceSpec] = REGISTERED) -> Dict[str, str]:
    """The ontology mapper's property table: every source's rows."""
    return _merged_rows(specs, "property_mappings")


def class_mappings(specs: Sequence[SourceSpec] = REGISTERED) -> Dict[str, str]:
    """The ontology mapper's class table: every source's rows."""
    return _merged_rows(specs, "class_mappings")


def relation_fragments(
    category: str, specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[str, ...]:
    """One edge-feature category's relation fragments, in recorded order."""
    if category not in RELATION_CATEGORIES:
        raise ValueError(f"unknown edge-feature category {category!r}")
    fragments = list(SHARED_RELATION_FRAGMENTS.get(category, ()))
    for spec in specs:
        fragments.extend(spec.relation_fragments.get(category, ()))
    return tuple(fragments)
