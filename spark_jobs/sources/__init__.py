"""The registered data sources, and the per-source tables built from them.

Each source is declared once, as a SourceSpec in its own module here. Tables
that used to be written out source by source are built from every registered
spec: the namespace table and its source and enrichment subsets, the synthetic
period prefixes, the ontology mapping rows, the edge relation fragments and
the loader's path labels.

This package imports neither pyspark nor rdf_utils. rdf_utils builds its
namespace tables from it, and every module that imports rdf_utils would
otherwise load pyspark too.
"""
from typing import Dict, FrozenSet, List, Sequence, Tuple

from rdflib.namespace import OWL, RDFS

from spark_jobs.sources import bls, market, noaa, sec
from spark_jobs.sources.spec import RELATION_CATEGORIES, SourceSpec
from spark_jobs.utils.namespaces import (
    GEOSPARQL,
    SOURCE_BASE,
    SOURCE_TEMPORAL,
    UNIFIED,
)

# Registration order is table order. Each source's namespaces follow the
# previous source's in NAMESPACE_PREFIXES, and a namespace's position there is
# its ontology-source feature slot, so changing this order moves slots.
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
    name the temporal unifier looks a prefix up by.
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


def source_label_patterns(
    specs: Sequence[SourceSpec] = REGISTERED,
) -> Tuple[Tuple[str, str], ...]:
    """(path fragment, source name) pairs the loader labels input paths by."""
    return tuple(
        (fragment, spec.name) for spec in specs for fragment in spec.path_fragments
    )


def source_names(specs: Sequence[SourceSpec] = REGISTERED) -> FrozenSet[str]:
    """Every registered source's name."""
    return frozenset(spec.name for spec in specs)
