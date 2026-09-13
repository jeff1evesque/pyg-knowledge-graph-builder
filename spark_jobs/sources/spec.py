"""What one data source declares about itself.

The registry in spark_jobs/sources/__init__.py builds the pipeline's
per-source tables from these, and the job calls a source's functions for the
sources a run picks. This is its own module so the source modules can import
it without importing the registry that imports them.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

# The edge-feature categories a source may give relation fragments to, in the
# order the edge encoding config records them. "skip" marks relations that
# never get features.
RELATION_CATEGORIES: Tuple[str, ...] = (
    "temporal", "option_stock", "escalation", "correlation", "causal",
    "strategy", "skip",
)


@dataclass(frozen=True)
class RunOptions:
    """The run settings a source's functions may read."""

    sector_definitions_bucket: str = ""
    sector_definitions_key: str = ""
    source_data_day: str = ""


@dataclass(frozen=True, eq=False)
class SourceSpec:
    """One data source, declared once.

    Compared by identity, since each source has one spec. The mapping fields
    are stored read-only, so nothing can edit a registered table in place.

    The function fields import what they run inside their own bodies. The
    registry is imported by rdf_utils, so a Spark import at the top of a source
    module would reach every module that imports rdf_utils.
    """

    # The source's short name, as the loader and the per-source stats key it.
    name: str
    # Fragments that mark an input path as this source's.
    path_fragments: Tuple[str, ...]
    # (namespace, prefix) pairs, in NAMESPACE_PREFIXES order. A namespace's
    # position in that table is its ontology-source feature slot.
    namespaces: Tuple[Tuple[str, str], ...]
    # Where this pipeline mints the source's enrichment terms. One of the
    # namespaces above.
    enrichment_namespace: str
    # How log lines name the source. Defaults to the name.
    label: str = ""
    # The format this source's paths are read in. Empty means the run's
    # --source_format.
    source_format: str = ""
    # Date predicates that place the source's entities in time, and where the
    # period individuals made from them are minted. Both empty for a source
    # whose periods arrive as URIs, as BLS's do.
    date_predicates: Tuple[str, ...] = ()
    temporal_prefix: str = ""
    # The source's rows of the ontology mapper's property and class tables.
    property_mappings: Mapping[str, str] = field(default_factory=dict)
    class_mappings: Mapping[str, str] = field(default_factory=dict)
    # Edge-feature category -> fragments of the source's own relation names.
    relation_fragments: Mapping[str, Tuple[str, ...]] = field(
        default_factory=dict
    )
    # check_paths(paths): raise ValueError for a path of this source the job
    # cannot read. Runs before Spark starts.
    check_paths: Optional[Callable[..., None]] = None
    # canonicalize(triples_df): the source's identifier repair, applied to the
    # rows loaded from its own paths.
    canonicalize: Optional[Callable[..., Any]] = None
    # linker(spark, options): the source's intra-source linker, whose
    # enrich(triples_df) returns new triples.
    linker: Optional[Callable[..., Any]] = None
    # temporal_collector(triples_df): frames of (temporal_uri, normalized_name,
    # kind) for periods the source states as URIs rather than as dates.
    temporal_collector: Optional[Callable[..., Any]] = None
    # cross_source_inputs(options): keyword arguments the source adds to the
    # cross-source linker, read on the driver before that phase starts.
    cross_source_inputs: Optional[Callable[..., Any]] = None

    def __post_init__(self):
        for name in ("property_mappings", "class_mappings", "relation_fragments"):
            object.__setattr__(
                self, name, MappingProxyType(dict(getattr(self, name)))
            )
        if not self.label:
            object.__setattr__(self, "label", self.name)

        if self.enrichment_namespace not in {ns for ns, _ in self.namespaces}:
            raise ValueError(
                f"source {self.name!r}: enrichment namespace "
                f"{self.enrichment_namespace!r} is not one of its namespaces"
            )
        if bool(self.date_predicates) != bool(self.temporal_prefix):
            raise ValueError(
                f"source {self.name!r}: set date predicates and a temporal "
                "prefix together, or neither"
            )
        unknown = sorted(set(self.relation_fragments) - set(RELATION_CATEGORIES))
        if unknown:
            raise ValueError(
                f"source {self.name!r}: unknown edge-feature categories {unknown}"
            )
