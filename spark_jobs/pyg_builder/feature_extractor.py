"""
Feature Extractor — Ontology-Aware Fixed-Width Node Feature Vectors

Constructs universal fixed-width feature vectors (default 1024-d) that
encode three layers of information for every node:

  Segment 1 — Ontology Structure  (25% of dims):  class identity,
              class hierarchy (rdfs:subClassOf chains), ontology/source
              membership

  Segment 2 — Property Schema     (37.5% of dims): property presence
              (which ontology-defined properties this node has),
              domain/range signals, property hierarchy

  Segment 3 — Literal Values      (37.5% of dims): numeric values in
              hashed slots (z-score normalized), categorical values as
              multi-hot hash encodings

All segment and sub-segment boundaries scale proportionally with
vector_dim. Passing vector_dim=512 produces a 512-d vector with the
same three-segment structure at half resolution. Passing vector_dim=2048
doubles resolution. The default 1024 is recommended for production.

All encoding runs on Spark executors using deterministic hash-based
functions expressed as pure Spark column expressions. Only the final
[num_nodes, vector_dim] float32 array per node type is collected to
the driver.

Driver memory safety:
  The dense tensor for a node type is num_nodes × vector_dim × 4 bytes.
  For large types (>500K nodes), this can exceed available driver memory.
  To prevent OOM:
  - Sparse (node_id, dim, value) entries are collected via toPandas()
    and scattered directly into a pre-allocated numpy array
  - The Pandas DataFrame is deleted immediately after scatter
  - For very large types, collection is chunked by node_id range
    so that at most ~500K nodes' sparse entries are in Pandas at once
  - The dense tensor itself is unavoidable (PyG requires it), but we
    ensure only ONE type's tensor + its Pandas intermediary coexist

Why this replaces the old per-type variable-width approach:
  - Universal width enables shared GNN layers across all node types
  - Ontology structure gives the GNN a type fingerprint beyond raw literals
  - Property presence distinguishes "missing" from "inapplicable"
  - Hash-based encoding avoids vocabulary management across 100+ ontologies
"""
import logging
import gc
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np
import torch
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from spark_jobs.utils.rdf_utils import (
    NAMESPACE_PREFIXES,
    ONTOLOGY_NAMESPACE_INDICES,
    PROV_DERIVED_BY,
    PROV_OBSERVED_LITERAL_DATATYPE,
    PROV_CLASS_HIERARCHY,
    PROV_PROPERTY_DOMAIN,
    PROV_PROPERTY_RANGE,
    PROV_PROPERTY_HIERARCHY,
    PROV_ROUTE_LABELS,
    PROV_ROUTE_CURATED_SUBCLASS,
    PROV_ROUTE_NAMED_SUBCLASS,
    PROV_ROUTE_CURATED_SUBPROPERTY,
    PROV_ROUTE_OBSERVED_DOMAIN,
    PROV_ROUTE_OBSERVED_RANGE,
    PROV_ROUTE_DATATYPE_RANGE,
)
from spark_jobs.utils.spark_rdf_utils import collect_sorted
# One resolution of "which class is this node type from", shared with the
# mapping graph_schema.json publishes. node_mapper does not import this module,
# so there is no cycle.
from spark_jobs.pyg_builder.node_mapper import build_type_uri_mapping
from spark_jobs.pyg_builder.sparse_scatter import scatter_sparse_entries
# What the slot assignments below collided on, and the check that stops a build
# whose class_identity segment can no longer separate its classes.
from spark_jobs.pyg_builder.collision_report import (
    check_class_identity_capacity,
    compute_collision_report,
)
# The vector's geometry: VectorLayout owns every segment boundary this
# module writes into, plus the segment proportions published below.
from spark_jobs.pyg_builder.vector_layout import (
    SEG1_FRAC,
    SEG2_FRAC,
    SEG3_FRAC,
    VECTOR_DIM,
    VectorLayout,
)

logger = logging.getLogger(__name__)

# Default for feature_config.numeric_predicate_min_share. A literal property
# is numeric only if MORE THAN this share of its values parse as a number;
# otherwise every one of its values is treated as a category label.
# Classification is per-predicate and mutually exclusive: a property is
# numeric or categorical, never both.
#
# Per-value classification (the previous behaviour) split a single property
# across both branches whenever some of its labels happened to parse. SEC
# hasDocumentType is the motivating case: of 2,372 values, 315 (13.3%) are
# bare-digit form types -- Form 4, 144, 3, 425, 497, 487, 25 -- while the rest
# are hyphenated (10-K, 8-K, S-1). Those 315 were z-scored into the numeric
# segment as if a form number were a magnitude (mean 62.24, std 128.64),
# injecting a spurious continuous ordering over what are labels.
#
# A simple majority is deliberate. It is the least presumptuous rule that
# still fixes the above, and it tolerates a genuinely numeric measurement
# carrying a minority of unparseable sentinels ("N/A", "unknown") without
# demoting the whole property out of the numeric segment.
_NUMERIC_PREDICATE_MIN_SHARE = 0.5


def _is_finite(col):
    """Whether a double column holds a real, usable number.

    Null, NaN and +/-infinity all mean the same thing here -- there is no
    magnitude to encode -- but they arrive by different routes and only the
    first is caught by an ``isNotNull()``. The other two are what let a single
    overflowing literal reach the arithmetic, where a mean goes infinite, a
    stddev goes NaN, and every value of that predicate is silently poisoned.
    """
    return (
        col.isNotNull()
        & ~F.isnan(col)
        & (F.abs(col) != float("inf"))
    )


# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_SUB_PROPERTY_OF = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
OWL_EQUIVALENT_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_EQUIVALENT_PROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"

_NON_FEATURE_PREDICATES = {
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUB_PROPERTY_OF,
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
    # Statements about the pipeline's own derivations. Their subjects are
    # predicate URIs and provenance URIs, never entities, so they must not
    # reach property presence or the literal segments.
    PROV_DERIVED_BY,
    PROV_OBSERVED_LITERAL_DATATYPE,
    *PROV_ROUTE_LABELS,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2002/07/owl#imports",
}

# Predicates whose presence proves the ontology-mapping phase ran over the
# triples this build was handed.
#
# The builder cannot read --enable_ontology_mapping: in pyg_only mode (how
# every experiment sweep runs) enrichment happened in a separate job, and this
# job's own flag says nothing about the Parquet it is reading -- the 2026-07-29
# manifest recorded enable_ontology_mapping=false for exactly that reason,
# describing a phase that run never even reached. So detect it from the data
# instead, which is truthful in every mode.
#
# OntologyMapper emits owl:equivalentProperty from a static, non-empty table on
# every run (_create_property_equivalences), and owl:equivalentClass from the
# curated class map -- and nothing else in the pipeline emits either. Their
# absence therefore means the phase did not run, not that it ran and found
# nothing.
_ONTOLOGY_MAPPING_MARKERS = (OWL_EQUIVALENT_PROPERTY, OWL_EQUIVALENT_CLASS)


def _empty_hierarchy_reason(ontology_mapping_ran: bool) -> str:
    """Why class_hierarchy is empty — the two causes demand opposite responses.

    "the sources declare no subsumption" is data to work with; "the phase that
    derives subsumption never ran" is a re-run. Both serialize to an empty
    superclass_chain, so the distinction has to be stated, not inferred.
    """
    if ontology_mapping_ran:
        return "no rdfs:subClassOf after ontology mapping"
    return (
        "no rdfs:subClassOf in source data and ontology mapping did not run"
    )


def _empty_property_schema_reason(ontology_mapping_ran: bool) -> str:
    """Why domain_range is empty. Same two causes as the class hierarchy."""
    if ontology_mapping_ran:
        return "no rdfs:domain or rdfs:range after ontology mapping"
    return (
        "no rdfs:domain or rdfs:range in source data and ontology mapping "
        "did not run"
    )


def _empty_property_hierarchy_reason(ontology_mapping_ran: bool) -> str:
    """Why property_hierarchy is empty. Same two causes again."""
    if ontology_mapping_ran:
        return "no rdfs:subPropertyOf after ontology mapping"
    return (
        "no rdfs:subPropertyOf in source data and ontology mapping did not run"
    )


# Provenance fallbacks for an axiom set the mapper left no marker for: either
# the axioms were in the source to begin with, or there are none.
_DECLARED = "declared by the source"
_NOT_DERIVED = "not derived — ontology mapping did not run"


def _route_counts(
    direct_axioms: Set[Tuple[str, str]],
    routes: Dict[Tuple[str, str], str],
) -> Dict[str, int]:
    """Split a set of axioms by the route that produced it.

    An axiom with no route marker was not derived by this pipeline, so it
    came out of the source and is counted as ``"declared"``. That fallback is
    what makes the count trustworthy in both directions: a build with mapping
    turned off has no markers and everything reads as declared, correctly,
    while these builds have a marker for every edge and ``declared`` is 0.

    Returned with sorted keys, and built by walking the axioms in sorted
    order. ``direct_axioms`` is a SET, so iterating it directly seeds dict
    insertion order from string hashing -- which differs per process under a
    randomized PYTHONHASHSEED. The values were always right; the KEY ORDER
    moved between runs, and metadata is compared in serialized form, so that
    alone broke byte-reproducibility (caught by test_jolts_features_
    reproducible, the one guard that compares JSON text rather than parsed
    objects).
    """
    counts: Dict[str, int] = {}
    for axiom in sorted(direct_axioms):
        route = routes.get(axiom)
        label = PROV_ROUTE_LABELS.get(route, "declared") if route else (
            "declared"
        )
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _merge_counts(*counts: Dict[str, int]) -> Dict[str, int]:
    """Sum per-route counts across axiom sets that share one _source field."""
    merged: Dict[str, int] = {}
    for group in counts:
        for label, n in group.items():
            merged[label] = merged.get(label, 0) + n
    return merged


def _route_detail(
    direct_axioms: Set[Tuple[str, str]],
    routes: Dict[Tuple[str, str], str],
    counts: Dict[str, int],
) -> Dict[str, Any]:
    """Counts per route plus the axioms each one produced.

    Edge lists are capped: the counts stay exact, and a set big enough to hit
    the cap is one nobody was going to read edge-by-edge anyway. ``declared``
    is always reported even at zero -- that it IS zero is the finding.
    """
    by_route: Dict[str, List[List[str]]] = {}
    for subject, obj in sorted(direct_axioms):
        route = routes.get((subject, obj))
        label = PROV_ROUTE_LABELS.get(route, "declared") if route else (
            "declared"
        )
        by_route.setdefault(label, []).append([subject, obj])

    return {
        "total": len(direct_axioms),
        # Sorted, and "declared" present even at zero -- that it IS zero is
        # the finding. Key order has to be stable for the same reason as in
        # _route_counts: metadata reproducibility is asserted on JSON text.
        "counts": dict(sorted({"declared": 0, **counts}.items())),
        "axioms": {
            label: edges[:_PROVENANCE_EDGE_LIMIT]
            for label, edges in sorted(by_route.items())
        },
        "truncated": any(
            len(edges) > _PROVENANCE_EDGE_LIMIT for edges in by_route.values()
        ),
    }


def _axiom_source(predicate_name: str, route_counts: Dict[str, int]) -> str:
    """Name where a populated axiom set actually came from.

    ``"rdfs:subClassOf"`` is true of the predicate the encoder consumed and
    false as an answer to "did a source declare this hierarchy". Since #273 no
    source does: the mapper derives every edge, and by the time the collector
    sees them a derived triple and a declared one are identical. Only the
    route markers tell them apart, and this turns them into the one string a
    reader looks at first.

    ``route_counts`` maps a route label (or ``"declared"``) to how many
    DIRECT axioms came in by it. Mixed sets report every route with its count,
    because "mostly curated with four guesses in it" is a different artifact
    from "all curated" and the difference is exactly what a debugging session
    needs.
    """
    live = {label: n for label, n in sorted(route_counts.items()) if n}
    if not live:
        return predicate_name

    if set(live) == {"declared"}:
        return predicate_name

    parts = ", ".join(
        f"{label} ({n})" for label, n in sorted(
            live.items(), key=lambda kv: (-kv[1], kv[0])
        )
    )
    if "declared" in live:
        return f"mixed: {parts}"
    return f"derived: {parts}"


# ============================================
# Driver memory safety constants
# ============================================
_CHUNK_NODE_THRESHOLD = 500_000

# Number of hash functions for multi-hot categorical encoding
_NUM_CATEGORICAL_HASHES = 4

# Cap on the named gaps in property_schema_coverage. The counts are always
# exact; only the URI lists are clipped, so the file cannot grow without
# bound on a vocabulary where most predicates are ambiguous.
_COVERAGE_GAP_LIMIT = 200

# Cap on the per-route axiom lists in derived_axioms. Counts stay exact.
_PROVENANCE_EDGE_LIMIT = 500

# Seed offsets for independent hash functions
_HASH_SEEDS = [0, 7, 13, 31]


def _slot_dim(col, seed: int, dim: int, start: int):
    """The vector column a hashed value lands in.

    Every hashed placement goes through here, and so does the slot mapping
    that publishes those columns, so the two cannot disagree.

    They used to. The mapping recomputed each slot on the driver with md5
    while the encoder placed values with Spark's hash, and the published
    column was right only by coincidence -- about one predicate in 257. A
    reader of slot_mapping.json was looking at the wrong feature (#354).
    """
    return F.abs(F.hash(col, F.lit(seed))) % F.lit(dim) + F.lit(start)


# What a categorical slot is keyed on. Joined into the key, not hashed apart.
_CATEGORICAL_KEY_SEP = "::"


def _categorical_key():
    """The value a categorical slot is keyed on: the predicate AND its value.

    A categorical property does not occupy a fixed set of columns the way a
    numeric one does -- every distinct predicate/value pair gets its own.
    slot_mapping.json used to publish four columns per predicate as though it
    did, which named columns holding some other value entirely (#354).
    """
    return F.concat(
        F.col("predicate"), F.lit(_CATEGORICAL_KEY_SEP), F.col("cat_value")
    )


def _local_name(uri: str) -> str:
    """The trailing name of a URI, after the last ``/`` or ``#``."""
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


class FeatureExtractor:
    """
    Builds universal fixed-width ontology-aware feature vectors for all
    nodes.

    All heavy computation runs on Spark executors. Only compact float
    arrays are collected to the driver, one node type at a time.

    Driver memory safety:
      - Dense tensor is pre-allocated once per type (num_nodes × vector_dim × 4B)
      - Sparse entries collected in chunks for large types
      - Pandas intermediaries freed immediately after scatter
      - Only one type's tensor is being built at a time
      - gc.collect() between types to reclaim fragmented memory
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        feat_config = config.get("feature_config", {})
        self._normalize = feat_config.get("normalize", True)
        self._vector_dim = feat_config.get("vector_dim", VECTOR_DIM)
        self._chunk_threshold = feat_config.get(
            "chunk_node_threshold", _CHUNK_NODE_THRESHOLD
        )
        self._numeric_min_share = feat_config.get(
            "numeric_predicate_min_share", _NUMERIC_PREDICATE_MIN_SHARE
        )
        self._class_identity_dim = feat_config.get(
            "class_identity_dim", None
        )
        # Escape hatch for a build that knowingly exceeds the class budget --
        # an exploratory run where partial class identity is acceptable. Off by
        # default: over-subscription costs nothing visible at build time and
        # only shows up as a model that will not learn class distinctions, so
        # it has to be asked for.
        self._allow_class_oversubscription = feat_config.get(
            "allow_class_identity_oversubscription", False
        )

        # Compute layout from vector_dim — all segment boundaries
        # scale proportionally
        self._layout = VectorLayout(
            self._vector_dim,
            class_identity_dim=self._class_identity_dim,
        )

        # Ontology-wide existence flags, populated once per build_features()
        # run and reused across all node types (see the hoist in that method).
        # Conservative defaults so a direct encoder call can't AttributeError.
        self._has_class_hierarchy = True
        self._has_property_schema = True
        self._has_property_hierarchy = True

        # Whether the ontology-mapping phase ran over the triples handed to
        # build_features(). Detected from the data, not from a config flag —
        # see _ONTOLOGY_MAPPING_MARKERS.
        self._ontology_mapping_ran = False

        # Metadata artifacts — populated by the _collect_* methods during
        # build_features(). Initialized here (not just inside the conditional
        # collect paths) so get_metadata_artifacts() is safe even when a build
        # has no numeric literals (normalization collection is then skipped).
        self._collected_norm_stats: Optional[List[Dict[str, Any]]] = None
        self._collected_zero_variance: List[str] = []
        self._collected_ontology_schema: Optional[Dict[str, Any]] = None
        self._collected_slot_mapping: Optional[Dict[str, Any]] = None
        # Empty until build_features() runs. An unknown sub-segment is treated
        # as claiming features, so a missing entry fails the vacuity guard
        # rather than silently exempting the sub-segment from it.
        self._collected_sub_segment_status: Dict[str, Optional[str]] = {}

    def get_layout(self) -> "VectorLayout":
        """Return the VectorLayout instance for metadata registration."""
        return self._layout

    def get_encoding_config(self) -> Dict[str, Any]:
        """
        Return the complete encoding configuration needed to
        deterministically reproduce the hash-based encoding.

        All values here are the same constants used in the encoding
        methods below. If any of these change, the same ontology class
        or property hashes to different vector positions and the
        trained model breaks.

        Called by constructor.py after build_features() to register
        with MetadataCollector.
        """
        layout = self._layout
        return {
            "version": "1.0",
            "hash_algorithm": "spark_murmur3",
            "node_features": {
                "total_dim": layout.vector_dim,
                "segment_proportions": {
                    "ontology_structure": SEG1_FRAC,
                    "property_schema": SEG2_FRAC,
                    "literal_values": SEG3_FRAC,
                },
                "class_identity": {
                    "dim": layout.seg1_class_identity_dim,
                    "num_hashes": len(_HASH_SEEDS),
                    "seeds": list(_HASH_SEEDS),
                    # How many ontology classes this width can keep linearly
                    # separable -- equal to the width, since a d-dim segment
                    # holds at most d independent codes. Published so a
                    # consumer can see the budget it is encoding against
                    # without knowing the rule.
                    "capacity_classes": layout.seg1_class_identity_dim,
                },
                "class_hierarchy": {
                    "dim": layout.seg1_class_hierarchy_dim,
                    "num_hashes": 2,
                    "seeds": [
                        _HASH_SEEDS[0] + 100,
                        _HASH_SEEDS[1] + 100,
                    ],
                    "decay_function": "inverse_depth",
                    "max_depth": 10,
                },
                "ontology_source": {
                    "dim": layout.seg1_ontology_source_dim,
                    "method": "index_modulo",
                    "node_uri_weight": 0.5,
                },
                "property_presence": {
                    "dim": layout.seg2_property_presence_dim,
                    "num_hashes": 3,
                    "seeds": [s + 200 for s in _HASH_SEEDS[:3]],
                    "encoding_convention": {
                        "present": 1.0,
                        "absent": -1.0,
                        "not_in_schema": 0.0,
                    },
                },
                "domain_range": {
                    "dim": layout.seg2_domain_range_dim,
                    "seeds": [300, 301],
                },
                "property_hierarchy": {
                    "dim": layout.seg2_property_hierarchy_dim,
                    "num_hashes": 2,
                    "seeds": [s + 400 for s in _HASH_SEEDS[:2]],
                },
                "numeric_values": {
                    "dim": layout.seg3_numeric_dim,
                    "seed": 500,
                    # Which properties reach this segment at all — a
                    # different threshold routes a mixed property to the
                    # categorical segment instead.
                    "predicate_min_numeric_share": self._numeric_min_share,
                },
                "categorical_values": {
                    "dim": layout.seg3_categorical_dim,
                    "num_hashes": _NUM_CATEGORICAL_HASHES,
                    "seeds": [s + 600 for s in _HASH_SEEDS],
                },
            },
        }
        # NOTE: no "checksum" key here. The total dim it used to carry is
        # already recorded as node_features.total_dim, and a dimension is not a
        # checksum -- two builds with different seeds but the same vector width
        # produced identical values. The real contract digest is computed once
        # over the MERGED node+edge config in MetadataCollector, which is the
        # only place that sees the whole contract.

    def get_metadata_artifacts(self) -> Dict[str, Any]:
        """
        Return all metadata artifacts collected during build_features().

        Returns a dict with keys:
          - normalization_stats: list of per-predicate stat dicts
          - zero_variance_properties: list of predicate URIs
          - ontology_schema: frozen ontology structure dict
          - slot_mapping: dimension-to-meaning mapping dict

        All values are small Python objects — no tensors, no DataFrames.
        Populated by _collect_* methods called during build_features().
        """
        return {
            "normalization_stats": self._collected_norm_stats,
            "zero_variance_properties": self._collected_zero_variance,
            "ontology_schema": self._collected_ontology_schema,
            "slot_mapping": self._collected_slot_mapping,
            "sub_segment_status": self._collected_sub_segment_status,
        }

    def build_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """
        Build universal fixed-width feature vectors for all node types.

        Args:
            triples_df: Enriched triples DataFrame (subject, predicate, object)
            node_id_df: Node ID table (uri, node_id, node_type) — cached
            node_counts: Dict[str, int] of node type counts

        Returns:
            Tuple of:
            - Dict[node_type -> FloatTensor[num_nodes, vector_dim]]
            - Dict[node_type -> List[str]] segment description lists
        """
        layout = self._layout
        vector_dim = layout.vector_dim

        logger.info(
            f"  Building {vector_dim}-d ontology-aware feature vectors"
        )
        logger.info(f"  {layout.summary()}")

        # Log driver memory budget estimate
        total_dense_bytes = sum(
            n * vector_dim * 4 for n in node_counts.values()
        )
        total_dense_mb = total_dense_bytes / (1024 * 1024)
        logger.info(
            f"  Estimated total dense tensor memory: "
            f"{total_dense_mb:,.1f} MB across {len(node_counts)} types"
        )

        # ============================================
        # Pre-compute ontology structure tables (on executors)
        # ============================================
        logger.info("  Extracting ontology structure from triples...")

        self._ontology_mapping_ran = self._detect_ontology_mapping(triples_df)
        class_hierarchy_df = self._extract_class_hierarchy(triples_df)
        property_schema_df = self._extract_property_schema(triples_df)
        property_hierarchy_df = self._extract_property_hierarchy(triples_df)

        # An unmapped build silently forfeits the class_hierarchy sub-segment
        # -- 64 of 1024 dims on the production layout. That is a re-run
        # decision, so say it at build time instead of leaving it to be
        # noticed later in a file full of empty lists.
        if not self._ontology_mapping_ran:
            logger.warning(
                f"    Ontology mapping did not run on these triples "
                f"(no {OWL_EQUIVALENT_PROPERTY} / {OWL_EQUIVALENT_CLASS}); "
                f"the class_hierarchy sub-segment "
                f"({layout.seg1_class_hierarchy_dim} of "
                f"{layout.vector_dim} dims) will be entirely zero. "
                f"Re-run enrichment with --enable_ontology_mapping true to "
                f"populate it."
            )

        # ============================================
        # Pre-compute per-node property presence (on executors)
        # ============================================
        logger.info("  Computing per-node property presence...")

        node_properties_df = self._compute_node_properties(
            triples_df, node_id_df
        )

        # ============================================
        # Pre-compute literal values (on executors)
        # ============================================
        logger.info("  Extracting literal values...")

        # Classify each literal predicate once, then route its values to
        # exactly one segment — the two extractions partition the literals.
        # DISK_ONLY -- nothing in this leg stays resident; see execute_pyg_only.
        literal_triples = self._literal_triples(
            triples_df, node_id_df
        ).persist(StorageLevel.DISK_ONLY)
        numeric_predicates = self._classify_literal_predicates(literal_triples)

        numeric_df = self._extract_numeric_literals(
            literal_triples, node_id_df, numeric_predicates
        )
        categorical_df = self._extract_categorical_literals(
            literal_triples, node_id_df, numeric_predicates
        )

        # ============================================
        # Pre-compute normalization stats for numeric values
        # ============================================
        norm_stats = None
        if self._normalize and numeric_df is not None:
            logger.info("  Computing normalization statistics...")
            norm_stats = self._compute_normalization_stats(numeric_df)

            # Collect stats for metadata — small collect, one row per
            # predicate (typically <200 predicates across all ontologies)
            self._collect_normalization_metadata(numeric_df)

        # ============================================
        # Get type URIs for ontology encoding
        # ============================================
        type_uri_df = self._get_type_uri_mapping(triples_df, node_id_df)

        # ============================================
        # Collect ontology schema and slot mapping for metadata.
        # All collect() calls here target small aggregated/distinct
        # DataFrames — never raw triples or per-node data.
        # ============================================
        self._collect_ontology_schema_metadata(
            triples_df, node_id_df, node_counts,
            class_hierarchy_df, property_schema_df, property_hierarchy_df,
        )
        self._collect_slot_mapping_metadata(
            numeric_df, categorical_df, type_uri_df,
            class_hierarchy_df,
        )

        # ============================================
        # Build ALL node-type vectors in ONE distributed pass.
        #
        # The encoders carry ``node_type`` through every projection, so every
        # type's sparse (node_type, node_id, dim, value) entries are produced
        # and aggregated together — a single Spark job-chain instead of one
        # per type (which was the dominant cost). node_id is 0-indexed within
        # a type, so the driver splits the collected frame back out by
        # node_type when scattering into per-type dense tensors.
        # ============================================
        logger.info("  Assembling feature vectors (single pass, all types)...")

        # Ontology-wide existence checks, evaluated ONCE (not once per type).
        self._has_class_hierarchy = bool(class_hierarchy_df.head(1))
        self._has_property_schema = bool(property_schema_df.head(1))
        self._has_property_hierarchy = bool(property_hierarchy_df.head(1))

        # Why each sub-segment will or will not carry signal, recorded at the
        # one place that knows. A sub-segment whose source predicate is absent
        # from the triples encodes nothing, and feature_spec.json must say so
        # rather than declaring dims it cannot fill.
        self._collected_sub_segment_status = {
            "class_hierarchy": (
                None if self._has_class_hierarchy
                else _empty_hierarchy_reason(self._ontology_mapping_ran)
            ),
            "domain_range": (
                None if self._has_property_schema
                else "no rdfs:domain or rdfs:range in source data"
            ),
            "property_hierarchy": (
                None if self._has_property_hierarchy
                else "no rdfs:subPropertyOf in source data"
            ),
            "numeric_values": (
                None if numeric_df is not None
                else "no numeric literal properties in source data"
            ),
            "categorical_values": (
                None if categorical_df is not None
                else "no categorical literal properties in source data"
            ),
        }

        segment_names = [
            f"ontology_structure[{layout.seg1_start}:{layout.seg2_start}]",
            f"property_schema[{layout.seg2_start}:{layout.seg3_start}]",
            f"literal_values[{layout.seg3_start}:{layout.vector_dim}]",
        ]

        all_nodes = node_id_df.select("uri", "node_id", "node_type")

        seg1 = self._encode_ontology_structure(
            all_type_uris=type_uri_df,
            all_nodes=all_nodes,
            class_hierarchy_df=class_hierarchy_df,
        )
        seg2 = self._encode_property_schema(
            all_node_props=node_properties_df,
            property_schema_df=property_schema_df,
            property_hierarchy_df=property_hierarchy_df,
        )
        seg3 = self._encode_literal_values(
            numeric_df=numeric_df,
            categorical_df=categorical_df,
            norm_stats=norm_stats,
        )

        all_parts = [p for p in [seg1, seg2, seg3] if p is not None]

        feature_tensors: Dict[str, torch.Tensor] = {}
        feature_names: Dict[str, List[str]] = {}
        active_types = [
            (t, n) for t, n in node_counts.items() if n > 0
        ]

        if not all_parts:
            # No sparse entries at all — every type is a zero tensor.
            for node_type, num_nodes in active_types:
                feature_tensors[node_type] = torch.zeros(
                    num_nodes, vector_dim, dtype=torch.float32
                )
                feature_names[node_type] = segment_names
        else:
            combined = all_parts[0]
            for p in all_parts[1:]:
                combined = combined.unionAll(p)

            # Aggregate on executors: sum values at same (node_type, node_id,
            # dim). Cache the single result so the bounded per-batch collects
            # below read from cache instead of recomputing the encode plan.
            combined = (
                combined
                .groupBy("node_type", "node_id", "dim")
                .agg(F.sum("value").alias("value"))
                .select(
                    F.col("node_type"),
                    F.col("node_id").cast("long"),
                    F.col("dim").cast("int"),
                    F.col("value").cast("float"),
                )
                # DISK_ONLY -- nothing in this leg stays resident; see
                # execute_pyg_only.
                .persist(StorageLevel.DISK_ONLY)
            )

            self._scatter_all_types(
                combined, active_types, vector_dim,
                feature_tensors, feature_names, segment_names,
            )

            combined.unpersist()

        # Cleanup cached intermediates
        for df in [literal_triples, numeric_df, categorical_df,
                    class_hierarchy_df, property_schema_df]:
            if df is not None:
                try:
                    df.unpersist()
                except Exception:
                    pass

        return feature_tensors, feature_names

    def _scatter_all_types(
        self,
        combined: DataFrame,
        active_types: List[Tuple[str, int]],
        vector_dim: int,
        feature_tensors: Dict[str, "torch.Tensor"],
        feature_names: Dict[str, List[str]],
        segment_names: List[str],
    ) -> None:
        """
        Collect the single aggregated (node_type, node_id, dim, value) frame
        and scatter it into per-type dense tensors.

        Driver-memory discipline (preserves the #186 guarantee at cluster
        scale; a single batch on small data):
          - Large types (> chunk_threshold nodes) are collected one at a time
            via the chunked node_id-range path, so one big type can't blow the
            driver heap.
          - Small types are collected in node-count-bounded batches — one
            toPandas per batch instead of one per type.

        Executor-memory discipline (#346): ``combined`` arrives cached, and
        that is the only copy of these rows the executors should hold. Neither
        path below may cache what it filters out of it -- one node type is
        ~98% of the graph, so caching that type's slice is a second copy of
        nearly the whole frame.
        """
        import torch

        large = [
            (t, n) for t, n in active_types if n > self._chunk_threshold
        ]
        small = [
            (t, n) for t, n in active_types if n <= self._chunk_threshold
        ]

        for node_type, num_nodes in large:
            logger.info(
                f"    [{node_type}] {num_nodes:,} nodes (chunked collection)"
            )
            tensor = np.zeros((num_nodes, vector_dim), dtype=np.float32)
            type_combined = (
                combined
                .filter(F.col("node_type") == node_type)
                .select("node_id", "dim", "value")
            )
            self._collect_and_scatter_chunked(
                type_combined, tensor, num_nodes, vector_dim
            )
            feature_tensors[node_type] = torch.from_numpy(tensor).contiguous()
            feature_names[node_type] = segment_names
            gc.collect()

        budget = max(1, self._chunk_threshold)

        def flush(batch: List[Tuple[str, int]]) -> None:
            if not batch:
                return
            names = [t for t, _ in batch]
            pdf = (
                combined
                .filter(F.col("node_type").isin(names))
                .toPandas()
            )
            groups = (
                {nt: g for nt, g in pdf.groupby("node_type")}
                if not pdf.empty else {}
            )
            for node_type, num_nodes in batch:
                tensor = np.zeros(
                    (num_nodes, vector_dim), dtype=np.float32
                )
                scatter_sparse_entries(
                    groups.get(node_type), tensor, "node_id", vector_dim
                )
                feature_tensors[node_type] = (
                    torch.from_numpy(tensor).contiguous()
                )
                feature_names[node_type] = segment_names
            del pdf
            gc.collect()

        batch: List[Tuple[str, int]] = []
        batch_nodes = 0
        for node_type, num_nodes in small:
            if batch and batch_nodes + num_nodes > budget:
                flush(batch)
                batch = []
                batch_nodes = 0
            batch.append((node_type, num_nodes))
            batch_nodes += num_nodes
        flush(batch)
        logger.info(
            f"    Collected {len(small)} small + {len(large)} large "
            f"node types"
        )

    # ================================================================
    # Ontology structure extraction (all on executors)
    # ================================================================

    def _detect_ontology_mapping(self, triples_df: DataFrame) -> bool:
        """
        Whether the ontology-mapping phase ran over these triples.

        Short-circuits on the first marker row (see
        _ONTOLOGY_MAPPING_MARKERS for why these predicates are the signal),
        so it costs a single scan that stops early rather than a count.
        """
        return bool(
            triples_df
            .filter(F.col("predicate").isin(list(_ONTOLOGY_MAPPING_MARKERS)))
            .head(1)
        )

    def _extract_class_hierarchy(
        self, triples_df: DataFrame
    ) -> DataFrame:
        """
        Extract rdfs:subClassOf chains from triples.

        Returns DataFrame(class_uri, superclass_uri, depth) where depth
        indicates distance in the hierarchy (1 = direct superclass).

        Computes transitive closure up to depth 10 via iterative joins
        on executors.
        """
        direct = (
            triples_df
            .filter(F.col("predicate") == RDFS_SUBCLASS_OF)
            .select(
                F.col("subject").alias("class_uri"),
                F.col("object").alias("superclass_uri"),
            )
            .distinct()
        )

        if not direct.head(1):
            logger.info("    No rdfs:subClassOf triples found")
            schema = "class_uri string, superclass_uri string, depth int"
            return self.spark.createDataFrame([], schema)

        current = direct.withColumn("depth", F.lit(1))
        all_hierarchy = current

        max_depth = 10
        for d in range(2, max_depth + 1):
            next_level = (
                current
                .select(
                    F.col("class_uri"),
                    F.col("superclass_uri").alias("_mid"),
                )
                .join(
                    direct.select(
                        F.col("class_uri").alias("_mid"),
                        F.col("superclass_uri"),
                    ),
                    "_mid",
                    "inner",
                )
                .drop("_mid")
                .withColumn("depth", F.lit(d))
                .distinct()
            )

            next_level = next_level.join(
                all_hierarchy.select("class_uri", "superclass_uri"),
                ["class_uri", "superclass_uri"],
                "left_anti",
            )

            if not next_level.head(1):
                break

            all_hierarchy = all_hierarchy.unionAll(next_level)
            current = next_level

        all_hierarchy = all_hierarchy.cache()
        count = all_hierarchy.count()
        logger.info(
            f"    Class hierarchy: {count:,} (class, superclass) pairs"
        )

        return all_hierarchy

    def _extract_property_schema(
        self, triples_df: DataFrame
    ) -> DataFrame:
        """
        Extract rdfs:domain and rdfs:range declarations.

        Returns DataFrame(property_uri, domain_uri, range_uri).
        """
        domain_df = (
            triples_df
            .filter(F.col("predicate") == RDFS_DOMAIN)
            .select(
                F.col("subject").alias("property_uri"),
                F.col("object").alias("domain_uri"),
            )
            .distinct()
        )

        range_df = (
            triples_df
            .filter(F.col("predicate") == RDFS_RANGE)
            .select(
                F.col("subject").alias("property_uri"),
                F.col("object").alias("range_uri"),
            )
            .distinct()
        )

        schema_df = domain_df.join(range_df, "property_uri", "full_outer")
        schema_df = schema_df.cache()

        count = schema_df.count()
        logger.info(
            f"    Property schema: {count:,} properties with domain/range"
        )

        return schema_df

    def _extract_property_hierarchy(
        self, triples_df: DataFrame
    ) -> DataFrame:
        """
        Extract rdfs:subPropertyOf relationships.

        Returns DataFrame(property_uri, super_property_uri).
        """
        prop_hier = (
            triples_df
            .filter(F.col("predicate") == RDFS_SUB_PROPERTY_OF)
            .select(
                F.col("subject").alias("property_uri"),
                F.col("object").alias("super_property_uri"),
            )
            .distinct()
        )

        return prop_hier

    # ================================================================
    # Per-node property presence (on executors)
    # ================================================================

    def _compute_node_properties(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Compute which properties each node has (regardless of value).

        Returns DataFrame(node_type, node_id, predicate) — one row per
        (node, property) pair where the node is the subject.
        """
        excluded_list = list(_NON_FEATURE_PREDICATES)

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        node_props = (
            triples_df
            .filter(~F.col("predicate").isin(excluded_list))
            .select("subject", "predicate")
            .distinct()
            .join(
                node_lookup,
                F.col("subject") == F.col("_node_uri"),
                "inner",
            )
            .drop("_node_uri", "subject")
            .select("node_type", "node_id", "predicate")
        )

        return node_props

    # ================================================================
    # Literal value extraction (on executors)
    # ================================================================

    def _literal_triples(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Triples whose object is a literal, excluding structural predicates.

        The anti-join drops any triple whose object is a known node URI —
        what remains is the literal-valued tail of the graph.
        """
        return (
            triples_df
            .join(
                node_id_df.select(F.col("uri").alias("_obj_uri")),
                triples_df["object"] == F.col("_obj_uri"),
                "left_anti",
            )
            .filter(~F.col("predicate").isin(list(_NON_FEATURE_PREDICATES)))
        )

    @staticmethod
    def _numeric_cast(col: str = "object"):
        """Lexical form of a literal cast to double — null unless it is finite.

        ``cast("double")`` fails to null on a value it cannot read, but it does
        NOT fail on one it reads as a number too large to hold: Java's parser
        follows the float64 rules and returns infinity. The CUSIP ``46120E602``
        is a real identifier and valid scientific notation, so it arrives here
        as 46120 x 10^602 and lands as ``inf`` -- not null, so it survived the
        ``isNotNull()`` filter downstream, made that predicate's mean infinite
        and its stddev NaN, and put NaN in 5,396 node feature rows (#351).

        Infinity is not a measurement whatever produced it, so it is treated
        exactly like an unparseable value: no number here. Both callers ask
        this the same question -- the classifier via ``isNotNull()`` and the
        value extraction via its filter -- so answering it once keeps the share
        that decides "is this predicate numeric" consistent with the values
        that are actually encoded.
        """
        parsed = F.split(F.col(col), r"\^\^").getItem(0).cast("double")
        return F.when(_is_finite(parsed), parsed)

    def _classify_literal_predicates(
        self,
        literal_triples: DataFrame,
    ) -> Set[str]:
        """
        Decide, per predicate, whether it is numeric or categorical.

        A predicate is numeric when more than ``self._numeric_min_share`` of
        its literal values parse as a number. The two classes are mutually
        exclusive, so a property never lands in both the numeric and the
        categorical segment.

        Returns the set of numeric predicates. Small driver-side collect —
        one row per distinct literal predicate (typically <200).
        """
        shares = (
            literal_triples
            .withColumn("_is_numeric", self._numeric_cast().isNotNull())
            .groupBy("predicate")
            .agg(
                F.count("*").alias("total"),
                F.sum(F.col("_is_numeric").cast("int")).alias("numeric"),
            )
            .collect()
        )

        numeric_predicates = set()
        for row in shares:
            share = row["numeric"] / row["total"] if row["total"] else 0.0
            if share > self._numeric_min_share:
                numeric_predicates.add(row["predicate"])
            elif row["numeric"]:
                # Mixed, but not numeric enough — the parseable minority is
                # treated as labels rather than magnitudes.
                logger.info(
                    f"    {row['predicate']}: {row['numeric']}/{row['total']} "
                    f"values parse as numeric ({share:.1%}) — "
                    f"classified categorical"
                )

        logger.info(
            f"    Literal predicates: {len(numeric_predicates)} numeric, "
            f"{len(shares) - len(numeric_predicates)} categorical"
        )
        return numeric_predicates

    def _extract_numeric_literals(
        self,
        literal_triples: DataFrame,
        node_id_df: DataFrame,
        numeric_predicates: Set[str],
    ) -> Optional[DataFrame]:
        """
        Extract numeric literal properties joined with node IDs.

        Values of a numeric predicate that do not parse are dropped rather
        than re-routed to the categorical segment: a sentinel in a numeric
        field is missing data, so the node carries no value in that slot —
        the same way an absent property is already handled.

        Returns DataFrame(node_type, node_id, predicate, numeric_value)
        or None if no numeric literals found.
        """
        if not numeric_predicates:
            logger.info("    No numeric literals found")
            return None

        candidates = literal_triples.filter(
            F.col("predicate").isin(list(numeric_predicates))
        ).withColumn(
            "numeric_value", self._numeric_cast(),
        ).filter(F.col("numeric_value").isNotNull())

        if not candidates.head(1):
            logger.info("    No numeric literals found")
            return None

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        numeric_df = (
            candidates
            .join(
                node_lookup,
                candidates["subject"] == node_lookup["_node_uri"],
                "inner",
            )
            .drop("_node_uri")
            .groupBy("node_type", "node_id", "predicate")
            .agg(F.mean("numeric_value").alias("numeric_value"))
        )

        # DISK_ONLY -- nothing in this leg stays resident; see execute_pyg_only.
        numeric_df = numeric_df.persist(StorageLevel.DISK_ONLY)
        count = numeric_df.count()
        logger.info(
            f"    Numeric literals: {count:,} (node, property) pairs"
        )

        return numeric_df

    def _extract_categorical_literals(
        self,
        literal_triples: DataFrame,
        node_id_df: DataFrame,
        numeric_predicates: Set[str],
    ) -> Optional[DataFrame]:
        """
        Extract categorical literal properties.

        Every value of a categorical predicate is a label, including any
        that happen to parse as a number (SEC form "4" is a form type, not
        the quantity four).

        Returns DataFrame(node_type, node_id, predicate, cat_value)
        or None.
        """
        non_numeric = literal_triples
        if numeric_predicates:
            non_numeric = non_numeric.filter(
                ~F.col("predicate").isin(list(numeric_predicates))
            )

        if not non_numeric.head(1):
            logger.info("    No categorical literals found")
            return None

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        cat_df = (
            non_numeric
            .join(
                node_lookup,
                non_numeric["subject"] == node_lookup["_node_uri"],
                "inner",
            )
            .drop("_node_uri")
            .select(
                "node_type", "node_id", "predicate",
                F.col("object").alias("cat_value"),
            )
        )

        return cat_df

    # ================================================================
    # Normalization statistics (on executors)
    # ================================================================

    def _compute_normalization_stats(
        self, numeric_df: DataFrame
    ) -> DataFrame:
        """
        Compute per-predicate mean and stddev for z-score normalization.

        Returns DataFrame(predicate, mu, sigma) — small table, broadcast
        joined downstream.

        The fallbacks reject any statistic that is not a finite number, not
        merely a null or a zero. NaN is neither null nor equal to 0.0, so the
        older guard passed it straight through and ``(value - mu) / sigma``
        produced NaN for every row of that predicate (#351). ``_numeric_cast``
        now keeps non-finite values out of ``numeric_df`` in the first place,
        so this should never fire; it stays because the failure it prevents is
        silent, and a graph full of NaN costs hours to discover downstream.
        """
        stats = (
            numeric_df
            .groupBy("predicate")
            .agg(
                F.mean("numeric_value").alias("mu"),
                F.stddev("numeric_value").alias("sigma"),
            )
            .withColumn(
                "sigma",
                F.when(
                    _is_finite(F.col("sigma")) & (F.col("sigma") != 0.0),
                    F.col("sigma"),
                ).otherwise(F.lit(1.0)),
            )
            .withColumn(
                "mu",
                F.when(_is_finite(F.col("mu")), F.col("mu"))
                .otherwise(F.lit(0.0)),
            )
        )

        return stats

    # ================================================================
    # Type URI mapping
    # ================================================================

    def _get_type_uri_mapping(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Get all rdf:type URIs for each node.

        Returns DataFrame(node_type, node_id, type_uri) with all
        rdf:type URIs per node.
        """
        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        type_uri_df = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select(
                F.col("subject").alias("_node_uri"),
                F.col("object").alias("type_uri"),
            )
            .join(node_lookup, "_node_uri", "inner")
            .drop("_node_uri")
            .select("node_type", "node_id", "type_uri")
            .distinct()
        )

        return type_uri_df

    # ================================================================
    # Per-type vector assembly (on executors, collect to driver)
    # ================================================================

    def _collect_and_scatter(
        self,
        combined: DataFrame,
        tensor: np.ndarray,
        vector_dim: int,
    ) -> None:
        """
        Collect all sparse entries and scatter into dense array.
        Used for small-to-medium node types.
        """
        pdf = combined.toPandas()
        scatter_sparse_entries(pdf, tensor, "node_id", vector_dim)
        del pdf
        gc.collect()

    def _collect_and_scatter_chunked(
        self,
        combined: DataFrame,
        tensor: np.ndarray,
        num_nodes: int,
        vector_dim: int,
    ) -> None:
        """
        Collect sparse entries in chunks by node_id range and scatter
        into the pre-allocated dense array incrementally.

        Does not cache (#346). ``combined`` is a filter over the all-types
        frame the caller already cached, so caching it here stores the same
        rows twice. The type that takes this path holds 10,153,351 of the
        graph's 10,348,460 nodes, so the second copy was ~98% of the first and
        the block manager held roughly two full copies against a 16g heap.
        That is what the eviction rounds and the wedged executor came out of.
        Reading the caller's cache costs one predicate per chunk instead.

        The entry total is summed from the chunks. Taking it with a ``count()``
        first read every row for a log line -- a whole extra pass.
        """
        chunk_size = self._chunk_threshold
        num_chunks = (num_nodes + chunk_size - 1) // chunk_size

        logger.info(
            f"      Chunked collection: {num_nodes:,} nodes, "
            f"{num_chunks} chunk(s) of {chunk_size:,}"
        )

        total_entries = 0

        for chunk_idx in range(num_chunks):
            lo = chunk_idx * chunk_size
            hi = min(lo + chunk_size, num_nodes)

            chunk_df = combined.filter(
                (F.col("node_id") >= lo) & (F.col("node_id") < hi)
            )

            pdf = chunk_df.toPandas()
            total_entries += len(pdf)
            scatter_sparse_entries(pdf, tensor, "node_id", vector_dim)
            del pdf
            gc.collect()

            if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
                logger.info(
                    f"      Chunk {chunk_idx + 1}/{num_chunks} complete "
                    f"(nodes {lo:,}–{hi:,})"
                )

        logger.info(
            f"      Collected {total_entries:,} sparse entries "
            f"for {num_nodes:,} nodes"
        )

    # ================================================================
    # Metadata collection helpers
    # ================================================================
    # These methods collect small aggregated data to the driver for
    # the six metadata files. All collect() calls target DataFrames
    # with at most a few hundred rows (per-predicate stats, per-type
    # URIs, per-class hierarchy entries). No per-node or per-edge
    # data is ever collected here.

    def _collect_normalization_metadata(
        self, numeric_df: DataFrame
    ) -> None:
        """
        Collect normalization statistics to driver for metadata.

        Small collect — one row per predicate (typically <200
        predicates across all ontologies).
        """
        stats_with_counts = (
            numeric_df
            .groupBy("predicate")
            .agg(
                F.mean("numeric_value").alias("mu"),
                F.stddev("numeric_value").alias("sigma"),
                F.count("numeric_value").alias("count"),
            )
        )

        rows = collect_sorted(stats_with_counts)

        collected = []
        zero_variance = []
        for row in rows:
            mu = float(row.mu) if row.mu is not None else 0.0
            sigma = float(row.sigma) if row.sigma is not None else 0.0
            count = int(row["count"])

            if sigma == 0.0:
                zero_variance.append(row.predicate)
                sigma = 1.0

            collected.append({
                "predicate": row.predicate,
                "mu": mu,
                "sigma": sigma,
                "count": count,
            })

        self._collected_norm_stats = collected
        self._collected_zero_variance = zero_variance

        logger.info(
            f"    Collected normalization stats for "
            f"{len(collected)} predicates "
            f"({len(zero_variance)} zero-variance)"
        )

    def _collect_derivation_provenance(
        self, triples_df: DataFrame
    ) -> Dict[str, str]:
        """
        Read the mapper's prov:derivedBy statements back out of the triples.

        Everything OntologyMapper emits -- subClassOf, subPropertyOf, domain,
        range -- is an inference, and once in the graph it is shaped exactly
        like an axiom a source declared. That distinction cannot be recovered
        from the axioms themselves, so the mapper states it explicitly and
        this reads it. A build whose triples carry no marker (no mapping ran,
        or a source genuinely declared the axioms) simply gets no entry, and
        the caller reports "declared by the source" instead.

        Returns {provenance_subject_uri: derivation description}.
        """
        rows = collect_sorted(
            triples_df
            .filter(F.col("predicate") == PROV_DERIVED_BY)
            .select("subject", "object")
            .distinct()
        )
        return {row["subject"]: row["object"] for row in rows}

    def _collect_route_markers(
        self, triples_df: DataFrame, routes: Tuple[str, ...]
    ) -> Dict[Tuple[str, str], str]:
        """
        Read back which route produced each individual derived axiom.

        Returns {(subject, object): route_predicate} over the given routes.
        Bounded by the number of derived axioms (tens to low hundreds), not by
        the triple count -- one marker per axiom, emitted by the mapper.
        """
        rows = collect_sorted(
            triples_df
            .filter(F.col("predicate").isin(list(routes)))
            .select("subject", "predicate", "object")
            .distinct()
        )
        return {
            (row["subject"], row["object"]): row["predicate"] for row in rows
        }

    def _property_schema_coverage(
        self,
        triples_df: DataFrame,
        prop_schema_map: Dict[str, Dict[str, Optional[str]]],
    ) -> Dict[str, Any]:
        """
        How much of the vocabulary actually got a domain and a range.

        Derivation is deliberately conservative -- a predicate used on several
        classes gets no domain, because rdfs:domain is an intersection and
        asserting two would say every subject is both. That is the right call
        but it leaves gaps, and a gap nobody enumerated is indistinguishable
        from a bug. So the predicates without each axiom are published by
        name, capped for file size with the full counts kept.

        Counted over the predicates that actually carry data: rdf:type, the
        axiom predicates and the provenance markers describe the vocabulary
        rather than use it and were never candidates.
        """
        data_predicates = [
            row["predicate"]
            for row in collect_sorted(
                triples_df
                .filter(~F.col("predicate").isin(list(_NON_FEATURE_PREDICATES)))
                .select("predicate")
                .distinct()
            )
        ]

        with_domain = sorted(
            p for p in data_predicates
            if (prop_schema_map.get(p) or {}).get("domain")
        )
        with_range = sorted(
            p for p in data_predicates
            if (prop_schema_map.get(p) or {}).get("range")
        )
        without_domain = sorted(set(data_predicates) - set(with_domain))
        without_range = sorted(set(data_predicates) - set(with_range))

        total = len(data_predicates)
        return {
            "data_predicates": total,
            "with_domain": len(with_domain),
            "with_range": len(with_range),
            "domain_coverage": round(
                len(with_domain) / total, 4) if total else 0.0,
            "range_coverage": round(
                len(with_range) / total, 4) if total else 0.0,
            "without_domain": without_domain[:_COVERAGE_GAP_LIMIT],
            "without_range": without_range[:_COVERAGE_GAP_LIMIT],
            "without_domain_truncated": (
                len(without_domain) > _COVERAGE_GAP_LIMIT
            ),
            "without_range_truncated": (
                len(without_range) > _COVERAGE_GAP_LIMIT
            ),
        }

    def _collect_ontology_schema_metadata(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
        class_hierarchy_df: DataFrame,
        property_schema_df: DataFrame,
        property_hierarchy_df: Optional[DataFrame] = None,
    ) -> None:
        """
        Collect ontology schema snapshot for metadata.

        All collect() calls here are on small distinct/aggregated
        DataFrames:
        - type_uri distinct: ~500 rows (one per type URI)
        - type_uri per node_type: ~1000 rows (types × multi-type)
        - class_hierarchy: ~5000 rows (classes × depth, transitive)
        - property_schema: ~500 rows (properties with domain/range)
        """
        # Collect type URI → PyG name mapping.
        # One row per distinct type URI — typically <500.
        type_mapping_rows = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select(F.col("object").alias("type_uri"))
            .distinct()
        )
        type_mapping_rows = collect_sorted(type_mapping_rows)

        uri_to_pyg: Dict[str, str] = {}
        for row in type_mapping_rows:
            uri = row.type_uri
            for ns, prefix in NAMESPACE_PREFIXES:
                if uri.startswith(ns):
                    local = uri[len(ns):].strip("/#")
                    if local:
                        uri_to_pyg[uri] = f"{prefix}_{local}"
                    break

        # Collect per-node-type source URI. Shared with the mapping
        # graph_schema.json publishes: the two files had independent copies of
        # this resolution and both kept the alphabetically smallest rdf:type,
        # which is not necessarily the class the node type was named after.
        # Sharing it also keeps the two artifacts from disagreeing.
        type_uri_map = build_type_uri_mapping(triples_df, node_id_df)

        # Collect class hierarchy — transitive closure.
        # Typically ~5000 rows (500 classes × avg depth ~10).
        hierarchy_rows = (
            collect_sorted(class_hierarchy_df)
            if class_hierarchy_df.head(1)
            else []
        )
        hierarchy_map: Dict[str, List[Tuple[str, int]]] = {}
        for row in hierarchy_rows:
            cls = row.class_uri
            if cls not in hierarchy_map:
                hierarchy_map[cls] = []
            hierarchy_map[cls].append(
                (row.superclass_uri, int(row.depth))
            )

        # Collect property schema — one row per property.
        # Typically <500 rows.
        prop_schema_rows = (
            collect_sorted(property_schema_df)
            if property_schema_df.head(1)
            else []
        )
        prop_schema_map: Dict[str, Dict[str, Optional[str]]] = {}
        for row in prop_schema_rows:
            prop_schema_map[row.property_uri] = {
                "domain": (
                    row.domain_uri if row.domain_uri else None
                ),
                "range": (
                    row.range_uri if row.range_uri else None
                ),
            }

        # The DIRECT axioms of each set — what a route marker can attach to.
        # The class hierarchy is stored as a transitive closure, so only
        # depth 1 is an asserted edge; the deeper entries are this pipeline's
        # own closure over those and have no independent provenance.
        direct_superclasses = {
            (cls, sup)
            for cls, supers in hierarchy_map.items()
            for sup, depth in supers
            if depth == 1
        }
        direct_super_properties = set()
        if property_hierarchy_df is not None:
            direct_super_properties = {
                (row["property_uri"], row["super_property_uri"])
                for row in collect_sorted(property_hierarchy_df)
            }
        declared_domains = {
            (prop, schema["domain"])
            for prop, schema in prop_schema_map.items()
            if schema.get("domain")
        }
        declared_ranges = {
            (prop, schema["range"])
            for prop, schema in prop_schema_map.items()
            if schema.get("range")
        }

        # Which route produced each individual axiom, and therefore how many
        # of each set the sources actually declared. A derived rdfs:subClassOf
        # is byte-identical to a declared one here, so without the markers the
        # only honest count is "34 subClassOf triples exist" -- which reads as
        # "the sources declare 34" and is wrong for every one of them.
        hierarchy_routes = self._collect_route_markers(
            triples_df,
            (PROV_ROUTE_CURATED_SUBCLASS, PROV_ROUTE_NAMED_SUBCLASS),
        )
        property_hierarchy_routes = self._collect_route_markers(
            triples_df, (PROV_ROUTE_CURATED_SUBPROPERTY,)
        )
        domain_routes = self._collect_route_markers(
            triples_df, (PROV_ROUTE_OBSERVED_DOMAIN,)
        )
        range_routes = self._collect_route_markers(
            triples_df,
            (PROV_ROUTE_OBSERVED_RANGE, PROV_ROUTE_DATATYPE_RANGE),
        )

        hierarchy_counts = _route_counts(direct_superclasses, hierarchy_routes)
        property_hierarchy_counts = _route_counts(
            direct_super_properties, property_hierarchy_routes
        )
        domain_counts = _route_counts(declared_domains, domain_routes)
        range_counts = _route_counts(declared_ranges, range_routes)

        # Build per-node-type schema entries
        node_type_schemas: Dict[str, Dict[str, Any]] = {}
        for pyg_name in node_counts:
            source_uri = type_uri_map.get(pyg_name, "")
            namespace = ""
            for ns, prefix in NAMESPACE_PREFIXES:
                if source_uri.startswith(ns):
                    namespace = ns
                    break

            superclass_chain = []
            if source_uri in hierarchy_map:
                sorted_supers = sorted(
                    hierarchy_map[source_uri], key=lambda x: x[1]
                )
                # Provenance per link, not just per set. Reading this file is
                # how someone chasing a wrong superclass finds out whether to
                # go and edit CLASS_MAPPINGS or go and fix a naming rule.
                # Depth > 1 has no route of its own: it is this pipeline's
                # transitive closure over the direct edges, so it inherits
                # their trust rather than claiming any.
                superclass_chain = [
                    {
                        "uri": uri,
                        "depth": depth,
                        "provenance": (
                            PROV_ROUTE_LABELS.get(
                                hierarchy_routes.get((source_uri, uri)),
                                "declared",
                            )
                            if depth == 1
                            else "transitive closure of the direct edges"
                        ),
                    }
                    for uri, depth in sorted_supers
                ]

            defined_properties = []
            for prop_uri, schema in prop_schema_map.items():
                if schema.get("domain") == source_uri:
                    defined_properties.append({
                        "property_uri": prop_uri,
                        "range": schema.get("range"),
                    })

            node_type_schemas[pyg_name] = {
                "source_type_uri": source_uri,
                "superclass_chain": superclass_chain,
                "namespace": namespace,
                "defined_properties": defined_properties,
            }

        provenance = self._collect_derivation_provenance(triples_df)
        coverage = self._property_schema_coverage(triples_df, prop_schema_map)
        has_property_hierarchy = bool(
            property_hierarchy_df is not None
            and property_hierarchy_df.head(1)
        )

        # Why a chain is empty is not recoverable from an empty list: "no
        # source declares a hierarchy" and "this class sits at the root of one"
        # both serialize to []. Record the distinction so a zero
        # class_hierarchy sub-segment is diagnosable from the artifact rather
        # than from the encoder source.
        #
        # ontology_mapping_enabled closes the third reading, which the other
        # two cannot express: the hierarchy was never computed at all. Since
        # OntologyMapper is what derives rdfs:subClassOf, a build it never
        # touched has an empty hierarchy no matter what the sources declare,
        # and the file has to say which of the two it is looking at.
        self._collected_ontology_schema = {
            # 1.1: adds ontology_mapping_enabled / ontology_mapping_evidence.
            # A 1.0 file with an all-empty hierarchy is genuinely ambiguous
            # and stays that way -- the version is what lets a consumer
            # require the flag rather than test for its key.
            # 1.2: adds provenance, property_hierarchy_source and
            # property_schema_coverage, once domain/range/subPropertyOf
            # started being DERIVED rather than read. Before that every axiom
            # in the file was declared by a source, so provenance was not a
            # question a consumer had to ask; now it is.
            # 1.3: *_source now names the derivation route with its count
            # instead of the predicate, and derived_axioms lists the edges of
            # each route. "rdfs:subClassOf" was true of the predicate the
            # encoder read and false as an answer to "did a source declare
            # this", which is the question anyone reading the field is
            # actually asking.
            "version": "1.3",
            "ontology_mapping_enabled": self._ontology_mapping_ran,
            "ontology_mapping_evidence": (
                "owl:equivalentProperty/owl:equivalentClass present in "
                "build input"
                if self._ontology_mapping_ran
                else "no owl:equivalentProperty/owl:equivalentClass in "
                     "build input"
            ),
            # The empty forms are untouched -- #273's "no rdfs:subClassOf in
            # source data" reasoning is about data that is absent, which no
            # amount of provenance changes.
            "hierarchy_source": (
                _axiom_source("rdfs:subClassOf", hierarchy_counts)
                if hierarchy_map
                else _empty_hierarchy_reason(self._ontology_mapping_ran)
            ),
            "property_schema_source": (
                _axiom_source(
                    "rdfs:domain/rdfs:range",
                    _merge_counts(domain_counts, range_counts),
                )
                if prop_schema_map
                else _empty_property_schema_reason(self._ontology_mapping_ran)
            ),
            "property_hierarchy_source": (
                _axiom_source(
                    "rdfs:subPropertyOf", property_hierarchy_counts
                )
                if has_property_hierarchy
                else _empty_property_hierarchy_reason(
                    self._ontology_mapping_ran
                )
            ),
            # Per route: how many axioms it produced and which ones. Counts
            # alone answer "did a source declare any of this" (declared: 0 on
            # every build since #273); the edge lists answer "is THIS edge a
            # person's judgement or a spelling rule's guess", which is what a
            # reader chasing AllItemsLessShelter -> Shelter needs and cannot
            # get from a count.
            "derived_axioms": {
                "class_hierarchy": _route_detail(
                    direct_superclasses, hierarchy_routes, hierarchy_counts
                ),
                "property_hierarchy": _route_detail(
                    direct_super_properties, property_hierarchy_routes,
                    property_hierarchy_counts,
                ),
                "property_domain": _route_detail(
                    declared_domains, domain_routes, domain_counts
                ),
                "property_range": _route_detail(
                    declared_ranges, range_routes, range_counts
                ),
            },
            # Where each axiom set came from. An axiom this pipeline inferred
            # is shaped exactly like one a source declared, and the two are
            # not interchangeable -- "used on class C in this data" is weaker
            # than "its domain is C". Absent a marker the axioms were not
            # ours, so they are reported as declared rather than guessed at.
            "provenance": {
                key: provenance.get(uri, default)
                for key, uri, default in (
                    ("class_hierarchy", PROV_CLASS_HIERARCHY,
                     _DECLARED if hierarchy_map else _NOT_DERIVED),
                    ("property_hierarchy", PROV_PROPERTY_HIERARCHY,
                     _DECLARED if has_property_hierarchy else _NOT_DERIVED),
                    ("property_domain", PROV_PROPERTY_DOMAIN,
                     _DECLARED if prop_schema_map else _NOT_DERIVED),
                    ("property_range", PROV_PROPERTY_RANGE,
                     _DECLARED if prop_schema_map else _NOT_DERIVED),
                )
            },
            # Which predicates got an axiom and which did not, so a gap is a
            # named set rather than something to be discovered by noticing
            # that a sub-segment is thinner than expected.
            "property_schema_coverage": coverage,
            "node_types": node_type_schemas,
            "uri_to_pyg_name": uri_to_pyg,
            "namespace_prefixes": {
                prefix: ns for ns, prefix in NAMESPACE_PREFIXES
            },
        }

        logger.info(
            f"    Collected ontology schema for "
            f"{len(node_type_schemas)} node types, "
            f"{len(uri_to_pyg)} URI mappings; ontology mapping "
            f"{'ran' if self._ontology_mapping_ran else 'did NOT run'} "
            f"on this input"
        )

    def _collect_slot_mapping_metadata(
        self,
        numeric_df: Optional[DataFrame],
        categorical_df: Optional[DataFrame],
        type_uri_df: DataFrame,
        class_hierarchy_df: DataFrame,
    ) -> None:
        """
        Collect slot mapping metadata — maps vector dimensions back
        to their semantic meaning.

        All collect() calls are on distinct single-column DataFrames:
        - numeric predicates: typically <100
        - categorical predicates: typically <100
        - type URIs: typically <500
        - superclass URIs: typically <200

        Slots are resolved by evaluating the encoder's own expression
        (``_slot_dim``) over the distinct values, so a published column is the
        column the value is in. They used to be recomputed here on the driver
        with md5, which matched Spark's hash only by coincidence and named the
        wrong column for most properties (#354).
        """
        layout = self._layout

        # --- Numeric property slots ---
        numeric_slots = []
        if numeric_df is not None:
            for pred, dims in self._resolve_slots(
                numeric_df,
                "predicate",
                [500],
                layout.seg3_numeric_dim,
                layout.seg3_numeric_start,
            ):
                numeric_slots.append({
                    "predicate_uri": pred,
                    "local_name": _local_name(pred),
                    "hash_slot": dims[0] - layout.seg3_numeric_start,
                    "global_dim": dims[0],
                })

        # --- Categorical property slots ---
        # No columns per predicate here. The encoder keys a categorical slot on
        # "predicate::value", so every distinct value of a predicate lands
        # somewhere different and a predicate has no fixed set of columns to
        # publish. Four were published per predicate anyway, naming columns
        # that hold some other value (#354).
        #
        # The rule below lets a reader work out the columns for a value they
        # care about. Listing every value instead would mean collecting the
        # whole categorical vocabulary to the driver, which does not fit.
        categorical_slots = []
        if categorical_df is not None:
            for row in collect_sorted(
                categorical_df.select("predicate").distinct()
            ):
                categorical_slots.append({
                    "predicate_uri": row.predicate,
                    "local_name": _local_name(row.predicate),
                })
        categorical_rule = {
            "keyed_on": (
                "{predicate_uri}" + _CATEGORICAL_KEY_SEP + "{value}"
            ),
            "seeds": [s + 600 for s in _HASH_SEEDS],
            "segment_start": layout.seg3_categorical_start,
            "segment_dim": layout.seg3_categorical_dim,
            "expression": (
                "abs(spark_hash(keyed_on, seed)) "
                "% segment_dim + segment_start"
            ),
        }

        # --- Class identity slots ---
        ci_start = layout.seg1_class_identity_start
        class_slots = []
        for uri, dims in self._resolve_slots(
            type_uri_df,
            "type_uri",
            _HASH_SEEDS,
            layout.seg1_class_identity_dim,
            ci_start,
        ):
            pyg_name = ""
            for ns, prefix in NAMESPACE_PREFIXES:
                if uri.startswith(ns):
                    local = uri[len(ns):].strip("/#")
                    if local:
                        pyg_name = f"{prefix}_{local}"
                    break

            class_slots.append({
                "class_uri": uri,
                "pyg_name": pyg_name,
                "hash_slots": [d - ci_start for d in dims],
                "global_dims": dims,
            })

        # --- Namespace slots ---
        # A namespace occupies two columns, not one. Membership read from a
        # node's rdf:type lands at `slot` with weight 1.0; membership read
        # from the node's own URI lands half a segment away with weight 0.5
        # (_encode_node_uri_namespace). Only the first was ever published, so
        # the second column looked unclaimed (#354).
        #
        # These are index-based, not hashed, so they were already correct as
        # far as they went -- the gap here is the missing column, not a wrong
        # one.
        os_start = layout.seg1_ontology_source_start
        os_dim = layout.seg1_ontology_source_dim
        namespace_slots = []
        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            prefix = ""
            for ns, p in NAMESPACE_PREFIXES:
                if ns == namespace:
                    prefix = p
                    break
            slot = onto_idx % os_dim
            node_uri_slot = (onto_idx + os_dim // 2) % os_dim
            namespace_slots.append({
                "namespace": namespace,
                "prefix": prefix,
                "slot": slot,
                "global_dim": slot + os_start,
                "node_uri_slot": node_uri_slot,
                "node_uri_global_dim": node_uri_slot + os_start,
            })

        # --- Superclass hierarchy slots ---
        ch_start = layout.seg1_class_hierarchy_start
        hierarchy_slots = []
        if class_hierarchy_df.head(1):
            for uri, dims in self._resolve_slots(
                class_hierarchy_df,
                "superclass_uri",
                [s + 100 for s in _HASH_SEEDS[:2]],
                layout.seg1_class_hierarchy_dim,
                ch_start,
            ):
                hierarchy_slots.append({
                    "superclass_uri": uri,
                    "hash_slots": [d - ch_start for d in dims],
                    "global_dims": dims,
                })

        # --- Collision report ---
        collision_report = compute_collision_report(
            numeric_slots, categorical_slots, class_slots,
            namespace_slots, hierarchy_slots,
            class_identity_dim=layout.seg1_class_identity_dim,
        )
        check_class_identity_capacity(
            collision_report,
            allow_oversubscription=self._allow_class_oversubscription,
            vector_dim=layout.vector_dim,
        )

        self._collected_slot_mapping = {
            # 1.1: class_identity in collision_report reports code
            # separability (distinct_codes, linearly_separable, headroom)
            # instead of slot occupancy. The 1.0 keys `collisions` /
            # `collision_rate` counted multi-hot slot reuse, which pigeonhole
            # forces high on a healthy code -- they are now `slot_reuse` /
            # `slot_reuse_rate` so the number cannot be read as lost identity.
            #
            # 1.2: every published column now comes from the encoder's own
            # expression instead of a driver-side md5, so the columns named
            # here are the columns the values are in. Three shape changes came
            # with it: `categorical_properties` entries no longer carry
            # `hash_slots` / `global_dims` (a categorical slot is keyed on
            # predicate AND value, so a predicate has no fixed columns -- see
            # `categorical_slot_rule`), and `namespaces` entries gained
            # `node_uri_slot` / `node_uri_global_dim` for the second column
            # each namespace occupies (#354).
            "version": "1.2",
            "numeric_properties": numeric_slots,
            "categorical_properties": categorical_slots,
            "categorical_slot_rule": categorical_rule,
            "classes": class_slots,
            "superclasses": hierarchy_slots,
            "namespaces": namespace_slots,
            "collision_report": collision_report,
        }

        logger.info(
            f"    Collected slot mappings: "
            f"{len(numeric_slots)} numeric, "
            f"{len(categorical_slots)} categorical, "
            f"{len(class_slots)} classes, "
            f"{len(hierarchy_slots)} superclasses, "
            f"{len(namespace_slots)} namespaces"
        )

    def _resolve_slots(
        self,
        df: DataFrame,
        value_col: str,
        seeds: List[int],
        dim: int,
        start: int,
    ) -> List[Tuple[str, List[int]]]:
        """Each distinct value and the columns it lands in, one per seed.

        Evaluated by Spark through ``_slot_dim`` -- the same expression the
        encoder places values with -- so the published column is the column
        holding the value, rather than a second guess at it.

        One ``collect()`` per slot kind, same as before, on a distinct
        single-column frame of at most a few thousand rows. Ordered by
        ``collect_sorted`` because entry order drifting between runs is its
        own reproducibility bug (#221).
        """
        distinct = df.select(value_col).distinct()
        projected = distinct.select(
            F.col(value_col).alias("value"),
            F.array(
                *[
                    _slot_dim(F.col(value_col), seed, dim, start)
                    for seed in seeds
                ]
            ).alias("dims"),
        )
        return [
            (row["value"], [int(d) for d in row["dims"]])
            for row in collect_sorted(projected)
        ]

    # ================================================================
    # Segment 1: Ontology Structure encoding
    # ================================================================

    def _encode_ontology_structure(
        self,
        all_type_uris: DataFrame,
        all_nodes: DataFrame,
        class_hierarchy_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 1: class identity, class hierarchy, and
        ontology/source membership — for ALL node types in one pass.

        Every projection carries ``node_type`` so the disjoint per-type
        vectors can be split apart on the driver (node_id is 0-indexed
        within a type). All dim indices are derived from self._layout.

        Inputs:
          all_type_uris: DataFrame(node_type, node_id, type_uri) — every type
          all_nodes:     DataFrame(uri, node_id, node_type) — every type

        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 1a: Class Identity ---
        class_identity = (
            all_type_uris
            .select("node_type", "node_id", "type_uri")
            .distinct()
        )

        ci_start = layout.seg1_class_identity_start
        ci_dim = layout.seg1_class_identity_dim

        for seed_offset in _HASH_SEEDS:
            ci_encoded = class_identity.select(
                F.col("node_type"),
                F.col("node_id"),
                _slot_dim(
                    F.col("type_uri"), seed_offset, ci_dim, ci_start
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(ci_encoded)

        # --- Sub-segment 1b: Class Hierarchy ---
        ch_start = layout.seg1_class_hierarchy_start
        ch_dim = layout.seg1_class_hierarchy_dim

        if self._has_class_hierarchy:
            node_supers = (
                all_type_uris
                .select(
                    F.col("node_type"),
                    F.col("node_id"),
                    F.col("type_uri").alias("class_uri"),
                )
                .join(class_hierarchy_df, "class_uri", "inner")
                .select("node_type", "node_id", "superclass_uri", "depth")
            )

            if node_supers.head(1):
                node_supers = node_supers.withColumn(
                    "weight",
                    F.lit(1.0) / F.col("depth").cast("double"),
                )

                for seed_offset in _HASH_SEEDS[:2]:
                    hier_encoded = node_supers.select(
                        F.col("node_type"),
                        F.col("node_id"),
                        _slot_dim(
                            F.col("superclass_uri"),
                            seed_offset + 100,
                            ch_dim,
                            ch_start,
                        ).alias("dim"),
                        F.col("weight").alias("value"),
                    )
                    parts.append(hier_encoded)

        # --- Sub-segment 1c: Ontology/Source Membership ---
        os_start = layout.seg1_ontology_source_start
        os_dim = layout.seg1_ontology_source_dim

        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            ns_match = (
                all_type_uris
                .filter(F.col("type_uri").startswith(namespace))
                .select("node_type", "node_id")
                .distinct()
                .withColumn(
                    "dim",
                    F.lit(os_start + (onto_idx % os_dim)),
                )
                .withColumn("value", F.lit(1.0))
            )
            parts.append(ns_match)

        # Also encode from node URI namespace
        node_ns_parts = self._encode_node_uri_namespace(
            all_nodes, os_start, os_dim
        )
        if node_ns_parts is not None:
            parts.append(node_ns_parts)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_type"),
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    def _encode_node_uri_namespace(
        self,
        all_nodes: DataFrame,
        os_start: int,
        os_dim: int,
    ) -> Optional[DataFrame]:
        """
        Encode ontology membership from the node's own URI namespace, for
        all node types at once. Carries node_type through every projection.

        Input:  all_nodes: DataFrame(uri, node_id, node_type)
        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        parts = []
        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            ns_match = (
                all_nodes
                .filter(F.col("uri").startswith(namespace))
                .select("node_type", "node_id")
                .distinct()
                .withColumn(
                    "dim",
                    F.lit(
                        os_start
                        + ((onto_idx + os_dim // 2) % os_dim)
                    ),
                )
                .withColumn("value", F.lit(0.5))
            )
            parts.append(ns_match)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result

    # ================================================================
    # Segment 2: Property Schema encoding
    # ================================================================

    def _encode_property_schema(
        self,
        all_node_props: DataFrame,
        property_schema_df: DataFrame,
        property_hierarchy_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 2: property presence, domain/range signals, and
        property hierarchy — for ALL node types in one pass. Every
        projection carries ``node_type``.

        Input:  all_node_props: DataFrame(node_type, node_id, predicate)
        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # Existence check once across all types (not once per type).
        has_any_props = bool(all_node_props.head(1))

        # --- Sub-segment 2a: Property Presence ---
        pp_start = layout.seg2_property_presence_start
        pp_dim = layout.seg2_property_presence_dim

        if has_any_props:
            for seed_offset in _HASH_SEEDS[:3]:
                pp_encoded = all_node_props.select(
                    F.col("node_type"),
                    F.col("node_id"),
                    _slot_dim(
                        F.col("predicate"),
                        seed_offset + 200,
                        pp_dim,
                        pp_start,
                    ).alias("dim"),
                    F.lit(1.0).alias("value"),
                )
                parts.append(pp_encoded)

        # --- Sub-segment 2b: Domain/Range Signals ---
        dr_start = layout.seg2_domain_range_start
        dr_dim = layout.seg2_domain_range_dim
        dr_half = max(1, dr_dim // 2)

        if has_any_props and self._has_property_schema:
            prop_with_schema = (
                all_node_props
                .select("node_type", "node_id", "predicate")
                .join(
                    property_schema_df,
                    all_node_props["predicate"]
                    == property_schema_df["property_uri"],
                    "inner",
                )
                .drop("property_uri")
            )

            if prop_with_schema.head(1):
                domain_entries = (
                    prop_with_schema
                    .filter(F.col("domain_uri").isNotNull())
                    .select(
                        F.col("node_type"),
                        F.col("node_id"),
                        _slot_dim(
                            F.col("domain_uri"), 300, dr_half, dr_start
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                )
                parts.append(domain_entries)

                range_entries = (
                    prop_with_schema
                    .filter(F.col("range_uri").isNotNull())
                    .select(
                        F.col("node_type"),
                        F.col("node_id"),
                        _slot_dim(
                            F.col("range_uri"),
                            301,
                            dr_dim - dr_half,
                            dr_start + dr_half,
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                )
                parts.append(range_entries)

        # --- Sub-segment 2c: Property Hierarchy ---
        ph_start = layout.seg2_property_hierarchy_start
        ph_dim = layout.seg2_property_hierarchy_dim

        if has_any_props and self._has_property_hierarchy:
            prop_with_super = (
                all_node_props
                .select("node_type", "node_id", "predicate")
                .join(
                    property_hierarchy_df,
                    all_node_props["predicate"]
                    == property_hierarchy_df["property_uri"],
                    "inner",
                )
                .drop("property_uri")
            )

            if prop_with_super.head(1):
                for seed_offset in _HASH_SEEDS[:2]:
                    ph_encoded = prop_with_super.select(
                        F.col("node_type"),
                        F.col("node_id"),
                        _slot_dim(
                            F.col("super_property_uri"),
                            seed_offset + 400,
                            ph_dim,
                            ph_start,
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                    parts.append(ph_encoded)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_type"),
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    # ================================================================
    # Segment 3: Literal Values encoding
    # ================================================================

    def _encode_literal_values(
        self,
        numeric_df: Optional[DataFrame],
        categorical_df: Optional[DataFrame],
        norm_stats: Optional[DataFrame],
    ) -> Optional[DataFrame]:
        """
        Encode Segment 3: numeric values in hashed slots and categorical
        values as multi-hot hash encoding — for ALL node types in one pass.
        numeric_df / categorical_df already carry node_type, so no per-type
        filtering is needed; node_type is projected through.

        Inputs:
          numeric_df:     DataFrame(node_type, node_id, predicate, numeric_value)
          categorical_df: DataFrame(node_type, node_id, predicate, cat_value)
        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 3a: Numeric Values ---
        num_start = layout.seg3_numeric_start
        num_dim = layout.seg3_numeric_dim

        if numeric_df is not None and numeric_df.head(1):
            all_numeric = numeric_df
            if norm_stats is not None and self._normalize:
                all_numeric = (
                    all_numeric
                    .join(
                        F.broadcast(norm_stats),
                        "predicate",
                        "left",
                    )
                    .withColumn(
                        "normalized_value",
                        (
                            F.col("numeric_value")
                            - F.coalesce(F.col("mu"), F.lit(0.0))
                        )
                        / F.coalesce(F.col("sigma"), F.lit(1.0)),
                    )
                    .drop("mu", "sigma")
                )
                value_col = "normalized_value"
            else:
                all_numeric = all_numeric.withColumn(
                    "normalized_value", F.col("numeric_value")
                )
                value_col = "normalized_value"

            num_encoded = all_numeric.select(
                F.col("node_type"),
                F.col("node_id"),
                _slot_dim(
                    F.col("predicate"), 500, num_dim, num_start
                ).alias("dim"),
                F.col(value_col).alias("value"),
            )
            parts.append(num_encoded)

        # --- Sub-segment 3b: Categorical Values ---
        cat_start = layout.seg3_categorical_start
        cat_dim = layout.seg3_categorical_dim

        if categorical_df is not None and categorical_df.head(1):
            for seed_offset in _HASH_SEEDS:
                cat_encoded = categorical_df.select(
                    F.col("node_type"),
                    F.col("node_id"),
                    _slot_dim(
                        _categorical_key(),
                        seed_offset + 600,
                        cat_dim,
                        cat_start,
                    ).alias("dim"),
                    F.lit(1.0).alias("value"),
                )
                parts.append(cat_encoded)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_type"),
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )
