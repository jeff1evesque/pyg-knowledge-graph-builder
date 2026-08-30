"""
Metadata Writer — Generates JSON metadata files for PyG graph builds.

Produces six metadata files that enable downstream training and inference
code to consistently use the PyG HeteroData object:

  graph_schema.json     — complete inventory of node/edge types, counts, origins
  feature_spec.json     — structure of node/edge feature vectors
  normalization.json    — per-property z-score statistics
  encoding_config.json  — deterministic hash encoding parameters
  ontology_schema.json  — frozen ontology structure snapshot
  slot_mapping.json     — dimension-to-semantic-meaning mappings

All metadata is collected during PyG construction (Steps 1-5 in
constructor.py) and written to S3 as a batch after the .pt file is saved.

Metadata files are small JSON (<1 MB each) — no driver memory concern.
"""
import json
import logging
import math
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Significant digits retained for serialized normalization statistics.
# float64 carries ~15-17; the drift this guards against appears at the
# 16th, so 12 discards only noise.
_STAT_SIGNIFICANT_DIGITS = 12


def _round_floats(obj: Any, sig: int = _STAT_SIGNIFICANT_DIGITS) -> Any:
    """Recursively round floats to `sig` significant digits.

    Aggregations like stddev are parallel reductions, and parallel
    float reductions are not order-deterministic — partitions combine
    in whatever order they finish. Two runs over identical data can
    therefore produce statistics differing in the last ULP
    (observed: std 265.6434939473693 vs 265.64349394736934). That is
    numerically meaningless — the node feature tensors are float32 and
    both values round to the same float32, which is why the tensors
    compare equal while the metadata did not — but it makes
    normalization.json non-reproducible byte-for-byte, which the
    pipeline promises and the e2e reproducibility tests assert.

    Rounding at serialization keeps the artifact stable without
    touching the statistics used to normalize features.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if obj == 0.0 or not math.isfinite(obj):
            return obj
        decimals = -int(math.floor(math.log10(abs(obj)))) + (sig - 1)
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, sig) for v in obj]
    return obj


class MetadataCollector:
    """
    Accumulates metadata artifacts during PyG construction steps.

    Each step in constructor.py calls the appropriate register_* method
    to deposit its metadata. After all steps complete, to_metadata_files()
    produces the six JSON-serializable dicts.

    This class holds only small Python dicts/lists — no tensors, no
    DataFrames, no Spark references.
    """

    def __init__(
        self,
        time_period: str,
        vector_dim: int,
        edge_vector_dim: int,
        edge_features_enabled: bool,
        config: Dict[str, Any],
        dataset: str = "",
        sources: Optional[List[str]] = None,
    ):
        self._time_period = time_period
        self._vector_dim = vector_dim
        self._edge_vector_dim = edge_vector_dim
        self._edge_features_enabled = edge_features_enabled
        self._config = config
        # Which sources the graph was built from, and the name of that
        # combination. Both default to empty because a graph built from enriched
        # output written before the dataset descriptor existed cannot know them,
        # and "unknown" must stay distinguishable from "none".
        self._dataset = dataset
        self._sources = sorted(sources) if sources else []
        self._build_timestamp = datetime.now(timezone.utc).isoformat()

        # Populated by register_* methods during construction
        self._node_counts: Dict[str, int] = {}
        self._node_type_uris: Dict[str, str] = {}
        self._node_type_categories: Dict[str, str] = {}
        # Node types whose literal_values segment carries a non-zero value.
        # None until register_node_literal_features() is called; see
        # _build_graph_schema for what that distinction means in the output.
        self._node_types_with_literal_features: Optional[set] = None
        self._edge_counts: Dict[str, int] = {}
        self._edge_type_details: Dict[str, Dict[str, Any]] = {}
        self._node_layout: Optional[Dict[str, Any]] = None
        self._edge_layout: Optional[Dict[str, Any]] = None
        self._norm_stats: Optional[List[Dict[str, Any]]] = None
        self._edge_norm_stats: Optional[Dict[str, List[Dict]]] = None
        self._encoding_config: Optional[Dict[str, Any]] = None
        self._ontology_schema: Optional[Dict[str, Any]] = None
        self._slot_mapping: Optional[Dict[str, Any]] = None
        self._sub_segment_status: Dict[str, Optional[str]] = {}
        self._edge_feature_categories: Dict[str, str] = {}
        self._edge_types_with_features: List[str] = []
        self._edge_types_without_features: List[str] = []
        self._zero_variance_properties: List[str] = []

    # ================================================================
    # Registration methods — called by constructor.py during each step
    # ================================================================

    def register_node_types(
        self,
        node_counts: Dict[str, int],
        node_type_uris: Dict[str, str],
        node_type_categories: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register node type information from NodeMapper (Step 1).

        Args:
            node_counts: Dict[pyg_name -> count]
            node_type_uris: Dict[pyg_name -> source_type_uri]
            node_type_categories: Optional Dict[pyg_name -> category]
                where category is one of: measurement, observation,
                temporal, structural, entity
        """
        self._node_counts = dict(node_counts)
        self._node_type_uris = dict(node_type_uris)
        self._node_type_categories = dict(node_type_categories or {})

    def register_node_literal_features(
        self,
        node_types_with_literal_features: Iterable[str],
    ) -> None:
        """
        Register which node types actually carry literal-value features (Step 3).

        Separate from register_node_types because that runs at Step 1, before
        any feature tensor exists. This is called after FeatureExtractor has
        built them.

        Args:
            node_types_with_literal_features: pyg_names whose literal_values
                segment holds at least one non-zero value
        """
        self._node_types_with_literal_features = set(
            node_types_with_literal_features
        )

    def register_edge_types(
        self,
        edge_counts: Dict[Tuple[str, str, str], int],
        edge_predicate_uris: Dict[str, str],
        edge_origins: Optional[Dict[Tuple[str, str, str], str]] = None,
        edge_feature_flags: Optional[Dict[str, bool]] = None,
        edge_feature_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Register edge type information from EdgeMapper (Step 2).

        Args:
            edge_counts: Dict[(src, rel, dst) -> num_edges]
            edge_predicate_uris: Dict[relation_name -> predicate_uri]
            edge_origins: Optional Dict[(src, rel, dst) -> origin] where
                origin is one of: raw, enrichment, unification
                (see rdf_utils.classify_edge_origin). Three values, not
                the four once listed here: intra- and cross-source enrichment
                share a namespace, so nothing available at this layer can
                separate them, and promising a distinction that is never
                emitted is the same defect as the "unknown" default below.

                Keyed by the FULL edge type, not the relation name alone.
                classify_edge_origin reads the endpoints as well as the
                predicate, so one relation legitimately has different origins
                per endpoint pair: jolts:hasIndustry is raw from jolts_HiresLevel
                but enrichment from the pipeline-minted
                bls_enrichment_RateMeasurement. Keyed by name, those collapsed to
                whichever the caller built last -- 3 edge types (90 edges) on the
                e2e fixtures were stamped `raw` for links this pipeline inferred,
                the "observed fact" mislabelling classify_edge_origin exists to
                prevent. A relation-name key cannot express the distinction, so
                the key has to carry the endpoints.
            edge_feature_flags: Dict[relation_name -> has_features]
            edge_feature_dims: Dict[relation_name -> feature_dim]
        """
        for (src, rel, dst), count in edge_counts.items():
            key = f"({src}, {rel}, {dst})"
            self._edge_counts[key] = count
            self._edge_type_details[key] = {
                "src_type": src,
                "relation": rel,
                "dst_type": dst,
                "count": count,
                "predicate_uri": edge_predicate_uris.get(rel, ""),
                "origin": (edge_origins or {}).get((src, rel, dst), "unknown"),
                "has_features": (edge_feature_flags or {}).get(rel, False),
                "feature_dim": (edge_feature_dims or {}).get(rel, 0),
                # Schema 1.2. The group a consumer should share weights over —
                # see _build_relation_groups. Same value as `relation`, named
                # separately because it answers a different question ("what may
                # be tied?") and a later change could widen it beyond the
                # relation name without repurposing an existing field.
                "relation_group": rel,
            }

    def register_node_feature_layout(
        self, layout_dict: Dict[str, Any]
    ) -> None:
        """
        Register node feature vector layout from FeatureExtractor (Step 3).

        Args:
            layout_dict: Serialized VectorLayout with segment boundaries
        """
        self._node_layout = layout_dict

    def register_edge_feature_layout(
        self, layout_dict: Dict[str, Any]
    ) -> None:
        """
        Register edge feature vector layout from EdgeFeatureExtractor (Step 4).

        Args:
            layout_dict: Serialized EdgeVectorLayout with segment boundaries
        """
        self._edge_layout = layout_dict

    def register_normalization_stats(
        self,
        stats: List[Dict[str, Any]],
        zero_variance_properties: Optional[List[str]] = None,
    ) -> None:
        """
        Register per-property normalization statistics from
        FeatureExtractor (Step 3).

        Args:
            stats: List of dicts with keys: predicate, mu, sigma, count
            zero_variance_properties: Properties where sigma was 0
                (set to constant 1.0)
        """
        self._norm_stats = stats
        self._zero_variance_properties = zero_variance_properties or []

    def register_edge_normalization_stats(
        self,
        edge_stats: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """
        Register per-edge-type derived feature normalization stats
        from EdgeFeatureExtractor (Step 4).

        Args:
            edge_stats: Dict[edge_type_key_str -> list of stat dicts]
                Each stat dict has: property, mu, sigma
        """
        self._edge_norm_stats = edge_stats

    def register_encoding_config(
        self, encoding_config: Dict[str, Any]
    ) -> None:
        """
        Register encoding configuration from FeatureExtractor and
        EdgeFeatureExtractor (Steps 3-4).

        Args:
            encoding_config: All hash seeds, dimensions, algorithm
                parameters needed to reproduce the encoding
        """
        self._encoding_config = encoding_config

    def register_ontology_schema(
        self, ontology_schema: Dict[str, Any]
    ) -> None:
        """
        Register frozen ontology structure from FeatureExtractor (Step 3).

        Args:
            ontology_schema: Per-node-type class hierarchy, property
                definitions, namespace mappings
        """
        self._ontology_schema = ontology_schema

    def register_slot_mapping(
        self, slot_mapping: Dict[str, Any]
    ) -> None:
        """
        Register dimension-to-semantic-meaning mappings from
        FeatureExtractor (Step 3).

        Args:
            slot_mapping: Per-property, per-class, per-namespace
                slot assignments with collision report
        """
        self._slot_mapping = slot_mapping

    def register_sub_segment_status(
        self,
        status: Dict[str, Optional[str]],
    ) -> None:
        """
        Register why each node sub-segment does or does not carry signal.

        Args:
            status: Dict[sub_segment_name -> None if populated, else the
                reason it is empty]. A sub-segment absent from this dict is
                reported as carrying features.
        """
        self._sub_segment_status = status

    def register_edge_feature_classification(
        self,
        categories: Dict[str, str],
        types_with_features: List[str],
        types_without_features: List[str],
    ) -> None:
        """
        Register edge type classification from EdgeFeatureExtractor (Step 4).

        Args:
            categories: Dict[relation_name -> category]
            types_with_features: List of edge type key strings that got features
            types_without_features: List of edge type key strings without features
        """
        self._edge_feature_categories = categories
        self._edge_types_with_features = types_with_features
        self._edge_types_without_features = types_without_features

    # ================================================================
    # Output generation
    # ================================================================

    def to_metadata_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Produce all six metadata files as JSON-serializable dicts.

        Returns:
            Dict mapping filename -> content dict
        """
        return {
            "graph_schema.json": self._build_graph_schema(),
            "feature_spec.json": self._build_feature_spec(),
            "normalization.json": self._build_normalization(),
            "encoding_config.json": self._build_encoding_config(),
            "ontology_schema.json": self._build_ontology_schema(),
            "slot_mapping.json": self._build_slot_mapping(),
        }

    # ================================================================
    # File builders
    # ================================================================

    def _build_graph_schema(self) -> Dict[str, Any]:
        """Build graph_schema.json content.

        ``has_features`` on a node type means it carries LITERAL-value features
        -- a non-zero literal_values segment (the third and last segment of the
        node vector).

        It cannot usefully mean "a feature tensor exists": constructor.py gives
        every node type an ``x``, falling back to a zeros placeholder, so that
        reading would be true for every entry and carry no information. Nor is
        "any non-zero value" useful -- the ontology_structure segment is
        populated for every typed node, so that is true for every entry too.
        Literal values are the segment that actually varies: on the e2e
        fixtures 76 of 100 node types have them, the other 24 being pure
        taxonomy types (EconomicSector, GeographicRegion, TimePeriod, ...)
        that carry structure but no measurements.

        This is a CHANGED MEANING as of schema version 1.1. Through 1.0 the
        field was ``count > 0`` -- the node count, not features at all -- so it
        was true for essentially every type, and
        ``summary.node_types_with_literal_features`` was identical to
        ``total_node_types`` by construction.

        Caveat worth knowing: a type whose literal values all normalize to
        exactly 0.0 (a constant numeric property under z-score) reads as
        False here. That is the honest answer for a consumer asking "does this
        type carry usable literal signal", but it is not the same question as
        "were literals present in the source".
        """
        with_literals = self._node_types_with_literal_features

        # Node types. `index` is assigned by sorted name so it is stable across
        # runs on the same data and reproducible from the file alone; the edge
        # entries below refer to it, and a consumer embedding endpoint types
        # indexes a table with it.
        node_types = {}
        for index, (pyg_name, count) in enumerate(
            sorted(self._node_counts.items())
        ):
            node_types[pyg_name] = {
                "index": index,
                "count": count,
                "source_type_uri": self._node_type_uris.get(pyg_name, ""),
                "category": self._node_type_categories.get(
                    pyg_name, "entity"
                ),
                # Unregistered (a caller that never reached feature building)
                # reports False rather than guessing: an absent fact must not
                # masquerade as a positive one.
                "has_features": (
                    pyg_name in with_literals
                    if with_literals is not None
                    else False
                ),
            }

        # Edge types, each carrying the endpoint indices assigned above so a
        # consumer can condition a shared relation weight on endpoint type
        # without re-deriving the vocabulary.
        edge_types = {}
        for key, details in sorted(self._edge_type_details.items()):
            entry = dict(details)
            src = entry.get("src_type", "")
            dst = entry.get("dst_type", "")
            entry["src_type_index"] = node_types.get(src, {}).get("index", -1)
            entry["dst_type_index"] = node_types.get(dst, {}).get("index", -1)
            edge_types[key] = entry

        relation_groups = self._build_relation_groups(edge_types)

        # Summary statistics
        total_nodes = sum(self._node_counts.values())
        total_edges = sum(self._edge_counts.values())
        node_types_with_features = sum(
            1 for entry in node_types.values() if entry["has_features"]
        )
        edge_types_with_features = len(self._edge_types_with_features)

        return {
            # 1.3: adds `build_metadata.dataset` and `build_metadata.sources`.
            # Additive only; both are empty strings/lists when the enriched
            # output carries no dataset descriptor, which is every graph built
            # before this field existed.
            #
            # 1.2: adds `relation_groups`, per-edge `relation_group` /
            # `src_type_index` / `dst_type_index`, and per-node `index`.
            # Additive only -- every 1.1 field keeps its name and meaning.
            #
            # 1.1: node-type `has_features` now means "carries literal-value
            # features". Through 1.0 it was `count > 0` -- the node count --
            # which made it, and summary.node_types_with_literal_features,
            # true/total for every build. See _build_graph_schema's docstring.
            "version": "1.3",
            "build_metadata": {
                "time_period": self._time_period,
                # What the graph was built from. Until 1.3 this file recorded the
                # period and the feature config but nothing about its inputs, so
                # "does this graph include market data?" could only be answered by
                # finding the enrichment manifest in a sibling directory -- a fact
                # the file should own, since files get copied and paths do not
                # travel with them.
                #
                # Short labels from source_label(), never the source URIs: this
                # file is published, and the URIs carry deployment detail that
                # config.source_paths already records in a field readers know to
                # treat as sensitive.
                "dataset": self._dataset,
                "sources": self._sources,
                "build_timestamp": self._build_timestamp,
                "pipeline_config": _sanitize_config(self._config),
            },
            "node_types": node_types,
            "edge_types": edge_types,
            "relation_groups": relation_groups,
            "summary": {
                "total_node_types": len(self._node_counts),
                "total_edge_types": len(self._edge_type_details),
                "total_relation_groups": len(relation_groups),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_types_with_literal_features": node_types_with_features,
                "edge_types_with_features": edge_types_with_features,
                "edge_types_without_features": len(
                    self._edge_types_without_features
                ),
            },
        }

    @staticmethod
    def _build_relation_groups(
        edge_types: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Which edge types are the same relation, and may share GNN weights.

        A PyG edge type is (src_type, relation, dst_type), and a heterogeneous
        conv allocates one weight matrix per edge type. That key is not a free
        choice -- edge_index values are node IDs LOCAL to their node type, so
        (cpi_Category, r, X) and (eci_Industry, r, X) cannot share a key without
        their two ID spaces colliding. The .pt therefore has to keep every
        triple, and the multiplicity lands on the model:

            on the e2e fixtures, 69 relations produce 770 edge types. A single
            HeteroConv 1024->128 over that is ~101M parameters for ~10k edges,
            most of it in matrices that see a handful of edges each -- 67 edge
            types hold exactly one.

        What the pipeline can do is say which of those keys are the SAME
        relation, so a consumer builds one weight matrix per group and reuses it
        across the group's edge types. Nothing is lost by tying them: the
        endpoint types the key encodes are published as src_type_index /
        dst_type_index, so a shared weight can still condition on them through a
        node-type embedding -- one table of N types rather than N x M matrices.

        Emitted here rather than left to the trainer because graph_schema.json
        is the contract: a trainer that re-derived groups by string-splitting
        edge-type keys would be guessing at a fact the builder knows.

        Grouping is by relation name, which is the predicate URI's local name
        under its namespace prefix -- so two genuinely different predicates
        never collide, and `predicate_uri` is carried on the group to make that
        checkable.

        `origin` is the group's only field that its edge types can genuinely
        disagree on: origin reads the endpoints (see register_edge_types) and a
        group spans endpoint pairs by definition, so jolts:hasIndustry is `raw`
        across 21 of its edge types and `enrichment` on the one leaving a minted
        node. A disagreeing group reports "mixed" rather than one member's value
        -- naming a single origin there would restate at group level the exact
        mislabelling that keying origin by relation name used to cause. "mixed"
        means "ask the edge types"; it is deliberately not one of the three
        origin values, so a consumer that switches on origin cannot read it as a
        trust level.

        Returns:
            Dict[relation_name -> {edge_types, count, ...}], sorted by name.
        """
        groups: Dict[str, Dict[str, Any]] = {}
        for key, entry in sorted(edge_types.items()):
            group = entry.get("relation_group") or entry.get("relation", "")
            origin = entry.get("origin", "unknown")
            bucket = groups.setdefault(
                group,
                {
                    "predicate_uri": entry.get("predicate_uri", ""),
                    "origin": origin,
                    "edge_types": [],
                    "edge_type_count": 0,
                    "count": 0,
                    "has_features": False,
                    "feature_dim": 0,
                },
            )
            if bucket["origin"] != origin:
                bucket["origin"] = "mixed"
            bucket["edge_types"].append(key)
            bucket["edge_type_count"] += 1
            bucket["count"] += int(entry.get("count", 0))
            if entry.get("has_features"):
                bucket["has_features"] = True
                bucket["feature_dim"] = int(entry.get("feature_dim", 0))

        return groups

    def _build_feature_spec(self) -> Dict[str, Any]:
        """Build feature_spec.json content."""
        # Node feature segments
        node_segments = []
        if self._node_layout:
            layout = self._node_layout
            node_segments = [
                {
                    "name": "ontology_structure",
                    "type": "structural",
                    "start": layout["seg1_start"],
                    "end": layout["seg1_start"] + layout["seg1_total"] - 1,
                    "dim": layout["seg1_total"],
                    "sub_segments": [
                        {
                            "name": "class_identity",
                            "start": layout["seg1_class_identity_start"],
                            "end": layout["seg1_class_identity_start"]
                            + layout["seg1_class_identity_dim"] - 1,
                            "dim": layout["seg1_class_identity_dim"],
                        },
                        {
                            "name": "class_hierarchy",
                            "start": layout["seg1_class_hierarchy_start"],
                            "end": layout["seg1_class_hierarchy_start"]
                            + layout["seg1_class_hierarchy_dim"] - 1,
                            "dim": layout["seg1_class_hierarchy_dim"],
                        },
                        {
                            "name": "ontology_source",
                            "start": layout["seg1_ontology_source_start"],
                            "end": layout["seg1_ontology_source_start"]
                            + layout["seg1_ontology_source_dim"] - 1,
                            "dim": layout["seg1_ontology_source_dim"],
                        },
                    ],
                },
                {
                    "name": "property_schema",
                    "type": "schema",
                    "start": layout["seg2_start"],
                    "end": layout["seg2_start"] + layout["seg2_total"] - 1,
                    "dim": layout["seg2_total"],
                    "sub_segments": [
                        {
                            "name": "property_presence",
                            "start": layout["seg2_property_presence_start"],
                            "end": layout["seg2_property_presence_start"]
                            + layout["seg2_property_presence_dim"] - 1,
                            "dim": layout["seg2_property_presence_dim"],
                        },
                        {
                            "name": "domain_range",
                            "start": layout["seg2_domain_range_start"],
                            "end": layout["seg2_domain_range_start"]
                            + layout["seg2_domain_range_dim"] - 1,
                            "dim": layout["seg2_domain_range_dim"],
                        },
                        {
                            "name": "property_hierarchy",
                            "start": layout["seg2_property_hierarchy_start"],
                            "end": layout["seg2_property_hierarchy_start"]
                            + layout["seg2_property_hierarchy_dim"] - 1,
                            "dim": layout["seg2_property_hierarchy_dim"],
                        },
                    ],
                },
                {
                    "name": "literal_values",
                    "type": "literal",
                    "start": layout["seg3_start"],
                    "end": layout["seg3_start"] + layout["seg3_total"] - 1,
                    "dim": layout["seg3_total"],
                    "sub_segments": [
                        {
                            "name": "numeric_values",
                            "start": layout["seg3_numeric_start"],
                            "end": layout["seg3_numeric_start"]
                            + layout["seg3_numeric_dim"] - 1,
                            "dim": layout["seg3_numeric_dim"],
                        },
                        {
                            "name": "categorical_values",
                            "start": layout["seg3_categorical_start"],
                            "end": layout["seg3_categorical_start"]
                            + layout["seg3_categorical_dim"] - 1,
                            "dim": layout["seg3_categorical_dim"],
                        },
                    ],
                },
            ]

        # Declare, per sub-segment, whether it actually carries signal. A
        # sub-segment whose source predicate is absent from the triples is
        # permanently zero, and the spec has to say so — a consumer reading
        # dims alone cannot tell a dead slice from a live one, and neither
        # could the e2e vacuity guard.
        for seg in node_segments:
            for sub in seg.get("sub_segments", []):
                reason = self._sub_segment_status.get(sub["name"])
                sub["populated"] = reason is None
                if reason is not None:
                    sub["empty_reason"] = reason

        # Edge feature segments
        edge_segments = []
        if self._edge_layout and self._edge_features_enabled:
            elayout = self._edge_layout
            edge_segments = [
                {
                    "name": "temporal_signals",
                    "start": elayout["seg1_start"],
                    "end": elayout["seg1_start"]
                    + elayout["seg1_total"] - 1,
                    "dim": elayout["seg1_total"],
                    "sub_segments": [
                        {
                            "name": "time_delta",
                            "start": elayout["seg1_time_delta_start"],
                            "end": elayout["seg1_time_delta_start"]
                            + elayout["seg1_time_delta_dim"] - 1,
                            "dim": elayout["seg1_time_delta_dim"],
                        },
                        {
                            "name": "period_flags",
                            "start": elayout["seg1_period_flags_start"],
                            "end": elayout["seg1_period_flags_start"]
                            + elayout["seg1_period_flags_dim"] - 1,
                            "dim": elayout["seg1_period_flags_dim"],
                        },
                        {
                            "name": "direction",
                            "start": elayout["seg1_direction_start"],
                            "end": elayout["seg1_direction_start"]
                            + elayout["seg1_direction_dim"] - 1,
                            "dim": elayout["seg1_direction_dim"],
                        },
                    ],
                },
                {
                    "name": "numeric_contrast",
                    "start": elayout["seg2_start"],
                    "end": elayout["seg2_start"]
                    + elayout["seg2_total"] - 1,
                    "dim": elayout["seg2_total"],
                    "sub_segments": [
                        {
                            "name": "difference",
                            "start": elayout["seg2_difference_start"],
                            "end": elayout["seg2_difference_start"]
                            + elayout["seg2_difference_dim"] - 1,
                            "dim": elayout["seg2_difference_dim"],
                        },
                        {
                            "name": "ratio",
                            "start": elayout["seg2_ratio_start"],
                            "end": elayout["seg2_ratio_start"]
                            + elayout["seg2_ratio_dim"] - 1,
                            "dim": elayout["seg2_ratio_dim"],
                        },
                        {
                            "name": "magnitude",
                            "start": elayout["seg2_magnitude_start"],
                            "end": elayout["seg2_magnitude_start"]
                            + elayout["seg2_magnitude_dim"] - 1,
                            "dim": elayout["seg2_magnitude_dim"],
                        },
                    ],
                },
                {
                    "name": "relational_context",
                    "start": elayout["seg3_start"],
                    "end": elayout["seg3_start"]
                    + elayout["seg3_total"] - 1,
                    "dim": elayout["seg3_total"],
                    "sub_segments": [
                        {
                            "name": "namespace",
                            "start": elayout["seg3_namespace_start"],
                            "end": elayout["seg3_namespace_start"]
                            + elayout["seg3_namespace_dim"] - 1,
                            "dim": elayout["seg3_namespace_dim"],
                        },
                        {
                            "name": "label_similarity",
                            "start": elayout[
                                "seg3_label_similarity_start"
                            ],
                            "end": elayout[
                                "seg3_label_similarity_start"
                            ]
                            + elayout["seg3_label_similarity_dim"] - 1,
                            "dim": elayout["seg3_label_similarity_dim"],
                        },
                        {
                            "name": "relation_identity",
                            "start": elayout[
                                "seg3_relation_identity_start"
                            ],
                            "end": elayout[
                                "seg3_relation_identity_start"
                            ]
                            + elayout["seg3_relation_identity_dim"] - 1,
                            "dim": elayout["seg3_relation_identity_dim"],
                        },
                    ],
                },
            ]

        # Edge type feature derivation methods: no relation is featurized when
        # edge features are disabled, so the map stays empty alongside the
        # zeroed total_dim and empty segments.
        edge_type_derivations = {}
        if self._edge_features_enabled:
            for rel, cat in self._edge_feature_categories.items():
                if cat != "skip":
                    edge_type_derivations[rel] = cat

        return {
            "version": "1.0",
            "node_features": {
                "total_dim": self._vector_dim,
                "structural_dims_shared_within_type": True,
                "segments": node_segments,
            },
            "edge_features": {
                "enabled": self._edge_features_enabled,
                "total_dim": (
                    self._edge_vector_dim
                    if self._edge_features_enabled
                    else 0
                ),
                "segments": edge_segments,
                "edge_types_with_features": self._edge_types_with_features,
                "edge_types_without_features": (
                    self._edge_types_without_features
                ),
                "derivation_methods": edge_type_derivations,
            },
        }

    def _build_normalization(self) -> Dict[str, Any]:
        """Build normalization.json content.

        Statistics are rounded to _STAT_SIGNIFICANT_DIGITS before
        serialization so the file is byte-reproducible across runs —
        see _round_floats for why the raw values are not.
        """
        per_property = []
        if self._norm_stats:
            for stat in self._norm_stats:
                per_property.append({
                    "predicate_uri": stat.get("predicate", ""),
                    "mean": stat.get("mu", 0.0),
                    "std": stat.get("sigma", 1.0),
                    "count": stat.get("count", 0),
                })

        per_edge_type = {}
        if self._edge_norm_stats:
            per_edge_type = self._edge_norm_stats

        return _round_floats({
            "version": "1.0",
            "method": "z-score",
            "node_properties": {
                "total_properties_normalized": len(per_property),
                "zero_variance_properties": self._zero_variance_properties,
                "per_property": per_property,
            },
            "edge_derived_features": per_edge_type,
        })

    def _build_encoding_config(self) -> Dict[str, Any]:
        """Build encoding_config.json content, stamped with a contract digest."""
        if self._encoding_config:
            config = dict(self._encoding_config)
            config["checksum"] = _encoding_contract_digest(config)
            return config

        # Fallback: build from known constants if register was not called
        return {
            "version": "1.0",
            "note": "encoding_config was not explicitly registered",
        }

    def _build_ontology_schema(self) -> Dict[str, Any]:
        """Build ontology_schema.json content."""
        if self._ontology_schema:
            return self._ontology_schema

        return {
            "version": "1.0",
            "note": "ontology_schema was not explicitly registered",
        }

    def _build_slot_mapping(self) -> Dict[str, Any]:
        """Build slot_mapping.json content."""
        if self._slot_mapping:
            return self._slot_mapping

        return {
            "version": "1.0",
            "note": "slot_mapping was not explicitly registered",
        }


def _encoding_contract_digest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Digest of the encoding contract — what must match for a model to stay valid.

    Replaces a field that was named ``checksum`` but only held
    ``{"total_node_feature_dim": N}``. A dimension detects nothing: two builds
    with different hash seeds, namespace indices or slot boundaries produced
    identical values as long as the vector width matched. A model trained
    against one encoding and served against a graph built with another would
    load cleanly and return plausible, silently wrong numbers -- every feature
    sitting in a different slot than the weights expect.

    Hashes the whole merged node+edge config (minus any existing checksum), so
    it automatically covers every seed, dimension, segment boundary, namespace
    table and encoding convention already recorded there, and keeps covering
    new fields as they are added.

    This is a CONTRACT hash, not a data hash: it is derived purely from
    configuration, so rebuilding a different month with the same settings
    yields the same digest. Only a change that would invalidate a trained model
    changes it -- which is what makes it usable as a compatibility gate in a
    deployed inference path.
    """
    import hashlib

    payload = {k: v for k, v in config.items() if k != "checksum"}
    canonical = json.dumps(
        _sanitize_config(payload), sort_keys=True, separators=(",", ":")
    )
    return {
        "algorithm": "sha256",
        "contract_digest": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Make config JSON-serializable by converting non-standard types."""
    result = {}
    for k, v in config.items():
        if isinstance(v, dict):
            result[k] = _sanitize_config(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [
                _sanitize_config(i) if isinstance(i, dict) else i
                for i in v
            ]
        elif isinstance(v, set):
            result[k] = sorted(list(v))
        elif isinstance(v, (int, float, str, bool, type(None))):
            result[k] = v
        else:
            result[k] = str(v)
    return result


def write_metadata_to_s3(
    s3_client,
    metadata_files: Dict[str, Dict[str, Any]],
    bucket: str,
    metadata_prefix: str,
) -> None:
    """
    Write all metadata files to S3.

    Args:
        s3_client: Boto3 S3 client
        metadata_files: Dict[filename -> content_dict]
        bucket: S3 bucket
        metadata_prefix: S3 prefix for metadata directory
            (e.g., "pyg/2024-12/metadata/")
    """
    # Ensure prefix ends with /
    if not metadata_prefix.endswith("/"):
        metadata_prefix += "/"

    for filename, content in metadata_files.items():
        key = f"{metadata_prefix}{filename}"
        body = json.dumps(content, indent=2, default=str).encode("utf-8")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

        size_kb = len(body) / 1024
        logger.info(
            f"  Saved {filename} ({size_kb:.1f} KB) "
            f"to s3://{bucket}/{key}"
        )


def write_metadata_to_local(
    metadata_files: Dict[str, Dict[str, Any]],
    metadata_dir: str,
    spark=None,
) -> None:
    """
    Write all metadata files to the job's work dir.

    Mirror of write_metadata_to_s3() for the local-first storage model.

    Despite the name (kept for its callers), the destination is not necessarily
    local: ``local_work_dir`` is an ``s3a://`` URI on the cluster, and
    ``os.makedirs`` + ``open()`` would silently write a junk ``./s3a:/...`` tree
    on the driver's disk. Routing through the scheme-aware writer keeps bare
    POSIX paths on direct local I/O and sends URIs through Hadoop.

    Args:
        metadata_files: Dict[filename -> content_dict]
        metadata_dir: Directory for the metadata JSON files — bare path or URI
            (e.g., "/data/pyg/year=2024/month=12/metadata/")
        spark: Active SparkSession; required when metadata_dir is a non-local URI
    """
    from spark_jobs.utils.fs_utils import join_path, write_bytes

    for filename, content in metadata_files.items():
        path = join_path(metadata_dir, filename)
        body = json.dumps(content, indent=2, default=str).encode("utf-8")

        write_bytes(path, body, spark=spark)

        size_kb = len(body) / 1024
        logger.info(f"  Saved {filename} ({size_kb:.1f} KB) to {path}")


# The partition segment standing in for `year=YYYY/month=MM` on the alias copy.
#
# The period layout is addressable only by someone who already knows which
# period is newest. A consumer fetching one fixed URL over HTTP cannot list the
# bucket to find out -- and should not be able to -- so every build also writes
# the schema to this fixed segment, which always names the most recent build.
LATEST_PARTITION = "latest"

# Only the schema is aliased. The alias exists for consumers reading the graph's
# SHAPE; the .pt, the node index and the other five metadata files have no such
# reader, and copying them would advertise the whole build as addressable at
# `latest/` when only this one file is kept current there.
LATEST_ALIAS_FILES = frozenset({"graph_schema.json"})


def write_latest_alias(
    metadata_files: Dict[str, Dict[str, Any]],
    latest_metadata_dir: str,
    spark=None,
) -> None:
    """
    Write the consumer-facing subset of the metadata to the `latest` alias.

    Delegates to write_metadata_to_local() rather than serializing again, which
    is what makes the alias byte-identical to the period-partitioned copy: one
    json.dumps() call site, one set of separators and indentation. It also means
    the alias inherits the scheme-aware write, so an unreachable URI raises here
    exactly as it does for the period copy instead of quietly landing on the
    driver's disk.

    A metadata_files dict carrying none of LATEST_ALIAS_FILES is a no-op, not an
    error: partial metadata is a legitimate state for a build that did not
    register every collector, and failing here would take down a build over an
    artifact that has no consumer yet.

    Args:
        metadata_files: Dict[filename -> content_dict], the full set
        latest_metadata_dir: Directory for the alias — bare path or URI
        spark: Active SparkSession; required when the directory is a non-local URI
    """
    alias_files = {
        name: content
        for name, content in metadata_files.items()
        if name in LATEST_ALIAS_FILES
    }
    if not alias_files:
        logger.info("  No aliasable metadata for the latest pointer; skipping")
        return

    write_metadata_to_local(alias_files, latest_metadata_dir, spark=spark)


def derive_metadata_prefix(pyg_output_key: str) -> str:
    """
    Derive the metadata S3 prefix from the PyG output key.

    Examples:
        "pyg/2024-12/hetero_data.pt"
            → "pyg/2024-12/metadata/"

        "pyg/2024-12/hetero_data_512d.pt"
            → "pyg/2024-12/hetero_data_512d_metadata/"

    Convention: if the filename is exactly "hetero_data.pt" (the default),
    use "metadata/" as the directory name. Otherwise, use
    "{stem}_metadata/" to support multiple experiments in the same
    time period directory.
    """
    return _derive_sibling_prefix(pyg_output_key, "metadata")


def derive_node_index_prefix(pyg_output_key: str) -> str:
    """Derive the node-index prefix from the PyG output key.

    Same convention as derive_metadata_prefix, so the identity map sits beside
    the metadata it belongs to and experiment variants stay separated:

        "pyg/2024-12/hetero_data.pt"      -> "pyg/2024-12/node_index/"
        "pyg/2024-12/hetero_data_512d.pt" -> "pyg/2024-12/hetero_data_512d_node_index/"
    """
    return _derive_sibling_prefix(pyg_output_key, "node_index")


def _derive_sibling_prefix(pyg_output_key: str, suffix: str) -> str:
    """Shared derivation for artifact directories that sit beside the .pt.

    The parent is split off the key rather than parsed, so a Hive-partitioned
    parent carries through untouched and every artifact for a period stays in
    one directory:

        pyg/year=2024/month=12/hetero_data.pt
            -> pyg/year=2024/month=12/{suffix}/
    """
    if "/" in pyg_output_key:
        parent = pyg_output_key.rsplit("/", 1)[0]
        filename = pyg_output_key.rsplit("/", 1)[1]
    else:
        parent = ""
        filename = pyg_output_key

    # Strip .pt extension
    if filename.endswith(".pt"):
        stem = filename[:-3]
    else:
        stem = filename

    if stem == "hetero_data":
        directory = suffix
    else:
        directory = f"{stem}_{suffix}"

    if parent:
        return f"{parent}/{directory}/"
    else:
        return f"{directory}/"