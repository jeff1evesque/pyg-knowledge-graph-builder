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
from typing import Dict, Any, List, Optional, Tuple

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
    ):
        self._time_period = time_period
        self._vector_dim = vector_dim
        self._edge_vector_dim = edge_vector_dim
        self._edge_features_enabled = edge_features_enabled
        self._config = config
        self._build_timestamp = datetime.now(timezone.utc).isoformat()

        # Populated by register_* methods during construction
        self._node_counts: Dict[str, int] = {}
        self._node_type_uris: Dict[str, str] = {}
        self._node_type_categories: Dict[str, str] = {}
        self._edge_counts: Dict[str, int] = {}
        self._edge_type_details: Dict[str, Dict[str, Any]] = {}
        self._node_layout: Optional[Dict[str, Any]] = None
        self._edge_layout: Optional[Dict[str, Any]] = None
        self._norm_stats: Optional[List[Dict[str, Any]]] = None
        self._edge_norm_stats: Optional[Dict[str, List[Dict]]] = None
        self._encoding_config: Optional[Dict[str, Any]] = None
        self._ontology_schema: Optional[Dict[str, Any]] = None
        self._slot_mapping: Optional[Dict[str, Any]] = None
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

    def register_edge_types(
        self,
        edge_counts: Dict[Tuple[str, str, str], int],
        edge_predicate_uris: Dict[str, str],
        edge_origins: Optional[Dict[str, str]] = None,
        edge_feature_flags: Optional[Dict[str, bool]] = None,
        edge_feature_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Register edge type information from EdgeMapper (Step 2).

        Args:
            edge_counts: Dict[(src, rel, dst) -> num_edges]
            edge_predicate_uris: Dict[relation_name -> predicate_uri]
            edge_origins: Optional Dict[relation_name -> origin] where
                origin is one of: raw, enrichment, unification
                (see rdf_utils.classify_edge_origin). Three values, not
                the four once listed here: intra- and cross-source enrichment
                share a namespace, so nothing available at this layer can
                separate them, and promising a distinction that is never
                emitted is the same defect as the "unknown" default below.
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
                "origin": (edge_origins or {}).get(rel, "unknown"),
                "has_features": (edge_feature_flags or {}).get(rel, False),
                "feature_dim": (edge_feature_dims or {}).get(rel, 0),
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
        """Build graph_schema.json content."""
        # Node types
        node_types = {}
        for pyg_name, count in sorted(self._node_counts.items()):
            node_types[pyg_name] = {
                "count": count,
                "source_type_uri": self._node_type_uris.get(pyg_name, ""),
                "category": self._node_type_categories.get(
                    pyg_name, "entity"
                ),
                "has_features": count > 0,
            }

        # Edge types
        edge_types = {}
        for key, details in sorted(self._edge_type_details.items()):
            edge_types[key] = details

        # Summary statistics
        total_nodes = sum(self._node_counts.values())
        total_edges = sum(self._edge_counts.values())
        node_types_with_features = sum(
            1 for c in self._node_counts.values() if c > 0
        )
        edge_types_with_features = len(self._edge_types_with_features)

        return {
            "version": "1.0",
            "build_metadata": {
                "time_period": self._time_period,
                "build_timestamp": self._build_timestamp,
                "pipeline_config": _sanitize_config(self._config),
            },
            "node_types": node_types,
            "edge_types": edge_types,
            "summary": {
                "total_node_types": len(self._node_counts),
                "total_edge_types": len(self._edge_type_details),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_types_with_literal_features": node_types_with_features,
                "edge_types_with_features": edge_types_with_features,
                "edge_types_without_features": len(
                    self._edge_types_without_features
                ),
            },
        }

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

        # Edge type feature derivation methods
        edge_type_derivations = {}
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
) -> None:
    """
    Write all metadata files to a local directory.

    Creates the directory if needed. Mirror of write_metadata_to_s3() for
    the local-first storage model.

    Args:
        metadata_files: Dict[filename -> content_dict]
        metadata_dir: Local directory for the metadata JSON files
            (e.g., "/data/pyg/year=2024/month=12/metadata/")
    """
    import os

    os.makedirs(metadata_dir, exist_ok=True)

    for filename, content in metadata_files.items():
        path = os.path.join(metadata_dir, filename)
        body = json.dumps(content, indent=2, default=str).encode("utf-8")

        with open(path, "wb") as f:
            f.write(body)

        size_kb = len(body) / 1024
        logger.info(f"  Saved {filename} ({size_kb:.1f} KB) to {path}")


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
    if "/" in pyg_output_key:
        parent = pyg_output_key.rsplit("/", 1)[0]
        filename = pyg_output_key.rsplit("/", 1)[1]
    else:
        parent = ""
        filename = pyg_output_key

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