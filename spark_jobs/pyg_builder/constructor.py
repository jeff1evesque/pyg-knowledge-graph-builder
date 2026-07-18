"""
PyG HeteroData Constructor

Orchestrates the construction of a PyTorch Geometric HeteroData object
from an enriched triples DataFrame.

Architecture:
- All filtering, joining, ID assignment, and aggregation runs on Spark
  executors using DataFrame operations
- Node URI → integer ID mapping uses Spark Window functions, never
  collected as a Python dict
- Only compact numeric tensors are collected to the driver via
  Arrow-optimized toPandas()
- Final HeteroData assembly happens on the driver

Node feature vectors:
- Universal 1024-d ontology-aware vectors for all node types
- Segment 1 (dims 0–255):   Ontology structure (class identity,
                             hierarchy, source membership)
- Segment 2 (dims 256–639): Property schema (presence, domain/range,
                             property hierarchy)
- Segment 3 (dims 640–1023): Literal values (numeric hashed slots,
                              categorical multi-hot)
- Same width across all node types enables shared GNN layers

Edge feature vectors (selective, default 32-d):
- Only high-value edge types receive features (temporal, option-stock,
  escalation)
- Segment 1: Temporal signals (time delta, period flags, direction)
- Segment 2: Numeric contrast (difference, ratio, magnitude between
  endpoints)
- Segment 3: Relational context (namespace, label similarity, relation
  identity)
- Dimension configurable via edge_feature_config.edge_vector_dim
- Edge types without features use simpler GNN message-passing layers
- Reuses cached resolved edges DataFrame from EdgeMapper — the
  expensive double-join runs exactly once

Metadata collection:
- During each step, metadata artifacts are collected into a
  MetadataCollector instance for the six metadata JSON files
- All metadata collect() calls target small aggregated/distinct
  DataFrames (per-predicate stats, per-type URIs, per-class hierarchy)
  — never raw triples or per-node/per-edge data
- MetadataCollector holds only small Python dicts — no tensors,
  no DataFrames, no Spark references

Driver memory safety:
- Node feature tensors are built one type at a time, largest first
- Large types use chunked collection to bound Pandas intermediaries
- gc.collect() runs between types to reclaim fragmented memory
- Edge indices are collected one type at a time with immediate Pandas
  release
- Edge feature tensors are small (32-d) and collected per edge type
- Resolved edges DataFrame is cached once by EdgeMapper, shared with
  EdgeFeatureExtractor, then unpersisted by the constructor
- HeteroData assembly only references existing tensors (no copies)
- Final .pt serialization streams to S3 via upload_fileobj
- Metadata collection adds negligible driver memory (<1 MB total)

Scaling:
- Spark-side: scales horizontally with DPUs
- Driver-side: receives only integer/float tensors, not URI strings
  - Typical total: 2-8 GB for 30-50M triples
  - Fits on Glue G.2X (32 GB) or G.4X (64 GB)
"""
import gc
import logging
from typing import Dict, Any, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame

from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.pyg_builder.edge_mapper import EdgeMapper
from spark_jobs.pyg_builder.feature_extractor import (
    FeatureExtractor,
    VECTOR_DIM,
)
from spark_jobs.pyg_builder.edge_feature_extractor import (
    EdgeFeatureExtractor,
    EDGE_VECTOR_DIM,
)
from spark_jobs.pyg_builder.metadata_writer import MetadataCollector

logger = logging.getLogger(__name__)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def build_hetero_data(
    spark: SparkSession,
    triples_df: DataFrame,
    config: Optional[Dict[str, Any]] = None,
    time_period: str = "",
) -> Tuple[Any, MetadataCollector, DataFrame]:
    """
    Build a PyTorch Geometric HeteroData object from an enriched triples
    DataFrame.

    All heavy computation runs on Spark executors. Only compact
    integer/float tensors cross the Spark → driver boundary.

    Node feature vectors are universal 1024-d ontology-aware vectors:
    - Same width for all node types (enables shared GNN layers)
    - Encodes ontology structure, property schema, and literal values
    - Large types use chunked collection to prevent driver OOM
    - See feature_extractor.py for segment layout

    Edge feature vectors are selective 32-d derived vectors:
    - Only high-value edge types receive features
    - Derived from endpoint node properties (no enrichment changes)
    - Reuses cached resolved edges from EdgeMapper (no double-join)
    - See edge_feature_extractor.py for segment layout

    Metadata collection:
    - During each step, metadata artifacts are deposited into a
      MetadataCollector for the six metadata JSON files
    - All metadata collect() calls target small aggregated DataFrames
    - MetadataCollector is returned alongside HeteroData for the
      caller to write to S3

    Args:
        spark: Active SparkSession
        triples_df: Enriched triples DataFrame (subject, predicate, object).
                    Should already be cached by the caller.
        config: Optional configuration dict. If None/empty, defaults
                are inferred from the data.
        time_period: Time period label (e.g., "2024-12") for metadata.

    Returns:
        Tuple of (HeteroData, MetadataCollector, node_index DataFrame).

        The third element maps (node_type, node_id) -> uri: the identity of
        every row in the graph. It is persisted beside the .pt so the graph can
        be joined to labels and predictions attributed back to entities --
        see _node_index().
    """
    import torch
    from torch_geometric.data import HeteroData

    config = config or {}

    logger.info("Building PyG HeteroData from triples DataFrame")
    logger.info(
        f"Config keys: {list(config.keys()) if config else 'defaults'}"
    )

    vector_dim = config.get("feature_config", {}).get(
        "vector_dim", VECTOR_DIM
    )
    edge_vector_dim = config.get("edge_feature_config", {}).get(
        "edge_vector_dim", EDGE_VECTOR_DIM
    )
    edge_features_enabled = config.get("edge_feature_config", {}).get(
        "enabled", True
    )
    logger.info(f"Node feature vector dimension: {vector_dim}")
    logger.info(
        f"Edge feature vector dimension: {edge_vector_dim} "
        f"({'enabled' if edge_features_enabled else 'disabled'})"
    )

    # Initialize metadata collector — holds only small Python dicts,
    # no tensors, no DataFrames, no Spark references
    metadata = MetadataCollector(
        time_period=time_period,
        vector_dim=vector_dim,
        edge_vector_dim=edge_vector_dim,
        edge_features_enabled=edge_features_enabled,
        config=config,
    )

    # Ensure triples_df is cached — if already cached, this is a no-op
    if not (
        triples_df.storageLevel.useMemory
        or triples_df.storageLevel.useDisk
    ):
        triples_df = triples_df.cache()
        triples_df.count()
        logger.info("Cached triples_df")

    # ============================================
    # STEP 1: Build node ID tables (on executors)
    # ============================================
    logger.info("Step 1/5: Building node ID tables...")

    node_mapper = NodeMapper(spark, config)
    node_id_df, node_counts = node_mapper.build_node_id_table(triples_df)

    total_nodes = sum(node_counts.values())
    logger.info(
        f"  {len(node_counts)} node types, {total_nodes:,} total nodes"
    )

    if total_nodes == 0:
        logger.warning("No nodes found — returning empty HeteroData")
        return HeteroData(), metadata, _node_index(node_id_df)

    # Log estimated driver memory for feature tensors
    total_feature_bytes = sum(
        n * vector_dim * 4 for n in node_counts.values()
    )
    total_feature_mb = total_feature_bytes / (1024 * 1024)
    logger.info(
        f"  Estimated node feature tensor memory: "
        f"{total_feature_mb:,.1f} MB"
    )

    # Collect node type URI mapping for metadata.
    # Small collect — one row per (node_type, type_uri), typically <1000.
    node_type_uris = node_mapper.get_type_uri_mapping(
        triples_df, node_id_df
    )

    # Register node types with metadata collector
    metadata.register_node_types(
        node_counts=node_counts,
        node_type_uris=node_type_uris,
    )

    # ============================================
    # STEP 2: Build edge index tensors
    #         (returns cached edges_final_df for reuse)
    # ============================================
    logger.info("Step 2/5: Building edge indices...")

    edge_mapper = EdgeMapper(spark, config)
    edge_indices, edges_final_df = edge_mapper.build_edge_indices(
        triples_df, node_id_df, node_counts
    )

    total_edges = sum(t.shape[1] for t in edge_indices.values())
    total_edge_bytes = sum(
        t.numel() * 8 for t in edge_indices.values()
    )
    total_edge_mb = total_edge_bytes / (1024 * 1024)
    logger.info(
        f"  {len(edge_indices)} edge types, {total_edges:,} total edges "
        f"({total_edge_mb:,.1f} MB)"
    )

    # Collect edge predicate URI mapping for metadata.
    # Small collect — one row per distinct relation, typically <100.
    edge_predicate_uris = edge_mapper.get_predicate_uri_mapping(
        edges_final_df
    )

    # ============================================
    # STEP 3: Build node feature tensors (ontology-aware)
    # ============================================
    logger.info("Step 3/5: Building ontology-aware node feature tensors...")

    feature_extractor = FeatureExtractor(spark, config)
    feature_tensors, feature_names = feature_extractor.build_features(
        triples_df, node_id_df, node_counts
    )

    for ntype, tensor in feature_tensors.items():
        segments = feature_names.get(ntype, [])
        logger.info(
            f"  [{ntype}] shape: {list(tensor.shape)}, "
            f"segments: {segments}"
        )

    # Register node feature metadata — all from small Python objects
    # already collected during build_features()
    metadata.register_node_feature_layout(
        feature_extractor.get_layout().to_dict()
    )

    artifacts = feature_extractor.get_metadata_artifacts()
    if artifacts["normalization_stats"] is not None:
        metadata.register_normalization_stats(
            stats=artifacts["normalization_stats"],
            zero_variance_properties=artifacts[
                "zero_variance_properties"
            ],
        )
    if artifacts["ontology_schema"] is not None:
        metadata.register_ontology_schema(artifacts["ontology_schema"])
    if artifacts["slot_mapping"] is not None:
        metadata.register_slot_mapping(artifacts["slot_mapping"])

    # Build combined encoding config from node + edge extractors
    encoding_config = feature_extractor.get_encoding_config()

    # ============================================
    # STEP 4: Build edge feature tensors (derived)
    #         Reuses edges_final_df — no double-join
    # ============================================
    logger.info("Step 4/5: Building derived edge feature tensors...")

    edge_feature_tensors: Dict = {}

    edge_feature_extractor = EdgeFeatureExtractor(spark, config)

    if edge_features_enabled:
        edge_feature_tensors = (
            edge_feature_extractor.build_edge_features(
                triples_df, node_id_df, edges_final_df, edge_indices
            )
        )

        for etype, tensor in edge_feature_tensors.items():
            logger.info(
                f"  [{etype}] shape: {list(tensor.shape)}"
            )
    else:
        logger.info("  Edge features disabled — skipping")

    # Register edge feature metadata
    metadata.register_edge_feature_layout(
        edge_feature_extractor.get_layout().to_dict()
    )

    # Merge edge encoding config into combined config
    edge_enc_config = edge_feature_extractor.get_encoding_config()
    encoding_config.update(edge_enc_config)
    metadata.register_encoding_config(encoding_config)

    # Classify edge types for metadata — lightweight driver-side
    # substring matching, ~1 μs per type
    categories, types_with, types_without = (
        edge_feature_extractor.get_edge_classification(edge_indices)
    )
    metadata.register_edge_feature_classification(
        categories=categories,
        types_with_features=types_with,
        types_without_features=types_without,
    )

    # Register edge types with metadata collector
    edge_counts = {
        k: t.shape[1] for k, t in edge_indices.items()
    }
    edge_feature_flags = {}
    edge_feature_dims = {}
    for (src, rel, dst) in edge_indices:
        has_feat = (src, rel, dst) in edge_feature_tensors
        edge_feature_flags[rel] = has_feat
        edge_feature_dims[rel] = (
            edge_vector_dim if has_feat else 0
        )
    metadata.register_edge_types(
        edge_counts=edge_counts,
        edge_predicate_uris=edge_predicate_uris,
        edge_feature_flags=edge_feature_flags,
        edge_feature_dims=edge_feature_dims,
    )

    # Release resolved edges — both consumers are done
    edges_final_df.unpersist()

    # ============================================
    # STEP 5: Assemble HeteroData (on driver only)
    # ============================================
    logger.info("Step 5/5: Assembling HeteroData...")

    data = HeteroData()

    for node_type, num_nodes in node_counts.items():
        if node_type in feature_tensors:
            data[node_type].x = feature_tensors[node_type]
        else:
            data[node_type].x = torch.zeros(
                num_nodes, vector_dim, dtype=torch.float32
            )
        data[node_type].num_nodes = num_nodes

    for (src_type, rel, dst_type), edge_index in edge_indices.items():
        edge_key = (src_type, rel, dst_type)
        data[edge_key].edge_index = edge_index

        if edge_key in edge_feature_tensors:
            data[edge_key].edge_attr = edge_feature_tensors[edge_key]

    # Release executor memory for node ID table
    node_id_df.unpersist()

    # Release references to intermediate dicts
    del feature_tensors
    del edge_indices
    del edge_feature_tensors
    gc.collect()

    logger.info("HeteroData assembly complete")
    logger.info(f"  Node types: {data.node_types}")
    logger.info(f"  Edge types: {data.edge_types}")
    logger.info(
        f"  Universal node feature dim: {vector_dim} "
        f"(all node types share same width)"
    )

    # Log which edge types have features
    edge_types_with_features = [
        et for et in data.edge_types
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
    ]
    edge_types_without_features = [
        et for et in data.edge_types
        if et not in edge_types_with_features
    ]
    logger.info(
        f"  Edge types with features ({edge_vector_dim}-d): "
        f"{len(edge_types_with_features)}"
    )
    for et in edge_types_with_features:
        logger.info(
            f"    {et}: {list(data[et].edge_attr.shape)}"
        )
    logger.info(
        f"  Edge types without features: "
        f"{len(edge_types_without_features)}"
    )

    # Log final driver memory estimate
    total_bytes = 0
    for node_type in data.node_types:
        store = data[node_type]
        if hasattr(store, "x") and store.x is not None:
            total_bytes += store.x.numel() * store.x.element_size()
    for edge_type in data.edge_types:
        store = data[edge_type]
        if hasattr(store, "edge_index") and store.edge_index is not None:
            total_bytes += (
                store.edge_index.numel()
                * store.edge_index.element_size()
            )
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            total_bytes += (
                store.edge_attr.numel()
                * store.edge_attr.element_size()
            )
    total_mb = total_bytes / (1024 * 1024)
    logger.info(f"  Total HeteroData tensor memory: {total_mb:,.1f} MB")

    return data, metadata, _node_index(node_id_df)


def _node_index(node_id_df: DataFrame) -> DataFrame:
    """The identity map: which real-world entity each graph row is.

    hetero_data.pt stores only ``x`` and ``num_nodes`` per node type, so row 5 of
    cpi_Index is some specific CPI series and nothing on disk says which. That
    makes the graph impossible to attach labels to (training) or to attribute
    predictions from (inference).

    node_id_df already carries (uri, node_id, node_type) and every later stage
    joins against it — this only persists what is already computed.

    Sorted so the written Parquet is content-stable run to run: node IDs come
    from row_number() over a uri-ordered window and are deterministic, but the
    ROW order Spark emits is not, and this artifact is covered by the
    reproducibility guard.
    """
    return (
        node_id_df
        .select("node_type", "node_id", "uri")
        .orderBy("node_type", "node_id")
    )
