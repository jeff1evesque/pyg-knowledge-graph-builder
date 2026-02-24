"""
PyG HeteroData Constructor

Orchestrates the construction of a PyTorch Geometric HeteroData object
from an enriched triples DataFrame.

Architecture:
- All filtering, joining, ID assignment, and aggregation runs on Spark
  executors using DataFrame operations
- Node URI → integer ID mapping is done via Spark Window functions
  (row_number / dense_rank), never collected as a Python dict
- Only compact numeric tensors (edge indices, feature matrices) are
  collected to the driver via Arrow-optimized toPandas()
- Final HeteroData assembly happens on the driver (PyG is not distributed)

Scaling characteristics:
- Spark-side: scales horizontally with DPUs, handles 100M+ triples
- Driver-side: only receives integer tensors, not URI strings
  - Edge indices: [2, num_edges] int64 — ~16 bytes per edge
  - Feature matrices: [num_nodes, num_features] float32 — ~4 bytes per cell
  - Per-type feature isolation: each node type carries only its ontology-
    relevant features (5-20 columns), not a wide matrix across all ontologies
  - Total driver memory for PyG object: typically 2-8 GB for 30-50M triples
  - Fits on Glue G.2X (32 GB) or G.4X (64 GB) workers
"""
import logging
from typing import Dict, Any, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame

from glue_jobs.pyg_builder.node_mapper import NodeMapper
from glue_jobs.pyg_builder.edge_mapper import EdgeMapper
from glue_jobs.pyg_builder.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def build_hetero_data(
    spark: SparkSession,
    triples_df: DataFrame,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Build a PyTorch Geometric HeteroData object from an enriched triples
    DataFrame.

    All heavy computation runs on Spark executors. Only compact
    integer/float tensors cross the Spark → driver boundary for
    final HeteroData assembly.

    Args:
        spark: Active SparkSession
        triples_df: Enriched triples DataFrame (subject, predicate, object).
                    Should already be cached by the caller.
        config: Optional configuration dict. If None or empty, defaults
                are inferred from the data.

    Returns:
        torch_geometric.data.HeteroData
    """
    import torch
    from torch_geometric.data import HeteroData

    config = config or {}

    logger.info("Building PyG HeteroData from triples DataFrame")
    logger.info(
        f"Config keys: {list(config.keys()) if config else 'defaults'}"
    )

    # Ensure triples_df is cached for repeated reads across mappers.
    # If already cached by the caller, this is a no-op.
    if triples_df.storageLevel.useMemory or triples_df.storageLevel.useDisk:
        logger.info("triples_df already cached")
    else:
        triples_df = triples_df.cache()
        triples_df.count()  # materialize
        logger.info("Cached triples_df")

    # ============================================
    # STEP 1: Build node ID tables (on executors)
    # ============================================
    logger.info("Step 1/4: Building node ID tables...")

    node_mapper = NodeMapper(spark, config)
    node_id_df, node_counts = node_mapper.build_node_id_table(triples_df)

    # node_id_df: DataFrame(uri, node_id, node_type) — cached on executors
    # node_counts: Dict[str, int] — small, one entry per node type
    total_nodes = sum(node_counts.values())
    logger.info(
        f"  {len(node_counts)} node types, {total_nodes:,} total nodes"
    )

    if total_nodes == 0:
        logger.warning("No nodes found — returning empty HeteroData")
        return HeteroData()

    # ============================================
    # STEP 2: Build edge index tensors
    #         (join on executors, collect int tensors to driver)
    # ============================================
    logger.info("Step 2/4: Building edge indices...")

    edge_mapper = EdgeMapper(spark, config)
    edge_indices = edge_mapper.build_edge_indices(
        triples_df, node_id_df, node_counts
    )

    total_edges = sum(t.shape[1] for t in edge_indices.values())
    logger.info(
        f"  {len(edge_indices)} edge types, {total_edges:,} total edges"
    )

    # ============================================
    # STEP 3: Build feature tensors
    #         (pivot on executors, collect float tensors to driver)
    # ============================================
    logger.info("Step 3/4: Building feature tensors...")

    feature_extractor = FeatureExtractor(spark, config)
    feature_tensors, feature_names = feature_extractor.build_features(
        triples_df, node_id_df, node_counts
    )

    for ntype, tensor in feature_tensors.items():
        names = feature_names.get(ntype, [])
        logger.info(
            f"  [{ntype}] shape: {list(tensor.shape)}, "
            f"features: {names[:5]}{'...' if len(names) > 5 else ''}"
        )

    # ============================================
    # STEP 4: Assemble HeteroData (on driver only)
    # ============================================
    logger.info("Step 4/4: Assembling HeteroData...")

    data = HeteroData()

    for node_type, num_nodes in node_counts.items():
        if node_type in feature_tensors:
            data[node_type].x = feature_tensors[node_type]
        else:
            data[node_type].x = torch.zeros(num_nodes, 1)
        data[node_type].num_nodes = num_nodes

    for (src_type, rel, dst_type), edge_index in edge_indices.items():
        data[src_type, rel, dst_type].edge_index = edge_index

    # Release executor memory
    node_id_df.unpersist()

    logger.info("HeteroData assembly complete")
    logger.info(f"  Node types: {data.node_types}")
    logger.info(f"  Edge types: {data.edge_types}")

    return data