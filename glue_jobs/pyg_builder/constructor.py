"""
PyG HeteroData Constructor

Orchestrates the construction of a PyTorch Geometric HeteroData object
from an enriched triples DataFrame.

Architecture:
- Heavy filtering, joining, and aggregation runs on Spark executors
- Only compact results (node ID maps, edge indices, feature matrices)
  are collected to the driver
- Final HeteroData assembly happens on the driver (PyG is not distributed)

Expected config structure:
    {
        "node_types": ["cpi_Index", "ppi_MonthlyChange", ...],
        "edge_types": ["precedes", "correlatesWith", "belongsToSector", ...],
        "feature_config": {
            "numeric_properties": ["indexValue", "changeValue", "rate"],
            "categorical_properties": ["hasCategory", "hasIndustry"],
            "normalize": true
        },
        "include_temporal_nodes": true,
        "include_sector_nodes": true
    }

If config is empty or keys are missing, sensible defaults are used:
- All rdf:type classes become node types
- All predicates linking two node entities become edge types
- All numeric literal properties become features
"""
import logging
from typing import Dict, Any, Optional

from pyspark.sql import SparkSession, DataFrame

from glue_jobs.pyg_builder.node_mapper import NodeMapper
from glue_jobs.pyg_builder.edge_mapper import EdgeMapper
from glue_jobs.pyg_builder.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def build_hetero_data(
    spark: SparkSession,
    triples_df: DataFrame,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Build a PyTorch Geometric HeteroData object from an enriched triples DataFrame.

    All heavy computation (filtering, joining, aggregation) runs on Spark
    executors. Only compact results are collected to the driver for final
    HeteroData assembly.

    Args:
        spark: Active SparkSession
        triples_df: Enriched triples DataFrame with columns
                    (subject, predicate, object)
        config: Optional configuration dict controlling which node types,
                edge types, and features to include. If None or empty,
                defaults are inferred from the data.

    Returns:
        torch_geometric.data.HeteroData with:
        - Node stores: x (feature tensor), num_nodes per node type
        - Edge stores: edge_index per (src_type, rel, dst_type) edge type
    """
    import torch
    from torch_geometric.data import HeteroData

    config = config or {}

    logger.info("Building PyG HeteroData from triples DataFrame")
    logger.info(f"Config keys: {list(config.keys()) if config else 'defaults'}")

    # Cache the input for repeated reads across mappers
    triples_df = triples_df.cache()
    triple_count = triples_df.count()
    logger.info(f"Input triples: {triple_count:,}")

    # ============================================
    # STEP 1: Build node mappings (on executors)
    # ============================================
    logger.info("Step 1/3: Building node mappings...")

    node_mapper = NodeMapper(spark, config)
    node_maps = node_mapper.build_node_maps(triples_df)

    # node_maps: Dict[str, Dict[str, int]]
    #   node_type -> {uri -> integer_id}
    total_nodes = sum(len(m) for m in node_maps.values())
    logger.info(
        f"  {len(node_maps)} node types, {total_nodes:,} total nodes"
    )

    if total_nodes == 0:
        logger.warning("No nodes found — returning empty HeteroData")
        triples_df.unpersist()
        return HeteroData()

    # ============================================
    # STEP 2: Build edge indices (on executors)
    # ============================================
    logger.info("Step 2/3: Building edge indices...")

    edge_mapper = EdgeMapper(spark, config)
    edge_indices = edge_mapper.build_edge_indices(triples_df, node_maps)

    # edge_indices: Dict[Tuple[str,str,str], Tensor]
    #   (src_type, relation, dst_type) -> LongTensor of shape [2, num_edges]
    total_edges = sum(t.shape[1] for t in edge_indices.values())
    logger.info(
        f"  {len(edge_indices)} edge types, {total_edges:,} total edges"
    )

    # ============================================
    # STEP 3: Build feature tensors (on executors)
    # ============================================
    logger.info("Step 3/3: Building feature tensors...")

    feature_extractor = FeatureExtractor(spark, config)
    feature_tensors = feature_extractor.build_features(triples_df, node_maps)

    # feature_tensors: Dict[str, Tensor]
    #   node_type -> FloatTensor of shape [num_nodes, num_features]
    for ntype, tensor in feature_tensors.items():
        logger.info(f"  [{ntype}] features shape: {list(tensor.shape)}")

    # ============================================
    # STEP 4: Assemble HeteroData (on driver)
    # ============================================
    logger.info("Assembling HeteroData...")

    data = HeteroData()

    # Add node stores
    for node_type, uri_to_id in node_maps.items():
        num_nodes = len(uri_to_id)

        if node_type in feature_tensors:
            data[node_type].x = feature_tensors[node_type]
        else:
            # No features — store a placeholder identity-like tensor
            # so num_nodes is set correctly
            data[node_type].x = torch.zeros(num_nodes, 1)

        data[node_type].num_nodes = num_nodes

    # Add edge stores
    for (src_type, rel, dst_type), edge_index in edge_indices.items():
        data[src_type, rel, dst_type].edge_index = edge_index

    triples_df.unpersist()

    logger.info("HeteroData assembly complete")
    logger.info(f"  Node types: {data.node_types}")
    logger.info(f"  Edge types: {data.edge_types}")

    return data