"""
PyG HeteroData Constructor

Orchestrates the construction of a PyTorch Geometric HeteroData object
from an enriched triples DataFrame.

Architecture:
- All filtering, joining, ID assignment, and aggregation runs on Spark
  executors using DataFrame operations
- Node URI → integer ID mapping is done via Spark (monotonically_increasing_id
  or dense_rank), never collected as a Python dict
- Only compact numeric tensors (edge indices, feature matrices) are collected
  to the driver via Arrow-optimized toPandas()
- Final HeteroData assembly happens on the driver (PyG is not distributed)

Scaling characteristics:
- Spark-side: scales horizontally with DPUs, handles 100M+ triples
- Driver-side: only receives integer tensors, not URI strings
  - Edge indices: [2, num_edges] int64 — ~16 bytes per edge
  - Feature matrices: [num_nodes, num_features] float32 — ~4 bytes per cell
  - HeteroData stores separate feature tensors per node type, so features
    are NOT a single wide matrix across all ontologies. Each node type
    only carries features relevant to its ontology (e.g., CPI Index nodes
    get ~5-10 CPI features, Market PriceObservation nodes get ~10-15
    market features, not 200+ columns from all ontologies combined).
  - Typical per-type sizes: 100K-1M nodes × 5-20 features = 2-80 MB each
  - With 100+ node types, total feature memory: ~1-5 GB
  - Edge indices across all edge types: ~500 MB - 2 GB
  - Total driver memory for PyG object: typically 2-8 GB for 100M triples
  - Fits comfortably on Glue G.2X (32 GB) or G.4X (64 GB) workers

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
from typing import Dict, Any, Optional, Tuple

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

    All heavy computation (filtering, joining, ID assignment, aggregation)
    runs on Spark executors. Only compact integer/float tensors are collected
    to the driver for final HeteroData assembly.

    Node URI strings are NEVER collected to the driver. ID assignment
    happens entirely on Spark via monotonically_increasing_id or
    dense_rank, and only the resulting integer edge indices and
    float feature matrices cross the Spark → driver boundary.

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
    # STEP 1: Build node ID tables (on executors)
    # ============================================
    # Returns a DataFrame per node type: (uri, node_id, node_type)
    # These stay on executors — never collected as dicts.
    logger.info("Step 1/4: Building node ID tables...")

    node_mapper = NodeMapper(spark, config)
    node_id_df, node_counts = node_mapper.build_node_id_table(triples_df)

    # node_id_df: DataFrame(uri: string, node_id: long, node_type: string)
    #   Partitioned and cached on executors
    # node_counts: Dict[str, int]
    #   Small dict — one entry per node type (bounded by ontology size, ~100 entries)
    total_nodes = sum(node_counts.values())
    logger.info(
        f"  {len(node_counts)} node types, {total_nodes:,} total nodes"
    )

    if total_nodes == 0:
        logger.warning("No nodes found — returning empty HeteroData")
        triples_df.unpersist()
        return HeteroData()

    # ============================================
    # STEP 2: Build edge index tensors (on executors, collect as tensors)
    # ============================================
    # Joins triples with node_id_df on executors to resolve URIs → integer IDs,
    # then collects only the [2, num_edges] integer arrays to the driver.
    logger.info("Step 2/4: Building edge indices...")

    edge_mapper = EdgeMapper(spark, config)
    edge_indices = edge_mapper.build_edge_indices(
        triples_df, node_id_df, node_counts
    )

    # edge_indices: Dict[Tuple[str,str,str], Tensor]
    #   (src_type, relation, dst_type) -> LongTensor of shape [2, num_edges]
    total_edges = sum(t.shape[1] for t in edge_indices.values())
    logger.info(
        f"  {len(edge_indices)} edge types, {total_edges:,} total edges"
    )

    # ============================================
    # STEP 3: Build feature tensors (on executors, collect as tensors)
    # ============================================
    # Joins triples with node_id_df on executors to align features with
    # integer node IDs, aggregates per node, then collects only the
    # [num_nodes, num_features] float arrays to the driver.
    #
    # Each node type gets its own feature tensor with only the properties
    # relevant to that type's ontology. A CPI Index node carries CPI
    # features (~5-10 columns), not all 200+ properties from every ontology.
    logger.info("Step 3/4: Building feature tensors...")

    feature_extractor = FeatureExtractor(spark, config)
    feature_tensors = feature_extractor.build_features(
        triples_df, node_id_df, node_counts
    )

    # feature_tensors: Dict[str, Tensor]
    #   node_type -> FloatTensor of shape [num_nodes, num_features]
    for ntype, tensor in feature_tensors.items():
        logger.info(f"  [{ntype}] features shape: {list(tensor.shape)}")

    # ============================================
    # STEP 4: Assemble HeteroData (on driver)
    # ============================================
    # At this point we only have compact tensors on the driver.
    # No URI strings cross the Spark → driver boundary.
    logger.info("Step 4/4: Assembling HeteroData...")

    data = HeteroData()

    # Add node stores
    for node_type, num_nodes in node_counts.items():
        if node_type in feature_tensors:
            data[node_type].x = feature_tensors[node_type]
        else:
            # No features — use single-column zeros so num_nodes is set
            data[node_type].x = torch.zeros(num_nodes, 1)

        data[node_type].num_nodes = num_nodes

    # Add edge stores
    for (src_type, rel, dst_type), edge_index in edge_indices.items():
        data[src_type, rel, dst_type].edge_index = edge_index

    # Cleanup
    node_id_df.unpersist()
    triples_df.unpersist()

    logger.info("HeteroData assembly complete")
    logger.info(f"  Node types: {data.node_types}")
    logger.info(f"  Edge types: {data.edge_types}")

    return data