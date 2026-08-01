"""Unit tests for EdgeMapper (spark_jobs/pyg_builder/edge_mapper.py).

EdgeMapper joins triples against the node ID table to produce the per-edge-type
[2, num_edges] edge_index tensors PyG is built on. The e2e smoke only checks that
edge_index is 2-D with 2 rows; these tests pin the value-level contract on tiny
in-memory fixtures over the shared local SparkSession:

  * URIs resolve to the integer IDs the node table assigned (src row 0, dst row 1),
  * the edge-type key is (src_type, relation, dst_type) with the documented name,
  * within a type edges are ordered (src_id ASC, dst_id ASC) — the alignment
    contract EdgeFeatureExtractor relies on,
  * excluded predicates, literal/untyped endpoints, and duplicate edges are dropped,
  * the edge_types whitelist filters by relation,
  * get_predicate_uri_mapping inverts the relation name back to its predicate URI,
  * the result is reproducible run-to-run.

The node ID table is built directly (not via NodeMapper) so the resolved IDs are
fully known and this test isolates EdgeMapper.
"""
import torch

from spark_jobs.pyg_builder.edge_mapper import (
    EdgeMapper,
    RDF_TYPE,
    _build_predicate_to_relation_expr,
)

# Built from the namespace table rather than spelled out, so re-homing the
# minted vocabulary cannot leave this pointing at a namespace nothing matches
# -- which silently renames the relation to its bare last segment.
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT  # noqa: E402

PRECEDES = f"{BLS_ENRICHMENT}precedes"  # -> bls_enrichment_precedes
RELATION = "bls_enrichment_precedes"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"  # excluded predicate

# Fully-known node ID table: two cpi_Index sources, two cpi_Series destinations.
NODE_ROWS = [
    ("https://ex/s1", 0, "cpi_Index"),
    ("https://ex/s2", 1, "cpi_Index"),
    ("https://ex/o1", 0, "cpi_Series"),
    ("https://ex/o2", 1, "cpi_Series"),
]
NODE_COUNTS = {"cpi_Index": 2, "cpi_Series": 2}
EDGE_KEY = ("cpi_Index", RELATION, "cpi_Series")


def _node_id_df(spark):
    return spark.createDataFrame(NODE_ROWS, ["uri", "node_id", "node_type"])


def _build(spark, triple_rows, config=None):
    triples = spark.createDataFrame(
        triple_rows, schema="subject STRING, predicate STRING, object STRING"
    )
    return EdgeMapper(spark, config or {}).build_edge_indices(
        triples, _node_id_df(spark), NODE_COUNTS
    )


# ======================================================================
# Relation naming (pure Column expression)
# ======================================================================

def test_predicate_to_relation_maps_known_namespace(spark):
    df = spark.createDataFrame([(PRECEDES,)], ["predicate"])
    rel = df.withColumn(
        "r", _build_predicate_to_relation_expr("predicate")
    ).collect()[0]["r"]
    assert rel == RELATION


def test_predicate_to_relation_falls_back_to_last_segment(spark):
    df = spark.createDataFrame([("http://nonexistent.invalid/links",)], ["predicate"])
    rel = df.withColumn(
        "r", _build_predicate_to_relation_expr("predicate")
    ).collect()[0]["r"]
    assert rel == "unknown_links"


# ======================================================================
# Resolution, edge-type key, and [2, E] tensor contents
# ======================================================================

def test_edges_resolve_to_node_ids_with_correct_key(spark):
    edge_indices, _ = _build(spark, [
        ("https://ex/s1", PRECEDES, "https://ex/o1"),  # (0, 0)
        ("https://ex/s2", PRECEDES, "https://ex/o1"),  # (1, 0)
    ])
    assert set(edge_indices) == {EDGE_KEY}
    ei = edge_indices[EDGE_KEY]
    assert ei.dtype == torch.int64
    assert ei.shape == (2, 2)
    # Ordered (src_id ASC, dst_id ASC): (0,0), (1,0).
    assert ei[0].tolist() == [0, 1]  # source IDs
    assert ei[1].tolist() == [0, 0]  # destination IDs


def test_within_type_ordering_is_src_then_dst(spark):
    # Feed edges out of order; expect (src ASC, dst ASC): (0,0),(0,1),(1,0).
    edge_indices, _ = _build(spark, [
        ("https://ex/s2", PRECEDES, "https://ex/o1"),  # (1, 0)
        ("https://ex/s1", PRECEDES, "https://ex/o2"),  # (0, 1)
        ("https://ex/s1", PRECEDES, "https://ex/o1"),  # (0, 0)
    ])
    ei = edge_indices[EDGE_KEY]
    assert ei[0].tolist() == [0, 0, 1]
    assert ei[1].tolist() == [0, 1, 0]


# ======================================================================
# Filtering: excluded predicates, non-node endpoints, duplicates
# ======================================================================

def test_rdf_type_and_structural_predicates_produce_no_edges(spark):
    edge_indices, _ = _build(spark, [
        ("https://ex/s1", RDF_TYPE, "https://ex/o1"),
        ("https://ex/s1", RDFS_LABEL, "https://ex/o1"),
    ])
    assert edge_indices == {}


def test_endpoints_absent_from_node_table_are_dropped(spark):
    edge_indices, _ = _build(spark, [
        ("https://ex/s1", PRECEDES, "42"),                 # literal object
        ("https://ex/unknown", PRECEDES, "https://ex/o1"),  # untyped subject
        ("https://ex/s1", PRECEDES, "https://ex/o1"),       # the only real edge
    ])
    ei = edge_indices[EDGE_KEY]
    assert ei.shape == (2, 1)
    assert ei[:, 0].tolist() == [0, 0]


def test_duplicate_edges_are_collapsed(spark):
    edge_indices, _ = _build(spark, [
        ("https://ex/s1", PRECEDES, "https://ex/o1"),
        ("https://ex/s1", PRECEDES, "https://ex/o1"),
    ])
    assert edge_indices[EDGE_KEY].shape == (2, 1)


# ======================================================================
# Config whitelist and reverse URI mapping
# ======================================================================

def test_edge_types_whitelist_filters_by_relation(spark):
    kept, _ = _build(
        spark,
        [("https://ex/s1", PRECEDES, "https://ex/o1")],
        config={"edge_types": [RELATION]},
    )
    assert set(kept) == {EDGE_KEY}

    dropped, _ = _build(
        spark,
        [("https://ex/s1", PRECEDES, "https://ex/o1")],
        config={"edge_types": ["some_other_relation"]},
    )
    assert dropped == {}


def test_get_predicate_uri_mapping_inverts_relation_name(spark):
    mapper = EdgeMapper(spark, {})
    triples = spark.createDataFrame(
        [("https://ex/s1", PRECEDES, "https://ex/o1")],
        schema="subject STRING, predicate STRING, object STRING",
    )
    _, edges_final = mapper.build_edge_indices(
        triples, _node_id_df(spark), NODE_COUNTS
    )
    assert mapper.get_predicate_uri_mapping(edges_final) == {RELATION: PRECEDES}


# ======================================================================
# Determinism
# ======================================================================

def test_edge_indices_are_deterministic_across_runs(spark):
    rows = [
        ("https://ex/s2", PRECEDES, "https://ex/o2"),
        ("https://ex/s1", PRECEDES, "https://ex/o1"),
        ("https://ex/s1", PRECEDES, "https://ex/o2"),
    ]
    first, _ = _build(spark, rows)
    second, _ = _build(spark, rows)
    assert set(first) == set(second)
    assert torch.equal(first[EDGE_KEY], second[EDGE_KEY])
