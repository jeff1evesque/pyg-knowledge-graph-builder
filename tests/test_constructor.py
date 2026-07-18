"""Unit tests for build_hetero_data (spark_jobs/pyg_builder/constructor.py).

``build_hetero_data`` is the assembly point: it stitches the mapped node IDs,
edge indices, and feature tensors produced by the four extractors into the final
`HeteroData`. The e2e smoke only checks that `x.dim() == 2` and
`x.shape[0] == num_nodes`; it never checks that the feature row at index *i*
actually belongs to node ID *i*, nor that per-type counts and edge endpoints line
up. A transposition or off-by-one in assembly would pass the smoke and silently
train the model on scrambled features.

These tests run the real pipeline (real NodeMapper / EdgeMapper / FeatureExtractor)
over tiny, fully-known triples on the shared local SparkSession — no cluster,
no RAPIDS. Edge features are disabled (their encoding is out of scope, Issue #202);
`edge_index` construction is independent of that flag, so edge-alignment coverage
is unaffected.

Feature↔ID alignment is asserted with an **encoding-independent sentinel**: three
same-type nodes that are identical except for one numeric literal whose value is
monotonic in the node's ID. All ontology/schema dimensions are therefore identical
across the three rows, so the *only* columns that vary are the numeric-literal
slots — and z-score normalization is strictly monotonic, so each varying column is
strictly monotonic in the node value. Any row permutation (transpose, reversal,
off-by-one, swap) breaks that monotonic ordering, so `np.argsort` of every varying
column must be the identity `[0,1,2]` (or its reverse for a sign-flipped slot).
This detects misalignment without depending on *which* value lands in *which*
slot (the encoding correctness that Issue #202 owns).
"""
import numpy as np
import torch

from spark_jobs.pyg_builder.constructor import build_hetero_data
from spark_jobs.pyg_builder.feature_extractor import VectorLayout, VECTOR_DIM

# Type / predicate URIs whose PyG names are fixed by the mappers' namespace
# tables (mirrors test_node_mapper.py / test_edge_mapper.py).
CPI_INDEX = "https://www.bls.gov/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://www.bls.gov/cpi/Series"      # -> cpi_Series
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PRECEDES = "https://www.bls.gov/enrichment/precedes"  # -> bls_enrichment_precedes
RELATION = "bls_enrichment_precedes"
METRIC = "https://example.org/metric"              # numeric literal predicate

# Edge features are out of scope here (Issue #202); disabling keeps the tests
# focused on node-feature alignment and edge_index structure, and faster.
NO_EDGE_FEATURES = {"edge_feature_config": {"enabled": False}}


def _build(spark, rows, config=None):
    """Run the real assembly over a list of (subject, predicate, object) rows."""
    triples = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    cfg = NO_EDGE_FEATURES if config is None else config
    data, metadata, _node_index = build_hetero_data(spark, triples, config=cfg)
    return data


# ======================================================================
# Node/edge types present and per-type counts match the input
# ======================================================================

def test_all_node_and_edge_types_present_with_matching_counts(spark):
    # 2 cpi_Index, 2 cpi_Series, joined by two precedes edges.
    data = _build(spark, [
        ("https://ex/s1", RDF_TYPE, CPI_INDEX),
        ("https://ex/s2", RDF_TYPE, CPI_INDEX),
        ("https://ex/o1", RDF_TYPE, CPI_SERIES),
        ("https://ex/o2", RDF_TYPE, CPI_SERIES),
        ("https://ex/s1", PRECEDES, "https://ex/o1"),
        ("https://ex/s2", PRECEDES, "https://ex/o2"),
    ])

    assert set(data.node_types) == {"cpi_Index", "cpi_Series"}
    assert data["cpi_Index"].num_nodes == 2
    assert data["cpi_Series"].num_nodes == 2

    edge_key = ("cpi_Index", RELATION, "cpi_Series")
    assert set(data.edge_types) == {edge_key}
    assert data[edge_key].edge_index.shape == (2, 2)


def test_x_shape_matches_num_nodes_for_every_type(spark):
    data = _build(spark, [
        ("https://ex/s1", RDF_TYPE, CPI_INDEX),
        ("https://ex/s2", RDF_TYPE, CPI_INDEX),
        ("https://ex/o1", RDF_TYPE, CPI_SERIES),
    ])
    for ntype in data.node_types:
        x = data[ntype].x
        assert x.dim() == 2
        assert x.shape[0] == data[ntype].num_nodes
        assert x.shape[1] == VECTOR_DIM


# ======================================================================
# Edge endpoints reference valid node IDs within each endpoint type's range
# ======================================================================

def test_edge_endpoints_within_per_type_node_id_ranges(spark):
    data = _build(spark, [
        ("https://ex/s1", RDF_TYPE, CPI_INDEX),
        ("https://ex/s2", RDF_TYPE, CPI_INDEX),
        ("https://ex/o1", RDF_TYPE, CPI_SERIES),
        ("https://ex/o2", RDF_TYPE, CPI_SERIES),
        ("https://ex/s1", PRECEDES, "https://ex/o1"),
        ("https://ex/s2", PRECEDES, "https://ex/o2"),
        ("https://ex/s1", PRECEDES, "https://ex/o2"),
    ])

    for (src_type, _rel, dst_type) in data.edge_types:
        ei = data[(src_type, _rel, dst_type)].edge_index
        num_src = data[src_type].num_nodes
        num_dst = data[dst_type].num_nodes
        # Row 0 = source IDs, row 1 = destination IDs; each within [0, count).
        assert int(ei[0].min()) >= 0
        assert int(ei[0].max()) < num_src, "source ID out of range"
        assert int(ei[1].min()) >= 0
        assert int(ei[1].max()) < num_dst, "destination ID out of range"


# ======================================================================
# Feature ↔ node-ID alignment (encoding-independent sentinel)
# ======================================================================

def test_feature_row_belongs_to_its_node_id(spark):
    # Three same-type nodes, identical but for a numeric literal that increases
    # with the node's (URI-sorted) ID: n0->10, n1->20, n2->30 → IDs 0,1,2.
    data = _build(spark, [
        ("https://ex/n0", RDF_TYPE, CPI_INDEX),
        ("https://ex/n1", RDF_TYPE, CPI_INDEX),
        ("https://ex/n2", RDF_TYPE, CPI_INDEX),
        ("https://ex/n0", METRIC, "10"),
        ("https://ex/n1", METRIC, "20"),
        ("https://ex/n2", METRIC, "30"),
    ])

    x = data["cpi_Index"].x.numpy()
    assert x.shape == (3, VECTOR_DIM)

    # Only the numeric-literal slots may differ across the three rows; ontology
    # and property-schema dims are identical (same type, same single predicate).
    varying = np.where(x.max(axis=0) - x.min(axis=0) > 1e-6)[0]
    assert varying.size >= 1, "sentinel produced no distinguishing feature"

    layout = VectorLayout(VECTOR_DIM)
    lo = layout.seg3_numeric_start
    hi = layout.seg3_numeric_start + layout.seg3_numeric_dim
    assert all(lo <= c < hi for c in varying), (
        "distinguishing columns must live in the numeric-literal segment"
    )

    # Each varying column is strictly monotonic in the node value (z-score is
    # monotonic; a slot may be sign-flipped). Any row permutation other than the
    # identity breaks this, so argsort must be [0,1,2] (or its sign-flip reverse).
    for c in varying:
        order = tuple(np.argsort(x[:, c]).tolist())
        assert order in {(0, 1, 2), (2, 1, 0)}, (
            f"feature rows are misaligned to node IDs at column {c}: {order}"
        )

    # The rows are genuinely distinct (the sentinel is observable).
    assert not np.array_equal(x[0], x[1])
    assert not np.array_equal(x[1], x[2])


# ======================================================================
# Node types with no literal features — contract: full-width tensor present
# ======================================================================

def test_featureless_type_gets_full_width_tensor_not_absent(spark):
    # Nodes carry only rdf:type — no numeric or categorical literals. Contract:
    # x is always present with shape (num_nodes, VECTOR_DIM); never absent/empty.
    data = _build(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
    ])
    store = data["cpi_Index"]
    assert hasattr(store, "x") and store.x is not None
    assert store.x.shape == (2, VECTOR_DIM)
    assert store.x.dtype == torch.float32
    assert store.x.dim() == 2


# ======================================================================
# Edge cases: empty input and single node type must not crash
# ======================================================================

def test_empty_input_returns_empty_hetero_data(spark):
    # No rdf:type triples → no nodes → empty HeteroData (early return path).
    data = _build(spark, [
        ("https://ex/a", PRECEDES, "https://ex/b"),
    ])
    assert list(data.node_types) == []
    assert list(data.edge_types) == []


def test_single_node_type_no_edges_does_not_crash(spark):
    data = _build(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
    ])
    assert list(data.node_types) == ["cpi_Index"]
    assert list(data.edge_types) == []
    assert data["cpi_Index"].num_nodes == 2
