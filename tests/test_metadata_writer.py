"""
Unit tests for MetadataCollector and the metadata-writer helpers.

MetadataCollector produces the six JSON metadata files that ship alongside the
``.pt`` graph (graph_schema, feature_spec, normalization, encoding_config,
ontology_schema, slot_mapping). This is almost pure Python — no DataFrame work,
no SparkSession — so it is the cheapest place to pin down *content*: node/edge
counts, type names, feature-segment structure, and the ``.pt``-key → metadata
prefix derivation used by the archive path.

The e2e smoke only asserts these files exist and are non-empty; these tests
assert what is actually *in* them, so a wrong count, a dropped segment, or a
mis-derived prefix fails loudly here instead of shipping silently.

Pure Python: runs under ``pytest -m "not e2e"`` with no Spark fixture.
"""
import json

import pytest

from spark_jobs.pyg_builder.feature_extractor import VectorLayout
from spark_jobs.pyg_builder.edge_feature_extractor import EdgeVectorLayout
from spark_jobs.pyg_builder.metadata_writer import (
    MetadataCollector,
    _sanitize_config,
    derive_metadata_prefix,
    write_metadata_to_local,
)


# Real layouts so the segment arithmetic in _build_feature_spec is exercised
# against the exact same to_dict() contract the constructor feeds it.
NODE_LAYOUT = VectorLayout(1024).to_dict()
EDGE_LAYOUT = EdgeVectorLayout(64).to_dict()

# The six files to_metadata_files() must always emit.
EXPECTED_FILES = {
    "graph_schema.json",
    "feature_spec.json",
    "normalization.json",
    "encoding_config.json",
    "ontology_schema.json",
    "slot_mapping.json",
}


def _fully_registered_collector():
    """
    A collector fed a known, self-consistent set of node/edge types, layouts,
    normalization stats, and the three registered sub-builder payloads.

    Two node types (measurement, entity) and two edge types (one with derived
    features, one without) — enough to exercise every summary counter.
    """
    collector = MetadataCollector(
        time_period="2099-01",
        vector_dim=1024,
        edge_vector_dim=64,
        edge_features_enabled=True,
        config={"source": "test", "api_key": "SECRET", "opt": None},
    )

    collector.register_node_types(
        node_counts={"Observation": 12, "Company": 5},
        node_type_uris={
            "Observation": "http://ex/Observation",
            "Company": "http://ex/Company",
        },
        node_type_categories={
            "Observation": "measurement",
            "Company": "entity",
        },
    )

    collector.register_edge_types(
        edge_counts={
            ("Company", "reports", "Observation"): 30,
            ("Company", "sameAs", "Company"): 4,
        },
        edge_predicate_uris={
            "reports": "http://ex/reports",
            "sameAs": "http://ex/sameAs",
        },
        edge_origins={"reports": "raw", "sameAs": "unification"},
        edge_feature_flags={"reports": True, "sameAs": False},
        edge_feature_dims={"reports": 64, "sameAs": 0},
    )

    collector.register_node_feature_layout(NODE_LAYOUT)
    collector.register_edge_feature_layout(EDGE_LAYOUT)

    collector.register_normalization_stats(
        stats=[
            {"predicate": "http://ex/value", "mu": 1.5, "sigma": 0.5,
             "count": 12},
            {"predicate": "http://ex/rank", "mu": 0.0, "sigma": 1.0,
             "count": 5},
        ],
        zero_variance_properties=["http://ex/rank"],
    )
    collector.register_edge_normalization_stats(
        {"(Company, reports, Observation)": [
            {"property": "delta", "mu": 2.0, "sigma": 1.0}]}
    )

    collector.register_encoding_config(
        {"version": "1.0", "algorithm": "murmur3", "hash_seed": 42}
    )
    collector.register_ontology_schema(
        {"version": "1.0", "classes": {"Observation": {"parent": "Thing"}}}
    )
    collector.register_slot_mapping(
        {"version": "1.0", "slots": {"http://ex/value": 640}}
    )
    collector.register_edge_feature_classification(
        categories={"reports": "temporal_numeric", "sameAs": "skip"},
        types_with_features=["(Company, reports, Observation)"],
        types_without_features=["(Company, sameAs, Company)"],
    )
    return collector


# ======================================================================
# to_metadata_files — the six-key envelope
# ======================================================================

def test_to_metadata_files_emits_exactly_six_named_files():
    files = _fully_registered_collector().to_metadata_files()
    assert set(files.keys()) == EXPECTED_FILES
    # Every file is a non-empty dict and JSON-serializable.
    for name, content in files.items():
        assert isinstance(content, dict) and content, name
        json.dumps(content, default=str)


# ======================================================================
# graph_schema.json — counts, type names, summary
# ======================================================================

def test_graph_schema_node_types():
    schema = _fully_registered_collector()._build_graph_schema()
    nodes = schema["node_types"]
    assert set(nodes) == {"Observation", "Company"}
    assert nodes["Observation"] == {
        "count": 12,
        "source_type_uri": "http://ex/Observation",
        "category": "measurement",
        "has_features": True,
    }
    assert nodes["Company"]["category"] == "entity"


def test_graph_schema_edge_types_keyed_and_detailed():
    schema = _fully_registered_collector()._build_graph_schema()
    edges = schema["edge_types"]
    assert set(edges) == {
        "(Company, reports, Observation)",
        "(Company, sameAs, Company)",
    }
    reports = edges["(Company, reports, Observation)"]
    assert reports == {
        "src_type": "Company",
        "relation": "reports",
        "dst_type": "Observation",
        "count": 30,
        "predicate_uri": "http://ex/reports",
        "origin": "raw",
        "has_features": True,
        "feature_dim": 64,
    }


def test_graph_schema_summary_counts():
    schema = _fully_registered_collector()._build_graph_schema()
    summary = schema["summary"]
    assert summary["total_node_types"] == 2
    assert summary["total_edge_types"] == 2
    assert summary["total_nodes"] == 17          # 12 + 5
    assert summary["total_edges"] == 34          # 30 + 4
    assert summary["node_types_with_literal_features"] == 2
    assert summary["edge_types_with_features"] == 1
    assert summary["edge_types_without_features"] == 1


def test_graph_schema_build_metadata_and_config_sanitized():
    schema = _fully_registered_collector()._build_graph_schema()
    meta = schema["build_metadata"]
    assert meta["time_period"] == "2099-01"
    assert "build_timestamp" in meta
    # Config is passed through _sanitize_config (None survives, stays JSON-safe).
    cfg = meta["pipeline_config"]
    assert cfg["source"] == "test"
    assert cfg["opt"] is None
    json.dumps(schema, default=str)


def test_graph_schema_zero_count_node_has_no_features():
    c = MetadataCollector("2099-01", 1024, 64, True, {})
    c.register_node_types(
        node_counts={"Empty": 0},
        node_type_uris={"Empty": "http://ex/Empty"},
    )
    nodes = c._build_graph_schema()["node_types"]
    assert nodes["Empty"]["has_features"] is False
    # Default category when none registered.
    assert nodes["Empty"]["category"] == "entity"


# ======================================================================
# feature_spec.json — segment structure
# ======================================================================

def test_feature_spec_node_segments_shape():
    spec = _fully_registered_collector()._build_feature_spec()
    node = spec["node_features"]
    assert node["total_dim"] == 1024
    segs = node["segments"]
    assert [s["name"] for s in segs] == [
        "ontology_structure", "property_schema", "literal_values",
    ]
    # First segment mirrors the VectorLayout(1024) boundaries exactly.
    seg1 = segs[0]
    assert seg1["start"] == NODE_LAYOUT["seg1_start"]
    assert seg1["dim"] == NODE_LAYOUT["seg1_total"]
    assert seg1["end"] == NODE_LAYOUT["seg1_start"] + NODE_LAYOUT["seg1_total"] - 1
    assert [ss["name"] for ss in seg1["sub_segments"]] == [
        "class_identity", "class_hierarchy", "ontology_source",
    ]


def test_feature_spec_node_segments_are_contiguous():
    spec = _fully_registered_collector()._build_feature_spec()
    subs = [
        ss
        for seg in spec["node_features"]["segments"]
        for ss in seg["sub_segments"]
    ]
    cursor = 0
    for ss in subs:
        assert ss["start"] == cursor
        assert ss["end"] == ss["start"] + ss["dim"] - 1
        cursor = ss["end"] + 1
    assert cursor == 1024


def test_feature_spec_edge_segments_present_when_enabled():
    spec = _fully_registered_collector()._build_feature_spec()
    edge = spec["edge_features"]
    assert edge["enabled"] is True
    assert edge["total_dim"] == 64
    assert [s["name"] for s in edge["segments"]] == [
        "temporal_signals", "numeric_contrast", "relational_context",
    ]
    assert edge["edge_types_with_features"] == [
        "(Company, reports, Observation)"]
    assert edge["edge_types_without_features"] == ["(Company, sameAs, Company)"]
    # "skip" categories are filtered out of derivation_methods.
    assert edge["derivation_methods"] == {"reports": "temporal_numeric"}


def test_feature_spec_edge_disabled_zeros_dim_and_drops_segments():
    c = _fully_registered_collector()
    c._edge_features_enabled = False
    edge = c._build_feature_spec()["edge_features"]
    assert edge["enabled"] is False
    assert edge["total_dim"] == 0
    assert edge["segments"] == []


# ======================================================================
# normalization.json
# ======================================================================

def test_normalization_maps_stat_keys():
    norm = _fully_registered_collector()._build_normalization()
    assert norm["method"] == "z-score"
    node = norm["node_properties"]
    assert node["total_properties_normalized"] == 2
    assert node["zero_variance_properties"] == ["http://ex/rank"]
    # Internal (predicate/mu/sigma) → documented (predicate_uri/mean/std).
    assert node["per_property"][0] == {
        "predicate_uri": "http://ex/value",
        "mean": 1.5,
        "std": 0.5,
        "count": 12,
    }
    assert norm["edge_derived_features"] == {
        "(Company, reports, Observation)": [
            {"property": "delta", "mu": 2.0, "sigma": 1.0}]
    }


def test_normalization_empty_when_unregistered():
    norm = MetadataCollector("2099-01", 1024, 64, True, {})._build_normalization()
    assert norm["node_properties"]["total_properties_normalized"] == 0
    assert norm["node_properties"]["per_property"] == []
    assert norm["edge_derived_features"] == {}


# ======================================================================
# encoding_config / ontology_schema / slot_mapping — passthrough + fallback
# ======================================================================

def test_registered_sub_builders_pass_through():
    c = _fully_registered_collector()
    assert c._build_encoding_config()["hash_seed"] == 42
    assert c._build_ontology_schema()["classes"] == {
        "Observation": {"parent": "Thing"}}
    assert c._build_slot_mapping()["slots"] == {"http://ex/value": 640}


def test_sub_builders_fallback_when_unregistered():
    c = MetadataCollector("2099-01", 1024, 64, True, {})
    for content in (
        c._build_encoding_config(),
        c._build_ontology_schema(),
        c._build_slot_mapping(),
    ):
        assert content["version"] == "1.0"
        assert "not explicitly registered" in content["note"]


# ======================================================================
# derive_metadata_prefix — .pt key → metadata dir
# ======================================================================

@pytest.mark.parametrize("key,expected", [
    ("pyg/2099-01/hetero_data.pt", "pyg/2099-01/metadata/"),
    ("pyg/2024-12/hetero_data.pt", "pyg/2024-12/metadata/"),
    ("pyg/2099-01/hetero_data_512d.pt", "pyg/2099-01/hetero_data_512d_metadata/"),
    ("hetero_data.pt", "metadata/"),
    ("experiment_a.pt", "experiment_a_metadata/"),
    # No .pt extension: stem is still exactly "hetero_data" → "metadata/".
    ("pyg/2099-01/hetero_data", "pyg/2099-01/metadata/"),
    # A non-default stem without extension keeps the "{stem}_metadata/" form.
    ("pyg/2099-01/experiment_a", "pyg/2099-01/experiment_a_metadata/"),
])
def test_derive_metadata_prefix(key, expected):
    assert derive_metadata_prefix(key) == expected


# ======================================================================
# _sanitize_config — secrets/None handling, JSON-serializable output
# ======================================================================

def test_sanitize_config_preserves_scalars_and_none():
    out = _sanitize_config({"s": "x", "i": 1, "f": 1.5, "b": True, "n": None})
    assert out == {"s": "x", "i": 1, "f": 1.5, "b": True, "n": None}


def test_sanitize_config_recurses_and_sorts_sets():
    out = _sanitize_config({
        "nested": {"deep": {"k": None}},
        "items": ["a", {"k": "v"}],
        "tags": {"z", "a", "m"},
    })
    assert out["nested"] == {"deep": {"k": None}}
    assert out["items"] == ["a", {"k": "v"}]
    assert out["tags"] == ["a", "m", "z"]          # set → sorted list


def test_sanitize_config_stringifies_non_serializable():
    class Weird:
        def __str__(self):
            return "weird-value"

    out = _sanitize_config({"obj": Weird()})
    assert out["obj"] == "weird-value"
    # Whole result must be JSON-serializable without a custom encoder.
    json.dumps(out)


# ======================================================================
# write_metadata_to_local — six files land on disk, round-trip as JSON
# ======================================================================

def test_write_metadata_to_local_writes_all_six(tmp_path):
    files = _fully_registered_collector().to_metadata_files()
    dest = tmp_path / "meta"
    write_metadata_to_local(files, str(dest))

    written = {p.name for p in dest.iterdir()}
    assert written == EXPECTED_FILES

    # Each file is non-empty and round-trips to the same content.
    for name, content in files.items():
        path = dest / name
        assert path.stat().st_size > 0
        assert json.loads(path.read_text()) == content


def test_write_metadata_to_local_creates_missing_dir(tmp_path):
    dest = tmp_path / "a" / "b" / "metadata"
    write_metadata_to_local({"graph_schema.json": {"version": "1.0"}}, str(dest))
    assert (dest / "graph_schema.json").exists()
