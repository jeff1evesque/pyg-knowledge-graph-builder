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
    _round_floats,
    _sanitize_config,
    derive_metadata_prefix,
    derive_node_index_prefix,
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

    # Only Observation carries literal values -- Company is a pure entity type,
    # the taxonomy-node shape that makes has_features discriminating. Mirrors
    # the edge types below, where one has features and one does not.
    collector.register_node_literal_features(["Observation"])

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
    # Company registered no literal features, so the flag distinguishes it.
    assert nodes["Company"]["has_features"] is False


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
    # 1 of 2, not 2 of 2: the counter follows the per-type has_features flag,
    # so it can differ from total_node_types. Through schema 1.0 it was derived
    # from `count > 0` and was therefore always equal to it.
    assert summary["node_types_with_literal_features"] == 1
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


def test_graph_schema_unregistered_literal_features_defaults_to_false():
    """A collector that never reached feature building reports False, not True.

    Was test_graph_schema_zero_count_node_has_no_features, back when
    has_features was `count > 0`. The flag no longer has anything to do with
    the count, so what matters now is the absent-information case: an unknown
    fact must not be published as a positive claim.
    """
    c = MetadataCollector("2099-01", 1024, 64, True, {})
    c.register_node_types(
        node_counts={"Empty": 0, "Populated": 7},
        node_type_uris={"Empty": "http://ex/Empty"},
    )
    nodes = c._build_graph_schema()["node_types"]

    assert nodes["Empty"]["has_features"] is False
    # And a non-zero count does NOT imply features -- the old bug exactly.
    assert nodes["Populated"]["count"] == 7
    assert nodes["Populated"]["has_features"] is False
    assert c._build_graph_schema()["summary"][
        "node_types_with_literal_features"
    ] == 0
    # Default category when none registered.
    assert nodes["Empty"]["category"] == "entity"


def test_graph_schema_literal_features_are_independent_of_node_count():
    """The regression guard: has_features must track features, not counts.

    Through schema 1.0 this field was `count > 0`, so every type with nodes
    claimed features and summary.node_types_with_literal_features was identical
    to total_node_types by construction (observed: 100 and 100 on a real
    build). Pins the two apart -- the type with FEWER nodes is the one with
    features here, so any reintroduction of a count-derived flag fails.
    """
    c = MetadataCollector("2099-01", 1024, 64, True, {})
    c.register_node_types(
        node_counts={"Taxonomy": 500, "Measurement": 3},
        node_type_uris={},
    )
    c.register_node_literal_features(["Measurement"])

    schema = c._build_graph_schema()
    nodes = schema["node_types"]
    assert nodes["Taxonomy"]["has_features"] is False
    assert nodes["Measurement"]["has_features"] is True
    assert schema["summary"]["node_types_with_literal_features"] == 1
    assert schema["summary"]["total_node_types"] == 2


def test_graph_schema_version_is_bumped_for_the_new_semantics():
    """1.1 signals the changed meaning of node-type has_features.

    A consumer reading 1.0 and 1.1 gets different values from the same field,
    so the version is the only way to tell the two apart.
    """
    assert _fully_registered_collector()._build_graph_schema()["version"] == "1.1"


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


def test_feature_spec_sub_segments_default_to_claiming_features():
    # No status registered — every sub-segment claims features, so the e2e
    # vacuity guard checks them all rather than silently exempting any.
    spec = _fully_registered_collector()._build_feature_spec()
    subs = [s for seg in spec["node_features"]["segments"]
            for s in seg["sub_segments"]]
    assert subs, "node segments carry no sub-segments"
    assert all(s["populated"] is True for s in subs)
    assert all("empty_reason" not in s for s in subs)


def test_feature_spec_records_why_a_sub_segment_is_empty():
    c = _fully_registered_collector()
    c.register_sub_segment_status(
        {"class_hierarchy": "no rdfs:subClassOf in source data"}
    )
    subs = {s["name"]: s for seg in c._build_feature_spec()[
        "node_features"]["segments"] for s in seg["sub_segments"]}

    dead = subs["class_hierarchy"]
    assert dead["populated"] is False
    assert dead["empty_reason"] == "no rdfs:subClassOf in source data"
    # Its dims are still declared — the slice exists in the vector, it just
    # carries nothing. Reclaiming them is a layout change, not a spec edit.
    assert dead["dim"] > 0
    # Siblings are unaffected.
    assert subs["class_identity"]["populated"] is True


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
    assert edge["derivation_methods"] == {}


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
    # Hive-partitioned parent: the helper splits the parent off the key rather
    # than parsing it, so the partition directories carry through untouched and
    # a period's artifacts stay together. This is what production keys look like.
    (
        "pyg/year=2024/month=12/hetero_data.pt",
        "pyg/year=2024/month=12/metadata/",
    ),
    (
        "pyg/year=2024/month=12/hetero_data_512d.pt",
        "pyg/year=2024/month=12/hetero_data_512d_metadata/",
    ),
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


@pytest.mark.parametrize("key,expected", [
    ("pyg/2099-01/hetero_data.pt", "pyg/2099-01/node_index/"),
    ("pyg/2099-01/hetero_data_512d.pt", "pyg/2099-01/hetero_data_512d_node_index/"),
    (
        "pyg/year=2024/month=12/hetero_data.pt",
        "pyg/year=2024/month=12/node_index/",
    ),
    ("hetero_data.pt", "node_index/"),
    ("experiment_a.pt", "experiment_a_node_index/"),
    ("pyg/2099-01/hetero_data", "pyg/2099-01/node_index/"),
])
def test_derive_node_index_prefix(key, expected):
    assert derive_node_index_prefix(key) == expected


def test_node_index_and_metadata_prefixes_are_siblings():
    """The identity map sits beside the metadata it belongs to, not inside it.

    Both share _derive_sibling_prefix, so experiment variants keep their
    artifacts separated the same way rather than colliding in one directory.
    """
    key = "pyg/2099-01/hetero_data_512d.pt"
    meta = derive_metadata_prefix(key)
    index = derive_node_index_prefix(key)

    assert meta.rsplit("/", 2)[0] == index.rsplit("/", 2)[0]
    assert meta != index
    assert not index.startswith(meta)


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


# ======================================================================
# Encoding contract digest (encoding_config.checksum)
# ======================================================================

def _encoding_config(**overrides):
    """A minimal but realistic merged node+edge encoding config."""
    config = {
        "version": "1.0",
        "hash_algorithm": "spark_murmur3",
        "node_features": {
            "total_dim": 1024,
            "class_identity": {"dim": 256, "seeds": [0, 7, 13, 31]},
        },
        "edge_features": {
            "total_dim": 32,
            "temporal_signals": {"time_delta_seed": 0},
        },
    }
    config.update(overrides)
    return config


def _digest_of(config):
    collector = MetadataCollector("2099-01", 1024, 32, True, {})
    collector.register_encoding_config(config)
    written = collector.to_metadata_files()["encoding_config.json"]
    return written["checksum"]["contract_digest"]


def test_encoding_digest_is_stable_for_identical_config():
    """Same contract in, same digest out — it must not depend on dict order."""
    assert _digest_of(_encoding_config()) == _digest_of(_encoding_config())


def test_encoding_digest_changes_when_a_seed_changes():
    """A different hash seed invalidates a trained model, so it must be detected.

    This is the case the old "checksum" could not see: it recorded only the
    total vector dimension, which is unchanged here.
    """
    baseline = _encoding_config()
    reseeded = _encoding_config()
    reseeded["node_features"]["class_identity"]["seeds"] = [1, 7, 13, 31]

    assert (
        baseline["node_features"]["total_dim"]
        == reseeded["node_features"]["total_dim"]
    ), "fixture must hold dimensions constant, or it proves nothing"
    assert _digest_of(baseline) != _digest_of(reseeded)


def test_encoding_digest_changes_when_a_layout_dim_changes():
    changed = _encoding_config()
    changed["node_features"]["class_identity"]["dim"] = 512
    assert _digest_of(_encoding_config()) != _digest_of(changed)


def test_encoding_digest_covers_the_edge_side_too():
    """Both halves are merged before hashing.

    The old per-extractor checksums collided on the same top-level key, so the
    shallow merge in constructor.py dropped the node one entirely.
    """
    changed = _encoding_config()
    changed["edge_features"]["temporal_signals"]["time_delta_seed"] = 99
    assert _digest_of(_encoding_config()) != _digest_of(changed)


def test_encoding_digest_ignores_a_previous_checksum_field():
    """Re-stamping a config that already carries a checksum is a no-op.

    The digest is computed over the contract with any existing checksum
    removed, so it does not hash its own previous output.
    """
    stamped = _encoding_config()
    stamped["checksum"] = {"algorithm": "sha256", "contract_digest": "stale"}
    assert _digest_of(stamped) == _digest_of(_encoding_config())


# ======================================================================
# Edge origin — raw vs pipeline-minted
# ======================================================================

def test_edge_origin_is_populated_not_unknown():
    """graph_schema.json must say where each edge came from.

    Every edge type used to report "unknown": the field was plumbed and
    documented but constructor.py never passed edge_origins. A consumer could
    not tell a reported fact from a link the enrichment pipeline inferred, which
    are worth very different trust to a model.
    """
    c = MetadataCollector("2099-01", 1024, 64, True, {})
    c.register_edge_types(
        edge_counts={
            ("Company", "reports", "Observation"): 3,
            ("Sector", "correlatesWith", "Sector"): 2,
        },
        edge_predicate_uris={
            "reports": "https://www.bls.gov/cpi/reports",
            "correlatesWith": (
                "https://www.bls.gov/enrichment/correlatesWith"
            ),
        },
        edge_origins={"reports": "raw", "correlatesWith": "enrichment"},
    )
    edges = c._build_graph_schema()["edge_types"]

    origins = {v["relation"]: v["origin"] for v in edges.values()}
    assert origins == {"reports": "raw", "correlatesWith": "enrichment"}
    assert "unknown" not in origins.values()


def test_edge_origin_falls_back_to_unknown_when_not_supplied():
    """The fallback still exists — this pins that it is the ONLY way to get it.

    If a future caller forgets edge_origins, "unknown" reappears; that is the
    signal something regressed, not a normal value.
    """
    c = MetadataCollector("2099-01", 1024, 64, True, {})
    c.register_edge_types(
        edge_counts={("A", "rel", "B"): 1},
        edge_predicate_uris={"rel": "http://ex/rel"},
    )
    edges = c._build_graph_schema()["edge_types"]
    assert next(iter(edges.values()))["origin"] == "unknown"


# ======================================================================
# _round_floats — keeps normalization.json byte-reproducible
# ======================================================================

def test_round_floats_collapses_last_ulp_drift():
    """The two std values actually observed drifting between twin runs.

    stddev is a parallel reduction and parallel float reductions are not
    order-deterministic, so two runs over identical data produced statistics
    differing in the final ULP. The node tensors still compared equal (they are
    float32; both values round to the same float32), but normalization.json did
    not — which broke the byte-reproducibility the e2e tests assert. Pinned with
    the real values so the guard cannot be weakened without noticing.
    """
    a, b = 265.6434939473693, 265.64349394736934
    assert a != b
    assert _round_floats(a) == _round_floats(b)
    assert json.dumps(_round_floats(a)) == json.dumps(_round_floats(b))


def test_round_floats_preserves_meaningful_precision():
    # Differences well inside 12 significant digits must survive — the rounding
    # discards noise, not signal.
    assert _round_floats(1.23456789012) != _round_floats(1.23456789013)


def test_round_floats_recurses_and_leaves_non_floats_alone():
    out = _round_floats(
        {"per_property": [{"std": 265.64349394736934, "count": 438}],
         "flag": True, "name": "x", "none": None}
    )
    assert out["per_property"][0]["std"] == _round_floats(265.6434939473693)
    # count must stay an int, not become a float
    assert out["per_property"][0]["count"] == 438
    assert isinstance(out["per_property"][0]["count"], int)
    # bool is an int subclass — it must not be rounded into a number
    assert out["flag"] is True
    assert out["name"] == "x" and out["none"] is None


@pytest.mark.parametrize("value", [0.0, float("inf"), float("-inf")])
def test_round_floats_handles_non_finite_and_zero(value):
    # log10 of these is undefined or -inf; they must pass through untouched.
    assert _round_floats(value) == value or value != value
