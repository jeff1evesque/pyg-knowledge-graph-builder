"""Value-level unit tests for FeatureExtractor node-feature encoding
(spark_jobs/pyg_builder/feature_extractor.py).

Only VectorLayout's descriptor geometry was previously tested — never the
*encoding*, i.e. that a given triple lands in the correct vector slot with the
correct value. These tests drive build_features over tiny synthetic triples on
the shared local SparkSession and assert the encoded rows at the value level:

  * class-identity one-hots occupy exactly the layout-reserved slots (the same
    F.hash(type_uri, seed) the encoder uses, reproduced here — never hardcoded);
  * numeric literals are z-score normalized to the expected value in the numeric
    slot (F.hash(predicate, 500));
  * categorical values set exactly the expected multi-hot slots;
  * _scatter_all_types places each type's rows at the right node IDs;
  * encoding the same fixture repeatedly yields identical vectors — the direct
    regression guard for the jolts_* categorical non-determinism (Issue open).

Vector width is pinned to 128 for speed; every assertion derives its slots from
VectorLayout(128), so nothing depends on the production 1024 default.
"""
import itertools
import json

import numpy as np
import pytest
from pyspark.sql import functions as F

from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.pyg_builder.feature_extractor import (
    FeatureExtractor,
    VectorLayout,
    _HASH_SEEDS,
    _compute_collision_report,
    ClassIdentityCapacityError,
    _check_class_identity_capacity,
)

CPI_INDEX = "https://jefflevesque.com/ontology/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://jefflevesque.com/ontology/cpi/Series"      # -> cpi_Series
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_SUBPROPERTY_OF = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
METRIC = "https://example.org/metric"              # numeric literal predicate
SECTOR = "https://example.org/sector"              # categorical predicate
REGION = "https://example.org/region"              # categorical predicate
SUPER_PROP = "https://example.org/attribute"       # super-property of SECTOR
BASE_CLASS = "https://example.org/EconomicIndicator"   # direct superclass
ROOT_CLASS = "https://example.org/Thing"               # transitive superclass

VDIM = 128
CONFIG = {"feature_config": {"vector_dim": VDIM}}


def _node_features(spark, rows):
    """Run the real node-feature encoding; return {pyg_name: np.ndarray}."""
    triples = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    node_id_df, counts = NodeMapper(spark, CONFIG).build_node_id_table(triples)
    tensors, _names = FeatureExtractor(spark, CONFIG).build_features(
        triples, node_id_df, counts
    )
    return {t: v.numpy() for t, v in tensors.items()}


def _hash_slot(spark, parts, dim, start):
    """Reproduce the encoder's slot: abs(F.hash(*parts)) % dim + start.

    `parts` are hashed together in order exactly as the encoder passes them
    (e.g. [type_uri, seed] or ["pred::value", seed]). Uses the same Spark
    primitive, so it tracks the implementation rather than reimplementing it.
    """
    cols = [F.lit(p) for p in parts]
    expr = F.abs(F.hash(*cols)) % F.lit(dim) + F.lit(start)
    return int(spark.range(1).select(expr.alias("d")).first()["d"])


def _seg(row, start, dim):
    return row[start:start + dim]


# ======================================================================
# Segment 1 — class identity one-hot
# ======================================================================

def test_class_identity_lands_in_reserved_slots(spark):
    x = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    ci_start, ci_dim = L.seg1_class_identity_start, L.seg1_class_identity_dim

    # Expected: each of the 4 hash seeds sets one slot to 1.0 (collisions sum).
    expected = np.zeros(ci_dim, dtype=np.float32)
    for seed in _HASH_SEEDS:
        slot = _hash_slot(spark, [CPI_INDEX, seed], ci_dim, ci_start)
        assert ci_start <= slot < ci_start + ci_dim
        expected[slot - ci_start] += 1.0

    np.testing.assert_allclose(_seg(x[0], ci_start, ci_dim), expected, atol=1e-6)
    # Total mass equals the number of seeds (each contributes 1.0).
    assert _seg(x[0], ci_start, ci_dim).sum() == pytest.approx(len(_HASH_SEEDS))


def test_class_identity_distinguishes_types(spark):
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
    ])
    L = VectorLayout(VDIM)
    ci_start, ci_dim = L.seg1_class_identity_start, L.seg1_class_identity_dim
    idx = _seg(feats["cpi_Index"][0], ci_start, ci_dim)
    ser = _seg(feats["cpi_Series"][0], ci_start, ci_dim)
    # Different type URIs must produce different class-identity fingerprints.
    assert not np.array_equal(idx, ser)


# ======================================================================
# Segment 1b — class hierarchy (rdfs:subClassOf, depth-weighted)
# ======================================================================

def test_class_hierarchy_encodes_depth_weighted_superclasses(spark):
    # cpi:Index subClassOf BASE subClassOf ROOT. The transitive closure gives
    # node a superclasses (BASE, depth 1) and (ROOT, depth 2); each is hashed
    # with 2 seeds at weight 1/depth. cpi:Series has no hierarchy → zeros.
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
        (BASE_CLASS, RDFS_SUBCLASS_OF, ROOT_CLASS),
    ])
    L = VectorLayout(VDIM)
    ch_start, ch_dim = L.seg1_class_hierarchy_start, L.seg1_class_hierarchy_dim

    expected = np.zeros(ch_dim, dtype=np.float32)
    for seed in _HASH_SEEDS[:2]:
        for super_uri, depth in [(BASE_CLASS, 1), (ROOT_CLASS, 2)]:
            slot = _hash_slot(spark, [super_uri, seed + 100], ch_dim, ch_start)
            expected[slot - ch_start] += 1.0 / depth

    np.testing.assert_allclose(
        _seg(feats["cpi_Index"][0], ch_start, ch_dim), expected, atol=1e-6
    )
    # Negative: a type outside the hierarchy contributes nothing here.
    np.testing.assert_allclose(
        _seg(feats["cpi_Series"][0], ch_start, ch_dim), 0.0, atol=1e-6
    )


# ======================================================================
# Segment 2 — property schema (presence, domain/range, hierarchy)
# ======================================================================

def test_property_presence_sets_reserved_slots(spark):
    # Node a has the SECTOR property → 3-seed multi-hot of the predicate URI
    # in the property-presence sub-segment. Node b has no properties → zeros.
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    pp_start, pp_dim = (
        L.seg2_property_presence_start, L.seg2_property_presence_dim
    )

    expected = np.zeros(pp_dim, dtype=np.float32)
    for seed in _HASH_SEEDS[:3]:
        slot = _hash_slot(spark, [SECTOR, seed + 200], pp_dim, pp_start)
        assert pp_start <= slot < pp_start + pp_dim
        expected[slot - pp_start] += 1.0

    np.testing.assert_allclose(_seg(feats[0], pp_start, pp_dim), expected,
                               atol=1e-6)
    # Negative: node without the property has an empty presence sub-segment.
    np.testing.assert_allclose(_seg(feats[1], pp_start, pp_dim), 0.0,
                               atol=1e-6)


def test_domain_range_signals_land_in_their_halves(spark):
    # SECTOR declares rdfs:domain (cpi:Index) and rdfs:range (xsd:string).
    # A node carrying SECTOR gets the domain URI hashed (seed 300) into the
    # lower half of the domain-range sub-segment and the range URI (seed 301)
    # into the upper half.
    x = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
        (SECTOR, RDFS_DOMAIN, CPI_INDEX),
        (SECTOR, RDFS_RANGE, XSD_STRING),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    dr_start, dr_dim = L.seg2_domain_range_start, L.seg2_domain_range_dim
    dr_half = max(1, dr_dim // 2)

    expected = np.zeros(dr_dim, dtype=np.float32)
    domain_slot = _hash_slot(spark, [CPI_INDEX, 300], dr_half, dr_start)
    range_slot = _hash_slot(
        spark, [XSD_STRING, 301], dr_dim - dr_half, dr_start + dr_half
    )
    expected[domain_slot - dr_start] += 1.0
    expected[range_slot - dr_start] += 1.0

    np.testing.assert_allclose(_seg(x[0], dr_start, dr_dim), expected,
                               atol=1e-6)


def test_property_hierarchy_encodes_super_property(spark):
    # SECTOR subPropertyOf SUPER_PROP → the super-property URI is hashed with
    # 2 seeds (offset 400) into the property-hierarchy sub-segment for every
    # node carrying SECTOR. A schema-less predicate leaves it untouched.
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
        ("https://ex/b", REGION, "West"),          # no subPropertyOf declared
        (SECTOR, RDFS_SUBPROPERTY_OF, SUPER_PROP),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    ph_start, ph_dim = (
        L.seg2_property_hierarchy_start, L.seg2_property_hierarchy_dim
    )

    expected = np.zeros(ph_dim, dtype=np.float32)
    for seed in _HASH_SEEDS[:2]:
        slot = _hash_slot(spark, [SUPER_PROP, seed + 400], ph_dim, ph_start)
        expected[slot - ph_start] += 1.0

    np.testing.assert_allclose(_seg(feats[0], ph_start, ph_dim), expected,
                               atol=1e-6)
    # Negative: REGION has no declared super-property → zeros for node b.
    np.testing.assert_allclose(_seg(feats[1], ph_start, ph_dim), 0.0,
                               atol=1e-6)


# ======================================================================
# Segment 3a — numeric literal z-score normalization
# ======================================================================

def test_numeric_literal_is_zscore_normalized(spark):
    # Values 10/20/30 → mean 20, sample std 10 → normalized -1/0/+1.
    # URIs sort n0<n1<n2 → node IDs 0/1/2 receive 10/20/30 respectively.
    x = _node_features(spark, [
        ("https://ex/n0", RDF_TYPE, CPI_INDEX),
        ("https://ex/n1", RDF_TYPE, CPI_INDEX),
        ("https://ex/n2", RDF_TYPE, CPI_INDEX),
        ("https://ex/n0", METRIC, "10"),
        ("https://ex/n1", METRIC, "20"),
        ("https://ex/n2", METRIC, "30"),
    ])["cpi_Index"]

    L = VectorLayout(VDIM)
    num_start, num_dim = L.seg3_numeric_start, L.seg3_numeric_dim
    slot = _hash_slot(spark, [METRIC, 500], num_dim, num_start)

    expected = [-1.0, 0.0, 1.0]
    for node_id, want in enumerate(expected):
        assert x[node_id, slot] == pytest.approx(want, abs=1e-5)
        # Single predicate → the numeric sub-segment holds exactly this value.
        assert _seg(x[node_id], num_start, num_dim).sum() == pytest.approx(
            want, abs=1e-5
        )


def _numeric_predicates(spark, rows, config=CONFIG):
    """Run the real per-predicate numeric/categorical classification."""
    triples = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    node_id_df, _counts = NodeMapper(spark, config).build_node_id_table(triples)
    fx = FeatureExtractor(spark, config)
    return fx._classify_literal_predicates(
        fx._literal_triples(triples, node_id_df)
    )


# A property whose values are mostly labels is a label property, even where
# some values parse — SEC form "8" is a form type, not the quantity eight.
DOC_TYPE_ROWS = [
    ("https://ex/a", RDF_TYPE, CPI_INDEX),
    ("https://ex/b", RDF_TYPE, CPI_INDEX),
    ("https://ex/c", RDF_TYPE, CPI_INDEX),
    ("https://ex/a", SECTOR, "10-K"),
    ("https://ex/b", SECTOR, "8"),
    ("https://ex/c", SECTOR, "S-1"),
]


def test_mixed_literals_classify_categorical(spark):
    # 1 of 3 values parses (33%) — below the 0.5 default, so the whole
    # property is categorical and nothing about it is z-scored.
    assert _numeric_predicates(spark, DOC_TYPE_ROWS) == set()


def test_majority_numeric_literals_survive_a_sentinel(spark):
    # 2 of 3 parse (67%) — a stray "N/A" must not demote a real magnitude.
    assert _numeric_predicates(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/c", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", METRIC, "10"),
        ("https://ex/b", METRIC, "20"),
        ("https://ex/c", METRIC, "N/A"),
    ]) == {METRIC}


def test_numeric_share_threshold_is_configurable(spark):
    # The same mixed literals flip to numeric once the bar drops below 33%.
    config = {"feature_config": {"vector_dim": VDIM,
                                 "numeric_predicate_min_share": 0.1}}
    assert _numeric_predicates(spark, DOC_TYPE_ROWS, config) == {SECTOR}


def test_categorical_property_is_not_normalized(spark):
    # The reported defect: a label property reaching normalization.json.
    triples = spark.createDataFrame(
        DOC_TYPE_ROWS, schema="subject STRING, predicate STRING, object STRING"
    )
    node_id_df, counts = NodeMapper(spark, CONFIG).build_node_id_table(triples)
    fx = FeatureExtractor(spark, CONFIG)
    fx.build_features(triples, node_id_df, counts)

    artifacts = fx.get_metadata_artifacts()
    normalized = {s["predicate"] for s in
                  (artifacts["normalization_stats"] or [])}
    assert SECTOR not in normalized

    # ...and it is claimed by exactly one segment, not both.
    slots = artifacts["slot_mapping"]
    numeric = {p["predicate_uri"] for p in slots["numeric_properties"]}
    categorical = {p["predicate_uri"] for p in slots["categorical_properties"]}
    assert SECTOR in categorical
    assert SECTOR not in numeric
    assert numeric & categorical == set()


# ======================================================================
# Segment 3b — categorical multi-hot
# ======================================================================

def test_categorical_sets_expected_multihot_slots(spark):
    x = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])["cpi_Index"]

    L = VectorLayout(VDIM)
    cat_start, cat_dim = L.seg3_categorical_start, L.seg3_categorical_dim

    expected = np.zeros(cat_dim, dtype=np.float32)
    for seed in _HASH_SEEDS:
        slot = _hash_slot(
            spark, [f"{SECTOR}::Energy", seed + 600], cat_dim, cat_start
        )
        assert cat_start <= slot < cat_start + cat_dim
        expected[slot - cat_start] += 1.0

    np.testing.assert_allclose(
        _seg(x[0], cat_start, cat_dim), expected, atol=1e-6
    )


def test_distinct_categorical_values_differ(spark):
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
        ("https://ex/b", SECTOR, "Finance"),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    cat = (L.seg3_categorical_start, L.seg3_categorical_dim)
    # Nodes 0 and 1 carry different categorical values → different multi-hot.
    assert not np.array_equal(_seg(feats[0], *cat), _seg(feats[1], *cat))


# ======================================================================
# _scatter_all_types — per-type placement at the right node IDs
# ======================================================================

def test_scatter_places_each_type_at_its_own_node_ids(spark):
    feats = _node_features(spark, [
        ("https://ex/i0", RDF_TYPE, CPI_INDEX),
        ("https://ex/i1", RDF_TYPE, CPI_INDEX),
        ("https://ex/s0", RDF_TYPE, CPI_SERIES),
        # Distinct numeric values, monotonic in each type's node IDs.
        ("https://ex/i0", METRIC, "10"),
        ("https://ex/i1", METRIC, "20"),
        ("https://ex/s0", METRIC, "5"),
    ])
    L = VectorLayout(VDIM)
    slot = _hash_slot(
        spark, [METRIC, 500], L.seg3_numeric_dim, L.seg3_numeric_start
    )

    # Each type gets its own tensor with the right row count and width.
    assert feats["cpi_Index"].shape == (2, VDIM)
    assert feats["cpi_Series"].shape == (1, VDIM)
    # Within cpi_Index, the smaller input value (node 0: 10) normalizes below the
    # larger (node 1: 20) — monotonic regardless of the global mean/std, so rows
    # are placed at their own node IDs (not swapped across the type boundary).
    assert feats["cpi_Index"][0, slot] < feats["cpi_Index"][1, slot]
    assert np.isfinite(feats["cpi_Series"][0, slot])


# ======================================================================
# Determinism — regression guard for the jolts_* categorical bug
# ======================================================================

def test_categorical_encoding_is_deterministic_across_runs(spark):
    # Multiple categorical values per node exercise the multi-hot + group-by-sum
    # combine where the jolts_* non-determinism lives. Repeated encodings of the
    # same fixture must be bit-identical.
    rows = [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/c", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
        ("https://ex/a", REGION, "West"),
        ("https://ex/b", SECTOR, "Energy"),
        ("https://ex/b", REGION, "East"),
        ("https://ex/c", SECTOR, "Finance"),
        ("https://ex/c", REGION, "West"),
    ]
    first = _node_features(spark, rows)["cpi_Index"]
    again = _node_features(spark, rows)["cpi_Index"]
    assert np.array_equal(first, again), (
        "categorical encoding is order-dependent (jolts_* regression)"
    )


# ======================================================================
# ontology_schema.json — why an empty hierarchy is empty
# ======================================================================

def _metadata_artifacts(spark, rows):
    """Run the real build and return every collected metadata artifact."""
    triples = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    node_id_df, counts = NodeMapper(spark, CONFIG).build_node_id_table(triples)
    fx = FeatureExtractor(spark, CONFIG)
    fx.build_features(triples, node_id_df, counts)
    return fx.get_metadata_artifacts()


def _ontology_schema(spark, rows):
    """Build the real ontology_schema artifact from a triple set."""
    return _metadata_artifacts(spark, rows)["ontology_schema"]


# A triple the OntologyMapper always emits and nothing else does — the marker
# that says the mapping phase ran. Prepending it to a fixture makes that
# fixture stand for "mapping ran"; omitting it, for "mapping never ran".
MAPPED = (
    "https://jefflevesque.com/ontology/cpi/hasMonth",
    "http://www.w3.org/2002/07/owl#equivalentProperty",
    "https://example.org/unified/hasMonth",
)


def test_empty_hierarchy_records_its_reason(spark):
    # An empty superclass_chain is ambiguous on its own: "no source declares
    # a hierarchy" and "this class is a root" both serialize to []. A zero
    # class_hierarchy sub-segment must be diagnosable from the artifact.
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    assert schema["hierarchy_source"] == (
        "no rdfs:subClassOf after ontology mapping"
    )
    assert schema["node_types"]["cpi_Index"]["superclass_chain"] == []


def test_populated_hierarchy_names_its_source(spark):
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
    ])
    assert schema["hierarchy_source"] == "rdfs:subClassOf"
    assert schema["node_types"]["cpi_Index"]["superclass_chain"]


# ======================================================================
# ontology_schema.json — "no hierarchy" vs "hierarchy never computed"
# ======================================================================
#
# The 2026-07-29 run wrote a structurally valid ontology_schema.json in which
# every one of 43 node types had superclass_chain == [] — correct for that
# build, because ontology mapping never ran, but indistinguishable in the file
# from sources that genuinely declare no subsumption. The two demand opposite
# responses: the first is a re-run, the second is data to work with. Nothing
# in the file said which, and the job manifest could not settle it either —
# that run was pyg_only, so its enable_ontology_mapping flag described a phase
# the job never reached, not the Parquet it read.
#
# The flag is therefore derived from the triples themselves, which is the only
# signal that stays truthful across full and pyg_only builds alike.

def test_unmapped_build_says_mapping_never_ran(spark):
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    # The flag itself — not merely an empty list left to be interpreted.
    assert schema["ontology_mapping_enabled"] is False
    # The flag arrived in 1.1, so a consumer can require it by version rather
    # than probe for the key: a 1.0 file's empty hierarchy stays ambiguous
    # forever. Compared as a bound, not pinned — later versions still carry it,
    # and test_schema_version_tracks_the_provenance_fields owns the exact
    # number.
    assert tuple(map(int, schema["version"].split("."))) >= (1, 1)
    assert "no owl:equivalentProperty" in schema["ontology_mapping_evidence"]
    # ...and the prose reason names the missing phase, not just missing data.
    assert schema["hierarchy_source"] == (
        "no rdfs:subClassOf in source data and ontology mapping did not run"
    )
    assert schema["node_types"]["cpi_Index"]["superclass_chain"] == []


def test_mapped_build_says_mapping_ran(spark):
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    assert schema["ontology_mapping_enabled"] is True
    assert schema["ontology_mapping_evidence"] == (
        "owl:equivalentProperty/owl:equivalentClass present in build input"
    )


def test_equivalent_class_alone_is_marker_enough(spark):
    # Either marker suffices: a run whose class map is non-empty but whose
    # property table is not must still read as mapped.
    schema = _ontology_schema(spark, [
        (CPI_INDEX, "http://www.w3.org/2002/07/owl#equivalentClass",
         BASE_CLASS),
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
    ])
    assert schema["ontology_mapping_enabled"] is True


def test_hierarchy_from_mapping_is_not_reported_as_empty(spark):
    # When mapping does derive subsumption, nothing about the flag suppresses
    # the ordinary source attribution.
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
    ])
    assert schema["ontology_mapping_enabled"] is True
    assert schema["hierarchy_source"] == "rdfs:subClassOf"


def test_dead_class_hierarchy_segment_blames_the_missing_phase(spark):
    # feature_spec.json's empty_reason comes from the same distinction, so a
    # reader of that file is not sent looking for absent source triples when
    # the real cause is a phase that never ran.
    unmapped = _metadata_artifacts(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])["sub_segment_status"]
    assert unmapped["class_hierarchy"] == (
        "no rdfs:subClassOf in source data and ontology mapping did not run"
    )

    mapped = _metadata_artifacts(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])["sub_segment_status"]
    assert mapped["class_hierarchy"] == (
        "no rdfs:subClassOf after ontology mapping"
    )


def test_empty_property_schema_records_its_reason(spark):
    # Same diagnosis gap for domain_range: no source declares rdfs:domain or
    # rdfs:range either, so the sub-segment is zero unless something derives
    # them. And since OntologyMapper is now what does, the reason has to name
    # the phase that did not run rather than only the data that is missing.
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    assert schema["property_schema_source"] == (
        "no rdfs:domain or rdfs:range in source data and ontology mapping "
        "did not run"
    )


def test_property_schema_after_mapping_blames_the_data_not_the_phase(spark):
    schema = _mapped_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
    ])
    assert schema["property_schema_source"] == (
        "no rdfs:domain or rdfs:range after ontology mapping"
    )


# ======================================================================
# class_identity capacity — the collision_report metric and its guard
# ======================================================================
#
# The 2026-07-29 cluster run reported collision_rate 0.6705 on class_identity
# (176 hash entries, 58 unique slots, 64 dims) and that number was read as
# "two thirds of class identity is aliased". It is not: class identity is a
# multi-hot code, and on that very build all 44 classes had distinct codes, the
# 44x64 code matrix was full rank, and its condition number was ~12. With
# entries > dims, pigeonhole forces a high slot-reuse rate on a perfectly
# healthy code, so the metric could not distinguish healthy from broken.
#
# These tests pin the metric that can: identical codes, and the class count
# against the segment width.

# A REQUIREMENT, not an implementation fact: the number of ontology classes the
# encoding must be able to keep separable. Measured from a full
# sec+noaa+market+BLS build (the turtle_parquet e2e fixtures). Raise it when the
# source set grows; the layout then has to be re-tuned to meet it, which is the
# point -- the test states the need and lets the composition of the vector
# change underneath it.
SUPPORTED_CLASS_COUNT = 118


def _distinct_codes(n, dim, k=4):
    """The first n distinct k-hot codes over `dim` slots, in lexical order.

    Enumerated with itertools.combinations rather than constructed by hand:
    distinctness is then guaranteed by the enumeration itself, so a failure in
    a test using this fixture always means "capacity", never "the fixture
    happened to collide". (An earlier hand-rolled spread did collide, at
    exactly n = dim/2.)
    """
    k = max(1, min(k, dim))
    codes = list(itertools.islice(itertools.combinations(range(dim), k), n))
    assert len(codes) == n, (
        f"cannot draw {n} distinct {k}-hot codes from {dim} slots"
    )
    return codes


def _class_slots(codes, start=0):
    """Build class_slots entries with the given per-class slot lists."""
    return [
        {
            "class_uri": f"https://ex/C{i}",
            "pyg_name": f"C{i}",
            "hash_slots": list(slots),
            "global_dims": [s + start for s in slots],
        }
        for i, slots in enumerate(codes)
    ]


def _class_report(codes, dim, start=0):
    report = _compute_collision_report(
        [], [], _class_slots(codes, start), [], [], class_identity_dim=dim,
    )
    return report["class_identity"]


def test_class_identity_reports_separability_not_slot_reuse():
    """Heavy slot reuse with distinct codes is healthy, and must read as such.

    Four classes, 4 hashes each, into 6 dims: 16 entries over 6 slots is 62%
    slot reuse -- the shape that produced the 0.67 figure -- yet every code is
    distinct and 4 <= 6, so identity is fully recoverable.
    """
    ci = _class_report(
        [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (0, 2, 4, 5)], dim=6,
    )
    assert ci["distinct_codes"] == 4
    assert ci["classes_sharing_a_code"] == []
    assert ci["linearly_separable"] is True
    assert ci["headroom_classes"] == 2
    # Slot reuse is still reported -- it is a fact -- but under a name that
    # cannot be mistaken for lost identity.
    assert ci["slot_reuse_rate"] > 0.5
    assert "collision_rate" not in ci, (
        "the misleading 1.0 key must be gone, not merely supplemented"
    )


def test_class_identity_flags_classes_sharing_a_code():
    """Two classes hashing to the same slot set ARE indistinguishable."""
    ci = _class_report([(0, 1), (0, 1), (2, 3)], dim=8)
    assert ci["distinct_codes"] == 2
    assert ci["classes_sharing_a_code"] == [["C0", "C1"]]
    assert ci["linearly_separable"] is False, (
        "an identical code pair is not separable at any segment width"
    )


def test_class_identity_flags_more_classes_than_dimensions():
    """Past d classes in d dims no code set can be linearly independent.

    Distinctness does not catch this -- these 10 codes are all distinct -- so
    the capacity check is the only thing standing between a saturated segment
    and a silently unrecoverable feature.
    """
    codes = [(i % 4, (i + 1) % 4) for i in range(10)]
    codes = [(a, b, 4 + i) for i, (a, b) in enumerate(codes)]  # keep distinct
    ci = _class_report(codes, dim=4)
    assert ci["distinct_codes"] == len(codes)
    assert ci["total_classes"] == 10
    assert ci["capacity_classes"] == 4
    assert ci["headroom_classes"] == -6
    assert ci["linearly_separable"] is False


def test_class_identity_max_pairwise_overlap_is_reported():
    ci = _class_report([(0, 1, 2), (1, 2, 3), (7, 8, 9)], dim=16)
    assert ci["max_pairwise_slot_overlap"] == 2


@pytest.mark.parametrize("num_classes,dim,expect", [
    (44, 64, None),                       # the real build: quiet
    (56, 64, "near capacity"),            # 87.5% full
])
def test_saturation_warning_fires_only_when_identity_is_at_risk(
    caplog, num_classes, dim, expect,
):
    import logging

    report = {"class_identity": {
        "total_classes": num_classes,
        "segment_dim": dim,
        "classes_sharing_a_code": [],
    }}
    logger_name = "spark_jobs.pyg_builder.feature_extractor"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        _check_class_identity_capacity(report)

    if expect is None:
        assert "class_identity" not in caplog.text, (
            f"{num_classes} classes in {dim} dims is healthy and must not warn"
        )
    else:
        assert expect in caplog.text


def test_over_subscription_fails_the_build():
    """Past the ceiling is an error, not a log line.

    A warning was the wrong severity. The artifact ships, every consistency
    check passes, and the only symptom is a model that never learns to tell two
    classes apart -- weeks downstream, with nothing pointing back at the build.
    """
    report = {"class_identity": {
        "total_classes": 70,
        "segment_dim": 64,
        "classes_sharing_a_code": [],
    }}
    with pytest.raises(ClassIdentityCapacityError) as exc:
        _check_class_identity_capacity(report)

    assert "over-subscribed" in str(exc.value)
    # The message must name the levers, or it reports a wall with no door.
    assert "class_identity_dim" in str(exc.value)
    assert "vector_dim" in str(exc.value)


def test_over_subscription_can_be_opted_into(caplog):
    """An exploratory run may accept partial identity -- but must ask."""
    import logging

    report = {"class_identity": {
        "total_classes": 70,
        "segment_dim": 64,
        "classes_sharing_a_code": [],
    }}
    with caplog.at_level(
        logging.WARNING, logger="spark_jobs.pyg_builder.feature_extractor"
    ):
        _check_class_identity_capacity(report, allow_oversubscription=True)

    assert "over-subscribed" in caplog.text


def test_shared_codes_fail_the_build_and_name_the_groups():
    """Identical codes are unrecoverable at any width, so they fail too."""
    report = {"class_identity": {
        "total_classes": 4,
        "segment_dim": 64,
        "classes_sharing_a_code": [["cpi_Area", "cpi_Region"]],
    }}
    with pytest.raises(ClassIdentityCapacityError) as exc:
        _check_class_identity_capacity(report)

    assert "indistinguishable" in str(exc.value)
    assert "cpi_Area" in str(exc.value)


# ======================================================================
# class_identity width override (feature_config.class_identity_dim)
# ======================================================================

def test_class_identity_dim_override_takes_dims_from_segment_one():
    """The production lever: more class capacity at the same vector width.

    Taken from within segment 1 so vector_dim -- and therefore the driver
    memory held for every node in the graph -- does not move.
    """
    default = VectorLayout(1024)
    wider = VectorLayout(1024, class_identity_dim=200)

    assert wider.seg1_class_identity_dim == 200
    assert wider.vector_dim == default.vector_dim
    assert wider.seg1_total == default.seg1_total
    # Every later segment starts where it did.
    assert wider.seg2_start == default.seg2_start
    assert wider.seg3_start == default.seg3_start


@pytest.mark.parametrize("bad", [0, -1, 255, 1000])
def test_class_identity_dim_override_rejects_widths_that_do_not_fit(bad):
    """Rejected, not clamped.

    The override exists to guarantee a class budget, so silently granting a
    smaller one would defeat the purpose -- the build would run and the
    capacity check would then fail on a width nobody asked for.
    """
    with pytest.raises(ValueError, match="class_identity_dim"):
        VectorLayout(1024, class_identity_dim=bad)


def test_class_identity_capacity_holds_for_the_real_class_count(spark):
    """The end-to-end budget check, on the real default layout.

    A 1024-d vector gives class_identity 64 dims. The 2026-07-29 build had 44
    classes -- inside capacity, but at 69% of it. This is the test that fails
    when a future source pushes the class count past the segment, which is the
    point at which the feature silently stops working.
    """
    layout = VectorLayout(1024)
    ci_dim = layout.seg1_class_identity_dim
    assert ci_dim >= SUPPORTED_CLASS_COUNT, (
        f"class_identity has {ci_dim} dims at the production default "
        f"vector_dim, which cannot separate the {SUPPORTED_CLASS_COUNT} "
        f"classes a full-source build produces. Raise "
        f"_SEG1_CLASS_IDENTITY_FRAC or vector_dim -- whichever segment the "
        f"dims come from, this is the requirement the layout has to meet."
    )


@pytest.mark.parametrize("vector_dim", [256, 512, 1024, 2048])
def test_capacity_is_a_ceiling_not_a_guarantee(vector_dim):
    """The capacity rule, asserted against whatever the layout happens to be.

    Deliberately derives every number from VectorLayout rather than pinning a
    width: the segment fractions are a tuning decision and the composition of
    the feature vector is expected to change. What must not change is the rule,
    so that is what is tested, at four widths, with no hardcoded dims anywhere.

    The rule is asymmetric, and the asymmetry is the point:

      * MORE than d classes can never be separable -- pigeonhole, no set of
        codes can help.
      * d or fewer MIGHT be separable. It is not guaranteed and has to be
        measured, because distinct codes under the ceiling can still be
        linearly dependent.

    This previously asserted separability at exactly d classes, which passed
    only because the metric was `num_classes <= dim and not shared` -- a count
    comparison wearing the name of a rank test. Measured, d 4-hot codes in d
    dims come out rank d-2 at every width here: the nominal ceiling is an upper
    bound nobody should plan to reach, which is why headroom_classes is
    reported alongside it rather than instead of it.
    """
    ci_dim = VectorLayout(vector_dim).seg1_class_identity_dim

    # Comfortably inside: full rank, genuinely separable.
    inside = max(1, ci_dim // 2)
    ci = _class_report(_distinct_codes(inside, ci_dim), dim=ci_dim)
    assert ci["distinct_codes"] == inside, (
        "test fixture must supply distinct codes so the assertions below "
        "isolate capacity from code collisions"
    )
    assert ci["code_matrix_rank"] == inside
    assert ci["linearly_separable"] is True
    assert ci["headroom_classes"] == ci_dim - inside

    # One past the ceiling: never separable, whatever the codes are.
    over = ci_dim + 1
    ci = _class_report(_distinct_codes(over, ci_dim), dim=ci_dim)
    assert ci["linearly_separable"] is False, (
        f"vector_dim={vector_dim} -> class_identity={ci_dim} dims: "
        f"{over} classes cannot be separable"
    )
    assert ci["rank_deficiency"] >= 1

    # Exactly at the ceiling: whatever rank says, and separability must agree
    # with it rather than with the count.
    ci = _class_report(_distinct_codes(ci_dim, ci_dim), dim=ci_dim)
    assert ci["distinct_codes"] == ci_dim
    assert ci["linearly_separable"] is (ci["code_matrix_rank"] == ci_dim)
    assert ci["headroom_classes"] == 0


def test_distinct_codes_under_capacity_can_still_be_dependent():
    """The case both cheap proxies miss -- and the reason rank is measured.

    Four 4-hot codes over 8 slots, pairwise distinct and well inside the
    ceiling, yet A + D - B - C = 0: one class is exactly a combination of the
    others and no linear readout can recover it.

    Under the previous metric (`num_classes <= dim and not shared`) this
    reported linearly_separable=True and the build shipped.
    """
    codes = [(0, 1, 2, 3), (0, 1, 4, 5), (2, 3, 6, 7), (4, 5, 6, 7)]
    ci = _class_report(codes, dim=8)

    assert ci["distinct_codes"] == 4, "the codes are pairwise distinct"
    assert ci["classes_sharing_a_code"] == [], "none are identical"
    assert ci["total_classes"] <= ci["capacity_classes"], "inside the ceiling"

    assert ci["code_matrix_rank"] == 3
    assert ci["rank_deficiency"] == 1
    assert ci["linearly_separable"] is False


def test_dependent_codes_fail_the_build_with_their_own_diagnosis():
    """The remedy differs from over-subscription, so the message must too.

    Over-subscription means "too many classes for this width". Dependence
    inside the ceiling is an unlucky hash draw -- changing the width re-draws
    every code -- so reporting it as over-subscription would send someone to
    the wrong fix.
    """
    codes = [(0, 1, 2, 3), (0, 1, 4, 5), (2, 3, 6, 7), (4, 5, 6, 7)]
    report = {"class_identity": _class_report(codes, dim=8)}

    with pytest.raises(ClassIdentityCapacityError) as exc:
        _check_class_identity_capacity(report)

    message = str(exc.value)
    assert "linearly dependent" in message
    assert "over-subscribed" not in message
    assert "span only 3 dimensions" in message


# ======================================================================
# ontology_schema.json — provenance and property-schema coverage
# ======================================================================
#
# domain_range (111 dims) and property_hierarchy (81 dims) were structurally
# zero in every build: 192 of 1024 dims, 18.75% of every node vector, because
# no source declares rdfs:domain, rdfs:range or rdfs:subPropertyOf anywhere.
# They are now DERIVED -- domain and range from how the graph uses each
# predicate, the property hierarchy from the curated table.
#
# Which makes provenance load-bearing rather than decorative. An inferred
# rdfs:domain is shaped exactly like a declared one, and they are not
# interchangeable: "used on class C in this month's data" is weaker than "its
# domain is C", and a consumer that reasons over the second when it only has
# the first will be wrong in ways nothing here would catch.

PROP_A = "https://example.org/measure"
XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"


def _mapped_schema(spark, rows):
    """ontology_schema for a triple set that carries the mapper's markers."""
    return _ontology_schema(spark, [MAPPED] + rows)


def test_derived_axioms_are_not_reported_as_declared(spark):
    from spark_jobs.utils.rdf_utils import (
        PROV_DERIVED_BY, PROV_PROPERTY_DOMAIN, PROV_PROPERTY_RANGE,
    )

    schema = _mapped_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", PROP_A, "1.5"),
        (PROP_A, RDFS_DOMAIN, CPI_INDEX),
        (PROP_A, RDFS_RANGE, XSD_DECIMAL),
        (PROV_PROPERTY_DOMAIN, PROV_DERIVED_BY,
         "observed rdf:type of each predicate's subjects"),
        (PROV_PROPERTY_RANGE, PROV_DERIVED_BY,
         "observed object types and declared XSD datatypes"),
    ])

    assert schema["provenance"]["property_domain"] == (
        "observed rdf:type of each predicate's subjects"
    )
    assert "declared" in schema["provenance"]["property_range"]
    # The axioms themselves still read as present — provenance qualifies the
    # source, it does not retract it.
    assert schema["property_schema_source"] == "rdfs:domain/rdfs:range"


def test_undeclared_axioms_are_not_attributed_to_the_pipeline(spark):
    """No marker means the pipeline did not derive them.

    Reporting an inference here would be the same error in reverse: claiming
    authorship of an axiom that came out of the source.
    """
    schema = _mapped_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", PROP_A, "1.5"),
        (PROP_A, RDFS_DOMAIN, CPI_INDEX),
    ])
    assert schema["provenance"]["property_domain"] == "declared by the source"


def test_provenance_says_nothing_was_derived_when_mapping_never_ran(spark):
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    assert schema["provenance"]["property_domain"] == (
        "not derived — ontology mapping did not run"
    )
    assert schema["provenance"]["property_hierarchy"] == (
        "not derived — ontology mapping did not run"
    )


def test_property_hierarchy_source_names_its_predicate(spark):
    schema = _mapped_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", PROP_A, "1.5"),
        (PROP_A, RDFS_SUBPROPERTY_OF, "https://example.org/unified/value"),
    ])
    assert schema["property_hierarchy_source"] == "rdfs:subPropertyOf"


def test_empty_property_hierarchy_records_its_reason(spark):
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    assert schema["property_hierarchy_source"] == (
        "no rdfs:subPropertyOf in source data and ontology mapping did not run"
    )


def test_coverage_names_the_predicates_that_got_no_axiom(spark):
    """A gap nobody enumerated is indistinguishable from a bug.

    Derivation declines an ambiguous predicate on purpose -- rdfs:domain is an
    intersection, so a property used on two classes gets none -- and that is
    only defensible if the skipped set is published rather than inferred from
    a sub-segment being thinner than expected.
    """
    schema = _mapped_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", PROP_A, "1.5"),
        ("https://ex/a", SECTOR, "Energy"),
        # PROP_A gets both; SECTOR gets neither.
        (PROP_A, RDFS_DOMAIN, CPI_INDEX),
        (PROP_A, RDFS_RANGE, XSD_DECIMAL),
    ])
    cov = schema["property_schema_coverage"]

    assert cov["with_domain"] == 1
    assert cov["with_range"] == 1
    assert SECTOR in cov["without_domain"]
    assert SECTOR in cov["without_range"]
    assert PROP_A not in cov["without_domain"]
    # Counts are a fraction of the data predicates, and the marker/axiom
    # predicates are not counted as vocabulary that could have had a domain.
    assert cov["data_predicates"] >= 2
    assert 0.0 < cov["domain_coverage"] <= 1.0
    assert cov["without_domain_truncated"] is False


def test_coverage_excludes_the_axiom_and_provenance_predicates(spark):
    """rdfs:domain describes the vocabulary; it is not part of it.

    Counted as a data predicate it would show up as a permanent coverage gap
    that no derivation could ever close.
    """
    from spark_jobs.utils.rdf_utils import PROV_OBSERVED_LITERAL_DATATYPE

    schema = _mapped_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", PROP_A, "1.5"),
        (PROP_A, RDFS_DOMAIN, CPI_INDEX),
        (PROP_A, PROV_OBSERVED_LITERAL_DATATYPE, XSD_DECIMAL),
    ])
    cov = schema["property_schema_coverage"]

    for meta in (RDF_TYPE, RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBPROPERTY_OF,
                 PROV_OBSERVED_LITERAL_DATATYPE):
        assert meta not in cov["without_domain"], meta
        assert meta not in cov["without_range"], meta


def test_schema_version_tracks_the_provenance_fields(spark):
    """One pin for the current shape; the field history is in the builder.

    1.1 added the ontology_mapping_enabled flag, 1.2 the provenance block and
    coverage, 1.3 the per-route *_source strings and derived_axioms.
    """
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
    ])
    assert schema["version"] == "1.3"


# ======================================================================
# hierarchy_source — a derived hierarchy must never read as a declared one
# ======================================================================
#
# Since #273 the mapper emits rdfs:subClassOf that no source declared: 24
# curated from CLASS_MAPPINGS plus naming-inferred edges. The collector read
# them back and reported "rdfs:subClassOf" — true of the predicate the encoder
# consumed, false as an answer to the question anyone reading the field is
# asking. Two concrete costs:
#
#   * #273's own AC1 ("can any source emit rdfs:subClassOf?") became
#     unrepeatable — re-running it against ENRICHED triples finds 34 and
#     concludes yes, when the true answer is still no;
#   * it flattened two very different levels of trust. Curated edges were
#     chosen by a person. Naming-inferred edges come from spelling, and that
#     rule produced AllItemsLessShelter -> Shelter and three more inversions
#     before the negating-qualifier guard.

from spark_jobs.utils.rdf_utils import (          # noqa: E402
    PROV_ROUTE_CURATED_SUBCLASS,
    PROV_ROUTE_NAMED_SUBCLASS,
)

SUB_A = "https://example.org/SubA"
SUB_B = "https://example.org/SubB"


def _hierarchy_schema(spark, extra):
    """A build whose class hierarchy is whatever `extra` declares."""
    return _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
    ] + extra)


def test_a_declared_hierarchy_still_reports_the_predicate(spark):
    """No route marker means the source really did declare it."""
    schema = _hierarchy_schema(spark, [])
    assert schema["hierarchy_source"] == "rdfs:subClassOf"
    assert schema["derived_axioms"]["class_hierarchy"]["counts"][
        "declared"] == 1


def test_a_curated_hierarchy_says_curated(spark):
    schema = _hierarchy_schema(spark, [
        (CPI_INDEX, PROV_ROUTE_CURATED_SUBCLASS, BASE_CLASS),
    ])
    assert schema["hierarchy_source"] == (
        "derived: curated class mappings (1)"
    )
    # ...and the count that answers #273's AC1 directly.
    assert schema["derived_axioms"]["class_hierarchy"]["counts"][
        "declared"] == 0


def test_a_name_inferred_hierarchy_says_class_naming(spark):
    schema = _hierarchy_schema(spark, [
        (CPI_INDEX, PROV_ROUTE_NAMED_SUBCLASS, BASE_CLASS),
    ])
    assert schema["hierarchy_source"] == "derived: class naming (1)"


def test_two_derivation_routes_are_reported_with_their_counts(spark):
    """The mixed-derivation case, which is what the real builds produce."""
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
        (CPI_SERIES, RDFS_SUBCLASS_OF, SUB_A),
        (CPI_INDEX, PROV_ROUTE_CURATED_SUBCLASS, BASE_CLASS),
        (CPI_SERIES, PROV_ROUTE_NAMED_SUBCLASS, SUB_A),
    ])
    assert schema["hierarchy_source"] == (
        "derived: class naming (1), curated class mappings (1)"
    )


def test_derived_and_declared_together_read_as_mixed(spark):
    """A source that DOES declare a hierarchy must not be hidden by ours.

    "derived" would understate the artifact; "rdfs:subClassOf" would overstate
    it. Both counts are reported.
    """
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
        (CPI_SERIES, RDFS_SUBCLASS_OF, SUB_A),
        # Only the first is ours.
        (CPI_INDEX, PROV_ROUTE_CURATED_SUBCLASS, BASE_CLASS),
    ])
    source = schema["hierarchy_source"]
    assert source.startswith("mixed: "), source
    assert "curated class mappings (1)" in source
    assert "declared (1)" in source


def test_each_edge_is_attributable_without_reading_the_code(spark):
    """Counts cannot answer "is THIS edge a person's call or a guess?"."""
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
        (CPI_SERIES, RDFS_SUBCLASS_OF, SUB_A),
        (CPI_INDEX, PROV_ROUTE_CURATED_SUBCLASS, BASE_CLASS),
        (CPI_SERIES, PROV_ROUTE_NAMED_SUBCLASS, SUB_A),
    ])
    axioms = schema["derived_axioms"]["class_hierarchy"]["axioms"]

    assert [CPI_INDEX, BASE_CLASS] in axioms["curated class mappings"]
    assert [CPI_SERIES, SUB_A] in axioms["class naming"]
    # ...and the same distinction on the chain a consumer actually reads.
    chain = schema["node_types"]["cpi_Index"]["superclass_chain"]
    assert chain[0]["provenance"] == "curated class mappings"
    assert schema["node_types"]["cpi_Series"]["superclass_chain"][0][
        "provenance"] == "class naming"


def test_a_transitive_link_claims_no_route_of_its_own(spark):
    """Depth > 1 is our closure, not an asserted edge.

    Attributing it to the route that produced the direct edge would double-
    count that rule's reach; calling it declared would be worse.
    """
    schema = _ontology_schema(spark, [
        MAPPED,
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        (CPI_INDEX, RDFS_SUBCLASS_OF, BASE_CLASS),
        (BASE_CLASS, RDFS_SUBCLASS_OF, ROOT_CLASS),
        (CPI_INDEX, PROV_ROUTE_CURATED_SUBCLASS, BASE_CLASS),
    ])
    chain = {
        e["depth"]: e["provenance"]
        for e in schema["node_types"]["cpi_Index"]["superclass_chain"]
    }
    assert chain[1] == "curated class mappings"
    assert chain[2] == "transitive closure of the direct edges"


def test_an_empty_hierarchy_keeps_its_existing_reason(spark):
    """#273's AC4/AC5 wording is about absent DATA, which provenance cannot
    change — an empty set has no routes to report."""
    schema = _ontology_schema(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])
    assert schema["hierarchy_source"].startswith("no rdfs:subClassOf")
    assert schema["derived_axioms"]["class_hierarchy"]["total"] == 0


def test_route_counts_have_a_stable_key_order():
    """Key ORDER, not just values — metadata is compared as JSON text.

    _route_counts walks a set of axioms. Iterating a set seeds dict insertion
    order from string hashing, which differs per process under a randomized
    PYTHONHASHSEED, so two runs of the same build produced the same counts in
    a different order and ontology_schema.json stopped being byte-identical.
    Only test_jolts_features_reproducible caught it, because it is the one
    guard that compares serialized metadata rather than parsed objects — and
    it takes eight minutes. This is the same assertion in a tenth of a second.
    """
    from spark_jobs.pyg_builder.feature_extractor import (
        _route_counts, _route_detail,
    )
    from spark_jobs.utils.rdf_utils import (
        PROV_ROUTE_CURATED_SUBCLASS, PROV_ROUTE_NAMED_SUBCLASS,
    )

    axioms = {(f"https://ex/c{i}", f"https://ex/p{i}") for i in range(40)}
    routes = {
        a: (PROV_ROUTE_CURATED_SUBCLASS if i % 2 else PROV_ROUTE_NAMED_SUBCLASS)
        for i, a in enumerate(sorted(axioms))
    }

    counts = _route_counts(axioms, routes)
    assert list(counts) == sorted(counts), "route counts are not key-sorted"

    # Re-deriving from a set built in a different insertion order must give a
    # byte-identical dict, not merely an equal one.
    shuffled = {a for a in sorted(axioms, reverse=True)}
    assert json.dumps(_route_counts(shuffled, routes)) == json.dumps(counts)

    detail = _route_detail(axioms, routes, counts)
    assert list(detail["counts"]) == sorted(detail["counts"])
    assert list(detail["axioms"]) == sorted(detail["axioms"])
    assert detail["counts"]["declared"] == 0
