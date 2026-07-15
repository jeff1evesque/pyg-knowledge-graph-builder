"""Unit tests for NodeMapper (spark_jobs/pyg_builder/node_mapper.py).

NodeMapper turns rdf:type triples into the contiguous, per-type integer node IDs
that the whole HeteroData graph is indexed on. The e2e smoke only checks that
*some* nodes exist; these tests pin the value-level contract on tiny in-memory
fixtures over the shared local SparkSession (`spark` / `make_triples`):

  * type URIs map to the documented PyG names,
  * IDs are 0-indexed, contiguous, and deterministically ordered within a type,
  * meta-ontology types are dropped,
  * a multi-type entity is assigned its single canonical (most specific) type,
  * config filters (whitelist / temporal / sector) select the right types,
  * the mapping is reproducible run-to-run.

Type URIs / prefixes are taken from the code (NAMESPACE_PREFIXES, RDF_TYPE) so the
tests track the implementation rather than hardcoding IRIs.
"""
from spark_jobs.pyg_builder.node_mapper import (
    NodeMapper,
    RDF_TYPE,
    _build_uri_to_pyg_name_expr,
)

# Concrete type URIs whose PyG names are fixed by NAMESPACE_PREFIXES.
CPI_INDEX = "https://www.bls.gov/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://www.bls.gov/cpi/Series"      # -> cpi_Series
UNIFIED_MONTH = "https://example.org/unified/UnifiedMonth"  # -> unified_UnifiedMonth
UNIFIED_SECTOR = "https://example.org/unified/EconomicSector"  # -> unified_EconomicSector
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"  # excluded (meta-ontology)


def _mapping(node_id_df):
    """Collect node_id_df into {uri: (node_id, node_type)}."""
    return {
        r["uri"]: (r["node_id"], r["node_type"])
        for r in node_id_df.collect()
    }


# ======================================================================
# PyG type naming (pure Column expression)
# ======================================================================

def test_uri_to_pyg_name_maps_known_namespace(spark):
    df = spark.createDataFrame([(CPI_INDEX,)], ["type_uri"])
    name = df.withColumn("n", _build_uri_to_pyg_name_expr("type_uri")).collect()[0]["n"]
    assert name == "cpi_Index"


def test_uri_to_pyg_name_falls_back_to_last_segment(spark):
    # Unknown namespace -> fallback extracts the trailing segment as unknown_<seg>.
    df = spark.createDataFrame([("http://nonexistent.invalid/Widget",)], ["type_uri"])
    name = df.withColumn("n", _build_uri_to_pyg_name_expr("type_uri")).collect()[0]["n"]
    assert name == "unknown_Widget"


# ======================================================================
# build_node_id_table — IDs and counts
# ======================================================================

def test_ids_are_contiguous_zero_indexed_and_uri_ordered(spark, make_triples):
    # Three cpi_Index entities; IDs are assigned per type ordered by uri ASC.
    triples = make_triples([
        ("https://ex/n3", RDF_TYPE, CPI_INDEX),
        ("https://ex/n1", RDF_TYPE, CPI_INDEX),
        ("https://ex/n2", RDF_TYPE, CPI_INDEX),
    ])
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    mapping = _mapping(node_id_df)

    assert counts == {"cpi_Index": 3}
    # Ordered by uri: n1->0, n2->1, n3->2.
    assert mapping["https://ex/n1"] == (0, "cpi_Index")
    assert mapping["https://ex/n2"] == (1, "cpi_Index")
    assert mapping["https://ex/n3"] == (2, "cpi_Index")


def test_ids_are_zero_indexed_within_each_type_independently(spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/c", RDF_TYPE, CPI_SERIES),
    ])
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    mapping = _mapping(node_id_df)

    assert counts == {"cpi_Index": 2, "cpi_Series": 1}
    # Each type restarts at 0.
    assert {mapping["https://ex/a"][0], mapping["https://ex/b"][0]} == {0, 1}
    assert mapping["https://ex/c"] == (0, "cpi_Series")


def test_meta_ontology_types_are_excluded(spark, make_triples):
    triples = make_triples([
        ("https://ex/real", RDF_TYPE, CPI_INDEX),
        ("https://ex/meta", RDF_TYPE, OWL_THING),  # dropped by prefix/exact filter
    ])
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    mapping = _mapping(node_id_df)

    assert counts == {"cpi_Index": 1}
    assert "https://ex/meta" not in mapping


# ======================================================================
# Canonical type selection (multi-type entities)
# ======================================================================

def test_multitype_entity_gets_most_specific_type(spark, make_triples):
    # e1 is both cpi_Index and cpi_Series. cpi_Index is the more common type
    # (3 instances) and cpi_Series the rarer/"more specific" one (1 instance),
    # so e1's single canonical type must be cpi_Series.
    triples = make_triples([
        ("https://ex/e1", RDF_TYPE, CPI_INDEX),
        ("https://ex/e1", RDF_TYPE, CPI_SERIES),
        ("https://ex/e2", RDF_TYPE, CPI_INDEX),
        ("https://ex/e3", RDF_TYPE, CPI_INDEX),
    ])
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    mapping = _mapping(node_id_df)

    assert mapping["https://ex/e1"][1] == "cpi_Series"
    # e1 no longer counts toward cpi_Index; each entity appears exactly once.
    assert counts == {"cpi_Index": 2, "cpi_Series": 1}
    assert sorted(mapping) == ["https://ex/e1", "https://ex/e2", "https://ex/e3"]


# ======================================================================
# Config filters
# ======================================================================

def test_node_types_whitelist_keeps_only_requested(spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
    ])
    node_id_df, counts = NodeMapper(
        spark, {"node_types": ["cpi_Index"]}
    ).build_node_id_table(triples)

    assert counts == {"cpi_Index": 1}
    assert "https://ex/b" not in _mapping(node_id_df)


def test_include_temporal_false_drops_temporal_types(spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/m", RDF_TYPE, UNIFIED_MONTH),  # matches temporal fragment
    ])
    node_id_df, counts = NodeMapper(
        spark, {"include_temporal_nodes": False}
    ).build_node_id_table(triples)

    assert "cpi_Index" in counts
    assert not any("Month" in t for t in counts)


def test_include_sector_false_drops_sector_types(spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/s", RDF_TYPE, UNIFIED_SECTOR),  # matches sector fragment
    ])
    node_id_df, counts = NodeMapper(
        spark, {"include_sector_nodes": False}
    ).build_node_id_table(triples)

    assert "cpi_Index" in counts
    assert not any("Sector" in t for t in counts)


# ======================================================================
# type -> URI mapping and determinism
# ======================================================================

def test_get_type_uri_mapping_recovers_source_uri(spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
    ])
    mapper = NodeMapper(spark, {})
    node_id_df, _ = mapper.build_node_id_table(triples)
    type_uris = mapper.get_type_uri_mapping(triples, node_id_df)

    assert type_uris["cpi_Index"] == CPI_INDEX
    assert type_uris["cpi_Series"] == CPI_SERIES


def test_mapping_is_deterministic_across_runs(spark, make_triples):
    rows = [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/c", RDF_TYPE, CPI_SERIES),
    ]
    first = _mapping(NodeMapper(spark, {}).build_node_id_table(make_triples(rows))[0])
    second = _mapping(NodeMapper(spark, {}).build_node_id_table(make_triples(rows))[0])
    assert first == second
