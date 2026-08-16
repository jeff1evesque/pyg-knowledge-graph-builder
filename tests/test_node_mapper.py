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
  * the mapping is reproducible run-to-run,
  * get_type_uri_mapping resolves a node type with several candidate type URIs
    identically regardless of the order Spark returns rows in.

Type URIs / prefixes are taken from the code (NAMESPACE_PREFIXES, RDF_TYPE) so the
tests track the implementation rather than hardcoding IRIs.
"""
from spark_jobs.pyg_builder.node_mapper import (
    NodeMapper,
    RDF_TYPE,
    _build_uri_to_pyg_name_expr,
)

# Concrete type URIs whose PyG names are fixed by NAMESPACE_PREFIXES.
CPI_INDEX = "https://jefflevesque.com/ontology/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://jefflevesque.com/ontology/cpi/Series"      # -> cpi_Series
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

def test_multitype_entity_gets_the_superclass(spark, make_triples):
    """A subclass loses to the type that subsumes it.

    e1 is both cpi_Index and cpi_Series. Every cpi_Series instance (just e1) is
    also a cpi_Index, and cpi_Index is strictly larger, so cpi_Series is a
    subclass by extension and e1 canonicalizes to cpi_Index.

    This reverses the previous rule, which was "most specific wins" -- fewest
    instances. A subclass is ALWAYS rarer than its parent, so that rule picked
    the subclass every time and sharded one concept across every sub-class the
    source happens to state: 40 SEC filings became five node types, and
    filings_hasIssuer five edge types onto one filings_Issuer.

    cpi_Series does not disappear from the graph -- it stays an rdf:type triple
    on e1 and is still encoded in the class_identity segment. Only the
    *canonical* type, the one that decides graph structure, is the parent.
    """
    triples = make_triples([
        ("https://ex/e1", RDF_TYPE, CPI_INDEX),
        ("https://ex/e1", RDF_TYPE, CPI_SERIES),
        ("https://ex/e2", RDF_TYPE, CPI_INDEX),
        ("https://ex/e3", RDF_TYPE, CPI_INDEX),
    ])
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    mapping = _mapping(node_id_df)

    assert mapping["https://ex/e1"][1] == "cpi_Index"
    # cpi_Series is no longer a node type at all; each entity appears once.
    assert counts == {"cpi_Index": 3}
    assert sorted(mapping) == ["https://ex/e1", "https://ex/e2", "https://ex/e3"]


def test_most_specific_still_wins_without_subsumption(spark, make_triples):
    """Types that merely overlap keep the old rarest-wins tie-break.

    e1 carries both types, but e4 is a cpi_Series that is NOT a cpi_Index, so
    cpi_Series' instances are not contained in cpi_Index's and neither type
    subsumes the other. Nothing has been learned about a hierarchy, so the
    contest falls through to the pre-existing count/name ordering and the rarer
    type wins -- unchanged behaviour for every entity whose types are simply
    independent.
    """
    triples = make_triples([
        ("https://ex/e1", RDF_TYPE, CPI_INDEX),
        ("https://ex/e1", RDF_TYPE, CPI_SERIES),
        ("https://ex/e2", RDF_TYPE, CPI_INDEX),
        ("https://ex/e3", RDF_TYPE, CPI_INDEX),
        ("https://ex/e4", RDF_TYPE, CPI_SERIES),
    ])
    node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(triples)
    mapping = _mapping(node_id_df)

    assert mapping["https://ex/e1"][1] == "cpi_Series"
    assert counts == {"cpi_Index": 2, "cpi_Series": 2}


def test_subsumption_does_not_fire_on_equal_extensions(spark, make_triples):
    """Two types on exactly the same entities subsume each other -- so neither wins.

    Returning both would make the winner depend on join order, which is a
    content non-determinism in the emitted graph. The strict `count(sup) >
    count(sub)` test excludes the pair, leaving the deterministic name
    tie-break to settle it.
    """
    rows = [
        ("https://ex/e1", RDF_TYPE, CPI_INDEX),
        ("https://ex/e1", RDF_TYPE, CPI_SERIES),
        ("https://ex/e2", RDF_TYPE, CPI_INDEX),
        ("https://ex/e2", RDF_TYPE, CPI_SERIES),
    ]

    def canonical(triple_rows):
        node_id_df, counts = NodeMapper(spark, {}).build_node_id_table(
            make_triples(triple_rows)
        )
        return _mapping(node_id_df)["https://ex/e1"][1], counts

    forward = canonical(rows)
    reversed_ = canonical(list(reversed(rows)))

    assert forward == reversed_, "row order changed the canonical type"
    # cpi_Index sorts before cpi_Series, so the name tie-break picks it.
    assert forward[0] == "cpi_Index"


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


def test_type_uri_mapping_is_deterministic_for_multi_typed_node(
    spark, make_triples
):
    """A node type with several candidate type URIs must resolve identically.

    get_type_uri_mapping keeps ONE type URI per node type. e1 carries both
    CPI_INDEX and CPI_SERIES, and its canonical node type is cpi_Index (the
    superclass — see test_multitype_entity_gets_the_superclass), so BOTH URIs
    join to node type cpi_Index and one of them has to win.

    That choice used to be "whichever row Spark returned first", which is
    task-completion order and therefore varied run-to-run — the same input
    produced different metadata on different runs. Sorting the rows made it
    deterministic but still arbitrary: the smallest URI won.

    It is no longer arbitrary. The winner is the candidate whose derived PyG
    name equals the node type, so cpi_Index reports CPI_INDEX rather than the
    subclass URI that also joins to it — which is what lets
    graph_schema.json's source_type_uri be trusted, and what stops
    ontology_schema.json attributing another class's superclass_chain and
    defined_properties to this type.

    Feeding the same triples in two orders still pins determinism: row order
    must not reach the result.
    """
    rows = [
        ("https://ex/e1", RDF_TYPE, CPI_INDEX),
        ("https://ex/e1", RDF_TYPE, CPI_SERIES),
        ("https://ex/e2", RDF_TYPE, CPI_INDEX),
        ("https://ex/e3", RDF_TYPE, CPI_INDEX),
    ]

    def resolve(triple_rows):
        mapper = NodeMapper(spark, {})
        triples = make_triples(triple_rows)
        node_id_df, _ = mapper.build_node_id_table(triples)
        return mapper.get_type_uri_mapping(triples, node_id_df)

    forward = resolve(rows)
    reversed_ = resolve(list(reversed(rows)))

    assert forward == reversed_, "input row order changed the type URI mapping"
    assert forward["cpi_Index"] == CPI_INDEX, (
        "the node type must report its own class, not the smallest candidate "
        f"URI (got {forward['cpi_Index']})"
    )


# ======================================================================
# source_type_uri resolution for multi-type entities
# ======================================================================
#
# Enrichment gives entities a unified supertype alongside their specific type,
# so a node type has several candidate rdf:type URIs. Keeping the
# alphabetically smallest -- the previous rule -- picks a URI unrelated to the
# node type whenever the supertype sorts earlier. On the 2026-07-29 run that
# made jolts_HiresRate, jolts_JobOpeningsRate and jolts_QuitsRate all report
# bls_enrichment/RateMeasurement (".../enrichment/..." < ".../jolts/..."), and
# cpi_SpecialAggregateIndex report cpi/ExpenditureCategory. 43 node types, 40
# distinct URIs.
#
# The URI now comes from inverting the naming rule, so these pin that the
# reported class is the one the node type is named for.

JOLTS_HIRES_RATE = "https://jefflevesque.com/ontology/jolts/HiresRate"      # -> jolts_HiresRate
JOLTS_QUITS_RATE = "https://jefflevesque.com/ontology/jolts/QuitsRate"      # -> jolts_QuitsRate
# Sorts BEFORE both of the above: "enrichment" < "jolts".
BLS_RATE_MEASUREMENT = "https://jefflevesque.com/ontology/bls-common/enrichment/RateMeasurement"


def _type_uris(spark, rows, make_triples):
    triples = make_triples(rows)
    mapper = NodeMapper(spark, {})
    node_id_df, counts = mapper.build_node_id_table(triples)
    return mapper.get_type_uri_mapping(triples, node_id_df), counts


def _sibling_rows():
    """Two specific rate types under one shared enrichment supertype.

    The supertype must be the MORE common type, or it would itself be the
    rarest type per entity and there would be nothing to mis-report.

    It must ALSO not subsume either sibling, or canonical selection collapses
    both into bls_enrichment_RateMeasurement and neither specific node type
    survives to have a source_type_uri at all. So h3 and q3 carry their rate
    type WITHOUT the supertype: containment fails in both directions, the
    hierarchy inference correctly declines to fire, and the sibling types stay
    on the graph -- which is the situation this URI rule exists for.

    (When every instance DOES carry the supertype -- the shape real jolts data
    has -- the siblings now merge into it by design. That collapse is the point
    of the subsumption rule, and it is asserted in
    test_multitype_entity_gets_the_superclass.)
    """
    return [
        ("https://ex/h1", RDF_TYPE, JOLTS_HIRES_RATE),
        ("https://ex/h1", RDF_TYPE, BLS_RATE_MEASUREMENT),
        ("https://ex/h2", RDF_TYPE, JOLTS_HIRES_RATE),
        ("https://ex/h2", RDF_TYPE, BLS_RATE_MEASUREMENT),
        ("https://ex/h3", RDF_TYPE, JOLTS_HIRES_RATE),
        ("https://ex/q1", RDF_TYPE, JOLTS_QUITS_RATE),
        ("https://ex/q1", RDF_TYPE, BLS_RATE_MEASUREMENT),
        ("https://ex/q2", RDF_TYPE, JOLTS_QUITS_RATE),
        ("https://ex/q2", RDF_TYPE, BLS_RATE_MEASUREMENT),
        ("https://ex/q3", RDF_TYPE, JOLTS_QUITS_RATE),
        ("https://ex/q3", RDF_TYPE, BLS_RATE_MEASUREMENT),
        ("https://ex/q4", RDF_TYPE, JOLTS_QUITS_RATE),
    ]


def test_supertype_that_sorts_first_does_not_hijack_source_uri(
    spark, make_triples,
):
    """The regression: a supertype sorting earlier must not become the URI."""
    type_uris, counts = _type_uris(spark, _sibling_rows(), make_triples)

    assert "jolts_HiresRate" in counts, (
        f"fixture must produce the specific node type; got {sorted(counts)}"
    )
    assert type_uris["jolts_HiresRate"] == JOLTS_HIRES_RATE, (
        "the node type must report the class it is named after, not whichever "
        f"rdf:type sorts first (got {type_uris['jolts_HiresRate']})"
    )


def test_sibling_types_sharing_a_supertype_get_distinct_uris(
    spark, make_triples,
):
    """Two node types under one supertype must not collapse onto one URI.

    This is the shape that produced "3 node types -> 1 URI": every jolts rate
    type carries the same enrichment supertype.
    """
    type_uris, counts = _type_uris(spark, _sibling_rows(), make_triples)

    present = {t: type_uris[t] for t in counts}
    assert present["jolts_HiresRate"] == JOLTS_HIRES_RATE
    assert present["jolts_QuitsRate"] == JOLTS_QUITS_RATE
    assert len(set(present.values())) == len(present), (
        f"node types collapsed onto one source_type_uri: {present}"
    )


def test_every_node_type_round_trips_through_the_naming_rule(
    spark, make_triples,
):
    """The general invariant, independent of these fixtures.

    Whatever the data, deriving a PyG name from a node type's reported
    source_type_uri must give that node type back. This is what makes
    graph_schema.json's source_type_uri and ontology_schema.json's
    uri_to_pyg_name consistent with each other, and it holds for any input --
    so it keeps working if the naming rule or the namespace table changes.
    """
    type_uris, counts = _type_uris(
        spark,
        _sibling_rows() + [
            ("https://ex/i1", RDF_TYPE, CPI_INDEX),
            ("https://ex/s1", RDF_TYPE, CPI_SERIES),
        ],
        make_triples,
    )

    for node_type in counts:
        uri = type_uris[node_type]
        # Derive through the production expression rather than re-deriving the
        # name here, so the assertion tracks the rule instead of copying it.
        derived = (
            spark.createDataFrame([(uri,)], "uri STRING")
            .select(_build_uri_to_pyg_name_expr("uri").alias("name"))
            .first()["name"]
        )
        assert derived == node_type, (
            f"{node_type} reports source_type_uri {uri}, which derives to "
            f"{derived} -- the two artifacts would disagree about this type"
        )


def test_single_typed_entities_are_unaffected(spark, make_triples):
    """The fix must not change the common case."""
    type_uris, _ = _type_uris(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
    ], make_triples)
    assert type_uris["cpi_Index"] == CPI_INDEX
    assert type_uris["cpi_Series"] == CPI_SERIES
