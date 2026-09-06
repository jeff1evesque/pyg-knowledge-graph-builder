"""
Unit tests for BLS intra-source enrichment (current PySpark API).

Rewritten from the obsolete rdflib-based suite. These tests drive the
distributed enrichment code paths with tiny in-memory (subject, predicate,
object) DataFrames on a local SparkSession, asserting on the *output* triples
DataFrames rather than an in-memory rdflib Graph.

Covered:
  - normalize_keyword_for_uri_matching (pure helper)
  - BLSDatasetEnricher.link_temporal_sequences (precedes chains, per-category
    partitioning, chronological ordering)
  - BLSIntraSourceLinker.enrich end-to-end (temporal + sector + hierarchy)
  - BLS is still found when another source outnumbers it
  - No-BLS input short-circuits to an empty result
  - The per-dataset config maps agree with each other, and detection names
    every dataset in them
"""
import itertools

from rdflib.namespace import RDF, RDFS

from spark_jobs.utils.rdf_utils import (
    CPI, MARKET_QUOTES, BLS_ENRICHMENT, identifier_namespace,
)
from spark_jobs.enrichment.intra_source.bls_linker import (
    BLSIntraSourceLinker,
    DATASET_NS_MAP,
    normalize_keyword_for_uri_matching,
)
from spark_jobs.enrichment.intra_source.bls.base_enricher import (
    BLSDatasetEnricher, DATASET_ENRICHERS,
)
from spark_jobs.enrichment.intra_source.bls.measurements import MEASUREMENT_TYPES

# ---- URI string constants (mirror what the RML mappers emit) ----
RDF_TYPE = str(RDF.type)
RDFS_LABEL = str(RDFS.label)

CPI_INDEX = str(CPI.Index)
HAS_CATEGORY = str(CPI.hasCategory)
HAS_MONTH = str(CPI.hasMonth)
HAS_YEAR = str(CPI.hasYear)

PRECEDES = str(BLS_ENRICHMENT.precedes)
BELONGS_TO_SECTOR = str(BLS_ENRICHMENT.belongsToSector)
FOOD_SECTOR = str(BLS_ENRICHMENT.FoodSector)
HAS_PARENT = str(BLS_ENRICHMENT.hasParent)


CPI_ID = identifier_namespace(str(CPI))
MARKET_ID = identifier_namespace(str(MARKET_QUOTES))


def _cpi(local):
    """Build a CPI INDIVIDUAL uri string.

    Individuals, not terms: every caller builds a thing -- an index entity, a
    category entity, a month, a year -- and the RML mappers put all of those
    under id/cpi/. The predicates stay on the term namespace via CPI.hasMonth
    and friends above.

    This used to return str(CPI[local]), i.e. ontology/cpi/Food_Nov2024_Index,
    which no mapper emits. It passed only because the linker's own dataset
    detection had the same conflation, so the test agreed with the bug instead
    of with the data.
    """
    return f"{CPI_ID}{local}"


def _index_entity(rows, entity_local, category_uri, month, year):
    """Append the triples for a single CPI Index measurement to `rows`."""
    entity = _cpi(entity_local)
    rows.extend([
        (entity, RDF_TYPE, CPI_INDEX),
        (entity, HAS_CATEGORY, category_uri),
        (entity, HAS_MONTH, _cpi(month)),
        (entity, HAS_YEAR, _cpi(year)),
    ])
    return entity


# ======================================================================
# Pure helper
# ======================================================================

def test_normalize_keyword_for_uri_matching():
    assert normalize_keyword_for_uri_matching("Food at home") == "Food_at_home"
    assert normalize_keyword_for_uri_matching("Owners' equivalent rent") == \
        "Owners_equivalent_rent"
    assert normalize_keyword_for_uri_matching(
        "All items less food, shelter, and energy"
    ) == "All_items_less_food_shelter_and_energy"
    # Parentheses and hyphens are stripped / converted.
    assert normalize_keyword_for_uri_matching("Gasoline (all types)") == \
        "Gasoline_all_types"
    assert normalize_keyword_for_uri_matching("Food-away") == "Food_away"


# ======================================================================
# BLSDatasetEnricher.link_temporal_sequences
# ======================================================================

def test_temporal_sequence_orders_by_time(spark, make_triples):
    """Three CPI Index measurements in one category link Nov -> Dec -> Jan."""
    category = _cpi("All_items_Food_Entity")
    rows = []
    _index_entity(rows, "Food_Dec2024_Index", category, "December", "2024")
    _index_entity(rows, "Food_Nov2024_Index", category, "November", "2024")
    _index_entity(rows, "Food_Jan2025_Index", category, "January", "2025")

    df = make_triples(rows)
    result = BLSDatasetEnricher(spark, "cpi").link_temporal_sequences(df)
    assert result is not None

    precedes = [
        (r["subject"], r["object"])
        for r in result.filter("predicate = '%s'" % PRECEDES).collect()
    ]
    # Chronological order regardless of input order: Nov -> Dec -> Jan.
    assert (_cpi("Food_Nov2024_Index"), _cpi("Food_Dec2024_Index")) in precedes
    assert (_cpi("Food_Dec2024_Index"), _cpi("Food_Jan2025_Index")) in precedes
    assert len(precedes) == 2
    assert all(r["predicate"] == PRECEDES for r in result.collect())


def test_temporal_sequence_partitioned_by_category(spark, make_triples):
    """precedes links never cross category boundaries."""
    food = _cpi("Food_Entity")
    energy = _cpi("Energy_Entity")
    rows = []
    _index_entity(rows, "Food_Nov_Index", food, "November", "2024")
    _index_entity(rows, "Food_Dec_Index", food, "December", "2024")
    _index_entity(rows, "Energy_Nov_Index", energy, "November", "2024")
    _index_entity(rows, "Energy_Dec_Index", energy, "December", "2024")

    df = make_triples(rows)
    result = BLSDatasetEnricher(spark, "cpi").link_temporal_sequences(df)
    pairs = [(r["subject"], r["object"]) for r in result.collect()]

    # One precedes per category, none linking a food entity to an energy one.
    assert (_cpi("Food_Nov_Index"), _cpi("Food_Dec_Index")) in pairs
    assert (_cpi("Energy_Nov_Index"), _cpi("Energy_Dec_Index")) in pairs
    assert len(pairs) == 2
    for src, dst in pairs:
        assert ("Food" in src) == ("Food" in dst)


def test_temporal_sequence_single_measurement_yields_nothing(spark, make_triples):
    """A lone measurement in a category produces no precedes edge."""
    rows = []
    _index_entity(rows, "Food_Nov_Index", _cpi("Food_Entity"), "November", "2024")
    result = BLSDatasetEnricher(spark, "cpi").link_temporal_sequences(make_triples(rows))
    # Either None or an empty DataFrame is acceptable "no links".
    assert result is None or result.count() == 0


def test_a_measurement_dated_twice_does_not_precede_itself(spark, make_triples):
    """A measurement stating two years must not precede its own copy (#360).

    Built from the real cause rather than a synthetic duplicate row: the year
    is reached by an inner join on entity, so a measurement carrying two year
    triples arrives at the window as two rows in one category. Their sort keys
    are adjacent here, which is what let lead() return the measurement to
    itself.
    """
    category = _cpi("Food_Entity")
    rows = []
    dup = _index_entity(rows, "Food_Nov_Index", category, "November", "2024")
    rows.append((dup, HAS_YEAR, _cpi("2025")))
    later = _index_entity(rows, "Food_Dec_Index", category, "December", "2026")

    result = BLSDatasetEnricher(spark, "cpi").link_temporal_sequences(make_triples(rows))
    pairs = [(r["subject"], r["object"]) for r in result.collect()]

    assert (dup, dup) not in pairs
    assert pairs == [(dup, later)]


# ======================================================================
# BLSIntraSourceLinker.enrich (end-to-end orchestration)
# ======================================================================

def test_enrich_produces_temporal_sector_and_hierarchy(spark, make_triples):
    """
    One enrich() pass over a small CPI graph should yield:
      - precedes chain across the temporal measurements
      - belongsToSector link for the Food category entity
      - hasParent link from the child category to its parent
    """
    parent = _cpi("All_items_Entity")
    child = _cpi("All_items_Food_Entity")

    rows = [
        # Category entities must appear as subjects to be seen by the
        # sector / hierarchy steps (they scan subjects ending in _Entity).
        (parent, RDFS_LABEL, "All items"),
        (child, RDFS_LABEL, "Food"),
    ]
    _index_entity(rows, "Food_Nov2024_Index", child, "November", "2024")
    _index_entity(rows, "Food_Dec2024_Index", child, "December", "2024")

    result = BLSIntraSourceLinker(spark).enrich(make_triples(rows))
    triples = {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}

    # Temporal
    assert (_cpi("Food_Nov2024_Index"), PRECEDES, _cpi("Food_Dec2024_Index")) in triples
    # Sector: "All_items_Food_Entity" contains the "Food" keyword.
    assert (child, BELONGS_TO_SECTOR, FOOD_SECTOR) in triples
    # Hierarchy: child -> parent (immediate parent via path structure).
    assert (child, HAS_PARENT, parent) in triples


def test_hierarchy_skips_a_missing_generation(spark, make_triples):
    """
    A child whose immediate parent does not exist links to its grandparent,
    and a child with no ancestor at all links to nothing.

    This pins the depth-2 branch of _link_category_hierarchies, which nothing
    reached before. #380 rewrote that branch from a "_[^_]+$" regexp_replace to
    substring_index so it can run on the GPU, and the point of the test is that
    the output does not move: it passes on both spellings.

    It does NOT exercise the length guard on parent_path_d2. Removing that guard
    and re-running this test still passes, because an empty candidate path is
    already dropped by the inner join against existing paths -- no entity has an
    empty path. The guard stays as cheap insurance, not as a behaviour change.
    """
    grandparent = _cpi("All_items_Entity")            # path All_items
    child = _cpi("All_items_Food_Home_Entity")        # path All_items_Food_Home
    orphan = _cpi("Housing_Rent_Entity")              # path Housing_Rent

    rows = [
        (grandparent, RDFS_LABEL, "All items"),
        (child, RDFS_LABEL, "Food at home"),
        (orphan, RDFS_LABEL, "Rent"),
    ]
    # All_items_Food_Entity is deliberately absent, so depth 1 finds nothing
    # for `child`. Housing_Entity is absent too, and `orphan` has no second
    # generation to fall back to.
    _index_entity(rows, "Home_Nov2024_Index", child, "November", "2024")

    result = BLSIntraSourceLinker(spark).enrich(make_triples(rows))
    parents = {
        (r["subject"], r["object"])
        for r in result.collect()
        if r["predicate"] == HAS_PARENT
    }

    assert (child, grandparent) in parents
    assert not [p for p in parents if p[0] == orphan]
    assert not [p for p in parents if p[1] == ""]


def test_enrich_finds_bls_behind_a_wall_of_market_rows(spark, make_triples):
    """A few BLS rows after many market ones still run the whole BLS leg.

    Dataset detection used to read the first 200,000 rows of the frame and
    decide from those. Market is 99.5% of a four-source run, so no BLS row was
    ever in that sample: the linker logged "No BLS data detected", skipped all
    four steps, and left the cross-source causal step nothing to read (#350).

    The wall below is one row longer than that old sample, and the BLS rows sit
    behind it, so this fails on the sampling version and passes on the one that
    reads every row.
    """
    parent = _cpi("All_items_Entity")
    child = _cpi("All_items_Food_Entity")

    bls_rows = [
        (parent, RDFS_LABEL, "All items"),
        (child, RDFS_LABEL, "Food"),
    ]
    _index_entity(bls_rows, "Food_Nov2024_Index", child, "November", "2024")
    _index_entity(bls_rows, "Food_Dec2024_Index", child, "December", "2024")

    market = spark.range(200_001).selectExpr(
        f"concat('{MARKET_ID}', id) AS subject",
        f"'{RDF_TYPE}' AS predicate",
        f"'{str(MARKET_QUOTES.EquitySnapshot)}' AS object",
    )
    df = market.unionAll(make_triples(bls_rows))

    result = BLSIntraSourceLinker(spark).enrich(df)
    triples = {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}

    assert (child, BELONGS_TO_SECTOR, FOOD_SECTOR) in triples
    assert (child, HAS_PARENT, parent) in triples
    assert (_cpi("Food_Nov2024_Index"), PRECEDES, _cpi("Food_Dec2024_Index")) in triples


def test_enrich_returns_empty_for_non_bls_data(spark, make_triples):
    """Data with no BLS-namespaced subjects short-circuits to zero new triples."""
    rows = [
        (str(MARKET_QUOTES["AAPL_20241115"]), RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (str(MARKET_QUOTES["AAPL_20241115"]), str(MARKET_QUOTES.symbol), "AAPL"),
    ]
    result = BLSIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# Per-dataset configuration
#
# Ten datasets share one BLSDatasetEnricher, so the tests above drive cpi
# and cover the code all ten run. What is NOT shared is the config: each
# dataset needs a namespace, a registry entry, and measurement types. Get
# one wrong and nothing raises -- the dataset is quietly missing from the
# output. That is the shape of #350, one dataset at a time.
# ======================================================================

def test_every_dataset_is_wired_into_all_three_config_maps():
    """A dataset needs a namespace, an enricher, and measurement types.

    Without the namespace, detection never names it. Without measurement
    types, the enricher returns None at its first guard and logs nothing.
    Either way the dataset contributes nothing and the run still passes.
    """
    namespaces = set(DATASET_NS_MAP)
    registry = set(DATASET_ENRICHERS)
    measurements = set(MEASUREMENT_TYPES)

    assert namespaces == registry, (
        "namespace map and enricher registry disagree on: "
        f"{sorted(namespaces ^ registry)}"
    )
    assert namespaces == measurements, (
        "namespace map and measurement types disagree on: "
        f"{sorted(namespaces ^ measurements)}"
    )


def test_every_measurement_config_has_what_the_enricher_reads():
    """Each config carries the keys _link_single_measurement_type uses.

    'class' is read with a plain subscript, so a config missing it raises
    KeyError part-way through a run rather than failing here. One with no
    category or no temporal property links nothing and says nothing.
    """
    for dataset, configs in MEASUREMENT_TYPES.items():
        assert configs, f"{dataset} has an empty measurement type map"
        for name, config in configs.items():
            where = f"{dataset}.{name}"
            assert 'class' in config, f"{where} has no 'class'"
            assert config.get('category_property'), \
                f"{where} has no category_property"
            assert config.get('month_property') or config.get('quarter_property'), \
                f"{where} has neither a month nor a quarter property"


def test_dataset_namespaces_are_distinct_and_non_overlapping():
    """No dataset's namespace may start with another's.

    Detection is a chain of startswith tests and keeps the first match, so
    an overlapping pair would file one dataset's rows under the other's
    name, and which one won would depend on dict order.
    """
    for (name_a, prefix_a), (name_b, prefix_b) in itertools.permutations(
        DATASET_NS_MAP.items(), 2
    ):
        assert not prefix_b.startswith(prefix_a), (
            f"{name_b}'s namespace starts with {name_a}'s ({prefix_a}); "
            f"detection would attribute {name_b} rows to {name_a}"
        )


def test_detect_datasets_finds_every_registered_dataset(spark, make_triples):
    """One row per dataset, and detection names all of them.

    Everything above drives cpi, so a dataset that the when/otherwise chain
    cannot reach -- registered but never matched -- would go unnoticed.

    It does not prove a namespace is the one the mappers emit: the rows are
    built from the same map the assertion reads. Only real data shows that,
    which is what the cluster run is for.
    """
    rows = [
        (f"{prefix}Entity", RDF_TYPE, "https://example.org/Measurement")
        for prefix in DATASET_NS_MAP.values()
    ]

    linker = BLSIntraSourceLinker(spark)
    linker.triples_df = make_triples(rows)
    linker._bls_triples = linker._filter_bls_triples()

    assert linker._detect_datasets() == set(DATASET_NS_MAP)
