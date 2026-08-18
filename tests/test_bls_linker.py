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
  - No-BLS input short-circuits to an empty result
"""
from rdflib.namespace import RDF, RDFS

from spark_jobs.utils.rdf_utils import (
    CPI, MARKET_QUOTES, BLS_ENRICHMENT, identifier_namespace,
)
from spark_jobs.enrichment.intra_source.bls_linker import (
    BLSIntraSourceLinker,
    normalize_keyword_for_uri_matching,
)
from spark_jobs.enrichment.intra_source.bls.base_enricher import BLSDatasetEnricher

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


def test_enrich_returns_empty_for_non_bls_data(spark, make_triples):
    """Data with no BLS-namespaced subjects short-circuits to zero new triples."""
    rows = [
        (str(MARKET_QUOTES["AAPL_20241115"]), RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (str(MARKET_QUOTES["AAPL_20241115"]), str(MARKET_QUOTES.symbol), "AAPL"),
    ]
    result = BLSIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0
