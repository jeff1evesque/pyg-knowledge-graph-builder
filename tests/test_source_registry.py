"""The source registry (#406).

Pure Python, no SparkSession. Each table built from the registered specs is
pinned to the values the pipeline had before the registry existed. The modules
that held the old tables now read the registry, so the pins are literal:
comparing a table with one of those modules would test the registry against
itself.

Also here: which source a path belongs to, what a spec may state, and the
modules that must not name a source.
"""
import ast
from pathlib import Path

import pytest
from rdflib.namespace import OWL, RDFS

from spark_jobs import sources
from spark_jobs.sources.spec import RELATION_CATEGORIES, SourceSpec
from spark_jobs.utils import namespaces, rdf_utils
from spark_jobs.utils.namespaces import (
    ALERT,
    BLS_COMMON,
    BLS_ENRICHMENT,
    CAP,
    CPI,
    ECI,
    EMPSIT,
    GEOSPARQL,
    IDENTIFIER_BASE,
    JOLTS,
    LAUS,
    MARKET_ENRICHMENT,
    MARKET_QUOTES,
    METRO,
    NOAA_ENRICHMENT,
    ONTOLOGY_BASE,
    PPI,
    REALER,
    SEC_COMMON,
    SEC_ENRICHMENT,
    SEC_FILINGS,
    SOURCE_TEMPORAL,
    UNIFIED,
    WEATHER,
    WKYENG,
    XIMPIM,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "e2e"

# NAMESPACE_PREFIXES as it was, entry for entry. Each position was also the
# namespace's ontology-source feature slot, which ONTOLOGY_NAMESPACE_INDICES now
# holds fixed.
TODAYS_NAMESPACE_PREFIXES = [
    (str(CPI), "cpi"),
    (str(PPI), "ppi"),
    (str(ECI), "eci"),
    (str(EMPSIT), "empsit"),
    (str(JOLTS), "jolts"),
    (str(LAUS), "laus"),
    (str(METRO), "metro"),
    (str(REALER), "realer"),
    (str(WKYENG), "wkyeng"),
    (str(XIMPIM), "ximpim"),
    (str(BLS_COMMON), "bls_common"),
    (str(BLS_ENRICHMENT), "bls_enrichment"),
    (str(SEC_FILINGS), "filings"),
    (str(SEC_COMMON), "sec_common"),
    (str(SEC_ENRICHMENT), "sec_enrichment"),
    (str(MARKET_ENRICHMENT), "market_enrichment"),
    (str(MARKET_QUOTES), "market_quotes"),
    (str(CAP), "cap"),
    (str(WEATHER), "weather"),
    (str(ALERT), "alert"),
    (str(NOAA_ENRICHMENT), "noaa_enrichment"),
    (str(GEOSPARQL), "geosparql"),
    (str(UNIFIED), "unified"),
    (str(SOURCE_TEMPORAL), "temporal"),
    (str(OWL), "owl"),
    (str(RDFS), "rdfs"),
]

# PROPERTY_MAPPINGS and CLASS_MAPPINGS as ontology_mapper.py wrote them out.
TODAYS_PROPERTY_MAPPINGS = {
    str(CPI.hasMonth): str(UNIFIED.hasMonth),
    str(PPI.hasStartMonth): str(UNIFIED.hasMonth),
    str(PPI.hasEndMonth): str(UNIFIED.hasMonth),
    str(ECI.hasMonth): str(UNIFIED.hasMonth),
    str(JOLTS.hasMonth): str(UNIFIED.hasMonth),
    str(EMPSIT.hasMonth): str(UNIFIED.hasMonth),
    str(XIMPIM.hasMonth): str(UNIFIED.hasMonth),
    str(LAUS.hasMonth): str(UNIFIED.hasMonth),
    str(METRO.hasMonth): str(UNIFIED.hasMonth),
    str(REALER.hasMonth): str(UNIFIED.hasMonth),
    str(CPI.hasYear): str(UNIFIED.hasYear),
    str(PPI.hasStartYear): str(UNIFIED.hasYear),
    str(PPI.hasEndYear): str(UNIFIED.hasYear),
    str(ECI.hasYear): str(UNIFIED.hasYear),
    str(JOLTS.hasYear): str(UNIFIED.hasYear),
    str(EMPSIT.hasYear): str(UNIFIED.hasYear),
    str(XIMPIM.hasYear): str(UNIFIED.hasYear),
    str(LAUS.hasYear): str(UNIFIED.hasYear),
    str(METRO.hasYear): str(UNIFIED.hasYear),
    str(REALER.hasYear): str(UNIFIED.hasYear),
    str(CPI.indexValue): str(UNIFIED.measurementValue),
    str(PPI.changeValue): str(UNIFIED.measurementValue),
    str(PPI.indexValue): str(UNIFIED.measurementValue),
    str(JOLTS.levelValue): str(UNIFIED.measurementValue),
    str(JOLTS.rateValue): str(UNIFIED.measurementValue),
    str(EMPSIT.value): str(UNIFIED.measurementValue),
    str(ECI.indexValue): str(UNIFIED.measurementValue),
    str(MARKET_QUOTES.lastPrice): str(UNIFIED.measurementValue),
    str(MARKET_QUOTES.mark): str(UNIFIED.measurementValue),
    str(CPI.hasCategory): str(UNIFIED.hasCategory),
    str(PPI.hasCommodityGrouping): str(UNIFIED.hasCategory),
    str(ECI.hasOccupationalGroup): str(UNIFIED.hasCategory),
    str(JOLTS.hasIndustry): str(UNIFIED.hasCategory),
    str(EMPSIT.hasIndustry): str(UNIFIED.hasCategory),
    str(EMPSIT.hasLaborForceCategory): str(UNIFIED.hasCategory),
    str(MARKET_QUOTES.symbol): str(UNIFIED.ticker),
    str(LAUS.hasState): str(UNIFIED.hasRegion),
    str(METRO.hasRegion): str(UNIFIED.hasRegion),
    str(CAP.hasSentTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasEffectiveTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasOnsetTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasExpirationTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasEvent): str(UNIFIED.hasEventName),
    str(CAP.hasSeverity): str(UNIFIED.hasSeverity),
    str(CAP.hasUrgency): str(UNIFIED.hasUrgency),
    str(CAP.hasAreaDescription): str(UNIFIED.hasRegionDescription),
}

TODAYS_CLASS_MAPPINGS = {
    str(CPI.Index): str(BLS_ENRICHMENT.PriceIndex),
    str(PPI.IndexValue): str(BLS_ENRICHMENT.PriceIndex),
    str(JOLTS.JobOpeningsRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(JOLTS.HiresRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(JOLTS.QuitsRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(LAUS.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(METRO.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(CPI.OneMonthPercentChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(CPI.TwelveMonthPercentChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(PPI.MonthlyChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(PPI.TwelveMonthChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(ECI.ThreeMonthPercentChangeData): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(ECI.TwelveMonthPercentChangeData): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(JOLTS.JobOpeningsLevel): str(BLS_ENRICHMENT.LevelMeasurement),
    str(JOLTS.HiresLevel): str(BLS_ENRICHMENT.LevelMeasurement),
    str(EMPSIT.EmployeeCount): str(BLS_ENRICHMENT.LevelMeasurement),
    str(LAUS.LaborForceData): str(BLS_ENRICHMENT.LevelMeasurement),
    str(CPI.Category): str(BLS_ENRICHMENT.EconomicIndicator),
    str(PPI.Grouping): str(BLS_ENRICHMENT.EconomicIndicator),
    str(JOLTS.Industry): str(BLS_ENRICHMENT.IndustryClassification),
    str(EMPSIT.Industry): str(BLS_ENRICHMENT.IndustryClassification),
    str(ECI.Industry): str(BLS_ENRICHMENT.IndustryClassification),
    str(ECI.OccupationalGroup): str(BLS_ENRICHMENT.OccupationalClassification),
    str(EMPSIT.LaborForceCategory): str(BLS_ENRICHMENT.OccupationalClassification),
    str(WEATHER.WeatherAlert): str(NOAA_ENRICHMENT.EmergencyAlert),
    str(CAP.Info): str(NOAA_ENRICHMENT.AlertInfo),
    str(CAP.Area): str(NOAA_ENRICHMENT.AlertArea),
}

# The relation fragments as edge_feature_extractor.py wrote them out, in the
# order the edge encoding config records them.
TODAYS_RELATION_FRAGMENTS = {
    "temporal": (
        "precedes", "follows", "hasNext", "hasPrevious", "temporallyRelated",
    ),
    "option_stock": ("hasUnderlyingPriceObservation", "hasUnderlying"),
    "escalation": ("escalatesTo", "escalatesFrom", "severityChange"),
    "correlation": ("correlatesWith", "relatedTo", "Correlation"),
    "causal": ("leadsTo", "impacts", "causes", "affects"),
    "strategy": ("straddleWith", "spreadWith", "strangleWith"),
    "skip": (
        "belongsToSector", "sameAs", "hasParent", "hasChild",
        "equivalentClass", "equivalentProperty", "imports",
        "refersToCompany", "hasRegion", "affectsRegion",
        "sameEventType", "affectsSameRegion",
    ),
}


def _spec(name):
    return next(spec for spec in sources.REGISTERED if spec.name == name)


def _toy(name, **fields):
    namespace = f"{ONTOLOGY_BASE}{name}/"
    return SourceSpec(
        name=name,
        path_fragments=(name,),
        namespaces=((namespace, name),),
        enrichment_namespace=namespace,
        **fields,
    )


# ======================================================================
# The tables, against today's values
# ======================================================================

def test_the_namespace_table_is_todays_entry_for_entry():
    assert sources.namespace_prefixes() == TODAYS_NAMESPACE_PREFIXES
    assert rdf_utils.NAMESPACE_PREFIXES == TODAYS_NAMESPACE_PREFIXES


def test_the_ontology_source_slots_are_todays():
    assert rdf_utils.ONTOLOGY_NAMESPACE_INDICES == [
        (namespace, slot)
        for slot, (namespace, _prefix) in enumerate(TODAYS_NAMESPACE_PREFIXES)
    ]


def test_no_namespace_registered_today_is_hashed():
    assert rdf_utils.hashed_ontology_namespaces() == []


@pytest.mark.parametrize("where", ["first", "last"])
def test_registering_a_source_moves_none_of_the_26_slots(where):
    """Slots used to be positions. A source registered first would have moved
    all 26, and one registered last the five shared namespaces, which follow
    every source's. Registered anywhere, it now moves none, and its own
    namespace is the only one hashed. What it hashes to takes Spark;
    test_feature_extractor pins it."""
    toy = _toy("toy")
    specs = (
        (toy, *sources.REGISTERED) if where == "first"
        else (*sources.REGISTERED, toy)
    )
    table = sources.namespace_prefixes(specs)

    assert rdf_utils.hashed_ontology_namespaces(table) == [f"{ONTOLOGY_BASE}toy/"]

    position = {namespace: i for i, (namespace, _prefix) in enumerate(table)}
    moved_by_position = {
        namespace
        for namespace, slot in rdf_utils.ONTOLOGY_NAMESPACE_INDICES
        if position[namespace] != slot
    }
    frozen = {namespace for namespace, _slot in rdf_utils.ONTOLOGY_NAMESPACE_INDICES}
    shared = {namespace for namespace, _prefix in sources.SHARED_NAMESPACES}
    assert moved_by_position == (frozen if where == "first" else shared)


def test_the_source_vocabularies_are_todays():
    """Compared as a set: nothing reads this tuple in order."""
    built = sources.source_vocabularies()
    assert len(built) == len(set(built))
    assert set(built) == {
        str(namespace)
        for namespace in (
            CPI, PPI, ECI, EMPSIT, JOLTS, LAUS, METRO, REALER, WKYENG, XIMPIM,
            BLS_COMMON, SEC_COMMON, SEC_FILINGS, MARKET_QUOTES, WEATHER, CAP,
        )
    }
    assert rdf_utils.SOURCE_VOCABULARIES == built


def test_the_enrichment_namespaces_are_todays():
    """Compared as a set: classify_edge_origin() only asks whether a URI
    starts with one of them."""
    built = sources.enrichment_namespaces()
    assert len(built) == len(set(built))
    assert set(built) == {
        str(BLS_ENRICHMENT), str(SEC_ENRICHMENT),
        str(MARKET_ENRICHMENT), str(NOAA_ENRICHMENT),
    }
    assert rdf_utils.ENRICHMENT_NAMESPACES == built


def test_the_synthetic_period_prefixes_are_todays():
    todays = {
        "sec": f"{IDENTIFIER_BASE}temporal/sec/",
        "noaa": f"{IDENTIFIER_BASE}temporal/noaa/",
        "market-quotes": f"{IDENTIFIER_BASE}temporal/market-quotes/",
    }
    assert sources.synthetic_temporal_ids() == todays
    assert rdf_utils.SYNTHETIC_TEMPORAL_IDS == todays


def test_the_mapping_rows_are_todays():
    """The ontology mapper's tables are the registry's, row for row."""
    from spark_jobs.enrichment import ontology_mapper

    assert len(TODAYS_PROPERTY_MAPPINGS) == 46
    assert len(TODAYS_CLASS_MAPPINGS) == 27
    assert sources.property_mappings() == TODAYS_PROPERTY_MAPPINGS
    assert sources.class_mappings() == TODAYS_CLASS_MAPPINGS
    assert ontology_mapper.PROPERTY_MAPPINGS == TODAYS_PROPERTY_MAPPINGS
    assert ontology_mapper.CLASS_MAPPINGS == TODAYS_CLASS_MAPPINGS


@pytest.mark.parametrize("category", RELATION_CATEGORIES)
def test_the_relation_fragments_are_todays_in_order(category):
    """In order, because the edge encoding config records the lists as they are."""
    from spark_jobs.pyg_builder import edge_feature_extractor as extractor

    read_by_the_extractor = {
        "temporal": extractor._TEMPORAL_RELATION_FRAGMENTS,
        "option_stock": extractor._OPTION_STOCK_RELATION_FRAGMENTS,
        "escalation": extractor._ESCALATION_RELATION_FRAGMENTS,
        "correlation": extractor._CORRELATION_RELATION_FRAGMENTS,
        "causal": extractor._CAUSAL_RELATION_FRAGMENTS,
        "strategy": extractor._STRATEGY_RELATION_FRAGMENTS,
        "skip": extractor._SKIP_RELATION_FRAGMENTS,
    }
    todays = TODAYS_RELATION_FRAGMENTS[category]
    assert sources.relation_fragments(category) == todays
    assert read_by_the_extractor[category] == todays


def test_the_relation_categories_are_the_extractors():
    from spark_jobs.pyg_builder.edge_feature_extractor import (
        _FEATURIZABLE_CATEGORIES,
    )

    assert set(RELATION_CATEGORIES) == (
        (_FEATURIZABLE_CATEGORIES - {"generic"}) | {"skip"}
    )


def test_the_date_predicates_are_todays():
    assert _spec("sec").date_predicates == (
        str(SEC_FILINGS.hasPeriodOfReport),
        str(SEC_FILINGS.hasFilingDate),
    )
    assert _spec("noaa").date_predicates == (
        str(CAP.hasSentTime),
        str(CAP.hasEffectiveTime),
        str(CAP.hasOnsetTime),
        str(CAP.hasExpirationTime),
        str(CAP.hasEndsTime),
    )
    assert _spec("market").date_predicates == (str(MARKET_QUOTES.captureTime),)
    assert _spec("bls").date_predicates == ()


def test_the_path_fragments_are_todays():
    assert {spec.name: spec.path_fragments for spec in sources.REGISTERED} == {
        "bls": ("source=bls",),
        "sec": ("source=sec",),
        "market": ("quotes",),
        "noaa": ("/noaa/",),
    }


def test_the_log_labels_are_todays():
    """bin/profiles/extra-checks.example.sh reads "Market enrichment produced"
    out of the driver log, and the linker loop builds that line from the label."""
    assert [spec.label for spec in sources.REGISTERED] == [
        "BLS", "SEC", "Market", "NOAA",
    ]


def test_the_cross_source_declarations_are_todays():
    """What cross-source linking reads from each spec, as the linker spelled it
    out before: the namespaces a source is detected by, which sources the
    sector keyword step classifies, the measurement type rows, and the steps
    that pair named sources."""
    assert {
        spec.name: set(spec.entity_namespaces) for spec in sources.REGISTERED
    } == {
        "bls": {
            str(namespace)
            for namespace in (
                CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER, WKYENG,
            )
        },
        "sec": {str(SEC_FILINGS)},
        "market": {str(MARKET_QUOTES)},
        "noaa": {str(ALERT), str(CAP), str(WEATHER)},
    }
    assert {spec.name: spec.sector_keywords for spec in sources.REGISTERED} == {
        "bls": True, "sec": False, "market": True, "noaa": True,
    }
    assert _spec("bls").measurement_types == (
        str(CPI.Index),
        str(PPI.IndexValue),
        str(JOLTS.JobOpeningsRate),
        str(JOLTS.HiresRate),
        str(JOLTS.QuitsRate),
        str(LAUS.UnemploymentRate),
    )
    assert [
        (spec.name, title, needs)
        for spec in sources.REGISTERED
        for title, needs, _step in spec.cross_source_steps
    ] == [
        ("bls", "Creating causal relationships", ("bls", "market")),
        ("market", "Linking constituents by sub-industry", ("market",)),
    ]


# ======================================================================
# Which source a path belongs to
# ======================================================================

def _fixture_paths():
    """(path, source) for every committed fixture path a test run reads: the
    .nt files, and each leaf folder of Turtle Parquet."""
    paths = [
        (str(path), path.stem)
        for path in sorted((FIXTURES / "ntriples").glob("*.nt"))
    ]
    turtle = FIXTURES / "turtle_parquet"
    for leaf in sorted({path.parent for path in turtle.rglob("*.parquet")}):
        paths.append((str(leaf), leaf.relative_to(turtle).parts[0]))
    return paths


def test_every_committed_fixture_path_matches_its_source():
    """The fixture paths carry no archive partition, so each is matched by a
    folder or file named after its source. Without that, every e2e run would
    be rejected."""
    paths = _fixture_paths()
    assert {source for _path, source in paths} == {"bls", "market", "noaa", "sec"}
    for path, source in paths:
        assert sources.match_path(path).name == source, path


@pytest.mark.parametrize("path, source", [
    ("s3a://b/raw/source=sec/feed=filings/year=2026/month=09/12.snappy.parquet", "sec"),
    ("s3a://b/raw/source=bls/feed=cpi/year=2026/month=09/12.snappy.parquet", "bls"),
    ("s3a://b/raw/noaa/year=2026/month=09/12.snappy.parquet", "noaa"),
    ("s3a://b/vendor/intraday/quotes/year=2026/month=09/day=12/", "market"),
])
def test_an_archive_path_matches_by_its_fragment(path, source):
    assert sources.match_path(path).name == source


def test_a_fragment_outranks_a_folder_named_after_another_source():
    """A mirror kept under a folder called sec still holds BLS data."""
    assert sources.match_path("/srv/sec/raw/source=bls/feed=cpi/").name == "bls"


def test_a_source_name_inside_a_longer_name_is_not_a_match():
    """A bucket called secure-data or a folder called marketing holds a
    source's name but is not that source."""
    with pytest.raises(ValueError, match="matches no registered source"):
        sources.match_path("/data/secure-data/marketing/x.parquet")


def test_a_path_matching_no_source_is_rejected():
    with pytest.raises(ValueError, match="matches no registered source"):
        sources.match_path("s3a://b/raw/year=2026/month=09/day=12/")


def test_a_path_matching_two_sources_is_rejected():
    with pytest.raises(ValueError, match=r"more than one source \(sec, market\)"):
        sources.match_path("s3a://b/raw/source=sec/quotes/")


def test_a_run_picks_each_source_once_in_registration_order():
    picked = sources.pick([
        "s3a://b/raw/noaa/year=2026/month=09/12.snappy.parquet",
        "s3a://b/raw/source=sec/feed=filings/year=2026/month=09/12.snappy.parquet",
        "s3a://b/raw/noaa/year=2026/month=09/13.snappy.parquet",
    ])
    assert [spec.name for spec in picked] == ["sec", "noaa"]


def test_a_new_source_is_matched_by_its_own_fragment():
    toy = _toy("toy")
    assert sources.match_path("s3a://b/raw/toy/x.parquet", (*sources.REGISTERED, toy)) is toy


def test_leaving_sec_out_of_a_run_moves_no_namespace_slot():
    """The namespace table is built from every registered source, never from a
    run's pick. Built from the pick, market's and NOAA's slots would move down
    by SEC's three namespaces."""
    picked = sources.pick([
        "s3a://b/raw/source=bls/feed=cpi/",
        "s3a://b/vendor/intraday/quotes/",
        "s3a://b/raw/noaa/",
    ])
    assert [spec.name for spec in picked] == ["bls", "market", "noaa"]

    slots = dict(rdf_utils.ONTOLOGY_NAMESPACE_INDICES)
    assert [slots[ns] for ns, _prefix in _spec("market").namespaces] == [15, 16]
    assert [slots[ns] for ns, _prefix in _spec("noaa").namespaces] == [17, 18, 19, 20]

    from_the_pick = {
        ns: slot for slot, (ns, _prefix) in enumerate(sources.namespace_prefixes(picked))
    }
    assert [from_the_pick[ns] for ns, _prefix in _spec("market").namespaces] == [12, 13]


# ======================================================================
# Which source a term belongs to
#
# graph_schema.json asks this of every node type it built, to say which sources
# are in the .pt rather than which ones the run read (#419).
# ======================================================================

@pytest.mark.parametrize("source, term, minted", [
    ("bls", str(CPI.Index), str(BLS_ENRICHMENT.PriceIndex)),
    ("sec", str(SEC_FILINGS.Form), str(SEC_ENRICHMENT.UnifiedCompany)),
    ("market", str(MARKET_QUOTES.OptionSnapshot), str(MARKET_ENRICHMENT.Moneyness)),
    ("noaa", str(WEATHER.WeatherAlert), str(NOAA_ENRICHMENT.EmergencyAlert)),
])
def test_a_source_term_and_one_minted_from_it_name_the_same_source(source, term, minted):
    """A type this pipeline built belongs to the source it was built from.

    Many node types are the pipeline's own -- unified companies, moneyness
    classes, emergency alerts -- and a graph holding only those still holds that
    source. A spec's enrichment namespace is one of its own namespaces, so both
    terms answer with the same name and neither needs a rule of its own.
    """
    assert sources.source_of_type_uri(term) == source
    assert sources.source_of_type_uri(minted) == source


@pytest.mark.parametrize("namespace", [ns for ns, _prefix in sources.SHARED_NAMESPACES])
def test_a_term_from_a_shared_vocabulary_belongs_to_no_source(namespace):
    """Temporal, unified, GeoSPARQL, OWL and RDFS are nobody's.

    temporal_SourceDay is in every graph whatever was read, so attributing it to
    a source would report a source that contributed nothing. None is the answer,
    not a failure to find one.
    """
    assert sources.source_of_type_uri(f"{namespace}Thing") is None


def test_a_term_under_no_registered_namespace_belongs_to_no_source():
    """An unregistered vocabulary is not an error here.

    The graph carries whatever the upstream Turtle names, which is not limited
    to what this registry knows.
    """
    assert sources.source_of_type_uri("https://example.org/ontology/toy/Thing") is None


@pytest.mark.parametrize("order", ["outer first", "inner first"])
def test_the_longer_namespace_wins_where_two_sources_nest(order):
    """Longest prefix, so the answer does not depend on registration order.

    NAMESPACE_PREFIXES matches in list order, which is why a namespace extending
    another has to be declared before it (tests/test_namespaces.py). This asks
    the question of a type after the fact, and a term under the inner namespace
    belongs to the inner source however the two were registered.
    """
    outer = _toy("toy")
    inner = SourceSpec(
        name="inner",
        path_fragments=("toy/inner",),
        namespaces=((f"{ONTOLOGY_BASE}toy/inner/", "toy_inner"),),
        enrichment_namespace=f"{ONTOLOGY_BASE}toy/inner/",
    )
    specs = (outer, inner) if order == "outer first" else (inner, outer)

    assert sources.source_of_type_uri(f"{ONTOLOGY_BASE}toy/inner/Thing", specs) == "inner"
    assert sources.source_of_type_uri(f"{ONTOLOGY_BASE}toy/Thing", specs) == "toy"


def test_a_set_of_terms_names_each_source_once_in_registration_order():
    """Registration order, not the order the terms arrived in.

    Node types reach this in whatever order the graph holds them, and two runs
    over the same sources have to report the same list.
    """
    assert sources.sources_in_type_uris([
        str(WEATHER.WeatherAlert),
        str(MARKET_QUOTES.OptionSnapshot),
        str(CPI.Index),
        str(CPI.Category),
        str(SOURCE_TEMPORAL.SourceDay),
    ]) == ("bls", "market", "noaa")


def test_no_terms_names_no_sources():
    assert sources.sources_in_type_uris([]) == ()


# ======================================================================
# Modules that run every source through its spec
# ======================================================================

GENERIC_MODULES = (
    "spark_jobs/enrichment/intra_source_linker.py",
    "spark_jobs/enrichment/temporal_unifier.py",
    "spark_jobs/enrichment/cross_source_linker.py",
    "spark_jobs/utils/canonicalization.py",
)


def _source_namespace_constants():
    """Names in utils/namespaces.py for the namespaces a source's own data
    uses: each registered spec's namespaces except its enrichment namespace."""
    used_by_sources = {
        namespace
        for spec in sources.REGISTERED
        for namespace, _prefix in spec.namespaces
        if namespace != spec.enrichment_namespace
    }
    return {
        name
        for name, value in vars(namespaces).items()
        if name.isupper() and isinstance(value, str)
        and str(value) in used_by_sources
    }


def test_the_rule_covers_every_source_vocabulary():
    assert _source_namespace_constants() == {
        "CPI", "PPI", "ECI", "EMPSIT", "JOLTS", "LAUS", "METRO", "REALER",
        "WKYENG", "XIMPIM", "BLS_COMMON", "SEC_FILINGS", "SEC_COMMON",
        "MARKET_QUOTES", "CAP", "WEATHER", "ALERT",
    }


@pytest.mark.parametrize("module", GENERIC_MODULES)
def test_a_generic_module_imports_no_source_namespace(module):
    """These modules reach each source through its spec. A source namespace
    imported here would wire that source in by name again."""
    imported = set()
    for node in ast.walk(ast.parse((REPO_ROOT / module).read_text())):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    stray = sorted(imported & _source_namespace_constants())
    assert not stray, f"{module} imports {stray}"


# ======================================================================
# What a spec may state
# ======================================================================

def test_every_spec_states_only_its_own_terms():
    for spec in sources.REGISTERED:
        own = tuple(namespace for namespace, _prefix in spec.namespaces)
        terms = [
            *spec.date_predicates, *spec.property_mappings, *spec.class_mappings,
            *spec.measurement_types,
        ]
        strays = [term for term in terms if not term.startswith(own)]
        assert not strays, (
            f"{spec.name} states terms under another source's namespaces: {strays}"
        )


def test_a_spec_must_hold_its_enrichment_namespace():
    with pytest.raises(ValueError, match="enrichment namespace"):
        SourceSpec(
            name="toy",
            path_fragments=("toy",),
            namespaces=((f"{ONTOLOGY_BASE}toy/", "toy"),),
            enrichment_namespace=f"{ONTOLOGY_BASE}toy-enrichment/",
        )


def test_date_predicates_need_a_place_to_mint_periods():
    with pytest.raises(ValueError, match="temporal prefix"):
        _toy("toy", date_predicates=(f"{ONTOLOGY_BASE}toy/hasDate",))


def test_a_spec_cannot_name_an_unknown_edge_category():
    with pytest.raises(ValueError, match="unknown edge-feature categories"):
        _toy("toy", relation_fragments={"sideways": ("affects",)})


def test_a_spec_cannot_name_an_unknown_format():
    with pytest.raises(ValueError, match="unknown source format"):
        _toy("toy", source_format="csv")


def test_entity_namespaces_are_among_the_specs_own():
    with pytest.raises(ValueError, match="entity namespaces"):
        _toy("toy", entity_namespaces=(f"{ONTOLOGY_BASE}other/",))


def test_a_measurement_type_needs_a_class_mappings_row():
    with pytest.raises(ValueError, match="no class_mappings row"):
        _toy("toy", measurement_types=(f"{ONTOLOGY_BASE}toy/Index",))


def test_two_sources_cannot_map_the_same_term():
    row = {f"{ONTOLOGY_BASE}toy/hasMonth": str(UNIFIED.hasMonth)}
    specs = (_toy("toy", property_mappings=row), _toy("other", property_mappings=row))
    with pytest.raises(ValueError, match="mapped by two sources"):
        sources.property_mappings(specs)


def test_a_registered_table_cannot_be_edited_in_place():
    with pytest.raises(TypeError):
        _spec("noaa").property_mappings[str(CAP.hasEvent)] = str(UNIFIED.hasTimestamp)


def test_the_registry_imports_neither_pyspark_nor_rdf_utils():
    """rdf_utils imports the registry, so importing rdf_utils back is a cycle.
    And specs carry functions that use Spark: those import it inside the
    function, or every module that imports rdf_utils would load pyspark."""
    paths = [
        *sorted(Path(sources.__file__).parent.glob("*.py")),
        Path(namespaces.__file__),
    ]
    offenders = {}
    for path in paths:
        imported = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        bad = sorted(
            name for name in imported
            if name.split(".")[0] == "pyspark" or name == "spark_jobs.utils.rdf_utils"
        )
        if bad:
            offenders[path.name] = bad

    assert not offenders, offenders
