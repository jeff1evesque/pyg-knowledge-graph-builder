"""The source registry rebuilds today's per-source tables (#406, step 1).

Pure Python, no SparkSession. Each table built from the registered specs is
pinned to the values the pipeline had before the registry existed. Where the
old table still lives in its own module, it is compared as well, so the two
copies cannot drift apart before that module reads the registry.
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

# NAMESPACE_PREFIXES as it was, entry for entry. A namespace's position is its
# ontology-source feature slot, so this order is part of every trained model.
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
    from spark_jobs.enrichment.ontology_mapper import (
        CLASS_MAPPINGS,
        PROPERTY_MAPPINGS,
    )

    assert len(sources.property_mappings()) == 46
    assert len(sources.class_mappings()) == 27
    assert sources.property_mappings() == PROPERTY_MAPPINGS
    assert sources.class_mappings() == CLASS_MAPPINGS


@pytest.mark.parametrize("category", RELATION_CATEGORIES)
def test_the_relation_fragments_are_todays_in_order(category):
    """In order, because the edge encoding config records the lists as they are."""
    from spark_jobs.pyg_builder import edge_feature_extractor as extractor

    todays = {
        "temporal": extractor._TEMPORAL_RELATION_FRAGMENTS,
        "option_stock": extractor._OPTION_STOCK_RELATION_FRAGMENTS,
        "escalation": extractor._ESCALATION_RELATION_FRAGMENTS,
        "correlation": extractor._CORRELATION_RELATION_FRAGMENTS,
        "causal": extractor._CAUSAL_RELATION_FRAGMENTS,
        "strategy": extractor._STRATEGY_RELATION_FRAGMENTS,
        "skip": extractor._SKIP_RELATION_FRAGMENTS,
    }
    assert sources.relation_fragments(category) == todays[category]


def test_the_relation_categories_are_the_extractors():
    from spark_jobs.pyg_builder.edge_feature_extractor import (
        _FEATURIZABLE_CATEGORIES,
    )

    assert set(RELATION_CATEGORIES) == (
        (_FEATURIZABLE_CATEGORIES - {"generic"}) | {"skip"}
    )


def test_the_date_predicates_are_todays():
    from spark_jobs.enrichment.temporal_unifier import (
        MARKET_CAPTURE_TIME,
        NOAA_DATE_PREDS,
        SEC_DATE_PREDS,
    )

    assert list(_spec("sec").date_predicates) == SEC_DATE_PREDS
    assert list(_spec("noaa").date_predicates) == NOAA_DATE_PREDS
    assert list(_spec("market").date_predicates) == [MARKET_CAPTURE_TIME]
    assert _spec("bls").date_predicates == ()


def test_the_path_labels_are_todays():
    """Compared as a set. The loader takes the first fragment that matches,
    and no source's paths contain another source's fragment."""
    from spark_jobs.graph.loading import _SOURCE_LABEL_PATTERNS, _SOURCE_NAMES

    assert set(sources.source_label_patterns()) == set(_SOURCE_LABEL_PATTERNS)
    assert sources.source_names() == _SOURCE_NAMES


# ======================================================================
# What a spec may state
# ======================================================================

def test_every_spec_states_only_its_own_terms():
    for spec in sources.REGISTERED:
        own = tuple(namespace for namespace, _prefix in spec.namespaces)
        terms = [
            *spec.date_predicates, *spec.property_mappings, *spec.class_mappings,
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
    And specs will carry functions that use Spark: those import it inside the
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
