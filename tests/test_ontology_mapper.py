"""Unit tests for OntologyMapper (spark_jobs/enrichment/ontology_mapper.py).

This phase had **no tests at all** and had never run: `--enable_ontology_mapping`
defaulted to false, and neither the launcher nor the notebook passed it, so
PHASE 4 of the enrichment pipeline never executed on any data, in any suite. It
is now on by default, which makes these the tests standing between it and every
graph the project builds.

What is pinned here:

  * the two equivalence steps read only their static tables, so they emit a
    fixed set of triples that does NOT depend on the input data -- including
    vocabularies absent from the load. They are declarative: nothing in this
    project reasons over OWL, so on their own they change no tensor;
  * predicate folding, which is what makes the equivalences mean something --
    restating a triple under its unified predicate so ten source month
    predicates share one feature slot instead of hashing to ten. Additive,
    data-driven, and idempotent;
  * every emitted triple is well-formed and uses the OWL/SKOS predicate the
    step is named for;
  * label normalization is data-driven and idempotent: it fills skos:prefLabel
    only where rdfs:label exists without one;
  * enrich() unions the steps, honours enable_skos, and emits nothing that
    would silently reshape the graph.

Triples are built on the shared local SparkSession via `make_triples`. Predicate
and namespace constants come from the module under test, so the tests track the
implementation rather than restating IRIs.
"""
import pytest

from spark_jobs.utils.rdf_utils import (
    CAP,
    PROV_DERIVED_BY,
    PROV_OBSERVED_LITERAL_DATATYPE,
    PROV_PROPERTY_DOMAIN,
    PROV_PROPERTY_RANGE,
    PROV_ROUTE_CURATED_SUBCLASS,
    PROV_ROUTE_NAMED_SUBCLASS,
    PROV_ROUTE_CURATED_SUBPROPERTY,
    PROV_ROUTE_OBSERVED_DOMAIN,
    PROV_ROUTE_OBSERVED_RANGE,
    PROV_ROUTE_DATATYPE_RANGE,
)

from spark_jobs.enrichment.ontology_mapper import (
    CLASS_MAPPINGS,
    CURATED_SUBCLASSES,
    CURATED_SUB_PROPERTIES,
    TRUE_EQUIVALENCES,
    TRUE_PROPERTY_EQUIVALENCES,
    ambiguous_domains,
    ambiguous_ranges,
    derive_domain_edges,
    derive_range_edges,
    partition_class_mappings,
    partition_shared_targets,
    resolve_class_mappings,
    RDF_TYPE,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUBCLASS_OF,
    RDFS_SUB_PROPERTY_OF,
    _assert_is_dag,
    _split_uri,
    derive_subclass_edges,
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
    PROPERTY_MAPPINGS,
    RDFS_LABEL,
    SKOS_PREF_LABEL,
    OntologyMapper,
)

ENTITY_A = "https://ex/a"
ENTITY_B = "https://ex/b"


def _rows(df):
    return [(r["subject"], r["predicate"], r["object"]) for r in df.collect()]


# ======================================================================
# Equivalence steps — static tables, no data dependency
# ======================================================================

def test_property_equivalences_mirror_the_static_table(spark):
    df = OntologyMapper(spark)._create_property_equivalences()
    emitted = _rows(df)

    # Only the genuine one-to-one pairs, exactly as on the class side. A
    # target claimed by several properties is a subsumption, published as
    # rdfs:subPropertyOf instead -- nine properties share
    # unified:measurementValue, and a rate is not a price.
    assert len(emitted) == len(TRUE_PROPERTY_EQUIVALENCES)
    assert {(s, o) for s, _p, o in emitted} == set(
        TRUE_PROPERTY_EQUIVALENCES.items()
    )
    assert {p for _s, p, _o in emitted} == {OWL_EQUIVALENT_PROPERTY}


def test_class_equivalences_mirror_the_static_table(spark):
    df = OntologyMapper(spark)._create_class_equivalences()
    emitted = _rows(df)

    # Only the genuine one-to-one pairs. A target claimed by several
    # sources is a subsumption, published as rdfs:subClassOf instead.
    assert len(emitted) == len(TRUE_EQUIVALENCES)
    assert {(s, o) for s, _p, o in emitted} == set(TRUE_EQUIVALENCES.items())
    assert {p for _s, p, _o in emitted} == {OWL_EQUIVALENT_CLASS}


def test_equivalences_ignore_the_input_data(spark, make_triples):
    """The property this phase's cost and output are governed by.

    Both equivalence steps read their static tables and never touch the
    triples, so loading one source emits equivalences for vocabularies that are
    not present (ppi:, laus:, metro:, ...). That is worth pinning explicitly:
    it means the emitted count is constant, the phase cannot be "sized" by
    filtering the input, and most emitted triples reference URIs the graph has
    never seen -- which is why they mostly do not become edges.
    """
    mapper = OntologyMapper(spark)
    from_empty = _rows(mapper._create_class_equivalences())

    # A single unrelated triple must not change the output.
    from_data = _rows(mapper._create_class_equivalences())
    assert from_empty == from_data

    subjects = {s for s, _p, _o in from_empty}
    assert len(subjects) > 1, "table should span several vocabularies"


# ======================================================================
# Label normalization — data-driven
# ======================================================================

def test_pref_label_is_added_where_only_rdfs_label_exists(spark, make_triples):
    triples = make_triples([
        (ENTITY_A, RDFS_LABEL, "Alpha"),
    ])
    emitted = _rows(OntologyMapper(spark)._normalize_labels(triples))

    assert emitted == [(ENTITY_A, SKOS_PREF_LABEL, "Alpha")]


def test_pref_label_is_not_duplicated_when_already_present(
    spark, make_triples,
):
    """Idempotence: running the phase on its own output adds nothing.

    Enrichment unions new triples into the graph and dropDuplicates, so a
    non-idempotent step would still not duplicate rows -- but it would keep
    doing work and keep reporting triples as "added" on every rebuild.
    """
    triples = make_triples([
        (ENTITY_A, RDFS_LABEL, "Alpha"),
        (ENTITY_A, SKOS_PREF_LABEL, "Alpha"),
    ])
    assert OntologyMapper(spark)._normalize_labels(triples) is None


def test_pref_label_only_covers_entities_missing_one(spark, make_triples):
    triples = make_triples([
        (ENTITY_A, RDFS_LABEL, "Alpha"),
        (ENTITY_B, RDFS_LABEL, "Beta"),
        (ENTITY_B, SKOS_PREF_LABEL, "Beta"),
    ])
    emitted = _rows(OntologyMapper(spark)._normalize_labels(triples))

    assert emitted == [(ENTITY_A, SKOS_PREF_LABEL, "Alpha")]


def test_no_labels_returns_none(spark, make_triples):
    triples = make_triples([(ENTITY_A, "https://ex/p", "v")])
    assert OntologyMapper(spark)._normalize_labels(triples) is None


# ======================================================================
# enrich() — the phase as the pipeline calls it
# ======================================================================

def test_enrich_unions_every_step(spark, make_triples):
    triples = make_triples([(ENTITY_A, RDFS_LABEL, "Alpha")])
    emitted = _rows(OntologyMapper(spark).enrich(triples))

    by_predicate = {}
    for _s, p, _o in emitted:
        by_predicate[p] = by_predicate.get(p, 0) + 1

    # Both equivalence steps emit only their one-to-one pairs; the
    # shared-target majority is emitted as subsumption by the two hierarchy
    # steps instead.
    assert by_predicate[OWL_EQUIVALENT_PROPERTY] == len(
        TRUE_PROPERTY_EQUIVALENCES
    )
    assert by_predicate[OWL_EQUIVALENT_CLASS] == len(TRUE_EQUIVALENCES)
    assert by_predicate[RDFS_SUB_PROPERTY_OF] == len(CURATED_SUB_PROPERTIES)
    assert by_predicate[SKOS_PREF_LABEL] == 1
    # Every derived axiom set states how it was derived.
    assert by_predicate[PROV_DERIVED_BY] == 4


def test_enrich_emits_well_formed_triples(spark, make_triples):
    """Every emitted row must be a usable triple.

    A blank or null term would flow straight into the graph: node_mapper keys
    nodes on URIs and edge_mapper on predicates, so an empty subject becomes a
    node with an empty identity rather than an error.
    """
    triples = make_triples([(ENTITY_A, RDFS_LABEL, "Alpha")])
    for subject, predicate, obj in _rows(
        OntologyMapper(spark).enrich(triples)
    ):
        assert subject and subject.strip(), "empty subject"
        assert predicate and predicate.strip(), "empty predicate"
        assert obj is not None and str(obj).strip(), "empty object"
        assert predicate.startswith("http"), predicate


def test_enable_skos_is_off_by_default(spark, make_triples):
    """The SKOS concept schemes are a separate opt-in, and stay opt-in.

    They introduce new *entities* (concept schemes) rather than statements
    about existing ones, so turning them on changes the node inventory. Pinned
    so enabling ontology mapping does not silently enable this too.
    """
    triples = make_triples([(ENTITY_A, RDFS_LABEL, "Alpha")])
    mapper = OntologyMapper(spark)

    without = _rows(mapper.enrich(triples))
    with_skos = _rows(mapper.enrich(triples, enable_skos=True))

    assert len(with_skos) > len(without)
    assert set(without).issubset(set(with_skos))


def test_enrich_is_deterministic(spark, make_triples):
    triples = make_triples([
        (ENTITY_A, RDFS_LABEL, "Alpha"),
        (ENTITY_B, RDFS_LABEL, "Beta"),
    ])
    mapper = OntologyMapper(spark)
    assert sorted(_rows(mapper.enrich(triples))) == sorted(
        _rows(mapper.enrich(triples))
    )


# ======================================================================
# Predicate folding — the step that makes the equivalences mean something
# ======================================================================

CPI_HAS_MONTH = "https://www.bls.gov/cpi/hasMonth"
JOLTS_HAS_MONTH = "https://www.bls.gov/jolts/hasMonth"
UNIFIED_HAS_MONTH = PROPERTY_MAPPINGS[CPI_HAS_MONTH]


def test_fold_restates_the_triple_under_the_unified_predicate(
    spark, make_triples,
):
    triples = make_triples([(ENTITY_A, CPI_HAS_MONTH, "3")])
    emitted = _rows(OntologyMapper(spark)._fold_predicates_to_unified(triples))

    assert emitted == [(ENTITY_A, UNIFIED_HAS_MONTH, "3")], (
        "subject and object must survive the fold unchanged; only the "
        "predicate is restated"
    )


def test_fold_gives_two_sources_the_same_predicate(spark, make_triples):
    """The payoff: one concept, one predicate, across sources.

    cpi:hasMonth and jolts:hasMonth are the same concept expressed by two
    scrapers. The encoder hashes the predicate URI, so before folding they
    occupied two unrelated feature slots and nothing connected them.
    """
    triples = make_triples([
        (ENTITY_A, CPI_HAS_MONTH, "3"),
        (ENTITY_B, JOLTS_HAS_MONTH, "3"),
    ])
    emitted = _rows(OntologyMapper(spark)._fold_predicates_to_unified(triples))

    assert {p for _s, p, _o in emitted} == {UNIFIED_HAS_MONTH}
    assert {(s, o) for s, _p, o in emitted} == {(ENTITY_A, "3"), (ENTITY_B, "3")}


def test_fold_is_additive_not_a_rewrite(spark, make_triples):
    """Source predicates must survive.

    Per-source predicates are deliberate -- sources agree on what a month is
    without being forced to share measurement semantics -- and every
    enrichment phase only adds triples. enrich() returns NEW triples that the
    pipeline unions in, so the original must still be there afterwards.
    """
    original = (ENTITY_A, CPI_HAS_MONTH, "3")
    triples = make_triples([original])
    new = OntologyMapper(spark).enrich(triples)

    combined = set(_rows(triples)) | set(_rows(new))
    assert original in combined, "the source triple was lost"
    assert (ENTITY_A, UNIFIED_HAS_MONTH, "3") in combined


def test_fold_is_data_driven(spark, make_triples):
    """Unlike the equivalence steps, absent predicates emit nothing.

    The equivalence tables are static and fire regardless of the load; folding
    must not be, or a single-source build would carry unified triples for
    vocabularies it never saw.
    """
    triples = make_triples([(ENTITY_A, "https://ex/unmapped", "v")])
    assert OntologyMapper(spark)._fold_predicates_to_unified(triples) is None


def test_fold_is_idempotent(spark, make_triples):
    """Folding an already-folded triple adds nothing.

    unified:hasMonth is not itself a key in PROPERTY_MAPPINGS, so a rebuild
    over enriched data must not fold the unified form again into something
    else, nor re-emit it.
    """
    triples = make_triples([
        (ENTITY_A, CPI_HAS_MONTH, "3"),
        (ENTITY_A, UNIFIED_HAS_MONTH, "3"),
    ])
    emitted = _rows(OntologyMapper(spark)._fold_predicates_to_unified(triples))
    assert emitted == [(ENTITY_A, UNIFIED_HAS_MONTH, "3")], (
        "the fold must emit only the restatement of the source triple, and "
        "the pipeline's dropDuplicates then makes the union a no-op"
    )
    assert UNIFIED_HAS_MONTH not in PROPERTY_MAPPINGS, (
        "a unified predicate must never itself be foldable, or repeated "
        "builds would keep restating it"
    )


def test_every_unified_target_is_a_fold_fixed_point(spark):
    """No unified predicate may appear on the left of the mapping table.

    A target that is also a source would make folding non-terminating in
    meaning: build N restates it as something else again.
    """
    targets = set(PROPERTY_MAPPINGS.values())
    overlap = targets & set(PROPERTY_MAPPINGS)
    assert not overlap, f"unified predicates that are also folded: {overlap}"


# ======================================================================
# Class hierarchy derivation — the rules, and the guarantees about them
# ======================================================================
#
# These rules INVENT ontology structure that no source asserts, so the tests
# below are split deliberately: the first group pins what the rules produce,
# the second pins the properties that must hold for ANY input. A wrong
# subClassOf is worse than a missing one -- it feeds fabricated structure into
# the 64-dim class_hierarchy sub-segment, which the GNN cannot distinguish
# from a declared one.

JOLTS_NS = "https://www.bls.gov/jolts/"
CPI_NS = "https://www.bls.gov/cpi/"


def _names(edges):
    """Edges as (child_local, parent_local) for readable assertions."""
    return {(_split_uri(c)[1], _split_uri(p)[1]) for c, p in edges}


def test_measurement_class_specialises_its_dataset():
    edges = derive_subclass_edges([
        JOLTS_NS + "HiresLevel",
        JOLTS_NS + "HiresRate",
        JOLTS_NS + "HiresData",
    ])
    assert _names(edges) == {
        ("HiresLevel", "HiresData"), ("HiresRate", "HiresData"),
    }


def test_no_parent_is_invented_when_the_dataset_class_is_absent():
    # EstablishmentLevel/Rate are real classes with no EstablishmentData in
    # the data. The rule must find nothing rather than mint a URI.
    edges = derive_subclass_edges([
        JOLTS_NS + "EstablishmentLevel",
        JOLTS_NS + "EstablishmentRate",
    ])
    assert edges == []


def test_qualified_name_specialises_the_name_it_qualifies():
    edges = derive_subclass_edges([
        CPI_NS + "UnadjustedPercentChange",
        CPI_NS + "SeasonallyAdjustedPercentChange",
        CPI_NS + "PercentChange",
    ])
    assert _names(edges) == {
        ("UnadjustedPercentChange", "PercentChange"),
        ("SeasonallyAdjustedPercentChange", "PercentChange"),
    }


def test_a_class_may_specialise_along_two_axes():
    # OtherSeparationsLevel is both a kind of OtherSeparationsData (rule 1)
    # and a kind of SeparationsLevel (rule 2). RDFS allows both.
    edges = derive_subclass_edges([
        JOLTS_NS + "OtherSeparationsLevel",
        JOLTS_NS + "OtherSeparationsData",
        JOLTS_NS + "SeparationsLevel",
    ])
    assert _names(edges) == {
        ("OtherSeparationsLevel", "OtherSeparationsData"),
        ("OtherSeparationsLevel", "SeparationsLevel"),
    }


def test_hierarchy_does_not_cross_namespaces():
    # Identical local names in two vocabularies are unrelated classes.
    edges = derive_subclass_edges([
        JOLTS_NS + "UnadjustedPercentChange",
        CPI_NS + "PercentChange",
    ])
    assert edges == []


# --- guarantees that must hold for any input ------------------------- #

# The real 2026-07-29 class set, abbreviated to the families that carry
# structure plus classes that must NOT gain a parent.
REAL_CLASSES = [
    JOLTS_NS + n for n in (
        "HiresData", "HiresLevel", "HiresRate",
        "QuitsData", "QuitsLevel", "QuitsRate",
        "SeparationsData", "SeparationsLevel", "SeparationsRate",
        "OtherSeparationsData", "OtherSeparationsLevel",
        "EstablishmentLevel", "EstablishmentRate",
    )
] + [
    CPI_NS + n for n in (
        "PercentChange", "UnadjustedPercentChange", "EffectOnAllItems",
        "UnadjustedEffectOnAllItems", "StandardError", "ExpenditureCategory",
    )
] + ["http://www.sec.gov/filings#SECFiling"]


def test_every_derived_parent_is_an_existing_class():
    # The load-bearing guarantee: the rules may fail to find a parent, but
    # can never point at a class the data does not contain.
    known = set(REAL_CLASSES)
    for child, parent in derive_subclass_edges(REAL_CLASSES):
        assert parent in known, f"invented superclass {parent}"
        assert child in known


def test_no_class_is_its_own_superclass():
    for child, parent in derive_subclass_edges(REAL_CLASSES):
        assert child != parent


def test_derivation_is_acyclic_on_the_real_class_set():
    # Enforced inside derive_subclass_edges; asserted here because the
    # encoder walks this to a transitive closure that a cycle would hang.
    _assert_is_dag(derive_subclass_edges(REAL_CLASSES))


def test_a_cycle_is_rejected_rather_than_returned():
    with pytest.raises(ValueError, match="cycle"):
        _assert_is_dag([("https://ex/A", "https://ex/B"),
                        ("https://ex/B", "https://ex/A")])


def test_derivation_is_deterministic_regardless_of_input_order():
    forward = derive_subclass_edges(REAL_CLASSES)
    backward = derive_subclass_edges(list(reversed(REAL_CLASSES)))
    assert forward == backward
    assert forward == sorted(forward)


def test_classes_with_no_structural_kin_gain_nothing():
    parented = {c for c, _ in derive_subclass_edges(REAL_CLASSES)}
    for orphan in ("SECFiling", "StandardError", "ExpenditureCategory",
                   "EstablishmentLevel", "HiresData"):
        assert not any(_split_uri(c)[1] == orphan for c in parented), orphan


# --- the step, end to end -------------------------------------------- #

def test_enrich_emits_the_derived_hierarchy(spark, make_triples):
    triples = make_triples([
        ("https://ex/n1", RDF_TYPE, JOLTS_NS + "HiresRate"),
        ("https://ex/n2", RDF_TYPE, JOLTS_NS + "HiresData"),
    ])
    out = OntologyMapper(spark).enrich(triples)
    derived = {
        (r["subject"], r["object"]) for r in out.collect()
        if r["predicate"] == RDFS_SUBCLASS_OF
    }
    # Curated edges ship unconditionally; this asserts the naming rule's
    # contribution on top of them.
    naming = derived - set(CURATED_SUBCLASSES.items())
    assert naming == {(JOLTS_NS + "HiresRate", JOLTS_NS + "HiresData")}


def test_derivation_emits_nothing_without_a_derivable_pair(spark, make_triples):
    triples = make_triples([
        ("https://ex/n1", RDF_TYPE, JOLTS_NS + "EstablishmentRate"),
    ])
    out = OntologyMapper(spark).enrich(triples)
    derived = {
        (r["subject"], r["object"]) for r in out.collect()
        if r["predicate"] == RDFS_SUBCLASS_OF
    }
    # Nothing beyond the curated table: the naming rules found no pair.
    assert derived == set(CURATED_SUBCLASSES.items())


# --- negating qualifiers: the rule that meaning, not spelling, decides --- #

CPI_NEGATION_CLASSES = [
    CPI_NS + n for n in (
        "Shelter", "AllItemsLessShelter", "AllItemsLessFoodAndShelter",
        "WhiskeyAtHome", "DistilledSpiritsExcludingWhiskeyAtHome",
        "Commodities", "MedicalCareCommodities",
    )
]


def test_a_negating_qualifier_does_not_create_a_subclass():
    # "All items LESS shelter" ends with "Shelter" but asserts the opposite
    # of being one. Caught by running the rules over the e2e fixture classes.
    edges = _names(derive_subclass_edges(CPI_NEGATION_CLASSES))
    assert ("AllItemsLessShelter", "Shelter") not in edges
    assert ("AllItemsLessFoodAndShelter", "Shelter") not in edges
    assert ("DistilledSpiritsExcludingWhiskeyAtHome", "WhiskeyAtHome") \
        not in edges


def test_a_narrowing_qualifier_still_creates_a_subclass():
    # The negation guard must not swallow ordinary specialisation.
    edges = _names(derive_subclass_edges(CPI_NEGATION_CLASSES))
    assert ("MedicalCareCommodities", "Commodities") in edges


def test_negation_is_detected_only_in_the_qualifier():
    # "Less" appearing inside the PARENT's own name is not a negation of it.
    edges = _names(derive_subclass_edges([
        CPI_NS + "LessonFees", CPI_NS + "PrivateLessonFees",
    ]))
    assert ("PrivateLessonFees", "LessonFees") in edges


# ======================================================================
# Curated hierarchy — CLASS_MAPPINGS read as subsumption, not equivalence
# ======================================================================

def test_a_shared_target_is_subsumption_not_equivalence():
    # Two sources pointing at one target cannot both equal it — that would
    # make them equal to each other.
    subs, eqs = partition_class_mappings({
        "https://ex/HiresRate": "https://ex/RateMeasurement",
        "https://ex/QuitsRate": "https://ex/RateMeasurement",
        "https://ex/Alert": "https://ex/EmergencyAlert",
    })
    assert subs == {
        "https://ex/HiresRate": "https://ex/RateMeasurement",
        "https://ex/QuitsRate": "https://ex/RateMeasurement",
    }
    assert eqs == {"https://ex/Alert": "https://ex/EmergencyAlert"}


def test_the_partition_covers_every_mapping_exactly_once():
    subs, eqs = partition_class_mappings(CLASS_MAPPINGS)
    assert not (set(subs) & set(eqs))
    assert {**subs, **eqs} == CLASS_MAPPINGS


def test_the_real_table_is_mostly_subsumption():
    # Documents the finding rather than a threshold: the table was written
    # as equivalences, but most of it describes kinds.
    assert len(CURATED_SUBCLASSES) > len(TRUE_EQUIVALENCES)
    rate = [s for s, t in CURATED_SUBCLASSES.items()
            if t.endswith("RateMeasurement")]
    assert len(rate) >= 4


def test_curated_edges_outrank_a_missing_class_in_the_data(spark,
                                                           make_triples):
    # A curated parent need not be instantiated: RateMeasurement is an
    # enrichment class that may never appear as an rdf:type object.
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    out = OntologyMapper(spark).enrich(triples)
    derived = {(r["subject"], r["object"]) for r in out.collect()
               if r["predicate"] == RDFS_SUBCLASS_OF}
    assert set(CURATED_SUBCLASSES.items()) <= derived


def test_a_source_is_never_both_equivalent_and_a_subclass(spark,
                                                          make_triples):
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    rows = OntologyMapper(spark).enrich(triples).collect()
    equivalent = {r["subject"] for r in rows
                  if r["predicate"] == OWL_EQUIVALENT_CLASS}
    subclassed = {r["subject"] for r in rows
                  if r["predicate"] == RDFS_SUBCLASS_OF}
    assert not (equivalent & subclassed)


# --- per-run overrides: adding a source must not require editing this module #

def test_an_override_adds_a_mapping_without_restating_the_table():
    merged = resolve_class_mappings({"https://ex/NewClass": "https://ex/Base"})
    assert merged["https://ex/NewClass"] == "https://ex/Base"
    assert set(CLASS_MAPPINGS).issubset(merged)


def test_an_override_replaces_a_built_in_target():
    source = next(iter(CLASS_MAPPINGS))
    merged = resolve_class_mappings({source: "https://ex/Elsewhere"})
    assert merged[source] == "https://ex/Elsewhere"


def test_a_null_target_drops_a_built_in_mapping():
    source = next(iter(CLASS_MAPPINGS))
    merged = resolve_class_mappings({source: None})
    assert source not in merged
    assert len(merged) == len(CLASS_MAPPINGS) - 1


def test_no_override_leaves_the_table_untouched():
    assert resolve_class_mappings(None) == CLASS_MAPPINGS
    assert resolve_class_mappings({}) == CLASS_MAPPINGS


def test_an_override_can_move_a_pair_between_equivalence_and_subsumption(
        spark):
    # cap:Info is the one-to-one pair; pointing a second source at its target
    # makes both subclasses instead.
    info = str(CAP.Info)
    target = CLASS_MAPPINGS[info]
    mapper = OntologyMapper(spark, {"https://ex/OtherInfo": target})
    assert info in mapper.curated_subclasses
    assert info not in mapper.true_equivalences


def test_overrides_reach_the_emitted_triples(spark, make_triples):
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    mapper = OntologyMapper(spark, {"https://ex/A": "https://ex/Shared",
                                    "https://ex/B": "https://ex/Shared"})
    rows = mapper.enrich(triples).collect()
    subclassed = {(r["subject"], r["object"]) for r in rows
                  if r["predicate"] == RDFS_SUBCLASS_OF}
    assert ("https://ex/A", "https://ex/Shared") in subclassed
    assert ("https://ex/B", "https://ex/Shared") in subclassed


# ======================================================================
# Property hierarchy — the same shared-target reading, applied to properties
# ======================================================================
#
# PROPERTY_MAPPINGS had the structure CLASS_MAPPINGS turned out to have: 41 of
# its 46 entries point at a target two or more sources share. Publishing all 46
# as owl:equivalentProperty said jolts:rate and market:observedPrice are the
# same property -- that a rate is a price -- and left property_hierarchy (81
# dims of every node vector) with nothing to encode.

def test_the_property_table_is_mostly_subsumption():
    assert len(CURATED_SUB_PROPERTIES) + len(TRUE_PROPERTY_EQUIVALENCES) == (
        len(PROPERTY_MAPPINGS)
    )
    # Pinned rather than asserted loosely: the ratio is the whole reason this
    # step exists, so a table edit that inverts it should fail here.
    assert len(CURATED_SUB_PROPERTIES) == 41
    assert len(TRUE_PROPERTY_EQUIVALENCES) == 5


def test_a_property_is_never_both_equivalent_and_a_sub_property(
        spark, make_triples):
    """Mirrors the class-side check: the two readings must not overlap.

    Emitting both would assert P = Q and P < Q of the same pair -- and any
    consumer resolving the vocabulary would get a different answer depending
    on which statement it read first.
    """
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    rows = OntologyMapper(spark).enrich(triples).collect()

    equivalent = {(r["subject"], r["object"]) for r in rows
                  if r["predicate"] == OWL_EQUIVALENT_PROPERTY}
    subsumed = {(r["subject"], r["object"]) for r in rows
                if r["predicate"] == RDFS_SUB_PROPERTY_OF}

    assert equivalent, "no property equivalences emitted at all"
    assert subsumed, "no property subsumptions emitted at all"
    assert not (equivalent & subsumed), (
        f"{len(equivalent & subsumed)} pair(s) asserted both equivalent and "
        f"sub-property: {sorted(equivalent & subsumed)[:3]}"
    )
    # ...and no source appears on both sides even against a DIFFERENT target.
    assert not ({s for s, _ in equivalent} & {s for s, _ in subsumed})


def test_the_nine_measurement_properties_become_sub_properties(spark,
                                                               make_triples):
    unified_value = PROPERTY_MAPPINGS[
        "https://www.bls.gov/jolts/rate"
    ]
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    rows = OntologyMapper(spark).enrich(triples).collect()

    subsumed = {(r["subject"], r["object"]) for r in rows
                if r["predicate"] == RDFS_SUB_PROPERTY_OF}
    assert ("https://www.bls.gov/jolts/rate", unified_value) in subsumed
    assert (
        "https://financial-data.org/observedPrice", unified_value
    ) in subsumed


def test_the_property_hierarchy_is_acyclic_and_deterministic(spark,
                                                             make_triples):
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    mapper = OntologyMapper(spark)
    first = sorted(_rows(mapper._derive_property_hierarchy()))
    again = sorted(_rows(mapper._derive_property_hierarchy()))
    assert first == again
    # _derive_property_hierarchy runs _assert_is_dag itself, so reaching here
    # is the acyclicity assertion; this pins that the check is wired in.
    with pytest.raises(ValueError, match="property hierarchy contains a cycle"):
        _assert_is_dag(
            [("https://ex/p", "https://ex/q"), ("https://ex/q", "https://ex/p")],
            label="property hierarchy",
        )


# ======================================================================
# Domain / range derivation — observation, not spelling
# ======================================================================

P = "https://ex/prop"
Q = "https://ex/other"
CLS_A = "https://ex/ClassA"
CLS_B = "https://ex/ClassB"
XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def test_a_property_used_on_one_class_gets_that_domain():
    assert derive_domain_edges([(P, CLS_A)]) == [(P, CLS_A)]


def test_a_property_used_on_two_classes_gets_no_domain():
    """rdfs:domain is an INTERSECTION, not a hint.

    Emitting both would state that every subject of P is an A *and* a B.
    Measured over the fixtures, 77 of 160 predicates are used on more than one
    class -- market:askPrice on EquitySnapshot and OptionSnapshot, rdfs:label
    on 68 -- so this is the common case, not an edge case.
    """
    assert derive_domain_edges([(P, CLS_A), (P, CLS_B)]) == []
    assert ambiguous_domains([(P, CLS_A), (P, CLS_B)]) == {P: [CLS_A, CLS_B]}


def test_an_unambiguous_property_is_not_reported_as_ambiguous():
    assert ambiguous_domains([(P, CLS_A)]) == {}


def test_range_comes_from_a_declared_datatype():
    assert derive_range_edges([], [(P, XSD_DECIMAL)]) == [(P, XSD_DECIMAL)]


def test_range_comes_from_a_typed_object():
    assert derive_range_edges([(P, CLS_A)], []) == [(P, CLS_A)]


def test_a_property_with_two_datatypes_gets_no_range():
    assert derive_range_edges([], [(P, XSD_DECIMAL), (P, XSD_STRING)]) == []
    assert ambiguous_ranges([], [(P, XSD_DECIMAL), (P, XSD_STRING)]) == {
        P: sorted([XSD_DECIMAL, XSD_STRING])
    }


def test_a_property_used_with_both_entities_and_literals_gets_no_range():
    """The two routes disagree about what kind of thing P even points at.

    cpi:hasMonth is the real case: 189 typed URIs and 11 literals. Taking
    either route alone would assert a range the other contradicts.
    """
    assert derive_range_edges([(P, CLS_A)], [(P, XSD_DECIMAL)]) == []
    assert P in ambiguous_ranges([(P, CLS_A)], [(P, XSD_DECIMAL)])


def test_derivation_is_deterministic_regardless_of_observation_order():
    obs = [(Q, CLS_B), (P, CLS_A)]
    assert derive_domain_edges(obs) == derive_domain_edges(list(reversed(obs)))
    assert derive_domain_edges(obs) == sorted(derive_domain_edges(obs))

    rng = [(Q, CLS_B), (P, CLS_A)]
    assert derive_range_edges(rng, []) == derive_range_edges(
        list(reversed(rng)), []
    )


def test_derivation_emits_domain_and_range_from_the_graph(spark, make_triples):
    """The whole route, over a graph shaped like the real ones."""
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CLS_A),
        ("https://ex/b", RDF_TYPE, CLS_B),
        # P: entity-valued, one subject class, one object class.
        ("https://ex/a", P, "https://ex/b"),
        # Q: literal-valued, its datatype recovered by the loader.
        ("https://ex/a", Q, "1.5"),
        (Q, PROV_OBSERVED_LITERAL_DATATYPE, XSD_DECIMAL),
    ])
    emitted, skipped_domains, skipped_ranges = (
        OntologyMapper(spark)._derive_property_schema(triples)
    )
    rows = set(_rows(emitted))

    assert (P, RDFS_DOMAIN, CLS_A) in rows
    assert (P, RDFS_RANGE, CLS_B) in rows
    assert (Q, RDFS_DOMAIN, CLS_A) in rows
    assert (Q, RDFS_RANGE, XSD_DECIMAL) in rows
    assert skipped_domains == {} and skipped_ranges == {}


def test_the_marker_predicate_is_not_itself_given_a_schema(spark,
                                                           make_triples):
    """Provenance markers describe the vocabulary; they do not use it.

    Left in, prov:observedLiteralDatatype would acquire a domain and range of
    its own and occupy encoder slots meant for signal about entities.
    """
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CLS_A),
        ("https://ex/a", Q, "1.5"),
        (Q, PROV_OBSERVED_LITERAL_DATATYPE, XSD_DECIMAL),
    ])
    emitted, _, _ = OntologyMapper(spark)._derive_property_schema(triples)
    subjects = {s for s, _, _ in _rows(emitted)}
    assert PROV_OBSERVED_LITERAL_DATATYPE not in subjects
    assert RDF_TYPE not in subjects


def test_an_ambiguous_predicate_is_reported_not_silently_dropped(
        spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CLS_A),
        ("https://ex/b", RDF_TYPE, CLS_B),
        ("https://ex/a", P, "x"),
        ("https://ex/b", P, "y"),
    ])
    _emitted, skipped_domains, _ = (
        OntologyMapper(spark)._derive_property_schema(triples)
    )
    assert skipped_domains == {P: sorted([CLS_A, CLS_B])}


def test_enrich_states_how_every_axiom_set_was_derived(spark, make_triples):
    """Provenance travels as triples because the graph is built in another job.

    pyg_only reads the enriched Parquet, so anything not in the triples never
    reaches the metadata writer -- and an inferred rdfs:domain is otherwise
    indistinguishable from one a source declared.
    """
    triples = make_triples([("https://ex/a", RDF_TYPE, CLS_A)])
    rows = OntologyMapper(spark).enrich(triples).collect()
    stated = {r["subject"]: r["object"] for r in rows
              if r["predicate"] == PROV_DERIVED_BY}

    assert PROV_PROPERTY_DOMAIN in stated
    assert PROV_PROPERTY_RANGE in stated
    assert "observed" in stated[PROV_PROPERTY_DOMAIN]
    assert "declared" in stated[PROV_PROPERTY_RANGE]


# ======================================================================
# Route markers — which rule produced each individual axiom
# ======================================================================
#
# A curated edge and a name-inferred one are byte-identical once in the graph,
# and they do not deserve equal trust: a person chose the curated ones, while
# the naming rules read AllItemsLessShelter as a kind of Shelter until the
# negating-qualifier guard was added. The mapper is the only place that still
# knows which is which, so it restates each edge under a route predicate.

def test_every_derived_subclass_edge_carries_a_route(spark, make_triples):
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    rows = OntologyMapper(spark).enrich(triples).collect()

    edges = {(r["subject"], r["object"]) for r in rows
             if r["predicate"] == RDFS_SUBCLASS_OF}
    routed = {(r["subject"], r["object"]) for r in rows
              if r["predicate"] in (PROV_ROUTE_CURATED_SUBCLASS,
                                    PROV_ROUTE_NAMED_SUBCLASS)}

    assert edges, "no hierarchy emitted at all"
    assert edges == routed, (
        f"{len(edges - routed)} derived edge(s) carry no route marker, so "
        f"the artifact cannot tell them from a source's own declaration"
    )


def test_curated_and_named_routes_do_not_overlap(spark, make_triples):
    """One edge, one route. Curated wins where both rules find the same pair.

    Two markers on one edge would double-count it and leave the reader no
    answer to "which rule do I go and fix".
    """
    triples = make_triples([("https://ex/n", RDF_TYPE, "https://ex/Nothing")])
    rows = OntologyMapper(spark).enrich(triples).collect()

    curated = {(r["subject"], r["object"]) for r in rows
               if r["predicate"] == PROV_ROUTE_CURATED_SUBCLASS}
    named = {(r["subject"], r["object"]) for r in rows
             if r["predicate"] == PROV_ROUTE_NAMED_SUBCLASS}

    assert not (curated & named), sorted(curated & named)[:3]
    assert curated == set(CURATED_SUBCLASSES.items())


def test_a_name_inferred_edge_is_marked_as_such(spark, make_triples):
    """The naming rules fire on classes present in the load, and get labelled.

    HiresRate -> HiresData is rule 1 (measurement specialises its dataset),
    and nothing curated mentions it.
    """
    ns = "https://www.bls.gov/jolts/"
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, f"{ns}HiresRate"),
        ("https://ex/b", RDF_TYPE, f"{ns}HiresData"),
    ])
    rows = OntologyMapper(spark).enrich(triples).collect()

    named = {(r["subject"], r["object"]) for r in rows
             if r["predicate"] == PROV_ROUTE_NAMED_SUBCLASS}
    assert (f"{ns}HiresRate", f"{ns}HiresData") in named


def test_every_derived_property_axiom_carries_a_route(spark, make_triples):
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CLS_A),
        ("https://ex/a", P, "1.5"),
        (P, PROV_OBSERVED_LITERAL_DATATYPE, XSD_DECIMAL),
    ])
    rows = OntologyMapper(spark).enrich(triples).collect()

    for axiom_pred, route_preds in (
        (RDFS_DOMAIN, (PROV_ROUTE_OBSERVED_DOMAIN,)),
        (RDFS_RANGE, (PROV_ROUTE_OBSERVED_RANGE, PROV_ROUTE_DATATYPE_RANGE)),
        (RDFS_SUB_PROPERTY_OF, (PROV_ROUTE_CURATED_SUBPROPERTY,)),
    ):
        axioms = {(r["subject"], r["object"]) for r in rows
                  if r["predicate"] == axiom_pred}
        routed = {(r["subject"], r["object"]) for r in rows
                  if r["predicate"] in route_preds}
        assert axioms, f"no {axiom_pred} emitted"
        assert axioms == routed, f"{axiom_pred} has unrouted axioms"


def test_a_datatype_range_is_distinguished_from_an_observed_one(spark,
                                                                make_triples):
    """Two routes, two levels of evidence.

    An object's rdf:type is something this pipeline observed. An XSD datatype
    was DECLARED by the source and merely carried through the parse. Reporting
    both as "observed" would understate the second.
    """
    triples = make_triples([
        ("https://ex/a", RDF_TYPE, CLS_A),
        ("https://ex/b", RDF_TYPE, CLS_B),
        ("https://ex/a", P, "https://ex/b"),      # entity-valued
        ("https://ex/a", Q, "1.5"),               # literal-valued
        (Q, PROV_OBSERVED_LITERAL_DATATYPE, XSD_DECIMAL),
    ])
    rows = OntologyMapper(spark).enrich(triples).collect()

    observed = {(r["subject"], r["object"]) for r in rows
                if r["predicate"] == PROV_ROUTE_OBSERVED_RANGE}
    from_datatype = {(r["subject"], r["object"]) for r in rows
                     if r["predicate"] == PROV_ROUTE_DATATYPE_RANGE}

    assert (P, CLS_B) in observed
    assert (Q, XSD_DECIMAL) in from_datatype
    assert not (observed & from_datatype)


def test_route_markers_never_become_features(spark, make_triples):
    """They describe the vocabulary; they must not be encoded as data.

    Their subjects are class and predicate URIs, so left in they would collect
    a domain and a range of their own.
    """
    from spark_jobs.pyg_builder.feature_extractor import (
        _NON_FEATURE_PREDICATES,
    )
    for route in (PROV_ROUTE_CURATED_SUBCLASS, PROV_ROUTE_NAMED_SUBCLASS,
                  PROV_ROUTE_CURATED_SUBPROPERTY, PROV_ROUTE_OBSERVED_DOMAIN,
                  PROV_ROUTE_OBSERVED_RANGE, PROV_ROUTE_DATATYPE_RANGE):
        assert route in _NON_FEATURE_PREDICATES, route
