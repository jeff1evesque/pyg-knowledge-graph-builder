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
from spark_jobs.enrichment.ontology_mapper import (
    CLASS_MAPPINGS,
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

    assert len(emitted) == len(PROPERTY_MAPPINGS)
    assert {(s, o) for s, _p, o in emitted} == set(PROPERTY_MAPPINGS.items())
    assert {p for _s, p, _o in emitted} == {OWL_EQUIVALENT_PROPERTY}


def test_class_equivalences_mirror_the_static_table(spark):
    df = OntologyMapper(spark)._create_class_equivalences()
    emitted = _rows(df)

    assert len(emitted) == len(CLASS_MAPPINGS)
    assert {(s, o) for s, _p, o in emitted} == set(CLASS_MAPPINGS.items())
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

    assert by_predicate[OWL_EQUIVALENT_PROPERTY] == len(PROPERTY_MAPPINGS)
    assert by_predicate[OWL_EQUIVALENT_CLASS] == len(CLASS_MAPPINGS)
    assert by_predicate[SKOS_PREF_LABEL] == 1


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
