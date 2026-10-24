"""
Tests for the SEC intra-source enrichment linker (SECIntraSourceLinker).

Two tiers, mirroring test_bls_linker.py:
  - Tier 2 smokes: enrich() tags a violation type by URI keyword, and foreign
    input short-circuits to zero triples.
  - Targeted deep test: CIK-based company unification
    (`_unify_company_entities`) — the "same CIK across >= 2 datasets" rule,
    including the negative case where a single-dataset CIK is NOT unified.
  - The codes filings link to (`_type_linked_codes`): each 8-K item and XBRL
    concept becomes a node with its code, and the links to them become edges.

Imports underscore-prefixed module constants (the linker's canonical URIs) so
the tests track the code rather than hardcoding IRIs. Drives enrich() over tiny
in-memory triples on the shared local SparkSession (`spark` / `make_triples`).
"""
import logging

from pyspark.sql import functions as F

from spark_jobs.enrichment.intra_source.market_linker import (
    RDF_TYPE as MARKET_RDF_TYPE,
    EQUITY_SNAPSHOT_TYPE,
)
from spark_jobs.enrichment.intra_source.sec_linker import (
    SECIntraSourceLinker,
    SEC_VIOLATION_PATTERNS,
    _RDF_TYPE,
    _RDFS_LABEL,
    _OWL_SAME_AS,
    _SEC_FILINGS_NS,
    _UNIFIED_NS,
    _FORM4_TYPE,
    _FORM8K_TYPE,
    _FORM10Q_TYPE,
    _FILINGS_HAS_ISSUER_CIK,
    _FILINGS_HAS_REPORTING_OWNER,
    _FILINGS_HAS_NON_DERIVATIVE_TRANSACTION,
    _FILINGS_HAS_DERIVATIVE_TRANSACTION,
    _FILINGS_HAS_TRANSACTION_DATE,
    _FILINGS_HAS_FORM8K_ITEM,
    _FILINGS_HAS_CONCEPT,
    _NON_DERIVATIVE_TRANSACTION_TYPE,
    _DERIVATIVE_TRANSACTION_TYPE,
    _FORM8K_ITEM_TYPE,
    _XBRL_CONCEPT_TYPE,
    _HAS_ITEM_NUMBER,
    _HAS_CONCEPT_NAME,
    _HAS_VIOLATION_TYPE,
    _HAS_CIK,
    _PRECEDES,
    _UNIFIED_COMPANY_TYPE,
    _concept_name,
)
from spark_jobs.graph.tables import count_edge_types, literals, resolve_edges
from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.utils.rdf_utils import (
    SEC_FILINGS,
    classify_edge_origin,
    identifier_namespace,
)

# The imported _SEC_*_NS constants are the TERM namespaces -- they name the
# classes and properties. Filings are things, and the mappers put them under
# id/sec/filings/, so entity URIs are built from this instead.
#
# Building entities on the term namespace passed only while the linker's own
# filtering shared that conflation; against real data the URIs it produced
# (ontology/sec/filings/Form4_001) do not occur.
_SEC_FILINGS_ID = identifier_namespace(_SEC_FILINGS_NS)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# Tier 2 smokes
# ======================================================================

def test_sec_enrich_tags_violation_type_by_uri_keyword(spark, make_triples):
    """A filing URI containing "Form4" -> hasViolationType.

    Was written against an administrative proceeding and the securities_fraud
    pattern. That feed is no longer collected, and SEC_VIOLATION_PATTERNS keys
    its keywords BY dataset -- securities_fraud only ever had admin and
    litigation keywords, so with those gone it can match nothing. insider_trading
    is one of the two patterns that carries filings keywords, which is what this
    matching can still reach.
    """
    entity = f"{_SEC_FILINGS_ID}Form4_Filing_001"
    rows = [(entity, _RDF_TYPE, _FORM4_TYPE)]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    violation_uri = str(SEC_VIOLATION_PATTERNS["insider_trading"]["violation_uri"])
    assert (entity, _HAS_VIOLATION_TYPE, violation_uri) in triples


def test_sec_enrich_empty_for_non_sec_data(spark, make_triples):
    rows = [(EQUITY_SNAPSHOT_TYPE, MARKET_RDF_TYPE, EQUITY_SNAPSHOT_TYPE)]
    result = SECIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# Deep: CIK-based company unification
# ======================================================================

def test_cik_unification_links_entities_sharing_a_cik(spark, make_triples):
    """
    Two filings stating the same issuer CIK produce one unified company with a
    UnifiedCompany type, hasCik, and an owl:sameAs to each source entity.

    Was written across filings and administrative proceedings, on the belief
    that unification needs two DATASETS. It does not -- it groups by distinct
    entities per CIK -- and with filings the only collected SEC feed, the
    two-dataset reading would make this method look dead when an issuer filing
    twice still unifies.
    """
    cik = "0000320193"
    first = f"{_SEC_FILINGS_ID}Form4_001"
    second = f"{_SEC_FILINGS_ID}Form4_002"
    rows = [
        (first, _RDF_TYPE, _FORM4_TYPE),
        (first, _FILINGS_HAS_ISSUER_CIK, cik),
        (second, _RDF_TYPE, _FORM4_TYPE),
        (second, _FILINGS_HAS_ISSUER_CIK, cik),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    unified = f"{_UNIFIED_NS}Company_{cik}"
    assert (unified, _RDF_TYPE, _UNIFIED_COMPANY_TYPE) in triples
    assert (unified, _HAS_CIK, cik) in triples
    assert (unified, _OWL_SAME_AS, first) in triples
    assert (unified, _OWL_SAME_AS, second) in triples


def test_cik_in_single_dataset_is_not_unified(spark, make_triples):
    """A CIK seen in only one dataset must NOT create a unified company."""
    cik = "0000320193"
    filing = f"{_SEC_FILINGS_ID}Form4_001"
    rows = [
        (filing, _RDF_TYPE, _FORM4_TYPE),
        (filing, _FILINGS_HAS_ISSUER_CIK, cik),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert not any(o == _UNIFIED_COMPANY_TYPE for _, _, o in triples)


# ======================================================================
# Deep: transaction-level sequencing
# ======================================================================
#
# The filing-level steps say which FORM came after which. They cannot say that
# a director bought on the 3rd and sold on the 11th, because the transactions
# inside one Form 4 are siblings with no order. These pin the three decisions
# _link_transaction_sequences makes -- group by owner, split by class, break
# ties on document order -- since each is a different graph and only the
# docstring otherwise records which one was built.

def _owner(cik):
    return f"{_SEC_FILINGS_ID}ReportingOwner_{cik}"


def _filing(accession):
    return f"{_SEC_FILINGS_ID}{accession}_Filing"


def _transaction(accession, kind, index):
    return f"{_SEC_FILINGS_ID}{accession}_{kind}Transaction_{index}"


def _ownership_filing(accession, owner_cik):
    """The triples every transaction reaches its owner through."""
    filing = _filing(accession)
    return [
        (filing, _RDF_TYPE, _FORM4_TYPE),
        (filing, _FILINGS_HAS_REPORTING_OWNER, _owner(owner_cik)),
    ]


def _reports(accession, kind, index, date):
    """A filing reporting one transaction on a date."""
    transaction = _transaction(accession, kind, index)
    pointer = (
        _FILINGS_HAS_NON_DERIVATIVE_TRANSACTION if kind == "NonDerivative"
        else _FILINGS_HAS_DERIVATIVE_TRANSACTION
    )
    klass = (
        _NON_DERIVATIVE_TRANSACTION_TYPE if kind == "NonDerivative"
        else _DERIVATIVE_TRANSACTION_TYPE
    )
    return [
        (_filing(accession), pointer, transaction),
        (transaction, _RDF_TYPE, klass),
        (transaction, _FILINGS_HAS_TRANSACTION_DATE, date),
    ]


def _precedes_pairs(triples):
    return {(s, o) for s, p, o in triples if p == _PRECEDES}


def test_transactions_chain_by_owner_across_filings_and_break_ties(
    spark, make_triples
):
    """One owner's non-derivative transactions form one dated chain.

    Covers all three decisions at once, because they are only separable by the
    edges they produce:

      - grouping by OWNER, not by filing: the chain runs from a transaction in
        one Form 4 into a transaction in the next, which filing-grouping cannot
        produce;
      - ordering by DATE, not by document: the 08-10 transaction reported third
        in its filing still comes first;
      - breaking a tie on DOCUMENT ORDER: _0 and _1 share 2026-08-12, and
        without the tie-break their relative order is whatever the shuffle
        produced on the day, which is not reproducible.

    N dated transactions in a group give N-1 edges, so the count is asserted as
    a set rather than by membership: an extra edge is as wrong as a missing one.
    """
    rows = [
        *_ownership_filing("0000000001-26-000001", "0000000001"),
        *_reports("0000000001-26-000001", "NonDerivative", 0, "2026-08-12"),
        *_reports("0000000001-26-000001", "NonDerivative", 1, "2026-08-12"),
        *_reports("0000000001-26-000001", "NonDerivative", 2, "2026-08-10"),
        *_ownership_filing("0000000001-26-000002", "0000000001"),
        *_reports("0000000001-26-000002", "NonDerivative", 0, "2026-08-11"),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    first = _transaction("0000000001-26-000001", "NonDerivative", 2)   # 08-10
    second = _transaction("0000000001-26-000002", "NonDerivative", 0)  # 08-11
    third = _transaction("0000000001-26-000001", "NonDerivative", 0)   # 08-12
    fourth = _transaction("0000000001-26-000001", "NonDerivative", 1)  # 08-12

    assert _precedes_pairs(triples) == {
        (first, second), (second, third), (third, fourth),
    }


def test_derivative_and_non_derivative_do_not_chain_together(spark, make_triples):
    """Two instruments in two tables of one form are two chains, not one.

    Both classes here are dated 2026-08-12 and belong to the same owner, so a
    single merged chain would link them in some order. Splitting on class
    leaves each transaction alone in its group and produces no edge at all.
    """
    rows = [
        *_ownership_filing("0000000001-26-000001", "0000000001"),
        *_reports("0000000001-26-000001", "NonDerivative", 0, "2026-08-12"),
        *_reports("0000000001-26-000001", "Derivative", 0, "2026-08-12"),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert _precedes_pairs(triples) == set()


def test_two_owners_transactions_do_not_chain_together(spark, make_triples):
    """The chain is one owner's history, not the day's tape.

    Without the owner in the partition key every transaction in the load would
    fall into one global chain, which says a director at one company sold just
    after a director at another bought.
    """
    rows = [
        *_ownership_filing("0000000001-26-000001", "0000000001"),
        *_reports("0000000001-26-000001", "NonDerivative", 0, "2026-08-12"),
        *_ownership_filing("0000000002-26-000001", "0000000002"),
        *_reports("0000000002-26-000001", "NonDerivative", 0, "2026-08-13"),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert _precedes_pairs(triples) == set()


def test_a_transaction_with_no_reachable_owner_is_not_chained(spark, make_triples):
    """The owner join is required, and its absence must not fake a chain.

    A transaction states neither its filing nor its owner, so if the filing
    does not name a reporting owner there is no group to sequence within.
    Grouping such transactions together -- by treating the missing owner as one
    shared key -- would chain unrelated filers, which is worse than the
    unordered bag this issue set out to fix.
    """
    rows = [
        (_filing("0000000001-26-000001"), _RDF_TYPE, _FORM4_TYPE),
        *_reports("0000000001-26-000001", "NonDerivative", 0, "2026-08-12"),
        (_filing("0000000002-26-000001"), _RDF_TYPE, _FORM4_TYPE),
        *_reports("0000000002-26-000001", "NonDerivative", 0, "2026-08-13"),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert _precedes_pairs(triples) == set()


def test_a_transaction_dated_twice_does_not_precede_itself(spark, make_triples):
    """A transaction stating two dates must not precede its own copy (#360).

    The date is reached by a join on the transaction, so two date triples put
    it in its owner's chain twice. Appending the entity URI to the sort keys
    gives the chain a total order but still sorts the copies next to each
    other, which is what let lead() return the transaction to itself.
    """
    rows = [
        *_ownership_filing("0000000001-26-000001", "0000000001"),
        *_reports("0000000001-26-000001", "NonDerivative", 0, "2026-08-10"),
        *_ownership_filing("0000000001-26-000002", "0000000001"),
        *_reports("0000000001-26-000002", "NonDerivative", 0, "2026-08-20"),
    ]
    dup = _transaction("0000000001-26-000001", "NonDerivative", 0)
    later = _transaction("0000000001-26-000002", "NonDerivative", 0)
    rows.append((dup, _FILINGS_HAS_TRANSACTION_DATE, "2026-08-13"))

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    pairs = _precedes_pairs(triples)
    assert (dup, dup) not in pairs
    assert pairs == {(dup, later)}


# ======================================================================
# The codes filings link to: 8-K items and XBRL concepts
# ======================================================================

_SEC_FILING_TYPE = str(SEC_FILINGS.SECFiling)
_XBRL_FACT_TYPE = str(SEC_FILINGS.XbrlFact)
_HAS_XBRL_FACT = str(SEC_FILINGS.hasXbrlFact)

REVENUES_2025 = "http://fasb.org/us-gaap/2025#Revenues"
REVENUES_2026 = "http://fasb.org/us-gaap/2026#Revenues"
TRADING_SYMBOL = "http://xbrl.sec.gov/dei/2026#TradingSymbol"
COMPANY_CONCEPT = "http://www.acme.com/20260930#WidgetRevenue"

ITEM_LINK = ("filings_SECFiling", "filings_hasForm8KItem", "sec_codes_Form8KItem")
CONCEPT_LINK = ("filings_XbrlFact", "filings_hasConcept", "sec_codes_XbrlConcept")


def _item(number):
    return f"{_SEC_FILINGS_ID}Form8KItem_{number}"


def _form8k(accession, *items):
    """An 8-K, typed as upstream types one, and the items it reports."""
    filing = _filing(accession)
    return [
        (filing, _RDF_TYPE, _FORM8K_TYPE),
        (filing, _RDF_TYPE, _SEC_FILING_TYPE),
        *((filing, _FILINGS_HAS_FORM8K_ITEM, _item(number)) for number in items),
    ]


def _form10q(accession, *concepts):
    """A 10-Q with one XBRL fact for each concept."""
    filing = _filing(accession)
    rows = [
        (filing, _RDF_TYPE, _FORM10Q_TYPE),
        (filing, _RDF_TYPE, _SEC_FILING_TYPE),
    ]
    for i, concept in enumerate(concepts):
        fact = f"{_SEC_FILINGS_ID}{accession}_Fact_{i}"
        rows += [
            (filing, _HAS_XBRL_FACT, fact),
            (fact, _RDF_TYPE, _XBRL_FACT_TYPE),
            (fact, _FILINGS_HAS_CONCEPT, concept),
        ]
    return rows


def _enriched(spark, triples):
    """The day's triples with the linker's new ones added, as a run has them."""
    return triples.unionByName(SECIntraSourceLinker(spark).enrich(triples))


def _mapped(spark, triples):
    """Node counts and edge type counts, as node and edge mapping make them."""
    node_id_df, node_counts = NodeMapper(spark, {}).build_node_id_table(triples)
    edge_counts = {
        (row["src_type"], row["relation"], row["dst_type"]): row["count"]
        for row in count_edge_types(resolve_edges(triples, node_id_df))
    }
    return node_counts, edge_counts


def test_each_item_an_8k_reports_becomes_one_typed_node(spark, make_triples):
    """Two 8-Ks report item 2.02 and one of them 9.01 as well: two item nodes,
    each typed once, with its number and its label."""
    rows = [
        *_form8k("0000000001-26-000001", "2.02"),
        *_form8k("0000000002-26-000001", "2.02", "9.01"),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    items = {_item("2.02"), _item("9.01")}
    assert {t for t in triples if t[0] in items} == {
        (_item("2.02"), _RDF_TYPE, _FORM8K_ITEM_TYPE),
        (_item("2.02"), _HAS_ITEM_NUMBER, "2.02"),
        (_item("2.02"), _RDFS_LABEL, "Item 2.02"),
        (_item("9.01"), _RDF_TYPE, _FORM8K_ITEM_TYPE),
        (_item("9.01"), _HAS_ITEM_NUMBER, "9.01"),
        (_item("9.01"), _RDFS_LABEL, "Item 9.01"),
    }


def test_each_concept_becomes_a_node_named_without_its_year(spark, make_triples):
    """Four concepts, four nodes. Revenues under two taxonomy years is two
    nodes with one name, and a company's own concept is ext:<Name>."""
    rows = _form10q(
        "0000000001-26-000001",
        REVENUES_2025, REVENUES_2026, TRADING_SYMBOL, COMPANY_CONCEPT,
    )

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    names = {s: o for s, p, o in triples if p == _HAS_CONCEPT_NAME}
    assert names == {
        REVENUES_2025: "us-gaap:Revenues",
        REVENUES_2026: "us-gaap:Revenues",
        TRADING_SYMBOL: "dei:TradingSymbol",
        COMPANY_CONCEPT: "ext:WidgetRevenue",
    }
    typed = {s for s, p, o in triples if (p, o) == (_RDF_TYPE, _XBRL_CONCEPT_TYPE)}
    assert typed == set(names)
    assert {s: o for s, p, o in triples if p == _RDFS_LABEL and s in names} == names


def test_a_concept_name_follows_each_publishers_uri_shape(spark):
    """The other shapes on 2026-10-22's filings. An SEC version is not always a
    year (ffd/2025q4), and IFRS puts its date before the prefix."""
    names = {
        "http://fasb.org/srt/2026#MortgageLoansOnRealEstateInterestRate":
            "srt:MortgageLoansOnRealEstateInterestRate",
        "http://xbrl.sec.gov/ecd/2026#PeoActuallyPaidCompAmt":
            "ecd:PeoActuallyPaidCompAmt",
        "http://xbrl.sec.gov/ffd/2025q4#AmtSctiesRegd": "ffd:AmtSctiesRegd",
        "https://xbrl.ifrs.org/taxonomy/2025-03-27/ifrs-full#Equity":
            "ifrs-full:Equity",
        "http://kbgloballtd.com/20260630#AdvancesFromRelatedParties":
            "ext:AdvancesFromRelatedParties",
    }
    frame = spark.createDataFrame([(uri,) for uri in names], "uri string")

    read = {
        row["uri"]: row["name"]
        for row in frame.select(
            "uri", _concept_name(F.col("uri")).alias("name")
        ).collect()
    }

    assert read == names


def test_both_links_resolve_into_edges_once_their_ends_are_typed(
    spark, make_triples
):
    """One edge per link after node and edge mapping, and each link reads as
    reported by the filings rather than made by this pipeline."""
    rows = [
        *_form8k("0000000001-26-000001", "2.02"),
        *_form8k("0000000002-26-000001", "2.02", "9.01"),
        *_form10q(
            "0000000003-26-000001",
            REVENUES_2025, REVENUES_2026, TRADING_SYMBOL, COMPANY_CONCEPT,
        ),
    ]

    node_counts, edge_counts = _mapped(spark, _enriched(spark, make_triples(rows)))

    assert edge_counts[ITEM_LINK] == 3
    assert edge_counts[CONCEPT_LINK] == 4
    assert node_counts["sec_codes_Form8KItem"] == 2
    assert node_counts["sec_codes_XbrlConcept"] == 4
    for predicate, (src_type, _relation, dst_type) in (
        (_FILINGS_HAS_FORM8K_ITEM, ITEM_LINK),
        (_FILINGS_HAS_CONCEPT, CONCEPT_LINK),
    ):
        assert classify_edge_origin(predicate, src_type, dst_type) == "raw"


def test_item_numbers_and_concept_names_are_labels_in_facts(spark, make_triples):
    """2.02 parses, and is still a label: is_numeric is false for every code
    the step writes. The .pt's side is in test_feature_extractor's CODE_VALUES."""
    rows = [
        *_form8k("0000000001-26-000001", "2.02", "9.01"),
        *_form10q("0000000002-26-000001", REVENUES_2026),
    ]
    triples = _enriched(spark, make_triples(rows))
    node_id_df, _counts = NodeMapper(spark, {}).build_node_id_table(triples)

    facts = {
        (row["uri"], row["predicate_name"]): (row["value"], row["is_numeric"])
        for row in literals(triples, node_id_df).collect()
    }

    assert facts[(_item("2.02"), "sec_codes_hasItemNumber")] == ("2.02", False)
    assert facts[(_item("9.01"), "sec_codes_hasItemNumber")] == ("9.01", False)
    assert facts[(REVENUES_2026, "sec_codes_hasConceptName")] == (
        "us-gaap:Revenues", False,
    )
    assert facts[(_item("2.02"), "rdfs_label")] == ("Item 2.02", False)


def test_a_day_with_neither_link_adds_nothing_and_says_so(
    spark, make_triples, caplog
):
    """An 8-K reporting no item and a 10-Q with no XBRL fact: nothing is typed,
    and the log says the step made nothing."""
    rows = [*_form8k("0000000001-26-000001"), *_form10q("0000000002-26-000001")]

    with caplog.at_level(
        logging.WARNING, logger="spark_jobs.enrichment.intra_source.sec_linker"
    ):
        triples = _triple_set(
            SECIntraSourceLinker(spark).enrich(make_triples(rows))
        )

    assert not {
        t for t in triples
        if t[1] in (_HAS_ITEM_NUMBER, _HAS_CONCEPT_NAME)
        or t[2] in (_FORM8K_ITEM_TYPE, _XBRL_CONCEPT_TYPE)
    }
    assert "[Step 5/5] produced no triples" in caplog.text


def test_no_other_predicates_objects_are_typed(spark, make_triples):
    """The step keys on the two predicates, not on what a URI looks like: an
    item's URI and a concept's under other predicates stay untyped, as does a
    link to anything else."""
    filing = _filing("0000000001-26-000001")
    fact = f"{_SEC_FILINGS_ID}0000000002-26-000001_Fact_0"
    other_item = _item("7.01")
    other_concept = "http://fasb.org/us-gaap/2026#NetIncomeLoss"
    exhibit = f"{_SEC_FILINGS_ID}0000000001-26-000001_Exhibit_99.1"
    rows = [
        *_form8k("0000000001-26-000001", "2.02"),
        *_form10q("0000000002-26-000001", REVENUES_2026),
        (filing, f"{_SEC_FILINGS_NS}hasRelatedItem", other_item),
        (fact, f"{_SEC_FILINGS_NS}hasRelatedConcept", other_concept),
        (filing, f"{_SEC_FILINGS_NS}hasExhibit", exhibit),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    subjects = {s for s, _p, _o in triples}
    assert {_item("2.02"), REVENUES_2026} <= subjects
    assert not {other_item, other_concept, exhibit} & subjects


def test_typing_the_codes_moves_no_other_node_or_edge_count(
    spark, make_triples, monkeypatch
):
    """The same day with the step on and off: the item and concept nodes, and
    the links to them, are the only difference. The day also has a unified
    company and a transaction chain, so the other steps have output to keep."""
    rows = [
        *_form8k("0000000001-26-000001", "2.02"),
        *_form8k("0000000002-26-000001", "2.02", "9.01"),
        *_form10q("0000000003-26-000001", REVENUES_2026, COMPANY_CONCEPT),
        *_ownership_filing("0000000004-26-000001", "0000000001"),
        *_reports("0000000004-26-000001", "NonDerivative", 0, "2026-10-20"),
        *_reports("0000000004-26-000001", "NonDerivative", 1, "2026-10-21"),
        (_filing("0000000001-26-000001"), _FILINGS_HAS_ISSUER_CIK, "0000320193"),
        (_filing("0000000004-26-000001"), _FILINGS_HAS_ISSUER_CIK, "0000320193"),
    ]
    triples = make_triples(rows)
    nodes_on, edges_on = _mapped(spark, _enriched(spark, triples))

    monkeypatch.setattr(
        SECIntraSourceLinker, "_type_linked_codes", lambda self: None
    )
    nodes_off, edges_off = _mapped(spark, _enriched(spark, triples))

    new_types = {"sec_codes_Form8KItem", "sec_codes_XbrlConcept"}
    assert set(nodes_on) - set(nodes_off) == new_types
    assert {t: n for t, n in nodes_on.items() if t not in new_types} == nodes_off
    assert {e for e in edges_on if e[2] in new_types} == {ITEM_LINK, CONCEPT_LINK}
    assert {e: n for e, n in edges_on.items() if e[2] not in new_types} == edges_off
    # The other steps did run: the unified company and the chain are there.
    assert ("sec_enrichment_UnifiedCompany", "owl_sameAs", "filings_SECFiling") in edges_off
    assert any(relation == "sec_enrichment_precedes" for _s, relation, _d in edges_off)
