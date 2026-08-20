"""
Tests for the SEC intra-source enrichment linker (SECIntraSourceLinker).

Two tiers, mirroring test_bls_linker.py:
  - Tier 2 smokes: enrich() tags a violation type by URI keyword, and foreign
    input short-circuits to zero triples.
  - Targeted deep test: CIK-based company unification
    (`_unify_company_entities`) — the "same CIK across >= 2 datasets" rule,
    including the negative case where a single-dataset CIK is NOT unified.

Imports underscore-prefixed module constants (the linker's canonical URIs) so
the tests track the code rather than hardcoding IRIs. Drives enrich() over tiny
in-memory triples on the shared local SparkSession (`spark` / `make_triples`).
"""
from spark_jobs.enrichment.intra_source.market_linker import (
    RDF_TYPE as MARKET_RDF_TYPE,
    EQUITY_SNAPSHOT_TYPE,
)
from spark_jobs.enrichment.intra_source.sec_linker import (
    SECIntraSourceLinker,
    SEC_VIOLATION_PATTERNS,
    _RDF_TYPE,
    _OWL_SAME_AS,
    _SEC_FILINGS_NS,
    _UNIFIED_NS,
    _FORM4_TYPE,
    _FILINGS_HAS_ISSUER_CIK,
    _FILINGS_HAS_REPORTING_OWNER,
    _FILINGS_HAS_NON_DERIVATIVE_TRANSACTION,
    _FILINGS_HAS_DERIVATIVE_TRANSACTION,
    _FILINGS_HAS_TRANSACTION_DATE,
    _NON_DERIVATIVE_TRANSACTION_TYPE,
    _DERIVATIVE_TRANSACTION_TYPE,
    _HAS_VIOLATION_TYPE,
    _HAS_CIK,
    _PRECEDES,
    _UNIFIED_COMPANY_TYPE,
)
from spark_jobs.utils.rdf_utils import identifier_namespace

# The imported _SEC_*_NS constants are the TERM namespaces -- they name the
# classes and properties. Filings are things, and the mappers put them under
# id/sec-filings/, so entity URIs are built from this instead.
#
# Building entities on the term namespace passed only while the linker's own
# filtering shared that conflation; against real data the URIs it produced
# (ontology/sec-filings/Form4_001) do not occur.
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
