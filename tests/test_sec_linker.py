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
    _HAS_VIOLATION_TYPE,
    _HAS_CIK,
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
