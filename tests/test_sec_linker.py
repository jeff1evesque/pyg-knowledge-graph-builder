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
    _SEC_ADMIN_NS,
    _SEC_FILINGS_NS,
    _UNIFIED_NS,
    _ADMIN_PROCEEDING_TYPE,
    _FORM4_TYPE,
    _FILINGS_HAS_ISSUER_CIK,
    _ADMIN_CIK_NUMBER,
    _HAS_VIOLATION_TYPE,
    _HAS_CIK,
    _UNIFIED_COMPANY_TYPE,
)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# Tier 2 smokes
# ======================================================================

def test_sec_enrich_tags_violation_type_by_uri_keyword(spark, make_triples):
    """Admin-proceeding URI containing "SecuritiesFraud" -> hasViolationType."""
    entity = f"{_SEC_ADMIN_NS}SecuritiesFraud_Proceeding_001"
    rows = [(entity, _RDF_TYPE, _ADMIN_PROCEEDING_TYPE)]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    violation_uri = str(SEC_VIOLATION_PATTERNS["securities_fraud"]["violation_uri"])
    assert (entity, _HAS_VIOLATION_TYPE, violation_uri) in triples


def test_sec_enrich_empty_for_non_sec_data(spark, make_triples):
    rows = [(EQUITY_SNAPSHOT_TYPE, MARKET_RDF_TYPE, EQUITY_SNAPSHOT_TYPE)]
    result = SECIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# Deep: CIK-based company unification
# ======================================================================

def test_cik_unification_links_same_cik_across_datasets(spark, make_triples):
    """
    A CIK present in BOTH filings and admin datasets produces a unified company
    with a UnifiedCompany type, hasCik, and an owl:sameAs to each source entity.
    """
    cik = "0000320193"
    filing = f"{_SEC_FILINGS_NS}Form4_001"
    proceeding = f"{_SEC_ADMIN_NS}Proceeding_001"
    rows = [
        (filing, _RDF_TYPE, _FORM4_TYPE),
        (filing, _FILINGS_HAS_ISSUER_CIK, cik),
        (proceeding, _RDF_TYPE, _ADMIN_PROCEEDING_TYPE),
        (proceeding, _ADMIN_CIK_NUMBER, cik),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    unified = f"{_UNIFIED_NS}Company_{cik}"
    assert (unified, _RDF_TYPE, _UNIFIED_COMPANY_TYPE) in triples
    assert (unified, _HAS_CIK, cik) in triples
    assert (unified, _OWL_SAME_AS, filing) in triples
    assert (unified, _OWL_SAME_AS, proceeding) in triples


def test_cik_in_single_dataset_is_not_unified(spark, make_triples):
    """A CIK seen in only one dataset must NOT create a unified company."""
    cik = "0000320193"
    filing = f"{_SEC_FILINGS_NS}Form4_001"
    rows = [
        (filing, _RDF_TYPE, _FORM4_TYPE),
        (filing, _FILINGS_HAS_ISSUER_CIK, cik),
    ]

    triples = _triple_set(SECIntraSourceLinker(spark).enrich(make_triples(rows)))

    assert not any(o == _UNIFIED_COMPANY_TYPE for _, _, o in triples)
