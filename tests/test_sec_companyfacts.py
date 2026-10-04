"""One row of the SEC companyfacts snapshot, through the graph (#434).

A row is a CompanyFact under ontology/sec/companyfacts/, pointing at the
filings vocabulary's own Issuer_{cik}, which the row restates with its type
and hasIssuerCik. These pin what reading the feed buys: the fact is a node of
its own type with an edge to its issuer, the issuer reaches its company node
on a day it filed nothing, and the fact's two dates reach the period spine.

Drives the mappers and linkers over tiny in-memory triples on the shared local
SparkSession (`spark` / `make_triples` fixtures from conftest.py).
"""
from spark_jobs.enrichment.cross_source_linker import CrossSourceLinker
from spark_jobs.enrichment.temporal_unifier import (
    OBSERVED_IN_PERIOD_PRED,
    TemporalUnifier,
)
from spark_jobs.pyg_builder.edge_mapper import EdgeMapper
from spark_jobs.pyg_builder.naming import RDF_TYPE
from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT,
    MARKET_QUOTES,
    SEC_COMPANYFACTS,
    SEC_FILINGS,
    SYNTHETIC_TEMPORAL_IDS,
    UNIFIED,
    identifier_namespace,
)

FACTS_ID = identifier_namespace(str(SEC_COMPANYFACTS))
FILINGS_ID = identifier_namespace(str(SEC_FILINGS))

CIK = "0000320193"
FACT = f"{FACTS_ID}Fact_0000320193_revenueQuarter_2026-06-30"
ISSUER = f"{FILINGS_ID}Issuer_{CIK}"


def _snapshot_row():
    """A fact as the snapshot states it, its issuer restated beside it."""
    return [
        (FACT, RDF_TYPE, str(SEC_COMPANYFACTS.CompanyFact)),
        (FACT, str(SEC_COMPANYFACTS.revenueQuarter), "94036000000"),
        (FACT, str(SEC_COMPANYFACTS.periodEnd), "2026-06-30"),
        (FACT, str(SEC_COMPANYFACTS.filedOn), "2026-08-01"),
        (FACT, str(SEC_COMPANYFACTS.aboutIssuer), ISSUER),
        (ISSUER, RDF_TYPE, str(SEC_FILINGS.Issuer)),
        (ISSUER, str(SEC_FILINGS.hasIssuerCik), CIK),
    ]


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


def test_a_fact_is_a_node_with_an_edge_to_its_issuer(spark, make_triples):
    """Before SEC registered the vocabulary, its node type failed the build."""
    triples = make_triples(_snapshot_row())

    node_mapper = NodeMapper(spark, {})
    node_id_df, counts = node_mapper.build_node_id_table(triples)
    assert counts == {"companyfacts_CompanyFact": 1, "filings_Issuer": 1}

    edge_indices, _ = EdgeMapper(spark, {}).build_edge_indices(
        triples, node_id_df, counts
    )
    assert (
        "companyfacts_CompanyFact", "companyfacts_aboutIssuer", "filings_Issuer"
    ) in edge_indices


def test_the_issuer_reaches_its_company_on_a_day_it_filed_nothing(
    spark, make_triples,
):
    """The snapshot names every company every day. A filing names one only
    on the days it files. The quote meets the fact's issuer at the same company,
    through the constituents CSV's ticker pairing."""
    snapshot = str(MARKET_QUOTES) + "snapshot/AAPL/2026-10-01"
    rows = _snapshot_row() + [
        (snapshot, RDF_TYPE, str(MARKET_QUOTES.EquitySnapshot)),
        (snapshot, str(MARKET_QUOTES.symbol), "AAPL"),
    ]

    triples = _triple_set(CrossSourceLinker(
        spark, make_triples(rows), ticker_cik_map={"AAPL": CIK},
    ).enrich())

    company = f"{UNIFIED}Company_{CIK}"
    assert (ISSUER, str(BLS_ENRICHMENT.refersToCompany), company) in triples
    assert (snapshot, str(BLS_ENRICHMENT.refersToCompany), company) in triples


def test_a_fact_reaches_the_day_it_is_about_and_the_day_it_was_filed(
    spark, make_triples,
):
    triples = _triple_set(
        TemporalUnifier(spark).enrich(make_triples(_snapshot_row()))
    )

    sec = SYNTHETIC_TEMPORAL_IDS["sec"]
    assert (FACT, OBSERVED_IN_PERIOD_PRED, sec + "2026-06-30") in triples
    assert (FACT, OBSERVED_IN_PERIOD_PRED, sec + "2026-08-01") in triples
