"""Value-level unit tests for EdgeFeatureExtractor encoding
(spark_jobs/pyg_builder/edge_feature_extractor.py).

Previously only EdgeVectorLayout's geometry was tested — never the encoding of a
known edge into its reserved segments. These tests cover:

  * _classify_relation — pure function, one case per category (incl. skip
    precedence and the generic fallback);
  * _encode_temporal_signals — period flags, direction sign, and time-delta on a
    known temporal edge;
  * _encode_numeric_contrast — difference / ratio / magnitude of shared numeric
    endpoint properties (asserted via disjoint sub-segment sums, so hash
    collisions within a sub-segment don't matter);
  * _encode_cross_property_contrast — option_stock moneyness / log-moneyness /
    price-difference on endpoints with no shared predicate;
  * _encode_relational_context — same-namespace / cross-source flags;
  * determinism — encoding the same edge twice yields identical vectors.

Edges are driven through the real NodeMapper → EdgeMapper → EdgeFeatureExtractor
path on the shared local SparkSession, so edge_idx alignment is exercised too.
"""
import math

import pytest
import torch
from pyspark.sql import functions as F

from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.pyg_builder.edge_mapper import EdgeMapper
from spark_jobs.pyg_builder.edge_feature_extractor import (
    EdgeFeatureExtractor,
    _classify_relation,
)

CPI_INDEX = "https://www.bls.gov/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://www.bls.gov/cpi/Series"      # -> cpi_Series
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PRECEDES = "https://www.bls.gov/enrichment/precedes"   # temporal
PRECEDES_REL = "bls_enrichment_precedes"
HAS_UNDERLYING = "https://example.org/hasUnderlying"   # option_stock
HAS_UNDERLYING_REL = "unknown_hasUnderlying"

HAS_MONTH = "https://example.org/hasMonth"
HAS_YEAR = "https://example.org/hasYear"
STRIKE = "https://example.org/strikePrice"
PRICE = "https://example.org/observedPrice"

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
CORRELATES = "https://example.org/correlatesWith"      # correlation
CORRELATES_REL = "unknown_correlatesWith"
ESCALATES = "https://example.org/escalatesTo"          # escalation
ESCALATES_REL = "unknown_escalatesTo"
SEVERITY_SRC = "https://example.org/severityNum"       # src-only severity
SEVERITY_DST = "https://example.org/severityLevel"     # dst-only severity

# Correlation is NOT in the default enabled_categories — label similarity
# only runs when it's switched on. Pinned here so a config regression that
# silently drops the category fails a value test, not just e2e.
CORR_CONFIG = {"edge_feature_config": {"enabled_categories": ["correlation"]}}


def _edge_features(spark, rows, config=None):
    """Run the real edge-feature path; return (features, edge_indices, layout)."""
    cfg = config or {}
    triples = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    node_id_df, counts = NodeMapper(spark, cfg).build_node_id_table(triples)
    edge_indices, edges_final = EdgeMapper(spark, cfg).build_edge_indices(
        triples, node_id_df, counts
    )
    efe = EdgeFeatureExtractor(spark, cfg)
    feats = efe.build_edge_features(
        triples, node_id_df, edges_final, edge_indices
    )
    return feats, edge_indices, efe.get_layout()


def _subsum(row, start, dim):
    return float(row[start:start + dim].sum())


# ======================================================================
# _classify_relation — pure function, one case per category
# ======================================================================

@pytest.mark.parametrize("relation,expected", [
    ("bls_enrichment_precedes", "temporal"),
    ("market_hasNext", "temporal"),
    ("unknown_hasUnderlying", "option_stock"),
    ("noaa_escalatesTo", "escalation"),
    ("x_correlatesWith", "correlation"),
    ("x_relatedTo", "correlation"),
    ("x_leadsTo", "causal"),
    ("x_impacts", "causal"),
    ("x_straddleWith", "strategy"),
    ("x_belongsToSector", "skip"),
    ("company_sameAs", "skip"),
    ("some_random_relation", "generic"),
])
def test_classify_relation(relation, expected):
    assert _classify_relation(relation) == expected


def test_classify_relation_skip_takes_precedence():
    # A relation matching both a skip fragment and a temporal fragment is
    # classified skip — skip is checked first so structural edges never get
    # features.
    assert _classify_relation("sameAs_precedes") == "skip"


# ======================================================================
# _encode_temporal_signals + _encode_numeric_contrast on a known temporal edge
# ======================================================================

def _temporal_rows(src_month, dst_month, year=2020):
    return [
        ("https://ex/src", RDF_TYPE, CPI_INDEX),
        ("https://ex/dst", RDF_TYPE, CPI_SERIES),
        ("https://ex/src", HAS_MONTH, str(src_month)),
        ("https://ex/src", HAS_YEAR, str(year)),
        ("https://ex/dst", HAS_MONTH, str(dst_month)),
        ("https://ex/dst", HAS_YEAR, str(year)),
        ("https://ex/src", PRECEDES, "https://ex/dst"),
    ]


def test_temporal_edge_encodes_expected_segments(spark):
    # src month 1, dst month 2, same year → 1-month forward step.
    feats, _ei, L = _edge_features(spark, _temporal_rows(1, 2))
    key = ("cpi_Index", PRECEDES_REL, "cpi_Series")
    attr = feats[key].numpy()
    assert attr.shape == (1, L.edge_vector_dim)
    row = attr[0]

    # --- Segment 1: temporal signals ---
    # Same year → same-year flag set.
    assert row[L.seg1_period_flags_start] == pytest.approx(1.0)
    # 1 month apart → consecutive-month flag set.
    if L.seg1_period_flags_dim >= 2:
        assert row[L.seg1_period_flags_start + 1] == pytest.approx(1.0)
    # Months 1 and 2 fall in the same quarter of the same year.
    if L.seg1_period_flags_dim >= 3:
        assert row[L.seg1_period_flags_start + 2] == pytest.approx(1.0)
    # dst is later → forward direction +1.
    assert row[L.seg1_direction_start] == pytest.approx(1.0)
    # Time-delta sub-segment holds norm_delta (1/12) + |norm_delta| (1/12).
    assert _subsum(row, L.seg1_time_delta_start, L.seg1_time_delta_dim) == (
        pytest.approx(2.0 / 12.0, abs=1e-5)
    )

    # --- Segment 2: numeric contrast over shared predicates (hasMonth, hasYear) ---
    # difference = (2-1) + (2020-2020) = 1
    assert _subsum(row, L.seg2_difference_start, L.seg2_difference_dim) == (
        pytest.approx(1.0, abs=1e-4)
    )
    # ratio = 2/1 + 2020/2020 = 3
    assert _subsum(row, L.seg2_ratio_start, L.seg2_ratio_dim) == (
        pytest.approx(3.0, abs=1e-4)
    )
    # magnitude = (1+2)/2 + (2020+2020)/2 = 1.5 + 2020 = 2021.5
    assert _subsum(row, L.seg2_magnitude_start, L.seg2_magnitude_dim) == (
        pytest.approx(2021.5, rel=1e-5)
    )

    # --- Segment 3: relational context — both endpoints are cpi_* ---
    assert row[L.seg3_namespace_start] == pytest.approx(1.0)          # same ns
    if L.seg3_namespace_dim >= 2:
        assert row[L.seg3_namespace_start + 1] == pytest.approx(0.0)  # cross-src

    # No NaNs/infs anywhere.
    assert bool(torch.isfinite(feats[key]).all())


def test_temporal_direction_sign_reverses_when_dst_earlier(spark):
    # src month 3, dst month 1 → dst is earlier → backward direction -1,
    # and not exactly one month apart → consecutive flag clear.
    feats, _ei, L = _edge_features(spark, _temporal_rows(3, 1))
    row = feats[("cpi_Index", PRECEDES_REL, "cpi_Series")].numpy()[0]
    assert row[L.seg1_direction_start] == pytest.approx(-1.0)
    if L.seg1_period_flags_dim >= 2:
        assert row[L.seg1_period_flags_start + 1] == pytest.approx(0.0)


# ======================================================================
# _encode_cross_property_contrast — option_stock moneyness
# ======================================================================

def test_option_stock_cross_property_moneyness(spark):
    # No shared predicate between endpoints → numeric contrast falls back to
    # cross-property moneyness derivation. strike 110 / stock 100.
    feats, _ei, L = _edge_features(spark, [
        ("https://ex/opt", RDF_TYPE, CPI_INDEX),
        ("https://ex/stk", RDF_TYPE, CPI_SERIES),
        ("https://ex/opt", STRIKE, "110"),
        ("https://ex/stk", PRICE, "100"),
        ("https://ex/opt", HAS_UNDERLYING, "https://ex/stk"),
    ])
    key = ("cpi_Index", HAS_UNDERLYING_REL, "cpi_Series")
    row = feats[key].numpy()[0]

    # moneyness = strike / stock = 1.1 (difference sub-segment)
    assert _subsum(row, L.seg2_difference_start, L.seg2_difference_dim) == (
        pytest.approx(1.1, abs=1e-4)
    )
    # log-moneyness = ln(1.1) (ratio sub-segment)
    assert _subsum(row, L.seg2_ratio_start, L.seg2_ratio_dim) == (
        pytest.approx(math.log(1.1), abs=1e-4)
    )
    # strike - stock = 10 (magnitude sub-segment)
    assert _subsum(row, L.seg2_magnitude_start, L.seg2_magnitude_dim) == (
        pytest.approx(10.0, abs=1e-4)
    )


# ======================================================================
# _encode_relational_context — label similarity (correlation edges)
# ======================================================================

def _correlation_rows(src_label, dst_label):
    return [
        ("https://ex/src", RDF_TYPE, CPI_INDEX),
        ("https://ex/dst", RDF_TYPE, CPI_SERIES),
        ("https://ex/src", RDFS_LABEL, src_label),
        ("https://ex/dst", RDFS_LABEL, dst_label),
        ("https://ex/src", CORRELATES, "https://ex/dst"),
    ]


def test_correlation_label_similarity_overlapping_labels(spark):
    # "Energy Prices" / "Energy Stocks" (lowercased by the extractor):
    # Jaccard = |{energy}| / |{energy, prices, stocks}| = 1/3.
    feats, _ei, L = _edge_features(
        spark, _correlation_rows("Energy Prices", "Energy Stocks"),
        config=CORR_CONFIG,
    )
    row = feats[("cpi_Index", CORRELATES_REL, "cpi_Series")].numpy()[0]

    ls_start, ls_dim = (
        L.seg3_label_similarity_start, L.seg3_label_similarity_dim
    )
    # The similarity value sits in the first label-similarity slot...
    assert row[ls_start] == pytest.approx(1.0 / 3.0, abs=1e-5)
    # ...and the label-pair hash slot repeats it, so the sub-segment sums
    # to exactly twice the similarity (hash slot is in [ls_start+1, end)).
    if ls_dim >= 2:
        assert _subsum(row, ls_start, ls_dim) == pytest.approx(
            2.0 / 3.0, abs=1e-5
        )


def test_correlation_label_similarity_disjoint_labels(spark):
    # No shared words → similarity 0 → the entire sub-segment stays zero.
    feats, _ei, L = _edge_features(
        spark, _correlation_rows("Energy Prices", "Wheat Futures"),
        config=CORR_CONFIG,
    )
    row = feats[("cpi_Index", CORRELATES_REL, "cpi_Series")].numpy()[0]
    assert _subsum(
        row, L.seg3_label_similarity_start, L.seg3_label_similarity_dim
    ) == pytest.approx(0.0, abs=1e-6)


# ======================================================================
# _encode_cross_property_contrast — escalation severity delta
# ======================================================================

@pytest.mark.parametrize("src_sev,dst_sev,expected_delta", [
    ("1", "3", 2.0),     # escalation: dst more severe → positive delta
    ("3", "1", -2.0),    # de-escalation: dst less severe → negative delta
])
def test_escalation_severity_delta(spark, src_sev, dst_sev, expected_delta):
    # Endpoints carry severity under DIFFERENT predicates (severityNum vs
    # severityLevel, both matching the severity regex), so the shared-
    # predicate numeric contrast finds nothing and the cross-property
    # escalation fallback must fire.
    feats, _ei, L = _edge_features(spark, [
        ("https://ex/src", RDF_TYPE, CPI_INDEX),
        ("https://ex/dst", RDF_TYPE, CPI_SERIES),
        ("https://ex/src", SEVERITY_SRC, src_sev),
        ("https://ex/dst", SEVERITY_DST, dst_sev),
        ("https://ex/src", ESCALATES, "https://ex/dst"),
    ])
    row = feats[("cpi_Index", ESCALATES_REL, "cpi_Series")].numpy()[0]

    # severity delta = dst - src, written into the difference sub-segment
    assert _subsum(row, L.seg2_difference_start, L.seg2_difference_dim) == (
        pytest.approx(expected_delta, abs=1e-5)
    )
    # The escalation branch writes ONLY the delta — ratio and magnitude
    # sub-segments stay empty (unlike the option_stock branch).
    assert _subsum(row, L.seg2_ratio_start, L.seg2_ratio_dim) == (
        pytest.approx(0.0, abs=1e-6)
    )
    assert _subsum(row, L.seg2_magnitude_start, L.seg2_magnitude_dim) == (
        pytest.approx(0.0, abs=1e-6)
    )


# ======================================================================
# skip / classification behaviour end-to-end
# ======================================================================

def test_skip_relation_gets_no_edge_features(spark):
    # belongsToSector classifies as skip → no edge_attr for that type.
    feats, edge_indices, _L = _edge_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
        ("https://ex/a", "https://example.org/belongsToSector", "https://ex/b"),
    ])
    # The edge exists in the index, but received no features.
    assert any("belongsToSector" in rel for (_s, rel, _d) in edge_indices)
    assert feats == {}


# ======================================================================
# Determinism
# ======================================================================

def test_edge_encoding_is_deterministic_across_runs(spark):
    rows = _temporal_rows(1, 2)
    key = ("cpi_Index", PRECEDES_REL, "cpi_Series")
    first = _edge_features(spark, rows)[0][key]
    again = _edge_features(spark, rows)[0][key]
    assert torch.equal(first, again)
