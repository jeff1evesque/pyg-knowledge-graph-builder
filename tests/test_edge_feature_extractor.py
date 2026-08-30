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

from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.pyg_builder.edge_mapper import EdgeMapper
from spark_jobs.pyg_builder.edge_feature_extractor import (
    EdgeFeatureExtractor,
    _CHUNK_EDGE_THRESHOLD,
    _NUMERIC_ROW_BYTES,
    _PropertyRows,
    _RESULT_SIZE_TARGET_FRACTION,
    _SPARSE_ENTRY_BYTES,
    _classify_relation,
    _parse_byte_string,
)

CPI_INDEX = "https://jefflevesque.com/ontology/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://jefflevesque.com/ontology/cpi/Series"      # -> cpi_Series
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT  # noqa: E402

# From the namespace table, not spelled out: a URI under a namespace nothing
# registers falls back to its bare last segment and silently renames the
# relation.
PRECEDES = f"{BLS_ENRICHMENT}precedes"   # temporal
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

# A relation no fragment matches -> "generic". Matches the shape of the real
# relations the linkers emit (cpi_hasArea, jolts_hasQuitsRate).
GENERIC = "https://example.org/hasThing"
GENERIC_REL = "unknown_hasThing"


def _generic_edge_rows(year=2020):
    """One edge of a single generic-category type, with numeric endpoints."""
    return [
        ("https://ex/src", RDF_TYPE, CPI_INDEX),
        ("https://ex/dst", RDF_TYPE, CPI_SERIES),
        ("https://ex/src", HAS_MONTH, "1"),
        ("https://ex/src", HAS_YEAR, str(year)),
        ("https://ex/dst", HAS_MONTH, "2"),
        ("https://ex/dst", HAS_YEAR, str(year)),
        ("https://ex/src", GENERIC, "https://ex/dst"),
    ]


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
# _classify_relation over the relations a real cluster run produced
# ======================================================================

# Every relation name below is taken verbatim from feature_spec.json's
# derivation_methods on the 2026-07-29 cluster run (SEC filings + BLS cpi/jolts,
# 704 edge types). On that run all 46 of them classified "generic", which is in
# no enabled_categories set, so not one edge type was featurized — the job still
# exited 0 and graph_schema.json honestly reported edge_types_with_features: 0.
# Pinning the real names is what makes this regression visible: the hand-written
# fragments matched the vocabulary the linkers were *designed* around, not the
# vocabulary they *emit*.
@pytest.mark.parametrize("relation,expected", [
    # The 22 per-sector correlation relations the cross-source linker emits.
    ("bls_enrichment_employmentSizeSectorCorrelation", "correlation"),
    ("bls_enrichment_energySectorCorrelation", "correlation"),
    ("bls_enrichment_geographicRegionsSectorCorrelation", "correlation"),
    ("bls_enrichment_educationCommunicationSectorCorrelation", "correlation"),
    ("bls_enrichment_miscellaneousGoodsSectorCorrelation", "correlation"),
    # Structural, and must stay excluded even though it is a sector relation.
    ("bls_enrichment_belongsToSector", "skip"),
    # Genuinely unclassified: featurizable only by opting "generic" in.
    ("bls_enrichment_goodsServicesRelation", "generic"),
    ("cpi_hasArea", "generic"),
    ("cpi_measuresInflationFor", "generic"),
    ("jolts_hasQuitsRate", "generic"),
    ("jolts_hasSeparationsLevel", "generic"),
])
def test_classify_relation_on_real_cluster_relations(relation, expected):
    assert _classify_relation(relation) == expected


def test_correlation_fragment_matches_case_insensitively():
    """The suffix is 'Correlation' but matching must not depend on case.

    The linkers camel-case it (energySectorCorrelation); a hand-written config
    or a future linker may not.
    """
    for relation in ("x_sectorcorrelation", "x_SECTORCORRELATION",
                     "x_SectorCorrelation"):
        assert _classify_relation(relation) == "correlation"


# ======================================================================
# enabled_categories validation
# ======================================================================

def test_generic_is_selectable_but_not_default(spark):
    """'generic' is a real classifier verdict, so config must be able to ask
    for it.

    Before #issue-6 it was unreachable: _classify_relation returned it, but no
    enabled_categories value could select it, so every relation the fragments
    missed was silently unfeaturizable. Asserted as a behavioural pair — the
    same rows, features only when the category is switched on.
    """
    rows = _generic_edge_rows()
    key = ("cpi_Index", GENERIC_REL, "cpi_Series")

    default_feats, _ei, _L = _edge_features(spark, rows)
    assert key not in default_feats, (
        "a generic relation must not be featurized by default"
    )

    opted_in, _ei2, layout = _edge_features(
        spark, rows,
        config={"edge_feature_config": {"enabled_categories": ["generic"]}},
    )
    assert key in opted_in, (
        "enabled_categories=['generic'] must featurize generic edge types"
    )
    assert opted_in[key].shape == (1, layout.edge_vector_dim)


@pytest.mark.parametrize("categories,expected_fragment", [
    (["temporal", "corelation"], "corelation"),      # typo
    (["Temporal"], "Temporal"),                      # wrong case
    (["structural"], "structural"),                  # never a verdict
])
def test_unknown_enabled_category_raises(spark, categories, expected_fragment):
    """A name nothing classifies as must fail at construction.

    Left to run, it logs "Building 32-d edge feature vectors", featurizes
    nothing, and exits 0 — an hour of cluster time whose only trace is a zero
    in graph_schema.json.
    """
    with pytest.raises(ValueError) as excinfo:
        EdgeFeatureExtractor(
            spark,
            {"edge_feature_config": {"enabled_categories": categories}},
        )
    assert expected_fragment in str(excinfo.value)
    # The message must say what IS selectable, not merely what is not.
    assert "temporal" in str(excinfo.value)


def test_skip_as_enabled_category_raises_with_its_own_message(spark):
    """'skip' is a verdict, not a category — structural edges are excluded
    before enabled_categories is consulted, so accepting it would promise
    features that can never appear."""
    with pytest.raises(ValueError, match="verdict"):
        EdgeFeatureExtractor(
            spark,
            {"edge_feature_config": {"enabled_categories": ["skip"]}},
        )


def test_default_enabled_categories_are_the_documented_set(spark):
    """Pinned deliberately: the default decides whether a run that configures
    nothing gets edge features at all.

    "correlation" is in the set because the cross-source linkers make it the
    dominant relation family on real data (542 of 704 edge types on a
    two-source build); without it the default produced a graph with no
    edge_attr anywhere. "generic" stays out -- it is the no-fragment-matched
    fallback and would featurize nearly every edge type.
    """
    efe = EdgeFeatureExtractor(spark, {})
    assert efe._enabled_categories == {
        "temporal", "option_stock", "escalation", "correlation",
    }
    assert "generic" not in efe._enabled_categories


# ======================================================================
# The silent-empty diagnostic
# ======================================================================

def test_zero_featurized_edge_types_warns_loudly(spark, caplog):
    """Edge features enabled + nothing featurized must WARN, not stay silent.

    This is the shape of the 2026-07-29 run: enabled, 704 edge types, zero
    featurized, exit 0. Nothing else in the pipeline reports it — the .pt is
    valid and graph_schema.json's edge_types_with_features: 0 is accurate — so
    this log line is the only signal the run was not what was asked for.
    """
    import logging

    logger_name = "spark_jobs.pyg_builder.edge_feature_extractor"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        feats, _ei, _L = _edge_features(spark, _generic_edge_rows())

    assert feats == {}
    assert "NO EDGE TYPES WERE FEATURIZED" in caplog.text
    # Actionable: which categories the graph actually holds, and the fix.
    assert "generic=1" in caplog.text
    assert "enabled_categories" in caplog.text


def test_featurized_run_does_not_warn(spark, caplog):
    """The converse guard: the warning must not cry wolf on a healthy run."""
    import logging

    logger_name = "spark_jobs.pyg_builder.edge_feature_extractor"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        feats, _ei, _L = _edge_features(spark, _temporal_rows(1, 2))

    assert feats, "temporal edges are featurized by default"
    assert "NO EDGE TYPES WERE FEATURIZED" not in caplog.text


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


# ======================================================================
# Chunked collection path for large edge types
# ======================================================================

def _temporal_edge_set(n, year=2020):
    """n edges of a single type (cpi_Index -precedes-> cpi_Series)."""
    rows = []
    for i in range(n):
        src, dst = f"https://ex/src{i}", f"https://ex/dst{i}"
        rows += [
            (src, RDF_TYPE, CPI_INDEX),
            (dst, RDF_TYPE, CPI_SERIES),
            (src, HAS_MONTH, str((i % 12) + 1)),
            (src, HAS_YEAR, str(year)),
            (dst, HAS_MONTH, str(((i + 1) % 12) + 1)),
            (dst, HAS_YEAR, str(year)),
            (src, PRECEDES, dst),
        ]
    return rows


def test_large_edge_type_streams_via_chunked_path(spark, caplog):
    """A type over chunk_edge_threshold streams, and streaming changes nothing.

    Small types are collected in batches, one grouped toPandas per batch; a type
    larger than the threshold instead streams by edge_idx range so a single huge
    type can never materialize whole on the driver (the #186 guarantee). The e2e
    fixtures cannot cover this branch — the threshold is 1,000,000 edges and the
    biggest fixture type has 169 — so the split is exercised here by lowering
    the threshold rather than by enlarging the data.

    Asserts both halves of the contract: the chunked branch actually ran, and it
    produced byte-identical output to the batched branch.
    """
    import logging

    rows = _temporal_edge_set(6)
    key = ("cpi_Index", PRECEDES_REL, "cpi_Series")

    batched, _ei, layout = _edge_features(spark, rows)

    logger_name = "spark_jobs.pyg_builder.edge_feature_extractor"
    with caplog.at_level(logging.INFO, logger=logger_name):
        chunked, _ei2, _layout2 = _edge_features(
            spark,
            rows,
            config={"edge_feature_config": {"chunk_edge_threshold": 2}},
        )

    # 6 edges over a threshold of 2 must take the streaming branch, in 3 chunks.
    assert "Chunked collection" in caplog.text
    assert "0 large" not in caplog.text

    assert batched[key].shape == (6, layout.edge_vector_dim)
    assert chunked[key].shape == batched[key].shape
    assert torch.equal(batched[key], chunked[key])


# ======================================================================
# Broadcast sizing — the endpoint property frames
#
# The six joins that attach endpoint properties to edges used to hint
# F.broadcast() unconditionally, on the stated assumption that the
# filtered frame holds "order 1e3 rows". That holds for the node types
# these fixtures exercise and fails by five orders of magnitude on real
# data, where one node type carried 10.1M nodes and 305M property rows.
# A broadcast build side is collected to the driver whatever its size and
# charged against spark.driver.maxResultSize, so the hint turned a count
# into an 8.1 GiB driver transfer and no run with edge features could
# finish. These tests pin the size test that replaced the assumption.
# ======================================================================

_BROADCAST_HINT = "ResolvedHint (strategy=broadcast)"


def _is_hinted(df):
    """True when df carries a broadcast hint in its analyzed plan."""
    return _BROADCAST_HINT in df._jdf.queryExecution().analyzed().toString()


@pytest.fixture
def broadcast_bar(spark):
    """Build extractors against a chosen autoBroadcastJoinThreshold.

    The threshold is read once at construction, so the conf is restored
    immediately afterwards and never leaks into another test.
    """
    key = "spark.sql.autoBroadcastJoinThreshold"

    def build(threshold, config=None):
        original = spark.conf.get(key)
        spark.conf.set(key, str(threshold))
        try:
            return EdgeFeatureExtractor(spark, config or {})
        finally:
            spark.conf.set(key, original)

    return build


@pytest.mark.parametrize("text,expected", [
    ("10485760b", 10485760),   # what Spark answers when nothing set it
    ("10485760", 10485760),
    ("10MB", 10 * 1024 * 1024),
    ("10m", 10 * 1024 * 1024),
    ("1g", 1024 ** 3),
    ("64k", 64 * 1024),
    # Spark accepts pebibytes too; a suffix we don't know reads as
    # unparseable and would silently disable every hint.
    ("2p", 2 * 1024 ** 5),
    ("-1", -1),                # broadcasting disabled
    ("", None),
    ("nonsense", None),
])
def test_parse_byte_string(text, expected):
    """Spark states byte settings in several forms; all of them have to read."""
    assert _parse_byte_string(text) == expected


def test_broadcast_threshold_read_from_the_session(broadcast_bar):
    """The bar comes from config, not a constant.

    Nothing in this repo sets spark.sql.autoBroadcastJoinThreshold, so the
    usual answer is Spark's 10 MB default — but a deployment that raises it
    is asking for the wider hint and must get it.
    """
    assert broadcast_bar("64k")._broadcast_limit == 64 * 1024
    assert broadcast_bar(-1)._broadcast_limit == -1


def test_threshold_falls_back_when_the_session_cannot_answer():
    """A threshold we cannot read must not read as zero.

    Zero would suppress every hint, quietly costing a shuffle per join per
    edge type on graphs where the frames really are small. Spark rejects an
    unparseable value at conf.set(), so the way this happens is the session
    failing to answer at all.
    """
    class _SilentSession:
        class conf:
            @staticmethod
            def get(_key):
                raise RuntimeError("session is not available")

    efe = EdgeFeatureExtractor(_SilentSession(), {})
    assert efe._broadcast_limit == 10 * 1024 * 1024


def test_maybe_broadcast_hints_only_below_the_bar(spark, broadcast_bar):
    """The size test itself: same frame, different measured row count."""
    efe = broadcast_bar(1024)          # 8 rows at 128 bytes each
    frame = spark.createDataFrame([(1, 2.0)], ["node_id", "numeric_value"])

    assert _is_hinted(efe._maybe_broadcast(frame, 4, _NUMERIC_ROW_BYTES))
    assert not _is_hinted(
        efe._maybe_broadcast(frame, 40, _NUMERIC_ROW_BYTES)
    )


def test_maybe_broadcast_honours_a_disabled_threshold(spark, broadcast_bar):
    """-1 means the deployment turned broadcast joins off. Say nothing."""
    efe = broadcast_bar(-1)
    frame = spark.createDataFrame([(1, 2.0)], ["node_id", "numeric_value"])

    assert not _is_hinted(efe._maybe_broadcast(frame, 1, _NUMERIC_ROW_BYTES))


def test_endpoint_property_rows_counts_per_type_and_subset(spark):
    """Segment 1 broadcasts the month/year subsets, segment 2 the whole
    per-type frame, so all three are counted in the one pass."""
    props = spark.createDataFrame(
        [
            ("big_Type", 1, HAS_MONTH, 1.0),
            ("big_Type", 1, HAS_YEAR, 2020.0),
            ("big_Type", 1, STRIKE, 100.0),
            ("big_Type", 2, HAS_MONTH, 2.0),
            ("small_Type", 3, STRIKE, 5.0),
        ],
        schema=(
            "node_type STRING, node_id LONG, predicate STRING, "
            "numeric_value DOUBLE"
        ),
    )

    sizes = EdgeFeatureExtractor(spark, {})._endpoint_property_rows(props)

    assert sizes["big_Type"].total == 4
    assert sizes["big_Type"].month == 2
    assert sizes["big_Type"].year == 1
    assert sizes["small_Type"].total == 1
    assert sizes["small_Type"].month == 0
    # A type with no numeric properties is absent, and reads as zero rows.
    assert "absent_Type" not in sizes


def _one_edge_df(spark, src_type, dst_type):
    return spark.createDataFrame(
        [(0, 1, 2, src_type, dst_type)],
        schema=(
            "edge_idx LONG, src_id LONG, dst_id LONG, "
            "src_type STRING, dst_type STRING"
        ),
    )


def _two_type_props(spark):
    return spark.createDataFrame(
        [
            ("big_Type", 1, HAS_MONTH, 1.0),
            ("big_Type", 1, HAS_YEAR, 2020.0),
            ("big_Type", 1, STRIKE, 100.0),
            ("small_Type", 2, HAS_MONTH, 2.0),
            ("small_Type", 2, HAS_YEAR, 2020.0),
            ("small_Type", 2, STRIKE, 110.0),
        ],
        schema=(
            "node_type STRING, node_id LONG, predicate STRING, "
            "numeric_value DOUBLE"
        ),
    )


def test_temporal_joins_drop_the_hint_for_a_large_endpoint_type(
    spark, broadcast_bar
):
    """Segment 1, the four-broadcast path.

    market_enrichment_precedes has the same 10.1M-node type on both ends and
    took this path on the run that failed. A frame that size must plan as an
    ordinary shuffle join instead.
    """
    efe = broadcast_bar(256)      # ~10 temporal rows at 24 bytes each
    big = _PropertyRows(total=10_000, month=5_000, year=5_000)

    entries = efe._encode_temporal_signals(
        edge_df=_one_edge_df(spark, "big_Type", "big_Type"),
        src_type="big_Type",
        dst_type="big_Type",
        category="temporal",
        numeric_props_df=_two_type_props(spark),
        src_rows=big,
        dst_rows=big,
    )

    assert entries is not None
    assert not _is_hinted(entries)


def test_temporal_joins_keep_the_hint_for_a_small_endpoint_type(
    spark, broadcast_bar
):
    """The other half of the contract — the original rationale still holds.

    These frames come through an anti-join and an aggregation, so Catalyst's
    estimate for them is far too pessimistic and no join would auto-broadcast.
    Where the frame really is small the hint must stay, or the phase pays one
    shuffle stage per join per edge type.
    """
    efe = broadcast_bar(1024 * 1024)
    small = _PropertyRows(total=6, month=2, year=2)

    entries = efe._encode_temporal_signals(
        edge_df=_one_edge_df(spark, "small_Type", "small_Type"),
        src_type="small_Type",
        dst_type="small_Type",
        category="temporal",
        numeric_props_df=_two_type_props(spark),
        src_rows=small,
        dst_rows=small,
    )

    assert entries is not None
    assert _is_hinted(entries)


def test_numeric_contrast_drops_the_hint_for_a_large_endpoint_type(
    spark, broadcast_bar
):
    """Segment 2, the two-broadcast path — the widest of the six frames.

    No predicate filter narrows these, so the type contributes every numeric
    property of every one of its nodes: 305M rows on the failing run.
    """
    efe = broadcast_bar(256)      # 2 rows at 128 bytes each
    big = _PropertyRows(total=10_000, month=5_000, year=5_000)

    entries = efe._encode_numeric_contrast(
        edge_df=_one_edge_df(spark, "big_Type", "big_Type"),
        src_type="big_Type",
        dst_type="big_Type",
        category="temporal",
        numeric_props_df=_two_type_props(spark),
        has_shared_numerics=True,
        src_rows=big,
        dst_rows=big,
    )

    assert entries is not None
    assert not _is_hinted(entries)


def test_sizing_leaves_the_encoded_values_unchanged(spark):
    """Whether a join is hinted is a planning decision, not a value one.

    Same fixture encoded under a bar that admits every frame and one that
    admits none must produce identical tensors.
    """
    key = ("cpi_Index", PRECEDES_REL, "cpi_Series")
    rows = _temporal_edge_set(4)
    conf_key = "spark.sql.autoBroadcastJoinThreshold"
    original = spark.conf.get(conf_key)

    try:
        spark.conf.set(conf_key, "1g")
        hinted, _ei, _layout = _edge_features(spark, rows)
        spark.conf.set(conf_key, "-1")
        shuffled, _ei2, _layout2 = _edge_features(spark, rows)
    finally:
        spark.conf.set(conf_key, original)

    assert torch.equal(hinted[key], shuffled[key])


# ======================================================================
# Collect budget — bytes, not rows
#
# The batcher capped on edge COUNT while spark.driver.maxResultSize counts
# BYTES, so the same budget meant 32x the bytes at edge_vector_dim 1024 that
# it meant at 32. A run that fit at one width failed at another and nothing
# in the failure named the width (#340).
#
# No Spark needed: the extractor only stores the session, so a stub conf is
# enough to drive the sizing.
# ======================================================================

class _StubConf:
    def __init__(self, value):
        self._value = value

    def get(self, key, default=None):
        assert key == "spark.driver.maxResultSize", key
        return default if self._value is None else self._value


class _StubSession:
    def __init__(self, value):
        self.conf = _StubConf(value)


def _budget(max_result_size, edge_vector_dim):
    fx = EdgeFeatureExtractor(_StubSession(max_result_size), {})
    return fx._max_edges_per_collect(edge_vector_dim)


def test_collect_budget_shrinks_as_the_vector_widens():
    """The whole point: the same cap must buy fewer edges at a wider vector."""
    narrow = _budget("1g", 32)
    wide = _budget("1g", 1024)

    assert wide < narrow
    # 0.6 x 1 GiB / (1024 dims x 20 bytes per collected row)
    assert wide == 31_457
    # The old code returned _CHUNK_EDGE_THRESHOLD for both.
    assert narrow == _CHUNK_EDGE_THRESHOLD


def test_collect_budget_is_sized_on_the_sparse_row_not_the_dense_slot():
    """A collect returns 20 bytes per non-zero slot to deliver 4.

    Sizing on the dense tensor's 4 bytes under-counts what crosses the wire
    by 5x. Measured on the 2026-08 graph the edge features are 51.6% full, so
    the dense estimate was ~2.6x too generous in practice.
    """
    cap = 128 * 1024 ** 2
    dim = 32

    expected = (
        int(cap * _RESULT_SIZE_TARGET_FRACTION)
        // (dim * _SPARSE_ENTRY_BYTES)
    )
    assert _budget("128m", dim) == expected
    assert expected < _CHUNK_EDGE_THRESHOLD, (
        "fixture must pick a cap where bytes bind, or it tests the row cap"
    )
    # Sized on the dense slot this would have allowed 5x more.
    assert _budget("128m", dim) * 5 == pytest.approx(
        int(cap * _RESULT_SIZE_TARGET_FRACTION) // (dim * 4), rel=1e-6
    )


@pytest.mark.parametrize("unbounded", [None, "0", "-1", "not-a-size"])
def test_collect_budget_falls_back_to_the_row_cap(unbounded):
    """Unset, zero, negative and unreadable all mean "no cap to size against".

    Spark reads 0 as unbounded; an unparseable value is not a licence to
    treat the cap as zero and collect one edge at a time.
    """
    assert _budget(unbounded, 1024) == _CHUNK_EDGE_THRESHOLD


def test_collect_budget_never_reaches_zero():
    """A cap smaller than one edge still has to leave the batch able to move."""
    assert _budget("1k", 4096) == 1


def test_collect_budget_is_cached_per_vector_dim():
    """The split loop asks once per edge type; the conf cannot change mid-build."""
    fx = EdgeFeatureExtractor(_StubSession("1g"), {})

    assert fx._max_edges_per_collect(1024) == 31_457
    # A different width must not return the cached answer for the first.
    assert fx._max_edges_per_collect(32) == _CHUNK_EDGE_THRESHOLD
