"""
Structural / integrity tests for the per-source enrichment pattern tables
(NOAA / market / SEC).

These are deliberately *shallow but broad*: pure-Python, no SparkSession. They
pin the shape of the hand-maintained pattern dictionaries and the pure helper
functions — nothing that touches the distributed enrichment code paths.

Rationale (ROI vs. maintenance debt):
  - The pattern tables are hand-edited data. The real bug class is a malformed
    entry (missing key, empty keyword list, typo'd/duplicated relation) that
    *silently drops enrichment* on the cluster with no error. A structural
    guard catches that class at ~zero runtime cost and only fails when someone
    genuinely changes the data — which is exactly when a human should re-check.
  - We assert *invariants* (every entry has a relationship URI, keyword lists
    are non-empty, ranks are distinct), not the full contents. That keeps the
    tests from breaking every time a legitimate new event type / sector is
    added, so they carry very little maintenance debt.
  - The end-to-end linkers (which need Spark) are intentionally out of scope
    here; see test_bls_enrichment.py for that heavier, more brittle tier.
"""
import pytest

from spark_jobs.enrichment.intra_source.noaa import patterns as noaa
from spark_jobs.enrichment.intra_source.market import patterns as market
from spark_jobs.enrichment.intra_source.market import correlations as market_corr
from spark_jobs.enrichment.intra_source.sec import patterns as sec


def _is_uri(value):
    """Accept both str and rdflib.URIRef; require an http(s) IRI."""
    return str(value).startswith(("http://", "https://"))


def _distinct_positive_ranks(hierarchy):
    """A rank map is valid if values are ints and (for a strict hierarchy) unique."""
    values = list(hierarchy.values())
    assert all(isinstance(v, int) for v in values)
    assert len(values) >= 2
    return values


# ======================================================================
# NOAA
# ======================================================================

def test_noaa_event_patterns_well_formed():
    assert noaa.NOAA_EVENT_PATTERNS, "no NOAA event patterns defined"
    for key, entry in noaa.NOAA_EVENT_PATTERNS.items():
        assert {"description", "event_types", "cap_category", "relationship"} <= set(entry), key
        assert entry["event_types"], f"{key} has no event_types"
        assert all(isinstance(e, str) and e for e in entry["event_types"]), key
        assert _is_uri(entry["relationship"]), f"{key} relationship not a URI"
        assert isinstance(entry["cap_category"], str) and entry["cap_category"], key


def test_noaa_event_types_unique_across_categories():
    """A given warning string must map to exactly one category, else enrichment
    would double-classify or race depending on dict order."""
    seen = {}
    for key, entry in noaa.NOAA_EVENT_PATTERNS.items():
        for event in entry["event_types"]:
            assert event not in seen, f"'{event}' in both {seen.get(event)} and {key}"
            seen[event] = key


def test_noaa_relationships_unique_per_category():
    rels = [str(e["relationship"]) for e in noaa.NOAA_EVENT_PATTERNS.values()]
    assert len(rels) == len(set(rels)), "duplicate relationship URIs across categories"


@pytest.mark.parametrize("hierarchy", [
    noaa.SEVERITY_HIERARCHY,
    noaa.URGENCY_HIERARCHY,
    noaa.CERTAINTY_HIERARCHY,
])
def test_noaa_strict_hierarchies_have_distinct_ranks(hierarchy):
    values = _distinct_positive_ranks(hierarchy)
    assert len(values) == len(set(values)), "strict CAP hierarchy must have unique ranks"
    assert all(_is_uri(k) for k in hierarchy), "hierarchy keys must be CAP URIs"


def test_noaa_response_type_severity_monotonic_range():
    # RESPONSE_TYPE_SEVERITY intentionally reuses ranks (None/AllClear both 0),
    # so only assert the range/typing, not uniqueness.
    values = _distinct_positive_ranks(noaa.RESPONSE_TYPE_SEVERITY)
    assert min(values) >= 0
    assert all(_is_uri(k) for k in noaa.RESPONSE_TYPE_SEVERITY)


# ======================================================================
# market
# ======================================================================

@pytest.mark.parametrize("gics,key,pascal", [
    ("Information Technology", "information_technology_sector", "InformationTechnology"),
    ("Health Care", "health_care_sector", "HealthCare"),
])
def test_market_gics_sector_helpers(gics, key, pascal):
    assert market._gics_sector_to_key(gics) == key
    assert market._gics_sector_to_pascal(gics) == pascal


def test_market_default_sector_keys_follow_convention():
    for key in market.DEFAULT_MARKET_SECTOR_PATTERNS:
        assert key.endswith("_sector"), f"sector key {key!r} missing _sector suffix"


@pytest.mark.parametrize("raw,expected", [
    ("320193", "0000320193"),      # the CSV's form
    ("0000320193", "0000320193"),  # the filings' form, unchanged
    (" 320193 ", "0000320193"),    # a stray cell space
])
def test_constituent_cik_is_padded_to_the_filed_width(raw, expected):
    """The join key the whole company bridge rests on.

    The CSV states the CIK unpadded and the filings state it padded to ten;
    compared as strings those are different keys, so the unpadded form joins
    against nothing and reports no error.
    """
    assert market.padded_cik(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "abc", "12a45", "00003201930"])
def test_a_value_that_is_not_a_cik_is_dropped_not_guessed(raw):
    """One malformed row drops that constituent; it does not fail the build.

    The over-long case is the one that matters: Spark's lpad would TRUNCATE it
    from the right into a shorter, entirely plausible, WRONG filer. zfill plus
    this guard rejects it instead.
    """
    assert market.padded_cik(raw) is None


def test_an_unpadded_cik_map_is_rejected_loudly():
    """An empty bridge is indistinguishable from a fixture with no overlap.

    That is why this raises rather than logging: the silent version of this
    bug survives review, because a join matching zero rows looks exactly like
    data that genuinely does not intersect.
    """
    with pytest.raises(ValueError, match="zero-padded"):
        market.assert_ciks_are_padded({"AAPL": "320193"})

    with pytest.raises(ValueError, match="zero-padded"):
        market.assert_ciks_are_padded({"AAPL": 320193})

    market.assert_ciks_are_padded({"AAPL": "0000320193"})
    market.assert_ciks_are_padded({})


def _constituents_csv(rows, header=None):
    """A stub S3 client serving one constituents CSV."""
    header = header or "Symbol,Security,GICS Sector,GICS Sub-Industry,CIK"
    body = "\n".join([header, *rows]).encode("utf-8")

    class _Body:
        @staticmethod
        def read():
            return body

    class _Client:
        @staticmethod
        def get_object(Bucket, Key):
            return {"Body": _Body}

    return _Client


def test_the_constituents_csv_yields_a_padded_ticker_cik_map():
    client = _constituents_csv([
        "AAPL,Apple Inc.,Information Technology,Technology Hardware,320193",
        "MSFT,Microsoft,Information Technology,Application Software,0000789019",
    ])

    assert market.load_ticker_cik_map_from_s3("b", "k", client) == {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
    }


def test_the_constituents_csv_yields_sub_industry_pairs():
    client = _constituents_csv([
        "NVDA,NVIDIA,Information Technology,Semiconductors,1045810",
        "AMD,AMD,Information Technology,Semiconductors,2488",
    ])

    assert market.load_sub_industries_from_s3("b", "k", client) == [
        ("NVDA", "Semiconductors"),
        ("AMD", "Semiconductors"),
    ]


def test_each_reader_requires_only_the_columns_it_uses():
    """A CSV vintage missing one column must not disable the other readers.

    The three readers share one file and one fetch, so a single required-column
    set would couple them: an older export without GICS Sub-Industry would take
    the company bridge down with the peer edges.
    """
    no_cik = _constituents_csv(
        ["AAPL,Apple Inc.,Information Technology,Technology Hardware"],
        header="Symbol,Security,GICS Sector,GICS Sub-Industry",
    )

    assert market.load_ticker_cik_map_from_s3("b", "k", no_cik) is None
    assert market.load_sector_patterns_from_s3("b", "k", no_cik) is not None
    assert market.load_sub_industries_from_s3("b", "k", no_cik) is not None


def test_no_csv_means_no_map_rather_than_a_guessed_one():
    """get_sector_patterns falls back to defaults; these two must not.

    A stale sector assignment is still roughly true. A CIK is a regulator's
    primary key, and a wrong one silently merges two unrelated companies into
    one node — so the absence of the file is reported as an absence.
    """
    assert market.get_ticker_cik_map() == {}
    assert market.get_sub_industries() == []
    assert market.get_sector_patterns() is market.DEFAULT_MARKET_SECTOR_PATTERNS


# ======================================================================
# Which constituents CSV a run reads
# ======================================================================
#
# A constituents list is a point-in-time membership: tickers join and leave the
# index, so a run rebuilding an older day must read that day's export rather
# than the current one.

def test_the_days_own_export_is_preferred_to_the_undated_one():
    assert market.constituents_keys("ref/tickers", "2026-09-12") == [
        "ref/tickers/year=2026/month=09/12.csv",
        "ref/tickers/latest.csv",
    ]


def test_without_a_data_day_only_the_undated_export_is_a_candidate():
    assert market.constituents_keys("ref/tickers") == ["ref/tickers/latest.csv"]


@pytest.mark.parametrize("data_day", [
    "2026-9-12",     # unpadded
    "26-09-12",      # two-digit year
    "2026-09",       # a month, not a day
    "latest",
    "",
])
def test_a_day_that_is_not_a_date_does_not_become_a_key(data_day):
    """Rendering an unparseable label into the path would request an object
    that cannot exist, and the miss would read as 'no export for that day'."""
    assert market.constituents_keys("ref/tickers", data_day) == [
        "ref/tickers/latest.csv"
    ]


@pytest.mark.parametrize("prefix", ["ref/tickers", "ref/tickers/", "/ref/tickers/"])
def test_the_prefix_is_normalised_rather_than_doubled(prefix):
    assert market.constituents_keys(prefix) == ["ref/tickers/latest.csv"]


def test_no_prefix_means_no_candidates():
    assert market.constituents_keys("") == []
    assert market.constituents_keys("   ") == []


_FULL_HEADER = "Symbol,Security,GICS Sector,GICS Sub-Industry,CIK"


def _keyed_csv(bodies, header=_FULL_HEADER):
    """A stub S3 client serving only the keys in ``bodies``.

    Records every key requested, in order, so a test can assert what was NOT
    fetched as well as what was. A value may be a list of rows (rendered under
    ``header``) or a complete CSV string.
    """
    from botocore.exceptions import ClientError

    class _Client:
        requested = []

        @classmethod
        def get_object(cls, Bucket, Key):
            cls.requested.append(Key)
            if Key not in bodies:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            content = bodies[Key]
            if not isinstance(content, str):
                content = "\n".join([header, *content])

            class _Body:
                @staticmethod
                def read():
                    return content.encode("utf-8")

            return {"Body": _Body}

    return _Client


def test_the_days_export_is_used_and_the_undated_one_is_never_fetched():
    client = _keyed_csv({
        "ref/year=2026/month=09/12.csv": [
            "AAPL,Apple Inc.,Information Technology,Technology Hardware,320193",
        ],
        "ref/latest.csv": [
            "MSFT,Microsoft,Information Technology,Application Software,789019",
        ],
    })

    assert market.load_ticker_cik_map_from_s3(
        "b", "ref", client, data_day="2026-09-12"
    ) == {"AAPL": "0000320193"}
    assert client.requested == ["ref/year=2026/month=09/12.csv"]


def test_a_missing_day_falls_back_to_the_undated_export():
    client = _keyed_csv({
        "ref/latest.csv": [
            "MSFT,Microsoft,Information Technology,Application Software,789019",
        ],
    })

    assert market.load_ticker_cik_map_from_s3(
        "b", "ref", client, data_day="2026-09-12"
    ) == {"MSFT": "0000789019"}
    assert client.requested == [
        "ref/year=2026/month=09/12.csv",
        "ref/latest.csv",
    ]


def test_neither_export_present_is_an_absence_not_a_guess():
    client = _keyed_csv({})

    assert market.load_ticker_cik_map_from_s3(
        "b", "ref", client, data_day="2026-09-12"
    ) is None
    assert market.get_ticker_cik_map(
        "b", "ref", client, data_day="2026-09-12"
    ) == {}


@pytest.mark.parametrize("broken", [
    "Symbol,Security,GICS Sector\nAAPL,Apple Inc.,Information Technology",
    "",
])
def test_a_broken_days_export_is_reported_rather_than_routed_around(broken):
    """The fallback is for a day that has not been published, not for one that
    has been published wrong. Silently reading a different day's membership
    because that day's file is malformed hides the defect and produces a graph
    nobody can account for — so only a MISSING object is retried elsewhere.
    """
    client = _keyed_csv({
        "ref/year=2026/month=09/12.csv": broken,
        "ref/latest.csv": [
            "MSFT,Microsoft,Information Technology,Application Software,789019",
        ],
    })

    assert market.load_ticker_cik_map_from_s3(
        "b", "ref", client, data_day="2026-09-12"
    ) is None
    assert client.requested == ["ref/year=2026/month=09/12.csv"]


# Run 20260913T032118Z was configured with .../latest.csv rather than the
# prefix. The lookup built both keys under that file name, neither existed, and
# the run lost its company and sector links with only an info line to say so.

def test_a_latest_csv_setting_still_tries_the_days_export_first():
    assert market.constituents_keys("ref/tickers/latest.csv", "2026-09-12") == [
        "ref/tickers/year=2026/month=09/12.csv",
        "ref/tickers/latest.csv",
    ]


@pytest.mark.parametrize("setting", [
    "ref/tickers/latest.csv",
    "/ref/tickers/latest.csv",
    " ref/tickers/latest.csv ",
])
@pytest.mark.parametrize("data_day", ["2026-09-12", ""])
def test_a_latest_csv_setting_means_the_same_place_as_its_prefix(setting, data_day):
    assert market.constituents_keys(setting, data_day) == (
        market.constituents_keys("ref/tickers", data_day)
    )


def test_a_named_days_csv_is_tried_first_then_latest_csv():
    """A day's CSV named in the setting takes the place of the day being
    processed. latest.csv is still the fallback, and it sits two folders up
    from the day's file, not beside it."""
    named = "ref/tickers/year=2026/month=09/11.csv"
    assert market.constituents_keys(named, "2026-09-12") == [
        named,
        "ref/tickers/latest.csv",
    ]


@pytest.mark.parametrize("setting", [
    "ref/tickers",
    "ref/tickers/",
    "ref/tickers/latest.csv",
    "ref/tickers/year=2026/month=09/11.csv",
])
@pytest.mark.parametrize("data_day", ["2026-09-12", ""])
def test_every_setting_ends_at_latest_csv_and_never_looks_under_a_file(
    setting, data_day
):
    keys = market.constituents_keys(setting, data_day)
    assert keys[-1] == "ref/tickers/latest.csv"
    for key in keys:
        assert ".csv/" not in key, key
        assert key.endswith(".csv"), key


def test_a_setting_at_the_bucket_root_builds_no_leading_slash():
    assert market.constituents_keys("latest.csv", "2026-09-12") == [
        "year=2026/month=09/12.csv",
        "latest.csv",
    ]


def test_a_latest_csv_setting_reads_the_days_export_in_every_reader():
    """The failed run's exact setting, through all three readers."""
    client = _keyed_csv({
        "ref/tickers/year=2026/month=09/09.csv": [
            "AAPL,Apple Inc.,Information Technology,Technology Hardware,320193",
        ],
        "ref/tickers/latest.csv": [
            "MSFT,Microsoft,Information Technology,Application Software,789019",
        ],
    })
    where = dict(bucket="b", prefix="ref/tickers/latest.csv",
                 s3_client=client, data_day="2026-09-09")

    assert market.get_ticker_cik_map(**where) == {"AAPL": "0000320193"}
    assert market.get_sub_industries(**where) == [
        ("AAPL", "Technology Hardware"),
    ]
    patterns = market.get_sector_patterns(**where)
    assert patterns is not market.DEFAULT_MARKET_SECTOR_PATTERNS
    assert patterns["information_technology_sector"]["tickers"] == ["AAPL"]
    assert set(client.requested) == {"ref/tickers/year=2026/month=09/09.csv"}


def test_a_latest_csv_setting_falls_back_to_latest_when_the_day_is_missing():
    client = _keyed_csv({
        "ref/tickers/latest.csv": [
            "MSFT,Microsoft,Information Technology,Application Software,789019",
        ],
    })

    assert market.get_ticker_cik_map(
        "b", "ref/tickers/latest.csv", client, data_day="2026-09-09"
    ) == {"MSFT": "0000789019"}
    assert client.requested == [
        "ref/tickers/year=2026/month=09/09.csv",
        "ref/tickers/latest.csv",
    ]


def test_a_named_days_csv_that_is_missing_falls_back_to_latest_csv():
    client = _keyed_csv({
        "ref/tickers/latest.csv": [
            "MSFT,Microsoft,Information Technology,Application Software,789019",
        ],
    })

    assert market.get_ticker_cik_map(
        "b", "ref/tickers/year=2026/month=09/11.csv", client,
        data_day="2026-09-12",
    ) == {"MSFT": "0000789019"}
    assert client.requested == [
        "ref/tickers/year=2026/month=09/11.csv",
        "ref/tickers/latest.csv",
    ]


def test_no_csv_at_any_key_is_a_warning_naming_the_keys_tried(caplog):
    """A missing day is normal and stays at info. Ending up with no CSV at all
    is not, and an info line is how the failed run hid it."""
    import logging

    client = _keyed_csv({})
    with caplog.at_level(logging.WARNING, logger=market.logger.name):
        assert market.load_ticker_cik_map_from_s3(
            "b", "ref/tickers/latest.csv", client, data_day="2026-09-12"
        ) is None

    warnings = [record.getMessage() for record in caplog.records
                if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "s3://b/ref/tickers/year=2026/month=09/12.csv" in warnings[0]
    assert "s3://b/ref/tickers/latest.csv" in warnings[0]


@pytest.mark.parametrize("script", [
    "generate_market_e2e_fixtures",
    "generate_sec_e2e_fixtures",
])
def test_the_fixture_generators_read_the_first_csv_the_run_tries(
    script, monkeypatch
):
    """The generators copy the rule rather than import it, because importing it
    brings the Spark stack along. With no data day they read the first key the
    run would try."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "bin" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"_{script}", path)
    generator = importlib.util.module_from_spec(spec)
    # A dataclass looks its own module up in sys.modules as it is defined.
    monkeypatch.setitem(sys.modules, spec.name, generator)
    spec.loader.exec_module(generator)

    for setting in ("ref/tickers", "ref/tickers/", "ref/tickers/latest.csv",
                    "ref/tickers/year=2026/month=09/11.csv"):
        assert generator.constituents_key(setting) == (
            market.constituents_keys(setting)[0]
        ), setting


def test_market_option_strategy_patterns_well_formed():
    assert market.MARKET_OPTION_STRATEGY_PATTERNS
    for key, entry in market.MARKET_OPTION_STRATEGY_PATTERNS.items():
        assert _is_uri(entry["relationship"]), key
        assert _is_uri(entry["pattern_uri"]), key
        assert entry["description"], key


# market/measurements.py and sec/measurements.py were deleted, and the test that
# stood here went with the market one. It asserted that a config no production
# code imported was well-formed -- the only thing keeping either file alive, and
# a check that could never fail in a way that mattered. Both files described the
# classes, date properties and grouping keys their linkers already hardcode, so
# they were a second description of the code sitting beside it, consulted by
# nothing while being maintained as though they were wired up.
#
# bls/measurements.py is the one that stays, because bls/base_enricher.py really
# does loop over it: adding an entry there changes what the pipeline emits.
#
# The SEC config also described two entries nothing implemented at the time --
# transaction-level temporal sequencing over NonDerivativeTransaction (18.16%
# of filings) and DerivativeTransaction (7.05%). That was real missing graph
# structure rather than dead config, so it was raised as its own issue instead
# of being deleted silently, and sec_linker._link_transaction_sequences now
# builds it.


def test_market_known_correlations_well_formed():
    corr = market_corr.KNOWN_CORRELATIONS
    assert corr
    names = [c["name"] for c in corr]
    assert len(names) == len(set(names)), "duplicate correlation names"
    for c in corr:
        assert {"source_type", "target_type", "relationship", "strength"} <= set(c), c["name"]
        assert _is_uri(c["relationship"]), c["name"]
        assert c["strength"] in {"weak", "medium", "strong"}, c["name"]


# ======================================================================
# SEC
# ======================================================================

# SEC_SECTOR_PATTERNS was deleted -- it matched nothing on real data and sorted
# filings into a third, unreconciled sector vocabulary. Filings now get their
# sector from filings:hasSic. Violations keep their keyword table, which does
# match.
@pytest.mark.parametrize("table,uri_key", [
    ("SEC_VIOLATION_PATTERNS", "violation_uri"),
])
def test_sec_pattern_tables_well_formed(table, uri_key):
    patterns = getattr(sec, table)
    assert patterns, f"{table} is empty"
    for key, entry in patterns.items():
        assert _is_uri(entry[uri_key]), f"{table}/{key} {uri_key} not a URI"
        assert _is_uri(entry["relationship"]), f"{table}/{key} relationship not a URI"
        assert entry["keywords"], f"{table}/{key} has no keyword groups"
        # A per-dataset keyword list may be empty by design ("don't match this
        # sector via this dataset"), but the entry must be matchable somewhere.
        assert any(entry["keywords"].values()), f"{table}/{key} matchable via no dataset"
        for group, kws in entry["keywords"].items():
            assert all(isinstance(k, str) and k for k in kws), f"{table}/{key}/{group}"



# ======================================================================
# Edge origin classification (rdf_utils.classify_edge_origin)
# ======================================================================

def test_classify_edge_origin_by_predicate_namespace():
    """A minted predicate means the pipeline inferred the link."""
    from spark_jobs.utils.rdf_utils import (
        classify_edge_origin, BLS_ENRICHMENT, SEC_ENRICHMENT,
        NOAA_ENRICHMENT, MARKET_ENRICHMENT, CPI,
    )

    for ns in (BLS_ENRICHMENT, SEC_ENRICHMENT, NOAA_ENRICHMENT,
               MARKET_ENRICHMENT):
        assert classify_edge_origin(f"{ns}linkedTo", "a_X", "b_Y") == (
            "enrichment"
        ), ns

    assert classify_edge_origin(f"{CPI}hasValue", "cpi_S", "cpi_I") == "raw"
    # Missing predicate and plain endpoints must not raise.
    assert classify_edge_origin("", "cpi_S", "cpi_I") == "raw"


def test_unification_is_detected_via_endpoints_not_predicate():
    """The regression that made this function take endpoints at all.

    Unification links carry a MINTED NODE but a STANDARD predicate
    (unified:November owl:sameAs cpi:November). Classifying on the predicate
    alone reported them as "raw" -- an inferred link presented as an observed
    fact, which is exactly the mislabelling edge origin exists to prevent. On
    the e2e fixtures that was 16 of 19 supposedly-raw edge types.
    """
    from spark_jobs.utils.rdf_utils import classify_edge_origin, OWL_SAME_AS

    # owl:sameAs is NOT in any pipeline namespace ...
    assert classify_edge_origin(OWL_SAME_AS, "cpi_Month", "ppi_Month") == "raw"
    # ... so only the minted endpoint reveals the edge as pipeline-made.
    assert classify_edge_origin(
        OWL_SAME_AS, "bls_enrichment_UnifiedMonth", "cpi_PercentChange"
    ) == "unification"
    assert classify_edge_origin(
        OWL_SAME_AS, "cpi_PercentChange", "bls_enrichment_UnifiedMonth"
    ) == "unification"


def test_an_inferred_bls_link_is_still_classified_as_enrichment():
    """The behaviour this used to guard, kept after the namespaces moved.

    It previously asserted that BLS_ENRICHMENT sits UNDER https://jefflevesque.com/ontology/bls-common/
    and is therefore at risk of being swallowed by the shorter source
    namespace. That arrangement was the defect: a URI under bls.gov claims BLS
    defined a term we invented. The minted vocabularies now live under a
    domain this project controls, which dissolves the shadowing hazard here
    rather than managing it -- the general ordering invariant is asserted over
    the whole table in tests/test_namespaces.py.

    What must not change is the verdict: an inferred BLS link is enrichment,
    not raw.
    """
    from spark_jobs.utils.rdf_utils import (
        classify_edge_origin, BLS_ENRICHMENT, ONTOLOGY_BASE,
    )

    assert str(BLS_ENRICHMENT).startswith(ONTOLOGY_BASE)
    assert "bls.gov" not in str(BLS_ENRICHMENT)
    assert classify_edge_origin(
        f"{BLS_ENRICHMENT}apparelSectorCorrelation", "a_X", "b_Y"
    ) == "enrichment"


def test_pipeline_node_type_prefixes_derive_from_the_namespace_registry():
    """Adding an enrichment namespace must not need a second list updated."""
    from spark_jobs.utils.rdf_utils import PIPELINE_NODE_TYPE_PREFIXES

    assert "bls_enrichment_" in PIPELINE_NODE_TYPE_PREFIXES
    assert "unified_" in PIPELINE_NODE_TYPE_PREFIXES
    # Source namespaces must NOT be treated as pipeline-minted.
    assert "cpi_" not in PIPELINE_NODE_TYPE_PREFIXES
    assert "cap_" not in PIPELINE_NODE_TYPE_PREFIXES
