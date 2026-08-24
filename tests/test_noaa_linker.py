"""
Tests for the NOAA intra-source enrichment linker (NOAAIntraSourceLinker).

Three tiers, mirroring test_bls_linker.py:
  - Tier 2 smokes: enrich() classifies a known event, and foreign input
    short-circuits to zero triples.
  - Targeted deep test: severity escalation (`_link_severity_escalations`) —
    the window-ordered, rank-comparison logic that is easy to regress. Covers
    the positive case, the non-escalation case (severity drops), the secondary
    urgency-tiebreak signal, and the #339 partition key: escalations are
    sequenced per FIPS/SAME geocode, not per free-text `cap:hasAreaDescription`.
  - Repeated-source-row tests: the NOAA archive states one row per
    (alert, geocode) and restates the whole alert/info/area block in each, so
    every one of `_build_alert_info`'s ten join legs sees the same triple k
    times. Undeduplicated that is k**10 rows. See `_noaa_triples`.

Drives enrich() over tiny in-memory triples on the shared local SparkSession
(`spark` / `make_triples` fixtures from conftest.py).
"""

from spark_jobs.utils.rdf_utils import CAP, NOAA_ENRICHMENT

from spark_jobs.enrichment.intra_source.noaa_linker import (
    NOAAIntraSourceLinker,
    NOAA_EVENT_PATTERNS,
    NOAA_PREDICATES,
    SEVERITY_HIERARCHY,
    URGENCY_HIERARCHY,
    RDF_TYPE,
    WEATHER_ALERT_TYPE,
    HAS_INFO,
    HAS_EVENT,
    HAS_EVENT_CLASSIFICATION,
    HAS_SENT_TIME,
    HAS_SEVERITY,
    HAS_URGENCY,
    HAS_CERTAINTY,
    HAS_CATEGORY,
    HAS_AREA,
    HAS_AREA_DESC,
    HAS_GEOCODE,
    HAS_FIPS_CODE,
    HAS_SAME_CODE,
    ESCALATES_TO_PRED,
    PRECEDES_PRED,
)


def _triple_set(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


def _enrich_rows(spark, make_triples, rows):
    return _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))


def _pairs_under(triples, predicate):
    return {(s, o) for s, p, o in triples if p == predicate}


def _precedes(triples):
    return _pairs_under(triples, PRECEDES_PRED)


def _escalates(triples):
    return _pairs_under(triples, ESCALATES_TO_PRED)


def _severity_uri(level):
    """The CAP severity URI at a given rank (Minor=1 .. Extreme=4)."""
    return next(uri for uri, lvl in SEVERITY_HIERARCHY.items() if lvl == level)


def _urgency_uri(level):
    """The CAP urgency URI at a given rank (Past=1 .. Immediate=4)."""
    return next(uri for uri, lvl in URGENCY_HIERARCHY.items() if lvl == level)


# The ten legs _build_alert_info joins. Named here so the k**10 arithmetic in
# the tests below is checkable against the code rather than asserted.
ALERT_INFO_JOIN_LEGS = 10


def _alert_over_geocodes(alert, fips_codes, *, restate_block, event="Flood Advisory",
                         sent_time="2024-11-15T10:00:00", area_desc=None,
                         severity=3, urgency=3):
    """The triples one alert covering `fips_codes` counties arrives as.

    `restate_block=True` is the real archive shape: the source stores one
    Parquet row per (alert, geocode), and every row's Turtle blob carries the
    complete alert, info and area blocks — only the geocode sub-block differs.
    Parsed and exploded, that yields len(fips_codes) identical copies of each of
    the ten triples _build_alert_info joins on.

    `restate_block=False` is the same graph as a set: block stated once, one
    geocode sub-block per county. The two must enrich to the same triples.

    `severity` and `urgency` are ranks in SEVERITY_HIERARCHY / URGENCY_HIERARCHY
    (Minor=1..Extreme=4, Past=1..Immediate=4), for the escalation tests.
    """
    info, area = f"{alert}#info", f"{alert}#area"
    block = [
        (alert, RDF_TYPE, WEATHER_ALERT_TYPE),
        (alert, HAS_INFO, info),
        (info, HAS_SENT_TIME, sent_time),
        (info, HAS_EVENT, event),
        (info, HAS_SEVERITY, _severity_uri(severity)),
        (info, HAS_URGENCY, _urgency_uri(urgency)),
        (info, HAS_CERTAINTY, str(CAP.Observed)),
        (info, HAS_CATEGORY, str(CAP.Met)),
        (info, HAS_AREA, area),
        (area, HAS_AREA_DESC,
         area_desc if area_desc is not None
         else "; ".join(f"County {c}" for c in fips_codes)),
    ]
    assert len(block) == ALERT_INFO_JOIN_LEGS

    rows = []
    for code in fips_codes:
        geocode = f"{alert}#geocode-{code}"
        if restate_block or not rows:
            rows.extend(block)
        rows.extend([
            (area, HAS_GEOCODE, geocode),
            (geocode, HAS_FIPS_CODE, code),
            (geocode, HAS_SAME_CODE, code),
        ])
    return rows


# ======================================================================
# Tier 2 smokes
# ======================================================================

def test_noaa_enrich_classifies_event_category(spark, make_triples):
    """WeatherAlert with cap:hasEvent "Tornado Warning" -> severe_weather."""
    alert = "http://example.org/alert/1"
    info = "http://example.org/alert/1/info"
    rows = [
        (alert, RDF_TYPE, WEATHER_ALERT_TYPE),
        (alert, HAS_INFO, info),
        (info, HAS_EVENT, "Tornado Warning"),
    ]

    triples = _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))

    category_uri = str(NOAA_ENRICHMENT["severe_weather"])
    relationship = str(NOAA_EVENT_PATTERNS["severe_weather"]["relationship"])
    assert (alert, HAS_EVENT_CLASSIFICATION, category_uri) in triples
    assert (alert, relationship, category_uri) in triples


def test_noaa_enrich_empty_for_non_noaa_data(spark, make_triples):
    rows = [("http://example.org/x", RDF_TYPE, "http://example.org/SomethingElse")]
    result = NOAAIntraSourceLinker(spark).enrich(make_triples(rows))
    assert result.count() == 0


# ======================================================================
# Deep: severity escalation
#
# Keys on the geocode, not cap:hasAreaDescription — the same defect precedes
# had, fixed the same way. See _link_severity_escalations.
# ======================================================================

def test_severity_escalation_links_increasing_severity(spark, make_triples):
    """Same county, Moderate -> Severe over time yields a directional escalatesTo."""
    a1, a2 = "http://example.org/alert/a1", "http://example.org/alert/a2"
    rows = (
        _alert_over_geocodes(a1, ["012001"], restate_block=False,
                             sent_time="2024-11-15T10:00:00", severity=2)
        + _alert_over_geocodes(a2, ["012001"], restate_block=False,
                               sent_time="2024-11-15T11:00:00", severity=3)
    )

    triples = _enrich_rows(spark, make_triples, rows)

    assert (a1, ESCALATES_TO_PRED, a2) in triples
    # Directional: the higher-severity alert does not escalate back down.
    assert (a2, ESCALATES_TO_PRED, a1) not in triples


def test_no_escalation_when_severity_decreases(spark, make_triples):
    """Severe -> Minor is a de-escalation: no escalatesTo edge."""
    a1, a2 = "http://example.org/alert/b1", "http://example.org/alert/b2"
    rows = (
        _alert_over_geocodes(a1, ["012003"], restate_block=False,
                             sent_time="2024-11-15T10:00:00", severity=3)
        + _alert_over_geocodes(a2, ["012003"], restate_block=False,
                               sent_time="2024-11-15T11:00:00", severity=1)
    )

    assert _escalates(_enrich_rows(spark, make_triples, rows)) == set()


def test_escalation_via_urgency_when_severity_equal(spark, make_triples):
    """Same severity but rising urgency (Expected -> Immediate) still escalates."""
    a1, a2 = "http://example.org/alert/c1", "http://example.org/alert/c2"
    rows = (
        _alert_over_geocodes(a1, ["012005"], restate_block=False,
                             sent_time="2024-11-15T10:00:00",
                             severity=3, urgency=3)
        + _alert_over_geocodes(a2, ["012005"], restate_block=False,
                               sent_time="2024-11-15T11:00:00",
                               severity=3, urgency=4)
    )

    assert (a1, ESCALATES_TO_PRED, a2) in _enrich_rows(spark, make_triples, rows)


def test_escalation_fires_when_area_descriptions_differ(spark, make_triples):
    """The half of #339 the follow-up added: same county, different county lists.

    This is the escalation the old key threw away. A warning is re-issued at a
    higher severity over a slightly different set of counties, so the two
    free-text descriptions differ, so the two alerts landed in different window
    partitions and the upgrade went undetected — even though both alerts name
    the same FIPS code. One day of 1,262 alerts yielded 56 escalatesTo edges.
    """
    a1, a2 = "http://example.org/alert/d1", "http://example.org/alert/d2"
    rows = (
        _alert_over_geocodes(a1, ["013083"], restate_block=False,
                             sent_time="2024-11-15T10:00:00", severity=2,
                             area_desc="Dade; Walker; Catoosa")
        + _alert_over_geocodes(a2, ["013083"], restate_block=False,
                               sent_time="2024-11-15T11:00:00", severity=4,
                               area_desc="Dade; Walker; Catoosa; Whitfield")
    )

    assert _escalates(_enrich_rows(spark, make_triples, rows)) == {(a1, a2)}


def test_escalation_does_not_link_alerts_in_different_counties(spark, make_triples):
    """No shared code means no escalation, however sharp the severity rise."""
    a1, a2 = "http://example.org/alert/e1", "http://example.org/alert/e2"
    rows = (
        _alert_over_geocodes(a1, ["012001"], restate_block=False,
                             sent_time="2024-11-15T10:00:00", severity=1)
        + _alert_over_geocodes(a2, ["012099"], restate_block=False,
                               sent_time="2024-11-15T11:00:00", severity=4)
    )

    assert _escalates(_enrich_rows(spark, make_triples, rows)) == set()


def test_escalation_emits_one_triple_for_a_pair_adjacent_in_several_counties(
    spark, make_triples
):
    """An escalation over three shared counties is one fact, so one triple."""
    a1, a2 = "http://example.org/alert/f1", "http://example.org/alert/f2"
    shared = ["012001", "012003", "012005"]
    rows = (
        _alert_over_geocodes(a1, shared, restate_block=False,
                             sent_time="2024-11-15T10:00:00", severity=2)
        + _alert_over_geocodes(a2, shared, restate_block=False,
                               sent_time="2024-11-15T11:00:00", severity=3)
    )

    triples = _enrich_rows(spark, make_triples, rows)
    assert _escalates(triples) == {(a1, a2)}
    # One triple, not one per shared county.
    assert len([1 for _, p, _ in triples if p == ESCALATES_TO_PRED]) == 1


# ======================================================================
# Deep: repeated source rows (the k**10 denormalization blow-up)
# ======================================================================

def test_alert_table_is_one_row_per_alert_when_source_restates_the_block(
    spark, make_triples
):
    """One alert over four counties denormalizes to ONE row, not 4**10.

    Reaches past enrich() into the two private steps because the row count is
    the assertion: `_build_alert_info` joins ten attribute relations and none of
    them deduplicates, so an alert the source states k times becomes k**10 rows
    and every later step inherits it. On real data that reached 3.7e18 rows for
    a single 72-county alert, which is why the job stopped logging at Step 1/6
    rather than failing.

    Four counties keeps the un-fixed number at 4**10 = 1,048,576 — large enough
    to prove the multiplication, small enough that a regression fails this
    assertion in seconds instead of hanging the suite.
    """
    alert = "http://example.org/alert/multi-county"
    rows = _alert_over_geocodes(
        alert, ["013083", "013295", "013047", "013115"], restate_block=True
    )

    linker = NOAAIntraSourceLinker(spark)
    alert_info = linker._build_alert_info(linker._noaa_triples(make_triples(rows)))

    assert alert_info.count() == 1
    assert alert_info.select("alert").distinct().count() == 1


def test_enrich_is_unchanged_by_a_source_that_restates_the_block(
    spark, make_triples
):
    """The same graph stated twice over produces the same enrichment.

    The fix rests on this: deduplicating the frame is safe precisely because
    repeated triples are the same RDF, so removing them cannot change what the
    linker emits. Two alerts, each over three counties — once in the archive's
    row-per-geocode shape, once stated as a set — must enrich identically,
    including the geocode-driven affectsSameRegion links that share a county.
    """
    shared = ["012001", "012003"]
    counties = {
        "http://example.org/alert/r1": shared + ["012005"],
        "http://example.org/alert/r2": shared + ["012007"],
    }

    def enrich(restate_block):
        rows = []
        for alert, codes in counties.items():
            rows.extend(
                _alert_over_geocodes(alert, codes, restate_block=restate_block)
            )
        return _triple_set(NOAAIntraSourceLinker(spark).enrich(make_triples(rows)))

    restated = enrich(restate_block=True)
    stated_once = enrich(restate_block=False)

    assert restated == stated_once
    # Guard the comparison itself: an all-empty result would satisfy it.
    assert restated


# ======================================================================
# Deep: temporal sequencing keys on the geocode, not the description
# ======================================================================

def test_precedes_chains_alerts_sharing_a_county(spark, make_triples):
    """Two alerts over the same FIPS code are chained oldest -> newest."""
    a1, a2 = "http://example.org/alert/t1", "http://example.org/alert/t2"
    rows = (
        _alert_over_geocodes(a1, ["012001"], restate_block=False,
                             sent_time="2024-11-15T10:00:00")
        + _alert_over_geocodes(a2, ["012001"], restate_block=False,
                               sent_time="2024-11-15T11:00:00")
    )

    assert _precedes(_enrich_rows(spark, make_triples, rows)) == {(a1, a2)}


def test_precedes_does_not_chain_alerts_in_different_counties(spark, make_triples):
    """No shared code means no sequence, however close in time."""
    a1, a2 = "http://example.org/alert/u1", "http://example.org/alert/u2"
    rows = (
        _alert_over_geocodes(a1, ["012001"], restate_block=False,
                             sent_time="2024-11-15T10:00:00")
        + _alert_over_geocodes(a2, ["012099"], restate_block=False,
                               sent_time="2024-11-15T11:00:00")
    )

    assert _precedes(_enrich_rows(spark, make_triples, rows)) == set()


def test_precedes_fires_when_area_descriptions_differ(spark, make_triples):
    """The regression #339 is about: same county, different free-text lists.

    Keying on cap:hasAreaDescription made this pair invisible — the strings
    differ, so the two alerts landed in different window partitions and neither
    got a successor. The geocode they share is the fact that matters.
    """
    a1, a2 = "http://example.org/alert/v1", "http://example.org/alert/v2"
    rows = (
        _alert_over_geocodes(a1, ["013083"], restate_block=False,
                             sent_time="2024-11-15T10:00:00",
                             area_desc="Dade; Walker; Catoosa")
        + _alert_over_geocodes(a2, ["013083"], restate_block=False,
                               sent_time="2024-11-15T11:00:00",
                               area_desc="Dade; Walker; Catoosa; Whitfield")
    )

    assert _precedes(_enrich_rows(spark, make_triples, rows)) == {(a1, a2)}


def test_precedes_emits_one_triple_for_a_pair_adjacent_in_several_counties(
    spark, make_triples
):
    """An alert covering many counties appears in many chains, not many edges.

    Sequencing per geocode means the same ordered pair can be adjacent in every
    county they share. That is one fact, so it must be stated once.
    """
    a1, a2 = "http://example.org/alert/w1", "http://example.org/alert/w2"
    shared = ["012001", "012003", "012005"]
    rows = (
        _alert_over_geocodes(a1, shared, restate_block=False,
                             sent_time="2024-11-15T10:00:00")
        + _alert_over_geocodes(a2, shared, restate_block=False,
                               sent_time="2024-11-15T11:00:00")
    )

    triples = _enrich_rows(spark, make_triples, rows)
    assert _precedes(triples) == {(a1, a2)}
    # One triple, not one per shared county.
    assert len([1 for _, p, _ in triples if p == PRECEDES_PRED]) == 1


def test_precedes_is_deterministic_when_sent_times_tie(spark, make_triples):
    """Identical timestamps must still yield a stable, single ordering.

    sent_time alone is not a total order — alerts are issued in the same second
    — and lead() over a non-deterministic order would vary run to run, which the
    pipeline's reproducibility guarantee forbids. The tie breaks on the alert
    URI, so the lower URI precedes the higher one.
    """
    a1, a2 = "http://example.org/alert/x1", "http://example.org/alert/x2"
    same_time = "2024-11-15T10:00:00"
    rows = (
        _alert_over_geocodes(a1, ["012001"], restate_block=False,
                             sent_time=same_time)
        + _alert_over_geocodes(a2, ["012001"], restate_block=False,
                               sent_time=same_time)
    )

    runs = [_precedes(_enrich_rows(spark, make_triples, rows)) for _ in range(2)]
    assert runs[0] == runs[1] == {(a1, a2)}


def test_alerts_sharing_only_a_cap_category_are_not_linked(spark, make_triples):
    """Two alerts alike in nothing but cap:hasCategory get no edge between them.

    Every alert the weather feed publishes is cap:Met, so linking on the
    category alone was a complete graph over the window: 795,691 edges on one
    real day's 1,262 alerts, 90% of everything NOAA contributed, growing as the
    square of the alert count. The pairs asserted nothing -- both endpoints
    shared a value every alert shares -- and no code read the predicate.

    A tornado in Georgia and a winter storm in Texas share the category and
    nothing else, so they must end up unconnected.
    """
    a1, a2 = "http://example.org/alert/g1", "http://example.org/alert/g2"
    rows = (
        _alert_over_geocodes(a1, ["013083"], restate_block=False,
                             event="Tornado Warning")
        + _alert_over_geocodes(a2, ["048201"], restate_block=False,
                               event="Winter Storm Warning")
    )

    # Guard the premise: the two really do share a category in the input.
    categories = {o for s, p, o in rows if p == HAS_CATEGORY}
    assert categories == {str(CAP.Met)}

    triples = _enrich_rows(spark, make_triples, rows)

    assert not [
        (s, p, o) for s, p, o in triples if s in (a1, a2) and o in (a1, a2)
    ], "alerts sharing only a CAP category must not be linked"
    # Each is still enriched on its own terms -- this is not an empty result.
    assert {s for s, p, _ in triples if p == HAS_EVENT_CLASSIFICATION} == {a1, a2}


def test_noaa_triples_keeps_noaa_deduplicated_and_drops_the_rest(
    spark, make_triples
):
    """The narrowing gate: NOAA's predicates in, as a set; everything else out.

    rdf:type is the one predicate that cannot be matched by name — every source
    emits it — so it is admitted only when its object is the WeatherAlert type.
    A foreign type triple riding in would put non-NOAA subjects into the alert
    table.
    """
    alert = "http://example.org/alert/n1"
    rows = _alert_over_geocodes(alert, ["012001", "012003"], restate_block=True)
    foreign = [
        ("http://example.org/equity/1", RDF_TYPE, "http://example.org/EquitySnapshot"),
        ("http://example.org/equity/1", "http://example.org/symbol", "AAPL"),
    ]

    narrowed = NOAAIntraSourceLinker(spark)._noaa_triples(make_triples(rows + foreign))
    kept = _triple_set(narrowed)

    assert kept == set(rows)
    assert narrowed.count() == len(kept), "narrowed frame still carries duplicates"
    assert not kept & set(foreign)
    assert {p for _, p, _ in kept} <= set(NOAA_PREDICATES) | {RDF_TYPE}
