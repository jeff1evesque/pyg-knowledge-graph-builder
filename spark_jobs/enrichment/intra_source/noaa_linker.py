"""
NOAA Intra-Source Enrichment Orchestrator (PySpark)

Coordinates enrichment for NOAA weather alerts (CAP 1.2 / WEATHER format).
All enrichment runs as distributed PySpark DataFrame operations.

Aligned with CAP 1.2 ontology v3.0 and the WEATHER RML mapper which produces:
  - Alert subjects:  alert:{alert_id}         typed as nws:WeatherAlert
  - Info subjects:   alert:{alert_id}#info     typed as cap:Info
  - Area subjects:   alert:{alert_id}#area     typed as cap:Area
  - Geocode subjects: alert:{alert_id}#geocode-{fips_code} typed as cap:Geocode

Key structural notes from the RML mapper:
  - cap:hasSentTime is on the Info subject (not the Alert subject)
  - cap:hasInfo links Alert → Info
  - cap:hasArea links Info → Area
  - cap:hasGeocode links Area → Geocode
  - Severity, urgency, certainty, category, responseType, status,
    messageType, scope are all URI-valued (named individuals like
    cap:Severe, cap:Immediate, cap:Met)
  - cap:hasEvent is a literal string on Info (e.g., "Tornado Warning")
  - Geocodes use nws:hasFIPSCode, nws:hasStateFIPS, nws:hasCountyFIPS,
    nws:hasUGCCode, and nws:hasSAMECode

Enrichment strategies:
1. Link temporal sequences of alerts per geographic area (precedes)
2. Link alerts affecting same geographic regions (via FIPS/SAME codes)
3. Link alerts of the same event type (sameEventType)
4. Link severity escalations within same area over time (escalatesTo)
5. Classify alerts by event category (severeWeatherEvent, floodEvent, etc.)

Note: Temporal unification is handled separately by TemporalUnifier
in pipeline.py Phase 2 — not called from here.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from functools import reduce
from typing import List, Optional

from spark_jobs.utils.rdf_utils import (
    CAP, WEATHER, NOAA_ENRICHMENT
)
from spark_jobs.enrichment.intra_source.noaa.patterns import (
    NOAA_EVENT_PATTERNS,
    SEVERITY_HIERARCHY,
    URGENCY_HIERARCHY,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

# Class URIs — aligned with RML mapper output
#
# nws:WeatherAlert is the only type the mapper writes on an alert node.
# cap:Alert was carried alongside it "in case of mixed data": it is declared in
# the CAP model, emitted by nothing, and shares its rdfs:label "Alert" with the
# named individual cap:AlertMessage that cap:hasMessageType points at -- so the
# fallback could never fire and made the real type look optional.
WEATHER_ALERT_TYPE = str(WEATHER.WeatherAlert)
INFO_TYPE = str(CAP.Info)
AREA_TYPE = str(CAP.Area)
GEOCODE_TYPE = str(CAP.Geocode)

# Alert-level property URIs
HAS_INFO = str(CAP.hasInfo)
HAS_STATUS = str(CAP.hasStatus)
HAS_MESSAGE_TYPE = str(CAP.hasMessageType)

# Info-level property URIs
# NOTE: In the RML mapper, hasSentTime is mapped on the Info subject,
# not the Alert subject. The enricher must join through Info to get
# the sent time for an alert.
HAS_SENT_TIME = str(CAP.hasSentTime)
HAS_AREA = str(CAP.hasArea)
HAS_AREA_DESC = str(CAP.hasAreaDescription)
HAS_EVENT = str(CAP.hasEvent)
HAS_SEVERITY = str(CAP.hasSeverity)
HAS_URGENCY = str(CAP.hasUrgency)
HAS_CERTAINTY = str(CAP.hasCertainty)
HAS_CATEGORY = str(CAP.hasCategory)
HAS_RESPONSE_TYPE = str(CAP.hasResponseType)
HAS_HEADLINE = str(CAP.hasHeadline)
HAS_EFFECTIVE_TIME = str(CAP.hasEffectiveTime)
HAS_ONSET_TIME = str(CAP.hasOnsetTime)
HAS_EXPIRATION_TIME = str(CAP.hasExpirationTime)
HAS_ENDS_TIME = str(CAP.hasEndsTime)

# Area-level property URIs
HAS_GEOCODE = str(CAP.hasGeocode)

# Geocode property URIs — aligned with RML mapper
HAS_SAME_CODE = str(WEATHER.hasSAMECode)
HAS_FIPS_CODE = str(WEATHER.hasFIPSCode)
HAS_STATE_FIPS = str(WEATHER.hasStateFIPS)
HAS_COUNTY_FIPS = str(WEATHER.hasCountyFIPS)
HAS_UGC_CODE = str(WEATHER.hasUGCCode)

# Status/MessageType named individuals for filtering
STATUS_ACTUAL = str(CAP.Actual)
MSG_TYPE_CANCEL = str(CAP.Cancel)

# Every predicate this linker reads, used to narrow the graph to NOAA's own
# triples in one pass. rdf:type is NOT in the list -- every source uses it, so
# it is matched on its object instead (see _noaa_triples).
NOAA_PREDICATES = (
    HAS_INFO,
    HAS_SENT_TIME,
    HAS_AREA,
    HAS_AREA_DESC,
    HAS_EVENT,
    HAS_SEVERITY,
    HAS_URGENCY,
    HAS_CERTAINTY,
    HAS_CATEGORY,
    HAS_GEOCODE,
    HAS_FIPS_CODE,
    HAS_SAME_CODE,
)

# Enrichment property URIs
PRECEDES_PRED = str(NOAA_ENRICHMENT.precedes)
AFFECTS_SAME_REGION_PRED = str(NOAA_ENRICHMENT.affectsSameRegion)
SAME_EVENT_TYPE_PRED = str(NOAA_ENRICHMENT.sameEventType)
ESCALATES_TO_PRED = str(NOAA_ENRICHMENT.escalatesTo)
HAS_EVENT_CLASSIFICATION = str(NOAA_ENRICHMENT.hasEventClassification)


class NOAAIntraSourceLinker:
    """
    NOAA intra-source enrichment using PySpark DataFrames.

    Each method reads from the narrowed NOAA frame _noaa_triples() builds,
    produces a DataFrame of new triples (subject, predicate, object), and
    returns it. The enrich() method unions all new triples together. Only the
    "is there NOAA data at all" probe touches the full graph.

    The RML mapper produces a 3-level structure:
        Alert (nws:WeatherAlert) → Info (cap:Info) → Area (cap:Area) → Geocode (cap:Geocode)

    Most enrichment operates at the Alert level (linking alerts to each other),
    but properties like hasSentTime, hasEvent, hasSeverity are on the Info
    subject. The _build_alert_info() method denormalizes this structure into
    a flat table for efficient enrichment joins.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def enrich(self, triples_df: DataFrame) -> DataFrame:
        """
        Run all NOAA intra-source enrichment steps.

        Args:
            triples_df: DataFrame with columns (subject, predicate, object)

        Returns:
            DataFrame of NEW triples to union with triples_df.
        """
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        # Quick check: is there any NOAA data?
        # The RML mapper types alerts as nws:WeatherAlert.
        has_noaa = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == WEATHER_ALERT_TYPE)
        ).limit(1).count() > 0

        if not has_noaa:
            logger.info("No NOAA data detected, skipping enrichment")
            return empty

        logger.info("=" * 60)
        logger.info("Starting NOAA Intra-Source Enrichment (PySpark)")
        logger.info("=" * 60)

        # NOAA's own triples, as a set, read once. Everything below joins
        # against this rather than the whole graph -- see _noaa_triples for why
        # the deduplication is what makes the joins finish at all.
        noaa_df = self._noaa_triples(triples_df).cache()

        # Build the denormalized alert info table once — used by multiple steps.
        # Structure: (alert, info, sent_time, event, severity_uri,
        #             urgency_uri, certainty_uri, category_uri,
        #             area, area_desc)
        alert_info_df = self._build_alert_info(noaa_df)

        if alert_info_df is None:
            logger.info("No fully-specified alerts found")
            noaa_df.unpersist()
            return empty

        alert_info_df.cache()

        # Materialized and counted HERE, before any step reads it. Six steps
        # join against this table, so if it is wrong every one of them is, and
        # a count is the one number that says so. When the table exploded, the
        # job had logged "[Step 1/6]" and then went silent for as long as it was
        # left running -- the row count was the missing evidence, and it costs
        # one pass over a table that holds one row per alert.
        alert_count = alert_info_df.count()
        logger.info(f"Denormalized alert table: {alert_count:,} rows")

        # Where each alert applies, as structured codes. Steps 1, 2 and 4 all
        # read it, so it is derived once and cached rather than rebuilt per
        # step -- the walk is four filters and three joins.
        alert_geo_df = self._alert_geocodes(noaa_df).cache()

        new_dfs: List[DataFrame] = []

        logger.info("[Step 1/5] Linking temporal sequences...")
        df = self._link_temporal_sequences(alert_info_df, alert_geo_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/5] Linking geographic relationships...")
        df = self._link_geographic_relationships(alert_geo_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 3/5] Linking event type relationships...")
        df = self._link_event_relationships(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 4/5] Linking severity escalations...")
        df = self._link_severity_escalations(alert_info_df, alert_geo_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 5/5] Classifying alerts by event category...")
        df = self._classify_event_categories(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        alert_info_df.unpersist()
        alert_geo_df.unpersist()
        noaa_df.unpersist()

        if not new_dfs:
            logger.info("No enrichment triples produced")
            return empty

        result = reduce(DataFrame.unionAll, new_dfs).cache()
        count = result.count()

        logger.info("=" * 60)
        logger.info(f"NOAA Intra-Source Enrichment Complete: {count} triples")
        logger.info("=" * 60)

        return result

    # ================================================================
    # Shared: narrow the graph to NOAA's triples, once
    # ================================================================

    def _noaa_triples(self, triples_df: DataFrame) -> DataFrame:
        """NOAA's own triples, deduplicated -- the frame every step reads.

        Two things happen here, and only one of them is about speed.

        **dropDuplicates is load-bearing.** RDF is a set, and the rest of the
        pipeline treats it as one: EnrichmentPipeline unions each phase's output
        and dropDuplicates over the whole frame. But that runs AFTER intra-source
        enrichment, so a linker sees the source frame with its repeats intact --
        and the NOAA source repeats heavily. Its archive stores one row per
        (alert, geocode) pair, and every row's Turtle blob restates the alert,
        info and area blocks in full; only the geocode sub-block differs. An
        alert covering 72 counties therefore arrives as 72 identical copies of
        each of its ten alert/info/area triples.

        _build_alert_info joins ten such relations. Ten legs each repeated k
        times is k**10 rows for that alert, so the 72-county alert alone expands
        to 3.7e18 rows. Measured over one real day (2026-04-15: 4,984 parquet
        rows, 1,262 alerts): 3,782,235,540,923,562,898 rows out of a table that
        should hold 1,262. That is the "stalls at Step 1/6" symptom -- the job
        was not slow, it was building a table that cannot be built. Nothing
        detected it because _build_alert_info's own head(1) probe takes one row
        off one partition and returns instantly; the first full evaluation is
        the dropDuplicates inside _link_temporal_sequences, which is where the
        log line stops.

        Deduplicating here removes it entirely: every leg becomes at most one
        row per (subject, value), and the ten-way join collapses to one row per
        alert (verified on real days -- no per-info predicate carries more than
        one distinct value, and the table lands at exactly one row per alert).
        Nothing downstream changes, because every consumer of the table already
        dropDuplicates the columns it uses.

        **The narrowing is the cost half.** All fourteen relation legs below --
        ten in _build_alert_info, four in _link_geographic_relationships -- used
        to filter the whole accumulated graph, so the linker traversed ~20M rows
        fourteen times to find NOAA's ~70k. One pass now selects them and the
        legs filter that. This matches what the SEC and BLS linkers already do
        (_filter_sec_triples / _filter_bls_triples), and the deduplication is
        what those two get for free from sources that do not repeat rows.

        The predicate list is the gate rather than a subject/object URI prefix:
        NOAA subjects live under a third-party host (api.weather.gov), so there
        is no namespace of ours to key on, and every predicate this linker reads
        is CAP- or WEATHER-namespaced. rdf:type is shared with every other
        source, so the alert type triple is matched on its object instead.
        """
        return triples_df.filter(
            F.col("predicate").isin(*NOAA_PREDICATES)
            | (
                (F.col("predicate") == RDF_TYPE)
                & (F.col("object") == WEATHER_ALERT_TYPE)
            )
        ).dropDuplicates(["subject", "predicate", "object"])

    # ================================================================
    # Shared: Build denormalized alert info table
    # ================================================================

    def _build_alert_info(self, noaa_df: DataFrame) -> Optional[DataFrame]:
        """
        Build a denormalized table of alert metadata by joining through
        the CAP/WEATHER structure produced by the RML mapper:

            Alert (nws:WeatherAlert)
              → cap:hasInfo → Info (cap:Info)
                  → cap:hasSentTime (dateTime literal)
                  → cap:hasEvent (string literal)
                  → cap:hasSeverity (URI: cap:Minor/Moderate/Severe/Extreme)
                  → cap:hasUrgency (URI: cap:Immediate/Expected/Future/Past)
                  → cap:hasCertainty (URI: cap:Observed/Likely/Possible/Unlikely)
                  → cap:hasCategory (URI: cap:Met/Geo/Fire/etc.)
                  → cap:hasArea → Area (cap:Area)
                      → cap:hasAreaDescription (string literal)

        IMPORTANT: In the RML mapper, cap:hasSentTime is on the Info
        subject, not the Alert subject. We join Alert → Info first,
        then read sent_time from the Info subject.

        Takes the frame from _noaa_triples(), NOT the raw graph. Ten relations
        are joined below and not one of them deduplicates, so a repeated source
        triple multiplies through all ten -- k**10 rows for an alert the source
        states k times. Passing the raw frame is what made this unfinishable;
        read _noaa_triples' docstring before changing the argument.

        Returns DataFrame with columns:
            alert, info, sent_time, event, severity_uri, urgency_uri,
            certainty_uri, category_uri, area, area_desc
        """
        # Alerts — nws:WeatherAlert is the only type an alert node carries
        alerts = noaa_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == WEATHER_ALERT_TYPE)
        ).select(F.col("subject").alias("alert"))

        # Alert → Info (cap:hasInfo)
        alert_infos = noaa_df.filter(
            F.col("predicate") == HAS_INFO
        ).select(
            F.col("subject").alias("alert"),
            F.col("object").alias("info"),
        )

        # Info → sent time (cap:hasSentTime is on Info subject per RML mapper)
        sent_times = noaa_df.filter(
            F.col("predicate") == HAS_SENT_TIME
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("sent_time"),
        )

        # Info → Area (cap:hasArea)
        info_areas = noaa_df.filter(
            F.col("predicate") == HAS_AREA
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("area"),
        )

        # Area → area description
        area_descs = noaa_df.filter(
            F.col("predicate") == HAS_AREA_DESC
        ).select(
            F.col("subject").alias("area"),
            F.col("object").alias("area_desc"),
        )

        # Info → event (literal string like "Flood Advisory")
        events = noaa_df.filter(
            F.col("predicate") == HAS_EVENT
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("event"),
        )

        # Info → severity (URI like cap:Severe)
        severities = noaa_df.filter(
            F.col("predicate") == HAS_SEVERITY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("severity_uri"),
        )

        # Info → urgency (URI like cap:Immediate)
        urgencies = noaa_df.filter(
            F.col("predicate") == HAS_URGENCY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("urgency_uri"),
        )

        # Info → certainty (URI like cap:Observed)
        certainties = noaa_df.filter(
            F.col("predicate") == HAS_CERTAINTY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("certainty_uri"),
        )

        # Info → category (URI like cap:Met)
        categories = noaa_df.filter(
            F.col("predicate") == HAS_CATEGORY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("category_uri"),
        )

        # Join everything together:
        # Alert → Info (inner: every alert must have an info)
        # Info → sent_time, event, severity, urgency, certainty, category (left)
        # Info → Area → area_desc (left)
        result = (
            alerts
            .join(alert_infos, "alert", "inner")
            .join(sent_times, "info", "left")
            .join(events, "info", "left")
            .join(severities, "info", "left")
            .join(urgencies, "info", "left")
            .join(certainties, "info", "left")
            .join(categories, "info", "left")
            .join(info_areas, "info", "left")
            .join(area_descs, "area", "left")
        )

        if result.head(1) == []:
            return None

        return result

    # ================================================================
    # Shared: alert → geographic code
    # ================================================================

    def _alert_geocodes(self, noaa_df: DataFrame) -> DataFrame:
        """(alert, geo_code) for every alert, deduplicated.

        Walks the mapper's structure:
            Alert -> cap:hasInfo -> Info -> cap:hasArea -> Area
                  -> cap:hasGeocode -> Geocode -> nws:hasFIPSCode / hasSAMECode

        This is the graph's only STRUCTURED statement of where an alert applies.
        cap:hasAreaDescription is the same fact as free text, and the difference
        matters: both steps that sequence alerts over an area -- precedes and
        escalatesTo -- key on the code, never the description.

        Both code predicates are read and the result deduplicated, because the
        mapper sets hasFIPSCode and hasSAMECode to the same value on the same
        geocode -- reading either alone works, reading both without the dedup
        doubles every row.
        """
        alert_infos = noaa_df.filter(
            F.col("predicate") == HAS_INFO
        ).select(
            F.col("subject").alias("alert"),
            F.col("object").alias("info"),
        )

        info_areas = noaa_df.filter(
            F.col("predicate") == HAS_AREA
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("area"),
        )

        area_geocodes = noaa_df.filter(
            F.col("predicate") == HAS_GEOCODE
        ).select(
            F.col("subject").alias("area"),
            F.col("object").alias("geocode"),
        )

        fips_codes = noaa_df.filter(
            (F.col("predicate") == HAS_FIPS_CODE)
            | (F.col("predicate") == HAS_SAME_CODE)
        ).select(
            F.col("subject").alias("geocode"),
            F.col("object").alias("geo_code"),
        ).dropDuplicates()

        return (
            alert_infos
            .join(info_areas, "info", "inner")
            .join(area_geocodes, "area", "inner")
            .join(fips_codes, "geocode", "inner")
            .select("alert", "geo_code")
            .dropDuplicates()
        )

    # ================================================================
    # Step 1: Temporal Sequences
    # ================================================================

    def _link_temporal_sequences(
        self, alert_info_df: DataFrame, alert_geo_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts in chronological order per geographic area.

        For each geographic code, orders alerts by sent_time and produces:
            alert_N  noaa_enrichment:precedes  alert_N+1

        The partition key is the FIPS/SAME code, NOT cap:hasAreaDescription.
        The description is a free-text list of every county an alert covers --
        "Dade; Walker; Catoosa; Whitfield; ..." -- so two alerts share a
        partition only if their county lists match exactly, string for string.
        Measured over real days, ~80% of descriptions belong to a single alert
        (2026-04-15: 702 of 904 distinct values; 2026-08-22: 891 of 1,073), so
        most partitions held one row, lead() returned null, and the alert got no
        successor. One day of 1,262 alerts produced 358 edges.

        Keying on the geocode sequences alerts per county, which is what
        "chronological order per geographic area" means, and an alert covering
        several counties correctly appears in several chains. The same pair can
        therefore be adjacent in more than one county, so the result is
        deduplicated -- the triple is the same fact however many counties
        witnessed it.

        Ordering breaks ties on the alert URI. sent_time alone is not a total
        order (alerts are issued in the same second), and lead() over a
        non-deterministic order makes the output vary between runs, which the
        pipeline's reproducibility guarantee does not allow.
        """
        sequenceable = (
            alert_info_df
            .filter(F.col("sent_time").isNotNull())
            .select("alert", "sent_time")
            .dropDuplicates()
            .join(alert_geo_df, "alert", "inner")
        )

        if sequenceable.head(1) == []:
            logger.info("  No alerts with a geocode and a sent time")
            return None

        w = Window.partitionBy("geo_code").orderBy("sent_time", "alert")
        sequenced = sequenceable.withColumn(
            "next_alert", F.lead("alert").over(w)
        ).filter(F.col("next_alert").isNotNull())

        if sequenced.head(1) == []:
            return None

        result = sequenced.select(
            F.col("alert").alias("subject"),
            F.lit(PRECEDES_PRED).alias("predicate"),
            F.col("next_alert").alias("object"),
        ).dropDuplicates()

        logger.info("  Temporal sequence linking complete")
        return result

    # ================================================================
    # Step 2: Geographic Relationships (via FIPS/SAME codes)
    # ================================================================

    def _link_geographic_relationships(
        self, alert_geo_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts that share FIPS or SAME geocodes (same geographic region).

        Self-joins _alert_geocodes() on the code to find alert pairs, using
        alert1 < alert2 so each pair is produced exactly once.

        Takes the derived (alert, geo_code) relation rather than the triples
        frame: Step 1 needs the same walk, so it is built once in enrich() and
        cached. The walk itself, and why both code predicates are read, are
        documented on _alert_geocodes.
        """
        alert_geo = alert_geo_df

        if alert_geo.head(1) == []:
            logger.info("  No FIPS/SAME codes found")
            return None

        # Self-join on geo_code, alert1 < alert2
        left = alert_geo.select(
            F.col("alert").alias("alert1"),
            F.col("geo_code").alias("gc1"),
        )
        right = alert_geo.select(
            F.col("alert").alias("alert2"),
            F.col("geo_code").alias("gc2"),
        )

        pairs = left.join(
            right,
            (left.gc1 == right.gc2) & (left.alert1 < right.alert2),
            "inner",
        ).select("alert1", "alert2").dropDuplicates()

        if pairs.head(1) == []:
            return None

        result = pairs.select(
            F.col("alert1").alias("subject"),
            F.lit(AFFECTS_SAME_REGION_PRED).alias("predicate"),
            F.col("alert2").alias("object"),
        )

        logger.info("  Geographic relationship linking complete")
        return result

    # ================================================================
    # Step 3: Event Type Relationships
    # ================================================================

    def _link_event_relationships(
        self, alert_info_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts with the same event type (e.g., all "Flood Advisory" alerts).

        cap:hasEvent is a literal string on the Info subject.
        The denormalized alert_info_df already has the event column
        joined from Info.

        Uses alert1 < alert2 to produce each pair exactly once.
        """
        alert_events = alert_info_df.filter(
            F.col("event").isNotNull()
        ).select("alert", "event").dropDuplicates()

        if alert_events.head(1) == []:
            return None

        left = alert_events.select(
            F.col("alert").alias("alert1"),
            F.col("event").alias("e1"),
        )
        right = alert_events.select(
            F.col("alert").alias("alert2"),
            F.col("event").alias("e2"),
        )

        pairs = left.join(
            right,
            (left.e1 == right.e2) & (left.alert1 < right.alert2),
            "inner",
        ).select("alert1", "alert2").dropDuplicates()

        if pairs.head(1) == []:
            return None

        result = pairs.select(
            F.col("alert1").alias("subject"),
            F.lit(SAME_EVENT_TYPE_PRED).alias("predicate"),
            F.col("alert2").alias("object"),
        )

        logger.info("  Event type linking complete")
        return result

    # ================================================================
    # Step 4: Severity Escalations
    # ================================================================

    def _link_severity_escalations(
        self, alert_info_df: DataFrame, alert_geo_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link severity escalations over the same geographic area.

        For each geographic code, orders alerts by sent_time and links
        consecutive alerts where the hazard got worse:
            alert_N  noaa_enrichment:escalatesTo  alert_N+1
            (only when severity_level(N+1) > severity_level(N))

        Severity hierarchy: Minor(1) < Moderate(2) < Severe(3) < Extreme(4)

        Severity URIs are named individuals (e.g., cap:Severe) produced
        by the RML mapper's enum mapping. The SEVERITY_HIERARCHY dict
        maps these URIs to numeric levels.

        Also considers urgency as a secondary escalation signal:
        if severity is the same but urgency increases, that is also
        treated as an escalation.

        The partition key is the FIPS/SAME code, for the reason set out on
        _link_temporal_sequences -- this step had the identical defect and is
        fixed the identical way. It keyed on cap:hasAreaDescription, the
        free-text county list, so two alerts shared a partition only if their
        lists matched string for string, and ~80% of those lists belong to a
        single alert (2026-04-15: 702 of 904 distinct values; 2026-08-22: 891
        of 1,073). Most partitions held one row, lead() returned null, and no
        escalation was detected. One day of 1,262 alerts produced 56 edges.

        The miss costs more here than it did for precedes. "The warning over
        this county got more severe" is the whole point of the relation, and it
        went undetected whenever the follow-up alert listed a different set of
        counties -- while the county they actually share, the thing that makes
        it the same area, was sitting in the graph as a structured code.
        """
        # Only alerts with a sent time and a severity. Requiring an area_desc
        # is what this step used to do INSTEAD of requiring a geocode; the
        # inner join below is the requirement now, so the description is not
        # read at all any more.
        escalatable = alert_info_df.filter(
            F.col("sent_time").isNotNull()
            & F.col("severity_uri").isNotNull()
        ).select(
            "alert", "sent_time", "severity_uri", "urgency_uri"
        ).dropDuplicates()

        if escalatable.head(1) == []:
            return None

        # Map severity URIs to numeric levels via broadcast join
        severity_rows = list(SEVERITY_HIERARCHY.items())
        severity_df = self.spark.createDataFrame(
            severity_rows, ["severity_uri", "severity_level"]
        )

        escalatable = escalatable.join(
            F.broadcast(severity_df), "severity_uri", "inner"
        )

        # Map urgency URIs to numeric levels via broadcast join (optional)
        urgency_rows = list(URGENCY_HIERARCHY.items())
        urgency_df = self.spark.createDataFrame(
            urgency_rows, ["urgency_uri", "urgency_level"]
        )

        escalatable = escalatable.join(
            F.broadcast(urgency_df), "urgency_uri", "left"
        ).fillna({"urgency_level": 0})

        # Where each alert applies. Joined AFTER both level lookups because
        # this is the step that fans an alert out to one row per county:
        # mapping the levels first does that work once per alert rather than
        # once per (alert, county). The join is inner, so an alert with no
        # geocode drops out here -- it has no area to be sequenced within.
        escalatable = escalatable.join(alert_geo_df, "alert", "inner")

        if escalatable.head(1) == []:
            logger.info("  No alerts with a geocode, a sent time and a severity")
            return None

        # Window: partition by geographic code, order by time.
        # The tie-break on the alert URI matters for the same reason it does in
        # Step 1 -- sent_time alone is not a total order, alerts are issued in
        # the same second, and lead() over a non-deterministic order would make
        # the output vary between runs.
        w = Window.partitionBy("geo_code").orderBy("sent_time", "alert")

        with_next = escalatable.withColumn(
            "next_alert", F.lead("alert").over(w)
        ).withColumn(
            "next_severity", F.lead("severity_level").over(w)
        ).withColumn(
            "next_urgency", F.lead("urgency_level").over(w)
        ).filter(
            F.col("next_alert").isNotNull()
            & F.col("next_severity").isNotNull()
        )

        # Keep pairs where severity increases, OR severity is same but
        # urgency increases (secondary escalation signal)
        escalations = with_next.filter(
            (F.col("next_severity") > F.col("severity_level"))
            | (
                (F.col("next_severity") == F.col("severity_level"))
                & (F.col("next_urgency") > F.col("urgency_level"))
            )
        )

        if escalations.head(1) == []:
            return None

        # One triple per escalating pair. Sequencing per geocode puts an alert
        # in one chain per county it covers, so the same escalation can be
        # adjacent in several of them -- that is one fact, however many
        # counties witnessed it.
        result = escalations.select(
            F.col("alert").alias("subject"),
            F.lit(ESCALATES_TO_PRED).alias("predicate"),
            F.col("next_alert").alias("object"),
        ).dropDuplicates()

        logger.info("  Severity escalation linking complete")
        return result

    # ================================================================
    # Step 5: Event Category Classification
    # ================================================================

    def _classify_event_categories(
        self, alert_info_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Classify alerts by event type using NOAA_EVENT_PATTERNS.

        For each alert whose cap:hasEvent literal matches a known event
        type pattern, produces:
            alert  noaa_enrichment:hasEventClassification  noaa_enrichment:{category}

        Also produces the specific relationship triple:
            alert  noaa_enrichment:{relationship}  noaa_enrichment:{category}

        Event types are matched as exact string equality against the
        literal values produced by the RML mapper's cap:hasEvent property.
        """
        alert_events = alert_info_df.filter(
            F.col("event").isNotNull()
        ).select("alert", "event").dropDuplicates()

        if alert_events.head(1) == []:
            return None

        # Build event type → (category_uri, relationship) lookup
        event_rows = []
        for category_name, pattern in NOAA_EVENT_PATTERNS.items():
            category_uri = str(NOAA_ENRICHMENT[category_name])
            relationship = pattern['relationship']
            for event_type in pattern['event_types']:
                event_rows.append((event_type, category_uri, relationship))

        if not event_rows:
            return None

        event_lookup_df = self.spark.createDataFrame(
            event_rows, ["event_type", "category_uri", "relationship"]
        )

        # Join alerts to event patterns
        matched = alert_events.join(
            F.broadcast(event_lookup_df),
            alert_events.event == event_lookup_df.event_type,
            "inner",
        )

        if matched.head(1) == []:
            return None

        # Classification triples
        classification_triples = matched.select(
            F.col("alert").alias("subject"),
            F.lit(HAS_EVENT_CLASSIFICATION).alias("predicate"),
            F.col("category_uri").alias("object"),
        )

        # Specific relationship triples
        relationship_triples = matched.select(
            F.col("alert").alias("subject"),
            F.col("relationship").alias("predicate"),
            F.col("category_uri").alias("object"),
        )

        result = classification_triples.unionByName(relationship_triples)

        logger.info("  Event category classification complete")
        return result
