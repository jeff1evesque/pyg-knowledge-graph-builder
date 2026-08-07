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
6. Link alerts sharing the same CAP category (sameCategory)

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
# The mapper types alerts as nws:WeatherAlert (subClassOf cap:Alert)
WEATHER_ALERT_TYPE = str(WEATHER.WeatherAlert)
ALERT_TYPE = str(CAP.Alert)
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

# Enrichment property URIs
PRECEDES_PRED = str(NOAA_ENRICHMENT.precedes)
AFFECTS_SAME_REGION_PRED = str(NOAA_ENRICHMENT.affectsSameRegion)
SAME_EVENT_TYPE_PRED = str(NOAA_ENRICHMENT.sameEventType)
ESCALATES_TO_PRED = str(NOAA_ENRICHMENT.escalatesTo)
SAME_CATEGORY_PRED = str(NOAA_ENRICHMENT.sameCategory)
HAS_EVENT_CLASSIFICATION = str(NOAA_ENRICHMENT.hasEventClassification)


class NOAAIntraSourceLinker:
    """
    NOAA intra-source enrichment using PySpark DataFrames.

    Each method reads from triples_df, produces a DataFrame of new triples
    (subject, predicate, object), and returns it. The enrich() method unions
    all new triples together.

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
        # Also check for cap:Alert in case of mixed data or future changes.
        has_noaa = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (
                (F.col("object") == WEATHER_ALERT_TYPE)
                | (F.col("object") == ALERT_TYPE)
            )
        ).limit(1).count() > 0

        if not has_noaa:
            logger.info("No NOAA data detected, skipping enrichment")
            return empty

        logger.info("=" * 60)
        logger.info("Starting NOAA Intra-Source Enrichment (PySpark)")
        logger.info("=" * 60)

        triples_df.cache()

        # Build the denormalized alert info table once — used by multiple steps.
        # Structure: (alert, info, sent_time, event, severity_uri,
        #             urgency_uri, certainty_uri, category_uri,
        #             area, area_desc)
        alert_info_df = self._build_alert_info(triples_df)

        if alert_info_df is None:
            logger.info("No fully-specified alerts found")
            return empty

        alert_info_df.cache()

        new_dfs: List[DataFrame] = []

        logger.info("[Step 1/6] Linking temporal sequences...")
        df = self._link_temporal_sequences(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/6] Linking geographic relationships...")
        df = self._link_geographic_relationships(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 3/6] Linking event type relationships...")
        df = self._link_event_relationships(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 4/6] Linking severity escalations...")
        df = self._link_severity_escalations(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 5/6] Classifying alerts by event category...")
        df = self._classify_event_categories(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 6/6] Linking alerts by CAP category...")
        df = self._link_by_cap_category(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        alert_info_df.unpersist()

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
    # Shared: Build denormalized alert info table
    # ================================================================

    def _build_alert_info(self, triples_df: DataFrame) -> Optional[DataFrame]:
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

        Returns DataFrame with columns:
            alert, info, sent_time, event, severity_uri, urgency_uri,
            certainty_uri, category_uri, area, area_desc
        """
        # Alerts — detect both nws:WeatherAlert and cap:Alert
        alerts = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (
                (F.col("object") == WEATHER_ALERT_TYPE)
                | (F.col("object") == ALERT_TYPE)
            )
        ).select(F.col("subject").alias("alert"))

        # Alert → Info (cap:hasInfo)
        alert_infos = triples_df.filter(
            F.col("predicate") == HAS_INFO
        ).select(
            F.col("subject").alias("alert"),
            F.col("object").alias("info"),
        )

        # Info → sent time (cap:hasSentTime is on Info subject per RML mapper)
        sent_times = triples_df.filter(
            F.col("predicate") == HAS_SENT_TIME
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("sent_time"),
        )

        # Info → Area (cap:hasArea)
        info_areas = triples_df.filter(
            F.col("predicate") == HAS_AREA
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("area"),
        )

        # Area → area description
        area_descs = triples_df.filter(
            F.col("predicate") == HAS_AREA_DESC
        ).select(
            F.col("subject").alias("area"),
            F.col("object").alias("area_desc"),
        )

        # Info → event (literal string like "Flood Advisory")
        events = triples_df.filter(
            F.col("predicate") == HAS_EVENT
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("event"),
        )

        # Info → severity (URI like cap:Severe)
        severities = triples_df.filter(
            F.col("predicate") == HAS_SEVERITY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("severity_uri"),
        )

        # Info → urgency (URI like cap:Immediate)
        urgencies = triples_df.filter(
            F.col("predicate") == HAS_URGENCY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("urgency_uri"),
        )

        # Info → certainty (URI like cap:Observed)
        certainties = triples_df.filter(
            F.col("predicate") == HAS_CERTAINTY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("certainty_uri"),
        )

        # Info → category (URI like cap:Met)
        categories = triples_df.filter(
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
    # Step 1: Temporal Sequences
    # ================================================================

    def _link_temporal_sequences(
        self, alert_info_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts in chronological order per geographic area.

        For each area_desc, orders alerts by sent_time and produces:
            alert_N  noaa_enrichment:precedes  alert_N+1

        Uses the denormalized alert_info_df where sent_time comes from
        the Info subject (joined through cap:hasInfo).
        """
        # Only alerts with area_desc and sent_time
        sequenceable = alert_info_df.filter(
            F.col("area_desc").isNotNull() & F.col("sent_time").isNotNull()
        ).select("alert", "area_desc", "sent_time").dropDuplicates()

        if sequenceable.head(1) == []:
            return None

        w = Window.partitionBy("area_desc").orderBy("sent_time")
        sequenced = sequenceable.withColumn(
            "next_alert", F.lead("alert").over(w)
        ).filter(F.col("next_alert").isNotNull())

        if sequenced.head(1) == []:
            return None

        result = sequenced.select(
            F.col("alert").alias("subject"),
            F.lit(PRECEDES_PRED).alias("predicate"),
            F.col("next_alert").alias("object"),
        )

        logger.info("  Temporal sequence linking complete")
        return result

    # ================================================================
    # Step 2: Geographic Relationships (via FIPS/SAME codes)
    # ================================================================

    def _link_geographic_relationships(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts that share FIPS or SAME geocodes (same geographic region).

        Traverses the RML mapper structure:
            Alert → cap:hasInfo → Info → cap:hasArea → Area
                → cap:hasGeocode → Geocode → nws:hasFIPSCode / nws:hasSAMECode

        Then self-joins on FIPS/SAME code to find alert pairs.
        Uses alert1 < alert2 to produce each pair exactly once.

        The RML mapper produces geocode URIs like:
            alert:{alert_id}#geocode-{fips_code}
        with nws:hasFIPSCode and nws:hasSAMECode both set to the FIPS code.
        """
        # Alert → Info
        alert_infos = triples_df.filter(
            F.col("predicate") == HAS_INFO
        ).select(
            F.col("subject").alias("alert"),
            F.col("object").alias("info"),
        )

        # Info → Area
        info_areas = triples_df.filter(
            F.col("predicate") == HAS_AREA
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("area"),
        )

        # Area → Geocode
        area_geocodes = triples_df.filter(
            F.col("predicate") == HAS_GEOCODE
        ).select(
            F.col("subject").alias("area"),
            F.col("object").alias("geocode"),
        )

        # Geocode → FIPS code (primary geographic identifier in new mapper)
        # Also check SAME code for backward compatibility
        fips_codes = triples_df.filter(
            (F.col("predicate") == HAS_FIPS_CODE)
            | (F.col("predicate") == HAS_SAME_CODE)
        ).select(
            F.col("subject").alias("geocode"),
            F.col("object").alias("geo_code"),
        ).dropDuplicates()

        # Join: alert → geo_code
        alert_geo = (
            alert_infos
            .join(info_areas, "info", "inner")
            .join(area_geocodes, "area", "inner")
            .join(fips_codes, "geocode", "inner")
            .select("alert", "geo_code")
            .dropDuplicates()
        )

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
        self, alert_info_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link severity escalations for the same geographic area.

        For each area_desc, orders alerts by sent_time, then links
        consecutive alerts where severity increases:
            alert_N  noaa_enrichment:escalatesTo  alert_N+1
            (only when severity_level(N+1) > severity_level(N))

        Severity hierarchy: Minor(1) < Moderate(2) < Severe(3) < Extreme(4)

        Severity URIs are named individuals (e.g., cap:Severe) produced
        by the RML mapper's enum mapping. The SEVERITY_HIERARCHY dict
        maps these URIs to numeric levels.

        Also considers urgency as a secondary escalation signal:
        if severity is the same but urgency increases, that is also
        treated as an escalation.
        """
        # Only alerts with area_desc, sent_time, and severity
        escalatable = alert_info_df.filter(
            F.col("area_desc").isNotNull()
            & F.col("sent_time").isNotNull()
            & F.col("severity_uri").isNotNull()
        ).select(
            "alert", "area_desc", "sent_time", "severity_uri", "urgency_uri"
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

        if escalatable.head(1) == []:
            return None

        # Window: partition by area, order by time
        w = Window.partitionBy("area_desc").orderBy("sent_time")

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

        result = escalations.select(
            F.col("alert").alias("subject"),
            F.lit(ESCALATES_TO_PRED).alias("predicate"),
            F.col("next_alert").alias("object"),
        )

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

    # ================================================================
    # Step 6: CAP Category Linking
    # ================================================================

    def _link_by_cap_category(
        self, alert_info_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts that share the same CAP category (cap:hasCategory).

        CAP categories are URI-valued named individuals (e.g., cap:Met,
        cap:Geo, cap:Fire) produced by the RML mapper's enum mapping.

        Uses alert1 < alert2 to produce each pair exactly once.
        """
        alert_categories = alert_info_df.filter(
            F.col("category_uri").isNotNull()
        ).select("alert", "category_uri").dropDuplicates()

        if alert_categories.head(1) == []:
            return None

        left = alert_categories.select(
            F.col("alert").alias("alert1"),
            F.col("category_uri").alias("cat1"),
        )
        right = alert_categories.select(
            F.col("alert").alias("alert2"),
            F.col("category_uri").alias("cat2"),
        )

        pairs = left.join(
            right,
            (left.cat1 == right.cat2) & (left.alert1 < right.alert2),
            "inner",
        ).select("alert1", "alert2").dropDuplicates()

        if pairs.head(1) == []:
            return None

        result = pairs.select(
            F.col("alert1").alias("subject"),
            F.lit(SAME_CATEGORY_PRED).alias("predicate"),
            F.col("alert2").alias("object"),
        )

        logger.info("  CAP category linking complete")
        return result