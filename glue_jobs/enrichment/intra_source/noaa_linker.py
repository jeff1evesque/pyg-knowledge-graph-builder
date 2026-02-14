"""
NOAA Intra-Source Enrichment Orchestrator (PySpark)

Coordinates enrichment for NOAA weather alerts (CAP 1.2 format).
All enrichment runs as distributed PySpark DataFrame operations.

Enrichment strategies:
1. Link temporal sequences of alerts per geographic area (precedes)
2. Link alerts affecting same geographic regions (via SAME codes)
3. Link alerts of the same event type (sameEventType)
4. Link severity escalations within same area over time (escalatesTo)

Note: Temporal unification is handled separately by TemporalUnifier
in pipeline.py Phase 2 — not called from here.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from functools import reduce
from typing import List, Optional

from glue_jobs.utils.rdf_utils import (
    CAP, NWS, NOAA_ENRICHMENT
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

# CAP class URIs
ALERT_TYPE = str(CAP.Alert)
INFO_TYPE = str(CAP.Info)
AREA_TYPE = str(CAP.Area)
GEOCODE_TYPE = str(CAP.Geocode)

# CAP property URIs
HAS_SENT_TIME = str(CAP.hasSentTime)
HAS_INFO = str(CAP.hasInfo)
HAS_AREA = str(CAP.hasArea)
HAS_AREA_DESC = str(CAP.hasAreaDescription)
HAS_EVENT = str(CAP.hasEvent)
HAS_SEVERITY = str(CAP.hasSeverity)
HAS_GEOCODE = str(CAP.hasGeocode)

# NWS geocode property URIs
HAS_SAME_CODE = str(NWS.hasSAMECode)

# Enrichment property URIs
PRECEDES_PRED = str(NOAA_ENRICHMENT.precedes)
AFFECTS_SAME_REGION_PRED = str(NOAA_ENRICHMENT.affectsSameRegion)
SAME_EVENT_TYPE_PRED = str(NOAA_ENRICHMENT.sameEventType)
ESCALATES_TO_PRED = str(NOAA_ENRICHMENT.escalatesTo)

# Severity hierarchy — higher number = more severe
# The severity objects in the triples are URIs like cap:Minor, cap:Moderate, etc.
SEVERITY_URIS = {
    str(CAP.Minor): 1,
    str(CAP.Moderate): 2,
    str(CAP.Severe): 3,
    str(CAP.Extreme): 4,
}


class NOAAIntraSourceLinker:
    """
    NOAA intra-source enrichment using PySpark DataFrames.

    Each method reads from triples_df, produces a DataFrame of new triples
    (subject, predicate, object), and returns it. The enrich() method unions
    all new triples together.
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
        has_noaa = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == ALERT_TYPE)
        ).limit(1).count() > 0

        if not has_noaa:
            logger.info("No NOAA data detected, skipping enrichment")
            return empty

        logger.info("=" * 60)
        logger.info("Starting NOAA Intra-Source Enrichment (PySpark)")
        logger.info("=" * 60)

        triples_df.cache()

        # Build the alert info table once — used by multiple steps.
        # Structure: (alert, info, area, area_desc, sent_time, event, severity_uri)
        alert_info_df = self._build_alert_info(triples_df)

        if alert_info_df is None:
            logger.info("No fully-specified alerts found")
            return empty

        alert_info_df.cache()

        new_dfs: List[DataFrame] = []

        logger.info("[Step 1/4] Linking temporal sequences...")
        df = self._link_temporal_sequences(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/4] Linking geographic relationships...")
        df = self._link_geographic_relationships(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 3/4] Linking event type relationships...")
        df = self._link_event_relationships(alert_info_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 4/4] Linking severity escalations...")
        df = self._link_severity_escalations(alert_info_df)
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
    # Shared: Build alert info table
    # ================================================================

    def _build_alert_info(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Build a denormalized table of alert metadata by joining through
        the CAP structure: Alert → Info → Area, Event, Severity.

        Returns DataFrame with columns:
            alert, info, area, area_desc, sent_time, event, severity_uri
        """
        # Alerts
        alerts = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == ALERT_TYPE)
        ).select(F.col("subject").alias("alert"))

        # Alert → sent time
        sent_times = triples_df.filter(
            F.col("predicate") == HAS_SENT_TIME
        ).select(
            F.col("subject").alias("alert"),
            F.col("object").alias("sent_time"),
        )

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

        # Info → severity (URI like cap:Minor)
        severities = triples_df.filter(
            F.col("predicate") == HAS_SEVERITY
        ).select(
            F.col("subject").alias("info"),
            F.col("object").alias("severity_uri"),
        )

        # Join everything together
        result = (
            alerts
            .join(sent_times, "alert", "inner")
            .join(alert_infos, "alert", "inner")
            .join(info_areas, "info", "left")
            .join(area_descs, "area", "left")
            .join(events, "info", "left")
            .join(severities, "info", "left")
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
    # Step 2: Geographic Relationships (via SAME codes)
    # ================================================================

    def _link_geographic_relationships(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link alerts that share SAME geocodes (same geographic region).

        Traverses: Alert → Info → Area → Geocode → hasSAMECode
        Then self-joins on SAME code to find alert pairs.

        Uses alert1 < alert2 to produce each pair exactly once.
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

        # Geocode → SAME code
        same_codes = triples_df.filter(
            F.col("predicate") == HAS_SAME_CODE
        ).select(
            F.col("subject").alias("geocode"),
            F.col("object").alias("same_code"),
        )

        # Join: alert → same_code
        alert_same = (
            alert_infos
            .join(info_areas, "info", "inner")
            .join(area_geocodes, "area", "inner")
            .join(same_codes, "geocode", "inner")
            .select("alert", "same_code")
            .dropDuplicates()
        )

        if alert_same.head(1) == []:
            logger.info("  No SAME codes found")
            return None

        # Self-join on same_code, alert1 < alert2
        left = alert_same.select(
            F.col("alert").alias("alert1"),
            F.col("same_code").alias("sc1"),
        )
        right = alert_same.select(
            F.col("alert").alias("alert2"),
            F.col("same_code").alias("sc2"),
        )

        pairs = left.join(
            right,
            (left.sc1 == right.sc2) & (left.alert1 < right.alert2),
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
        """
        # Only alerts with area_desc, sent_time, and severity
        escalatable = alert_info_df.filter(
            F.col("area_desc").isNotNull()
            & F.col("sent_time").isNotNull()
            & F.col("severity_uri").isNotNull()
        ).select(
            "alert", "area_desc", "sent_time", "severity_uri"
        ).dropDuplicates()

        if escalatable.head(1) == []:
            return None

        # Map severity URIs to numeric levels via broadcast join
        severity_rows = list(SEVERITY_URIS.items())
        severity_df = self.spark.createDataFrame(
            severity_rows, ["severity_uri", "severity_level"]
        )

        escalatable = escalatable.join(
            F.broadcast(severity_df), "severity_uri", "inner"
        )

        if escalatable.head(1) == []:
            return None

        # Window: partition by area, order by time
        w = Window.partitionBy("area_desc").orderBy("sent_time")

        with_next = escalatable.withColumn(
            "next_alert", F.lead("alert").over(w)
        ).withColumn(
            "next_severity", F.lead("severity_level").over(w)
        ).filter(
            F.col("next_alert").isNotNull()
            & F.col("next_severity").isNotNull()
        )

        # Only keep pairs where severity increases
        escalations = with_next.filter(
            F.col("next_severity") > F.col("severity_level")
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