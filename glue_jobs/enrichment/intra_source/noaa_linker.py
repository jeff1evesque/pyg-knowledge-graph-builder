"""
NOAA Intra-Source Enrichment Orchestrator
Coordinates enrichment for NOAA weather alerts
"""
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, OWL
from glue_jobs.utils.rdf_utils import (
    NOAA, CAP, NWS, ATOM,
    NOAA_ENRICHMENT, UNIFIED
)
from glue_jobs.enrichment.intra_source.base import IntraSourceEnricher
from glue_jobs.enrichment.temporal_unifier import TemporalUnifier
from typing import Dict, Set
import logging

logger = logging.getLogger(__name__)


class NOAAIntraSourceLinker(IntraSourceEnricher):
    """
    Orchestrates NOAA intra-source enrichment for weather alerts

    NOAA weather alerts use the Common Alerting Protocol (CAP) 1.2 standard
    with NWS-specific extensions. The data structure is:

    - Alert (top level) - contains metadata about the alert message
    - Info (information segment) - contains specific alert details
    - Area (geographic area) - defines affected regions
    - Geocodes (SAME, UGC, FIPS) - specific geographic identifiers
    - Parameters (NWS-specific) - additional alert metadata

    Enrichment strategies:
    1. Temporal sequences - Link alerts by time and geographic area
    2. Geographic relationships - Link alerts affecting same/nearby regions
    3. Event type relationships - Link related weather events
    4. Severity escalation - Link watches → warnings → emergencies
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.stats.update({
            'geographic_links': 0,
            'event_links': 0,
            'severity_links': 0
        })
        self.available_datasets = self.detect_datasets()
        logger.info(f"Detected NOAA datasets: {', '.join(self.available_datasets)}")

    def detect_datasets(self) -> Set[str]:
        """Detect if NOAA weather alert data is present in the graph"""
        datasets = set()

        # Check for CAP alerts
        query = f"""
        ASK {{
            ?s a <{CAP.Alert}> .
        }}
        """
        if self.graph.query(query).askAnswer:
            datasets.add('noaa_alerts')

        return datasets

    def enrich(self) -> Dict[str, int]:
        """Run all NOAA intra-source enrichment steps"""
        if 'noaa_alerts' not in self.available_datasets:
            logger.info("No NOAA data detected, skipping enrichment")
            return {'total_triples_added': 0}

        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting NOAA Intra-Source Enrichment")
        logger.info("=" * 60)

        # Step 1: Unify temporal entities
        logger.info("\n[Step 1/5] Unifying temporal entities...")
        self.unify_temporal_entities()

        # Step 2: Link temporal sequences
        logger.info("\n[Step 2/5] Linking temporal sequences...")
        self.link_temporal_sequences()

        # Step 3: Link geographic relationships
        logger.info("\n[Step 3/5] Linking geographic relationships...")
        self.link_geographic_relationships()

        # Step 4: Link event type relationships
        logger.info("\n[Step 4/5] Linking event type relationships...")
        self.link_event_relationships()

        # Step 5: Link severity escalations
        logger.info("\n[Step 5/5] Linking severity escalations...")
        self.link_severity_escalations()

        final_count = len(self.graph)
        enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("NOAA Intra-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {enrichment_count}")
        logger.info(f"  - Temporal unification: {self.stats['temporal_unified']}")
        logger.info(f"  - Temporal sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Geographic links: {self.stats['geographic_links']}")
        logger.info(f"  - Event type links: {self.stats['event_links']}")
        logger.info(f"  - Severity escalation links: {self.stats['severity_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': enrichment_count,
            'available_datasets': list(self.available_datasets),
            **self.stats
        }

    def unify_temporal_entities(self):
        """Unify temporal entities from NOAA alerts"""
        # Use TemporalUnifier with NOAA-only scope
        unifier = TemporalUnifier(self.graph)
        unifier.available_sources = {'noaa'}  # Restrict to NOAA only

        stats = unifier.unify_all_sources()
        self.stats['temporal_unified'] = stats['temporal_links']

        logger.info(f"  Unified {stats['months_unified']} months and {stats['years_unified']} years")
        logger.info(f"  Created {stats['temporal_links']} temporal unification links")

    def link_temporal_sequences(self):
        """
        Link temporal sequences for weather alerts

        Links alerts in chronological order for the same geographic area
        """
        # Get all alerts with their areas and times
        query = f"""
        SELECT ?alert ?area ?areaDesc ?sentTime WHERE {{
            ?alert a <{CAP.Alert}> ;
                   <{CAP.hasSentTime}> ?sentTime ;
                   <{CAP.hasInfo}> ?info .
            ?info <{CAP.hasArea}> ?area .
            ?area <{CAP.hasAreaDescription}> ?areaDesc .
        }}
        """

        results = list(self.graph.query(query))

        # Group by area description
        by_area = {}
        for row in results:
            area_desc = str(row.areaDesc)
            if area_desc not in by_area:
                by_area[area_desc] = []

            by_area[area_desc].append({
                'alert': row.alert,
                'sent_time': str(row.sentTime)
            })

        # Sort and link alerts chronologically for each area
        links_added = 0
        for area_desc, alerts in by_area.items():
            if len(alerts) < 2:
                continue

            # Sort by sent time
            alerts.sort(key=lambda x: x['sent_time'])

            # Link consecutive alerts
            for i in range(len(alerts) - 1):
                current = alerts[i]['alert']
                next_alert = alerts[i + 1]['alert']

                self.graph.add((
                    current,
                    NOAA_ENRICHMENT.precedes,
                    next_alert
                ))
                links_added += 1
                self.stats['temporal_sequences'] += 1

        logger.info(f"  Added {links_added} temporal sequence links")

    def link_geographic_relationships(self):
        """
        Link alerts affecting same or overlapping geographic areas

        Uses SAME codes, UGC codes, and FIPS codes to identify related areas
        """

        # Link alerts with same SAME codes
        same_query = f"""
        SELECT ?alert1 ?alert2 ?sameCode WHERE {{
            ?alert1 <{CAP.hasInfo}> ?info1 .
            ?info1 <{CAP.hasArea}> ?area1 .
            ?area1 <{CAP.hasGeocode}> ?geocode1 .
            ?geocode1 <{NWS.hasSAMECode}> ?sameCode .

            ?alert2 <{CAP.hasInfo}> ?info2 .
            ?info2 <{CAP.hasArea}> ?area2 .
            ?area2 <{CAP.hasGeocode}> ?geocode2 .
            ?geocode2 <{NWS.hasSAMECode}> ?sameCode .

            FILTER(?alert1 != ?alert2)
        }}
        """

        results = list(self.graph.query(same_query))

        # Track unique pairs to avoid duplicates
        linked_pairs = set()

        for row in results:
            alert1 = row.alert1
            alert2 = row.alert2

            # Create ordered pair to avoid duplicates
            pair = tuple(sorted([str(alert1), str(alert2)]))

            if pair not in linked_pairs:
                self.graph.add((
                    alert1,
                    NOAA_ENRICHMENT.affectsSameRegion,
                    alert2
                ))
                linked_pairs.add(pair)
                self.stats['geographic_links'] += 1

        logger.info(f"  Added {len(linked_pairs)} geographic relationship links")

    def link_event_relationships(self):
        """
        Link related weather events

        Links alerts with the same event type (e.g., all Flood Advisories)
        """

        # Get all alerts grouped by event type
        query = f"""
        SELECT ?alert ?event WHERE {{
            ?alert <{CAP.hasInfo}> ?info .
            ?info <{CAP.hasEvent}> ?event .
        }}
        """

        results = list(self.graph.query(query))

        # Group by event type
        by_event = {}
        for row in results:
            event = str(row.event)
            if event not in by_event:
                by_event[event] = []
            by_event[event].append(row.alert)

        # Link alerts of same event type
        links_added = 0
        for event_type, alerts in by_event.items():
            if len(alerts) < 2:
                continue

            # Link each alert to others of same type
            for i, alert1 in enumerate(alerts):
                for alert2 in alerts[i + 1:]:
                    self.graph.add((
                        alert1,
                        NOAA_ENRICHMENT.sameEventType,
                        alert2
                    ))
                    links_added += 1
                    self.stats['event_links'] += 1

        logger.info(f"  Added {links_added} event type links")

    def link_severity_escalations(self):
        """
        Link severity escalations for same geographic areas

        Links watches → advisories → warnings → emergencies
        for the same region
        """

        # Define severity hierarchy
        severity_order = {
            'Minor': 1,
            'Moderate': 2,
            'Severe': 3,
            'Extreme': 4
        }

        # Get alerts with severity and area
        query = f"""
        SELECT ?alert ?severity ?areaDesc ?sentTime WHERE {{
            ?alert <{CAP.hasInfo}> ?info ;
                   <{CAP.hasSentTime}> ?sentTime .
            ?info <{CAP.hasSeverity}> ?severityUri ;
                  <{CAP.hasArea}> ?area .
            ?area <{CAP.hasAreaDescription}> ?areaDesc .
            ?severityUri <{RDFS.label}> ?severity .
        }}
        """

        results = list(self.graph.query(query))

        # Group by area
        by_area = {}
        for row in results:
            area_desc = str(row.areaDesc)
            severity = str(row.severity)

            if area_desc not in by_area:
                by_area[area_desc] = []

            by_area[area_desc].append({
                'alert': row.alert,
                'severity': severity,
                'severity_level': severity_order.get(severity, 0),
                'sent_time': str(row.sentTime)
            })

        # Link escalating severity alerts
        links_added = 0
        for area_desc, alerts in by_area.items():
            if len(alerts) < 2:
                continue

            # Sort by time
            alerts.sort(key=lambda x: x['sent_time'])

            # Find escalations (increasing severity over time)
            for i in range(len(alerts) - 1):
                current = alerts[i]
                next_alert = alerts[i + 1]

                if next_alert['severity_level'] > current['severity_level']:
                    self.graph.add((
                        current['alert'],
                        NOAA_ENRICHMENT.escalatesTo,
                        next_alert['alert']
                    ))
                    links_added += 1
                    self.stats['severity_links'] += 1

        logger.info(f"  Added {links_added} severity escalation links")