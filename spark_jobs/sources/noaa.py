"""NOAA: National Weather Service alerts, in our model of CAP."""
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import (
    ALERT,
    CAP,
    IDENTIFIER_BASE,
    NOAA_ENRICHMENT,
    UNIFIED,
    WEATHER,
)


def _linker(spark, options):
    from spark_jobs.enrichment.intra_source.noaa_linker import NOAAIntraSourceLinker

    return NOAAIntraSourceLinker(spark)


def _region_keys(context):
    from spark_jobs.enrichment.intra_source.noaa.cross_source import region_keys

    return region_keys(context)


SPEC = SourceSpec(
    name="noaa",
    label="NOAA",
    path_fragments=("/noaa/",),
    namespaces=(
        (str(CAP), "cap"),
        (str(WEATHER), "weather"),
        (str(ALERT), "alert"),
        (str(NOAA_ENRICHMENT), "noaa_enrichment"),
    ),
    enrichment_namespace=str(NOAA_ENRICHMENT),
    # All five are stated on the alert's Info subject, not on the alert. The
    # date collection keys on the predicate alone, so the entity it links to a
    # period is whichever subject stated the date.
    date_predicates=(
        str(CAP.hasSentTime),
        str(CAP.hasEffectiveTime),
        str(CAP.hasOnsetTime),
        str(CAP.hasExpirationTime),
        str(CAP.hasEndsTime),
    ),
    temporal_prefix=f"{IDENTIFIER_BASE}temporal/noaa/",
    linker=_linker,
    # Alert instances sit under the publisher's own alert namespace.
    entity_namespaces=(str(ALERT), str(CAP), str(WEATHER)),
    sector_keywords=True,
    region_keys=_region_keys,
    property_mappings={
        str(CAP.hasSentTime): str(UNIFIED.hasTimestamp),
        str(CAP.hasEffectiveTime): str(UNIFIED.hasTimestamp),
        str(CAP.hasOnsetTime): str(UNIFIED.hasTimestamp),
        str(CAP.hasExpirationTime): str(UNIFIED.hasTimestamp),
        str(CAP.hasEvent): str(UNIFIED.hasEventName),
        str(CAP.hasSeverity): str(UNIFIED.hasSeverity),
        str(CAP.hasUrgency): str(UNIFIED.hasUrgency),
        str(CAP.hasAreaDescription): str(UNIFIED.hasRegionDescription),
    },
    class_mappings={
        # weather:WeatherAlert is the only type written on an alert node.
        # cap:Alert is declared and never written, so it has no row.
        str(WEATHER.WeatherAlert): str(NOAA_ENRICHMENT.EmergencyAlert),
        str(CAP.Info): str(NOAA_ENRICHMENT.AlertInfo),
        str(CAP.Area): str(NOAA_ENRICHMENT.AlertArea),
    },
    relation_fragments={
        "escalation": ("escalatesTo", "escalatesFrom", "severityChange"),
        "skip": ("sameEventType", "affectsSameRegion"),
    },
)
