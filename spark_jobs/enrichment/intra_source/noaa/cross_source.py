"""NOAA's side of cross-source linking: the states each weather alert affects.

The NOAA source spec (spark_jobs/sources/noaa.py) hands this to the
cross-source linker, which builds the region nodes the links point at.
"""
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_jobs.enrichment.cross_source_linker import CrossSourceContext
from spark_jobs.enrichment.region_crosswalk import (
    STATE_FIPS_TO_NAME,
    normalized_state_fips,
)
from spark_jobs.sources.spec import RegionKeys
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT, CAP, UNIFIED, WEATHER
from spark_jobs.utils.spark_rdf_utils import extract_property

# NOAA area description — on Area subject (alert:{id}#area)
_CAP_HAS_AREA_DESC = str(CAP.hasAreaDescription)

# NOAA geocode properties for geographic linking.
#
# nws:hasFIPSCode is deliberately NOT bound here. It was, and was never
# referenced -- the FIPS strategy below reads hasStateFIPS only. Worse than
# unused: the name reads as "the county FIPS is available", when it is not.
# Verified against the mapper -- hasFIPSCode and hasSAMECode are both mapped
# from the same fips_code column and render byte-identical ("024031" for a
# geocode whose county FIPS is "24031"). Anyone reaching for the dead constant
# would have joined a 6-character SAME code against a 5-character county-FIPS
# lookup. Nothing on any branch changes this, so a county-level join here would
# have to derive the county FIPS itself rather than wait for a corrected term.
_NWS_HAS_STATE_FIPS = str(WEATHER.hasStateFIPS)

# NOAA structural properties for traversal
_CAP_HAS_INFO = str(CAP.hasInfo)
_CAP_HAS_AREA = str(CAP.hasArea)
_CAP_HAS_GEOCODE = str(CAP.hasGeocode)


def region_keys(context: CrossSourceContext) -> RegionKeys:
    """Each alert's states, from its area description and from its geocodes'
    state FIPS codes."""
    return RegionKeys(state_links=_alert_state_links(context))


def _alert_state_links(context: CrossSourceContext) -> Optional[DataFrame]:
    """
    Link NOAA alerts to the unified regions of the states they affect.

    For NOAA: The updated RML mapper produces:
      - Area descriptions on cap:Area subjects (alert:{id}#area)
        via cap:hasAreaDescription
      - State FIPS codes on cap:Geocode subjects (alert:{id}#geocode-{fips})
        via nws:hasStateFIPS

    To link NOAA alerts to geographic regions, we:
      1. Match area descriptions containing state names
      2. Match state FIPS codes to state names
    Both approaches trace back to the Alert subject via the
    Alert → Info → Area → Geocode chain.
    """
    states_df = context.states
    noaa_link_dfs: List[DataFrame] = []

    # Structural chain shared by both strategies (lazy — defining
    # these costs nothing until joined):
    #   Alert → hasInfo → Info → hasArea → Area
    info_areas = context.triples_df.filter(
        F.col("predicate") == _CAP_HAS_AREA
    ).select(
        F.col("subject").alias("info"),
        F.col("object").alias("area"),
    )

    alert_infos = context.triples_df.filter(
        F.col("predicate") == _CAP_HAS_INFO
    ).select(
        F.col("subject").alias("alert"),
        F.col("object").alias("info"),
    )

    # Strategy 1: Area description contains state name
    # Area descriptions are on cap:Area subjects, so join through the
    # forward chain: Alert → Info → Area → hasAreaDescription
    area_descs = extract_property(
        context.triples_df, _CAP_HAS_AREA_DESC, "area_desc"
    )

    if area_descs.head(1):
        # area_descs has (subject=area_uri, area_desc)
        # Join: area_desc → area → info → alert
        area_to_alert = (
            area_descs
            .join(info_areas, area_descs.subject == info_areas.area, "inner")
            .join(alert_infos, "info", "inner")
            .select("alert", "area_desc")
        )

        # Match the ABBREVIATION as well as the full name.
        #
        # NWS writes area descriptions as "Lincoln, KS; Russell, KS" --
        # county plus two-letter state -- and the full name never
        # appears. Matching only `contains("Kansas")` therefore matched
        # nothing on every alert ever ingested, and this strategy
        # produced no link at all.
        #
        # The abbreviation is matched with its ", " separator rather
        # than bare, because two letters appear inside ordinary words:
        # a bare "contains('OR')" hits "ORANGE" and a bare "IN" hits
        # almost everything. Anchoring on the comma is how NWS actually
        # writes it and keeps the match to the state field.
        noaa_area_matched = area_to_alert.crossJoin(F.broadcast(states_df)).filter(
            F.col("area_desc").contains(F.col("state_name"))
            | F.col("area_desc").contains(
                F.concat(F.lit(", "), F.col("state_abbr"))
            )
        )

        noaa_area_links = noaa_area_matched.select(
            F.col("alert").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.affectsRegion)).alias("predicate"),
            F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
        )
        noaa_link_dfs.append(noaa_area_links)

    # Strategy 2: State FIPS code matching
    #
    # Geocode subjects carry nws:hasStateFIPS as the 3-character head of
    # a SAME code ("040", "024"), never a 2-digit FIPS -- upstream slices
    # it as value[:3], so the leading part-digit is structural and always
    # present. The lookup below is keyed 2-digit; see the normalisation.
    #
    # Note also that nws:hasFIPSCode carries the SAME value as
    # nws:hasSAMECode -- both read upstream's fips_code column, which is
    # populated from properties.geocode.SAME. hasFIPSCode is a misnomer
    # for the SAME code and is not a county FIPS.
    state_fips_rows = [
        (fips, name, name.replace(' ', ''))
        for fips, name in STATE_FIPS_TO_NAME.items()
    ]
    state_fips_df = context.spark.createDataFrame(
        state_fips_rows, ["fips_code", "state_name", "state_key"]
    )

    # Normalize the code to the two digits the lookup above is keyed on.
    #
    # The lookup holds 2-digit state FIPS ("20" Kansas, "42"
    # Pennsylvania) while the mapper emits the 3-digit head of a SAME
    # code -- "020", "042" -- which is a leading part-digit followed by
    # the state. Compared as strings those never match, so this whole
    # strategy linked nothing on every alert ever ingested.
    #
    # normalized_state_fips reads the LAST two digits, which is correct
    # for the current 3-character form AND for the 2-character form
    # upstream is moving to -- see the note on that function for why it
    # must stay tolerant of both permanently, and for the lpad trap it
    # exists to avoid.
    state_fips_triples = context.triples_df.filter(
        F.col("predicate") == _NWS_HAS_STATE_FIPS
    ).select(
        F.col("subject").alias("geocode"),
        normalized_state_fips(F.col("object")).alias("state_fips"),
    )

    if state_fips_triples.head(1):
        # Trace geocode → area → info → alert
        area_geocodes = context.triples_df.filter(
            F.col("predicate") == _CAP_HAS_GEOCODE
        ).select(
            F.col("subject").alias("area"),
            F.col("object").alias("geocode"),
        )

        geocode_to_alert = (
            state_fips_triples
            .join(area_geocodes, "geocode", "inner")
            .join(info_areas, "area", "inner")
            .join(alert_infos, "info", "inner")
            .select("alert", "state_fips")
            .dropDuplicates()
        )

        fips_matched = geocode_to_alert.join(
            F.broadcast(state_fips_df),
            geocode_to_alert.state_fips == state_fips_df.fips_code,
            "inner",
        )

        noaa_fips_links = fips_matched.select(
            F.col("alert").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.affectsRegion)).alias("predicate"),
            F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
        )
        noaa_link_dfs.append(noaa_fips_links)

    if not noaa_link_dfs:
        return None

    noaa_links = noaa_link_dfs[0]
    for df in noaa_link_dfs[1:]:
        noaa_links = noaa_links.unionByName(df)
    return noaa_links.dropDuplicates()
