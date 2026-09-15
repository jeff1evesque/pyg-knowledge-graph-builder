"""BLS's side of cross-source linking: local-area series by state, job-openings
regions as census regions, and the indicators that lead an equity sector.

The BLS source spec (spark_jobs/sources/bls.py) hands these functions to the
cross-source linker.
"""
import logging
from typing import Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from rdflib.namespace import RDF

from spark_jobs import sources
from spark_jobs.enrichment.cross_source_linker import CrossSourceContext
from spark_jobs.enrichment.intra_source.market.patterns import _gics_sector_to_pascal
from spark_jobs.enrichment.region_crosswalk import CENSUS_REGION_CODES, state_key
from spark_jobs.enrichment.sector_crosswalk import EQUITY_TO_ECONOMIC_SECTORS
from spark_jobs.sources.spec import RegionKeys
from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT,
    JOLTS,
    LAUS,
    MARKET_ENRICHMENT,
    UNIFIED,
)

logger = logging.getLogger(__name__)

_RDF_TYPE = str(RDF.type)

# The economic side of the region join is keyed on NAMES, not codes.
#
# An earlier version of this file also read laus:hasStateFIPS,
# metro:hasStateFIPS and jolts:hasCensusRegionCode, on the strength of an issue
# saying those terms were coming. They were checked against the mapper and none
# of the three exists, is emitted, or is planned on any branch -- the same issue
# was wrong three times about filings:hasSic, so its claims are not a source.
# The readers were deleted rather than left matching zero rows and looking like
# working code.
#
# Nothing is lost by that. Every one of those surveys already exposes its
# geography as a NAMED node -- laus:hasState -> id/laus/Alabama,
# jolts:hasRegion -> id/jolts/Midwest_Region -- and the name is what this module
# joins on. The full path weather alert -> state region -> census region ->
# jolts region is live on the fixtures today.
#
# Two further notes for anyone tempted to add a code reader later:
#
#   * The JOLTS region identifier upstream is NOT numeric. Its catalog carries
#     "state_code": "MW" for "Midwest region", so a reader expecting census
#     regions 1-4 would match nothing even if the term shipped.
#   * The metro catalog holds no FIPS column at all, so that term would need
#     the codes sourced first, not merely mapped.
_JOLTS_REGION_TYPE = str(JOLTS.Region)


def delimited_local_name(subject: Column) -> Column:
    """The URI's local name with every run of non-letters as one "_", wrapped.

    "…/id/laus/Arkansas_June2026_HiresLevel" -> "_Arkansas_June_HiresLevel_"

    Wrapping in the separator is what turns a substring test into a WORD test:
    "_Arkansas_June_" does not contain "_Kansas_", while the raw URI does
    contain "Kansas". Collapsing digit runs too, so a year between two names
    still reads as one boundary rather than gluing them together.
    """
    local = F.regexp_extract(subject, r"[/#]([^/#]+)$", 1)
    return F.concat(
        F.lit("_"), F.regexp_replace(local, r"[^A-Za-z]+", "_"), F.lit("_")
    )


def region_keys(context: CrossSourceContext) -> RegionKeys:
    """The local-area series' states, and the job-openings regions that are
    census regions."""
    return RegionKeys(
        state_links=_local_area_state_links(context),
        census_regions=_census_region_entities(context),
    )


def _local_area_state_links(context: CrossSourceContext) -> DataFrame:
    """Each local-area series -> the unified region of the state it is about."""
    laus_prefixes = sources.entity_prefixes((str(LAUS),))
    laus_match = F.col("subject").startswith(laus_prefixes[0])
    for p in laus_prefixes[1:]:
        laus_match = laus_match | F.col("subject").startswith(p)
    laus_entities = (
        context.triples_df
        .select("subject")
        .distinct()
        .filter(laus_match)
    )

    # The state name must be the DELIMITED HEAD of the local name, not
    # a substring of the whole URI anywhere.
    #
    # `subject.contains(state_key)` linked every West Virginia series to
    # VirginiaRegion as well as to its own, because "WestVirginia"
    # contains "Virginia". That is the only such pair among the 51
    # names -- see test_exactly_one_state_name_contains_another, which
    # also records that Arkansas/Kansas is NOT one of them, the
    # comparison being case-sensitive and the 'k' lowercase. One pair is
    # enough: merging two genuinely distinct regions is worse than
    # missing one, since the weather feed then reaches the wrong state's
    # unemployment series with full confidence.
    #
    # It does NOT separate a metro area whose name BEGINS with a state
    # name: "Kansas_City_MO" still lands on KansasRegion, though the
    # metro straddles two states. That needs the federal delineation
    # file which puts metro resolution out of scope, so this is the
    # state-vs-state fix and not the metro one.
    #
    # Anchored at the HEAD because these URIs put the area first
    # ("Midwest_June2026_HiresLevel"). An unanchored delimited match
    # does not separate the Virginia pair either -- "_West_Virginia_"
    # contains "_Virginia_" as a substring -- so the anchor is doing
    # real work rather than being a convenience.
    #
    # Both separator spellings are accepted because the two are minted
    # differently across sources: "New_York" from a label slug,
    # "NewYork" from a URI-safe key.
    laus_matched = laus_entities.crossJoin(F.broadcast(context.states)).filter(
        delimited_local_name(F.col("subject")).startswith(
            F.col("state_slug")
        )
        | delimited_local_name(F.col("subject")).startswith(
            F.col("state_slug_nospace")
        )
    )

    return laus_matched.select(
        F.col("subject"),
        F.lit(str(BLS_ENRICHMENT.hasRegion)).alias("predicate"),
        F.concat(F.lit(str(UNIFIED)), F.col("state_key"), F.lit("Region")).alias("object")
    )


def _census_region_entities(context: CrossSourceContext) -> DataFrame:
    """(entity, census_name, census_key) for each job-openings region that is
    one of the census regions, matched by NAME.

    This is the only path, and it is sufficient: the job-openings survey
    states its region as a named node (id/jolts/Midwest_Region) and never
    as a code. A code-keyed path was written first, on the strength of an
    issue claiming jolts:hasCensusRegionCode was coming; it is not, and the
    upstream region identifier is "MW" rather than a census number anyway, so
    a numeric reader would have matched nothing even if it had shipped.
    """
    census_rows = [
        (name, code, state_key(name))
        for name, code in CENSUS_REGION_CODES.items()
    ]
    census_df = context.spark.createDataFrame(
        census_rows, schema=["census_name", "census_code", "census_key"]
    )

    return context.triples_df.filter(
        (F.col("predicate") == _RDF_TYPE)
        & (F.col("object") == _JOLTS_REGION_TYPE)
    ).select(F.col("subject").alias("entity")).crossJoin(
        F.broadcast(census_df)
    ).filter(
        # The same delimited-head rule the state match uses. "Midwest_Region"
        # reads as "_Midwest_Region_", which starts with "_Midwest_"; a
        # region merely CONTAINING a census name does not match.
        delimited_local_name(F.col("entity")).startswith(
            F.concat(F.lit("_"), F.col("census_name"), F.lit("_"))
        )
    ).select("entity", "census_name", "census_key").dropDuplicates()


def link_indicators_to_equity_sectors(
    context: CrossSourceContext,
) -> Optional[DataFrame]:
    """
    Link a BLS indicator to the equity sector it leads.

    THE OBJECT IS A SECTOR, NOT A SNAPSHOT, and that is the whole of #359.
    This used to point at every market entity in the sector, one edge per
    snapshot. On the first run where the BLS side was not empty that was
    393,860,192 edges -- 46% of the entire graph, 97% of everything
    cross-source produced -- from 499 BLS subjects, averaging 789,299 edges
    each. One producer-price series carried 2,933,391 on its own.

    Those edges said "this producer-price series leads this PSX put, as
    quoted at 18:20:40.111Z". The strike and the timestamp have nothing to
    do with the indicator, and the same claim was repeated for every other
    snapshot of that contract and every other contract in the sector. It is
    one sector-level statement inflated by six orders of magnitude.

    It was also information the graph already held. The edge is exactly the
    composition of three relations that are already there:

        BLS series -belongsToSector-> EconomicSector
                   <-relatedToEconomicSector- GICS sector
                   <-market:belongsToSector- snapshot

    and the join below is built from the SAME crosswalk table that mints
    the middle one. So every one of those 393 million edges was derivable
    from ~499 + 15 + 10,054,950 edges already in the frame. They added no
    reachability, and for a GNN a complete bipartite block between two
    groups carries nothing beyond the group membership that defines it:
    every snapshot in the sector receives the identical set of messages.

    Pointing at the GICS sector states the claim at the granularity the
    crosswalk actually supports. On that same run it is 570 edges -- the
    same 499 subjects across the 8 mapped GICS sectors. A snapshot is still
    two hops from any indicator that leads it, which a two-layer model
    reaches.

    NOT the economic sector, though both are one hop away. bls:leadsTo to
    an EconomicSector would repeat the (subject, object) pair that
    bls:belongsToSector already covers, which is the duplicate-relation
    problem removed twice elsewhere in cross-source linking. The GICS sector
    is a different node, so the edge is a different claim.

    WHY THIS NEVER FIRED. Both sides were selected by filtering for
    bls:belongsToSector -- but only the BLS side emits that. The market
    enricher writes market:belongsToSector, a DIFFERENT predicate, because
    its sector is a GICS sector and lives in its own vocabulary. So the
    market half of the filter matched nothing on every graph ever built,
    the inner join had one empty side, and the step returned None while
    logging that it ran. The e2e suite recorded the absence and blamed the
    missing constituents CSV; that was only half the reason, and the join
    would have stayed dead with the CSV configured.

    The fix is NOT to make market emit the BLS predicate. A GICS sector and
    an economic sector are different classifications of different things --
    that is the whole premise of enrichment/sector_crosswalk.py -- and
    collapsing them onto one predicate would assert the equivalence that
    module exists to deny.

    Instead the market side is carried to the economic sector THROUGH the
    crosswalk, which is what it is for:

        BLS series  -belongsToSector->  EconomicSector
                                              ^
                                -relatedToEconomicSector-
                                              |
        snapshot -market:belongsToSector-> GICS sector

    so both sides end up keyed on the same economic sector and the join has
    something to match. Note this makes the link only as good as the
    curated table -- a GICS sector deliberately mapped to nothing (see the
    three gaps) contributes no causal edge, which is correct.
    """
    bls_sector = (
        context.triples_df
        .filter(F.col("predicate") == str(BLS_ENRICHMENT.belongsToSector))
        .select(
            F.col("subject").alias("entity"),
            F.col("object").alias("sector"),
        )
    )

    # The market side. Read per snapshot, because that is the only place
    # the GICS classification is stated; collapsed to the sector below.
    market_gics = (
        context.triples_df
        .filter(F.col("predicate") == str(MARKET_ENRICHMENT.belongsToSector))
        .select(
            F.col("subject").alias("entity"),
            F.col("object").alias("equity_sector"),
        )
    )

    # Built from the TABLE, not from the relatedToEconomicSector triples.
    #
    # Those triples are minted by market's sector step in this same
    # cross-source run and are not unioned into triples_df until every
    # step has run -- so reading them here finds nothing, and the join dies
    # exactly the way the predicate mismatch used to. Every cross-source
    # step reads the INPUT frame; none can see another's output.
    crosswalk = context.spark.createDataFrame(
        sorted({
            (
                str(MARKET_ENRICHMENT[f"{_gics_sector_to_pascal(gics)}Sector"]),
                sector_uri,
            )
            for gics, mapped in EQUITY_TO_ECONOMIC_SECTORS.items()
            for sector_uri, _confidence in mapped
        }),
        schema=["equity_sector", "sector"],
    )

    bls_prefixes = context.entity_prefixes("bls")
    market_prefixes = context.entity_prefixes("market")

    # BLS entities in sectors
    bls_filter = F.col("entity").startswith(bls_prefixes[0])
    for p in bls_prefixes[1:]:
        bls_filter = bls_filter | F.col("entity").startswith(p)

    bls_sector = bls_sector.filter(bls_filter).select(
        F.col("entity").alias("bls_entity"), F.col("sector").alias("bls_sector")
    )

    # Market entities in sectors
    market_filter = F.col("entity").startswith(market_prefixes[0])
    for p in market_prefixes[1:]:
        market_filter = market_filter | F.col("entity").startswith(p)

    # DISTINCT on the sector, and this is the line that ends the cartesian
    # product. The snapshots are still what proves a sector is present in
    # this build -- a GICS sector nothing classified into must not collect
    # an edge -- but once that is established they are dropped, and the
    # right side of the join is at most eleven rows instead of ten million.
    market_sector = (
        market_gics
        .filter(market_filter)
        .select("equity_sector")
        .distinct()
        .join(crosswalk, "equity_sector", "inner")
        .select(
            F.col("equity_sector"),
            F.col("sector").alias("market_sector"),
        )
    )

    if bls_sector.head(1) and market_sector.head(1):
        # Join on the economic sector, emit the GICS one. The economic
        # sector is how the two vocabularies meet; it is not what the edge
        # points at, because belongsToSector already points there.
        causal = bls_sector.join(
            market_sector,
            bls_sector.bls_sector == market_sector.market_sector,
            how="inner"
        )

        # An indicator can sit in several economic sectors that a single
        # GICS sector maps to -- Industrials covers manufacturing,
        # transportation and construction trades -- so the join can reach
        # the same pair by more than one route.
        result = causal.select(
            F.col("bls_entity").alias("subject"),
            F.lit(str(BLS_ENRICHMENT.leadsTo)).alias("predicate"),
            F.col("equity_sector").alias("object")
        ).distinct()

        logger.info("  Causal link triples prepared (lazy)")
        return result

    return None
