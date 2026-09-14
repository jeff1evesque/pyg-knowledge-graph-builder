"""Market's side of cross-source linking: its companies by ticker, its GICS
sectors' economic sectors, and peers in one GICS sub-industry.

The market source spec (spark_jobs/sources/market.py) hands these functions to
the cross-source linker.
"""
import logging
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from rdflib.namespace import RDF

from spark_jobs.enrichment.cross_source_linker import (
    UNIFIED_COMPANY_PREFIX,
    CrossSourceContext,
)
from spark_jobs.enrichment.intra_source.market.patterns import _gics_sector_to_pascal
from spark_jobs.enrichment.intra_source.market.symbols import equity_symbol
from spark_jobs.enrichment.sector_crosswalk import (
    EQUITY_SECTOR_TYPE,
    EQUITY_TO_ECONOMIC_SECTORS,
    RELATED_TO_ECONOMIC_SECTOR,
    RELATION_CONFIDENCE,
)
from spark_jobs.sources.spec import CompanyKeys
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT, MARKET_ENRICHMENT, MARKET_QUOTES
from spark_jobs.utils.spark_rdf_utils import extract_entities_by_type, extract_property

logger = logging.getLogger(__name__)

_RDF_TYPE = str(RDF.type)

# The symbol-bearing classes, as (type, predicate). Equity and option snapshots
# arrive in the same feed and state their ticker the same way.
_MARKET_SYMBOL_TYPES = (
    (str(MARKET_QUOTES.EquitySnapshot), str(MARKET_QUOTES.symbol)),
    (str(MARKET_QUOTES.OptionSnapshot), str(MARKET_QUOTES.symbol)),
)

# Peer relation between two companies in one GICS sub-industry. In the MARKET
# enrichment namespace because the sub-industry is GICS vocabulary, and the
# subject and object are both companies the market feed classifies.
_SHARES_SUB_INDUSTRY = str(MARKET_ENRICHMENT.sharesSubIndustryWith)


def symbol_bearers(triples_df: DataFrame) -> DataFrame:
    """(entity, symbol) for every market entity that names a ticker.

    Keyed on the classes each market vocabulary actually declares, per
    _MARKET_SYMBOL_TYPES. The previous version asked for
    market-feeds:StockTicker, which belongs to NEITHER model -- feeds is
    PriceObservation / OptionContract, quotes is EquitySnapshot /
    OptionSnapshot -- so it matched nothing no matter which namespace it
    was pointed at, and the SEC-to-market bridge had no market side.
    """
    frames = [
        extract_entities_by_type(triples_df, type_uri).join(
            extract_property(triples_df, symbol_pred, "raw_symbol"),
            on=F.col("entity") == F.col("subject"),
        ).select("entity", "raw_symbol")
        for type_uri, symbol_pred in _MARKET_SYMBOL_TYPES
    ]

    union = frames[0]
    for frame in frames[1:]:
        union = union.unionByName(frame)

    return union.select(
        F.col("entity"),
        equity_symbol(F.col("raw_symbol")).alias("symbol"),
    ).filter(F.length(F.col("symbol")) > 0).dropDuplicates()


def company_keys(context: CrossSourceContext) -> CompanyKeys:
    """The snapshots, by the ticker each names, and the constituents CSV's
    ticker pairings.

    The CSV pairs the symbol with the CIK for index members whose filings may
    not have been ingested at all, which is how a quote reaches a company no
    filing in the build names. Its pairings rank after the filings' own
    (priority 1): a ticker is reassigned when a company delists and another
    takes the symbol, so the CSV can carry a pairing that was true at
    publication and is not true of the filing in front of us.
    """
    symbol_ciks = None
    if context.ticker_cik_map:
        symbol_ciks = context.spark.createDataFrame(
            sorted(context.ticker_cik_map.items()), schema=["symbol", "cik"]
        ).select(
            F.col("symbol"),
            F.col("cik"),
            F.lit(1).alias("priority"),
        )

    return CompanyKeys(
        symbols=symbol_bearers(context.triples_df),
        symbol_ciks=symbol_ciks,
    )


def sector_keys(context: CrossSourceContext) -> Optional[DataFrame]:
    """Join the two sector vocabularies, by hand, with the gaps kept.

    bls:EconomicSector is the best-connected hub in the graph -- every
    economic-indicator type attaches to it -- and the market side had zero
    edges to it, because the market enricher classifies into its own GICS
    namespace and nothing reconciled the two. This is the Part 1 split one
    level up: the same concept represented twice, in parallel, never
    joined.

    The join is a CURATED table, not a name match, and it is deliberately
    incomplete: three GICS sectors map to nothing. See
    enrichment/sector_crosswalk.py for each row's reasoning and for why
    "completing" the table would make the graph worse.

    Emitted under relatedToEconomicSector, never belongsToSector. A
    constituent BELONGS TO its GICS sector; that GICS sector is merely
    RELATED TO an economic one. Collapsing membership and similarity into
    one relation would deny a GNN the ability to weight them differently.
    """
    rows = [
        (
            str(MARKET_ENRICHMENT[f"{_gics_sector_to_pascal(gics)}Sector"]),
            sector_uri,
            confidence,
        )
        for gics, mapped in EQUITY_TO_ECONOMIC_SECTORS.items()
        for sector_uri, confidence in mapped
    ]

    if not rows:
        return None

    crosswalk = context.spark.createDataFrame(
        sorted(rows), schema=["equity_sector", "economic_sector", "confidence"]
    )

    # Only the equity sectors this build actually classified something
    # into. The table covers eleven GICS sectors; a build holding quotes
    # for two of them should not carry nine sector nodes with one edge
    # each and no constituents.
    present = (
        context.triples_df
        .filter(F.col("predicate") == _RDF_TYPE)
        .filter(F.col("object") == EQUITY_SECTOR_TYPE)
        .select(F.col("subject").alias("equity_sector"))
        .distinct()
    )

    matched = crosswalk.join(F.broadcast(present), "equity_sector", "inner")

    # Type the economic sector too. The keyword step types the ones its
    # keywords hit, but a build with market data and no BLS feed would
    # otherwise point these edges at an untyped URI, and node_mapper drops
    # an edge whose destination is not a node.
    economic_types = matched.select("economic_sector").distinct().select(
        F.col("economic_sector").alias("subject"),
        F.lit(_RDF_TYPE).alias("predicate"),
        F.lit(str(BLS_ENRICHMENT.EconomicSector)).alias("object"),
    )

    related = matched.select(
        F.col("equity_sector").alias("subject"),
        F.lit(RELATED_TO_ECONOMIC_SECTOR).alias("predicate"),
        F.col("economic_sector").alias("object"),
    )

    # Confidence rides as a property of the equity sector node rather than
    # splitting the relation into strong/moderate variants, which would
    # double the edge types for something a scalar states better.
    confidence = matched.select(
        F.col("equity_sector").alias("subject"),
        F.lit(RELATION_CONFIDENCE).alias("predicate"),
        F.col("confidence").alias("object"),
    ).distinct()

    logger.info("  Equity-to-economic sector triples prepared (lazy)")
    return economic_types.unionByName(related).unionByName(confidence)


def link_by_sub_industry(context: CrossSourceContext) -> Optional[DataFrame]:
    """Peer edges between companies sharing one GICS sub-industry.

    WHY SUB-INDUSTRY AND NOT SECTOR. The constituents CSV carries both, and
    only the coarse one was ever read. Sector is too coarse to be a
    similarity signal: eleven buckets over 503 constituents, the largest
    holding 73 of them, so "same sector" separates almost nothing and a
    peer edge built on it is noise with a high degree. Sub-industry is
    roughly 160 buckets of 3-5 constituents, where "same sub-industry" is a
    sharp claim -- these are companies competing in one market.

    WHY PEER EDGES AND NOT A HUB NODE. A shared sub-industry node would be
    another high-degree hub that aggregates its whole bucket into one
    representation, which is the failure Part 2 of this work is unpicking
    for the period nodes. A bucket of 3-5 is small enough that the pairs
    themselves are cheap -- about ten edges per bucket -- and a pair says
    what a hub cannot: which specific company this one competes with.

    Restricted to companies the graph actually carries. The CSV lists every
    index member, and minting a node for each would add hundreds of
    vertices whose only edges are to each other -- a disconnected clique
    lattice bolted onto the graph, describing companies no source in this
    build mentions. The companies come from the company hub, which runs
    first.
    """
    if not context.sub_industries:
        return None
    if context.symbol_ciks is None or context.company_ciks is None:
        return None

    constituents = context.spark.createDataFrame(
        sorted(set(context.sub_industries)), schema=["symbol", "sub_industry"]
    )

    # Symbol -> CIK -> the company node, then keep only the companies some
    # source in this build actually points at.
    members = (
        constituents.join(F.broadcast(context.symbol_ciks), "symbol", "inner")
        .join(context.company_ciks, "cik", "left_semi")
        .select("cik", "sub_industry")
        .distinct()
    )

    left = members.select(
        F.col("cik").alias("cik_a"), F.col("sub_industry")
    )
    right = members.select(
        F.col("cik").alias("cik_b"), F.col("sub_industry")
    )

    # One edge per unordered pair. `<` rather than `!=` because the relation
    # is symmetric and emitting both directions would double every peer
    # edge -- the same doubling the duplicated sector relation already costs
    # this graph elsewhere.
    pairs = left.join(right, "sub_industry", "inner").filter(
        F.col("cik_a") < F.col("cik_b")
    )

    result = pairs.select(
        F.concat(F.lit(UNIFIED_COMPANY_PREFIX), F.col("cik_a")).alias("subject"),
        F.lit(_SHARES_SUB_INDUSTRY).alias("predicate"),
        F.concat(F.lit(UNIFIED_COMPANY_PREFIX), F.col("cik_b")).alias("object"),
    )

    logger.info("  Sub-industry peer triples prepared (lazy)")
    return result
