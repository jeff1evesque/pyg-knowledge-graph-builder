"""Market: intraday equity and option quote snapshots, in one vocabulary."""
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import (
    IDENTIFIER_BASE,
    MARKET_ENRICHMENT,
    MARKET_QUOTES,
    UNIFIED,
)


def _linker(spark, options):
    from spark_jobs.enrichment.intra_source.market_linker import (
        MarketIntraSourceLinker,
    )

    return MarketIntraSourceLinker(
        spark,
        sector_definitions_bucket=options.sector_definitions_bucket,
        sector_definitions_key=options.sector_definitions_key,
        source_data_day=options.source_data_day,
    )


def _cross_source_inputs(options):
    """The index-constituents CSV's two tables, for the company bridge and the
    sub-industry peers. Empty when the CSV is unavailable."""
    from spark_jobs.enrichment.intra_source.market.patterns import (
        get_sub_industries,
        get_ticker_cik_map,
    )

    location = dict(
        bucket=options.sector_definitions_bucket,
        prefix=options.sector_definitions_key,
        data_day=options.source_data_day,
    )
    return {
        "ticker_cik_map": get_ticker_cik_map(**location),
        "sub_industries": get_sub_industries(**location),
    }


SPEC = SourceSpec(
    name="market",
    label="Market",
    path_fragments=("quotes",),
    # Enrichment first, which is where it sits in the namespace table.
    namespaces=(
        (str(MARKET_ENRICHMENT), "market_enrichment"),
        (str(MARKET_QUOTES), "market_quotes"),
    ),
    enrichment_namespace=str(MARKET_ENRICHMENT),
    # captureTime is the only ISO-8601 time a snapshot carries. quoteTime and
    # tradeTime are epoch milliseconds, which the date parser cannot read.
    date_predicates=(str(MARKET_QUOTES.captureTime),),
    # market-quotes rather than market, so the URI names the vocabulary that
    # observed the period.
    temporal_prefix=f"{IDENTIFIER_BASE}temporal/market-quotes/",
    linker=_linker,
    cross_source_inputs=_cross_source_inputs,
    property_mappings={
        str(MARKET_QUOTES.lastPrice): str(UNIFIED.measurementValue),
        str(MARKET_QUOTES.mark): str(UNIFIED.measurementValue),
        str(MARKET_QUOTES.symbol): str(UNIFIED.ticker),
    },
    relation_fragments={
        "option_stock": ("hasUnderlyingPriceObservation", "hasUnderlying"),
        "strategy": ("straddleWith", "spreadWith", "strangleWith"),
    },
)
