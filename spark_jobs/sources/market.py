"""Market: intraday equity and option quote snapshots, in one vocabulary."""
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import (
    IDENTIFIER_BASE,
    MARKET_ENRICHMENT,
    MARKET_QUOTES,
    UNIFIED,
)

SPEC = SourceSpec(
    name="market",
    path_fragments=("quotes",),
    # Enrichment first, which is where it sits in the namespace table.
    namespaces=(
        (str(MARKET_ENRICHMENT), "market_enrichment"),
        (str(MARKET_QUOTES), "market_quotes"),
    ),
    enrichment_namespace=str(MARKET_ENRICHMENT),
    # captureTime is the only ISO-8601 time a snapshot carries. quoteTime and
    # tradeTime are epoch milliseconds.
    date_predicates=(str(MARKET_QUOTES.captureTime),),
    # market-quotes rather than market, so the URI names the vocabulary that
    # observed the period.
    temporal_prefix=f"{IDENTIFIER_BASE}temporal/market-quotes/",
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
