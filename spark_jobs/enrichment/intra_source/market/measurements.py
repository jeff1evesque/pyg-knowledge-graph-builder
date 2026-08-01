"""
Market Measurement Types - Temporal linking configurations for market data.

Aligned with the flat snapshot model where each row in the source
Parquet is a self-contained QuoteSnapshot (EquitySnapshot or
OptionSnapshot) with all fields as direct datatype properties.

Key properties for temporal linking:
  - captureTime: ISO timestamp of when the snapshot was taken
  - symbol: ticker symbol (equity) or OCC symbol (option)
  - underlyingSymbol: for options, the underlying equity ticker

Temporal sequences are built per-symbol ordered by captureTime.
"""
from spark_jobs.utils.rdf_utils import MARKET_QUOTES

MEASUREMENT_TYPES = {
    'equity_snapshots': {
        'EquitySnapshot': {
            'class': MARKET_QUOTES.EquitySnapshot,
            'symbol_property': MARKET_QUOTES.symbol,
            'timestamp_property': MARKET_QUOTES.captureTime,
            'price_property': MARKET_QUOTES.lastPrice,
            'volume_property': MARKET_QUOTES.totalVolume,
        },
    },

    'option_snapshots': {
        'OptionSnapshot': {
            'class': MARKET_QUOTES.OptionSnapshot,
            'symbol_property': MARKET_QUOTES.symbol,
            'timestamp_property': MARKET_QUOTES.captureTime,
            'underlying_symbol_property': MARKET_QUOTES.underlyingSymbol,
            'strike_property': MARKET_QUOTES.strikePrice,
            'expiration_property': MARKET_QUOTES.expirationDate,
            'contract_type_property': MARKET_QUOTES.contractType,
            'underlying_price_property': MARKET_QUOTES.underlyingPrice,
            'delta_property': MARKET_QUOTES.delta,
            'gamma_property': MARKET_QUOTES.gamma,
            'theta_property': MARKET_QUOTES.theta,
            'vega_property': MARKET_QUOTES.vega,
        },
    },
}