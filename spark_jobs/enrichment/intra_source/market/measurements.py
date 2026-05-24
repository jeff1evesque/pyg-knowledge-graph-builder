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
from spark_jobs.utils.rdf_utils import MARKET

MEASUREMENT_TYPES = {
    'equity_snapshots': {
        'EquitySnapshot': {
            'class': MARKET.EquitySnapshot,
            'symbol_property': MARKET.symbol,
            'timestamp_property': MARKET.captureTime,
            'price_property': MARKET.lastPrice,
            'volume_property': MARKET.totalVolume,
        },
    },

    'option_snapshots': {
        'OptionSnapshot': {
            'class': MARKET.OptionSnapshot,
            'symbol_property': MARKET.symbol,
            'timestamp_property': MARKET.captureTime,
            'underlying_symbol_property': MARKET.underlyingSymbol,
            'strike_property': MARKET.strikePrice,
            'expiration_property': MARKET.expirationDate,
            'contract_type_property': MARKET.contractType,
            'underlying_price_property': MARKET.underlyingPrice,
            'delta_property': MARKET.delta,
            'gamma_property': MARKET.gamma,
            'theta_property': MARKET.theta,
            'vega_property': MARKET.vega,
        },
    },
}