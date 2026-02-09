"""
Market Measurement Types - Temporal linking configurations for market data
"""
from glue_jobs.utils.rdf_utils import MARKET

MEASUREMENT_TYPES = {
    'stock_prices': {
        'PriceObservation': {
            'class': MARKET.PriceObservation,
            'ticker_property': MARKET.observedTicker,
            'date_property': MARKET.observedAt,
            'price_property': MARKET.observedPrice,
        },
    },

    'options': {
        'OptionQuote': {
            'class': MARKET.OptionQuote,
            'contract_property': MARKET.quotedContract,
            'date_property': MARKET.observedAt,
            'price_property': MARKET.lastPrice,
        },
        'OptionContract': {
            'class': MARKET.OptionContract,
            'ticker_property': MARKET.underlyingTicker,
            'strike_property': MARKET.strikePrice,
            'expiration_property': MARKET.expirationDate,
            'type_property': MARKET.optionType,
        },
    },
}