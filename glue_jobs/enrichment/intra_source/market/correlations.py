"""
Market Known Correlations - Relationships between market data types
"""
from glue_jobs.utils.rdf_utils import MARKET_ENRICHMENT

KNOWN_CORRELATIONS = [

    # ============================================
    # STOCK PRICE - OPTIONS CORRELATIONS
    # ============================================

    {
        'name': 'stock_price_to_call_options',
        'source_dataset': 'stock_prices',
        'target_dataset': 'options',
        'source_pattern': 'StockTicker',
        'target_pattern': 'call',
        'description': 'Stock price movements affect call option values',
        'relationship': MARKET_ENRICHMENT.stockPriceCallOptionLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    {
        'name': 'stock_price_to_put_options',
        'source_dataset': 'stock_prices',
        'target_dataset': 'options',
        'source_pattern': 'StockTicker',
        'target_pattern': 'put',
        'description': 'Stock price movements affect put option values',
        'relationship': MARKET_ENRICHMENT.stockPricePutOptionLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    {
        'name': 'at_the_money_options',
        'source_dataset': 'stock_prices',
        'target_dataset': 'options',
        'source_pattern': 'PriceObservation',
        'target_pattern': 'OptionContract',
        'description': 'Options with strike near current stock price (ATM)',
        'relationship': MARKET_ENRICHMENT.atTheMoneyLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    # ============================================
    # INTERNAL CORRELATIONS - OPTIONS
    # ============================================

    {
        'name': 'call_put_parity',
        'source_dataset': 'options',
        'target_dataset': 'options',
        'source_pattern': 'call',
        'target_pattern': 'put',
        'description': 'Call and put options at same strike and expiration (put-call parity)',
        'relationship': MARKET_ENRICHMENT.putCallParityLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    {
        'name': 'same_expiration_options',
        'source_dataset': 'options',
        'target_dataset': 'options',
        'source_pattern': 'OptionContract',
        'target_pattern': 'OptionContract',
        'description': 'Options with same expiration date form option chains',
        'relationship': MARKET_ENRICHMENT.sameExpirationLink,
        'lag_months': 0,
        'strength': 'medium'
    },

    {
        'name': 'near_strike_options',
        'source_dataset': 'options',
        'target_dataset': 'options',
        'source_pattern': 'OptionContract',
        'target_pattern': 'OptionContract',
        'description': 'Options with adjacent strike prices',
        'relationship': MARKET_ENRICHMENT.adjacentStrikeLink,
        'lag_months': 0,
        'strength': 'medium'
    },

    {
        'name': 'expiration_series',
        'source_dataset': 'options',
        'target_dataset': 'options',
        'source_pattern': 'OptionContract',
        'target_pattern': 'OptionContract',
        'description': 'Options at same strike across different expirations',
        'relationship': MARKET_ENRICHMENT.expirationSeriesLink,
        'lag_months': 0,
        'strength': 'medium'
    },

    # ============================================
    # INTERNAL CORRELATIONS - STOCK PRICES
    # ============================================

    {
        'name': 'price_observation_sequence',
        'source_dataset': 'stock_prices',
        'target_dataset': 'stock_prices',
        'source_pattern': 'PriceObservation',
        'target_pattern': 'PriceObservation',
        'description': 'Sequential price observations for same ticker',
        'relationship': MARKET_ENRICHMENT.priceSequenceLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    {
        'name': 'same_sector_stocks',
        'source_dataset': 'stock_prices',
        'target_dataset': 'stock_prices',
        'source_pattern': 'StockTicker',
        'target_pattern': 'StockTicker',
        'description': 'Stocks in the same sector tend to correlate',
        'relationship': MARKET_ENRICHMENT.sameSectorStockLink,
        'lag_months': 0,
        'strength': 'medium'
    },

    {
        'name': 'same_exchange_stocks',
        'source_dataset': 'stock_prices',
        'target_dataset': 'stock_prices',
        'source_pattern': 'StockTicker',
        'target_pattern': 'StockTicker',
        'description': 'Stocks listed on the same exchange',
        'relationship': MARKET_ENRICHMENT.sameExchangeLink,
        'lag_months': 0,
        'strength': 'weak'
    },

    # ============================================
    # DATA SOURCE CORRELATIONS
    # ============================================

    {
        'name': 'multi_source_price_observation',
        'source_dataset': 'stock_prices',
        'target_dataset': 'stock_prices',
        'source_pattern': 'yahoo',
        'target_pattern': 'marketwatch',
        'description': 'Same ticker observed by multiple data sources',
        'relationship': MARKET_ENRICHMENT.multiSourceObservationLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    {
        'name': 'multi_source_option_quote',
        'source_dataset': 'options',
        'target_dataset': 'options',
        'source_pattern': 'yahoo',
        'target_pattern': 'marketwatch',
        'description': 'Same option contract quoted by multiple sources',
        'relationship': MARKET_ENRICHMENT.multiSourceQuoteLink,
        'lag_months': 0,
        'strength': 'strong'
    },
]