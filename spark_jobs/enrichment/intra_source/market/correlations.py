"""
Market Known Correlations - Relationships between market data types.

Aligned with the flat snapshot model (EquitySnapshot, OptionSnapshot).
Correlations link snapshots that share temporal, underlying, or
structural relationships.
"""
from spark_jobs.utils.rdf_utils import MARKET_ENRICHMENT

KNOWN_CORRELATIONS = [

    # ============================================
    # EQUITY ↔ OPTION CORRELATIONS
    # ============================================

    {
        'name': 'equity_to_call_options',
        'source_type': 'equity',
        'target_type': 'option',
        'target_contract_type': 'CALL',
        'description': 'Equity price movements affect call option values',
        'relationship': MARKET_ENRICHMENT.equityCallOptionLink,
        'strength': 'strong',
    },

    {
        'name': 'equity_to_put_options',
        'source_type': 'equity',
        'target_type': 'option',
        'target_contract_type': 'PUT',
        'description': 'Equity price movements affect put option values',
        'relationship': MARKET_ENRICHMENT.equityPutOptionLink,
        'strength': 'strong',
    },

    {
        'name': 'at_the_money_options',
        'source_type': 'equity',
        'target_type': 'option',
        'description': 'Options with strike near current equity price (ATM)',
        'relationship': MARKET_ENRICHMENT.atTheMoneyLink,
        'strength': 'strong',
    },

    # ============================================
    # INTERNAL CORRELATIONS - OPTIONS
    # ============================================

    {
        'name': 'call_put_parity',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'strikePrice', 'expirationDate', 'captureTime'],
        'source_contract_type': 'CALL',
        'target_contract_type': 'PUT',
        'description': 'Call and put at same strike/expiration (put-call parity)',
        'relationship': MARKET_ENRICHMENT.putCallParityLink,
        'strength': 'strong',
    },

    {
        'name': 'same_expiration_chain',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'contractType', 'captureTime'],
        'description': 'Options with same underlying, expiration, and type form a chain',
        'relationship': MARKET_ENRICHMENT.sameExpirationChainLink,
        'strength': 'medium',
    },

    {
        'name': 'adjacent_strike_options',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'contractType', 'captureTime'],
        'description': 'Options with adjacent strike prices in the same chain',
        'relationship': MARKET_ENRICHMENT.adjacentStrikeLink,
        'strength': 'medium',
    },

    {
        'name': 'expiration_series',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'strikePrice', 'contractType', 'captureTime'],
        'description': 'Options at same strike across different expirations (calendar spread)',
        'relationship': MARKET_ENRICHMENT.expirationSeriesLink,
        'strength': 'medium',
    },

    # ============================================
    # INTERNAL CORRELATIONS - EQUITY SNAPSHOTS
    # ============================================

    {
        'name': 'equity_snapshot_sequence',
        'source_type': 'equity',
        'target_type': 'equity',
        'match_on': ['symbol'],
        'description': 'Sequential equity snapshots for same ticker (temporal)',
        'relationship': MARKET_ENRICHMENT.snapshotSequenceLink,
        'strength': 'strong',
    },

    {
        'name': 'same_sector_equities',
        'source_type': 'equity',
        'target_type': 'equity',
        'match_on': ['sector'],
        'description': 'Equities in the same sector tend to correlate',
        'relationship': MARKET_ENRICHMENT.sameSectorEquityLink,
        'strength': 'medium',
    },

    {
        'name': 'same_capture_equities',
        'source_type': 'equity',
        'target_type': 'equity',
        'match_on': ['captureTime'],
        'description': 'Equities captured in the same snapshot window (cross-sectional)',
        'relationship': MARKET_ENRICHMENT.sameCaptureLink,
        'strength': 'weak',
    },

    # ============================================
    # OPTION STRATEGY CORRELATIONS
    # ============================================

    {
        'name': 'straddle_pair',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'strikePrice', 'expirationDate', 'captureTime'],
        'source_contract_type': 'CALL',
        'target_contract_type': 'PUT',
        'description': 'Straddle: call + put at same strike and expiration',
        'relationship': MARKET_ENRICHMENT.straddleWith,
        'strength': 'strong',
    },

    {
        'name': 'call_spread_pair',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'captureTime'],
        'source_contract_type': 'CALL',
        'target_contract_type': 'CALL',
        'description': 'Vertical call spread: adjacent call strikes',
        'relationship': MARKET_ENRICHMENT.callSpreadWith,
        'strength': 'strong',
    },

    {
        'name': 'put_spread_pair',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'captureTime'],
        'source_contract_type': 'PUT',
        'target_contract_type': 'PUT',
        'description': 'Vertical put spread: adjacent put strikes',
        'relationship': MARKET_ENRICHMENT.putSpreadWith,
        'strength': 'strong',
    },

    {
        'name': 'strangle_pair',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'captureTime'],
        'source_contract_type': 'CALL',
        'target_contract_type': 'PUT',
        'description': 'Strangle: OTM call + OTM put at different strikes',
        'relationship': MARKET_ENRICHMENT.strangleWith,
        'strength': 'strong',
    },

    {
        'name': 'iron_condor',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'captureTime'],
        'description': 'Iron condor: sell call spread + sell put spread',
        'relationship': MARKET_ENRICHMENT.ironCondorLink,
        'strength': 'strong',
    },

    {
        'name': 'butterfly_spread',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'contractType', 'captureTime'],
        'description': 'Butterfly spread: three strikes, same expiration and type',
        'relationship': MARKET_ENRICHMENT.butterflyLink,
        'strength': 'strong',
    },

    # ============================================
    # GREEKS-BASED CORRELATIONS
    # ============================================

    {
        'name': 'similar_delta_options',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'captureTime'],
        'description': 'Options with similar delta values (similar directional exposure)',
        'relationship': MARKET_ENRICHMENT.similarDeltaLink,
        'strength': 'medium',
    },

    {
        'name': 'high_gamma_cluster',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'captureTime'],
        'description': 'Options with high gamma (near ATM, sensitive to price moves)',
        'relationship': MARKET_ENRICHMENT.highGammaClusterLink,
        'strength': 'medium',
    },

    {
        'name': 'high_vega_cluster',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'captureTime'],
        'description': 'Options with high vega (sensitive to volatility changes)',
        'relationship': MARKET_ENRICHMENT.highVegaClusterLink,
        'strength': 'medium',
    },

    # ============================================
    # TEMPORAL CORRELATIONS
    # ============================================

    {
        'name': 'option_snapshot_sequence',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['symbol'],
        'description': 'Sequential option snapshots for same contract (temporal)',
        'relationship': MARKET_ENRICHMENT.snapshotSequenceLink,
        'strength': 'strong',
    },

    {
        'name': 'underlying_equity_temporal',
        'source_type': 'option',
        'target_type': 'equity',
        'match_on': ['underlyingSymbol_to_symbol', 'captureTime'],
        'description': 'Option snapshot linked to same-time underlying equity snapshot',
        'relationship': MARKET_ENRICHMENT.underlyingEquityTemporalLink,
        'strength': 'strong',
    },

    # ============================================
    # MONEYNESS CORRELATIONS
    # ============================================

    {
        'name': 'itm_to_otm_transition',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['symbol'],
        'description': 'Same option contract transitioning between ITM and OTM over time',
        'relationship': MARKET_ENRICHMENT.moneynessTransitionLink,
        'strength': 'medium',
    },

    {
        'name': 'same_moneyness_cluster',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'captureTime', 'moneyness'],
        'description': 'Options with same moneyness classification in same chain',
        'relationship': MARKET_ENRICHMENT.sameMoneynessLink,
        'strength': 'weak',
    },

    # ============================================
    # VOLUME / LIQUIDITY CORRELATIONS
    # ============================================

    {
        'name': 'high_volume_equity_to_options',
        'source_type': 'equity',
        'target_type': 'option',
        'description': 'High-volume equity snapshots correlate with active option chains',
        'relationship': MARKET_ENRICHMENT.highVolumeOptionActivityLink,
        'strength': 'medium',
    },

    {
        'name': 'high_open_interest_options',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'expirationDate', 'captureTime'],
        'description': 'Options with high open interest in same chain (liquidity cluster)',
        'relationship': MARKET_ENRICHMENT.highOpenInterestClusterLink,
        'strength': 'medium',
    },
]