"""
Market Known Correlations - Relationships between market data types.

Aligned with the flat snapshot model (EquitySnapshot, OptionSnapshot).
Correlations link snapshots that share temporal, underlying, or
structural relationships.
"""
from glue_jobs.utils.rdf_utils import MARKET_ENRICHMENT

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

    # ============================================
    # INTERNAL CORRELATIONS - OPTIONS
    # ============================================

    {
        'name': 'call_put_parity',
        'source_type': 'option',
        'target_type': 'option',
        'match_on': ['underlyingSymbol', 'strikePrice', 'expirationDate'],
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
        'match_on': ['underlyingSymbol', 'expirationDate', 'contractType'],
        'description': 'Options with same underlying, expiration, and type form a chain',
        'relationship': MARKET_ENRICHMENT.sameExpirationChainLink,
        'strength': 'medium',
    },

    # ============================================
    # INTERNAL CORRELATIONS - EQUITY SNAPSHOTS
    # ============================================

    {
        'name': 'same_sector_equities',
        'source_type': 'equity',
        'target_type': 'equity',
        'match_on': ['sector'],
        'description': 'Equities in the same sector tend to correlate',
        'relationship': MARKET_ENRICHMENT.sameSectorEquityLink,
        'strength': 'medium',
    },
]