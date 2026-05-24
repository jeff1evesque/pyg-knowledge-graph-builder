"""
Market Sector Patterns - Industry sectors and market segments.

Aligned with the flat snapshot model. Sector classification is
based on the equity ticker symbol (the `symbol` property on
EquitySnapshot nodes).

Option snapshots inherit sector classification from their
underlying equity via the `underlyingSymbol` property.
"""
from glue_jobs.utils.rdf_utils import MARKET_ENRICHMENT

MARKET_SECTOR_PATTERNS = {
    'technology_sector': {
        'description': 'Technology companies and stocks',
        'sector_uri': MARKET_ENRICHMENT.TechnologySector,
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN',
                    'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'QCOM'],
        'relationship': MARKET_ENRICHMENT.technologySectorCorrelation,
    },

    'automotive_sector': {
        'description': 'Automotive and EV companies',
        'sector_uri': MARKET_ENRICHMENT.AutomotiveSector,
        'tickers': ['TSLA', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI'],
        'relationship': MARKET_ENRICHMENT.automotiveSectorCorrelation,
    },

    'financial_sector': {
        'description': 'Financial services companies',
        'sector_uri': MARKET_ENRICHMENT.FinancialSector,
        'tickers': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',
                    'AXP', 'V', 'MA', 'COF'],
        'relationship': MARKET_ENRICHMENT.financialSectorCorrelation,
    },

    'healthcare_sector': {
        'description': 'Healthcare and pharmaceutical companies',
        'sector_uri': MARKET_ENRICHMENT.HealthcareSector,
        'tickers': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK',
                    'LLY', 'BMY', 'AMGN', 'GILD'],
        'relationship': MARKET_ENRICHMENT.healthcareSectorCorrelation,
    },

    'energy_sector': {
        'description': 'Energy and oil companies',
        'sector_uri': MARKET_ENRICHMENT.EnergySector,
        'tickers': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
                    'OXY', 'HAL', 'DVN', 'FANG'],
        'relationship': MARKET_ENRICHMENT.energySectorCorrelation,
    },

    'consumer_discretionary_sector': {
        'description': 'Consumer discretionary companies',
        'sector_uri': MARKET_ENRICHMENT.ConsumerDiscretionarySector,
        'tickers': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
                    'BKNG', 'TJX', 'CMG', 'LULU'],
        'relationship': MARKET_ENRICHMENT.consumerDiscretionarySectorCorrelation,
    },

    'consumer_staples_sector': {
        'description': 'Consumer staples companies',
        'sector_uri': MARKET_ENRICHMENT.ConsumerStaplesSector,
        'tickers': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL',
                    'MDLZ', 'KHC', 'GIS', 'SYY'],
        'relationship': MARKET_ENRICHMENT.consumerStaplesSectorCorrelation,
    },

    'industrials_sector': {
        'description': 'Industrial companies',
        'sector_uri': MARKET_ENRICHMENT.IndustrialsSector,
        'tickers': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
                    'DE', 'UNP', 'FDX', 'WM'],
        'relationship': MARKET_ENRICHMENT.industrialsSectorCorrelation,
    },

    'real_estate_sector': {
        'description': 'Real estate and REIT companies',
        'sector_uri': MARKET_ENRICHMENT.RealEstateSector,
        'tickers': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL',
                    'DLR', 'AVB', 'EQR', 'VTR'],
        'relationship': MARKET_ENRICHMENT.realEstateSectorCorrelation,
    },

    'utilities_sector': {
        'description': 'Utility companies',
        'sector_uri': MARKET_ENRICHMENT.UtilitiesSector,
        'tickers': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
                    'WEC', 'ED', 'ES', 'AWK'],
        'relationship': MARKET_ENRICHMENT.utilitiesSectorCorrelation,
    },

    'materials_sector': {
        'description': 'Materials and commodities companies',
        'sector_uri': MARKET_ENRICHMENT.MaterialsSector,
        'tickers': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DD', 'DOW',
                    'NUE', 'VMC', 'MLM', 'PPG'],
        'relationship': MARKET_ENRICHMENT.materialsSectorCorrelation,
    },

    'communication_services_sector': {
        'description': 'Communication services companies',
        'sector_uri': MARKET_ENRICHMENT.CommunicationServicesSector,
        'tickers': ['META', 'GOOGL', 'GOOG', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T',
                    'TMUS', 'CHTR', 'EA', 'TTWO'],
        'relationship': MARKET_ENRICHMENT.communicationServicesSectorCorrelation,
    },
}

# Option strategy patterns for linking related options
MARKET_OPTION_STRATEGY_PATTERNS = {
    'straddle': {
        'description': 'Long straddle (call and put at same strike/expiration)',
        'pattern_uri': MARKET_ENRICHMENT.StraddlePattern,
        'relationship': MARKET_ENRICHMENT.straddleWith,
    },

    'call_spread': {
        'description': 'Vertical call spread (adjacent strikes, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.CallSpreadPattern,
        'relationship': MARKET_ENRICHMENT.callSpreadWith,
    },

    'put_spread': {
        'description': 'Vertical put spread (adjacent strikes, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.PutSpreadPattern,
        'relationship': MARKET_ENRICHMENT.putSpreadWith,
    },

    'strangle': {
        'description': 'Long strangle (OTM call and OTM put, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.StranglePattern,
        'relationship': MARKET_ENRICHMENT.strangleWith,
    },
}