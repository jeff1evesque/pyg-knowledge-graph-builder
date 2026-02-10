"""
Market Sector Patterns - Industry sectors and market segments
"""
from glue_jobs.utils.rdf_utils import MARKET_ENRICHMENT

MARKET_SECTOR_PATTERNS = {
    'technology_sector': {
        'description': 'Technology companies and stocks',
        'sector_uri': MARKET_ENRICHMENT.TechnologySector,
        'keywords': {
            'stock_prices': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.technologySectorCorrelation
    },

    'automotive_sector': {
        'description': 'Automotive and EV companies',
        'sector_uri': MARKET_ENRICHMENT.AutomotiveSector,
        'keywords': {
            'stock_prices': ['TSLA', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.automotiveSectorCorrelation
    },

    'financial_sector': {
        'description': 'Financial services companies',
        'sector_uri': MARKET_ENRICHMENT.FinancialSector,
        'keywords': {
            'stock_prices': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.financialSectorCorrelation
    },

    'healthcare_sector': {
        'description': 'Healthcare and pharmaceutical companies',
        'sector_uri': MARKET_ENRICHMENT.HealthcareSector,
        'keywords': {
            'stock_prices': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.healthcareSectorCorrelation
    },

    'energy_sector': {
        'description': 'Energy and oil companies',
        'sector_uri': MARKET_ENRICHMENT.EnergySector,
        'keywords': {
            'stock_prices': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.energySectorCorrelation
    },

    'consumer_discretionary_sector': {
        'description': 'Consumer discretionary companies',
        'sector_uri': MARKET_ENRICHMENT.ConsumerDiscretionarySector,
        'keywords': {
            'stock_prices': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.consumerDiscretionarySectorCorrelation
    },

    'consumer_staples_sector': {
        'description': 'Consumer staples companies',
        'sector_uri': MARKET_ENRICHMENT.ConsumerStaplesSector,
        'keywords': {
            'stock_prices': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.consumerStaplesSectorCorrelation
    },

    'industrials_sector': {
        'description': 'Industrial companies',
        'sector_uri': MARKET_ENRICHMENT.IndustrialsSector,
        'keywords': {
            'stock_prices': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.industrialsSectorCorrelation
    },

    'real_estate_sector': {
        'description': 'Real estate and REIT companies',
        'sector_uri': MARKET_ENRICHMENT.RealEstateSector,
        'keywords': {
            'stock_prices': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.realEstateSectorCorrelation
    },

    'utilities_sector': {
        'description': 'Utility companies',
        'sector_uri': MARKET_ENRICHMENT.UtilitiesSector,
        'keywords': {
            'stock_prices': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.utilitiesSectorCorrelation
    },

    'materials_sector': {
        'description': 'Materials and commodities companies',
        'sector_uri': MARKET_ENRICHMENT.MaterialsSector,
        'keywords': {
            'stock_prices': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DD', 'DOW'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.materialsSectorCorrelation
    },

    'communication_services_sector': {
        'description': 'Communication services companies',
        'sector_uri': MARKET_ENRICHMENT.CommunicationServicesSector,
        'keywords': {
            'stock_prices': ['META', 'GOOGL', 'GOOG', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T'],
            'options': []
        },
        'relationship': MARKET_ENRICHMENT.communicationServicesSectorCorrelation
    },
}

# Option strategy patterns for linking related options
MARKET_OPTION_PATTERNS = {
    'call_spread': {
        'description': 'Vertical call spread (buy lower strike, sell higher strike)',
        'pattern_uri': MARKET_ENRICHMENT.CallSpreadPattern,
        'relationship': MARKET_ENRICHMENT.callSpreadLink
    },

    'put_spread': {
        'description': 'Vertical put spread (buy higher strike, sell lower strike)',
        'pattern_uri': MARKET_ENRICHMENT.PutSpreadPattern,
        'relationship': MARKET_ENRICHMENT.putSpreadLink
    },

    'straddle': {
        'description': 'Long straddle (buy call and put at same strike)',
        'pattern_uri': MARKET_ENRICHMENT.StraddlePattern,
        'relationship': MARKET_ENRICHMENT.straddleLink
    },

    'strangle': {
        'description': 'Long strangle (buy call and put at different strikes)',
        'pattern_uri': MARKET_ENRICHMENT.StranglePattern,
        'relationship': MARKET_ENRICHMENT.strangleLink
    },

    'iron_condor': {
        'description': 'Iron condor (sell call spread and put spread)',
        'pattern_uri': MARKET_ENRICHMENT.IronCondorPattern,
        'relationship': MARKET_ENRICHMENT.ironCondorLink
    },

    'butterfly': {
        'description': 'Butterfly spread (three strikes, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.ButterflyPattern,
        'relationship': MARKET_ENRICHMENT.butterflyLink
    },
}

# Exchange patterns for linking related trading venues
MARKET_EXCHANGE_PATTERNS = {
    'us_major_exchanges': {
        'description': 'Major US stock exchanges',
        'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'NYSEARCA'],
        'region': 'United States',
        'relationship': MARKET_ENRICHMENT.usMajorExchangeLink
    },

    'us_otc_markets': {
        'description': 'US over-the-counter markets',
        'exchanges': ['OTC', 'OTCQX', 'OTCQB', 'OTCBB', 'PINK', 'GREY'],
        'region': 'United States',
        'relationship': MARKET_ENRICHMENT.usOtcMarketLink
    },

    'european_exchanges': {
        'description': 'Major European stock exchanges',
        'exchanges': ['LSE', 'XETRA', 'FRA', 'EPA', 'AMS', 'BRU', 'MIL', 'SWX'],
        'region': 'Europe',
        'relationship': MARKET_ENRICHMENT.europeanExchangeLink
    },

    'asian_exchanges': {
        'description': 'Major Asian stock exchanges',
        'exchanges': ['TSE', 'SSE', 'SZSE', 'HKEX', 'KRX', 'SGX', 'NSE', 'BSE'],
        'region': 'Asia',
        'relationship': MARKET_ENRICHMENT.asianExchangeLink
    },
}