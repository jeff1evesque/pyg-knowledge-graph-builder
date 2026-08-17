"""
SEC Sector Patterns - Industry and violation type definitions
"""
from spark_jobs.utils.rdf_utils import SEC_ENRICHMENT

SEC_SECTOR_PATTERNS = {
    'financial_services_sector': {
        'description': 'Financial services and investment industry',
        'sector_uri': SEC_ENRICHMENT.FinancialServicesSector,
        'keywords': {
            'filings': ['Investment Adviser', 'Broker-Dealer', 'Investment Company']
        },
        'relationship': SEC_ENRICHMENT.financialServicesSectorCorrelation
    },



    'technology_sector': {
        'description': 'Technology companies',
        'sector_uri': SEC_ENRICHMENT.TechnologySector,
        'keywords': {
            'filings': ['Technology', 'Software', 'Internet']
        },
        'relationship': SEC_ENRICHMENT.technologySectorCorrelation
    },

    'healthcare_sector': {
        'description': 'Healthcare and pharmaceutical companies',
        'sector_uri': SEC_ENRICHMENT.HealthcareSector,
        'keywords': {
            'filings': ['Healthcare', 'Pharmaceutical', 'Biotech', 'Medical']
        },
        'relationship': SEC_ENRICHMENT.healthcareSectorCorrelation
    },

    'energy_sector': {
        'description': 'Energy and natural resources companies',
        'sector_uri': SEC_ENRICHMENT.EnergySector,
        'keywords': {
            'filings': ['Energy', 'Oil', 'Gas', 'Mining']
        },
        'relationship': SEC_ENRICHMENT.energySectorCorrelation
    },

    'real_estate_sector': {
        'description': 'Real estate and REIT companies',
        'sector_uri': SEC_ENRICHMENT.RealEstateSector,
        'keywords': {
            'filings': ['Real Estate', 'REIT']
        },
        'relationship': SEC_ENRICHMENT.realEstateSectorCorrelation
    },
}

# Violation type patterns for linking related violations
SEC_VIOLATION_PATTERNS = {

    'insider_trading': {
        'description': 'Insider trading violations',
        'violation_uri': SEC_ENRICHMENT.InsiderTradingViolation,
        'keywords': {
            'filings': ['Form 4', 'Form 3', 'Form 5'],  # Ownership forms
        },
        'relationship': SEC_ENRICHMENT.insiderTradingLink
    },

    'reporting_violations': {
        'description': 'Reporting and disclosure violations',
        'violation_uri': SEC_ENRICHMENT.ReportingViolation,
        'keywords': {
            'filings': ['Form 10-K', 'Form 10-Q', 'Form 8-K']
        },
        'relationship': SEC_ENRICHMENT.reportingViolationLink
    },



}