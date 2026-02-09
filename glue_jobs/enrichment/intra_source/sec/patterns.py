"""
SEC Sector Patterns - Industry and violation type definitions
"""
from glue_jobs.utils.rdf_utils import SEC_ENRICHMENT

SEC_SECTOR_PATTERNS = {
    'financial_services_sector': {
        'description': 'Financial services and investment industry',
        'sector_uri': SEC_ENRICHMENT.FinancialServicesSector,
        'keywords': {
            'filings': ['Investment Adviser', 'Broker-Dealer', 'Investment Company'],
            'administrative_proceedings': ['Investment Adviser', 'Broker-Dealer', 'Transfer Agent'],
            'litigation': ['Investment Adviser', 'Broker-Dealer'],
            'trading_suspensions': ['Broker-Dealer']
        },
        'relationship': SEC_ENRICHMENT.financialServicesSectorCorrelation
    },

    'accounting_sector': {
        'description': 'Accounting and auditing profession',
        'sector_uri': SEC_ENRICHMENT.AccountingSector,
        'keywords': {
            'filings': [],
            'administrative_proceedings': ['Accountant', 'CPA', 'Auditor'],
            'litigation': ['Accountant', 'Auditor'],
            'trading_suspensions': []
        },
        'relationship': SEC_ENRICHMENT.accountingSectorCorrelation
    },

    'legal_sector': {
        'description': 'Legal profession and attorneys',
        'sector_uri': SEC_ENRICHMENT.LegalSector,
        'keywords': {
            'filings': [],
            'administrative_proceedings': ['Attorney'],
            'litigation': ['Attorney'],
            'trading_suspensions': []
        },
        'relationship': SEC_ENRICHMENT.legalSectorCorrelation
    },

    'technology_sector': {
        'description': 'Technology companies',
        'sector_uri': SEC_ENRICHMENT.TechnologySector,
        'keywords': {
            'filings': ['Technology', 'Software', 'Internet'],
            'administrative_proceedings': [],
            'litigation': ['Technology', 'Software'],
            'trading_suspensions': ['Technology']
        },
        'relationship': SEC_ENRICHMENT.technologySectorCorrelation
    },

    'healthcare_sector': {
        'description': 'Healthcare and pharmaceutical companies',
        'sector_uri': SEC_ENRICHMENT.HealthcareSector,
        'keywords': {
            'filings': ['Healthcare', 'Pharmaceutical', 'Biotech', 'Medical'],
            'administrative_proceedings': [],
            'litigation': ['Healthcare', 'Pharmaceutical'],
            'trading_suspensions': ['Healthcare', 'Pharmaceutical']
        },
        'relationship': SEC_ENRICHMENT.healthcareSectorCorrelation
    },

    'energy_sector': {
        'description': 'Energy and natural resources companies',
        'sector_uri': SEC_ENRICHMENT.EnergySector,
        'keywords': {
            'filings': ['Energy', 'Oil', 'Gas', 'Mining'],
            'administrative_proceedings': [],
            'litigation': ['Energy', 'Oil', 'Gas'],
            'trading_suspensions': ['Energy', 'Mining']
        },
        'relationship': SEC_ENRICHMENT.energySectorCorrelation
    },

    'real_estate_sector': {
        'description': 'Real estate and REIT companies',
        'sector_uri': SEC_ENRICHMENT.RealEstateSector,
        'keywords': {
            'filings': ['Real Estate', 'REIT'],
            'administrative_proceedings': [],
            'litigation': ['Real Estate'],
            'trading_suspensions': ['Real Estate']
        },
        'relationship': SEC_ENRICHMENT.realEstateSectorCorrelation
    },
}

# Violation type patterns for linking related violations
SEC_VIOLATION_PATTERNS = {
    'securities_fraud': {
        'description': 'Securities fraud violations',
        'violation_uri': SEC_ENRICHMENT.SecuritiesFraudViolation,
        'keywords': {
            'administrative_proceedings': ['Securities Fraud', 'Section 10(b)', 'Rule 10b-5', 'Section 17(a)'],
            'litigation': ['Securities Fraud', 'Section 10(b)', 'Rule 10b-5', 'Section 17(a)'],
        },
        'relationship': SEC_ENRICHMENT.securitiesFraudLink
    },

    'insider_trading': {
        'description': 'Insider trading violations',
        'violation_uri': SEC_ENRICHMENT.InsiderTradingViolation,
        'keywords': {
            'filings': ['Form 4', 'Form 3', 'Form 5'],  # Ownership forms
            'administrative_proceedings': ['Insider Trading'],
            'litigation': ['Insider Trading'],
        },
        'relationship': SEC_ENRICHMENT.insiderTradingLink
    },

    'reporting_violations': {
        'description': 'Reporting and disclosure violations',
        'violation_uri': SEC_ENRICHMENT.ReportingViolation,
        'keywords': {
            'filings': ['Form 10-K', 'Form 10-Q', 'Form 8-K'],
            'administrative_proceedings': ['Reporting Violation', 'Periodic Reporting'],
            'litigation': ['Reporting Violation'],
            'trading_suspensions': ['Lack of Current Information']
        },
        'relationship': SEC_ENRICHMENT.reportingViolationLink
    },

    'market_manipulation': {
        'description': 'Market manipulation violations',
        'violation_uri': SEC_ENRICHMENT.MarketManipulationViolation,
        'keywords': {
            'administrative_proceedings': ['Market Manipulation'],
            'litigation': ['Market Manipulation'],
            'trading_suspensions': ['Market Manipulation', 'Pump and Dump']
        },
        'relationship': SEC_ENRICHMENT.marketManipulationLink
    },

    'offering_fraud': {
        'description': 'Fraudulent securities offerings',
        'violation_uri': SEC_ENRICHMENT.OfferingFraudViolation,
        'keywords': {
            'administrative_proceedings': ['Section 5', 'Unregistered Offering'],
            'litigation': ['Section 5', 'Offering Fraud'],
        },
        'relationship': SEC_ENRICHMENT.offeringFraudLink
    },

    'fiduciary_breach': {
        'description': 'Breach of fiduciary duty',
        'violation_uri': SEC_ENRICHMENT.FiduciaryBreachViolation,
        'keywords': {
            'administrative_proceedings': ['Fiduciary Breach'],
            'litigation': ['Fiduciary Breach', 'Investment Adviser Fraud'],
        },
        'relationship': SEC_ENRICHMENT.fiduciaryBreachLink
    },
}

# Company status patterns for linking related company states
SEC_COMPANY_STATUS_PATTERNS = {
    'shell_company': {
        'description': 'Shell companies with no operations',
        'status_uri': SEC_ENRICHMENT.ShellCompanyStatus,
        'keywords': {
            'trading_suspensions': ['Shell Company'],
        },
        'relationship': SEC_ENRICHMENT.shellCompanyLink
    },

    'delinquent_filer': {
        'description': 'Companies delinquent in SEC filings',
        'status_uri': SEC_ENRICHMENT.DelinquentFilerStatus,
        'keywords': {
            'administrative_proceedings': ['Reporting Violation', 'Failure to File'],
            'trading_suspensions': ['Lack of Current Information'],
        },
        'relationship': SEC_ENRICHMENT.delinquentFilerLink
    },

    'suspended_trading': {
        'description': 'Companies with suspended trading',
        'status_uri': SEC_ENRICHMENT.SuspendedTradingStatus,
        'keywords': {
            'trading_suspensions': ['Trading Suspension', 'Section 12(k)'],
            'administrative_proceedings': ['Section 12(j)'],
        },
        'relationship': SEC_ENRICHMENT.suspendedTradingLink
    },
}