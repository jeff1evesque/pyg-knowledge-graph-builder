"""
SEC Sector Patterns - Industry and violation type definitions
"""
from spark_jobs.utils.rdf_utils import SEC_ENRICHMENT

# SEC_SECTOR_PATTERNS was removed.
#
# It sorted filings into a THIRD sector vocabulary (sec:financialServicesSector
# and four others) by matching keywords against filing URIs and rdfs:labels --
# a parallel taxonomy to the BLS economic sectors and the GICS ones, reconciled
# with neither. On a real build it matched NOTHING: zero sector nodes, zero
# edges. A filing's local name is an accession number, so there was nothing for
# a keyword to hit.
#
# Filings now reach a sector through filings:hasSic, the industry code the SEC
# itself assigns, which CrossSourceLinker attaches to the company rather than
# to the document. That is real data instead of a guess at a substring.

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