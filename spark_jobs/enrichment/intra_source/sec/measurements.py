"""
SEC Measurement Types - Temporal linking configurations for SEC datasets

Defines which RDF classes, date properties, and grouping properties
are needed to build temporal sequences within each SEC dataset.

Property references come from the rdf_utils constants and are deliberately not
spelled out here -- a copy in a docstring goes stale exactly the way a copy in
code does, and this one did:

- Filings:     SEC_FILINGS, plus SEC_COMMON for the shared classes

Filings is the only SEC feed collected; the administrative-proceedings,
litigation and trading-suspension blocks were removed along with the linker
paths that read them.

These were once described as the SEC's "actual ontologies". They are not: the
SEC publishes filings as XML and no RDF vocabulary at all, so every term above
was minted by this project's scrapers and now resolves under a domain we
control.

Used by sec_linker.py for:
- _link_ownership_filing_sequences()
- _link_periodic_reporting_sequences()
"""
from spark_jobs.utils.rdf_utils import SEC_FILINGS, SEC_COMMON

# The shared SEC classes (Date, Company, Person, Address). Re-declared as a
# literal until this change, which meant it kept naming sec.gov after the
# vocabulary moved -- matching nothing, silently.
SEC_BASE = SEC_COMMON

MEASUREMENT_TYPES = {

    # ============================================
    # FILINGS - Ownership Documents (Forms 3, 4, 5)
    # ============================================
    'filings': {

        'Form3': {
            'class': SEC_FILINGS.Form3,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'owner_property': SEC_FILINGS.hasReportingOwner,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'owner_cik_property': SEC_FILINGS.hasReportingOwnerCik,
            'description': 'Initial Statement of Beneficial Ownership',
        },

        'Form4': {
            'class': SEC_FILINGS.Form4,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'owner_property': SEC_FILINGS.hasReportingOwner,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'owner_cik_property': SEC_FILINGS.hasReportingOwnerCik,
            'transaction_date_property': SEC_FILINGS.hasTransactionDate,
            'description': 'Statement of Changes in Beneficial Ownership',
        },

        'Form5': {
            'class': SEC_FILINGS.Form5,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'owner_property': SEC_FILINGS.hasReportingOwner,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'owner_cik_property': SEC_FILINGS.hasReportingOwnerCik,
            'description': 'Annual Statement of Changes in Beneficial Ownership',
        },

        'Form10K': {
            'class': SEC_FILINGS.Form10K,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'description': 'Annual Report',
        },

        'Form10Q': {
            'class': SEC_FILINGS.Form10Q,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'description': 'Quarterly Report',
        },

        'Form8K': {
            'class': SEC_FILINGS.Form8K,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'description': 'Current Report',
        },

        # Generic ownership document (when specific form type not determined)
        'OwnershipDocument': {
            'class': SEC_FILINGS.OwnershipDocument,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasIssuer,
            'owner_property': SEC_FILINGS.hasReportingOwner,
            'cik_property': SEC_FILINGS.hasIssuerCik,
            'owner_cik_property': SEC_FILINGS.hasReportingOwnerCik,
            'description': 'Generic ownership document',
        },

        # Transactions within filings (for sub-entity temporal linking)
        'NonDerivativeTransaction': {
            'class': SEC_FILINGS.NonDerivativeTransaction,
            'date_property': SEC_FILINGS.hasTransactionDate,
            'date_value_property': SEC_BASE.hasDateValue,
            'description': 'Non-derivative transaction within an ownership filing',
        },

        'DerivativeTransaction': {
            'class': SEC_FILINGS.DerivativeTransaction,
            'date_property': SEC_FILINGS.hasTransactionDate,
            'date_value_property': SEC_BASE.hasDateValue,
            'expiration_property': SEC_FILINGS.hasExpirationDate,
            'exercise_property': SEC_FILINGS.hasExerciseDate,
            'description': 'Derivative transaction within an ownership filing',
        },
    },



}


# ============================================
# ENTITY MATCHING PROPERTIES
# ============================================
# Properties used to match entities across SEC datasets
# for entity unification (company/person resolution)

ENTITY_MATCHING_PROPERTIES = {
    'company_cik': {
        'filings': SEC_FILINGS.hasIssuerCik,
    },
    'company_ticker': {
        'filings': SEC_FILINGS.hasIssuerTradingSymbol,
    },
    'company_name': {
        'filings': SEC_FILINGS.hasIssuerName,
    },
    'person_cik': {
        'filings': SEC_FILINGS.hasReportingOwnerCik,
    },
    'person_name': {
    },
}