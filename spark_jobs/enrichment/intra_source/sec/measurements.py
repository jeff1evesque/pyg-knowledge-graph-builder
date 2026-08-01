"""
SEC Measurement Types - Temporal linking configurations for SEC datasets

Defines which RDF classes, date properties, and grouping properties
are needed to build temporal sequences within each SEC dataset.

Property references match actual ontologies:
- Filings:     http://www.sec.gov/filings#  and  http://www.sec.gov#
- Admin:       https://www.sec.gov/ontology/administrative-proceedings#
- Litigation:  https://www.sec.gov/ontology/litigation#
- Suspensions: https://www.sec.gov/ontology/trading-suspensions#

Used by sec_linker.py for:
- _link_ownership_filing_sequences()
- _link_periodic_reporting_sequences()
- _link_administrative_proceeding_sequences()
- _link_litigation_sequences()
- _link_trading_suspension_sequences()
"""
from spark_jobs.utils.rdf_utils import (
    SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP, SEC_COMMON,
)

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
            'grouping_property': SEC_FILINGS.hasCompany,
            'cik_property': SEC_BASE.hasCik,
            'description': 'Annual Report',
        },

        'Form10Q': {
            'class': SEC_FILINGS.Form10Q,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasCompany,
            'cik_property': SEC_BASE.hasCik,
            'description': 'Quarterly Report',
        },

        'Form8K': {
            'class': SEC_FILINGS.Form8K,
            'date_property': SEC_FILINGS.hasPeriodOfReport,
            'date_value_property': SEC_BASE.hasDateValue,
            'grouping_property': SEC_FILINGS.hasCompany,
            'cik_property': SEC_BASE.hasCik,
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

    # ============================================
    # ADMINISTRATIVE PROCEEDINGS
    # ============================================
    'administrative_proceedings': {

        'AdministrativeProceeding': {
            'class': SEC_ADMIN.AdministrativeProceeding,
            'date_property': SEC_ADMIN.initiationDate,
            'grouping_property': SEC_ADMIN.hasRespondent,
            'cik_property': SEC_ADMIN.cikNumber,
            'ticker_property': SEC_ADMIN.tickerSymbol,
            'name_property': SEC_ADMIN.respondentName,
            'description': 'SEC administrative enforcement proceeding',
        },

        'Rule102eProceeding': {
            'class': SEC_ADMIN.Rule102eProceeding,
            'date_property': SEC_ADMIN.initiationDate,
            'grouping_property': SEC_ADMIN.hasRespondent,
            'description': 'Rule 102(e) proceeding for professional conduct violations',
        },

        'CeaseAndDesistProceeding': {
            'class': SEC_ADMIN.CeaseAndDesistProceeding,
            'date_property': SEC_ADMIN.initiationDate,
            'grouping_property': SEC_ADMIN.hasRespondent,
            'description': 'Proceeding seeking cease and desist orders',
        },

        'Section12jProceeding': {
            'class': SEC_ADMIN.Section12jProceeding,
            'date_property': SEC_ADMIN.initiationDate,
            'grouping_property': SEC_ADMIN.hasRespondent,
            'description': 'Section 12(j) proceeding to revoke/suspend securities registration',
        },

        # Sanctions (have their own temporal properties)
        'Suspension': {
            'class': SEC_ADMIN.Suspension,
            'start_date_property': SEC_ADMIN.suspensionStartDate,
            'end_date_property': SEC_ADMIN.suspensionEndDate,
            'effective_date_property': SEC_ADMIN.effectiveDate,
            'grouping_property': SEC_ADMIN.imposedOn,
            'description': 'Suspension sanction from practice or industry',
        },

        # Documents (have filing dates)
        'OrderInstitutingProceedings': {
            'class': SEC_ADMIN.OrderInstitutingProceedings,
            'date_property': SEC_ADMIN.documentDate,
            'description': 'Order that initiates an administrative proceeding',
        },

        'CommissionOrder': {
            'class': SEC_ADMIN.CommissionOrder,
            'date_property': SEC_ADMIN.documentDate,
            'effective_date_property': SEC_ADMIN.effectiveDate,
            'description': 'Final order by the SEC Commission',
        },

        # Violations (have temporal bounds)
        'Violation': {
            'class': SEC_ADMIN.Violation,
            'start_date_property': SEC_ADMIN.violationStartDate,
            'end_date_property': SEC_ADMIN.violationEndDate,
            'grouping_property': SEC_ADMIN.committedBy,
            'description': 'Securities law violation with temporal period',
        },
    },

    # ============================================
    # LITIGATION RELEASES
    # ============================================
    'litigation': {

        'LitigationRelease': {
            'class': SEC_LIT.LitigationRelease,
            'date_property': SEC_LIT.releaseDate,
            'description': 'SEC litigation release announcing civil enforcement action',
        },

        'CivilEnforcementAction': {
            'class': SEC_LIT.CivilEnforcementAction,
            'date_property': SEC_LIT.filingDate,
            'grouping_property': SEC_LIT.hasDefendant,
            'court_property': SEC_LIT.filedInCourt,
            'name_property': SEC_LIT.caseTitle,
            'description': 'Civil enforcement action filed by the SEC',
        },

        'EmergencyAction': {
            'class': SEC_LIT.EmergencyAction,
            'date_property': SEC_LIT.filingDate,
            'grouping_property': SEC_LIT.hasDefendant,
            'description': 'Emergency enforcement action seeking immediate relief',
        },

        'Settlement': {
            'class': SEC_LIT.Settlement,
            'date_property': SEC_LIT.settlementDate,
            'description': 'Settlement agreement resolving litigation',
        },

        'ConsentJudgment': {
            'class': SEC_LIT.ConsentJudgment,
            'date_property': SEC_LIT.settlementDate,
            'description': 'Judgment entered by consent of parties',
        },

        'Judgment': {
            'class': SEC_LIT.Judgment,
            'date_property': SEC_LIT.documentDate,
            'description': 'Court judgment',
        },

        # Relief (monetary amounts with dates)
        'CivilPenalty': {
            'class': SEC_LIT.CivilPenalty,
            'amount_property': SEC_LIT.penaltyAmount,
            'description': 'Civil monetary penalty',
        },

        'Disgorgement': {
            'class': SEC_LIT.Disgorgement,
            'amount_property': SEC_LIT.disgorgementAmount,
            'description': 'Disgorgement of ill-gotten gains',
        },
    },

    # ============================================
    # TRADING SUSPENSIONS
    # ============================================
    'trading_suspensions': {

        'TradingSuspension': {
            'class': SEC_SUSP.TradingSuspension,
            'start_date_property': SEC_SUSP.startDate,
            'end_date_property': SEC_SUSP.endDate,
            'termination_date_property': SEC_SUSP.terminationDate,
            'grouping_property': SEC_SUSP.affectsCompany,
            'ticker_property': SEC_SUSP.tickerSymbol,
            'cik_property': SEC_SUSP.cikNumber,
            'reason_property': SEC_SUSP.hasReason,
            'description': 'SEC order suspending trading in a security',
        },

        'Section12kSuspension': {
            'class': SEC_SUSP.Section12kSuspension,
            'start_date_property': SEC_SUSP.startDate,
            'end_date_property': SEC_SUSP.endDate,
            'grouping_property': SEC_SUSP.affectsCompany,
            'description': 'Temporary trading suspension under Section 12(k)',
        },

        'Section12jSuspension': {
            'class': SEC_SUSP.Section12jSuspension,
            'start_date_property': SEC_SUSP.startDate,
            'end_date_property': SEC_SUSP.endDate,
            'grouping_property': SEC_SUSP.affectsCompany,
            'description': 'Trading suspension under Section 12(j)',
        },

        'EmergencySuspension': {
            'class': SEC_SUSP.EmergencySuspension,
            'start_date_property': SEC_SUSP.startDate,
            'end_date_property': SEC_SUSP.endDate,
            'grouping_property': SEC_SUSP.affectsCompany,
            'description': 'Emergency trading suspension for investor protection',
        },

        # Suspension reasons (no dates, but useful for linking)
        'LackOfCurrentInformation': {
            'class': SEC_SUSP.LackOfCurrentInformation,
            'description': 'Suspension due to lack of current and accurate information',
        },

        'MarketManipulation': {
            'class': SEC_SUSP.MarketManipulation,
            'description': 'Suspected market manipulation activities',
        },

        'PumpAndDumpScheme': {
            'class': SEC_SUSP.PumpAndDumpScheme,
            'description': 'Suspected pump and dump promotional activities',
        },

        # Market activity indicators
        'UnusualPriceIncrease': {
            'class': SEC_SUSP.UnusualPriceIncrease,
            'amount_property': SEC_SUSP.priceIncreasePercentage,
            'description': 'Unusual increase in stock price',
        },

        'UnusualTradingVolume': {
            'class': SEC_SUSP.UnusualTradingVolume,
            'amount_property': SEC_SUSP.tradingVolume,
            'description': 'Unusual trading volume',
        },

        # Securities (have last trade dates)
        'Security': {
            'class': SEC_SUSP.Security,
            'date_property': SEC_SUSP.lastTradeDate,
            'grouping_property': SEC_SUSP.issuedBy,
            'description': 'Security subject to trading suspension',
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
        'administrative_proceedings': SEC_ADMIN.cikNumber,
        'trading_suspensions': SEC_SUSP.cikNumber,
        # Litigation uses name matching (no CIK in ontology)
    },
    'company_ticker': {
        'filings': SEC_FILINGS.hasIssuerTradingSymbol,
        'administrative_proceedings': SEC_ADMIN.tickerSymbol,
        'trading_suspensions': SEC_SUSP.tickerSymbol,
        # Litigation uses name matching
    },
    'company_name': {
        'filings': SEC_FILINGS.hasIssuerName,
        'administrative_proceedings': SEC_ADMIN.respondentName,
        'litigation': SEC_LIT.companyName,
        'trading_suspensions': SEC_SUSP.companyName,
    },
    'person_cik': {
        'filings': SEC_FILINGS.hasReportingOwnerCik,
        'administrative_proceedings': SEC_ADMIN.cikNumber,
        # Litigation and suspensions use name matching for persons
    },
    'person_name': {
        'administrative_proceedings': SEC_ADMIN.respondentName,
        'litigation': SEC_LIT.defendantName,
    },
}