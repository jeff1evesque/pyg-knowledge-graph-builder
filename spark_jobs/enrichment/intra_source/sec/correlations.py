"""
SEC Known Correlations - Relationships between SEC datasets

Defines known relationships between:
- Filings ↔ Administrative Proceedings (company files, then gets enforcement)
- Administrative Proceedings ↔ Litigation (admin action accompanies civil suit)
- Administrative Proceedings ↔ Trading Suspensions (enforcement leads to suspension)
- Litigation ↔ Trading Suspensions (civil action accompanies suspension)
- Filings (ownership) ↔ Litigation (insider trading detection)

Property references match actual ontologies:
- Filings:    http://www.sec.gov/filings# (filings:hasIssuerCik, filings:hasIssuerTradingSymbol, etc.)
- Admin:      https://www.sec.gov/ontology/administrative-proceedings# (sec:cikNumber, sec:tickerSymbol, etc.)
- Litigation: https://www.sec.gov/ontology/litigation# (seclit:filingDate, seclit:caseNumber, etc.)
- Suspensions: https://www.sec.gov/ontology/trading-suspensions# (secsusp:cikNumber, secsusp:tickerSymbol, etc.)
"""
from spark_jobs.utils.rdf_utils import SEC_ENRICHMENT

KNOWN_CORRELATIONS = [

    # ============================================
    # FILINGS ↔ ADMINISTRATIVE PROCEEDINGS
    # ============================================
    # Companies that file ownership forms may later face admin proceedings

    {
        'name': 'filing_issuer_to_admin_respondent_by_cik',
        'source_dataset': 'filings',
        'target_dataset': 'administrative_proceedings',
        'source_pattern': 'Issuer',
        'target_pattern': 'Respondent',
        'match_strategy': 'cik',
        'source_cik_property': 'filings:hasIssuerCik',
        'target_cik_property': 'sec:cikNumber',
        'description': 'Issuer in ownership filing is respondent in admin proceeding (matched by CIK)',
        'relationship': SEC_ENRICHMENT.filingIssuerAdminRespondentLink,
        'strength': 'strong'
    },

    {
        'name': 'filing_issuer_to_admin_respondent_by_ticker',
        'source_dataset': 'filings',
        'target_dataset': 'administrative_proceedings',
        'source_pattern': 'Issuer',
        'target_pattern': 'EntityRespondent',
        'match_strategy': 'ticker',
        'source_ticker_property': 'filings:hasIssuerTradingSymbol',
        'target_ticker_property': 'sec:tickerSymbol',
        'description': 'Issuer in filing matches entity respondent in admin proceeding (by ticker)',
        'relationship': SEC_ENRICHMENT.filingIssuerAdminRespondentTickerLink,
        'strength': 'medium'
    },

    {
        'name': 'reporting_owner_to_admin_individual_respondent',
        'source_dataset': 'filings',
        'target_dataset': 'administrative_proceedings',
        'source_pattern': 'ReportingOwner',
        'target_pattern': 'IndividualRespondent',
        'match_strategy': 'cik',
        'source_cik_property': 'filings:hasReportingOwnerCik',
        'target_cik_property': 'sec:cikNumber',
        'description': 'Reporting owner in ownership filing becomes individual respondent in admin proceeding',
        'relationship': SEC_ENRICHMENT.ownerToAdminRespondentLink,
        'strength': 'strong'
    },

    # ============================================
    # FILINGS ↔ LITIGATION
    # ============================================
    # Ownership filings may precede insider trading litigation

    {
        'name': 'ownership_filing_to_insider_trading_litigation',
        'source_dataset': 'filings',
        'target_dataset': 'litigation',
        'source_pattern': 'Form4',
        'target_pattern': 'InsiderTradingClaim',
        'match_strategy': 'cik',
        'source_cik_property': 'filings:hasReportingOwnerCik',
        'target_cik_property': 'seclit:defendantName',
        'description': 'Form 4 ownership change filing precedes insider trading litigation',
        'relationship': SEC_ENRICHMENT.ownershipFilingInsiderTradingLink,
        'strength': 'medium'
    },

    {
        'name': 'filing_issuer_to_litigation_defendant_by_name',
        'source_dataset': 'filings',
        'target_dataset': 'litigation',
        'source_pattern': 'Issuer',
        'target_pattern': 'CorporateDefendant',
        'match_strategy': 'name',
        'source_name_property': 'filings:hasIssuerName',
        'target_name_property': 'seclit:companyName',
        'description': 'Issuer in filing is corporate defendant in litigation (matched by company name)',
        'relationship': SEC_ENRICHMENT.filingIssuerLitigationDefendantLink,
        'strength': 'medium'
    },

    # ============================================
    # FILINGS ↔ TRADING SUSPENSIONS
    # ============================================
    # Companies with filings may have trading suspended

    {
        'name': 'filing_issuer_to_suspension_company_by_ticker',
        'source_dataset': 'filings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'Issuer',
        'target_pattern': 'Company',
        'match_strategy': 'ticker',
        'source_ticker_property': 'filings:hasIssuerTradingSymbol',
        'target_ticker_property': 'secsusp:tickerSymbol',
        'description': 'Issuer in filing has trading suspended (matched by ticker)',
        'relationship': SEC_ENRICHMENT.filingIssuerSuspensionLink,
        'strength': 'strong'
    },

    {
        'name': 'filing_issuer_to_suspension_company_by_cik',
        'source_dataset': 'filings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'Issuer',
        'target_pattern': 'Company',
        'match_strategy': 'cik',
        'source_cik_property': 'filings:hasIssuerCik',
        'target_cik_property': 'secsusp:cikNumber',
        'description': 'Issuer in filing has trading suspended (matched by CIK)',
        'relationship': SEC_ENRICHMENT.filingIssuerSuspensionCikLink,
        'strength': 'strong'
    },

    # ============================================
    # ADMINISTRATIVE PROCEEDINGS ↔ LITIGATION
    # ============================================
    # Admin proceedings often accompany or follow civil litigation

    {
        'name': 'admin_respondent_to_litigation_defendant_by_cik',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'litigation',
        'source_pattern': 'EntityRespondent',
        'target_pattern': 'CorporateDefendant',
        'match_strategy': 'cik',
        'source_cik_property': 'sec:cikNumber',
        'target_cik_property': 'seclit:companyName',
        'description': 'Entity respondent in admin proceeding is also defendant in civil litigation',
        'relationship': SEC_ENRICHMENT.adminRespondentLitigationDefendantLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_individual_to_litigation_individual',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'litigation',
        'source_pattern': 'IndividualRespondent',
        'target_pattern': 'IndividualDefendant',
        'match_strategy': 'name',
        'source_name_property': 'sec:respondentName',
        'target_name_property': 'seclit:defendantName',
        'description': 'Individual respondent in admin proceeding is also defendant in litigation',
        'relationship': SEC_ENRICHMENT.adminIndividualLitigationIndividualLink,
        'strength': 'medium'
    },

    {
        'name': 'admin_fraud_to_litigation_fraud',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'litigation',
        'source_pattern': 'SecuritiesFraud',
        'target_pattern': 'SecuritiesFraudClaim',
        'match_strategy': 'violation_type',
        'description': 'Securities fraud violation in admin proceeding correlates with fraud claim in litigation',
        'relationship': SEC_ENRICHMENT.adminFraudLitigationFraudLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_reporting_to_litigation_reporting',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'litigation',
        'source_pattern': 'ReportingViolation',
        'target_pattern': 'ReportingViolationClaim',
        'match_strategy': 'violation_type',
        'description': 'Reporting violation in admin proceeding correlates with reporting claim in litigation',
        'relationship': SEC_ENRICHMENT.adminReportingLitigationReportingLink,
        'strength': 'strong'
    },

    # ============================================
    # ADMINISTRATIVE PROCEEDINGS ↔ TRADING SUSPENSIONS
    # ============================================
    # Admin proceedings (especially Section 12(j)) lead to trading suspensions

    {
        'name': 'admin_section12j_to_trading_suspension',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'Section12jProceeding',
        'target_pattern': 'Section12jSuspension',
        'match_strategy': 'section_type',
        'description': 'Section 12(j) admin proceeding leads to Section 12(j) trading suspension',
        'relationship': SEC_ENRICHMENT.adminSection12jSuspensionLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_respondent_to_suspended_company_by_ticker',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'EntityRespondent',
        'target_pattern': 'Company',
        'match_strategy': 'ticker',
        'source_ticker_property': 'sec:tickerSymbol',
        'target_ticker_property': 'secsusp:tickerSymbol',
        'description': 'Entity respondent in admin proceeding has trading suspended (by ticker)',
        'relationship': SEC_ENRICHMENT.adminRespondentSuspensionTickerLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_respondent_to_suspended_company_by_cik',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'EntityRespondent',
        'target_pattern': 'Company',
        'match_strategy': 'cik',
        'source_cik_property': 'sec:cikNumber',
        'target_cik_property': 'secsusp:cikNumber',
        'description': 'Entity respondent in admin proceeding has trading suspended (by CIK)',
        'relationship': SEC_ENRICHMENT.adminRespondentSuspensionCikLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_reporting_violation_to_lack_of_info_suspension',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'PeriodicReportingViolation',
        'target_pattern': 'LackOfCurrentInformation',
        'match_strategy': 'violation_reason',
        'description': 'Periodic reporting violation leads to suspension for lack of current information',
        'relationship': SEC_ENRICHMENT.reportingViolationLackOfInfoLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_registration_revocation_to_suspension',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'RegistrationRevocation',
        'target_pattern': 'TradingSuspension',
        'match_strategy': 'sanction_consequence',
        'description': 'Registration revocation sanction leads to trading suspension',
        'relationship': SEC_ENRICHMENT.registrationRevocationSuspensionLink,
        'strength': 'strong'
    },

    # ============================================
    # LITIGATION ↔ TRADING SUSPENSIONS
    # ============================================
    # Civil enforcement actions may accompany trading suspensions

    {
        'name': 'litigation_defendant_to_suspended_company_by_name',
        'source_dataset': 'litigation',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'CorporateDefendant',
        'target_pattern': 'Company',
        'match_strategy': 'name',
        'source_name_property': 'seclit:companyName',
        'target_name_property': 'secsusp:companyName',
        'description': 'Corporate defendant in litigation has trading suspended',
        'relationship': SEC_ENRICHMENT.litigationDefendantSuspensionLink,
        'strength': 'medium'
    },

    {
        'name': 'litigation_emergency_to_emergency_suspension',
        'source_dataset': 'litigation',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'EmergencyAction',
        'target_pattern': 'EmergencySuspension',
        'match_strategy': 'action_type',
        'description': 'Emergency enforcement action accompanies emergency trading suspension',
        'relationship': SEC_ENRICHMENT.emergencyActionSuspensionLink,
        'strength': 'strong'
    },

    {
        'name': 'litigation_market_manipulation_to_suspension_manipulation',
        'source_dataset': 'litigation',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'MarketManipulationClaim',
        'target_pattern': 'MarketManipulation',
        'match_strategy': 'violation_reason',
        'description': 'Market manipulation claim in litigation correlates with manipulation suspension reason',
        'relationship': SEC_ENRICHMENT.litigationManipulationSuspensionLink,
        'strength': 'strong'
    },

    # ============================================
    # INTERNAL CORRELATIONS - WITHIN ADMIN PROCEEDINGS
    # ============================================

    {
        'name': 'admin_violation_to_sanction',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'administrative_proceedings',
        'source_pattern': 'Violation',
        'target_pattern': 'Sanction',
        'match_strategy': 'same_proceeding',
        'description': 'Violation leads to sanction within same admin proceeding',
        'relationship': SEC_ENRICHMENT.violationToSanctionLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_cease_desist_to_cease_desist_order',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'administrative_proceedings',
        'source_pattern': 'CeaseAndDesistProceeding',
        'target_pattern': 'CeaseAndDesistOrder',
        'match_strategy': 'same_proceeding',
        'description': 'Cease and desist proceeding results in cease and desist order',
        'relationship': SEC_ENRICHMENT.ceaseDesistProceedingOrderLink,
        'strength': 'strong'
    },

    {
        'name': 'admin_rule102e_to_professional_suspension',
        'source_dataset': 'administrative_proceedings',
        'target_dataset': 'administrative_proceedings',
        'source_pattern': 'Rule102eProceeding',
        'target_pattern': 'Suspension',
        'match_strategy': 'same_proceeding',
        'description': 'Rule 102(e) proceeding results in professional suspension (accountant/attorney)',
        'relationship': SEC_ENRICHMENT.rule102eSuspensionLink,
        'strength': 'strong'
    },

    # ============================================
    # INTERNAL CORRELATIONS - WITHIN LITIGATION
    # ============================================

    {
        'name': 'litigation_claim_to_relief',
        'source_dataset': 'litigation',
        'target_dataset': 'litigation',
        'source_pattern': 'LegalClaim',
        'target_pattern': 'Relief',
        'match_strategy': 'same_action',
        'description': 'Legal claim in litigation leads to relief sought/obtained',
        'relationship': SEC_ENRICHMENT.claimToReliefLink,
        'strength': 'strong'
    },

    {
        'name': 'litigation_action_to_settlement',
        'source_dataset': 'litigation',
        'target_dataset': 'litigation',
        'source_pattern': 'CivilEnforcementAction',
        'target_pattern': 'ConsentJudgment',
        'match_strategy': 'same_action',
        'description': 'Civil enforcement action resolved by consent judgment',
        'relationship': SEC_ENRICHMENT.actionToSettlementLink,
        'strength': 'strong'
    },

    # ============================================
    # INTERNAL CORRELATIONS - WITHIN TRADING SUSPENSIONS
    # ============================================

    {
        'name': 'suspension_reason_to_related_action',
        'source_dataset': 'trading_suspensions',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'TradingSuspension',
        'target_pattern': 'EnforcementAction',
        'match_strategy': 'same_suspension',
        'description': 'Trading suspension leads to enforcement action',
        'relationship': SEC_ENRICHMENT.suspensionToEnforcementLink,
        'strength': 'strong'
    },

    {
        'name': 'pump_dump_to_market_manipulation_suspension',
        'source_dataset': 'trading_suspensions',
        'target_dataset': 'trading_suspensions',
        'source_pattern': 'PumpAndDumpScheme',
        'target_pattern': 'MarketManipulation',
        'match_strategy': 'reason_hierarchy',
        'description': 'Pump and dump scheme is a specific type of market manipulation',
        'relationship': SEC_ENRICHMENT.pumpDumpManipulationLink,
        'strength': 'strong'
    },
]