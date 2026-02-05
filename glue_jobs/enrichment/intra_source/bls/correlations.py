"""
BLS Known Correlations - Economic relationships based on domain expertise
"""
from glue_jobs.utils.rdf_utils import BLS_ENRICHMENT

# Specific known correlations based on economic theory
# IMPORTANT: Only include correlations between datasets we've analyzed!
KNOWN_CORRELATIONS = [
    # CPI-PPI Correlations
    {
        'name': 'cpi_ppi_food_chain',
        'source_dataset': 'ppi',
        'target_dataset': 'cpi',
        'source_pattern': 'Food Manufacturing',
        'target_pattern': 'Food',
        'description': 'Producer food costs lead to consumer food prices',
        'relationship': BLS_ENRICHMENT.producerConsumerLink,
        'lag_months': 1,
        'strength': 'strong'
    },
    {
        'name': 'cpi_ppi_energy_chain',
        'source_dataset': 'ppi',
        'target_dataset': 'cpi',
        'source_pattern': 'Energy Goods',
        'target_pattern': 'Energy',
        'description': 'Producer energy costs lead to consumer energy prices',
        'relationship': BLS_ENRICHMENT.producerConsumerLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'ppi_supply_chain',
        'source_dataset': 'ppi',
        'target_dataset': 'ppi',
        'source_pattern': 'Intermediate Demand',
        'target_pattern': 'Final Demand',
        'description': 'Intermediate goods costs flow to final demand prices',
        'relationship': BLS_ENRICHMENT.supplyChainLink,
        'direction': 'upstream',
        'strength': 'medium'
    },
    {
        'name': 'ppi_processed_unprocessed',
        'source_dataset': 'ppi',
        'target_dataset': 'ppi',
        'source_pattern': 'Unprocessed',
        'target_pattern': 'Processed',
        'description': 'Unprocessed goods costs affect processed goods prices',
        'relationship': BLS_ENRICHMENT.processingChainLink,
        'direction': 'upstream',
        'strength': 'medium'
    },

    # ECI-CPI Correlations
    {
        'name': 'eci_cpi_wage_inflation',
        'source_dataset': 'eci',
        'target_dataset': 'cpi',
        'source_pattern': 'Wages and salaries',
        'target_pattern': 'All items',
        'description': 'Wage growth leads to consumer price inflation',
        'relationship': BLS_ENRICHMENT.wageInflationLink,
        'lag_months': 2,
        'strength': 'strong'
    },
    {
        'name': 'eci_cpi_compensation_inflation',
        'source_dataset': 'eci',
        'target_dataset': 'cpi',
        'source_pattern': 'Total compensation',
        'target_pattern': 'All items',
        'description': 'Total compensation growth affects consumer prices',
        'relationship': BLS_ENRICHMENT.compensationInflationLink,
        'lag_months': 3,
        'strength': 'strong'
    },
    {
        'name': 'eci_cpi_healthcare_costs',
        'source_dataset': 'eci',
        'target_dataset': 'cpi',
        'source_pattern': 'Health care',
        'target_pattern': 'Medical',
        'description': 'Healthcare employment costs affect medical service prices',
        'relationship': BLS_ENRICHMENT.employmentPriceLink,
        'lag_months': 3,
        'strength': 'strong'
    },
    {
        'name': 'eci_cpi_education_costs',
        'source_dataset': 'eci',
        'target_dataset': 'cpi',
        'source_pattern': 'Educational services',
        'target_pattern': 'Education',
        'description': 'Education employment costs affect education prices',
        'relationship': BLS_ENRICHMENT.employmentPriceLink,
        'lag_months': 6,
        'strength': 'medium'
    },

    # ECI-PPI Correlations
    {
        'name': 'eci_ppi_manufacturing_costs',
        'source_dataset': 'eci',
        'target_dataset': 'ppi',
        'source_pattern': 'Manufacturing',
        'target_pattern': 'Manufacturing',
        'description': 'Manufacturing employment costs affect production costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 1,
        'strength': 'strong'
    },
    {
        'name': 'eci_ppi_construction_costs',
        'source_dataset': 'eci',
        'target_dataset': 'ppi',
        'source_pattern': 'Construction',
        'target_pattern': 'Construction',
        'description': 'Construction employment costs affect building costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 2,
        'strength': 'medium'
    },
    {
        'name': 'eci_ppi_wage_costs',
        'source_dataset': 'eci',
        'target_dataset': 'ppi',
        'source_pattern': 'Wages and salaries',
        'target_pattern': 'Final Demand',
        'description': 'Wage growth affects producer costs',
        'relationship': BLS_ENRICHMENT.wageCostLink,
        'lag_months': 1,
        'strength': 'strong'
    },

    # ECI-JOLTS Correlations
    {
        'name': 'eci_jolts_wage_job_openings',
        'source_dataset': 'eci',
        'target_dataset': 'jolts',
        'source_pattern': 'Wages and salaries',
        'target_pattern': 'Job openings',
        'description': 'Wage growth correlates with job openings (tight labor market)',
        'relationship': BLS_ENRICHMENT.wageJobOpeningsLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'eci_jolts_benefits_quits',
        'source_dataset': 'eci',
        'target_dataset': 'jolts',
        'source_pattern': 'Benefits',
        'target_pattern': 'Quits',
        'description': 'Benefits growth correlates with voluntary quits',
        'relationship': BLS_ENRICHMENT.benefitsQuitsLink,
        'lag_months': 1,
        'strength': 'medium'
    },
    {
        'name': 'eci_jolts_manufacturing_employment',
        'source_dataset': 'eci',
        'target_dataset': 'jolts',
        'source_pattern': 'Manufacturing',
        'target_pattern': 'Manufacturing',
        'description': 'Manufacturing employment costs correlate with job market dynamics',
        'relationship': BLS_ENRICHMENT.employmentJobMarketLink,
        'lag_months': 1,
        'strength': 'strong'
    },
    {
        'name': 'eci_jolts_healthcare_employment',
        'source_dataset': 'eci',
        'target_dataset': 'jolts',
        'source_pattern': 'Health care',
        'target_pattern': 'Health care',
        'description': 'Healthcare employment costs correlate with job openings',
        'relationship': BLS_ENRICHMENT.employmentJobOpeningsLink,
        'lag_months': 2,
        'strength': 'strong'
    },

    # ECI-EMPSIT Correlations
    {
        'name': 'eci_empsit_wage_earnings',
        'source_dataset': 'eci',
        'target_dataset': 'empsit',
        'source_pattern': 'Wages and salaries',
        'target_pattern': 'Average hourly earnings',
        'description': 'ECI wage index correlates with average hourly earnings',
        'relationship': BLS_ENRICHMENT.wageEarningsLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'eci_empsit_compensation_earnings',
        'source_dataset': 'eci',
        'target_dataset': 'empsit',
        'source_pattern': 'Total compensation',
        'target_pattern': 'Average weekly earnings',
        'description': 'Total compensation correlates with weekly earnings',
        'relationship': BLS_ENRICHMENT.compensationEarningsLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'eci_empsit_manufacturing_wages',
        'source_dataset': 'eci',
        'target_dataset': 'empsit',
        'source_pattern': 'Manufacturing',
        'target_pattern': 'Manufacturing',
        'description': 'Manufacturing employment costs correlate with manufacturing earnings',
        'relationship': BLS_ENRICHMENT.employmentEarningsLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'eci_empsit_government_wages',
        'source_dataset': 'eci',
        'target_dataset': 'empsit',
        'source_pattern': 'State and local government',
        'target_pattern': 'State government',
        'description': 'State/local government employment costs correlate with government employment',
        'relationship': BLS_ENRICHMENT.governmentEmploymentLink,
        'lag_months': 0,
        'strength': 'strong'
    },

    # ECI Internal Correlations
    {
        'name': 'eci_wages_benefits',
        'source_dataset': 'eci',
        'target_dataset': 'eci',
        'source_pattern': 'Wages and salaries',
        'target_pattern': 'Benefits',
        'description': 'Wages and benefits move together as total compensation',
        'relationship': BLS_ENRICHMENT.compensationComponentLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'eci_union_nonunion',
        'source_dataset': 'eci',
        'target_dataset': 'eci',
        'source_pattern': 'Union',
        'target_pattern': 'Nonunion',
        'description': 'Union and nonunion wage patterns influence each other',
        'relationship': BLS_ENRICHMENT.bargainingStatusLink,
        'lag_months': 1,
        'strength': 'medium'
    },
    {
        'name': 'eci_private_government',
        'source_dataset': 'eci',
        'target_dataset': 'eci',
        'source_pattern': 'Private industry',
        'target_pattern': 'State and local government',
        'description': 'Private and government sector wages influence each other',
        'relationship': BLS_ENRICHMENT.sectorWageLink,
        'lag_months': 2,
        'strength': 'medium'
    },
    {
        'name': 'eci_regional_patterns',
        'source_dataset': 'eci',
        'target_dataset': 'eci',
        'source_pattern': 'Northeast',
        'target_pattern': 'South',
        'description': 'Regional wage patterns show geographic wage competition',
        'relationship': BLS_ENRICHMENT.regionalWageLink,
        'lag_months': 1,
        'strength': 'weak'
    },

    # JOLTS-CPI Correlations
    {
        'name': 'jolts_cpi_food_services',
        'source_dataset': 'jolts',
        'target_dataset': 'cpi',
        'source_pattern': 'Food services',
        'target_pattern': 'Food away from home',
        'description': 'Job openings in food services affect restaurant prices',
        'relationship': BLS_ENRICHMENT.employmentPriceLink,
        'lag_months': 2,
        'strength': 'medium'
    },
    {
        'name': 'jolts_cpi_retail',
        'source_dataset': 'jolts',
        'target_dataset': 'cpi',
        'source_pattern': 'Retail trade',
        'target_pattern': 'All items',
        'description': 'Retail employment affects consumer prices',
        'relationship': BLS_ENRICHMENT.employmentPriceLink,
        'lag_months': 1,
        'strength': 'medium'
    },
    {
        'name': 'jolts_cpi_healthcare',
        'source_dataset': 'jolts',
        'target_dataset': 'cpi',
        'source_pattern': 'Health care',
        'target_pattern': 'Medical',
        'description': 'Healthcare employment affects medical service prices',
        'relationship': BLS_ENRICHMENT.employmentPriceLink,
        'lag_months': 3,
        'strength': 'strong'
    },

    # JOLTS-PPI Correlations
    {
        'name': 'jolts_ppi_manufacturing',
        'source_dataset': 'jolts',
        'target_dataset': 'ppi',
        'source_pattern': 'Manufacturing',
        'target_pattern': 'Manufacturing',
        'description': 'Manufacturing job openings correlate with production costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 1,
        'strength': 'strong'
    },
    {
        'name': 'jolts_ppi_construction',
        'source_dataset': 'jolts',
        'target_dataset': 'ppi',
        'source_pattern': 'Construction',
        'target_pattern': 'Construction',
        'description': 'Construction employment affects building costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 2,
        'strength': 'medium'
    },
    {
        'name': 'jolts_ppi_transportation',
        'source_dataset': 'jolts',
        'target_dataset': 'ppi',
        'source_pattern': 'Transportation',
        'target_pattern': 'Transportation',
        'description': 'Transportation employment affects shipping costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 1,
        'strength': 'medium'
    },

    # JOLTS Internal Correlations
    {
        'name': 'jolts_openings_hires',
        'source_dataset': 'jolts',
        'target_dataset': 'jolts',
        'source_pattern': 'JobOpenings',
        'target_pattern': 'Hires',
        'description': 'Job openings lead to hires',
        'relationship': BLS_ENRICHMENT.laborMarketFlow,
        'lag_months': 1,
        'strength': 'strong'
    },
    {
        'name': 'jolts_quits_openings',
        'source_dataset': 'jolts',
        'target_dataset': 'jolts',
        'source_pattern': 'Quits',
        'target_pattern': 'JobOpenings',
        'description': 'Quits create job openings',
        'relationship': BLS_ENRICHMENT.laborMarketFlow,
        'lag_months': 0,
        'strength': 'strong'
    },

    # EMPSIT-CPI Correlations
    {
        'name': 'empsit_cpi_employment_spending',
        'source_dataset': 'empsit',
        'target_dataset': 'cpi',
        'source_pattern': 'Employed',
        'target_pattern': 'All items',
        'description': 'Employment levels affect consumer spending and prices',
        'relationship': BLS_ENRICHMENT.employmentConsumptionLink,
        'lag_months': 1,
        'strength': 'medium'
    },
    {
        'name': 'empsit_cpi_earnings_spending',
        'source_dataset': 'empsit',
        'target_dataset': 'cpi',
        'source_pattern': 'Average hourly earnings',
        'target_pattern': 'All items',
        'description': 'Wage growth affects consumer price inflation',
        'relationship': BLS_ENRICHMENT.wageInflationLink,
        'lag_months': 2,
        'strength': 'strong'
    },
    {
        'name': 'empsit_cpi_healthcare_employment',
        'source_dataset': 'empsit',
        'target_dataset': 'cpi',
        'source_pattern': 'Health care',
        'target_pattern': 'Medical',
        'description': 'Healthcare employment affects medical service prices',
        'relationship': BLS_ENRICHMENT.employmentPriceLink,
        'lag_months': 3,
        'strength': 'strong'
    },

    # EMPSIT-JOLTS Correlations
    {
        'name': 'empsit_jolts_unemployment_openings',
        'source_dataset': 'empsit',
        'target_dataset': 'jolts',
        'source_pattern': 'Unemployed',
        'target_pattern': 'Job openings',
        'description': 'Unemployment levels inversely correlate with job openings',
        'relationship': BLS_ENRICHMENT.unemploymentJobOpeningsLink,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'empsit_jolts_labor_force_participation',
        'source_dataset': 'empsit',
        'target_dataset': 'jolts',
        'source_pattern': 'Civilian labor force',
        'target_pattern': 'Total nonfarm',
        'description': 'Labor force participation affects job market dynamics',
        'relationship': BLS_ENRICHMENT.laborForceJobMarketLink,
        'lag_months': 0,
        'strength': 'medium'
    },
    {
        'name': 'empsit_jolts_manufacturing_employment',
        'source_dataset': 'empsit',
        'target_dataset': 'jolts',
        'source_pattern': 'Manufacturing',
        'target_pattern': 'Manufacturing',
        'description': 'Manufacturing employment correlates with job openings',
        'relationship': BLS_ENRICHMENT.employmentJobOpeningsLink,
        'lag_months': 1,
        'strength': 'strong'
    },

    # EMPSIT-PPI Correlations
    {
        'name': 'empsit_ppi_manufacturing_employment',
        'source_dataset': 'empsit',
        'target_dataset': 'ppi',
        'source_pattern': 'Manufacturing',
        'target_pattern': 'Manufacturing',
        'description': 'Manufacturing employment affects production costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 1,
        'strength': 'medium'
    },
    {
        'name': 'empsit_ppi_construction_employment',
        'source_dataset': 'empsit',
        'target_dataset': 'ppi',
        'source_pattern': 'Construction',
        'target_pattern': 'Construction',
        'description': 'Construction employment affects building costs',
        'relationship': BLS_ENRICHMENT.employmentProductionLink,
        'lag_months': 2,
        'strength': 'medium'
    },
    {
        'name': 'empsit_ppi_wages_costs',
        'source_dataset': 'empsit',
        'target_dataset': 'ppi',
        'source_pattern': 'Average hourly earnings',
        'target_pattern': 'Final Demand',
        'description': 'Wage growth affects producer costs',
        'relationship': BLS_ENRICHMENT.wageCostLink,
        'lag_months': 1,
        'strength': 'strong'
    },

    # EMPSIT Internal Correlations
    {
        'name': 'empsit_establishment_household',
        'source_dataset': 'empsit',
        'target_dataset': 'empsit',
        'source_pattern': 'Total nonfarm',
        'target_pattern': 'Employed',
        'description': 'Establishment and household employment measures',
        'relationship': BLS_ENRICHMENT.surveyCorrelation,
        'lag_months': 0,
        'strength': 'strong'
    },
    {
        'name': 'empsit_unemployment_duration',
        'source_dataset': 'empsit',
        'target_dataset': 'empsit',
        'source_pattern': 'Unemployed',
        'target_pattern': 'Average duration',
        'description': 'Unemployment levels affect average duration',
        'relationship': BLS_ENRICHMENT.unemploymentDurationLink,
        'lag_months': 0,
        'strength': 'medium'
    },
    {
        'name': 'empsit_hours_earnings',
        'source_dataset': 'empsit',
        'target_dataset': 'empsit',
        'source_pattern': 'Average weekly hours',
        'target_pattern': 'Average weekly earnings',
        'description': 'Hours worked directly affect weekly earnings',
        'relationship': BLS_ENRICHMENT.hoursEarningsLink,
        'lag_months': 0,
        'strength': 'strong'
    }
]