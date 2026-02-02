"""
BLS Intra-Source Enrichment
Links entities within the BLS family (CPI, PPI, ECI, JOLTS, EMPSIT, etc.)

This module encodes domain knowledge about BLS data relationships.
Patterns are only defined for datasets whose ontologies have been analyzed.

Currently supported datasets: CPI, PPI, ECI, JOLTS, EMPSIT (27 tables)
TODO: Add LAUS, METRO, REALER, WKYENG, XIMPIM as ontologies are provided
"""

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
from glue_jobs.utils.rdf_utils import (
    CPI, PPI, JOLTS, EMPSIT, ECI, LAUS, METRO, REALER, WKYENG, XIMPIM,
    BLS_ENRICHMENT, UNIFIED,
    get_month_name, get_year_value
)
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================
# DOMAIN KNOWLEDGE: BLS LINKING PATTERNS
# ============================================
# IMPORTANT: Only include datasets whose ontologies have been analyzed!
# Currently includes: CPI, PPI, ECI, JOLTS, EMPSIT
# TODO: Add LAUS, METRO, REALER, WKYENG, XIMPIM as their ontologies are provided

BLS_SECTOR_PATTERNS = {
    'food_sector': {
        'description': 'Food production, distribution, and consumption chain',
        'sector_uri': BLS_ENRICHMENT.FoodSector,
        'keywords': {
            'cpi': ['Food', 'Grocery', 'Restaurant', 'Dining', 'Meals', 'Beverages'],
            'ppi': ['Food', 'Manufacturing', 'Foodstuffs', 'Feeds', 'Finished Consumer Foods'],
            'eci': ['Food services', 'Accommodation and food', 'Food service'],
            'jolts': ['Food services', 'Accommodation and food services', 'Restaurants',
                      'Drinking places', 'Food service'],
            'empsit': [
                # From Table B-1
                'Food manufacturing',
                'Food and beverage retailers',
                'Food services and drinking places',
                'Accommodation and food services',
                # Beverage
                'Beverage, tobacco, and leather',
            ],
        },
        'relationship': BLS_ENRICHMENT.foodSectorCorrelation
    },

    'energy_sector': {
        'description': 'Energy production, distribution, and consumption',
        'sector_uri': BLS_ENRICHMENT.EnergySector,
        'keywords': {
            'cpi': ['Energy', 'Gasoline', 'Fuel', 'Electricity', 'Gas', 'Utility'],
            'ppi': ['Energy', 'Goods', 'Materials', 'Fuel', 'Petroleum'],
            'eci': ['Utilities', 'Mining', 'Oil and gas'],
            'jolts': ['Mining', 'Oil and gas extraction', 'Utilities', 'Electric power',
                      'Natural gas', 'Coal mining'],
            'empsit': [
                # From Table B-1
                'Mining and logging',
                'Oil and gas extraction',
                'Coal mining',
                'Mining, quarrying, and oil and gas extraction',
                'Mining (except oil and gas)',
                'Metal ore mining',
                'Nonmetallic mineral mining',
                'Support activities for mining',
                'Utilities',
                'Petroleum and coal products manufacturing',
                'Gasoline stations and fuel dealers',
                'Pipeline transportation',
            ],
        },
        'relationship': BLS_ENRICHMENT.energySectorCorrelation
    },

    'housing_sector': {
        'description': 'Housing, construction, and shelter',
        'sector_uri': BLS_ENRICHMENT.HousingSector,
        'keywords': {
            'cpi': ['Housing', 'Shelter', 'Rent', 'Owners', 'Lodging'],
            'ppi': ['Construction', 'Building', 'Materials', 'Residential'],
            'eci': ['Construction', 'Real estate', 'Rental', 'Leasing'],
            'jolts': ['Construction', 'Building construction', 'Heavy construction',
                      'Specialty trade contractors', 'Residential construction'],
            'empsit': [
                # From Table B-1
                'Construction',
                'Construction of buildings',
                'Residential building construction',
                'Nonresidential building construction',
                'Heavy and civil engineering construction',
                'Specialty trade contractors',
                'Residential specialty trade contractors',
                'Nonresidential specialty trade contractors',
                'Building material and garden equipment and supplies dealers',
                'Real estate',
                'Real estate and rental and leasing',
                'Lessors of nonfinancial intangible assets',
            ],
        },
        'relationship': BLS_ENRICHMENT.housingSectorCorrelation
    },

    'transportation_sector': {
        'description': 'Transportation and warehousing',
        'sector_uri': BLS_ENRICHMENT.TransportationSector,
        'keywords': {
            'cpi': ['Transportation', 'Vehicle', 'Automobile', 'Transit', 'Airfare'],
            'ppi': ['Transportation', 'Warehousing', 'Trade', 'Shipping'],
            'eci': ['Transportation', 'Warehousing', 'Trade, transportation'],
            'jolts': ['Transportation', 'Warehousing', 'Truck transportation',
                      'Couriers and messengers', 'Transit', 'Air transportation'],
            'empsit': [
                # From Table B-1
                'Transportation and warehousing',
                'Air transportation',
                'Rail transportation',
                'Water transportation',
                'Truck transportation',
                'Transit and ground passenger transportation',
                'Scenic and sightseeing transportation',
                'Support activities for transportation',
                'Couriers and messengers',
                'Warehousing and storage',
                'Transportation equipment manufacturing',
                'Motor vehicles and parts',
                # From Table A-14 (combined category)
                'Transportation and utilities',
            ],
        },
        'relationship': BLS_ENRICHMENT.transportationSectorCorrelation
    },

    'goods_services': {
        'description': 'Goods vs Services distinction across datasets',
        'sector_uri': BLS_ENRICHMENT.GoodsServicesSector,
        'keywords': {
            'cpi': ['Goods', 'Services', 'Commodities'],
            'ppi': ['Goods', 'Services', 'Final Demand', 'Intermediate Demand',
                    'Processed', 'Unprocessed'],
            'eci': ['Goods-producing', 'Service-providing'],
            'jolts': ['Goods-producing', 'Service-providing', 'Manufacturing',
                      'Trade, transportation, and utilities', 'Professional and business services'],
            'empsit': [
                # From Table B-1
                'Goods-producing',
                'Private service-providing',
                'Durable goods',
                'Nondurable goods',
            ],
        },
        'relationship': BLS_ENRICHMENT.goodsServicesRelation
    },

    'manufacturing_sector': {
        'description': 'Manufacturing and production',
        'sector_uri': BLS_ENRICHMENT.ManufacturingSector,
        'keywords': {
            'cpi': [],  # CPI doesn't have manufacturing categories
            'ppi': ['Manufacturing', 'Production', 'Fabrication', 'Processing'],
            'eci': ['Manufacturing', 'Aircraft manufacturing', 'Production'],
            'jolts': ['Manufacturing', 'Durable goods', 'Nondurable goods',
                      'Fabricated metal', 'Machinery', 'Computer and electronic'],
            'empsit': [
                # From Table B-1
                'Manufacturing',
                'Durable goods',
                'Nondurable goods',
                'Wood product manufacturing',
                'Nonmetallic mineral product manufacturing',
                'Primary metal manufacturing',
                'Fabricated metal product manufacturing',
                'Machinery manufacturing',
                'Computer and electronic product manufacturing',
                'Computer and peripheral equipment manufacturing',
                'Communications equipment manufacturing',
                'Semiconductor and other electronic component manufacturing',
                'Navigational, measuring, electromedical, and control instruments manufacturing',
                'Manufacturing and reproducing magnetic and optical media and audio and video equipment manufacturing',
                'Electrical equipment, appliance, and component manufacturing',
                'Transportation equipment manufacturing',
                'Furniture and related product manufacturing',
                'Miscellaneous manufacturing',
                'Food manufacturing',
                'Textile mills',
                'Textile product mills',
                'Apparel manufacturing',
                'Paper manufacturing',
                'Printing and related support activities',
                'Chemical manufacturing',
                'Plastics and rubber products manufacturing',
                # From Table A-13
                'Production occupations',
                'Production, transportation, and material moving occupations',
            ],
        },
        'relationship': BLS_ENRICHMENT.manufacturingSectorCorrelation
    },

    'retail_sector': {
        'description': 'Retail trade and sales',
        'sector_uri': BLS_ENRICHMENT.RetailSector,
        'keywords': {
            'cpi': [],  # CPI measures prices, not retail activity
            'ppi': ['Trade', 'Wholesale', 'Retail'],
            'eci': ['Retail trade', 'Wholesale trade', 'Trade, transportation'],
            'jolts': ['Retail trade', 'Motor vehicle dealers', 'Furniture stores',
                      'Electronics stores', 'Food and beverage stores', 'General merchandise'],
            'empsit': [
                # From Table B-1
                'Retail trade',
                'Motor vehicle and parts dealers',
                'Automobile dealers',
                'Other motor vehicle dealers',
                'Automotive parts, accessories, and tire retailers',
                'Building material and garden equipment and supplies dealers',
                'Food and beverage retailers',
                'Furniture, home furnishings, electronics, and appliance retailers',
                'Furniture and home furnishings retailers',
                'Electronics and appliance retailers',
                'General merchandise retailers',
                'Department stores',
                'Warehouse clubs, supercenters, and other general merchandise retailers',
                'Health and personal care retailers',
                'Gasoline stations and fuel dealers',
                'Clothing, clothing accessories, shoe, and jewelry retailers',
                'Sporting goods, hobby, musical instrument, book, and miscellaneous retailers',
                'Wholesale trade',
                'Merchant wholesalers, durable goods',
                'Merchant wholesalers, nondurable goods',
                'Wholesale trade agents and brokers',
                # From Table A-13
                'Sales and office occupations',
                'Sales and related occupations',
                # From Table A-14 (combined category)
                'Wholesale and retail trade',
            ],
        },
        'relationship': BLS_ENRICHMENT.retailSectorCorrelation
    },

    'healthcare_sector': {
        'description': 'Healthcare and social assistance',
        'sector_uri': BLS_ENRICHMENT.HealthcareSector,
        'keywords': {
            'cpi': ['Medical', 'Healthcare', 'Hospital', 'Physician', 'Prescription'],
            'ppi': ['Healthcare', 'Medical', 'Hospital'],
            'eci': ['Health care', 'Healthcare', 'Hospitals', 'Nursing', 'Social assistance'],
            'jolts': ['Health care', 'Social assistance', 'Hospitals',
                      'Nursing', 'Ambulatory health care', 'Medical'],
            'empsit': [
                # From Table B-1
                'Health care and social assistance',
                'Health care',
                'Ambulatory health care services',
                'Offices of physicians',
                'Offices of dentists',
                'Offices of other health practitioners',
                'Outpatient care centers',
                'Medical and diagnostic laboratories',
                'Home health care services',
                'Other ambulatory health care services',
                'Hospitals',
                'Nursing and residential care facilities',
                'Skilled nursing care facilities',
                'Residential intellectual and developmental disability, mental health, and substance abuse facilities',
                'Continuing care retirement communities and assisted living facilities for the elderly',
                'Other residential care facilities',
                'Social assistance',
                'Individual and family services',
                'Community food and housing, and emergency and other relief services',
                'Vocational rehabilitation services',
                'Child care services',
                'Private education and health services',
                # From Table A-14 (combined category)
                'Education and health services',
            ],
        },
        'relationship': BLS_ENRICHMENT.healthcareSectorCorrelation
    },

    'professional_services_sector': {
        'description': 'Professional and business services',
        'sector_uri': BLS_ENRICHMENT.ProfessionalServicesSector,
        'keywords': {
            'cpi': [],  # CPI doesn't have professional services categories
            'ppi': ['Professional services', 'Business services', 'Consulting'],
            'eci': ['Professional and business services', 'Professional, scientific, and technical',
                    'Administrative and support'],
            'jolts': ['Professional and business services', 'Professional and technical services',
                      'Management', 'Administrative', 'Waste services'],
            'empsit': [
                # From Table B-1
                'Professional and business services',
                'Professional, scientific, and technical services',
                'Legal services',
                'Accounting, tax preparation, bookkeeping, and payroll services',
                'Architectural, engineering, and related services',
                'Specialized design services',
                'Computer systems design and related services',
                'Management, scientific, and technical consulting services',
                'Scientific research and development services',
                'Advertising, public relations, and related services',
                'Other professional, scientific, and technical services',
                'Management of companies and enterprises',
                'Administrative and support and waste management and remediation services',
                'Administrative and support services',
                'Office administrative services',
                'Facilities support services',
                'Employment services',
                'Temporary help services',
                'Business support services',
                'Travel arrangement and reservation services',
                'Investigation and security services',
                'Services to buildings and dwellings',
                'Other support services',
                'Waste management and remediation services',
                # From Table A-13
                'Management, professional, and related occupations',
                'Management, business, and financial operations occupations',
                'Professional and related occupations',
                'Office and administrative support occupations',
            ],
        },
        'relationship': BLS_ENRICHMENT.professionalServicesSectorCorrelation
    },

    'leisure_hospitality_sector': {
        'description': 'Leisure, hospitality, and entertainment',
        'sector_uri': BLS_ENRICHMENT.LeisureHospitalitySector,
        'keywords': {
            'cpi': ['Recreation', 'Entertainment', 'Lodging', 'Hotel'],
            'ppi': [],  # PPI doesn't have leisure/hospitality categories
            'eci': ['Leisure and hospitality', 'Accommodation and food'],
            'jolts': ['Leisure and hospitality', 'Arts and entertainment',
                      'Accommodation', 'Recreation', 'Amusement'],
            'empsit': [
                # From Table B-1
                'Leisure and hospitality',
                'Arts, entertainment, and recreation',
                'Performing arts, spectator sports, and related industries',
                'Museums, historical sites, and similar institutions',
                'Amusement, gambling, and recreation industries',
                'Accommodation and food services',
                'Accommodation',
                'Food services and drinking places',
            ],
        },
        'relationship': BLS_ENRICHMENT.leisureHospitalitySectorCorrelation
    },

    'government_sector': {
        'description': 'Government employment',
        'sector_uri': BLS_ENRICHMENT.GovernmentSector,
        'keywords': {
            'cpi': [],
            'ppi': [],
            'eci': ['State and local government', 'Public administration'],
            'jolts': ['Government', 'Federal', 'State', 'Local'],
            'empsit': [
                # From Table B-1
                'Government',
                'Federal',
                'Federal, except U.S. Postal Service',
                'U.S. Postal Service',
                'State government',
                'State government education',
                'State government, excluding education',
                'Local government',
                'Local government education',
                'Local government, excluding education',
                # From Table A-14
                'Government workers',
            ],
        },
        'relationship': BLS_ENRICHMENT.governmentSectorCorrelation
    },

    'information_sector': {
        'description': 'Information and communications',
        'sector_uri': BLS_ENRICHMENT.InformationSector,
        'keywords': {
            'cpi': ['Information', 'Communication'],
            'ppi': ['Information'],
            'eci': ['Information'],
            'jolts': ['Information', 'Publishing', 'Broadcasting', 'Telecommunications'],
            'empsit': [
                # From Table B-1
                'Information',
                'Motion picture and sound recording industries',
                'Publishing industries',
                'Broadcasting and content providers',
                'Telecommunications',
                'Computing infrastructure providers, data processing, web hosting, and related services',
                'Web search portals, libraries, archives, and other information services',
            ],
        },
        'relationship': BLS_ENRICHMENT.informationSectorCorrelation
    },

    'financial_sector': {
        'description': 'Financial activities and services',
        'sector_uri': BLS_ENRICHMENT.FinancialSector,
        'keywords': {
            'cpi': ['Financial'],
            'ppi': ['Financial'],
            'eci': ['Financial activities', 'Finance and insurance', 'Credit intermediation',
                    'Insurance carriers', 'Real estate'],
            'jolts': ['Financial activities', 'Finance', 'Insurance', 'Real estate'],
            'empsit': [
                # From Table B-1
                'Financial activities',
                'Finance and insurance',
                'Monetary authorities-central bank',
                'Credit intermediation and related  activities',
                'Depository credit intermediation',
                'Commercial banking',
                'Nondepository credit intermediation',
                'Activities related to credit intermediation',
                'Securities, commodity contracts, funds, trusts, and other financial vehicles, investments, and related activities',
                'Insurance carriers and related activities',
                'Real estate and rental and leasing',
                'Real estate',
                'Rental and leasing services',
            ],
        },
        'relationship': BLS_ENRICHMENT.financialSectorCorrelation
    },

    'education_sector': {
        'description': 'Educational services',
        'sector_uri': BLS_ENRICHMENT.EducationSector,
        'keywords': {
            'cpi': ['Education'],
            'ppi': [],
            'eci': ['Educational services', 'Education and health services', 'Elementary and secondary schools',
                    'Junior colleges', 'Colleges', 'Universities', 'Schools'],
            'jolts': ['Educational services'],
            'empsit': [
                # From Table B-1
                'Private educational services',
                'Private education and health services',
                'State government education',
                'Local government education',
                # From Table A-14 (combined category)
                'Education and health services',
            ],
        },
        'relationship': BLS_ENRICHMENT.educationSectorCorrelation
    },

    'other_services_sector': {
        'description': 'Other services (repair, personal, religious)',
        'sector_uri': BLS_ENRICHMENT.OtherServicesSector,
        'keywords': {
            'cpi': [],
            'ppi': [],
            'eci': ['Other services'],
            'jolts': ['Other services', 'Repair', 'Personal services'],
            'empsit': [
                # From Table B-1
                'Other services',
                'Repair and maintenance',
                'Personal and laundry services',
                'Religious, grantmaking, civic, professional, and similar organizations',
                # From Table A-13
                'Service occupations',
                'Installation, maintenance, and repair occupations',
            ],
        },
        'relationship': BLS_ENRICHMENT.otherServicesSectorCorrelation
    },

    'natural_resources_sector': {
        'description': 'Natural resources, farming, fishing, forestry',
        'sector_uri': BLS_ENRICHMENT.NaturalResourcesSector,
        'keywords': {
            'cpi': [],
            'ppi': ['Agriculture', 'Forestry', 'Fishing'],
            'eci': ['Natural resources, construction, and maintenance',
                    'Construction, extraction, farming, fishing, and forestry'],
            'jolts': ['Agriculture', 'Forestry', 'Fishing'],
            'empsit': [
                # From Table B-1
                'Logging',
                # From Table A-13
                'Natural resources, construction, and maintenance occupations',
                'Farming, fishing, and forestry occupations',
                # From Table A-8
                'Agriculture and related industries',
                # From Table A-14
                'Agriculture and related private wage and salary workers',
            ],
        },
        'relationship': BLS_ENRICHMENT.naturalResourcesSectorCorrelation
    },

    'construction_trades_sector': {
        'description': 'Construction and extraction trades',
        'sector_uri': BLS_ENRICHMENT.ConstructionTradesSector,
        'keywords': {
            'cpi': [],
            'ppi': ['Construction'],
            'eci': ['Construction', 'Installation, maintenance, and repair'],
            'jolts': ['Construction'],
            'empsit': [
                # From Table A-13
                'Construction and extraction occupations',
                'Natural resources, construction, and maintenance occupations',
            ],
        },
        'relationship': BLS_ENRICHMENT.constructionTradesSectorCorrelation
    },
}

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

# Measurement type mappings for temporal linking
# IMPORTANT: Only include datasets we've analyzed!
MEASUREMENT_TYPES = {
    'cpi': {
        'Index': {
            'class': CPI.Index,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },
        'PercentChange': {
            'class': CPI.PercentChange,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },
        'RelativeImportance': {
            'class': CPI.RelativeImportance,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        }
    },
    'ppi': {
        'MonthlyChange': {
            'class': PPI.MonthlyChange,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasEndMonth,
            'year_property': PPI.hasEndYear
        },
        'TwelveMonthChange': {
            'class': PPI.TwelveMonthChange,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasEndMonth,
            'year_property': PPI.hasEndYear
        },
        'IndexValue': {
            'class': PPI.IndexValue,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        },
        'RelativeImportance': {
            'class': PPI.RelativeImportance,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        }
    },
    'eci': {
        # Table 1: Total Compensation (seasonally adjusted)
        'EmploymentCostIndexData': {
            'class': ECI.EmploymentCostIndexData,
            'category_property': ECI.hasOccupationalGroup,  # or hasIndustry
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PercentChangeData': {
            'class': ECI.PercentChangeData,
            'category_property': ECI.hasOccupationalGroup,  # or hasIndustry
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 2: Wages and Salaries (seasonally adjusted)
        'WagesAndSalariesIndexData': {
            'class': ECI.WagesAndSalariesIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'WagesAndSalariesPercentChangeData': {
            'class': ECI.WagesAndSalariesPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 3: Benefits (seasonally adjusted)
        'BenefitsIndexData': {
            'class': ECI.BenefitsIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsPercentChangeData': {
            'class': ECI.BenefitsPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 4: Civilian Workers (not seasonally adjusted)
        'CivilianWorkerIndexData': {
            'class': ECI.CivilianWorkerIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'ThreeMonthPercentChangeData': {
            'class': ECI.ThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthPercentChangeData': {
            'class': ECI.TwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 5: Private Industry Workers
        'PrivateIndustryWorkerIndexData': {
            'class': ECI.PrivateIndustryWorkerIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryThreeMonthPercentChangeData': {
            'class': ECI.PrivateIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryTwelveMonthPercentChangeData': {
            'class': ECI.PrivateIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 7: State and Local Government Workers
        'StateLocalGovernmentWorkerIndexData': {
            'class': ECI.StateLocalGovernmentWorkerIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentThreeMonthPercentChangeData': {
            'class': ECI.StateLocalGovernmentThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentTwelveMonthPercentChangeData': {
            'class': ECI.StateLocalGovernmentTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 8: Civilian Workers Wages and Salaries
        'CivilianWorkerWagesSalariesIndexData': {
            'class': ECI.CivilianWorkerWagesSalariesIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesThreeMonthPercentChangeData': {
            'class': ECI.CivilianWorkerWagesSalariesThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesTwelveMonthPercentChangeData': {
            'class': ECI.CivilianWorkerWagesSalariesTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 9: Private Industry Workers Wages and Salaries
        'PrivateIndustryWorkerWagesSalariesIndexData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesIndexData,
            'category_property': ECI.hasPrivateIndustryOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesThreeMonthPercentChangeData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesThreeMonthPercentChangeData,
            'category_property': ECI.hasPrivateIndustryOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesTwelveMonthPercentChangeData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesTwelveMonthPercentChangeData,
            'category_property': ECI.hasPrivateIndustryOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 10: Private Industry by Bargaining Status and Region (Wages and Salaries)
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 11: State and Local Government Workers Wages and Salaries
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 12: Benefits by Occupation, Industry, and Bargaining Status
        'BenefitsByOccupationIndustryBargainingStatusIndexData': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusIndexData,
            'category_property': ECI.hasBenefitsOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData,
            'category_property': ECI.hasBenefitsOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData,
            'category_property': ECI.hasBenefitsOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table 13: Private Industry by Area (12-month only)
        'PrivateIndustryWorkerTotalCompensationByAreaTwelveMonthPercentChangeData': {
            'class': ECI.PrivateIndustryWorkerTotalCompensationByAreaTwelveMonthPercentChangeData,
            'category_property': ECI.hasMetropolitanArea,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByAreaTwelveMonthPercentChangeData': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByAreaTwelveMonthPercentChangeData,
            'category_property': ECI.hasMetropolitanArea,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Table A: Major Series
        'ThreeMonthSeasonallyAdjustedPercentChangeData': {
            'class': ECI.ThreeMonthSeasonallyAdjustedPercentChangeData,
            'category_property': ECI.hasMajorSeriesWorkerCategory,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthNotSeasonallyAdjustedCurrentDollarPercentChangeData': {
            'class': ECI.TwelveMonthNotSeasonallyAdjustedCurrentDollarPercentChangeData,
            'category_property': ECI.hasMajorSeriesWorkerCategory,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthNotSeasonallyAdjustedConstantDollarPercentChangeData': {
            'class': ECI.TwelveMonthNotSeasonallyAdjustedConstantDollarPercentChangeData,
            'category_property': ECI.hasMajorSeriesWorkerCategory,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        }
    },
    'jolts': {
        'JobOpeningsLevel': {
            'class': JOLTS.JobOpeningsLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'JobOpeningsRate': {
            'class': JOLTS.JobOpeningsRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresLevel': {
            'class': JOLTS.HiresLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresRate': {
            'class': JOLTS.HiresRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsLevel': {
            'class': JOLTS.TotalSeparationsLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsRate': {
            'class': JOLTS.TotalSeparationsRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsLevel': {
            'class': JOLTS.QuitsLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsRate': {
            'class': JOLTS.QuitsRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsDischargesLevel': {
            'class': JOLTS.LayoffsDischargesLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsDischargesRate': {
            'class': JOLTS.LayoffsDischargesRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsLevel': {
            'class': JOLTS.OtherSeparationsLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsRate': {
            'class': JOLTS.OtherSeparationsRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        }
    },
    'empsit': {
        # Establishment Survey Tables (9 tables)
        'EstablishmentData': {
            'class': EMPSIT.EstablishmentData,
            'category_property': EMPSIT.hasCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmployeeCount': {
            'class': EMPSIT.EmployeeCount,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'WeeklyHoursData': {
            'class': EMPSIT.WeeklyHoursData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EarningsData': {
            'class': EMPSIT.EarningsData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'IndexData': {
            'class': EMPSIT.IndexData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'WomenEmploymentData': {
            'class': EMPSIT.WomenEmploymentData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'ProductionNonsupervisoryEmploymentData': {
            'class': EMPSIT.ProductionNonsupervisoryEmploymentData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'AverageWeeklyHoursData': {
            'class': EMPSIT.AverageWeeklyHoursData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'AverageOvertimeHoursData': {
            'class': EMPSIT.AverageOvertimeHoursData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'AverageHourlyEarningsData': {
            'class': EMPSIT.AverageHourlyEarningsData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'AverageWeeklyEarningsData': {
            'class': EMPSIT.AverageWeeklyEarningsData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'IndexOfAggregateWeeklyHoursData': {
            'class': EMPSIT.IndexOfAggregateWeeklyHoursData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'IndexOfAggregateWeeklyPayrollsData': {
            'class': EMPSIT.IndexOfAggregateWeeklyPayrollsData,
            'category_property': EMPSIT.hasIndustry,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },

        # Household Survey Tables (18 tables)
        'HouseholdData': {
            'class': EMPSIT.HouseholdData,
            'category_property': EMPSIT.hasCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusData': {
            'class': EMPSIT.EmploymentStatusData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusByRaceData': {
            'class': EMPSIT.EmploymentStatusByRaceData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusByEthnicityData': {
            'class': EMPSIT.EmploymentStatusByEthnicityData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusByEducationData': {
            'class': EMPSIT.EmploymentStatusByEducationData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusByVeteranData': {
            'class': EMPSIT.EmploymentStatusByVeteranData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusByDisabilityData': {
            'class': EMPSIT.EmploymentStatusByDisabilityData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentStatusByNativityData': {
            'class': EMPSIT.EmploymentStatusByNativityData,
            'category_property': EMPSIT.hasStatusCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'ClassOfWorkerData': {
            'class': EMPSIT.ClassOfWorkerData,
            'category_property': EMPSIT.hasCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'EmploymentIndicatorData': {
            'class': EMPSIT.EmploymentIndicatorData,
            'category_property': EMPSIT.hasCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'UnemploymentIndicatorData': {
            'class': EMPSIT.UnemploymentIndicatorData,
            'category_property': EMPSIT.hasCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'UnemploymentReasonData': {
            'class': EMPSIT.UnemploymentReasonData,
            'category_property': EMPSIT.hasReason,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'UnemploymentDurationData': {
            'class': EMPSIT.UnemploymentDurationData,
            'category_property': EMPSIT.hasDuration,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'OccupationData': {
            'class': EMPSIT.OccupationData,
            'category_property': EMPSIT.hasOccupation,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'IndustryClassOfWorkerData': {
            'class': EMPSIT.IndustryClassOfWorkerData,
            'category_property': EMPSIT.hasIndustryClassOfWorker,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'UnderutilizationData': {
            'class': EMPSIT.UnderutilizationData,
            'category_property': EMPSIT.hasUnderutilizationMeasure,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        },
        'LaborForceCategoryData': {
            'class': EMPSIT.LaborForceCategoryData,
            'category_property': EMPSIT.hasLaborForceCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        }
    }
}


class BLSIntraSourceLinker:
    """
    Enriches BLS data with intra-source relationships across all BLS datasets.

    Uses encoded domain knowledge to create links. Patterns are only defined
    for datasets whose ontologies have been analyzed (currently: CPI, PPI, ECI, JOLTS, EMPSIT).

    When new datasets are added, update:
    1. BLS_SECTOR_PATTERNS - Add keywords for the new dataset
    2. KNOWN_CORRELATIONS - Add correlations involving the new dataset
    3. MEASUREMENT_TYPES - Add measurement type mappings

    The linker automatically detects which datasets are present in the graph
    and only applies patterns for available datasets.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.enrichment_count = 0
        self.stats = {
            'temporal_unified': 0,
            'temporal_sequences': 0,
            'sector_links': 0,
            'correlation_links': 0,
            'hierarchy_links': 0
        }

        # Month ordering for temporal sequences
        self.month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        # Detect which datasets are present in the graph
        self.available_datasets = self._detect_available_datasets()
        logger.info(f"Detected datasets in graph: {', '.join(self.available_datasets)}")

    def _detect_available_datasets(self) -> Set[str]:
        """
        Detect which BLS datasets are present in the graph.

        Returns:
            Set of dataset names (e.g., {'cpi', 'ppi', 'eci', 'empsit'})
        """
        datasets = set()

        # Check for each dataset by looking for its namespace
        dataset_checks = {
            'cpi': CPI,
            'ppi': PPI,
            'eci': ECI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'laus': LAUS,
            'metro': METRO,
            'realer': REALER,
            'wkyeng': WKYENG,
            'ximpim': XIMPIM
        }

        for dataset_name, namespace in dataset_checks.items():
            query = f"""
            ASK {{
                ?s ?p ?o .
                FILTER(STRSTARTS(STR(?s), "{namespace}"))
            }}
            """
            if self.graph.query(query).askAnswer:
                datasets.add(dataset_name)

        return datasets

    def enrich(self) -> Dict[str, int]:
        """
        Run all BLS intra-source enrichment steps.

        Only applies enrichment patterns for datasets that are present in the graph
        and have been configured in the pattern dictionaries.

        Returns:
            Dict with enrichment statistics
        """
        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting BLS Intra-Source Enrichment")
        logger.info(f"Available datasets: {', '.join(sorted(self.available_datasets))}")
        logger.info("=" * 60)

        # Step 1: Unify temporal entities across all BLS datasets
        logger.info("\n[Step 1/5] Unifying temporal entities...")
        self.unify_temporal_entities()

        # Step 2: Link temporal sequences
        logger.info("\n[Step 2/5] Linking temporal sequences...")
        self.link_temporal_sequences()

        # Step 3: Apply sector-based linking patterns
        logger.info("\n[Step 3/5] Applying sector-based patterns...")
        self.apply_sector_patterns()

        # Step 4: Apply known correlations
        logger.info("\n[Step 4/5] Applying known correlations...")
        self.apply_known_correlations()

        # Step 5: Link hierarchies across datasets
        logger.info("\n[Step 5/5] Linking cross-dataset hierarchies...")
        self.link_cross_dataset_hierarchies()

        final_count = len(self.graph)
        self.enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("BLS Intra-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {self.enrichment_count}")
        logger.info(f"  - Temporal unification: {self.stats['temporal_unified']}")
        logger.info(f"  - Temporal sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Correlation links: {self.stats['correlation_links']}")
        logger.info(f"  - Hierarchy links: {self.stats['hierarchy_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': self.enrichment_count,
            'available_datasets': list(self.available_datasets),
            **self.stats
        }

    def unify_temporal_entities(self):
        """
        Unify temporal entities across all BLS datasets.

        Creates unified Month and Year entities that link to dataset-specific temporal entities.
        Example: cpi:January, ppi:January, eci:January, empsit:January → bls:January

        This works for any BLS dataset that uses Month and Year classes.
        """
        # Find all Month entities across all BLS datasets
        month_query = """
        SELECT DISTINCT ?month ?monthClass WHERE {
            ?month a ?monthClass .
            FILTER(
                ?monthClass = <https://www.bls.gov/cpi/Month> ||
                ?monthClass = <https://www.bls.gov/ppi/Month> ||
                ?monthClass = <https://www.bls.gov/eci/Month> ||
                ?monthClass = <https://www.bls.gov/jolts/Month> ||
                ?monthClass = <https://www.bls.gov/empsit/Month> ||
                ?monthClass = <https://www.bls.gov/laus/Month> ||
                ?monthClass = <https://www.bls.gov/metro/Month> ||
                ?monthClass = <https://www.bls.gov/realer/Month> ||
                ?monthClass = <https://www.bls.gov/wkyeng/Month> ||
                ?monthClass = <https://www.bls.gov/ximpim/Month>
            )
        }
        """

        # Find all Year entities
        year_query = """
        SELECT DISTINCT ?year ?yearClass WHERE {
            ?year a ?yearClass .
            FILTER(
                ?yearClass = <https://www.bls.gov/cpi/Year> ||
                ?yearClass = <https://www.bls.gov/ppi/Year> ||
                ?yearClass = <https://www.bls.gov/eci/Year> ||
                ?yearClass = <https://www.bls.gov/jolts/Year> ||
                ?yearClass = <https://www.bls.gov/empsit/Year> ||
                ?yearClass = <https://www.bls.gov/laus/Year> ||
                ?yearClass = <https://www.bls.gov/metro/Year> ||
                ?yearClass = <https://www.bls.gov/realer/Year> ||
                ?yearClass = <https://www.bls.gov/wkyeng/Year> ||
                ?yearClass = <https://www.bls.gov/ximpim/Year>
            )
        }
        """

        months = list(self.graph.query(month_query))
        years = list(self.graph.query(year_query))

        logger.info(f"Found {len(months)} month entities, {len(years)} year entities")

        # Group months by name
        month_groups = {}
        for row in months:
            month_name = get_month_name(row.month)
            if month_name not in month_groups:
                month_groups[month_name] = []
            month_groups[month_name].append(row.month)

        # Group years by value
        year_groups = {}
        for row in years:
            year_value = get_year_value(row.year)
            if year_value not in year_groups:
                year_groups[year_value] = []
            year_groups[year_value].append(row.year)

        # Create unified temporal entities
        for month_name, month_uris in month_groups.items():
            if len(month_uris) > 1:
                unified_month = BLS_ENRICHMENT[month_name]
                self.graph.add((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))
                self.graph.add((unified_month, RDFS.label, Literal(month_name, datatype=XSD.string)))

                for month_uri in month_uris:
                    self.graph.add((month_uri, BLS_ENRICHMENT.unifiedAs, unified_month))
                    self.stats['temporal_unified'] += 1

        for year_value, year_uris in year_groups.items():
            if len(year_uris) > 1:
                unified_year = BLS_ENRICHMENT[year_value]
                self.graph.add((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))
                self.graph.add((unified_year, RDFS.label, Literal(year_value, datatype=XSD.string)))

                for year_uri in year_uris:
                    self.graph.add((year_uri, BLS_ENRICHMENT.unifiedAs, unified_year))
                    self.stats['temporal_unified'] += 1

        logger.info(f"Created {len(month_groups)} unified months, {len(year_groups)} unified years")
        logger.info(f"Added {self.stats['temporal_unified']} unification links")

    def link_temporal_sequences(self):
        """
        Add temporal ordering for measurements across all datasets.

        Links consecutive time periods using bls:precedes relationship.
        Only processes measurement types that are configured in MEASUREMENT_TYPES.
        """
        # Link sequences for each measurement type in each dataset
        for dataset, measurement_types in MEASUREMENT_TYPES.items():
            if dataset not in self.available_datasets:
                logger.info(f"  Skipping {dataset} (not in graph)")
                continue

            for measurement_name, config in measurement_types.items():
                count = self._link_sequences_for_measurement_type(
                    measurement_type=config['class'],
                    category_property=config['category_property'],
                    month_property=config['month_property'],
                    year_property=config['year_property'],
                    dataset_name=dataset,
                    measurement_name=measurement_name
                )
                self.stats['temporal_sequences'] += count

    def _link_sequences_for_measurement_type(self,
                                             measurement_type: URIRef,
                                             category_property: URIRef,
                                             month_property: URIRef,
                                             year_property: URIRef,
                                             dataset_name: str,
                                             measurement_name: str) -> int:
        """Helper to link temporal sequences for a specific measurement type"""

        query = f"""
        SELECT ?measurement ?category ?month ?year WHERE {{
            ?measurement a <{measurement_type}> ;
                        <{category_property}> ?category ;
                        <{month_property}> ?month ;
                        <{year_property}> ?year .
        }}
        """

        results = list(self.graph.query(query))

        if not results:
            return 0

        # Group by category
        by_category = {}
        for row in results:
            category = row.category
            if category not in by_category:
                by_category[category] = []

            by_category[category].append({
                'measurement': row.measurement,
                'month': get_month_name(row.month),
                'year': get_year_value(row.year)
            })

        # For each category, sort by time and add precedes links
        links_added = 0
        for category, measurements in by_category.items():
            measurements.sort(key=lambda x: (
                int(x['year']),
                self.month_order.index(x['month'])
            ))

            for i in range(len(measurements) - 1):
                current = measurements[i]['measurement']
                next_measurement = measurements[i + 1]['measurement']

                self.graph.add((
                    current,
                    BLS_ENRICHMENT.precedes,
                    next_measurement
                ))
                links_added += 1

        if links_added > 0:
            logger.info(
                f"  {dataset_name}.{measurement_name}: {links_added} sequence links across {len(by_category)} categories")
        return links_added

    def apply_sector_patterns(self):
        """
        Apply sector-based linking patterns from configuration.

        Creates economic sector entities and links related entities across datasets.
        Only applies patterns for datasets that are:
        1. Present in the graph (self.available_datasets)
        2. Configured in BLS_SECTOR_PATTERNS
        """
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            logger.info(f"  Processing sector: {sector_name}")

            # Create sector entity
            sector_uri = pattern['sector_uri']
            self.graph.add((sector_uri, RDF.type, BLS_ENRICHMENT.EconomicSector))
            self.graph.add((sector_uri, RDFS.label, Literal(pattern['description'], datatype=XSD.string)))

            # Find entities in each dataset matching keywords
            dataset_entities = {}
            total_entities = 0

            for dataset, keywords in pattern['keywords'].items():
                # Only process if dataset is available AND configured
                if dataset not in self.available_datasets:
                    continue

                if not keywords:
                    continue

                namespace = self._get_namespace_for_dataset(dataset)
                if namespace:
                    entities = self._find_entities_by_keywords(namespace, keywords)
                    dataset_entities[dataset] = entities
                    total_entities += len(entities)

                    # Link entities to sector
                    for entity in entities:
                        self.graph.add((entity, BLS_ENRICHMENT.belongsToSector, sector_uri))
                        self.stats['sector_links'] += 1

            # Create cross-dataset correlations within sector
            datasets = list(dataset_entities.keys())
            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    dataset1 = datasets[i]
                    dataset2 = datasets[j]

                    for entity1 in dataset_entities[dataset1]:
                        for entity2 in dataset_entities[dataset2]:
                            self.graph.add((entity1, pattern['relationship'], entity2))
                            self.stats['sector_links'] += 1

            if total_entities > 0:
                logger.info(f"    Found {total_entities} entities across {len(dataset_entities)} datasets")

    def apply_known_correlations(self):
        """
        Apply specific known correlations from domain expertise.

        Creates directed relationships based on economic theory.
        Only applies correlations where both source and target datasets are available.
        """
        for correlation in KNOWN_CORRELATIONS:
            source_dataset = correlation['source_dataset']
            target_dataset = correlation['target_dataset']

            # Check if both datasets are available
            if source_dataset not in self.available_datasets:
                continue

            if target_dataset not in self.available_datasets:
                continue

            logger.info(f"  Applying: {correlation['name']}")

            source_ns = self._get_namespace_for_dataset(source_dataset)
            target_ns = self._get_namespace_for_dataset(target_dataset)

            if not source_ns or not target_ns:
                continue

            source_entities = self._find_entities_by_pattern(source_ns, correlation['source_pattern'])
            target_entities = self._find_entities_by_pattern(target_ns, correlation['target_pattern'])

            links_added = 0
            for source_entity in source_entities:
                for target_entity in target_entities:
                    self.graph.add((source_entity, correlation['relationship'], target_entity))
                    links_added += 1

                    # Add metadata about the correlation
                    if 'lag_months' in correlation:
                        correlation_node = BLS_ENRICHMENT[f"Correlation_{correlation['name']}"]
                        self.graph.add((correlation_node, RDF.type, BLS_ENRICHMENT.EconomicCorrelation))
                        self.graph.add((correlation_node, BLS_ENRICHMENT.lagMonths,
                                        Literal(correlation['lag_months'], datatype=XSD.integer)))
                        self.graph.add((correlation_node, BLS_ENRICHMENT.strength,
                                        Literal(correlation['strength'], datatype=XSD.string)))

            self.stats['correlation_links'] += links_added
            if links_added > 0:
                logger.info(f"    Created {links_added} correlation links")

    def link_cross_dataset_hierarchies(self):
        """
        Link hierarchies across datasets based on similar paths.

        Currently implemented for:
        - CPI ↔ PPI
        - CPI ↔ ECI
        - CPI ↔ EMPSIT
        - PPI ↔ ECI
        - PPI ↔ EMPSIT
        - ECI ↔ JOLTS
        - ECI ↔ EMPSIT
        - JOLTS ↔ EMPSIT

        This method can be extended as more datasets with hierarchies are added.
        """
        # CPI ↔ PPI hierarchies
        if 'cpi' in self.available_datasets and 'ppi' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'cpi', 'ppi',
                CPI.hasParent, PPI.hasParent,
                CPI.hasFullPath, PPI.hasFullPath
            )

        # CPI ↔ ECI hierarchies
        if 'cpi' in self.available_datasets and 'eci' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'cpi', 'eci',
                CPI.hasParent, ECI.hasParentIndustry,
                CPI.hasFullPath, ECI.hasFullPath
            )

        # CPI ↔ EMPSIT hierarchies
        if 'cpi' in self.available_datasets and 'empsit' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'cpi', 'empsit',
                CPI.hasParent, EMPSIT.hasParentCategory,
                CPI.hasFullPath, EMPSIT.hasFullPath
            )

        # PPI ↔ ECI hierarchies
        if 'ppi' in self.available_datasets and 'eci' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'ppi', 'eci',
                PPI.hasParent, ECI.hasParentIndustry,
                PPI.hasFullPath, ECI.hasFullPath
            )

        # PPI ↔ EMPSIT hierarchies
        if 'ppi' in self.available_datasets and 'empsit' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'ppi', 'empsit',
                PPI.hasParent, EMPSIT.hasParentIndustry,
                PPI.hasFullPath, EMPSIT.hasFullPath
            )

        # ECI ↔ JOLTS hierarchies
        if 'eci' in self.available_datasets and 'jolts' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'eci', 'jolts',
                ECI.hasParentIndustry, JOLTS.hasParent,
                ECI.hasFullPath, JOLTS.hasFullPath
            )

        # ECI ↔ EMPSIT hierarchies
        if 'eci' in self.available_datasets and 'empsit' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'eci', 'empsit',
                ECI.hasParentIndustry, EMPSIT.hasParentIndustry,
                ECI.hasFullPath, EMPSIT.hasFullPath
            )

        # JOLTS ↔ EMPSIT hierarchies
        if 'jolts' in self.available_datasets and 'empsit' in self.available_datasets:
            self._link_hierarchies_between_datasets(
                'jolts', 'empsit',
                JOLTS.hasParent, EMPSIT.hasParentIndustry,
                JOLTS.hasFullPath, EMPSIT.hasFullPath
            )

    def _link_hierarchies_between_datasets(self,
                                          dataset1: str,
                                          dataset2: str,
                                          parent_prop1: URIRef,
                                          parent_prop2: URIRef,
                                          path_prop1: URIRef,
                                          path_prop2: URIRef):
        """Helper to link hierarchies between two datasets"""

        query1 = f"""
        SELECT ?category ?parent ?fullPath ?label WHERE {{
            ?category <{parent_prop1}> ?parent ;
                     <{path_prop1}> ?fullPath .
            OPTIONAL {{ ?category rdfs:label ?label }}
        }}
        """

        query2 = f"""
        SELECT ?category ?parent ?fullPath ?label WHERE {{
            ?category <{parent_prop2}> ?parent ;
                     <{path_prop2}> ?fullPath .
            OPTIONAL {{ ?category rdfs:label ?label }}
        }}
        """

        hierarchies1 = list(self.graph.query(query1))
        hierarchies2 = list(self.graph.query(query2))

        if not hierarchies1 or not hierarchies2:
            return

        logger.info(f"  Analyzing {len(hierarchies1)} {dataset1} and {len(hierarchies2)} {dataset2} hierarchies")

        links_added = 0
        for row1 in hierarchies1:
            path1 = str(row1.fullPath).lower()
            label1 = str(row1.label).lower() if row1.label else ""

            for row2 in hierarchies2:
                path2 = str(row2.fullPath).lower()
                label2 = str(row2.label).lower() if row2.label else ""

                # Check for keyword overlap in paths or labels
                if self._paths_are_related(path1, path2) or \
                        self._labels_are_related(label1, label2):
                    self.graph.add((
                        row1.category,
                        BLS_ENRICHMENT.relatedHierarchy,
                        row2.category
                    ))
                    links_added += 1

        self.stats['hierarchy_links'] += links_added
        if links_added > 0:
            logger.info(f"  Created {links_added} {dataset1}↔{dataset2} hierarchy links")

    # ============================================
    # HELPER METHODS
    # ============================================

    def _get_namespace_for_dataset(self, dataset: str) -> Optional[Namespace]:
        """Get namespace for a dataset name"""
        mapping = {
            'cpi': CPI,
            'ppi': PPI,
            'eci': ECI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'laus': LAUS,
            'metro': METRO,
            'realer': REALER,
            'wkyeng': WKYENG,
            'ximpim': XIMPIM
        }
        return mapping.get(dataset.lower())

    def _find_entities_by_keywords(self, namespace: Namespace, keywords: List[str]) -> List[URIRef]:
        """Find entities in a namespace containing any of the keywords"""
        if not keywords:
            return []

        keyword_filters = " || ".join([
            f'CONTAINS(LCASE(?label), "{kw.lower()}")'
            for kw in keywords
        ])

        query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity rdfs:label ?label .
            FILTER(STRSTARTS(STR(?entity), "{namespace}"))
            FILTER({keyword_filters})
        }}
        """

        results = self.graph.query(query)
        return [row.entity for row in results]

    def _find_entities_by_pattern(self, namespace: Namespace, pattern: str) -> List[URIRef]:
        """Find entities matching a specific pattern"""
        query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity rdfs:label ?label .
            FILTER(STRSTARTS(STR(?entity), "{namespace}"))
            FILTER(CONTAINS(LCASE(?label), "{pattern.lower()}"))
        }}
        """

        results = self.graph.query(query)
        return [row.entity for row in results]

    def _paths_are_related(self, path1: str, path2: str) -> bool:
        """Check if two hierarchical paths are related"""
        # Keywords that indicate related concepts
        keywords = ['food', 'energy', 'housing', 'transportation', 'goods', 'services',
                    'manufacturing', 'construction', 'trade', 'warehousing', 'healthcare',
                    'retail', 'wholesale', 'professional', 'leisure', 'hospitality',
                    'information', 'financial', 'government', 'education', 'wages',
                    'salaries', 'compensation', 'benefits', 'union', 'nonunion']

        for keyword in keywords:
            if keyword in path1 and keyword in path2:
                return True

        return False

    def _labels_are_related(self, label1: str, label2: str) -> bool:
        """Check if two labels are related"""
        if not label1 or not label2:
            return False

        # Simple word overlap check
        words1 = set(label1.split())
        words2 = set(label2.split())

        # Remove common words
        common_words = {'the', 'and', 'or', 'for', 'of', 'in', 'to', 'a', 'an', 'by', 'with'}
        words1 -= common_words
        words2 -= common_words

        # Check for overlap
        overlap = words1 & words2
        return len(overlap) >= 1