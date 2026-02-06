"""
BLS Sector Patterns - Economic sector definitions and keywords
"""
from glue_jobs.utils.rdf_utils import BLS_ENRICHMENT

BLS_SECTOR_PATTERNSBLS_SECTOR_PATTERNS = {
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
            'ximpim': [
                'Food',
                'Foodstuffs',
                'Beverages',
                'Agricultural products',
                'Food and live animals',
                'Meat and meat preparations',
                'Dairy products',
                'Fish and seafood',
                'Cereals and cereal preparations',
                'Vegetables and fruit',
                'Sugar and confectionery',
                'Coffee, tea, cocoa, spices',
                'Animal feed',
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
            'ximpim': [
                'Fuel',
                'Petroleum',
                'Crude oil',
                'Refined petroleum products',
                'Natural gas',
                'Coal',
                'Energy',
                'Mineral fuels',
                'Petroleum oils',
                'Gas oils',
                'Fuel oils',
                'Liquefied propane and butane',
                'Petroleum gases',
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
            'ximpim': [
                'Construction materials',
                'Building materials',
                'Lumber',
                'Wood products',
                'Cement',
                'Glass',
                'Ceramic products',
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
            'ximpim': [
                'Transportation services',
                'Air freight',
                'Air passenger fares',
                'Ocean freight',
                'Vehicles',
                'Motor vehicles',
                'Automobiles',
                'Trucks',
                'Aircraft',
                'Ships and boats',
                'Railway vehicles',
                'Transportation equipment',
                'Parts and accessories for vehicles',
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
            'ximpim': [
                'Goods',
                'Services',
                'Transportation services',
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
            'ximpim': [
                'Manufactured goods',
                'Machinery',
                'Electrical machinery',
                'Industrial machinery',
                'Computers',
                'Telecommunications equipment',
                'Semiconductors',
                'Electronic components',
                'Chemicals',
                'Pharmaceuticals',
                'Plastics',
                'Rubber products',
                'Textiles',
                'Apparel',
                'Footwear',
                'Wood products',
                'Paper products',
                'Printed materials',
                'Metal products',
                'Fabricated metal products',
                'Iron and steel',
                'Nonferrous metals',
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
            'ximpim': [
                'Consumer goods',
                'Household goods',
                'Furniture',
                'Home furnishings',
                'Appliances',
                'Clothing',
                'Footwear',
                'Jewelry',
                'Watches',
                'Toys and games',
                'Sporting goods',
                'Books',
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
            'ximpim': [
                'Pharmaceuticals',
                'Medicinal products',
                'Medical equipment',
                'Medical instruments',
                'Surgical instruments',
                'Diagnostic equipment',
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
            'ximpim': [
                'Computers',
                'Computer equipment',
                'Semiconductors',
                'Electronic components',
                'Telecommunications equipment',
                'Office machines',
                'Data processing equipment',
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
            'ximpim': [],
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
            'ximpim': [],
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
            'ximpim': [
                'Telecommunications equipment',
                'Communications equipment',
                'Broadcasting equipment',
                'Printed materials',
                'Books',
                'Newspapers',
                'Periodicals',
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
            'ximpim': [
                'Books',
                'Printed materials',
                'Educational materials',
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
            'ximpim': [],
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
            'ximpim': [
                'Agricultural products',
                'Forestry products',
                'Fish and seafood',
                'Timber',
                'Logs',
                'Wood',
                'Cork',
                'Animal products',
                'Hides and skins',
                'Crude fertilizers',
                'Ores',
                'Metal ores',
                'Minerals',
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
            'ximpim': [
                'Construction materials',
                'Building materials',
            ],
        },
        'relationship': BLS_ENRICHMENT.constructionTradesSectorCorrelation
    },
}