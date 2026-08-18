"""
BLS Measurement Types - Temporal linking configurations
"""
from spark_jobs.utils.rdf_utils import (
    CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER, WKYENG
)


MEASUREMENT_TYPES = {
    'cpi': {
        # ============================================
        # Tables 1, 3: Index values (hasCategory)
        # ============================================
        'Index': {
            'class': CPI.Index,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },

    },
    'ppi': {
        # ============================================
        # Tables 1, 2: MonthlyChange (hasCommodityGrouping, hasEndMonth/hasEndYear)
        # ============================================
        'MonthlyChange': {
            'class': PPI.MonthlyChange,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasEndMonth,
            'year_property': PPI.hasEndYear
        },

        # ============================================
        # Tables 1, 2: TwelveMonthChange (hasCommodityGrouping, hasEndMonth/hasEndYear)
        # ============================================
        'TwelveMonthChange': {
            'class': PPI.TwelveMonthChange,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasEndMonth,
            'year_property': PPI.hasEndYear
        },

        # ============================================
        # Table 3: IndexValue (hasCommodityGrouping, hasMonth/hasYear)
        # ============================================
        'IndexValue': {
            'class': PPI.IndexValue,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        },

    },

    'eci': {
        # ============================================
        # Table 1: Total Compensation by Occupational Group and Industry (Seasonally Adjusted)
        # ============================================

        # Index data partitioned by occupational group
        # Index data partitioned by industry
        # Percent change partitioned by occupational group
        # Percent change partitioned by industry

        # ============================================
        # Table 2: Wages and Salaries by Occupational Group and Industry (Seasonally Adjusted)
        # ============================================

        'WagesAndSalariesIndexData_OccupationalGroup': {
            'class': ECI.WagesAndSalariesIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'WagesAndSalariesIndexData_Industry': {
            'class': ECI.WagesAndSalariesIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 3: Benefits by Occupational Group and Industry (Seasonally Adjusted)
        # ============================================

        'BenefitsIndexData_OccupationalGroup': {
            'class': ECI.BenefitsIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsIndexData_Industry': {
            'class': ECI.BenefitsIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 4: Total Compensation for Civilian Workers (Not Seasonally Adjusted)
        # ============================================

        'CivilianWorkerIndexData_OccupationalGroup': {
            'class': ECI.CivilianWorkerIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerIndexData_Industry': {
            'class': ECI.CivilianWorkerIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'ThreeMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.ThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'ThreeMonthPercentChangeData_Industry': {
            'class': ECI.ThreeMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.TwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthPercentChangeData_Industry': {
            'class': ECI.TwelveMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 5: Total Compensation for Private Industry Workers by Occ/Industry (Not Seasonally Adjusted)
        # Shared classes with Table 6 — partitioned by occupational group and industry here
        # ============================================

        'PrivateIndustryWorkerIndexData_OccupationalGroup': {
            'class': ECI.PrivateIndustryWorkerIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerIndexData_Industry': {
            'class': ECI.PrivateIndustryWorkerIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryThreeMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.PrivateIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryThreeMonthPercentChangeData_Industry': {
            'class': ECI.PrivateIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryTwelveMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.PrivateIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryTwelveMonthPercentChangeData_Industry': {
            'class': ECI.PrivateIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 7: Total Compensation for State/Local Government Workers (Not Seasonally Adjusted)
        # ============================================

        'StateLocalGovernmentWorkerIndexData_OccupationalGroup': {
            'class': ECI.StateLocalGovernmentWorkerIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerIndexData_Industry': {
            'class': ECI.StateLocalGovernmentWorkerIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentThreeMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.StateLocalGovernmentThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentThreeMonthPercentChangeData_Industry': {
            'class': ECI.StateLocalGovernmentThreeMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentTwelveMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.StateLocalGovernmentTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentTwelveMonthPercentChangeData_Industry': {
            'class': ECI.StateLocalGovernmentTwelveMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 8: Wages/Salaries for Civilian Workers (Not Seasonally Adjusted)
        # ============================================

        'CivilianWorkerWagesSalariesIndexData_OccupationalGroup': {
            'class': ECI.CivilianWorkerWagesSalariesIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesIndexData_Industry': {
            'class': ECI.CivilianWorkerWagesSalariesIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesThreeMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.CivilianWorkerWagesSalariesThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesThreeMonthPercentChangeData_Industry': {
            'class': ECI.CivilianWorkerWagesSalariesThreeMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesTwelveMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.CivilianWorkerWagesSalariesTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'CivilianWorkerWagesSalariesTwelveMonthPercentChangeData_Industry': {
            'class': ECI.CivilianWorkerWagesSalariesTwelveMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 9: Wages/Salaries for Private Industry Workers by Occ/Industry (Not Seasonally Adjusted)
        # ============================================

        'PrivateIndustryWorkerWagesSalariesIndexData_OccupationalGroup': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesIndexData_Industry': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesThreeMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesThreeMonthPercentChangeData_Industry': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesThreeMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesTwelveMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesTwelveMonthPercentChangeData_Industry': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesTwelveMonthPercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

    },

    'ximpim': {
        # ============================================
        # Tables 1, 3, 5, 7: ImportPriceIndex (hasMonth/hasYear)
        # Linked via different category properties per table.
        # Each category property variant needs its own entry.
        # ============================================

        # Tables 1, 2: Import/Export Price Index by End Use Category
        'ImportPriceIndex_EndUse': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },
        'ExportPriceIndex_EndUse': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Tables 1, 2: RelativeImportance by End Use Category

        # Tables 1, 2: PercentChange by End Use Category
        'PercentChange_EndUse': {
            'class': XIMPIM.PercentChange,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Tables 3, 4: Import/Export Price Index by NAICS Industry

        # Tables 3, 4: RelativeImportance by NAICS Industry

        # Tables 3, 4: PercentChange by NAICS Industry

        # Tables 5, 6: Import/Export Price Index by Harmonized Category

        # Tables 5, 6: RelativeImportance by Harmonized Category

        # Tables 5, 6: PercentChange by Harmonized Category

        # Table 7: Import Price Index by Locality of Origin
        'ImportPriceIndex_LocalityOfOrigin': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasLocalityOfOrigin,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 7: RelativeImportance by Locality of Origin

        # Table 7: PercentChange by Locality of Origin
        'PercentChange_LocalityOfOrigin': {
            'class': XIMPIM.PercentChange,
            'category_property': XIMPIM.hasLocalityOfOrigin,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Table 8: Export Price Index by Locality of Destination
        'ExportPriceIndex_LocalityOfDestination': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasLocalityOfDestination,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 8: RelativeImportance by Locality of Destination

        # Table 8: PercentChange by Locality of Destination
        'PercentChange_LocalityOfDestination': {
            'class': XIMPIM.PercentChange,
            'category_property': XIMPIM.hasLocalityOfDestination,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Table 9: Terms of Trade Index by Locality
        'TermsOfTradeIndex': {
            'class': XIMPIM.TermsOfTradeIndex,
            'category_property': XIMPIM.hasLocality,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

    },

    'laus': {
        # ============================================
        # Tables 1, 2: Labor Force Data container
        # Contains embedded CivilianLaborForce, UnemploymentNumber,
        # UnemploymentRate as BNodes. Only the container entity has
        # hasState/hasMonth/hasYear properties, so we create temporal
        # chains on the container, partitioned by state.
        #
        # This produces precedes chains like:
        #   Alabama_2024_July → Alabama_2024_August → Alabama_2025_July → ...
        # allowing traversal of all labor force metrics for a state
        # in chronological order.
        # ============================================
        'LaborForceData': {
            'class': LAUS.LaborForceData,
            'category_property': LAUS.hasState,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },

        # ============================================
        # Tables 3, 3-continued, 4, 4-continued: Nonfarm Payroll Data container
        # Contains embedded TotalEmployment, ConstructionEmployment,
        # ManufacturingEmployment, TradeTransportationUtilitiesEmployment,
        # FinancialActivitiesEmployment, ProfessionalBusinessServicesEmployment,
        # InformationEmployment, MiningLoggingEmployment as BNodes.
        # Only the container has hasState/hasMonth/hasYear.
        #
        # This produces precedes chains like:
        #   California_2024_July → California_2024_August → ...
        # allowing traversal of all nonfarm payroll metrics for a state
        # in chronological order.
        # ============================================
        'NonfarmPayrollData': {
            'class': LAUS.NonfarmPayrollData,
            'category_property': LAUS.hasState,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },
    },

    'jolts': {
        # ============================================
        # Tables 1, 8, A: Job Openings by Industry (hasIndustry)
        # ============================================
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

        # ============================================
        # Tables 1, 8: Job Openings by Region (hasRegion)
        # ============================================
        'JobOpeningsLevel_Region': {
            'class': JOLTS.JobOpeningsLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'JobOpeningsRate_Region': {
            'class': JOLTS.JobOpeningsRate,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Tables 2, 9, A: Hires by Industry (hasIndustry)
        # ============================================
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

        # ============================================
        # Tables 2, 9: Hires by Region (hasRegion)
        # ============================================
        'HiresLevel_Region': {
            'class': JOLTS.HiresLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresRate_Region': {
            'class': JOLTS.HiresRate,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Tables 3, 10, A: Total Separations by Industry (hasIndustry)
        # ============================================
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

        # ============================================
        # Tables 3, 10: Total Separations by Region (hasRegion)
        # ============================================
        'TotalSeparationsLevel_Region': {
            'class': JOLTS.TotalSeparationsLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsRate_Region': {
            'class': JOLTS.TotalSeparationsRate,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Tables 4, 11: Quits by Industry (hasIndustry)
        # ============================================
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

        # ============================================
        # Tables 4, 11: Quits by Region (hasRegion)
        # ============================================
        'QuitsLevel_Region': {
            'class': JOLTS.QuitsLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsRate_Region': {
            'class': JOLTS.QuitsRate,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Tables 5: Layoffs and Discharges by Industry (hasIndustry)
        # Note: Tables 5, 7 use "LayoffsAndDischarges" class names
        # ============================================
        'LayoffsAndDischargesLevel': {
            'class': JOLTS.LayoffsAndDischargesLevel,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsAndDischargesRate': {
            'class': JOLTS.LayoffsAndDischargesRate,
            'category_property': JOLTS.hasIndustry,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Table 5: Layoffs and Discharges by Region (hasRegion)
        # ============================================
        'LayoffsAndDischargesLevel_Region': {
            'class': JOLTS.LayoffsAndDischargesLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsAndDischargesRate_Region': {
            'class': JOLTS.LayoffsAndDischargesRate,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Tables 6, 13: Other Separations by Industry (hasIndustry)
        # ============================================
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
        },

        # ============================================
        # Tables 6, 13: Other Separations by Region (hasRegion)
        # ============================================
        'OtherSeparationsLevel_Region': {
            'class': JOLTS.OtherSeparationsLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsRate_Region': {
            'class': JOLTS.OtherSeparationsRate,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

        # ============================================
        # Table 7: All metrics by Establishment Size Class (hasEstablishmentSizeClass)
        # ============================================
        'JobOpeningsLevel_EstablishmentSize': {
            'class': JOLTS.JobOpeningsLevel,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'JobOpeningsRate_EstablishmentSize': {
            'class': JOLTS.JobOpeningsRate,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresLevel_EstablishmentSize': {
            'class': JOLTS.HiresLevel,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresRate_EstablishmentSize': {
            'class': JOLTS.HiresRate,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsLevel_EstablishmentSize': {
            'class': JOLTS.TotalSeparationsLevel,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsRate_EstablishmentSize': {
            'class': JOLTS.TotalSeparationsRate,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsLevel_EstablishmentSize': {
            'class': JOLTS.QuitsLevel,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsRate_EstablishmentSize': {
            'class': JOLTS.QuitsRate,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsAndDischargesLevel_EstablishmentSize': {
            'class': JOLTS.LayoffsAndDischargesLevel,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsAndDischargesRate_EstablishmentSize': {
            'class': JOLTS.LayoffsAndDischargesRate,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsLevel_EstablishmentSize': {
            'class': JOLTS.OtherSeparationsLevel,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsRate_EstablishmentSize': {
            'class': JOLTS.OtherSeparationsRate,
            'category_property': JOLTS.hasEstablishmentSizeClass,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },

    },
    'empsit': {
        # Establishment Survey Tables (9 tables)
        'EmployeeCount': {
            'class': EMPSIT.EmployeeCount,
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
        'LaborForceCategoryData': {
            'class': EMPSIT.LaborForceCategoryData,
            'category_property': EMPSIT.hasLaborForceCategory,
            'month_property': EMPSIT.hasMonth,
            'year_property': EMPSIT.hasYear
        }
    },

        'metro': {
        # ============================================
        # Tables 1, 2: Civilian Labor Force Data container
        # Contains embedded CivilianLaborForce, UnemployedNumber,
        # UnemploymentRate as BNodes. Only the container has
        # hasRegion/hasMonth/hasYear.
        # Partitioned by region (state, metro area, or metro division).
        # ============================================
        'CivilianLaborForceData': {
            'class': METRO.CivilianLaborForceData,
            'category_property': METRO.hasRegion,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # ============================================
        # Tables 3, 4: Nonfarm Payroll Data container
        # Contains embedded NonfarmPayrollEmployment, EmploymentChange,
        # EmploymentChangePercent as BNodes. Only the container has
        # hasRegion/hasMonth/hasYear.
        # Partitioned by region (state, metro area, or metro division).
        # ============================================
        'NonfarmPayrollData': {
            'class': METRO.NonfarmPayrollData,
            'category_property': METRO.hasRegion,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },
    },

    'realer': {
        'MonthlyEarningsData': {
            'class': REALER.MonthlyEarningsData,
            'category_property': REALER.appliesTo,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
    },

    'wkyeng': {
        # Table 1: Quarterly Earnings by Sex (Seasonally Adjusted)
        'QuarterlyEarningsData': {
            'class': WKYENG.QuarterlyEarningsData,
            'category_property': WKYENG.hasSex,
            'quarter_property': WKYENG.hasQuarter,
            'year_property': WKYENG.hasYear,
            # Note: WKYENG uses quarter_property instead of month_property
            'month_property': None,
        },

        # Tables 2, 3: Quarterly Characteristic Data (Not Seasonally Adjusted)
        'QuarterlyCharacteristicData': {
            'class': WKYENG.QuarterlyCharacteristicData,
            'category_property': WKYENG.hasCharacteristic,
            'quarter_property': WKYENG.hasQuarter,
            'year_property': WKYENG.hasYear,
            'month_property': None,
        },

        # Table 4: Quarterly Occupation Data (Not Seasonally Adjusted)
        'QuarterlyOccupationData': {
            'class': WKYENG.QuarterlyOccupationData,
            'category_property': WKYENG.hasOccupation,
            'quarter_property': WKYENG.hasQuarter,
            'year_property': WKYENG.hasYear,
            'month_property': None,
        },

        # Table 5: Quarterly Distribution Data (Not Seasonally Adjusted)
        'QuarterlyDistributionData': {
            'class': WKYENG.QuarterlyDistributionData,
            'category_property': WKYENG.hasCharacteristic,
            'quarter_property': WKYENG.hasQuarter,
            'year_property': WKYENG.hasYear,
            'month_property': None,
        },

        # Table 6: Quarterly Part-Time Data (Not Seasonally Adjusted)
        'QuarterlyPartTimeData': {
            'class': WKYENG.QuarterlyPartTimeData,
            'category_property': WKYENG.hasCharacteristic,
            'quarter_property': WKYENG.hasQuarter,
            'year_property': WKYENG.hasYear,
            'month_property': None,
        },
    }
}