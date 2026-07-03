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

        # ============================================
        # Tables 1, 2, 3: Relative Importance (hasCategory)
        # ============================================
        'RelativeImportance': {
            'class': CPI.RelativeImportance,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },

        # ============================================
        # Tables 1, 2: Percent Change by category (hasEndMonth/hasEndYear)
        # ============================================
        'PercentChange': {
            'class': CPI.PercentChange,
            'category_property': CPI.hasCategory,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },

        # ============================================
        # Table 3: Special Aggregate variants
        # Same types but linked via hasSpecialAggregate instead of hasCategory
        # ============================================
        'Index_SpecialAggregate': {
            'class': CPI.Index,
            'category_property': CPI.hasSpecialAggregate,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },
        'RelativeImportance_SpecialAggregate': {
            'class': CPI.RelativeImportance,
            'category_property': CPI.hasSpecialAggregate,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },
        'PercentChange_SpecialAggregate': {
            'class': CPI.PercentChange,
            'category_property': CPI.hasSpecialAggregate,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },

        # ============================================
        # Table 4: Percent Change by area
        # ============================================
        'PercentChange_Area': {
            'class': CPI.PercentChange,
            'category_property': CPI.hasArea,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },

        # ============================================
        # Table 5: Chained CPI (C-CPI-U vs CPI-U)
        # These use hasTimePeriod → TimePeriod → hasMonth/hasYear
        # The base enricher can't follow 2-hop paths, so we link
        # TimePeriod entities directly (they ARE the temporal grouping)
        # Category = hasIndexType (CCPIU or CPIU)
        # ============================================
        'OneMonthPercentChange': {
            'class': CPI.OneMonthPercentChange,
            'category_property': CPI.hasIndexType,
            'month_property': CPI.hasTimePeriod,
            # Note: This won't work with the base enricher's month extraction
            # because hasTimePeriod points to a TimePeriod entity, not a Month.
            # We skip this — Table 5 has only ~24 rows, not worth custom logic.
            'year_property': None
        },
        'TwelveMonthPercentChange': {
            'class': CPI.TwelveMonthPercentChange,
            'category_property': CPI.hasIndexType,
            'month_property': CPI.hasTimePeriod,
            'year_property': None
        },

        # ============================================
        # Tables 6, 7: Percent Change by expenditure category
        # ============================================
        'PercentChange_ExpenditureCategory': {
            'class': CPI.PercentChange,
            'category_property': CPI.hasExpenditureCategory,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },

        # ============================================
        # Table 6: Effect on All Items
        # ============================================
        'EffectOnAllItems': {
            'class': CPI.EffectOnAllItems,
            'category_property': CPI.hasExpenditureCategory,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
        },

        # ============================================
        # Tables 6, 7: Standard Error
        # ============================================
        'StandardError': {
            'class': CPI.StandardError,
            'category_property': CPI.hasExpenditureCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },

        # ============================================
        # Table A: Seasonally Adjusted Change (month-to-month)
        # ============================================
        'SeasonallyAdjustedChange': {
            'class': CPI.SeasonallyAdjustedChange,
            'category_property': CPI.hasExpenditureCategory,
            'month_property': CPI.hasMonth,
            'year_property': CPI.hasYear
        },

        # ============================================
        # Table A: Unadjusted Annual Change (12-month)
        # ============================================
        'UnadjustedAnnualChange': {
            'class': CPI.UnadjustedAnnualChange,
            'category_property': CPI.hasExpenditureCategory,
            'month_property': CPI.hasEndMonth,
            'year_property': CPI.hasEndYear
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

        # ============================================
        # Table 1: RelativeImportance (hasCommodityGrouping, hasMonth/hasYear)
        # ============================================
        'RelativeImportance': {
            'class': PPI.RelativeImportance,
            'category_property': PPI.hasCommodityGrouping,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        },

        # ============================================
        # Tables A, B, C, D: MonthlyChange (hasPriceIndex, hasMonth/hasYear)
        # These are flat summary tables where changes link to PriceIndex
        # entities instead of CommodityGrouping entities.
        # ============================================
        'MonthlyChange_PriceIndex': {
            'class': PPI.MonthlyChange,
            'category_property': PPI.hasPriceIndex,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        },

        # ============================================
        # Tables A, B, C: TwelveMonthChange (hasPriceIndex, hasMonth/hasYear)
        # Same flat summary pattern — 12-month changes also use
        # hasMonth/hasYear (not hasEndMonth/hasEndYear) in these tables.
        # ============================================
        'TwelveMonthChange_PriceIndex': {
            'class': PPI.TwelveMonthChange,
            'category_property': PPI.hasPriceIndex,
            'month_property': PPI.hasMonth,
            'year_property': PPI.hasYear
        },
    },

    'eci': {
        # ============================================
        # Table 1: Total Compensation by Occupational Group and Industry (Seasonally Adjusted)
        # ============================================

        # Index data partitioned by occupational group
        'EmploymentCostIndexData_OccupationalGroup': {
            'class': ECI.EmploymentCostIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Index data partitioned by industry
        'EmploymentCostIndexData_Industry': {
            'class': ECI.EmploymentCostIndexData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Percent change partitioned by occupational group
        'PercentChangeData_OccupationalGroup': {
            'class': ECI.PercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        # Percent change partitioned by industry
        'PercentChangeData_Industry': {
            'class': ECI.PercentChangeData,
            'category_property': ECI.hasIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

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
        'WagesAndSalariesPercentChangeData_OccupationalGroup': {
            'class': ECI.WagesAndSalariesPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'WagesAndSalariesPercentChangeData_Industry': {
            'class': ECI.WagesAndSalariesPercentChangeData,
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
        'BenefitsPercentChangeData_OccupationalGroup': {
            'class': ECI.BenefitsPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsPercentChangeData_Industry': {
            'class': ECI.BenefitsPercentChangeData,
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
        'CivilianWorkerIndexData_SpecialCategory': {
            'class': ECI.CivilianWorkerIndexData,
            'category_property': ECI.hasSpecialCategory,
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
        'ThreeMonthPercentChangeData_SpecialCategory': {
            'class': ECI.ThreeMonthPercentChangeData,
            'category_property': ECI.hasSpecialCategory,
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
        'TwelveMonthPercentChangeData_SpecialCategory': {
            'class': ECI.TwelveMonthPercentChangeData,
            'category_property': ECI.hasSpecialCategory,
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
        # Table 6: Total Compensation by Bargaining Status and Census Region (Not Seasonally Adjusted)
        # Same data classes as Table 5 — partitioned by bargaining/census properties
        # ============================================

        'PrivateIndustryWorkerIndexData_BargainingStatus': {
            'class': ECI.PrivateIndustryWorkerIndexData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerIndexData_CensusRegion': {
            'class': ECI.PrivateIndustryWorkerIndexData,
            'category_property': ECI.hasCensusRegion,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerIndexData_CensusDivision': {
            'class': ECI.PrivateIndustryWorkerIndexData,
            'category_property': ECI.hasCensusDivision,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryThreeMonthPercentChangeData_BargainingStatus': {
            'class': ECI.PrivateIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryThreeMonthPercentChangeData_CensusRegion': {
            'class': ECI.PrivateIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasCensusRegion,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryThreeMonthPercentChangeData_CensusDivision': {
            'class': ECI.PrivateIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasCensusDivision,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryTwelveMonthPercentChangeData_BargainingStatus': {
            'class': ECI.PrivateIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryTwelveMonthPercentChangeData_CensusRegion': {
            'class': ECI.PrivateIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasCensusRegion,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryTwelveMonthPercentChangeData_CensusDivision': {
            'class': ECI.PrivateIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasCensusDivision,
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

        # ============================================
        # Table 10: Wages/Salaries by Bargaining Status and Census Region (Not Seasonally Adjusted)
        # Distinct classes from Table 9
        # ============================================

        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData_BargainingStatus': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData_BargainingStatusIndustry': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData,
            'category_property': ECI.hasBargainingStatusIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData_CensusRegion': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData,
            'category_property': ECI.hasCensusRegion,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData_CensusDivision': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionIndexData,
            'category_property': ECI.hasCensusDivision,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData_BargainingStatus': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData_BargainingStatusIndustry': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatusIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData_CensusRegion': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData,
            'category_property': ECI.hasCensusRegion,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData_CensusDivision': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionThreeMonthPercentChangeData,
            'category_property': ECI.hasCensusDivision,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData_BargainingStatus': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData_BargainingStatusIndustry': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData,
            'category_property': ECI.hasBargainingStatusIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData_CensusRegion': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData,
            'category_property': ECI.hasCensusRegion,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData_CensusDivision': {
            'class': ECI.PrivateIndustryWorkerWagesSalariesByBargainingStatusRegionTwelveMonthPercentChangeData,
            'category_property': ECI.hasCensusDivision,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 11: Wages/Salaries for State/Local Government by Occupation and Industry (Not Seasonally Adjusted)
        # ============================================

        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData_OccupationalGroup': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData_StateLocalIndustry': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData,
            'category_property': ECI.hasStateLocalIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData_WorkerCategory': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryIndexData,
            'category_property': ECI.hasWorkerCategory,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData_StateLocalIndustry': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasStateLocalIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData_WorkerCategory': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryThreeMonthPercentChangeData,
            'category_property': ECI.hasWorkerCategory,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData_OccupationalGroup': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData_StateLocalIndustry': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasStateLocalIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData_WorkerCategory': {
            'class': ECI.StateLocalGovernmentWorkerWagesSalariesByOccupationIndustryTwelveMonthPercentChangeData,
            'category_property': ECI.hasWorkerCategory,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 12: Benefits by Occupation, Industry, and Bargaining Status (Not Seasonally Adjusted)
        # ============================================

        'BenefitsByOccupationIndustryBargainingStatusIndexData_WorkerType': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusIndexData,
            'category_property': ECI.hasWorkerType,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusIndexData_BenefitsOccupationalGroup': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusIndexData,
            'category_property': ECI.hasBenefitsOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusIndexData_BenefitsIndustry': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusIndexData,
            'category_property': ECI.hasBenefitsIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusIndexData_BenefitsBargainingStatus': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusIndexData,
            'category_property': ECI.hasBenefitsBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData_WorkerType': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData,
            'category_property': ECI.hasWorkerType,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData_BenefitsOccupationalGroup': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData,
            'category_property': ECI.hasBenefitsOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData_BenefitsIndustry': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData,
            'category_property': ECI.hasBenefitsIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData_BenefitsBargainingStatus': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusThreeMonthPercentChangeData,
            'category_property': ECI.hasBenefitsBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData_WorkerType': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData,
            'category_property': ECI.hasWorkerType,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData_BenefitsOccupationalGroup': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData,
            'category_property': ECI.hasBenefitsOccupationalGroup,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData_BenefitsIndustry': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData,
            'category_property': ECI.hasBenefitsIndustry,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData_BenefitsBargainingStatus': {
            'class': ECI.BenefitsByOccupationIndustryBargainingStatusTwelveMonthPercentChangeData,
            'category_property': ECI.hasBenefitsBargainingStatus,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },

        # ============================================
        # Table 13: Total Compensation and Wages/Salaries by Area (Not Seasonally Adjusted)
        # ============================================

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

        # ============================================
        # Table A: Major Series Percent Change
        # ============================================

        'ThreeMonthSeasonallyAdjustedPercentChangeData': {
            'class': ECI.ThreeMonthSeasonallyAdjustedPercentChangeData,
            'category_property': ECI.hasMajorSeriesCompensationComponent,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthNotSeasonallyAdjustedCurrentDollarPercentChangeData': {
            'class': ECI.TwelveMonthNotSeasonallyAdjustedCurrentDollarPercentChangeData,
            'category_property': ECI.hasMajorSeriesCompensationComponent,
            'month_property': ECI.hasMonth,
            'year_property': ECI.hasYear
        },
        'TwelveMonthNotSeasonallyAdjustedConstantDollarPercentChangeData': {
            'class': ECI.TwelveMonthNotSeasonallyAdjustedConstantDollarPercentChangeData,
            'category_property': ECI.hasMajorSeriesCompensationComponent,
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
        'RelativeImportance_EndUse': {
            'class': XIMPIM.RelativeImportance,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Tables 1, 2: PercentChange by End Use Category
        'PercentChange_EndUse': {
            'class': XIMPIM.PercentChange,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Tables 3, 4: Import/Export Price Index by NAICS Industry
        'ImportPriceIndex_NAICS': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasNAICSIndustry,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },
        'ExportPriceIndex_NAICS': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasNAICSIndustry,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Tables 3, 4: RelativeImportance by NAICS Industry
        'RelativeImportance_NAICS': {
            'class': XIMPIM.RelativeImportance,
            'category_property': XIMPIM.hasNAICSIndustry,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Tables 3, 4: PercentChange by NAICS Industry
        'PercentChange_NAICS': {
            'class': XIMPIM.PercentChange,
            'category_property': XIMPIM.hasNAICSIndustry,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Tables 5, 6: Import/Export Price Index by Harmonized Category
        'ImportPriceIndex_Harmonized': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasHarmonizedCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },
        'ExportPriceIndex_Harmonized': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasHarmonizedCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Tables 5, 6: RelativeImportance by Harmonized Category
        'RelativeImportance_Harmonized': {
            'class': XIMPIM.RelativeImportance,
            'category_property': XIMPIM.hasHarmonizedCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Tables 5, 6: PercentChange by Harmonized Category
        'PercentChange_Harmonized': {
            'class': XIMPIM.PercentChange,
            'category_property': XIMPIM.hasHarmonizedCategory,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Table 7: Import Price Index by Locality of Origin
        'ImportPriceIndex_LocalityOfOrigin': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasLocalityOfOrigin,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 7: RelativeImportance by Locality of Origin
        'RelativeImportance_LocalityOfOrigin': {
            'class': XIMPIM.RelativeImportance,
            'category_property': XIMPIM.hasLocalityOfOrigin,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

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
        'RelativeImportance_LocalityOfDestination': {
            'class': XIMPIM.RelativeImportance,
            'category_property': XIMPIM.hasLocalityOfDestination,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

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

        # Table 9: Terms of Trade Percent Change by Locality
        'TermsOfTradePercentChange': {
            'class': XIMPIM.TermsOfTradePercentChange,
            'category_property': XIMPIM.hasLocality,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Table 10: Transportation Price Index by Transportation Service
        'TransportationPriceIndex': {
            'class': XIMPIM.TransportationPriceIndex,
            'category_property': XIMPIM.hasTransportationService,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 10: Transportation Relative Importance by Transportation Service
        'TransportationRelativeImportance': {
            'class': XIMPIM.TransportationRelativeImportance,
            'category_property': XIMPIM.hasTransportationService,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 10: Transportation Percent Change by Transportation Service
        'TransportationPercentChange': {
            'class': XIMPIM.TransportationPercentChange,
            'category_property': XIMPIM.hasTransportationService,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Table A: Monthly Percent Change by Trade Category
        'MonthlyPercentChange': {
            'class': XIMPIM.MonthlyPercentChange,
            'category_property': XIMPIM.hasTradeCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table A: Annual Percent Change by Trade Category
        'AnnualPercentChange': {
            'class': XIMPIM.AnnualPercentChange,
            'category_property': XIMPIM.hasTradeCategory,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
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
        # Table 12: Layoffs Discharges by Industry (hasIndustry)
        # Note: Table 12 uses "LayoffsDischarges" (no "And") class names
        # ============================================
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

        # ============================================
        # Table 12: Layoffs Discharges by Region (hasRegion)
        # ============================================
        'LayoffsDischargesLevel_Region': {
            'class': JOLTS.LayoffsDischargesLevel,
            'category_property': JOLTS.hasRegion,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsDischargesRate_Region': {
            'class': JOLTS.LayoffsDischargesRate,
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

        # ============================================
        # Table 14: All metrics by Establishment Size (hasEstablishmentSize)
        # Note: Table 14 uses "hasEstablishmentSize" (not "hasEstablishmentSizeClass")
        # and "LayoffsDischarges" (not "LayoffsAndDischarges") for L&D classes
        # The Level/Rate classes for JO, Hires, TotalSep, Quits, OtherSep
        # are the same as Tables 1-13, so the inner join on hasEstablishmentSize
        # naturally filters to only Table 14 entities.
        # ============================================
        'JobOpeningsLevel_EstablishmentSize14': {
            'class': JOLTS.JobOpeningsLevel,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'JobOpeningsRate_EstablishmentSize14': {
            'class': JOLTS.JobOpeningsRate,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresLevel_EstablishmentSize14': {
            'class': JOLTS.HiresLevel,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'HiresRate_EstablishmentSize14': {
            'class': JOLTS.HiresRate,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsLevel_EstablishmentSize14': {
            'class': JOLTS.TotalSeparationsLevel,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'TotalSeparationsRate_EstablishmentSize14': {
            'class': JOLTS.TotalSeparationsRate,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsLevel_EstablishmentSize14': {
            'class': JOLTS.QuitsLevel,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'QuitsRate_EstablishmentSize14': {
            'class': JOLTS.QuitsRate,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsDischargesLevel_EstablishmentSize14': {
            'class': JOLTS.LayoffsDischargesLevel,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'LayoffsDischargesRate_EstablishmentSize14': {
            'class': JOLTS.LayoffsDischargesRate,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsLevel_EstablishmentSize14': {
            'class': JOLTS.OtherSeparationsLevel,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
        'OtherSeparationsRate_EstablishmentSize14': {
            'class': JOLTS.OtherSeparationsRate,
            'category_property': JOLTS.hasEstablishmentSize,
            'month_property': JOLTS.hasMonth,
            'year_property': JOLTS.hasYear
        },
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