"""
BLS Measurement Types - Temporal linking configurations
"""
from glue_jobs.utils.rdf_utils import (
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
    },
    'ximpim': {
        # Table 1: Import Price Index by End Use
        'ImportPriceIndex_EndUse': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 2: Export Price Index by End Use
        'ExportPriceIndex_EndUse': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasEndUseCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 3: Import Price Index by NAICS
        'ImportPriceIndex_NAICS': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasNAICSIndustry,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 4: Export Price Index by NAICS
        'ExportPriceIndex_NAICS': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasNAICSIndustry,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 5: Import Price Index by Harmonized System
        'ImportPriceIndex_Harmonized': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasHarmonizedCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 6: Export Price Index by Harmonized System
        'ExportPriceIndex_Harmonized': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasHarmonizedCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 7: Import Price Index by Locality of Origin
        'ImportPriceIndex_Locality': {
            'class': XIMPIM.ImportPriceIndex,
            'category_property': XIMPIM.hasLocalityOfOrigin,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 8: Export Price Index by Locality of Destination
        'ExportPriceIndex_Destination': {
            'class': XIMPIM.ExportPriceIndex,
            'category_property': XIMPIM.hasLocalityOfDestination,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 9: Terms of Trade Index
        'TermsOfTradeIndex': {
            'class': XIMPIM.TermsOfTradeIndex,
            'category_property': XIMPIM.hasLocality,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table 10: Transportation Price Index
        'TransportationPriceIndex': {
            'class': XIMPIM.TransportationPriceIndex,
            'category_property': XIMPIM.hasTransportationService,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Relative Importance (all tables)
        'RelativeImportance': {
            'class': XIMPIM.RelativeImportance,
            'category_property': None,  # Multiple possible properties
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Percent Changes (all tables)
        'PercentChange': {
            'class': XIMPIM.PercentChange,
            'category_property': None,  # Multiple possible properties
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },

        # Table A: Monthly Percent Change
        'MonthlyPercentChange': {
            'class': XIMPIM.MonthlyPercentChange,
            'category_property': XIMPIM.hasTradeCategory,
            'month_property': XIMPIM.hasMonth,
            'year_property': XIMPIM.hasYear
        },

        # Table A: Annual Percent Change
        'AnnualPercentChange': {
            'class': XIMPIM.AnnualPercentChange,
            'category_property': XIMPIM.hasTradeCategory,
            'month_property': XIMPIM.hasEndMonth,
            'year_property': XIMPIM.hasEndYear
        },
    },
    'laus': {
        # Table 1: Labor Force Data
        'LaborForceData': {
            'class': LAUS.LaborForceData,
            'category_property': LAUS.hasState,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },
        # Table 2: Employment Data
        'EmploymentData': {
            'class': LAUS.EmploymentData,
            'category_property': LAUS.hasState,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },
        # Table 3: Unemployment Data
        'UnemploymentData': {
            'class': LAUS.UnemploymentData,
            'category_property': LAUS.hasState,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },
        # Unemployment Rate
        'UnemploymentRate': {
            'class': LAUS.UnemploymentRate,
            'category_property': LAUS.hasState,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },
        # Industry Employment by State
        'IndustryEmploymentData': {
            'class': LAUS.IndustryEmploymentData,
            'category_property': LAUS.hasIndustry,
            'month_property': LAUS.hasMonth,
            'year_property': LAUS.hasYear
        },
    },

    'metro': {
        # Table 1: Civilian Labor Force - States and Selected Areas (Not Seasonally Adjusted)
        'CivilianLaborForceData': {
            'class': METRO.CivilianLaborForceData,
            'category_property': METRO.hasMetropolitanArea,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 1: Unemployment Data - States and Selected Areas (Not Seasonally Adjusted)
        'UnemploymentData': {
            'class': METRO.UnemploymentData,
            'category_property': METRO.hasMetropolitanArea,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 1: Unemployment Rate - States and Selected Areas (Not Seasonally Adjusted)
        'UnemploymentRate': {
            'class': METRO.UnemploymentRate,
            'category_property': METRO.hasMetropolitanArea,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 2: Civilian Labor Force - Large Metropolitan Divisions (Not Seasonally Adjusted)
        'CivilianLaborForceDivisionData': {
            'class': METRO.CivilianLaborForceDivisionData,
            'category_property': METRO.hasMetropolitanDivision,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 2: Unemployment Data - Large Metropolitan Divisions (Not Seasonally Adjusted)
        'UnemploymentDivisionData': {
            'class': METRO.UnemploymentDivisionData,
            'category_property': METRO.hasMetropolitanDivision,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 2: Unemployment Rate - Large Metropolitan Divisions (Not Seasonally Adjusted)
        'UnemploymentDivisionRate': {
            'class': METRO.UnemploymentDivisionRate,
            'category_property': METRO.hasMetropolitanDivision,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 3: Civilian Labor Force - States and Selected Areas (Seasonally Adjusted)
        'CivilianLaborForceSeasonallyAdjustedData': {
            'class': METRO.CivilianLaborForceSeasonallyAdjustedData,
            'category_property': METRO.hasMetropolitanArea,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 3: Unemployment Data - States and Selected Areas (Seasonally Adjusted)
        'UnemploymentSeasonallyAdjustedData': {
            'class': METRO.UnemploymentSeasonallyAdjustedData,
            'category_property': METRO.hasMetropolitanArea,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 3: Unemployment Rate - States and Selected Areas (Seasonally Adjusted)
        'UnemploymentSeasonallyAdjustedRate': {
            'class': METRO.UnemploymentSeasonallyAdjustedRate,
            'category_property': METRO.hasMetropolitanArea,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 4: Civilian Labor Force - Large Metropolitan Divisions (Seasonally Adjusted)
        'CivilianLaborForceDivisionSeasonallyAdjustedData': {
            'class': METRO.CivilianLaborForceDivisionSeasonallyAdjustedData,
            'category_property': METRO.hasMetropolitanDivision,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 4: Unemployment Data - Large Metropolitan Divisions (Seasonally Adjusted)
        'UnemploymentDivisionSeasonallyAdjustedData': {
            'class': METRO.UnemploymentDivisionSeasonallyAdjustedData,
            'category_property': METRO.hasMetropolitanDivision,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },

        # Table 4: Unemployment Rate - Large Metropolitan Divisions (Seasonally Adjusted)
        'UnemploymentDivisionSeasonallyAdjustedRate': {
            'class': METRO.UnemploymentDivisionSeasonallyAdjustedRate,
            'category_property': METRO.hasMetropolitanDivision,
            'month_property': METRO.hasMonth,
            'year_property': METRO.hasYear
        },
    },

    'realer': {
        # Table A-1: Real Earnings - CPI-U (All Urban Consumers)
        'RealAverageHourlyEarnings_CPIU': {
            'class': REALER.RealAverageHourlyEarnings_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'RealAverageWeeklyEarnings_CPIU': {
            'class': REALER.RealAverageWeeklyEarnings_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'ConsumerPriceIndex_CPIU': {
            'class': REALER.ConsumerPriceIndex_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageHourlyEarnings_CPIU': {
            'class': REALER.AverageHourlyEarnings_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyHours_CPIU': {
            'class': REALER.AverageWeeklyHours_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyEarnings_CPIU': {
            'class': REALER.AverageWeeklyEarnings_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },

        # Table A-1: Over-the-Month Percent Change
        'RealAverageHourlyEarningsMonthlyChange_CPIU': {
            'class': REALER.RealAverageHourlyEarningsMonthlyChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'RealAverageWeeklyEarningsMonthlyChange_CPIU': {
            'class': REALER.RealAverageWeeklyEarningsMonthlyChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'ConsumerPriceIndexMonthlyChange_CPIU': {
            'class': REALER.ConsumerPriceIndexMonthlyChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageHourlyEarningsMonthlyChange_CPIU': {
            'class': REALER.AverageHourlyEarningsMonthlyChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyHoursMonthlyChange_CPIU': {
            'class': REALER.AverageWeeklyHoursMonthlyChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyEarningsMonthlyChange_CPIU': {
            'class': REALER.AverageWeeklyEarningsMonthlyChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },

        # Table A-1: Over-the-Year Percent Change
        'RealAverageHourlyEarningsAnnualChange_CPIU': {
            'class': REALER.RealAverageHourlyEarningsAnnualChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'RealAverageWeeklyEarningsAnnualChange_CPIU': {
            'class': REALER.RealAverageWeeklyEarningsAnnualChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'ConsumerPriceIndexAnnualChange_CPIU': {
            'class': REALER.ConsumerPriceIndexAnnualChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageHourlyEarningsAnnualChange_CPIU': {
            'class': REALER.AverageHourlyEarningsAnnualChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyHoursAnnualChange_CPIU': {
            'class': REALER.AverageWeeklyHoursAnnualChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyEarningsAnnualChange_CPIU': {
            'class': REALER.AverageWeeklyEarningsAnnualChange_CPIU,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },

        # Table A-2: Real Earnings - CPI-W (Urban Wage Earners and Clerical Workers)
        'RealAverageHourlyEarnings_CPIW': {
            'class': REALER.RealAverageHourlyEarnings_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'RealAverageWeeklyEarnings_CPIW': {
            'class': REALER.RealAverageWeeklyEarnings_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'ConsumerPriceIndex_CPIW': {
            'class': REALER.ConsumerPriceIndex_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageHourlyEarnings_CPIW': {
            'class': REALER.AverageHourlyEarnings_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyHours_CPIW': {
            'class': REALER.AverageWeeklyHours_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyEarnings_CPIW': {
            'class': REALER.AverageWeeklyEarnings_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },

        # Table A-2: Over-the-Month Percent Change
        'RealAverageHourlyEarningsMonthlyChange_CPIW': {
            'class': REALER.RealAverageHourlyEarningsMonthlyChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'RealAverageWeeklyEarningsMonthlyChange_CPIW': {
            'class': REALER.RealAverageWeeklyEarningsMonthlyChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'ConsumerPriceIndexMonthlyChange_CPIW': {
            'class': REALER.ConsumerPriceIndexMonthlyChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageHourlyEarningsMonthlyChange_CPIW': {
            'class': REALER.AverageHourlyEarningsMonthlyChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyHoursMonthlyChange_CPIW': {
            'class': REALER.AverageWeeklyHoursMonthlyChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyEarningsMonthlyChange_CPIW': {
            'class': REALER.AverageWeeklyEarningsMonthlyChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },

        # Table A-2: Over-the-Year Percent Change
        'RealAverageHourlyEarningsAnnualChange_CPIW': {
            'class': REALER.RealAverageHourlyEarningsAnnualChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'RealAverageWeeklyEarningsAnnualChange_CPIW': {
            'class': REALER.RealAverageWeeklyEarningsAnnualChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'ConsumerPriceIndexAnnualChange_CPIW': {
            'class': REALER.ConsumerPriceIndexAnnualChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageHourlyEarningsAnnualChange_CPIW': {
            'class': REALER.AverageHourlyEarningsAnnualChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyHoursAnnualChange_CPIW': {
            'class': REALER.AverageWeeklyHoursAnnualChange_CPIW,
            'category_property': REALER.hasCategory,
            'month_property': REALER.hasMonth,
            'year_property': REALER.hasYear
        },
        'AverageWeeklyEarningsAnnualChange_CPIW': {
            'class': REALER.AverageWeeklyEarningsAnnualChange_CPIW,
            'category_property': REALER.hasCategory,
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