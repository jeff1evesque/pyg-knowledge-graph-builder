"""
BLS Measurement Types - Temporal linking configurations
"""
from glue_jobs.utils.rdf_utils import CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM


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
    }
}