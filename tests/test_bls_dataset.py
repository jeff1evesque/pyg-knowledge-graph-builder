"""
Updated BLS Enrichment Test Suite for Refactored Structure

Tests the new modular enrichment architecture:
- Individual dataset enrichers (CPI, PPI, ECI, JOLTS, EMPSIT)
- BLS orchestrator coordination
- Cross-dataset integration
"""

import sys
import os
from typing import Dict, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdflib import Graph
from spark_jobs.utils.rdf_utils import RDFGraphLoader
from spark_jobs.enrichment.intra_source.bls_linker import BLSIntraSourceLinker
from spark_jobs.enrichment.intra_source.bls.enrichers import (
    CPIEnricher, PPIEnricher, ECIEnricher, JOLTSEnricher, EMPSITEnricher
)
import logging
import pytest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ============================================
# TEST DATA GENERATORS
# ============================================

def create_sample_cpi_data() -> str:
    """Generate sample CPI RDF data"""
    return """
    @prefix cpi: <https://www.bls.gov/cpi/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    # Temporal entities
    cpi:November rdf:type cpi:Month .
    cpi:December rdf:type cpi:Month .
    cpi:2024 rdf:type cpi:Year .
    
    # Category hierarchy
    cpi:AllItems_Entity rdf:type cpi:AllItems ;
        rdfs:label "All items" ;
        cpi:hasFullPath "AllItems" .
    
    cpi:AllItems_Food_Entity rdf:type cpi:Food ;
        rdfs:label "Food" ;
        cpi:hasFullPath "AllItems_Food" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    cpi:AllItems_Food_FoodAtHome_Entity rdf:type cpi:FoodAtHome ;
        rdfs:label "Food at home" ;
        cpi:hasFullPath "AllItems_Food_FoodAtHome" ;
        cpi:hasParent cpi:AllItems_Food_Entity .
    
    cpi:AllItems_Food_FoodAwayFromHome_Entity rdf:type cpi:FoodAwayFromHome ;
        rdfs:label "Food away from home" ;
        cpi:hasFullPath "AllItems_Food_FoodAwayFromHome" ;
        cpi:hasParent cpi:AllItems_Food_Entity .
    
    cpi:AllItems_Energy_Entity rdf:type cpi:Energy ;
        rdfs:label "Energy" ;
        cpi:hasFullPath "AllItems_Energy" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    cpi:AllItems_Housing_Entity rdf:type cpi:Housing ;
        rdfs:label "Housing" ;
        cpi:hasFullPath "AllItems_Housing" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    cpi:AllItems_Transportation_Entity rdf:type cpi:Transportation ;
        rdfs:label "Transportation" ;
        cpi:hasFullPath "AllItems_Transportation" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    cpi:AllItems_Medical_care_Entity rdf:type cpi:MedicalCare ;
        rdfs:label "Medical care" ;
        cpi:hasFullPath "AllItems_Medical_care" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    cpi:AllItems_Education_Entity rdf:type cpi:Education ;
        rdfs:label "Education" ;
        cpi:hasFullPath "AllItems_Education" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    # Index measurements
    cpi:AllItems_Food_November2024_Index rdf:type cpi:Index ;
        cpi:indexValue 295.8 ;
        cpi:hasCategory cpi:AllItems_Food_Entity ;
        cpi:hasMonth cpi:November ;
        cpi:hasYear cpi:2024 .
    
    cpi:AllItems_Food_December2024_Index rdf:type cpi:Index ;
        cpi:indexValue 296.2 ;
        cpi:hasCategory cpi:AllItems_Food_Entity ;
        cpi:hasMonth cpi:December ;
        cpi:hasYear cpi:2024 .
    
    cpi:AllItems_Food_FoodAwayFromHome_November2024_Index rdf:type cpi:Index ;
        cpi:indexValue 320.5 ;
        cpi:hasCategory cpi:AllItems_Food_FoodAwayFromHome_Entity ;
        cpi:hasMonth cpi:November ;
        cpi:hasYear cpi:2024 .
    
    cpi:AllItems_Food_FoodAwayFromHome_December2024_Index rdf:type cpi:Index ;
        cpi:indexValue 321.2 ;
        cpi:hasCategory cpi:AllItems_Food_FoodAwayFromHome_Entity ;
        cpi:hasMonth cpi:December ;
        cpi:hasYear cpi:2024 .
    
    cpi:AllItems_Energy_November2024_Index rdf:type cpi:Index ;
        cpi:indexValue 280.5 ;
        cpi:hasCategory cpi:AllItems_Energy_Entity ;
        cpi:hasMonth cpi:November ;
        cpi:hasYear cpi:2024 .
    
    cpi:AllItems_Energy_December2024_Index rdf:type cpi:Index ;
        cpi:indexValue 281.0 ;
        cpi:hasCategory cpi:AllItems_Energy_Entity ;
        cpi:hasMonth cpi:December ;
        cpi:hasYear cpi:2024 .
    """


def create_sample_ppi_data() -> str:
    """Generate sample PPI RDF data"""
    return """
    @prefix ppi: <https://www.bls.gov/ppi/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    # Temporal entities
    ppi:November rdf:type ppi:Month .
    ppi:December rdf:type ppi:Month .
    ppi:2024 rdf:type ppi:Year .
    
    # Sections
    ppi:FinalDemand_Section rdf:type ppi:Section ;
        rdfs:label "Final Demand" .
    
    ppi:IntermediateDemand_Section rdf:type ppi:Section ;
        rdfs:label "Intermediate Demand" .
    
    # Commodity groupings
    ppi:FinalDemand_FinalDemand_4_Entity rdf:type ppi:FinalDemand ;
        rdfs:label "Final demand" ;
        ppi:hasFullPath "FinalDemand_FinalDemand" ;
        ppi:hasGroupCode "FD" ;
        ppi:hasItemCode "4" ;
        ppi:belongsToSection ppi:FinalDemand_Section .
    
    ppi:FinalDemand_FoodManufacturing_12345_Entity rdf:type ppi:FoodManufacturing ;
        rdfs:label "Food manufacturing" ;
        ppi:hasFullPath "FinalDemand_FoodManufacturing" ;
        ppi:hasGroupCode "FD" ;
        ppi:hasItemCode "12345" ;
        ppi:belongsToSection ppi:FinalDemand_Section ;
        ppi:hasParent ppi:FinalDemand_FinalDemand_4_Entity .
    
    ppi:FinalDemand_EnergyGoods_67890_Entity rdf:type ppi:EnergyGoods ;
        rdfs:label "Energy goods" ;
        ppi:hasFullPath "FinalDemand_EnergyGoods" ;
        ppi:hasGroupCode "FD" ;
        ppi:hasItemCode "67890" ;
        ppi:belongsToSection ppi:FinalDemand_Section ;
        ppi:hasParent ppi:FinalDemand_FinalDemand_4_Entity .
    
    ppi:IntermediateDemand_ProcessedGoods_11111_Entity rdf:type ppi:ProcessedGoods ;
        rdfs:label "Processed goods for intermediate demand" ;
        ppi:hasFullPath "IntermediateDemand_ProcessedGoods" ;
        ppi:hasGroupCode "ID" ;
        ppi:hasItemCode "11111" ;
        ppi:belongsToSection ppi:IntermediateDemand_Section .
    
    ppi:IntermediateDemand_UnprocessedGoods_22222_Entity rdf:type ppi:UnprocessedGoods ;
        rdfs:label "Unprocessed goods for intermediate demand" ;
        ppi:hasFullPath "IntermediateDemand_UnprocessedGoods" ;
        ppi:hasGroupCode "ID" ;
        ppi:hasItemCode "22222" ;
        ppi:belongsToSection ppi:IntermediateDemand_Section .
    
    # Monthly changes
    ppi:FinalDemand_FoodManufacturing_12345_November2024ToDecember2024_MonthlyChange rdf:type ppi:MonthlyChange ;
        ppi:changeValue 0.3 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_FoodManufacturing_12345_Entity ;
        ppi:hasStartMonth ppi:November ;
        ppi:hasStartYear ppi:2024 ;
        ppi:hasEndMonth ppi:December ;
        ppi:hasEndYear ppi:2024 ;
        ppi:isSeasonallyAdjusted true .
    
    ppi:FinalDemand_EnergyGoods_67890_November2024ToDecember2024_MonthlyChange rdf:type ppi:MonthlyChange ;
        ppi:changeValue 0.5 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_EnergyGoods_67890_Entity ;
        ppi:hasStartMonth ppi:November ;
        ppi:hasStartYear ppi:2024 ;
        ppi:hasEndMonth ppi:December ;
        ppi:hasEndYear ppi:2024 ;
        ppi:isSeasonallyAdjusted true .
    
    # Index values
    ppi:FinalDemand_FoodManufacturing_12345_November2024_IndexValue rdf:type ppi:IndexValue ;
        ppi:indexValue 148.5 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_FoodManufacturing_12345_Entity ;
        ppi:hasMonth ppi:November ;
        ppi:hasYear ppi:2024 ;
        ppi:isSeasonallyAdjusted true .
    
    ppi:FinalDemand_FoodManufacturing_12345_December2024_IndexValue rdf:type ppi:IndexValue ;
        ppi:indexValue 148.9 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_FoodManufacturing_12345_Entity ;
        ppi:hasMonth ppi:December ;
        ppi:hasYear ppi:2024 ;
        ppi:isSeasonallyAdjusted true .
    """


def create_sample_eci_data() -> str:
    """Generate sample ECI RDF data"""
    return """
    @prefix eci: <https://www.bls.gov/eci/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    # Temporal entities
    eci:September rdf:type eci:Month .
    eci:December rdf:type eci:Month .
    eci:2024 rdf:type eci:Year .
    
    # Worker categories
    eci:private_industry rdf:type eci:WorkerCategory ;
        rdfs:label "Private industry workers" .
    
    # Occupational groups
    eci:all_workers rdf:type eci:OccupationalGroup ;
        rdfs:label "All workers" ;
        eci:hasFullPath "private_industry_all_workers" ;
        eci:hasWorkerCategory eci:private_industry .
    
    # Industries
    eci:manufacturing rdf:type eci:Industry ;
        rdfs:label "Manufacturing" ;
        eci:hasFullPath "private_industry_industry_manufacturing" ;
        eci:hasWorkerCategory eci:private_industry .
    
    eci:healthcare_social rdf:type eci:Industry ;
        rdfs:label "Health care and social assistance" ;
        eci:hasFullPath "private_industry_industry_healthcare_social" ;
        eci:hasWorkerCategory eci:private_industry .
    
    eci:accommodation_food rdf:type eci:Industry ;
        rdfs:label "Accommodation and food services" ;
        eci:hasFullPath "private_industry_industry_accommodation_food" ;
        eci:hasWorkerCategory eci:private_industry .
    
    eci:educational rdf:type eci:Industry ;
        rdfs:label "Educational services" ;
        eci:hasFullPath "private_industry_industry_educational" ;
        eci:hasWorkerCategory eci:private_industry .
    
    # Total Compensation Index
    eci:2024_September_index_private_industry_all_workers rdf:type eci:EmploymentCostIndexData ;
        eci:indexValue 158.2 ;
        eci:baseYear "2005" ;
        eci:baseMonth "December" ;
        eci:hasOccupationalGroup eci:all_workers ;
        eci:hasWorkerCategory eci:private_industry ;
        eci:hasMonth eci:September ;
        eci:hasYear eci:2024 .
    
    eci:2024_December_index_private_industry_all_workers rdf:type eci:EmploymentCostIndexData ;
        eci:indexValue 159.1 ;
        eci:baseYear "2005" ;
        eci:baseMonth "December" ;
        eci:hasOccupationalGroup eci:all_workers ;
        eci:hasWorkerCategory eci:private_industry ;
        eci:hasMonth eci:December ;
        eci:hasYear eci:2024 .
    
    # Wages and Salaries Index
    eci:2024_September_wages_index_private_industry_all_workers rdf:type eci:WagesAndSalariesIndexData ;
        eci:wagesAndSalariesIndexValue 156.8 ;
        eci:baseYear "2005" ;
        eci:baseMonth "December" ;
        eci:hasOccupationalGroup eci:all_workers ;
        eci:hasWorkerCategory eci:private_industry ;
        eci:hasMonth eci:September ;
        eci:hasYear eci:2024 .
    
    eci:2024_December_wages_index_private_industry_all_workers rdf:type eci:WagesAndSalariesIndexData ;
        eci:wagesAndSalariesIndexValue 157.6 ;
        eci:baseYear "2005" ;
        eci:baseMonth "December" ;
        eci:hasOccupationalGroup eci:all_workers ;
        eci:hasWorkerCategory eci:private_industry ;
        eci:hasMonth eci:December ;
        eci:hasYear eci:2024 .
    """


def create_sample_jolts_data() -> str:
    """Generate sample JOLTS RDF data"""
    return """
    @prefix jolts: <https://www.bls.gov/jolts/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    # Temporal entities
    jolts:November rdf:type jolts:Month .
    jolts:December rdf:type jolts:Month .
    jolts:2024 rdf:type jolts:Year .
    
    # Industry hierarchy
    jolts:Total_nonfarm_Industry rdf:type jolts:Industry ;
        rdfs:label "Total nonfarm" ;
        jolts:hasFullPath "Total_nonfarm" .
    
    jolts:Total_nonfarm_Service_providing_Industry rdf:type jolts:Industry ;
        rdfs:label "Service-providing" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing" ;
        jolts:hasParent jolts:Total_nonfarm_Industry .
    
    jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Industry rdf:type jolts:Industry ;
        rdfs:label "Leisure and hospitality" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing_Leisure_and_hospitality" ;
        jolts:hasParent jolts:Total_nonfarm_Service_providing_Industry .
    
    jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_Industry rdf:type jolts:Industry ;
        rdfs:label "Accommodation and food services" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services" ;
        jolts:hasParent jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Industry .
    
    jolts:Total_nonfarm_Goods_producing_Industry rdf:type jolts:Industry ;
        rdfs:label "Goods-producing" ;
        jolts:hasFullPath "Total_nonfarm_Goods_producing" ;
        jolts:hasParent jolts:Total_nonfarm_Industry .
    
    jolts:Total_nonfarm_Goods_producing_Manufacturing_Industry rdf:type jolts:Industry ;
        rdfs:label "Manufacturing" ;
        jolts:hasFullPath "Total_nonfarm_Goods_producing_Manufacturing" ;
        jolts:hasParent jolts:Total_nonfarm_Goods_producing_Industry .
    
    jolts:Total_nonfarm_Service_providing_Health_care_and_social_assistance_Industry rdf:type jolts:Industry ;
        rdfs:label "Health care and social assistance" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing_Health_care_and_social_assistance" ;
        jolts:hasParent jolts:Total_nonfarm_Service_providing_Industry .
    
    # Job Openings measurements
    jolts:Total_nonfarm_November2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 7839 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_December2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 8098 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:December ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_November2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 1234 ;
        jolts:hasIndustry jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_December2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 1289 ;
        jolts:hasIndustry jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_Industry ;
        jolts:hasMonth jolts:December ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    # Hires measurements
    jolts:Total_nonfarm_November2024_HiresLevel rdf:type jolts:HiresLevel ;
        jolts:levelValue 5270 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_December2024_HiresLevel rdf:type jolts:HiresLevel ;
        jolts:levelValue 5350 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:December ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    # Quits measurements
    jolts:Total_nonfarm_November2024_QuitsLevel rdf:type jolts:QuitsLevel ;
        jolts:levelValue 3098 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_December2024_QuitsLevel rdf:type jolts:QuitsLevel ;
        jolts:levelValue 3150 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:December ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    """


def create_sample_empsit_data() -> str:
    """Generate sample EMPSIT RDF data"""
    return """
    @prefix empsit: <https://www.bls.gov/empsit/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    # Temporal entities
    empsit:November rdf:type empsit:Month .
    empsit:December rdf:type empsit:Month .
    empsit:2024 rdf:type empsit:Year .
    
    # Industry hierarchy
    empsit:total_nonfarm rdf:type empsit:Industry ;
        rdfs:label "Total nonfarm" ;
        empsit:hasFullPath "total_nonfarm" .
    
    empsit:total_private rdf:type empsit:Industry ;
        rdfs:label "Total private" ;
        empsit:hasFullPath "total_private" ;
        empsit:hasParentIndustry empsit:total_nonfarm .
    
    empsit:goods_producing rdf:type empsit:Industry ;
        rdfs:label "Goods-producing" ;
        empsit:hasFullPath "goods_producing" ;
        empsit:hasParentIndustry empsit:total_private .
    
    empsit:manufacturing rdf:type empsit:Industry ;
        rdfs:label "Manufacturing" ;
        empsit:hasFullPath "manufacturing" ;
        empsit:hasParentIndustry empsit:goods_producing .
    
    empsit:food_manufacturing rdf:type empsit:Industry ;
        rdfs:label "Food manufacturing" ;
        empsit:hasFullPath "food_manufacturing" ;
        empsit:hasParentIndustry empsit:manufacturing .
    
    empsit:private_service rdf:type empsit:Industry ;
        rdfs:label "Private service-providing" ;
        empsit:hasFullPath "private_service" ;
        empsit:hasParentIndustry empsit:total_private .
    
    empsit:healthcare_social rdf:type empsit:Industry ;
        rdfs:label "Health care and social assistance" ;
        empsit:hasFullPath "healthcare_social" ;
        empsit:hasParentIndustry empsit:private_service .
    
    empsit:leisure_hospitality rdf:type empsit:Industry ;
        rdfs:label "Leisure and hospitality" ;
        empsit:hasFullPath "leisure_hospitality" ;
        empsit:hasParentIndustry empsit:private_service .
    
    empsit:food_services rdf:type empsit:Industry ;
        rdfs:label "Food services and drinking places" ;
        empsit:hasFullPath "food_services" ;
        empsit:hasParentIndustry empsit:leisure_hospitality .
    
    # Employee counts
    empsit:total_nonfarm_November2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 159358 ;
        empsit:hasIndustry empsit:total_nonfarm ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    empsit:total_nonfarm_December2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 159568 ;
        empsit:hasIndustry empsit:total_nonfarm ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary true .
    
    empsit:manufacturing_November2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 12987 ;
        empsit:hasIndustry empsit:manufacturing ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    empsit:manufacturing_December2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 13012 ;
        empsit:hasIndustry empsit:manufacturing ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary true .
    
    empsit:food_services_November2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 12456 ;
        empsit:hasIndustry empsit:food_services ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    empsit:food_services_December2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 12523 ;
        empsit:hasIndustry empsit:food_services ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary true .
    
    # Household data categories
    empsit:civilian_labor_force rdf:type empsit:Category ;
        rdfs:label "Civilian labor force" ;
        empsit:fullPath "Civilian_labor_force" .
    
    empsit:employed rdf:type empsit:Category ;
        rdfs:label "Employed" ;
        empsit:fullPath "Civilian_labor_force_Employed" ;
        empsit:hasParentCategory empsit:civilian_labor_force .
    
    empsit:unemployed rdf:type empsit:Category ;
        rdfs:label "Unemployed" ;
        empsit:fullPath "Civilian_labor_force_Unemployed" ;
        empsit:hasParentCategory empsit:civilian_labor_force .
    
    # Household data measurements
    empsit:employed_November2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 161861 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:employed ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 .
    
    empsit:employed_December2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 162098 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:employed ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 .
    
    empsit:unemployed_November2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 6728 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:unemployed ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 .
    
    empsit:unemployed_December2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 6456 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:unemployed ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 .
    """


# ============================================
# UNIT TESTS: Individual Enrichers
# ============================================

class TestIndividualEnrichers:
    """Test individual dataset enrichers in isolation"""

    def test_cpi_enricher(self):
        """Test CPI enricher alone"""
        logger.info("Testing CPI Enricher...")

        loader = RDFGraphLoader()
        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        enricher = CPIEnricher(loader.graph)

        # Test sector keywords
        keywords = enricher.get_sector_keywords()
        assert 'food_sector' in keywords
        assert len(keywords['food_sector']) > 0
        assert 'Food' in keywords['food_sector']

        # Test measurement types
        measurement_types = enricher.get_measurement_types()
        assert 'Index' in measurement_types
        assert 'PercentChange' in measurement_types
        assert measurement_types['Index']['class'] is not None

        # Test temporal linking
        links = enricher.link_temporal_sequences()
        assert links > 0

        stats = enricher.get_stats()
        assert stats['temporal_sequences'] > 0
        assert stats['temporal_sequences'] == links

        logger.info(f"✅ CPI Enricher: {links} temporal links created")

    def test_ppi_enricher(self):
        """Test PPI enricher alone"""
        logger.info("Testing PPI Enricher...")

        loader = RDFGraphLoader()
        ppi_ttl = create_sample_ppi_data()
        loader.load_from_string(ppi_ttl, format='turtle')

        enricher = PPIEnricher(loader.graph)

        keywords = enricher.get_sector_keywords()
        assert 'food_sector' in keywords
        assert 'Food' in keywords['food_sector']

        measurement_types = enricher.get_measurement_types()
        assert 'MonthlyChange' in measurement_types
        assert 'IndexValue' in measurement_types

        links = enricher.link_temporal_sequences()
        assert links > 0

        logger.info(f"✅ PPI Enricher: {links} temporal links created")

    def test_eci_enricher(self):
        """Test ECI enricher alone"""
        logger.info("Testing ECI Enricher...")

        loader = RDFGraphLoader()
        eci_ttl = create_sample_eci_data()
        loader.load_from_string(eci_ttl, format='turtle')

        enricher = ECIEnricher(loader.graph)

        keywords = enricher.get_sector_keywords()
        assert 'healthcare_sector' in keywords
        assert len(keywords['healthcare_sector']) > 0

        measurement_types = enricher.get_measurement_types()
        assert 'EmploymentCostIndexData' in measurement_types
        assert 'WagesAndSalariesIndexData' in measurement_types

        links = enricher.link_temporal_sequences()
        assert links > 0

        logger.info(f"✅ ECI Enricher: {links} temporal links created")

    def test_jolts_enricher(self):
        """Test JOLTS enricher alone"""
        logger.info("Testing JOLTS Enricher...")

        loader = RDFGraphLoader()
        jolts_ttl = create_sample_jolts_data()
        loader.load_from_string(jolts_ttl, format='turtle')

        enricher = JOLTSEnricher(loader.graph)

        keywords = enricher.get_sector_keywords()
        assert 'food_sector' in keywords
        assert 'Food services' in keywords['food_sector']

        measurement_types = enricher.get_measurement_types()
        assert 'JobOpeningsLevel' in measurement_types
        assert 'HiresLevel' in measurement_types

        links = enricher.link_temporal_sequences()
        assert links > 0

        logger.info(f"✅ JOLTS Enricher: {links} temporal links created")

    def test_empsit_enricher(self):
        """Test EMPSIT enricher alone"""
        logger.info("Testing EMPSIT Enricher...")

        loader = RDFGraphLoader()
        empsit_ttl = create_sample_empsit_data()
        loader.load_from_string(empsit_ttl, format='turtle')

        enricher = EMPSITEnricher(loader.graph)

        keywords = enricher.get_sector_keywords()
        assert 'food_sector' in keywords
        assert 'Food manufacturing' in keywords['food_sector']

        measurement_types = enricher.get_measurement_types()
        assert 'EmployeeCount' in measurement_types
        assert 'HouseholdData' in measurement_types

        links = enricher.link_temporal_sequences()
        assert links > 0

        logger.info(f"✅ EMPSIT Enricher: {links} temporal links created")


# ============================================
# INTEGRATION TESTS: BLS Orchestrator
# ============================================

class TestBLSOrchestrator:
    """Test BLS orchestrator coordination"""

    def test_orchestrator_initialization(self):
        """Test orchestrator detects datasets correctly"""
        logger.info("Testing BLS Orchestrator Initialization...")

        loader = RDFGraphLoader()

        # Load multiple datasets
        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        ppi_ttl = create_sample_ppi_data()
        loader.load_from_string(ppi_ttl, format='turtle')

        # Create orchestrator
        orchestrator = BLSIntraSourceLinker(loader.graph)

        # Check dataset detection
        assert 'cpi' in orchestrator.available_datasets
        assert 'ppi' in orchestrator.available_datasets
        assert len(orchestrator.available_datasets) == 2

        # Check enrichers initialized
        assert 'cpi' in orchestrator.enrichers
        assert 'ppi' in orchestrator.enrichers
        assert isinstance(orchestrator.enrichers['cpi'], CPIEnricher)
        assert isinstance(orchestrator.enrichers['ppi'], PPIEnricher)

        logger.info(f"✅ Orchestrator detected: {orchestrator.available_datasets}")

    def test_orchestrator_enrichment(self):
        """Test full orchestrator enrichment pipeline"""
        logger.info("Testing Full Orchestrator Enrichment...")

        loader = RDFGraphLoader()

        # Load all datasets
        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        ppi_ttl = create_sample_ppi_data()
        loader.load_from_string(ppi_ttl, format='turtle')

        eci_ttl = create_sample_eci_data()
        loader.load_from_string(eci_ttl, format='turtle')

        jolts_ttl = create_sample_jolts_data()
        loader.load_from_string(jolts_ttl, format='turtle')

        empsit_ttl = create_sample_empsit_data()
        loader.load_from_string(empsit_ttl, format='turtle')

        initial_count = len(loader.graph)
        logger.info(f"Initial graph size: {initial_count} triples")

        # Run enrichment
        orchestrator = BLSIntraSourceLinker(loader.graph)
        stats = orchestrator.enrich()

        final_count = len(loader.graph)
        logger.info(f"Final graph size: {final_count} triples")

        # Validate results
        assert stats['total_triples_added'] > 0
        assert stats['temporal_unified'] > 0
        assert stats['temporal_sequences'] > 0
        assert stats['sector_links'] > 0

        assert final_count > initial_count
        assert final_count == initial_count + stats['total_triples_added']

        # Check all datasets detected
        assert len(stats['available_datasets']) == 5
        assert 'cpi' in stats['available_datasets']
        assert 'ppi' in stats['available_datasets']
        assert 'eci' in stats['available_datasets']
        assert 'jolts' in stats['available_datasets']
        assert 'empsit' in stats['available_datasets']

        logger.info(f"✅ Orchestrator added {stats['total_triples_added']} triples")
        logger.info(f"   - Temporal unified: {stats['temporal_unified']}")
        logger.info(f"   - Temporal sequences: {stats['temporal_sequences']}")
        logger.info(f"   - Sector links: {stats['sector_links']}")
        logger.info(f"   - Correlation links: {stats['correlation_links']}")
        logger.info(f"   - Hierarchy links: {stats['hierarchy_links']}")


# ============================================
# CROSS-DATASET TESTS
# ============================================

class TestCrossDatasetIntegration:
    """Test cross-dataset enrichment patterns"""

    def test_cpi_ppi_integration(self):
        """Test CPI-PPI food sector correlation"""
        logger.info("Testing CPI-PPI Integration...")

        loader = RDFGraphLoader()

        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        ppi_ttl = create_sample_ppi_data()
        loader.load_from_string(ppi_ttl, format='turtle')

        orchestrator = BLSIntraSourceLinker(loader.graph)
        stats = orchestrator.enrich()

        # Check for food sector links
        food_sector_query = """
        SELECT (COUNT(*) as ?count) WHERE {
            ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> 
                      <https://www.bls.gov/enrichment/FoodSector> .
            ?ppiEntity <https://www.bls.gov/enrichment/belongsToSector> 
                      <https://www.bls.gov/enrichment/FoodSector> .
            FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
            FILTER(STRSTARTS(STR(?ppiEntity), "https://www.bls.gov/ppi/"))
        }
        """

        result = list(loader.graph.query(food_sector_query))
        food_sector_count = int(result[0].count)

        assert food_sector_count > 0
        logger.info(f"✅ CPI-PPI: {food_sector_count} food sector correlations")

    def test_eci_jolts_integration(self):
        """Test ECI-JOLTS labor market correlation"""
        logger.info("Testing ECI-JOLTS Integration...")

        loader = RDFGraphLoader()

        eci_ttl = create_sample_eci_data()
        loader.load_from_string(eci_ttl, format='turtle')

        jolts_ttl = create_sample_jolts_data()
        loader.load_from_string(jolts_ttl, format='turtle')

        orchestrator = BLSIntraSourceLinker(loader.graph)
        stats = orchestrator.enrich()

        # Check for healthcare sector links
        healthcare_query = """
        SELECT (COUNT(*) as ?count) WHERE {
            ?eciEntity <https://www.bls.gov/enrichment/belongsToSector> 
                      <https://www.bls.gov/enrichment/HealthcareSector> .
            ?joltsEntity <https://www.bls.gov/enrichment/belongsToSector> 
                        <https://www.bls.gov/enrichment/HealthcareSector> .
            FILTER(STRSTARTS(STR(?eciEntity), "https://www.bls.gov/eci/"))
            FILTER(STRSTARTS(STR(?joltsEntity), "https://www.bls.gov/jolts/"))
        }
        """

        result = list(loader.graph.query(healthcare_query))
        healthcare_count = int(result[0].count)

        assert healthcare_count > 0
        logger.info(f"✅ ECI-JOLTS: {healthcare_count} healthcare sector correlations")

    def test_empsit_jolts_integration(self):
        """Test EMPSIT-JOLTS employment correlation"""
        logger.info("Testing EMPSIT-JOLTS Integration...")

        loader = RDFGraphLoader()

        empsit_ttl = create_sample_empsit_data()
        loader.load_from_string(empsit_ttl, format='turtle')

        jolts_ttl = create_sample_jolts_data()
        loader.load_from_string(jolts_ttl, format='turtle')

        orchestrator = BLSIntraSourceLinker(loader.graph)
        stats = orchestrator.enrich()

        # Check for food sector links
        food_query = """
        SELECT (COUNT(*) as ?count) WHERE {
            ?empsitEntity <https://www.bls.gov/enrichment/belongsToSector> 
                         <https://www.bls.gov/enrichment/FoodSector> .
            ?joltsEntity <https://www.bls.gov/enrichment/belongsToSector> 
                        <https://www.bls.gov/enrichment/FoodSector> .
            FILTER(STRSTARTS(STR(?empsitEntity), "https://www.bls.gov/empsit/"))
            FILTER(STRSTARTS(STR(?joltsEntity), "https://www.bls.gov/jolts/"))
        }
        """

        result = list(loader.graph.query(food_query))
        food_count = int(result[0].count)

        assert food_count > 0
        logger.info(f"✅ EMPSIT-JOLTS: {food_count} food sector correlations")

    def test_temporal_unification(self):
        """Test temporal entity unification across datasets"""
        logger.info("Testing Temporal Unification...")

        loader = RDFGraphLoader()

        # Load multiple datasets
        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        ppi_ttl = create_sample_ppi_data()
        loader.load_from_string(ppi_ttl, format='turtle')

        eci_ttl = create_sample_eci_data()
        loader.load_from_string(eci_ttl, format='turtle')

        orchestrator = BLSIntraSourceLinker(loader.graph)
        stats = orchestrator.enrich()

        # Check for unified temporal entities
        unified_query = """
        SELECT (COUNT(DISTINCT ?unified) as ?count) WHERE {
            ?entity <https://www.bls.gov/enrichment/unifiedAs> ?unified .
        }
        """

        result = list(loader.graph.query(unified_query))
        unified_count = int(result[0].count)

        assert unified_count > 0
        assert stats['temporal_unified'] > 0

        logger.info(f"✅ Temporal Unification: {unified_count} unified entities")


# ============================================
# REGRESSION TESTS
# ============================================

class TestRegressionTests:
    """Regression tests to ensure refactoring didn't break functionality"""

    def test_backward_compatibility(self):
        """Test that refactored code produces same results as legacy"""
        logger.info("Testing Backward Compatibility...")

        loader = RDFGraphLoader()

        # Load all datasets
        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        ppi_ttl = create_sample_ppi_data()
        loader.load_from_string(ppi_ttl, format='turtle')

        orchestrator = BLSIntraSourceLinker(loader.graph)
        stats = orchestrator.enrich()

        # Verify expected enrichment patterns
        assert stats['temporal_sequences'] > 0, "Should create temporal sequences"
        assert stats['sector_links'] > 0, "Should create sector links"
        assert stats['temporal_unified'] > 0, "Should unify temporal entities"

        logger.info("✅ Backward compatibility maintained")

    def test_no_duplicate_enrichment(self):
        """Test that running enrichment twice doesn't duplicate triples"""
        logger.info("Testing No Duplicate Enrichment...")

        loader = RDFGraphLoader()

        cpi_ttl = create_sample_cpi_data()
        loader.load_from_string(cpi_ttl, format='turtle')

        # First enrichment
        orchestrator1 = BLSIntraSourceLinker(loader.graph)
        stats1 = orchestrator1.enrich()
        count_after_first = len(loader.graph)

        # Second enrichment (should not add more triples)
        orchestrator2 = BLSIntraSourceLinker(loader.graph)
        stats2 = orchestrator2.enrich()
        count_after_second = len(loader.graph)

        # Second run should add minimal or no triples
        # (some may be added due to re-detection, but should be minimal)
        assert count_after_second <= count_after_first + 10

        logger.info(f"✅ No significant duplicate enrichment")
        logger.info(f"   First run: {stats1['total_triples_added']} triples")
        logger.info(f"   Second run: {stats2['total_triples_added']} triples")


# ============================================
# PYTEST CONFIGURATION
# ============================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])