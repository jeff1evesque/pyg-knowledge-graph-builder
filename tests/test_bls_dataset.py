"""
Unified BLS Dataset Test Suite

This comprehensive test suite validates:
1. Individual dataset enrichment (CPI, PPI, JOLTS, EMPSIT)
2. Pairwise dataset integration (CPI+PPI, JOLTS+CPI, JOLTS+PPI, EMPSIT+CPI, EMPSIT+PPI, EMPSIT+JOLTS)
3. Full multi-dataset integration (CPI+PPI+JOLTS+EMPSIT)

Usage:
    # Test individual datasets
    python tests/test_bls_dataset.py --dataset cpi
    python tests/test_bls_dataset.py --dataset ppi
    python tests/test_bls_dataset.py --dataset jolts
    python tests/test_bls_dataset.py --dataset empsit

    # Test dataset pairs
    python tests/test_bls_dataset.py --combination cpi-ppi
    python tests/test_bls_dataset.py --combination jolts-cpi
    python tests/test_bls_dataset.py --combination jolts-ppi
    python tests/test_bls_dataset.py --combination empsit-cpi
    python tests/test_bls_dataset.py --combination empsit-ppi
    python tests/test_bls_dataset.py --combination empsit-jolts

    # Test all datasets together
    python tests/test_bls_dataset.py --combination all

    # Run all tests
    python tests/test_bls_dataset.py --run-all
"""

import sys
import os
from typing import Dict, Tuple, Optional, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdflib import Graph
from glue_jobs.utils.rdf_utils import RDFGraphLoader, RDFGraphWriter
from glue_jobs.enrichment.intra_source_linker import BLSIntraSourceLinker
import logging
import argparse
from dataclasses import dataclass
from enum import Enum

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
    
    # Category hierarchy: All Items > Food > Food at Home
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
    
    # Energy category
    cpi:AllItems_Energy_Entity rdf:type cpi:Energy ;
        rdfs:label "Energy" ;
        cpi:hasFullPath "AllItems_Energy" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    # Housing category
    cpi:AllItems_Housing_Entity rdf:type cpi:Housing ;
        rdfs:label "Housing" ;
        cpi:hasFullPath "AllItems_Housing" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    # Transportation category
    cpi:AllItems_Transportation_Entity rdf:type cpi:Transportation ;
        rdfs:label "Transportation" ;
        cpi:hasFullPath "AllItems_Transportation" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    # Medical care category
    cpi:AllItems_Medical_care_Entity rdf:type cpi:MedicalCare ;
        rdfs:label "Medical care" ;
        cpi:hasFullPath "AllItems_Medical_care" ;
        cpi:hasParent cpi:AllItems_Entity .
    
    # Index measurements for Food (November 2024)
    cpi:AllItems_Food_November2024_Index rdf:type cpi:Index ;
        cpi:indexValue 295.8 ;
        cpi:hasCategory cpi:AllItems_Food_Entity ;
        cpi:hasMonth cpi:November ;
        cpi:hasYear cpi:2024 .
    
    # Index measurements for Food (December 2024)
    cpi:AllItems_Food_December2024_Index rdf:type cpi:Index ;
        cpi:indexValue 296.2 ;
        cpi:hasCategory cpi:AllItems_Food_Entity ;
        cpi:hasMonth cpi:December ;
        cpi:hasYear cpi:2024 .
    
    # Index measurements for Food Away From Home (November 2024)
    cpi:AllItems_Food_FoodAwayFromHome_November2024_Index rdf:type cpi:Index ;
        cpi:indexValue 320.5 ;
        cpi:hasCategory cpi:AllItems_Food_FoodAwayFromHome_Entity ;
        cpi:hasMonth cpi:November ;
        cpi:hasYear cpi:2024 .
    
    # Index measurements for Food Away From Home (December 2024)
    cpi:AllItems_Food_FoodAwayFromHome_December2024_Index rdf:type cpi:Index ;
        cpi:indexValue 321.2 ;
        cpi:hasCategory cpi:AllItems_Food_FoodAwayFromHome_Entity ;
        cpi:hasMonth cpi:December ;
        cpi:hasYear cpi:2024 .
    
    # Index measurements for Energy (November 2024)
    cpi:AllItems_Energy_November2024_Index rdf:type cpi:Index ;
        cpi:indexValue 280.5 ;
        cpi:hasCategory cpi:AllItems_Energy_Entity ;
        cpi:hasMonth cpi:November ;
        cpi:hasYear cpi:2024 .
    
    # Index measurements for Energy (December 2024)
    cpi:AllItems_Energy_December2024_Index rdf:type cpi:Index ;
        cpi:indexValue 281.0 ;
        cpi:hasCategory cpi:AllItems_Energy_Entity ;
        cpi:hasMonth cpi:December ;
        cpi:hasYear cpi:2024 .
    
    # Percent change measurements
    cpi:AllItems_Food_November2024ToDecember2024_SeasonallyAdjusted_PercentChange rdf:type cpi:PercentChange ;
        cpi:changeValue 0.4 ;
        cpi:hasCategory cpi:AllItems_Food_Entity ;
        cpi:hasStartMonth cpi:November ;
        cpi:hasStartYear cpi:2024 ;
        cpi:hasEndMonth cpi:December ;
        cpi:hasEndYear cpi:2024 ;
        cpi:isSeasonallyAdjusted true .
    
    # Relative importance
    cpi:AllItems_Food_November2024_RelativeImportance rdf:type cpi:RelativeImportance ;
        cpi:relativeImportanceValue 13.5 ;
        cpi:hasCategory cpi:AllItems_Food_Entity ;
        cpi:hasMonth cpi:November ;
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
    
    # Final Demand commodity groupings
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
    
    # Intermediate Demand commodity groupings
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
    
    # Monthly changes for Food Manufacturing (November to December 2024)
    ppi:FinalDemand_FoodManufacturing_12345_November2024ToDecember2024_MonthlyChange rdf:type ppi:MonthlyChange ;
        ppi:changeValue 0.3 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_FoodManufacturing_12345_Entity ;
        ppi:hasStartMonth ppi:November ;
        ppi:hasStartYear ppi:2024 ;
        ppi:hasEndMonth ppi:December ;
        ppi:hasEndYear ppi:2024 ;
        ppi:isSeasonallyAdjusted true .
    
    # Monthly changes for Energy Goods (November to December 2024)
    ppi:FinalDemand_EnergyGoods_67890_November2024ToDecember2024_MonthlyChange rdf:type ppi:MonthlyChange ;
        ppi:changeValue 0.5 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_EnergyGoods_67890_Entity ;
        ppi:hasStartMonth ppi:November ;
        ppi:hasStartYear ppi:2024 ;
        ppi:hasEndMonth ppi:December ;
        ppi:hasEndYear ppi:2024 ;
        ppi:isSeasonallyAdjusted true .
    
    # Twelve month changes
    ppi:FinalDemand_FoodManufacturing_12345_2023ToDecember2024_TwelveMonthChange rdf:type ppi:TwelveMonthChange ;
        ppi:changeValue 2.5 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_FoodManufacturing_12345_Entity ;
        ppi:hasStartMonth ppi:December ;
        ppi:hasStartYear ppi:2023 ;
        ppi:hasEndMonth ppi:December ;
        ppi:hasEndYear ppi:2024 ;
        ppi:isUnadjusted true .
    
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
    
    # Relative importance
    ppi:FinalDemand_FoodManufacturing_12345_November2024_RelativeImportance rdf:type ppi:RelativeImportance ;
        ppi:relativeImportanceValue 8.2 ;
        ppi:hasCommodityGrouping ppi:FinalDemand_FoodManufacturing_12345_Entity ;
        ppi:hasMonth ppi:November ;
        ppi:hasYear ppi:2024 .
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
    
    # Industry hierarchy: Total nonfarm > Service-providing > Leisure and hospitality > Food services
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
    
    # Manufacturing hierarchy
    jolts:Total_nonfarm_Goods_producing_Industry rdf:type jolts:Industry ;
        rdfs:label "Goods-producing" ;
        jolts:hasFullPath "Total_nonfarm_Goods_producing" ;
        jolts:hasParent jolts:Total_nonfarm_Industry .
    
    jolts:Total_nonfarm_Goods_producing_Manufacturing_Industry rdf:type jolts:Industry ;
        rdfs:label "Manufacturing" ;
        jolts:hasFullPath "Total_nonfarm_Goods_producing_Manufacturing" ;
        jolts:hasParent jolts:Total_nonfarm_Goods_producing_Industry .
    
    # Retail trade
    jolts:Total_nonfarm_Service_providing_Retail_trade_Industry rdf:type jolts:Industry ;
        rdfs:label "Retail trade" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing_Retail_trade" ;
        jolts:hasParent jolts:Total_nonfarm_Service_providing_Industry .
    
    # Healthcare
    jolts:Total_nonfarm_Service_providing_Health_care_and_social_assistance_Industry rdf:type jolts:Industry ;
        rdfs:label "Health care and social assistance" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing_Health_care_and_social_assistance" ;
        jolts:hasParent jolts:Total_nonfarm_Service_providing_Industry .
    
    # Construction
    jolts:Total_nonfarm_Goods_producing_Construction_Industry rdf:type jolts:Industry ;
        rdfs:label "Construction" ;
        jolts:hasFullPath "Total_nonfarm_Goods_producing_Construction" ;
        jolts:hasParent jolts:Total_nonfarm_Goods_producing_Industry .
    
    # Transportation
    jolts:Total_nonfarm_Service_providing_Transportation_warehousing_and_utilities_Industry rdf:type jolts:Industry ;
        rdfs:label "Transportation, warehousing, and utilities" ;
        jolts:hasFullPath "Total_nonfarm_Service_providing_Transportation_warehousing_and_utilities" ;
        jolts:hasParent jolts:Total_nonfarm_Service_providing_Industry .
    
    # Job Openings measurements (November 2024)
    jolts:Total_nonfarm_November2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 7839 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_November2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 1234 ;
        jolts:hasIndustry jolts:Total_nonfarm_Service_providing_Leisure_and_hospitality_Accommodation_and_food_services_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_Goods_producing_Manufacturing_November2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 567 ;
        jolts:hasIndustry jolts:Total_nonfarm_Goods_producing_Manufacturing_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    # Job Openings measurements (December 2024)
    jolts:Total_nonfarm_December2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 8098 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:December ;
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
    
    # Separations measurements
    jolts:Total_nonfarm_November2024_TotalSeparationsLevel rdf:type jolts:TotalSeparationsLevel ;
        jolts:levelValue 5076 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
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
    
    # Rates
    jolts:Total_nonfarm_November2024_JobOpeningsRate rdf:type jolts:JobOpeningsRate ;
        jolts:rateValue 4.8 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    jolts:Total_nonfarm_December2024_JobOpeningsRate rdf:type jolts:JobOpeningsRate ;
        jolts:rateValue 4.9 ;
        jolts:hasIndustry jolts:Total_nonfarm_Industry ;
        jolts:hasMonth jolts:December ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    
    # Region data
    jolts:Northeast_Region rdf:type jolts:Region ;
        rdfs:label "Northeast" ;
        jolts:hasFullPath "Northeast" .
    
    jolts:Northeast_November2024_JobOpeningsLevel rdf:type jolts:JobOpeningsLevel ;
        jolts:levelValue 1500 ;
        jolts:hasRegion jolts:Northeast_Region ;
        jolts:hasMonth jolts:November ;
        jolts:hasYear jolts:2024 ;
        jolts:isSeasonallyAdjusted true .
    """


def create_sample_empsit_data() -> str:
    """Generate sample EMPSIT RDF data (Establishment + Household)"""
    return """
    @prefix empsit: <https://www.bls.gov/empsit/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    # Temporal entities
    empsit:November rdf:type empsit:Month .
    empsit:December rdf:type empsit:Month .
    empsit:2024 rdf:type empsit:Year .
    
    # ============================================
    # ESTABLISHMENT DATA (Table B-1)
    # ============================================
    
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
    
    empsit:private_service empsit:type empsit:Industry ;
        rdfs:label "Private service-providing" ;
        empsit:hasFullPath "private_service" ;
        empsit:hasParentIndustry empsit:total_private .
    
    empsit:retail rdf:type empsit:Industry ;
        rdfs:label "Retail trade" ;
        empsit:hasFullPath "retail" ;
        empsit:hasParentIndustry empsit:private_service .
    
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
    
    empsit:construction rdf:type empsit:Industry ;
        rdfs:label "Construction" ;
        empsit:hasFullPath "construction" ;
        empsit:hasParentIndustry empsit:goods_producing .
    
    # Employee counts (November 2024)
    empsit:total_nonfarm_November2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 159358 ;
        empsit:hasIndustry empsit:total_nonfarm ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    empsit:manufacturing_November2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 12987 ;
        empsit:hasIndustry empsit:manufacturing ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    empsit:food_services_November2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 12456 ;
        empsit:hasIndustry empsit:food_services ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    # Employee counts (December 2024)
    empsit:total_nonfarm_December2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 159568 ;
        empsit:hasIndustry empsit:total_nonfarm ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary true .
    
    empsit:manufacturing_December2024_EmployeeCount rdf:type empsit:EmployeeCount ;
        empsit:employeeCount 13012 ;
        empsit:hasIndustry empsit:manufacturing ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary true .
    
    # Average hourly earnings (November 2024)
    empsit:total_private_November2024_AverageHourlyEarnings rdf:type empsit:EarningsData ;
        empsit:value 35.46 ;
        empsit:unit "Dollars" ;
        empsit:hasIndustry empsit:total_private ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary false .
    
    # Average hourly earnings (December 2024)
    empsit:total_private_December2024_AverageHourlyEarnings rdf:type empsit:EarningsData ;
        empsit:value 35.69 ;
        empsit:unit "Dollars" ;
        empsit:hasIndustry empsit:total_private ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 ;
        empsit:isPreliminary true .
    
    # ============================================
    # HOUSEHOLD DATA (Table A)
    # ============================================
    
    # Employment status categories
    empsit:civilian_labor_force rdf:type empsit:Category ;
        rdfs:label "Civilian labor force" ;
        empsit:fullPath "Civilian_noninstitutional_population_Civilian_labor_force" .
    
    empsit:employed rdf:type empsit:Category ;
        rdfs:label "Employed" ;
        empsit:fullPath "Civilian_noninstitutional_population_Civilian_labor_force_Employed" ;
        empsit:hasParentCategory empsit:civilian_labor_force .
    
    empsit:unemployed rdf:type empsit:Category ;
        rdfs:label "Unemployed" ;
        empsit:fullPath "Civilian_noninstitutional_population_Civilian_labor_force_Unemployed" ;
        empsit:hasParentCategory empsit:civilian_labor_force .
    
    empsit:not_in_lf rdf:type empsit:Category ;
        rdfs:label "Not in labor force" ;
        empsit:fullPath "Civilian_noninstitutional_population_Not_in_labor_force" .
    
    # Household data measurements (November 2024)
    empsit:employed_November2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 161861 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:employed ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 .
    
    empsit:unemployed_November2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 6728 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:unemployed ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 .
    
    # Household data measurements (December 2024)
    empsit:employed_December2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 162098 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:employed ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 .
    
    empsit:unemployed_December2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 6456 ;
        empsit:metricType "Count" ;
        empsit:unit "Thousands" ;
        empsit:hasCategory empsit:unemployed ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 .
    
    # Unemployment rate
    empsit:unemployment_rate_November2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 4.0 ;
        empsit:metricType "Rate" ;
        empsit:unit "Percent" ;
        empsit:hasCategory empsit:unemployed ;
        empsit:hasMonth empsit:November ;
        empsit:hasYear empsit:2024 .
    
    empsit:unemployment_rate_December2024_HouseholdData rdf:type empsit:HouseholdData ;
        empsit:value 3.8 ;
        empsit:metricType "Rate" ;
        empsit:unit "Percent" ;
        empsit:hasCategory empsit:unemployed ;
        empsit:hasMonth empsit:December ;
        empsit:hasYear empsit:2024 .
    """


# ============================================
# TEST RESULT CLASSES
# ============================================

@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    passed: bool
    graph: Optional[Graph]
    stats: Optional[Dict[str, any]]  # Changed from implicit to explicit
    output_file: str
    error_message: Optional[str] = None

    def get_datasets(self) -> List[str]:
        """Safely get available datasets list"""
        if self.stats and 'available_datasets' in self.stats:
            datasets = self.stats['available_datasets']
            if isinstance(datasets, list):
                return datasets
        return []

    def get_triples_added(self) -> int:
        """Safely get triples added count"""
        if self.stats and 'total_triples_added' in self.stats:
            return self.stats['total_triples_added']
        return 0


class TestSuite:
    """Unified test suite for all BLS datasets"""

    def __init__(self):
        self.results: List[TestResult] = []

    # ============================================
    # INDIVIDUAL DATASET TESTS
    # ============================================

    def test_cpi_only(self) -> TestResult:
        """Test CPI enrichment alone"""
        test_name = "CPI Only"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading CPI RDF data...")
            cpi_ttl = create_sample_cpi_data()
            loader.load_from_string(cpi_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} CPI triples")

            logger.info("[2] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[3] Validation...")
            passed = (
                'cpi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0
            )

            output_file = 'enriched_cpi_only.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[4] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_ppi_only(self) -> TestResult:
        """Test PPI enrichment alone"""
        test_name = "PPI Only"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading PPI RDF data...")
            ppi_ttl = create_sample_ppi_data()
            loader.load_from_string(ppi_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} PPI triples")

            logger.info("[2] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[3] Validation...")
            passed = (
                'ppi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0
            )

            output_file = 'enriched_ppi_only.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[4] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_jolts_only(self) -> TestResult:
        """Test JOLTS enrichment alone"""
        test_name = "JOLTS Only"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading JOLTS RDF data...")
            jolts_ttl = create_sample_jolts_data()
            loader.load_from_string(jolts_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} JOLTS triples")

            logger.info("[2] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[3] Validation...")
            passed = (
                'jolts' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0
            )

            output_file = 'enriched_jolts_only.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[4] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_empsit_only(self) -> TestResult:
        """Test EMPSIT enrichment alone"""
        test_name = "EMPSIT Only"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading EMPSIT RDF data...")
            empsit_ttl = create_sample_empsit_data()
            loader.load_from_string(empsit_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} EMPSIT triples")

            logger.info("[2] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[3] Validation...")
            passed = (
                'empsit' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0
            )

            output_file = 'enriched_empsit_only.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[4] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    # ============================================
    # PAIRWISE DATASET TESTS
    # ============================================

    def test_cpi_ppi(self) -> TestResult:
        """Test CPI + PPI integration"""
        test_name = "CPI + PPI"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading CPI RDF data...")
            cpi_ttl = create_sample_cpi_data()
            loader.load_from_string(cpi_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} CPI triples")

            logger.info("[2] Loading PPI RDF data...")
            ppi_ttl = create_sample_ppi_data()
            loader.load_from_string(ppi_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[3] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[4] Validation...")
            # Check for CPI-PPI food sector correlation
            food_sector_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?ppiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
                FILTER(STRSTARTS(STR(?ppiEntity), "https://www.bls.gov/ppi/"))
            }
            """
            result_query = list(loader.graph.query(food_sector_query))
            food_sector_count = int(result_query[0].count)
            logger.info(f"   Found {food_sector_count} CPI-PPI food sector entities")

            passed = (
                'cpi' in stats['available_datasets'] and
                'ppi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0 and
                food_sector_count > 0
            )

            output_file = 'enriched_cpi_ppi.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[5] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_jolts_cpi(self) -> TestResult:
        """Test JOLTS + CPI integration"""
        test_name = "JOLTS + CPI"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading JOLTS RDF data...")
            jolts_ttl = create_sample_jolts_data()
            loader.load_from_string(jolts_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} JOLTS triples")

            logger.info("[2] Loading CPI RDF data...")
            cpi_ttl = create_sample_cpi_data()
            loader.load_from_string(cpi_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[3] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[4] Validation...")
            # Check for JOLTS-CPI food sector correlation
            food_sector_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?joltsEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?joltsEntity), "https://www.bls.gov/jolts/"))
                FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
            }
            """
            result_query = list(loader.graph.query(food_sector_query))
            food_sector_count = int(result_query[0].count)
            logger.info(f"   Found {food_sector_count} JOLTS-CPI food sector entities")

            passed = (
                'jolts' in stats['available_datasets'] and
                'cpi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0 and
                food_sector_count > 0
            )

            output_file = 'enriched_jolts_cpi.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[5] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_jolts_ppi(self) -> TestResult:
        """Test JOLTS + PPI integration"""
        test_name = "JOLTS + PPI"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading JOLTS RDF data...")
            jolts_ttl = create_sample_jolts_data()
            loader.load_from_string(jolts_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} JOLTS triples")

            logger.info("[2] Loading PPI RDF data...")
            ppi_ttl = create_sample_ppi_data()
            loader.load_from_string(ppi_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[3] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[4] Validation...")
            # Check for JOLTS-PPI manufacturing sector correlation
            manufacturing_sector_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?joltsEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/ManufacturingSector> .
                ?ppiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/ManufacturingSector> .
                FILTER(STRSTARTS(STR(?joltsEntity), "https://www.bls.gov/jolts/"))
                FILTER(STRSTARTS(STR(?ppiEntity), "https://www.bls.gov/ppi/"))
            }
            """
            result_query = list(loader.graph.query(manufacturing_sector_query))
            manufacturing_sector_count = int(result_query[0].count)
            logger.info(f"   Found {manufacturing_sector_count} JOLTS-PPI manufacturing sector entities")

            passed = (
                'jolts' in stats['available_datasets'] and
                'ppi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0
            )

            output_file = 'enriched_jolts_ppi.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[5] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_empsit_cpi(self) -> TestResult:
        """Test EMPSIT + CPI integration"""
        test_name = "EMPSIT + CPI"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading EMPSIT RDF data...")
            empsit_ttl = create_sample_empsit_data()
            loader.load_from_string(empsit_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} EMPSIT triples")

            logger.info("[2] Loading CPI RDF data...")
            cpi_ttl = create_sample_cpi_data()
            loader.load_from_string(cpi_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[3] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[4] Validation...")
            # Check for EMPSIT-CPI food sector correlation
            food_sector_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?empsitEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?empsitEntity), "https://www.bls.gov/empsit/"))
                FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
            }
            """
            result_query = list(loader.graph.query(food_sector_query))
            food_sector_count = int(result_query[0].count)
            logger.info(f"   Found {food_sector_count} EMPSIT-CPI food sector entities")

            passed = (
                'empsit' in stats['available_datasets'] and
                'cpi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0 and
                food_sector_count > 0
            )

            output_file = 'enriched_empsit_cpi.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[5] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_empsit_ppi(self) -> TestResult:
        """Test EMPSIT + PPI integration"""
        test_name = "EMPSIT + PPI"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading EMPSIT RDF data...")
            empsit_ttl = create_sample_empsit_data()
            loader.load_from_string(empsit_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} EMPSIT triples")

            logger.info("[2] Loading PPI RDF data...")
            ppi_ttl = create_sample_ppi_data()
            loader.load_from_string(ppi_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[3] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[4] Validation...")
            # Check for EMPSIT-PPI manufacturing sector correlation
            manufacturing_sector_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?empsitEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/ManufacturingSector> .
                ?ppiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/ManufacturingSector> .
                FILTER(STRSTARTS(STR(?empsitEntity), "https://www.bls.gov/empsit/"))
                FILTER(STRSTARTS(STR(?ppiEntity), "https://www.bls.gov/ppi/"))
            }
            """
            result_query = list(loader.graph.query(manufacturing_sector_query))
            manufacturing_sector_count = int(result_query[0].count)
            logger.info(f"   Found {manufacturing_sector_count} EMPSIT-PPI manufacturing sector entities")

            passed = (
                'empsit' in stats['available_datasets'] and
                'ppi' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0 and
                manufacturing_sector_count > 0
            )

            output_file = 'enriched_empsit_ppi.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[5] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    def test_empsit_jolts(self) -> TestResult:
        """Test EMPSIT + JOLTS integration"""
        test_name = "EMPSIT + JOLTS"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading EMPSIT RDF data...")
            empsit_ttl = create_sample_empsit_data()
            loader.load_from_string(empsit_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} EMPSIT triples")

            logger.info("[2] Loading JOLTS RDF data...")
            jolts_ttl = create_sample_jolts_data()
            loader.load_from_string(jolts_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[3] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[4] Validation...")
            # Check for EMPSIT-JOLTS food sector correlation
            food_sector_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?empsitEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?joltsEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?empsitEntity), "https://www.bls.gov/empsit/"))
                FILTER(STRSTARTS(STR(?joltsEntity), "https://www.bls.gov/jolts/"))
            }
            """
            result_query = list(loader.graph.query(food_sector_query))
            food_sector_count = int(result_query[0].count)
            logger.info(f"   Found {food_sector_count} EMPSIT-JOLTS food sector entities")

            passed = (
                'empsit' in stats['available_datasets'] and
                'jolts' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0 and
                food_sector_count > 0
            )

            output_file = 'enriched_empsit_jolts.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[5] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    # ============================================
    # FULL INTEGRATION TEST
    # ============================================

    def test_all_datasets(self) -> TestResult:
        """Test CPI + PPI + JOLTS + EMPSIT full integration"""
        test_name = "CPI + PPI + JOLTS + EMPSIT (Full Integration)"
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*80}")

        try:
            loader = RDFGraphLoader()

            logger.info("[1] Loading CPI RDF data...")
            cpi_ttl = create_sample_cpi_data()
            loader.load_from_string(cpi_ttl, format='turtle')
            logger.info(f"   Loaded {len(loader.graph)} CPI triples")

            logger.info("[2] Loading PPI RDF data...")
            ppi_ttl = create_sample_ppi_data()
            loader.load_from_string(ppi_ttl, format='turtle')
            logger.info(f"   Total: {len(loader.graph)} triples")

            logger.info("[3] Loading JOLTS RDF data...")
            jolts_ttl = create_sample_jolts_data()
            loader.load_from_string(jolts_ttl, format='turtle')
            logger.info(f"   Total: {len(loader.graph)} triples")

            logger.info("[4] Loading EMPSIT RDF data...")
            empsit_ttl = create_sample_empsit_data()
            loader.load_from_string(empsit_ttl, format='turtle')
            logger.info(f"   Total graph size: {len(loader.graph)} triples")

            logger.info("[5] Running enrichment...")
            linker = BLSIntraSourceLinker(loader.graph)
            stats = linker.enrich()

            logger.info("[6] Validation...")

            # Check CPI-PPI food sector
            cpi_ppi_food_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?ppiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
                FILTER(STRSTARTS(STR(?ppiEntity), "https://www.bls.gov/ppi/"))
            }
            """
            result_query = list(loader.graph.query(cpi_ppi_food_query))
            cpi_ppi_food_count = int(result_query[0].count)
            logger.info(f"   CPI-PPI food sector entities: {cpi_ppi_food_count}")

            # Check JOLTS-CPI food sector
            jolts_cpi_food_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?joltsEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?joltsEntity), "https://www.bls.gov/jolts/"))
                FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
            }
            """
            result_query = list(loader.graph.query(jolts_cpi_food_query))
            jolts_cpi_food_count = int(result_query[0].count)
            logger.info(f"   JOLTS-CPI food sector entities: {jolts_cpi_food_count}")

            # Check EMPSIT-CPI food sector
            empsit_cpi_food_query = """
            SELECT (COUNT(*) as ?count) WHERE {
                ?empsitEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                ?cpiEntity <https://www.bls.gov/enrichment/belongsToSector> <https://www.bls.gov/enrichment/FoodSector> .
                FILTER(STRSTARTS(STR(?empsitEntity), "https://www.bls.gov/empsit/"))
                FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
            }
            """
            result_query = list(loader.graph.query(empsit_cpi_food_query))
            empsit_cpi_food_count = int(result_query[0].count)
            logger.info(f"   EMPSIT-CPI food sector entities: {empsit_cpi_food_count}")

            # Check unified temporal entities
            unified_temporal_query = """
            SELECT (COUNT(DISTINCT ?unified) as ?count) WHERE {
                ?entity <https://www.bls.gov/enrichment/unifiedAs> ?unified .
            }
            """
            result_query = list(loader.graph.query(unified_temporal_query))
            unified_temporal_count = int(result_query[0].count)
            logger.info(f"   Unified temporal entities: {unified_temporal_count}")

            passed = (
                'cpi' in stats['available_datasets'] and
                'ppi' in stats['available_datasets'] and
                'jolts' in stats['available_datasets'] and
                'empsit' in stats['available_datasets'] and
                stats['temporal_sequences'] > 0 and
                stats['sector_links'] > 0 and
                stats['temporal_unified'] > 0 and
                cpi_ppi_food_count > 0 and
                jolts_cpi_food_count > 0 and
                empsit_cpi_food_count > 0 and
                unified_temporal_count > 0
            )

            output_file = 'enriched_all_datasets.ttl'
            writer = RDFGraphWriter()
            writer.save_to_file(loader.graph, output_file, format='turtle')
            logger.info(f"[7] Saved to: {output_file}")

            result = TestResult(
                test_name=test_name,
                passed=passed,
                graph=loader.graph,
                stats=stats,
                output_file=output_file
            )

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            result = TestResult(
                test_name=test_name,
                passed=False,
                graph=None,
                stats=None,
                output_file="",
                error_message=str(e)
            )

        self.results.append(result)
        return result

    # ============================================
    # SUMMARY REPORTING
    # ============================================

    def print_summary(self):
        """Print summary of all test results"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUITE SUMMARY")
        logger.info("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        logger.info(f"\nTotal Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        logger.info("\nDetailed Results:")
        logger.info("-" * 80)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"{status} | {result.test_name}")

            # Use helper methods for type-safe access
            datasets = result.get_datasets()
            if datasets:
                logger.info(f"       Datasets: {', '.join(datasets)}")

            triples_added = result.get_triples_added()
            logger.info(f"       Triples added: {triples_added}")

            if result.output_file:
                logger.info(f"       Output: {result.output_file}")

            if result.error_message:
                logger.info(f"       Error: {result.error_message}")

            logger.info("")

        logger.info("=" * 80)

        return passed_tests == total_tests


# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified BLS Dataset Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test individual datasets
  python tests/test_bls_dataset.py --dataset cpi
  python tests/test_bls_dataset.py --dataset ppi
  python tests/test_bls_dataset.py --dataset jolts
  python tests/test_bls_dataset.py --dataset empsit
  
  # Test dataset pairs
  python tests/test_bls_dataset.py --combination cpi-ppi
  python tests/test_bls_dataset.py --combination jolts-cpi
  python tests/test_bls_dataset.py --combination jolts-ppi
  python tests/test_bls_dataset.py --combination empsit-cpi
  python tests/test_bls_dataset.py --combination empsit-ppi
  python tests/test_bls_dataset.py --combination empsit-jolts
  
  # Test all datasets together
  python tests/test_bls_dataset.py --combination all
  
  # Run all tests
  python tests/test_bls_dataset.py --run-all
        """
    )

    parser.add_argument('--dataset',
                       choices=['cpi', 'ppi', 'jolts', 'empsit'],
                       help='Test a single dataset')

    parser.add_argument('--combination',
                       choices=['cpi-ppi', 'jolts-cpi', 'jolts-ppi',
                               'empsit-cpi', 'empsit-ppi', 'empsit-jolts', 'all'],
                       help='Test dataset combinations')

    parser.add_argument('--run-all',
                       action='store_true',
                       help='Run all tests (individual + combinations)')

    args = parser.parse_args()

    suite = TestSuite()

    if args.run_all:
        # Run all tests
        logger.info("Running ALL tests...")
        suite.test_cpi_only()
        suite.test_ppi_only()
        suite.test_jolts_only()
        suite.test_empsit_only()
        suite.test_cpi_ppi()
        suite.test_jolts_cpi()
        suite.test_jolts_ppi()
        suite.test_empsit_cpi()
        suite.test_empsit_ppi()
        suite.test_empsit_jolts()
        suite.test_all_datasets()

    elif args.dataset:
        # Run single dataset test
        if args.dataset == 'cpi':
            suite.test_cpi_only()
        elif args.dataset == 'ppi':
            suite.test_ppi_only()
        elif args.dataset == 'jolts':
            suite.test_jolts_only()
        elif args.dataset == 'empsit':
            suite.test_empsit_only()

    elif args.combination:
        # Run combination test
        if args.combination == 'cpi-ppi':
            suite.test_cpi_ppi()
        elif args.combination == 'jolts-cpi':
            suite.test_jolts_cpi()
        elif args.combination == 'jolts-ppi':
            suite.test_jolts_ppi()
        elif args.combination == 'empsit-cpi':
            suite.test_empsit_cpi()
        elif args.combination == 'empsit-ppi':
            suite.test_empsit_ppi()
        elif args.combination == 'empsit-jolts':
            suite.test_empsit_jolts()
        elif args.combination == 'all':
            suite.test_all_datasets()

    else:
        # Default: run all tests
        logger.info("No arguments provided. Running ALL tests...")
        suite.test_cpi_only()
        suite.test_ppi_only()
        suite.test_jolts_only()
        suite.test_empsit_only()
        suite.test_cpi_ppi()
        suite.test_jolts_cpi()
        suite.test_jolts_ppi()
        suite.test_empsit_cpi()
        suite.test_empsit_ppi()
        suite.test_empsit_jolts()
        suite.test_all_datasets()

    # Print summary
    all_passed = suite.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()