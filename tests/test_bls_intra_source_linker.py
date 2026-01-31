"""
Test BLS Intra-Source Linker with CPI + PPI data

This test demonstrates:
1. Loading CPI and PPI RDF data
2. Running BLS intra-source enrichment
3. Validating enrichment results
4. Saving enriched graph
"""

import sys
import os
from typing import Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdflib import Graph
from glue_jobs.utils.rdf_utils import RDFGraphLoader, RDFGraphWriter
from glue_jobs.enrichment.intra_source_linker import BLSIntraSourceLinker
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_cpi_data() -> str:
    """
    Create sample CPI RDF data for testing.
    
    Based on actual CPI ontology structure:
    - Categories with hierarchies (hasParent, hasFullPath)
    - Index measurements (hasCategory, hasMonth, hasYear)
    - Temporal entities (Month, Year)
    """
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
    """
    Create sample PPI RDF data for testing.
    
    Based on actual PPI ontology structure:
    - Commodity groupings with hierarchies (hasParent, hasFullPath, belongsToSection)
    - Price changes (hasCommodityGrouping, hasMonth, hasYear)
    - Temporal entities (Month, Year)
    """
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


def validate_enrichment_results(graph: Graph, stats: Dict[str, int]) -> bool:
    """
    Validate that enrichment produced expected results.
    
    Args:
        graph: Enriched RDF graph
        stats: Enrichment statistics
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("\n" + "="*80)
    logger.info("Validating Enrichment Results")
    logger.info("="*80)
    
    validation_passed = True
    
    # Test 1: Check temporal unification
    logger.info("\n[Test 1] Checking temporal unification...")
    unified_months_query = """
    SELECT (COUNT(DISTINCT ?unifiedMonth) as ?count) WHERE {
        ?month <https://www.bls.gov/enrichment/unifiedAs> ?unifiedMonth .
        ?unifiedMonth a <https://www.bls.gov/enrichment/UnifiedMonth> .
    }
    """
    result = list(graph.query(unified_months_query))
    unified_month_count = int(result[0].count)
    
    if unified_month_count > 0:
        logger.info(f"  ✓ Found {unified_month_count} unified months")
    else:
        logger.error(f"  ✗ Expected unified months, found {unified_month_count}")
        validation_passed = False
    
    # Test 2: Check temporal sequences
    logger.info("\n[Test 2] Checking temporal sequences...")
    precedes_query = """
    SELECT (COUNT(*) as ?count) WHERE {
        ?measurement1 <https://www.bls.gov/enrichment/precedes> ?measurement2 .
    }
    """
    result = list(graph.query(precedes_query))
    precedes_count = int(result[0].count)
    
    if precedes_count > 0:
        logger.info(f"  ✓ Found {precedes_count} temporal sequence links")
    else:
        logger.error(f"  ✗ Expected temporal sequences, found {precedes_count}")
        validation_passed = False
    
    # Test 3: Check sector links
    logger.info("\n[Test 3] Checking sector links...")
    sector_query = """
    SELECT (COUNT(DISTINCT ?sector) as ?count) WHERE {
        ?sector a <https://www.bls.gov/enrichment/EconomicSector> .
    }
    """
    result = list(graph.query(sector_query))
    sector_count = int(result[0].count)
    
    if sector_count > 0:
        logger.info(f"  ✓ Found {sector_count} economic sectors")
        
        # Check entities linked to sectors
        sector_entities_query = """
        SELECT (COUNT(*) as ?count) WHERE {
            ?entity <https://www.bls.gov/enrichment/belongsToSector> ?sector .
        }
        """
        result = list(graph.query(sector_entities_query))
        sector_entity_count = int(result[0].count)
        logger.info(f"  ✓ Found {sector_entity_count} entities linked to sectors")
    else:
        logger.error(f"  ✗ Expected economic sectors, found {sector_count}")
        validation_passed = False
    
    # Test 4: Check CPI-PPI correlations
    logger.info("\n[Test 4] Checking CPI-PPI correlations...")
    correlation_query = """
    SELECT (COUNT(*) as ?count) WHERE {
        ?cpiEntity <https://www.bls.gov/enrichment/producerConsumerLink> ?ppiEntity .
        FILTER(STRSTARTS(STR(?cpiEntity), "https://www.bls.gov/cpi/"))
        FILTER(STRSTARTS(STR(?ppiEntity), "https://www.bls.gov/ppi/"))
    }
    """
    result = list(graph.query(correlation_query))
    correlation_count = int(result[0].count)
    
    if correlation_count > 0:
        logger.info(f"  ✓ Found {correlation_count} CPI-PPI correlation links")
    else:
        logger.warning(f"  ⚠ Expected CPI-PPI correlations, found {correlation_count}")
        # Not a failure - might not have matching entities
    
    # Test 5: Check hierarchy links
    logger.info("\n[Test 5] Checking cross-dataset hierarchy links...")
    hierarchy_query = """
    SELECT (COUNT(*) as ?count) WHERE {
        ?entity1 <https://www.bls.gov/enrichment/relatedHierarchy> ?entity2 .
    }
    """
    result = list(graph.query(hierarchy_query))
    hierarchy_count = int(result[0].count)
    
    if hierarchy_count > 0:
        logger.info(f"  ✓ Found {hierarchy_count} cross-hierarchy links")
    else:
        logger.warning(f"  ⚠ Expected hierarchy links, found {hierarchy_count}")
        # Not a failure - might not have matching hierarchies
    
    # Test 6: Verify stats match actual graph
    logger.info("\n[Test 6] Verifying statistics...")
    if stats['temporal_unified'] > 0:
        logger.info(f"  ✓ Temporal unification: {stats['temporal_unified']} links")
    else:
        logger.error(f"  ✗ No temporal unification links")
        validation_passed = False
    
    if stats['temporal_sequences'] > 0:
        logger.info(f"  ✓ Temporal sequences: {stats['temporal_sequences']} links")
    else:
        logger.error(f"  ✗ No temporal sequence links")
        validation_passed = False
    
    if stats['sector_links'] > 0:
        logger.info(f"  ✓ Sector links: {stats['sector_links']} links")
    else:
        logger.error(f"  ✗ No sector links")
        validation_passed = False
    
    logger.info("\n" + "="*80)
    if validation_passed:
        logger.info("✓ All validation tests passed!")
    else:
        logger.error("✗ Some validation tests failed")
    logger.info("="*80)
    
    return validation_passed


def test_cpi_ppi_linking() -> Tuple[Graph, Dict[str, int], bool]:
    """
    Main test function for CPI + PPI linking.
    
    Tests:
    1. Loading CPI and PPI RDF data
    2. Running BLS intra-source enrichment
    3. Validating enrichment results
    4. Saving enriched graph
    
    Returns:
        Tuple of (enriched_graph, stats, validation_passed)
    """
    
    print("\n" + "="*80)
    print("Testing BLS Intra-Source Linker: CPI + PPI")
    print("="*80)
    
    # Initialize loader
    loader = RDFGraphLoader()
    
    # Load CPI data
    print("\n[1] Loading CPI RDF data...")
    cpi_ttl = create_sample_cpi_data()
    loader.load_from_string(cpi_ttl, format='turtle')
    print(f"   Loaded {len(loader.graph)} CPI triples")
    
    # Load PPI data
    print("\n[2] Loading PPI RDF data...")
    ppi_ttl = create_sample_ppi_data()
    loader.load_from_string(ppi_ttl, format='turtle')
    print(f"   Total graph size: {len(loader.graph)} triples")
    
    # Print initial graph statistics
    print("\n[3] Initial graph statistics:")
    print(f"   Total triples: {len(loader.graph)}")
    
    # Count CPI entities
    cpi_count_query = """
    SELECT (COUNT(DISTINCT ?s) as ?count) WHERE {
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?s), "https://www.bls.gov/cpi/"))
    }
    """
    result = list(loader.graph.query(cpi_count_query))
    print(f"   CPI entities: {result[0].count}")
    
    # Count PPI entities
    ppi_count_query = """
    SELECT (COUNT(DISTINCT ?s) as ?count) WHERE {
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?s), "https://www.bls.gov/ppi/"))
    }
    """
    result = list(loader.graph.query(ppi_count_query))
    print(f"   PPI entities: {result[0].count}")
    
    # Run enrichment
    print("\n[4] Running BLS Intra-Source Enrichment...")
    linker = BLSIntraSourceLinker(loader.graph)
    stats = linker.enrich()
    
    # Validate results
    print("\n[5] Validating enrichment results...")
    validation_passed = validate_enrichment_results(loader.graph, stats)
    
    # Save enriched graph
    print("\n[6] Saving enriched graph...")
    writer = RDFGraphWriter()
    
    # Save to file
    output_file = 'enriched_bls_cpi_ppi.ttl'
    writer.save_to_file(loader.graph, output_file, format='turtle')
    print(f"   Saved to: {output_file}")
    
    # Print final statistics
    print("\n" + "="*80)
    print("Final Statistics:")
    print("="*80)
    print(f"  Initial triples: {len(loader.graph) - stats['total_triples_added']}")
    print(f"  Enrichment triples: {stats['total_triples_added']}")
    print(f"  Final triples: {len(loader.graph)}")
    print(f"  Datasets detected: {', '.join(stats['available_datasets'])}")
    print("="*80)
    
    # Print sample enriched triples
    print("\n[7] Sample enriched triples:")
    print("\n  Temporal unification:")
    unified_sample_query = """
    SELECT ?month ?unified WHERE {
        ?month <https://www.bls.gov/enrichment/unifiedAs> ?unified .
    } LIMIT 3
    """
    for row in loader.graph.query(unified_sample_query):
        print(f"    {row.month} → {row.unified}")
    
    print("\n  Temporal sequences:")
    precedes_sample_query = """
    SELECT ?m1 ?m2 WHERE {
        ?m1 <https://www.bls.gov/enrichment/precedes> ?m2 .
    } LIMIT 3
    """
    for row in loader.graph.query(precedes_sample_query):
        print(f"    {row.m1}")
        print(f"      precedes → {row.m2}")
    
    print("\n  Sector links:")
    sector_sample_query = """
    SELECT ?entity ?sector WHERE {
        ?entity <https://www.bls.gov/enrichment/belongsToSector> ?sector .
    } LIMIT 5
    """
    for row in loader.graph.query(sector_sample_query):
        print(f"    {row.entity}")
        print(f"      belongsTo → {row.sector}")
    
    print("\n" + "="*80)
    if validation_passed:
        print("✅ Test PASSED - All validations successful!")
    else:
        print("❌ Test FAILED - Some validations failed")
    print("="*80)
    
    return loader.graph, stats, validation_passed


def test_with_real_data() -> Tuple[Optional[Graph], Optional[Dict[str, int]], bool]:
    """
    Test with real CPI and PPI RDF files (if available).
    
    This function can be used when you have actual RDF files from your mappers.
    
    Returns:
        Tuple of (enriched_graph, stats, validation_passed) or (None, None, False) if files not found
    """
    print("\n" + "="*80)
    print("Testing with Real CPI + PPI Data")
    print("="*80)
    
    # Check if real data files exist
    cpi_file = 'data/cpi_table1_november2024.ttl'
    ppi_file = 'data/ppi_table1_november2024.ttl'
    
    if not os.path.exists(cpi_file) or not os.path.exists(ppi_file):
        print(f"\n⚠ Real data files not found:")
        print(f"  Expected: {cpi_file}")
        print(f"  Expected: {ppi_file}")
        print(f"\nSkipping real data test. Use test_cpi_ppi_linking() for sample data test.")
        return None, None, False
    
    # Load real data
    loader = RDFGraphLoader()
    
    print(f"\n[1] Loading CPI data from {cpi_file}...")
    loader.load_from_file(cpi_file, format='turtle')
    print(f"   Loaded {len(loader.graph)} triples")
    
    print(f"\n[2] Loading PPI data from {ppi_file}...")
    loader.load_from_file(ppi_file, format='turtle')
    print(f"   Total graph size: {len(loader.graph)} triples")
    
    # Run enrichment
    print("\n[3] Running BLS Intra-Source Enrichment...")
    linker = BLSIntraSourceLinker(loader.graph)
    stats = linker.enrich()
    
    # Validate results
    print("\n[4] Validating enrichment results...")
    validation_passed = validate_enrichment_results(loader.graph, stats)
    
    # Save enriched graph
    print("\n[5] Saving enriched graph...")
    writer = RDFGraphWriter()
    output_file = 'enriched_bls_real_data.ttl'
    writer.save_to_file(loader.graph, output_file, format='turtle')
    print(f"   Saved to: {output_file}")
    
    return loader.graph, stats, validation_passed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BLS Intra-Source Linker')
    parser.add_argument('--real-data', action='store_true', 
                       help='Test with real CPI/PPI data files instead of sample data')
    args = parser.parse_args()
    
    if args.real_data:
        graph, stats, passed = test_with_real_data()
    else:
        graph, stats, passed = test_cpi_ppi_linking()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)