"""
Ontology Mapping Utilities (PySpark)

Maps between different ontology vocabularies and creates standardized mappings.
This is OPTIONAL - the pipeline works without it, but it improves interoperability.

Use cases:
1. Standardize property names across BLS datasets (hasMonth → unified:hasMonth)
2. Create owl:equivalentProperty and owl:equivalentClass mappings
3. Add SKOS prefLabel for entities that only have rdfs:label
4. Align classification systems (NAICS, SIC, GICS, etc.) — future
5. Map NOAA CAP properties to unified equivalents

All operations run as PySpark DataFrame transformations.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from functools import reduce
from typing import List, Optional

from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT, NOAA_ENRICHMENT,
    UNIFIED, CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER,
    SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP, MARKET, CAP, NWS
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_EQUIVALENT_PROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
OWL_EQUIVALENT_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
SKOS_PREF_LABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
SKOS_CONCEPT_SCHEME = "http://www.w3.org/2004/02/skos/core#ConceptScheme"
DCTERMS_DESCRIPTION = "http://purl.org/dc/terms/description"

# ============================================
# PROPERTY EQUIVALENCE MAPPINGS
# ============================================

PROPERTY_MAPPINGS = {
    # Temporal: all month properties → unified:hasMonth
    str(CPI.hasMonth): str(UNIFIED.hasMonth),
    str(PPI.hasStartMonth): str(UNIFIED.hasMonth),
    str(PPI.hasEndMonth): str(UNIFIED.hasMonth),
    str(ECI.hasMonth): str(UNIFIED.hasMonth),
    str(JOLTS.hasMonth): str(UNIFIED.hasMonth),
    str(EMPSIT.hasMonth): str(UNIFIED.hasMonth),
    str(XIMPIM.hasMonth): str(UNIFIED.hasMonth),
    str(LAUS.hasMonth): str(UNIFIED.hasMonth),
    str(METRO.hasMonth): str(UNIFIED.hasMonth),
    str(REALER.hasMonth): str(UNIFIED.hasMonth),

    # Temporal: all year properties → unified:hasYear
    str(CPI.hasYear): str(UNIFIED.hasYear),
    str(PPI.hasStartYear): str(UNIFIED.hasYear),
    str(PPI.hasEndYear): str(UNIFIED.hasYear),
    str(ECI.hasYear): str(UNIFIED.hasYear),
    str(JOLTS.hasYear): str(UNIFIED.hasYear),
    str(EMPSIT.hasYear): str(UNIFIED.hasYear),
    str(XIMPIM.hasYear): str(UNIFIED.hasYear),
    str(LAUS.hasYear): str(UNIFIED.hasYear),
    str(METRO.hasYear): str(UNIFIED.hasYear),
    str(REALER.hasYear): str(UNIFIED.hasYear),

    # Measurement values → unified:measurementValue
    str(CPI.indexValue): str(UNIFIED.measurementValue),
    str(PPI.changeValue): str(UNIFIED.measurementValue),
    str(PPI.indexValue): str(UNIFIED.measurementValue),
    str(JOLTS.level): str(UNIFIED.measurementValue),
    str(JOLTS.rate): str(UNIFIED.measurementValue),
    str(EMPSIT.value): str(UNIFIED.measurementValue),
    str(ECI.indexValue): str(UNIFIED.measurementValue),
    str(MARKET.observedPrice): str(UNIFIED.measurementValue),
    str(MARKET.lastPrice): str(UNIFIED.measurementValue),

    # Category properties → unified:hasCategory
    str(CPI.hasCategory): str(UNIFIED.hasCategory),
    str(PPI.hasCommodityGrouping): str(UNIFIED.hasCategory),
    str(ECI.hasOccupationalGroup): str(UNIFIED.hasCategory),
    str(JOLTS.hasIndustry): str(UNIFIED.hasCategory),
    str(EMPSIT.hasIndustry): str(UNIFIED.hasCategory),
    str(EMPSIT.hasCategory): str(UNIFIED.hasCategory),

    # Company/ticker
    str(MARKET.symbol): str(UNIFIED.ticker),

    # Geographic
    str(LAUS.hasState): str(UNIFIED.hasRegion),
    str(METRO.hasMetropolitanArea): str(UNIFIED.hasRegion),

    # NOAA temporal properties → unified equivalents
    # cap:hasSentTime is the primary temporal property for NOAA alerts
    # (on Info subjects in the new mapper, but the equivalence is
    # at the property level regardless of subject)
    str(CAP.hasSentTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasEffectiveTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasOnsetTime): str(UNIFIED.hasTimestamp),
    str(CAP.hasExpirationTime): str(UNIFIED.hasTimestamp),

    # NOAA event → unified category
    str(CAP.hasEvent): str(UNIFIED.hasEventName),

    # NOAA severity/urgency/certainty → unified severity
    str(CAP.hasSeverity): str(UNIFIED.hasSeverity),
    str(CAP.hasUrgency): str(UNIFIED.hasUrgency),

    # NOAA area description → unified region description
    str(CAP.hasAreaDescription): str(UNIFIED.hasRegionDescription),
}

# ============================================
# CLASS EQUIVALENCE MAPPINGS
# ============================================

CLASS_MAPPINGS = {
    # Price indices
    str(CPI.Index): str(BLS_ENRICHMENT.PriceIndex),
    str(PPI.IndexValue): str(BLS_ENRICHMENT.PriceIndex),

    # Rate measurements
    str(JOLTS.JobOpeningsRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(JOLTS.HiresRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(JOLTS.QuitsRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(LAUS.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),
    str(METRO.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),

    # Change measurements
    str(CPI.PercentChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(PPI.MonthlyChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(PPI.TwelveMonthChange): str(BLS_ENRICHMENT.ChangeMeasurement),
    str(ECI.PercentChangeData): str(BLS_ENRICHMENT.ChangeMeasurement),

    # Level measurements
    str(JOLTS.JobOpeningsLevel): str(BLS_ENRICHMENT.LevelMeasurement),
    str(JOLTS.HiresLevel): str(BLS_ENRICHMENT.LevelMeasurement),
    str(EMPSIT.EmployeeCount): str(BLS_ENRICHMENT.LevelMeasurement),
    str(LAUS.LaborForceData): str(BLS_ENRICHMENT.LevelMeasurement),

    # Economic indicators
    str(CPI.Category): str(BLS_ENRICHMENT.EconomicIndicator),
    str(PPI.CommodityGrouping): str(BLS_ENRICHMENT.EconomicIndicator),

    # Industry classifications
    str(JOLTS.Industry): str(BLS_ENRICHMENT.IndustryClassification),
    str(EMPSIT.Industry): str(BLS_ENRICHMENT.IndustryClassification),
    str(ECI.Industry): str(BLS_ENRICHMENT.IndustryClassification),

    # Occupational classifications
    str(ECI.OccupationalGroup): str(BLS_ENRICHMENT.OccupationalClassification),
    str(EMPSIT.Occupation): str(BLS_ENRICHMENT.OccupationalClassification),

    # NOAA alert classes → unified emergency alert type
    # nws:WeatherAlert is the primary type from the RML mapper
    # (subClassOf cap:Alert in the ontology)
    str(NWS.WeatherAlert): str(NOAA_ENRICHMENT.EmergencyAlert),
    str(CAP.Alert): str(NOAA_ENRICHMENT.EmergencyAlert),

    # NOAA sub-structures
    str(CAP.Info): str(NOAA_ENRICHMENT.AlertInfo),
    str(CAP.Area): str(NOAA_ENRICHMENT.AlertArea),
}


class OntologyMapper:
    """
    Creates ontology equivalence mappings and label normalization
    using PySpark DataFrames.

    All mappings are static configuration — no data-dependent queries
    except for the label normalization step which reads rdfs:label
    triples from the graph.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def enrich(
        self,
        triples_df: DataFrame,
        enable_skos: bool = False,
    ) -> DataFrame:
        """
        Run ontology mapping.

        Args:
            triples_df: DataFrame with columns (subject, predicate, object)
            enable_skos: Whether to create SKOS concept schemes

        Returns:
            DataFrame of NEW triples (equivalences + labels)
        """
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        logger.info("=" * 60)
        logger.info("Starting Ontology Mapping (PySpark)")
        logger.info("=" * 60)

        new_dfs: List[DataFrame] = []

        logger.info("[Step 1/3] Creating property equivalences...")
        df = self._create_property_equivalences()
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/3] Creating class equivalences...")
        df = self._create_class_equivalences()
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 3/3] Normalizing labels (skos:prefLabel)...")
        df = self._normalize_labels(triples_df)
        if df is not None:
            new_dfs.append(df)

        if enable_skos:
            logger.info("[Optional] Creating SKOS concept schemes...")
            df = self._create_skos_concept_schemes()
            if df is not None:
                new_dfs.append(df)

        if not new_dfs:
            logger.info("No ontology mapping triples produced")
            return empty

        result = reduce(DataFrame.unionAll, new_dfs).cache()
        count = result.count()

        logger.info("=" * 60)
        logger.info(f"Ontology Mapping Complete: {count} triples")
        logger.info("=" * 60)

        return result

    # ================================================================
    # Step 1: Property Equivalences
    # ================================================================

    def _create_property_equivalences(self) -> Optional[DataFrame]:
        """
        Create owl:equivalentProperty triples from the static mapping table.

        Produces:
            cpi:Index  owl:equivalentClass  bls:PriceIndex
            ppi:IndexValue  owl:equivalentClass  bls:PriceIndex
            nws:WeatherAlert  owl:equivalentClass  noaa_enrichment:EmergencyAlert
            ...
        """
        if not PROPERTY_MAPPINGS:
            return None

        rows = [
            (source, OWL_EQUIVALENT_PROPERTY, target)
            for source, target in PROPERTY_MAPPINGS.items()
        ]

        df = self.spark.createDataFrame(
            rows, ["subject", "predicate", "object"]
        )

        logger.info(f"  Created {len(rows)} property equivalences")
        return df

    # ================================================================
    # Step 2: Class Equivalences
    # ================================================================

    def _create_class_equivalences(self) -> Optional[DataFrame]:
        """
        Create owl:equivalentClass triples from the static mapping table.

        Produces:
            cpi:Index  owl:equivalentClass  bls:PriceIndex
            nws:WeatherAlert  owl:equivalentClass  noaa_enrichment:EmergencyAlert
            ...
        """
        if not CLASS_MAPPINGS:
            return None

        rows = [
            (source, OWL_EQUIVALENT_CLASS, target)
            for source, target in CLASS_MAPPINGS.items()
        ]

        df = self.spark.createDataFrame(
            rows, ["subject", "predicate", "object"]
        )

        logger.info(f"  Created {len(rows)} class equivalences")
        return df

    # ================================================================
    # Step 3: Label Normalization
    # ================================================================

    def _normalize_labels(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Add skos:prefLabel for entities that have rdfs:label but no
        skos:prefLabel.

        Reads the triples DataFrame to find entities with rdfs:label,
        checks they don't already have skos:prefLabel, and produces
        new skos:prefLabel triples.
        """
        # Entities with rdfs:label
        has_label = triples_df.filter(
            F.col("predicate") == RDFS_LABEL
        ).select(
            F.col("subject").alias("entity"),
            F.col("object").alias("label"),
        )

        # Entities that already have skos:prefLabel
        has_pref = triples_df.filter(
            F.col("predicate") == SKOS_PREF_LABEL
        ).select(
            F.col("subject").alias("entity")
        ).dropDuplicates()

        # Left anti join: entities with label but no prefLabel
        needs_pref = has_label.join(has_pref, "entity", "left_anti")

        if needs_pref.head(1) == []:
            logger.info("  No labels to normalize")
            return None

        result = needs_pref.select(
            F.col("entity").alias("subject"),
            F.lit(SKOS_PREF_LABEL).alias("predicate"),
            F.col("label").alias("object"),
        )

        logger.info("  Label normalization complete")
        return result

    # ================================================================
    # Optional: SKOS Concept Schemes
    # ================================================================

    def _create_skos_concept_schemes(self) -> Optional[DataFrame]:
        """
        Create SKOS concept scheme triples for major classifications.

        Produces:
            bls:SectorConceptScheme  rdf:type  skos:ConceptScheme
            bls:SectorConceptScheme  rdfs:label  "BLS Economic Sectors"
            bls:SectorConceptScheme  dcterms:description  "..."
            ...
        """
        schemes = [
            (str(BLS_ENRICHMENT.SectorConceptScheme),
             "BLS Economic Sectors",
             "Economic sector classifications used in BLS data"),
            (str(BLS_ENRICHMENT.IndustryConceptScheme),
             "BLS Industry Classifications",
             "Industry classifications across BLS datasets"),
            (str(BLS_ENRICHMENT.OccupationConceptScheme),
             "BLS Occupational Classifications",
             "Occupational group classifications in ECI and EMPSIT"),
            (str(NOAA_ENRICHMENT.EventTypeConceptScheme),
             "NOAA Weather Event Types",
             "Weather event type classifications from NWS alerts, "
             "aligned with the NWS SKOS EventTypeScheme vocabulary"),
        ]

        rows = []
        for uri, label, description in schemes:
            rows.append((uri, RDF_TYPE, SKOS_CONCEPT_SCHEME))
            rows.append((uri, RDFS_LABEL, label))
            rows.append((uri, DCTERMS_DESCRIPTION, description))

        df = self.spark.createDataFrame(
            rows, ["subject", "predicate", "object"]
        )

        logger.info(f"  Created {len(schemes)} SKOS concept schemes")
        return df


def map_ontologies(
    spark: SparkSession,
    triples_df: DataFrame,
    enable_skos: bool = False,
) -> DataFrame:
    """
    Convenience function for ontology mapping.

    Args:
        spark: SparkSession
        triples_df: DataFrame with (subject, predicate, object)
        enable_skos: Whether to create SKOS concept schemes

    Returns:
        DataFrame of new ontology mapping triples
    """
    mapper = OntologyMapper(spark)
    return mapper.enrich(triples_df, enable_skos=enable_skos)