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
from typing import Dict, Iterable, List, Optional, Tuple

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
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
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


# ============================================
# CURATED HIERARCHY (from CLASS_MAPPINGS)
# ============================================
#
# CLASS_MAPPINGS above is a hand-written statement of what each source class
# MEANS, and most of it describes subsumption rather than equivalence.
#
# When two or more distinct source classes map to one target, equivalence is
# not merely imprecise, it is false: HiresRate = RateMeasurement and
# QuitsRate = RateMeasurement together imply HiresRate = QuitsRate. What the
# table actually says is that each is a KIND of RateMeasurement -- five of
# them, plus four ChangeMeasurement, four LevelMeasurement, and so on.
#
# So those relationships are published as rdfs:subClassOf, and only the
# genuine one-to-one pairs stay owl:equivalentClass. This is the better
# source of hierarchy: a person decided each entry, so unlike the naming
# rules below it cannot invert the meaning of a name it misreads.
#
# Unlike the naming rules, a curated parent need NOT appear in the data --
# RateMeasurement is an enrichment class that may never be instantiated, and
# RDFS allows a superclass with no direct instances.


def partition_class_mappings(
    mappings: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Split CLASS_MAPPINGS into (subsumptions, equivalences).

    A target claimed by two or more sources cannot be equivalent to any of
    them, so every one of its sources is a subclass. A target with exactly
    one source is left as a candidate equivalence.

    Returns two {source: target} dicts, together covering ``mappings``.
    """
    sources_per_target: Dict[str, List[str]] = {}
    for source, target in mappings.items():
        sources_per_target.setdefault(target, []).append(source)

    subsumptions, equivalences = {}, {}
    for source, target in mappings.items():
        if len(sources_per_target[target]) > 1:
            subsumptions[source] = target
        else:
            equivalences[source] = target
    return subsumptions, equivalences


def resolve_class_mappings(
    overrides: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """
    Merge caller-supplied class mappings over the built-in table.

    The built-in table covers the sources this project ships with. Adding a
    source should not require editing this module, so a run can pass its own:

      * ``{"<source uri>": "<target uri>"}`` adds or overrides one entry;
      * ``{"<source uri>": None}`` drops a built-in entry.

    Merging rather than replacing is deliberate — the common case is adding a
    vocabulary, not disowning the existing ones. Pass ``None`` targets to
    remove what does not apply.
    """
    merged = dict(CLASS_MAPPINGS)
    for source, target in (overrides or {}).items():
        if target is None:
            merged.pop(source, None)
        else:
            merged[source] = target
    return merged


# The defaults, for callers that want the built-in reading without a mapper.
CURATED_SUBCLASSES, TRUE_EQUIVALENCES = partition_class_mappings(
    CLASS_MAPPINGS
)


# ============================================
# CLASS HIERARCHY DERIVATION
# ============================================
#
# No source declares rdfs:subClassOf -- not the real SEC/BLS/NOAA/market data,
# not either e2e fixture set. The class_hierarchy sub-segment (64 dims of a
# 1024-d node vector) was therefore zero in every build. The hierarchy is not
# absent from the data though, only unasserted: the class names encode it.
#
# These rules recover it. Both are deliberately conservative -- a parent is
# only ever an EXISTING class from the same namespace, never an invented URI,
# so a rule can fail to find a parent but cannot fabricate one. A wrong
# subClassOf is worse than a missing one: it feeds fake structure into 64 dims
# that the GNN will treat as fact.
#
# What is recovered is recorded in ontology_schema.json as
# hierarchy_source="derived from class naming", never as "rdfs:subClassOf", so
# no consumer mistakes inference for a declaration.

# A measurement class specialises its dataset class: HiresRate is a kind of
# HiresData. Applied only when the "<prefix>Data" class actually exists.
_MEASURE_SUFFIXES = ("Level", "Rate")
_DATASET_SUFFIX = "Data"

# BLS names a series by what it EXCLUDES as readily as by what it contains:
# AllItemsLessShelter, DistilledSpiritsExcludingWhiskeyAtHome. Rule 2 reads
# those backwards -- "AllItemsLessShelter ends with Shelter, so it must be a
# kind of Shelter" -- when the name asserts the exact opposite. Found by
# running the rules over the e2e fixture classes, where the suffix rule
# produced AllItemsLessShelter -> Shelter and two more like it.
#
# A qualifier containing any of these negates rather than narrows, so the
# edge is dropped. This is deliberately blunt: it also drops
# CommoditiesLessFoodAndEnergyCommodities -> Commodities, which happens to be
# a correct subset relation. A missing superclass costs recall in 64 dims; a
# wrong one teaches the GNN a relationship that is false.
_NEGATING_QUALIFIERS = ("Less", "Excluding", "Without", "Except")


def _split_uri(uri: str) -> Tuple[str, str]:
    """Split a URI into (namespace, local name) on the last # or /."""
    for sep in ("#", "/"):
        head, found, tail = uri.rpartition(sep)
        if found and tail:
            return head + sep, tail
    return "", uri


def derive_subclass_edges(class_uris: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Infer (child, parent) rdfs:subClassOf pairs from class naming.

    Two rules, both scoped to a single namespace:

    1. Measurement -> dataset. ``<X>Level`` and ``<X>Rate`` are subclasses of
       ``<X>Data`` when that class exists (HiresRate -> HiresData).
    2. Qualifier -> base. A class whose local name ends with another class's
       full local name specialises it (UnadjustedPercentChange ->
       PercentChange, OtherSeparationsData -> SeparationsData) -- unless the
       qualifier negates rather than narrows (AllItemsLessShelter is not a
       kind of Shelter).

    Multiple parents are allowed -- RDFS permits it, and the two rules
    legitimately disagree about which axis a class specialises along.

    Returns edges sorted for determinism. Raises ValueError if the result is
    not a DAG; the encoder walks this to a transitive closure with a depth,
    which a cycle would not terminate.
    """
    classes = sorted(set(class_uris))
    parts = {c: _split_uri(c) for c in classes}
    by_ns: Dict[str, Dict[str, str]] = {}
    for uri, (ns, local) in parts.items():
        by_ns.setdefault(ns, {})[local] = uri

    edges = set()
    for child, (ns, local) in parts.items():
        siblings = by_ns[ns]

        # Rule 1 — measurement specialises its dataset.
        for suffix in _MEASURE_SUFFIXES:
            if local.endswith(suffix):
                parent_local = local[: -len(suffix)] + _DATASET_SUFFIX
                parent = siblings.get(parent_local)
                if parent is not None and parent != child:
                    edges.add((child, parent))

        # Rule 2 — qualified name specialises the name it qualifies, unless
        # the qualifier negates it (AllItemsLessShelter is NOT a Shelter).
        for other_local, parent in siblings.items():
            if other_local == local or not local.endswith(other_local):
                continue
            qualifier = local[: -len(other_local)]
            if any(neg in qualifier for neg in _NEGATING_QUALIFIERS):
                continue
            edges.add((child, parent))

    result = sorted(edges)
    _assert_is_dag(result)
    return result


def _assert_is_dag(edges: List[Tuple[str, str]]) -> None:
    """Raise if the child->parent edges contain a cycle.

    Enforced rather than merely tested: the rules are string-shaped, so a
    future suffix could quietly introduce A -> B -> A, and the encoder's
    transitive closure would not terminate on it.
    """
    parents: Dict[str, List[str]] = {}
    for child, parent in edges:
        parents.setdefault(child, []).append(parent)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {}

    def visit(node: str, trail: List[str]) -> None:
        colour[node] = GREY
        for parent in parents.get(node, []):
            state = colour.get(parent, WHITE)
            if state == GREY:
                cycle = " -> ".join(trail + [node, parent])
                raise ValueError(
                    f"derived class hierarchy contains a cycle: {cycle}"
                )
            if state == WHITE:
                visit(parent, trail + [node])
        colour[node] = BLACK

    for node in list(parents):
        if colour.get(node, WHITE) == WHITE:
            visit(node, [])


class OntologyMapper:
    """
    Creates ontology equivalence mappings and label normalization
    using PySpark DataFrames.

    All mappings are static configuration — no data-dependent queries
    except for the label normalization step which reads rdfs:label
    triples from the graph.
    """

    def __init__(
        self,
        spark: SparkSession,
        class_mappings: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        Args:
            spark: active session
            class_mappings: per-run additions/overrides to the built-in
                CLASS_MAPPINGS table; see resolve_class_mappings(). A value
                of None for a source drops that built-in entry.
        """
        self.spark = spark
        self.class_mappings = resolve_class_mappings(class_mappings)
        # Re-read per instance: an override can turn a one-to-one pair into a
        # shared target (or the reverse), which moves it between subsumption
        # and equivalence.
        self.curated_subclasses, self.true_equivalences = (
            partition_class_mappings(self.class_mappings)
        )

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

        logger.info("[Step 1/5] Creating property equivalences...")
        df = self._create_property_equivalences()
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/5] Creating class equivalences...")
        df = self._create_class_equivalences()
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 3/5] Folding predicates to their unified form...")
        df = self._fold_predicates_to_unified(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 4/5] Normalizing labels (skos:prefLabel)...")
        df = self._normalize_labels(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 5/5] Deriving class hierarchy (rdfs:subClassOf)...")
        df = self._derive_class_hierarchy(triples_df)
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
        if not self.true_equivalences:
            return None

        # Only the one-to-one pairs. Where several sources share a target the
        # relationship is subsumption, and _derive_class_hierarchy publishes
        # it as rdfs:subClassOf instead -- asserting both here would state
        # that those sources are equivalent to each other.
        rows = [
            (source, OWL_EQUIVALENT_CLASS, target)
            for source, target in self.true_equivalences.items()
        ]

        df = self.spark.createDataFrame(
            rows, ["subject", "predicate", "object"]
        )

        logger.info(f"  Created {len(rows)} class equivalences")
        return df

    # ================================================================
    # Step 5: Class Hierarchy Derivation
    # ================================================================

    def _derive_class_hierarchy(
        self,
        triples_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Emit rdfs:subClassOf triples inferred from class naming.

        The classes are the distinct objects of rdf:type — a small driver-side
        collect (44 on the real data, <500 for any plausible source set), so
        the rules themselves stay pure Python and directly testable.
        """
        class_rows = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select("object")
            .distinct()
            .collect()
        )
        class_uris = [row["object"] for row in class_rows]

        # Curated first — a person wrote these, so they outrank a guess from
        # spelling. Emitted for every mapped source whether or not the class
        # appears in this load, matching how the equivalences already behave.
        curated = sorted(self.curated_subclasses.items())
        inferred = derive_subclass_edges(class_uris)

        edges = sorted(set(curated) | set(inferred))
        if not edges:
            logger.info(
                f"  No hierarchy derivable from {len(class_uris)} class(es)"
            )
            return None

        # The two sources are independent, so neither can rule out a cycle
        # across the union of them.
        _assert_is_dag(edges)

        rows = [(child, RDFS_SUBCLASS_OF, parent) for child, parent in edges]
        logger.info(
            f"  Hierarchy: {len(curated)} curated + "
            f"{len(set(inferred) - set(curated))} inferred from naming "
            f"= {len(rows)} subClassOf edge(s) over {len(class_uris)} "
            f"class(es); {len({c for c, _ in edges})} class(es) gained a "
            f"superclass"
        )
        return self.spark.createDataFrame(
            rows, ["subject", "predicate", "object"]
        )

    # ================================================================
    # Step 3: Predicate Folding
    # ================================================================

    def _fold_predicates_to_unified(
        self,
        triples_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Restate triples whose predicate has a unified form under that form.

        This is the step that makes the equivalences mean something. Emitting
        ``cpi:hasMonth owl:equivalentProperty unified:hasMonth`` declares a
        relationship but changes nothing downstream: nothing in this project
        reasons over OWL, and the feature encoder hashes the predicate URI it
        is given, so cpi:hasMonth and jolts:hasMonth landed in two unrelated
        slots however many equivalence triples were present. Ten source month
        predicates meant ten slots for one concept, and a model could not see
        that two sources were talking about the same thing.

        Restating the triple under the unified predicate gives that concept ONE
        slot shared across sources. Measured on the e2e fixtures, 22% of
        triples carry a foldable predicate, and four different month predicates
        (cpi/jolts/eci/empsit, 200 triples each) collapse onto unified:hasMonth.

        ADDITIVE, not a rewrite. The source predicate stays, for two reasons:
        every other enrichment phase only adds triples, and per-source
        predicates are a deliberate design decision (see the canonical-type
        note in the README) -- sources agree on what a year *is* without being
        forced to share measurement semantics. So a folded fact is available
        both ways: cpi:hasMonth for source-specific signal, unified:hasMonth
        for the cross-source view.

        Unlike the two equivalence steps, this one is data-driven: only
        predicates actually present produce output, so loading one source does
        not emit anything for vocabularies that are absent.
        """
        if not PROPERTY_MAPPINGS:
            return None

        mapping_df = self.spark.createDataFrame(
            list(PROPERTY_MAPPINGS.items()),
            ["predicate", "unified_predicate"],
        )

        # Broadcast: 46 rows against the full triple set.
        folded = (
            triples_df
            .join(F.broadcast(mapping_df), "predicate", "inner")
            .select(
                F.col("subject"),
                F.col("unified_predicate").alias("predicate"),
                F.col("object"),
            )
            .distinct()
        )

        if folded.head(1) == []:
            logger.info("  No foldable predicates present")
            return None

        logger.info("  Predicate folding complete")
        return folded

    # ================================================================
    # Step 4: Label Normalization
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