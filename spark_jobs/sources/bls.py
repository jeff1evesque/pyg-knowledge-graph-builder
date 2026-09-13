"""BLS: ten economic indicator datasets, one vocabulary each, and their shared classes."""
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import (
    BLS_COMMON,
    BLS_ENRICHMENT,
    CPI,
    ECI,
    EMPSIT,
    JOLTS,
    LAUS,
    METRO,
    PPI,
    REALER,
    UNIFIED,
    WKYENG,
    XIMPIM,
)


def _linker(spark, options):
    from spark_jobs.enrichment.intra_source.bls_linker import BLSIntraSourceLinker

    return BLSIntraSourceLinker(spark)


def _collect_periods(triples_df):
    from spark_jobs.enrichment.intra_source.bls.temporal import collect_bls_periods

    return collect_bls_periods(triples_df)


SPEC = SourceSpec(
    name="bls",
    label="BLS",
    path_fragments=("source=bls",),
    namespaces=(
        (str(CPI), "cpi"),
        (str(PPI), "ppi"),
        (str(ECI), "eci"),
        (str(EMPSIT), "empsit"),
        (str(JOLTS), "jolts"),
        (str(LAUS), "laus"),
        (str(METRO), "metro"),
        (str(REALER), "realer"),
        (str(WKYENG), "wkyeng"),
        (str(XIMPIM), "ximpim"),
        (str(BLS_COMMON), "bls_common"),
        (str(BLS_ENRICHMENT), "bls_enrichment"),
    ),
    enrichment_namespace=str(BLS_ENRICHMENT),
    # No date predicates: BLS states its periods as URIs (id/cpi/February),
    # which _collect_periods reads directly.
    temporal_collector=_collect_periods,
    linker=_linker,
    property_mappings={
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

        str(CPI.indexValue): str(UNIFIED.measurementValue),
        str(PPI.changeValue): str(UNIFIED.measurementValue),
        str(PPI.indexValue): str(UNIFIED.measurementValue),
        # jolts states levelValue and rateValue. Bare level and rate were never
        # upstream terms.
        str(JOLTS.levelValue): str(UNIFIED.measurementValue),
        str(JOLTS.rateValue): str(UNIFIED.measurementValue),
        str(EMPSIT.value): str(UNIFIED.measurementValue),
        str(ECI.indexValue): str(UNIFIED.measurementValue),

        str(CPI.hasCategory): str(UNIFIED.hasCategory),
        str(PPI.hasCommodityGrouping): str(UNIFIED.hasCategory),
        str(ECI.hasOccupationalGroup): str(UNIFIED.hasCategory),
        str(JOLTS.hasIndustry): str(UNIFIED.hasCategory),
        str(EMPSIT.hasIndustry): str(UNIFIED.hasCategory),
        # empsit declares hasCategory, but the dimension link it writes is
        # hasLaborForceCategory.
        str(EMPSIT.hasLaborForceCategory): str(UNIFIED.hasCategory),

        str(LAUS.hasState): str(UNIFIED.hasRegion),
        # metro states its area through hasRegion. hasMetropolitanArea is
        # declared and never written.
        str(METRO.hasRegion): str(UNIFIED.hasRegion),
    },
    class_mappings={
        str(CPI.Index): str(BLS_ENRICHMENT.PriceIndex),
        str(PPI.IndexValue): str(BLS_ENRICHMENT.PriceIndex),

        str(JOLTS.JobOpeningsRate): str(BLS_ENRICHMENT.RateMeasurement),
        str(JOLTS.HiresRate): str(BLS_ENRICHMENT.RateMeasurement),
        str(JOLTS.QuitsRate): str(BLS_ENRICHMENT.RateMeasurement),
        str(LAUS.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),
        str(METRO.UnemploymentRate): str(BLS_ENRICHMENT.RateMeasurement),

        # The windowed change classes. The declared umbrellas
        # (cpi:PercentChange, eci:PercentChangeData) are on no instance.
        str(CPI.OneMonthPercentChange): str(BLS_ENRICHMENT.ChangeMeasurement),
        str(CPI.TwelveMonthPercentChange): str(BLS_ENRICHMENT.ChangeMeasurement),
        str(PPI.MonthlyChange): str(BLS_ENRICHMENT.ChangeMeasurement),
        str(PPI.TwelveMonthChange): str(BLS_ENRICHMENT.ChangeMeasurement),
        str(ECI.ThreeMonthPercentChangeData): str(BLS_ENRICHMENT.ChangeMeasurement),
        str(ECI.TwelveMonthPercentChangeData): str(BLS_ENRICHMENT.ChangeMeasurement),

        str(JOLTS.JobOpeningsLevel): str(BLS_ENRICHMENT.LevelMeasurement),
        str(JOLTS.HiresLevel): str(BLS_ENRICHMENT.LevelMeasurement),
        str(EMPSIT.EmployeeCount): str(BLS_ENRICHMENT.LevelMeasurement),
        str(LAUS.LaborForceData): str(BLS_ENRICHMENT.LevelMeasurement),

        str(CPI.Category): str(BLS_ENRICHMENT.EconomicIndicator),
        # ppi types its groupings ppi:Grouping. CommodityGrouping is not a
        # declared class.
        str(PPI.Grouping): str(BLS_ENRICHMENT.EconomicIndicator),

        str(JOLTS.Industry): str(BLS_ENRICHMENT.IndustryClassification),
        str(EMPSIT.Industry): str(BLS_ENRICHMENT.IndustryClassification),
        str(ECI.Industry): str(BLS_ENRICHMENT.IndustryClassification),

        str(ECI.OccupationalGroup): str(BLS_ENRICHMENT.OccupationalClassification),
        # empsit's closest thing to an occupational grouping. Its Occupation
        # classes are declared and never written.
        str(EMPSIT.LaborForceCategory): str(BLS_ENRICHMENT.OccupationalClassification),
    },
)
