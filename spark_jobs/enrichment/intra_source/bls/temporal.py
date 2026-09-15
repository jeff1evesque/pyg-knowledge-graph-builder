"""BLS periods: the month, year and quarter URIs the BLS datasets state.

BLS states its periods as URIs (id/cpi/November, id/wkyeng/Q1), not as date
literals, so the temporal unifier cannot find them through a date predicate.
The BLS source spec hands these collectors to the unifier instead.
"""
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_jobs.enrichment.temporal_unifier import MONTH_NAMES
from spark_jobs.utils.namespaces import (
    CPI,
    ECI,
    EMPSIT,
    JOLTS,
    LAUS,
    METRO,
    PPI,
    REALER,
    WKYENG,
    XIMPIM,
    identifier_namespace,
)

# BLS monthly dataset IDENTIFIER prefixes.
#
# These match period URIs -- id/cpi/February, id/jolts/2024 -- which are
# individuals, not terms. Keying them on the term namespaces (ontology/cpi/)
# matched nothing at all: no BLS period was ever collected, so none was typed
# temporal:SourceMonth, so node_mapper's _CANONICAL_TYPE_PRIORITY had nothing
# to prefer and every period sharded across cpi_Month / jolts_Month /
# empsit_Month / eci_Month. The graph kept its per-source month nodes and lost
# the cross-source bridge, without anything raising.
BLS_MONTHLY_PREFIXES = [
    identifier_namespace(str(ns))
    for ns in (CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER)
]

# BLS quarterly dataset predicates
WKYENG_HAS_QUARTER = str(WKYENG.hasQuarter)
WKYENG_HAS_YEAR = str(WKYENG.hasYear)


def collect_bls_periods(triples_df: DataFrame) -> List[DataFrame]:
    """Each BLS period frame that found something: months and years, then
    quarters and their years."""
    frames = []
    for collect in (collect_bls_months_years, collect_bls_quarters):
        frame = collect(triples_df)
        if frame is not None:
            frames.append(frame)
    return frames


def collect_bls_months_years(triples_df: DataFrame) -> Optional[DataFrame]:
    """
    Collect month and year URIs from BLS monthly datasets.

    BLS datasets use URI-based temporal entities like:
      id/cpi/November, id/ppi/November, id/cpi/2024, id/ppi/2024

    We find these by looking for URIs under BLS IDENTIFIER prefixes whose
    local name matches a month name or 4-digit year. Identifier, not term:
    periods are things, and nothing is ever minted under ontology/cpi/.
    """
    # Build filter: object URI starts with any BLS monthly prefix
    # and is used as an object in any triple (i.e., referenced as a value)
    bls_prefix_filter = F.lit(False)
    for prefix in BLS_MONTHLY_PREFIXES:
        bls_prefix_filter = bls_prefix_filter | F.col("object").startswith(prefix)

    bls_objects = (
        triples_df
        .filter(bls_prefix_filter)
        .select(F.col("object").alias("temporal_uri"))
        .dropDuplicates()
    )

    if bls_objects.head(1) == []:
        return None

    # Extract local name (everything after the last "/")
    bls_objects = bls_objects.withColumn(
        "local_name",
        F.regexp_extract(F.col("temporal_uri"), r"([^/]+)$", 1)
    )

    # Month URIs: local name is a valid month name
    month_names_str = "|".join(MONTH_NAMES)
    months = bls_objects.filter(
        F.col("local_name").rlike(f"^({month_names_str})$")
    ).select(
        F.col("temporal_uri"),
        F.col("local_name").alias("normalized_name"),
        F.lit("month").alias("kind"),
    )

    # Year URIs: local name is a 4-digit number
    years = bls_objects.filter(
        F.col("local_name").rlike(r"^\d{4}$")
    ).select(
        F.col("temporal_uri"),
        F.col("local_name").alias("normalized_name"),
        F.lit("year").alias("kind"),
    )

    result = months.unionAll(years)
    if result.head(1) == []:
        return None

    return result


def collect_bls_quarters(triples_df: DataFrame) -> Optional[DataFrame]:
    """
    Collect quarter and year URIs from WKYENG (quarterly BLS dataset).

    WKYENG uses:
      ?entity wkyeng:hasQuarter wkyeng:Q1
      ?entity wkyeng:hasYear wkyeng:2024
    """
    # Quarters: objects of wkyeng:hasQuarter
    quarters = triples_df.filter(
        F.col("predicate") == WKYENG_HAS_QUARTER
    ).select(
        F.col("object").alias("temporal_uri")
    ).dropDuplicates().withColumn(
        "local_name",
        F.regexp_extract(F.col("temporal_uri"), r"([^/]+)$", 1)
    ).filter(
        F.col("local_name").isin(["Q1", "Q2", "Q3", "Q4"])
    ).select(
        F.col("temporal_uri"),
        F.col("local_name").alias("normalized_name"),
        F.lit("quarter").alias("kind"),
    )

    # Years: objects of wkyeng:hasYear
    years = triples_df.filter(
        F.col("predicate") == WKYENG_HAS_YEAR
    ).select(
        F.col("object").alias("temporal_uri")
    ).dropDuplicates().withColumn(
        "local_name",
        F.regexp_extract(F.col("temporal_uri"), r"([^/]+)$", 1)
    ).filter(
        F.col("local_name").rlike(r"^\d{4}$")
    ).select(
        F.col("temporal_uri"),
        F.col("local_name").alias("normalized_name"),
        F.lit("year").alias("kind"),
    )

    result = quarters.unionAll(years)
    if result.head(1) == []:
        return None

    return result
