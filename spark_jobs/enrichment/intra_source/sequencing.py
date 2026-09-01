"""
Shared windowed sequencing for intra-source enrichment.

Every source builds its "this came before that" chains the same way: partition
the rows, order them, and read the next row with lead(). That shape has one
failure mode. If an entity appears twice inside a partition, ordering puts the
copies next to each other and lead() hands a row its own entity back, so the
step emits "X precedes X" -- an edge saying a thing came before itself.

Duplicates are easy to introduce upstream. BLS resolves each entity's month and
year with two inner joins, and either one fans a row out when an entity carries
more than one value for that property.

This module is the one place that shape lives. It drops duplicates first, so
the window sees each entity once per partition, then rejects any pair whose
ends are equal. The second guard is redundant while the first is correct, and
it is what fails loudly if a caller partitions on columns that do not determine
the dedupe grain.
"""
from typing import Sequence, Union

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F


# Temp column for the dedupe rank. Named to avoid colliding with caller columns.
_RANK_COL = "_sequencing_rank"


def sequence_within_partitions(
    df: DataFrame,
    entity_col: str,
    partition_cols: Sequence[str],
    order_cols: Sequence[Union[str, Column]],
    next_col: str = "next_entity",
) -> DataFrame:
    """Pair each row with the next entity in its partition.

    Args:
        df: rows to sequence
        entity_col: column holding the entity URI
        partition_cols: columns that define one chain
        order_cols: columns that order a chain, most significant first
        next_col: name for the added column holding the next entity

    Returns:
        DataFrame of the input columns plus `next_col`, keeping only rows that
        have a successor. No row has its own entity as its successor.
    """
    partition_cols = list(partition_cols)
    order_cols = list(order_cols)

    # One row per entity per partition, chosen deterministically.
    #
    # dropDuplicates is shorter but keeps an arbitrary copy. When copies
    # disagree on an ordering column -- two months on one measurement, say --
    # that makes the chain itself differ between runs. Ranking and taking the
    # first keeps the output reproducible.
    dedupe_window = Window.partitionBy(
        *partition_cols, entity_col
    ).orderBy(*order_cols)

    deduped = (
        df
        .withColumn(_RANK_COL, F.row_number().over(dedupe_window))
        .filter(F.col(_RANK_COL) == 1)
        .drop(_RANK_COL)
    )

    sequence_window = Window.partitionBy(*partition_cols).orderBy(*order_cols)

    return (
        deduped
        .withColumn(next_col, F.lead(entity_col).over(sequence_window))
        .filter(F.col(next_col).isNotNull())
        .filter(F.col(entity_col) != F.col(next_col))
    )


def sequence_to_triples(
    df: DataFrame,
    entity_col: str,
    predicate: str,
    partition_cols: Sequence[str],
    order_cols: Sequence[Union[str, Column]],
) -> DataFrame:
    """Sequence rows and emit them as (subject, predicate, object) triples.

    The common case. Callers that need the paired rows for something else --
    vertical spreads split one window across two predicates -- should use
    sequence_within_partitions directly.
    """
    paired = sequence_within_partitions(
        df,
        entity_col=entity_col,
        partition_cols=partition_cols,
        order_cols=order_cols,
    )

    return paired.select(
        F.col(entity_col).alias("subject"),
        F.lit(predicate).alias("predicate"),
        F.col("next_entity").alias("object"),
    )
