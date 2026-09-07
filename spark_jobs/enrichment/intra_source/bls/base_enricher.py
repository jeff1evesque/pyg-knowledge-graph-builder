"""
Base class for BLS per-dataset enrichment (PySpark) + dataset registry.

Each dataset's enrichment is driven by its MEASUREMENT_TYPES config.
The common pattern across all BLS datasets:
1. For each measurement type in MEASUREMENT_TYPES[dataset]:
   a. Find all entities of that rdf:type
   b. Resolve their category (grouping) property
   c. Resolve their month/year (or quarter/year) temporal properties
   d. Build a composite sort key (year + month or year + quarter)
   e. Window partition by category, order by sort key
   f. Produce precedes triples linking consecutive measurements

This base class provides the reusable machinery. All 10 BLS datasets
use it directly via the DATASET_ENRICHERS registry. If a dataset ever
needs custom logic, replace its registry entry with a subclass.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from functools import reduce
from typing import Dict, List, Optional
from rdflib.namespace import RDF
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT
from spark_jobs.enrichment.intra_source.bls.measurements import MEASUREMENT_TYPES
from spark_jobs.enrichment.intra_source.sequencing import sequence_to_triples

import logging

logger = logging.getLogger(__name__)

_RDF_TYPE = str(RDF.type)
_PRECEDES = str(BLS_ENRICHMENT.precedes)

# Month name → sort order for temporal ordering
MONTH_ORDER = {
    'January': '01', 'February': '02', 'March': '03', 'April': '04',
    'May': '05', 'June': '06', 'July': '07', 'August': '08',
    'September': '09', 'October': '10', 'November': '11', 'December': '12',
}

# Quarter → sort order
QUARTER_ORDER = {
    'Q1': '1', 'Q2': '2', 'Q3': '3', 'Q4': '4',
}


class BLSDatasetEnricher:
    """
    BLS per-dataset enricher.

    Instantiated with a dataset_name, reads measurement type configs
    from MEASUREMENT_TYPES, and produces precedes triples via PySpark
    window functions.

    Subclasses set `dataset_name` and optionally override
    `link_temporal_sequences` for dataset-specific behavior.
    """

    def __init__(self, spark: SparkSession, dataset_name: str):
        self.spark = spark
        self.dataset_name = dataset_name

    def link_temporal_sequences(self, bls_triples: DataFrame) -> Optional[DataFrame]:
        """
        Link temporal sequences for all measurement types in this dataset.

        For each measurement type:
        - Find entities of that type
        - Resolve category (grouping) and temporal (month/year or quarter/year)
        - Order by time within each category
        - Produce precedes triples

        Args:
            bls_triples: Cached DataFrame of BLS-only triples

        Returns:
            DataFrame of new precedes triples, or None
        """
        measurement_configs = MEASUREMENT_TYPES.get(self.dataset_name, {})
        if not measurement_configs:
            return None

        # Build month name → sort key mapping as a broadcast DataFrame
        month_rows = list(MONTH_ORDER.items())
        month_df = self.spark.createDataFrame(month_rows, ["month_name", "month_sort"])

        # Build quarter → sort key mapping
        quarter_rows = list(QUARTER_ORDER.items())
        quarter_df = self.spark.createDataFrame(quarter_rows, ["quarter_name", "quarter_sort"])

        new_dfs: List[DataFrame] = []

        for mtype_name, config in measurement_configs.items():
            result = self._link_single_measurement_type(
                bls_triples, mtype_name, config, month_df, quarter_df
            )
            if result is not None:
                new_dfs.append(result)

        if not new_dfs:
            return None

        return reduce(DataFrame.unionAll, new_dfs)

    def _link_single_measurement_type(
        self,
        bls_triples: DataFrame,
        mtype_name: str,
        config: Dict,
        month_df: DataFrame,
        quarter_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Link temporal sequences for a single measurement type.

        Args:
            bls_triples: BLS triples DataFrame
            mtype_name: Name of the measurement type (for logging)
            config: Measurement type config from MEASUREMENT_TYPES
            month_df: Broadcast DataFrame of month_name → month_sort
            quarter_df: Broadcast DataFrame of quarter_name → quarter_sort
        """
        type_uri = str(config['class'])
        category_prop = config.get('category_property')
        month_prop = config.get('month_property')
        year_prop = config.get('year_property')
        quarter_prop = config.get('quarter_property')

        # Must have at least a category and some temporal property
        if not category_prop:
            return None

        category_prop_str = str(category_prop)
        year_prop_str = str(year_prop) if year_prop else None

        is_quarterly = quarter_prop is not None and month_prop is None

        # Step 1: Find entities of this type
        entities = (
            bls_triples
            .filter(
                (F.col("predicate") == _RDF_TYPE) &
                (F.col("object") == type_uri)
            )
            .select(F.col("subject").alias("entity"))
            .distinct()
        )

        if entities.head(1) == []:
            return None

        # Step 2: Resolve category
        categories = (
            bls_triples
            .filter(F.col("predicate") == category_prop_str)
            .select(
                F.col("subject").alias("entity"),
                F.col("object").alias("category"),
            )
        )

        entities_with_cat = entities.join(categories, "entity", "inner")

        # Step 3: Resolve temporal properties
        if is_quarterly:
            # Quarter-based temporal (WKYENG)
            quarter_prop_str = str(quarter_prop)

            quarter_values = (
                bls_triples
                .filter(F.col("predicate") == quarter_prop_str)
                .select(
                    F.col("subject").alias("entity"),
                    F.col("object").alias("quarter_uri"),
                )
            )

            # Quarter name is the last URI segment. substring_index, not a
            # regex: RAPIDS will not put an end anchor after a variable-length
            # match on the GPU, and this sits in the query that costs BLS most
            # of its window. #380
            quarter_values = quarter_values.withColumn(
                "quarter_name",
                F.substring_index(F.col("quarter_uri"), "/", -1)
            )

            entities_with_time = entities_with_cat.join(quarter_values, "entity", "inner")

            # Resolve year
            if year_prop_str:
                year_values = (
                    bls_triples
                    .filter(F.col("predicate") == year_prop_str)
                    .select(
                        F.col("subject").alias("entity"),
                        F.col("object").alias("year_uri"),
                    )
                )
                year_values = year_values.withColumn(
                    "year_value",
                    F.regexp_extract(F.col("year_uri"), r"(\d{4})", 1)
                )
                entities_with_time = entities_with_time.join(year_values, "entity", "inner")
            else:
                return None

            # Build sort key: year + quarter
            entities_with_time = (
                entities_with_time
                .join(F.broadcast(quarter_df), "quarter_name", "inner")
                .withColumn(
                    "sort_key",
                    F.concat(F.col("year_value"), F.lit("-"), F.col("quarter_sort"))
                )
            )

        else:
            # Month-based temporal (most BLS datasets)
            if not month_prop:
                return None

            month_prop_str = str(month_prop)

            month_values = (
                bls_triples
                .filter(F.col("predicate") == month_prop_str)
                .select(
                    F.col("subject").alias("entity"),
                    F.col("object").alias("month_uri"),
                )
            )

            # Month name is the last URI segment -- same rewrite as the
            # quarterly branch above, and the one that runs for nine of the ten
            # BLS datasets. #380
            month_values = month_values.withColumn(
                "month_name",
                F.substring_index(F.col("month_uri"), "/", -1)
            )

            entities_with_time = entities_with_cat.join(month_values, "entity", "inner")

            # Resolve year
            if year_prop_str:
                year_values = (
                    bls_triples
                    .filter(F.col("predicate") == year_prop_str)
                    .select(
                        F.col("subject").alias("entity"),
                        F.col("object").alias("year_uri"),
                    )
                )
                year_values = year_values.withColumn(
                    "year_value",
                    F.regexp_extract(F.col("year_uri"), r"(\d{4})", 1)
                )
                entities_with_time = entities_with_time.join(year_values, "entity", "inner")
            else:
                return None

            # Build sort key: year + month_sort
            entities_with_time = (
                entities_with_time
                .join(F.broadcast(month_df), "month_name", "inner")
                .withColumn(
                    "sort_key",
                    F.concat(F.col("year_value"), F.lit("-"), F.col("month_sort"))
                )
            )

        # Step 4: Sequence each category into precedes triples.
        #
        # The month and year joins above are inner joins on entity, so an
        # entity carrying two months or two years arrives here as two rows in
        # one category. The helper drops those copies before the window; left
        # in, they sorted next to each other and lead() made the measurement
        # precede itself (#360).
        return sequence_to_triples(
            entities_with_time,
            entity_col="entity",
            predicate=_PRECEDES,
            partition_cols=["category"],
            order_cols=["sort_key"],
        )


# ============================================
# Dataset registry
#
# All 10 BLS datasets use BLSDatasetEnricher directly.
# If a dataset ever needs custom logic, replace its entry
# with a subclass.
# ============================================

DATASET_ENRICHERS: Dict[str, type] = {
    'cpi': BLSDatasetEnricher,
    'ppi': BLSDatasetEnricher,
    'eci': BLSDatasetEnricher,
    'jolts': BLSDatasetEnricher,
    'empsit': BLSDatasetEnricher,
    'ximpim': BLSDatasetEnricher,
    'laus': BLSDatasetEnricher,
    'metro': BLSDatasetEnricher,
    'realer': BLSDatasetEnricher,
    'wkyeng': BLSDatasetEnricher,
}