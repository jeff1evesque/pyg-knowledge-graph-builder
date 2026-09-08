"""
The three segments of an edge feature vector, one function each.

Each takes the edges of one type and returns lazy (edge_idx, dim, value) sparse
entries -- nothing is computed here, and nothing reaches the driver. Every dim
index comes from the EdgeVectorLayout passed in, so the segment boundaries are
never written down twice.

EdgeFeatureExtractor calls all three and unions what comes back. It also owns
the broadcast decision, which segments 1 and 2 need for their endpoint property
joins; those two take it as ``maybe_broadcast`` rather than reading it off an
object, so a caller can hand them any policy and a test can hand them none.

The frame shapes the encoders consume are described here as well -- PropertyRows
and the row widths and predicate patterns beside it. The extractor builds those
frames and imports the descriptions from here, so the dependency runs one way:
extractor -> encoders, never back.
"""
from typing import Callable, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_jobs.pyg_builder.edge_vector_layout import EdgeVectorLayout


# Hash seeds for deterministic encoding
_EDGE_HASH_SEEDS = [0, 7, 13, 31]

# Predicate patterns that identify a month or a year property. Named here
# rather than inside encode_temporal_signals so the extractor's
# _endpoint_property_rows can count exactly the rows that function filters to
# -- a count taken with a different pattern would size the wrong frame.
MONTH_PREDICATE_PATTERN = "(?i)(hasMonth|month|hasMonthNum|monthNumber)"
YEAR_PREDICATE_PATTERN = "(?i)(hasYear|year|hasYearNum|yearNumber)"

# Approximate width, in bytes, of one row of each endpoint property frame.
# Used to turn Spark's byte-denominated broadcast threshold into a row budget,
# since a row count is what we can measure cheaply.
#
# The temporal frames carry a long node id and a double value. The segment-2
# frame carries the predicate URI as well, which is most of its width -- these
# are full URIs, not local names.
_TEMPORAL_ROW_BYTES = 24
NUMERIC_ROW_BYTES = 128


class PropertyRows:
    """
    Row counts of the endpoint property frames for one node type.

    ``month`` and ``year`` are the subsets segment 1 filters to; ``total`` is
    the whole per-type frame segment 2 uses.
    """

    __slots__ = ("total", "month", "year")

    def __init__(self, total: int = 0, month: int = 0, year: int = 0):
        self.total = total
        self.month = month
        self.year = year


# ================================================================
# Segment 1: Temporal Signals encoding
# ================================================================

def encode_temporal_signals(
    layout: EdgeVectorLayout,
    maybe_broadcast: Callable[[DataFrame, int, int], DataFrame],
    edge_df: DataFrame,
    src_type: str,
    dst_type: str,
    category: str,
    numeric_props_df: DataFrame,
    src_rows: PropertyRows,
    dst_rows: PropertyRows,
) -> Optional[DataFrame]:
    """
    Encode Segment 1: time delta, period flags, and temporal
    direction between edge endpoints.

    For temporal edges: computes month/year delta from endpoint
    properties. Month and year values are extracted by regex-matching
    predicate URIs containing "month" or "year" patterns.

    For non-temporal edges: encodes a hash of the relation category
    into the time delta sub-segment as a type indicator, so the GNN
    can still distinguish edge categories in this segment.

    All dim indices are derived from ``layout``.

    Returns DataFrame(edge_idx: long, dim: int, value: float)
    or None.
    """
    parts: List[DataFrame] = []

    # --- Sub-segment 1a: Time Delta ---
    td_start = layout.seg1_time_delta_start
    td_dim = layout.seg1_time_delta_dim

    if category == "temporal":
        # Find month/year numeric properties by predicate pattern.
        # These patterns match predicates like cpi:hasMonth,
        # market:monthNumber, etc.
        month_preds = numeric_props_df.filter(
            F.col("predicate").rlike(MONTH_PREDICATE_PATTERN)
        )
        year_preds = numeric_props_df.filter(
            F.col("predicate").rlike(YEAR_PREDICATE_PATTERN)
        )

        # Source endpoint temporal properties
        src_month = (
            month_preds
            .filter(F.col("node_type") == src_type)
            .select(
                F.col("node_id").alias("_src_nid"),
                F.col("numeric_value").alias("src_month"),
            )
        )
        src_year = (
            year_preds
            .filter(F.col("node_type") == src_type)
            .select(
                F.col("node_id").alias("_src_nid"),
                F.col("numeric_value").alias("src_year"),
            )
        )

        # Destination endpoint temporal properties
        dst_month = (
            month_preds
            .filter(F.col("node_type") == dst_type)
            .select(
                F.col("node_id").alias("_dst_nid"),
                F.col("numeric_value").alias("dst_month"),
            )
        )
        dst_year = (
            year_preds
            .filter(F.col("node_type") == dst_type)
            .select(
                F.col("node_id").alias("_dst_nid"),
                F.col("numeric_value").alias("dst_year"),
            )
        )

        # Join temporal properties to edges (on executors).
        # Left joins ensure edges without temporal properties
        # still get zero-valued features rather than being dropped.
        # The endpoint property side is broadcast when it is small
        # enough to be — see _maybe_broadcast. For a node type of a
        # few thousand nodes that is every time, and it saves one
        # shuffle stage per join per edge type. For a node type of
        # ten million it is never, and the join shuffles.
        temporal_edges = (
            edge_df
            .join(
                maybe_broadcast(
                    src_month, src_rows.month, _TEMPORAL_ROW_BYTES
                ),
                edge_df["src_id"] == src_month["_src_nid"],
                "left",
            )
            .drop("_src_nid")
            .join(
                maybe_broadcast(
                    src_year, src_rows.year, _TEMPORAL_ROW_BYTES
                ),
                edge_df["src_id"] == src_year["_src_nid"],
                "left",
            )
            .drop("_src_nid")
            .join(
                maybe_broadcast(
                    dst_month, dst_rows.month, _TEMPORAL_ROW_BYTES
                ),
                edge_df["dst_id"] == dst_month["_dst_nid"],
                "left",
            )
            .drop("_dst_nid")
            .join(
                maybe_broadcast(
                    dst_year, dst_rows.year, _TEMPORAL_ROW_BYTES
                ),
                edge_df["dst_id"] == dst_year["_dst_nid"],
                "left",
            )
            .drop("_dst_nid")
        )

        # Compute month delta: (dst_year - src_year) * 12
        #                       + (dst_month - src_month)
        # This gives a signed integer representing the number of
        # months between endpoints. Positive = dst is later.
        temporal_edges = temporal_edges.withColumn(
            "month_delta",
            F.when(
                F.col("src_year").isNotNull()
                & F.col("dst_year").isNotNull()
                & F.col("src_month").isNotNull()
                & F.col("dst_month").isNotNull(),
                (
                    (F.col("dst_year") - F.col("src_year"))
                    * F.lit(12.0)
                    + (F.col("dst_month") - F.col("src_month"))
                ),
            ).otherwise(F.lit(0.0)),
        )

        # Normalize month delta: divide by 12 to put in
        # ~[-1, 1] range for typical monthly data.
        # A 1-month gap = 0.083, a 1-year gap = 1.0.
        temporal_edges = temporal_edges.withColumn(
            "norm_delta",
            F.col("month_delta") / F.lit(12.0),
        )

        # Signed normalized delta in a hashed slot
        td_encoded = temporal_edges.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.lit("time_delta"), F.lit(0)))
                % F.lit(td_dim)
                + F.lit(td_start)
            ).alias("dim"),
            F.col("norm_delta").alias("value"),
        )
        parts.append(td_encoded)

        # Absolute delta in a second slot — useful for the GNN
        # to learn "how far apart" regardless of direction
        if td_dim >= 2:
            td_abs_encoded = temporal_edges.select(
                F.col("edge_idx"),
                (
                    F.abs(F.hash(F.lit("abs_time_delta"), F.lit(7)))
                    % F.lit(td_dim)
                    + F.lit(td_start)
                ).alias("dim"),
                F.abs(F.col("norm_delta")).alias("value"),
            )
            parts.append(td_abs_encoded)

        # --- Sub-segment 1b: Period Flags ---
        pf_start = layout.seg1_period_flags_start
        pf_dim = layout.seg1_period_flags_dim

        # Same-year flag: 1.0 if both endpoints are in the same year
        same_year = temporal_edges.select(
            F.col("edge_idx"),
            F.lit(pf_start).alias("dim"),
            F.when(
                F.col("src_year").isNotNull()
                & F.col("dst_year").isNotNull()
                & (F.col("src_year") == F.col("dst_year")),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)).alias("value"),
        )
        parts.append(same_year)

        # Consecutive-month flag: 1.0 if endpoints are exactly
        # 1 month apart (the most common temporal edge pattern)
        if pf_dim >= 2:
            consecutive = temporal_edges.select(
                F.col("edge_idx"),
                F.lit(pf_start + 1).alias("dim"),
                F.when(
                    F.abs(F.col("month_delta")) == F.lit(1.0),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)).alias("value"),
            )
            parts.append(consecutive)

        # Same-quarter flag: 1.0 if both endpoints are in the
        # same calendar quarter and year
        if pf_dim >= 3:
            same_quarter = temporal_edges.select(
                F.col("edge_idx"),
                F.lit(pf_start + 2).alias("dim"),
                F.when(
                    F.col("src_month").isNotNull()
                    & F.col("dst_month").isNotNull()
                    & (
                        F.floor(
                            (F.col("src_month") - 1) / F.lit(3.0)
                        )
                        == F.floor(
                            (F.col("dst_month") - 1) / F.lit(3.0)
                        )
                    )
                    & (F.col("src_year") == F.col("dst_year")),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)).alias("value"),
            )
            parts.append(same_quarter)

        # --- Sub-segment 1c: Direction ---
        dir_start = layout.seg1_direction_start

        # Forward direction indicator: +1 if dst is later in time,
        # -1 if earlier, 0 if same or unknown
        direction = temporal_edges.select(
            F.col("edge_idx"),
            F.lit(dir_start).alias("dim"),
            F.when(
                F.col("month_delta") > 0, F.lit(1.0)
            ).when(
                F.col("month_delta") < 0, F.lit(-1.0)
            ).otherwise(F.lit(0.0)).alias("value"),
        )
        parts.append(direction)

    else:
        # Non-temporal edges: encode category as a hash indicator
        # in the time delta sub-segment so the GNN can still
        # distinguish edge categories in this segment
        cat_indicator = edge_df.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.lit(category), F.lit(50)))
                % F.lit(td_dim)
                + F.lit(td_start)
            ).alias("dim"),
            F.lit(1.0).alias("value"),
        )
        parts.append(cat_indicator)

    if not parts:
        return None

    result = parts[0]
    for df in parts[1:]:
        result = result.unionAll(df)

    return result.select(
        F.col("edge_idx").cast("long"),
        F.col("dim").cast("int"),
        F.col("value").cast("float"),
    )

# ================================================================
# Segment 2: Numeric Contrast encoding
# ================================================================

def encode_numeric_contrast(
    layout: EdgeVectorLayout,
    maybe_broadcast: Callable[[DataFrame, int, int], DataFrame],
    edge_df: DataFrame,
    src_type: str,
    dst_type: str,
    category: str,
    numeric_props_df: DataFrame,
    has_shared_numerics: bool,
    src_rows: PropertyRows,
    dst_rows: PropertyRows,
) -> Optional[DataFrame]:
    """
    Encode Segment 2: numeric differences, ratios, and magnitudes
    between edge endpoints.

    For each numeric property shared by both endpoints (same
    predicate URI), computes:
    - Difference (dst - src) hashed into difference sub-segment
    - Ratio (dst / src, clamped to [-10, 10]) hashed into ratio
      sub-segment
    - Average magnitude hashed into magnitude sub-segment

    If no shared properties exist, falls back to cross-property
    derivation for specific categories (e.g., moneyness for
    option-stock edges, severity delta for escalation edges).

    All dim indices are derived from ``layout``.

    Returns DataFrame(edge_idx: long, dim: int, value: float)
    or None.
    """
    parts: List[DataFrame] = []

    if not has_shared_numerics:
        # No edge of this type has endpoints sharing a numeric
        # predicate — derive contrast across different properties
        # instead. Decided once for all types by
        # _types_with_shared_numerics, so this branch costs no
        # Spark job of its own.
        return _encode_cross_property_contrast(
            layout, edge_df, src_type, dst_type, category, numeric_props_df
        )

    # Get numeric properties for source and destination types
    src_numerics = (
        numeric_props_df
        .filter(F.col("node_type") == src_type)
        .select(
            F.col("node_id").alias("_src_nid"),
            F.col("predicate").alias("src_pred"),
            F.col("numeric_value").alias("src_val"),
        )
    )

    dst_numerics = (
        numeric_props_df
        .filter(F.col("node_type") == dst_type)
        .select(
            F.col("node_id").alias("_dst_nid"),
            F.col("predicate").alias("dst_pred"),
            F.col("numeric_value").alias("dst_val"),
        )
    )

    # Join edge endpoints with their numeric properties.
    # Inner join on predicate ensures we only get properties
    # that both endpoints share.
    # Broadcast on the same terms as the segment-1 joins, and note
    # that these frames are the widest of the six: no predicate
    # filter narrows them, so a type contributes every numeric
    # property of every one of its nodes.
    edge_with_src = (
        edge_df
        .join(
            maybe_broadcast(
                src_numerics, src_rows.total, NUMERIC_ROW_BYTES
            ),
            edge_df["src_id"] == src_numerics["_src_nid"],
            "inner",
        )
        .drop("_src_nid")
    )

    edge_with_both = (
        edge_with_src
        .join(
            maybe_broadcast(
                dst_numerics, dst_rows.total, NUMERIC_ROW_BYTES
            ),
            (edge_with_src["dst_id"] == dst_numerics["_dst_nid"])
            & (edge_with_src["src_pred"] == dst_numerics["dst_pred"]),
            "inner",
        )
        .drop("_dst_nid", "dst_pred")
        .withColumnRenamed("src_pred", "predicate")
    )

    # --- Sub-segment 2a: Difference (dst - src) ---
    diff_start = layout.seg2_difference_start
    diff_dim = layout.seg2_difference_dim

    diff_entries = edge_with_both.select(
        F.col("edge_idx"),
        (
            F.abs(F.hash(F.col("predicate"), F.lit(500)))
            % F.lit(diff_dim)
            + F.lit(diff_start)
        ).alias("dim"),
        (F.col("dst_val") - F.col("src_val")).alias("value"),
    )
    parts.append(diff_entries)

    # --- Sub-segment 2b: Ratio (dst / src, clamped) ---
    rat_start = layout.seg2_ratio_start
    rat_dim = layout.seg2_ratio_dim

    # Safe ratio: clamp to [-10, 10] to avoid extreme values
    # from near-zero denominators. Division by zero returns 0.0.
    ratio_entries = edge_with_both.select(
        F.col("edge_idx"),
        (
            F.abs(F.hash(F.col("predicate"), F.lit(501)))
            % F.lit(rat_dim)
            + F.lit(rat_start)
        ).alias("dim"),
        F.when(
            F.abs(F.col("src_val")) > F.lit(1e-8),
            F.greatest(
                F.lit(-10.0),
                F.least(
                    F.lit(10.0),
                    F.col("dst_val") / F.col("src_val"),
                ),
            ),
        ).otherwise(F.lit(0.0)).alias("value"),
    )
    parts.append(ratio_entries)

    # --- Sub-segment 2c: Magnitude (average absolute value) ---
    mag_start = layout.seg2_magnitude_start
    mag_dim = layout.seg2_magnitude_dim

    mag_entries = edge_with_both.select(
        F.col("edge_idx"),
        (
            F.abs(F.hash(F.col("predicate"), F.lit(502)))
            % F.lit(mag_dim)
            + F.lit(mag_start)
        ).alias("dim"),
        (
            (F.abs(F.col("src_val")) + F.abs(F.col("dst_val")))
            / F.lit(2.0)
        ).alias("value"),
    )
    parts.append(mag_entries)

    if not parts:
        return None

    result = parts[0]
    for df in parts[1:]:
        result = result.unionAll(df)

    return result.select(
        F.col("edge_idx").cast("long"),
        F.col("dim").cast("int"),
        F.col("value").cast("float"),
    )


def _encode_cross_property_contrast(
    layout: EdgeVectorLayout,
    edge_df: DataFrame,
    src_type: str,
    dst_type: str,
    category: str,
    numeric_props_df: DataFrame,
) -> Optional[DataFrame]:
    """
    Derive numeric contrast from different properties on source
    and destination endpoints for specific edge categories.

    This handles the case where endpoints don't share the same
    predicate URI but have semantically related properties that
    can be meaningfully compared:

    For option_stock edges:
      - Source (option node): strikePrice property
      - Destination (stock node): observedPrice property
      - Derived: moneyness = strike / stock_price
      - Derived: log-moneyness = log(moneyness)
      - Derived: strike-stock difference

    For escalation edges:
      - Source (alert node): severity/severityLevel property
      - Destination (alert node): severity/severityLevel property
      - Derived: severity delta = dst_severity - src_severity

    Returns DataFrame(edge_idx: long, dim: int, value: float)
    or None.
    """
    parts: List[DataFrame] = []

    if category == "option_stock":
        # Source (option): look for strike price property
        src_strike = (
            numeric_props_df
            .filter(F.col("node_type") == src_type)
            .filter(
                F.col("predicate").rlike(
                    "(?i)(strikePrice|strike)"
                )
            )
            .select(
                F.col("node_id").alias("_src_nid"),
                F.col("numeric_value").alias("strike_price"),
            )
        )

        # Destination (stock): look for observed price property
        dst_price = (
            numeric_props_df
            .filter(F.col("node_type") == dst_type)
            .filter(
                F.col("predicate").rlike(
                    "(?i)(observedPrice|price|lastPrice|closePrice)"
                )
            )
            .select(
                F.col("node_id").alias("_dst_nid"),
                F.col("numeric_value").alias("stock_price"),
            )
        )

        moneyness_edges = (
            edge_df
            .join(
                src_strike,
                edge_df["src_id"] == src_strike["_src_nid"],
                "inner",
            )
            .drop("_src_nid")
            .join(
                dst_price,
                edge_df["dst_id"] == dst_price["_dst_nid"],
                "inner",
            )
            .drop("_dst_nid")
        )

        if moneyness_edges.head(1):
            diff_start = layout.seg2_difference_start
            diff_dim = layout.seg2_difference_dim

            # Moneyness = strike / stock_price
            # ATM ≈ 1.0, ITM call < 1.0, OTM call > 1.0
            moneyness = moneyness_edges.withColumn(
                "moneyness",
                F.when(
                    F.abs(F.col("stock_price")) > F.lit(1e-8),
                    F.col("strike_price") / F.col("stock_price"),
                ).otherwise(F.lit(1.0)),
            )

            # Moneyness in difference sub-segment
            m_entry = moneyness.select(
                F.col("edge_idx"),
                (
                    F.abs(F.hash(F.lit("moneyness"), F.lit(510)))
                    % F.lit(diff_dim)
                    + F.lit(diff_start)
                ).alias("dim"),
                F.col("moneyness").alias("value"),
            )
            parts.append(m_entry)

            # Log-moneyness in ratio sub-segment — more useful
            # for GNNs because it's symmetric around ATM (log(1)=0)
            rat_start = layout.seg2_ratio_start
            rat_dim = layout.seg2_ratio_dim

            log_m = moneyness.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(F.lit("log_moneyness"), F.lit(511))
                    )
                    % F.lit(rat_dim)
                    + F.lit(rat_start)
                ).alias("dim"),
                F.log(
                    F.greatest(F.col("moneyness"), F.lit(1e-8))
                ).alias("value"),
            )
            parts.append(log_m)

            # Strike-stock absolute difference in magnitude
            # sub-segment — raw dollar distance
            mag_start = layout.seg2_magnitude_start
            mag_dim = layout.seg2_magnitude_dim

            price_diff = moneyness.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(
                            F.lit("strike_stock_diff"), F.lit(512)
                        )
                    )
                    % F.lit(mag_dim)
                    + F.lit(mag_start)
                ).alias("dim"),
                (
                    F.col("strike_price") - F.col("stock_price")
                ).alias("value"),
            )
            parts.append(price_diff)

    elif category == "escalation":
        # Look for severity-related numeric properties on both
        # endpoints. NOAA alerts may encode severity as a numeric
        # level (1=Minor, 2=Moderate, 3=Severe, 4=Extreme).
        src_severity = (
            numeric_props_df
            .filter(F.col("node_type") == src_type)
            .filter(
                F.col("predicate").rlike(
                    "(?i)(severity|severityLevel|severityNum)"
                )
            )
            .select(
                F.col("node_id").alias("_src_nid"),
                F.col("numeric_value").alias("src_severity"),
            )
        )

        dst_severity = (
            numeric_props_df
            .filter(F.col("node_type") == dst_type)
            .filter(
                F.col("predicate").rlike(
                    "(?i)(severity|severityLevel|severityNum)"
                )
            )
            .select(
                F.col("node_id").alias("_dst_nid"),
                F.col("numeric_value").alias("dst_severity"),
            )
        )

        sev_edges = (
            edge_df
            .join(
                src_severity,
                edge_df["src_id"] == src_severity["_src_nid"],
                "inner",
            )
            .drop("_src_nid")
            .join(
                dst_severity,
                edge_df["dst_id"] == dst_severity["_dst_nid"],
                "inner",
            )
            .drop("_dst_nid")
        )

        if sev_edges.head(1):
            diff_start = layout.seg2_difference_start
            diff_dim = layout.seg2_difference_dim

            # Severity delta: positive means escalation (dst is
            # more severe), negative means de-escalation
            sev_delta = sev_edges.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(F.lit("severity_delta"), F.lit(520))
                    )
                    % F.lit(diff_dim)
                    + F.lit(diff_start)
                ).alias("dim"),
                (
                    F.col("dst_severity") - F.col("src_severity")
                ).alias("value"),
            )
            parts.append(sev_delta)

    if not parts:
        return None

    result = parts[0]
    for df in parts[1:]:
        result = result.unionAll(df)

    return result.select(
        F.col("edge_idx").cast("long"),
        F.col("dim").cast("int"),
        F.col("value").cast("float"),
    )

# ================================================================
# Segment 3: Relational Context encoding
# ================================================================

def encode_relational_context(
    layout: EdgeVectorLayout,
    edge_df: DataFrame,
    src_type: str,
    dst_type: str,
    relation: str,
    category: str,
    label_df: DataFrame,
) -> Optional[DataFrame]:
    """
    Encode Segment 3: namespace signals, label similarity, and
    relation type identity.

    Namespace signals are derived from the PyG node type name
    prefix (e.g., "cpi_Index" → "cpi"). This is a driver-side
    string operation on the type name, not a Spark UDF — the
    result is broadcast as a literal to all executors.

    Label similarity is computed for correlation and causal edges
    using Jaccard word overlap between endpoint rdfs:label values.
    For other edge types, a category indicator hash is used instead.

    Relation identity hashes the relation name and category into
    fixed slots, giving the GNN a fingerprint of the relationship
    type within the vector.

    All dim indices are derived from ``layout``.

    Returns DataFrame(edge_idx: long, dim: int, value: float)
    or None.
    """
    parts: List[DataFrame] = []

    # --- Sub-segment 3a: Namespace Signals ---
    ns_start = layout.seg3_namespace_start
    ns_dim = layout.seg3_namespace_dim

    # Same-namespace flag: are src and dst from the same ontology?
    # Derive from node_type prefix (e.g., "cpi_Index" → "cpi").
    # This is a driver-side string operation — the result is
    # broadcast as F.lit() to all executors.
    src_prefix = (
        src_type.split("_")[0] if "_" in src_type else src_type
    )
    dst_prefix = (
        dst_type.split("_")[0] if "_" in dst_type else dst_type
    )
    same_ns = 1.0 if src_prefix == dst_prefix else 0.0

    same_ns_entry = edge_df.select(
        F.col("edge_idx"),
        F.lit(ns_start).alias("dim"),
        F.lit(same_ns).alias("value"),
    )
    parts.append(same_ns_entry)

    # Cross-source flag (inverse of same-namespace)
    if ns_dim >= 2:
        cross_source_entry = edge_df.select(
            F.col("edge_idx"),
            F.lit(ns_start + 1).alias("dim"),
            F.lit(1.0 - same_ns).alias("value"),
        )
        parts.append(cross_source_entry)

    # Hash source and destination namespace prefixes into remaining
    # namespace slots for finer-grained namespace identity
    if ns_dim >= 3:
        for seed_offset in _EDGE_HASH_SEEDS[:2]:
            src_ns_hash = edge_df.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(
                            F.lit(src_prefix),
                            F.lit(seed_offset + 700),
                        )
                    )
                    % F.lit(max(1, ns_dim - 2))
                    + F.lit(ns_start + 2)
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(src_ns_hash)

            dst_ns_hash = edge_df.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(
                            F.lit(dst_prefix),
                            F.lit(seed_offset + 710),
                        )
                    )
                    % F.lit(max(1, ns_dim - 2))
                    + F.lit(ns_start + 2)
                ).alias("dim"),
                F.lit(0.5).alias("value"),
            )
            parts.append(dst_ns_hash)

    # --- Sub-segment 3b: Label Similarity ---
    ls_start = layout.seg3_label_similarity_start
    ls_dim = layout.seg3_label_similarity_dim

    if category in ("correlation", "causal"):
        # Join labels for both endpoints to compute word overlap.
        # This distinguishes strong correlations (exact keyword
        # match like "Energy" ↔ "Energy") from weak ones
        # ("Food at Home" ↔ "Food Manufacturing").
        src_labels = (
            label_df
            .filter(F.col("node_type") == src_type)
            .select(
                F.col("node_id").alias("_src_nid"),
                F.col("label").alias("src_label"),
            )
        )
        dst_labels = (
            label_df
            .filter(F.col("node_type") == dst_type)
            .select(
                F.col("node_id").alias("_dst_nid"),
                F.col("label").alias("dst_label"),
            )
        )

        label_edges = (
            edge_df
            .join(
                src_labels,
                edge_df["src_id"] == src_labels["_src_nid"],
                "left",
            )
            .drop("_src_nid")
            .join(
                dst_labels,
                edge_df["dst_id"] == dst_labels["_dst_nid"],
                "left",
            )
            .drop("_dst_nid")
        )

        # Compute Jaccard word overlap as a simple similarity
        # measure: |intersection| / |union| of word sets.
        # This is a rough approximation — good enough for a
        # feature signal. All computed via Spark array functions
        # on executors.
        label_edges = (
            label_edges
            .withColumn(
                "src_words",
                F.split(
                    F.coalesce(F.col("src_label"), F.lit("")),
                    r"\s+",
                ),
            )
            .withColumn(
                "dst_words",
                F.split(
                    F.coalesce(F.col("dst_label"), F.lit("")),
                    r"\s+",
                ),
            )
            .withColumn(
                "shared_words",
                F.size(
                    F.array_intersect("src_words", "dst_words")
                ),
            )
            .withColumn(
                "total_words",
                F.size(
                    F.array_union("src_words", "dst_words")
                ),
            )
            .withColumn(
                "label_sim",
                F.when(
                    F.col("total_words") > 0,
                    F.col("shared_words").cast("double")
                    / F.col("total_words").cast("double"),
                ).otherwise(F.lit(0.0)),
            )
        )

        # Jaccard similarity in the first label similarity slot
        sim_entry = label_edges.select(
            F.col("edge_idx"),
            F.lit(ls_start).alias("dim"),
            F.col("label_sim").alias("value"),
        )
        parts.append(sim_entry)

        # Hash the concatenated labels into remaining slots for
        # finer-grained label pair identity
        if ls_dim >= 2:
            shared_hash = label_edges.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(
                            F.concat(
                                F.coalesce(
                                    F.col("src_label"), F.lit("")
                                ),
                                F.lit("::"),
                                F.coalesce(
                                    F.col("dst_label"), F.lit("")
                                ),
                            ),
                            F.lit(800),
                        )
                    )
                    % F.lit(max(1, ls_dim - 1))
                    + F.lit(ls_start + 1)
                ).alias("dim"),
                F.col("label_sim").alias("value"),
            )
            parts.append(shared_hash)
    else:
        # Non-correlation edges: encode a category indicator
        # so the GNN can still distinguish categories in this
        # sub-segment
        cat_label_entry = edge_df.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.lit(category), F.lit(810)))
                % F.lit(ls_dim)
                + F.lit(ls_start)
            ).alias("dim"),
            F.lit(1.0).alias("value"),
        )
        parts.append(cat_label_entry)

    # --- Sub-segment 3c: Relation Identity ---
    ri_start = layout.seg3_relation_identity_start
    ri_dim = layout.seg3_relation_identity_dim

    # Hash the relation name into the relation identity sub-segment.
    # Two hash functions for reduced collision probability.
    for seed_offset in _EDGE_HASH_SEEDS[:2]:
        rel_hash = edge_df.select(
            F.col("edge_idx"),
            (
                F.abs(
                    F.hash(F.lit(relation), F.lit(seed_offset + 900))
                )
                % F.lit(ri_dim)
                + F.lit(ri_start)
            ).alias("dim"),
            F.lit(1.0).alias("value"),
        )
        parts.append(rel_hash)

    # Hash the category into a separate slot at half weight,
    # so edges of the same category share bits but edges of the
    # same specific relation share stronger bits
    cat_hash = edge_df.select(
        F.col("edge_idx"),
        (
            F.abs(F.hash(F.lit(category), F.lit(950)))
            % F.lit(ri_dim)
            + F.lit(ri_start)
        ).alias("dim"),
        F.lit(0.5).alias("value"),
    )
    parts.append(cat_hash)

    if not parts:
        return None

    result = parts[0]
    for df in parts[1:]:
        result = result.unionAll(df)

    return result.select(
        F.col("edge_idx").cast("long"),
        F.col("dim").cast("int"),
        F.col("value").cast("float"),
    )
