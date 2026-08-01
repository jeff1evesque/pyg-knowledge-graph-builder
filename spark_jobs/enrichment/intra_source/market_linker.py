"""
Market Intra-Source Enrichment Orchestrator (PySpark)

Coordinates enrichment for stock market data using the flat snapshot model:
  - EquitySnapshot: one node per equity quote capture
  - OptionSnapshot: one node per option quote capture

All enrichment runs as distributed PySpark DataFrame operations.

Enrichment strategies:
1. Link temporal sequences of snapshots per symbol (precedes)
2. Link option snapshots to their underlying equity snapshots
3. Identify option strategies (straddles, spreads, strangles)
4. Classify snapshots by sector (via ticker symbol)
5. Compute moneyness for option snapshots

The flat model means each snapshot node has ALL its properties as
direct datatype properties (lastPrice, strikePrice, delta, etc.).
No intermediate entity resolution is needed — the enricher works
directly with the snapshot subjects.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from functools import reduce
from typing import List, Optional

from spark_jobs.utils.rdf_utils import MARKET_QUOTES, MARKET_ENRICHMENT
from spark_jobs.enrichment.intra_source.market.patterns import (
    get_sector_patterns,
    MARKET_OPTION_STRATEGY_PATTERNS,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Class URIs (flat snapshot model)
EQUITY_SNAPSHOT_TYPE = str(MARKET_QUOTES.EquitySnapshot)
OPTION_SNAPSHOT_TYPE = str(MARKET_QUOTES.OptionSnapshot)
QUOTE_SNAPSHOT_TYPE = str(MARKET_QUOTES.QuoteSnapshot)

# Property URIs
SYMBOL_PRED = str(MARKET_QUOTES.symbol)
CAPTURE_TIME_PRED = str(MARKET_QUOTES.captureTime)
LAST_PRICE_PRED = str(MARKET_QUOTES.lastPrice)
UNDERLYING_SYMBOL_PRED = str(MARKET_QUOTES.underlyingSymbol)
UNDERLYING_PRICE_PRED = str(MARKET_QUOTES.underlyingPrice)
STRIKE_PRICE_PRED = str(MARKET_QUOTES.strikePrice)
EXPIRATION_DATE_PRED = str(MARKET_QUOTES.expirationDate)
CONTRACT_TYPE_PRED = str(MARKET_QUOTES.contractType)
DELTA_PRED = str(MARKET_QUOTES.delta)
IN_THE_MONEY_PRED = str(MARKET_QUOTES.inTheMoney)

# Enrichment property URIs
PRECEDES_PRED = str(MARKET_ENRICHMENT.precedes)
HAS_UNDERLYING_EQUITY_PRED = str(MARKET_ENRICHMENT.hasUnderlyingEquity)
HAS_MONEYNESS_PRED = str(MARKET_ENRICHMENT.hasMoneyness)
ATM_URI = str(MARKET_ENRICHMENT.AtTheMoney)
ITM_URI = str(MARKET_ENRICHMENT.InTheMoney)
OTM_URI = str(MARKET_ENRICHMENT.OutOfTheMoney)
STRADDLE_WITH_PRED = str(MARKET_ENRICHMENT.straddleWith)
CALL_SPREAD_WITH_PRED = str(MARKET_ENRICHMENT.callSpreadWith)
PUT_SPREAD_WITH_PRED = str(MARKET_ENRICHMENT.putSpreadWith)
STRANGLE_WITH_PRED = str(MARKET_ENRICHMENT.strangleWith)
BELONGS_TO_SECTOR_PRED = str(MARKET_ENRICHMENT.belongsToSector)

# Maximum strangle pairs per underlying/expiration chain
MAX_STRANGLE_PAIRS_PER_CHAIN = 10


class MarketIntraSourceLinker:
    """
    Market intra-source enrichment using PySpark DataFrames.

    Works with the flat snapshot model where each subject is a
    self-contained EquitySnapshot or OptionSnapshot with all
    properties as direct datatype properties.

    Each method reads from triples_df, produces a DataFrame of new
    triples (subject, predicate, object), and returns it.
    """

    def __init__(
        self,
        spark: SparkSession,
        sector_definitions_bucket: str = "",
        sector_definitions_key: str = "",
    ):
        self.spark = spark
        self._sector_definitions_bucket = sector_definitions_bucket
        self._sector_definitions_key = sector_definitions_key

    def enrich(self, triples_df: DataFrame) -> DataFrame:
        """
        Run all Market intra-source enrichment steps.

        Args:
            triples_df: DataFrame with columns (subject, predicate, object)

        Returns:
            DataFrame of NEW triples to union with triples_df.
        """
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        # Quick check: is there any market data?
        has_market = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (
                (F.col("object") == EQUITY_SNAPSHOT_TYPE)
                | (F.col("object") == OPTION_SNAPSHOT_TYPE)
                | (F.col("object") == QUOTE_SNAPSHOT_TYPE)
            )
        ).limit(1).count() > 0

        if not has_market:
            logger.info("No Market data detected, skipping enrichment")
            return empty

        logger.info("=" * 60)
        logger.info("Starting Market Intra-Source Enrichment (PySpark)")
        logger.info("=" * 60)

        triples_df.cache()

        new_dfs: List[DataFrame] = []

        logger.info("[Step 1/5] Linking snapshot temporal sequences...")
        df = self._link_snapshot_sequences(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/5] Linking options to underlying equities...")
        df = self._link_options_to_underlying(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 3/5] Identifying option strategies...")
        df = self._identify_option_strategies(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 4/5] Applying sector patterns...")
        df = self._classify_sectors(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 5/5] Computing option moneyness...")
        df = self._compute_moneyness(triples_df)
        if df is not None:
            new_dfs.append(df)

        if not new_dfs:
            logger.info("No enrichment triples produced")
            return empty

        result = reduce(DataFrame.unionAll, new_dfs)

        count = result.count()
        logger.info("=" * 60)
        logger.info(f"Market Intra-Source Enrichment Complete: {count} triples")
        logger.info("=" * 60)

        return result

    # ================================================================
    # Step 1: Snapshot Temporal Sequences
    # ================================================================

    def _link_snapshot_sequences(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Link snapshots in chronological order per symbol.

        For each symbol, orders snapshots by captureTime and produces:
            snapshot_N  enrichment:precedes  snapshot_N+1

        Works for both equity and option snapshots — any subject with
        a symbol and captureTime property gets sequenced.
        """
        # Get all snapshot subjects with their symbol and timestamp
        snapshot_symbols = triples_df.filter(
            F.col("predicate") == SYMBOL_PRED
        ).select(
            F.col("subject").alias("snapshot"),
            F.col("object").alias("symbol"),
        )

        snapshot_times = triples_df.filter(
            F.col("predicate") == CAPTURE_TIME_PRED
        ).select(
            F.col("subject").alias("snapshot"),
            F.col("object").alias("capture_time"),
        )

        snapshots_df = snapshot_symbols.join(snapshot_times, "snapshot", "inner")

        if snapshots_df.head(1) == []:
            logger.info("  No snapshots with symbol + captureTime found")
            return None

        # Window: partition by symbol, order by capture_time
        w = Window.partitionBy("symbol").orderBy("capture_time")
        sequenced = snapshots_df.withColumn(
            "next_snapshot", F.lead("snapshot").over(w)
        )
        pairs = sequenced.filter(F.col("next_snapshot").isNotNull())

        if pairs.head(1) == []:
            return None

        result = pairs.select(
            F.col("snapshot").alias("subject"),
            F.lit(PRECEDES_PRED).alias("predicate"),
            F.col("next_snapshot").alias("object"),
        )

        logger.info("  Snapshot sequence linking complete")
        return result

    # ================================================================
    # Step 2: Link Options to Underlying Equities
    # ================================================================

    def _link_options_to_underlying(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Link option snapshots to their underlying equity snapshots.

        Uses the underlyingSymbol property on OptionSnapshots to find
        the corresponding EquitySnapshot with the same symbol and
        closest captureTime.

        Produces:
            option_snapshot  hasUnderlyingEquity  equity_snapshot
        """
        # Option snapshots with their underlying symbol
        option_snapshots = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == OPTION_SNAPSHOT_TYPE)
        ).select(F.col("subject").alias("option"))

        option_underlying = triples_df.filter(
            F.col("predicate") == UNDERLYING_SYMBOL_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("underlying_symbol"),
        )

        option_times = triples_df.filter(
            F.col("predicate") == CAPTURE_TIME_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("option_time"),
        )

        options_df = (
            option_snapshots
            .join(option_underlying, "option", "inner")
            .join(option_times, "option", "inner")
        )

        if options_df.head(1) == []:
            logger.info("  No option snapshots with underlyingSymbol found")
            return None

        # Equity snapshots with their symbol and time
        equity_snapshots = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == EQUITY_SNAPSHOT_TYPE)
        ).select(F.col("subject").alias("equity"))

        equity_symbols = triples_df.filter(
            F.col("predicate") == SYMBOL_PRED
        ).select(
            F.col("subject").alias("equity"),
            F.col("object").alias("equity_symbol"),
        )

        equity_times = triples_df.filter(
            F.col("predicate") == CAPTURE_TIME_PRED
        ).select(
            F.col("subject").alias("equity"),
            F.col("object").alias("equity_time"),
        )

        equities_df = (
            equity_snapshots
            .join(equity_symbols, "equity", "inner")
            .join(equity_times, "equity", "inner")
        )

        if equities_df.head(1) == []:
            logger.info("  No equity snapshots found for linking")
            return None

        # Join: option's underlyingSymbol = equity's symbol
        # AND same captureTime (snapshots from same capture event)
        joined = options_df.join(
            equities_df,
            (options_df.underlying_symbol == equities_df.equity_symbol)
            & (options_df.option_time == equities_df.equity_time),
            "inner",
        )

        if joined.head(1) == []:
            # Fallback: match by symbol only, take most recent equity
            # per underlying symbol (for cases where timestamps don't
            # align exactly)
            w = Window.partitionBy("equity_symbol").orderBy(
                F.col("equity_time").desc()
            )
            latest_equities = (
                equities_df
                .withColumn("rn", F.row_number().over(w))
                .filter(F.col("rn") == 1)
                .drop("rn")
            )

            joined = options_df.join(
                latest_equities,
                options_df.underlying_symbol == latest_equities.equity_symbol,
                "inner",
            )

            if joined.head(1) == []:
                logger.info("  No option-equity matches found")
                return None

        result = joined.select(
            F.col("option").alias("subject"),
            F.lit(HAS_UNDERLYING_EQUITY_PRED).alias("predicate"),
            F.col("equity").alias("object"),
        ).dropDuplicates()

        logger.info("  Option-to-underlying linking complete")
        return result

    # ================================================================
    # Step 3: Option Strategy Identification
    # ================================================================

    def _identify_option_strategies(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Identify option strategies: straddles, vertical spreads, strangles.

        Works directly with OptionSnapshot subjects and their properties.
        """
        # Build options table from triples
        option_snapshots = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == OPTION_SNAPSHOT_TYPE)
        ).select(F.col("subject").alias("option"))

        underlying = triples_df.filter(
            F.col("predicate") == UNDERLYING_SYMBOL_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("underlying"),
        )

        strikes = triples_df.filter(
            F.col("predicate") == STRIKE_PRICE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").cast("double").alias("strike"),
        )

        expirations = triples_df.filter(
            F.col("predicate") == EXPIRATION_DATE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("expiration"),
        )

        contract_types = triples_df.filter(
            F.col("predicate") == CONTRACT_TYPE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("contract_type"),
        )

        capture_times = triples_df.filter(
            F.col("predicate") == CAPTURE_TIME_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("capture_time"),
        )

        options_df = (
            option_snapshots
            .join(underlying, "option", "inner")
            .join(strikes, "option", "inner")
            .join(expirations, "option", "inner")
            .join(contract_types, "option", "inner")
            .join(capture_times, "option", "inner")
        )

        if options_df.head(1) == []:
            logger.info("  No fully-specified option snapshots found")
            return None

        options_df.cache()

        new_dfs: List[DataFrame] = []

        df = self._identify_straddles(options_df)
        if df is not None:
            new_dfs.append(df)

        df = self._identify_vertical_spreads(options_df)
        if df is not None:
            new_dfs.append(df)

        df = self._identify_strangles(options_df)
        if df is not None:
            new_dfs.append(df)

        options_df.unpersist()

        if not new_dfs:
            return None

        return reduce(DataFrame.unionAll, new_dfs)

    def _identify_straddles(
        self, options_df: DataFrame
    ) -> Optional[DataFrame]:
        """Same underlying, expiration, strike, capture_time — call + put."""
        calls = options_df.filter(
            F.upper(F.col("contract_type")) == "CALL"
        ).select(
            F.col("option").alias("call_option"),
            F.col("underlying"),
            F.col("expiration"),
            F.col("strike"),
            F.col("capture_time"),
        )

        puts = options_df.filter(
            F.upper(F.col("contract_type")) == "PUT"
        ).select(
            F.col("option").alias("put_option"),
            F.col("underlying").alias("p_underlying"),
            F.col("expiration").alias("p_expiration"),
            F.col("strike").alias("p_strike"),
            F.col("capture_time").alias("p_capture_time"),
        )

        straddles = calls.join(
            puts,
            (calls.underlying == puts.p_underlying)
            & (calls.expiration == puts.p_expiration)
            & (calls.strike == puts.p_strike)
            & (calls.capture_time == puts.p_capture_time),
            "inner",
        )

        if straddles.head(1) == []:
            return None

        return straddles.select(
            F.col("call_option").alias("subject"),
            F.lit(STRADDLE_WITH_PRED).alias("predicate"),
            F.col("put_option").alias("object"),
        )

    def _identify_vertical_spreads(
        self, options_df: DataFrame
    ) -> Optional[DataFrame]:
        """Same underlying, expiration, type, capture_time — adjacent strikes."""
        w = Window.partitionBy(
            "underlying", "expiration", "contract_type", "capture_time"
        ).orderBy("strike")

        with_next = options_df.withColumn(
            "next_option", F.lead("option").over(w)
        ).filter(F.col("next_option").isNotNull())

        if with_next.head(1) == []:
            return None

        call_spreads = with_next.filter(
            F.upper(F.col("contract_type")) == "CALL"
        ).select(
            F.col("option").alias("subject"),
            F.lit(CALL_SPREAD_WITH_PRED).alias("predicate"),
            F.col("next_option").alias("object"),
        )

        put_spreads = with_next.filter(
            F.upper(F.col("contract_type")) == "PUT"
        ).select(
            F.col("option").alias("subject"),
            F.lit(PUT_SPREAD_WITH_PRED).alias("predicate"),
            F.col("next_option").alias("object"),
        )

        result = call_spreads.unionAll(put_spreads)
        if result.head(1) == []:
            return None

        return result

    def _identify_strangles(
        self, options_df: DataFrame
    ) -> Optional[DataFrame]:
        """Nearest OTM call + put pairs per (underlying, expiration, capture_time)."""
        # Compute midpoint of strike range per chain
        chain_stats = options_df.groupBy(
            "underlying", "expiration", "capture_time"
        ).agg(
            ((F.min("strike") + F.max("strike")) / 2.0).alias("midpoint")
        )

        options_with_mid = options_df.join(
            chain_stats, ["underlying", "expiration", "capture_time"], "inner"
        )

        # OTM calls: strike > midpoint, ranked ascending
        otm_calls = options_with_mid.filter(
            (F.upper(F.col("contract_type")) == "CALL")
            & (F.col("strike") > F.col("midpoint"))
        )
        w_call = Window.partitionBy(
            "underlying", "expiration", "capture_time"
        ).orderBy(F.col("strike").asc())
        otm_calls = (
            otm_calls.withColumn("rank", F.row_number().over(w_call))
            .filter(F.col("rank") <= MAX_STRANGLE_PAIRS_PER_CHAIN)
            .select(
                F.col("option").alias("call_option"),
                F.col("underlying"),
                F.col("expiration"),
                F.col("capture_time"),
                F.col("rank"),
            )
        )

        # OTM puts: strike < midpoint, ranked descending
        otm_puts = options_with_mid.filter(
            (F.upper(F.col("contract_type")) == "PUT")
            & (F.col("strike") < F.col("midpoint"))
        )
        w_put = Window.partitionBy(
            "underlying", "expiration", "capture_time"
        ).orderBy(F.col("strike").desc())
        otm_puts = (
            otm_puts.withColumn("rank", F.row_number().over(w_put))
            .filter(F.col("rank") <= MAX_STRANGLE_PAIRS_PER_CHAIN)
            .select(
                F.col("option").alias("put_option"),
                F.col("underlying").alias("p_underlying"),
                F.col("expiration").alias("p_expiration"),
                F.col("capture_time").alias("p_capture_time"),
                F.col("rank").alias("p_rank"),
            )
        )

        strangles = otm_calls.join(
            otm_puts,
            (otm_calls.underlying == otm_puts.p_underlying)
            & (otm_calls.expiration == otm_puts.p_expiration)
            & (otm_calls.capture_time == otm_puts.p_capture_time)
            & (otm_calls.rank == otm_puts.p_rank),
            "inner",
        )

        if strangles.head(1) == []:
            return None

        return strangles.select(
            F.col("call_option").alias("subject"),
            F.lit(STRANGLE_WITH_PRED).alias("predicate"),
            F.col("put_option").alias("object"),
        )

    # ================================================================
    # Step 4: Sector Classification
    # ================================================================

    def _classify_sectors(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Classify equity snapshots by sector.

        Loads sector patterns from the S&P 500 tickers CSV in S3
        (grouped by GICS Sector). Falls back to hardcoded defaults
        if S3 is unavailable.
        """
        sector_patterns = get_sector_patterns(
            bucket=self._sector_definitions_bucket,
            key=self._sector_definitions_key,
        )

        # Build ticker → sector lookup
        sector_rows = []
        for sector_name, pattern in sector_patterns.items():
            sector_uri = str(pattern["sector_uri"])
            relationship = str(pattern["relationship"])
            for ticker in pattern["tickers"]:
                sector_rows.append((ticker, sector_uri, relationship))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows, ["ticker", "sector_uri", "relationship_uri"]
        )

        # Equity snapshots: match by symbol
        equity_symbols = triples_df.filter(
            F.col("predicate") == SYMBOL_PRED
        ).select(
            F.col("subject").alias("snapshot"),
            F.col("object").alias("ticker"),
        )

        # Also match option snapshots by underlyingSymbol
        option_underlying = triples_df.filter(
            F.col("predicate") == UNDERLYING_SYMBOL_PRED
        ).select(
            F.col("subject").alias("snapshot"),
            F.col("object").alias("ticker"),
        )

        all_symbols = equity_symbols.unionAll(option_underlying)

        joined = all_symbols.join(
            F.broadcast(sector_df), "ticker", "inner"
        )

        if joined.head(1) == []:
            logger.info("  No sector matches found")
            return None

        belongs_triples = joined.select(
            F.col("snapshot").alias("subject"),
            F.lit(BELONGS_TO_SECTOR_PRED).alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        correlation_triples = joined.select(
            F.col("snapshot").alias("subject"),
            F.col("relationship_uri").alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        return belongs_triples.unionAll(correlation_triples)

    # ================================================================
    # Step 5: Compute Moneyness
    # ================================================================

    def _compute_moneyness(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Compute moneyness classification for option snapshots.

        Uses strikePrice and underlyingPrice (both direct properties
        on the OptionSnapshot) to classify as ATM/ITM/OTM.

        ATM: strike within 2% of underlying price
        ITM: call with strike < underlying, or put with strike > underlying
        OTM: call with strike > underlying, or put with strike < underlying

        Produces:
            option_snapshot  hasMoneyness  ATM/ITM/OTM
        """
        option_snapshots = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == OPTION_SNAPSHOT_TYPE)
        ).select(F.col("subject").alias("option"))

        strikes = triples_df.filter(
            F.col("predicate") == STRIKE_PRICE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").cast("double").alias("strike"),
        )

        underlying_prices = triples_df.filter(
            F.col("predicate") == UNDERLYING_PRICE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").cast("double").alias("underlying_price"),
        )

        contract_types = triples_df.filter(
            F.col("predicate") == CONTRACT_TYPE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").alias("contract_type"),
        )

        options_df = (
            option_snapshots
            .join(strikes, "option", "inner")
            .join(underlying_prices, "option", "inner")
            .join(contract_types, "option", "inner")
        )

        if options_df.head(1) == []:
            logger.info("  No options with strike + underlyingPrice found")
            return None

        # Classify moneyness
        moneyness_df = options_df.withColumn(
            "moneyness",
            F.when(
                F.abs(F.col("strike") - F.col("underlying_price"))
                <= F.col("underlying_price") * 0.02,
                F.lit(ATM_URI),
            )
            .when(
                (F.upper(F.col("contract_type")) == "CALL")
                & (F.col("strike") < F.col("underlying_price")),
                F.lit(ITM_URI),
            )
            .when(
                (F.upper(F.col("contract_type")) == "CALL")
                & (F.col("strike") > F.col("underlying_price")),
                F.lit(OTM_URI),
            )
            .when(
                (F.upper(F.col("contract_type")) == "PUT")
                & (F.col("strike") > F.col("underlying_price")),
                F.lit(ITM_URI),
            )
            .when(
                (F.upper(F.col("contract_type")) == "PUT")
                & (F.col("strike") < F.col("underlying_price")),
                F.lit(OTM_URI),
            ),
        ).filter(F.col("moneyness").isNotNull())

        if moneyness_df.head(1) == []:
            return None

        result = moneyness_df.select(
            F.col("option").alias("subject"),
            F.lit(HAS_MONEYNESS_PRED).alias("predicate"),
            F.col("moneyness").alias("object"),
        )

        logger.info("  Moneyness computation complete")
        return result