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
from spark_jobs.enrichment.sector_crosswalk import EQUITY_SECTOR_TYPE
from spark_jobs.enrichment.intra_source.market.patterns import (
    get_sector_patterns,
)
from spark_jobs.enrichment.intra_source.market.symbols import (
    occ_expiration_date, occ_underlying,
)
from spark_jobs.enrichment.intra_source.sequencing import (
    sequence_to_triples,
    sequence_within_partitions,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Class URIs (flat snapshot model)
#
# QuoteSnapshot is declared upstream and nothing is ever typed with it -- the
# mapper picks EquitySnapshot or OptionSnapshot off the asset_type column, so
# these two are the whole model. The name survives in upstream's own docstring
# for the shared predicate block, which is how it came to be treated as a type.
EQUITY_SNAPSHOT_TYPE = str(MARKET_QUOTES.EquitySnapshot)
OPTION_SNAPSHOT_TYPE = str(MARKET_QUOTES.OptionSnapshot)

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

# The class the three moneyness individuals belong to. They are categories an
# option is assigned to, so they are individuals that need a type of their own
# -- see the note in _compute_moneyness for what their being untyped cost.
MONEYNESS_CLASS_TYPE = str(MARKET_ENRICHMENT.Moneyness)
STRADDLE_WITH_PRED = str(MARKET_ENRICHMENT.straddleWith)
CALL_SPREAD_WITH_PRED = str(MARKET_ENRICHMENT.callSpreadWith)
PUT_SPREAD_WITH_PRED = str(MARKET_ENRICHMENT.putSpreadWith)
STRANGLE_WITH_PRED = str(MARKET_ENRICHMENT.strangleWith)
BELONGS_TO_SECTOR_PRED = str(MARKET_ENRICHMENT.belongsToSector)

# Maximum strangle pairs per underlying/expiration chain
MAX_STRANGLE_PAIRS_PER_CHAIN = 10


def _canonical_contract_type(col: F.Column) -> F.Column:
    """Normalize a contractType literal to CALL / PUT.

    The quote-snapshot vocabulary encodes this as a single letter -- "C" / "P"
    -- which is what the API returns and what the captured fixtures contain.
    Every comparison in this module was written against the spelled-out words,
    so `F.upper(contract_type) == "CALL"` was false for every option ever
    ingested. All three option steps (underlying linking, strategy detection,
    moneyness) silently produced nothing.

    Both encodings are accepted rather than swapping one literal for the other:
    the feeds vocabulary is a different model from the quotes one, and pinning
    this to whichever spelling the current fixture happens to use is what
    created the bug in the first place. An unrecognized value is passed through
    uppercased, so it fails to match a branch rather than being silently
    coerced into the wrong one.
    """
    upper = F.upper(F.trim(col))
    return (
        F.when(upper.isin("C", "CALL"), F.lit("CALL"))
        .when(upper.isin("P", "PUT"), F.lit("PUT"))
        .otherwise(upper)
    )


def _resolve_option_property(
    triples_df: DataFrame,
    stated_predicate: str,
    derive,
    alias: str,
) -> DataFrame:
    """(option, <alias>) for every option snapshot, stated or derived.

    Prefers what the data says. Falls back to reading the value out of the OCC
    symbol, which is where the quote-snapshot vocabulary keeps it: quotes emit
    neither underlyingSymbol nor expirationDate, so the stated side is empty on
    every graph built from them and the inner joins downstream dropped every
    option row. Three steps -- underlying linking, straddles/spreads/strangles,
    and the option leg of sector classification -- each logged "no option
    snapshots found" and returned None on data made entirely of option
    snapshots.

    A left join rather than a union, so an option that states the property
    keeps its stated value and never gets two rows. Rows where neither side
    resolves are dropped: they are not options this can say anything about.
    """
    options = triples_df.filter(
        (F.col("predicate") == RDF_TYPE)
        & (F.col("object") == OPTION_SNAPSHOT_TYPE)
    ).select(F.col("subject").alias("option"))

    stated = triples_df.filter(
        F.col("predicate") == stated_predicate
    ).select(
        F.col("subject").alias("option"),
        F.col("object").alias("stated"),
    )

    derived = triples_df.filter(
        F.col("predicate") == SYMBOL_PRED
    ).select(
        F.col("subject").alias("option"),
        derive(F.col("object")).alias("derived"),
    )

    return (
        options
        .join(stated, "option", "left")
        .join(derived, "option", "left")
        .select(
            F.col("option"),
            F.coalesce(F.col("stated"), F.col("derived")).alias(alias),
        )
        .filter(F.col(alias).isNotNull())
    )


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
        self._append(
            new_dfs, self._link_snapshot_sequences(triples_df), "[Step 1/5]"
        )

        logger.info("[Step 2/5] Linking options to underlying equities...")
        self._append(
            new_dfs, self._link_options_to_underlying(triples_df), "[Step 2/5]"
        )

        logger.info("[Step 3/5] Identifying option strategies...")
        self._append(
            new_dfs, self._identify_option_strategies(triples_df), "[Step 3/5]"
        )

        logger.info("[Step 4/5] Applying sector patterns...")
        self._append(new_dfs, self._classify_sectors(triples_df), "[Step 4/5]")

        logger.info("[Step 5/5] Computing option moneyness...")
        self._append(new_dfs, self._compute_moneyness(triples_df), "[Step 5/5]")

        if not new_dfs:
            logger.info("No enrichment triples produced")
            return empty

        result = reduce(DataFrame.unionAll, new_dfs)

        count = result.count()
        logger.info("=" * 60)
        logger.info(f"Market Intra-Source Enrichment Complete: {count} triples")
        logger.info("=" * 60)

        return result

    @staticmethod
    def _append(lst: list, item: Optional[DataFrame], step: str):
        """Keep a step's triples, and say so when it made none.

        Same reason as CrossSourceLinker._append: a step that returns None
        logged only its opening line, so a missing link family read like a
        healthy graph (#350).
        """
        if item is not None:
            lst.append(item)
            return
        logger.warning(f"  {step} produced no triples")

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

        # Sequence each symbol's snapshots.
        #
        # The two joins above are inner joins on snapshot, so a snapshot
        # carrying two symbols or two capture times arrives here as two rows in
        # one partition. Left in, they sorted next to each other and lead() made
        # the snapshot precede itself (#360).
        #
        # The tie-break on the snapshot URI matters for the same reason it does
        # in NOAA Step 1: capture_time alone is not a total order, snapshots of
        # one symbol share a capture time, and lead() over a non-deterministic
        # order would make the output vary between runs.
        result = sequence_to_triples(
            snapshots_df,
            entity_col="snapshot",
            predicate=PRECEDES_PRED,
            partition_cols=["symbol"],
            order_cols=["capture_time", "snapshot"],
        )

        if result.head(1) == []:
            return None

        logger.info("  Snapshot sequence linking complete")
        return result

    # ================================================================
    # Step 2: Link Options to Underlying Equities
    # ================================================================

    def _link_options_to_underlying(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Link option snapshots to their underlying equity snapshots.

        Uses the underlying symbol of each OptionSnapshot -- stated, or read
        out of its OCC symbol -- to find the corresponding EquitySnapshot with
        the same symbol and closest captureTime.

        Produces:
            option_snapshot  hasUnderlyingEquity  equity_snapshot
        """
        option_underlying = _resolve_option_property(
            triples_df, UNDERLYING_SYMBOL_PRED, occ_underlying, "underlying_symbol"
        )

        option_snapshots = option_underlying.select("option").distinct()

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
            logger.info("  No option snapshots with a resolvable underlying found")
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
        # Build options table from triples. Underlying and expiration are
        # resolved rather than read directly: quotes state neither, and an
        # inner join against an empty frame silently emptied this whole step.
        underlying = _resolve_option_property(
            triples_df, UNDERLYING_SYMBOL_PRED, occ_underlying, "underlying"
        )
        expirations = _resolve_option_property(
            triples_df, EXPIRATION_DATE_PRED, occ_expiration_date, "expiration"
        )

        option_snapshots = underlying.select("option").distinct()

        strikes = triples_df.filter(
            F.col("predicate") == STRIKE_PRICE_PRED
        ).select(
            F.col("subject").alias("option"),
            F.col("object").cast("double").alias("strike"),
        )

        contract_types = triples_df.filter(
            F.col("predicate") == CONTRACT_TYPE_PRED
        ).select(
            F.col("subject").alias("option"),
            _canonical_contract_type(F.col("object")).alias("contract_type"),
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
        # Pair each option with the next strike up in its chain.
        #
        # This step takes the paired rows rather than triples, because calls
        # and puts leave through different predicates. The helper drops
        # duplicate options before the window, so an option cannot spread with
        # its own copy (#360), and the tie-break on the option URI keeps the
        # chain the same between runs when two options share a strike.
        with_next = sequence_within_partitions(
            options_df,
            entity_col="option",
            partition_cols=[
                "underlying", "expiration", "contract_type", "capture_time",
            ],
            order_cols=["strike", "option"],
            next_col="next_option",
        )

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
            for ticker in pattern["tickers"]:
                sector_rows.append((ticker, sector_uri))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows, ["ticker", "sector_uri"]
        )

        # Equity snapshots: match by symbol
        equity_symbols = triples_df.filter(
            F.col("predicate") == SYMBOL_PRED
        ).select(
            F.col("subject").alias("snapshot"),
            F.col("object").alias("ticker"),
        )

        # Also match option snapshots by their underlying, which for a quote
        # snapshot is only recoverable from the OCC symbol. Read literally, an
        # option's own symbol ("A     260717C00065000") matches no ticker in
        # the sector table, so options were classified into no sector at all.
        option_underlying = _resolve_option_property(
            triples_df, UNDERLYING_SYMBOL_PRED, occ_underlying, "ticker"
        ).select(
            F.col("option").alias("snapshot"),
            F.col("ticker"),
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

        # NO CORRELATION TWIN, and here that removed ELEVEN relation types
        # rather than one. This used to also emit a per-sector predicate --
        # market:energySectorCorrelation, market:healthCareSectorCorrelation,
        # one per GICS sector -- over the identical (snapshot, sector) pair
        # belongsToSector already covered. So the sector was encoded twice in
        # the same edge, once in the predicate name and once in the object,
        # and the pair could not disagree because both came from this one join.
        #
        # A GNN allocates a weight matrix per edge type, so eleven near-empty
        # duplicate relations is eleven sets of parameters learning what
        # belongsToSector already carries. See the fuller note in
        # CrossSourceLinker._link_by_sector.

        # Type the sector URIs this step points AT.
        #
        # They were never typed, and node_mapper only makes a node out of a
        # typed URI -- so every equity sector was a dangling object and every
        # belongsToSector edge above was dropped during edge resolution. The
        # market side reported "no edges to any sector" while emitting a
        # belongsToSector triple per snapshot, because the triples were real
        # and the destination node was not.
        #
        # market:EquitySector rather than bls:EconomicSector, deliberately. A
        # GICS sector sorts companies by revenue source and an economic sector
        # sorts price series by consumption category; typing one as the other
        # asserts an equivalence that does not hold. They are joined instead,
        # by a curated table, under a weaker predicate -- see
        # enrichment/sector_crosswalk.py.
        sectors = joined.select("sector_uri").distinct()

        sector_type_triples = sectors.select(
            F.col("sector_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(EQUITY_SECTOR_TYPE).alias("object"),
        )

        return (
            belongs_triples
            .unionAll(sector_type_triples)
        )

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
            _canonical_contract_type(F.col("object")).alias("contract_type"),
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
            # Reached only when options WERE found but none classified, i.e.
            # every contract_type fell through the branches above. Silence here
            # is what hid the C/CALL mismatch: the step logged that it started,
            # returned nothing, and looked identical to having no options.
            logger.info(
                "  No option classified: %d option(s) had strike + "
                "underlyingPrice but no recognized contractType",
                options_df.count(),
            )
            return None

        links = moneyness_df.select(
            F.col("option").alias("subject"),
            F.lit(HAS_MONEYNESS_PRED).alias("predicate"),
            F.col("moneyness").alias("object"),
        )

        # Type the three moneyness classes, which is what makes them nodes.
        #
        # node_mapper only creates a node for a URI that carries an rdf:type,
        # and nothing typed market:InTheMoney / AtTheMoney / OutOfTheMoney
        # anywhere. The links above were emitted correctly and then dropped
        # whole during edge resolution -- the step logged success, the triples
        # were in the enriched output, and the graph had no moneyness edge at
        # all. Same untyped-URI defect as the source temporal URIs, one hop
        # further out.
        #
        # Emitted for the classes actually USED rather than all three, so the
        # graph never carries a category no option was assigned.
        class_triples = moneyness_df.select(
            F.col("moneyness").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(MONEYNESS_CLASS_TYPE).alias("object"),
        ).distinct()

        logger.info("  Moneyness computation complete")
        return links.unionByName(class_triples)