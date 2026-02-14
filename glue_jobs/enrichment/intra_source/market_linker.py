"""
Market Intra-Source Enrichment Orchestrator (PySpark)

Coordinates enrichment for stock market data (prices and options).
All enrichment runs as distributed PySpark DataFrame operations.

Enrichment strategies:
1. Link temporal sequences of price observations (precedes)
2. Link options to underlying stock price observations
3. Identify option strategies (spreads, straddles, strangles)
4. Classify tickers by sector
5. Link multi-source observations of same ticker/contract

Note: Ticker unification is NOT needed — all mappers produce the same
canonical ticker URI (e.g., finance:TSLA) regardless of source.
Option contract URIs are also canonical (e.g., finance:TSLA_20250606_C50000).
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from functools import reduce
from typing import List, Optional

from glue_jobs.utils.rdf_utils import (
    MARKET, MARKET_OPTIONS, MARKET_ENRICHMENT, UNIFIED
)
from glue_jobs.enrichment.intra_source.market.patterns import (
    MARKET_SECTOR_PATTERNS,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

# Standard predicates
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"

# Market class URIs (canonical — all mappers use these)
STOCK_TICKER_TYPE = str(MARKET.StockTicker)
PRICE_OBS_TYPE = str(MARKET.PriceObservation)
OPTION_CONTRACT_TYPE = str(MARKET.OptionContract)
OPTION_QUOTE_TYPE = str(MARKET.OptionQuote)

# Market property URIs (canonical — all mappers use finance: namespace)
SYMBOL_PRED = str(MARKET.symbol)
OBSERVED_TICKER_PRED = str(MARKET.observedTicker)
OBSERVED_AT_PRED = str(MARKET.observedAt)
OBSERVED_PRICE_PRED = str(MARKET.observedPrice)
DATA_SOURCE_PRED = str(MARKET.dataSource)
UNDERLYING_TICKER_PRED = str(MARKET.underlyingTicker)
QUOTED_CONTRACT_PRED = str(MARKET.quotedContract)

# Option property URIs (canonical — all mappers use options: namespace)
STRIKE_PRICE_PRED = str(MARKET_OPTIONS.strikePrice)
EXPIRATION_DATE_PRED = str(MARKET_OPTIONS.expirationDate)
OPTION_TYPE_PRED = str(MARKET_OPTIONS.optionType)

# Enrichment property URIs
PRECEDES_PRED = str(MARKET_ENRICHMENT.precedes)
HAS_UNDERLYING_OBS_PRED = str(MARKET_ENRICHMENT.hasUnderlyingPriceObservation)
HAS_MONEYNESS_PRED = str(MARKET_ENRICHMENT.hasMoneyness)
ATM_URI = str(MARKET_ENRICHMENT.AtTheMoney)
ITM_URI = str(MARKET_ENRICHMENT.InTheMoney)
OTM_URI = str(MARKET_ENRICHMENT.OutOfTheMoney)
STRADDLE_WITH_PRED = str(MARKET_ENRICHMENT.straddleWith)
CALL_SPREAD_WITH_PRED = str(MARKET_ENRICHMENT.callSpreadWith)
PUT_SPREAD_WITH_PRED = str(MARKET_ENRICHMENT.putSpreadWith)
STRANGLE_WITH_PRED = str(MARKET_ENRICHMENT.strangleWith)
BELONGS_TO_SECTOR_PRED = str(MARKET_ENRICHMENT.belongsToSector)
SAME_TICKER_OBS_PRED = str(MARKET_ENRICHMENT.sameTickerObservation)
SAME_CONTRACT_QUOTE_PRED = str(MARKET_ENRICHMENT.sameContractQuote)

# Maximum strangle pairs per ticker/expiration chain
MAX_STRANGLE_PAIRS_PER_CHAIN = 10


class MarketIntraSourceLinker:
    """
    Market intra-source enrichment using PySpark DataFrames.

    Each method reads from triples_df, produces a DataFrame of new triples
    (subject, predicate, object), and returns it. The enrich() method unions
    all new triples together.

    Design notes:
    - Ticker URIs are already canonical across sources (finance:TSLA).
      No ticker unification step is needed.
    - OptionContract URIs are already canonical (finance:TSLA_20250606_C50000).
      No contract unification step is needed.
    - PriceObservation and OptionQuote URIs are source-specific
      (yahoo:price_obs_X, marketwatch:price_obs_X). Multi-source linking
      connects these.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

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
                (F.col("object") == PRICE_OBS_TYPE)
                | (F.col("object") == OPTION_CONTRACT_TYPE)
                | (F.col("object") == STOCK_TICKER_TYPE)
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

        logger.info("[Step 1/5] Linking price observation sequences...")
        df = self._link_price_sequences(triples_df)
        if df is not None:
            new_dfs.append(df)

        logger.info("[Step 2/5] Linking options to underlying stocks...")
        df = self._link_options_to_stocks(triples_df)
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

        logger.info("[Step 5/5] Linking multi-source observations...")
        df = self._link_multi_source_observations(triples_df)
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
    # Step 1: Price Observation Sequences
    # ================================================================

    def _link_price_sequences(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Link price observations in chronological order per ticker.

        Produces:  obs_N  enrichment:precedes  obs_N+1
        """
        price_obs = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == PRICE_OBS_TYPE)
        ).select(F.col("subject").alias("obs"))

        obs_tickers = triples_df.filter(
            F.col("predicate") == OBSERVED_TICKER_PRED
        ).select(
            F.col("subject").alias("obs"),
            F.col("object").alias("ticker"),
        )

        obs_timestamps = triples_df.filter(
            F.col("predicate") == OBSERVED_AT_PRED
        ).select(
            F.col("subject").alias("obs"),
            F.col("object").alias("observed_at"),
        )

        obs_df = (
            price_obs
            .join(obs_tickers, "obs", "inner")
            .join(obs_timestamps, "obs", "inner")
        )

        if obs_df.head(1) == []:
            logger.info("  No price observations found")
            return None

        w = Window.partitionBy("ticker").orderBy("observed_at")
        sequenced = obs_df.withColumn("next_obs", F.lead("obs").over(w))
        pairs = sequenced.filter(F.col("next_obs").isNotNull())

        if pairs.head(1) == []:
            return None

        result = pairs.select(
            F.col("obs").alias("subject"),
            F.lit(PRECEDES_PRED).alias("predicate"),
            F.col("next_obs").alias("object"),
        )

        logger.info("  Price sequence linking complete")
        return result

    # ================================================================
    # Step 2: Link Options to Underlying Stocks
    # ================================================================

    def _link_options_to_stocks(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Link option contracts to their underlying stock price observations.

        Produces:
          contract  hasUnderlyingPriceObservation  price_obs
          contract  hasMoneyness                   ATM/ITM/OTM
        """
        contracts = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == OPTION_CONTRACT_TYPE)
        ).select(F.col("subject").alias("contract"))

        contract_tickers = triples_df.filter(
            F.col("predicate") == UNDERLYING_TICKER_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").alias("ticker"),
        )

        contract_strikes = triples_df.filter(
            F.col("predicate") == STRIKE_PRICE_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").cast("double").alias("strike"),
        )

        contract_types = triples_df.filter(
            F.col("predicate") == OPTION_TYPE_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").alias("option_type"),
        )

        contracts_df = (
            contracts
            .join(contract_tickers, "contract", "inner")
            .join(contract_strikes, "contract", "left")
            .join(contract_types, "contract", "left")
        )

        if contracts_df.head(1) == []:
            logger.info("  No option contracts found")
            return None

        # Price observations: (obs, ticker, price)
        price_obs = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == PRICE_OBS_TYPE)
        ).select(F.col("subject").alias("obs"))

        obs_tickers = triples_df.filter(
            F.col("predicate") == OBSERVED_TICKER_PRED
        ).select(
            F.col("subject").alias("obs"),
            F.col("object").alias("ticker"),
        )

        obs_prices = triples_df.filter(
            F.col("predicate") == OBSERVED_PRICE_PRED
        ).select(
            F.col("subject").alias("obs"),
            F.col("object").cast("double").alias("stock_price"),
        )

        prices_df = (
            price_obs
            .join(obs_tickers, "obs", "inner")
            .join(obs_prices, "obs", "inner")
        )

        # One representative price observation per ticker
        w = Window.partitionBy("ticker").orderBy("obs")
        prices_df = (
            prices_df
            .withColumn("rn", F.row_number().over(w))
            .filter(F.col("rn") == 1)
            .drop("rn")
        )

        joined = contracts_df.join(prices_df, "ticker", "inner")

        if joined.head(1) == []:
            logger.info("  No option-stock matches found")
            return None

        # hasUnderlyingPriceObservation triples
        link_triples = joined.select(
            F.col("contract").alias("subject"),
            F.lit(HAS_UNDERLYING_OBS_PRED).alias("predicate"),
            F.col("obs").alias("object"),
        )

        # hasMoneyness triples
        moneyness_df = joined.filter(
            F.col("strike").isNotNull() & F.col("option_type").isNotNull()
        ).withColumn(
            "moneyness",
            F.when(
                F.abs(F.col("strike") - F.col("stock_price"))
                <= F.col("stock_price") * 0.02,
                F.lit(ATM_URI),
            )
            .when(
                (F.col("option_type") == "call")
                & (F.col("strike") < F.col("stock_price")),
                F.lit(ITM_URI),
            )
            .when(
                (F.col("option_type") == "call")
                & (F.col("strike") > F.col("stock_price")),
                F.lit(OTM_URI),
            )
            .when(
                (F.col("option_type") == "put")
                & (F.col("strike") > F.col("stock_price")),
                F.lit(ITM_URI),
            )
            .when(
                (F.col("option_type") == "put")
                & (F.col("strike") < F.col("stock_price")),
                F.lit(OTM_URI),
            ),
        ).filter(F.col("moneyness").isNotNull())

        moneyness_triples = moneyness_df.select(
            F.col("contract").alias("subject"),
            F.lit(HAS_MONEYNESS_PRED).alias("predicate"),
            F.col("moneyness").alias("object"),
        )

        return link_triples.unionAll(moneyness_triples)

    # ================================================================
    # Step 3: Option Strategy Identification
    # ================================================================

    def _identify_option_strategies(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Identify option strategies: straddles, vertical spreads, strangles.
        """
        contracts = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == OPTION_CONTRACT_TYPE)
        ).select(F.col("subject").alias("contract"))

        tickers = triples_df.filter(
            F.col("predicate") == UNDERLYING_TICKER_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").alias("ticker"),
        )

        strikes = triples_df.filter(
            F.col("predicate") == STRIKE_PRICE_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").cast("double").alias("strike"),
        )

        expirations = triples_df.filter(
            F.col("predicate") == EXPIRATION_DATE_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").alias("expiration"),
        )

        types = triples_df.filter(
            F.col("predicate") == OPTION_TYPE_PRED
        ).select(
            F.col("subject").alias("contract"),
            F.col("object").alias("option_type"),
        )

        options_df = (
            contracts
            .join(tickers, "contract", "inner")
            .join(strikes, "contract", "inner")
            .join(expirations, "contract", "inner")
            .join(types, "contract", "inner")
        )

        if options_df.head(1) == []:
            logger.info("  No fully-specified option contracts found")
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
        """Same ticker, expiration, strike — call + put."""
        calls = options_df.filter(F.col("option_type") == "call").select(
            F.col("contract").alias("call_contract"),
            F.col("ticker"),
            F.col("expiration"),
            F.col("strike"),
        )

        puts = options_df.filter(F.col("option_type") == "put").select(
            F.col("contract").alias("put_contract"),
            F.col("ticker").alias("p_ticker"),
            F.col("expiration").alias("p_expiration"),
            F.col("strike").alias("p_strike"),
        )

        straddles = calls.join(
            puts,
            (calls.ticker == puts.p_ticker)
            & (calls.expiration == puts.p_expiration)
            & (calls.strike == puts.p_strike),
            "inner",
        )

        if straddles.head(1) == []:
            return None

        return straddles.select(
            F.col("call_contract").alias("subject"),
            F.lit(STRADDLE_WITH_PRED).alias("predicate"),
            F.col("put_contract").alias("object"),
        )

    def _identify_vertical_spreads(
        self, options_df: DataFrame
    ) -> Optional[DataFrame]:
        """Same ticker, expiration, type — adjacent strikes."""
        w = Window.partitionBy(
            "ticker", "expiration", "option_type"
        ).orderBy("strike")

        with_next = options_df.withColumn(
            "next_contract", F.lead("contract").over(w)
        ).filter(F.col("next_contract").isNotNull())

        if with_next.head(1) == []:
            return None

        call_spreads = with_next.filter(
            F.col("option_type") == "call"
        ).select(
            F.col("contract").alias("subject"),
            F.lit(CALL_SPREAD_WITH_PRED).alias("predicate"),
            F.col("next_contract").alias("object"),
        )

        put_spreads = with_next.filter(
            F.col("option_type") == "put"
        ).select(
            F.col("contract").alias("subject"),
            F.lit(PUT_SPREAD_WITH_PRED).alias("predicate"),
            F.col("next_contract").alias("object"),
        )

        result = call_spreads.unionAll(put_spreads)
        if result.head(1) == []:
            return None

        return result

    def _identify_strangles(
        self, options_df: DataFrame
    ) -> Optional[DataFrame]:
        """Nearest OTM call + put pairs per (ticker, expiration)."""
        chain_stats = options_df.groupBy("ticker", "expiration").agg(
            ((F.min("strike") + F.max("strike")) / 2.0).alias("midpoint")
        )

        options_with_mid = options_df.join(
            chain_stats, ["ticker", "expiration"], "inner"
        )

        # OTM calls: strike > midpoint, ranked ascending
        otm_calls = options_with_mid.filter(
            (F.col("option_type") == "call")
            & (F.col("strike") > F.col("midpoint"))
        )
        w_call = Window.partitionBy("ticker", "expiration").orderBy(
            F.col("strike").asc()
        )
        otm_calls = (
            otm_calls.withColumn("rank", F.row_number().over(w_call))
            .filter(F.col("rank") <= MAX_STRANGLE_PAIRS_PER_CHAIN)
            .select(
                F.col("contract").alias("call_contract"),
                F.col("ticker"),
                F.col("expiration"),
                F.col("rank"),
            )
        )

        # OTM puts: strike < midpoint, ranked descending
        otm_puts = options_with_mid.filter(
            (F.col("option_type") == "put")
            & (F.col("strike") < F.col("midpoint"))
        )
        w_put = Window.partitionBy("ticker", "expiration").orderBy(
            F.col("strike").desc()
        )
        otm_puts = (
            otm_puts.withColumn("rank", F.row_number().over(w_put))
            .filter(F.col("rank") <= MAX_STRANGLE_PAIRS_PER_CHAIN)
            .select(
                F.col("contract").alias("put_contract"),
                F.col("ticker").alias("p_ticker"),
                F.col("expiration").alias("p_expiration"),
                F.col("rank").alias("p_rank"),
            )
        )

        strangles = otm_calls.join(
            otm_puts,
            (otm_calls.ticker == otm_puts.p_ticker)
            & (otm_calls.expiration == otm_puts.p_expiration)
            & (otm_calls.rank == otm_puts.p_rank),
            "inner",
        )

        if strangles.head(1) == []:
            return None

        return strangles.select(
            F.col("call_contract").alias("subject"),
            F.lit(STRANGLE_WITH_PRED).alias("predicate"),
            F.col("put_contract").alias("object"),
        )

    # ================================================================
    # Step 4: Sector Classification
    # ================================================================

    def _classify_sectors(self, triples_df: DataFrame) -> Optional[DataFrame]:
        """
        Classify tickers by sector using MARKET_SECTOR_PATTERNS.

        Broadcast-joins a small lookup table of (symbol -> sector) against
        the ticker symbols in the graph.
        """
        sector_rows = []
        for sector_name, pattern in MARKET_SECTOR_PATTERNS.items():
            sector_uri = str(pattern["sector_uri"])
            relationship = str(pattern["relationship"])
            for symbol in pattern["keywords"].get("stock_prices", []):
                sector_rows.append((symbol, sector_uri, relationship))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows, ["symbol", "sector_uri", "relationship_uri"]
        )

        ticker_symbols = triples_df.filter(
            F.col("predicate") == SYMBOL_PRED
        ).select(
            F.col("subject").alias("ticker_uri"),
            F.col("object").alias("symbol"),
        )

        joined = ticker_symbols.join(
            F.broadcast(sector_df), "symbol", "inner"
        )

        if joined.head(1) == []:
            logger.info("  No sector matches found")
            return None

        belongs_triples = joined.select(
            F.col("ticker_uri").alias("subject"),
            F.lit(BELONGS_TO_SECTOR_PRED).alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        correlation_triples = joined.select(
            F.col("ticker_uri").alias("subject"),
            F.col("relationship_uri").alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        return belongs_triples.unionAll(correlation_triples)

    # ================================================================
    # Step 5: Multi-Source Observation Linking
    # ================================================================

    def _link_multi_source_observations(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Link observations of the same ticker/contract from multiple sources.

        Two sub-steps:
        a) PriceObservations for the same ticker: sameTickerObservation
        b) OptionQuotes for the same contract: sameContractQuote

        Uses self-join with obs1 < obs2 to produce each pair exactly once.
        """
        new_dfs: List[DataFrame] = []

        # (a) Price observations — same ticker, different source URIs
        price_obs = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == PRICE_OBS_TYPE)
        ).select(F.col("subject").alias("obs"))

        obs_tickers = triples_df.filter(
            F.col("predicate") == OBSERVED_TICKER_PRED
        ).select(
            F.col("subject").alias("obs"),
            F.col("object").alias("ticker"),
        )

        obs_df = price_obs.join(obs_tickers, "obs", "inner")

        left = obs_df.select(
            F.col("obs").alias("obs1"), F.col("ticker").alias("t1")
        )
        right = obs_df.select(
            F.col("obs").alias("obs2"), F.col("ticker").alias("t2")
        )

        price_pairs = left.join(
            right,
            (left.t1 == right.t2) & (left.obs1 < right.obs2),
            "inner",
        )

        if price_pairs.head(1) != []:
            new_dfs.append(
                price_pairs.select(
                    F.col("obs1").alias("subject"),
                    F.lit(SAME_TICKER_OBS_PRED).alias("predicate"),
                    F.col("obs2").alias("object"),
                )
            )

        # (b) Option quotes — same contract, different source URIs
        quotes = triples_df.filter(
            (F.col("predicate") == RDF_TYPE)
            & (F.col("object") == OPTION_QUOTE_TYPE)
        ).select(F.col("subject").alias("quote"))

        quote_contracts = triples_df.filter(
            F.col("predicate") == QUOTED_CONTRACT_PRED
        ).select(
            F.col("subject").alias("quote"),
            F.col("object").alias("contract"),
        )

        quote_df = quotes.join(quote_contracts, "quote", "inner")

        ql = quote_df.select(
            F.col("quote").alias("q1"), F.col("contract").alias("c1")
        )
        qr = quote_df.select(
            F.col("quote").alias("q2"), F.col("contract").alias("c2")
        )

        quote_pairs = ql.join(
            qr,
            (ql.c1 == qr.c2) & (ql.q1 < qr.q2),
            "inner",
        )

        if quote_pairs.head(1) != []:
            new_dfs.append(
                quote_pairs.select(
                    F.col("q1").alias("subject"),
                    F.lit(SAME_CONTRACT_QUOTE_PRED).alias("predicate"),
                    F.col("q2").alias("object"),
                )
            )

        if not new_dfs:
            logger.info("  No multi-source observations to link")
            return None

        result = reduce(DataFrame.unionAll, new_dfs)
        logger.info("  Multi-source observation linking complete")
        return result