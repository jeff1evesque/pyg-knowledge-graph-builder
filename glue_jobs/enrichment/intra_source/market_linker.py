"""
Market Intra-Source Enrichment Orchestrator
Coordinates enrichment for stock market data (prices and options)
"""
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, OWL
from glue_jobs.utils.rdf_utils import (
    MARKET, MARKET_ENRICHMENT, UNIFIED
)
from glue_jobs.enrichment.intra_source.base import IntraSourceEnricher
from glue_jobs.enrichment.intra_source.market.patterns import (
    MARKET_SECTOR_PATTERNS, MARKET_OPTION_PATTERNS, MARKET_EXCHANGE_PATTERNS
)
from glue_jobs.enrichment.temporal_unifier import TemporalUnifier
from typing import Dict, Set, Optional, List
import logging

logger = logging.getLogger(__name__)


class MarketIntraSourceLinker(IntraSourceEnricher):
    """
    Orchestrates Market intra-source enrichment for stock prices and options

    Market data comes from multiple sources (Yahoo Finance, MarketWatch, Benzinga)
    and includes:
    - Stock price observations
    - Option contracts (canonical)
    - Option quotes (source-specific observations)

    Enrichment strategies:
    1. Unify ticker entities across data sources
    2. Link temporal sequences of price observations
    3. Link options to underlying stocks
    4. Identify option strategies (spreads, straddles, etc.)
    5. Link stocks by sector and exchange
    6. Link multi-source observations of same ticker/contract
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.stats.update({
            'ticker_unified': 0,
            'option_stock_links': 0,
            'option_strategy_links': 0,
            'multi_source_links': 0
        })
        self.available_datasets = self.detect_datasets()
        logger.info(f"Detected Market datasets: {', '.join(self.available_datasets)}")

    def detect_datasets(self) -> Set[str]:
        """Detect if market data is present in the graph"""
        datasets = set()

        # Check for stock price observations
        price_query = f"""
        ASK {{
            ?s a <{MARKET.PriceObservation}> .
        }}
        """
        if self.graph.query(price_query).askAnswer:
            datasets.add('stock_prices')

        # Check for option contracts
        options_query = f"""
        ASK {{
            ?s a <{MARKET.OptionContract}> .
        }}
        """
        if self.graph.query(options_query).askAnswer:
            datasets.add('options')

        return datasets

    def enrich(self) -> Dict[str, int]:
        """Run all Market intra-source enrichment steps"""
        if not self.available_datasets:
            logger.info("No Market data detected, skipping enrichment")
            return {'total_triples_added': 0}

        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting Market Intra-Source Enrichment")
        logger.info("=" * 60)

        # Step 1: Unify ticker entities
        logger.info("\n[Step 1/7] Unifying ticker entities...")
        self.unify_ticker_entities()

        # Step 2: Unify temporal entities
        logger.info("\n[Step 2/7] Unifying temporal entities...")
        self.unify_temporal_entities()

        # Step 3: Link price observation sequences
        logger.info("\n[Step 3/7] Linking price observation sequences...")
        self.link_price_sequences()

        # Step 4: Link options to underlying stocks
        logger.info("\n[Step 4/7] Linking options to underlying stocks...")
        self.link_options_to_stocks()

        # Step 5: Identify option strategies
        logger.info("\n[Step 5/7] Identifying option strategies...")
        self.identify_option_strategies()

        # Step 6: Apply sector patterns
        logger.info("\n[Step 6/7] Applying sector patterns...")
        self.apply_sector_patterns()

        # Step 7: Link multi-source observations
        logger.info("\n[Step 7/7] Linking multi-source observations...")
        self.link_multi_source_observations()

        final_count = len(self.graph)
        enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("Market Intra-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {enrichment_count}")
        logger.info(f"  - Ticker unification: {self.stats['ticker_unified']}")
        logger.info(f"  - Temporal unification: {self.stats['temporal_unified']}")
        logger.info(f"  - Price sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Option-stock links: {self.stats['option_stock_links']}")
        logger.info(f"  - Option strategy links: {self.stats['option_strategy_links']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Multi-source links: {self.stats['multi_source_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': enrichment_count,
            'available_datasets': list(self.available_datasets),
            **self.stats
        }

    def unify_ticker_entities(self):
        """
        Unify ticker entities across data sources

        Creates unified StockTicker entities since the same ticker
        may be observed by multiple sources (yahoo, marketwatch, benzinga)
        """

        # Collect all tickers by symbol
        tickers_by_symbol = {}

        query = f"""
        SELECT DISTINCT ?ticker ?symbol WHERE {{
            ?ticker a <{MARKET.StockTicker}> ;
                    <{MARKET.symbol}> ?symbol .
        }}
        """

        for row in self.graph.query(query):
            symbol = str(row.symbol)
            if symbol not in tickers_by_symbol:
                tickers_by_symbol[symbol] = []
            tickers_by_symbol[symbol].append(row.ticker)

        # Create unified ticker entities
        for symbol, ticker_uris in tickers_by_symbol.items():
            if len(ticker_uris) > 1:  # Only unify if multiple references exist
                unified_ticker = UNIFIED[f"Ticker_{symbol}"]
                self.graph.add((unified_ticker, RDF.type, MARKET_ENRICHMENT.UnifiedTicker))
                self.graph.add((unified_ticker, MARKET_ENRICHMENT.hasSymbol, Literal(symbol)))

                for ticker_uri in ticker_uris:
                    self.graph.add((unified_ticker, OWL.sameAs, ticker_uri))
                    self.stats['ticker_unified'] += 1

        logger.info(f"  Unified {len(tickers_by_symbol)} tickers across data sources")
        logger.info(f"  Created {self.stats['ticker_unified']} ticker unification links")

    def unify_temporal_entities(self):
        """Unify temporal entities from market data"""
        # Use TemporalUnifier with Market-only scope
        unifier = TemporalUnifier(self.graph)
        unifier.available_sources = {'market'}  # Restrict to Market only

        stats = unifier.unify_all_sources()
        self.stats['temporal_unified'] = stats['temporal_links']

        logger.info(f"  Unified {stats['months_unified']} months and {stats['years_unified']} years")
        logger.info(f"  Created {stats['temporal_links']} temporal unification links")

    def link_price_sequences(self):
        """
        Link price observations in chronological order for same ticker
        """
        # Get all price observations with ticker and timestamp
        query = f"""
        SELECT ?obs ?ticker ?observedAt WHERE {{
            ?obs a <{MARKET.PriceObservation}> ;
                 <{MARKET.observedTicker}> ?ticker ;
                 <{MARKET.observedAt}> ?observedAt .
        }}
        """

        results = list(self.graph.query(query))

        # Group by ticker
        by_ticker = {}
        for row in results:
            ticker = str(row.ticker)
            if ticker not in by_ticker:
                by_ticker[ticker] = []

            by_ticker[ticker].append({
                'obs': row.obs,
                'timestamp': str(row.observedAt)
            })

        # Sort and link observations chronologically
        links_added = 0
        for ticker, observations in by_ticker.items():
            if len(observations) < 2:
                continue

            # Sort by timestamp
            observations.sort(key=lambda x: x['timestamp'])

            # Link consecutive observations
            for i in range(len(observations) - 1):
                current = observations[i]['obs']
                next_obs = observations[i + 1]['obs']

                self.graph.add((
                    current,
                    MARKET_ENRICHMENT.precedes,
                    next_obs
                ))
                links_added += 1
                self.stats['temporal_sequences'] += 1

        logger.info(f"  Added {links_added} price sequence links")

    def link_options_to_stocks(self):
        """
        Link option contracts to their underlying stock price observations

        Creates relationships between options and recent stock prices
        to enable analysis of option pricing relative to underlying
        """

        # Get all option contracts with their underlying tickers
        query = f"""
        SELECT ?contract ?ticker ?strike WHERE {{
            ?contract a <{MARKET.OptionContract}> ;
                      <{MARKET.underlyingTicker}> ?ticker ;
                      <{MARKET.strikePrice}> ?strike .
        }}
        """

        results = list(self.graph.query(query))

        # For each contract, find recent price observations
        links_added = 0
        for row in results:
            contract = row.contract
            ticker = row.ticker
            strike = float(row.strike)

            # Find price observations for this ticker
            price_query = f"""
            SELECT ?obs ?price WHERE {{
                ?obs a <{MARKET.PriceObservation}> ;
                     <{MARKET.observedTicker}> <{ticker}> ;
                     <{MARKET.observedPrice}> ?price .
            }}
            LIMIT 1
            """

            price_results = list(self.graph.query(price_query))
            if price_results:
                obs = price_results[0].obs
                price = float(price_results[0].price)

                # Link option to price observation
                self.graph.add((
                    contract,
                    MARKET_ENRICHMENT.hasUnderlyingPriceObservation,
                    obs
                ))
                links_added += 1
                self.stats['option_stock_links'] += 1

                # Classify option as ITM, ATM, or OTM
                moneyness = self._classify_option_moneyness(strike, price, row.contract)
                if moneyness:
                    self.graph.add((
                        contract,
                        MARKET_ENRICHMENT.hasMoneyness,
                        moneyness
                    ))

        logger.info(f"  Added {links_added} option-stock links")

    def _classify_option_moneyness(self, strike: float, stock_price: float, contract_uri: URIRef) -> Optional[URIRef]:
        """
        Classify option as in-the-money, at-the-money, or out-of-the-money
        """
        # Get option type
        type_query = f"""
        SELECT ?type WHERE {{
            <{contract_uri}> <{MARKET.optionType}> ?type .
        }}
        """

        type_results = list(self.graph.query(type_query))
        if not type_results:
            return None

        option_type = str(type_results[0].type)

        # Calculate moneyness
        threshold = stock_price * 0.02  # 2% threshold for ATM

        if abs(strike - stock_price) <= threshold:
            return MARKET_ENRICHMENT.AtTheMoney
        elif option_type == 'call':
            if strike < stock_price:
                return MARKET_ENRICHMENT.InTheMoney
            else:
                return MARKET_ENRICHMENT.OutOfTheMoney
        else:  # put
            if strike > stock_price:
                return MARKET_ENRICHMENT.InTheMoney
            else:
                return MARKET_ENRICHMENT.OutOfTheMoney

    def identify_option_strategies(self):
        """
        Identify common option strategies by analyzing option contracts

        Detects:
        - Vertical spreads (call/put spreads)
        - Straddles (same strike, call + put)
        - Strangles (different strikes, call + put)
        - Iron condors
        - Butterflies
        """

        # Get all option contracts grouped by ticker and expiration
        query = f"""
        SELECT ?contract ?ticker ?strike ?expiration ?type WHERE {{
            ?contract a <{MARKET.OptionContract}> ;
                      <{MARKET.underlyingTicker}> ?ticker ;
                      <{MARKET.strikePrice}> ?strike ;
                      <{MARKET.expirationDate}> ?expiration ;
                      <{MARKET.optionType}> ?type .
        }}
        ORDER BY ?ticker ?expiration ?strike
        """

        results = list(self.graph.query(query))

        # Group by ticker and expiration
        by_ticker_exp = {}
        for row in results:
            key = (str(row.ticker), str(row.expiration))
            if key not in by_ticker_exp:
                by_ticker_exp[key] = []

            by_ticker_exp[key].append({
                'contract': row.contract,
                'strike': float(row.strike),
                'type': str(row.type)
            })

        # Identify strategies
        links_added = 0

        for (ticker, expiration), contracts in by_ticker_exp.items():
            # Identify straddles (same strike, call + put)
            links_added += self._identify_straddles(contracts)

            # Identify vertical spreads
            links_added += self._identify_vertical_spreads(contracts)

            # Identify strangles
            links_added += self._identify_strangles(contracts)

        logger.info(f"  Added {links_added} option strategy links")

    def _identify_straddles(self, contracts: List[Dict]) -> int:
        """Identify straddle strategies (same strike, call + put)"""
        links_added = 0

        # Group by strike
        by_strike = {}
        for contract in contracts:
            strike = contract['strike']
            if strike not in by_strike:
                by_strike[strike] = {'calls': [], 'puts': []}

            if contract['type'] == 'call':
                by_strike[strike]['calls'].append(contract['contract'])
            else:
                by_strike[strike]['puts'].append(contract['contract'])

        # Find straddles
        for strike, options in by_strike.items():
            if options['calls'] and options['puts']:
                for call in options['calls']:
                    for put in options['puts']:
                        self.graph.add((
                            call,
                            MARKET_ENRICHMENT.straddleWith,
                            put
                        ))
                        links_added += 1
                        self.stats['option_strategy_links'] += 1

        return links_added

    def _identify_vertical_spreads(self, contracts: List[Dict]) -> int:
        """Identify vertical spread strategies"""
        links_added = 0

        # Separate calls and puts
        calls = [c for c in contracts if c['type'] == 'call']
        puts = [c for c in contracts if c['type'] == 'put']

        # Sort by strike
        calls.sort(key=lambda x: x['strike'])
        puts.sort(key=lambda x: x['strike'])

        # Link adjacent strikes for calls
        for i in range(len(calls) - 1):
            self.graph.add((
                calls[i]['contract'],
                MARKET_ENRICHMENT.callSpreadWith,
                calls[i + 1]['contract']
            ))
            links_added += 1
            self.stats['option_strategy_links'] += 1

        # Link adjacent strikes for puts
        for i in range(len(puts) - 1):
            self.graph.add((
                puts[i]['contract'],
                MARKET_ENRICHMENT.putSpreadWith,
                puts[i + 1]['contract']
            ))
            links_added += 1
            self.stats['option_strategy_links'] += 1

        return links_added

    def _identify_strangles(self, contracts: List[Dict]) -> int:
        """Identify strangle strategies (different strikes, call + put)"""
        links_added = 0

        calls = [c for c in contracts if c['type'] == 'call']
        puts = [c for c in contracts if c['type'] == 'put']

        # Link calls with puts at different strikes
        for call in calls:
            for put in puts:
                if call['strike'] != put['strike']:
                    self.graph.add((
                        call['contract'],
                        MARKET_ENRICHMENT.strangleWith,
                        put['contract']
                    ))
                    links_added += 1
                    self.stats['option_strategy_links'] += 1

        return links_added

    def apply_sector_patterns(self):
        """
        Apply sector-based linking patterns

        Links tickers to industry sectors based on symbol matching
        """
        logger.info("  Applying sector patterns...")

        for sector_name, pattern in MARKET_SECTOR_PATTERNS.items():
            sector_uri = pattern['sector_uri']
            relationship = pattern['relationship']
            keywords = pattern['keywords'].get('stock_prices', [])

            if not keywords:
                continue

            # Find tickers matching sector keywords (ticker symbols)
            for ticker_symbol in keywords:
                ticker_query = f"""
                SELECT ?ticker WHERE {{
                    ?ticker a <{MARKET.StockTicker}> ;
                            <{MARKET.symbol}> "{ticker_symbol}" .
                }}
                """

                results = list(self.graph.query(ticker_query))

                for row in results:
                    # Link ticker to sector
                    self.graph.add((
                        row.ticker,
                        MARKET_ENRICHMENT.belongsToSector,
                        sector_uri
                    ))

                    # Add sector correlation relationship
                    self.graph.add((
                        row.ticker,
                        relationship,
                        sector_uri
                    ))

                    self.stats['sector_links'] += 2

        logger.info(f"  Added {self.stats['sector_links']} sector links")

    def link_multi_source_observations(self):
        """
        Link observations of the same ticker/contract from multiple sources

        Enables comparison and validation across data sources
        """

        # Link price observations for same ticker
        ticker_query = f"""
        SELECT ?ticker (GROUP_CONCAT(?obs; separator=",") AS ?observations) WHERE {{
            ?obs a <{MARKET.PriceObservation}> ;
                 <{MARKET.observedTicker}> ?ticker .
        }}
        GROUP BY ?ticker
        HAVING (COUNT(?obs) > 1)
        """

        results = list(self.graph.query(ticker_query))

        links_added = 0
        for row in results:
            obs_list = str(row.observations).split(',')

            # Link all observations for same ticker
            for i, obs1 in enumerate(obs_list):
                for obs2 in obs_list[i + 1:]:
                    self.graph.add((
                        URIRef(obs1),
                        MARKET_ENRICHMENT.sameTickerObservation,
                        URIRef(obs2)
                    ))
                    links_added += 1
                    self.stats['multi_source_links'] += 1

        logger.info(f"  Added {links_added} multi-source observation links")