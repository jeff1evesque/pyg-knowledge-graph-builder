"""
Cross-Source Enrichment Linker
Links entities across different data source families (BLS, SEC, Market, NOAA)
"""
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL
from glue_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT, NOAA_ENRICHMENT,
    UNIFIED, CPI, PPI, JOLTS, EMPSIT, SEC_FILINGS, MARKET, CAP
)
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossSourceLinker:
    """
    Links entities across different data source families

    Strategies:
    1. Temporal Alignment - Unify temporal entities across all sources
    2. Sector-Based Linking - Link entities in same economic sectors
    3. Company/Ticker Linking - Link entities referencing same companies
    4. Geographic Linking - Link entities by geographic region
    5. Causal Relationships - Discover potential causal links
    6. Measurement Type Alignment - Link similar measurement types
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.stats = {
            'temporal_unified': 0,
            'sector_links': 0,
            'company_links': 0,
            'geographic_links': 0,
            'causal_links': 0,
            'measurement_links': 0
        }

        # Detect available data sources
        self.available_sources = self._detect_sources()
        logger.info(f"Detected data sources: {', '.join(self.available_sources)}")

    def _detect_sources(self) -> Set[str]:
        """Detect which data sources are present in the graph"""
        sources = set()

        # Check for BLS data
        bls_query = """
        ASK {
            { ?s a cpi:Index } UNION
            { ?s a ppi:MonthlyChange } UNION
            { ?s a jolts:JobOpenings } UNION
            { ?s a empsit:Employment }
        }
        """
        if self.graph.query(bls_query).askAnswer:
            sources.add('bls')

        # Check for SEC data
        sec_query = """
        ASK {
            { ?s a filings:Form3 } UNION
            { ?s a filings:Form4 } UNION
            { ?s a sec:AdministrativeProceeding }
        }
        """
        if self.graph.query(sec_query).askAnswer:
            sources.add('sec')

        # Check for Market data
        market_query = """
        ASK {
            { ?s a market:PriceObservation } UNION
            { ?s a market:OptionContract }
        }
        """
        if self.graph.query(market_query).askAnswer:
            sources.add('market')

        # Check for NOAA data
        noaa_query = """
        ASK {
            ?s a cap:Alert .
        }
        """
        if self.graph.query(noaa_query).askAnswer:
            sources.add('noaa')

        return sources

    def enrich(self) -> Dict[str, int]:
        """Run all cross-source enrichment steps"""
        if len(self.available_sources) < 2:
            logger.info("Less than 2 data sources detected, skipping cross-source enrichment")
            return {'total_triples_added': 0}

        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting Cross-Source Enrichment")
        logger.info("=" * 60)

        # Step 1: Temporal alignment (always run if multiple sources)
        logger.info("\n[Step 1/6] Aligning temporal entities across sources...")
        self.align_temporal_entities()

        # Step 2: Sector-based linking
        logger.info("\n[Step 2/6] Creating sector-based links...")
        self.link_by_sector()

        # Step 3: Company/ticker linking
        logger.info("\n[Step 3/6] Linking by company/ticker...")
        self.link_by_company()

        # Step 4: Geographic linking
        logger.info("\n[Step 4/6] Linking by geographic region...")
        self.link_by_geography()

        # Step 5: Causal relationships
        logger.info("\n[Step 5/6] Creating causal relationships...")
        self.create_causal_links()

        # Step 6: Measurement type alignment
        logger.info("\n[Step 6/6] Aligning measurement types...")
        self.align_measurement_types()

        final_count = len(self.graph)
        enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("Cross-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {enrichment_count}")
        logger.info(f"  - Temporal alignment: {self.stats['temporal_unified']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Company links: {self.stats['company_links']}")
        logger.info(f"  - Geographic links: {self.stats['geographic_links']}")
        logger.info(f"  - Causal links: {self.stats['causal_links']}")
        logger.info(f"  - Measurement links: {self.stats['measurement_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': enrichment_count,
            'available_sources': list(self.available_sources),
            **self.stats
        }

    def align_temporal_entities(self):
        """
        Create unified temporal entities that all sources reference

        Example:
            cpi:November → unified:November2024
            sec:November → unified:November2024
            market:November → unified:November2024
        """
        # Collect all temporal entities from all sources
        temporal_entities = self._collect_temporal_entities()

        # Create unified temporal entities
        for (month_name, year_value), source_entities in temporal_entities.items():
            if len(source_entities) < 2:
                continue  # Only unify if multiple sources reference it

            # Create unified month and year
            unified_month = UNIFIED[month_name]
            unified_year = UNIFIED[f"Year{year_value}"]

            # Add unified entities
            self.graph.add((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))
            self.graph.add((unified_month, RDFS.label, Literal(month_name)))

            self.graph.add((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))
            self.graph.add((unified_year, RDFS.label, Literal(year_value)))

            # Link all source-specific temporal entities to unified ones
            for source, entity in source_entities:
                self.graph.add((unified_month, OWL.sameAs, entity))
                self.stats['temporal_unified'] += 1

        logger.info(f"  Unified {len(temporal_entities)} temporal entities across sources")

    def _collect_temporal_entities(self) -> Dict[Tuple[str, str], List[Tuple[str, URIRef]]]:
        """Collect temporal entities from all sources"""
        temporal_entities = {}

        # Query each source for temporal entities
        # This is a simplified example - actual implementation would be more comprehensive

        if 'bls' in self.available_sources:
            query = """
            SELECT DISTINCT ?month ?year WHERE {
                { ?entity cpi:hasMonth ?month ; cpi:hasYear ?year } UNION
                { ?entity ppi:hasStartMonth ?month ; ppi:hasStartYear ?year }
            }
            """
            for row in self.graph.query(query):
                month_name = str(row.month).split('/')[-1]
                year_value = str(row.year).split('/')[-1]
                key = (month_name, year_value)

                if key not in temporal_entities:
                    temporal_entities[key] = []
                temporal_entities[key].append(('bls', row.month))

        # Similar queries for other sources...

        return temporal_entities

    def link_by_sector(self):
        """
        Link entities across sources that belong to same economic sector

        Example:
            cpi:EnergyEntity + ppi:EnergyGoodsEntity + market:EnergyStockEntity
            → all linked to unified:EnergySector
        """
        # Define unified sectors
        sectors = {
            'energy': {
                'uri': UNIFIED.EnergySector,
                'bls_keywords': ['Energy', 'Gasoline', 'Fuel', 'Electricity'],
                'sec_keywords': ['Energy', 'Oil', 'Gas'],
                'market_tickers': ['XOM', 'CVX', 'COP', 'SLB']
            },
            'technology': {
                'uri': UNIFIED.TechnologySector,
                'bls_keywords': ['Computer', 'Software', 'Electronics'],
                'sec_keywords': ['Technology', 'Software'],
                'market_tickers': ['AAPL', 'MSFT', 'GOOGL', 'META']
            },
            'healthcare': {
                'uri': UNIFIED.HealthcareSector,
                'bls_keywords': ['Medical', 'Healthcare', 'Pharmaceutical'],
                'sec_keywords': ['Healthcare', 'Pharmaceutical', 'Biotech'],
                'market_tickers': ['JNJ', 'UNH', 'PFE', 'ABBV']
            },
            # Add more sectors...
        }

        for sector_name, sector_config in sectors.items():
            sector_uri = sector_config['uri']

            # Add sector entity
            self.graph.add((sector_uri, RDF.type, BLS_ENRICHMENT.EconomicSector))
            self.graph.add((sector_uri, RDFS.label, Literal(sector_name.title())))

            # Link BLS entities
            if 'bls' in self.available_sources:
                self._link_bls_to_sector(sector_uri, sector_config['bls_keywords'])

            # Link SEC entities
            if 'sec' in self.available_sources:
                self._link_sec_to_sector(sector_uri, sector_config['sec_keywords'])

            # Link Market entities
            if 'market' in self.available_sources:
                self._link_market_to_sector(sector_uri, sector_config['market_tickers'])

        logger.info(f"  Created {self.stats['sector_links']} sector-based links")

    def _link_bls_to_sector(self, sector_uri: URIRef, keywords: List[str]):
        """Link BLS entities to sector"""
        for keyword in keywords:
            # Find BLS entities with labels containing keyword
            query = f"""
            SELECT DISTINCT ?entity WHERE {{
                ?entity rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{keyword}")))
                FILTER(
                    EXISTS {{ ?entity a cpi:Category }} ||
                    EXISTS {{ ?entity a ppi:CommodityGrouping }}
                )
            }}
            """

            for row in self.graph.query(query):
                self.graph.add((row.entity, BLS_ENRICHMENT.belongsToSector, sector_uri))
                self.stats['sector_links'] += 1

    def _link_sec_to_sector(self, sector_uri: URIRef, keywords: List[str]):
        """Link SEC entities to sector"""
        # Similar implementation for SEC data
        pass

    def _link_market_to_sector(self, sector_uri: URIRef, tickers: List[str]):
        """Link Market entities to sector"""
        for ticker in tickers:
            query = f"""
            SELECT DISTINCT ?entity WHERE {{
                ?entity a market:StockTicker ;
                        market:symbol "{ticker}" .
            }}
            """

            for row in self.graph.query(query):
                self.graph.add((row.entity, BLS_ENRICHMENT.belongsToSector, sector_uri))
                self.stats['sector_links'] += 1

    def link_by_company(self):
        """
        Link entities referencing same company across sources

        Example:
            sec:AAPL_Filing + market:AAPL_Stock → unified:Company_AAPL
        """
        if 'sec' not in self.available_sources or 'market' not in self.available_sources:
            logger.info("  Skipping company linking (requires both SEC and Market data)")
            return

        # Get all tickers from market data
        ticker_query = """
        SELECT DISTINCT ?ticker ?symbol WHERE {
            ?ticker a market:StockTicker ;
                    market:symbol ?symbol .
        }
        """

        for row in self.graph.query(ticker_query):
            symbol = str(row.symbol)

            # Create unified company entity
            unified_company = UNIFIED[f"Company_{symbol}"]
            self.graph.add((unified_company, RDF.type, BLS_ENRICHMENT.UnifiedCompany))
            self.graph.add((unified_company, BLS_ENRICHMENT.ticker, Literal(symbol)))

            # Link market ticker
            self.graph.add((row.ticker, BLS_ENRICHMENT.refersToCompany, unified_company))
            self.stats['company_links'] += 1

            # Find SEC filings for this ticker
            sec_query = f"""
            SELECT DISTINCT ?filing WHERE {{
                ?filing a filings:SECFiling ;
                        filings:hasIssuerTicker "{symbol}" .
            }}
            """

            for sec_row in self.graph.query(sec_query):
                self.graph.add((sec_row.filing, BLS_ENRICHMENT.refersToCompany, unified_company))
                self.stats['company_links'] += 1

        logger.info(f"  Created {self.stats['company_links']} company-based links")

    def link_by_geography(self):
        """
        Link entities by geographic region

        Example:
            jolts:NortheastRegion + noaa:NortheastAlert → unified:NortheastRegion
        """
        if len(self.available_sources) < 2:
            return

        # Define unified regions
        regions = {
            'northeast': ['Northeast', 'New England', 'Mid-Atlantic'],
            'southeast': ['Southeast', 'South Atlantic'],
            'midwest': ['Midwest', 'Great Lakes'],
            'southwest': ['Southwest'],
            'west': ['West', 'Pacific']
        }

        for region_name, region_keywords in regions.items():
            unified_region = UNIFIED[f"{region_name.title()}Region"]
            self.graph.add((unified_region, RDF.type, BLS_ENRICHMENT.GeographicRegion))
            self.graph.add((unified_region, RDFS.label, Literal(region_name.title())))

            # Link entities from different sources
            for keyword in region_keywords:
                # Find entities with region in label
                query = f"""
                SELECT DISTINCT ?entity WHERE {{
                    ?entity rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(?label), LCASE("{keyword}")))
                }}
                """

                for row in self.graph.query(query):
                    self.graph.add((row.entity, BLS_ENRICHMENT.hasRegion, unified_region))
                    self.stats['geographic_links'] += 1

        logger.info(f"  Created {self.stats['geographic_links']} geographic links")

    def create_causal_links(self):
        """
        Create potential causal relationships across sources

        Examples:
            - PPI → CPI (producer prices lead consumer prices)
            - JOLTS → CPI (employment affects consumer spending)
            - Weather → Market (weather affects commodity prices)
            - SEC Filings → Market (filings affect stock prices)
        """
        # PPI → CPI relationships
        if 'bls' in self.available_sources:
            self._link_ppi_to_cpi()

        # JOLTS → CPI relationships
        if 'bls' in self.available_sources:
            self._link_jolts_to_cpi()

        # Weather → Market relationships
        if 'noaa' in self.available_sources and 'market' in self.available_sources:
            self._link_weather_to_market()

        # SEC → Market relationships
        if 'sec' in self.available_sources and 'market' in self.available_sources:
            self._link_sec_to_market()

        logger.info(f"  Created {self.stats['causal_links']} causal links")

    def _link_ppi_to_cpi(self):
        """Link PPI commodities to related CPI categories"""
        # Example: PPI Food Manufacturing → CPI Food
        correlations = [
            ('Food', 'Food'),
            ('Energy', 'Energy'),
            ('Transportation', 'Transportation')
        ]

        for ppi_keyword, cpi_keyword in correlations:
            # Find PPI entities
            ppi_query = f"""
            SELECT DISTINCT ?ppi WHERE {{
                ?ppi a ppi:CommodityGrouping ;
                     rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{ppi_keyword}")))
            }}
            """

            # Find CPI entities
            cpi_query = f"""
            SELECT DISTINCT ?cpi WHERE {{
                ?cpi a cpi:Category ;
                     rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{cpi_keyword}")))
            }}
            """

            ppi_entities = list(self.graph.query(ppi_query))
            cpi_entities = list(self.graph.query(cpi_query))

            # Create causal links
            for ppi_row in ppi_entities:
                for cpi_row in cpi_entities:
                    self.graph.add((
                        ppi_row.ppi,
                        BLS_ENRICHMENT.leadsTo,
                        cpi_row.cpi
                    ))
                    self.stats['causal_links'] += 1

    def _link_jolts_to_cpi(self):
        """Link JOLTS employment to CPI consumer spending"""
        # Employment levels affect consumer spending
        pass

    def _link_weather_to_market(self):
        """Link weather events to affected market sectors"""
        # Example: Hurricane → Energy stocks, Agricultural commodities
        pass

    def _link_sec_to_market(self):
        """Link SEC filings to stock price movements"""
        # Filings can affect stock prices
        pass

    def align_measurement_types(self):
        """
        Align similar measurement types across sources

        Examples:
            - Price indices (CPI, PPI)
            - Rate measurements (JOLTS, LAUS unemployment rates)
            - Change measurements (CPI %, PPI %, EMPSIT employment change)
        """
        # Price indices
        price_indices = []

        if 'bls' in self.available_sources:
            # Find CPI indices
            cpi_query = """
            SELECT DISTINCT ?index WHERE {
                ?index a cpi:Index .
            }
            """
            price_indices.extend([row.index for row in self.graph.query(cpi_query)])

            # Find PPI indices
            ppi_query = """
            SELECT DISTINCT ?index WHERE {
                ?index a ppi:Index .
            }
            """
            price_indices.extend([row.index for row in self.graph.query(ppi_query)])

        # Mark all as price indices
        for index in price_indices:
            self.graph.add((index, RDF.type, BLS_ENRICHMENT.PriceIndex))
            self.stats['measurement_links'] += 1

        logger.info(f"  Created {self.stats['measurement_links']} measurement type links")


def enrich_cross_source(graph: Graph) -> Dict[str, int]:
    """
    Main entry point for cross-source enrichment
    """
    linker = CrossSourceLinker(graph)
    return linker.enrich()