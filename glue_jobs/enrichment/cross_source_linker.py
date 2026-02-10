"""
Cross-Source Enrichment Linker
Links entities across different data source families (BLS, SEC, Market, NOAA)

Enhanced with actual BLS patterns, correlations, and helper functions.
"""
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
from glue_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT, NOAA_ENRICHMENT,
    UNIFIED, CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER,
    SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP, MARKET, CAP,
    get_month_name, get_year_value
)
from glue_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from glue_jobs.enrichment.intra_source.bls.correlations import KNOWN_CORRELATIONS
from typing import Dict, Set
import logging

logger = logging.getLogger(__name__)


def normalize_keyword_for_uri_matching(keyword: str) -> str:
    """
    Normalize a keyword for matching against URIs
    
    Reused from bls_linker.py for consistency
    """
    return (keyword
            .replace(' ', '_')
            .replace("'", '')
            .replace(',', '')
            .replace('-', '_')
            .replace('(', '')
            .replace(')', ''))


class CrossSourceLinker:
    """
    Links entities across different data source families
    
    Strategies:
    1. Temporal Alignment - Unify temporal entities across all sources
    2. Sector-Based Linking - Link entities in same economic sectors (using BLS_SECTOR_PATTERNS)
    3. Company/Ticker Linking - Link entities referencing same companies
    4. Geographic Linking - Link entities by geographic region
    5. Causal Relationships - Discover potential causal links (extending KNOWN_CORRELATIONS)
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
        
        # Load BLS patterns for cross-source use
        self.sector_patterns = BLS_SECTOR_PATTERNS
        self.bls_correlations = KNOWN_CORRELATIONS
    
    def _detect_sources(self) -> Set[str]:
        """Detect which data sources are present in the graph"""
        sources = set()
        
        # Check for BLS data (any BLS dataset)
        bls_namespaces = [CPI, PPI, JOLTS, EMPSIT, ECI, XIMPIM, LAUS, METRO, REALER]
        for namespace in bls_namespaces:
            query = f"""
            ASK {{
                ?s ?p ?o .
                FILTER(STRSTARTS(STR(?s), "{namespace}"))
            }}
            """
            if self.graph.query(query).askAnswer:
                sources.add('bls')
                break
        
        # Check for SEC data
        sec_query = f"""
        ASK {{
            {{ ?s a filings:Form3 }} UNION
            {{ ?s a filings:Form4 }} UNION
            {{ ?s a sec:AdministrativeProceeding }} UNION
            {{ ?s a seclit:LitigationRelease }} UNION
            {{ ?s a secsusp:TradingSuspension }}
        }}
        """
        if self.graph.query(sec_query).askAnswer:
            sources.add('sec')
        
        # Check for Market data
        market_query = f"""
        ASK {{
            {{ ?s a <{MARKET.PriceObservation}> }} UNION
            {{ ?s a <{MARKET.OptionContract}> }}
        }}
        """
        if self.graph.query(market_query).askAnswer:
            sources.add('market')
        
        # Check for NOAA data
        noaa_query = f"""
        ASK {{
            ?s a <{CAP.Alert}> .
        }}
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
        
        # Step 2: Sector-based linking (using BLS_SECTOR_PATTERNS)
        logger.info("\n[Step 2/6] Creating sector-based links...")
        self.link_by_sector()
        
        # Step 3: Company/ticker linking
        logger.info("\n[Step 3/6] Linking by company/ticker...")
        self.link_by_company()
        
        # Step 4: Geographic linking
        logger.info("\n[Step 4/6] Linking by geographic region...")
        self.link_by_geography()
        
        # Step 5: Causal relationships (extending KNOWN_CORRELATIONS)
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
        
        Extends the BLS temporal unification to include SEC, Market, and NOAA
        
        Example:
            cpi:November + sec:November + market:November + noaa:November 
            → unified:November2024
        """
        # Collect temporal entities from all sources
        months_by_name = {}
        years_by_value = {}
        
        # BLS temporal entities (already collected by BLS intra-source enrichment)
        # We just need to link SEC, Market, and NOAA to the same unified entities
        
        # SEC temporal entities (from filing dates, proceeding dates, etc.)
        if 'sec' in self.available_sources:
            self._collect_sec_temporal_entities(months_by_name, years_by_value)
        
        # Market temporal entities (from price observations, option expirations)
        if 'market' in self.available_sources:
            self._collect_market_temporal_entities(months_by_name, years_by_value)
        
        # NOAA temporal entities (from alert timestamps)
        if 'noaa' in self.available_sources:
            self._collect_noaa_temporal_entities(months_by_name, years_by_value)
        
        # Create unified temporal entities and link with owl:sameAs
        for month_name, month_uris in months_by_name.items():
            if len(month_uris) < 2:  # Only unify if multiple sources reference it
                continue
            
            unified_month = UNIFIED[month_name]
            
            # Add unified month if not already exists
            if not list(self.graph.triples((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))):
                self.graph.add((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))
                self.graph.add((unified_month, RDFS.label, Literal(month_name)))
            
            # Link all source-specific temporal entities
            for month_uri in month_uris:
                if not list(self.graph.triples((unified_month, OWL.sameAs, month_uri))):
                    self.graph.add((unified_month, OWL.sameAs, month_uri))
                    self.stats['temporal_unified'] += 1
        
        for year_value, year_uris in years_by_value.items():
            if len(year_uris) < 2:
                continue
            
            unified_year = UNIFIED[f"Year{year_value}"]
            
            # Add unified year if not already exists
            if not list(self.graph.triples((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))):
                self.graph.add((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))
                self.graph.add((unified_year, RDFS.label, Literal(year_value)))
            
            # Link all source-specific temporal entities
            for year_uri in year_uris:
                if not list(self.graph.triples((unified_year, OWL.sameAs, year_uri))):
                    self.graph.add((unified_year, OWL.sameAs, year_uri))
                    self.stats['temporal_unified'] += 1
        
        logger.info(f"  Unified {len(months_by_name)} months and {len(years_by_value)} years across sources")
    
    def _collect_sec_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """Collect temporal entities from SEC data"""
        # SEC filings have dates - extract month/year from them
        query = """
        SELECT DISTINCT ?date WHERE {
            { ?filing filings:hasPeriodOfReport ?date } UNION
            { ?filing filings:hasReportDate ?date } UNION
            { ?proceeding sec:initiationDate ?date } UNION
            { ?litigation seclit:filingDate ?date } UNION
            { ?suspension secsusp:startDate ?date }
        }
        """
        
        for row in self.graph.query(query):
            date_str = str(row.date)
            try:
                from dateutil import parser
                dt = parser.parse(date_str)
                
                month_name = dt.strftime('%B')
                year_value = str(dt.year)
                
                # Create synthetic SEC temporal URIs
                sec_month = URIRef(f"https://www.sec.gov/temporal/{month_name}")
                sec_year = URIRef(f"https://www.sec.gov/temporal/{year_value}")
                
                if month_name not in months_by_name:
                    months_by_name[month_name] = []
                if sec_month not in months_by_name[month_name]:
                    months_by_name[month_name].append(sec_month)
                
                if year_value not in years_by_value:
                    years_by_value[year_value] = []
                if sec_year not in years_by_value[year_value]:
                    years_by_value[year_value].append(sec_year)
            
            except Exception as e:
                logger.warning(f"Could not parse SEC date {date_str}: {e}")
    
    def _collect_market_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """Collect temporal entities from Market data"""
        query = f"""
        SELECT DISTINCT ?observedAt WHERE {{
            ?obs a <{MARKET.PriceObservation}> ;
                 <{MARKET.observedAt}> ?observedAt .
        }}
        """
        
        for row in self.graph.query(query):
            observed_at = str(row.observedAt)
            try:
                from dateutil import parser
                dt = parser.parse(observed_at)
                
                month_name = dt.strftime('%B')
                year_value = str(dt.year)
                
                # Create synthetic Market temporal URIs
                market_month = URIRef(f"https://financial-data.org/temporal/{month_name}")
                market_year = URIRef(f"https://financial-data.org/temporal/{year_value}")
                
                if month_name not in months_by_name:
                    months_by_name[month_name] = []
                if market_month not in months_by_name[month_name]:
                    months_by_name[month_name].append(market_month)
                
                if year_value not in years_by_value:
                    years_by_value[year_value] = []
                if market_year not in years_by_value[year_value]:
                    years_by_value[year_value].append(market_year)
            
            except Exception as e:
                logger.warning(f"Could not parse Market date {observed_at}: {e}")
    
    def _collect_noaa_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """Collect temporal entities from NOAA data"""
        query = f"""
        SELECT DISTINCT ?sentTime WHERE {{
            ?alert a <{CAP.Alert}> ;
                   <{CAP.hasSentTime}> ?sentTime .
        }}
        """
        
        for row in self.graph.query(query):
            sent_time = str(row.sentTime)
            try:
                from dateutil import parser
                dt = parser.parse(sent_time)
                
                month_name = dt.strftime('%B')
                year_value = str(dt.year)
                
                # Create synthetic NOAA temporal URIs
                noaa_month = URIRef(f"https://www.noaa.gov/temporal/{month_name}")
                noaa_year = URIRef(f"https://www.noaa.gov/temporal/{year_value}")
                
                if month_name not in months_by_name:
                    months_by_name[month_name] = []
                if noaa_month not in months_by_name[month_name]:
                    months_by_name[month_name].append(noaa_month)
                
                if year_value not in years_by_value:
                    years_by_value[year_value] = []
                if noaa_year not in years_by_value[year_value]:
                    years_by_value[year_value].append(noaa_year)
            
            except Exception as e:
                logger.warning(f"Could not parse NOAA date {sent_time}: {e}")
    
    def link_by_sector(self):
        """
        Link entities across sources that belong to same economic sector
        
        Uses BLS_SECTOR_PATTERNS extended with SEC, Market, and NOAA keywords
        
        Example:
            cpi:Energy_Entity + market:XOM_Ticker + sec:EnergyCompanyFiling
            → all linked to unified:EnergySector
        """
        logger.info("  Applying sector patterns across sources...")
        
        for sector_name, pattern in self.sector_patterns.items():
            sector_uri = pattern['sector_uri']
            relationship = pattern['relationship']
            
            # Add sector entity if not exists
            if not list(self.graph.triples((sector_uri, RDF.type, BLS_ENRICHMENT.EconomicSector))):
                self.graph.add((sector_uri, RDF.type, BLS_ENRICHMENT.EconomicSector))
                self.graph.add((sector_uri, RDFS.label, Literal(sector_name.replace('_', ' ').title())))
            
            # Link BLS entities (already done by intra-source, but we add cross-source relationship)
            if 'bls' in self.available_sources:
                self._link_bls_to_sector(sector_uri, relationship, pattern['keywords'])
            
            # Link SEC entities
            if 'sec' in self.available_sources:
                self._link_sec_to_sector(sector_uri, relationship, sector_name)
            
            # Link Market entities
            if 'market' in self.available_sources:
                self._link_market_to_sector(sector_uri, relationship, sector_name)
            
            # Link NOAA entities
            if 'noaa' in self.available_sources:
                self._link_noaa_to_sector(sector_uri, relationship, sector_name)
        
        logger.info(f"  Created {self.stats['sector_links']} cross-source sector links")
    
    def _link_bls_to_sector(self, sector_uri: URIRef, relationship: URIRef, keywords: Dict):
        """Link BLS entities to sector (cross-source relationship)"""
        # BLS entities already linked to sector by intra-source enrichment
        # Here we just add the cross-source correlation relationship if not exists
        
        for dataset_name, dataset_keywords in keywords.items():
            if dataset_name not in ['cpi', 'ppi', 'eci', 'jolts', 'empsit', 'ximpim', 'laus', 'metro', 'realer']:
                continue
            
            namespace_map = {
                'cpi': CPI, 'ppi': PPI, 'eci': ECI, 'jolts': JOLTS, 'empsit': EMPSIT,
                'ximpim': XIMPIM, 'laus': LAUS, 'metro': METRO, 'realer': REALER
            }
            
            namespace = namespace_map.get(dataset_name)
            if not namespace:
                continue
            
            # Find entities already linked to this sector
            query = f"""
            SELECT DISTINCT ?entity WHERE {{
                ?entity <{BLS_ENRICHMENT.belongsToSector}> <{sector_uri}> .
                FILTER(STRSTARTS(STR(?entity), "{namespace}"))
            }}
            """
            
            for row in self.graph.query(query):
                # Add cross-source correlation if not exists
                if not list(self.graph.triples((row.entity, relationship, sector_uri))):
                    self.graph.add((row.entity, relationship, sector_uri))
                    self.stats['sector_links'] += 1
    
    def _link_sec_to_sector(self, sector_uri: URIRef, relationship: URIRef, sector_name: str):
        """Link SEC entities to sector based on industry keywords"""
        # Map sector names to SEC industry keywords
        sec_sector_keywords = {
            'energy_sector': ['Energy', 'Oil', 'Gas', 'Mining', 'Petroleum'],
            'technology_sector': ['Technology', 'Software', 'Internet', 'Computer'],
            'healthcare_sector': ['Healthcare', 'Pharmaceutical', 'Biotech', 'Medical'],
            'financial_sector': ['Financial', 'Bank', 'Insurance', 'Investment'],
            'manufacturing_sector': ['Manufacturing', 'Industrial'],
            'food_sector': ['Food', 'Beverage', 'Restaurant'],
        }
        
        keywords = sec_sector_keywords.get(sector_name, [])
        if not keywords:
            return
        
        # Find SEC filings with industry keywords in issuer name or description
        for keyword in keywords:
            query = f"""
            SELECT DISTINCT ?filing WHERE {{
                ?filing a filings:SECFiling .
                ?filing filings:hasIssuer ?issuer .
                ?issuer rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{keyword}")))
            }}
            """
            
            for row in self.graph.query(query):
                self.graph.add((row.filing, BLS_ENRICHMENT.belongsToSector, sector_uri))
                self.graph.add((row.filing, relationship, sector_uri))
                self.stats['sector_links'] += 2
    
    def _link_market_to_sector(self, sector_uri: URIRef, relationship: URIRef, sector_name: str):
        """Link Market entities to sector based on ticker symbols"""
        # Map sector names to ticker symbols
        market_sector_tickers = {
            'energy_sector': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
            'technology_sector': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN'],
            'healthcare_sector': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK'],
            'financial_sector': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW'],
            'transportation_sector': ['UPS', 'FDX', 'DAL', 'UAL', 'LUV', 'AAL'],
            'food_sector': ['WMT', 'KO', 'PEP', 'MCD', 'SBUX', 'KHC', 'GIS'],
        }
        
        tickers = market_sector_tickers.get(sector_name, [])
        if not tickers:
            return
        
        for ticker in tickers:
            query = f"""
            SELECT DISTINCT ?ticker WHERE {{
                ?ticker a <{MARKET.StockTicker}> ;
                        <{MARKET.symbol}> "{ticker}" .
            }}
            """
            
            for row in self.graph.query(query):
                self.graph.add((row.ticker, BLS_ENRICHMENT.belongsToSector, sector_uri))
                self.graph.add((row.ticker, relationship, sector_uri))
                self.stats['sector_links'] += 2
    
    def _link_noaa_to_sector(self, sector_uri: URIRef, relationship: URIRef, sector_name: str):
        """Link NOAA weather events to affected sectors"""
        # Weather events can affect certain sectors
        noaa_sector_impacts = {
            'energy_sector': ['Hurricane', 'Winter Storm', 'Extreme Cold'],
            'food_sector': ['Drought', 'Flood', 'Freeze', 'Excessive Heat'],
            'transportation_sector': ['Winter Storm', 'Hurricane', 'Flood', 'Ice Storm'],
        }
        
        event_types = noaa_sector_impacts.get(sector_name, [])
        if not event_types:
            return
        
        for event_type in event_types:
            query = f"""
            SELECT DISTINCT ?alert WHERE {{
                ?alert <{CAP.hasInfo}> ?info .
                ?info <{CAP.hasEvent}> ?event .
                FILTER(CONTAINS(STR(?event), "{event_type}"))
            }}
            """
            
            for row in self.graph.query(query):
                self.graph.add((row.alert, BLS_ENRICHMENT.affectsSector, sector_uri))
                self.graph.add((row.alert, relationship, sector_uri))
                self.stats['sector_links'] += 2
    
    def link_by_company(self):
        """
        Link entities referencing same company across sources
        
        Example:
            sec:AAPL_Filing + market:AAPL_Ticker → unified:Company_AAPL
        """
        if 'sec' not in self.available_sources or 'market' not in self.available_sources:
            logger.info("  Skipping company linking (requires both SEC and Market data)")
            return
        
        logger.info("  Linking by company/ticker...")
        
        # Get all tickers from market data
        ticker_query = f"""
        SELECT DISTINCT ?ticker ?symbol WHERE {{
            ?ticker a <{MARKET.StockTicker}> ;
                    <{MARKET.symbol}> ?symbol .
        }}
        """
        
        for row in self.graph.query(ticker_query):
            symbol = str(row.symbol)
            
            # Create unified company entity
            unified_company = UNIFIED[f"Company_{symbol}"]
            
            if not list(self.graph.triples((unified_company, RDF.type, BLS_ENRICHMENT.UnifiedCompany))):
                self.graph.add((unified_company, RDF.type, BLS_ENRICHMENT.UnifiedCompany))
                self.graph.add((unified_company, BLS_ENRICHMENT.ticker, Literal(symbol)))
            
            # Link market ticker
            if not list(self.graph.triples((row.ticker, BLS_ENRICHMENT.refersToCompany, unified_company))):
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
                if not list(self.graph.triples((sec_row.filing, BLS_ENRICHMENT.refersToCompany, unified_company))):
                    self.graph.add((sec_row.filing, BLS_ENRICHMENT.refersToCompany, unified_company))
                    self.stats['company_links'] += 1
        
        logger.info(f"  Created {self.stats['company_links']} company-based links")
    
    def link_by_geography(self):
        """
        Link entities by geographic region
        
        Example:
            laus:California + noaa:CaliforniaAlert → unified:CaliforniaRegion
        """
        logger.info("  Linking by geographic region...")
        
        # Define unified regions (US states)
        us_states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
            'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
            'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
            'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
            'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
            'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
            'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
            'West Virginia', 'Wisconsin', 'Wyoming'
        ]
        
        for state in us_states:
            unified_region = UNIFIED[f"{state.replace(' ', '')}Region"]
            
            if not list(self.graph.triples((unified_region, RDF.type, BLS_ENRICHMENT.GeographicRegion))):
                self.graph.add((unified_region, RDF.type, BLS_ENRICHMENT.GeographicRegion))
                self.graph.add((unified_region, RDFS.label, Literal(state)))
            
            # Link LAUS state data
            if 'bls' in self.available_sources:
                laus_query = f"""
                SELECT DISTINCT ?entity WHERE {{
                    ?entity a <{LAUS.LaborForceData}> .
                    ?entity <{LAUS.hasState}> ?state .
                    ?state rdfs:label ?label .
                    FILTER(CONTAINS(STR(?label), "{state}"))
                }}
                """
                
                for row in self.graph.query(laus_query):
                    if not list(self.graph.triples((row.entity, BLS_ENRICHMENT.hasRegion, unified_region))):
                        self.graph.add((row.entity, BLS_ENRICHMENT.hasRegion, unified_region))
                        self.stats['geographic_links'] += 1
            
            # Link NOAA alerts for this state
            if 'noaa' in self.available_sources:
                noaa_query = f"""
                SELECT DISTINCT ?alert WHERE {{
                    ?alert <{CAP.hasInfo}> ?info .
                    ?info <{CAP.hasArea}> ?area .
                    ?area <{CAP.hasAreaDescription}> ?areaDesc .
                    FILTER(CONTAINS(STR(?areaDesc), "{state}"))
                }}
                """
                
                for row in self.graph.query(noaa_query):
                    if not list(self.graph.triples((row.alert, BLS_ENRICHMENT.affectsRegion, unified_region))):
                        self.graph.add((row.alert, BLS_ENRICHMENT.affectsRegion, unified_region))
                        self.stats['geographic_links'] += 1
        
        logger.info(f"  Created {self.stats['geographic_links']} geographic links")
    
    def create_causal_links(self):
        """
        Create potential causal relationships across sources
        
        Extends KNOWN_CORRELATIONS to cross-source relationships:
        - BLS → Market (economic indicators affect stock prices)
        - BLS → SEC (economic conditions affect filings/enforcement)
        - NOAA → Market (weather affects commodity prices)
        - SEC → Market (filings affect stock prices)
        """
        logger.info("  Creating causal relationships...")
        
        # BLS → Market causal links
        if 'bls' in self.available_sources and 'market' in self.available_sources:
            self._link_bls_to_market()
        
        # NOAA → Market causal links
        if 'noaa' in self.available_sources and 'market' in self.available_sources:
            self._link_noaa_to_market()
        
        # SEC → Market causal links
        if 'sec' in self.available_sources and 'market' in self.available_sources:
            self._link_sec_to_market()
        
        logger.info(f"  Created {self.stats['causal_links']} causal links")
    
    def _link_bls_to_market(self):
        """Link BLS economic indicators to market entities"""
        # CPI Energy → Energy stocks
        cpi_energy_query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity <{BLS_ENRICHMENT.belongsToSector}> <{BLS_ENRICHMENT.EnergySector}> .
            FILTER(STRSTARTS(STR(?entity), "{CPI}"))
        }}
        """
        
        market_energy_query = f"""
        SELECT DISTINCT ?ticker WHERE {{
            ?ticker <{BLS_ENRICHMENT.belongsToSector}> <{BLS_ENRICHMENT.EnergySector}> .
            FILTER(STRSTARTS(STR(?ticker), "{MARKET}"))
        }}
        """
        
        cpi_entities = list(self.graph.query(cpi_energy_query))
        market_entities = list(self.graph.query(market_energy_query))
        
        for cpi_row in cpi_entities:
            for market_row in market_entities:
                self.graph.add((
                    cpi_row.entity,
                    BLS_ENRICHMENT.leadsTo,
                    market_row.ticker
                ))
                self.stats['causal_links'] += 1
    
    def _link_noaa_to_market(self):
        """Link weather events to affected market sectors"""
        # Hurricane → Energy stocks
        hurricane_query = f"""
        SELECT DISTINCT ?alert WHERE {{
            ?alert <{CAP.hasInfo}> ?info .
            ?info <{CAP.hasEvent}> ?event .
            FILTER(CONTAINS(STR(?event), "Hurricane"))
        }}
        """
        
        energy_stocks_query = f"""
        SELECT DISTINCT ?ticker WHERE {{
            ?ticker <{BLS_ENRICHMENT.belongsToSector}> <{BLS_ENRICHMENT.EnergySector}> .
            FILTER(STRSTARTS(STR(?ticker), "{MARKET}"))
        }}
        """
        
        alerts = list(self.graph.query(hurricane_query))
        stocks = list(self.graph.query(energy_stocks_query))
        
        for alert_row in alerts:
            for stock_row in stocks:
                self.graph.add((
                    alert_row.alert,
                    BLS_ENRICHMENT.impacts,
                    stock_row.ticker
                ))
                self.stats['causal_links'] += 1
    
    def _link_sec_to_market(self):
        """Link SEC filings to stock price movements"""
        # Form 10-K filings → Stock prices
        filing_query = """
        SELECT DISTINCT ?filing ?ticker WHERE {
            ?filing a filings:Form10K ;
                    filings:hasIssuerTicker ?tickerSymbol .
            ?ticker a market:StockTicker ;
                    market:symbol ?tickerSymbol .
        }
        """
        
        for row in self.graph.query(filing_query):
            self.graph.add((
                row.filing,
                BLS_ENRICHMENT.affects,
                row.ticker
            ))
            self.stats['causal_links'] += 1
    
    def align_measurement_types(self):
        """
        Align similar measurement types across sources
        
        Examples:
        - Price indices (CPI, PPI) ↔ Stock prices (Market)
        - Rate measurements (JOLTS, LAUS) ↔ Unemployment indicators
        - Change measurements (CPI %, PPI %) ↔ Price changes
        """
        logger.info("  Aligning measurement types...")
        
        # Price indices
        if 'bls' in self.available_sources:
            # Mark CPI indices as price indices
            cpi_index_query = f"""
            SELECT DISTINCT ?index WHERE {{
                ?index a <{CPI.Index}> .
            }}
            """
            
            for row in self.graph.query(cpi_index_query):
                if not list(self.graph.triples((row.index, RDF.type, BLS_ENRICHMENT.PriceIndex))):
                    self.graph.add((row.index, RDF.type, BLS_ENRICHMENT.PriceIndex))
                    self.stats['measurement_links'] += 1
            
            # Mark PPI indices as price indices
            ppi_index_query = f"""
            SELECT DISTINCT ?index WHERE {{
                ?index a <{PPI.IndexValue}> .
            }}
            """
            
            for row in self.graph.query(ppi_index_query):
                if not list(self.graph.triples((row.index, RDF.type, BLS_ENRICHMENT.PriceIndex))):
                    self.graph.add((row.index, RDF.type, BLS_ENRICHMENT.PriceIndex))
                    self.stats['measurement_links'] += 1
        
        # Rate measurements
        if 'bls' in self.available_sources:
            # JOLTS rates
            jolts_rate_query = f"""
            SELECT DISTINCT ?rate WHERE {{
                {{ ?rate a <{JOLTS.JobOpeningsRate}> }} UNION
                {{ ?rate a <{JOLTS.HiresRate}> }} UNION
                {{ ?rate a <{JOLTS.QuitsRate}> }}
            }}
            """
            
            for row in self.graph.query(jolts_rate_query):
                if not list(self.graph.triples((row.rate, RDF.type, BLS_ENRICHMENT.RateMeasurement))):
                    self.graph.add((row.rate, RDF.type, BLS_ENRICHMENT.RateMeasurement))
                    self.stats['measurement_links'] += 1
            
            # LAUS unemployment rates
            laus_rate_query = f"""
            SELECT DISTINCT ?rate WHERE {{
                ?rate a <{LAUS.UnemploymentRate}> .
            }}
            """
            
            for row in self.graph.query(laus_rate_query):
                if not list(self.graph.triples((row.rate, RDF.type, BLS_ENRICHMENT.RateMeasurement))):
                    self.graph.add((row.rate, RDF.type, BLS_ENRICHMENT.RateMeasurement))
                    self.stats['measurement_links'] += 1
        
        logger.info(f"  Created {self.stats['measurement_links']} measurement type links")


def enrich_cross_source(graph: Graph) -> Dict[str, int]:
    """
    Main entry point for cross-source enrichment
    """
    linker = CrossSourceLinker(graph)
    return linker.enrich()