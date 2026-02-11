"""
Temporal Entity Unifier

Unifies temporal entities (months, years, dates) across all data sources.
Creates unified temporal entities and links source-specific temporal entities
to them using owl:sameAs.

This module is used by:
- intra_source_linker.py (BLS temporal unification)
- cross_source_linker.py (cross-source temporal alignment)

Example:
    from glue_jobs.enrichment.temporal_unifier import TemporalUnifier

    unifier = TemporalUnifier(graph)
    stats = unifier.unify_all_sources()

    # Result:
    # cpi:November + ppi:November + sec:November + market:November
    # → unified:November2024
"""
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
from glue_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, UNIFIED,
    CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER,
    SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP,
    MARKET, CAP,
    get_month_name, get_year_value
)
from typing import Dict, Set, List, Optional
from dateutil import parser as date_parser
import logging

logger = logging.getLogger(__name__)


class TemporalUnifier:
    """
    Unifies temporal entities across all data sources

    Strategies:
    1. Collect temporal entities from all sources
    2. Group by normalized month/year values
    3. Create unified temporal entities
    4. Link source-specific entities with owl:sameAs

    Supports:
    - BLS datasets (CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER)
    - SEC data (filings, proceedings, litigation, suspensions)
    - Market data (price observations, option expirations)
    - NOAA data (weather alerts)
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.stats = {
            'months_unified': 0,
            'years_unified': 0,
            'temporal_links': 0,
            'sources_processed': []
        }

        # Month name normalization
        self.month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        # Detect available sources
        self.available_sources = self._detect_sources()
        logger.info(f"Detected sources for temporal unification: {', '.join(self.available_sources)}")

    def _detect_sources(self) -> Set[str]:
        """Detect which data sources are present in the graph"""
        sources = set()

        # Check for BLS data
        bls_namespaces = [CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER]
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
        sec_query = """
        ASK {
            { ?s a filings:Form3 } UNION
            { ?s a filings:Form4 } UNION
            { ?s a sec:AdministrativeProceeding } UNION
            { ?s a seclit:LitigationRelease } UNION
            { ?s a secsusp:TradingSuspension }
        }
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

    def unify_all_sources(self) -> Dict[str, int]:
        """
        Main entry point: Unify temporal entities across all sources

        Returns:
            Dictionary with unification statistics
        """
        logger.info("Starting temporal unification across all sources...")

        # Collect temporal entities from all sources
        months_by_name = {}
        years_by_value = {}

        if 'bls' in self.available_sources:
            self._collect_bls_temporal_entities(months_by_name, years_by_value)
            self.stats['sources_processed'].append('bls')

        if 'sec' in self.available_sources:
            self._collect_sec_temporal_entities(months_by_name, years_by_value)
            self.stats['sources_processed'].append('sec')

        if 'market' in self.available_sources:
            self._collect_market_temporal_entities(months_by_name, years_by_value)
            self.stats['sources_processed'].append('market')

        if 'noaa' in self.available_sources:
            self._collect_noaa_temporal_entities(months_by_name, years_by_value)
            self.stats['sources_processed'].append('noaa')

        # Create unified temporal entities
        self._create_unified_months(months_by_name)
        self._create_unified_years(years_by_value)

        logger.info(f"Temporal unification complete:")
        logger.info(f"  - Unified {self.stats['months_unified']} months")
        logger.info(f"  - Unified {self.stats['years_unified']} years")
        logger.info(f"  - Created {self.stats['temporal_links']} temporal links")

        return self.stats

    def _collect_bls_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """
        Collect temporal entities from BLS datasets

        BLS datasets use explicit Month and Year entities:
        - cpi:November, cpi:2024
        - ppi:November, ppi:2024
        - etc.
        """
        logger.info("  Collecting BLS temporal entities...")

        bls_namespaces = {
            'cpi': CPI,
            'ppi': PPI,
            'eci': ECI,
            'jolts': JOLTS,
            'empsit': EMPSIT,
            'ximpim': XIMPIM,
            'laus': LAUS,
            'metro': METRO,
            'realer': REALER
        }

        for dataset_name, namespace in bls_namespaces.items():
            # Check if this dataset exists
            check_query = f"""
            ASK {{
                ?s ?p ?o .
                FILTER(STRSTARTS(STR(?s), "{namespace}"))
            }}
            """
            if not self.graph.query(check_query).askAnswer:
                continue

            # Collect months
            month_query = f"""
            SELECT DISTINCT ?month WHERE {{
                ?s ?p ?month .
                FILTER(STRSTARTS(STR(?month), "{namespace}"))
                FILTER(REGEX(STR(?month), "(January|February|March|April|May|June|July|August|September|October|November|December)$"))
            }}
            """

            for row in self.graph.query(month_query):
                month_name = get_month_name(row.month)
                if month_name not in months_by_name:
                    months_by_name[month_name] = []
                if row.month not in months_by_name[month_name]:
                    months_by_name[month_name].append(row.month)

            # Collect years
            year_query = f"""
            SELECT DISTINCT ?year WHERE {{
                ?s ?p ?year .
                FILTER(STRSTARTS(STR(?year), "{namespace}"))
                FILTER(REGEX(STR(?year), "[0-9]{{4}}$"))
            }}
            """

            for row in self.graph.query(year_query):
                year_value = get_year_value(row.year)
                if year_value not in years_by_value:
                    years_by_value[year_value] = []
                if row.year not in years_by_value[year_value]:
                    years_by_value[year_value].append(row.year)

        logger.info(
            f"    Collected {len(months_by_name)} unique months and {len(years_by_value)} unique years from BLS")

    def _collect_sec_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """
        Collect temporal entities from SEC data

        SEC data uses date literals, not explicit temporal entities.
        We extract month/year from dates and create synthetic temporal URIs.
        """
        logger.info("  Collecting SEC temporal entities...")

        # Query for all dates in SEC data
        date_query = """
        SELECT DISTINCT ?date WHERE {
            { ?filing filings:hasPeriodOfReport ?date } UNION
            { ?filing filings:hasReportDate ?date } UNION
            { ?proceeding sec:initiationDate ?date } UNION
            { ?litigation seclit:filingDate ?date } UNION
            { ?suspension secsusp:startDate ?date }
        }
        """

        dates_processed = 0
        for row in self.graph.query(date_query):
            date_str = str(row.date)
            try:
                dt = date_parser.parse(date_str)

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

                dates_processed += 1

            except Exception as e:
                logger.warning(f"Could not parse SEC date {date_str}: {e}")

        logger.info(f"    Processed {dates_processed} SEC dates")

    def _collect_market_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """
        Collect temporal entities from Market data

        Market data uses timestamp literals for price observations.
        We extract month/year and create synthetic temporal URIs.
        """
        logger.info("  Collecting Market temporal entities...")

        # Query for price observation timestamps
        timestamp_query = f"""
        SELECT DISTINCT ?observedAt WHERE {{
            ?obs a <{MARKET.PriceObservation}> ;
                 <{MARKET.observedAt}> ?observedAt .
        }}
        """

        timestamps_processed = 0
        for row in self.graph.query(timestamp_query):
            observed_at = str(row.observedAt)
            try:
                dt = date_parser.parse(observed_at)

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

                timestamps_processed += 1

            except Exception as e:
                logger.warning(f"Could not parse Market timestamp {observed_at}: {e}")

        # Query for option expiration dates
        expiration_query = f"""
        SELECT DISTINCT ?expirationDate WHERE {{
            ?contract a <{MARKET.OptionContract}> ;
                      <{MARKET.expirationDate}> ?expirationDate .
        }}
        """

        for row in self.graph.query(expiration_query):
            expiration_date = str(row.expirationDate)
            try:
                dt = date_parser.parse(expiration_date)

                month_name = dt.strftime('%B')
                year_value = str(dt.year)

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

                timestamps_processed += 1

            except Exception as e:
                logger.warning(f"Could not parse option expiration date {expiration_date}: {e}")

        logger.info(f"    Processed {timestamps_processed} Market timestamps")

    def _collect_noaa_temporal_entities(self, months_by_name: Dict, years_by_value: Dict):
        """
        Collect temporal entities from NOAA weather alerts

        NOAA alerts use CAP standard with timestamp literals.
        We extract month/year and create synthetic temporal URIs.
        """
        logger.info("  Collecting NOAA temporal entities...")

        # Query for alert sent times
        sent_time_query = f"""
        SELECT DISTINCT ?sentTime WHERE {{
            ?alert a <{CAP.Alert}> ;
                   <{CAP.hasSentTime}> ?sentTime .
        }}
        """

        timestamps_processed = 0
        for row in self.graph.query(sent_time_query):
            sent_time = str(row.sentTime)
            try:
                dt = date_parser.parse(sent_time)

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

                timestamps_processed += 1

            except Exception as e:
                logger.warning(f"Could not parse NOAA timestamp {sent_time}: {e}")

        logger.info(f"    Processed {timestamps_processed} NOAA timestamps")

    def _create_unified_months(self, months_by_name: Dict):
        """
        Create unified month entities and link source-specific months

        Example:
            unified:November owl:sameAs cpi:November, ppi:November, sec:November, ...
        """
        logger.info("  Creating unified month entities...")

        for month_name, month_uris in months_by_name.items():
            # Only unify if multiple sources reference this month
            if len(month_uris) < 1:
                continue

            unified_month = UNIFIED[month_name]

            # Add unified month entity if not exists
            if not list(self.graph.triples((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))):
                self.graph.add((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))
                self.graph.add((unified_month, RDFS.label, Literal(month_name)))
                self.stats['months_unified'] += 1

            # Link all source-specific month entities with owl:sameAs
            for month_uri in month_uris:
                if not list(self.graph.triples((unified_month, OWL.sameAs, month_uri))):
                    self.graph.add((unified_month, OWL.sameAs, month_uri))
                    self.stats['temporal_links'] += 1

        logger.info(f"    Created {self.stats['months_unified']} unified months")

    def _create_unified_years(self, years_by_value: Dict):
        """
        Create unified year entities and link source-specific years

        Example:
            unified:Year2024 owl:sameAs cpi:2024, ppi:2024, sec:2024, ...
        """
        logger.info("  Creating unified year entities...")

        for year_value, year_uris in years_by_value.items():
            # Only unify if multiple sources reference this year
            if len(year_uris) < 1:
                continue

            unified_year = UNIFIED[f"Year{year_value}"]

            # Add unified year entity if not exists
            if not list(self.graph.triples((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))):
                self.graph.add((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))
                self.graph.add((unified_year, RDFS.label, Literal(year_value)))
                self.stats['years_unified'] += 1

            # Link all source-specific year entities with owl:sameAs
            for year_uri in year_uris:
                if not list(self.graph.triples((unified_year, OWL.sameAs, year_uri))):
                    self.graph.add((unified_year, OWL.sameAs, year_uri))
                    self.stats['temporal_links'] += 1

        logger.info(f"    Created {self.stats['years_unified']} unified years")

    def get_unified_month(self, month_name: str) -> Optional[URIRef]:
        """
        Get unified month URI for a given month name

        Args:
            month_name: Month name (e.g., "November")

        Returns:
            Unified month URI or None if not found
        """
        if month_name not in self.month_names:
            return None

        unified_month = UNIFIED[month_name]

        # Check if it exists
        if list(self.graph.triples((unified_month, RDF.type, BLS_ENRICHMENT.UnifiedMonth))):
            return unified_month

        return None

    def get_unified_year(self, year_value: str) -> Optional[URIRef]:
        """
        Get unified year URI for a given year value

        Args:
            year_value: Year value (e.g., "2024")

        Returns:
            Unified year URI or None if not found
        """
        unified_year = UNIFIED[f"Year{year_value}"]

        # Check if it exists
        if list(self.graph.triples((unified_year, RDF.type, BLS_ENRICHMENT.UnifiedYear))):
            return unified_year

        return None

    def get_source_months_for_unified(self, unified_month: URIRef) -> List[URIRef]:
        """
        Get all source-specific month URIs linked to a unified month

        Args:
            unified_month: Unified month URI

        Returns:
            List of source-specific month URIs
        """
        query = f"""
        SELECT ?sourceMonth WHERE {{
            <{unified_month}> owl:sameAs ?sourceMonth .
        }}
        """

        return [row.sourceMonth for row in self.graph.query(query)]

    def get_source_years_for_unified(self, unified_year: URIRef) -> List[URIRef]:
        """
        Get all source-specific year URIs linked to a unified year

        Args:
            unified_year: Unified year URI

        Returns:
            List of source-specific year URIs
        """
        query = f"""
        SELECT ?sourceYear WHERE {{
            <{unified_year}> owl:sameAs ?sourceYear .
        }}
        """

        return [row.sourceYear for row in self.graph.query(query)]


def unify_temporal_entities(graph: Graph) -> Dict[str, int]:
    """
    Convenience function for temporal unification

    Args:
        graph: RDFLib graph to enrich

    Returns:
        Dictionary with unification statistics

    Example:
        from glue_jobs.enrichment.temporal_unifier import unify_temporal_entities

        stats = unify_temporal_entities(graph)
    """
    unifier = TemporalUnifier(graph)
    return unifier.unify_all_sources()