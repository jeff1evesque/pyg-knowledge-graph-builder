""""
SEC Intra-Source Enrichment Orchestrator
Coordinates enrichment across all SEC datasets (filings, administrative proceedings, litigation, trading suspensions)
"""
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, OWL
from glue_jobs.utils.rdf_utils import (
    SEC_ENRICHMENT, SEC_ADMIN, SEC_LIT, SEC_SUSP, SEC_FILINGS, UNIFIED
)
from glue_jobs.enrichment.intra_source.base import IntraSourceEnricher
from glue_jobs.enrichment.intra_source.sec.patterns import (
    SEC_SECTOR_PATTERNS, SEC_VIOLATION_PATTERNS, SEC_COMPANY_STATUS_PATTERNS
)
from glue_jobs.enrichment.intra_source.sec.correlations import KNOWN_CORRELATIONS
from typing import Dict, Set
import logging

logger = logging.getLogger(__name__)


def normalize_keyword_for_uri_matching(keyword: str) -> str:
    """
    Normalize a keyword for matching against SEC URIs

    SEC URIs may use various formats:
    - Spaces may become underscores or be removed
    - Special characters may be removed
    - CamelCase may be used

    Examples:
        "Investment Adviser" → "InvestmentAdviser" or "Investment_Adviser"
        "Section 10(b)" → "Section10b"
        "Rule 10b-5" → "Rule10b5"
    """
    # Try multiple normalization strategies
    normalized = keyword.replace(' ', '')  # Remove spaces for CamelCase
    normalized = normalized.replace('(', '').replace(')', '')  # Remove parentheses
    normalized = normalized.replace('-', '')  # Remove hyphens
    return normalized


class SECIntraSourceLinker(IntraSourceEnricher):
    """
    Orchestrates SEC intra-source enrichment across all datasets

    Handles:
    - Filings (Forms 3, 4, 5, 10-K, 10-Q, 8-K, etc.)
    - Administrative Proceedings
    - Litigation Releases
    - Trading Suspensions
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.available_datasets = self.detect_datasets()
        logger.info(f"Detected SEC datasets: {', '.join(self.available_datasets)}")

    def detect_datasets(self) -> Set[str]:
        """Detect which SEC datasets are present in the graph"""
        datasets = set()

        # Check for filings
        filings_query = """
        ASK {
            { ?s a filings:Form3 } UNION
            { ?s a filings:Form4 } UNION
            { ?s a filings:Form5 } UNION
            { ?s a filings:Form10K } UNION
            { ?s a filings:Form10Q } UNION
            { ?s a filings:Form8K } UNION
            { ?s a filings:SECFiling }
        }
        """
        if self.graph.query(filings_query).askAnswer:
            datasets.add('filings')

        # Check for administrative proceedings
        admin_query = """
        ASK {
            { ?s a sec:AdministrativeProceeding } UNION
            { ?s a sec:Rule102eProceeding } UNION
            { ?s a sec:CeaseAndDesistProceeding } UNION
            { ?s a sec:Section12jProceeding }
        }
        """
        if self.graph.query(admin_query).askAnswer:
            datasets.add('administrative_proceedings')

        # Check for litigation
        lit_query = """
        ASK {
            { ?s a seclit:LitigationRelease } UNION
            { ?s a seclit:CivilEnforcementAction }
        }
        """
        if self.graph.query(lit_query).askAnswer:
            datasets.add('litigation')

        # Check for trading suspensions
        susp_query = """
        ASK {
            { ?s a secsusp:TradingSuspension } UNION
            { ?s a secsusp:Section12kSuspension } UNION
            { ?s a secsusp:Section12jSuspension }
        }
        """
        if self.graph.query(susp_query).askAnswer:
            datasets.add('trading_suspensions')

        return datasets

    def enrich(self) -> Dict[str, int]:
        """Run all SEC intra-source enrichment steps"""
        if not self.available_datasets:
            logger.info("No SEC data detected, skipping enrichment")
            return {'total_triples_added': 0}

        initial_count = len(self.graph)

        logger.info("=" * 60)
        logger.info("Starting SEC Intra-Source Enrichment")
        logger.info("=" * 60)

        # Step 1: Unify company entities
        logger.info("\n[Step 1/6] Unifying company entities...")
        self.unify_company_entities()

        # Step 2: Unify person entities
        logger.info("\n[Step 2/6] Unifying person entities...")
        self.unify_person_entities()

        # Step 3: Link temporal sequences
        logger.info("\n[Step 3/6] Linking temporal sequences...")
        self.link_temporal_sequences()

        # Step 4: Apply sector patterns
        logger.info("\n[Step 4/6] Applying sector patterns...")
        self.apply_sector_patterns()

        # Step 5: Apply violation patterns
        logger.info("\n[Step 5/6] Applying violation patterns...")
        self.apply_violation_patterns()

        # Step 6: Apply correlations
        logger.info("\n[Step 6/6] Applying correlations...")
        self.apply_known_correlations()

        final_count = len(self.graph)
        enrichment_count = final_count - initial_count

        logger.info("\n" + "=" * 60)
        logger.info("SEC Intra-Source Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total triples added: {enrichment_count}")
        logger.info(f"  - Company unification: {self.stats.get('company_unified', 0)}")
        logger.info(f"  - Person unification: {self.stats.get('person_unified', 0)}")
        logger.info(f"  - Temporal sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Violation links: {self.stats.get('violation_links', 0)}")
        logger.info(f"  - Correlation links: {self.stats['correlation_links']}")
        logger.info("=" * 60)

        return {
            'total_triples_added': enrichment_count,
            'available_datasets': list(self.available_datasets),
            **self.stats
        }

    def unify_company_entities(self):
        """
        Unify company entities across SEC datasets using CIK numbers

        Creates unified Company entities and links dataset-specific
        company entities to them using owl:sameAs

        Example:
            filings:Issuer_0001234567 → unified:Company_0001234567
            sec:CorporateRespondent_0001234567 → unified:Company_0001234567
            secsusp:Company_0001234567 → unified:Company_0001234567
        """
        if 'company_unified' not in self.stats:
            self.stats['company_unified'] = 0

        # Collect all companies by CIK
        companies_by_cik = {}

        # Query for companies with CIK numbers across all datasets
        cik_query = """
        SELECT DISTINCT ?company ?cik WHERE {
            ?company ?hasCikProp ?cik .
            FILTER(
                ?hasCikProp = filings:hasIssuerCik ||
                ?hasCikProp = sec:cikNumber ||
                ?hasCikProp = secsusp:cikNumber ||
                ?hasCikProp = seclit:cikNumber
            )
        }
        """

        for row in self.graph.query(cik_query):
            cik = str(row.cik)
            if cik not in companies_by_cik:
                companies_by_cik[cik] = []
            companies_by_cik[cik].append(row.company)

        # Create unified company entities
        for cik, company_uris in companies_by_cik.items():
            if len(company_uris) > 1:  # Only unify if multiple references exist
                unified_company = UNIFIED[f"Company_{cik}"]
                self.graph.add((unified_company, RDF.type, SEC_ENRICHMENT.UnifiedCompany))
                self.graph.add((unified_company, SEC_ENRICHMENT.hasCik, Literal(cik)))

                for company_uri in company_uris:
                    self.graph.add((unified_company, OWL.sameAs, company_uri))
                    self.stats['company_unified'] += 1

        logger.info(f"  Unified {len(companies_by_cik)} companies across datasets")
        logger.info(f"  Created {self.stats['company_unified']} company unification links")

    def unify_person_entities(self):
        """
        Unify person entities across SEC datasets using CIK numbers

        Creates unified Person entities and links dataset-specific
        person entities to them using owl:sameAs

        Example:
            filings:ReportingOwner_0009876543 → unified:Person_0009876543
            sec:IndividualRespondent_0009876543 → unified:Person_0009876543
            seclit:IndividualDefendant_0009876543 → unified:Person_0009876543
        """
        if 'person_unified' not in self.stats:
            self.stats['person_unified'] = 0

        # Collect all persons by CIK
        persons_by_cik = {}

        # Query for persons with CIK numbers
        cik_query = """
        SELECT DISTINCT ?person ?cik WHERE {
            ?person ?hasCikProp ?cik .
            FILTER(
                ?hasCikProp = filings:hasReportingOwnerCik ||
                ?hasCikProp = sec:cikNumber
            )
            # Filter to ensure it's a person, not a company
            FILTER EXISTS {
                { ?person a filings:ReportingOwner } UNION
                { ?person a sec:IndividualRespondent } UNION
                { ?person a seclit:IndividualDefendant }
            }
        }
        """

        for row in self.graph.query(cik_query):
            cik = str(row.cik)
            if cik not in persons_by_cik:
                persons_by_cik[cik] = []
            persons_by_cik[cik].append(row.person)

        # Create unified person entities
        for cik, person_uris in persons_by_cik.items():
            if len(person_uris) > 1:  # Only unify if multiple references exist
                unified_person = UNIFIED[f"Person_{cik}"]
                self.graph.add((unified_person, RDF.type, SEC_ENRICHMENT.UnifiedPerson))
                self.graph.add((unified_person, SEC_ENRICHMENT.hasCik, Literal(cik)))

                for person_uri in person_uris:
                    self.graph.add((unified_person, OWL.sameAs, person_uri))
                    self.stats['person_unified'] += 1

        logger.info(f"  Unified {len(persons_by_cik)} persons across datasets")
        logger.info(f"  Created {self.stats['person_unified']} person unification links")

    def link_temporal_sequences(self):
        """
        Link temporal sequences for SEC filings and actions

        Links filings, proceedings, and actions in chronological order
        for the same company or person
        """
        # Link Form 3 → Form 4 → Form 5 sequences for same reporting owner
        self._link_ownership_filing_sequences()

        # Link periodic reporting sequences (10-Q → 10-K)
        self._link_periodic_reporting_sequences()

        # Link administrative proceedings chronologically
        self._link_administrative_proceeding_sequences()

        # Link litigation actions chronologically
        self._link_litigation_sequences()

        # Link trading suspensions chronologically
        self._link_trading_suspension_sequences()

    def _link_ownership_filing_sequences(self):
        """Link Form 3 → Form 4 → Form 5 sequences"""
        # Query for ownership filings by reporting owner and date
        query = """
        SELECT ?filing ?owner ?date ?formType WHERE {
            ?filing a ?type ;
                    filings:hasReportingOwner ?owner ;
                    filings:hasPeriodOfReport ?dateObj .
            ?dateObj :hasDateValue ?date .

            FILTER(
                ?type = filings:Form3 ||
                ?type = filings:Form4 ||
                ?type = filings:Form5
            )

            BIND(
                IF(?type = filings:Form3, "3",
                IF(?type = filings:Form4, "4",
                IF(?type = filings:Form5, "5", "")))
                AS ?formType
            )
        }
        ORDER BY ?owner ?date
        """

        results = list(self.graph.query(query))

        # Group by owner
        by_owner = {}
        for row in results:
            owner = str(row.owner)
            if owner not in by_owner:
                by_owner[owner] = []
            by_owner[owner].append({
                'filing': row.filing,
                'date': str(row.date),
                'form_type': str(row.formType)
            })

        # Link consecutive filings
        links_added = 0
        for owner, filings in by_owner.items():
            for i in range(len(filings) - 1):
                current = filings[i]['filing']
                next_filing = filings[i + 1]['filing']

                self.graph.add((
                    current,
                    SEC_ENRICHMENT.precedes,
                    next_filing
                ))
                links_added += 1
                self.stats['temporal_sequences'] += 1

        if links_added > 0:
            logger.info(f"  Ownership filings: {links_added} sequence links")

    def _link_periodic_reporting_sequences(self):
        """Link 10-Q → 10-K sequences"""
        # Similar implementation for periodic reports
        # Query for 10-Q and 10-K filings by company and date
        query = """
        SELECT ?filing ?company ?date ?formType WHERE {
            ?filing a ?type ;
                    filings:hasCompany ?company ;
                    filings:hasReportDate ?dateObj .
            ?dateObj :hasDateValue ?date .

            FILTER(
                ?type = filings:Form10Q ||
                ?type = filings:Form10K
            )

            BIND(
                IF(?type = filings:Form10Q, "10-Q",
                IF(?type = filings:Form10K, "10-K", ""))
                AS ?formType
            )
        }
        ORDER BY ?company ?date
        """

        results = list(self.graph.query(query))

        # Group by company
        by_company = {}
        for row in results:
            company = str(row.company)
            if company not in by_company:
                by_company[company] = []
            by_company[company].append({
                'filing': row.filing,
                'date': str(row.date),
                'form_type': str(row.formType)
            })

        # Link consecutive filings
        links_added = 0
        for company, filings in by_company.items():
            for i in range(len(filings) - 1):
                current = filings[i]['filing']
                next_filing = filings[i + 1]['filing']

                self.graph.add((
                    current,
                    SEC_ENRICHMENT.precedes,
                    next_filing
                ))
                links_added += 1
                self.stats['temporal_sequences'] += 1

        if links_added > 0:
            logger.info(f"  Periodic reports: {links_added} sequence links")

    def _link_administrative_proceeding_sequences(self):
        """Link administrative proceedings chronologically"""
        # Implementation for administrative proceedings
        pass

    def _link_litigation_sequences(self):
        """Link litigation actions chronologically"""
        # Implementation for litigation
        pass

    def _link_trading_suspension_sequences(self):
        """Link trading suspensions chronologically"""
        # Implementation for trading suspensions
        pass

    def apply_sector_patterns(self):
        """
        Apply sector-based linking patterns

        Links entities to industry sectors based on keyword matching
        """
        logger.info("  Applying sector patterns...")

        for sector_name, pattern in SEC_SECTOR_PATTERNS.items():
            sector_uri = pattern['sector_uri']
            relationship = pattern['relationship']

            # Process each dataset that has keywords for this sector
            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                if not keywords:
                    continue

                # Find entities matching keywords
                for keyword in keywords:
                    normalized_keyword = normalize_keyword_for_uri_matching(keyword)

                    # Search for entities containing the keyword
                    # This is dataset-specific and would need proper SPARQL queries
                    # For now, showing the pattern

                    self.stats['sector_links'] += 0  # Placeholder

        logger.info(f"  Added {self.stats['sector_links']} sector links")

    def apply_violation_patterns(self):
        """
        Apply violation type patterns

        Links entities to violation types based on keyword matching
        """
        if 'violation_links' not in self.stats:
            self.stats['violation_links'] = 0

        logger.info("  Applying violation patterns...")

        for violation_name, pattern in SEC_VIOLATION_PATTERNS.items():
            violation_uri = pattern['violation_uri']
            relationship = pattern['relationship']

            # Process each dataset
            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                if not keywords:
                    continue

                # Find entities matching keywords
                # Implementation would search for violations, claims, etc.

                self.stats['violation_links'] += 0  # Placeholder

        logger.info(f"  Added {self.stats['violation_links']} violation links")

    def apply_known_correlations(self):
        """
        Apply known correlations from domain expertise

        Links entities across datasets based on SEC enforcement relationships
        """
        logger.info("  Applying known correlations...")

        for correlation in KNOWN_CORRELATIONS:
            source_dataset = correlation['source_dataset']
            target_dataset = correlation['target_dataset']

            # Skip if datasets not available
            if source_dataset not in self.available_datasets or target_dataset not in self.available_datasets:
                continue

            # Apply correlation based on patterns
            # Implementation would use SPARQL queries to find matching entities

            self.stats['correlation_links'] += 0  # Placeholder

        logger.info(f"  Total correlation links added: {self.stats['correlation_links']}")