"""
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
from rdflib import Namespace
from typing import Dict, Set
from dateutil import parser as date_parser
import logging

logger = logging.getLogger(__name__)

# Base namespace for SEC filings (http://www.sec.gov#)
SEC_BASE = Namespace("http://www.sec.gov#")


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
    normalized = keyword.replace(' ', '')
    normalized = normalized.replace('(', '').replace(')', '')
    normalized = normalized.replace('-', '')
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
        self.stats.update({
            'company_unified': 0,
            'person_unified': 0,
            'violation_links': 0
        })
        self.available_datasets = self.detect_datasets()
        logger.info(f"Detected SEC datasets: {', '.join(self.available_datasets)}")

    def detect_datasets(self) -> Set[str]:
        """Detect which SEC datasets are present in the graph"""
        datasets = set()

        # Check for filings (using full URIs)
        filings_query = f"""
        ASK {{
            {{ ?s a <{SEC_FILINGS.Form3}> }} UNION
            {{ ?s a <{SEC_FILINGS.Form4}> }} UNION
            {{ ?s a <{SEC_FILINGS.Form5}> }} UNION
            {{ ?s a <{SEC_FILINGS.Form10K}> }} UNION
            {{ ?s a <{SEC_FILINGS.Form10Q}> }} UNION
            {{ ?s a <{SEC_FILINGS.Form8K}> }} UNION
            {{ ?s a <{SEC_FILINGS.OwnershipDocument}> }}
        }}
        """
        if self.graph.query(filings_query).askAnswer:
            datasets.add('filings')

        # Check for administrative proceedings
        admin_query = f"""
        ASK {{
            {{ ?s a <{SEC_ADMIN.AdministrativeProceeding}> }} UNION
            {{ ?s a <{SEC_ADMIN.Rule102eProceeding}> }} UNION
            {{ ?s a <{SEC_ADMIN.CeaseAndDesistProceeding}> }} UNION
            {{ ?s a <{SEC_ADMIN.Section12jProceeding}> }}
        }}
        """
        if self.graph.query(admin_query).askAnswer:
            datasets.add('administrative_proceedings')

        # Check for litigation
        lit_query = f"""
        ASK {{
            {{ ?s a <{SEC_LIT.LitigationRelease}> }} UNION
            {{ ?s a <{SEC_LIT.CivilEnforcementAction}> }}
        }}
        """
        if self.graph.query(lit_query).askAnswer:
            datasets.add('litigation')

        # Check for trading suspensions
        susp_query = f"""
        ASK {{
            {{ ?s a <{SEC_SUSP.TradingSuspension}> }} UNION
            {{ ?s a <{SEC_SUSP.Section12kSuspension}> }} UNION
            {{ ?s a <{SEC_SUSP.Section12jSuspension}> }}
        }}
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
        logger.info(f"  - Company unification: {self.stats['company_unified']}")
        logger.info(f"  - Person unification: {self.stats['person_unified']}")
        logger.info(f"  - Temporal sequences: {self.stats['temporal_sequences']}")
        logger.info(f"  - Sector links: {self.stats['sector_links']}")
        logger.info(f"  - Violation links: {self.stats['violation_links']}")
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

        companies_by_cik = {}

        # Collect CIKs from each dataset using full URIs
        cik_properties = [
            (SEC_FILINGS.hasIssuerCik, 'filings'),
            (SEC_ADMIN.cikNumber, 'administrative_proceedings'),
            (SEC_SUSP.cikNumber, 'trading_suspensions'),
        ]

        for cik_prop, dataset_name in cik_properties:
            if dataset_name not in self.available_datasets:
                continue

            query = f"""
            SELECT DISTINCT ?entity ?cik WHERE {{
                ?entity <{cik_prop}> ?cik .
            }}
            """

            for row in self.graph.query(query):
                cik = str(row.cik).strip()
                if not cik:
                    continue
                if cik not in companies_by_cik:
                    companies_by_cik[cik] = []
                if row.entity not in companies_by_cik[cik]:
                    companies_by_cik[cik].append(row.entity)

        # Create unified company entities
        for cik, company_uris in companies_by_cik.items():
            if len(company_uris) > 1:
                unified_company = UNIFIED[f"Company_{cik}"]

                if (unified_company, RDF.type, SEC_ENRICHMENT.UnifiedCompany) not in self.graph:
                    self.graph.add((unified_company, RDF.type, SEC_ENRICHMENT.UnifiedCompany))
                    self.graph.add((unified_company, SEC_ENRICHMENT.hasCik, Literal(cik)))

                for company_uri in company_uris:
                    if (unified_company, OWL.sameAs, company_uri) not in self.graph:
                        self.graph.add((unified_company, OWL.sameAs, company_uri))
                        self.stats['company_unified'] += 1

        logger.info(f"  Unified {len([c for c in companies_by_cik.values() if len(c) > 1])} companies across datasets")
        logger.info(f"  Created {self.stats['company_unified']} company unification links")

    def unify_person_entities(self):
        """
        Unify person entities across SEC datasets using CIK numbers

        Creates unified Person entities and links dataset-specific
        person entities to them using owl:sameAs
        """

        persons_by_cik = {}

        # Collect person CIKs from filings
        if 'filings' in self.available_datasets:
            query = f"""
            SELECT DISTINCT ?person ?cik WHERE {{
                ?person <{SEC_FILINGS.hasReportingOwnerCik}> ?cik .
            }}
            """
            for row in self.graph.query(query):
                cik = str(row.cik).strip()
                if not cik:
                    continue
                if cik not in persons_by_cik:
                    persons_by_cik[cik] = []
                if row.person not in persons_by_cik[cik]:
                    persons_by_cik[cik].append(row.person)

        # Collect person CIKs from admin proceedings (individual respondents)
        if 'administrative_proceedings' in self.available_datasets:
            query = f"""
            SELECT DISTINCT ?person ?cik WHERE {{
                ?person a <{SEC_ADMIN.IndividualRespondent}> ;
                        <{SEC_ADMIN.cikNumber}> ?cik .
            }}
            """
            for row in self.graph.query(query):
                cik = str(row.cik).strip()
                if not cik:
                    continue
                if cik not in persons_by_cik:
                    persons_by_cik[cik] = []
                if row.person not in persons_by_cik[cik]:
                    persons_by_cik[cik].append(row.person)

        # Create unified person entities
        for cik, person_uris in persons_by_cik.items():
            if len(person_uris) > 1:
                unified_person = UNIFIED[f"Person_{cik}"]

                if (unified_person, RDF.type, SEC_ENRICHMENT.UnifiedPerson) not in self.graph:
                    self.graph.add((unified_person, RDF.type, SEC_ENRICHMENT.UnifiedPerson))
                    self.graph.add((unified_person, SEC_ENRICHMENT.hasCik, Literal(cik)))

                for person_uri in person_uris:
                    if (unified_person, OWL.sameAs, person_uri) not in self.graph:
                        self.graph.add((unified_person, OWL.sameAs, person_uri))
                        self.stats['person_unified'] += 1

        logger.info(f"  Unified {len([p for p in persons_by_cik.values() if len(p) > 1])} persons across datasets")
        logger.info(f"  Created {self.stats['person_unified']} person unification links")

    def link_temporal_sequences(self):
        """
        Link temporal sequences for SEC filings and actions

        Links filings, proceedings, and actions in chronological order
        for the same company or person
        """
        self._link_ownership_filing_sequences()
        self._link_periodic_reporting_sequences()
        self._link_administrative_proceeding_sequences()
        self._link_litigation_sequences()
        self._link_trading_suspension_sequences()

    def _parse_date_from_entity(self, entity: URIRef, date_property: URIRef) -> str:
        """
        Extract a date string from an entity's date property.

        Handles two patterns:
        1. Direct literal: ?entity date_property "2024-01-15"
        2. Intermediate node: ?entity date_property ?dateNode . ?dateNode hasDateValue "2024-01-15"
        """
        # Try direct literal first
        for _, _, date_val in self.graph.triples((entity, date_property, None)):
            date_str = str(date_val)
            # If it looks like a date string, return it
            if any(c.isdigit() for c in date_str) and ('-' in date_str or '/' in date_str):
                return date_str

            # Otherwise it might be an intermediate node — check for hasDateValue
            if isinstance(date_val, URIRef):
                for _, _, inner_val in self.graph.triples((date_val, SEC_BASE.hasDateValue, None)):
                    return str(inner_val)

        return ''

    def _link_ownership_filing_sequences(self):
        """Link Form 3 → Form 4 → Form 5 sequences for same reporting owner"""
        if 'filings' not in self.available_datasets:
            return

        form_types = [
            (SEC_FILINGS.Form3, '3'),
            (SEC_FILINGS.Form4, '4'),
            (SEC_FILINGS.Form5, '5'),
        ]

        # Collect all ownership filings with owner and date
        filings_by_owner = {}

        for form_class, form_label in form_types:
            query = f"""
            SELECT ?filing ?owner WHERE {{
                ?filing a <{form_class}> ;
                        <{SEC_FILINGS.hasReportingOwner}> ?owner .
            }}
            """

            for row in self.graph.query(query):
                date_str = self._parse_date_from_entity(
                    row.filing, SEC_FILINGS.hasPeriodOfReport
                )
                if not date_str:
                    continue

                owner_key = str(row.owner)
                if owner_key not in filings_by_owner:
                    filings_by_owner[owner_key] = []

                filings_by_owner[owner_key].append({
                    'filing': row.filing,
                    'date': date_str,
                    'form_type': form_label
                })

        # Sort and link consecutive filings per owner
        links_added = 0
        for owner, filings in filings_by_owner.items():
            try:
                filings.sort(key=lambda x: date_parser.parse(x['date']))
            except Exception:
                continue

            for i in range(len(filings) - 1):
                current = filings[i]['filing']
                next_filing = filings[i + 1]['filing']

                if (current, SEC_ENRICHMENT.precedes, next_filing) not in self.graph:
                    self.graph.add((current, SEC_ENRICHMENT.precedes, next_filing))
                    links_added += 1
                    self.stats['temporal_sequences'] += 1

        if links_added > 0:
            logger.info(f"  Ownership filings: {links_added} sequence links")

    def _link_periodic_reporting_sequences(self):
        """Link 10-Q → 10-K sequences for same company"""
        if 'filings' not in self.available_datasets:
            return

        form_types = [
            (SEC_FILINGS.Form10K, '10-K'),
            (SEC_FILINGS.Form10Q, '10-Q'),
            (SEC_FILINGS.Form8K, '8-K'),
        ]

        filings_by_company = {}

        for form_class, form_label in form_types:
            query = f"""
            SELECT ?filing ?company WHERE {{
                ?filing a <{form_class}> ;
                        <{SEC_FILINGS.hasCompany}> ?company .
            }}
            """

            for row in self.graph.query(query):
                date_str = self._parse_date_from_entity(
                    row.filing, SEC_FILINGS.hasPeriodOfReport
                )
                if not date_str:
                    continue

                company_key = str(row.company)
                if company_key not in filings_by_company:
                    filings_by_company[company_key] = []

                filings_by_company[company_key].append({
                    'filing': row.filing,
                    'date': date_str,
                    'form_type': form_label
                })

        links_added = 0
        for company, filings in filings_by_company.items():
            try:
                filings.sort(key=lambda x: date_parser.parse(x['date']))
            except Exception:
                continue

            for i in range(len(filings) - 1):
                current = filings[i]['filing']
                next_filing = filings[i + 1]['filing']

                if (current, SEC_ENRICHMENT.precedes, next_filing) not in self.graph:
                    self.graph.add((current, SEC_ENRICHMENT.precedes, next_filing))
                    links_added += 1
                    self.stats['temporal_sequences'] += 1

        if links_added > 0:
            logger.info(f"  Periodic reports: {links_added} sequence links")

    def _link_administrative_proceeding_sequences(self):
        """Link administrative proceedings chronologically by respondent"""
        if 'administrative_proceedings' not in self.available_datasets:
            return

        proceeding_types = [
            SEC_ADMIN.AdministrativeProceeding,
            SEC_ADMIN.Rule102eProceeding,
            SEC_ADMIN.CeaseAndDesistProceeding,
            SEC_ADMIN.Section12jProceeding,
        ]

        proceedings_by_respondent = {}

        for proc_class in proceeding_types:
            query = f"""
            SELECT ?proceeding ?respondent WHERE {{
                ?proceeding a <{proc_class}> ;
                            <{SEC_ADMIN.hasRespondent}> ?respondent .
            }}
            """

            for row in self.graph.query(query):
                date_str = self._parse_date_from_entity(
                    row.proceeding, SEC_ADMIN.initiationDate
                )
                if not date_str:
                    continue

                respondent_key = str(row.respondent)
                if respondent_key not in proceedings_by_respondent:
                    proceedings_by_respondent[respondent_key] = []

                proceedings_by_respondent[respondent_key].append({
                    'proceeding': row.proceeding,
                    'date': date_str
                })

        links_added = 0
        for respondent, proceedings in proceedings_by_respondent.items():
            try:
                proceedings.sort(key=lambda x: date_parser.parse(x['date']))
            except Exception:
                continue

            for i in range(len(proceedings) - 1):
                current = proceedings[i]['proceeding']
                next_proc = proceedings[i + 1]['proceeding']

                if (current, SEC_ENRICHMENT.precedes, next_proc) not in self.graph:
                    self.graph.add((current, SEC_ENRICHMENT.precedes, next_proc))
                    links_added += 1
                    self.stats['temporal_sequences'] += 1

        if links_added > 0:
            logger.info(f"  Administrative proceedings: {links_added} sequence links")

    def _link_litigation_sequences(self):
        """Link litigation actions chronologically by defendant"""
        if 'litigation' not in self.available_datasets:
            return

        action_types = [
            (SEC_LIT.CivilEnforcementAction, SEC_LIT.filingDate, SEC_LIT.hasDefendant),
            (SEC_LIT.EmergencyAction, SEC_LIT.filingDate, SEC_LIT.hasDefendant),
        ]

        actions_by_defendant = {}

        for action_class, date_prop, defendant_prop in action_types:
            query = f"""
            SELECT ?action ?defendant WHERE {{
                ?action a <{action_class}> ;
                        <{defendant_prop}> ?defendant .
            }}
            """

            for row in self.graph.query(query):
                date_str = self._parse_date_from_entity(row.action, date_prop)
                if not date_str:
                    continue

                defendant_key = str(row.defendant)
                if defendant_key not in actions_by_defendant:
                    actions_by_defendant[defendant_key] = []

                actions_by_defendant[defendant_key].append({
                    'action': row.action,
                    'date': date_str
                })

        # Also collect litigation releases (no defendant grouping, just chronological)
        release_query = f"""
        SELECT ?release WHERE {{
            ?release a <{SEC_LIT.LitigationRelease}> .
        }}
        """

        releases = []
        for row in self.graph.query(release_query):
            date_str = self._parse_date_from_entity(row.release, SEC_LIT.releaseDate)
            if date_str:
                releases.append({
                    'action': row.release,
                    'date': date_str
                })

        links_added = 0

        # Link actions by defendant
        for defendant, actions in actions_by_defendant.items():
            try:
                actions.sort(key=lambda x: date_parser.parse(x['date']))
            except Exception:
                continue

            for i in range(len(actions) - 1):
                current = actions[i]['action']
                next_action = actions[i + 1]['action']

                if (current, SEC_ENRICHMENT.precedes, next_action) not in self.graph:
                    self.graph.add((current, SEC_ENRICHMENT.precedes, next_action))
                    links_added += 1
                    self.stats['temporal_sequences'] += 1

        # Link releases chronologically
        if len(releases) > 1:
            try:
                releases.sort(key=lambda x: date_parser.parse(x['date']))
                for i in range(len(releases) - 1):
                    current = releases[i]['action']
                    next_release = releases[i + 1]['action']

                    if (current, SEC_ENRICHMENT.precedes, next_release) not in self.graph:
                        self.graph.add((current, SEC_ENRICHMENT.precedes, next_release))
                        links_added += 1
                        self.stats['temporal_sequences'] += 1
            except Exception:
                pass

        if links_added > 0:
            logger.info(f"  Litigation actions: {links_added} sequence links")

    def _link_trading_suspension_sequences(self):
        """Link trading suspensions chronologically by company"""
        if 'trading_suspensions' not in self.available_datasets:
            return

        suspension_types = [
            SEC_SUSP.TradingSuspension,
            SEC_SUSP.Section12kSuspension,
            SEC_SUSP.Section12jSuspension,
            SEC_SUSP.EmergencySuspension,
        ]

        suspensions_by_company = {}

        for susp_class in suspension_types:
            query = f"""
            SELECT ?suspension ?company WHERE {{
                ?suspension a <{susp_class}> ;
                            <{SEC_SUSP.affectsCompany}> ?company .
            }}
            """

            for row in self.graph.query(query):
                date_str = self._parse_date_from_entity(
                    row.suspension, SEC_SUSP.startDate
                )
                if not date_str:
                    continue

                company_key = str(row.company)
                if company_key not in suspensions_by_company:
                    suspensions_by_company[company_key] = []

                suspensions_by_company[company_key].append({
                    'suspension': row.suspension,
                    'date': date_str
                })

        links_added = 0
        for company, suspensions in suspensions_by_company.items():
            try:
                suspensions.sort(key=lambda x: date_parser.parse(x['date']))
            except Exception:
                continue

            for i in range(len(suspensions) - 1):
                current = suspensions[i]['suspension']
                next_susp = suspensions[i + 1]['suspension']

                if (current, SEC_ENRICHMENT.precedes, next_susp) not in self.graph:
                    self.graph.add((current, SEC_ENRICHMENT.precedes, next_susp))
                    links_added += 1
                    self.stats['temporal_sequences'] += 1

        if links_added > 0:
            logger.info(f"  Trading suspensions: {links_added} sequence links")

    def apply_sector_patterns(self):
        """
        Apply sector-based linking patterns

        Links entities to industry sectors based on keyword matching
        against entity labels and URIs.

        Uses SEC_SECTOR_PATTERNS to match entities from filings,
        admin proceedings, litigation, and trading suspensions to
        industry sectors like financial services, healthcare, etc.
        """
        logger.info("  Applying sector patterns...")

        # Map dataset names to namespace prefixes for URI matching
        dataset_namespace_map = {
            'filings': str(SEC_FILINGS),
            'administrative_proceedings': str(SEC_ADMIN),
            'litigation': str(SEC_LIT),
            'trading_suspensions': str(SEC_SUSP),
        }

        for sector_name, pattern in SEC_SECTOR_PATTERNS.items():
            sector_uri = pattern['sector_uri']
            relationship = pattern['relationship']

            # Ensure sector entity exists
            if (sector_uri, RDF.type, SEC_ENRICHMENT.EconomicSector) not in self.graph:
                self.graph.add((sector_uri, RDF.type, SEC_ENRICHMENT.EconomicSector))
                self.graph.add((sector_uri, RDFS.label, Literal(
                    sector_name.replace('_', ' ').title()
                )))

            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                if not keywords:
                    continue

                namespace_str = dataset_namespace_map.get(dataset_name, '')
                if not namespace_str:
                    continue

                for keyword in keywords:
                    normalized = normalize_keyword_for_uri_matching(keyword)

                    # Search by URI substring
                    uri_query = f"""
                    SELECT DISTINCT ?entity WHERE {{
                        ?entity ?p ?o .
                        FILTER(STRSTARTS(STR(?entity), "{namespace_str}"))
                        FILTER(CONTAINS(STR(?entity), "{normalized}"))
                    }}
                    """

                    for row in self.graph.query(uri_query):
                        if (row.entity, SEC_ENRICHMENT.belongsToSector, sector_uri) not in self.graph:
                            self.graph.add((row.entity, SEC_ENRICHMENT.belongsToSector, sector_uri))
                            self.graph.add((row.entity, relationship, sector_uri))
                            self.stats['sector_links'] += 2

                    # Search by rdfs:label
                    label_query = f"""
                    SELECT DISTINCT ?entity WHERE {{
                        ?entity rdfs:label ?label .
                        FILTER(STRSTARTS(STR(?entity), "{namespace_str}"))
                        FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{keyword}")))
                    }}
                    """

                    for row in self.graph.query(label_query):
                        if (row.entity, SEC_ENRICHMENT.belongsToSector, sector_uri) not in self.graph:
                            self.graph.add((row.entity, SEC_ENRICHMENT.belongsToSector, sector_uri))
                            self.graph.add((row.entity, relationship, sector_uri))
                            self.stats['sector_links'] += 2

        logger.info(f"  Added {self.stats['sector_links']} sector links")

    def apply_violation_patterns(self):
        """
        Apply violation type patterns

        Links entities to violation types based on keyword matching.
        Uses SEC_VIOLATION_PATTERNS to connect related violations,
        claims, and enforcement actions across datasets.
        """

        logger.info("  Applying violation patterns...")

        dataset_namespace_map = {
            'filings': str(SEC_FILINGS),
            'administrative_proceedings': str(SEC_ADMIN),
            'litigation': str(SEC_LIT),
            'trading_suspensions': str(SEC_SUSP),
        }

        for violation_name, pattern in SEC_VIOLATION_PATTERNS.items():
            violation_uri = pattern['violation_uri']
            relationship = pattern['relationship']

            # Ensure violation type entity exists
            if (violation_uri, RDF.type, SEC_ENRICHMENT.ViolationType) not in self.graph:
                self.graph.add((violation_uri, RDF.type, SEC_ENRICHMENT.ViolationType))
                self.graph.add((violation_uri, RDFS.label, Literal(
                    violation_name.replace('_', ' ').title()
                )))

            # Collect all entities matching this violation across datasets
            matched_entities = []

            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                if not keywords:
                    continue

                namespace_str = dataset_namespace_map.get(dataset_name, '')
                if not namespace_str:
                    continue

                for keyword in keywords:
                    normalized = normalize_keyword_for_uri_matching(keyword)

                    # Search by URI substring
                    uri_query = f"""
                    SELECT DISTINCT ?entity WHERE {{
                        ?entity ?p ?o .
                        FILTER(STRSTARTS(STR(?entity), "{namespace_str}"))
                        FILTER(CONTAINS(STR(?entity), "{normalized}"))
                    }}
                    """

                    for row in self.graph.query(uri_query):
                        if row.entity not in matched_entities:
                            matched_entities.append(row.entity)

                    # Search by rdfs:label
                    label_query = f"""
                    SELECT DISTINCT ?entity WHERE {{
                        ?entity rdfs:label ?label .
                        FILTER(STRSTARTS(STR(?entity), "{namespace_str}"))
                        FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{keyword}")))
                    }}
                    """

                    for row in self.graph.query(label_query):
                        if row.entity not in matched_entities:
                            matched_entities.append(row.entity)

            # Link all matched entities to the violation type
            for entity in matched_entities:
                if (entity, SEC_ENRICHMENT.hasViolationType, violation_uri) not in self.graph:
                    self.graph.add((entity, SEC_ENRICHMENT.hasViolationType, violation_uri))
                    self.stats['violation_links'] += 1

            # Cross-link entities of the same violation type across datasets
            if len(matched_entities) > 1:
                for i in range(len(matched_entities)):
                    for j in range(i + 1, len(matched_entities)):
                        entity_a = matched_entities[i]
                        entity_b = matched_entities[j]

                        # Only cross-link entities from different datasets
                        a_str = str(entity_a)
                        b_str = str(entity_b)

                        a_dataset = None
                        b_dataset = None
                        for ds, ns in dataset_namespace_map.items():
                            if a_str.startswith(ns):
                                a_dataset = ds
                            if b_str.startswith(ns):
                                b_dataset = ds

                        if a_dataset and b_dataset and a_dataset != b_dataset:
                            if (entity_a, relationship, entity_b) not in self.graph:
                                self.graph.add((entity_a, relationship, entity_b))
                                self.stats['violation_links'] += 1

        logger.info(f"  Added {self.stats['violation_links']} violation links")

    def apply_known_correlations(self):
        """
        Apply known correlations from domain expertise

        Links entities across datasets based on SEC enforcement relationships.
        Uses match strategies defined in KNOWN_CORRELATIONS:
        - cik: Match by CIK number
        - ticker: Match by ticker symbol
        - name: Match by entity/company name
        - violation_type: Match by violation category
        - section_type: Match by statutory section
        - same_proceeding/same_action: Match within same parent entity
        """
        logger.info("  Applying known correlations...")

        for correlation in KNOWN_CORRELATIONS:
            source_dataset = correlation['source_dataset']
            target_dataset = correlation['target_dataset']

            if source_dataset not in self.available_datasets or target_dataset not in self.available_datasets:
                continue

            match_strategy = correlation.get('match_strategy', '')
            relationship = correlation['relationship']

            links_added = 0

            if match_strategy == 'cik':
                links_added = self._apply_cik_correlation(correlation)
            elif match_strategy == 'ticker':
                links_added = self._apply_ticker_correlation(correlation)
            elif match_strategy == 'name':
                links_added = self._apply_name_correlation(correlation)
            elif match_strategy in ('violation_type', 'section_type', 'violation_reason',
                                     'action_type', 'sanction_consequence', 'reason_hierarchy'):
                links_added = self._apply_pattern_correlation(correlation)
            elif match_strategy in ('same_proceeding', 'same_action', 'same_suspension'):
                links_added = self._apply_same_parent_correlation(correlation)

            if links_added > 0:
                logger.info(f"    {correlation['name']}: {links_added} links")

            self.stats['correlation_links'] += links_added

        logger.info(f"  Total correlation links added: {self.stats['correlation_links']}")

    def _get_namespace_for_dataset(self, dataset_name: str):
        """Get the RDF namespace for a dataset name"""
        namespace_map = {
            'filings': SEC_FILINGS,
            'administrative_proceedings': SEC_ADMIN,
            'litigation': SEC_LIT,
            'trading_suspensions': SEC_SUSP,
        }
        return namespace_map.get(dataset_name)

    def _apply_cik_correlation(self, correlation: Dict) -> int:
        """Apply correlation by matching CIK numbers across datasets"""
        source_cik_prop = correlation.get('source_cik_property', '')
        target_cik_prop = correlation.get('target_cik_property', '')
        relationship = correlation['relationship']

        if not source_cik_prop or not target_cik_prop:
            return 0

        # Resolve prefixed property names to full URIs
        source_prop_uri = self._resolve_property(source_cik_prop)
        target_prop_uri = self._resolve_property(target_cik_prop)

        if not source_prop_uri or not target_prop_uri:
            return 0

        # Collect source entities by CIK
        source_by_cik = {}
        source_query = f"""
        SELECT ?entity ?cik WHERE {{
            ?entity <{source_prop_uri}> ?cik .
        }}
        """
        for row in self.graph.query(source_query):
            cik = str(row.cik).strip()
            if cik:
                if cik not in source_by_cik:
                    source_by_cik[cik] = []
                source_by_cik[cik].append(row.entity)

        # Collect target entities by CIK
        target_by_cik = {}
        target_query = f"""
        SELECT ?entity ?cik WHERE {{
            ?entity <{target_prop_uri}> ?cik .
        }}
        """
        for row in self.graph.query(target_query):
            cik = str(row.cik).strip()
            if cik:
                if cik not in target_by_cik:
                    target_by_cik[cik] = []
                target_by_cik[cik].append(row.entity)

        # Match and link
        links_added = 0
        for cik in source_by_cik:
            if cik in target_by_cik:
                for source_entity in source_by_cik[cik]:
                    for target_entity in target_by_cik[cik]:
                        if (source_entity, relationship, target_entity) not in self.graph:
                            self.graph.add((source_entity, relationship, target_entity))
                            links_added += 1

        return links_added

    def _apply_ticker_correlation(self, correlation: Dict) -> int:
        """Apply correlation by matching ticker symbols across datasets"""
        source_ticker_prop = correlation.get('source_ticker_property', '')
        target_ticker_prop = correlation.get('target_ticker_property', '')
        relationship = correlation['relationship']

        if not source_ticker_prop or not target_ticker_prop:
            return 0

        source_prop_uri = self._resolve_property(source_ticker_prop)
        target_prop_uri = self._resolve_property(target_ticker_prop)

        if not source_prop_uri or not target_prop_uri:
            return 0

        # Collect source entities by ticker
        source_by_ticker = {}
        source_query = f"""
        SELECT ?entity ?ticker WHERE {{
            ?entity <{source_prop_uri}> ?ticker .
        }}
        """
        for row in self.graph.query(source_query):
            ticker = str(row.ticker).strip().upper()
            if ticker:
                if ticker not in source_by_ticker:
                    source_by_ticker[ticker] = []
                source_by_ticker[ticker].append(row.entity)

        # Collect target entities by ticker
        target_by_ticker = {}
        target_query = f"""
        SELECT ?entity ?ticker WHERE {{
            ?entity <{target_prop_uri}> ?ticker .
        }}
        """
        for row in self.graph.query(target_query):
            ticker = str(row.ticker).strip().upper()
            if ticker:
                if ticker not in target_by_ticker:
                    target_by_ticker[ticker] = []
                target_by_ticker[ticker].append(row.entity)

        # Match and link
        links_added = 0
        for ticker in source_by_ticker:
            if ticker in target_by_ticker:
                for source_entity in source_by_ticker[ticker]:
                    for target_entity in target_by_ticker[ticker]:
                        if (source_entity, relationship, target_entity) not in self.graph:
                            self.graph.add((source_entity, relationship, target_entity))
                            links_added += 1

        return links_added

    def _apply_name_correlation(self, correlation: Dict) -> int:
        """Apply correlation by matching entity names across datasets"""
        source_name_prop = correlation.get('source_name_property', '')
        target_name_prop = correlation.get('target_name_property', '')
        relationship = correlation['relationship']

        if not source_name_prop or not target_name_prop:
            return 0

        source_prop_uri = self._resolve_property(source_name_prop)
        target_prop_uri = self._resolve_property(target_name_prop)

        if not source_prop_uri or not target_prop_uri:
            return 0

        # Collect source entities by normalized name
        source_by_name = {}
        source_query = f"""
        SELECT ?entity ?name WHERE {{
            ?entity <{source_prop_uri}> ?name .
        }}
        """
        for row in self.graph.query(source_query):
            name = str(row.name).strip().lower()
            if name:
                if name not in source_by_name:
                    source_by_name[name] = []
                source_by_name[name].append(row.entity)

        # Collect target entities by normalized name
        target_by_name = {}
        target_query = f"""
        SELECT ?entity ?name WHERE {{
            ?entity <{target_prop_uri}> ?name .
        }}
        """
        for row in self.graph.query(target_query):
            name = str(row.name).strip().lower()
            if name:
                if name not in target_by_name:
                    target_by_name[name] = []
                target_by_name[name].append(row.entity)

        # Match and link (exact match on normalized names)
        links_added = 0
        for name in source_by_name:
            if name in target_by_name:
                for source_entity in source_by_name[name]:
                    for target_entity in target_by_name[name]:
                        if (source_entity, relationship, target_entity) not in self.graph:
                            self.graph.add((source_entity, relationship, target_entity))
                            links_added += 1

        return links_added

    def _apply_pattern_correlation(self, correlation: Dict) -> int:
        """
        Apply correlation by matching violation/section/reason patterns

        Finds entities whose URIs or labels contain the source and target
        patterns, then links matching pairs across datasets.
        """
        source_dataset = correlation['source_dataset']
        target_dataset = correlation['target_dataset']
        source_pattern = correlation['source_pattern']
        target_pattern = correlation['target_pattern']
        relationship = correlation['relationship']

        source_ns = self._get_namespace_for_dataset(source_dataset)
        target_ns = self._get_namespace_for_dataset(target_dataset)

        if not source_ns or not target_ns:
            return 0

        source_normalized = normalize_keyword_for_uri_matching(source_pattern)
        target_normalized = normalize_keyword_for_uri_matching(target_pattern)

        # Find source entities
        source_query = f"""
        SELECT DISTINCT ?entity WHERE {{
            {{ ?entity ?p ?o .
               FILTER(STRSTARTS(STR(?entity), "{source_ns}"))
               FILTER(CONTAINS(STR(?entity), "{source_normalized}"))
            }} UNION {{
               ?entity rdfs:label ?label .
               FILTER(STRSTARTS(STR(?entity), "{source_ns}"))
               FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{source_pattern}")))
            }}
        }}
        """
        source_entities = [row.entity for row in self.graph.query(source_query)]

        # Find target entities
        target_query = f"""
        SELECT DISTINCT ?entity WHERE {{
            {{ ?entity ?p ?o .
               FILTER(STRSTARTS(STR(?entity), "{target_ns}"))
               FILTER(CONTAINS(STR(?entity), "{target_normalized}"))
            }} UNION {{
               ?entity rdfs:label ?label .
               FILTER(STRSTARTS(STR(?entity), "{target_ns}"))
               FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{target_pattern}")))
            }}
        }}
        """
        target_entities = [row.entity for row in self.graph.query(target_query)]

        # Link matching pairs
        links_added = 0
        for source_entity in source_entities:
            for target_entity in target_entities:
                if (source_entity, relationship, target_entity) not in self.graph:
                    self.graph.add((source_entity, relationship, target_entity))
                    links_added += 1

        return links_added

    def _apply_same_parent_correlation(self, correlation: Dict) -> int:
        """
        Apply correlation for entities within the same parent entity

        For 'same_proceeding': links violations to sanctions within the same admin proceeding
        For 'same_action': links claims to relief within the same litigation action
        For 'same_suspension': links suspension to enforcement within the same suspension order
        """
        source_dataset = correlation['source_dataset']
        source_pattern = correlation['source_pattern']
        target_pattern = correlation['target_pattern']
        relationship = correlation['relationship']
        match_strategy = correlation['match_strategy']

        source_ns = self._get_namespace_for_dataset(source_dataset)
        if not source_ns:
            return 0

        source_normalized = normalize_keyword_for_uri_matching(source_pattern)
        target_normalized = normalize_keyword_for_uri_matching(target_pattern)

        # Determine the parent relationship based on strategy
        if match_strategy == 'same_proceeding':
            # Find violations and sanctions that share a parent proceeding
            parent_types = [SEC_ADMIN.AdministrativeProceeding, SEC_ADMIN.Rule102eProceeding,
                            SEC_ADMIN.CeaseAndDesistProceeding, SEC_ADMIN.Section12jProceeding]
        elif match_strategy == 'same_action':
            parent_types = [SEC_LIT.CivilEnforcementAction, SEC_LIT.EmergencyAction]
        elif match_strategy == 'same_suspension':
            parent_types = [SEC_SUSP.TradingSuspension, SEC_SUSP.Section12kSuspension,
                            SEC_SUSP.Section12jSuspension, SEC_SUSP.EmergencySuspension]
        else:
            return 0

        # Find source and target entities that share a common parent
        # Strategy: find entities whose URIs contain the patterns and are
        # connected to the same parent entity via any property
        links_added = 0

        for parent_type in parent_types:
            # Find parent entities
            parent_query = f"""
            SELECT DISTINCT ?parent WHERE {{
                ?parent a <{parent_type}> .
            }}
            """

            for parent_row in self.graph.query(parent_query):
                parent = parent_row.parent

                # Find all entities connected to this parent
                children_query = f"""
                SELECT DISTINCT ?child WHERE {{
                    {{ ?parent ?p ?child . FILTER(?parent = <{parent}>) }}
                    UNION
                    {{ ?child ?p ?parent . FILTER(?parent = <{parent}>) }}
                    FILTER(isIRI(?child))
                    FILTER(STRSTARTS(STR(?child), "{source_ns}"))
                }}
                """

                children = [row.child for row in self.graph.query(children_query)]

                # Separate into source and target matches
                source_matches = [c for c in children if source_normalized in str(c)]
                target_matches = [c for c in children if target_normalized in str(c)]

                # Link source to target within same parent
                for source_entity in source_matches:
                    for target_entity in target_matches:
                        if source_entity != target_entity:
                            if (source_entity, relationship, target_entity) not in self.graph:
                                self.graph.add((source_entity, relationship, target_entity))
                                links_added += 1

        return links_added

    def _resolve_property(self, prefixed_name: str) -> str:
        """
        Resolve a prefixed property name to a full URI string

        Examples:
            "filings:hasIssuerCik" → "http://www.sec.gov/filings#hasIssuerCik"
            "sec:cikNumber" → "https://www.sec.gov/ontology/administrative-proceedings#cikNumber"
            "seclit:filingDate" → "https://www.sec.gov/ontology/litigation#filingDate"
            "secsusp:tickerSymbol" → "https://www.sec.gov/ontology/trading-suspensions#tickerSymbol"
        """
        prefix_map = {
            'filings:': str(SEC_FILINGS),
            'sec:': str(SEC_ADMIN),
            'seclit:': str(SEC_LIT),
            'secsusp:': str(SEC_SUSP),
        }

        for prefix, namespace in prefix_map.items():
            if prefixed_name.startswith(prefix):
                local_name = prefixed_name[len(prefix):]
                return f"{namespace}{local_name}"

        return ''