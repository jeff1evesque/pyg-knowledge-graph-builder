"""
WKYENG-specific enrichment logic

Weekly Earnings - Median usual weekly earnings data (6 tables)

Key difference from other BLS datasets: WKYENG uses QUARTERLY temporal
granularity (Q1-Q4) instead of monthly granularity.
"""
from rdflib import Graph
from typing import Dict, List
from glue_jobs.utils.rdf_utils import WKYENG, BLS_ENRICHMENT
from glue_jobs.enrichment.intra_source.base import DatasetEnricher
from glue_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from glue_jobs.enrichment.intra_source.bls.measurements import MEASUREMENT_TYPES
import logging

logger = logging.getLogger(__name__)


class WKYENGEnricher(DatasetEnricher):
    """
    WKYENG-specific enrichment

    Weekly Earnings provides median usual weekly earnings data, including:
    - Table 1: By sex (seasonally adjusted, quarterly)
    - Table 2: By selected characteristics (not seasonally adjusted, quarterly)
    - Table 3: By age, race, ethnicity, and sex (not seasonally adjusted, quarterly)
    - Table 4: By occupation and sex (not seasonally adjusted, quarterly)
    - Table 5: Quartiles and deciles by characteristics (not seasonally adjusted, quarterly)
    - Table 6: Part-time workers by characteristics (not seasonally adjusted, quarterly)

    Key difference: WKYENG uses quarterly temporal granularity (Q1-Q4)
    instead of monthly granularity used by other BLS datasets.

    Measurement classes:
    - QuarterlyEarningsData (Table 1)
    - QuarterlyCharacteristicData (Tables 2, 3)
    - QuarterlyOccupationData (Table 4)
    - QuarterlyDistributionData (Table 5)
    - QuarterlyPartTimeData (Table 6)

    Dimensions:
    - Sex (Total, Men, Women)
    - AgeGroup (16 to 24 years, 25 years and over, etc.)
    - Race (White, Black or African American, Asian)
    - Ethnicity (Hispanic or Latino ethnicity)
    - EducationLevel (Less than high school, Bachelor's degree, etc.)
    - Occupation (Management, Sales, Production, etc.)
    """

    # Quarter ordering for temporal sequence linking
    QUARTER_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']

    def __init__(self, graph: Graph):
        super().__init__(graph, WKYENG)

    def get_sector_keywords(self) -> Dict[str, List[str]]:
        """Extract WKYENG keywords from sector patterns"""
        keywords = {}
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            if 'wkyeng' in pattern['keywords']:
                keywords[sector_name] = pattern['keywords']['wkyeng']
        return keywords

    def get_measurement_types(self) -> Dict[str, Dict]:
        """Return WKYENG measurement type configurations"""
        return MEASUREMENT_TYPES.get('wkyeng', {})

    def link_temporal_sequences(self) -> int:
        """
        Link temporal sequences for WKYENG measurements.

        WKYENG uses quarterly granularity, so we link:
        Q1 2024 → Q2 2024 → Q3 2024 → Q4 2024 → Q1 2025 → ...

        Each measurement type has its own category property that groups
        measurements for temporal ordering.
        """
        total_links = 0

        for measurement_name, config in self.get_measurement_types().items():
            category_property = config.get('category_property')

            # Skip measurement types without a category property
            # (e.g., generic types that span multiple category properties)
            if category_property is None:
                continue

            links = self._link_quarterly_sequences(
                measurement_type=config['class'],
                category_property=category_property,
                quarter_property=config['quarter_property'],
                year_property=config['year_property'],
                measurement_name=measurement_name
            )
            total_links += links
            self.stats['temporal_sequences'] += links

        return total_links

    def _get_quarter_label(self, quarter_uri) -> str:
        """Extract quarter label (Q1, Q2, Q3, Q4) from a quarter URI"""
        uri_str = str(quarter_uri)
        # Quarter URIs are like https://www.bls.gov/wkyeng/Q1
        local_name = uri_str.split('/')[-1]

        # Check for rdfs:label
        for _, _, label in self.graph.triples((quarter_uri, None, None)):
            label_str = str(label)
            if label_str in self.QUARTER_ORDER:
                return label_str

        # Fall back to local name
        if local_name in self.QUARTER_ORDER:
            return local_name

        return local_name

    def _get_year_value(self, year_uri) -> str:
        """Extract year value from a year URI"""
        uri_str = str(year_uri)
        local_name = uri_str.split('/')[-1]

        # Check for rdfs:label
        for _, _, label in self.graph.triples((year_uri, None, None)):
            label_str = str(label)
            if label_str.isdigit() and len(label_str) == 4:
                return label_str

        # Fall back to local name
        if local_name.isdigit():
            return local_name

        return local_name

    def _link_quarterly_sequences(self, measurement_type, category_property,
                                   quarter_property, year_property,
                                   measurement_name) -> int:
        """
        Link temporal sequences for quarterly measurements.

        Groups measurements by category, sorts by year+quarter,
        and creates precedes links between consecutive measurements.
        """
        query = f"""
        SELECT ?measurement ?category ?quarter ?year WHERE {{
            ?measurement a <{measurement_type}> ;
                        <{category_property}> ?category ;
                        <{quarter_property}> ?quarter ;
                        <{year_property}> ?year .
        }}
        """

        results = list(self.graph.query(query))
        if not results:
            return 0

        # Group by category
        by_category = {}
        for row in results:
            category = row.category
            if category not in by_category:
                by_category[category] = []

            quarter_label = self._get_quarter_label(row.quarter)
            year_value = self._get_year_value(row.year)

            by_category[category].append({
                'measurement': row.measurement,
                'quarter': quarter_label,
                'year': year_value
            })

        # Sort and link
        links_added = 0
        for category, measurements in by_category.items():
            # Sort by year, then by quarter order
            measurements.sort(key=lambda x: (
                int(x['year']) if x['year'].isdigit() else 0,
                self.QUARTER_ORDER.index(x['quarter']) if x['quarter'] in self.QUARTER_ORDER else 0
            ))

            for i in range(len(measurements) - 1):
                current = measurements[i]['measurement']
                next_measurement = measurements[i + 1]['measurement']

                self.graph.add((
                    current,
                    BLS_ENRICHMENT.precedes,
                    next_measurement
                ))
                links_added += 1

        if links_added > 0:
            logger.info(f"  WKYENG.{measurement_name}: {links_added} sequence links")

        return links_added