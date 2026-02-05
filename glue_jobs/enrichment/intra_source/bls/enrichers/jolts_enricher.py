"""
JOLTS-specific enrichment logic
"""
from rdflib import Graph
from typing import Dict, List
from glue_jobs.utils.rdf_utils import JOLTS, BLS_ENRICHMENT, get_month_name, get_year_value
from glue_jobs.enrichment.intra_source.base import DatasetEnricher
from glue_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from glue_jobs.enrichment.intra_source.bls.measurements import MEASUREMENT_TYPES
import logging

logger = logging.getLogger(__name__)


class JOLTSEnricher(DatasetEnricher):
    """JOLTS-specific enrichment"""

    def __init__(self, graph: Graph):
        super().__init__(graph, JOLTS)
        self.month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

    def get_sector_keywords(self) -> Dict[str, List[str]]:
        """Extract JOLTS keywords from sector patterns"""
        keywords = {}
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            if 'jolts' in pattern['keywords']:
                keywords[sector_name] = pattern['keywords']['jolts']
        return keywords

    def get_measurement_types(self) -> Dict[str, Dict]:
        """Return JOLTS measurement type configurations"""
        return MEASUREMENT_TYPES.get('jolts', {})

    def link_temporal_sequences(self) -> int:
        """Link temporal sequences for JOLTS measurements"""
        total_links = 0

        for measurement_name, config in self.get_measurement_types().items():
            links = self._link_sequences_for_measurement(
                measurement_type=config['class'],
                category_property=config['category_property'],
                month_property=config['month_property'],
                year_property=config['year_property'],
                measurement_name=measurement_name
            )
            total_links += links
            self.stats['temporal_sequences'] += links

        return total_links

    def _link_sequences_for_measurement(self, measurement_type, category_property,
                                        month_property, year_property, measurement_name) -> int:
        """Helper to link temporal sequences for a specific measurement type"""
        query = f"""
        SELECT ?measurement ?category ?month ?year WHERE {{
            ?measurement a <{measurement_type}> ;
                        <{category_property}> ?category ;
                        <{month_property}> ?month ;
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

            by_category[category].append({
                'measurement': row.measurement,
                'month': get_month_name(row.month),
                'year': get_year_value(row.year)
            })

        # Sort and link
        links_added = 0
        for category, measurements in by_category.items():
            measurements.sort(key=lambda x: (
                int(x['year']),
                self.month_order.index(x['month'])
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
            logger.info(f"  JOLTS.{measurement_name}: {links_added} sequence links")

        return links_added