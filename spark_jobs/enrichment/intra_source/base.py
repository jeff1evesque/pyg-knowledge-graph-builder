"""
Base classes for intra-source enrichment
"""
from abc import ABC, abstractmethod
from rdflib import Graph, Namespace
from typing import Dict, Set, List


class IntraSourceEnricher(ABC):
    """Base class for all intra-source enrichers"""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.stats = {
            'temporal_unified': 0,
            'temporal_sequences': 0,
            'sector_links': 0,
            'correlation_links': 0,
            'hierarchy_links': 0
        }

    @abstractmethod
    def detect_datasets(self) -> Set[str]:
        """Detect which datasets are present in the graph"""
        pass

    @abstractmethod
    def enrich(self) -> Dict[str, int]:
        """Run enrichment and return statistics"""
        pass


class DatasetEnricher(ABC):
    """Base class for dataset-specific enrichers (CPI, PPI, etc.)"""

    def __init__(self, graph: Graph, namespace: Namespace):
        self.graph = graph
        self.namespace = namespace
        self.stats = {
            'temporal_sequences': 0,
            'sector_links': 0,
            'correlation_links': 0,
            'hierarchy_links': 0
        }

    @abstractmethod
    def get_sector_keywords(self) -> Dict[str, List[str]]:
        """Return sector keywords for this dataset"""
        pass

    @abstractmethod
    def get_measurement_types(self) -> Dict[str, Dict]:
        """Return measurement type configurations"""
        pass

    @abstractmethod
    def link_temporal_sequences(self) -> int:
        """Link temporal sequences for this dataset"""
        pass

    def get_stats(self) -> Dict[str, int]:
        """Return enrichment statistics"""
        return self.stats.copy()