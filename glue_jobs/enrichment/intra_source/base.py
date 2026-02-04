"""
Base classes for intra-source enrichment
"""
from abc import ABC, abstractmethod
from rdflib import Graph
from typing import Dict, Set


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