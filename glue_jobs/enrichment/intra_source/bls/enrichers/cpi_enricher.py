"""
CPI-specific enrichment logic (PySpark).

CPI uses only the base class temporal sequence linking.
All measurement types are configured in MEASUREMENT_TYPES['cpi'].
No custom enrichment logic beyond what the base class provides.
"""
from glue_jobs.enrichment.intra_source.bls.enrichers.base_enricher import BLSDatasetEnricher

import logging

logger = logging.getLogger(__name__)


class CPIEnricher(BLSDatasetEnricher):
    """CPI-specific enrichment — delegates entirely to base class."""

    dataset_name = 'cpi'