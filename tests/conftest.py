"""
Pytest configuration for BLS enrichment tests
"""
import pytest
import logging
from rdflib import Graph

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture
def sample_cpi_graph():
    """Fixture providing CPI test graph"""
    from spark_jobs.utils.rdf_utils import RDFGraphLoader
    from tests.test_bls_enrichment import create_sample_cpi_data

    loader = RDFGraphLoader()
    cpi_ttl = create_sample_cpi_data()
    loader.load_from_string(cpi_ttl, format='turtle')
    return loader.graph


@pytest.fixture
def sample_ppi_graph():
    """Fixture providing PPI test graph"""
    from spark_jobs.utils.rdf_utils import RDFGraphLoader
    from tests.test_bls_enrichment import create_sample_ppi_data

    loader = RDFGraphLoader()
    ppi_ttl = create_sample_ppi_data()
    loader.load_from_string(ppi_ttl, format='turtle')
    return loader.graph


@pytest.fixture
def sample_eci_graph():
    """Fixture providing ECI test graph"""
    from spark_jobs.utils.rdf_utils import RDFGraphLoader
    from tests.test_bls_enrichment import create_sample_eci_data

    loader = RDFGraphLoader()
    eci_ttl = create_sample_eci_data()
    loader.load_from_string(eci_ttl, format='turtle')
    return loader.graph


@pytest.fixture
def sample_jolts_graph():
    """Fixture providing JOLTS test graph"""
    from spark_jobs.utils.rdf_utils import RDFGraphLoader
    from tests.test_bls_enrichment import create_sample_jolts_data

    loader = RDFGraphLoader()
    jolts_ttl = create_sample_jolts_data()
    loader.load_from_string(jolts_ttl, format='turtle')
    return loader.graph


@pytest.fixture
def sample_empsit_graph():
    """Fixture providing EMPSIT test graph"""
    from spark_jobs.utils.rdf_utils import RDFGraphLoader
    from tests.test_bls_enrichment import create_sample_empsit_data

    loader = RDFGraphLoader()
    empsit_ttl = create_sample_empsit_data()
    loader.load_from_string(empsit_ttl, format='turtle')
    return loader.graph


@pytest.fixture
def multi_dataset_graph():
    """Fixture providing multi-dataset test graph"""
    from spark_jobs.utils.rdf_utils import RDFGraphLoader
    from tests.test_bls_enrichment import (
        create_sample_cpi_data,
        create_sample_ppi_data,
        create_sample_eci_data,
        create_sample_jolts_data,
        create_sample_empsit_data
    )

    loader = RDFGraphLoader()

    loader.load_from_string(create_sample_cpi_data(), format='turtle')
    loader.load_from_string(create_sample_ppi_data(), format='turtle')
    loader.load_from_string(create_sample_eci_data(), format='turtle')
    loader.load_from_string(create_sample_jolts_data(), format='turtle')
    loader.load_from_string(create_sample_empsit_data(), format='turtle')

    return loader.graph


@pytest.fixture
def empty_graph():
    """Fixture providing empty graph"""
    return Graph()