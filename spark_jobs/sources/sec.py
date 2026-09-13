"""SEC: the filings feed, the one SEC feed that carries RDF."""
from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import (
    IDENTIFIER_BASE,
    SEC_COMMON,
    SEC_ENRICHMENT,
    SEC_FILINGS,
)

SPEC = SourceSpec(
    name="sec",
    path_fragments=("source=sec",),
    namespaces=(
        (str(SEC_FILINGS), "filings"),
        (str(SEC_COMMON), "sec_common"),
        (str(SEC_ENRICHMENT), "sec_enrichment"),
    ),
    enrichment_namespace=str(SEC_ENRICHMENT),
    date_predicates=(
        str(SEC_FILINGS.hasPeriodOfReport),
        str(SEC_FILINGS.hasFilingDate),
    ),
    temporal_prefix=f"{IDENTIFIER_BASE}temporal/sec/",
)
