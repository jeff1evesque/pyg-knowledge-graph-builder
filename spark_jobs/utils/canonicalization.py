"""
Source-shape canonicalization — the same fact, spelled one way.

WHAT THIS IS NOT
----------------
Not enrichment. Every repair applied here is value-preserving: it asserts no
new fact, mints no new relationship, and adds no row. It rewrites an identifier
that upstream spelled two ways into the one spelling both halves of a join can
see. Enrichment lives one layer up and runs after this.

It sits in the LOADER rather than in a pipeline phase because the split it
repairs is an identity split: the two spellings are different URIs, so they are
different nodes, and every consumer downstream -- enrichment, the enriched
Parquet, node_mapper -- has to agree on which one is the entity. Repairing it
once, before anything reads the frame, is the only placement where they cannot
disagree.

WHY IT EXISTS
-------------
A join keyed on a term the data no longer emits matches nothing and is
detectable by diffing the vocabulary (see bin/check_vocabulary_drift.py). This
is the adjacent failure: the term is live, the join is correct, and the DATA is
shaped two ways, so the join matches some of what it should and silently
under-covers.

Each source declares its own repair as its spec's ``canonicalize``
(spark_jobs/sources/). Only SEC has one: utils/sec_identifiers.py, which also
holds the measured instance.
"""
from pyspark.sql import DataFrame

from spark_jobs.sources.spec import SourceSpec

import logging

logger = logging.getLogger(__name__)


def canonicalize_source_triples(triples_df: DataFrame, spec: SourceSpec) -> DataFrame:
    """Apply one source's identifier repair to the rows loaded from its paths.

    One entry point so the loader does not grow a rule list. A source that
    declares no repair gets its frame back unchanged.
    """
    if spec.canonicalize is None:
        return triples_df
    logger.info(f"Canonicalizing {spec.label} identifier shapes")
    return spec.canonicalize(triples_df)
