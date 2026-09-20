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

THE LEGACY NAMESPACE REWRITE
----------------------------
Every source also gets one repair it does not declare, applied first: the
pre-deploy flat URIs are rewritten to the nested spelling the mappers emit now.

Registering both spellings (namespaces.LEGACY_VOCABULARIES) makes them name one
node TYPE. It cannot make them one NODE, because node identity is the URI:
id/cpi/February and id/bls/common/February are two subjects and therefore two
nodes. BLS appends by streaming stored row groups through undecoded, so after
the deploy one object holds rows of both spellings, split by which series
happened to be restated -- and a closed year is never restated at all, so the
split does not drain, it persists.

Measured on the 2026 feeds: 11,382 of 101,342 BLS individuals (11.2%) are
declared by more than one row, so they are the ones that can end up existing
under both spellings at once. They are also the months, years, categories,
states and areas -- BLS's entire hub layer, and the reason its measurements form
a graph rather than a pile. Splitting them is the failure this prevents.
"""
import logging
import re
from typing import List, Tuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import legacy_rewrites

logger = logging.getLogger(__name__)

# The triple columns a URI can appear in. object_datatype holds xsd types, which
# no source vocabulary reaches, and a literal object matches no namespace.
_URI_COLUMNS = ("subject", "predicate", "object")


def _legacy_rewrite(column: Column, rewrites: List[Tuple[str, str, bool]]) -> Column:
    """Swap a URI's legacy prefix for the current one, or pass it through."""
    expr = None
    for old, new, anchored in rewrites:
        if anchored:
            condition = column.rlike(f"^{re.escape(old)}[^/]+$")
        else:
            condition = column.startswith(old)
        replacement = F.concat(F.lit(new), F.substring(column, len(old) + 1, 1000))
        expr = (
            F.when(condition, replacement) if expr is None
            else expr.when(condition, replacement)
        )
    return column if expr is None else expr.otherwise(column)


def canonicalize_legacy_namespaces(
    triples_df: DataFrame, spec: SourceSpec,
) -> DataFrame:
    """Rewrite this source's pre-deploy URIs to the spelling it emits now.

    A no-op on a frame that is already nested: no branch matches and every
    column passes through. That is what lets one run read flat input, nested
    input, and a BLS object holding both.
    """
    # Only this source's, since the loader canonicalizes per path: market is
    # 99.5% of rows and has exactly one vocabulary with a legacy form.
    rewrites = legacy_rewrites([namespace for namespace, _prefix in spec.namespaces])
    if not rewrites:
        return triples_df
    for column in _URI_COLUMNS:
        triples_df = triples_df.withColumn(
            column, _legacy_rewrite(F.col(column), rewrites)
        )
    return triples_df


def canonicalize_source_triples(triples_df: DataFrame, spec: SourceSpec) -> DataFrame:
    """Apply one source's identifier repair to the rows loaded from its paths.

    One entry point so the loader does not grow a rule list. A source that
    declares no repair gets its frame back unchanged.

    The legacy namespace rewrite runs FIRST, and every source gets it. SEC's own
    repair keys on identifier_namespace(SEC_FILINGS), which is the nested
    spelling -- run the other way round it would silently skip every flat row.
    """
    triples_df = canonicalize_legacy_namespaces(triples_df, spec)
    if spec.canonicalize is None:
        return triples_df
    logger.info(f"Canonicalizing {spec.label} identifier shapes")
    return spec.canonicalize(triples_df)
