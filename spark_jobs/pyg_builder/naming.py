"""
How a URI becomes a name, and how a name becomes a URI again.

Node types and relations are both named the same way: replace the namespace
with its short prefix and keep the local name, so
``https://jefflevesque.com/ontology/bls/precedes`` is ``bls_enrichment_precedes``
whether it arrives as an rdf:type object or as a predicate. node_mapper and
edge_mapper each carried their own copy of that rule; the copies agreed, but
nothing made them, and a third copy was about to be written for the query
tables -- whose whole promise is that a relation in ``edges/`` resolves through
``edge_types/`` to the same predicate ``graph_schema.json`` names.

Pure Spark expressions, no Python UDF: the conversion runs on the JVM across
every executor. The inverse runs on the driver over a few hundred distinct
relation names.

This module imports no torch, which is what lets the enrichment leg name
relations without the PyG builder installed.
"""
import logging

from pyspark.sql import functions as F

from spark_jobs.utils.rdf_utils import NAMESPACE_PREFIXES

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Predicates that never make an edge. rdf:type states what a node IS, and the
# rdfs/owl four are statements about the vocabulary rather than about the data.
EXCLUDED_EDGE_PREDICATES = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#imports",
}


def prefixed_local_name_expr(uri_col: str) -> F.Column:
    """A Column expression turning a URI into its ``prefix_localName``.

    A chain of WHEN clauses, one per known namespace, testing startsWith and
    taking the rest as the local name. A URI in no known namespace keeps its
    last path segment under an ``unknown_`` prefix, and one with no segment to
    take falls back to a hash -- so the result is always a usable name, and a
    vocabulary nobody declared still shows up as itself rather than collapsing
    with every other stranger.

    Namespace order matters and is NOT decided here: NAMESPACE_PREFIXES puts a
    longer namespace ahead of any namespace it contains, which
    tests/test_namespaces.py asserts over every pair.
    """
    col = F.col(uri_col)
    expr = None

    for namespace, prefix in NAMESPACE_PREFIXES:
        ns_len = len(namespace)
        # Everything after the namespace, with any leading or trailing / and #
        # trimmed off.
        local_name = F.substring(col, ns_len + 1, 1000)
        local_name = F.regexp_replace(local_name, r"^[/#]+|[/#]+$", "")
        name = F.concat(F.lit(f"{prefix}_"), local_name)

        condition = col.startswith(namespace) & (F.length(local_name) > 0)

        if expr is None:
            expr = F.when(condition, name)
        else:
            expr = expr.when(condition, name)

    fallback_local = F.regexp_extract(col, r"[#/]([^#/]+)$", 1)
    fallback_name = F.concat(F.lit("unknown_"), fallback_local)

    return expr.otherwise(
        F.when(
            F.length(fallback_local) > 0, fallback_name
        ).otherwise(
            F.concat(F.lit("unknown_"), F.abs(F.hash(col)).cast("string"))
        )
    )


def relation_to_predicate_uri(relation: str) -> str:
    """The predicate URI a relation name came from.

    The inverse of the expression above, for the one direction that cannot be
    recovered from the data: a resolved edge carries the relation name and not
    the predicate it was built from. A name whose prefix is not one of ours is
    returned unchanged -- it came through the ``unknown_`` fallback, where the
    namespace was already lost.
    """
    for namespace, prefix in NAMESPACE_PREFIXES:
        if relation.startswith(f"{prefix}_"):
            return f"{namespace}{relation[len(prefix) + 1:]}"
    return relation
