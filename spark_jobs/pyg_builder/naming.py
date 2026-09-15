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

A URI in no registered namespace is named ``unknown_<last segment>``, and a
build that would put such a name in the graph fails instead (#406): two
vocabularies that both have a Widget class would become one ``unknown_Widget``
node type. ``check_registered_names`` is that check.

Pure Spark expressions, no Python UDF: the conversion runs on the JVM across
every executor. The inverse runs on the driver over a few hundred distinct
relation names.

This module imports no torch, which is what lets the enrichment leg name
relations without the PyG builder installed.
"""
import logging
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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

# What a URI in no registered namespace is named under. No registered prefix
# is "unknown" (tests/test_namespaces.py), so a name starting with this came
# through the fallback.
UNREGISTERED_NAME_PREFIX = "unknown_"

# The pyg_config key that keeps unknown_ names instead of failing the build.
ALLOW_UNREGISTERED_NAMESPACES = "allow_unregistered_namespaces"

# How many unregistered names a message lists.
_NAMES_SHOWN = 10


def _table(
    namespace_prefixes: Optional[Sequence[Tuple[str, str]]],
) -> Sequence[Tuple[str, str]]:
    """The namespace table to name with: every registered source's, unless the
    caller passes its own, as a test with a toy source does."""
    return NAMESPACE_PREFIXES if namespace_prefixes is None else namespace_prefixes


def prefixed_local_name_expr(
    uri_col: str,
    namespace_prefixes: Optional[Sequence[Tuple[str, str]]] = None,
) -> F.Column:
    """A Column expression turning a URI into its ``prefix_localName``.

    A chain of WHEN clauses, one per known namespace, testing startsWith and
    taking the rest as the local name. A URI in no known namespace keeps its
    last path segment under an ``unknown_`` prefix, and one with no segment to
    take falls back to a hash -- so the result is always a usable name. It is
    not a safe one: two vocabularies with a class of the same name get the same
    ``unknown_`` name, which is why the mappers fail the build on it unless
    pyg_config allows it.

    Namespace order matters and is NOT decided here: NAMESPACE_PREFIXES puts a
    longer namespace ahead of any namespace it contains, which
    tests/test_namespaces.py asserts over every pair.
    """
    col = F.col(uri_col)
    expr = None

    for namespace, prefix in _table(namespace_prefixes):
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
    fallback_name = F.concat(F.lit(UNREGISTERED_NAME_PREFIX), fallback_local)

    return expr.otherwise(
        F.when(
            F.length(fallback_local) > 0, fallback_name
        ).otherwise(
            F.concat(
                F.lit(UNREGISTERED_NAME_PREFIX),
                F.abs(F.hash(col)).cast("string"),
            )
        )
    )


def prefixed_local_name(
    uri: str,
    namespace_prefixes: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    """The name ``prefixed_local_name_expr`` would give this URI.

    The driver-side twin, for the few places that need to name a COLUMN after a
    vocabulary term rather than compute a name per row. Kept honest by a test
    that runs both over the same URIs.

    One case is not covered: a URI with no ``/`` or ``#`` to take a local name
    from, where the expression falls back to Spark's own hash. Nothing can
    reproduce that in Python, and no vocabulary term looks like that, so this
    says so instead of guessing.
    """
    for namespace, prefix in _table(namespace_prefixes):
        if uri.startswith(namespace):
            local = uri[len(namespace):].strip("/#")
            if local:
                return f"{prefix}_{local}"

    match = re.search(r"[#/]([^#/]+)$", uri)
    if match:
        return f"{UNREGISTERED_NAME_PREFIX}{match.group(1)}"

    raise ValueError(
        f"{uri!r} has no local name; only the Spark expression can name it, "
        f"and it does so with a hash"
    )


def relation_to_predicate_uri(
    relation: str,
    namespace_prefixes: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    """The predicate URI a relation name came from.

    The inverse of the expression above, for the one direction that cannot be
    recovered from the data: a resolved edge carries the relation name and not
    the predicate it was built from. A name whose prefix is not one of ours is
    returned unchanged -- it came through the ``unknown_`` fallback, where the
    namespace was already lost.
    """
    for namespace, prefix in _table(namespace_prefixes):
        if relation.startswith(f"{prefix}_"):
            return f"{namespace}{relation[len(prefix) + 1:]}"
    return relation


def _namespace_of(uri: str) -> str:
    """A URI up to and including its last ``/`` or ``#``: the part the
    fallback dropped, and what a source would register."""
    match = re.search(r"[#/][^#/]+$", uri)
    return uri[:match.start() + 1] if match else uri


def check_registered_names(
    kind: str,
    names: Iterable[str],
    allowed: bool,
    find_uris: Callable[[List[str]], Iterable[str]],
) -> None:
    """Fail the build when a node type or relation name came through the
    ``unknown_`` fallback.

    ``names`` are names the build has already collected, so a build whose
    namespaces are all registered does no extra work. ``find_uris(names)``
    looks up the URIs behind the unregistered names, and runs only when there
    are some, so the message can say which namespaces to register.

    ``allowed`` is pyg_config's ``allow_unregistered_namespaces``: an
    exploratory run keeps the ``unknown_`` names, with a warning.

    Raises:
        ValueError: naming the names and their namespaces, unless ``allowed``.
    """
    found = sorted({
        name for name in names if name.startswith(UNREGISTERED_NAME_PREFIX)
    })
    if not found:
        return

    shown = ", ".join(found[:_NAMES_SHOWN])
    if len(found) > _NAMES_SHOWN:
        shown += f" and {len(found) - _NAMES_SHOWN} more"

    if allowed:
        logger.warning(
            f"{kind} from namespaces no source registers, kept because "
            f"pyg_config sets {ALLOW_UNREGISTERED_NAMESPACES}: {shown}"
        )
        return

    namespaces = sorted({_namespace_of(uri) for uri in find_uris(found)})
    raise ValueError(
        f"{kind} from namespaces no source registers: {shown}. Namespaces: "
        f"{', '.join(namespaces) or 'none found'}. Add each one to a source "
        f"spec's namespaces in spark_jobs/sources/, or set "
        f'"{ALLOW_UNREGISTERED_NAMESPACES}": true in pyg_config to build them '
        f"under {UNREGISTERED_NAME_PREFIX} names."
    )
