"""
A day of the non-market graph, as something you can ask questions of.

``edges/`` and ``nodes/`` hold the same structure, and answer a one-hop
question well. They answer a two-hop question awkwardly: a self-join per hop
carrying a mandatory same-day guard, with hand-rolled recursion for anything
deeper. That shape -- "which measurements cover the region this weather alert
hit" -- is what a knowledge graph exists for, so it gets a store that answers
it directly. The store also keeps what no table can: a triple whose object is
a URI nothing typed, such as an alert's severity.

Measured on the published 2026-09-09 store: 1,413,546 triples in 185 MB, about
131 bytes a triple. The two-hop ``weather -> region <- measurement`` traversal
returned its 38,352 pairs in 10 ms; the same join over a local copy of
``edges/`` took 0.05 s, so speed is not the argument.

**One day fits; a window does not.** At that size a 30-day window is about
5.6 GB and a year about 68 GB. So this supplements the tables rather than
replacing them: a consumer loads one day for traversal and falls back to
``edges/`` for anything spanning more.

**Market is not in it.** At the same rate its ~415M triples a day would be
about 54 GB. It is a time series of numbers carrying one edge per snapshot --
not a shape a triple store earns anything on.

**No new dependency and no server.** requirements.txt already pins pyoxigraph
to parse Turtle; the same library builds the store. A managed graph database
would bill whether or not anything queried it, and the cheap tiers do not scale
to zero. A directory costs what it costs to store.

What the store holds is the ENRICHED frame's view of the data: literals are the
plain strings the loader converted them to, with their source datatypes already
folded in (see graph/turtle.py). It is the same view every other artifact in
this pipeline is built from, and not a re-serialization of the original RDF.
"""
import logging
import shutil
from pathlib import Path
from typing import Iterator

from pyspark.sql import DataFrame

from spark_jobs.utils.fs_utils import is_local_path, local_filesystem_path

# The job's logger, not this module's -- see the note in graph/config.py.
logger = logging.getLogger("build_graph")

# Rows pulled from one partition at a time. toLocalIterator streams partition
# by partition, so the driver holds one partition's rows rather than the day's.
_LOG_EVERY = 250_000


def _terms(module, subject: str, predicate: str, obj: str):
    """One triple's three terms.

    The object is a URI, a blank node or a literal, decided the same way the
    tables decide it: a bare string that still looks like a URI is a URI.
    Subjects and predicates are never literals, which RDF guarantees.
    """
    if subject.startswith("_:"):
        src = module.BlankNode(subject[2:])
    else:
        src = module.NamedNode(subject)

    if obj.startswith("_:"):
        dst = module.BlankNode(obj[2:])
    elif obj.startswith("http://") or obj.startswith("https://"):
        dst = module.NamedNode(obj)
    else:
        dst = module.Literal(obj)

    return src, module.NamedNode(predicate), dst


def _quads(module, rows: Iterator) -> Iterator:
    """Stream rows into quads, counting as it goes."""
    loaded = 0
    for row in rows:
        try:
            subject, predicate, obj = _terms(
                module, row["subject"], row["predicate"], row["object"]
            )
        except ValueError:
            # An IRI pyoxigraph will not accept. One malformed subject must not
            # cost the day's store, and the count below says how many there
            # were.
            continue

        loaded += 1
        if loaded % _LOG_EVERY == 0:
            logger.info(f"    {loaded:,} triples loaded")

        yield module.Quad(subject, predicate, obj)


def write_graph_store(triples_df: DataFrame, path: str) -> str:
    """Build a store of ``triples_df`` at ``path``. Returns the path written.

    The caller decides what goes in -- this takes the frame it is given and
    puts all of it in the store. What it refuses is a destination it cannot
    write a DIRECTORY to: the store is RocksDB, not a file, so an ``s3a://``
    work dir has nowhere to put it. That returns "" and leaves everything else
    the run produces unaffected.
    """
    import pyoxigraph

    if not is_local_path(path):
        logger.warning(
            f"No graph store: {path} is not a filesystem path, and the store "
            f"is a directory rather than a file. The tables are unaffected."
        )
        return ""

    local = Path(local_filesystem_path(path))
    if local.exists():
        # Store() opens an existing directory rather than replacing it, so a
        # rerun would union the old day into the new one.
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)

    store = pyoxigraph.Store(str(local))
    store.bulk_extend(
        _quads(
            pyoxigraph,
            triples_df
            .select("subject", "predicate", "object")
            .toLocalIterator(),
        )
    )
    store.flush()
    store.optimize()

    size_mb = sum(
        f.stat().st_size for f in local.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    logger.info(f"    graph store: {size_mb:.1f} MB at {path}")

    return path
