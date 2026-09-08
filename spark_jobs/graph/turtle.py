"""
One Turtle blob -> the four triple columns, and the batching that carries them.

THIS MODULE RUNS ON EXECUTORS, AND THAT CONSTRAINS WHAT IT MAY IMPORT.
build_graph.py is submitted by path, so on the driver that file is ``__main__``,
and cloudpickle ships a ``__main__`` function BY VALUE -- the code object
travels and executors never import the module it came from. A function that
lives in a package is pickled by reference instead, so every Python worker
running the parse now executes ``import spark_jobs.graph.turtle``.

Executors run a venv packaged separately from this repo
(requirements-executor.txt), holding only pyoxigraph, rdflib, pandas, pyarrow
and setuptools. If this module's transitive imports ever reach past that set
plus the standard library, every parse task raises ModuleNotFoundError -- while
the unit suite stays green, because the driver venv has everything. That is why
the heavy imports below sit inside the function bodies and only ``typing`` is
imported at the top, and why tests/test_executor_imports.py fails the build when
that stops being true.
"""
from typing import List, Optional, Tuple


# ============================================
# Blank-node labels (shared by both parse paths)
# ============================================
def deterministic_bnode_labels(graph, rounds: int = 3) -> dict:
    """Map every blank node in an rdflib graph to a content-derived label.

    rdflib mints a FRESH RANDOM identifier for each blank node on every parse
    (``_:nd0dc1ff...`` one run, ``_:n54d9d7...`` the next). Those labels flow
    into the triples as subject/object values and are later hashed into feature
    vector slots, so an identical input produced a different graph on every run
    — the ``jolts_*`` non-determinism (JOLTS models measurements as blank nodes).
    Replacing the random label with one derived from the blank node's *content*
    makes the parse reproducible.

    A blank node's signature is the sorted set of its outgoing
    ``(predicate, object)`` and incoming ``(subject, predicate)`` edges. Nested
    blank-node references would be circular, so they contribute the neighbour's
    signature from the previous round and the whole thing is refined ``rounds``
    times (standard colour-refinement). Three rounds distinguishes any nesting
    depth this data actually uses; more only matters for pathologically
    symmetric graphs.

    Structurally identical blank nodes hash alike, and collapsing them would
    silently merge two distinct measurements, so a group of colliding nodes gets
    a positional suffix. Which member takes which index depends on set iteration
    order and is therefore NOT stable — but it does not need to be: identical
    signatures mean the nodes are interchangeable, so any assignment yields the
    same triple multiset.
    """
    import hashlib

    from rdflib import BNode

    bnodes = {
        term
        for triple in graph
        for term in (triple[0], triple[2])
        if isinstance(term, BNode)
    }
    if not bnodes:
        return {}

    def _ref(term, labels):
        return labels[term] if isinstance(term, BNode) else str(term)

    labels = {b: "" for b in bnodes}
    for _ in range(rounds):
        labels = {
            b: hashlib.sha1(
                "\n".join(
                    sorted(
                        f">|{p}|{_ref(o, labels)}"
                        for p, o in graph.predicate_objects(b)
                    )
                    + sorted(
                        f"<|{p}|{_ref(s, labels)}"
                        for s, p in graph.subject_predicates(b)
                    )
                ).encode("utf-8")
            ).hexdigest()
            for b in bnodes
        }

    by_signature: dict = {}
    for b in bnodes:
        by_signature.setdefault(labels[b], []).append(b)

    return {
        b: (f"{sig[:32]}_{i}" if len(group) > 1 else sig[:32])
        for sig, group in by_signature.items()
        for i, b in enumerate(group)
    }


# ============================================
# Turtle blob → the UDF's four columns
# ============================================
# rdflib parses Turtle in pure Python, and profiling the UDF on real blobs
# (2026-09-05, 13,024 blobs across all four sources) put 88% of its time inside
# g.parse(): 578 us of a 656 us blob for market, and the same share everywhere.
# Graph() construction was 0.4%, the blank-node relabel 4.5%, the emit loop 5.8%
# and pickling the result 1.1%. So the parser is the phase, and a faster one is
# the only change with room in it.
#
# pyoxigraph is that parser -- Rust, and a wheel, so it travels to executors
# through requirements-executor.txt like any other dependency. Measured on the
# same blobs it is 10-13x on the whole UDF, not just the parse, because dropping
# rdflib also drops a Graph and three term objects per triple.
#
# WHAT MUST NOT CHANGE is what the rows say: these strings are hashed into
# feature slots, so a different spelling is a different graph. rdflib still
# defines that spelling -- see _literal_columns, which borrows rdflib's own
# lexical-to-Python table rather than restating it.
_LEXICAL_TO_PYTHON: Optional[dict] = None

RDF_LANG_STRING = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"


def _lexical_converters() -> dict:
    """rdflib's lexical→Python table, re-keyed by plain string.

    THE RE-KEYING IS THE POINT. rdflib keys it by ``URIRef``, and ``URIRef``
    overrides ``__hash__``, so a lookup with an ordinary string never finds a
    ``URIRef`` entry -- it misses silently and the caller keeps the lexical
    form. Measured while building this: every boolean came out ``"false"``
    instead of ``"False"`` and every ``xsd:dateTime`` kept its ``T`` where
    ``str(datetime)`` puts a space. Both are hashed into feature slots.

    Built once per worker process rather than per row.
    """
    global _LEXICAL_TO_PYTHON
    if _LEXICAL_TO_PYTHON is None:
        from rdflib.term import XSDToPython

        _LEXICAL_TO_PYTHON = {
            str(datatype): convert
            for datatype, convert in XSDToPython.items()
            if datatype is not None and convert is not None
        }
    return _LEXICAL_TO_PYTHON


def _literal_columns(literal) -> Tuple[str, str]:
    """``(object, object_datatype)`` for one literal, spelled as rdflib spells it.

    Mirrors ``str(Literal.toPython())`` exactly, including the two cases that
    are easy to get wrong: an unconvertible datatype and an ill-typed value both
    keep the lexical form, because rdflib's ``toPython()`` hands back the
    ``Literal`` itself when ``.value`` is ``None``.
    """
    datatype = literal.datatype.value if literal.datatype is not None else None

    # rdflib reports a language-tagged literal's datatype as None rather than
    # rdf:langString, and this column has always followed rdflib.
    if datatype is None or datatype == RDF_LANG_STRING:
        return literal.value, ""

    # ONE PLACE THIS CANNOT FOLLOW RDFLIB, and it is RDF 1.1's doing: a plain
    # literal and an ^^xsd:string literal are the same term, so pyoxigraph
    # reports xsd:string for both while rdflib reports it only for the second.
    # Reporting it is the reading that leaves this corpus untouched -- every
    # string literal in it is written ^^xsd:string in the source text (measured
    # 2026-09-05: 3,461 of 3,461 on SEC, 1,744 of 1,744 on NOAA), so erasing
    # xsd:string here would strip a marker off every one of them. What changes
    # is only a source that ships a bare "literal": it now contributes an
    # xsd:string observation where it used to contribute none.

    convert = _lexical_converters().get(datatype)
    if convert is None:
        return literal.value, datatype
    try:
        value = convert(literal.value)
    except Exception:
        return literal.value, datatype
    return (literal.value if value is None else str(value)), datatype


def _blank_node_labels(triples) -> dict:
    """Content-derived blank-node labels, keyed by pyoxigraph's node id.

    Delegates to ``deterministic_bnode_labels`` so there is one definition of
    what a label means, which costs building an rdflib graph -- about what the
    whole fast path costs. Paid only by blobs that actually carry a blank node:
    a full day of all four sources shipped none (measured 2026-09-05), and the
    e2e fixtures do, so the branch stays covered.
    """
    from pyoxigraph import BlankNode, NamedNode

    if not any(
        isinstance(term, BlankNode)
        for triple in triples
        for term in (triple.subject, triple.object)
    ):
        return {}

    from rdflib import BNode, Graph, Literal, URIRef

    def to_rdflib(term):
        if isinstance(term, NamedNode):
            return URIRef(term.value)
        if isinstance(term, BlankNode):
            return BNode(term.value)
        if term.language:
            return Literal(term.value, lang=term.language)
        if term.datatype is not None:
            return Literal(term.value, datatype=URIRef(term.datatype.value))
        return Literal(term.value)

    graph = Graph()
    for triple in triples:
        graph.add(
            (
                to_rdflib(triple.subject),
                to_rdflib(triple.predicate),
                to_rdflib(triple.object),
            )
        )

    return {
        str(bnode): label
        for bnode, label in deterministic_bnode_labels(graph).items()
    }


def turtle_rows_or_skip(turtle_str: str) -> List[dict]:
    """``turtle_to_rows`` with the UDF's error policy, which is not "everything".

    A malformed blob contributes nothing and the run continues -- that is
    deliberate, and the zero-triple contribution shows up in the count logged
    after loading. A MISSING PARSER IS NOT A MALFORMED BLOB. Executors run a
    venv packaged separately from this repo (``requirements-executor.txt`` via
    ``bin/package_venv.sh``), so shipping code that imports something the
    archive does not carry is a real way to be wrong -- and under a blanket
    ``except`` it is invisible: every row answers "no triples", the job
    succeeds, and the graph is empty.

    Kept out of the UDF closure so the policy can be tested without a cluster.
    """
    if not turtle_str or not turtle_str.strip():
        return []
    try:
        return turtle_to_rows(turtle_str)
    except ImportError:
        raise
    except Exception:
        return []


def turtle_to_rows(turtle_str: str) -> List[dict]:
    """One self-contained Turtle blob → the rows the UDF returns.

    Raises rather than swallowing a parse error; the UDF is what decides that a
    malformed blob contributes nothing.
    """
    from pyoxigraph import BlankNode, NamedNode, RdfFormat, parse

    # An rdflib Graph is a SET, so a blob stating the same triple twice
    # contributed it once. pyoxigraph streams the document and would emit both,
    # which would move every triple count the pipeline reports.
    seen = set()
    triples = []
    for triple in parse(turtle_str, format=RdfFormat.TURTLE):
        if triple in seen:
            continue
        seen.add(triple)
        triples.append(triple)

    labels = _blank_node_labels(triples)

    rows = []
    for triple in triples:
        subject = triple.subject
        if isinstance(subject, NamedNode):
            subj = subject.value
        elif isinstance(subject, BlankNode):
            subj = f"_:{labels[subject.value]}"
        else:
            # Not reachable from well-formed Turtle; the rdflib version skipped
            # these rather than guessing, so this one does too.
            continue

        obj_term = triple.object
        if isinstance(obj_term, NamedNode):
            obj, datatype = obj_term.value, ""
        elif isinstance(obj_term, BlankNode):
            obj, datatype = f"_:{labels[obj_term.value]}", ""
        else:
            obj, datatype = _literal_columns(obj_term)

        rows.append(
            {
                "subject": subj,
                "predicate": triple.predicate.value,
                "object": obj,
                "object_datatype": datatype,
            }
        )

    return rows


# The four columns a parsed triple contributes, in the order the batches carry
# them. object_datatype is dropped downstream by load_source_triples once the
# marker triples are built; see the note at load_turtle_parquet_to_dataframe.
TRIPLE_FIELD_NAMES = ("subject", "predicate", "object", "object_datatype")

# Triples buffered before a batch is handed back. Only the fallback --
# load_turtle_parquet_to_dataframe reads spark.sql.execution.arrow.maxRecordsPerBatch
# and passes that instead.
PARSE_BATCH_ROWS = 10000


def turtle_batches_to_arrow(batches, max_rows: int = PARSE_BATCH_ROWS):
    """Arrow batches of Turtle blobs -> Arrow batches of triples, bounded.

    WHY THIS IS NOT A UDF RETURNING AN ARRAY
    ----------------------------------------
    It was one, and it deadlocked the cluster. ``@F.udf(returnType=ArrayType(...))``
    has to hand back every triple from a blob as ONE value, and nothing bounds
    that value. Most blobs are tiny -- bls, market and noaa are all 1-4 KB and a
    few dozen triples -- but one SEC filing in the 2026-09 data is 5.6 MB and
    parses into 57,350 triples, which is 14.8 MB pickled across the socket in a
    single piece.

    When that lands, the executor's writer thread blocks pushing more input into
    a socket that is already full while the task thread blocks waiting for output
    that cannot fit. Both wait forever. Nothing raises, no executor is lost, the
    stage sits one task short of done and both nodes go to ~99% idle until the
    job's own cap kills it. That cost seven runs and a day on 2026-09-06 before a
    thread dump showed the two blocked threads. See #380.

    Yielding bounded batches bounds the value: nothing crossing the boundary
    exceeds ``max_rows`` triples, whatever any blob does. A bigger outlier
    tomorrow costs more batches.

    THAT IS NECESSARY BUT NOT SUFFICIENT -- it does not on its own stop the
    deadlock, and an earlier version of this docstring claimed it did. Run
    20260906T233804Z stalled the same way with this function in place: bounding
    each value does not bound the total bytes in flight, and mapInArrow still
    streams a whole partition in while Python streams several times as many bytes
    of triples back. When both directions fill, both threads block. That made the
    stall rare rather than gone -- one run finished clean and the next stalled on
    identical code.

    What actually closes it is keeping this operator off the GPU:
    ``spark.rapids.sql.exec.PythonMapInArrowExec=false``, set by
    bin/submit_spark_job.sh. Both blocked frames are cudf native calls that exist
    only in the RAPIDS runner, so on Spark's stock ArrowPythonRunner they cannot
    occur. Keep both fixes; they are not alternatives.

    Args:
        batches: iterator of ``pyarrow.RecordBatch``, each carrying the Turtle
                 blob as its first (and only) column.
        max_rows: triples to buffer before yielding a batch.

    Yields:
        ``pyarrow.RecordBatch`` with the four string columns of
        TRIPLE_FIELD_NAMES. Nothing is yielded for an input that produces no
        triples, which is what an empty partition and a partition of malformed
        blobs both look like.
    """
    import pyarrow as pa

    columns = ([], [], [], [])

    def drain():
        batch = pa.RecordBatch.from_arrays(
            [pa.array(values, type=pa.string()) for values in columns],
            names=list(TRIPLE_FIELD_NAMES),
        )
        for values in columns:
            values.clear()
        return batch

    for batch in batches:
        # turtle_df selects the blob column and nothing else, so it is column 0.
        # Reading by index keeps the source's column name out of the closure --
        # sources disagree on the name and it is resolved per source.
        for blob in batch.column(0).to_pylist():
            for row in turtle_rows_or_skip(blob):
                for values, field in zip(columns, TRIPLE_FIELD_NAMES):
                    values.append(row[field])
                if len(columns[0]) >= max_rows:
                    yield drain()

    if columns[0]:
        yield drain()
