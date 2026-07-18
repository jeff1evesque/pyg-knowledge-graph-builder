"""Blank-node labelling must be deterministic across parses.

rdflib mints a fresh random identifier for every blank node on every parse. Those
labels flow into the triples as subject/object values and are hashed into feature
vector slots downstream, so random labels made the whole pipeline
non-reproducible: two runs over identical input produced different feature
tensors (observed on ``jolts_*`` node types, which model measurements as blank
nodes) and reordered entries in the metadata JSON.

These are pure-Python tests over rdflib — no SparkSession — so they run in
milliseconds, unlike the e2e reproducibility guard that takes ~100s to catch the
same class of bug.
"""
import pytest

from spark_jobs.build_graph import deterministic_bnode_labels

rdflib = pytest.importorskip("rdflib")


# A measurement modelled as a blank node, mirroring the JOLTS shape that
# exposed the bug: a named subject points at an anonymous node carrying values.
TURTLE = """
@prefix ex: <https://example.org/> .

ex:Series_A ex:hasHires [
    ex:hasValue 890.0 ;
    ex:hasUnit "thousands"
] .

ex:Series_B ex:hasQuits [
    ex:hasValue 1.2 ;
    ex:hasUnit "percent"
] .
"""


def _parse(turtle):
    g = rdflib.Graph()
    g.parse(data=turtle, format="turtle")
    return g


def _labelled_triples(turtle):
    """Triples with blank nodes replaced by their content-derived labels."""
    g = _parse(turtle)
    labels = deterministic_bnode_labels(g)

    def term(t):
        return f"_:{labels[t]}" if isinstance(t, rdflib.BNode) else str(t)

    return sorted((term(s), str(p), term(o)) for s, p, o in g)


def test_rdflib_labels_are_random_across_parses():
    """Guard the premise: without relabelling, two parses genuinely differ.

    If rdflib ever became deterministic on its own this would fail, signalling
    that deterministic_bnode_labels is no longer load-bearing.
    """
    raw = [
        sorted(str(t) for tr in _parse(TURTLE) for t in (tr[0], tr[2])
               if isinstance(t, rdflib.BNode))
        for _ in range(2)
    ]
    assert raw[0] != raw[1], "rdflib blank-node labels were stable — unexpected"


def test_labelled_triples_are_stable_across_parses():
    """The same Turtle yields byte-identical labelled triples every parse."""
    assert _labelled_triples(TURTLE) == _labelled_triples(TURTLE)


def test_labels_survive_reordered_input():
    """Serialization order must not change the labels.

    Content-derived labels should depend on graph structure, not on the order
    statements happen to appear in the document.
    """
    reordered = """
@prefix ex: <https://example.org/> .

ex:Series_B ex:hasQuits [
    ex:hasUnit "percent" ;
    ex:hasValue 1.2
] .

ex:Series_A ex:hasHires [
    ex:hasUnit "thousands" ;
    ex:hasValue 890.0
] .
"""
    assert _labelled_triples(TURTLE) == _labelled_triples(reordered)


def test_distinct_blank_nodes_are_not_merged():
    """Structurally identical blank nodes must stay distinct.

    Two measurements with the same values under the same predicate hash to the
    same signature. Collapsing them would silently drop a measurement, so the
    disambiguating suffix must keep them separate.
    """
    turtle = """
@prefix ex: <https://example.org/> .

ex:Series ex:hasReading [ ex:hasValue 5.0 ] .
ex:Series ex:hasReading [ ex:hasValue 5.0 ] .
"""
    g = _parse(turtle)
    bnodes = {t for tr in g for t in (tr[0], tr[2])
              if isinstance(t, rdflib.BNode)}
    labels = deterministic_bnode_labels(g)

    assert len(bnodes) == 2, "fixture should contain two distinct blank nodes"
    assert len(set(labels.values())) == 2, "distinct blank nodes were merged"


def test_no_blank_nodes_is_empty_mapping():
    """A graph without blank nodes needs no relabelling."""
    g = _parse(
        '@prefix ex: <https://example.org/> .\nex:A ex:p "literal" .\n'
    )
    assert deterministic_bnode_labels(g) == {}
