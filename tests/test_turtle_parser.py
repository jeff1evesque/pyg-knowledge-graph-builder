"""The parser changed; what the rows say must not.

#376 replaced rdflib's Turtle parser with pyoxigraph, because profiling the UDF
on 13,024 real blobs put 88% of its time inside ``g.parse()``. The rows a blob
produces are hashed into feature vector slots, so a literal spelled one
character differently is a different graph -- not a cosmetic difference.

So the contract these tests assert is not "the new parser is reasonable" but
"the new parser says exactly what rdflib said". ``rdflib_rows`` below is the
implementation that was replaced, kept here as the thing to agree with. Two
cases found real bugs while this was being written and are pinned individually
below: booleans and ``xsd:dateTime``.

Pure Python, no SparkSession -- this is the fast tier.
"""
import pytest

from spark_jobs.build_graph import (
    _lexical_converters,
    deterministic_bnode_labels,
    turtle_to_rows,
)

rdflib = pytest.importorskip("rdflib")
pytest.importorskip("pyoxigraph")

PREFIX = "@prefix ex: <https://example.org/> .\n@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
XSD = "http://www.w3.org/2001/XMLSchema#"


def rdflib_rows(turtle):
    """What the UDF did before #376. The reference, not a reimplementation."""
    from rdflib import BNode, Graph, Literal, URIRef

    g = Graph()
    g.parse(data=turtle, format="turtle")
    labels = deterministic_bnode_labels(g)

    rows = []
    for s, p, o in g:
        if isinstance(s, URIRef):
            subj = str(s)
        elif isinstance(s, BNode):
            subj = f"_:{labels[s]}"
        else:
            continue
        datatype = ""
        if isinstance(o, URIRef):
            obj = str(o)
        elif isinstance(o, BNode):
            obj = f"_:{labels[o]}"
        elif isinstance(o, Literal):
            obj = str(o.toPython())
            if o.datatype is not None:
                datatype = str(o.datatype)
        else:
            obj = str(o)
        rows.append((subj, str(p), obj, datatype))
    return sorted(rows)


def parsed_rows(turtle):
    return sorted(
        (r["subject"], r["predicate"], r["object"], r["object_datatype"])
        for r in turtle_to_rows(turtle)
    )


def one_object(turtle):
    rows = parsed_rows(turtle)
    assert len(rows) == 1, rows
    return rows[0][2], rows[0][3]


@pytest.mark.parametrize(
    "literal",
    [
        "42",
        '"42"^^xsd:integer',
        '"007"^^xsd:integer',
        "3.14",
        '"1.50"^^xsd:decimal',
        "1.0e2",
        '"1.0E2"^^xsd:double',
        "true",
        "false",
        '"2026-09-01T19:59:00-04:00"^^xsd:dateTime',
        '"2026-09-01"^^xsd:date',
        '"19:59:00"^^xsd:time',
        '"typed string"^^xsd:string',
        '"tagged"@en',
        '"abc"^^xsd:integer',
        '"whatever"^^ex:CustomType',
        "<https://example.org/o>",
    ],
)
def test_an_object_is_spelled_the_way_rdflib_spells_it(literal):
    turtle = f"{PREFIX}ex:s ex:p {literal} ."
    assert parsed_rows(turtle) == rdflib_rows(turtle)


def test_a_boolean_keeps_its_python_capital():
    """The bug the lexical table was re-keyed to fix.

    rdflib's table is keyed by URIRef, whose __hash__ is not str's, so looking
    it up with a plain string misses silently and the lexical form survives --
    "false" where rdflib says "False". Every market snapshot carries booleans.
    """
    assert one_object(f"{PREFIX}ex:s ex:p false .") == ("False", f"{XSD}boolean")
    assert one_object(f"{PREFIX}ex:s ex:p true .") == ("True", f"{XSD}boolean")


def test_a_datetime_is_rendered_with_a_space_not_a_t():
    """The other one. str(datetime) does not put back the T it parsed.

    Caught on NOAA, where every alert carries an effective time; a hand-written
    conversion table got this wrong on all 128 sampled blobs.
    """
    turtle = f'{PREFIX}ex:s ex:p "2026-09-01T19:59:00-04:00"^^xsd:dateTime .'
    assert one_object(turtle) == (
        "2026-09-01 19:59:00-04:00",
        f"{XSD}dateTime",
    )


def test_a_bare_literal_now_reports_xsd_string_and_that_is_deliberate():
    """The one place the new parser does NOT agree with rdflib.

    RDF 1.1 makes a plain literal and an ^^xsd:string literal the same term, so
    pyoxigraph reports xsd:string for both; rdflib kept the older distinction
    and reports a datatype only for the second. Following rdflib here would
    mean erasing xsd:string, and every string literal this pipeline actually
    reads is written ^^xsd:string in the source text -- measured 2026-09-05,
    3,461 of 3,461 on SEC and 1,744 of 1,744 on NOAA -- so erasing it would
    strip a real observation off each one. Reporting it changes nothing in the
    corpus and changes a bare literal, which the corpus does not contain.
    """
    assert one_object(f'{PREFIX}ex:s ex:p "plain string" .') == (
        "plain string",
        f"{XSD}string",
    )
    assert one_object(f'{PREFIX}ex:s ex:p "plain string"^^xsd:string .') == (
        "plain string",
        f"{XSD}string",
    )


def test_a_language_tag_leaves_the_datatype_column_empty():
    """pyoxigraph reports rdf:langString; rdflib reports no datatype at all.

    The column has always followed rdflib, and it feeds the observation markers
    that stand in for rdfs:range -- a language tag is not a range.
    """
    assert one_object(f'{PREFIX}ex:s ex:p "tagged"@en .') == ("tagged", "")


def test_an_ill_typed_literal_keeps_its_lexical_form():
    """rdflib's cast swallows the error and leaves the value as written."""
    turtle = f'{PREFIX}ex:s ex:p "abc"^^xsd:integer .'
    assert one_object(turtle) == ("abc", f"{XSD}integer")
    assert parsed_rows(turtle) == rdflib_rows(turtle)


def test_a_datatype_with_no_converter_keeps_its_lexical_form():
    turtle = f'{PREFIX}ex:s ex:p "whatever"^^ex:CustomType .'
    assert one_object(turtle) == ("whatever", "https://example.org/CustomType")


def test_the_lexical_table_is_reachable_by_plain_string():
    """Guards the re-keying directly, not through a literal.

    If rdflib ever changes how that table is keyed, this fails with a clear
    cause instead of a graph that is quietly one character different.
    """
    converters = _lexical_converters()
    for datatype in ("boolean", "dateTime", "integer", "double", "date"):
        assert f"{XSD}{datatype}" in converters, datatype
    assert all(isinstance(k, str) for k in converters)


# The JOLTS shape that motivated deterministic labels in the first place: a
# named subject pointing at an anonymous node that carries the values. The unit
# is written ^^xsd:string because that is how the real sources write one.
BNODE_TURTLE = """
@prefix ex: <https://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Series_A ex:hasHires [
    ex:hasValue 890.0 ;
    ex:hasUnit "thousands"^^xsd:string
] .
"""


def test_blank_nodes_get_the_same_content_labels_rdflib_got():
    assert parsed_rows(BNODE_TURTLE) == rdflib_rows(BNODE_TURTLE)


def test_blank_node_labels_are_stable_across_parses():
    """The reproducibility guarantee, now through the new parser.

    pyoxigraph mints its own per-parse blank-node ids, so this fails the moment
    they reach the rows instead of the content-derived labels.
    """
    assert parsed_rows(BNODE_TURTLE) == parsed_rows(BNODE_TURTLE)
    for row in parsed_rows(BNODE_TURTLE):
        for term in (row[0], row[2]):
            if term.startswith("_:"):
                assert len(term) == len("_:") + 32, term


def test_a_triple_stated_twice_is_emitted_once():
    """An rdflib Graph is a set; pyoxigraph streams the document.

    Without the dedupe every repeated statement would inflate the triple counts
    the pipeline reports, which is how a parser swap turns into a data change.
    """
    turtle = f"{PREFIX}ex:s ex:p ex:o .\nex:s ex:p ex:o .\n"
    assert len(turtle_to_rows(turtle)) == 1
    assert parsed_rows(turtle) == rdflib_rows(turtle)


def test_predicate_and_object_lists_expand_the_same_way():
    turtle = f"{PREFIX}ex:s ex:p ex:o1, ex:o2 ; ex:q 1, 2 ."
    assert parsed_rows(turtle) == rdflib_rows(turtle)
    assert len(parsed_rows(turtle)) == 4


def test_a_malformed_blob_is_skipped_but_a_missing_parser_is_not(monkeypatch):
    """The two failures a blanket `except` cannot tell apart.

    Executors run a venv packaged from requirements-executor.txt, not this
    checkout, so a parser that never reached them is a real possibility -- and
    swallowing its ImportError would answer "no triples" for every row, finish
    green, and write an empty graph.
    """
    from spark_jobs import build_graph

    assert build_graph.turtle_rows_or_skip("") == []
    assert build_graph.turtle_rows_or_skip("this is not turtle {") == []

    def missing(_):
        raise ImportError("No module named 'pyoxigraph'")

    monkeypatch.setattr(build_graph, "turtle_to_rows", missing)
    with pytest.raises(ImportError):
        build_graph.turtle_rows_or_skip(f"{PREFIX}ex:s ex:p ex:o .")


def test_malformed_turtle_raises_so_the_udf_can_skip_the_row():
    """turtle_to_rows does not decide policy; the UDF's except does.

    pyoxigraph parses lazily, so this also pins that the rows are materialised
    inside the call -- a generator handed back unevaluated would raise later,
    outside the UDF's try, and fail the job instead of skipping the blob.
    """
    with pytest.raises(Exception):
        turtle_to_rows("@prefix ex: <https://example.org/> .\nex:s ex:p")
