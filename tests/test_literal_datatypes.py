"""The XSD datatype the source declares on a literal must survive the parse.

Both loaders used to throw it away — the N-Triples regex captured the lexical
form inside the quotes and stopped, and the turtle-parquet UDF called
``Literal.toPython()``, which is precisely where ``^^<xsd:decimal>`` stops
existing. Nothing downstream could recover it, so ``rdfs:range`` was underivable
for every literal-valued property and ``domain_range`` (111 dims of every 1024-d
node vector) had nothing to encode.

The datatype is not inferred here — it is *read*. ``market.nt`` alone declares
60 ``xsd:decimal``, 17 ``xsd:long``, 14 ``xsd:string`` and 2 ``xsd:boolean``;
``noaa.nt`` adds ``xsd:dateTime`` and ``xsd:anyURI``. What the loaders emit is
an observation (``prov:observedLiteralDatatype``), not an ``rdfs:range``
assertion: OntologyMapper decides whether the observation supports an axiom,
and ontology_schema.json publishes which of the two a consumer is looking at.

Deduplicated at the source, so the marker count is bounded by distinct
(predicate, datatype) pairs — never by the triple count.
"""
from pathlib import Path

import pytest

from spark_jobs.utils.rdf_utils import PROV_OBSERVED_LITERAL_DATATYPE
from spark_jobs.utils.spark_rdf_utils import load_ntriples_to_triples_df

XSD = "http://www.w3.org/2001/XMLSchema#"
PRED = "https://ex/value"
SUBJ = "https://ex/a"

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "e2e" / "ntriples"


def _write_nt(tmp_path, lines):
    path = tmp_path / "sample.nt"
    path.write_text("\n".join(lines) + "\n")
    return str(tmp_path)


def _markers(df):
    return {
        (r["subject"], r["object"])
        for r in df.filter(
            df.predicate == PROV_OBSERVED_LITERAL_DATATYPE
        ).collect()
    }


def _data(df):
    return {
        (r["subject"], r["predicate"], r["object"])
        for r in df.filter(
            df.predicate != PROV_OBSERVED_LITERAL_DATATYPE
        ).collect()
    }


# ======================================================================
# N-Triples loader
# ======================================================================

def test_declared_datatype_becomes_an_observation(spark, tmp_path):
    src = _write_nt(tmp_path, [
        f'<{SUBJ}> <{PRED}> "1.5"^^<{XSD}decimal> .',
    ])
    df = load_ntriples_to_triples_df(spark, [src])

    assert _markers(df) == {(PRED, f"{XSD}decimal")}
    # ...and the triple itself is unchanged: the object is still the lexical
    # form the rest of the pipeline parses as a number.
    assert _data(df) == {(SUBJ, PRED, "1.5")}


def test_a_plain_literal_produces_no_observation(spark, tmp_path):
    src = _write_nt(tmp_path, [
        f'<{SUBJ}> <{PRED}> "just text" .',
    ])
    df = load_ntriples_to_triples_df(spark, [src])

    assert _markers(df) == set()
    assert _data(df) == {(SUBJ, PRED, "just text")}


def test_a_uri_object_produces_no_observation(spark, tmp_path):
    src = _write_nt(tmp_path, [
        f'<{SUBJ}> <{PRED}> <https://ex/b> .',
    ])
    assert _markers(load_ntriples_to_triples_df(spark, [src])) == set()


def test_observations_are_deduplicated(spark, tmp_path):
    """Bounded by distinct (predicate, datatype) pairs, not by triple count.

    Without the distinct, a predicate on a million rows would put a million
    identical marker triples into the graph.
    """
    src = _write_nt(tmp_path, [
        f'<https://ex/{i}> <{PRED}> "{i}.0"^^<{XSD}decimal> .'
        for i in range(50)
    ])
    df = load_ntriples_to_triples_df(spark, [src])

    assert _markers(df) == {(PRED, f"{XSD}decimal")}
    assert len(_data(df)) == 50


def test_a_predicate_with_two_datatypes_reports_both(spark, tmp_path):
    """Reported, not resolved.

    Picking one here would hide the conflict; OntologyMapper is where the
    decision belongs, and it declines to assert a range at all.
    """
    src = _write_nt(tmp_path, [
        f'<{SUBJ}> <{PRED}> "1.5"^^<{XSD}decimal> .',
        f'<https://ex/b> <{PRED}> "text"^^<{XSD}string> .',
    ])
    assert _markers(load_ntriples_to_triples_df(spark, [src])) == {
        (PRED, f"{XSD}decimal"),
        (PRED, f"{XSD}string"),
    }


def test_the_committed_fixtures_declare_datatypes(spark):
    """The signal is really in the data, not only in synthetic lines above.

    If a fixture refresh ever dropped the datatypes, range derivation would go
    quiet and only the coverage numbers would show it.
    """
    df = load_ntriples_to_triples_df(spark, [str(FIXTURES)])
    datatypes = {dt for _pred, dt in _markers(df)}

    assert f"{XSD}decimal" in datatypes
    assert f"{XSD}dateTime" in datatypes
    assert len(datatypes) >= 4, sorted(datatypes)


# ======================================================================
# turtle-parquet loader — the path the BLS fixtures use
# ======================================================================

def test_turtle_parquet_preserves_datatypes(spark, tmp_path):
    """`Literal.toPython()` is where the declaration used to be lost."""
    pytest.importorskip("rdflib")
    from spark_jobs.build_graph import load_turtle_parquet_to_dataframe

    turtle = (
        f'<{SUBJ}> <{PRED}> "1.5"^^<{XSD}decimal> ;\n'
        f'  <https://ex/name> "alpha" .\n'
    )
    pq = tmp_path / "src"
    spark.createDataFrame(
        [(turtle,)], schema="triples STRING"
    ).write.parquet(str(pq))

    df = load_turtle_parquet_to_dataframe(spark, str(pq))

    assert (PRED, f"{XSD}decimal") in _markers(df)
    # rdflib types a bare Turtle string as xsd:string, so <name> gets an
    # observation too — correct, and it is what makes string-valued
    # properties rangeable at all.
    assert (SUBJ, PRED, "1.5") in _data(df)


# ======================================================================
# partition count — issue #344
# ======================================================================

def test_marker_frame_lands_on_one_partition(spark):
    """The marker frame must not carry a shuffle's worth of partitions.

    ``distinct()`` lands on ``spark.sql.shuffle.partitions`` no matter how few
    rows survive it, and every loader unions this frame into the triples for
    its source. A union's partition count is the sum of its children's, so
    before this was coalesced each source added 200 partitions of a
    few-hundred-row frame to the cached triples, and each of the ~501
    enrichment stages that read that cache inherited every one of them.

    AQE hides this locally -- it coalesces the tiny exchange away -- so the
    assertion is made with AQE off, which is the shape the cluster runs in.
    """
    from spark_jobs.utils.spark_rdf_utils import literal_datatype_observations

    parsed = spark.createDataFrame(
        [(f"{SUBJ}{i}", PRED, "1.5", f"{XSD}decimal") for i in range(50)],
        schema="subject STRING, predicate STRING, object STRING, "
               "object_datatype STRING",
    ).repartition(8)

    previous = spark.conf.get("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    try:
        markers = literal_datatype_observations(parsed)
        assert markers.rdd.getNumPartitions() == 1
    finally:
        spark.conf.set("spark.sql.adaptive.enabled", previous)

    # ...and the rows it carries are unchanged by the coalesce.
    assert markers.count() == 1
