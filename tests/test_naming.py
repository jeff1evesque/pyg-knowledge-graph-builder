"""The two halves of the naming rule have to agree.

``prefixed_local_name_expr`` names node types and relations on the executors;
``prefixed_local_name`` names a handful of COLUMNS on the driver. A column
named by one and filled by the other is exactly the shape of the query tables'
wide market table, so a disagreement there is a column of nulls rather than an
error.
"""
import pytest

from spark_jobs.pyg_builder.naming import (
    prefixed_local_name,
    prefixed_local_name_expr,
    relation_to_predicate_uri,
)
from spark_jobs.utils.rdf_utils import NAMESPACE_PREFIXES


def test_the_driver_and_spark_naming_rules_agree(spark):
    """prefixed_local_name is the driver-side twin of the Spark expression, and
    a column named by one is filled by the other. They are separate code, so
    this runs both over every registered namespace and over the shapes the
    fallback handles."""
    uris = [f"{namespace}Thing" for namespace, _p in NAMESPACE_PREFIXES]
    uris += [
        "https://nobody.example/vocab/Widget",
        "https://nobody.example/vocab#Widget",
    ]

    frame = spark.createDataFrame([(uri,) for uri in uris], "uri string")
    from_spark = {
        row["uri"]: row["name"]
        for row in frame.select(
            "uri", prefixed_local_name_expr("uri").alias("name")
        ).collect()
    }

    assert {uri: prefixed_local_name(uri) for uri in uris} == from_spark


def test_a_uri_with_no_local_name_is_refused_rather_than_guessed():
    """The expression falls back to a Spark hash there, which nothing in Python
    reproduces. Saying so beats returning a name that does not match."""
    with pytest.raises(ValueError, match="no local name"):
        prefixed_local_name("urn:isbn:0451450523")



def test_a_relation_name_inverts_back_to_its_predicate():
    """The other direction, which only the driver ever needs: a resolved edge
    carries the relation name and not the predicate it came from, and
    edge_types/ has to publish the predicate."""
    for namespace, _prefix in NAMESPACE_PREFIXES:
        uri = f"{namespace}Thing"
        assert relation_to_predicate_uri(prefixed_local_name(uri)) == uri
