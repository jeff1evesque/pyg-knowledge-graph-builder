"""SEC: the filings feed, the one SEC feed that carries RDF."""
from typing import List

from spark_jobs.sources.spec import SourceSpec
from spark_jobs.utils.namespaces import (
    IDENTIFIER_BASE,
    SEC_COMMON,
    SEC_ENRICHMENT,
    SEC_FILINGS,
)

# ============================================
# Which SEC feed this job handles
# ============================================
# The archive holds eight SEC feeds under raw/source=sec/. This pipeline
# handles exactly one of them, and the restriction has until now been
# incidental — a property of whichever prefix the caller happened to pass —
# rather than stated anywhere. Measured against the archive:
#
#   feed=filings             218 objects,   2.4 GB   RDF (rdf_turtle column)
#   feed=filings_documents   712,351 objects, 150 GB  raw filing documents
#   feed=filing-detail       364 objects            crawler telemetry columns
#   feed=litigation            1 object              only, no Turtle column
#   feed=press-release         2 objects             (parsed / parse-start /
#   feed=speeches              2 objects              failures / user-agent)
#   feed=statements            2 objects
#   feed=testimony             1 object
#
# filings_documents is not Parquet at all and not one format either: sampled
# over 400,000 keys it is 233,759 .xml, 160,196 .zip (upstream now packages
# each filing's documents together with its XBRL members), plus .txt, .htm,
# .pdf and images. Nothing there is RDF.
#
# So a run pointed at the SEC source root does not under-cover quietly — the
# six telemetry feeds have no Turtle column at all and resolve_turtle_column
# raises, while filings_documents is not something the Parquet reader can open.
# It fails, but it fails deep in the loader with a column-name error that says
# nothing about feeds. This turns that into a statement of scope at the point
# the job is configured.
#
# The one thing NOT guarded here, because storage says it is already resolved:
# the retired crawler also wrote into feed=filings itself, on a 10-column
# schema whose RDF column was named `triples` rather than `rdf_turtle`. All 218
# objects now carry the same 29-column upstream schema, so the migration ran to
# completion and no mixed-schema read is possible. Worth re-checking if that
# object count ever jumps backwards.
#
# Keyed on the partition name rather than on "sec" anywhere in the path: a
# local fixture directory called /data/sec/ is not the archive convention and
# is none of this check's business.
SEC_SOURCE_PARTITION = "source=sec"
SEC_HANDLED_FEED = "feed=filings"
# filings_documents starts with the handled feed's name, so a plain substring
# test would accept it. It is a different feed and carries no RDF.
SEC_UNHANDLED_FEEDS = (
    "feed=filings_documents", "feed=filing-detail", "feed=litigation",
    "feed=press-release", "feed=speeches", "feed=statements",
    "feed=testimony",
)


def assert_sec_paths_name_the_handled_feed(source_paths: List[str]) -> None:
    """Every SEC source path must name feed=filings explicitly.

    SEC's path check. JobConfig runs it on the paths matched to SEC, before
    Spark starts.

    Raises:
        ValueError: if a path under the SEC source partition names a feed this
            pipeline does not handle, or names no feed at all.
    """
    for path in source_paths:
        if SEC_SOURCE_PARTITION not in path:
            continue
        unhandled = [f for f in SEC_UNHANDLED_FEEDS if f in path]
        if unhandled:
            raise ValueError(
                f"source path names an unhandled SEC feed {unhandled[0]!r}: "
                f"{path}. Only {SEC_HANDLED_FEED!r} carries RDF; the others "
                f"hold crawler telemetry or raw XML and no pipeline step reads "
                f"them."
            )
        if SEC_HANDLED_FEED not in path:
            raise ValueError(
                f"SEC source path does not name a feed: {path}. This pipeline "
                f"handles {SEC_HANDLED_FEED!r} only, and the SEC source "
                f"partition holds seven other feeds it cannot read. Point at "
                f"raw/{SEC_SOURCE_PARTITION}/{SEC_HANDLED_FEED}/... instead."
            )


def _canonicalize(triples_df):
    from spark_jobs.utils.sec_identifiers import canonicalize_sec_identifiers

    return canonicalize_sec_identifiers(triples_df)


def _linker(spark, options):
    from spark_jobs.enrichment.intra_source.sec_linker import SECIntraSourceLinker

    return SECIntraSourceLinker(spark)


def _company_keys(context):
    from spark_jobs.enrichment.intra_source.sec.cross_source import company_keys

    return company_keys(context)


def _sector_keys(context):
    from spark_jobs.enrichment.intra_source.sec.cross_source import sector_keys

    return sector_keys(context)


SPEC = SourceSpec(
    name="sec",
    label="SEC",
    path_fragments=("source=sec",),
    namespaces=(
        (str(SEC_FILINGS), "filings"),
        (str(SEC_COMMON), "sec_common"),
        (str(SEC_ENRICHMENT), "sec_enrichment"),
    ),
    enrichment_namespace=str(SEC_ENRICHMENT),
    date_predicates=(
        str(SEC_FILINGS.hasPeriodOfReport),
        str(SEC_FILINGS.hasFilingDate),
    ),
    temporal_prefix=f"{IDENTIFIER_BASE}temporal/sec/",
    check_paths=assert_sec_paths_name_the_handled_feed,
    canonicalize=_canonicalize,
    linker=_linker,
    entity_namespaces=(str(SEC_FILINGS),),
    # Left out of the sector keyword step: a filing's local name is an
    # accession number, and its company's sector comes from its SIC code.
    sector_keywords=False,
    company_keys=_company_keys,
    sector_keys=_sector_keys,
)
