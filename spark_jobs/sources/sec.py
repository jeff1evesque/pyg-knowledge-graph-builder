"""SEC: the filings feed and the companyfacts snapshot, the SEC feeds that
carry RDF."""
from typing import List, Optional

from spark_jobs.sources.spec import Feed, SourceSpec
from spark_jobs.utils.namespaces import (
    IDENTIFIER_BASE,
    SEC_COMMON,
    SEC_COMPANYFACTS,
    SEC_ENRICHMENT,
    SEC_FILINGS,
)

# ============================================
# Which SEC feeds this job handles
# ============================================
# The archive holds ten SEC feeds under raw/source=sec/. This pipeline
# handles two of them, and the restriction has until now been
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
#   feed=companyfacts_snapshot  one object a day        RDF (rdf_turtle column)
#   feed=companyfacts           one object a year       XBRL history, no Turtle
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
# Each company's latest XBRL numbers, one RDF row per fact, written daily.
SEC_COMPANYFACTS_FEED = "feed=companyfacts_snapshot"
SEC_HANDLED_FEEDS = (SEC_HANDLED_FEED, SEC_COMPANYFACTS_FEED)
# filings_documents starts with the handled feed's name, and companyfacts is
# the start of the snapshot's, so feeds are matched as whole path segments.
# companyfacts is the snapshot's history and carries no RDF. It is read for the
# companyfacts/ query table instead (graph/companyfacts.py), never as a source.
SEC_UNHANDLED_FEEDS = (
    "feed=filings_documents", "feed=filing-detail", "feed=litigation",
    "feed=press-release", "feed=speeches", "feed=statements",
    "feed=testimony", "feed=companyfacts",
)


def _feed_of(path: str) -> Optional[str]:
    for segment in path.replace("\\", "/").split("/"):
        if segment.startswith("feed="):
            return segment
    return None


def assert_sec_paths_name_the_handled_feed(source_paths: List[str]) -> None:
    """Every SEC source path must name a handled feed explicitly.

    SEC's path check. JobConfig runs it on the paths matched to SEC, before
    Spark starts.

    Raises:
        ValueError: if a path under the SEC source partition names a feed this
            pipeline does not handle, or names no feed at all.
    """
    handled = " or ".join(repr(feed) for feed in SEC_HANDLED_FEEDS)
    for path in source_paths:
        if SEC_SOURCE_PARTITION not in path:
            continue
        feed = _feed_of(path)
        if feed is None:
            raise ValueError(
                f"SEC source path does not name a feed: {path}. This pipeline "
                f"handles {handled} only, and the SEC source partition holds "
                f"other feeds it cannot read. Point at "
                f"raw/{SEC_SOURCE_PARTITION}/{SEC_HANDLED_FEED}/... instead."
            )
        if feed not in SEC_HANDLED_FEEDS:
            raise ValueError(
                f"source path names an unhandled SEC feed {feed!r}: {path}. "
                f"Only {handled} carry RDF; the others hold crawler telemetry, "
                f"raw XML or untyped history and no pipeline step reads them."
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
        (str(SEC_COMPANYFACTS), "companyfacts"),
    ),
    enrichment_namespace=str(SEC_ENRICHMENT),
    date_predicates=(
        str(SEC_FILINGS.hasPeriodOfReport),
        str(SEC_FILINGS.hasFilingDate),
        str(SEC_COMPANYFACTS.periodEnd),
        str(SEC_COMPANYFACTS.filedOn),
    ),
    temporal_prefix=f"{IDENTIFIER_BASE}temporal/sec/",
    check_paths=assert_sec_paths_name_the_handled_feed,
    canonicalize=_canonicalize,
    linker=_linker,
    entity_namespaces=(str(SEC_FILINGS), str(SEC_COMPANYFACTS)),
    # Left out of the sector keyword step: a filing's local name is an
    # accession number, and its company's sector comes from its SIC code.
    sector_keywords=False,
    company_keys=_company_keys,
    sector_keys=_sector_keys,
    feeds=(
        Feed(
            name="sec-companyfacts",
            path_fragment=SEC_COMPANYFACTS_FEED,
            namespaces=(str(SEC_COMPANYFACTS),),
        ),
    ),
)
