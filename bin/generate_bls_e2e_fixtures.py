#!/usr/bin/env python3
"""
Regenerate the BLS end-to-end fixtures under
``tests/fixtures/e2e/turtle_parquet/bls/`` from the raw crawl output in the
project's S3 archive bucket.

WHY THIS EXISTS
---------------
The previous fixtures (committed as opaque binaries in 6317dd0 / #184, with no
generator) were sampled by *triple*, not by *entity*. Each subject retained an
average of 1.46 of its triples, so only ~15% happened to keep their ``rdf:type``
and become graph nodes at all; the rest were fragments, and their blank-node
measurement objects were emptied of values. The e2e graph was therefore
structurally unrepresentative — most edges could not resolve because most
endpoints were not nodes.

This script samples at the *entity* level instead: it selects whole measurement
subjects and then takes the transitive closure of everything they reference, so
every subject that appears in the output carries its full triple set. Upstream
is 99.97% typed, so closure-complete sampling inherits that property.

Whole-*row* sampling (a crawl row is already a self-consistent graph) was the
first choice and is not viable on size: the smallest single Turtle document is
612 KB against a ~180 KB budget for the entire fixture set. Hence entity-level
closure within rows.

WHAT UPSTREAM CHANGED
---------------------
Three things this script had encoded and now reads instead:

  * the RDF column is ``rdf_turtle``. It was ``triples`` when this was written;
    both spellings exist across sources, so the column is now *detected* by the
    same candidate list the production loader uses
    (``build_graph.TURTLE_COLUMN_CANDIDATES``) rather than named.

  * a row is filed under the year it is *about*, not the year it was crawled,
    and a revision replaces the row it revises. ``<feed>/<year>.snappy.parquet``
    is therefore a whole observation year, and the newest vintage is the current
    (partial) year — which is what the sample wants anyway.

  * measurements are flat. Values used to hang off a blank node
    (``[ ns1:value 1.2 ]``); they are now literals on the measurement subject
    itself, and no BLS feed emits a blank node at all. The validation below
    keeps the property that mattered — a sampled measurement carries its value —
    without requiring the blank nodes that used to carry it.

The ontology moved with it (jolts ``LayoffsDischargesLevel`` →
``LayoffsAndDischargesLevel``, cpi's per-item subclasses collapsed to a
``Category`` entity, eci keyed on occupation and industry, ...), which is why
regenerating is not optional: fixtures on the old vocabulary exercise class
names the pipeline will never see again.

NETWORK BOUNDARY
----------------
S3 access is **generation-time only**. The fixtures stay committed Parquet and
the e2e suite continues to read local files and touch no network. Preserve that
property: do not import this module from the test suite.

USAGE
-----
    export BLS_FIXTURE_BUCKET=<archive-bucket>      # or pass --bucket
    .venv/bin/python bin/generate_bls_e2e_fixtures.py

    # preview without writing
    .venv/bin/python bin/generate_bls_e2e_fixtures.py --dry-run

Requires credentials with read access to the archive bucket. Output is
deterministic: the same source objects produce byte-identical fixtures.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

RAW_PREFIX = "raw/source=bls/"

# The four feeds the e2e fixtures cover, matching the committed directory
# layout. Upstream has twelve feeds; these four are what the pipeline's BLS
# linker and the smoke test exercise.
FEEDS = ("cpi", "eci", "empsit", "jolts")

# Per-feed cap on sampled measurement subjects. Tuned so the total fixture set
# lands in the same size ballpark as the ~180 KB it replaces — the e2e suite is
# stage-count bound and must stay fast.
#
# Re-tuned from 60 to 200 when upstream flattened its measurements. A subject
# that used to carry a blank-node value tree is now six literals, so 60/feed
# produced 17 KiB against the 51 KiB the same setting used to produce. 200
# holds the byte budget and buys type coverage with it:
#
#   subjects/feed   measurements   node types   fixture size
#   60              240            33           17 KiB
#   200             800            55           45 KiB
#
# Still well under the 105 types the pre-refactor fixtures carried, so the
# stage count this guards is lower than it was, not higher.
SUBJECTS_PER_FEED = 200

# Fraction of each feed's quota spent anchoring on shared periods; any
# remainder goes to type coverage. See select_subjects().
#
# Defaults to 1.0 — anchoring first, type coverage only as a fallback for feeds
# with too few anchored subjects to fill the quota. Lowering it broadens the
# ontology slice but is a worse trade than it looks:
#
#   anchor_share   node types   fixture size
#   1.0            105          51 KiB
#   0.8            224          57 KiB
#   0.5            298          64 KiB
#
# The e2e suite is stage-count bound — the pipeline emits roughly one Spark job
# per node/edge type — so tripling node types inflates driver work on the exact
# axis that has OOMed before, against no measured benefit. Worse, the types the
# greedy pass adds are long-tail vocabulary terms with few incident edges, so
# broadening coverage tends to *add* orphaned node types rather than remove
# them. Entity completeness is what un-orphans the existing types; breadth is
# not. Raise the quota (--subjects-per-feed) before lowering this.
ANCHOR_SHARE = 1.0

# How many crawl rows to spread each feed's sample across. The production
# loader reads one Turtle document per row, so keeping several rows preserves
# the multi-document shape rather than collapsing to a single blob.
OUTPUT_ROWS = 3

# Depth to expand referenced URIs. Depth 1 pulls in the dimension entities a
# measurement points at (industry, month, year); beyond that the frontier is
# closed out with rdf:type triples alone so nothing lands untyped.
CLOSURE_DEPTH = 2

# Minimum number of feeds that must carry measurements in a shared period.
# Entity completeness does not imply period overlap, and without overlap the
# cross-source temporal bridge cannot form no matter how correct the pipeline
# is. This is asserted, not hoped for.
MIN_FEEDS_SHARING_PERIOD = 2

# Column names a vintage may spell its Turtle in, newest crawl schema last.
# Kept identical to build_graph.TURTLE_COLUMN_CANDIDATES: the generator must
# read exactly what the production loader reads, or a fixture can be sampled
# from a column the pipeline would ignore.
TURTLE_COLUMN_CANDIDATES = ("triples", "rdf_turtle")

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "e2e" / "turtle_parquet" / "bls"
)

# Matches both naming conventions present in the bucket. A migration renamed
# `<year>.snappy.parquet` to `year=<year>.snappy.parquet` and left the
# originals behind, so `metro` and `ppi` have duplicate keys for the same
# vintage. Parsing both into a common (feed, year) identity lets the dedupe
# below resolve them by rule rather than by hand-maintained exceptions.
_KEY_RE = re.compile(
    r"feed=(?P<feed>[^/]+)/(?:year=)?(?P<year>\d{4})\.snappy\.parquet$"
)


@dataclass(frozen=True)
class SourceObject:
    feed: str
    year: int
    key: str
    last_modified: object
    size: int


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def discover_objects(s3, bucket: str) -> list[SourceObject]:
    """Every (feed, year) vintage under the raw BLS prefix, duplicates resolved.

    Duplicate keys arise from the naming migration described on ``_KEY_RE``.
    Both spellings denote the same vintage, so they are collapsed by identity
    and the most recently modified wins — a general rule that needs no
    per-feed special-casing and keeps working if the migration continues.
    """
    found: dict[tuple[str, int], SourceObject] = {}
    dupes: list[str] = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=RAW_PREFIX):
        for item in page.get("Contents", []):
            match = _KEY_RE.search(item["Key"])
            if not match:
                continue
            obj = SourceObject(
                feed=match.group("feed"),
                year=int(match.group("year")),
                key=item["Key"],
                last_modified=item["LastModified"],
                size=item["Size"],
            )
            identity = (obj.feed, obj.year)
            existing = found.get(identity)
            if existing is None:
                found[identity] = obj
            else:
                dupes.append(obj.key)
                if obj.last_modified > existing.last_modified:
                    found[identity] = obj

    if dupes:
        _log(f"  note: {len(dupes)} duplicate vintage key(s) resolved by recency")
    return sorted(found.values(), key=lambda o: (o.feed, o.year))


def turtle_column(names) -> str | None:
    """The column a vintage spells its Turtle in, or None if it carries no RDF.

    Detected rather than named: the crawl renamed ``triples`` to ``rdf_turtle``,
    and a vintage written either side of that rename is equally valid input.
    """
    for candidate in TURTLE_COLUMN_CANDIDATES:
        if candidate in names:
            return candidate
    return None


def newest_rdf_vintage(s3, bucket: str, objects: list[SourceObject], feed: str):
    """Newest vintage for ``feed`` that actually carries RDF.

    Some vintages predate the RML mapping and use a pre-RDF crawl schema with no
    Turtle column at all. Rather than hardcoding which years those are, this
    reads the Parquet *schema* and walks backwards until it finds a vintage that
    has one — so a future no-RDF vintage is handled by the same rule.

    Returns the object, its bytes, and the column its Turtle is in.
    """
    import pyarrow.parquet as pq

    candidates = sorted(
        (o for o in objects if o.feed == feed),
        key=lambda o: o.year,
        reverse=True,
    )
    for obj in candidates:
        body = s3.get_object(Bucket=bucket, Key=obj.key)["Body"].read()
        buf = io.BytesIO(body)
        column = turtle_column(pq.read_schema(buf).names)
        if column:
            return obj, body, column
        _log(f"  {feed}: skipping {obj.year} (pre-RDF schema, no Turtle column)")
    raise RuntimeError(
        f"no vintage with a Turtle column "
        f"({' / '.join(TURTLE_COLUMN_CANDIDATES)}) for feed={feed}"
    )


# --------------------------------------------------------------------------- #
# graph sampling
# --------------------------------------------------------------------------- #

def load_graph(body: bytes, column: str):
    """Parse every Turtle document in a vintage into one graph."""
    import pandas as pd
    import rdflib

    frame = pd.read_parquet(io.BytesIO(body), columns=[column])
    graph = rdflib.Graph()
    for doc in frame[column]:
        if doc:
            graph.parse(data=doc, format="turtle")
    return graph


def _local(term) -> str:
    return str(term).rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def periods_by_subject(graph) -> dict:
    """Map each measurement subject to its (month, year) period, where it has one."""
    import rdflib

    months, years = {}, {}
    for subj, pred, obj in graph:
        if not isinstance(subj, rdflib.URIRef):
            continue
        name = _local(pred)
        if name == "hasMonth":
            months[subj] = _local(obj)
        elif name == "hasYear":
            years[subj] = _local(obj)
    return {s: (months[s], years[s]) for s in months if s in years}


def choose_anchor_periods(feed_periods: dict[str, dict]) -> list[tuple[str, str]]:
    """Periods carried by the most feeds, best coverage first.

    Anchoring every feed's sample on shared periods is what makes the
    cross-source temporal bridge assertable — see MIN_FEEDS_SHARING_PERIOD.
    """
    coverage = Counter()
    for periods in feed_periods.values():
        for period in set(periods.values()):
            coverage[period] += 1
    return [
        period
        for period, count in sorted(
            coverage.items(), key=lambda kv: (-kv[1], kv[0])
        )
        if count >= MIN_FEEDS_SHARING_PERIOD
    ]


def select_subjects(
    graph,
    periods: dict,
    anchors: list[tuple[str, str]],
    limit: int,
    anchor_share: float = ANCHOR_SHARE,
):
    """Pick measurement subjects: shared periods first, then type coverage.

    Anchoring on shared periods is what lets the cross-source temporal bridge
    form, so it gets the quota first. Whatever is left over — for feeds with
    fewer anchored subjects than the quota — greedily picks subjects that
    introduce an ``rdf:type`` not yet represented, so a thin feed still
    contributes a spread of its ontology rather than a short run of near
    duplicates.

    At the default ``anchor_share`` of 1.0 the type-coverage pass is a pure
    fallback. See the ANCHOR_SHARE comment for why broadening it by default is
    a bad trade.

    Sorted by URI throughout so the selection is deterministic across runs.
    """
    ranked = {period: rank for rank, period in enumerate(anchors)}
    anchor_quota = max(1, int(limit * anchor_share))

    # Round-robin across the anchor periods rather than draining the
    # best-covered one first. Taking them in rank order fills the whole quota
    # from a single month, which satisfies the overlap requirement but gives
    # the graph only one temporal node to bridge through — the unified
    # temporal entities are far more interesting with several periods present.
    by_period: dict = defaultdict(list)
    for subj, period in periods.items():
        if period in ranked:
            by_period[period].append(subj)

    queues = [
        sorted(by_period[period], key=str)
        for period in sorted(by_period, key=lambda p: ranked[p])
    ]
    chosen: list = []
    depth_index = 0
    while queues and len(chosen) < anchor_quota:
        progressed = False
        for queue in queues:
            if depth_index < len(queue):
                chosen.append(queue[depth_index])
                progressed = True
                if len(chosen) >= anchor_quota:
                    break
        if not progressed:
            break
        depth_index += 1

    # Remaining quota: greedily take subjects that introduce an unseen type.
    import rdflib

    picked = set(chosen)
    covered = {
        t for s in picked for t in graph.objects(s, rdflib.RDF.type)
    }
    remaining = sorted(
        (s for s in set(graph.subjects())
         if isinstance(s, rdflib.URIRef) and s not in picked),
        key=str,
    )

    for subj in remaining:
        if len(chosen) >= limit:
            break
        types = set(graph.objects(subj, rdflib.RDF.type))
        if types - covered:
            chosen.append(subj)
            picked.add(subj)
            covered |= types

    # Still short (every type already represented): top up deterministically so
    # small feeds do not end up with a thinner sample than requested.
    if len(chosen) < limit:
        chosen.extend(
            s for s in remaining if s not in picked
        )
        chosen = chosen[:limit]

    return chosen


def closure(graph, seeds, depth: int = CLOSURE_DEPTH):
    """Entity-complete subgraph: whole subjects plus everything they reach.

    Blank nodes are followed unconditionally and to any depth — they carry the
    measurement literals, and truncating them is exactly what emptied the
    previous fixtures. Referenced URIs get their full triple set down to
    ``depth``; past that the frontier is closed with rdf:type only, so every
    URI that appears in the output is still typed.
    """
    import rdflib

    out = rdflib.Graph()
    for prefix, uri in graph.namespaces():
        out.bind(prefix, uri)

    expanded: set = set()
    frontier = set(seeds)

    for level in range(depth):
        following: set = set()
        # Blank nodes discovered while expanding this level are drained here
        # rather than deferred to the next one: they are *part of* the entity
        # (they hold its measurement literals), not a reference to a separate
        # one, so they must be followed to any depth and must not be subject
        # to the URI depth budget.
        pending = sorted(frontier, key=str)
        while pending:
            subj = pending.pop(0)
            if subj in expanded:
                continue
            expanded.add(subj)
            for pred, obj in graph.predicate_objects(subj):
                out.add((subj, pred, obj))
                if isinstance(obj, rdflib.BNode):
                    pending.append(obj)
                elif isinstance(obj, rdflib.URIRef) and level + 1 < depth:
                    following.add(obj)
        frontier = following - expanded
        if not frontier:
            break

    # Close out anything still unexpanded with its type, so no bare URI is left
    # untyped in the output.
    for subj in frontier:
        for type_obj in graph.objects(subj, rdflib.RDF.type):
            out.add((subj, rdflib.RDF.type, type_obj))

    return out


def split_rows(graph, rows: int) -> list[str]:
    """Serialize into ``rows`` Turtle documents, partitioned by subject.

    Subjects are kept whole within a document so each row stays a valid,
    self-consistent graph — the same property that made whole-row sampling
    attractive in the first place.
    """
    import rdflib

    subjects = sorted(
        {s for s in graph.subjects() if isinstance(s, rdflib.URIRef)}, key=str
    )
    buckets: list[set] = [set() for _ in range(rows)]
    for index, subj in enumerate(subjects):
        buckets[index % rows].add(subj)

    def emit(keep: set) -> str:
        part = rdflib.Graph()
        for prefix, uri in graph.namespaces():
            part.bind(prefix, uri)
        pending = list(keep)
        seen = set()
        while pending:
            subj = pending.pop()
            if subj in seen:
                continue
            seen.add(subj)
            for pred, obj in graph.predicate_objects(subj):
                part.add((subj, pred, obj))
                if isinstance(obj, rdflib.BNode):
                    pending.append(obj)
        return part.serialize(format="turtle")

    return [emit(bucket) for bucket in buckets if bucket]


# --------------------------------------------------------------------------- #
# validation — the acceptance criteria, asserted in the generator itself
# --------------------------------------------------------------------------- #

def validate(samples: dict, anchors: list) -> None:
    """Fail loudly if the regenerated sample repeats the defect it replaces."""
    import rdflib

    total_subjects: set = set()
    typed_subjects: set = set()
    distinct_types: set = set()
    triple_counts: dict = defaultdict(int)
    # Measurement subjects: the ones that state a period. These are what the
    # sample is *for* — the dimension entities they reach (Month, Year,
    # Industry) are 1-3 triples each by construction and would drown the
    # measurements out of any statistic taken over all subjects.
    measurements: set = set()
    valueless: set = set()
    empty_bnodes = 0
    bnode_values = 0
    feed_periods = defaultdict(set)

    for feed, graph in samples.items():
        feed_measurements: set = set()
        for subj, pred, obj in graph:
            if isinstance(subj, rdflib.URIRef):
                total_subjects.add(subj)
                triple_counts[subj] += 1
                if pred == rdflib.RDF.type:
                    typed_subjects.add(subj)
            if pred == rdflib.RDF.type:
                distinct_types.add(_local(obj))
            if _local(pred) in ("hasMonth", "hasYear"):
                feed_periods[feed].add((_local(pred), _local(obj)))
                if isinstance(subj, rdflib.URIRef):
                    feed_measurements.add(subj)
        measurements |= feed_measurements
        # A measurement must arrive with its value. Upstream used to hang the
        # value off a blank node and now states it as a literal on the
        # measurement itself, so this counts literals wherever they sit:
        # directly on the subject, or on a blank node it points at. rdfs:label
        # is excluded because a dimension entity carries one and it says
        # nothing about whether the measurement survived.
        for subj in feed_measurements:
            literals = [
                o for p, o in graph.predicate_objects(subj)
                if isinstance(o, rdflib.Literal) and p != rdflib.RDFS.label
            ]
            for _, obj in graph.predicate_objects(subj):
                if isinstance(obj, rdflib.BNode):
                    literals.extend(
                        o for _, o in graph.predicate_objects(obj)
                        if isinstance(o, rdflib.Literal)
                    )
            if not literals:
                valueless.add(subj)
        for bnode in {s for s in graph.subjects() if isinstance(s, rdflib.BNode)}:
            if any(
                isinstance(o, rdflib.Literal)
                for _, o in graph.predicate_objects(bnode)
            ):
                bnode_values += 1
            else:
                empty_bnodes += 1

    typed_pct = 100.0 * len(typed_subjects) / max(len(total_subjects), 1)
    median = statistics.median(
        triple_counts[s] for s in measurements
    ) if measurements else 0

    _log("")
    _log("  validation")
    _log(f"    URI subjects .............. {len(total_subjects)}")
    _log(f"    carrying rdf:type ......... {len(typed_subjects)} ({typed_pct:.1f}%)")
    _log(f"    measurement subjects ...... {len(measurements)} "
         f"({len(valueless)} with no value)")
    _log(f"    median triples/measurement  {median}")
    _log(f"    blank nodes ............... {bnode_values + empty_bnodes} "
         f"({empty_bnodes} empty)")
    _log(f"    distinct rdf:type values .. {len(distinct_types)}")

    # ≥99% of URI subjects carry an rdf:type (was 14.9%).
    assert typed_pct >= 99.0, (
        f"only {typed_pct:.1f}% of URI subjects are typed; entity closure is "
        f"incomplete (target >=99%)"
    )
    # Measurements are present at all, and arrive whole rather than as
    # fragments (median was 1 on the fixtures this replaces).
    assert measurements, (
        "no subject states a period; the sample carries no measurements, only "
        "dimension entities"
    )
    assert median > 3, (
        f"median triples/measurement is {median}; sampling still looks "
        f"triple-level rather than entity-level"
    )
    # Every measurement carries its value. This is the flattened successor to
    # the blank-node check: what mattered was never that a blank node existed,
    # but that the numbers did not get stripped off on the way into the
    # fixture — which is exactly what produced the `[ ]` objects in the
    # fixtures this replaces.
    assert not valueless, (
        f"{len(valueless)} measurement subject(s) carry no literal value, "
        f"e.g. {sorted(map(str, valueless))[0]}"
    )
    # Blank nodes are no longer emitted by any BLS feed, so this is dormant
    # rather than dead: should upstream reintroduce them, a truncated closure
    # that leaves them empty still fails here.
    assert empty_bnodes == 0, (
        f"{empty_bnodes} blank nodes carry no literal; measurement values were "
        f"stripped"
    )
    # At least two feeds carry measurements in the same month.
    month_feeds = defaultdict(set)
    for feed, pairs in feed_periods.items():
        for kind, value in pairs:
            if kind == "hasMonth":
                month_feeds[value].add(feed)
    shared = {m: f for m, f in month_feeds.items() if len(f) >= MIN_FEEDS_SHARING_PERIOD}
    assert shared, (
        "no month is shared by two or more feeds; the cross-source temporal "
        "bridge cannot form on these fixtures"
    )
    best = max(shared.items(), key=lambda kv: len(kv[1]))
    _log(f"    shared months ............. {len(shared)} "
         f"(best: {best[0]} across {sorted(best[1])})")
    _log(f"    anchor periods ............ {anchors[:3]}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=os.environ.get("BLS_FIXTURE_BUCKET"),
        help="S3 archive bucket holding raw/source=bls/ (env: BLS_FIXTURE_BUCKET)",
    )
    parser.add_argument("--feeds", default=",".join(FEEDS))
    parser.add_argument("--subjects-per-feed", type=int, default=SUBJECTS_PER_FEED)
    parser.add_argument(
        "--anchor-share", type=float, default=ANCHOR_SHARE,
        help="fraction of each feed's quota spent on shared-period anchoring; "
             "the rest goes to type coverage. Raise toward 1.0 to hold the "
             "node-type count (and so the Spark stage count) down.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="sample and validate, but do not write the fixtures",
    )
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set BLS_FIXTURE_BUCKET)")

    import boto3
    import pandas as pd

    s3 = boto3.client("s3")

    _log(f"discovering objects in s3://{args.bucket}/{RAW_PREFIX}")
    objects = discover_objects(s3, args.bucket)
    _log(f"  {len(objects)} vintage(s) across "
         f"{len({o.feed for o in objects})} feed(s)")

    feeds = [f.strip() for f in args.feeds.split(",") if f.strip()]

    graphs, all_periods = {}, {}
    for feed in feeds:
        obj, body, column = newest_rdf_vintage(s3, args.bucket, objects, feed)
        _log(f"  {feed}: using {obj.year} ({obj.size / 1024:.0f} KiB, "
             f"column {column!r})")
        graph = load_graph(body, column)
        graphs[feed] = graph
        all_periods[feed] = periods_by_subject(graph)

    anchors = choose_anchor_periods(all_periods)
    if not anchors:
        raise RuntimeError(
            "no period is shared by two or more feeds; cannot guarantee the "
            "cross-source temporal bridge"
        )
    _log(f"\nanchor periods (shared by >={MIN_FEEDS_SHARING_PERIOD} feeds): "
         f"{anchors[:5]}")

    samples = {}
    for feed in feeds:
        seeds = select_subjects(
            graphs[feed], all_periods[feed], anchors, args.subjects_per_feed,
            anchor_share=args.anchor_share,
        )
        samples[feed] = closure(graphs[feed], seeds)
        _log(f"  {feed}: {len(seeds)} seed subjects -> "
             f"{len(samples[feed])} triples")

    validate(samples, anchors)

    if args.dry_run:
        _log("\ndry run — nothing written")
        return 0

    _log("")
    total_bytes = 0
    for feed, graph in samples.items():
        docs = split_rows(graph, OUTPUT_ROWS)
        target = FIXTURE_ROOT / feed / f"bls_{feed}_sample.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"triples": docs}).to_parquet(
            target, index=False, compression="snappy"
        )
        size = target.stat().st_size
        total_bytes += size
        _log(f"  wrote {target.relative_to(FIXTURE_ROOT.parents[3])} "
             f"({len(docs)} rows, {size / 1024:.0f} KiB)")

    _log(f"\ntotal fixture size: {total_bytes / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
