#!/usr/bin/env python3
"""
Regenerate the SEC end-to-end fixtures — ``tests/fixtures/e2e/turtle_parquet/
sec/sec_sample.parquet`` and ``tests/fixtures/e2e/ntriples/sec.nt`` — from the
raw crawl output in the project's S3 archive bucket.

WHY THIS EXISTS
---------------
The fixtures this replaces were three Turtle documents, ~500 bytes each, of the
form::

    <.../1782986746256_SECFiling_unknown> a sec-filings:SECFiling ;
        sec-filings:hasDocumentType "unknown" ;
        sec-filings:hasTextContent \"\"\"h1: Filing Detail ...\"\"\" .

That is a scraped HTML page with its tags stripped, typed ``SECFiling`` and
attached to nothing. Every SEC entity the pipeline knows how to link — the
issuer, the filing date, the reporting period, the holdings — was absent, so
``filings_SECFiling`` sat in the e2e graph as an orphaned node type and the SEC
half of cross-source linking was never exercised on real shapes.

Upstream is now RDF from the filing package itself: a filing carries its accession number, form type, issuer keyed on
CIK, filing and reporting dates as their own entities, and — depending on form —
institutional holdings, ownership transactions, proxy votes and XBRL facts.
This script samples that.

WHICH DAY TO SAMPLE
-------------------
Pass ``--date`` and name a day upstream BACKFILLED. Do not take the default
newest day without checking, and do not read the default as a recommendation:
it is what this did before the difference was understood.

Upstream fetches the filing *document* for at most 150 accessions per run,
selected as the head of EDGAR's accession-descending order with no cursor, so
the same few hundred filings are re-read every five minutes and the rest are
never fetched at all. A filing without its document carries metadata only — no
reporting owner, no transactions, no holdings. Measured document coverage per
day:

    2026-08-07   4,267 filings   100%   (backfilled on 08-11)
    2026-08-10   3,459 filings    17%
    2026-08-13   4,989 filings     9%
    2026-08-19   2,894 filings     5%

Sampling the newest day therefore produces a fixture in which ownership
filings are near-empty, and encodes 5% coverage as though it were the shape of
the data. The coverage is not a property of the day's filings — an ownership
filing whose document WAS fetched has its full contents, on every day sampled
(606/606 on 08-07, 23/23 on 08-19).

SAMPLING
--------
Three passes over one day's filings, all deterministic:

  1. **Type coverage.** The rare classes are genuinely rare — on a typical day
     ``XbrlFact`` and ``XbrlDimension`` appear in 2 documents of ~11,000,
     ``ProxyVote`` in 4. A random or head-of-file sample misses all of them, so
     documents are picked greedily by the classes they introduce.

  2. **Sequence coverage.** A class being present is not the same as its
     STRUCTURE being present. Pass 1 introduces NonDerivativeTransaction with
     the cheapest document that carries one, which is a Form 4 reporting a
     single transaction — and a lone transaction can form no sequence, so the
     transaction-level ``precedes`` chain had nothing to build on and its e2e
     guard measured nothing. This pass adds the cheapest ownership filing that
     reports two or more transactions of one class.

  3. **Period anchoring.** The remainder goes to filings whose reporting period
     falls in a month the *committed BLS fixtures* also carry, so the
     cross-source temporal bridge has something to join on. This is read out of
     the BLS fixtures on disk, not assumed — regenerate BLS first if both are
     being refreshed.

Within a document the sample is entity-complete, with one deliberate exception:
FAN_OUT_CAP. A single 13F-HR states up to several thousand
``hasInstitutionalHolding`` objects, which is 1 MB of Turtle for one filing and
no additional structure after the first few. Repeated relations are capped, and
the pointer triples to the dropped objects are dropped with them so nothing is
left referencing a URI that is not in the fixture.

NETWORK BOUNDARY
----------------
S3 access is **generation-time only**. The fixtures stay committed and the e2e
suite continues to read local files and touch no network. Preserve that
property: do not import this module from the test suite.

USAGE
-----
    export SEC_FIXTURE_BUCKET=<archive-bucket>      # or pass --bucket
    .venv/bin/python bin/generate_sec_e2e_fixtures.py

    # preview without writing
    .venv/bin/python bin/generate_sec_e2e_fixtures.py --dry-run

    # sample a specific day -- see WHICH DAY TO SAMPLE above. The committed
    # fixture comes from 2026-08-07, the newest day upstream backfilled.
    .venv/bin/python bin/generate_sec_e2e_fixtures.py --date 2026-08-07

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Only this feed carries RDF. `filing-detail`, `litigation`, `press-release`,
# `speeches`, `statements` and `testimony` are scraped HTML with a `parsed`
# column and no Turtle at all — the stub fixture this replaces came from one of
# them, which is why it was a wall of text rather than a filing.
RAW_PREFIX = "raw/source=sec/feed=filings/"

# Number of filings to sample. Sized against the BLS fixtures (~11 KiB per
# feed): SEC filings are ~10 triples each, so 40 lands in the same ballpark and
# keeps the whole committed fixture set well under the ~180 KB budget.
FILINGS_SAMPLED = 40

# Cap on objects kept per (subject, predicate). Only bites on the repeated
# relations — holdings, votes, facts — where the tail adds bytes and no
# structure. See SAMPLING above.
#
# Which objects the cap keeps is not arbitrary: an object that introduces a
# class no other object has is kept ahead of its siblings. Taking the first
# three by URI instead dropped XbrlDimension out of the sample entirely — the
# document was selected *for* it, but the only facts that carry a dimension
# sort after the ones that do not.
FAN_OUT_CAP = 3

# How many crawl rows to spread the sample across, matching the BLS generator:
# the production loader reads one Turtle document per row, so several rows keep
# the multi-document shape rather than collapsing to a single blob.
OUTPUT_ROWS = 3

# Depth to expand referenced URIs. Depth 1 is the entities a filing points at
# (issuer, dates, holdings, XBRL facts); depth 2 is what those point at in turn
# (a fact's dimension, a holding's issuer). Past that the frontier is closed
# with rdf:type alone so no bare URI is left untyped.
CLOSURE_DEPTH = 3

# Filings to put in the N-Triples fixture. That file is a loader test, not a
# graph test — market.nt is 96 lines and noaa.nt is 79 — so it gets a slice of
# the same sample rather than all of it.
NTRIPLES_FILINGS = 8

# Column names a vintage may spell its Turtle in, newest crawl schema last.
# Kept identical to build_graph.TURTLE_COLUMN_CANDIDATES: the generator must
# read exactly what the production loader reads, or a fixture can be sampled
# from a column the pipeline would ignore.
TURTLE_COLUMN_CANDIDATES = ("triples", "rdf_turtle")

# Identifier base for entities this project mints. A reference outside it is a
# reference to somebody else's published vocabulary — see validate().
PROJECT_ID_BASE = "https://jefflevesque.com/id/"

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2e"
PARQUET_TARGET = FIXTURE_ROOT / "turtle_parquet" / "sec" / "sec_sample.parquet"
NTRIPLES_TARGET = FIXTURE_ROOT / "ntriples" / "sec.nt"
BLS_FIXTURES = FIXTURE_ROOT / "turtle_parquet" / "bls"

# raw/source=sec/feed=filings/year=2026/month=08/14.snappy.parquet
_KEY_RE = re.compile(
    r"year=(?P<year>\d{4})/month=(?P<month>\d{2})/(?P<day>\d{2})\.snappy\.parquet$"
)

# `a ns1:Form10Q, ns1:SECFiling ;` — one match per subject, all of its classes
# in the group, including the ones on continuation lines.
_TYPES_RE = re.compile(r"\ba\s+((?:ns\d+:\w+\s*,\s*)*ns\d+:\w+)")

# Both date entities embed their ISO date in the URI. Read from there rather
# than from hasDateValue so a document can be classified before it is parsed.
_REPORTING_DATE_RE = re.compile(r"ReportingDate_(\d{4})-(\d{2})-\d{2}")

# `<...0002121472-26-000004_NonDerivativeTransaction_1>` — one match per
# transaction the document reports, whether it appears as the filing's pointer
# object or as its own subject, which is why the matches are deduplicated
# before they are counted.
_TRANSACTION_RE = re.compile(
    r"/id/sec-filings/([^\s<>\"]*_((?:Non)?DerivativeTransaction)_\d+)"
)

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


@dataclass(frozen=True)
class SourceObject:
    year: int
    month: int
    day: int
    key: str
    size: int

    @property
    def date(self) -> tuple[int, int, int]:
        return (self.year, self.month, self.day)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _local(term) -> str:
    return str(term).rsplit("/", 1)[-1].rsplit("#", 1)[-1]


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def discover_objects(s3, bucket: str) -> list[SourceObject]:
    """Every day partition under the raw SEC filings prefix, newest last."""
    found: list[SourceObject] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=RAW_PREFIX):
        for item in page.get("Contents", []):
            match = _KEY_RE.search(item["Key"])
            if not match:
                continue
            found.append(
                SourceObject(
                    year=int(match.group("year")),
                    month=int(match.group("month")),
                    day=int(match.group("day")),
                    key=item["Key"],
                    size=item["Size"],
                )
            )
    return sorted(found, key=lambda o: o.date)


def turtle_column(names) -> str | None:
    """The column a vintage spells its Turtle in, or None if it carries no RDF."""
    for candidate in TURTLE_COLUMN_CANDIDATES:
        if candidate in names:
            return candidate
    return None


def newest_rdf_day(s3, bucket: str, objects: list[SourceObject], date: str = ""):
    """Newest day partition that actually carries RDF, or the one named.

    Reads the Parquet *schema* and walks backwards, so a day written by a
    pre-RDF crawl schema is skipped by rule rather than by hardcoded date.

    ``date`` pins the walk to a single day, which is how the committed fixture
    is produced -- see WHICH DAY TO SAMPLE. It is also what lets a change to
    the SAMPLING rules be applied to the day the fixture already came from, so
    the diff shows the documents that pass added rather than replacing every
    document in the file.
    """
    import pyarrow.parquet as pq

    if date:
        year, month, day = (int(part) for part in date.split("-"))
        objects = [o for o in objects if o.date == (year, month, day)]
        if not objects:
            raise RuntimeError(f"no day partition for {date} under {RAW_PREFIX}")

    for obj in reversed(objects):
        body = s3.get_object(Bucket=bucket, Key=obj.key)["Body"].read()
        column = turtle_column(pq.read_schema(io.BytesIO(body)).names)
        if column:
            return obj, body, column
        _log(f"  skipping {obj.date} (pre-RDF schema, no Turtle column)")
    raise RuntimeError(
        f"no day with a Turtle column ({' / '.join(TURTLE_COLUMN_CANDIDATES)}) "
        f"under {RAW_PREFIX}"
    )


# --------------------------------------------------------------------------- #
# document selection
# --------------------------------------------------------------------------- #

def document_profile(doc: str) -> tuple[frozenset, frozenset]:
    """(classes, reporting periods) a Turtle document carries, read by regex.

    Deliberately not an rdflib parse: profiling all ~11,000 documents in a day
    that way costs minutes, and only the few dozen that get selected need to be
    parsed at all.
    """
    classes = {
        term.split(":", 1)[1]
        for match in _TYPES_RE.findall(doc)
        for term in re.split(r"\s*,\s*", match)
    }
    periods = {
        (MONTH_NAMES[int(month) - 1], year)
        for year, month in _REPORTING_DATE_RE.findall(doc)
    }
    return frozenset(classes), frozenset(periods)


def sequenceable_transactions(doc: str) -> int:
    """Largest number of transactions of ONE class the document reports.

    2 or more is what the transaction-level ``precedes`` chain needs to form an
    edge: the chain groups by (reporting owner, transaction class), so a filing
    reporting one non-derivative and one derivative transaction contributes two
    groups of one and no edge at all.

    Counted per class for that reason, and over the distinct transaction URIs
    rather than the raw matches — each one appears at least twice in the
    document, once as the filing's pointer object and once as its own subject.
    """
    per_class: dict[str, set[str]] = defaultdict(set)
    for local_name, klass in _TRANSACTION_RE.findall(doc):
        per_class[klass].add(local_name)
    return max((len(v) for v in per_class.values()), default=0)


def bls_fixture_periods() -> set[tuple[str, str]]:
    """(month, year) pairs the committed BLS fixtures carry.

    Read off disk, not assumed: anchoring the SEC sample on a period no BLS
    fixture states would leave the cross-source temporal bridge with nothing to
    join on, and the assertion below would then be vacuous rather than true.
    """
    import pyarrow.parquet as pq

    months: dict[str, set[str]] = defaultdict(set)
    for path in sorted(BLS_FIXTURES.rglob("*.parquet")):
        table = pq.read_table(path)
        column = turtle_column(table.schema.names)
        if not column:
            continue
        for doc in table.column(column).to_pylist():
            if not doc:
                continue
            # BLS states periods as bare URIs — .../id/cpi/June, .../id/cpi/2026
            # — reachable by hasMonth / hasYear. The measurement URI carries
            # both, spelled "June2026", which is the cheap thing to read.
            for month, year in re.findall(
                rf"({'|'.join(MONTH_NAMES)})(\d{{4}})", doc
            ):
                months[month].add(year)
    return {(month, year) for month, years in months.items() for year in years}


def select_documents(
    docs: list[str], anchors: set[tuple[str, str]], limit: int
) -> list[int]:
    """Pick document indices: type coverage, then sequence coverage, then
    period anchoring.

    Type coverage goes first because it is the scarce resource — a class that
    appears in 2 documents out of 11,000 is only ever going to arrive by being
    asked for. Anchoring then fills the rest, and ordinary filings top it up if
    too few of the day's filings report into an anchored period.

    Sorted by document index throughout, so the selection is deterministic.
    """
    profiles = [document_profile(doc) for doc in docs]

    chosen: list[int] = []
    picked: set[int] = set()
    covered: set[str] = set()

    # Pass 1 — every class the day carries, cheapest document first. Cheapest,
    # because a class is often introduced by both a 1 KB filing and a 1 MB one
    # and there is no reason to pay for the latter.
    by_cost = sorted(range(len(docs)), key=lambda i: (len(docs[i]), i))
    for index in by_cost:
        if len(chosen) >= limit:
            break
        classes = profiles[index][0]
        if classes - covered:
            chosen.append(index)
            picked.add(index)
            covered |= classes

    # Pass 2 — one ownership filing that reports a SEQUENCE, if pass 1 has not
    # already produced one. Cheapest again, and skipped entirely when the day
    # carries no such filing: this is a preference, not a requirement, and a
    # generator that raised here would be unable to sample a day on which
    # upstream reported only single-transaction filings. validate() is where
    # the outcome is asserted.
    if not any(sequenceable_transactions(docs[i]) > 1 for i in chosen):
        for index in by_cost:
            if len(chosen) >= limit:
                break
            if index not in picked and sequenceable_transactions(docs[index]) > 1:
                chosen.append(index)
                picked.add(index)
                covered |= profiles[index][0]
                break

    # Pass 3 — filings reporting into a period the BLS fixtures also carry.
    anchored = [
        index for index in range(len(docs))
        if index not in picked and profiles[index][1] & anchors
    ]
    for index in anchored:
        if len(chosen) >= limit:
            break
        chosen.append(index)
        picked.add(index)

    # Pass 4 — top up, so a day whose filings mostly report into unanchored
    # periods still yields a full-sized sample.
    for index in range(len(docs)):
        if len(chosen) >= limit:
            break
        if index not in picked:
            chosen.append(index)
            picked.add(index)

    return sorted(chosen)


# --------------------------------------------------------------------------- #
# graph sampling
# --------------------------------------------------------------------------- #

def load_graph(docs: list[str]):
    """Parse the selected Turtle documents into one graph."""
    import rdflib

    graph = rdflib.Graph()
    for doc in docs:
        graph.parse(data=doc, format="turtle")
    return graph


def closure(graph, seeds, depth: int = CLOSURE_DEPTH, cap: int = FAN_OUT_CAP):
    """Entity-complete subgraph: whole subjects plus everything they reach.

    Mirrors the BLS generator's closure, with one addition: ``cap`` bounds how
    many objects a single (subject, predicate) contributes, keeping the ones
    that introduce an unseen class first. Dropping the tail of a repeated
    relation is a size decision; dropping its *pointer* triples with it is a
    correctness one, because a retained pointer to a dropped entity is exactly
    the dangling untyped URI that made the old fixtures unresolvable.
    """
    import rdflib

    out = rdflib.Graph()
    for prefix, uri in graph.namespaces():
        out.bind(prefix, uri)

    expanded: set = set()
    covered: set = set()
    frontier = set(seeds)

    def priority(obj):
        """Sort key: most unseen classes in reach first, then URI.

        "In reach" is the object's own classes plus its neighbours'. One hop is
        needed because siblings under a repeated relation usually share a class
        — every object of hasXbrlFact is an XbrlFact — and what separates them
        is what they point at: only some facts carry an XbrlDimension. Scoring
        on the object's own class alone cannot see that and falls back to URI
        order, which is how XbrlDimension went missing.
        """
        classes = set(graph.objects(obj, rdflib.RDF.type))
        for neighbour in graph.objects(obj, None):
            if isinstance(neighbour, rdflib.URIRef):
                classes |= set(graph.objects(neighbour, rdflib.RDF.type))
        return (-len(classes - covered), str(obj))

    for level in range(depth):
        following: set = set()
        pending = sorted(frontier, key=str)
        while pending:
            subj = pending.pop(0)
            if subj in expanded:
                continue
            expanded.add(subj)
            covered |= set(graph.objects(subj, rdflib.RDF.type))
            by_predicate: dict = defaultdict(list)
            for pred, obj in graph.predicate_objects(subj):
                by_predicate[pred].append(obj)
            for pred, objects in by_predicate.items():
                kept = sorted(objects, key=priority)
                if len(kept) > cap:
                    kept = kept[:cap]
                for obj in sorted(kept, key=str):
                    out.add((subj, pred, obj))
                    # Blank nodes are part of the entity rather than a
                    # reference to another one, so they are followed to any
                    # depth and are not subject to the URI depth budget. No SEC
                    # feed emits one today; the branch costs nothing and keeps
                    # the closure correct if one appears.
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
    self-consistent graph — the same property a crawl row has upstream.
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


def ntriples_slice(graph, filings: list, count: int) -> str:
    """N-Triples serialization of the first ``count`` filings' closures.

    Lines are sorted before joining. rdflib's N-Triples serializer emits in set
    order, so the same sample otherwise produces a different file on every run
    and the fixture churns in every diff for no reason. (The Turtle serializer
    used for the Parquet rows already sorts, which is why only this one needs
    it.)
    """
    import rdflib

    part = rdflib.Graph()
    for subj in sorted(filings, key=str)[:count]:
        pending = [subj]
        seen = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            for pred, obj in graph.predicate_objects(node):
                part.add((node, pred, obj))
                if isinstance(obj, (rdflib.BNode, rdflib.URIRef)):
                    pending.append(obj)
    lines = sorted(
        line for line in part.serialize(format="nt").splitlines() if line.strip()
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# validation — the acceptance criteria, asserted in the generator itself
# --------------------------------------------------------------------------- #

def validate(graph, anchors: set[tuple[str, str]], selected_types: set) -> None:
    """Fail loudly if the regenerated sample repeats the defect it replaces."""
    import rdflib

    filings, total_subjects, typed_subjects = set(), set(), set()
    distinct_types: set = set()
    triple_counts: dict = defaultdict(int)

    for subj, pred, obj in graph:
        if isinstance(subj, rdflib.URIRef):
            total_subjects.add(subj)
            triple_counts[subj] += 1
            if pred == rdflib.RDF.type:
                typed_subjects.add(subj)
                distinct_types.add(_local(obj))
                if _local(obj) == "SECFiling":
                    filings.add(subj)

    typed_pct = 100.0 * len(typed_subjects) / max(len(total_subjects), 1)
    median = statistics.median(triple_counts[s] for s in filings) if filings else 0

    # A filing that states nothing but its own text is what the stub fixture
    # was. Both of these are what makes it a graph instead.
    with_issuer = {s for s in filings if _has_local(graph, s, "hasIssuer")}
    with_date = {s for s in filings if _has_local(graph, s, "hasFilingDate")}

    # Dangling references: an object URI that is not itself a subject anywhere
    # in the sample. These become nodes with no type and edges that cannot
    # resolve — the exact failure the entity closure exists to prevent.
    #
    # Scoped to this project's own identifier space. A filing also references
    # published vocabulary that upstream never describes and never will — an
    # XBRL fact points its hasConcept at http://xbrl.sec.gov/dei/2026#... —
    # and demanding a local description of those would fail on correct data.
    # They are reported, not asserted.
    unresolved = {
        obj for _, pred, obj in graph
        if isinstance(obj, rdflib.URIRef)
        and pred != rdflib.RDF.type
        and obj not in total_subjects
    }
    dangling = {u for u in unresolved if str(u).startswith(PROJECT_ID_BASE)}
    external = unresolved - dangling

    sample_periods = {
        (MONTH_NAMES[int(month) - 1], year)
        for year, month in _REPORTING_DATE_RE.findall(graph.serialize(format="turtle"))
    }
    shared = sample_periods & anchors

    # Transactions that can be SEQUENCED, not merely transactions present. The
    # chain groups by (owner, class) and needs two DATED members in one group,
    # so this is measured on the sampled graph — the closure's FAN_OUT_CAP
    # bounds how many of a filing's transactions survive into it — and it
    # counts only the ones carrying a transaction date.
    dated_transactions = {
        subj for subj in graph.subjects()
        if _has_local(graph, subj, "hasTransactionDate")
    }
    per_group: dict = defaultdict(set)
    for subj, pred, obj in graph:
        local = _local(pred)
        if local in ("hasNonDerivativeTransaction", "hasDerivativeTransaction"):
            if obj in dated_transactions:
                per_group[(subj, local)].add(obj)
    sequenceable = max((len(v) for v in per_group.values()), default=0)

    _log("")
    _log("  validation")
    _log(f"    URI subjects .............. {len(total_subjects)}")
    _log(f"    carrying rdf:type ......... {len(typed_subjects)} ({typed_pct:.1f}%)")
    _log(f"    filings ................... {len(filings)} "
         f"({len(with_issuer)} with issuer, {len(with_date)} with filing date)")
    _log(f"    median triples/filing ..... {median}")
    _log(f"    distinct rdf:type values .. {len(distinct_types)}")
    _log(f"    dangling references ....... {len(dangling)} "
         f"({len(external)} external vocabulary, not counted)")
    _log(f"    periods shared with BLS ... {sorted(shared)}")
    _log(f"    largest dated transaction group  {sequenceable}")

    assert typed_pct >= 99.0, (
        f"only {typed_pct:.1f}% of URI subjects are typed; entity closure is "
        f"incomplete (target >=99%)"
    )
    assert filings, "no subject is typed SECFiling; the sample carries no filings"
    # The stub fixture's filings carried 2 triples each and pointed at nothing.
    assert median > 5, (
        f"median triples/filing is {median}; the sample looks like fragments "
        f"rather than whole filings"
    )
    assert with_issuer == filings, (
        f"{len(filings - with_issuer)} filing(s) state no issuer, so nothing "
        f"connects them to a company"
    )
    assert with_date == filings, (
        f"{len(filings - with_date)} filing(s) state no filing date, so the "
        f"temporal unifier has nothing to read"
    )
    assert not dangling, (
        f"{len(dangling)} referenced URI(s) are not subjects in the sample, "
        f"e.g. {sorted(map(str, dangling))[0]} — their edges cannot resolve"
    )
    # Real classes, not just SECFiling: Issuer, Date, ReportingDate and at
    # least one form-specific class have to be present for the SEC half of the
    # pipeline to see anything it branches on.
    for required in ("Issuer", "Date", "ReportingDate"):
        assert required in distinct_types, (
            f"no {required} in the sample; SEC entities are not being reached"
        )
    # The selection and the closure have to agree. A document picked for a
    # class the closure then drops is a silent gap: the sample looks like it
    # covers the vocabulary and does not. XbrlDimension was exactly this — the
    # only facts carrying one sorted behind three that did not.
    missing = selected_types - distinct_types
    assert not missing, (
        f"documents were selected for {sorted(missing)} but the closure did "
        f"not reach those entities"
    )
    assert shared, (
        "no reporting period in the sample is a period the committed BLS "
        "fixtures carry; the cross-source temporal bridge cannot form"
    )
    # A chain of N transactions produces N-1 edges, so a sample whose largest
    # group is 1 produces none — and the e2e chain guard then measures nothing
    # while still reporting green. See pass 2 in select_documents.
    assert sequenceable > 1, (
        f"the largest group of dated transactions reported by one filing is "
        f"{sequenceable}; the transaction-level precedes chain needs two in "
        f"one (filing, class) group to produce a single edge, so the fixture "
        f"cannot exercise it"
    )


def _has_local(graph, subj, local: str) -> bool:
    return any(_local(p) == local for p in graph.predicates(subj))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=os.environ.get("SEC_FIXTURE_BUCKET"),
        help="S3 archive bucket holding raw/source=sec/ (env: SEC_FIXTURE_BUCKET)",
    )
    parser.add_argument(
        "--date", default="",
        help="YYYY-MM-DD day partition to sample; default = newest day with RDF",
    )
    parser.add_argument("--filings", type=int, default=FILINGS_SAMPLED)
    parser.add_argument("--fan-out-cap", type=int, default=FAN_OUT_CAP)
    parser.add_argument(
        "--ntriples-filings", type=int, default=NTRIPLES_FILINGS,
        help="filings to put in tests/fixtures/e2e/ntriples/sec.nt",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="sample and validate, but do not write the fixtures",
    )
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set SEC_FIXTURE_BUCKET)")

    import boto3
    import pandas as pd
    import rdflib

    s3 = boto3.client("s3")

    _log(f"discovering objects in s3://{args.bucket}/{RAW_PREFIX}")
    objects = discover_objects(s3, args.bucket)
    _log(f"  {len(objects)} day partition(s)")

    obj, body, column = newest_rdf_day(s3, args.bucket, objects, args.date)
    _log(f"  using {obj.year}-{obj.month:02d}-{obj.day:02d} "
         f"({obj.size / 1024 / 1024:.1f} MiB, column {column!r})")

    frame = pd.read_parquet(io.BytesIO(body), columns=[column])
    docs = [doc for doc in frame[column].tolist() if doc]
    _log(f"  {len(docs)} filing document(s)")

    anchors = bls_fixture_periods()
    _log(f"  BLS fixture periods: {sorted(anchors)}")

    selected = select_documents(docs, anchors, args.filings)
    graph = load_graph([docs[i] for i in selected])
    selected_types = {
        cls for i in selected for cls in document_profile(docs[i])[0]
    }
    _log(f"  {len(selected)} document(s) selected -> {len(graph)} triples, "
         f"{len(selected_types)} class(es)")

    seeds = sorted(
        set(graph.subjects(rdflib.RDF.type, None)) - set(graph.objects()),
        key=str,
    )
    sample = closure(graph, seeds, cap=args.fan_out_cap)
    _log(f"  closure over {len(seeds)} seed subject(s) -> {len(sample)} triples")

    validate(sample, anchors, selected_types)

    if args.dry_run:
        _log("\ndry run — nothing written")
        return 0

    _log("")
    turtle_docs = split_rows(sample, OUTPUT_ROWS)
    PARQUET_TARGET.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"triples": turtle_docs}).to_parquet(
        PARQUET_TARGET, index=False, compression="snappy"
    )
    _log(f"  wrote {PARQUET_TARGET.relative_to(REPO_ROOT)} "
         f"({len(turtle_docs)} rows, {PARQUET_TARGET.stat().st_size / 1024:.0f} KiB)")

    filings = sorted(
        subj for subj in set(sample.subjects(rdflib.RDF.type, None))
        if any(_local(t) == "SECFiling" for t in sample.objects(subj, rdflib.RDF.type))
    )
    nt = ntriples_slice(sample, filings, args.ntriples_filings)
    NTRIPLES_TARGET.parent.mkdir(parents=True, exist_ok=True)
    NTRIPLES_TARGET.write_text(nt)
    _log(f"  wrote {NTRIPLES_TARGET.relative_to(REPO_ROOT)} "
         f"({nt.count(chr(10))} triples, "
         f"{NTRIPLES_TARGET.stat().st_size / 1024:.0f} KiB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
