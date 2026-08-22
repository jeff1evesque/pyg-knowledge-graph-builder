#!/usr/bin/env python3
"""
Report ontology terms this pipeline keys on that the upstream data does not
emit — the drift that makes a join match nothing, emit nothing, and raise
nothing.

WHY THIS EXISTS
---------------
Ten defects in this repository have been the same shape. A linker filters on
``SEC_FILINGS.hasIssuerTicker`` / ``MARKET_FEEDS.StockTicker`` /
``MARKET_QUOTES.underlyingSymbol``; upstream renamed it, moved it onto another
subject, or never emitted it at all; the filter matches zero rows; the join
drops every row; the step logs that it ran and returns nothing. Spark does not
care, the schema does not change, and the pipeline produces a well-formed graph
that is missing an entire relationship. Every one of them was found by a person
noticing an absence, one at a time.

They are all mechanically detectable, because both halves are enumerable: the
terms the CODE references, and the terms the DATA emits. This diffs them.

WHAT IT CANNOT TELL YOU
-----------------------
That a term is absent from a sample does not prove the code is wrong — the
sample may simply not cover it. So the report only JUDGES a namespace it has
evidence for: if a sample carries no term at all from ``sec-litigation``, that
namespace is reported as UNCOVERED rather than as 22 dead references. An
uncovered namespace is its own finding — it means those code paths have no
coverage from this data — but it is not drift.

Run it against real S3 volume for a verdict. Run it against the committed
fixtures for a fast regression check.

USAGE
-----
    # committed fixtures (offline, fast)
    .venv/bin/python bin/check_vocabulary_drift.py

    # real data
    export ARCHIVE_BUCKET=<archive-bucket> MARKET_BUCKET=<market-bucket>
    .venv/bin/python bin/check_vocabulary_drift.py --s3

    # fail on anything absent in a covered namespace
    .venv/bin/python bin/check_vocabulary_drift.py --strict

    # fail only on absences that are NEW relative to a committed baseline
    .venv/bin/python bin/check_vocabulary_drift.py --baseline conf/vocabulary_baseline.json
    .venv/bin/python bin/check_vocabulary_drift.py --write-baseline conf/vocabulary_baseline.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "spark_jobs"

# Importable when run as a script from anywhere, the way the other bin/ tools
# are invoked. rdf_utils is imported for the namespace BASES only -- no Spark
# session is created and none is needed.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2e" / "turtle_parquet"

# The SOURCE vocabularies — terms minted by the upstream mappers, which is the
# only half that can drift out from under us. The pipeline's own enrichment
# namespaces are deliberately excluded: it emits those itself, so a term it
# references and never emits is a different (and much louder) kind of bug.
SOURCE_NAMESPACE_NAMES = (
    "SEC_FILINGS", "SEC_COMMON",
    "MARKET_QUOTES",
    "CAP", "WEATHER",
    "CPI", "PPI", "ECI", "EMPSIT", "JOLTS", "LAUS", "METRO", "REALER",
    "WKYENG", "XIMPIM", "BLS_COMMON",
)

# Attribute names that are not ontology terms.
_NOT_TERMS = frozenset({"value", "title", "n3", "toPython", "encode", "strip"})

# A feed not written to in this long is treated as decommissioned rather than as
# evidence about the current vocabulary. Generous, because some feeds are
# genuinely low-frequency -- this is meant to catch a source that STOPPED, not
# one that is merely quiet.
STALE_AFTER_DAYS = 60


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def referenced_terms() -> dict[str, set[str]]:
    """Every ``NAMESPACE.term`` the pipeline code names, by namespace.

    Parsed rather than grepped: a regex also matches the name inside a comment
    or a docstring, and this report's whole value is that its two halves are
    exact. rdflib resolves any attribute on a Namespace, so ``SEC.anything`` is
    syntactically valid and only this kind of check can tell you it is wrong.
    """
    found: dict[str, set[str]] = defaultdict(set)
    for path in sorted(CODE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            if node.value.id not in SOURCE_NAMESPACE_NAMES:
                continue
            if node.attr in _NOT_TERMS or node.attr.startswith("_"):
                continue
            found[node.value.id].add(node.attr)
    return found


_PREFIX_DECL = re.compile(r"@prefix\s+([A-Za-z][\w.-]*)?:\s*<([^>]+)>")
_FULL_URI = re.compile(r"<(https?://[^>]+)>")
_QNAME = re.compile(r"(?<![\w:/#-])([A-Za-z][\w.-]*):([A-Za-z_][\w.-]*)")


def _terms_from_turtle(blobs) -> set[str]:
    """Every ontology term the documents mention, expanded to full URIs.

    Scanned as TEXT rather than parsed into a graph, which is what makes it
    affordable to read every row instead of a sample. That matters more than it
    sounds: the terms this report is looking for are the RARE ones, and the
    first version sampled the head of each object and duly reported
    hasIssuerTradingSymbol as absent from a feed that emits it — the exact
    head-of-file sampling that generate_sec_e2e_fixtures.py exists to avoid
    (XbrlFact appears in 2 documents of ~11,000). A term that occurs once in
    ten thousand rows is precisely the one whose disappearance nobody notices,
    so the scan has to cover all of them.

    Both spellings are collected: full ``<uri>`` forms and ``prefix:local``
    qnames resolved through the document's own @prefix declarations. Upstream
    writes qnames almost exclusively, so reading only full URIs would find
    nearly nothing.

    Deliberately over-collects: a term appearing anywhere in a document counts,
    whether as predicate, class, or subject. This answers "does upstream still
    know this word", which is the question a dead join turns on.
    """
    emitted: set[str] = set()
    for blob in blobs:
        text = str(blob)
        prefixes = {name or "": uri for name, uri in _PREFIX_DECL.findall(text)}
        emitted.update(_FULL_URI.findall(text))
        for prefix, local in _QNAME.findall(text):
            base = prefixes.get(prefix)
            if base:
                emitted.add(base + local)
    return emitted


def emitted_from_fixtures() -> set[str]:
    import pandas as pd

    emitted: set[str] = set()
    for path in sorted(FIXTURE_ROOT.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        column = "triples" if "triples" in frame.columns else frame.columns[0]
        emitted |= _terms_from_turtle(frame[column].dropna())
    return emitted


def emitted_from_s3(archive_bucket: str, market_bucket: str, rows: int) -> set[str]:
    """Read the newest object of each feed, every row of its Turtle column.

    One object per feed rather than every partition: a feed's vocabulary is
    stable across its history in a way its values are not, so the newest object
    answers "which terms does upstream still emit" without moving hundreds of
    gigabytes. But EVERY ROW of that object, not a head slice — the terms worth
    finding are the rare ones, and `--rows` exists only as an escape hatch.
    """
    import boto3
    import io
    import pandas as pd

    s3 = boto3.client("s3")
    emitted: set[str] = set()
    stale: list[tuple[str, int]] = []

    def _natural(text: str) -> list:
        """Sort key that orders embedded numbers numerically.

        Partition names are zero-padded today (``month=08``), and a plain
        lexical sort happens to be right for that. It is not right in general —
        ``month=8`` sorts after ``month=10``, and a day file named ``5.parquet``
        sorts after ``17.parquet`` — so the ordering that decides WHICH object
        gets read would silently pick a stale partition the moment upstream
        stopped padding. Reading old data and reporting it as current is the
        one failure this whole report cannot survive.
        """
        return [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", text)
        ]

    def newest_under(bucket: str, prefix: str):
        current = prefix
        for _ in range(8):
            page = s3.list_objects_v2(Bucket=bucket, Prefix=current, Delimiter="/")
            files = [
                o for o in page.get("Contents", [])
                if o["Key"].endswith(".parquet")
            ]
            if files:
                # By write time, not by name. The partition path encodes the
                # period the data is ABOUT, which is not always the order it
                # was written in, and "latest" here means most recently
                # published vocabulary.
                newest = max(files, key=lambda o: o["LastModified"])
                return newest["Key"], newest["LastModified"]
            subdirs = sorted(
                (p["Prefix"] for p in page.get("CommonPrefixes", [])), key=_natural
            )
            if not subdirs:
                return None, None
            current = subdirs[-1]
        return None, None

    import os

    quotes_prefix = os.environ.get("MARKET_QUOTES_PREFIX", "")
    if market_bucket and not quotes_prefix:
        raise SystemExit("MARKET_QUOTES_PREFIX is required to sample the market bucket")

    targets = [(market_bucket, quotes_prefix)] if market_bucket else []
    if archive_bucket:
        # Paths that do NOT follow the raw/source=<name>/feed=<name>/ convention
        # and would be missed by the discovery below.
        #
        # NOAA is the one that matters: its CAP alert RDF lives at raw/noaa/,
        # while raw/source=weather/ holds an Atom feed wrapper carrying no CAP
        # term at all. Discovering only source= partitions therefore reported
        # CAP and WEATHER as having no coverage — which reads as "these code
        # paths are unverified" when in fact the data was simply never looked
        # at. A report that quietly omits a source is worse than no report.
        targets += [(archive_bucket, prefix) for prefix in ("raw/noaa/",)]

        page = s3.list_objects_v2(
            Bucket=archive_bucket, Prefix="raw/source=", Delimiter="/"
        )
        for src in (p["Prefix"] for p in page.get("CommonPrefixes", [])):
            feeds = s3.list_objects_v2(
                Bucket=archive_bucket, Prefix=src, Delimiter="/"
            )
            subs = [p["Prefix"] for p in feeds.get("CommonPrefixes", [])] or [src]
            targets += [(archive_bucket, sub) for sub in subs]

    for bucket, prefix in targets:
        key, written = newest_under(bucket, prefix)
        if not key:
            continue
        age = (datetime.now(timezone.utc) - written).days
        if age > STALE_AFTER_DAYS:
            # A feed nobody writes to any more is not evidence about the
            # CURRENT vocabulary. Reported rather than silently mixed in:
            # raw/source=weather/ still holds a decommissioned Atom-era NOAA
            # feed, and treating its terms as live would say the CAP model is
            # gone when it simply moved to raw/noaa/.
            stale.append((prefix, age))
            _log(f"  {prefix}: SKIPPED, last written {age}d ago")
            continue
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

        # Read the Turtle column ALONE, chosen off the schema.
        #
        # Not just for speed, though a quote snapshot is 69 columns and 420 MB
        # and this wants one of them. Reading the whole frame also decodes
        # every other column, and a timestamp column carrying a zone name the
        # host has no tzdata entry for ("US/Eastern") raises out of pyarrow and
        # takes the whole feed's report with it. Projecting to the one column
        # this needs makes the report independent of the rest of the schema.
        import pyarrow.parquet as pq

        source = io.BytesIO(body)
        names = pq.ParquetFile(source).schema_arrow.names
        column = next((c for c in ("rdf_turtle", "triples") if c in names), None)
        if column is None:
            _log(f"  {prefix}: no Turtle column, skipped")
            continue
        source.seek(0)
        frame = pd.read_parquet(source, columns=[column])
        series = frame[column].dropna()
        if rows:
            series = series.head(rows)
        _log(f"  {prefix}: {len(series)} of {len(frame)} rows, {age}d old")
        emitted |= _terms_from_turtle(series)
    return emitted, stale


def analyze(emitted: set[str]) -> tuple[dict, list[str]]:
    """(drift by namespace, namespaces with no evidence either way)."""
    from spark_jobs.utils import rdf_utils

    referenced = referenced_terms()
    drift: dict[str, list[str]] = {}
    uncovered: list[str] = []

    for name in sorted(SOURCE_NAMESPACE_NAMES):
        base = str(getattr(rdf_utils, name))
        terms = referenced.get(name, set())
        if not terms:
            continue
        # Is this namespace represented in the sample at all? Without that,
        # "absent" means "not sampled" and reporting it as drift is noise.
        if not any(term.startswith(base) for term in emitted):
            uncovered.append(name)
            continue
        absent = sorted(t for t in terms if base + t not in emitted)
        if absent:
            drift[name] = absent
    return drift, uncovered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", action="store_true", help="sample real data")
    parser.add_argument("--archive-bucket", default="")
    parser.add_argument("--market-bucket", default="")
    parser.add_argument("--rows", type=int, default=0,
                        help="max rows per feed; 0 = every row")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any covered namespace has drift")
    parser.add_argument("--baseline", help="fail only on absences NEW vs this file")
    parser.add_argument("--write-baseline", help="write current drift and exit")
    args = parser.parse_args()

    if args.s3:
        import os
        archive = args.archive_bucket or os.environ.get("ARCHIVE_BUCKET", "")
        market = args.market_bucket or os.environ.get("MARKET_BUCKET", "")
        if not (archive or market):
            parser.error("--s3 needs --archive-bucket / --market-bucket "
                         "(or ARCHIVE_BUCKET / MARKET_BUCKET)")
        _log("sampling S3...")
        emitted, stale = emitted_from_s3(archive, market, args.rows)
    else:
        _log("reading committed fixtures...")
        emitted, stale = emitted_from_fixtures(), []

    drift, uncovered = analyze(emitted)

    if args.write_baseline:
        Path(args.write_baseline).write_text(
            json.dumps(drift, indent=2, sort_keys=True) + "\n"
        )
        _log(f"wrote baseline: {sum(len(v) for v in drift.values())} entries")
        return 0

    print("\n=== TERMS THE CODE KEYS ON THAT THE DATA DOES NOT EMIT ===")
    if not drift:
        print("  none — every referenced term in a covered namespace is emitted")
    for namespace, terms in sorted(drift.items()):
        print(f"\n{namespace} ({len(terms)}):")
        for term in terms:
            print(f"   {term}")

    if stale:
        print("\n=== FEEDS SKIPPED AS DECOMMISSIONED ===")
        print(f"  Not written to in over {STALE_AFTER_DAYS} days, so not")
        print("  evidence about the vocabulary upstream emits today.")
        for prefix, age in sorted(stale):
            print(f"   {prefix}  ({age}d)")

    if uncovered:
        print("\n=== NAMESPACES WITH NO COVERAGE IN THIS SAMPLE ===")
        print("  Not drift — nothing here can be judged. But the code paths")
        print("  keyed on them are equally unverified.")
        print("   " + ", ".join(uncovered))

    if args.baseline:
        known = json.loads(Path(args.baseline).read_text())
        new = {
            ns: sorted(set(terms) - set(known.get(ns, [])))
            for ns, terms in drift.items()
        }
        new = {ns: terms for ns, terms in new.items() if terms}
        if new:
            print("\n=== NEW since the baseline ===")
            for ns, terms in sorted(new.items()):
                print(f"   {ns}: {', '.join(terms)}")
            return 1
        fixed = {
            ns: sorted(set(terms) - set(drift.get(ns, [])))
            for ns, terms in known.items()
        }
        fixed = {ns: terms for ns, terms in fixed.items() if terms}
        if fixed:
            print("\n=== resolved since the baseline (update it) ===")
            for ns, terms in sorted(fixed.items()):
                print(f"   {ns}: {', '.join(terms)}")
        return 0

    if args.strict and drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
