#!/usr/bin/env python3
"""
Regenerate the market end-to-end fixtures — ``tests/fixtures/e2e/turtle_parquet/
market/market_sample.parquet`` and ``tests/fixtures/e2e/ntriples/market.nt`` —
from the quote snapshots in the project's market S3 bucket.

WHY THIS EXISTS
---------------
The market fixture was the last one carved by hand rather than by a committed
tool, and it showed: it held three snapshots of a single ticker (``A``), which
matched no issuer in the SEC fixture. The SEC-to-market bridge therefore had
nothing to join even after the joins themselves were fixed — both sides reached
a ``unified:Company_*`` node, but never the SAME one, so the graph looked
connected per-namespace while no signal could cross.

Nothing here converts anything. The market bucket publishes ``rdf_turtle``
alongside the raw quote columns, exactly as the SEC and BLS feeds publish their
own Turtle, and this script samples that column. The raw columns are read ONLY
to choose which snapshots to keep. If the upstream vocabulary changes, this
fixture changes with it — which is the entire point of generating it from
source rather than writing it by hand.

SAMPLING
--------
Deterministic, and anchored so the fixture exercises the cross-source bridge:

  1. **Ticker anchoring.** Equity snapshots are preferred when their symbol is
     one the *committed SEC fixture* states via ``hasIssuerTradingSymbol``.
     This is read out of the SEC fixture on disk, not assumed — regenerate SEC
     first if both are being refreshed. It is the company-side twin of the
     period anchoring in ``generate_sec_e2e_fixtures.py``.

  2. **Option coverage.** For each anchored equity, option snapshots on the
     same underlying are included. Their underlying is recoverable only from
     the OCC symbol: a real option snapshot carries ``contractType``,
     ``strikePrice`` and ``symbol``, but NO ``underlyingSymbol`` and NO
     ``expirationDate`` (both raw columns are null upstream). Keeping options
     whose root matches a kept equity is what gives the OCC-derivation path
     real coverage.

  3. **An unanchored control.** One equity that matches NO SEC issuer is kept,
     so the "quote with no counterpart" branch stays exercised and the fixture
     cannot silently become one where everything joins.

NETWORK BOUNDARY
----------------
S3 access is **generation-time only**. The fixtures stay committed and the e2e
suite continues to read local files and touch no network. Preserve that
property: do not import this module from the test suite.

A snapshot object is ~420 MB and is read whole — the Turtle column cannot be
range-read out of it usefully — so expect a slow first step. This runs when the
upstream vocabulary changes, not per test run.

USAGE
-----
    export MARKET_FIXTURE_BUCKET=<market-bucket>    # or pass --bucket
    .venv/bin/python bin/generate_market_e2e_fixtures.py

    # preview without writing
    .venv/bin/python bin/generate_market_e2e_fixtures.py --dry-run

Requires credentials with read access to the market bucket. Output is
deterministic: the same source object produces byte-identical fixtures.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path

QUOTES_PREFIX = "market/intraday/quotes/"

# Anchored tickers to keep, and how many option snapshots to keep per anchored
# equity. Sized against the other fixtures (~11 KiB per BLS feed): a snapshot is
# ~33 triples, so this lands near the previous market fixture's footprint while
# covering both sides of the bridge.
ANCHORED_EQUITIES = 1
OPTIONS_PER_EQUITY = 2
CONTROL_EQUITIES = 1

# How many crawl rows to spread the sample across, matching the other
# generators: the production loader reads one Turtle document per row, so
# several rows keep the multi-document shape rather than collapsing to a blob.
OUTPUT_ROWS = 3

# Snapshots to put in the N-Triples fixture. That file is a loader test, not a
# graph test, so it gets a slice of the same sample rather than all of it.
NTRIPLES_SNAPSHOTS = 3

# Column names a vintage may spell its Turtle in, newest crawl schema last.
# Kept identical to build_graph.TURTLE_COLUMN_CANDIDATES: the generator must
# read exactly what the production loader reads, or a fixture can be sampled
# from a column the pipeline would ignore.
TURTLE_COLUMN_CANDIDATES = ("triples", "rdf_turtle")

# An OCC option symbol, whose first six characters are the underlying root.
# Restated from market/symbols.py rather than imported: bin/ scripts run without
# the Spark stack on the path, and that module is Spark column expressions.
OCC_OPTION_SYMBOL = re.compile(r"^(.{6})\d{6}[CP]\d{8}$")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2e"
PARQUET_TARGET = FIXTURE_ROOT / "turtle_parquet" / "market" / "market_sample.parquet"
NTRIPLES_TARGET = FIXTURE_ROOT / "ntriples" / "market.nt"
SEC_FIXTURE = FIXTURE_ROOT / "turtle_parquet" / "sec" / "sec_sample.parquet"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def newest_snapshot_key(s3, bucket: str) -> str:
    """The newest quote-snapshot object, found by descending the partitions.

    The prefix holds one object every twenty minutes of every trading day, so a
    full listing is tens of thousands of keys for one answer. Descending
    year=/month=/day= takes four bounded calls.
    """
    prefix = QUOTES_PREFIX
    while True:
        page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        files = sorted(
            o["Key"] for o in page.get("Contents", [])
            if o["Key"].endswith(".parquet")
        )
        if files:
            return files[-1]
        subdirs = sorted(p["Prefix"] for p in page.get("CommonPrefixes", []))
        if not subdirs:
            raise SystemExit(f"no quote snapshots under s3://{bucket}/{QUOTES_PREFIX}")
        prefix = subdirs[-1]


def turtle_column(names) -> str | None:
    """Which Turtle column this vintage spells, or None."""
    for candidate in reversed(TURTLE_COLUMN_CANDIDATES):
        if candidate in names:
            return candidate
    return None


def sec_fixture_tickers() -> set[str]:
    """Trading symbols the committed SEC fixture states, for ticker anchoring.

    Read from the fixture rather than hardcoded so the two stay in step: a
    regenerated SEC sample brings different issuers, and a hardcoded list would
    quietly stop anchoring while still looking deliberate.
    """
    import pandas as pd
    import rdflib

    if not SEC_FIXTURE.exists():
        _log(f"  WARNING: {SEC_FIXTURE} missing — no ticker anchoring")
        return set()

    graph = rdflib.Graph()
    for doc in pd.read_parquet(SEC_FIXTURE)["triples"]:
        graph.parse(data=doc, format="turtle")

    return {
        str(obj).strip().upper()
        for _s, pred, obj in graph
        if str(pred).endswith("hasIssuerTradingSymbol")
    }


def occ_root(symbol: str) -> str | None:
    """The underlying ticker of an OCC option symbol, or None."""
    match = OCC_OPTION_SYMBOL.match(symbol)
    return match.group(1).strip().upper() if match else None


def select_snapshots(frame, anchors: set[str]) -> list[str]:
    """The Turtle documents to keep, anchored per SAMPLING above."""
    column = turtle_column(frame.columns)
    if column is None:
        raise SystemExit(
            f"no Turtle column ({' / '.join(TURTLE_COLUMN_CANDIDATES)}) in the "
            f"snapshot object; columns are {sorted(frame.columns)[:10]}..."
        )

    frame = frame[["symbol", "asset_type", column]].dropna(subset=[column])
    frame = frame.assign(symbol=frame["symbol"].astype(str).str.strip())

    is_option = frame["symbol"].map(lambda s: occ_root(s) is not None)
    equities = frame[~is_option].assign(
        ticker=lambda f: f["symbol"].str.upper()
    ).sort_values("ticker")
    options = frame[is_option].assign(
        ticker=lambda f: f["symbol"].map(occ_root)
    ).sort_values("symbol")

    anchored = equities[equities["ticker"].isin(anchors)]
    if anchored.empty:
        _log(
            f"  WARNING: none of the {len(anchors)} SEC fixture ticker(s) quote "
            f"in this snapshot — the company bridge will have nothing to join"
        )
    kept_equities = list(anchored["ticker"].head(ANCHORED_EQUITIES))

    control = equities[~equities["ticker"].isin(anchors)]
    kept_control = list(control["ticker"].head(CONTROL_EQUITIES))

    docs: list[str] = []
    for ticker in kept_equities + kept_control:
        rows = equities[equities["ticker"] == ticker]
        docs += list(rows[column].head(1))
        if ticker in kept_equities:
            chain = options[options["ticker"] == ticker]
            docs += list(chain[column].head(OPTIONS_PER_EQUITY))
            _log(f"  {ticker}: 1 equity + {min(len(chain), OPTIONS_PER_EQUITY)} option(s) [anchored]")
        else:
            _log(f"  {ticker}: 1 equity [unanchored control]")

    return docs


def load_graph(docs: list[str]):
    import rdflib

    graph = rdflib.Graph()
    for doc in docs:
        graph.parse(data=doc, format="turtle")
    return graph


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
        for subj in keep:
            for pred, obj in graph.predicate_objects(subj):
                part.add((subj, pred, obj))
        return part.serialize(format="turtle")

    return [emit(bucket) for bucket in buckets if bucket]


def ntriples_slice(graph, count: int) -> str:
    """N-Triples for the first ``count`` snapshots, line-sorted.

    Sorted because rdflib's N-Triples serializer emits in set order, so the same
    sample would otherwise produce a different file every run and the fixture
    would churn in every diff for no reason.
    """
    import rdflib

    part = rdflib.Graph()
    for subj in sorted(set(graph.subjects()), key=str)[:count]:
        for pred, obj in graph.predicate_objects(subj):
            part.add((subj, pred, obj))
    lines = sorted(part.serialize(format="nt").splitlines())
    return "\n".join(line for line in lines if line.strip()) + "\n"


def validate(graph, anchors: set[str]) -> None:
    """Fail rather than commit a fixture that cannot exercise what it is for."""
    import rdflib

    types = {
        str(o).rsplit("/", 1)[-1]
        for o in graph.objects(None, rdflib.RDF.type)
    }
    missing = {"EquitySnapshot", "OptionSnapshot"} - types
    if missing:
        raise SystemExit(f"sample carries no {' or '.join(sorted(missing))}")

    symbols = {
        str(o).strip().upper()
        for _s, p, o in graph
        if str(p).endswith("market-quotes/symbol")
    }
    roots = {occ_root(s) or s for s in symbols}
    if anchors and not (roots & anchors):
        raise SystemExit(
            f"no kept snapshot resolves to a SEC fixture issuer ({sorted(anchors)}) "
            f"— the cross-source company bridge would have nothing to join, which "
            f"is the defect this fixture exists to cover"
        )
    _log(f"  validated: types={sorted(types)} tickers={sorted(roots)}")


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MARKET_FIXTURE_BUCKET"),
        help="S3 bucket holding market/intraday/quotes/ (env: MARKET_FIXTURE_BUCKET)",
    )
    parser.add_argument("--key", help="explicit snapshot object (else newest)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ntriples-snapshots", type=int, default=NTRIPLES_SNAPSHOTS)
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set MARKET_FIXTURE_BUCKET)")

    import boto3

    s3 = boto3.client("s3")

    key = args.key or newest_snapshot_key(s3, args.bucket)
    _log(f"reading s3://{args.bucket}/{key}")
    body = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
    frame = pd.read_parquet(io.BytesIO(body))
    _log(f"  {len(frame)} rows, {len(frame.columns)} columns")

    anchors = sec_fixture_tickers()
    _log(f"  SEC fixture tickers: {sorted(anchors) or '(none)'}")

    docs = select_snapshots(frame, anchors)
    if not docs:
        raise SystemExit("no snapshots selected")

    graph = load_graph(docs)
    _log(f"  {len(docs)} snapshot(s) -> {len(graph)} triples")

    validate(graph, anchors)

    if args.dry_run:
        _log("\ndry run — nothing written")
        return 0

    _log("")
    turtle_docs = split_rows(graph, OUTPUT_ROWS)
    PARQUET_TARGET.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"triples": turtle_docs}).to_parquet(
        PARQUET_TARGET, index=False, compression="snappy"
    )
    _log(f"  wrote {PARQUET_TARGET.relative_to(REPO_ROOT)} "
         f"({len(turtle_docs)} rows, {PARQUET_TARGET.stat().st_size / 1024:.0f} KiB)")

    nt = ntriples_slice(graph, args.ntriples_snapshots)
    NTRIPLES_TARGET.parent.mkdir(parents=True, exist_ok=True)
    NTRIPLES_TARGET.write_text(nt)
    _log(f"  wrote {NTRIPLES_TARGET.relative_to(REPO_ROOT)} "
         f"({nt.count(chr(10))} triples, "
         f"{NTRIPLES_TARGET.stat().st_size / 1024:.0f} KiB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
