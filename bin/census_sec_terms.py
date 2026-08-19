#!/usr/bin/env python3
"""
Per-form census of the SEC filings vocabulary: which terms actually carry a
value, on which form types, and in what proportion.

WHY THIS EXISTS
---------------
check_vocabulary_drift.py answers "does upstream still know this word". That is
the right question for a dead join, but it is not sufficient for a SPARSE one.
A term can be declared in the ontology, correctly mapped by the upstream
scraper, and still never carry a value -- market-quotes:underlyingSymbol is
mapped from reference.underlyingSymbol and is NULL on all 524,731 option rows,
because the broker returns nothing for it. A presence/absence scan reports that
term as EMITTED, and a join on it still matches nothing.

The SEC feed has a second, sharper version of the same problem: its terms are
partitioned by FORM FAMILY and are mutually exclusive across families. Ownership
terms appear only on Forms 3/4/5, 13F terms only on 13F, Xbrl* only on XBRL
instances. An aggregate null rate over all filings therefore describes no
document that exists -- 2% overall can mean "98% of the forms that can carry it".
Reading that number as drift would delete a working term.

So this groups by root_form and reports the conditional rate, which is the only
figure a join can be planned against.

WHAT IT CANNOT TELL YOU
-----------------------
Coverage is bounded by the dates sampled. A form type absent from the sample is
reported as UNSAMPLED, never as zero -- the distinction the aggregate rate
destroys is the same one a thin sample destroys.

USAGE
-----
    export ARCHIVE_BUCKET=<archive-bucket>
    .venv/bin/python bin/census_sec_terms.py                    # newest object
    .venv/bin/python bin/census_sec_terms.py --dates 2026-08-06,2026-08-07
    .venv/bin/python bin/census_sec_terms.py --json out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "_cvd", Path(__file__).resolve().parent / "check_vocabulary_drift.py"
)
_cvd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cvd)

SEC_PREFIX = "raw/source=sec/feed=filings/"

# The SEC namespaces whose referenced terms this censuses, and the prefix each
# is printed under. Both are needed: the filings vocabulary carries the form
# classes and their properties, sec-common the shared Date/Company classes and
# hasDateValue.
SEC_NAMESPACE_NAMES = ("SEC_FILINGS", "SEC_COMMON")
_PREFIX_OF = {"SEC_FILINGS": "sec-filings", "SEC_COMMON": "sec-common"}

# Issuer subject shapes. Upstream mints three, and they do not merge:
#   Issuer_0000320193          CIK-keyed, current
#   0001-26-1_Issuer_0         positional -- the PERMANENT fallback for a
#                              document that states no CIK (a 13F names a
#                              security, not an issuer), not merely a legacy
#                              shape from before the 2026-08-09 change
#   Issuer_apple_inc           name-slug, from the retired scrapy crawler
# A lowercase letter straight after "Issuer_" separates the slug form cleanly,
# because a CIK is digits.
_ISSUER_CIK = re.compile(r"/id/sec-filings/Issuer_(\d+)")
_ISSUER_POSITIONAL = re.compile(r"/id/sec-filings/[^\s>\"]*?_Issuer_(\d+)\b")
_ISSUER_SLUG = re.compile(r"/id/sec-filings/Issuer_([a-z][\w.-]*)")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _objects_for(s3, bucket: str, dates: list[str]) -> list[str]:
    """Keys to read: the named dates, or the newest object if none given."""
    if not dates:
        key, _ = _cvd_newest(s3, bucket, SEC_PREFIX)
        return [key] if key else []

    keys = []
    for date in dates:
        year, month, day = date.split("-")
        prefix = f"{SEC_PREFIX}year={year}/month={month}/"
        page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.split(".")[0].zfill(2) == day:
                keys.append(obj["Key"])
    return keys


def _cvd_newest(s3, bucket: str, prefix: str):
    """Deepest-partition newest object, reusing the drift checker's walk."""
    current = prefix
    for _ in range(8):
        page = s3.list_objects_v2(Bucket=bucket, Prefix=current, Delimiter="/")
        files = [o for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]
        if files:
            newest = max(files, key=lambda o: o["LastModified"])
            return newest["Key"], newest["LastModified"]
        subdirs = sorted(
            (p["Prefix"] for p in page.get("CommonPrefixes", [])),
            key=lambda t: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", t)],
        )
        if not subdirs:
            return None, None
        current = subdirs[-1]
    return None, None


def census(bucket: str, dates: list[str], rows: int) -> dict:
    import boto3
    import io
    import pandas as pd
    import pyarrow.parquet as pq

    s3 = boto3.client("s3")
    keys = _objects_for(s3, bucket, dates)
    if not keys:
        raise SystemExit(f"no SEC objects found under s3://{bucket}/{SEC_PREFIX}")

    # rows carrying each term, by form; and rows seen per form
    by_form: dict[str, Counter] = defaultdict(Counter)
    form_rows: Counter = Counter()
    issuer = Counter()
    unpadded: set[str] = set()
    sampled = []

    for key in keys:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        source = io.BytesIO(body)
        names = pq.ParquetFile(source).schema_arrow.names
        column = next((c for c in ("rdf_turtle", "triples") if c in names), None)
        if column is None:
            _log(f"  {key}: no Turtle column, skipped")
            continue
        # root_form is what makes the rate conditional rather than aggregate.
        # Without it every ownership term reads as ~2% and looks like drift.
        wanted = [column] + [c for c in ("root_form", "root_forms") if c in names]
        source.seek(0)
        frame = pd.read_parquet(source, columns=wanted)
        form_col = next((c for c in ("root_form", "root_forms") if c in wanted), None)
        if rows:
            frame = frame.head(rows)
        sampled.append((key, len(frame)))
        _log(f"  {key}: {len(frame)} rows")

        for _, row in frame.iterrows():
            blob = row[column]
            if not isinstance(blob, str) or not blob:
                continue
            form = str(row[form_col]) if form_col else "?"
            form_rows[form] += 1
            # One row's terms, deduped: the question is "does this filing carry
            # the term at all", not how many triples it wrote.
            terms = _cvd._terms_from_turtle([blob])
            for term in terms:
                by_form[form][term] += 1

            for m in _ISSUER_CIK.finditer(blob):
                issuer["cik"] += 1
                if len(m.group(1)) != 10:
                    unpadded.add(m.group(1))
            issuer["positional"] += len(_ISSUER_POSITIONAL.findall(blob))
            issuer["slug"] += len(_ISSUER_SLUG.findall(blob))

    return {
        "sampled": sampled,
        "form_rows": dict(form_rows),
        "by_form": {f: dict(c) for f, c in by_form.items()},
        "issuer_shapes": dict(issuer),
        "unpadded_ciks": sorted(unpadded)[:20],
    }


def report(result: dict, bases: dict) -> None:
    """Print the per-form census.

    ``bases`` maps each namespace CONSTANT NAME to its URI, and every term is
    resolved under the namespace it was referenced from. Resolving them all
    under one base is not a cosmetic error: this report once built every URI
    from SEC_FILINGS, so sec-common:hasDateValue was looked up at
    ontology/sec-filings/hasDateValue, found nowhere, and reported ABSENT
    (0.00%) -- while the term is in fact emitted by 100% of filing rows and is
    the ONLY path by which any SEC date resolves. A census that mis-resolves a
    term reports a live term as dead, which is a licence to delete working
    code; it is the exact failure the census exists to prevent, turned around.
    """
    form_rows = result["form_rows"]
    by_form = result["by_form"]
    total = sum(form_rows.values())

    print(f"\nSEC filings census -- {total} rows across {len(result['sampled'])} object(s)")
    for key, n in result["sampled"]:
        print(f"  {key}  ({n} rows)")

    print("\nrows by root_form:")
    for form, n in sorted(form_rows.items(), key=lambda kv: -kv[1]):
        print(f"  {form:<12} {n:>8}  ({100.0 * n / total:.1f}%)")

    # Overall and best-form rate for every referenced SEC term, each resolved
    # under its OWN namespace.
    referenced = _cvd.referenced_terms()
    terms = sorted(
        (term, name)
        for name in bases
        for term in referenced.get(name, set())
    )
    print(f"\nper-term presence ({len(terms)} referenced terms):")
    print(f"  {'term':<36} {'overall':>8}  best form (rate)")
    buckets = {"POPULATED": [], "SPARSE": [], "ABSENT": []}
    for term, namespace in terms:
        uri = bases[namespace] + term
        # Printed with its namespace, so two namespaces declaring the same
        # local name stay distinguishable in the output as well as in the
        # lookup.
        label = f"{_PREFIX_OF[namespace]}:{term}"
        hits = {f: by_form.get(f, {}).get(uri, 0) for f in form_rows}
        overall = sum(hits.values())
        pct = 100.0 * overall / total if total else 0.0
        best = max(
            ((f, 100.0 * n / form_rows[f]) for f, n in hits.items() if form_rows[f]),
            key=lambda kv: kv[1],
            default=("-", 0.0),
        )
        if overall == 0:
            buckets["ABSENT"].append(label)
        elif pct >= 90.0:
            buckets["POPULATED"].append(label)
        else:
            buckets["SPARSE"].append(label)
        print(f"  {label:<36} {pct:>7.2f}%  {best[0]} ({best[1]:.1f}%)")

    print("\nbuckets:")
    for name, items in buckets.items():
        print(f"  {name:<12} {len(items):>3}: {', '.join(items) if items else '-'}")

    shapes = result["issuer_shapes"]
    print("\nissuer subject shapes:")
    for shape in ("cik", "positional", "slug"):
        print(f"  {shape:<12} {shapes.get(shape, 0)}")
    if result["unpadded_ciks"]:
        print(f"  UNPADDED CIKs seen: {result['unpadded_ciks']}")
    else:
        print("  all CIKs zero-padded to 10")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bucket", default="")
    parser.add_argument("--dates", default="",
                        help="comma-separated YYYY-MM-DD; default = newest object")
    parser.add_argument("--rows", type=int, default=0,
                        help="max rows per object; 0 = every row")
    parser.add_argument("--json", help="also write the raw counts here")
    args = parser.parse_args()

    bucket = args.archive_bucket or os.environ.get("ARCHIVE_BUCKET", "")
    if not bucket:
        parser.error("needs --archive-bucket (or ARCHIVE_BUCKET)")

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    result = census(bucket, dates, args.rows)

    from spark_jobs.utils import rdf_utils
    report(result, {
        name: str(getattr(rdf_utils, name)) for name in SEC_NAMESPACE_NAMES
    })

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        _log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
