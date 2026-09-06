"""Count self-loops in an enriched-triples Parquet set: nodes that are their own object.

WHY THIS IS TRACKED
-------------------
This started as issue #360's acceptance check and lived only in the per-run copy
of the run harness, which is how it went missing once already. It is a general
measurement -- "does any predicate make a node point at itself, and how many
edges does that predicate carry" -- and it is worth having in a clean checkout.

WHAT IT REPORTS
---------------
One grouped pass gives every predicate with at least one self-loop, so the answer
is not limited to whatever predicate prompted the question. `precedes` is broken
out by source on top of that, because sequencing is where self-loops have
actually appeared.

Watch the edge counts, not just the loop total. A loop count that falls to zero
because the edge count collapsed means a dedupe took rows it should have kept,
and the total on its own hides that.

Measured on run 20260901T195728Z, before the #360 fix, as the reference for what
a regression would look like:

    bls/precedes        52,499 edges     5,988 self-loops  (11.41%)
    sec/precedes           274 edges       231 self-loops  (84.31%)
    market/precedes  9,627,931 edges         0
    noaa/precedes        1,871 edges         0
                                  total  6,219

RUNNING IT
----------
    bin/selfloops.py <enriched-triples-parquet> [output.json]

Local Spark: the cluster is idle by the time this runs and a scan this size does
not need it. Size it with SELFLOOPS_LOCAL_CORES and SELFLOOPS_DRIVER_MEMORY if
the defaults do not fit the host.

Unset SPARK_HOME and SPARK_CONF_DIR before calling this. A cluster conf that
requests a GPU per task cannot be satisfied in local mode, and the job then sits
at zero tasks with no error rather than failing::

    env -u SPARK_HOME -u SPARK_CONF_DIR SPARK_LOCAL_IP=127.0.0.1 \\
        python bin/selfloops.py <parquet> <out.json>
"""
import json
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if len(sys.argv) < 2:
    sys.exit(__doc__.strip().splitlines()[-1])

PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "selfloops.json"

CORES = os.environ.get("SELFLOOPS_LOCAL_CORES", "12")
DRIVER_MEMORY = os.environ.get("SELFLOOPS_DRIVER_MEMORY", "24g")

# Counts from run 20260901T195728Z, before the #360 fix. Kept as the shape of a
# regression, not as a pass mark: a rerun that matches these is a warning.
BASELINE = {"bls": 5988, "sec": 231, "market": 0, "noaa": 0}

spark = (
    SparkSession.builder
    .appName("selfloops")
    .master(f"local[{CORES}]")
    .config("spark.driver.memory", DRIVER_MEMORY)
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

triples = spark.read.parquet(PATH)

# One scan: per predicate, how many triples and how many say "X <p> X".
rows = [
    r.asDict() for r in
    triples.groupBy("predicate").agg(
        F.count("*").alias("edges"),
        F.sum(F.when(F.col("subject") == F.col("object"), 1).otherwise(0))
         .alias("self_loops"),
    ).filter(F.col("self_loops") > 0).orderBy(F.desc("self_loops")).collect()
]

prec_rows = [
    r.asDict() for r in
    triples.filter(F.col("predicate").endswith("precedes"))
           .groupBy("predicate").agg(
               F.count("*").alias("edges"),
               F.sum(F.when(F.col("subject") == F.col("object"), 1).otherwise(0))
                .alias("self_loops"),
           ).orderBy(F.desc("edges")).collect()
]

total_triples = triples.count()


def source_of(predicate):
    """The source name from the predicate's namespace, else the predicate itself."""
    for src in ("bls", "sec", "market", "noaa"):
        if f"/{src}/" in predicate or f"/{src}#" in predicate:
            return src
    return predicate


by_source = {}
for r in prec_rows:
    bucket = by_source.setdefault(source_of(r["predicate"]),
                                  {"edges": 0, "self_loops": 0, "predicates": 0})
    bucket["edges"] += r["edges"]
    bucket["self_loops"] += r["self_loops"]
    bucket["predicates"] += 1

total_loops = sum(r["self_loops"] for r in rows)

result = {
    "parquet": PATH,
    "total_triples": total_triples,
    "precedes_by_source": by_source,
    "precedes_by_predicate": prec_rows,
    "all_self_looping_predicates": rows,
    "total_self_loops_any_predicate": total_loops,
    "total_self_loops_precedes": sum(r["self_loops"] for r in prec_rows),
    "baseline_before_360_fix": BASELINE,
}
with open(OUT, "w") as fh:
    json.dump(result, fh, indent=2)

print(f"total triples {total_triples:,}")
print()
print("precedes, by source          edges     self-loops   was (before #360)")
for name, bucket in sorted(by_source.items(), key=lambda kv: -kv[1]["edges"]):
    was = BASELINE.get(name)
    print(f"  {name:10} {bucket['edges']:>14,} {bucket['self_loops']:>14,}"
          f" {'?' if was is None else format(was, ','):>16}")
print()
if rows:
    print("every predicate with at least one self-loop")
    for r in rows[:60]:
        share = 100.0 * r["self_loops"] / r["edges"] if r["edges"] else 0.0
        print(f"  {r['predicate'][:70]:70} {r['edges']:>12,} "
              f"{r['self_loops']:>10,} ({share:.4f}%)")
    print(f"({len(rows)} self-looping predicates in total)")
else:
    print("no predicate in the graph makes a node its own object")
print()
print(f"VERDICT: {total_loops:,} self-loops ({'PASS' if total_loops == 0 else 'FAIL'})")
print(f"wrote {OUT}")

spark.stop()
