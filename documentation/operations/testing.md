# Testing

Live pass/fail status is the **tests** badge on the [README](https://github.com/jeff1evesque/pyg-knowledge-graph-builder#readme), which reflects the latest [GitHub Actions](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/.github/workflows/tests.yml) run on the default branch.

The suite runs entirely against a **local `SparkSession`** — no Spark cluster and no RAPIDS Accelerator are required. The same application code runs unchanged on the GPU cluster (RAPIDS is a drop-in SQL plugin), so plain local Spark is a valid way to test the enrichment and PyG logic.

Tests are split into three **run groups** by pytest marker — how expensive a test is and therefore where it runs. (Depth of coverage is a separate axis; see [Test tiers](#test-tiers) below, which maps onto these three groups.)

- **Fast suite** — everything *unmarked*, selected with `-m "not e2e"`. Runs on a stock `ubuntu-latest` runner (Java 17 + Python 3.12) on **every push and pull request** ([`tests.yml`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/.github/workflows/tests.yml)); this is what the **tests** badge reflects.
- **End-to-end smoke** — marked `e2e`. Runs the real `build_graph` over small fixtures for all sources. It's heavy (~1,300 Spark stages regardless of data size) and does **not** reliably finish on a 7 GB GitHub runner, so it is **manual-only** ([`e2e.yml`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/.github/workflows/e2e.yml), `workflow_dispatch`) and best run locally / on a capable machine.
- **Cluster submit smoke** — marked `cluster` (and `e2e`, so `-m "not e2e"` excludes it too). Submits the real job to a real standalone cluster. Never runs in CI, and **skips itself** unless `SPARK_MASTER_URL` and `CLUSTER_SMOKE_OUTPUT_PATH` are set, so it costs contributors without a cluster nothing. See [Cluster smoke test](#cluster-smoke-test-a-real-cluster-not-local) below.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-test.txt          # runtime deps + pyspark + pytest

# Fast suite (what CI runs on every push/PR):
SPARK_LOCAL_IP=127.0.0.1 .venv/bin/python -m pytest tests/ -m "not e2e"

# ...or in parallel (local machines only — NOT what CI runs). bin/run_tests.sh
# is the wrapper for exactly this, and is the sibling of bin/run_e2e_tests.sh:
bin/run_tests.sh                      # the whole fast suite
bin/run_tests.sh tests/test_foo.py    # pytest args pass through

# It runs `-m "not e2e" -n 8`, and unsets SPARK_HOME with an empty
# SPARK_CONF_DIR so a shell that has sourced a cluster env cannot leak the
# standalone conf into local mode (which then asks for a per-task GPU local mode
# cannot satisfy, and hangs with 0 active tasks and no error).
#
# Each xdist worker is its own process and builds its OWN SparkSession, trading
# memory for wall-clock. Measured on this 20-core box: 1,178 tests, ~5m30s at 8
# workers — but only ~90s of that is user CPU, the rest being per-worker session
# startup. So raising the worker count buys little, and `-n auto` is worth
# avoiding outright since it silently scales with whatever machine it lands on.
# Set PYTEST_WORKERS to override the 8. The equivalent by hand:
SPARK_LOCAL_IP=127.0.0.1 .venv/bin/python -m pytest tests/ -m "not e2e" -n 8

# End-to-end pipeline smoke test (heavy; run locally / on a capable machine).
# bin/run_e2e_tests.sh wraps the env boilerplate below — SPARK_LOCAL_IP, the
# raised driver heap (the full pipeline fans out into ~1,300 stages and OOMs the
# default heap), and the RAPIDS toggle:
bin/run_e2e_tests.sh          # CPU (default)
bin/run_e2e_tests.sh gpu      # GPU via the RAPIDS Accelerator (auto-finds the jar)

# ...or invoke pytest directly. CPU:
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell" \
  .venv/bin/python -m pytest tests/ -m e2e -s

# ...the same e2e test on GPU via the RAPIDS Accelerator (requires a GPU + the
# RAPIDS jar). SPARK_RAPIDS=1 enables the plugin; point RAPIDS_JAR at the jar:
SPARK_RAPIDS=1 RAPIDS_JAR=/path/to/rapids-4-spark_2.12-<version>.jar \
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell" \
  .venv/bin/python -m pytest tests/ -m e2e -s
```

The `gpu` mode of [`bin/run_e2e_tests.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/run_e2e_tests.sh) resolves the RAPIDS jar via a glob (`/opt/spark/jars/rapids-4-spark_*.jar`, overridable with `RAPIDS_JAR` / `RAPIDS_JAR_DIR`), so a version bump on the host needs no edit.

The e2e test runs on **CPU by default**; set `SPARK_RAPIDS=1` to run it through the **RAPIDS Accelerator** on GPU. RAPIDS is a drop-in SQL plugin, so the application logic — and therefore every *pipeline* assertion — is identical on CPU and GPU; the toggle only changes where the DataFrame operators execute. (The one Python parsing UDF always runs on CPU under both.) The GPU settings mirror [`conf/spark-rapids.conf.template`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/conf/spark-rapids.conf.template) / [`bin/submit_spark_job.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/submit_spark_job.sh), minus the cluster-only GPU resource-scheduling confs that don't apply in `local[*]` mode.

Under `SPARK_RAPIDS=1` the suite additionally asserts that the query **really ran on the GPU** ([`tests/e2e/test_gpu_placement.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/tests/e2e/test_gpu_placement.py); skipped on CPU runs). This matters because a plugin that fails to load — a missing or mismatched jar, say — does not raise: Spark carries on correctly on the CPU and every pipeline assertion still passes, so the suite would report a green *GPU* run while the GPU sat idle. The check disables adaptive query execution before reading the plan: under AQE the plan renders as CPU (`isFinalPlan=false`) even when the GPU is executing it, so a plan inspected with AQE on cannot prove GPU placement either way.

> **Note — the GPU run is *slower* on these fixtures, and that is expected.** The `gpu` mode is a **correctness / plumbing sanity check** (does the pipeline produce the same graph through RAPIDS?), not a benchmark. On the tiny e2e fixtures GPU wall-clock is higher than CPU for two reasons: (1) a **one-time JIT compile** of GPU kernels when the RAPIDS jar ships no precompiled binaries for the local GPU architecture, and (2) the fixtures are so small that **per-operator GPU launch and host↔device transfer overhead dominates** any compute savings. GPU acceleration only pays off at cluster data scale, where those fixed costs amortize. Do not read the local e2e timing as a CPU-vs-GPU verdict. (You may also see recoverable `RMM ... maximum pool size exceeded` messages — that is the deliberately conservative `spark.rapids.memory.gpu.allocFraction` cap being hit; raise it for real workloads.)

`pyspark` is cluster-provided in production and is therefore not in `requirements.txt`; `requirements-test.txt` layers it (and `pytest`) on top for local and CI runs.

## Cluster smoke test (a real cluster, not `local[*]`)

Everything above runs against a local `SparkSession`. That leaves one path untested: **the job as actually submitted to a cluster**. Local mode reads none of the cluster's `spark-defaults.conf`, has no master or workers (so executor/GPU resource negotiation — the thing that decides whether a GPU job is ever *scheduled* — never happens), and imports the job in-process instead of zipping it and shipping it to executors. A cluster whose workers advertise no GPU will accept the job and simply never schedule it, hanging forever while the entire local suite stays green.

[`tests/e2e/test_cluster_submit.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/tests/e2e/test_cluster_submit.py) (marker: `cluster`) submits the real job through [`bin/submit_spark_job.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/submit_spark_job.sh) and asserts it completes, writes its artifacts to shared storage, **produces a structurally valid graph**, and **actually placed operators on the GPU**. The submission is bounded by a timeout, because an unschedulable GPU request hangs rather than failing. It is **skipped unless `SPARK_MASTER_URL` and `CLUSTER_SMOKE_OUTPUT_PATH` are set**, so CI and contributors without a cluster are unaffected.

It submits **two** jobs. `--mode enrichment_only` is the fast leg — the quick signal on launcher, submit, and IAM wiring. `--mode full` additionally runs PyG construction, which is the only place the **driver-side artifact writers** meet real object storage. That distinction is not academic: `save_pyg_local()` and `write_metadata_to_local()` both used `os.makedirs` + plain `open()`, and plain Python I/O treats `s3a://bucket/key` as a *relative path* — it creates a junk `./s3a:/bucket/key` tree under the driver's working directory, logs `Saved ... to s3a://...`, and exits 0. The graph and its metadata never reach the object store, and nothing raises. Because the cluster leg only ever ran `enrichment_only`, neither writer was ever invoked with a non-local URI and the defect survived undetected (it is the same defect `#197` fixed for the job manifest). All three writers now share one scheme-aware implementation, [`spark_jobs/utils/fs_utils.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/spark_jobs/utils/fs_utils.py) — **any new driver-side write must go through `write_bytes()`**, which routes local paths to direct I/O, URIs through the Hadoop FileSystem API, and *raises* rather than silently localizing a URI it cannot reach.

The `full` leg then **downloads its own artifacts back out of object storage and validates them**, using the very same `_assert_valid_graph_and_metadata` helper the local e2e suite uses — edge indices in range, tensor dtypes, finiteness, metadata/tensor agreement, node_index coverage, temporal structure, edge origins. Without this the cluster leg only ever asserted that *bytes arrived*: a run that produced a structurally broken graph passed every check, because the file was the right size in the right place. The helper is imported rather than reimplemented, so the cluster's graph is held to exactly the standard the local one is and the two cannot drift apart. The S3 tree is mirrored into a temp dir with its relative layout intact and handed a real `JobConfig`, so path derivation is the pipeline's own code rather than a second copy of it. Cost is ~2.5s of download and validation against a ~3-minute submission.

```bash
export SPARK_HOME=/opt/spark
export SPARK_MASTER_URL=spark://<master>:7077
export CLUSTER_SMOKE_OUTPUT_PATH=s3a://<bucket>/<prefix>   # must be reachable by EVERY node
.venv/bin/python -m pytest tests/e2e/test_cluster_submit.py -m cluster -q
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARK_MASTER_URL` | — | Required; the test skips without it. |
| `CLUSTER_SMOKE_OUTPUT_PATH` | — | Required. Where the job writes. On a cluster this must be shared storage (`s3a://`): executors are spread across machines, so a `file://` path resolves to a different local disk on each one and the commit protocol cannot assemble the output. |
| `CLUSTER_SMOKE_SOURCE_PATH` | staged fixtures | Source data. If unset, the repo's own e2e fixtures are uploaded under the output path, so a fresh clone can run this against any cluster with nothing copied onto the nodes by hand. |
| `CLUSTER_SMOKE_TIMEOUT` | `900` | Seconds before a hung submission is killed (with its driver). |
| `CLUSTER_SMOKE_MAX_SOURCES` | all | Cap the number of sources, for a faster signal. The pipeline's cost scales with **source count**, not data size — one source ≈ 30s, all seven ≈ 6min. Applies to both legs, so it roughly halves the suite's wall-clock. |
| `CLUSTER_SMOKE_EXPECT_GPU` | `1` | Set `0` to skip the GPU-placement assertion (e.g. a CPU-only cluster). |

Three failure modes this test exists to catch, all of which otherwise produce a **green** suite: RAPIDS silently falling back to CPU (the job still succeeds and still produces a correct graph), a job that writes a structurally broken graph to the right place under the right name (caught only by opening the `.pt`, which is what the validation step above added), and the driver OOMing while *planning* the multi-source query — see [`bin/submit_spark_job.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/submit_spark_job.sh) on `spark.sql.maxPlanStringLength`, and `_settle()` in [`spark_jobs/enrichment/pipeline.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/spark_jobs/enrichment/pipeline.py) on why the enrichment phases truncate their logical plans.

## Watching a long run for a stall

A cluster run can stop making progress without failing. The stage sits one task
short of done, both nodes go to ~99% idle, and nothing is raised: no exception,
no lost executor, no OOM, no failed task. Killing it produces a log that looks
like a crash and is not one, which is how three runs on 2026-09-06 produced
conclusions that were all wrong.

[`bin/stall_watchdog.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/stall_watchdog.py) watches for that state and
captures the evidence, so nobody has to be at the keyboard when it happens:

```bash
bin/stall_watchdog.py --host <driver-host> --out <run-dir>/stalls
```

It polls the driver's REST API and fires only when a stage has stopped
completing tasks **and** has one running past that stage's own longest completed
task — quiet alone is ambiguous, because a stage draining its last few tasks is
quiet too. A stage where nothing has finished yet has no such yardstick; there it
fires only once a running task has itself passed `--stall-seconds`. Without that
floor, any stage slow to finish its first task gets reported as stalled, naming
tasks that started seconds earlier — which happened twice on 2026-09-06 and cost
a run that had actually succeeded its outcome report, because the harness around
it treats a capture as proof. On a confirmed stall it pulls thread dumps from every executor twice,
30 seconds apart, alongside the stuck task ids and host memory. A thread present
in both dumps is stuck; one present in a single dump was merely slow.

It also captures what a JVM dump cannot see, for each host holding a stuck task:
the socket queues from `ss -tn`, and every Python worker's state and accumulated
CPU. Those two readings are what identified the deadlock in
[#386](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/issues/386) —
about 4 MB queued in *both* directions of one loopback connection at once, and
workers asleep having burned no CPU — and both are gone the moment anybody kills
the job. The worker's Python stack still needs `sudo py-spy dump --pid <worker>`
by hand while the stall is live: the executors run as another user under
restricted ptrace, which the watchdog has no way around.

Everything it does is a read — HTTP GETs plus a few local files, and one `ssh`
when the stuck task is on another node — so it never touches the job. It re-resolves the application between legs, so one invocation
covers a whole notebook run, and it costs well under 100 KB of log per run.

This is how #380 was diagnosed: the dumps showed the executor's reader thread in
`BasePythonUDFRunner.read` and its writer thread in `PythonRDD.write`, both
blocked on the same Python worker, with the worker itself burning no CPU — a
deadlock, not a slow parse. See `turtle_batches_to_arrow` in
[`spark_jobs/graph/turtle.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/spark_jobs/graph/turtle.py) for what caused it.

It then caught the same defect a second time, which is the better argument for
keeping it. After that first fix a run stalled again at stage 13, 168 of 169
tasks done, and the dumps showed the two blocked threads had simply moved to
RAPIDS' GPU Arrow runner — `GpuArrowPythonOutput.read` against
`GpuArrowWriter.write`. Bounding the batches had made the stall rare, not gone.
`spark.rapids.sql.exec.PythonMapInArrowExec=false`, which
[`bin/submit_spark_job.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/submit_spark_job.sh) now sets by default, moves
the parse off that runner.

**It is not fixed, and this section said otherwise for a day.** On 2026-09-07 the
same stall happened with that setting confirmed in the submit line, and the
blocked frames were Spark's own `ArrowPythonRunner` and `PythonArrowInput` — no
`Gpu*` frame anywhere. The setting did what it promised; the deadlock followed
the parse onto the CPU runner. Anything resting on "off the GPU it cannot occur"
is unsound. See
[#386](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/issues/386).

A related trap worth knowing when reading any of these captures: they are always
reported as "168 of 169". That is not a failure near the end. The stuck task
launches in the first wave, milliseconds after the stage is submitted, and the
other 168 finish and stream past it — so the counter parks at N-1 whichever task
was hit. The progress number describes what survived, not when it broke.

## The parse stall

For weeks the seed leg would hang at 168 of 169 tasks in the stage that
materialises the parse, and never clear. It looked like a pure race: the same
job read the same mirror twice within half an hour, with byte-identical Spark
properties, and stalled on one attempt while clearing the same stage in 68.7s on
the next. Three successive fixes were each declared verified on one clean run,
and each came back.

The cause was sitting in the thread dumps
[`bin/stall_watchdog.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/stall_watchdog.py) had been collecting all along.
Two threads on the executor wedge against each other:

- the task thread, blocked in `SocketInputStream.read` reading Arrow results back
  from the Python parse worker, *inside* `MemoryStore.putIteratorAsValues`;
- `stdout writer for python`, blocked in `SocketOutputStream.write` holding the
  stream monitors, writing the next input batch *to* Python.

The reader stops draining Python's output while it unrolls rows into the memory
store. Python's output buffer fills, so Python stops reading its input, so the
writer blocks too. Both directions are full and nothing breaks the tie.

The two captures show this with *different* Python runners — one
`GpuArrowPythonRunner`, one `BasePythonUDFRunner` — which is why "keep mapInArrow
off the GPU" swapped the runner and the stall followed it. The runner was never
the cause. The shared consumer was, and that consumer existed only because the
parse frame was `.cache()`d: #380 converted five other frames and this file's
*enriched* frame to `DISK_ONLY`, and left the parse frame behind. It is
`DISK_ONLY` now — `doPutIterator` branches on `level.useMemory`, so the unroll
never runs.

`--mode parse_only` remains, because it is still the cheapest way to exercise the
parse: it stops at the count that materialises it and writes nothing, so it
reaches a verdict in under two minutes against the three hours a full run needs
to reach the same point once.

## Driving a real cluster run

[`bin/run_cluster_notebook.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/run_cluster_notebook.sh) runs the experiment
notebook against the cluster and leaves a report behind however it ends:

```bash
bin/run_cluster_notebook.sh <run-dir>
```

The run directory holds everything about *this run* and nothing about the code:
an untracked `env.sh` with the addresses, paths and intent — copy
[`bin/profiles/run-env.example.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/profiles/run-env.example.sh), which
documents every variable — and afterwards the executed notebook, run log,
traces, event log and `outcome.txt`. Acceptance checks for whatever the run is
meant to prove go in `<run-dir>/extra-checks.sh`, from
[`bin/profiles/extra-checks.example.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/profiles/extra-checks.example.sh).
The same split as the sizing profiles: what is general is tracked, what
identifies a deployment is sourced beside it.

It starts a 1 Hz network trace on every node ([`bin/netsample.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/netsample.py))
and a cluster sampler locally ([`bin/cluster_sampler.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/cluster_sampler.sh)),
runs the notebook through [`bin/execute_notebook.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/execute_notebook.py) —
which rewrites the executed `.ipynb` after every cell, so a run that dies in the
seed still leaves a report — kills the run if the round trip to the gateway
starts climbing, stops early if the stall watchdog captures a stalled stage, and
records the outcome on both paths before sweeping this run's checkpoints.
[`bin/mem_reclaim.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/mem_reclaim.py) is the gate to run before it, on a
unified-memory host: RAPIDS sizes its pool from `MemFree`, which a previous run's
file cache holds near zero for hours while `MemAvailable` still reads over 100 GB.

This used to live only as a copy inside each run directory, copied forward from
whichever run came before, and three silent failures on one run are why it does
not any more: a stop flag cleared on one node but set on two, so a second node's
trace described a different run for two days; a checkpoint sweep keyed on a
literal path carrying the previous run's name, which therefore matched nothing
and once left 462 GB behind; and no recording at all when the harness gave up,
69 minutes before the job went on to succeed. Everything this run owns now
carries its run id, one supervisor may hold a run directory at a time, and
`tests/test_run_cluster_notebook.py` pins all three against stub binaries.

## Recording what a run did

A run's own log is not the record. The notebook runner puts each cell's output in
the executed `.ipynb`, so grepping the run log shows a clean run no matter what
happened — that is how a run whose seed died at 47 minutes, and whose three
experiments then failed outright, was once written down as "rc=0, errors: none".

[`bin/record_run_outcome.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/record_run_outcome.sh) reads the run directory
and writes `outcome.txt` beside it:

```bash
bin/record_run_outcome.sh <run-dir> [work-dir]
```

It flattens the executed notebook into text first, then reports each
submission's status, the phases the job logged, how many tasks each stage really
ran at once (from the event log — the only place that exists), a ranked error
scan, executor memory-pool pressure per node, and the network traces. A trace
whose samples fall outside the run window is called out as stale rather than
summarised, because a leftover file from an earlier attempt otherwise produces a
confident set of numbers about a different run.

Everything it does is a read, and it does not care how the run ended — **the run
whose harness gave up is the one whose report is worth having**. Whatever a
particular run was meant to prove goes in `<run-dir>/extra-checks.sh`, which it
sources at the end, so this file stays the same from one run to the next.

[`bin/selfloops.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/selfloops.py) is the acceptance check for sequencing:
it counts every predicate that makes a node its own object, and breaks
`precedes` out by source.

```bash
env -u SPARK_HOME -u SPARK_CONF_DIR SPARK_LOCAL_IP=127.0.0.1 \
    .venv/bin/python bin/selfloops.py <enriched-triples-parquet> selfloops.json
```

Unsetting `SPARK_HOME` matters: a cluster config asks for a GPU per task, which
local mode can never satisfy, and the job then sits at zero tasks with no error
rather than failing. Watch the edge counts as well as the loop total — a loop
count that falls to zero because the edge count collapsed means a dedupe took
rows it should have kept.

## Test tiers

Test depth is calibrated to risk rather than applied uniformly — deeper coverage only where the logic is genuinely subtle, to keep maintenance debt proportional to value.

Tiers are a **second, finer axis** than the three run groups above, not a competing one. A marker answers *where and when a test runs*; a tier answers *how deep it cuts*. The two line up cleanly — tiers **1–4 together are the fast suite**, and each of the last two tiers is one marker-gated group of its own:

| Run group (from above) | Tiers it contains | Marker |
|---|---|---|
| Fast suite — every push/PR | 1, 2, 3, 4 | *(none)* — selected with `-m "not e2e"` |
| End-to-end smoke — manual-only | 5 | `e2e` |
| Cluster submit smoke — opt-in | 6 | `cluster` (+ `e2e`) |

Each tier below repeats its run group, so no row has to be cross-referenced against the table above:

| Tier | Scope | Examples |
|------|-------|----------|
| **1 — pure / no-Spark**<br>*fast suite* | Import-time integrity, vector geometry, hand-maintained pattern tables, and RDF parse determinism. Sub-second. | `test_imports.py` (imports every `spark_jobs` module), `test_vector_layout.py` / `test_edge_vector_layout.py` (`VectorLayout` / `EdgeVectorLayout` boundaries), `test_source_patterns.py` (NOAA/market/SEC pattern-dict integrity), `test_bnode_determinism.py` (blank-node labels are content-derived, so the same Turtle parses identically every time — rdflib's own labels are random per parse), `test_sparse_scatter.py` (`scatter_sparse_entries()` — which dims it drops, which bad keys it lets raise), `test_executor_imports.py` (`graph/turtle.py` is imported by executors for real, so its transitive imports must stay inside what `requirements-executor.txt` ships — a driver-only import passes every other test and then fails every parse task) |
| **2 — linker smokes**<br>*fast suite* | Each enrichment module's `enrich()` driven end-to-end over tiny in-memory triples: one happy path + one short-circuit (foreign input for the intra-source linkers; a single detected source, and two sources with nothing linkable, for the cross-source linker; non-temporal input for the temporal unifier). | `test_{bls,noaa,market,sec}_linker.py`, `test_cross_source_linker.py`, `test_temporal_unifier.py` |
| **3 — targeted deep**<br>*fast suite* | One focused test on each module's trickiest computation (including the negative case), where a silent regression would be costly. | severity escalation (NOAA), option moneyness (market), CIK unification (SEC), temporal sequencing (BLS), state-FIPS geographic chain (cross-source), expiration-date period derivation (temporal unifier) |
| **4 — construction internals**<br>*fast suite* | Value-level unit tests of the PyG construction modules — exact node IDs, edge-index contents + `(src_id, dst_id)` ordering, config filters, determinism, the node/edge feature **encoding** (a known triple lands in the layout-reserved vector slot with the expected value: class-identity/categorical multi-hots, depth-weighted `subClassOf` class hierarchy, property-schema presence/domain-range/property-hierarchy slots, z-score numeric normalization, edge temporal/numeric-contrast/moneyness signals, label-similarity Jaccard on correlation edges, the escalation severity-delta fallback, plus `_classify_relation` per category and a categorical-determinism guard), the final `build_hetero_data` assembly (feature↔node-ID alignment via encoding-independent sentinels, edge endpoints within per-type ID ranges, per-type counts), and the six metadata JSON files' content (keys, counts, type names, feature-segment structure) — asserting what the `e2e` smoke only checks structurally or for presence. Runs in the fast suite; `test_metadata_writer.py` is pure-Python (no `SparkSession`). | `test_node_mapper.py`, `test_edge_mapper.py`, `test_feature_extractor.py`, `test_edge_feature_extractor.py`, `test_constructor.py`, `test_metadata_writer.py` |
| **5 — pipeline smoke**<br>*`e2e` marker — manual-only* | The real `build_graph` end-to-end over tiny committed RDF fixtures: both source loaders (`.nt` and turtle-parquet) × all three modes (`full`, and the `enrichment_only` → `pyg_only` split), asserting a valid `.pt` + all six metadata JSONs and layout-consistent tensor shapes. Also a **reproducibility** check that runs the pipeline twice (each in its own process, to isolate the per-run driver heap; the two independent runs are launched **concurrently**, and started early so they overlap the rest of the e2e suite rather than adding to it — cutting the twin-run cost from ~103s to ~21s of join time, at the price of two extra driver JVMs alive during the overlap. Set `TWIN_RUNS_SEQUENTIAL=1` to run them one-at-a-time on a memory-constrained host) and asserts the two graphs, feature tensors, and metadata are identical — an integration property no unit test can see. The check is split in two: `test_output_is_reproducible` compares the node/edge inventory, per-type counts, `edge_index`, edge features, **every** node type's feature tensors, and each metadata file exactly (modulo the allow-listed `build_timestamp`); `test_jolts_features_reproducible` additionally compares the metadata in **serialized** form, which catches dict key/insertion-order drift the parsed comparison cannot. Both carried concessions to a known non-determinism until it was fixed — `jolts_*` tensors were skipped and metadata was compared order-insensitively — so the guard is now strictly stronger than before that bug existed. The cause was twofold: rdflib assigns blank nodes a fresh **random** label on every parse (which then hash into different feature slots), and the metadata builders used unordered `collect()` (Spark returns rows in task-completion order). Blank-node determinism is additionally guarded by fast unit tests that *do* run in CI (`tests/test_bnode_determinism.py`), since this e2e suite does not. Catches wiring / API-mismatch bugs the unit tiers can't. | `tests/e2e/test_pipeline_smoke.py` |
| **6 — cluster submit smoke**<br>*`cluster` marker — opt-in* | The real job submitted to a real standalone cluster through `bin/submit_spark_job.sh`: asserts it completes, writes artifacts to shared storage, and ran on the GPU. The only tier that exercises `--py-files`/venv packaging, executor↔GPU resource negotiation, and the cluster's own `spark-defaults.conf`. Skipped unless `SPARK_MASTER_URL` is set. | `tests/e2e/test_cluster_submit.py` |

Exhaustive per-relationship assertions are intentionally **not** written — the targeted deep tests capture the high-risk logic without the brittleness of pinning every output.

## Local test report

[`bin/generate_report.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/generate_report.sh) runs the suites and writes a combined report to `reports/tests/report.html` + `reports/tests/report.json` (same data, two formats), overwriting the previous pair each run. (Reports are namespaced by kind under `reports/` — e.g. `reports/tests/` — to leave room for other report types.)

```bash
bin/generate_report.sh            # fast + e2e (CPU) [+ e2e (GPU) if a GPU + RAPIDS jar are present]
bin/generate_report.sh --no-gpu   # never attempt the GPU run

# Full picture: also submit the real job to a standalone cluster (adds the cluster suite).
SPARK_MASTER_URL=spark://<master>:7077 \
CLUSTER_SMOKE_OUTPUT_PATH=s3a://<bucket>/<prefix> \
  bin/generate_report.sh
```

One variable has to be **absent**: `PYG_INPUT_MODE` must not be `local`. The cluster test passes its whole environment to `bin/submit_spark_job.sh`. If that variable says `local` — left over from a [staged run](running-a-job.md#reading-sources-from-local-disk) — the launcher copies this suite's fixtures onto local disk and the job reads them from there. The suite still passes, but it stops testing the `s3a://` read path, which no other suite covers. Set `PYG_INPUT_MODE=s3` for the report run. `bin/generate_report.sh` warns when it sees this.

This is **local-only and not committed** — `reports/tests/` is gitignored. The e2e suite is too heavy for the GitHub Actions runners, so the report can't be produced in CI; instead each developer runs `bin/generate_report.sh` on their own branch to verify locally (the **tests** badge covers the fast suite in CI). The report always includes the fast unit suite and the CPU e2e run; it adds the GPU (RAPIDS) e2e run when the hardware is available, and the [cluster smoke test](#cluster-smoke-test-a-real-cluster-not-local) when `SPARK_MASTER_URL` and `CLUSTER_SMOKE_OUTPUT_PATH` are set (source data auto-stages, so `CLUSTER_SMOKE_SOURCE_PATH` is optional). [`bin/generate_test_report.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/generate_test_report.py) is the underlying renderer (reads pytest JUnit XML).

