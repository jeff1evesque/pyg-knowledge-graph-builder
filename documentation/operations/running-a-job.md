# Running a Job

How to configure a run, what each parameter does, and what the cluster needs
before a large one.

## PyG Configuration

The PyG builder accepts an optional configuration dict:

```json
{
    "node_types": ["cpi_Index", "ppi_MonthlyChange", "market_PriceObservation"],
    "edge_types": ["bls_enrichment_precedes", "bls_enrichment_correlatesWith"],
    "feature_config": {
        "normalize": true,
        "vector_dim": 1024,
        "chunk_node_threshold": 500000
    },
    "edge_feature_config": {
        "enabled": true,
        "edge_vector_dim": 32,
        "chunk_edge_threshold": 1000000,
        "max_edge_types_per_batch": 8,
        "enabled_categories": ["temporal", "option_stock", "escalation"]
    },
    "include_temporal_nodes": true,
    "include_sector_nodes": true
}
```

| Config key | Default | Description |
|-----------|---------|-------------|
| `node_types` | All rdf:type classes | Whitelist of PyG node type names to include |
| `edge_types` | All entity-to-entity predicates | Whitelist of relation names to include |
| `feature_config.normalize` | `true` | Z-score normalize numeric features (single-pass per-predicate) |
| `feature_config.vector_dim` | `1024` | Node feature vector dimension — all segments scale proportionally. Minimum 32. |
| `feature_config.chunk_node_threshold` | `500000` | Node count above which chunked collection is used |
| `--class_mappings` (CLI) | built-in table | JSON object (or path to a JSON file) of `{source_class_uri: target_class_uri}` merged over `ontology_mapper.CLASS_MAPPINGS`. A `null` target drops a built-in entry. Lets a new source declare its class semantics without editing the module. Where two or more sources share a target the relationship is published as `rdfs:subClassOf`, not `owl:equivalentClass`. |
| `feature_config.numeric_predicate_min_share` | `0.5` | Share of a property's literal values that must parse as numbers for the property to be treated as numeric. At or below it, every value is a category label. Lower it to admit sparsely numeric properties; raise it toward `1.0` to demand near-uniformly numeric ones. |
| `edge_feature_config.enabled` | `true` | Enable/disable edge feature extraction entirely |
| `edge_feature_config.edge_vector_dim` | `32` | Edge feature vector dimension — all segments scale proportionally. Minimum 9. |
| `edge_feature_config.chunk_edge_threshold` | `1000000` | Edge count above which chunked collection is used |
| `edge_feature_config.max_edge_types_per_batch` | `8` | Maximum edge types unioned into one batched collect. Caps the width of the batch's Catalyst plan, which the edge budget alone does not — many tiny types satisfy any edge budget while still producing a plan large enough to exhaust the driver during plan rendering. |
| `edge_feature_config.enabled_categories` | `["temporal", "option_stock", "escalation"]` | Which edge categories receive features. Options: `temporal`, `option_stock`, `escalation`, `correlation`, `causal`, `strategy` |
| `include_temporal_nodes` | `true` | Include Month/Year/Quarter node types |
| `include_sector_nodes` | `true` | Include EconomicSector node types |

When config is empty, sensible defaults are inferred from the data.

## Job Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--mode` | Yes | `full` | `full`, `enrichment_only`, or `pyg_only`. `parse_only` also exists but is a diagnostic, not a pipeline stage: it stops at the count that materialises the parse and writes nothing — see [The parse stall](testing.md#the-parse-stall) |
| `--source_paths` | Modes 1,2 | — | Comma-separated source path(s)/URI(s): local directories or `s3a://...`. Each is loaded independently and the results are unioned into a single triples DataFrame before enrichment. A path naming the archive's `source=sec` partition must also name `feed=filings`: that is the only SEC feed carrying RDF, and the job rejects the other seven up front rather than failing later on a missing Turtle column |
| `--input_mode` | No | `s3` | Where `--source_paths` are opened from. `s3` reads the `s3a://` URIs directly — correct in the cloud, where executors sit beside the bucket. `local` reads a mirror of those same objects from node-local disk instead; see [Reading sources from local disk](#reading-sources-from-local-disk) |
| `--local_source_root` | When `--input_mode local` | — | Root of the staged mirror. Must exist at the same path on every worker |
| `--local_work_dir` | Yes | — | Working directory for the interim enriched Parquet and the final artifacts. Must be reachable by every worker — a shared mount (e.g. NFS) or a URI on shared storage (`s3a://...`); on a multi-node cluster a driver-local path won't do |
| `--s3_archive_bucket` | No | `""` | Optional S3 bucket to *additionally* mirror the final artifacts (`.pt` + metadata + manifest). When empty, artifacts are written only to `--local_work_dir` (which may itself be an `s3a://` path) |
| `--s3_pyg_key` | No | `pyg/{time_period}/{pyg_filename}` | Optional S3 key for the archived `.pt`; the metadata prefix is derived from it |
| `--pyg_filename` | No | `hetero_data.pt` | Local `.pt` filename (override for experiment variants, e.g. `hetero_data_512d.pt`); determines the metadata directory name |
| `--enable_ontology_mapping` | No | `true` | Run the ontology-mapping phase: equivalences, predicate folding, and the derived `rdfs:subClassOf` hierarchy that fills the `class_hierarchy` sub-segment. Applies to modes `full` and `enrichment_only`; `pyg_only` never reaches this phase, so the flag is inert there (and meaningless in that mode's manifest — see [`ontology_schema.json`](../reference/outputs.md#ontology_schemajson)) |
| `--time_period` | No | Current `YYYY-MM` | Time period label for output paths |
| `--pyg_config` | No | `{}` | JSON string with PyG construction config |
| `--parquet_partitions` | No | `200` | Number of Parquet output partitions |
| `--source_format` | No | `ntriples` | Source RDF format: `ntriples` (one triple per line in `.nt` files) or `turtle_parquet` (self-contained Turtle blobs in a Parquet column). Applies to modes `full` and `enrichment_only` only — `pyg_only` always reads enriched Parquet written by this pipeline |
| `--turtle_column` | No | *(auto)* | Column name containing Turtle strings when `--source_format=turtle_parquet`. Ignored for `ntriples` format. Left unset, the column is resolved **per source** against `TURTLE_COLUMN_CANDIDATES` (`triples`, then `rdf_turtle`), so one run can span sources whose schemas disagree; set it to force a single name everywhere |
| `--market_sector_definitions_bucket` | No | `""` | S3 bucket holding the S&P 500 constituents CSV. **Set this for real runs** — three cross-source links are empty or degraded without it; see the note under Cross-Source Linking |
| `--market_sector_definitions_key` | No | `""` | S3 key for the S&P 500 constituents CSV. Supplies three things: the ticker to company-ID map that keys the company bridge, the GICS sector classification, and the sub-industry peer links. Without it the first and third are empty and sector classification falls back to a small built-in list |

Metadata files are always written when mode is `full` or `pyg_only`. Mode `enrichment_only` does not produce metadata files (no PyG graph is built in that mode).

Jobs are launched with `bin/submit_spark_job.sh`, which packages the code
and submits to the Spark standalone master with the RAPIDS Accelerator
enabled. Set `SPARK_MASTER_URL` (and optionally `RAPIDS_JAR`); see the
launcher header for all environment variables.

**Example — full pipeline (local source, local + S3 archive):**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode full \
    --source_paths /data/rdf/monthly/2024-12/ \
    --local_work_dir /data \
    --s3_archive_bucket my-archive \
    --s3_pyg_key pyg/year=2024/month=12/hetero_data.pt \
    --enable_ontology_mapping true \
    --time_period 2024-12 \
    --parquet_partitions 200 \
    --pyg_config '{"feature_config": {"normalize": true, "vector_dim": 1024}, "edge_feature_config": {"enabled": true, "edge_vector_dim": 32}}'
```

Outputs (local; and mirrored to `s3://my-archive/...` because an archive
bucket was given):
```
/data/enriched/year=2024/month=12/triples/            # interim, local only
/data/pyg/year=2024/month=12/hetero_data.pt
/data/pyg/year=2024/month=12/metadata/graph_schema.json
/data/pyg/year=2024/month=12/metadata/feature_spec.json
/data/pyg/year=2024/month=12/metadata/normalization.json
/data/pyg/year=2024/month=12/metadata/encoding_config.json
/data/pyg/year=2024/month=12/metadata/ontology_schema.json
/data/pyg/year=2024/month=12/metadata/slot_mapping.json
```

**Example — reduced-dimension experiment from existing enriched Parquet:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode pyg_only \
    --local_work_dir /data \
    --pyg_filename hetero_data_512d.pt \
    --time_period 2024-12 \
    --pyg_config '{"feature_config": {"vector_dim": 512, "normalize": true}, "edge_feature_config": {"edge_vector_dim": 16}}'
```

Outputs:
```
/data/pyg/year=2024/month=12/hetero_data_512d.pt
/data/pyg/year=2024/month=12/hetero_data_512d_metadata/graph_schema.json
/data/pyg/year=2024/month=12/hetero_data_512d_metadata/feature_spec.json
... (remaining four files)
```

**Example — additional edge feature categories:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode pyg_only \
    --local_work_dir /data \
    --pyg_filename hetero_data_full_edge_features.pt \
    --time_period 2024-12 \
    --pyg_config '{"edge_feature_config": {"enabled_categories": ["temporal", "option_stock", "escalation", "correlation", "causal"]}}'
```

**Example — edge features disabled:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode pyg_only \
    --local_work_dir /data \
    --pyg_filename hetero_data_no_edge_features.pt \
    --time_period 2024-12 \
    --pyg_config '{"edge_feature_config": {"enabled": false}}'
```

**Turtle Parquet source (SEC filings), reading from S3 via s3a://:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode enrichment_only \
    --source_paths s3a://my-data-lake/raw/sec/filings/2024-12/ \
    --local_work_dir /data \
    --source_format turtle_parquet \
    --turtle_column triples \
    --time_period 2024-12 \
    --parquet_partitions 200
```

If your Parquet column is named something other than `triples`, nothing needs to be
said: with `--turtle_column` unset the loader resolves the name against
`TURTLE_COLUMN_CANDIDATES` for each source path independently, so
`triples` and `rdf_turtle` sources can be submitted together and enrich into a single
graph. Pass `--turtle_column rdf_turtle` only to force one name across every source —
useful when a schema carries both columns and only one of them is meant, and the source
of truth when a third name appears.

Source Parquet written by pandas/pyarrow commonly carries **nanosecond** timestamps,
which Spark's Parquet reader rejects outright (`Illegal Parquet type: INT64
(TIMESTAMP(NANOS,true))`) during schema conversion — before a single triple is read, and
regardless of the fact that this pipeline goes on to select nothing but the Turtle
column. [`bin/submit_spark_job.sh`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/submit_spark_job.sh) therefore sets
`spark.sql.legacy.parquet.nanosAsLong=true` by default; set `PARQUET_NANOS_AS_LONG=false`
to restore Spark's behavior.

**Full pipeline from a Turtle Parquet source:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode full \
    --source_paths s3a://my-data-lake/raw/sec/filings/2024-12/ \
    --local_work_dir /data \
    --source_format turtle_parquet \
    --turtle_column triples \
    --time_period 2024-12 \
    --parquet_partitions 200 \
    --pyg_config '{"feature_config": {"normalize": true, "vector_dim": 1024}}'
```

## Reading sources from local disk

The Turtle parse is a Python UDF running rdflib, so RAPIDS cannot place it on a
GPU — but Spark applies one resource profile to the whole SQL job, and that
profile reserves a quarter GPU per task. With one GPU per executor and two
executors, the parse was pinned at **8 concurrent tasks on a cluster advertising
144 slots**.

Read that 144 as slots, not hardware. It is `SPARK_WORKER_CORES` summed over the
workers, and that setting is a declaration rather than a count — the cluster these
measurements come from declares 72 per node against 20 physical cores, so most of
the headroom the cap appeared to waste was never there. Loosening the reservation
is still the lever, but how far is a measurement. The same day, parsed at three slot
counts: 8 slots took 4,299.7s; 64 slots (`GPU_PER_TASK=0.03125`, one GPU per
executor) took 1,408.3s; and `GPU_PER_TASK=0`, which declares no GPU requirement at
all so the slot count falls back to the advertised 144, parsed in 1,570.9s and then
exhausted host RAM and lost the run — slower than 64 slots despite more than twice
as many, because the extra ones are oversubscription against cores that are already
outnumbered. Shrinking `RAPIDS_BATCH_SIZE_BYTES` to `256m` let those 144 slots
survive, at 4,833.8s — slower than the 8 they replaced, because batch size and slot
count trade against one fixed host-RAM budget. Take the most slots that still fit at
the default batch.

Dropping the reservation also takes the phase from 8 to 144 concurrent
object-storage readers, and on a deployment whose workers reach the bucket over a
site uplink rather than an in-region link, that is enough to take the network down.
Measured twice on a two-node cluster: within 60 seconds the gateway's flow table
went from ~330 to ~1,900 tracked connections and both nodes lost their default route
within a second of each other, mid-parse.

The count of *established* connections was not the problem — it stayed flat at
around 155. **Churn** was. Reads slower than `fs.s3a.connection.timeout` are
aborted and redialled, so the more congested the link becomes, the faster new
connections are created.

`--input_mode local` removes the uplink from the job's critical path. Mirror the
source prefixes to every worker once, then run the parse against local disk:

```bash
# once per cluster: the mirror is written by the account that runs the staging
# script and read by the account the executors run as, which is usually not the
# same one
sudo install -d -o "$(id -un)" -g "$(id -gn)" -m 755 /srv/pyg-source   # on every worker

# once per dataset; incremental afterwards
bin/stage_sources.sh \
  --dest /srv/pyg-source \
  --nodes worker-a,worker-b \
  --source s3a://my-data-lake/raw/source=sec/feed=filings/2024-12/

# then submit as usual, naming the same s3a:// URIs
SPARK_MASTER_URL=spark://<host>:7077 \
PYG_INPUT_MODE=local PYG_LOCAL_SOURCE_ROOT=/srv/pyg-source \
PYG_STAGE_NODES=worker-a,worker-b \
  bin/submit_spark_job.sh \
    --mode full \
    --source_paths s3a://my-data-lake/raw/source=sec/feed=filings/2024-12/ \
    --local_work_dir /data \
    --source_format turtle_parquet \
    --time_period 2024-12
```

With `PYG_INPUT_MODE=local` the launcher stages before it submits and appends
`--input_mode local --local_source_root ...` for you, so the mode is stated once.
`PYG_STAGE_ENABLED=false` skips the sync and reads a mirror staged earlier.

Points worth knowing:

- **The job keeps naming the `s3a://` URIs.** `bin/stage_sources.sh` and
  `build_graph.py` derive the same local path from them
  (`<root>/<bucket>/<key>`), so both modes read the same layout and report the
  same per-source statistics. Switching modes does not change the graph.
- **Every worker needs its own copy at the same path.** No shared filesystem is
  wanted here — one shared device would serialize reads that are otherwise
  parallel per node. Spark lists the input on the driver and each task then opens
  the path on whichever node it landed on, so divergent copies do not fail loudly;
  they build a graph from a mixture. The staging script compares a content digest
  across nodes and refuses to report success if they differ.
- **The staging step is not a Spark job.** A distributed copy is the same failure
  with a different label. It transfers one node at a time with a bounded number of
  connections (`--concurrency`, default 4, hard-capped at 16), and watches
  round-trip time to the default gateway — three consecutive samples above
  `--rtt-limit-ms` and it kills the transfer. `aws s3 sync` is incremental, so a
  retry resumes.
- **Re-running downloads nothing.** `aws s3 sync` is incremental on its own, but
  the script skips even the per-object comparison: the preflight `LIST` already
  knows the prefix's object count and total size, so one local walk per node
  decides whether there is anything to fetch. Ten runs against the same day cost
  one transfer and nine listings — measured at 4.5s cold and 0.49s warm on an
  89-object prefix. Immutable snapshots are the assumption; pass `--force` for a
  prefix that gets rewritten in place.
- **`s3` stays the default.** Nothing changes for a deployment that has not staged,
  and reading object storage directly remains correct where compute sits beside
  the bucket.

`notebook/multi_experiment.ipynb` needs no change: it submits through
`bin/submit_spark_job.sh` with the inherited environment, so exporting
`PYG_INPUT_MODE` and `PYG_LOCAL_SOURCE_ROOT` before starting Jupyter is enough.
The seed leg stages and reads locally; the `pyg_only` experiment legs pass no
`--source_paths`, so they skip staging and read the enriched Parquet as before.

## Cluster prerequisites for GPU runs

Five cluster-side settings decide whether this job runs at all — or whether it
merely *appears* to. They share an unpleasant property: when any of them is wrong,
the job **hangs indefinitely with no error message** (or, for the fourth, silently
runs on one node; for the fifth, silently runs many times slower) rather than
failing, so they are worth checking before you conclude the job itself is slow.

**1. Workers must advertise their GPUs.** The RAPIDS configuration makes every
executor request a GPU (`spark.executor.resource.gpu.amount`). On a standalone
cluster the worker must independently declare that it *has* one, or the master
can never satisfy the request and the application waits forever without ever
scheduling a task:

```bash
# in $SPARK_HOME/conf/spark-env.sh on each worker
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 \
  -Dspark.worker.resource.gpu.discoveryScript=$SPARK_HOME/examples/src/main/scripts/getGpusResources.sh"
```

**2. Buckets whose names contain dots need path-style S3A access.** With S3A's
default virtual-host addressing the request is sent to
`<bucket>.s3.<region>.amazonaws.com`. If the bucket name contains dots (for
example a bucket named after a domain), that hostname has more labels than the
`*.s3.<region>.amazonaws.com` wildcard certificate covers, and TLS fails with
`SSLPeerUnverifiedException` before any S3 call is made:

```
spark.hadoop.fs.s3a.path.style.access    true
```

Note that a working `aws s3 ls` proves nothing here: `boto3` and the AWS CLI
switch to path-style automatically for dotted buckets, while S3A does not.

**3. Cap the RAPIDS pool on GPUs whose memory is shared with the host.** RAPIDS
defaults to pooling essentially all available GPU memory, which is correct for a
discrete card with dedicated VRAM. On integrated or unified-memory GPUs, that
memory is the host's RAM — the default can reserve nearly all of it, starve the
OS and the JVM, and drive the machine into swap:

```
spark.rapids.memory.gpu.allocFraction    0.25
```

The pool has a second, harder bound. RAPIDS sizes it as
`(gpu.free - reserve) * allocFraction` but caps it at `gpu.total * maxAllocFraction`,
and refuses to start when the first exceeds the second — so **raising
`allocFraction` alone cannot grow the pool past the cap**. To genuinely want a
bigger pool, raise both.

Note the two fractions are taken against *different* quantities: `allocFraction`
scales free memory, the cap scales total. An `allocFraction` above
`maxAllocFraction` is therefore legal whenever enough memory is already in use —
which means the same configuration **starts on a busy GPU and fails on an idle
one**. `bin/submit_spark_job.sh` refuses to submit when
`RAPIDS_GPU_ALLOC_FRACTION` exceeds `RAPIDS_GPU_MAX_ALLOC_FRACTION`, because
RAPIDS' own rejection surfaces as an executor-plugin shutdown inside a py4j stack
trace — after the phase banner, before any progress line — which reads as a
startup hang rather than a bad setting.

**4. On a multi-homed node, bind Spark to the interface the cluster actually uses.**
If the nodes have more than one network — say a management LAN plus a dedicated
fabric — every Spark process advertises whichever non-loopback interface it finds
first unless told otherwise. Pointing the master at the fabric is not enough:
`SPARK_MASTER_HOST` only decides where the master *listens*, while the addresses
the driver, the executors and their block managers hand out to *each other* come
from `SPARK_LOCAL_IP`:

```bash
# in $SPARK_HOME/conf/spark-env.sh, per node — the node's OWN address on the fabric
export SPARK_LOCAL_IP=10.0.0.7
```

This one does not hang so much as **lie**. Executors on every node *except* the
driver's cannot reach the driver's advertised address, time out after 120s, exit 1,
and are relaunched forever; the executor that happens to be co-located with the
driver reaches it over loopback and quietly runs the entire job. The master reports
every worker healthy, the job succeeds, the tests pass — and the cluster is running
at the capacity of a single node. If those executors do survive long enough to
shuffle, peer block fetches hang instead, and the job stops making progress with no
error at all. Confirm the fix by checking that the master's worker list shows fabric
addresses, not LAN ones.

**5. Size the executors — Spark's default is 1 GB.** Nothing in this repository
or in a stock `spark-defaults.conf` sets `spark.executor.memory`, so without
`EXECUTOR_MEMORY` every executor takes Spark's 1 GB default no matter what the
workers advertise. This is the quietest failure of the five: the job does not
crash, it spills, and it succeeds while running many times slower than it should.
It stayed hidden because the cluster smoke test runs on committed fixtures
measured in kilobytes, where 1 GB is ample — real input is not, at 322.7M
triples for a single day across four sources.

```bash
EXECUTOR_MEMORY=64g bin/submit_spark_job.sh --mode full ...
```

The launcher's default is deliberately a low floor rather than a value sized to
any particular cluster, because Spark treats an executor-memory request larger
than a worker can offer as *unsatisfiable* and waits on it rather than failing —
the same hang class as prerequisite 1. Erring small costs speed; erring large
costs the whole run.

**6. Size the driver — there are two limits, and raising one does not raise the
other.** `--mode full` and `--mode pyg_only` finish on the driver: node features
arrive as one dense tensor per node type, edge features as one per edge type. Both
of these are driver-bound in a way the enrichment modes are not, and each runs into
a different limit.

- **`DRIVER_MEMORY`** is the heap the tensors live in. Node features cost
  `num_nodes × vector_dim × 4` bytes — on a four-source day (10.3M nodes at
  `vector_dim` 1024) that is 42.4 GiB before any Arrow buffer or Pandas copy, into
  a default of 4 GB.
- **`DRIVER_MAX_RESULT_SIZE`** is Spark's cap on what a *single* `collect` may
  return, defaulting to 1 GB. It is not raised by raising the heap, which is what
  makes it confusing in practice: a run given 32 GB of heap after an
  `OutOfMemoryError` fails again a step later on a limit nobody moved.

Edge-feature batches are sized against `DRIVER_MAX_RESULT_SIZE`, in bytes rather
than edge count, so leaving it unset also leaves that sizing with nothing to work
from. Both `bin/profiles/large-run.env` and `bin/profiles/pyg-assembly.env` set it.

```bash
DRIVER_MEMORY=64g DRIVER_MAX_RESULT_SIZE=8g \
  bin/submit_spark_job.sh --mode pyg_only ...
```

| Variable | Default | Applies to |
|---|---|---|
| `EXECUTOR_MEMORY` | `4g` | cluster masters only; local mode uses `DRIVER_MEMORY` |
| `EXECUTOR_CORES` | unset — each executor takes every core on its worker | cluster masters only |
| `DRIVER_MEMORY` | `4g` | all masters |
| `DRIVER_MAX_RESULT_SIZE` | unset — Spark caps a single `collect` at 1 GB | all masters; sizes the edge-feature batches |
| `RAPIDS_GPU_ALLOC_FRACTION` | `0.25` | pool as a fraction of **free** GPU memory |
| `RAPIDS_GPU_MAX_ALLOC_FRACTION` | `0.4` | hard cap as a fraction of **total** GPU memory |
| `RAPIDS_PINNED_POOL` | `2G` | host memory staged for host↔device transfer |
| `RAPIDS_GPU_MAP_IN_ARROW` | `false` | lets RAPIDS run `mapInArrow` on the GPU. Off on purpose: the Turtle parse deadlocks against its Python worker there, with no error and no lost executor. Set `true` only to reproduce that stall. |
| `NETWORK_TIMEOUT` | `120s` | how long the driver waits on a silent executor |
| `EXECUTOR_HEARTBEAT_INTERVAL` | `10s` | must stay well under `NETWORK_TIMEOUT` |

**Verify GPU execution; don't assume it.** A `count()` on Parquet can be answered
from file metadata without touching the GPU. Set `spark.rapids.sql.explain=ALL`,
or confirm that `Gpu*` operators (e.g. `GpuFileSourceScanExec`) appear in the
physical plan.

## Sizing a large run

Every default above is a **floor**, chosen so that a run on unfamiliar hardware fails
safe rather than fast. A full-day, multi-source run needs more than the floor, and
[`bin/profiles/large-run.env`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/profiles/large-run.env) carries that sizing. Source
it before the variables that identify your cluster:

```bash
. bin/profiles/large-run.env
. ~/my-cluster.env          # master URL, driver host, bucket paths — NOT tracked
bin/submit_spark_job.sh --mode full ...
```

The split is deliberate: the profile holds only *how big the job is*, so it names no
host, address, bucket, account or path and is safe to track. Anything identifying a
deployment stays in an untracked file sourced afterwards.

**Why the floors stay low even though a large run needs more.** The two GPU fractions
guard a *hardware* constraint, not a workload one — on a unified-memory GPU the pool is
host RAM (see prerequisite 3), so raising the default would reserve more of the host on
every run, including the kilobyte-fixture cluster smoke test. Workload size is the wrong
axis to set them on, which is why the profile opts in per run instead.

**The symptom that says the pool is too small.** Task slots per executor are
`min(cores/task.cpus, gpu/task.gpu)` — at the `GPU_PER_TASK` default of `0.125` that is
**8 tasks sharing one RMM pool regardless of core count**, so adding cores does not
relieve GPU pressure. Under-sizing shows up in the *executor* log (not the driver's) as:

```
[RMM] [error] [A][Stream 0][Upstream 220200960B][FAILURE maximum pool size exceeded]
```

Measured on a four-source day whose market source alone parses to 345M triples: 215,144
such lines from a single executor, rising hour over hour, until it stalled inside block
eviction for 173s and was evicted at the stock `120s` timeout. Because
`localCheckpoint` blocks have no replica, eviction destroyed the blocks that executor
held and the job aborted with `Checkpoint block rdd_N not found` after 2h35m. Raising
`NETWORK_TIMEOUT` buys tolerance for the stall; raising the fractions addresses the
cause. The profile does both.

Read executor-side logs at `$SPARK_HOME/work/<app-id>/<executor-id>/stderr` — the driver
log cannot distinguish a hung executor from a dead one, and the difference decides
whether more memory or a longer timeout is the fix.

**Where the loaded frame's partition count comes from.** The triples are unioned with the
datatype markers `literal_datatype_observations()` derives, and a union's partition count
is the *sum* of its children's. The marker frame ends in `distinct()`, which is a shuffle,
and a shuffle lands on `spark.sql.shuffle.partitions` — 200 by default — however few rows
survive it. That union used to happen inside each loader, so every source contributed its
file-scan partitions **plus 200**, for a frame whose row count is bounded by distinct
(predicate, datatype) pairs. Measured on five sources: 160 scan partitions and 1,000
marker partitions, and because the union is cached and read by every enrichment phase,
every stage that read it inherited all 1,160. Two changes closed that: the marker frame is
coalesced to one partition where it is built, and `load_source_triples` now derives the
markers from the cached parse and unions them once for the whole run rather than once per
source. What remains is the scan partitioning of the actual data.

**What that is worth — measure it, do not estimate it.** On a cluster run the coalesce took
the seed leg from 661,509 tasks to 151,808, and its wall clock from 174.5 to 168.2 minutes.
Almost all of the removed tasks were empty: they held a task slot for 3.3 ms each, 1,600s in
total, which is 2% of the run's slot time, against 430 ms for a task that does real work. An
earlier estimate of 60–80 minutes came from costing the empty tasks at the average across
*all* tasks — a blend of those two populations that overstates them by about 24x. Task
counts and task cost are separate measurements, and on this pipeline they differ by two
orders of magnitude. The reason to fix the partition count is that nothing bounds it: it
grows with every source added, and every enrichment stage reads the result.

**The same fork also cost a second parse, which was the expensive half.** Building the
markers from the *uncached* frame planned two independent legs per source, so the rdflib
UDF ran over every source twice — once for the triples, once for the markers — and the
`cache()` further down sat below the fork and never applied. Deriving them from the cached
parse instead took the load phase from 1,346.3s to 727.7s on identical input, and folded
two stages of 9.17 and 9.29 task-hours into one of 9.35.

That change moves the triple count slightly, and the move is correct. Markers now
deduplicate once per *source*, where they used to deduplicate once per *file path*, so a
source that ships ten files and declares the same (predicate, datatype) in each
contributes one marker row instead of ten. On a four-source day this is 9 rows out of
322.7M, all of them from the ten-file source, and `duplicates_within_source` falls by the
same 9. Distinct triples are unchanged, so no graph built either side of it differs.

The lesson generalizes past this one frame: a small `distinct()`, `groupBy` or join
*inside a branch you are about to union* costs the whole downstream pipeline that branch's
shuffle width, not just its own stage. Adaptive query execution hides it locally — AQE
coalesces the tiny exchange away — so it shows up only at cluster scale.

