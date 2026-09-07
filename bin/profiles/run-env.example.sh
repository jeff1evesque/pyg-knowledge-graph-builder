# The contract for a run directory's env.sh. Copy this to <run-dir>/env.sh, fill it
# in, and DO NOT track the result:
#
#     cp bin/profiles/run-env.example.sh ~/pyg-runs/issue-NNN/env.sh
#     $EDITOR ~/pyg-runs/issue-NNN/env.sh
#     bin/run_cluster_notebook.sh ~/pyg-runs/issue-NNN
#
# WHY THE SPLIT: everything in a real env.sh is either an address, a path, a storage
# location or this week's intent. None of that belongs in the repository, and all of
# it changes per run. The scripts that read it are tracked and change rarely. Same
# split as bin/profiles/large-run.env, where the sizing is tracked and the identity
# is sourced after it.
#
# RUN_ID is exported by the launcher before this file is sourced, so paths below can
# use it. Every value here is a placeholder that will fail loudly if left as is.

# --------------------------------------------------------------------------- #
# Required
# --------------------------------------------------------------------------- #
export PYG_REPO_ROOT="$HOME/path/to/pyg-knowledge-graph-builder"

# The work directory MUST end in $RUN_ID. The end-of-run checkpoint sweep will only
# delete under a path that does, which is what stops it from ever reaching another
# run's output. A literal path here used to mean the sweep matched nothing at all.
export PYG_WORK_DIR="/path/to/shared/work/issue-NNN/${RUN_ID:?set RUN_ID first}"

export SPARK_HOME=/opt/spark
export SPARK_MASTER_URL=spark://MASTER-HOST:7077
export SPARK_DRIVER_HOST=DRIVER-HOST

# Every node the run touches, comma separated, including this one. The launcher
# traces the network on each, and bin/record_run_outcome.sh reads executor logs from
# each. Addresses or resolvable names, whichever your cluster uses.
export PYG_STAGE_NODES=NODE-1,NODE-2

# What to build the graph from.
export PYG_SOURCE_PATHS="s3a://BUCKET/PREFIX/year=YYYY/month=MM/DD.snappy.parquet"
export PYG_SOURCE_FORMAT=turtle_parquet
export PYG_TIME_PERIOD=YYYY-MM

# --------------------------------------------------------------------------- #
# Reading a staged local mirror instead of object storage
# --------------------------------------------------------------------------- #
# export PYG_INPUT_MODE=local
# export PYG_LOCAL_SOURCE_ROOT=/path/to/mirror        # same path on every node

# --------------------------------------------------------------------------- #
# Sizing profiles. Tracked; sourced by the notebook per leg.
# --------------------------------------------------------------------------- #
# export PYG_SEED_PROFILE="$PYG_REPO_ROOT/bin/profiles/large-run.env"
# export PYG_ASSEMBLY_PROFILE="$PYG_REPO_ROOT/bin/profiles/pyg-assembly.env"
# export PYG_PARQUET_PARTITIONS=200

# --------------------------------------------------------------------------- #
# Extra Spark configuration
# --------------------------------------------------------------------------- #
# The event log is the ONLY record of per-stage task concurrency, and
# bin/record_run_outcome.sh reads the run's time window out of the file names there.
# Point it inside the run directory. The launcher creates the directory, because
# Spark refuses to start when it is missing -- one run died 50 seconds in that way.
# export SPARK_EXTRA_CONF="--conf spark.eventLog.enabled=true \
#   --conf spark.eventLog.dir=file://$HOME/pyg-runs/issue-NNN/eventlog \
#   --conf spark.eventLog.compress=true"
# export RAPIDS_EXPLAIN=NONE

# --------------------------------------------------------------------------- #
# Caps. Size these to "obviously hung", not to how long the run should take: the
# notebook kills a leg that hits one, and a cap set to the expected duration kills
# good runs. One 4h seed cap killed a run that needed 4h25m, with nothing written.
# --------------------------------------------------------------------------- #
# export PYG_SEED_TIMEOUT=25200
# export PYG_EXPERIMENT_TIMEOUT=14400
# export PYG_SEED_ONLY=1                              # skip the assembly legs

# --------------------------------------------------------------------------- #
# Stall watchdog. Started by bin/submit_spark_job.sh once per submit; the launcher
# watches this directory and stops the run with rc=99 when a capture lands in it.
# --------------------------------------------------------------------------- #
# export PYG_STALL_DUMP_DIR="$HOME/pyg-runs/issue-NNN/stalls"
# export PYG_STALL_SECONDS=300
# export PYG_STALL_FACTOR=2.0
# export PYG_STALL_MAX_CAPTURES=1

# --------------------------------------------------------------------------- #
# Notebook runner. The runner venv needs nbformat and nbclient, which the job's own
# venv does not carry.
# --------------------------------------------------------------------------- #
# export PYG_NOTEBOOK="$PYG_REPO_ROOT/notebook/multi_experiment.ipynb"
# export PYG_NOTEBOOK_KERNEL=pyg-notebook-runner
# export PYG_RUNNER_PYTHON="$HOME/pyg-runs/issue-NNN/runner-venv/bin/python"

# --------------------------------------------------------------------------- #
# Network trace. The uplink defaults to whichever interface holds the default
# route; name it only if that is wrong. Fabric interfaces are node-to-node links,
# counted together and reported apart from the uplink.
# --------------------------------------------------------------------------- #
# export PYG_NET_WAN_IFACE=IFACE
# export PYG_NET_FABRIC_IFACES=IFACE-A,IFACE-B
# export PYG_NET_MAX_HOURS=24

# --------------------------------------------------------------------------- #
# Cluster sampler. Both UIs are derived from the two Spark variables above; set
# these only if the UI lives somewhere else.
# --------------------------------------------------------------------------- #
# export PYG_SPARK_MASTER_UI=http://MASTER-HOST:8080
# export PYG_SPARK_DRIVER_UI=http://DRIVER-HOST:4040
# export PYG_SAMPLE_SECONDS=30

# --------------------------------------------------------------------------- #
# The network guard: how bad the round trip to the default gateway has to get, and
# for how many consecutive samples, before the run is killed to protect the network.
# A LAN round trip is sub-millisecond when the forwarding plane is healthy.
# --------------------------------------------------------------------------- #
# export PYG_GATEWAY_RTT_LIMIT_MS=60
# export PYG_GATEWAY_STRIKES=6

# --------------------------------------------------------------------------- #
# Credentials, if the run reads object storage. Values, never keys: this file is
# sourced into the launcher's environment and read by anyone with the run directory.
# --------------------------------------------------------------------------- #
# export AWS_EC2_METADATA_SERVICE_ENDPOINT=http://127.0.0.1:PORT
# export AWS_DEFAULT_REGION=us-east-1
