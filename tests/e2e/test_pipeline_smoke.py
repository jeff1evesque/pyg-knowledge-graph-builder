"""End-to-end pipeline smoke tests.

Runs the real ``build_graph`` entry points over small committed RDF fixtures on
a local CPU ``SparkSession`` (the shared ``spark`` fixture in tests/conftest.py).
This exercises the orchestration path the unit suite never touches: the source
loaders, enrichment -> construction, the enriched-Parquet round-trip, and the
``.pt`` + six metadata JSON outputs — exactly where import-time / API-mismatch
bugs have hidden until a cluster run.

Coverage matrix:
  * modes:   full, and the split enrichment_only -> pyg_only round-trip
  * loaders: turtle_parquet (production UDF path) and ntriples (default loader)

Sources: noaa / market / sec / bls — all of them.

Marked ``e2e`` so it is excluded from the fast unit suite (``-m "not e2e"``) and
run on its own (``-m e2e``). It is HEAVY: the enrichment fans out into ~1,300
Spark stages regardless of data size, so it does not reliably finish on a stock
7 GB CI runner. It is therefore NOT run on push/PR — the ``e2e.yml`` workflow is
manual-only (``workflow_dispatch``), and it's best run locally / on a capable
machine (see the README "Testing" section). No cluster, no GPU/RAPIDS required —
identical logic runs on CPU.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "e2e"
TURTLE_PARQUET = FIXTURES / "turtle_parquet"
NTRIPLES = FIXTURES / "ntriples"

METADATA_FILES = {
    "graph_schema.json",
    "feature_spec.json",
    "normalization.json",
    "encoding_config.json",
    "ontology_schema.json",
    "slot_mapping.json",
}


def _turtle_parquet_source_paths():
    """One source path per leaf dir that holds a .parquet fixture."""
    dirs = sorted({str(p.parent) for p in TURTLE_PARQUET.rglob("*.parquet")})
    assert dirs, f"no turtle-parquet fixtures under {TURTLE_PARQUET}"
    return dirs


def _make_config(**overrides):
    from spark_jobs.build_graph import JobConfig

    args = {
        "mode": "full",
        "source_format": "turtle_parquet",
        "turtle_column": "triples",
        "time_period": "2099-01",  # fixed so split-mode paths line up
        "parquet_partitions": "2",
    }
    args.update({k: str(v) for k, v in overrides.items()})
    return JobConfig(args)


def _load_hetero(pt_path):
    import torch

    # HeteroData is a pickled object graph; torch>=2.6 defaults weights_only=True
    # (which refuses arbitrary classes), so opt out explicitly.
    return torch.load(pt_path, weights_only=False)


def _assert_valid_graph_and_metadata(config, work_dir):
    import os

    from spark_jobs.pyg_builder.edge_feature_extractor import EdgeVectorLayout
    from spark_jobs.pyg_builder.feature_extractor import VectorLayout

    # --- .pt loads into a non-empty heterogeneous graph ---
    pt = config.pyg_output_path
    assert os.path.exists(pt), f"missing .pt at {pt}"
    data = _load_hetero(pt)

    assert len(data.node_types) > 0, "graph has no node types"
    assert len(data.edge_types) > 0, "graph has no edge types"
    total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    assert total_nodes > 0, "graph has no nodes"
    assert total_edges > 0, "graph has no edges"

    # --- node feature tensors: 2D, row-aligned, width is a valid layout dim ---
    node_types_with_features = 0
    for nt in data.node_types:
        store = data[nt]
        if "x" in store and store.x is not None:
            x = store.x
            assert x.dim() == 2, f"{nt}.x not 2D"
            assert x.shape[0] == store.num_nodes, f"{nt}.x row count != num_nodes"
            VectorLayout(int(x.shape[1]))  # raises if width violates the layout
            node_types_with_features += 1
    assert node_types_with_features > 0, "no node type carries a feature tensor"

    # --- edge tensors: edge_index [2, E]; edge features (if any) valid width ---
    for et in data.edge_types:
        store = data[et]
        assert store.edge_index.dim() == 2 and store.edge_index.shape[0] == 2
        if "edge_attr" in store and store.edge_attr is not None:
            EdgeVectorLayout(int(store.edge_attr.shape[1]))

    # --- all six metadata JSONs present, non-empty, valid JSON ---
    found = {p.name: p for p in Path(work_dir).rglob("*.json") if p.name in METADATA_FILES}
    missing = METADATA_FILES - set(found)
    assert not missing, f"missing metadata files: {sorted(missing)}"
    for name, path in found.items():
        with open(path) as fh:
            content = json.load(fh)
        assert content, f"{name} is empty"


# --------------------------------------------------------------------------- #
# full mode — both loaders
# --------------------------------------------------------------------------- #

def test_full_turtle_parquet(spark, tmp_path):
    from spark_jobs.build_graph import execute_full_pipeline

    config = _make_config(
        local_work_dir=str(tmp_path),
        source_format="turtle_parquet",
        source_paths=",".join(_turtle_parquet_source_paths()),
    )
    execute_full_pipeline(config, spark, s3_client=None)
    _assert_valid_graph_and_metadata(config, tmp_path)


def test_full_ntriples(spark, tmp_path):
    from spark_jobs.build_graph import execute_full_pipeline

    config = _make_config(
        local_work_dir=str(tmp_path),
        source_format="ntriples",
        source_paths=str(NTRIPLES),
    )
    execute_full_pipeline(config, spark, s3_client=None)
    _assert_valid_graph_and_metadata(config, tmp_path)


# --------------------------------------------------------------------------- #
# split mode — enrichment_only -> pyg_only (exercises the enriched-Parquet
# save/reload the full path never touches)
# --------------------------------------------------------------------------- #

def test_split_enrichment_then_pyg(spark, tmp_path):
    import os

    from spark_jobs.build_graph import execute_enrichment_only, execute_pyg_only

    work = str(tmp_path)
    source_paths = ",".join(_turtle_parquet_source_paths())

    enrich = _make_config(
        local_work_dir=work, mode="enrichment_only",
        source_format="turtle_parquet", source_paths=source_paths,
    )
    execute_enrichment_only(enrich, spark, s3_client=None)
    assert os.path.exists(enrich.enriched_parquet_path), (
        f"enriched parquet not written to {enrich.enriched_parquet_path}"
    )

    # Same work dir + time_period -> pyg_only reads the enriched parquet above.
    pyg = _make_config(local_work_dir=work, mode="pyg_only")
    execute_pyg_only(pyg, spark, s3_client=None)
    _assert_valid_graph_and_metadata(pyg, tmp_path)


# --------------------------------------------------------------------------- #
# reproducibility — the same input must yield an identical graph
#
# Determinism is an integration property no unit test can cover: it only emerges
# when every construction module composes over the real pipeline. We run the full
# pipeline twice on the same fixture and assert the two outputs are identical.
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]


def _start_pipeline_subprocess(work_dir):
    """Launch the real build_graph CLI once, in its own JVM process (non-blocking).

    Determinism requires two runs, but the enrichment fans out into ~1,300
    cumulative Spark stages *per run* regardless of source count (every source's
    linker executes structurally). Two runs in one shared SparkSession exhaust
    the driver/executor heap of the single local JVM (OutOfMemoryError mid-second
    run). Each single run fits — the other e2e tests prove it — so we isolate the
    two runs in separate processes. Independent JVMs, independent shuffle
    scheduling: this is also the strongest form of a reproducibility check.

    Returns the Popen handle (does not wait) so the caller can start both runs and
    then wait on them together: the two runs are fully independent, so overlapping
    them ~halves the twin-run wall-clock. The trade-off is ~2x peak memory (two
    --driver-memory 2g local[4] JVMs alive at once), acceptable on the machines
    that run the e2e suite; drop to sequential on a memory-constrained host. Pair
    with _finish_pipeline_subprocess.

    PYSPARK_PYTHON is pinned to this interpreter so the turtle_parquet parse UDF
    finds rdflib (see the e2e conftest note); heap/retention are passed through
    PYSPARK_SUBMIT_ARGS, which the driver JVM honors at launch.
    """
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYSPARK_PYTHON"] = sys.executable
    env["SPARK_LOCAL_IP"] = "127.0.0.1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PYSPARK_SUBMIT_ARGS"] = (
        "--master local[4] --driver-memory 2g "
        "--conf spark.ui.enabled=false "
        "--conf spark.sql.shuffle.partitions=8 "
        "--conf spark.ui.retainedJobs=40 "
        "--conf spark.ui.retainedStages=80 "
        "--conf spark.sql.ui.retainedExecutions=40 "
        "pyspark-shell"
    )

    cmd = [
        sys.executable, "-m", "spark_jobs.build_graph",
        "--mode", "full",
        "--source_format", "turtle_parquet",
        "--turtle_column", "triples",
        "--time_period", "2099-01",
        "--parquet_partitions", "2",
        "--local_work_dir", str(work_dir),
        "--source_paths", ",".join(_turtle_parquet_source_paths()),
    ]
    return subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _finish_pipeline_subprocess(work_dir, proc):
    """Wait for a _start_pipeline_subprocess run and assert it succeeded."""
    import subprocess

    try:
        stdout, stderr = proc.communicate(timeout=1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError(f"build_graph subprocess for {work_dir} timed out")
    assert proc.returncode == 0, (
        f"build_graph subprocess for {work_dir} failed\n"
        f"STDOUT tail:\n{stdout[-3000:]}\n"
        f"STDERR tail:\n{stderr[-3000:]}"
    )


def _metadata_by_name(work_dir):
    return {
        p.name: p
        for p in Path(work_dir).rglob("*.json")
        if p.name in METADATA_FILES
    }


def _store_get(store, key):
    return store[key] if key in store and store[key] is not None else None


def _is_jolts(node_type):
    """The categorical node type(s) with the known run-to-run drift."""
    return str(node_type).startswith("jolts_")


def _canon_json(value):
    """Recursively order-normalize a JSON value so reordered lists compare equal.

    slot_mapping.json's predicate lists get reordered run-to-run by the known
    jolts_* non-determinism (identical hash_slots/global_dims, different order).
    Canonicalizing lets the green guard assert the *content* is reproducible —
    which it is — while the byte-level ordering stays under the jolts_* xfail
    (test_jolts_features_reproducible).
    """
    if isinstance(value, list):
        return sorted(
            (_canon_json(v) for v in value),
            key=lambda v: json.dumps(v, sort_keys=True),
        )
    if isinstance(value, dict):
        return {k: _canon_json(v) for k, v in value.items()}
    return value


def _load_metadata(path, name):
    """Parse a metadata JSON, dropping the one field allowed to differ per run.

    graph_schema.json embeds a build_timestamp that legitimately changes between
    runs; every other field (and every other file) must be reproducible.
    """
    obj = json.loads(Path(path).read_bytes())
    if name == "graph_schema.json":
        obj.get("build_metadata", {}).pop("build_timestamp", None)
    return obj


class _TwinRuns:
    """Both loaded graphs + their metadata dirs from two identical pipeline runs."""

    def __init__(self, d1, d2, run1, run2):
        self.d1, self.d2 = d1, d2
        self.run1, self.run2 = run1, run2


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory):
    """Run the full pipeline twice over the same fixture, once, for both tests.

    Determinism is an integration property no unit test can cover: it only emerges
    when every construction module composes over the real pipeline. Each run is
    isolated in its own process (see _start_pipeline_subprocess) and the two are
    launched concurrently, then waited on together — overlapping the independent
    runs ~halves the twin-run wall-clock. The pair is module-scoped so the
    expensive twin runs execute exactly once and are shared by every
    reproducibility test below.
    """
    base = tmp_path_factory.mktemp("reproducibility")
    run1, run2 = base / "run1", base / "run2"
    # Start both independent runs, then wait on both (concurrent, not sequential).
    running = [(wd, _start_pipeline_subprocess(wd)) for wd in (run1, run2)]
    for wd, proc in running:
        _finish_pipeline_subprocess(wd, proc)

    d1 = _load_hetero(next(run1.rglob("hetero_data.pt")))
    d2 = _load_hetero(next(run2.rglob("hetero_data.pt")))
    return _TwinRuns(d1, d2, run1, run2)


def test_output_is_reproducible(twin_runs):
    """Two runs over the same fixture → identical graph, tensors, and metadata.

    Green guard for everything already reproducible: node/edge inventory,
    per-type counts, every edge_index, edge features, all metadata (byte-identical
    modulo the allow-listed build_timestamp; metadata list ordering compared
    order-insensitively — see below), and the feature tensors of every
    non-jolts_* node type. A new reproducibility regression anywhere outside
    jolts_* fails this test rather than being masked by an xfail. The jolts_*
    tensors and the metadata entry ordering are checked separately
    (test_jolts_features_reproducible).
    """
    import torch

    d1, d2 = twin_runs.d1, twin_runs.d2

    # --- same node/edge type inventory ---
    assert sorted(d1.node_types) == sorted(d2.node_types)
    assert sorted(map(tuple, d1.edge_types)) == sorted(map(tuple, d2.edge_types))

    # --- per-type counts + non-jolts_* feature tensors bit-identical ---
    for nt in d1.node_types:
        assert d1[nt].num_nodes == d2[nt].num_nodes, f"{nt} count drift"
        x1, x2 = _store_get(d1[nt], "x"), _store_get(d2[nt], "x")
        assert (x1 is None) == (x2 is None), f"{nt} feature presence drift"
        if x1 is not None and not _is_jolts(nt):
            assert torch.equal(x1, x2), f"node features differ for {nt}"

    for et in d1.edge_types:
        assert torch.equal(d1[et].edge_index, d2[et].edge_index), (
            f"edge_index differs for {et}"
        )
        a1, a2 = _store_get(d1[et], "edge_attr"), _store_get(d2[et], "edge_attr")
        assert (a1 is None) == (a2 is None), f"{et} edge_attr presence drift"
        if a1 is not None:
            assert torch.equal(a1, a2), f"edge features differ for {et}"

    # --- metadata content reproducible (modulo the allow-listed timestamp) ---
    # The jolts_* non-determinism reorders list entries inside the metadata JSON
    # (identical content, drifting order — seen in slot_mapping.json and
    # ontology_schema.json). Compare order-insensitively here so this guard tracks
    # content reproducibility; the byte-level ordering is guarded by the jolts_*
    # xfail (test_jolts_features_reproducible).
    m1, m2 = _metadata_by_name(twin_runs.run1), _metadata_by_name(twin_runs.run2)
    assert set(m1) == METADATA_FILES and set(m2) == METADATA_FILES
    for name in METADATA_FILES:
        j1 = _load_metadata(m1[name], name)
        j2 = _load_metadata(m2[name], name)
        assert _canon_json(j1) == _canon_json(j2), (
            f"{name} content differs between runs"
        )


@pytest.mark.xfail(
    reason=(
        "Known jolts_* categorical non-determinism: the multi-hot flags + "
        "group-by-sum combine in feature_extractor drift run-to-run "
        "(observed on jolts_MonthlyHiresData node features). Tracked as its "
        "own bug. Non-strict so an occasionally-deterministic run does not "
        "flip this red; remove the marker once reproducibility is fixed. Also "
        "covers the metadata JSON (e.g. slot_mapping.json, ontology_schema.json), "
        "whose jolts_* list entries the same bug reorders run-to-run (identical "
        "content, drifting order)."
    ),
    strict=False,
)
def test_jolts_features_reproducible(twin_runs):
    """The jolts_* feature tensors + metadata entry ordering must match across runs.

    Isolates the one genuinely non-reproducible slice so it xfails on its own
    without masking regressions elsewhere (guarded by test_output_is_reproducible).
    Flips green the moment the jolts_* non-determinism is fixed. Asserts the
    jolts_* type(s) are actually present so an empty selection can't silently
    xpass on a comparison that never ran.
    """
    import torch

    d1, d2 = twin_runs.d1, twin_runs.d2

    jolts_types = [nt for nt in d1.node_types if _is_jolts(nt)]
    assert jolts_types, "no jolts_* node types present — nothing to compare"

    for nt in jolts_types:
        x1, x2 = _store_get(d1[nt], "x"), _store_get(d2[nt], "x")
        assert (x1 is None) == (x2 is None), f"{nt} feature presence drift"
        if x1 is not None:
            assert torch.equal(x1, x2), f"jolts node features differ for {nt}"

    # Same non-determinism reorders entries inside the metadata JSON (identical
    # content, drifting order): list reordering (slot_mapping.json) and dict
    # key/insertion-order (ontology_schema.json). Compare serialized form so both
    # are honored — this is the full "byte-reproducible modulo build_timestamp"
    # property. It flips green alongside the tensors once the bug is fixed;
    # test_output_is_reproducible meanwhile guards the order-insensitive content.
    m1, m2 = _metadata_by_name(twin_runs.run1), _metadata_by_name(twin_runs.run2)
    for name in METADATA_FILES:
        s1 = json.dumps(_load_metadata(m1[name], name))
        s2 = json.dumps(_load_metadata(m2[name], name))
        assert s1 == s2, f"{name} entry ordering differs between runs"
