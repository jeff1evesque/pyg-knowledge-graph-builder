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


def _run_pipeline_subprocess(work_dir):
    """Run the real build_graph CLI once, in its own JVM process.

    Determinism requires two runs, but the enrichment fans out into ~1,300
    cumulative Spark stages *per run* regardless of source count (every source's
    linker executes structurally). Two runs in one shared SparkSession exhaust
    the driver/executor heap of the single local JVM (OutOfMemoryError mid-second
    run). Each single run fits — the other e2e tests prove it — so we isolate the
    two runs in separate processes. Independent JVMs, independent shuffle
    scheduling: this is also the strongest form of a reproducibility check.

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
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=1200,
    )
    assert proc.returncode == 0, (
        "build_graph subprocess failed\n"
        f"STDOUT tail:\n{proc.stdout[-3000:]}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )


def _metadata_by_name(work_dir):
    return {
        p.name: p
        for p in Path(work_dir).rglob("*.json")
        if p.name in METADATA_FILES
    }


def _store_get(store, key):
    return store[key] if key in store and store[key] is not None else None


@pytest.mark.xfail(
    reason=(
        "Known jolts_* categorical non-determinism: the multi-hot flags + "
        "group-by-sum combine in feature_extractor drift run-to-run "
        "(observed on jolts_MonthlyHiresData node features). Tracked as its "
        "own bug. Non-strict so an occasionally-deterministic run does not "
        "flip this red; remove the marker once reproducibility is fixed."
    ),
    strict=False,
)
def test_output_is_reproducible(tmp_path):
    """Two runs over the same fixture → identical graph, tensors, and metadata.

    Guards the class of defect the jolts_* categorical non-determinism belongs
    to: individually-plausible modules that, composed, drift run-to-run. The
    metadata build_timestamp is the one field allowed to differ. Runs each
    pipeline in its own process (see _run_pipeline_subprocess).

    Currently xfails on the jolts_* categorical non-determinism (see the marker).
    The full assertion set is kept intact so the guard flips green the moment
    the bug is fixed.
    """
    import torch

    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    _run_pipeline_subprocess(run1)
    _run_pipeline_subprocess(run2)

    pt1 = next(run1.rglob("hetero_data.pt"))
    pt2 = next(run2.rglob("hetero_data.pt"))
    d1 = _load_hetero(pt1)
    d2 = _load_hetero(pt2)

    # --- same node/edge type inventory ---
    assert sorted(d1.node_types) == sorted(d2.node_types)
    assert sorted(map(tuple, d1.edge_types)) == sorted(map(tuple, d2.edge_types))

    # --- per-type counts + feature tensors bit-identical ---
    for nt in d1.node_types:
        assert d1[nt].num_nodes == d2[nt].num_nodes, f"{nt} count drift"
        x1, x2 = _store_get(d1[nt], "x"), _store_get(d2[nt], "x")
        assert (x1 is None) == (x2 is None), f"{nt} feature presence drift"
        if x1 is not None:
            assert torch.equal(x1, x2), f"node features differ for {nt}"

    for et in d1.edge_types:
        assert torch.equal(d1[et].edge_index, d2[et].edge_index), (
            f"edge_index differs for {et}"
        )
        a1, a2 = _store_get(d1[et], "edge_attr"), _store_get(d2[et], "edge_attr")
        assert (a1 is None) == (a2 is None), f"{et} edge_attr presence drift"
        if a1 is not None:
            assert torch.equal(a1, a2), f"edge features differ for {et}"

    # --- metadata byte-identical, except the allow-listed build timestamp ---
    m1, m2 = _metadata_by_name(run1), _metadata_by_name(run2)
    assert set(m1) == METADATA_FILES and set(m2) == METADATA_FILES
    for name in METADATA_FILES:
        b1, b2 = m1[name].read_bytes(), m2[name].read_bytes()
        if name == "graph_schema.json":
            j1, j2 = json.loads(b1), json.loads(b2)
            j1["build_metadata"].pop("build_timestamp", None)
            j2["build_metadata"].pop("build_timestamp", None)
            assert j1 == j2, "graph_schema.json differs beyond the timestamp"
        else:
            assert b1 == b2, f"{name} differs between runs"
