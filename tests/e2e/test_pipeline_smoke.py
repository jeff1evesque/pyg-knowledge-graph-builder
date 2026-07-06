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
