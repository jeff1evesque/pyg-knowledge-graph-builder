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

import collections
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

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

    import torch

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

    # --- every edge index actually addresses a node that exists ---
    _assert_indices_are_in_range(data)

    # --- node feature tensors: 2D, row-aligned, valid width, float32 ---
    #
    # dtype is asserted alongside shape because torch enforces it only at the
    # point of use, far downstream. Spark's DoubleType lands as float64 through
    # a pandas round-trip, and a float64 `x` survives every other check here --
    # it is the right shape, the right width, and (being deterministic) it is
    # even bit-reproducible, so test_output_is_reproducible passes beside it.
    # The failure surfaces only when a trainer multiplies it against float32
    # weights, by which point nothing points back at the pipeline.
    node_types_with_features = 0
    for nt in data.node_types:
        store = data[nt]
        if "x" in store and store.x is not None:
            x = store.x
            assert x.dim() == 2, f"{nt}.x not 2D"
            assert x.shape[0] == store.num_nodes, f"{nt}.x row count != num_nodes"
            assert x.dtype == torch.float32, (
                f"{nt}.x has dtype {x.dtype}, expected torch.float32 -- "
                f"standard conv layers hold float32 weights and will not accept "
                f"it. Cast at the construction boundary (constructor.py), not "
                f"here."
            )
            _assert_all_finite(x, f"{nt}.x")
            VectorLayout(int(x.shape[1]))  # raises if width violates the layout
            node_types_with_features += 1
    assert node_types_with_features > 0, "no node type carries a feature tensor"

    # --- edge tensors: edge_index [2, E] int64; edge features valid + float32 ---
    for et in data.edge_types:
        store = data[et]
        assert store.edge_index.dim() == 2 and store.edge_index.shape[0] == 2
        assert store.edge_index.dtype == torch.int64, (
            f"{et} edge_index has dtype {store.edge_index.dtype}, expected "
            f"torch.int64 -- PyG indexes with it, so anything else raises "
            f"inside the first scatter/index_select rather than here."
        )
        if "edge_attr" in store and store.edge_attr is not None:
            assert store.edge_attr.dtype == torch.float32, (
                f"{et} edge_attr has dtype {store.edge_attr.dtype}, expected "
                f"torch.float32 (same reason as node features above)."
            )
            _assert_all_finite(store.edge_attr, f"{et} edge_attr")
            EdgeVectorLayout(int(store.edge_attr.shape[1]))

    # --- all six metadata JSONs present, non-empty, valid JSON ---
    found = {p.name: p for p in Path(work_dir).rglob("*.json") if p.name in METADATA_FILES}
    missing = METADATA_FILES - set(found)
    assert not missing, f"missing metadata files: {sorted(missing)}"
    for name, path in found.items():
        with open(path) as fh:
            content = json.load(fh)
        assert content, f"{name} is empty"

    # --- owl:sameAs only ever links temporal entities to temporal entities ---
    # cross_source_linker used to match subjects with "(January|...)$" -- anchored
    # only at the end -- so any URI merely ENDING with a month name was asserted
    # to BE that month (unified:September owl:sameAs cpi:...PercentChange_...
    # _September). owl:sameAs is RDF's strongest claim, so those became real
    # graph edges. Guarded here as well as in the unit tests because this asserts
    # the property of the WHOLE pipeline -- any enricher reintroducing it fails.
    _assert_sameas_links_only_temporal(config)

    # --- every edge type records where it came from ---
    # "unknown" is the not-supplied fallback, so seeing it in a real build means
    # constructor.py stopped passing edge_origins. Distinguishing an observed
    # fact from an enrichment-inferred link is what makes an edge trustworthy.
    schema = json.loads(Path(found["graph_schema.json"]).read_bytes())
    origins = {
        e["origin"] for e in schema["edge_types"].values()
    }
    assert "unknown" not in origins, "edge types with unrecorded origin"
    assert origins <= {"raw", "enrichment", "unification"}, (
        f"unexpected origin values: {origins}"
    )

    # --- the metadata actually describes the tensors that were emitted ---
    _assert_metadata_describes_the_graph(data, found)

    # --- node_index: every graph row is attributable to a real entity ---
    # Without this the .pt is anonymous -- row 5 of cpi_Index is some specific
    # series and nothing on disk says which, which blocks joining training
    # labels and attributing predictions. Assert it covers the graph exactly.
    index = _load_node_index(config.pyg_output_path)
    per_type = index.groupby("node_type")["node_id"]

    assert set(index["node_type"]) == set(map(str, data.node_types)), (
        "node_index node types do not match the graph's"
    )
    for nt in data.node_types:
        ids = sorted(per_type.get_group(str(nt)).tolist())
        assert ids == list(range(data[nt].num_nodes)), (
            f"{nt}: node_index ids are not exactly 0..num_nodes-1"
        )

    assert index["uri"].is_unique, "node_index maps two rows to one entity"
    assert len(index) == sum(data[nt].num_nodes for nt in data.node_types)

    # --- the graph has a temporal dimension ---
    _assert_temporal_edges_present(config, data)

    # --- that dimension is ONE node type per granularity, not one per source ---
    _assert_temporal_types_are_not_sharded(data)

    # --- unification is a real origin category, not an empty one ---
    # It went empty when the fabricated sameAs links (matched by an end-anchored
    # month regex, so they pointed at MEASUREMENTS, which are typed nodes) were
    # removed: the correct links pointed at untyped source temporal URIs and
    # were dropped. Non-empty here means the real links now resolve.
    assert "unification" in origins, (
        "no edge type has origin 'unification' — the owl:sameAs links from "
        "unified temporal entities to source temporal URIs did not resolve "
        "into edges (are the source temporal URIs typed?)"
    )

    _assert_no_node_type_is_orphaned(data)


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
        # Same plan-string cap the in-session fixture sets (see the e2e
        # conftest): every SQL execution renders its physical plan to
        # text, and these plans are wide enough that the default cap
        # lets one rendering exhaust the heap. The in-session fixture
        # had this; this subprocess path did not, so it kept Spark's
        # ~2GB default until batched edge-feature collects widened the
        # plans enough to hit it.
        "--conf spark.sql.maxPlanStringLength=4096 "
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



def _incident_edge_counts(data):
    """Total incident edges (in + out) per node type."""
    counts = {str(nt): 0 for nt in data.node_types}
    for src, _rel, dst in data.edge_types:
        n = data[(src, _rel, dst)].edge_index.shape[1]
        counts[str(src)] += n
        counts[str(dst)] += n
    return counts


# Node types that are STILL orphaned on the e2e fixtures. Measured directly:
# the fixtures produced 20 orphaned types before source temporal URIs were
# typed and 18 after — i.e. that fix cleared exactly bls_enrichment_UnifiedMonth
# and bls_enrichment_UnifiedYear and introduced no new orphan. The remaining 18
# are pre-existing and each is its own defect, mostly the same untyped-URI class
# one hop up: the BLS measurements that would link these types (cpi:...
# _PercentChange and friends) carry no rdf:type either, so they are not nodes
# and their edges cannot resolve.
#
# Listed explicitly rather than weakening the assertion, so the guard below
# stays exact — a NEW orphan fails — and so shrinking this list as each is
# fixed needs no change to the check.
# Seven BLS entries were removed when the e2e fixtures were regenerated with
# entity-complete sampling (#240): cpi_ExpenditureCategory, cpi_Year,
# eci_BenefitsPercentChangeData, empsit_EmploymentStatusByEducationData,
# empsit_Industry, jolts_SeparationsLevel, jolts_SeparationsRate. They were
# never a pipeline defect — the old fixtures were sampled by triple, so the
# entities on the other end of their edges were not typed and never became
# nodes. What remains is either non-BLS (filings_*, unknown_*, which draw on
# separate fixtures) or a BLS type the sample genuinely does not connect.
KNOWN_ORPHANED_NODE_TYPES = {
    "bls_enrichment_GeographicRegion",
    "cpi_TwelveMonthPercentChange",
    "eci_WagesAndSalariesIndexData",
    "empsit_HouseholdData",
    "empsit_OccupationData",
    "empsit_PeriodOfService",
    "empsit_UnemploymentDurationData",
    "empsit_UnemploymentReasonData",
    "filings_SECFiling",
    "unknown_EquitySnapshot",
    "unknown_OptionSnapshot",
}


def _assert_metadata_describes_the_graph(data, found):
    """The metadata JSONs must describe the tensors that were actually emitted.

    These six files are the contract a downstream trainer reads: it sizes its
    input layers from the declared dims and interprets each column through
    slot_mapping.json. The suite already checks the files exist, parse, and are
    non-empty -- but nothing checked they AGREE with the .pt beside them.

    They can drift because they come from different code paths: the tensors from
    FeatureExtractor / EdgeFeatureExtractor, the declarations from
    MetadataWriter (metadata_writer.py:304 onwards). When they drift nothing
    raises. A trainer builds an input layer of the declared width and either
    fails on a shape mismatch at the first batch -- far from the cause -- or, if
    the widths happen to match while the layout reordered, trains happily on
    features whose meaning does not match the declared slots and converges to
    something meaningless.

    Read from the run's own output (the `found` mapping of written files) rather
    than re-derived from the builders, so this tests the artifact that ships, not
    the code that produced it.

    Every message reports declared vs actual side by side: the whole failure mode
    is two numbers that should be equal and are not, so the diagnosis is the
    comparison.
    """
    schema = json.loads(Path(found["graph_schema.json"]).read_bytes())
    spec = json.loads(Path(found["feature_spec.json"]).read_bytes())

    # --- 1. node feature width -------------------------------------------- #
    # The node layout is global, not per-node-type, so one declared width applies
    # to every type that carries features.
    node_spec = spec["node_features"]
    declared_node_dim = int(node_spec["total_dim"])

    segment_sum = sum(int(s["dim"]) for s in node_spec["segments"])
    assert segment_sum == declared_node_dim, (
        f"feature_spec.json is internally inconsistent: node segments sum to "
        f"{segment_sum} but total_dim declares {declared_node_dim}. The "
        f"segment boundaries are what slot_mapping.json indexes into, so a "
        f"mismatch here misattributes every feature past the bad segment."
    )

    for nt in data.node_types:
        store = data[nt]
        if "x" in store and store.x is not None:
            actual = int(store.x.shape[1])
            assert actual == declared_node_dim, (
                f"{nt}.x width disagrees with feature_spec.json: "
                f"declared {declared_node_dim}, actual {actual}. A trainer "
                f"sizes its input layer from the declared value, so this fails "
                f"at the first batch -- or silently mis-slices if it does not."
            )

    # --- 2. edge feature width -------------------------------------------- #
    edge_spec = spec["edge_features"]
    declared_edge_dim = int(edge_spec["total_dim"])

    if edge_spec.get("enabled"):
        edge_segment_sum = sum(int(s["dim"]) for s in edge_spec["segments"])
        assert edge_segment_sum == declared_edge_dim, (
            f"feature_spec.json is internally inconsistent: edge segments sum "
            f"to {edge_segment_sum} but total_dim declares {declared_edge_dim}."
        )

    schema_edges = schema["edge_types"]
    for et in data.edge_types:
        store = data[et]
        key = _schema_edge_key(et)
        entry = schema_edges.get(key)
        assert entry is not None, (
            f"edge type {key} is in the graph but absent from "
            f"graph_schema.json's edge_types"
        )

        has_attr = "edge_attr" in store and store.edge_attr is not None
        assert bool(entry["has_features"]) == has_attr, (
            f"{key}: graph_schema.json declares has_features="
            f"{entry['has_features']} but the tensor "
            f"{'has' if has_attr else 'has no'} edge_attr"
        )

        if has_attr:
            actual = int(store.edge_attr.shape[1])
            assert int(entry["feature_dim"]) == actual, (
                f"{key} edge_attr width disagrees with graph_schema.json: "
                f"declared {entry['feature_dim']}, actual {actual}"
            )
            assert declared_edge_dim == actual, (
                f"{key} edge_attr width disagrees with feature_spec.json: "
                f"declared {declared_edge_dim}, actual {actual}"
            )
        else:
            assert int(entry["feature_dim"]) == 0, (
                f"{key}: graph_schema.json declares feature_dim="
                f"{entry['feature_dim']} but the tensor carries no edge_attr"
            )

    # --- 3. node type inventory ------------------------------------------- #
    declared_nodes = set(schema["node_types"])
    actual_nodes = {str(nt) for nt in data.node_types}
    assert declared_nodes == actual_nodes, (
        f"graph_schema.json node types disagree with the graph.\n"
        f"  declared not in graph: {sorted(declared_nodes - actual_nodes)}\n"
        f"  in graph not declared: {sorted(actual_nodes - declared_nodes)}"
    )

    # --- 4. edge type inventory ------------------------------------------- #
    declared_edges = set(schema_edges)
    actual_edges = {_schema_edge_key(et) for et in data.edge_types}
    assert declared_edges == actual_edges, (
        f"graph_schema.json edge types disagree with the graph.\n"
        f"  declared not in graph: {sorted(declared_edges - actual_edges)[:5]}\n"
        f"  in graph not declared: {sorted(actual_edges - declared_edges)[:5]}"
    )

    # --- 5. counts --------------------------------------------------------- #
    # Not redundant with the inventory checks above: identical type NAMES with
    # wrong COUNTS is what a metadata writer fed a stale or pre-deduplication
    # view of the graph produces, and that passes 3 and 4 untouched.
    for nt in data.node_types:
        declared = int(schema["node_types"][str(nt)]["count"])
        actual = int(data[nt].num_nodes)
        assert declared == actual, (
            f"{nt}: graph_schema.json declares count {declared}, graph has "
            f"{actual} nodes -- the writer saw a different version of the graph"
        )

    for et in data.edge_types:
        declared = int(schema_edges[_schema_edge_key(et)]["count"])
        actual = int(data[et].edge_index.shape[1])
        assert declared == actual, (
            f"{_schema_edge_key(et)}: graph_schema.json declares count "
            f"{declared}, graph has {actual} edges -- the writer saw a "
            f"different version of the graph (deduplication ordering?)"
        )


def _schema_edge_key(edge_type):
    """Render a PyG edge type the way MetadataWriter keys it.

    MetadataWriter builds ``f"({src}, {rel}, {dst})"`` (metadata_writer.py:158).
    Reproduced rather than imported because the point is to check the WRITTEN
    artifact against the graph; importing the writer's own formatting would make
    a change to that format invisible here.
    """
    src, rel, dst = edge_type
    return f"({src}, {rel}, {dst})"


def _assert_all_finite(tensor, label):
    """Every entry of a feature tensor must be finite.

    The realistic source is normalization: a feature that is constant across
    every node has zero variance, so a z-score divides by zero and the whole
    column becomes NaN. Nothing else in this suite notices. The tensor is still
    2D, still row-aligned, still a valid VectorLayout width -- the .pt looks
    perfect. NaN then propagates through the first forward pass and turns every
    gradient in the model to NaN, with no crash and no error message: training
    just silently produces nothing.

    The reproducibility tests are blind to it too. A NaN-poisoned tensor is
    produced identically on every run, so test_output_is_reproducible passes
    beside it -- confirming only that the graph is *consistently* broken.

    The message names the offending columns rather than just reporting "has
    NaN", because the column index is what maps back to a slot in
    slot_mapping.json and therefore to the feature that failed to normalize.
    NaN and infinity are counted separately: NaN means 0/0 (a constant column
    under z-score), infinity means x/0 (a non-zero value over a zero range).
    They point at different bugs, so collapsing them would discard the most
    useful part of the diagnosis.
    """
    import torch

    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return

    bad_columns = (~finite.all(dim=0)).nonzero().flatten().tolist()
    shown = ", ".join(str(c) for c in bad_columns[:8])
    if len(bad_columns) > 8:
        shown += f", ... ({len(bad_columns)} columns total)"

    nan_count = int(torch.isnan(tensor).sum())
    inf_count = int(torch.isinf(tensor).sum())

    raise AssertionError(
        f"{label} contains non-finite values: {nan_count} NaN, {inf_count} inf "
        f"across {int((~finite).sum())} of {tensor.numel()} entries.\n"
        f"Affected column(s): {shown}\n"
        f"Look up those indices in slot_mapping.json to identify the feature. "
        f"A NaN column is normally a zero-variance feature divided by its own "
        f"standard deviation during normalization; an inf column is a non-zero "
        f"value divided by a zero range. Fix the normalization path -- these "
        f"values reach the trainer and silently NaN out every gradient."
    )


def _assert_indices_are_in_range(data):
    """Every edge index must address a node that exists in its node type.

    The suite already checks each ``edge_index`` is shaped ``[2, E]``, but shape
    says nothing about the VALUES. An off-by-one or a stale mapping in
    node_mapper.py / edge_mapper.py can emit a destination id past the end of
    its node type and still satisfy every other assertion here. Downstream that
    either raises deep inside the first ``scatter``/``index_select`` -- far from
    the cause and hard to trace back -- or, worse, silently gathers the wrong
    feature rows, so the model trains on mismatched features and simply learns
    less than it should. Nothing fails; the graph is just quietly wrong.

    Delegates to PyG's own validator rather than hand-rolling the comparison, so
    the check tracks PyG's definition of a well-formed HeteroData as the library
    evolves. In torch-geometric 2.8 ``validate`` covers exactly the conditions
    that matter here -- negative indices, source indices >= num_src_nodes,
    destination indices >= num_dst_nodes -- plus undefined ``num_nodes`` and node
    types referenced by an edge type but absent from the graph. Its messages name
    the edge type, which endpoint, and the offending value.

    Asserts the RETURN VALUE as well as relying on the raise. ``validate``
    reports some conditions through ``warn_or_raise``, and requirements.txt
    admits any torch-geometric >= 2.6; if a version downgrades one of these to a
    warning, the raise would not fire but the returned status would still be
    False. Checking both means this guard cannot silently weaken under a version
    bump.

    One condition is filtered out: ``validate`` also reports isolated node types,
    always as a warning (it hardcodes ``raise_on_error=False``, since an isolated
    type "may be intended"). Here it IS intended -- these fixtures legitimately
    contain orphaned types, enumerated in KNOWN_ORPHANED_NODE_TYPES -- and
    ``_assert_no_node_type_is_orphaned`` already guards that property properly,
    against the allow-list rather than against zero. Letting the warning through
    would add known-benign noise to every report, which is how real warnings stop
    being read. Only this exact message is filtered; anything else ``validate``
    emits still surfaces.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*are isolated and are not referenced.*",
                category=UserWarning,
            )
            status = data.validate(raise_on_error=True)
    except ValueError as exc:  # pragma: no cover - only on a corrupt graph
        raise AssertionError(
            f"the graph is structurally invalid: {exc}\n\n"
            "An edge index addresses a node that does not exist in its node "
            "type. Look at the node_id assignment in node_mapper.py and the "
            "src/dst resolution in edge_mapper.py -- an edge whose endpoint id "
            "is >= that type's num_nodes means the two disagree about how many "
            "nodes the type has, or about which ids belong to it."
        ) from exc

    assert status, (
        "HeteroData.validate() reported the graph as invalid without raising "
        "(this torch-geometric version warns where 2.8 raises). Re-run with "
        "`-W error` to surface the specific condition."
    )


def _assert_temporal_types_are_not_sharded(data):
    """Periods must land in one node type per granularity, not one per source.

    TemporalUnifier types every source period as temporal_Source*, but many of
    those URIs already carry a source type -- cpi:2024 is both cpi:Year and
    temporal_SourceYear. node_mapper's default rule (fewest instances wins) then
    picks the per-namespace source type, because it is rarer, and one concept
    shards across every namespace that names it: on these fixtures 37 months
    over cpi_Month/jolts_Month/empsit_Month/eci_Month/temporal_SourceMonth, and
    14 years likewise, leaving temporal_SourceYear holding a single node. The
    sameAs edges from Unified{Month,Year} then land on whichever shard a period
    fell into, and a GNN sees unrelated types with no path between them.

    node_mapper._CANONICAL_TYPE_PRIORITY pins the temporal types ahead of the
    count heuristic to prevent that. This asserts the outcome rather than the
    mechanism: no per-namespace Month/Year/Quarter node type may survive
    alongside the unified temporal ones.

    Deliberately not scoped to a hardcoded list of namespaces -- a new BLS feed
    that types its periods must fail here too, not slip through.
    """
    granularities = ("Month", "Year", "Quarter")
    sharded = collections.defaultdict(list)
    for node_type in map(str, data.node_types):
        prefix, _, leaf = node_type.partition("_")
        # Unified* are the unifier's own hub nodes and are meant to be distinct;
        # only the SOURCE side is at risk of sharding.
        if prefix == "temporal" or leaf.startswith("Unified"):
            continue
        if leaf in granularities:
            sharded[leaf].append(f"{node_type} (n={data[node_type].num_nodes})")

    assert not sharded, (
        "temporal entities are sharded across per-source node types instead of "
        f"the unified temporal_Source* types: {dict(sharded)} — the owl:sameAs "
        "edges from Unified* cannot reach them all. Check "
        "node_mapper._CANONICAL_TYPE_PRIORITY."
    )


def _assert_no_node_type_is_orphaned(data):
    """No node type may exist with zero edges attached to it.

    A whole type with no incident edges is the visible symptom of a silent drop
    in edge resolution — those nodes were created but every triple that would
    have connected them went nowhere. That is exactly how the missing temporal
    dimension hid: bls_enrichment_UnifiedMonth (12) and UnifiedYear (8) sat in
    the graph as 20 isolated vertices a GNN can propagate nothing through, and
    nothing failed. Asserted for EVERY type, not just the temporal ones, so the
    next instance of this class of bug fails loudly instead of shipping.

    See KNOWN_ORPHANED_NODE_TYPES for the ones this does not yet cover — and
    note that both Unified* types are deliberately NOT on that list, so this is
    a live regression guard for the fix that removed them from it.
    """
    orphaned = {
        nt for nt, cnt in _incident_edge_counts(data).items()
        if data[nt].num_nodes > 0 and cnt == 0
    }
    assert not (orphaned - KNOWN_ORPHANED_NODE_TYPES), (
        "node types present in the graph with ZERO edges: "
        f"{sorted(orphaned - KNOWN_ORPHANED_NODE_TYPES)} — their triples were "
        "dropped during edge resolution"
    )

    # Keep the allowlist honest: an entry that is no longer orphaned must be
    # removed, or it silently exempts a type that later regresses.
    stale = KNOWN_ORPHANED_NODE_TYPES & set(map(str, data.node_types)) - orphaned
    assert not stale, (
        f"{sorted(stale)} are no longer orphaned — remove them from "
        "KNOWN_ORPHANED_NODE_TYPES"
    )


# The predicates that connect a measurement to when it happened. Source data
# references periods as bare URIs (cpi:February, eci:2024) and these are the
# only things pointing at them.
_TEMPORAL_PREDICATES = (
    "hasMonth", "hasYear", "hasStartMonth", "hasEndMonth",
    "hasStartYear", "hasEndYear",
)


def _assert_temporal_edges_present(config, data):
    """Measurements must actually be connected to their period.

    Every one of these triples used to be dropped (~1,205 on the turtle-parquet
    fixtures) because the URI on the object side carried no rdf:type, so
    node_mapper never made it a node. The graph had no temporal dimension at all:
    nothing recorded WHEN any measurement happened.

    The expectation is derived from the enriched triples rather than hardcoded:
    only BLS states periods as URIs this way, and not every fixture includes a
    BLS source (the ntriples fixture is market/noaa/sec only), so a fixed edge
    type would fail on data that never had the defect.

    Asserted precisely: every temporal triple whose BOTH endpoints are typed
    must become an edge. Typing the period fixed the object side (0 of 980 typed
    before, 891 after on the turtle-parquet fixtures). It does NOT reach the
    ~968 triples whose SUBJECT — the measurement itself, e.g.
    cpi:..._PercentChange — is also untyped and therefore also not a node. That
    is the same class of bug one hop up and a separate fix; counting it here
    would make this test a to-do rather than a guard.
    """
    import pandas as pd

    files = sorted(Path(config.enriched_parquet_path).rglob("*.parquet"))
    if not files:
        return  # split-mode runs may not re-emit the enriched frame
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    local = df["predicate"].str.rsplit("/", n=1).str[-1].str.rsplit("#", n=1).str[-1]
    temporal = df[local.isin(_TEMPORAL_PREDICATES)]
    if temporal.empty:
        return  # source has no URI-valued periods (e.g. the BLS-less fixture)

    typed = set(df.loc[df["predicate"] == _RDF_TYPE, "subject"])
    resolvable = temporal[
        temporal["subject"].isin(typed) & temporal["object"].isin(typed)
    ]
    assert not resolvable.empty, (
        f"all {len(temporal)} measurement->period triples have an untyped "
        "endpoint, so none can become an edge — the graph has no temporal "
        "dimension at all"
    )

    # PyG relation names are prefixed with the predicate's namespace
    # ("cpi_hasEndMonth"), so match on the suffix rather than equality.
    edges = sum(
        data[et].edge_index.shape[1]
        for et in data.edge_types
        if any(str(et[1]).endswith(p) for p in _TEMPORAL_PREDICATES)
    )
    assert edges >= len(resolvable), (
        f"{len(resolvable)} measurement->period triples have both endpoints "
        f"typed but only {edges} became edges — they were dropped during edge "
        "resolution"
    )


def _source_prefix(node_type):
    """The namespace prefix a node type belongs to, e.g. cpi_Index -> 'cpi'."""
    return str(node_type).split("_", 1)[0]


def test_cross_source_temporal_bridge(twin_runs):
    """A measurement from one source must reach one from another via a period."""
    _assert_cross_source_temporal_bridge(twin_runs.d1)


def _assert_cross_source_temporal_bridge(data):
    """A measurement from one source must reach one from another via a period.

    The design intends:

        cpi obs -> cpi:February -> unified:February <- eci:February <- eci obs

    Both hops used to be missing, and the gap was masked: cross_source_linker's
    temporal matcher was anchored only at the end, so it emitted
    `unified:September sameAs cpi:...PercentChange_..._September` — links to
    MEASUREMENTS, which are typed and therefore resolved into real edges. That
    fabricated 16 edge types' worth of connectivity and made unification look
    functional. Asserted end to end here so a bridge that only appears to exist
    cannot pass again.

    Traversed at the INSTANCE level, not over node types: a type-level path
    would be satisfied by cpi reaching February and eci reaching March, which
    is not a bridge. The two measurements must meet at the same period node.
    """
    # (node_type, node_id) -> set of neighbouring (node_type, node_id)
    neighbors = {}
    for et in data.edge_types:
        src_t, _rel, dst_t = (str(x) for x in et)
        edge_index = data[et].edge_index
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            neighbors.setdefault((src_t, s), set()).add((dst_t, d))
            neighbors.setdefault((dst_t, d), set()).add((src_t, s))

    unified_nodes = [
        (nt, i)
        for nt in map(str, data.node_types)
        if "Unified" in nt
        for i in range(data[nt].num_nodes)
    ]
    assert unified_nodes, "no unified temporal nodes in the graph"

    for unified in unified_nodes:
        # unified period -> the source periods it is sameAs -> what they date
        reached = {
            _source_prefix(measured[0])
            for source_period in neighbors.get(unified, ())
            for measured in neighbors.get(source_period, ())
            if measured != unified
        }
        if len(reached) >= 2:
            return

    raise AssertionError(
        "no unified temporal entity connects measurements from two different "
        "sources — the cross-source temporal bridge does not form"
    )


def _assert_sameas_links_only_temporal(config):
    """Every owl:sameAs in the enriched output must link two temporal entities."""
    import re

    import pandas as pd

    files = sorted(
        Path(config.enriched_parquet_path).rglob("*.parquet")
    )
    if not files:
        return  # split-mode runs may not re-emit the enriched frame
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    same_as = df[df["predicate"].str.endswith("#sameAs", na=False)]
    if same_as.empty:
        return

    months = (
        "January|February|March|April|May|June|July|August|September"
        "|October|November|December"
    )

    def temporal(uri):
        local = str(uri).rsplit("/", 1)[-1]
        return bool(
            re.fullmatch(months, local)
            or re.fullmatch(r"\d{4}", local)
            or re.fullmatch(rf"Year\d{{4}}", local)
            or re.fullmatch(r"Q[1-4]", local)
        )

    bad = same_as[~same_as["object"].map(temporal)]
    assert bad.empty, (
        f"{len(bad)} owl:sameAs triples point at a NON-temporal entity, e.g. "
        f"{bad['object'].iloc[0]}"
    )

def _load_node_index(pyg_output_path):
    """Read the node_index Parquet written beside the .pt into a DataFrame.

    Read with pandas rather than Spark: these tests already hold a
    SparkSession, but the artifact's whole point is being consumable by a
    downstream trainer or inference process that has no Spark at all. Reading it
    the way such a consumer would is the more meaningful assertion.
    """
    import pandas as pd

    from spark_jobs.pyg_builder.metadata_writer import derive_node_index_prefix

    path = Path(derive_node_index_prefix(pyg_output_path))
    assert path.is_dir(), f"missing node_index at {path}"
    return pd.read_parquet(path)


def _node_index_under(work_dir):
    """Locate and load the node_index for a pipeline run's work dir."""
    pt = next(Path(work_dir).rglob("hetero_data.pt"))
    return _load_node_index(str(pt))


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


def _twin_runs_sequential():
    """Whether to run the twin pipelines one-at-a-time instead of overlapped.

    Escape hatch for memory-constrained hosts: concurrent twin runs keep two
    ``--driver-memory 2g`` local[4] JVMs alive at once (and, with the early
    launch below, alongside the suite's own SparkSession). Set
    ``TWIN_RUNS_SEQUENTIAL=1`` to trade the wall-clock saving back for a lower
    memory ceiling. Correctness is identical either way — the runs are
    independent regardless of whether they overlap.
    """
    import os

    return os.environ.get("TWIN_RUNS_SEQUENTIAL", "") == "1"


@pytest.fixture(scope="module", autouse=True)
def _twin_runs_launcher(request, tmp_path_factory):
    """Start the twin pipeline runs *before* the other e2e tests in this module.

    The twin runs are external subprocesses, so while they execute this pytest
    process is only blocked in ``communicate()`` — time the rest of the module's
    tests can use. Launching here (module setup, before the first test) instead
    of lazily inside ``twin_runs`` overlaps the ~64s of subprocess time with the
    ~98s of in-session tests that run ahead of the reproducibility tests, rather
    than adding the two costs. ``twin_runs`` then joins whatever is already
    running.

    Only launches when a selected test actually needs ``twin_runs`` — a filtered
    run (e.g. ``-k full_turtle_parquet``) must not pay for two pipeline runs it
    will never read. Under ``TWIN_RUNS_SEQUENTIAL=1`` nothing is pre-launched;
    ``twin_runs`` does the work itself, one run at a time.

    Peak memory during the overlap is the suite's SparkSession plus two 2g
    driver JVMs. That is the trade for the wall-clock; see
    ``_twin_runs_sequential`` for the way out.
    """
    wanted = any(
        "twin_runs" in getattr(item, "fixturenames", ())
        for item in request.session.items
    )
    if not wanted or _twin_runs_sequential():
        yield None
        return

    base = tmp_path_factory.mktemp("reproducibility")
    run1, run2 = base / "run1", base / "run2"
    running = [(wd, _start_pipeline_subprocess(wd)) for wd in (run1, run2)]
    try:
        yield running
    finally:
        # A test may have failed before joining; never leak a live JVM.
        for _, proc in running:
            if proc.poll() is None:
                proc.kill()


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory, _twin_runs_launcher):
    """Run the full pipeline twice over the same fixture, once, for both tests.

    Determinism is an integration property no unit test can cover: it only emerges
    when every construction module composes over the real pipeline. Each run is
    isolated in its own process (see _start_pipeline_subprocess) and the two are
    launched concurrently — overlapping the independent runs ~halves the twin-run
    wall-clock. The pair is module-scoped so the expensive twin runs execute
    exactly once and are shared by every reproducibility test below.

    The runs are normally started early by ``_twin_runs_launcher`` and merely
    joined here. Under ``TWIN_RUNS_SEQUENTIAL=1`` there is nothing pre-launched,
    so this runs them itself, strictly one at a time.
    """
    if _twin_runs_launcher is None:
        base = tmp_path_factory.mktemp("reproducibility")
        run1, run2 = base / "run1", base / "run2"
        for wd in (run1, run2):
            _finish_pipeline_subprocess(wd, _start_pipeline_subprocess(wd))
    else:
        (run1, _), (run2, _) = _twin_runs_launcher
        for wd, proc in _twin_runs_launcher:
            _finish_pipeline_subprocess(wd, proc)

    d1 = _load_hetero(next(run1.rglob("hetero_data.pt")))
    d2 = _load_hetero(next(run2.rglob("hetero_data.pt")))
    return _TwinRuns(d1, d2, run1, run2)


def test_output_is_reproducible(twin_runs):
    """Two runs over the same fixture → identical graph, tensors, and metadata.

    Full-strength determinism guard: node/edge inventory, per-type counts, every
    edge_index, edge features, the feature tensors of EVERY node type, and every
    metadata file compared exactly (modulo the allow-listed build_timestamp).

    This used to carry two concessions to a known bug — jolts_* feature tensors
    were skipped, and metadata was compared order-insensitively because list
    entries drifted between runs. Both causes are fixed (#221: rdflib's random
    per-parse blank-node labels, and unordered collect() in the metadata
    builders), so the comparisons are now exact and the guard is strictly
    stronger than before the bug existed. Do not reintroduce a canonicalizing
    comparison here: ordering drift is itself the regression this catches.
    """
    import torch

    d1, d2 = twin_runs.d1, twin_runs.d2

    # --- same node/edge type inventory ---
    assert sorted(d1.node_types) == sorted(d2.node_types)
    assert sorted(map(tuple, d1.edge_types)) == sorted(map(tuple, d2.edge_types))

    # --- per-type counts + ALL feature tensors bit-identical ---
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

    # --- metadata reproducible EXACTLY (modulo the allow-listed timestamp) ---
    # Compared with ordering intact: entry order drifting run-to-run is exactly
    # the #221 regression, so canonicalizing it away here would blind the guard.
    m1, m2 = _metadata_by_name(twin_runs.run1), _metadata_by_name(twin_runs.run2)
    assert set(m1) == METADATA_FILES and set(m2) == METADATA_FILES
    for name in METADATA_FILES:
        j1 = _load_metadata(m1[name], name)
        j2 = _load_metadata(m2[name], name)
        assert j1 == j2, f"{name} differs between runs"

    # --- identity map reproducible, rows and order alike ---
    # A drifting node_index is worse than none: it would silently attribute
    # predictions to the WRONG entity rather than failing. Compared with row
    # order intact, since the artifact is written sorted and coalesced to one
    # file precisely so it is stable.
    i1, i2 = _node_index_under(twin_runs.run1), _node_index_under(twin_runs.run2)
    assert i1.equals(i2), "node_index differs between runs"


def test_jolts_features_reproducible(twin_runs):
    """The jolts_* feature tensors + metadata entry ordering must match across runs.

    Was a non-strict xfail for the known jolts_* non-determinism; both causes are
    fixed in #221 so it is now an ordinary passing guard. jolts_* is where the bug
    surfaced (JOLTS models measurements as RDF blank nodes, and rdflib assigned
    them a fresh random label on every parse, which then hashed into different
    feature slots), which is why this slice keeps a dedicated test — a regression
    in blank-node handling shows up here first and most legibly.

    Asserts the jolts_* type(s) are actually present so an empty selection cannot
    silently pass on a comparison that never ran.
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

    # Metadata compared in SERIALIZED form, which catches what the parsed-object
    # comparison in test_output_is_reproducible cannot: dict key/insertion order
    # (ontology_schema.json), on top of list ordering (slot_mapping.json). Both
    # drifted before #221 — the unordered collect() in the metadata builders fed
    # dict insertion order as well as list order. Together the two tests assert
    # the full "byte-reproducible modulo build_timestamp" property.
    m1, m2 = _metadata_by_name(twin_runs.run1), _metadata_by_name(twin_runs.run2)
    for name in METADATA_FILES:
        s1 = json.dumps(_load_metadata(m1[name], name))
        s2 = json.dumps(_load_metadata(m2[name], name))
        assert s1 == s2, f"{name} entry ordering differs between runs"
