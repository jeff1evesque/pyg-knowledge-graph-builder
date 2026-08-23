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
import re
from pathlib import Path

import pytest

from spark_jobs.utils.rdf_utils import classify_edge_origin
from spark_jobs.pyg_builder.metadata_writer import (
    LATEST_PARTITION,
    derive_metadata_prefix,
)

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
    #
    # The `latest` alias holds a second copy of graph_schema.json, so the scan
    # excludes it: without that, `found["graph_schema.json"]` would be whichever
    # of the two rglob happened to yield last, and every assertion below would
    # be reading a nondeterministically chosen file.
    found = {
        p.name: p
        for p in Path(work_dir).rglob("*.json")
        if p.name in METADATA_FILES and LATEST_PARTITION not in p.parts
    }
    missing = METADATA_FILES - set(found)
    assert not missing, f"missing metadata files: {sorted(missing)}"
    for name, path in found.items():
        with open(path) as fh:
            content = json.load(fh)
        assert content, f"{name} is empty"

    # --- the `latest` alias is written, and matches the period copy ---
    # Asserted here rather than only in unit tests because the unit tests
    # exercise write_latest_alias() directly; this is the only thing that proves
    # save_final_artifacts() actually calls it. Dropping the call would leave
    # every unit test passing and the consumer's fixed key permanently stale.
    alias = Path(derive_metadata_prefix(config.latest_pyg_path)) / "graph_schema.json"
    assert alias.is_file(), f"latest alias not written: {alias}"
    assert alias.read_bytes() == found["graph_schema.json"].read_bytes(), (
        "latest alias differs from the period copy it should mirror"
    )

    # --- owl:sameAs only ever links temporal entities to temporal entities ---
    # cross_source_linker used to match subjects with "(January|...)$" -- anchored
    # only at the end -- so any URI merely ENDING with a month name was asserted
    # to BE that month (unified:September owl:sameAs cpi:...PercentChange_...
    # _September). owl:sameAs is RDF's strongest claim, so those became real
    # graph edges. Guarded here as well as in the unit tests because this asserts
    # the property of the WHOLE pipeline -- any enricher reintroducing it fails.
    _assert_sameas_links_only_unified_vocabularies(config)

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

    # ...and that the origin recorded is the one its endpoints imply. The two
    # assertions above only ask whether the value is a legal word, which is why
    # they passed while origin was keyed by relation name alone: one relation
    # spans many endpoint pairs, so all of them kept whichever origin the
    # constructor built last. 3 edge types (90 edges) published `raw` for links
    # the pipeline had inferred -- a legal value, the wrong one, and exactly the
    # "inference presented as observed fact" the field exists to rule out.
    mislabelled = [
        (key, entry["origin"], expected)
        for key, entry in schema["edge_types"].items()
        if (expected := classify_edge_origin(
            entry["predicate_uri"], entry["src_type"], entry["dst_type"]
        )) != entry["origin"]
    ]
    assert not mislabelled, (
        f"{len(mislabelled)} edge types record an origin their endpoints "
        f"contradict: {mislabelled[:5]}"
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

    # --- and two rows are never the same entity spelled two ways ---
    # is_unique above compares URIs as strings, so it is satisfied by exactly
    # the defect this looks for: one filer under both a padded and an unpadded
    # CIK is two distinct strings and two distinct nodes.
    _assert_no_entity_is_split_by_identifier_padding(index)

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

    # --- and no whole SOURCE is connected only to itself ---
    _assert_no_source_builds_as_an_island(data)

    # --- and every inferred link the pipeline claims to make actually exists ---
    _assert_declared_bridges_resolve(data)
    _assert_bridge_absences_are_still_true(data)

    # --- and exists at full strength, rather than resolving into one edge ---
    _assert_declared_bridges_are_not_degenerate(config)

    # --- and points at the right company, not merely at some company ---
    _assert_bridged_companies_agree(config)

    # --- a chain of N transactions is N-1 edges, not "some precedes present" ---
    _assert_transaction_chains_are_complete(config)


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
# nodes.
#
# Eight more went when the fixtures were regenerated against the refactored
# crawl output. Seven are BLS classes upstream no longer emits at all —
# cpi_TwelveMonthPercentChange, eci_WagesAndSalariesIndexData,
# empsit_HouseholdData, empsit_OccupationData, empsit_PeriodOfService,
# empsit_UnemploymentDurationData, empsit_UnemploymentReasonData — so they can
# no longer name a node type in any graph built from this data. The eighth is
# filings_SECFiling, which was orphaned because the SEC fixture was three
# scraped-HTML stubs pointing at nothing; a real filing states hasIssuer,
# hasFilingDate and hasPeriodOfReport, and all three resolve.
#
# The two market_quotes_* entries are gone as of the cross-source join fixes.
# They read "unknown_*" until the source namespaces were re-homed, then sat here
# with zero edges: cross_source_linker detected market data by the FEEDS
# namespace only, so a graph of quote snapshots reported no market source at
# all and the SEC↔market bridge was skipped without running. Both snapshot types
# now reach unified companies through their ticker.
# bls_enrichment_GeographicRegion was the last entry here and is now empty
# rather than exempt: cross_source_linker used to type and label all 50 states
# unconditionally while linking them only where a LAUS or NOAA entity named one,
# so a fixture set with no LAUS feed carried 50 isolated region nodes. The
# scaffolding is now scoped to the regions something actually points at, which
# removes the orphans without weakening this check — an unlinked region is no
# longer a node at all.
KNOWN_ORPHANED_NODE_TYPES: set[str] = set()


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

    # Where the literal_values segment starts, for the has_features check below.
    literal_segment = next(
        s for s in node_spec["segments"] if s["name"] == "literal_values"
    )
    literal_start = int(literal_segment["start"])

    schema_nodes = schema["node_types"]
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

            # has_features means "carries literal-value features" (schema 1.1).
            # Asserting it against the tensor is what makes the flag
            # trustworthy: through schema 1.0 it was `count > 0` -- the node
            # count -- so it was true for every type and said nothing. The
            # equivalent edge-side check is a few lines below.
            declared_has = bool(schema_nodes[str(nt)]["has_features"])
            actual_has = bool(store.x[:, literal_start:].any())
            assert declared_has == actual_has, (
                f"{nt}: graph_schema.json declares has_features="
                f"{declared_has}, but the tensor's literal_values segment "
                f"(columns {literal_start}..{declared_node_dim - 1}) "
                f"{'has' if actual_has else 'has no'} non-zero values. The "
                f"flag must describe the tensor, not the node count."
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

    _assert_edge_features_are_not_vacuous(data, spec, schema, found)
    _assert_node_sub_segments_are_not_vacuous(data, spec)
    _assert_class_identity_is_recoverable(data, found)
    _assert_source_type_uris_identify_their_node_type(schema, found)
    _assert_ontology_schema_states_whether_mapping_ran(found)
    _assert_ontology_sub_segments_are_populated(spec)
    _assert_derived_axioms_declare_their_provenance(found)
    _assert_relation_groups_partition_the_edge_types(schema)

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


def _assert_ontology_schema_states_whether_mapping_ran(found):
    """ontology_schema.json must say whether ontology mapping ran.

    An all-empty hierarchy has two causes that demand opposite responses --
    sources that declare no subsumption (data to work with) and a mapping
    phase that never ran (a re-run) -- and both serialize to
    ``superclass_chain: []``. The 2026-07-29 build could not be told apart
    from either by reading the file.

    The job manifest cannot settle it: the split path below runs pyg_only,
    whose own enable_ontology_mapping flag describes a phase that job never
    reaches. So the flag is derived from the enriched triples, and this
    asserts it survives the Parquet round-trip -- every mode here enriches
    with mapping on, so anything but True means the detection is broken, not
    the pipeline.
    """
    ontology = json.loads(Path(found["ontology_schema.json"]).read_bytes())

    assert "ontology_mapping_enabled" in ontology, (
        "ontology_schema.json does not record whether ontology mapping ran, "
        "so an empty superclass_chain cannot be diagnosed from the file"
    )
    assert ontology["ontology_mapping_enabled"] is True, (
        f"these runs enrich with ontology mapping enabled, but "
        f"ontology_schema.json reports it did not run "
        f"({ontology.get('ontology_mapping_evidence')})"
    )


def _assert_ontology_sub_segments_are_populated(spec):
    """The three ontology sub-segments must carry signal, not just be legal.

    class_hierarchy (64 dims), domain_range (111) and property_hierarchy (81)
    were all structurally zero -- 256 of 1024, a quarter of every node vector
    -- because no source declares rdfs:subClassOf, rdfs:domain, rdfs:range or
    rdfs:subPropertyOf anywhere. All three are now derived.

    _assert_node_sub_segments_are_not_vacuous already proves that a
    sub-segment claiming features has some. What it CANNOT catch is a
    regression that stops deriving and honestly re-declares
    populated=false with an empty_reason: that is a legal spec, so the
    vacuity guard passes and a quarter of the vector silently goes dead
    again. This names the three and requires the populated path.
    """
    subs = {
        s["name"]: s
        for seg in spec["node_features"]["segments"]
        for s in seg.get("sub_segments", [])
    }

    for name in ("class_hierarchy", "domain_range", "property_hierarchy"):
        sub = subs.get(name)
        assert sub is not None, f"feature_spec.json has no {name} sub-segment"
        assert sub.get("populated", True) is True, (
            f"sub-segment {name!r} ({sub['dim']} dims) is declared "
            f"unpopulated: {sub.get('empty_reason')!r}. These are derived by "
            f"OntologyMapper, so an honest empty_reason here means the "
            f"derivation stopped working, not that the data changed."
        )


def _assert_derived_axioms_declare_their_provenance(found):
    """An inferred axiom must never be published as a declared one.

    rdfs:domain derived from observed usage is shaped exactly like one a
    source asserted, and they license different reasoning: "used on class C
    in this data" is weaker than "its domain is C". ontology_schema.json is
    the only place that distinction survives.
    """
    ontology = json.loads(Path(found["ontology_schema.json"]).read_bytes())
    provenance = ontology.get("provenance")

    assert provenance, (
        "ontology_schema.json carries no provenance block, so a derived "
        "rdfs:domain is indistinguishable from a declared one"
    )
    for key in ("class_hierarchy", "property_hierarchy",
                "property_domain", "property_range"):
        assert provenance.get(key), f"no provenance recorded for {key}"
        assert provenance[key] != "declared by the source", (
            f"{key} is reported as declared by the source, but no source in "
            f"these fixtures declares it -- the pipeline derived it, and "
            f"saying otherwise overstates what the axioms license"
        )

    # And the question #273 asked ("can any source emit rdfs:subClassOf?")
    # must stay answerable from the artifact. Counting the triples in an
    # ENRICHED graph now finds 34 and says yes; the true answer is still no,
    # and only the per-route counts carry it.
    derived = ontology.get("derived_axioms") or {}
    assert derived, "ontology_schema.json carries no per-route axiom counts"

    for name in ("class_hierarchy", "property_hierarchy",
                 "property_domain", "property_range"):
        detail = derived.get(name) or {}
        assert detail.get("total"), f"{name} reports no axioms at all"
        counts = detail["counts"]
        assert counts.get("declared") == 0, (
            f"{name} reports {counts['declared']} axiom(s) as declared by a "
            f"source, but no source in these fixtures declares any -- every "
            f"one was derived by OntologyMapper"
        )
        # The routes must be named, not lumped: a curated edge and a
        # spelling-rule guess do not deserve the same trust.
        routes = {k for k, v in counts.items() if k != "declared" and v}
        assert routes, f"{name} has axioms but names no route"
        assert detail["axioms"], f"{name} lists no attributable axioms"

    # hierarchy_source must not read as a declaration.
    assert not ontology["hierarchy_source"].startswith("rdfs:"), (
        f"hierarchy_source is {ontology['hierarchy_source']!r}, which reads "
        f"as a source declaration for a hierarchy this pipeline derived"
    )

    coverage = ontology.get("property_schema_coverage") or {}
    assert coverage.get("data_predicates"), "no coverage reported"
    assert coverage["with_domain"] > 0, (
        f"no predicate got an rdfs:domain: {coverage}"
    )
    assert coverage["with_range"] > 0, (
        f"no predicate got an rdfs:range: {coverage}"
    )


def _assert_source_type_uris_identify_their_node_type(schema, found):
    """Each node type's source_type_uri must be the class it is named after.

    Enrichment gives entities a unified supertype alongside their specific
    type, so a node type has several candidate rdf:type URIs and one is picked.
    Picking by sort order -- the old rule -- reported a URI belonging to another
    class whenever the supertype sorted earlier: three jolts rate types all
    claimed bls_enrichment/RateMeasurement, and 43 node types shared 40 URIs.

    Two things go wrong when it is unreliable, neither of which fails anything
    else: graph_schema.json publishes the wrong class for the type, and
    ontology_schema.json looks up superclass_chain and defined_properties BY
    that URI, so it attributes another class's hierarchy and property schema to
    this node type.

    Asserted through ontology_schema.json's own uri_to_pyg_name rather than by
    re-deriving names here, so this tracks the production naming rule and keeps
    holding if the namespace table changes.
    """
    ontology = json.loads(Path(found["ontology_schema.json"]).read_bytes())
    uri_to_name = ontology.get("uri_to_pyg_name") or {}
    if not uri_to_name:
        return

    mismatched = {}
    for node_type, entry in schema["node_types"].items():
        uri = entry.get("source_type_uri")
        if not uri:
            continue
        # A URI absent from the map cannot be checked -- it is a class no
        # namespace prefix matched, which the naming rule handles separately.
        derived = uri_to_name.get(uri)
        if derived is not None and derived != node_type:
            mismatched[node_type] = (uri, derived)

    assert not mismatched, (
        f"{len(mismatched)} node type(s) report a source_type_uri belonging to "
        f"a different class. graph_schema.json publishes it, and "
        f"ontology_schema.json keys superclass_chain / defined_properties off "
        f"it, so both describe the wrong class for these types:\n"
        + "\n".join(
            f"  {nt} -> {uri} (which names {derived})"
            for nt, (uri, derived) in sorted(mismatched.items())[:8]
        )
    )

    # And the mapping must not collapse: two node types sharing one URI means
    # at least one of them is misattributed.
    by_uri = {}
    for node_type, entry in schema["node_types"].items():
        uri = entry.get("source_type_uri")
        if uri:
            by_uri.setdefault(uri, []).append(node_type)
    collapsed = {u: n for u, n in by_uri.items() if len(n) > 1}
    assert not collapsed, (
        f"{len(schema['node_types'])} node types share only "
        f"{len(by_uri)} distinct source_type_uri values; collapsed: "
        + "; ".join(f"{u} <- {sorted(n)}" for u, n in sorted(collapsed.items()))
    )


def _assert_relation_groups_partition_the_edge_types(schema):
    """Weight tying must cover every edge type exactly once.

    A PyG edge type is (src, relation, dst) and a heterogeneous conv allocates
    one weight matrix per edge type, so a relation spanning many endpoint pairs
    multiplies out -- 69 relations produced 770 edge types on these fixtures,
    ~101M parameters at 1024->128 for ~10k edges. The .pt cannot collapse them
    (edge_index values are node IDs local to their node type), so
    graph_schema.json publishes which keys are the same relation and a consumer
    builds one weight matrix per group.

    That only works if the grouping partitions: an edge type in no group gets
    no weights, one in two groups has its messages counted twice. Asserted over
    the artifact rather than the builder, because the trainer reads the file.
    """
    groups = schema.get("relation_groups")
    assert groups, "graph_schema.json declares no relation_groups"

    grouped = [key for g in groups.values() for key in g["edge_types"]]
    assert len(grouped) == len(set(grouped)), (
        "an edge type appears in more than one relation group"
    )
    assert sorted(grouped) == sorted(schema["edge_types"]), (
        "relation_groups does not cover the edge types exactly:\n"
        f"  grouped not declared: {sorted(set(grouped) - set(schema['edge_types']))}\n"
        f"  declared not grouped: {sorted(set(schema['edge_types']) - set(grouped))}"
    )

    # Each edge type's own relation_group must be the group holding it, or a
    # consumer looking up either way round gets different answers.
    holder = {
        key: name for name, g in groups.items() for key in g["edge_types"]
    }
    for key, entry in schema["edge_types"].items():
        assert entry["relation_group"] == holder[key], (
            f"{key} declares relation_group={entry['relation_group']!r} but is "
            f"listed under {holder[key]!r}"
        )

    # Endpoint indices must address the node-type table.
    by_index = {e["index"]: n for n, e in schema["node_types"].items()}
    for key, entry in schema["edge_types"].items():
        assert by_index[entry["src_type_index"]] == entry["src_type"], key
        assert by_index[entry["dst_type_index"]] == entry["dst_type"], key

    assert int(schema["summary"]["total_relation_groups"]) == len(groups)


def _assert_class_identity_is_recoverable(data, found):
    """The class_identity segment must be able to separate the classes present.

    Layout-agnostic on purpose. It reads the capacity the run itself recorded
    rather than any fixed width, so re-tuning the segment fractions -- or
    changing what the node vector carries -- keeps this test meaningful instead
    of turning it into a stale constant to update.

    The failure it guards is silent in every other signal: past the segment
    width the codes stay distinct (4-hot into 64 slots has C(64,4) patterns, so
    a full-source build had 118 distinct codes in 64 dims), the vectors are the
    declared width, the metadata is self-consistent, and the job exits 0 -- but
    no readout layer can recover class identity from a rank-limited code.
    """
    slot_mapping = json.loads(Path(found["slot_mapping.json"]).read_bytes())
    ci = slot_mapping.get("collision_report", {}).get("class_identity")
    if not ci or not ci.get("segment_dim"):
        # slot mapping < 1.1 could not report capacity.
        return

    shared = ci.get("classes_sharing_a_code") or []
    assert not shared, (
        f"{len(shared)} group(s) of classes share an identical "
        f"class_identity code and are indistinguishable to any model: "
        f"{shared[:3]}"
    )
    assert ci["linearly_separable"], (
        f"class_identity cannot separate this graph's classes: "
        f"{ci['total_classes']} classes into {ci['segment_dim']} dims "
        f"(headroom {ci['headroom_classes']}). At most segment_dim codes can "
        f"be linearly independent, so class identity is not recoverable -- "
        f"even though {ci['distinct_codes']} of the codes are distinct, which "
        f"is why nothing else here fails.\n"
        f"  Fix by giving class_identity more dims "
        f"(_SEG1_CLASS_IDENTITY_FRAC) or raising feature_config.vector_dim."
    )
    # The graph's own node types are the ground truth for the class count.
    assert ci["total_classes"] >= len(data.node_types), (
        f"slot_mapping reports {ci['total_classes']} classes but the graph "
        f"has {len(data.node_types)} node types; the capacity check above is "
        f"then measuring the wrong population"
    )


def _assert_node_sub_segments_are_not_vacuous(data, spec):
    """Every sub-segment that claims features must carry some, somewhere.

    Written over whatever feature_spec.json declares rather than over a named
    list, so it covers class_hierarchy, property_hierarchy, domain_range and
    ontology_source alike -- and any sub-segment added later, without an edit
    here.

    This is the check that was missing. The encoder's unit tests feed
    synthetic rdfs:subClassOf triples and pass, so they cannot notice that no
    real source emits any: class_hierarchy was 64 permanently-zero dims of
    every 1024-d vector, and every consistency assertion in this file still
    passed. domain_range and property_hierarchy are in the same position on
    the real data (no rdfs:domain, rdfs:range or rdfs:subPropertyOf anywhere
    in the 2026-07-29 run), which is why this is written generally.

    A sub-segment is allowed to be empty -- sources genuinely differ in what
    they declare -- but then the spec must say so and say why, so the dead
    slice is visible in the artifact instead of only in the encoder.
    """
    live = {
        nt: _store_get(data[nt], "x") for nt in data.node_types
    }
    if not any(x is not None for x in live.values()):
        return  # no node features at all: nothing to be vacuous about

    for segment in spec["node_features"]["segments"]:
        for sub in segment.get("sub_segments", []):
            lo, hi = int(sub["start"]), int(sub["end"]) + 1
            carriers = [
                str(nt) for nt, x in live.items()
                if x is not None and bool(x[:, lo:hi].any())
            ]
            populated = sub.get("populated", True)

            if populated:
                assert carriers, (
                    f"sub-segment {sub['name']!r} claims "
                    f"{sub['dim']} dims [{sub['start']}-{sub['end']}] but is "
                    f"zero for EVERY node type. Either the source predicate "
                    f"it encodes is absent from the data -- in which case "
                    f"feature_spec.json must declare populated=false with an "
                    f"empty_reason -- or the encoder stopped filling it. "
                    f"Nothing else in this file fails when a sub-segment is "
                    f"dead, which is how it shipped."
                )
            else:
                assert sub.get("empty_reason"), (
                    f"sub-segment {sub['name']!r} is declared unpopulated "
                    f"but carries no empty_reason. The reason is the point: "
                    f"[] alone cannot distinguish 'no source declares this' "
                    f"from 'the encoder broke'."
                )
                assert not carriers, (
                    f"sub-segment {sub['name']!r} is declared unpopulated "
                    f"({sub['empty_reason']!r}) but node type(s) {carriers} "
                    f"carry non-zero values in [{sub['start']}-"
                    f"{sub['end']}]. The spec understates what the vector "
                    f"holds, so a consumer would discard real signal."
                )


def _assert_edge_features_are_not_vacuous(data, spec, schema, found):
    """If any relation classified into an enabled category, edge_attr must
    exist.

    Every other edge-side assertion in this file is guarded by
    `if has_attr:` / `if "edge_attr" in store:`, which is right for consistency
    checking -- an edge type without features has nothing to compare -- but
    means a graph where NO edge type got features satisfies all of them
    vacuously. That is what shipped: the 2026-07-29 cluster run produced 704
    edge types, zero with edge_attr, and would have passed every check here,
    because each relation classified "generic" and no enabled_categories value
    could select it.

    The expectation is derived from the run's OWN artifacts rather than
    recomputed here -- `derivation_methods` (relation -> category, skip already
    excluded) crossed with `encoding_config.json`'s enabled_categories. Two
    reasons: the test must not carry a second copy of the classification rules
    it is checking, and a fixture set that legitimately contains no featurizable
    relation must not fail. The ntriples fixtures are exactly that case today
    (4 generic + 2 skip edge types, nothing else), so a flat "must be non-empty"
    would be red for a reason that is not this contract.

    What that leaves uncovered is deliberate: zero featurizable relations is
    reported at runtime by EdgeFeatureExtractor's warning, not here.
    """
    edge_spec = spec["edge_features"]
    if not edge_spec.get("enabled"):
        # A deliberately disabled run has nothing to be vacuous about.
        return

    encoding = json.loads(Path(found["encoding_config.json"]).read_bytes())
    enabled = set(encoding["edge_features"]["enabled_categories"])
    # relation -> category, as production classified it. "skip" is already
    # filtered out by MetadataWriter.
    categories = edge_spec.get("derivation_methods", {})
    featurizable = sorted(
        rel for rel, cat in categories.items() if cat in enabled
    )

    with_attr = [
        str(et) for et in data.edge_types
        if _store_get(data[et], "edge_attr") is not None
    ]
    declared = list(edge_spec.get("edge_types_with_features", []))

    if featurizable:
        assert with_attr, (
            f"{len(featurizable)} relation(s) classified into an enabled "
            f"category, but NOT ONE of the {len(data.edge_types)} edge types "
            f"carries edge_attr. Every consistency check above passes "
            f"vacuously, so nothing else fails and the job exits 0.\n"
            f"  enabled categories: {sorted(enabled)}\n"
            f"  should have been featurized: {featurizable[:8]}"
            f"{'...' if len(featurizable) > 8 else ''}\n"
            f"  feature_spec.json declares edge_types_with_features="
            f"{declared[:5]}{'...' if len(declared) > 5 else ''}"
        )
        assert declared, (
            f"{len(with_attr)} edge type(s) carry edge_attr but "
            f"feature_spec.json declares edge_types_with_features empty. A "
            f"trainer reading the spec would never look at the features that "
            f"are there.\n  carry edge_attr: {sorted(with_attr)[:5]}"
        )
    else:
        # No relation could be featurized under this config. Absence is then
        # correct -- but it must be absence on BOTH sides.
        assert not with_attr, (
            f"no relation classified into an enabled category "
            f"({sorted(enabled)}), yet {len(with_attr)} edge type(s) carry "
            f"edge_attr: {sorted(with_attr)[:5]}"
        )

    # The count must agree either way: graph_schema.json's summary is what a
    # reader trusts for "how many relations are featurized".
    summary_count = int(schema["summary"]["edge_types_with_features"])
    assert summary_count == len(with_attr), (
        f"graph_schema.json summary declares edge_types_with_features="
        f"{summary_count} but {len(with_attr)} edge type(s) actually carry "
        f"edge_attr"
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


def _source_families():
    """(node_type_prefix -> source family) and the set of pipeline hub families.

    Built from the rdf_utils namespace constants rather than spelled out, so a
    re-homed namespace moves both at once. Restating prefixes here is the same
    staleness that produced the defect this file exists to catch.

    A "family" is one upstream source, which is NOT one namespace: SEC spans
    filings and sec-common, and a filing pointing at its own sec-common Date
    node has not left SEC. Grouping by namespace instead of by
    family is exactly the mistake that made the first version of the assertion
    below pass on the broken graph: node types were split on their first
    underscore, so filings_SECFiling and sec_common_Date read as two different
    sources and `filings_SECFiling -> sec_common_Date` counted as leaving. Those
    two edge types were the ENTIRE cross-namespace footprint of SEC before the
    joins were fixed, and the issue named them specifically as not counting.
    """
    from spark_jobs.utils.rdf_utils import (
        ALERT, BLS_COMMON, BLS_ENRICHMENT, CAP, CPI, ECI, EMPSIT, JOLTS, LAUS,
        MARKET_ENRICHMENT, MARKET_QUOTES,
        METRO, NAMESPACE_PREFIXES, NOAA_ENRICHMENT, PPI, REALER,
        SEC_COMMON, SEC_ENRICHMENT, SEC_FILINGS,
        SOURCE_TEMPORAL, UNIFIED, WEATHER, WKYENG, XIMPIM,
    )

    prefix_of = dict(NAMESPACE_PREFIXES)
    families = {
        "bls": (CPI, PPI, ECI, EMPSIT, JOLTS, LAUS, METRO, REALER, WKYENG,
                XIMPIM, BLS_COMMON),
        "sec": (SEC_FILINGS, SEC_COMMON),
        "market": (MARKET_QUOTES,),
        "noaa": (CAP, WEATHER, ALERT),
        # Everything this pipeline mints itself. These are how sources REACH
        # each other, so they are destinations rather than islands — "does it
        # reach outside itself" is vacuous for a hub, and their own
        # connectivity is asserted by _assert_cross_source_temporal_bridge and
        # _assert_temporal_edges_present.
        "PIPELINE": (BLS_ENRICHMENT, SEC_ENRICHMENT, MARKET_ENRICHMENT,
                     NOAA_ENRICHMENT, UNIFIED, SOURCE_TEMPORAL),
    }

    by_prefix = {}
    for family, namespaces in families.items():
        for ns in namespaces:
            by_prefix[prefix_of[str(ns)]] = family
    return by_prefix, {"PIPELINE"}


def _family_of(node_type, by_prefix):
    """The source family a node type belongs to, by longest prefix match.

    Longest-first because the prefixes nest: bls_common_ and bls_enrichment_
    both start with bls, and market_feeds_ is a prefix of market_feeds_options_.
    """
    name = str(node_type)
    for prefix in sorted(by_prefix, key=len, reverse=True):
        if name.startswith(prefix + "_"):
            return by_prefix[prefix]
    return None


def _assert_no_source_builds_as_an_island(data):
    """Every source must have at least one edge leaving it.

    This is the check whose absence let four dead vocabulary constants ship
    green. The suite was thorough about internal consistency — the graph builds,
    every index is in range, the metadata describes the tensors truthfully — and
    a graph of perfectly well-formed disconnected islands satisfies all of it.
    Measured on the fixtures at the time: 12 edge types touched filings_*, of
    which 2 left the filings_ namespace and both landed on SEC's own date nodes;
    market_quotes_* had zero edges of any kind. SEC and market contributed node
    features and participated in no message passing with the rest of the graph,
    so no cross-source signal — filings against economic indicators, quotes
    against filings — was learnable, because the edges did not exist.

    Asserted per SOURCE rather than per node type because that is the level the
    defect lives at. Four separate joins each keyed on a term the upstream data
    no longer emits; each matched nothing, emitted nothing and raised nothing,
    and the shared symptom was a whole source sitting alone.
    _assert_no_node_type_is_orphaned already covers the finer-grained "this one
    type got dropped" case, and would be satisfied by a source whose types are
    all busily connected to each other and to nobody else.

    Reaching a PIPELINE hub counts as leaving, because routing through a hub is
    how this graph is designed to connect sources — a filing and a quote meet at
    a unified company, not by a direct edge. That makes this necessary but not
    sufficient: two sources can both reach the hub TYPE while attaching to
    different hub NODES and never meeting. _assert_cross_source_temporal_bridge
    is the one that checks they actually meet, at the instance level.

    A source with no nodes at all is not an island — it is absent, which is a
    fixture-coverage question and not this assertion's business.
    """
    by_prefix, hubs = _source_families()

    unmapped = sorted(
        {str(nt) for nt in data.node_types if _family_of(nt, by_prefix) is None}
    )
    assert not unmapped, (
        f"node types belong to no known source family: {unmapped} — either a "
        "namespace was added without updating _source_families(), or these "
        "node types are named after a namespace nothing declares"
    )

    present = {_family_of(nt, by_prefix) for nt in data.node_types}
    reaches_out = set()
    for src, _rel, dst in data.edge_types:
        src_family = _family_of(src, by_prefix)
        dst_family = _family_of(dst, by_prefix)
        if src_family != dst_family:
            reaches_out.add(src_family)
            reaches_out.add(dst_family)

    islands = present - reaches_out - hubs
    assert not islands, (
        f"sources that build as islands: {sorted(islands)} — every edge type "
        "touching them has both endpoints inside the same source, so they "
        "contribute node features and participate in no cross-source message "
        "passing. Check that the joins bridging them key on terms the data "
        "actually emits: a join matching nothing emits nothing and raises "
        "nothing."
    )


# The links the pipeline INFERS, which must each resolve into at least one edge.
#
# Every defect in this class so far — seven of them — was a join keyed on a term
# the upstream data no longer emits. A join like that matches nothing, emits
# nothing and raises nothing, so the only visible trace is a relation that is
# simply absent from the graph. Nothing looked for absence.
#
# _assert_no_source_builds_as_an_island catches the version where a whole source
# ends up alone. It does NOT catch a single dead join on a source that stays
# connected by some other edge: market reached unified companies while all three
# of its option steps produced nothing, and that combination passes every other
# check in this file.
#
# Keyed on the predicate's local name, because the namespace is exactly what
# these defects get wrong and a test that spelled the full URI would have to be
# edited by the same hand that broke the code.
# Each entry maps a relation's LOCAL NAME to the node types that must be present
# for it to be required. An empty tuple means unconditional: those three are the
# temporal spine and the company hub, and a graph of two or more sources that
# lacks them is by definition the defect this guards.
REQUIRED_BRIDGE_RELATIONS = {
    # SEC issuer / market snapshot -> unified company. Dead for both sides at
    # once: the market half asked for a class neither market vocabulary
    # declares, the SEC half for a predicate renamed upstream.
    "refersToCompany": (),
    # Dated entity -> the period it falls in. The on-ramp whose absence left
    # SEC, NOAA and market minting period nodes nothing pointed at.
    "observedInPeriod": (),
    # Source period -> unified period. The other half of the temporal spine.
    "sameAs": (),
    # Option snapshot -> its underlying equity. Keyed on underlyingSymbol,
    # which quote snapshots do not emit and never did — confirmed against
    # full-volume production data, where the raw column is null.
    #
    # Conditional, because it needs BOTH sides in the graph and the N-Triples
    # fixture carries only a slice of the market sample (it is a loader test,
    # not a graph test, so it holds three snapshots and they are not guaranteed
    # to include the equity a kept option is written against). Requiring it
    # everywhere would fail that fixture for a reason that is not a defect.
    "hasUnderlyingEquity": (
        "market_quotes_EquitySnapshot",
        "market_quotes_OptionSnapshot",
    ),
    # Entity -> its economic sector. There is no longer a correlation twin;
    # hasSectorCorrelation was emitted over identical (subject, object) pairs
    # from the same frame, read by nothing, and was removed. See
    # test_no_sector_relation_has_a_duplicate_twin below, which now guards
    # against it coming back.
    "belongsToSector": ("cpi_Index",),
    # Chronological ordering within a source.
    "precedes": ("cpi_Index",),
    # Alert -> the state it affects. Both of its strategies were dead: the
    # area-description matcher looked for full state names in descriptions NWS
    # writes as "Lincoln, KS", and the FIPS matcher compared 2-digit lookup
    # keys against the 3-digit SAME-code head the mapper emits.
    "affectsRegion": ("cap_Alert",),
    # Option -> ATM/ITM/OTM. The links were emitted and then dropped whole,
    # because nothing typed the three category URIs and node_mapper makes nodes
    # only for typed URIs.
    "hasMoneyness": ("market_quotes_OptionSnapshot",),
}

# Inferred links that legitimately do NOT resolve on these fixtures, with the
# reason. Not exemptions for defects — each is a claim that the INPUTS are
# absent, which is a fixture-coverage fact rather than a pipeline one.
#
# Checked for staleness below: if one of these starts resolving, the entry is
# wrong and must move to REQUIRED_BRIDGE_RELATIONS rather than sit here
# silently excusing a relation that now works.
KNOWN_ABSENT_BRIDGES = {
    # Needs a call AND a put at the same strike, expiry and capture time. The
    # market sample holds two calls on one underlying, so the vertical-spread
    # branch fires and the straddle/strangle branches cannot.
    "straddleWith": "no put in the market sample",
    "strangleWith": "no put in the market sample",
    "putSpreadWith": "no put in the market sample",
    # Needs a BLS entity and a market entity that share a sector. Market sector
    # classification reads the GICS ticker CSV from S3, which the e2e run does
    # not configure, so no market entity carries a sector to join on.
    "leadsTo": "market sector classification needs the GICS CSV, unset in e2e",
}


def _assert_bridge_absences_are_still_true(data):
    """A relation listed as legitimately absent must still be absent.

    The allowlist above exists so the required list can stay strict without
    failing on fixture gaps. That only works if the entries stay honest: an
    entry that starts resolving is no longer a fixture fact, and leaving it
    here would exempt a relation that could later die unnoticed.
    """
    present = {str(rel) for _s, rel, _d in data.edge_types}
    resolved = sorted(
        relation for relation in KNOWN_ABSENT_BRIDGES
        if any(n == relation or n.endswith(f"_{relation}") for n in present)
    )
    assert not resolved, (
        f"{resolved} now resolve into edges — move them from "
        f"KNOWN_ABSENT_BRIDGES to REQUIRED_BRIDGE_RELATIONS so they stay "
        f"guarded"
    )


def _assert_declared_bridges_resolve(data):
    """Every inferred link the pipeline claims to make must exist in the graph.

    See REQUIRED_BRIDGE_RELATIONS. This is the guard for the defect CLASS rather
    than for any one instance of it: a join that silently matches nothing leaves
    no trace except a missing relation, and absence is what this looks for.

    Deliberately not derived from the enrichment code — a check generated from
    the same constants the code uses would agree with a typo. The list is
    written by hand and each entry says which join it stands for, so a
    disappeared relation names the step that stopped working.
    """
    node_types = set(map(str, data.node_types))

    # A PyG relation is already namespace-SHORTENED by node_mapper —
    # "bls_enrichment_refersToCompany", not a URI — so the local name is a
    # suffix, not the last path segment. Splitting on "/" is a no-op on these
    # names and made every required relation look absent while all four were
    # present; it cost a full run to notice, which is the same lesson as the
    # defects this guards.
    present = {str(rel) for _s, rel, _d in data.edge_types}

    def resolved(relation: str) -> bool:
        return any(
            name == relation or name.endswith(f"_{relation}") for name in present
        )

    missing = sorted(
        relation
        for relation, requires in REQUIRED_BRIDGE_RELATIONS.items()
        if set(requires) <= node_types and not resolved(relation)
    )
    assert not missing, (
        f"inferred links that resolved into NO edge: {missing} — the join "
        f"behind each either matched nothing or was dropped during edge "
        f"resolution. A join keyed on a term the data no longer emits fails "
        f"exactly this way: silently. Relations present: {sorted(present)}"
    )


# What a working bridge's endpoints look like, so a join resolving into ONE
# edge fails the way a join resolving into none already does.
#
# _assert_declared_bridges_resolve proves a relation is PRESENT. Present is not
# complete, and the gap between them is this issue: a bridge that should link
# every ticker-bearing issuer to a company and links one of them satisfies that
# guard completely. The term is live, the join is correct, and the data behind
# it is shaped so most rows fall out -- which no drift check can see, because
# every term involved is emitted.
#
# Expressed as a RATIO of the population that ought to be covered, never as an
# edge count. A count is a property of the fixture and changes every time it is
# regenerated, so it would have to be re-tuned by hand -- and a threshold people
# re-tune to make the suite pass stops being a guard. "Every issuer that states
# a ticker reaches a company" is a property of the PIPELINE and holds at any
# scale, on three rows or on three million.
#
# Measured on the ENRICHED TRIPLES, not on the graph. That is the layer the join
# runs at, so a failure here means the join under-covered -- distinct from an
# edge lost later during node typing, which _assert_no_node_type_is_orphaned and
# the required-relations guard look for. The two failures want different fixes,
# so they are worth telling apart.
#
# Matched on the predicate's LOCAL NAME for the same reason the required list
# is: the namespace is exactly what this class of defect gets wrong, and a check
# spelling the full URI would have to be edited by the same hand that broke it.
#
# `resolved_against` narrows the candidate population to the subjects whose
# OBJECT value also appears as an object of that predicate. It exists for the
# market half of the company bridge: since that bridge was re-keyed onto the
# CIK, a snapshot reaches a company only if its ticker RESOLVES to one, so the
# population that must be fully covered is "snapshots whose symbol matches an
# ingested issuer's ticker" rather than "snapshots stating a symbol". Without
# it the rule would demand coverage the join cannot supply and would report a
# fixture-overlap fact as a pipeline defect.
_BridgeCoverage = collections.namedtuple(
    "_BridgeCoverage", "relation candidates label floor why resolved_against",
    defaults=(None,),
)

BRIDGE_COVERAGE = (
    _BridgeCoverage(
        relation="refersToCompany",
        candidates="hasIssuerCik",
        label="SEC issuer -> unified company",
        floor=1.0,
        # Keyed on the CIK, not the ticker, and that IS the coverage story.
        # The ticker has one upstream emitter (the ownership-form document
        # parser), so keying on it capped this half of the bridge at the Form
        # 3/4/5 share of filings -- a measured 24.57%. hasIssuerCik is stated
        # by every issuer on every filing, from both the metadata and document
        # paths, so the candidate population here is now every issuer in the
        # graph rather than a quarter of them.
        #
        # 1.0 because there is no filtering step between stating a CIK and
        # reaching the company node it keys.
        why="every issuer states a CIK and none is filtered before the join",
    ),
    _BridgeCoverage(
        relation="refersToCompany",
        candidates="symbol",
        resolved_against="hasIssuerTradingSymbol",
        label="market snapshot -> unified company",
        floor=1.0,
        # The market half reads market-quotes:symbol off equity and option
        # snapshots alike, and equity_symbol() resolves an OCC option symbol to
        # its underlying, so no well-formed symbol is dropped on the way in.
        #
        # What CAN drop a snapshot is the ticker -> CIK step: an inner join,
        # deliberately, because falling back to a symbol-keyed company node is
        # exactly the split this work removed. So the rule measures the
        # snapshots whose symbol an ingested filing also states -- for those,
        # the resolution exists and full coverage is required. A snapshot for a
        # company with no filings here is not a defect, and the e2e run
        # configures no constituents CSV to widen it.
        why="a symbol an ingested issuer also states must resolve to its CIK",
    ),
)


def _assert_declared_bridges_are_not_degenerate(config):
    """Every declared bridge must cover its endpoints, not merely exist.

    See BRIDGE_COVERAGE.

    A rule whose candidate population is empty here FAILS. It used to be
    skipped, on the reasoning that the fixtures deliberately differ and an
    absent input is a fixture-coverage fact rather than a pipeline defect. That
    reasoning is what let the market half of the company bridge go dark: the
    SEC fixture was regenerated, its issuers stopped quoting in the market
    fixture, the candidate population went to zero, and a rule with a floor of
    1.0 went quiet in a green suite. A ratio of 0/0 is not coverage.

    A fixture that genuinely should not carry a bridge is declared, not
    inferred from an empty count -- that is what KNOWN_ABSENT_BRIDGES and the
    conditional entries in REQUIRED_BRIDGE_RELATIONS are for.

    tests/test_bridge_coverage_guard.py still asserts that every rule has a
    candidate population somewhere in the committed fixtures. That check
    belongs in the fast suite anyway -- it is a statement about the fixtures,
    needs no pipeline run, and so runs constantly rather than only when someone
    runs the heavy suite by hand.
    """
    import pandas as pd

    files = sorted(Path(config.enriched_parquet_path).rglob("*.parquet"))
    if not files:
        return  # split-mode runs may not re-emit the enriched frame
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def rows_of(local_name):
        suffix = f"/{local_name}"
        return df[df["predicate"].str.endswith(suffix)]

    def subjects_of(local_name):
        return set(rows_of(local_name)["subject"])

    def resolvable_subjects(local_name, against):
        """Subjects whose stated value also appears as a value of `against`.

        The two sides are compared case-insensitively and trimmed, matching
        equity_symbol()'s normalisation -- a fixture stating "aapl" against
        "AAPL" is the same ticker, and treating it as a miss would report a
        formatting difference as missing coverage.
        """
        available = {
            str(v).strip().upper() for v in rows_of(against)["object"]
        }
        stated = rows_of(local_name)
        matched = stated["object"].astype(str).str.strip().str.upper()
        return set(stated[matched.isin(available)]["subject"])

    thin = []
    unverifiable = []
    for rule in BRIDGE_COVERAGE:
        if rule.resolved_against:
            candidates = resolvable_subjects(rule.candidates, rule.resolved_against)
        else:
            candidates = subjects_of(rule.candidates)
        if not candidates:
            # A rule with nothing to measure is NOT a rule that passed, and
            # until this was collected the two were indistinguishable. The
            # market half of the company bridge sat here for weeks: the SEC
            # fixture was regenerated, its issuers stopped quoting in the
            # market fixture, the candidate population went to zero and the
            # rule went quiet -- on a floor of 1.0, in a green suite.
            #
            # Collected rather than raised on the spot: a fixture legitimately
            # need not carry every bridge, so the report below is what decides,
            # and it names which rule went dark rather than only that one did.
            unverifiable.append(
                f"{rule.label}: no subject states {rule.candidates}"
                + (f" resolvable against {rule.resolved_against}"
                   if rule.resolved_against else "")
            )
            continue
        covered = candidates & subjects_of(rule.relation)
        ratio = len(covered) / len(candidates)
        if ratio < rule.floor:
            thin.append(
                f"{rule.label}: {len(covered)}/{len(candidates)} "
                f"({ratio:.1%}) of subjects stating {rule.candidates} reach "
                f"{rule.relation}, floor {rule.floor:.0%} — {rule.why}"
            )

    assert not unverifiable, (
        "bridge coverage rules that measured NOTHING:\n  "
        + "\n  ".join(unverifiable)
        + "\n\nThese did not pass -- they had no candidate population, so the "
        "rule was skipped and the suite stayed green over an unexercised join. "
        "The usual cause is fixture drift: one committed fixture regenerated "
        "without the other, leaving nothing for the join to match. Regenerate "
        "the pair together, or delete the rule if the bridge is genuinely gone."
    )

    assert not thin, (
        "bridges that resolved but under-cover their endpoints:\n  "
        + "\n  ".join(thin)
        + "\n\nThe relation exists, so _assert_declared_bridges_resolve passes. "
        "The join behind it is matching a fraction of what it should — check "
        "whether the data on one side is shaped as the join assumes (padding, "
        "datatype, which subject states the term) rather than whether the term "
        "is still emitted."
    )


# Coverage asks whether a link FORMED. It cannot ask whether the link points
# where it should: a market snapshot that reaches some company node satisfies a
# floor of 1.0 whether or not that node is the company whose filing states the
# same ticker. Both defects that would produce a wrong node are live risks --
# the symbol-keyed fallback company this work replaced, and a ticker resolving
# to a CIK other than the filing's.
#
# The two sides are checked against EACH OTHER rather than against a name,
# because agreement is the strongest claim fixture data supports: nothing here
# can confirm that a ticker is the company the outside world says it is. A
# disagreement, though, is always a defect -- one ticker cannot be two companies
# in one graph.
#
# Deliberately silent about absence. A ticker that reaches a company from only
# one side is left to _assert_declared_bridges_are_not_degenerate's floor, so a
# failure here always means "wrong", never "missing", and the two cannot be
# confused when one fires.

def _assert_bridged_companies_agree(config):
    """A ticker must reach the SAME company from the market and SEC sides."""
    import collections

    import pandas as pd

    files = sorted(Path(config.enriched_parquet_path).rglob("*.parquet"))
    if not files:
        return  # split-mode runs may not re-emit the enriched frame
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def rows_of(local_name):
        return df[df["predicate"].str.endswith(f"/{local_name}")]

    refers = rows_of("refersToCompany")
    company = collections.defaultdict(set)
    for subject, obj in zip(refers["subject"], refers["object"]):
        company[subject].add(obj)

    def reached(local_name, fold_options):
        """Companies each ticker reaches, keyed by ticker."""
        out = collections.defaultdict(set)
        rows = rows_of(local_name)
        for subject, obj in zip(rows["subject"], rows["object"]):
            ticker = str(obj).strip().upper()
            if fold_options:
                # An OCC option symbol carries its underlying in the first six
                # characters, so a chain is checked against its own equity's
                # company rather than being skipped for not looking like a
                # ticker.
                ticker = ticker[:6].strip()
            if subject in company:
                out[ticker] |= company[subject]
        return out

    market = reached("symbol", fold_options=True)
    sec = reached("hasIssuerTradingSymbol", fold_options=False)

    mismatched = [
        f"{ticker}: market reaches "
        f"{sorted(_local_name(c) for c in market[ticker])}, "
        f"SEC reaches {sorted(_local_name(c) for c in sec[ticker])}"
        for ticker in sorted(set(market) & set(sec))
        if market[ticker] != sec[ticker]
    ]

    assert not mismatched, (
        "the same ticker reaches DIFFERENT companies from each side:\n  "
        + "\n  ".join(mismatched)
        + "\n\nThe bridge formed, so coverage passes -- it points at the wrong "
        "company. Check whether the market side fell back to a symbol-keyed "
        "company node instead of the CIK-keyed one, or whether the ticker "
        "resolved to a CIK other than the filing's."
    )


def _local_name(term) -> str:
    return str(term).rsplit("/", 1)[-1].rsplit("#", 1)[-1]


# The transaction chain, which the ratio rules above cannot express.
#
# BRIDGE_COVERAGE asks what fraction of a candidate population reaches a
# relation, and a chain does not have that shape: the LAST transaction in a
# group correctly reaches nothing, so the honest floor is (N - groups) / N,
# which is a property of how the fixture's filings happen to be distributed
# rather than of the pipeline. What IS a property of the pipeline is the
# chain itself -- N dated transactions in one group are N-1 edges laid end to
# end, no more and no fewer.
#
# That distinction is the whole point. "precedes edges exist between
# transactions" is satisfied by one edge among forty transactions, and a
# window that resolved a date for one member of each group would produce
# exactly that while looking healthy from every other angle.
#
# The grouping is re-derived HERE from the filing structure -- transaction ->
# filing -> reporting owner, split by class -- rather than read from the
# linker. It therefore encodes the same choice the linker's docstring states,
# and changing that choice means changing this too. That coupling is
# unavoidable for an exact count, and an exact count is the only thing that
# separates a chain from a scattering of edges.
_TRANSACTION_CLASSES = ("NonDerivativeTransaction", "DerivativeTransaction")
_TRANSACTION_POINTERS = ("hasNonDerivativeTransaction", "hasDerivativeTransaction")


def _assert_transaction_chains_are_complete(config):
    """Every group of N dated transactions carries exactly N-1 precedes edges.

    Skipped when the fixture holds no group of two, for the same reason the
    coverage rules skip a rule with no candidates: a filing reporting a single
    transaction has nothing to sequence, and failing on that would report a
    fixture-coverage fact as a pipeline defect. tests/test_bridge_coverage_guard
    .py is what keeps the skip honest -- it asserts the committed fixtures DO
    carry such a group, and generate_sec_e2e_fixtures.py selects one on purpose.

    Measured on the enriched triples rather than on the graph, like the bridge
    coverage above: this is a statement about the join, and an edge lost later
    in node typing is a different failure with a different fix.
    """
    import pandas as pd

    files = sorted(Path(config.enriched_parquet_path).rglob("*.parquet"))
    if not files:
        return  # split-mode runs may not re-emit the enriched frame
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    local = df["predicate"].str.rsplit("/", n=1).str[-1].str.rsplit("#", n=1).str[-1]
    object_local = df["object"].str.rsplit("/", n=1).str[-1]

    typed = df[(df["predicate"] == _RDF_TYPE) & object_local.isin(_TRANSACTION_CLASSES)]
    transaction_class = dict(zip(typed["subject"], object_local[typed.index]))
    if not transaction_class:
        return  # no ownership transactions in this fixture

    dated = set(df.loc[local == "hasTransactionDate", "subject"])

    pointers = df[local.isin(_TRANSACTION_POINTERS)]
    filing_of = dict(zip(pointers["object"], pointers["subject"]))

    reported_by = df[local == "hasReportingOwner"]
    owners_of = collections.defaultdict(list)
    for filing, owner in zip(reported_by["subject"], reported_by["object"]):
        owners_of[filing].append(owner)

    groups = collections.defaultdict(set)
    for transaction, klass in transaction_class.items():
        if transaction not in dated:
            continue
        for owner in owners_of.get(filing_of.get(transaction), ()):
            groups[(owner, klass)].add(transaction)

    chains = {key: members for key, members in groups.items() if len(members) > 1}
    if not chains:
        return  # nothing to sequence here; see the docstring

    sequenced = df[local == "precedes"]
    between_transactions = {
        (s, o) for s, o in zip(sequenced["subject"], sequenced["object"])
        if s in transaction_class and o in transaction_class
    }

    broken = []
    for (owner, klass), members in sorted(chains.items()):
        inside = {(s, o) for s, o in between_transactions
                  if s in members and o in members}
        successors = collections.Counter(s for s, _ in inside)
        predecessors = collections.Counter(o for _, o in inside)
        forks = sorted(s for s, n in successors.items() if n > 1)
        joins = sorted(o for o, n in predecessors.items() if n > 1)
        if len(inside) != len(members) - 1 or forks or joins:
            broken.append(
                f"{owner.rsplit('/', 1)[-1]} / {klass}: {len(members)} dated "
                f"transactions linked by {len(inside)} edge(s), expected "
                f"{len(members) - 1}"
                + (f"; branches at {forks}" if forks else "")
                + (f"; merges at {joins}" if joins else "")
            )

    assert not broken, (
        "transaction chains that are not chains:\n  " + "\n  ".join(broken)
        + "\n\nA group of N dated transactions is a path of N-1 precedes edges. "
        "Too few and the window resolved a date for only some members — check "
        "the date join, not whether the relation exists. Branching or merging "
        "means the ordering is not a total one, which on this data means the "
        "tie-break stopped applying: transactions sharing a transaction date "
        "are the common case, not the exception."
    )

    stray = sorted(
        (s, o) for s, o in between_transactions
        if not any(s in members and o in members for members in groups.values())
    )
    assert not stray, (
        f"precedes edges between transactions in different groups: {stray} — "
        "the window is partitioning on something wider than (reporting owner, "
        "transaction class), so unrelated filers' transactions are being "
        "ordered against each other"
    )


# Identifier locals that upstream keys on a CIK, and the shape of the URI it
# mints from one. Written out here rather than imported from
# utils/canonicalization.py deliberately: a check built from the same constants
# as the code under test agrees with that code's mistakes.
_CIK_KEYED_URI = re.compile(
    r"^(.*/id/sec-filings/(?:Issuer|ReportingOwner|Owner)_)(\d+)(?:\.0+)?$"
)


def _assert_no_entity_is_split_by_identifier_padding(index):
    """One filer must be one node, however upstream spelled its CIK.

    Upstream states a CIK two ways -- "0001729997" from its metadata path and
    an unpadded, decimal-typed 1729997 from its document path -- and a single
    Form 144 can carry both, pointing hasIssuer at each. Those are different
    URIs, so they are different nodes for one company, and the graph partitions
    that company's filings between them without anything erroring.

    Asserted over the node_index rather than over the triples because this is
    about node identity: the node_index is the artifact that says which entity
    each row of the tensor IS, so two rows for one filer is precisely what it
    would show.

    Keyed on the digits with leading zeros stripped, so the padded and unpadded
    spellings of one CIK collide here even though they do not collide as URIs.
    """
    groups = collections.defaultdict(set)
    for uri in map(str, index["uri"]):
        match = _CIK_KEYED_URI.match(uri)
        if match:
            prefix, digits = match.group(1), match.group(2)
            groups[(prefix, digits.lstrip("0") or "0")].add(uri)

    split = {key: sorted(uris) for key, uris in groups.items() if len(uris) > 1}
    assert not split, (
        "one entity appears as several nodes differing only in CIK padding: "
        f"{split} — canonicalization did not merge them, so this filer's edges "
        "and message passing are partitioned across the copies"
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

    "Different sources" means different SOURCE FAMILIES, not different
    namespaces. This test used to split node types on their first underscore,
    so cpi and eci counted as two sources and the assertion was satisfied by
    two BLS feeds meeting — while SEC, market and NOAA were all off the spine
    entirely. It passed throughout the period when three of the four sources
    reached no other source at all, which is a weaker claim than its name
    makes. Grouping by family is the same rule _assert_no_source_builds_as_an_
    island uses, so the two cannot disagree about what a source is.
    """
    # (node_type, node_id) -> set of neighbouring (node_type, node_id)
    neighbors = {}
    for et in data.edge_types:
        src_t, _rel, dst_t = (str(x) for x in et)
        edge_index = data[et].edge_index
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            neighbors.setdefault((src_t, s), set()).add((dst_t, d))
            neighbors.setdefault((dst_t, d), set()).add((src_t, s))

    by_prefix, hubs = _source_families()

    unified_nodes = [
        (nt, i)
        for nt in map(str, data.node_types)
        if "Unified" in nt
        for i in range(data[nt].num_nodes)
    ]
    assert unified_nodes, "no unified temporal nodes in the graph"

    best = set()
    for unified in unified_nodes:
        # unified period -> the source periods it is sameAs -> what they date
        reached = {
            _family_of(measured[0], by_prefix)
            for source_period in neighbors.get(unified, ())
            for measured in neighbors.get(source_period, ())
            if measured != unified
        } - hubs - {None}
        if len(reached) >= 2:
            return
        best |= reached

    raise AssertionError(
        "no unified temporal entity connects measurements from two different "
        f"sources — the cross-source temporal bridge does not form. Sources "
        f"reaching the spine at all: {sorted(best) or 'none'}"
    )


def _assert_sameas_links_only_unified_vocabularies(config):
    """Every owl:sameAs must point at a PERIOD or a REGION, never a measurement.

    THE DEFECT THIS GUARDS. A deleted cross-source matcher was anchored only at
    the end of the subject, so it matched any URI merely ENDING with a month
    name and asserted `unified:September owl:sameAs cpi:...PercentChange_...
    _September` -- that a measurement IS the month. owl:sameAs is RDF's
    strongest claim, so those became real edges and licensed a reasoner to
    merge the two. What makes that shape detectable is the OBJECT: a
    measurement URI carries its period inside a longer name, and a period URI
    is the period and nothing else.

    WHY REGIONS ARE HERE TOO. The unifier now has a second axis. Aligning
    `unified:MidwestCensusRegion` with `jolts:Midwest_Region` is the same
    operation as aligning `unified:February` with `cpi:February` -- one
    real-world thing named by two source vocabularies -- and
    classify_edge_origin already reports both as `unification`. So the rule is
    not "sameAs is temporal", which was only ever true because the temporal
    unifier was its sole producer; it is "sameAs relates two names for one
    entity in a vocabulary this pipeline unifies".

    The admitted shapes stay ENUMERATED rather than loosened to something like
    "contains no measurement keyword". A laxer rule would readmit the original
    defect, which is the one thing this must not do.
    """
    import re

    import pandas as pd

    from spark_jobs.enrichment.region_crosswalk import (
        CENSUS_REGIONS, US_STATES, state_key,
    )

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

    def temporal(local):
        return bool(
            re.fullmatch(months, local)
            or re.fullmatch(r"\d{4}", local)
            or re.fullmatch(r"Year\d{4}", local)
            or re.fullmatch(r"Q[1-4]", local)
            # The day grain: id/temporal/{source}/2026-07-03 on the source
            # side, unified/Day2026-07-03 on the unified side.
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}", local)
            or re.fullmatch(r"Day\d{4}-\d{2}-\d{2}", local)
        )

    # The unified region nodes, spelled from the crosswalk rather than by
    # pattern, so a region node this pipeline does not actually mint cannot
    # sneak through as "looks region-shaped".
    unified_regions = {f"{state_key(s)}Region" for s in US_STATES} | {
        f"{state_key(c)}CensusRegion" for c in CENSUS_REGIONS
    }

    def region(local):
        # Source-side region entities are "{Name}_Region" -- the shape jolts
        # mints. Digits are excluded because that is exactly what separates a
        # region entity from a measurement ABOUT a region:
        # "Midwest_Region" is the place, "Midwest_June2026_HiresLevel" is a
        # number, and only the first may ever be a sameAs object.
        return local in unified_regions or bool(
            re.fullmatch(r"[A-Za-z]+(?:_[A-Za-z]+)*_Region", local)
        )

    def unifiable(uri):
        local = str(uri).rsplit("/", 1)[-1]
        return temporal(local) or region(local)

    bad = same_as[~same_as["object"].map(unifiable)]
    assert bad.empty, (
        f"{len(bad)} owl:sameAs triples point at an entity that is neither a "
        f"period nor a region, e.g. {bad['object'].iloc[0]}"
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
