"""Pin what `bin/publish_run.py` sends to the published-runs prefix, and in what order.

WHY THIS IS WORTH A TEST
------------------------
The destination takes writes but not deletes, so a file sent to the wrong key stays
there. The published layout is four renames away from the work directory, and the
only earlier publish was a hand-edited script whose own check printed 233 files
against 236 expected and still reported success.

HOW
---
The script runs for real against a stub aws CLI. The stub keeps each bucket as a
folder under the test's tmp directory and records every call, so nothing here
touches the network.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = REPO_ROOT / "bin" / "publish_run.py"

RUN_ID = "20260101T120000Z"
YEAR, MONTH, DAY = "2026", "01", "2026-01-02"
GRAPHS = ("no_edge_features", "1024d")
LABELS = {"no_edge_features": "no_edge_features", "1024d": "baseline_1024d"}
METADATA = ("encoding_config.json", "feature_spec.json", "graph_schema.json",
            "normalization.json", "ontology_schema.json", "slot_mapping.json")

pytestmark = pytest.mark.launcher

STUB_AWS = '''#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path

s3 = Path(os.environ["FAKE_S3"])
args = sys.argv[1:]
with open(os.environ["FAKE_S3_CALLS"], "a") as fh:
    fh.write(" ".join(args) + "\\n")

if args[:2] == ["s3api", "list-objects-v2"]:
    bucket = s3 / args[args.index("--bucket") + 1]
    prefix = args[args.index("--prefix") + 1]
    deny = os.environ.get("FAKE_S3_DENY_LIST", "")
    if deny and prefix.startswith(deny):
        sys.exit("An error occurred (AccessDenied) when calling the ListObjectsV2 operation")
    keys = [p for p in sorted(bucket.rglob("*")) if p.is_file()] if bucket.exists() else []
    contents = [{"Key": p.relative_to(bucket).as_posix(), "Size": p.stat().st_size,
                 "ChecksumAlgorithm": ["CRC64NVME"]}
                for p in keys if p.relative_to(bucket).as_posix().startswith(prefix)]
    print(json.dumps({"Contents": contents} if contents else {"Prefix": prefix}))
elif args[:2] == ["s3", "sync"]:
    source, dest = Path(args[2]), args[3].rstrip("/")
    skip = {args[i + 1] for i, a in enumerate(args) if a == "--exclude"}
    skip.add(os.environ.get("FAKE_S3_DROP", ""))
    for p in sorted(source.rglob("*")):
        name = p.relative_to(source).as_posix()
        if p.is_dir() or name in skip:
            continue
        target = s3 / f"{dest}/{name}"[len("s3://"):]
        if target.exists() and target.stat().st_size == p.stat().st_size:
            continue
        if "--dryrun" in args:
            print(f"(dryrun) upload: {p} to {dest}/{name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        print(f"upload: {p} to {dest}/{name}")
elif args[:2] == ["s3", "cp"]:
    target = s3 / args[3][len("s3://"):]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args[2], target)
    print(f"upload: {args[2]} to {args[3]}")
else:
    sys.exit(f"stub aws: no case for {args}")
'''


def _write(path: Path, body) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, str):
        path.write_text(body)
    else:
        path.write_bytes(body)
    return path


def files(folder: Path) -> set:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def expected_names() -> set:
    names = {"index.json", "enriched/dataset.json", "enriched/triples/_SUCCESS",
             "manifests/enrichment_only_1.json", "manifests/pyg_only_1.json"}
    names |= {f"enriched/triples/part-0000{i}.snappy.parquet" for i in range(3)}
    for graph in GRAPHS:
        names |= {f"{graph}/hetero_data_{graph}.pt", f"{graph}/checksums.json",
                  f"{graph}/node_index/part-00000.snappy.parquet",
                  f"{graph}/node_index/_SUCCESS"}
        names |= {f"{graph}/{name}" for name in METADATA}
    return names


class Run:
    """A finished run: its run directory, its work directory, and an empty fake S3."""

    def __init__(self, tmp_path: Path):
        self.rd = tmp_path / "run"
        self.work = tmp_path / "work" / RUN_ID
        self.pyg = self.work / "pyg" / f"year={YEAR}" / f"month={MONTH}"
        self.enriched = self.work / "enriched" / f"year={YEAR}" / f"month={MONTH}"
        self.manifests = self.work / "manifests" / f"year={YEAR}" / f"month={MONTH}"
        self.s3 = tmp_path / "s3"
        self.s3.mkdir()
        self.published = (self.s3 / "bucket" / "runs" / "all-sources"
                          / f"year={YEAR}" / f"month={MONTH}" / RUN_ID)
        self.published_tables = self.s3 / "bucket" / "tables" / "all-sources"
        self.calls = tmp_path / "aws-calls.txt"
        self.bin = tmp_path / "stubbin"
        _write(self.bin / "aws", STUB_AWS).chmod(0o755)

        # The run directory, as the launcher and the recorder leave it.
        _write(self.rd / "run.log", f"[12:00:00] RUN_ID={RUN_ID}\n")
        _write(self.rd / "run-config.txt",
               f"run id      : {RUN_ID}\nwork dir    : {self.work}\n")
        _write(self.rd / "run.done", "0\n")
        _write(self.rd / "outcome.txt", "harness  : runner rc=0\nstate    : all 3 submissions ok\n")
        cells = ["=== seed (enrichment_only, 2026-01)"]
        for graph in GRAPHS:
            cells += [f"=== {LABELS[graph]}",
                      f"    [INFO] build_graph -   PyG output:       "
                      f"{self.pyg}/hetero_data_{graph}.pt"]
        _write(self.rd / "notebook-cells.txt", "\n".join(cells) + "\n")

        # The work directory, as the job leaves it.
        for graph in GRAPHS:
            self._graph(graph)
        for i in range(3):
            _write(self.enriched / "triples" / f"part-0000{i}.snappy.parquet", b"t" * (10 + i))
            _write(self.enriched / "triples" / f".part-0000{i}.snappy.parquet.crc", b"c")
        _write(self.enriched / "triples" / "_SUCCESS", b"")
        _write(self.enriched / "triples" / "._SUCCESS.crc", b"c")
        self.set_recorded_dataset("")
        self.source_paths = [
            "s3a://BUCKET/raw/source=a/year=2026/month=01/02.snappy.parquet",
            "s3a://BUCKET/raw/source=b/2026.snappy.parquet",
            "s3a://BUCKET/quotes/year=2026/month=01/day=02/",
        ]
        self._write_manifests()
        # Written by the job, never published.
        _write(self.work / "pyg" / "latest" / "hetero_data_1024d_metadata" / "graph_schema.json",
               "{}")
        _write(self.work / "checkpoints" / "rdd-1" / "part-00000", b"x")

    def _graph(self, graph: str) -> None:
        meta = self.pyg / f"hetero_data_{graph}_metadata"
        written = {f"hetero_data_{graph}.pt":
                   _write(self.pyg / f"hetero_data_{graph}.pt", f"{graph} tensors ".encode() * 50)}
        for name in METADATA:
            written[f"{meta.name}/{name}"] = _write(meta / name,
                                                    json.dumps({"file": name, "graph": graph}))
        _write(meta / "checksums.json", json.dumps({"algorithm": "sha256", "artifacts": {
            key: {"bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in written.items()}}))
        index = self.pyg / f"hetero_data_{graph}_node_index"
        _write(index / "part-00000.snappy.parquet", b"n" * 7)
        _write(index / ".part-00000.snappy.parquet.crc", b"c")
        _write(index / "_SUCCESS", b"")
        _write(index / "._SUCCESS.crc", b"c")

    def add_tables(self, day: str = DAY) -> set:
        """Write one day of query tables into the work dir, as the job leaves them.

        graph/ is a triple store directory rather than Parquet, so it is a tree
        of ordinary files under the same day partition.
        """
        names = set()
        for table in ("nodes", "edges", "facts"):
            partition = self.work / "tables" / table / f"day={day}"
            _write(partition / "part-00000.zstd.parquet", f"{table} rows".encode())
            _write(partition / ".part-00000.zstd.parquet.crc", b"c")
            _write(partition / "_SUCCESS", b"")
            names |= {f"{table}/day={day}/part-00000.zstd.parquet",
                      f"{table}/day={day}/_SUCCESS"}
        store = self.work / "tables" / "graph" / f"day={day}"
        _write(store / "CURRENT", b"MANIFEST-000001\n")
        _write(store / "MANIFEST-000001", b"rocksdb")
        names |= {f"graph/day={day}/CURRENT", f"graph/day={day}/MANIFEST-000001"}
        return names

    def set_recorded_dataset(self, name: str) -> None:
        _write(self.enriched / "dataset.json", json.dumps(
            {"dataset": name, "sources": ["a", "b"], "time_period": f"{YEAR}-{MONTH}"}))

    def add_source_path(self, path: str) -> None:
        self.source_paths.append(path)
        self._write_manifests()

    def _write_manifests(self) -> None:
        _write(self.manifests / "enrichment_only_1.json",
               json.dumps({"config": {"source_paths": self.source_paths}}))
        _write(self.manifests / "pyg_only_1.json", json.dumps({"config": {"source_paths": []}}))

    def publish(self, *args: str, **env_changes):
        env = dict(os.environ)
        env.update({
            "PATH": f"{self.bin}{os.pathsep}{env['PATH']}",
            "FAKE_S3": str(self.s3),
            "FAKE_S3_CALLS": str(self.calls),
            "PYG_PUBLISH_ROOT": "s3://bucket/runs",
            "PYG_TABLES_ROOT": "s3://bucket/tables",
            "PYG_PUBLISH_DATASET": "all-sources",
        })
        env.pop("PYG_PUBLISH_DATA_DAY", None)
        for key, value in env_changes.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run([sys.executable, str(PUBLISHER), str(self.rd), *args],
                              env=env, capture_output=True, text=True, timeout=120)

    def calls_text(self) -> str:
        return self.calls.read_text() if self.calls.exists() else ""


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


# --------------------------------------------------------------------------- #
# The layout
# --------------------------------------------------------------------------- #

def test_a_dry_run_lays_out_the_published_names_and_writes_nothing(run):
    """The four renames, and nothing from pyg/latest/, checkpoints/ or the .crc files."""
    r = run.publish()
    assert r.returncode == 0, r.stdout + r.stderr
    assert files(run.rd / "publish") == expected_names()
    assert files(run.s3) == set()
    assert "--dryrun" in run.calls_text()


def test_an_upload_publishes_every_file_and_index_json_last(run):
    r = run.publish("--upload")
    assert r.returncode == 0, r.stdout + r.stderr
    assert files(run.published) == expected_names()
    assert ((run.published / "1024d" / "hetero_data_1024d.pt").read_bytes()
            == (run.pyg / "hetero_data_1024d.pt").read_bytes())

    calls = run.calls_text().splitlines()
    sync = next(i for i, call in enumerate(calls) if call.startswith("s3 sync"))
    index = next(i for i, call in enumerate(calls) if call.startswith("s3 cp"))
    assert sync < index
    assert calls[index].split()[2].endswith("index.json")
    # The listing was checked between the upload and index.json.
    assert any(call.startswith("s3api list-objects-v2") for call in calls[sync + 1:index])
    assert (run.rd / "publish.done").read_text().strip() == "0"


def test_index_json_names_the_day_the_dataset_and_the_notebook_labels(run):
    assert run.publish("--upload").returncode == 0
    index = json.loads((run.published / "index.json").read_text())
    assert index["run_id"] == RUN_ID
    assert index["dataset"] == "all-sources"
    assert index["time_period"] == f"{YEAR}-{MONTH}"
    assert index["data_day"] == DAY
    assert index["sources"] == ["a", "b"]
    assert {g: v["notebook_label"] for g, v in index["variants"].items()} == LABELS
    assert "checksums.json" in index["per_variant_files"]


# --------------------------------------------------------------------------- #
# Refusals: nothing is written
# --------------------------------------------------------------------------- #

def test_a_run_that_did_not_succeed_is_refused(run):
    (run.rd / "run.done").write_text("1\n")
    r = run.publish("--upload")
    assert r.returncode == 2
    assert "NOT PUBLISHED" in r.stdout
    assert "s3 sync" not in run.calls_text()


def test_a_file_that_no_longer_matches_checksums_json_is_refused(run):
    """Same size, different bytes, so only the sha256 can catch it."""
    path = run.pyg / "hetero_data_1024d_metadata" / "graph_schema.json"
    body = path.read_bytes()
    path.write_bytes(body[:-1] + b"X")
    r = run.publish("--upload")
    assert r.returncode == 2
    assert "sha256" in r.stdout
    assert "s3 sync" not in run.calls_text()


def test_a_run_already_published_is_refused(run):
    _write(run.published / "index.json", "{}")
    r = run.publish("--upload")
    assert r.returncode == 2
    assert "already published" in r.stdout
    assert "s3 sync" not in run.calls_text()


def test_source_paths_naming_two_days_need_the_day_given(run):
    run.add_source_path("s3a://BUCKET/raw/source=c/year=2026/month=01/03.snappy.parquet")
    refused = run.publish()
    assert refused.returncode == 2
    assert "PYG_PUBLISH_DATA_DAY" in refused.stdout
    assert run.publish(PYG_PUBLISH_DATA_DAY=DAY).returncode == 0


def test_with_no_dataset_name_anywhere_the_run_is_refused(run):
    r = run.publish(PYG_PUBLISH_DATASET=None)
    assert r.returncode == 2
    assert "PYG_PUBLISH_DATASET" in r.stdout


def test_a_dataset_name_that_contradicts_the_job_is_refused(run):
    run.set_recorded_dataset("no-market")
    assert run.publish().returncode == 2


def test_the_dataset_comes_from_the_job_when_it_named_one(run):
    run.set_recorded_dataset("no-market")
    r = run.publish("--upload", PYG_PUBLISH_DATASET=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (run.s3 / "bucket" / "runs" / "no-market" / f"year={YEAR}" / f"month={MONTH}"
            / RUN_ID / "index.json").exists()


# --------------------------------------------------------------------------- #
# A failed upload
# --------------------------------------------------------------------------- #

def test_a_file_lost_in_the_upload_fails_the_check_and_leaves_index_json_unwritten(run):
    r = run.publish("--upload", FAKE_S3_DROP="1024d/normalization.json")
    assert r.returncode == 1
    assert "missing: 1024d/normalization.json" in r.stdout
    assert not (run.published / "index.json").exists()
    assert (run.rd / "publish.done").read_text().strip() == "1"


# --------------------------------------------------------------------------- #
# The query tables: a second destination, keyed by day
# --------------------------------------------------------------------------- #

def test_tables_are_published_outside_the_run_folder(run):
    """Runs expire at 21 days and the tables keep a year. S3 applies the
    SHORTEST expiration where two prefix rules overlap, so a table under the
    run's prefix would be deleted with the run whatever the second rule said."""
    expected = run.add_tables()
    assert run.publish("--upload").returncode == 0

    assert files(run.published_tables) == expected | {f"_days/{DAY}.json"}
    assert not [name for name in files(run.published) if name.startswith("nodes/")]
    assert not [name for name in files(run.published_tables) if name.endswith(".crc")]


def test_the_day_marker_goes_up_last(run):
    """Same contract as index.json: a day without its marker is a publish that
    did not finish, and running again resumes it."""
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    calls = run.calls_text().splitlines()
    copies = [i for i, call in enumerate(calls) if call.startswith("s3 cp")]
    marker = next(i for i in copies if f"_days/{DAY}.json" in calls[i])
    table_sync = next(i for i, call in enumerate(calls)
                      if call.startswith("s3 sync") and "tables" in call)
    assert table_sync < marker


def test_index_json_names_the_tables_and_where_they_went(run):
    """A consumer holding the run index has no other way to learn they exist,
    and the tables are the part still there a year later."""
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    tables = json.loads((run.published / "index.json").read_text())["tables"]
    assert tables["root"] == "s3://bucket/tables/all-sources/"
    assert tables["day"] == f"day={DAY}/"
    assert tables["published"] == ["edges", "facts", "graph", "nodes"]
    assert "day-scoped" in tables["note"]


def test_a_day_already_published_is_refused_but_the_run_still_goes(run):
    """The destination has no delete, so a rewritten day whose part count went
    down would leave the old parts behind and a reader would see their rows
    twice. The run itself is published either way and says so."""
    run.add_tables()
    _write(run.published_tables / "_days" / f"{DAY}.json", "{}")

    r = run.publish("--upload")
    assert r.returncode != 0
    assert "already published" in r.stdout
    assert (run.published / "index.json").exists()


def test_a_second_day_is_not_refused_for_the_first_one_being_there(run):
    """The tables are keyed by day, not by run: days accumulate under one root
    and an earlier one must not block a later one."""
    run.add_tables()
    _write(run.published_tables / "_days" / "2026-01-01.json", "{}")
    _write(run.published_tables / "nodes" / "day=2026-01-01" / "part-00000.zstd.parquet",
           b"older")

    r = run.publish("--upload")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (run.published_tables / "_days" / f"{DAY}.json").exists()
    # The earlier day is still there, and was not counted as unexpected.
    assert (run.published_tables / "nodes" / "day=2026-01-01"
            / "part-00000.zstd.parquet").exists()


def test_tables_with_nowhere_to_go_are_refused_before_anything_is_written(run):
    """Said at the prompt rather than after the .pt is up."""
    run.add_tables()
    r = run.publish("--upload", PYG_TABLES_ROOT=None)
    assert r.returncode == 2
    assert "PYG_TABLES_ROOT" in r.stdout
    assert "s3 sync" not in run.calls_text()


def test_a_run_with_no_tables_needs_no_tables_root(run):
    """Every run before this feature existed is that run."""
    r = run.publish("--upload", PYG_TABLES_ROOT=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((run.published / "index.json").read_text())["tables"]["published"] == []


# --------------------------------------------------------------------------- #
# Running again
#
# A run folder with index.json is finished and is never sent again. What can be
# left is its tables, when they failed after the run went up -- and a rerun used
# to stop at that index.json, so the script could not finish them.
# --------------------------------------------------------------------------- #

def _calls_after(run: Run, mark: int) -> list:
    return run.calls_text().splitlines()[mark:]


def test_running_again_after_the_tables_failed_sends_only_the_tables(run):
    expected = run.add_tables()
    lost = f"nodes/day={DAY}/part-00000.zstd.parquet"
    first = run.publish("--upload", FAKE_S3_DROP=lost)
    assert first.returncode == 1
    assert "the run is published at" in first.stdout
    assert not (run.published_tables / "_days" / f"{DAY}.json").exists()
    index_before = (run.published / "index.json").read_bytes()
    mark = len(run.calls_text().splitlines())

    dry = run.publish()
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "Resuming its tables" in dry.stdout
    assert f"plus _days/{DAY}.json" in dry.stdout

    again = run.publish("--upload")
    assert again.returncode == 0, again.stdout + again.stderr
    assert files(run.published_tables) == expected | {f"_days/{DAY}.json"}
    assert (run.published / "index.json").read_bytes() == index_before
    assert (run.rd / "publish.done").read_text().strip() == "0"
    assert not [call for call in _calls_after(run, mark)
                if call.startswith(("s3 sync", "s3 cp")) and "s3://bucket/runs/" in call]


def test_running_again_after_everything_went_up_sends_nothing(run):
    run.add_tables()
    assert run.publish("--upload").returncode == 0
    mark = len(run.calls_text().splitlines())

    r = run.publish("--upload")
    assert r.returncode == 2
    assert "already published" in r.stdout
    assert not [call for call in _calls_after(run, mark)
                if call.startswith(("s3 sync", "s3 cp"))]


def test_a_tables_root_that_cannot_be_listed_after_the_run_went_up_is_a_failure(run):
    """Exit code 2 says nothing was written, and by the time the tables are listed
    the run is up. Once the listing works, running again finishes the tables."""
    run.add_tables()

    r = run.publish("--upload", FAKE_S3_DENY_LIST="tables/")
    assert r.returncode == 1
    assert (run.published / "index.json").exists()
    assert "the run is published at" in r.stdout
    assert "NOT PUBLISHED" not in r.stdout
    assert (run.rd / "publish.done").read_text().strip() == "1"

    assert run.publish("--upload").returncode == 0
    assert (run.published_tables / "_days" / f"{DAY}.json").exists()


def test_a_tables_root_that_cannot_be_listed_in_a_dry_run_is_a_refusal(run):
    """Nothing was written, which is what exit code 2 says."""
    run.add_tables()

    r = run.publish(FAKE_S3_DENY_LIST="tables/")
    assert r.returncode == 2
    assert "NOT PUBLISHED" in r.stdout


# --------------------------------------------------------------------------- #
# --published: what a prune asks before it deletes a run's work directory
# --------------------------------------------------------------------------- #

def test_published_is_true_only_once_index_json_is_listed(run):
    assert run.publish("--published").returncode == 1
    assert run.publish("--upload").returncode == 0

    r = run.publish("--published")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "published:" in r.stdout


def test_published_also_wants_the_day_marker_when_the_run_wrote_tables(run):
    run.add_tables()
    lost = f"nodes/day={DAY}/part-00000.zstd.parquet"
    assert run.publish("--upload", FAKE_S3_DROP=lost).returncode == 1
    assert run.publish("--published").returncode == 1

    assert run.publish("--upload").returncode == 0
    assert run.publish("--published").returncode == 0


def test_published_still_holds_after_a_refused_rerun_overwrites_publish_done(run):
    """A prune that read publish.done would keep this finished run forever."""
    assert run.publish("--upload").returncode == 0
    assert run.publish("--upload").returncode == 2
    assert (run.rd / "publish.done").read_text().strip() == "2"

    assert run.publish("--published").returncode == 0


def test_published_writes_nothing(run):
    run.publish("--published")
    assert not (run.rd / "publish.done").exists()
    assert not (run.rd / "publish.log").exists()
    assert "s3 sync" not in run.calls_text()
    assert "s3 cp" not in run.calls_text()


def test_published_is_not_confirmed_when_the_listing_fails(run):
    assert run.publish("--upload").returncode == 0
    assert run.publish("--published", FAKE_S3_DENY_LIST="runs/").returncode == 1
