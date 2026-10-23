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
import re
import shutil
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
# When the seed leg wrote the day's tables, as its job manifest records them (#471).
TABLES_STARTED, TABLES_FINISHED = "2026-01-01T12:20:00Z", "2026-01-01T12:45:00Z"
TIMES = ("tables_started", "tables_finished")

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

    def __init__(self, tmp_path: Path, run_id: str = RUN_ID):
        self.run_id = run_id
        self.rd = tmp_path / "run"
        self.work = tmp_path / "work" / run_id
        self.pyg = self.work / "pyg" / f"year={YEAR}" / f"month={MONTH}"
        self.enriched = self.work / "enriched" / f"year={YEAR}" / f"month={MONTH}"
        self.manifests = self.work / "manifests" / f"year={YEAR}" / f"month={MONTH}"
        self.s3 = tmp_path / "s3"
        self.s3.mkdir()
        self.published = (self.s3 / "bucket" / "runs" / "all-sources"
                          / f"year={YEAR}" / f"month={MONTH}" / run_id)
        self.published_tables = self.s3 / "bucket" / "tables" / "all-sources"
        self.calls = tmp_path / "aws-calls.txt"
        self.bin = tmp_path / "stubbin"
        _write(self.bin / "aws", STUB_AWS).chmod(0o755)

        # The run directory, as the launcher and the recorder leave it.
        _write(self.rd / "run.log", f"[12:00:00] RUN_ID={run_id}\n")
        _write(self.rd / "run-config.txt",
               f"run id      : {run_id}\nwork dir    : {self.work}\n")
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
        self.tables_times = (TABLES_STARTED, TABLES_FINISHED)
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

    def add_feed_table(self, table: str, part: bytes = b"rows", day: str = DAY) -> set:
        """A partition of a table read from a feed, as the job writes it when it
        read the feed: splits/ or companyfacts/.

        A day the feed listed nothing still gets one, with a part file holding
        only the schema.
        """
        partition = self.work / "tables" / table / f"day={day}"
        _write(partition / "part-00000.zstd.parquet", part)
        _write(partition / ".part-00000.zstd.parquet.crc", b"c")
        _write(partition / "_SUCCESS", b"")
        return {f"{table}/day={day}/part-00000.zstd.parquet", f"{table}/day={day}/_SUCCESS"}

    def add_splits(self, part: bytes = b"splits rows", day: str = DAY) -> set:
        return self.add_feed_table("splits", part, day)

    def set_recorded_dataset(self, name: str, sources=("a", "b"), day=None,
                             **recorded) -> None:
        """dataset.json as the job writes it; ``recorded`` adds fields such as
        market_coverage."""
        body = {"dataset": name, "sources": list(sources), "time_period": f"{YEAR}-{MONTH}"}
        if day is not None:
            body["day"] = day
        body.update(recorded)
        _write(self.enriched / "dataset.json", json.dumps(body))

    def set_graph_schema(self, graph: str, body) -> None:
        """Rewrite one variant's graph_schema.json, or remove it when body is None.

        Its checksums.json row is rewritten with it. Every metadata file is
        hashed before anything is sent, and the list is compared with the files
        on disk, so a test that edits or removes one is otherwise refused before
        it reaches what it came to test.
        """
        meta = self.pyg / f"hetero_data_{graph}_metadata"
        schema, checksums = meta / "graph_schema.json", meta / "checksums.json"
        listed = json.loads(checksums.read_text())
        key = f"{meta.name}/graph_schema.json"
        if body is None:
            schema.unlink()
            listed["artifacts"].pop(key)
        else:
            _write(schema, body if isinstance(body, str) else json.dumps(body))
            listed["artifacts"][key] = {
                "bytes": schema.stat().st_size,
                "sha256": hashlib.sha256(schema.read_bytes()).hexdigest(),
            }
        _write(checksums, json.dumps(listed))

    def add_source_path(self, path: str) -> None:
        self.source_paths.append(path)
        self._write_manifests()

    def set_tables_times(self, times) -> None:
        """(started, finished) for the seed leg's manifest, or None for a run
        whose manifests record neither, as every run before #471."""
        self.tables_times = times
        self._write_manifests()

    def _write_manifests(self) -> None:
        seed = {"config": {"source_paths": self.source_paths}}
        if self.tables_times is not None:
            # The seed leg writes the tables; the .pt legs record no times.
            seed["result"] = dict(zip(TIMES, self.tables_times))
        _write(self.manifests / "enrichment_only_1.json", json.dumps(seed))
        _write(self.manifests / "pyg_only_1.json", json.dumps({"config": {"source_paths": []}}))

    def publish(self, *args: str, python_flags=(), **env_changes):
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
        return subprocess.run([sys.executable, *python_flags, str(PUBLISHER), str(self.rd), *args],
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
# What reached each variant
#
# The run-level `sources` is what the job read. A variant can hold fewer, and
# two variants of the same run can hold different sets, so the answer belongs to
# the variant. Only its graph_schema.json has it (#419).
# --------------------------------------------------------------------------- #

def _schema(sources_in_graph=None, version="1.5", day=DAY) -> dict:
    """A graph_schema.json. None leaves a key out, as an older version does."""
    build = {"sources": ["a", "b"]}
    if sources_in_graph is not None:
        build["sources_in_graph"] = sources_in_graph
    if day is not None:
        build["day"] = day
    return {"version": version, "build_metadata": build}


def test_each_variant_carries_the_sources_that_reached_it(run):
    run.set_graph_schema("1024d", _schema(["a"]))
    run.set_graph_schema("no_edge_features", _schema(["a", "b"]))

    assert run.publish("--upload").returncode == 0
    index = json.loads((run.published / "index.json").read_text())
    assert index["sources"] == ["a", "b"]
    assert index["variants"]["1024d"]["sources_in_graph"] == ["a"]
    assert index["variants"]["no_edge_features"]["sources_in_graph"] == ["a", "b"]


def test_a_variant_written_before_the_key_existed_is_left_without_it(run):
    """Absent, not guessed.

    The only thing to guess from is the exclusion list, and the guess is wrong
    exactly when it matters -- a source can also leave by arriving empty. The
    rest of the entry is unchanged, so a reader loses one key, not the variant.
    """
    run.set_graph_schema("1024d", _schema(version="1.3", day=None))
    run.set_graph_schema("no_edge_features", _schema(["a"]))

    assert run.publish("--upload").returncode == 0
    variants = json.loads((run.published / "index.json").read_text())["variants"]
    assert "sources_in_graph" not in variants["1024d"]
    assert variants["1024d"]["path"] == "1024d/"
    assert variants["no_edge_features"]["sources_in_graph"] == ["a"]


@pytest.mark.parametrize("damage", ["unparseable", "missing"])
def test_a_graph_schema_json_the_index_cannot_read_still_publishes(run, damage):
    """Building index.json adds no new way for a publish to fail.

    index.json points at the files it lists and they go up either way. A missing
    schema is normally caught earlier by the checksums check, so this drops it
    from checksums.json too -- otherwise the run is refused before reaching the
    line under test.
    """
    run.set_graph_schema("1024d", None if damage == "missing" else "{not json")

    r = run.publish("--upload")
    assert r.returncode == 0, r.stdout + r.stderr
    variants = json.loads((run.published / "index.json").read_text())["variants"]
    assert "sources_in_graph" not in variants["1024d"]
    assert (run.published / "1024d" / "hetero_data_1024d.pt").exists()


# --------------------------------------------------------------------------- #
# The day each variant was cut from
#
# From schema 1.5 a variant's graph_schema.json records the day its data was
# cut from, and everything downstream labels the build by it. It has to be the
# day index.json names for the run (#424).
# --------------------------------------------------------------------------- #

def test_a_variant_whose_day_is_the_runs_day_publishes(run):
    run.set_graph_schema("1024d", _schema(["a"], day=DAY))
    run.set_graph_schema("no_edge_features", _schema(["a", "b"], day=DAY))

    r = run.publish("--upload")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((run.published / "index.json").read_text())["data_day"] == DAY


def test_a_variant_whose_day_is_another_day_is_refused(run):
    """Published, it would be labelled with a day its data was not cut from,
    and nothing downstream has anything to check that label against."""
    run.set_graph_schema("1024d", _schema(["a"], day="2026-01-01"))

    r = run.publish("--upload")
    assert r.returncode == 2
    assert "says day 2026-01-01" in r.stdout
    assert "s3 sync" not in run.calls_text()


@pytest.mark.parametrize("version,day", [("1.4", None), ("1.5", "")])
def test_a_variant_with_no_day_publishes_as_it_does_today(run, version, day):
    """1.4 has no day at all, and "" says the job had none. Neither disagrees."""
    run.set_graph_schema("1024d", _schema(["a"], version=version, day=day))

    r = run.publish("--upload")
    assert r.returncode == 0, r.stdout + r.stderr


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


def test_source_paths_naming_two_days_take_the_day_the_job_recorded(run):
    """A snapshot from the day before puts a second day in the paths. The job is
    told the day and records it, so the publish needs no variable, and neither
    does the prune's --published, which asks about older runs (#434)."""
    run.add_source_path("s3a://BUCKET/raw/source=c/year=2026/month=01/01.snappy.parquet")
    run.set_recorded_dataset("", day=DAY)
    assert run.publish("--upload").returncode == 0
    assert json.loads((run.published / "index.json").read_text())["data_day"] == DAY
    assert run.publish("--published").returncode == 0


def test_a_recorded_day_the_paths_do_not_name_is_refused(run):
    run.set_recorded_dataset("", day="2026-01-05")
    refused = run.publish()
    assert refused.returncode == 2
    assert "the day in dataset.json is 2026-01-05" in refused.stdout


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


def test_the_day_marker_names_the_day_the_run_and_when_it_was_published(run):
    """Another service reads run_id and published from every marker, so a
    renamed or reshaped field has to fail here rather than there."""
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    marker = json.loads((run.published_tables / "_days" / f"{DAY}.json").read_text())
    assert marker["day"] == DAY
    assert marker["run_id"] == RUN_ID
    assert re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", marker["run_id"])
    assert re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                        marker["published"])


# --------------------------------------------------------------------------- #
# The day marker's sources
#
# Every source the run read, and the split feed when the day has a splits/
# partition. The feed never becomes a node, so a reader that works a day's
# sources out from its node types never sees it (#432). The same for the
# company facts history and companyfacts/ (#437).
# --------------------------------------------------------------------------- #

def _marker(run: Run) -> dict:
    return json.loads((run.published_tables / "_days" / f"{DAY}.json").read_text())


def test_the_day_marker_lists_the_runs_sources_sorted(run):
    run.set_recorded_dataset("", ["z", "a"])
    run.add_tables()
    run.add_splits()
    assert run.publish("--upload").returncode == 0
    assert _marker(run)["sources"] == ["a", "stock-split", "z"]


def test_the_day_marker_lists_the_companyfacts_feed_the_run_read(run):
    """The job names the snapshot feed in dataset.json when a path read it,
    and the marker carries it as it does any source (#434)."""
    run.set_recorded_dataset("", ["sec", "sec-companyfacts"])
    run.add_tables()
    assert run.publish("--upload").returncode == 0
    assert _marker(run)["sources"] == ["sec", "sec-companyfacts"]


@pytest.mark.parametrize("part,listed", [
    (None, False),
    (b"the schema", True),
    (b"the schema and rows", True),
], ids=["no partition", "empty partition", "partition with rows"])
def test_the_split_feed_is_listed_when_the_day_has_a_splits_partition(run, part, listed):
    """An empty partition says the feed was read and listed no splits. No
    partition says it was not read."""
    run.add_tables()
    if part is not None:
        run.add_splits(part)
    assert run.publish("--upload").returncode == 0
    assert _marker(run)["sources"] == (["a", "b", "stock-split"] if listed else ["a", "b"])


@pytest.mark.parametrize("part,listed", [
    (None, False),
    (b"the schema", True),
    (b"the schema and rows", True),
], ids=["no partition", "empty partition", "partition with rows"])
def test_the_companyfacts_history_is_listed_when_the_day_has_its_partition(
    run, part, listed,
):
    """The history is read for its table alone, as the split feed is, so a run
    that read no snapshot still names the feed when the day has the table
    (#437)."""
    run.add_tables()
    if part is not None:
        run.add_feed_table("companyfacts", part)
    assert run.publish("--upload").returncode == 0
    assert _marker(run)["sources"] == (
        ["a", "b", "sec-companyfacts"] if listed else ["a", "b"]
    )


def test_a_run_that_read_both_companyfacts_feeds_lists_the_name_once(run):
    run.set_recorded_dataset("", ["sec", "sec-companyfacts"])
    run.add_tables()
    run.add_feed_table("companyfacts")
    assert run.publish("--upload").returncode == 0
    assert _marker(run)["sources"] == ["sec", "sec-companyfacts"]


def test_the_day_marker_keeps_its_other_fields(run):
    expected = run.add_tables() | run.add_splits()
    assert run.publish("--upload").returncode == 0
    marker = _marker(run)
    assert set(marker) == {"day", "run_id", "tables_started", "tables_finished",
                           "published", "sources", "tables", "files"}
    assert (marker["day"], marker["run_id"]) == (DAY, RUN_ID)
    assert marker["tables"] == ["edges", "facts", "graph", "nodes", "splits"]
    assert marker["files"] == len(expected)


# --------------------------------------------------------------------------- #
# The day marker's market coverage
#
# How complete the day's market data was, as the job recorded it in dataset.json.
# A partial day publishes like any other, so this is where a reader of the
# markers sees one (#452).
# --------------------------------------------------------------------------- #

COVERAGE = {
    "constituents": 503,
    "snapshots": 20,
    "stocks": 503,
    "stocks_with_options": 138,
    "option_rows": 161232,
    "by_snapshot": [
        {"capture_time": "2026-10-16T13:40:40.100Z", "stocks": 503,
         "stocks_with_options": 138, "option_rows": 161232},
    ],
}


def test_the_day_marker_carries_the_market_coverage_the_run_recorded(run):
    """The day's three counts after its sources; the per-snapshot list stays in
    graph_schema.json."""
    run.set_recorded_dataset("", ["market"], market_coverage=COVERAGE)
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    marker = _marker(run)
    assert marker["market_coverage"] == {
        "snapshots": 20, "stocks": 503, "stocks_with_options": 138,
    }
    assert list(marker) == ["day", "run_id", "tables_started", "tables_finished",
                            "published", "sources", "market_coverage", "tables", "files"]


def test_a_run_that_recorded_no_market_coverage_leaves_it_off_the_marker(run):
    """A run that read no market source records null; one from before #452
    records nothing. Neither marker has the field."""
    run.set_recorded_dataset("", ["sec"], market_coverage=None)
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    assert "market_coverage" not in _marker(run)


# --------------------------------------------------------------------------- #
# When the day's tables were written
#
# published is when the upload started, so it says nothing about when the
# tables were made. The seed leg's job manifest records that, and the marker
# carries it after run_id, in the order things happened (#471).
# --------------------------------------------------------------------------- #

def test_the_day_marker_says_when_the_run_wrote_the_tables(run):
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    marker = _marker(run)
    assert [marker[key] for key in TIMES] == [TABLES_STARTED, TABLES_FINISHED]
    assert list(marker)[:5] == ["day", "run_id", *TIMES, "published"]


def test_a_run_that_recorded_no_times_leaves_both_off_the_marker(run):
    """A run from before #471 has manifests with neither. Its marker says
    nothing rather than guess, and keeps the fields it always had."""
    run.set_tables_times(None)
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    marker = _marker(run)
    assert not set(TIMES) & set(marker)
    assert list(marker)[:3] == ["day", "run_id", "published"]


def test_two_legs_that_recorded_times_give_the_later_finish(run):
    """One leg writes the tables in practice. Were two to, the tables on disk
    are the later one's, so its times are the ones that describe them."""
    later = ("2026-01-01T13:05:00Z", "2026-01-01T13:30:00Z")
    _write(run.manifests / "full_2.json", json.dumps({
        "config": {"source_paths": run.source_paths},
        "result": dict(zip(TIMES, later)),
    }))
    run.add_tables()
    assert run.publish("--upload").returncode == 0

    assert [_marker(run)[key] for key in TIMES] == list(later)


def test_the_publisher_runs_on_a_python_with_no_packages(run):
    """The nightly runs it on the system python, which has no pyarrow. It
    imports the feeds' names from spark_jobs/graph/splits.py and
    spark_jobs/graph/companyfacts.py, so both have to load with the standard
    library alone."""
    run.add_tables()
    run.add_splits()
    run.add_feed_table("companyfacts")
    r = run.publish("--upload", python_flags=["-S"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert _marker(run)["sources"] == ["a", "b", "sec-companyfacts", "stock-split"]


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


def test_tables_whose_run_recorded_no_sources_are_refused_before_anything_is_written(run):
    """The day's marker would have to list the sources without knowing them."""
    run.add_tables()
    (run.enriched / "dataset.json").unlink()
    r = run.publish("--upload")
    assert r.returncode == 2
    assert "recorded no sources" in r.stdout
    assert "s3 sync" not in run.calls_text()


def test_a_run_with_no_tables_needs_no_recorded_sources(run):
    (run.enriched / "dataset.json").unlink()
    r = run.publish("--upload")
    assert r.returncode == 0, r.stdout + r.stderr


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


# --------------------------------------------------------------------------- #
# --archive: the local archive
# --------------------------------------------------------------------------- #

def _archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "drive" / "archive"
    root.mkdir(parents=True)
    return root


def _archived_run(root: Path, run_id: str = RUN_ID) -> Path:
    return root / "runs" / "all-sources" / f"year={YEAR}" / f"month={MONTH}" / run_id


def _archived_tables(root: Path) -> Path:
    return root / "tables" / "all-sources"


def _snapshot(folder: Path) -> dict:
    """Each file's bytes and mtime, so a test can tell nothing was written again."""
    return {p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


def _second_run_of_the_day(tmp_path: Path) -> Run:
    (tmp_path / "rerun").mkdir()
    return Run(tmp_path / "rerun", run_id="20260101T180000Z")


def _archive_an_older_day(root: Path, day: str, run_id: str) -> Path:
    """A day already in the archive, as an earlier --archive left it."""
    year, month, _ = day.split("-")
    folder = root / "runs" / "all-sources" / f"year={year}" / f"month={month}" / run_id
    _write(folder / "1024d" / "hetero_data_1024d.pt", b"older")
    _write(_archived_tables(root) / "nodes" / f"day={day}" / "part-00000.zstd.parquet", b"rows")
    _write(_archived_tables(root) / "_days" / f"{day}.json",
           json.dumps({"day": day, "run_id": run_id}))
    return folder


def test_an_archive_holds_the_run_and_its_tables_as_real_files_and_sends_nothing(run, tmp_path):
    """The published layout less the enriched triples and index.json, as files rather
    than links, with no S3 at all: it needs neither publish root."""
    tables = run.add_tables()
    root = _archive_root(tmp_path)

    r = run.publish("--archive", str(root), PYG_PUBLISH_ROOT=None, PYG_TABLES_ROOT=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert files(_archived_run(root)) == {name for name in expected_names()
                                          if not name.startswith("enriched/")
                                          and name != "index.json"}
    assert files(_archived_tables(root)) == tables | {f"_days/{DAY}.json"}
    assert not [p for p in root.rglob("*") if p.is_symlink()]
    assert ((_archived_run(root) / "1024d" / "hetero_data_1024d.pt").read_bytes()
            == (run.pyg / "hetero_data_1024d.pt").read_bytes())
    assert files(run.s3) == set()
    assert run.calls_text() == ""


def test_the_archives_tables_root_reads_like_the_published_one(run, tmp_path):
    """The same names under <root>/tables/<dataset> as under the tables root, and a day
    marker with the same fields."""
    run.add_tables()
    run.add_splits()
    root = _archive_root(tmp_path)
    assert run.publish("--upload").returncode == 0
    assert run.publish("--archive", str(root)).returncode == 0

    assert files(_archived_tables(root)) == files(run.published_tables)
    archived = json.loads((_archived_tables(root) / "_days" / f"{DAY}.json").read_text())
    published = json.loads((run.published_tables / "_days" / f"{DAY}.json").read_text())
    archived.pop("published")
    published.pop("published")
    assert archived == published


def test_the_archives_day_marker_carries_the_same_market_coverage(run, tmp_path):
    """The archive's marker comes from the same day_marker(), so a trainer
    reading the archive sees what a reader of the published tables sees. #452"""
    run.set_recorded_dataset("", ["market"], market_coverage=COVERAGE)
    run.add_tables()
    root = _archive_root(tmp_path)
    assert run.publish("--upload").returncode == 0
    assert run.publish("--archive", str(root)).returncode == 0

    archived = json.loads((_archived_tables(root) / "_days" / f"{DAY}.json").read_text())
    assert archived["market_coverage"] == _marker(run)["market_coverage"]


def test_the_archives_day_marker_carries_the_same_tables_times(run, tmp_path):
    """Both markers come from day_marker() and the same manifest. #471"""
    run.add_tables()
    root = _archive_root(tmp_path)
    assert run.publish("--upload").returncode == 0
    assert run.publish("--archive", str(root)).returncode == 0

    archived = json.loads((_archived_tables(root) / "_days" / f"{DAY}.json").read_text())
    assert [archived[key] for key in TIMES] == [_marker(run)[key] for key in TIMES]
    assert list(archived)[:5] == ["day", "run_id", *TIMES, "published"]


def test_the_day_marker_is_the_last_file_the_archive_writes(run, tmp_path):
    run.add_tables()
    root = _archive_root(tmp_path)
    assert run.publish("--archive", str(root)).returncode == 0

    marker = _archived_tables(root) / "_days" / f"{DAY}.json"
    others = [p for p in root.rglob("*") if p.is_file() and p != marker]
    assert marker.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in others)


def test_archiving_a_run_again_changes_nothing(run, tmp_path):
    run.add_tables()
    root = _archive_root(tmp_path)
    assert run.publish("--archive", str(root)).returncode == 0
    before = _snapshot(root)

    r = run.publish("--archive", str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nothing to copy" in r.stdout
    assert _snapshot(root) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_a_copy_stopped_before_its_marker_is_finished_by_running_again(run, tmp_path):
    """No marker until every file is there and checked. The next run finishes the copy
    and leaves no temporary file behind."""
    tables = run.add_tables()
    root = _archive_root(tmp_path)
    blocked = run.pyg / "hetero_data_1024d_node_index" / "part-00000.snappy.parquet"
    blocked.chmod(0)
    try:
        r = run.publish("--archive", str(root))
    finally:
        blocked.chmod(0o644)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ARCHIVE FAILED" in r.stdout
    assert not (_archived_tables(root) / "_days").exists()
    assert files(_archived_run(root))

    r = run.publish("--archive", str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert files(_archived_tables(root)) == tables | {f"_days/{DAY}.json"}
    assert "1024d/node_index/part-00000.snappy.parquet" in files(_archived_run(root))
    assert not list(root.rglob("*.partial"))


def test_a_second_run_of_the_day_replaces_that_days_copy(run, tmp_path):
    """The archive keeps days, not runs. Nothing of the first run's copy stays beside the
    second's: not its run, and not a table only the first run wrote."""
    run.add_tables()
    run.add_splits()
    root = _archive_root(tmp_path)
    assert run.publish("--archive", str(root)).returncode == 0
    rerun = _second_run_of_the_day(tmp_path)
    tables = rerun.add_tables()

    r = rerun.publish("--archive", str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not _archived_run(root).exists()
    assert files(_archived_run(root, rerun.run_id))
    assert files(_archived_tables(root)) == tables | {f"_days/{DAY}.json"}
    marker = json.loads((_archived_tables(root) / "_days" / f"{DAY}.json").read_text())
    assert marker["run_id"] == rerun.run_id


def test_keep_days_keeps_the_newest_days_and_takes_out_the_rest(run, tmp_path):
    run.add_tables()
    root = _archive_root(tmp_path)
    kept = _archive_an_older_day(root, "2025-12-31", "20260101T050000Z")
    gone = [_archive_an_older_day(root, "2025-12-30", "20251231T050000Z"),
            _archive_an_older_day(root, "2025-11-28", "20251129T050000Z")]

    r = run.publish("--archive", str(root), "--keep-days", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in (_archived_tables(root) / "_days").iterdir()) == [
        "2025-12-31.json", f"{DAY}.json"]
    assert kept.is_dir()
    assert (_archived_tables(root) / "nodes" / "day=2025-12-31").is_dir()
    for folder in gone:
        assert not folder.exists()
    assert not (_archived_tables(root) / "nodes" / "day=2025-12-30").exists()
    # A month whose last day left goes with it.
    assert not (root / "runs" / "all-sources" / "year=2025" / "month=11").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes a file whatever its folder's mode")
def test_an_older_days_marker_is_removed_before_its_files(run, tmp_path):
    """A trim that stops partway leaves files with no marker, a day plainly unfinished,
    never a marker naming files that are gone."""
    root = _archive_root(tmp_path)
    stuck = _archive_an_older_day(root, "2025-12-30", "20251231T050000Z") / "1024d"
    stuck.chmod(0o555)
    try:
        r = run.publish("--archive", str(root), "--keep-days", "1")
    finally:
        stuck.chmod(0o755)
    assert r.returncode == 1, r.stdout + r.stderr
    assert not (_archived_tables(root) / "_days" / "2025-12-30.json").exists()
    assert (stuck / "hetero_data_1024d.pt").exists()
    assert (_archived_tables(root) / "_days" / f"{DAY}.json").exists()


def test_an_archive_root_that_is_not_there_is_refused(run, tmp_path):
    """An unmounted drive leaves no root, and the copy must not land on whatever disk
    holds the mount point."""
    root = tmp_path / "drive" / "archive"
    r = run.publish("--archive", str(root))
    assert r.returncode == 2
    assert "is its drive mounted" in r.stdout
    assert not root.exists()


def test_a_run_that_did_not_succeed_is_not_archived(run, tmp_path):
    root = _archive_root(tmp_path)
    _write(run.rd / "run.done", "1\n")
    assert run.publish("--archive", str(root)).returncode == 2
    assert files(root) == set()


def test_a_graph_that_no_longer_matches_checksums_json_is_not_archived(run, tmp_path):
    root = _archive_root(tmp_path)
    _write(run.pyg / "hetero_data_1024d.pt", b"not what was hashed")
    r = run.publish("--archive", str(root))
    assert r.returncode == 2
    assert "checksums.json" in r.stdout
    assert files(root) == set()


def test_keep_days_goes_with_archive(run):
    r = run.publish("--keep-days", "3")
    assert r.returncode == 2
    assert "goes with --archive" in r.stderr


# --------------------------------------------------------------------------- #
# --archived: what a prune asks when the schedule keeps an archive
# --------------------------------------------------------------------------- #

def test_archived_is_true_only_once_the_day_and_its_run_are_in_the_archive(run, tmp_path):
    root = _archive_root(tmp_path)
    assert run.publish("--archived", str(root)).returncode == 1
    assert run.publish("--archive", str(root)).returncode == 0

    r = run.publish("--archived", str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    shutil.rmtree(_archived_run(root))
    assert run.publish("--archived", str(root)).returncode == 1


def test_archived_holds_for_a_run_whose_day_a_second_run_replaced(run, tmp_path):
    root = _archive_root(tmp_path)
    assert run.publish("--archive", str(root)).returncode == 0
    assert _second_run_of_the_day(tmp_path).publish("--archive", str(root)).returncode == 0
    assert run.publish("--archived", str(root)).returncode == 0


def test_archived_writes_nothing(run, tmp_path):
    root = _archive_root(tmp_path)
    run.publish("--archived", str(root))
    assert files(root) == set()
    assert not (run.rd / "publish.log").exists()
