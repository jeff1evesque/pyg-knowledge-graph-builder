#!/usr/bin/env python3
"""Publish a finished cluster run to the published-runs prefix.

    bin/publish_run.py <run-dir>            dry run: lay the upload out and show it
    bin/publish_run.py <run-dir> --upload   upload, check the listing, then index.json

The job's work directory is not the published layout. It differs in four ways:

  1. the period moves above the run id and is dropped below it, so
     enriched/year=YYYY/month=MM/triples/ becomes enriched/triples/
  2. hetero_data_<v>_metadata/*.json becomes <v>/*.json
  3. hetero_data_<v>_node_index/ becomes <v>/node_index/
  4. hetero_data_<v>.pt keeps its name, inside <v>/

pyg/latest/, checkpoints/ and Hadoop's .crc files are left out.

The destination takes writes and listings, but not reads or deletes. A file sent to
the wrong key stays there, and nothing can be read back to check it. So a dry run is
the default, every .pt and metadata file is checked against checksums.json before it
goes, the upload is checked by name and size from a listing, and index.json goes up
last. A run folder without index.json is a publish that did not finish. Running again
resumes it, because files already there at the right size are skipped.

<run-dir> is read the way the launcher and the recorder write it: run.log for the run
id, run-config.txt for the work directory, run.done and outcome.txt for whether the
run succeeded. A run that did not succeed is refused.

Environment:
  PYG_PUBLISH_ROOT      where published runs go, s3://BUCKET/PREFIX. Required.
  PYG_PUBLISH_DATASET   the source set's name, e.g. all-sources. Defaults to the name
                        the job wrote into dataset.json. One of the two must be set.
  PYG_PUBLISH_DATA_DAY  YYYY-MM-DD, the day the sources were cut from. Defaults to the
                        one day the day-level source paths in the manifests name.

Writes <run-dir>/publish/ (symlinks plus index.json) and appends to publish.log there.
An --upload also writes its exit code to publish.done: 0 published, 1 the upload or
its check failed, 2 refused before anything was written.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CHECKSUMS = "checksums.json"
INDEX = "index.json"
TREE = "publish"

# A day-level source path: .../year=2026/month=09/09.snappy.parquet or .../day=09/
DAY_IN_PATH = re.compile(r"year=(\d{4})/month=(\d{2})/(?:day=)?(\d{2})(?:\.[\w.]+)?/?$")
# The job's summary line naming the graph a submission wrote.
PT_OUTPUT = re.compile(r"PyG output:\s+\S*/hetero_data_(.+)\.pt\s*$")
# A submission's heading in the notebook's output, as bin/record_run_outcome.sh reads it.
SUBMISSION = re.compile(r"^===\s+(\S.*?)\s*$")


class Refused(Exception):
    """A reason not to publish, found before anything was written."""


class Failed(Exception):
    """Something went wrong after writing started."""


class Log:
    """Print a line and append it to the run's publish.log."""

    def __init__(self, path: Path):
        self.fh = open(path, "a")

    def __call__(self, msg: str = "", echo: bool = True) -> None:
        if echo:
            print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# What the run left behind
# --------------------------------------------------------------------------- #

def read_run(rd: Path):
    """The run id and the work directory."""
    run_log, config = rd / "run.log", rd / "run-config.txt"
    run_id = ""
    if run_log.exists():
        for line in run_log.read_text(errors="replace").splitlines():
            if "RUN_ID=" in line:
                run_id = line.split("RUN_ID=", 1)[1].strip()
                break
    if not run_id:
        raise Refused(f"no RUN_ID= line in {run_log}")

    work = None
    if config.exists():
        for line in config.read_text(errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "work dir":
                work = Path(value.strip())
    if work is None:
        raise Refused(f"no 'work dir' line in {config}")
    # The launcher's contract: the work directory ends in the run id.
    if work.name != run_id:
        raise Refused(f"work dir {work} does not end in the run id {run_id}")
    if not work.is_dir():
        raise Refused(f"work dir {work} does not exist")
    return run_id, work


def check_succeeded(rd: Path) -> None:
    done = rd / "run.done"
    rc = done.read_text().strip() if done.exists() else "missing"
    if rc != "0":
        raise Refused(f"run.done is {rc}, not 0")
    state = "missing"
    outcome = rd / "outcome.txt"
    if outcome.exists():
        for line in outcome.read_text(errors="replace").splitlines():
            if line.startswith("state"):
                state = line.partition(":")[2].strip()
                break
    if not re.fullmatch(r"all \d+ submissions ok", state):
        raise Refused(f"outcome.txt state is '{state}', not 'all N submissions ok'. "
                      "If outcome.txt is missing, run bin/record_run_outcome.sh first.")


def find_period(work: Path):
    months = [p for p in (work / "pyg").glob("year=*/month=*") if p.is_dir()]
    if len(months) != 1:
        raise Refused(f"expected one year=/month= folder under {work / 'pyg'}, "
                      f"found {len(months)}")
    year = months[0].parent.name.partition("=")[2]
    month = months[0].name.partition("=")[2]
    if not (re.fullmatch(r"\d{4}", year) and re.fullmatch(r"\d{2}", month)):
        raise Refused(f"cannot read a period from {months[0]}")
    return year, month


def find_graphs(pyg: Path):
    names = sorted(p.name[len("hetero_data_"):-len(".pt")] for p in pyg.glob("hetero_data_*.pt"))
    if not names:
        raise Refused(f"no hetero_data_*.pt in {pyg}")
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check_graph(pyg: Path, graph: str, log: Log) -> None:
    """Every .pt and metadata file against the size and sha256 in checksums.json."""
    meta = pyg / f"hetero_data_{graph}_metadata"
    if not (meta / CHECKSUMS).exists():
        raise Refused(f"{meta / CHECKSUMS} is missing")
    if not (pyg / f"hetero_data_{graph}_node_index" / "_SUCCESS").exists():
        raise Refused(f"hetero_data_{graph}_node_index has no _SUCCESS")

    listed = json.loads((meta / CHECKSUMS).read_text())["artifacts"]
    on_disk = {f"hetero_data_{graph}.pt"} | {
        f"{meta.name}/{p.name}" for p in meta.glob("*.json") if p.name != CHECKSUMS}
    if set(listed) != on_disk:
        raise Refused(f"{meta.name}/{CHECKSUMS} and the files on disk differ: "
                      f"{sorted(set(listed) ^ on_disk)}")

    log(f"  {graph}: checking {len(listed)} files")
    for name in sorted(listed):
        path, want = pyg / name, listed[name]
        size = path.stat().st_size
        if size != want["bytes"]:
            raise Refused(f"{name} is {size} bytes, checksums.json says {want['bytes']}")
        if sha256(path) != want["sha256"]:
            raise Refused(f"{name} does not match its sha256 in checksums.json")


def dataset_name(enriched: Path, period: str, env):
    """The dataset folder name, and the sources the job recorded."""
    descriptor = enriched / "dataset.json"
    recorded = json.loads(descriptor.read_text()) if descriptor.exists() else {}
    if recorded.get("time_period") not in (None, period):
        raise Refused(f"dataset.json says period {recorded['time_period']}, "
                      f"the folders say {period}")
    given = env.get("PYG_PUBLISH_DATASET", "").strip()
    job = (recorded.get("dataset") or "").strip()
    if given and job and given != job:
        raise Refused(f"PYG_PUBLISH_DATASET is {given}, but the job recorded dataset {job}")
    name = given or job
    if not name:
        raise Refused("the job ran without --dataset, so set PYG_PUBLISH_DATASET "
                      "(e.g. all-sources)")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise Refused(f"'{name}' cannot be a folder name")
    return name, recorded.get("sources", [])


def data_day(work: Path, env) -> str:
    """The path's month= segment cannot tell two runs apart, so index.json names the day."""
    named = set()
    for manifest in sorted(work.glob("manifests/year=*/month=*/*.json")):
        try:
            paths = json.loads(manifest.read_text()).get("config", {}).get("source_paths") or []
        except ValueError:
            continue
        for path in paths:
            found = DAY_IN_PATH.search(path)
            if found:
                named.add("-".join(found.groups()))

    given = env.get("PYG_PUBLISH_DATA_DAY", "").strip()
    if given:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", given):
            raise Refused(f"PYG_PUBLISH_DATA_DAY {given} is not YYYY-MM-DD")
        if named and given not in named:
            raise Refused(f"PYG_PUBLISH_DATA_DAY is {given}, but the source paths "
                          f"name {', '.join(sorted(named))}")
        return given
    if len(named) != 1:
        raise Refused(f"the source paths name {len(named)} days "
                      f"({', '.join(sorted(named)) or 'none'}); set PYG_PUBLISH_DATA_DAY")
    return named.pop()


def notebook_labels(rd: Path) -> dict:
    """Graph name -> the notebook's name for the leg that built it. No artifact records it."""
    cells, labels, current = rd / "notebook-cells.txt", {}, None
    if not cells.exists():
        return labels
    with open(cells, errors="replace") as fh:
        for line in fh:
            if line.startswith("==="):
                heading = SUBMISSION.match(line)
                if heading:
                    current = heading.group(1)
            elif current and "PyG output:" in line:
                found = PT_OUTPUT.search(line)
                if found:
                    labels[found.group(1)] = current
    return labels


# --------------------------------------------------------------------------- #
# The published layout
# --------------------------------------------------------------------------- #

def files_under(folder: Path, published: str) -> list:
    """Every file below a folder, less Hadoop's .crc files and _temporary leftovers."""
    return [(f"{published}/{p.relative_to(folder).as_posix()}", p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and not p.name.endswith(".crc")
            and "_temporary" not in p.relative_to(folder).parts]


def plan(work: Path, period_dirs: str, graphs) -> list:
    """(published name, file on disk) for everything except index.json."""
    pyg = work / "pyg" / period_dirs
    items = []
    for graph in graphs:
        items.append((f"{graph}/hetero_data_{graph}.pt", pyg / f"hetero_data_{graph}.pt"))
        items += files_under(pyg / f"hetero_data_{graph}_metadata", graph)
        items += files_under(pyg / f"hetero_data_{graph}_node_index", f"{graph}/node_index")
    enriched = work / "enriched" / period_dirs
    if (enriched / "dataset.json").exists():
        items.append(("enriched/dataset.json", enriched / "dataset.json"))
    items += files_under(enriched / "triples", "enriched/triples")
    items += files_under(work / "manifests" / period_dirs, "manifests")
    return items


def make_index(run_id, dataset, period, day, sources, rd, graphs, labels, items) -> dict:
    metadata = sorted({name.split("/", 1)[1] for name, _ in items
                       if name.count("/") == 1 and name.split("/", 1)[0] in graphs
                       and name.endswith(".json")})
    run_level = {"enriched": "enriched/triples/", "manifests": "manifests/"}
    if any(name == "enriched/dataset.json" for name, _ in items):
        run_level["dataset"] = "enriched/dataset.json"
    return {
        "run_id": run_id,
        "dataset": dataset,
        "time_period": period,
        "data_day": day,
        "sources": sources,
        "source_of_run": rd.name,
        "published": utc_now(),
        "note": "Graph folders are named for the artifact files. notebook_label is the "
                "notebook's name for the leg that built each one; no artifact records it.",
        "variants": {g: {"path": f"{g}/", "notebook_label": labels.get(g)} for g in graphs},
        "run_level": run_level,
        "per_variant_files": metadata + ["hetero_data_<variant>.pt", "node_index/"],
    }


def build_tree(tree: Path, items, index: dict) -> None:
    """Lay the files out as they will be published, as symlinks. aws s3 sync follows them."""
    if tree.is_symlink():
        raise Refused(f"{tree} is a symlink; not clearing it")
    if tree.exists():
        shutil.rmtree(tree)  # symlinks and index.json only; rmtree does not follow the links
    for name, source in items:
        link = tree / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
    (tree / INDEX).write_text(json.dumps(index, indent=2) + "\n")


def summarize(items, log: Log) -> None:
    groups = {}
    for name, source in items:
        folder = name.rsplit("/", 1)[0]
        count, size = groups.get(folder, (0, 0))
        groups[folder] = (count + 1, size + source.stat().st_size)
    width = max(len(folder) for folder in groups)
    for folder in sorted(groups):
        count, size = groups[folder]
        noun = "file " if count == 1 else "files"
        log(f"  {folder:<{width}}  {count:>4} {noun}  {size / 1e9:8.2f} GB")
    total = sum(size for _, size in groups.values())
    log(f"  {'total':<{width}}  {len(items):>4} files  {total / 1e9:8.2f} GB, plus index.json")


# --------------------------------------------------------------------------- #
# The destination
# --------------------------------------------------------------------------- #

def aws(*args) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["aws", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise Refused("the aws CLI is not on PATH")


def aws_streamed(log: Log, echo: bool, *args):
    """Run aws, copying its output into the log as it comes. Returns (rc, lines)."""
    proc = subprocess.Popen(["aws", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        log("  " + lines[-1], echo=echo)
    return proc.wait(), lines


def published(dst: str) -> dict:
    """{name below dst: (size, checksum algorithm)} for what is already there."""
    bucket, _, prefix = dst[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    r = aws("s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix,
            "--output", "json")
    if r.returncode != 0:
        raise Failed(f"cannot list {dst}/: {r.stderr.strip()}")
    try:
        body = json.loads(r.stdout) if r.stdout.strip() else {}
    except ValueError:
        raise Failed(f"cannot read the listing of {dst}/")
    return {obj["Key"][len(prefix):]: (obj["Size"], ",".join(obj.get("ChecksumAlgorithm") or []))
            for obj in body.get("Contents") or []}


def check_upload(items, tree: Path, dst: str, log: Log) -> bool:
    """Compare the destination's listing with the upload tree, name by name."""
    want = {name: (tree / name).stat().st_size for name, _ in items}
    have = published(dst)
    missing = sorted(set(want) - set(have))
    wrong = sorted(name for name in want if name in have and have[name][0] != want[name])
    extra = sorted(set(have) - set(want) - {INDEX})
    for label, names in (("missing", missing), ("wrong size", wrong),
                         ("not in the upload tree", extra)):
        for name in names[:20]:
            log(f"  {label}: {name}")
    if missing or wrong or extra:
        log(f"CHECK FAILED: {len(missing)} missing, {len(wrong)} wrong size, "
            f"{len(extra)} unexpected. index.json was not written.")
        return False
    algorithms = sorted({have[name][1] for name in want if have[name][1]})
    carrying = sum(1 for name in want if have[name][1])
    log(f"check ok: all {len(want)} files are there at the right size; {carrying} carry "
        f"an S3 checksum ({', '.join(algorithms) or 'none'})")
    return True


def publish(rd: Path, upload: bool, env, log: Log) -> int:
    run_id, work = read_run(rd)
    check_succeeded(rd)
    year, month = find_period(work)
    period_dirs = f"year={year}/month={month}"
    pyg, enriched = work / "pyg" / period_dirs, work / "enriched" / period_dirs
    if not (enriched / "triples" / "_SUCCESS").exists():
        raise Refused(f"{enriched / 'triples'} has no _SUCCESS")
    graphs = find_graphs(pyg)
    dataset, sources = dataset_name(enriched, f"{year}-{month}", env)
    day = data_day(work, env)
    root = env.get("PYG_PUBLISH_ROOT", "").strip().rstrip("/")
    if not root.startswith("s3://") or len(root) == len("s3://"):
        raise Refused("set PYG_PUBLISH_ROOT to s3://BUCKET/PREFIX")
    dst = f"{root}/{dataset}/{period_dirs}/{run_id}"

    log(f"run {run_id}, sources from {day}, dataset {dataset}, graphs: {', '.join(graphs)}")
    log(f"from {work}")
    log(f"to   {dst}/")

    log("checking the graph files against checksums.json")
    for graph in graphs:
        check_graph(pyg, graph, log)

    items = plan(work, period_dirs, graphs)
    tree = rd / TREE
    build_tree(tree, items, make_index(run_id, dataset, f"{year}-{month}", day, sources,
                                       rd, graphs, notebook_labels(rd), items))
    log(f"upload tree {tree}:")
    summarize(items, log)

    try:
        already = published(dst)
    except Failed as why:
        raise Refused(str(why))
    if INDEX in already:
        raise Refused(f"{dst}/{INDEX} already exists, so this run is already published")
    if already:
        log(f"{len(already)} files are already at the destination; "
            "those at the right size are skipped")

    sync = ["s3", "sync", str(tree), f"{dst}/", "--size-only", "--exclude", INDEX,
            "--no-progress"]
    if not upload:
        rc, lines = aws_streamed(log, False, *sync, "--dryrun")
        if rc != 0:
            raise Refused(f"aws s3 sync --dryrun failed (rc={rc}); see publish.log")
        count = sum(1 for line in lines if line.startswith("(dryrun) upload:"))
        log(f"dry run: {count} files would be uploaded, then index.json. Nothing was written.")
        log("Run again with --upload to publish.")
        return 0

    log("uploading")
    rc, _ = aws_streamed(log, True, *sync)
    if rc != 0:
        log(f"UPLOAD FAILED (rc={rc}). index.json was not written. Running again resumes.")
        return 1
    if not check_upload(items, tree, dst, log):
        return 1
    r = aws("s3", "cp", str(tree / INDEX), f"{dst}/{INDEX}", "--no-progress")
    if r.returncode != 0:
        log(f"index.json upload failed: {r.stderr.strip()}")
        return 1
    if published(dst).get(INDEX, (None,))[0] != (tree / INDEX).stat().st_size:
        log("index.json does not show up in the listing at the right size")
        return 1
    log(f"published {dst}/")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish a finished cluster run. Without --upload nothing is written.")
    parser.add_argument("run_dir")
    parser.add_argument("--upload", action="store_true", help="write to the destination")
    args = parser.parse_args(argv)

    rd = Path(args.run_dir).expanduser().resolve()
    if not rd.is_dir():
        print(f"no such run directory: {rd}", file=sys.stderr)
        return 2
    log = Log(rd / "publish.log")
    log(f"==== {'upload' if args.upload else 'dry run'} started {utc_now()} ====")
    try:
        rc = publish(rd, args.upload, os.environ, log)
    except Refused as why:
        log(f"NOT PUBLISHED: {why}")
        rc = 2
    except Failed as why:
        log(f"FAILED: {why}")
        rc = 1
    if args.upload:
        (rd / "publish.done").write_text(f"{rc}\n")
    log(f"==== finished rc={rc} {utc_now()} ====")
    return rc


if __name__ == "__main__":
    sys.exit(main())
