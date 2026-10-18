"""Pin how `bin/stage_sources.sh` uses the local store of each day's source files: what
it copies from the store, what it downloads from S3, and what it saves. #464

WHY THIS IS WORTH A TEST
------------------------
Two mistakes here would pass unseen. A day the store holds that is downloaded anyway
costs money every night it happens, and nothing fails. A saved day that does not hold
what the run read would make a later rebuild read different files from the first build,
and nothing fails then either.

HOW
---
The script runs for real. aws, ssh, scp, hostname and ip are stub binaries. The fake S3
is a folder, as in tests/test_daily_run.py, and the store is a folder. The other node is
this machine with its own folder for the local copies: the ssh stub runs the command it
is given, with the local copy folder's path changed to that node's folder, so a copy
sent over ssh really lands there. No default route is reported, so the gateway watchdog
is off. Nothing here touches a bucket, a cluster or the network.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = REPO_ROOT / "bin" / "stage_sources.sh"

LOCAL = "node-a"     # the host that runs the script
REMOTE = "node-b"    # the other node, reached over ssh
DAY = "2026-10-16"

# One day's sources: a day file, a folder of the day's snapshots, and a yearly file
# that upstream overwrites with each new release.
DAY_FILE = "bucket/raw/source=a/year=2026/month=10/16.snappy.parquet"
SNAPSHOTS = "bucket/quotes/year=2026/month=10/day=16"
YEARLY = "bucket/raw/source=b/feed=x/2026.snappy.parquet"
SOURCES = [f"s3a://{DAY_FILE}", f"s3a://{SNAPSHOTS}/", f"s3a://{YEARLY}"]

pytestmark = pytest.mark.launcher


def day_files(yearly: bytes = b"release 1") -> dict:
    return {
        DAY_FILE: b"the day's filings",
        f"{SNAPSHOTS}/snapshot-01.parquet": b"quotes at 09:40",
        f"{SNAPSHOTS}/snapshot-02.parquet": b"quotes at 10:00",
        YEARLY: yearly,
    }


def listed(files: dict, uri: str) -> dict:
    """The file count and total size the store's list gives a source."""
    path = uri.split("://", 1)[1].rstrip("/")
    hits = [data for key, data in files.items() if key == path or key.startswith(path + "/")]
    return {"files": len(hits), "bytes": sum(len(data) for data in hits)}


STUB_AWS = '''#!/usr/bin/env python3
import os, shutil, sys
from pathlib import Path

args = sys.argv[1:]
node = os.environ.get("FAKE_NODE", "node-a")
with open(os.path.join(os.environ["FAKE_CALLS"], "aws.txt"), "a") as fh:
    fh.write(node + " " + " ".join(args) + "\\n")
s3 = Path(os.environ["FAKE_S3"])


def split(uri):
    bucket, _, key = uri.split("://", 1)[1].partition("/")
    return s3 / bucket, key


def found(bucket, prefix):
    if not bucket.is_dir():
        return []
    return sorted((p.relative_to(bucket).as_posix(), p) for p in bucket.rglob("*")
                  if p.is_file() and p.relative_to(bucket).as_posix().startswith(prefix))


if args[:2] == ["s3api", "head-object"]:
    path = s3 / args[args.index("--bucket") + 1] / args[args.index("--key") + 1]
    sys.exit(0 if path.is_file() else 255)
if args[:2] == ["s3", "ls"]:
    hits = found(*split(args[-1]))
    print(f"Total Objects: {len(hits)}")
    print(f"   Total Size: {sum(p.stat().st_size for _, p in hits)}")
    sys.exit(0 if hits else 1)
if args[:2] in (["s3", "cp"], ["s3", "sync"]):
    if os.environ.get("FAKE_AWS_FAIL") and os.environ["FAKE_AWS_FAIL"] in args[2]:
        sys.exit("stub aws: the download failed")
    bucket, key = split(args[2])
    if args[1] == "cp":
        shutil.copy2(bucket / key, args[3])
    else:
        prefix = key.rstrip("/") + "/"
        for name, path in found(bucket, prefix):
            dest = Path(args[3]) / name[len(prefix):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    sys.exit()
sys.exit(f"stub aws: no case for {args}")
'''

# The other node: this machine, with its local copies in a folder of its own.
STUB_SSH = '''#!/usr/bin/env bash
while [[ "$1" == -* ]]; do
  case "$1" in -o) shift 2 ;; *) shift ;; esac
done
node="$1"
shift
printf '%s %s\\n' "$node" "$*" >> "$FAKE_CALLS/ssh.txt"
cmd="$*"
FAKE_NODE="$node" exec bash -c "${cmd//"$FAKE_DEST"/"$FAKE_NODES/$node"}"
'''

STUB_SCP = '''#!/usr/bin/env bash
dest="${@: -1}"
cp "${@: -2:1}" "${dest#*:}"
'''

# No default route, so the gateway watchdog is off; no address names the other node.
STUB_IP = '''#!/usr/bin/env bash
[ "$1" = route ] || echo "1: lo    inet 127.0.0.1/8 scope host lo"
'''


def _script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


class Cluster:
    """Two nodes, a fake S3, a store folder and the stubs."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.mirror = tmp_path / "mirror"          # this host's copies, --dest
        self.nodes = tmp_path / "nodes"            # nodes/node-b holds the other node's
        self.s3 = tmp_path / "s3"
        self.store = tmp_path / "drive" / "sources"
        self.calls = tmp_path / "calls"
        self.calls.mkdir()
        (tmp_path / "tmp").mkdir()
        self.bin = tmp_path / "stubbin"
        _script(self.bin / "aws", STUB_AWS)
        _script(self.bin / "ssh", STUB_SSH)
        _script(self.bin / "scp", STUB_SCP)
        _script(self.bin / "ip", STUB_IP)
        _script(self.bin / "hostname", f"#!/usr/bin/env bash\necho {LOCAL}\n")

    def put(self, files: dict) -> None:
        """S3 holds these files."""
        for key, data in files.items():
            path = self.s3 / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def day_dir(self) -> Path:
        return self.store / f"day={DAY}"

    def save_day(self, files: dict, sources=SOURCES) -> None:
        """The store holds these files for the day, and its list names these sources."""
        for key, data in files.items():
            path = self.day_dir() / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            os.utime(path, (1_700_000_000, 1_700_000_000))
        record = {"day": DAY, "sources": {uri: listed(files, uri) for uri in sources}}
        (self.day_dir() / "_sources.json").write_text(json.dumps(record))

    def record(self) -> dict:
        return json.loads((self.day_dir() / "_sources.json").read_text())

    def copy_on(self, node: str) -> Path:
        """Where a node keeps its local copies."""
        return self.mirror if node == LOCAL else self.nodes / node

    def run(self, *args: str, nodes=(LOCAL, REMOTE), sources=SOURCES, **env_changes):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYG_", "AWS_"))}
        env.update({
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.tmp),
            "TMPDIR": str(self.tmp / "tmp"),
            "FAKE_S3": str(self.s3),
            "FAKE_CALLS": str(self.calls),
            "FAKE_DEST": str(self.mirror),
            "FAKE_NODES": str(self.nodes),
        })
        env.update(env_changes)
        return subprocess.run(
            ["bash", str(STAGE), "--dest", str(self.mirror), "--nodes", ",".join(nodes),
             "--sources", ",".join(sources), *args],
            env=env, capture_output=True, text=True, timeout=60)

    def recorded(self, name: str) -> str:
        path = self.calls / f"{name}.txt"
        return path.read_text() if path.exists() else ""

    def forget_calls(self) -> None:
        for path in self.calls.iterdir():
            path.unlink()

    def downloads(self) -> list:
        """(node, source path) for every file or folder a node downloaded."""
        out = []
        for line in self.recorded("aws").splitlines():
            node, *args = line.split()
            if args[:2] in (["s3", "cp"], ["s3", "sync"]):
                out.append((node, args[2].split("://", 1)[1].rstrip("/")))
        return out


@pytest.fixture
def cluster(tmp_path):
    return Cluster(tmp_path)


def holds(c: Cluster, node: str, files: dict) -> bool:
    copy = c.copy_on(node)
    return all((copy / key).is_file() and (copy / key).read_bytes() == data
               for key, data in files.items())


# --------------------------------------------------------------------------- #
# A day in the store
# --------------------------------------------------------------------------- #

def test_a_day_in_the_store_is_copied_to_every_node_and_s3_is_never_asked(cluster):
    """S3 has overwritten the yearly file since the day was saved. The day gets what it
    read the first time, and S3 is not even listed."""
    c = cluster
    c.put(day_files(yearly=b"release 2"))
    c.save_day(day_files(yearly=b"release 1"))

    r = c.run("--store", str(c.store), "--day", DAY)
    assert r.returncode == 0, r.stdout + r.stderr
    assert c.recorded("aws") == ""
    for node in (LOCAL, REMOTE):
        assert holds(c, node, day_files(yearly=b"release 1")), node
        # The saved files' times come along.
        assert int((c.copy_on(node) / YEARLY).stat().st_mtime) == 1_700_000_000
    assert "-xf -" in c.recorded("ssh")      # the other node took it from this host
    assert "all nodes match" in r.stdout
    assert f"store: {c.day_dir()} held 3 of the 3 sources" in r.stdout


def test_a_node_that_already_holds_the_saved_copy_gets_nothing(cluster):
    c = cluster
    c.save_day(day_files())
    assert c.run("--store", str(c.store), "--day", DAY).returncode == 0
    c.forget_calls()

    r = c.run("--store", str(c.store), "--day", DAY)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nothing to download -- every source was already on every node." in r.stdout
    assert c.recorded("aws") == ""
    assert "-xf -" not in c.recorded("ssh")


@pytest.mark.parametrize("damage, broken", [
    ("one missing", SNAPSHOTS),
    ("one too short", DAY_FILE),
    ("not in the list", YEARLY),
])
def test_a_source_whose_files_do_not_match_the_list_is_not_in_the_store(
        cluster, damage, broken):
    """The list is written after the files, so a source it does not name, or whose
    files no longer match it, may be incomplete. That source alone comes from S3."""
    c = cluster
    c.put(day_files(yearly=b"release 2"))
    if damage == "not in the list":
        c.save_day(day_files(yearly=b"release 1"), sources=SOURCES[:2])
    else:
        c.save_day(day_files(yearly=b"release 1"))
    if damage == "one missing":
        (c.day_dir() / SNAPSHOTS / "snapshot-02.parquet").unlink()
    elif damage == "one too short":
        (c.day_dir() / DAY_FILE).write_bytes(b"the day's")

    r = c.run("--store", str(c.store), "--day", DAY)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(c.downloads()) == [(LOCAL, broken), (REMOTE, broken)]
    assert not any(other in c.recorded("aws") for other in {DAY_FILE, SNAPSHOTS, YEARLY} - {broken})
    expected = {**day_files(yearly=b"release 1"),
                **{key: data for key, data in day_files(yearly=b"release 2").items()
                   if key.startswith(broken)}}
    for node in (LOCAL, REMOTE):
        assert holds(c, node, expected), node
    assert "held 2 of the 3 sources" in r.stdout


# --------------------------------------------------------------------------- #
# Saving what was downloaded
# --------------------------------------------------------------------------- #

def test_a_new_day_is_downloaded_once_saved_and_copied_to_the_other_node(cluster):
    """Named second, this host still goes first, so the other node copies from it
    instead of downloading the day again."""
    c = cluster
    c.put(day_files())
    c.store.mkdir(parents=True)

    r = c.run("--store", str(c.store), "--day", DAY, "--save", nodes=(REMOTE, LOCAL))
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(c.downloads()) == sorted((LOCAL, s.split("://")[1].rstrip("/"))
                                           for s in SOURCES)
    assert c.record() == {"day": DAY,
                          "sources": {uri: listed(day_files(), uri) for uri in SOURCES}}
    for key, data in day_files().items():
        assert (c.day_dir() / key).read_bytes() == data
    assert holds(c, LOCAL, day_files())
    assert holds(c, REMOTE, day_files())
    assert "-xf -" in c.recorded("ssh")
    assert r.stdout.index(f"== {LOCAL} ==") < r.stdout.index(f"== {REMOTE} ==")
    assert f"store: {c.day_dir()} held 0 of the 3 sources; saved 3" in r.stdout


def test_a_copy_this_host_already_holds_is_saved_without_a_download(cluster):
    """A day already staged before the store was set: it is saved from this host's
    copy, which matched S3's count and size, and nothing is downloaded."""
    c = cluster
    c.put(day_files())
    for key, data in day_files().items():
        for copy in (c.mirror, c.nodes / REMOTE):
            (copy / key).parent.mkdir(parents=True, exist_ok=True)
            (copy / key).write_bytes(data)
    c.store.mkdir(parents=True)

    r = c.run("--store", str(c.store), "--day", DAY, "--save")
    assert r.returncode == 0, r.stdout + r.stderr
    assert c.downloads() == []
    assert c.record()["sources"] == {uri: listed(day_files(), uri) for uri in SOURCES}


def test_without_save_nothing_is_saved_and_each_node_downloads_as_before(cluster):
    c = cluster
    c.put(day_files())
    c.store.mkdir(parents=True)

    r = c.run("--store", str(c.store), "--day", DAY)
    assert r.returncode == 0, r.stdout + r.stderr
    assert {node for node, _ in c.downloads()} == {LOCAL, REMOTE}
    assert len(c.downloads()) == 6
    assert list(c.store.iterdir()) == []
    assert holds(c, REMOTE, day_files())


def test_a_save_that_stops_partway_lists_only_what_was_saved_in_full(cluster):
    """Files first, then the list: the source whose download failed is not named, and
    the next staging finishes the day without downloading the saved source again."""
    c = cluster
    c.put(day_files())
    c.store.mkdir(parents=True)

    failed = c.run("--store", str(c.store), "--day", DAY, "--save", FAKE_AWS_FAIL=SNAPSHOTS)
    assert failed.returncode != 0
    assert list(c.record()["sources"]) == [f"s3a://{DAY_FILE}"]

    c.forget_calls()
    r = c.run("--store", str(c.store), "--day", DAY, "--save")
    assert r.returncode == 0, r.stdout + r.stderr
    assert DAY_FILE not in c.recorded("aws")
    assert sorted(c.record()["sources"]) == sorted(SOURCES)
    assert holds(c, REMOTE, day_files())


# --------------------------------------------------------------------------- #
# --no-store: a day upstream fixed after it was saved
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("save", [False, True], ids=["kept", "replaced"])
def test_no_store_downloads_even_when_the_store_has_the_day(cluster, save):
    c = cluster
    c.put(day_files(yearly=b"release 1, fixed upstream"))
    c.save_day(day_files(yearly=b"release 1"))

    r = c.run("--store", str(c.store), "--day", DAY, "--no-store", *(["--save"] if save else []))
    assert r.returncode == 0, r.stdout + r.stderr
    for node in (LOCAL, REMOTE):
        assert holds(c, node, day_files(yearly=b"release 1, fixed upstream")), node
    yearly = c.record()["sources"][f"s3a://{YEARLY}"]
    if save:
        assert {node for node, _ in c.downloads()} == {LOCAL}
        assert (c.day_dir() / YEARLY).read_bytes() == b"release 1, fixed upstream"
        assert yearly["bytes"] == len(b"release 1, fixed upstream")
    else:
        assert {node for node, _ in c.downloads()} == {LOCAL, REMOTE}
        assert (c.day_dir() / YEARLY).read_bytes() == b"release 1"
        assert yearly["bytes"] == len(b"release 1")


# --------------------------------------------------------------------------- #
# No store: S3, as before
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("missing", ["the store folder", "the day"])
def test_a_missing_store_or_no_day_falls_back_to_s3_with_a_note(cluster, missing):
    """A drive that is not mounted, or a staging that names no day: S3 as before, and
    nothing saved."""
    c = cluster
    c.put(day_files())
    if missing == "the day":
        c.store.mkdir(parents=True)
        args = ("--store", str(c.store), "--save")
        note = f"NOTE: no --day given, so the store at {c.store} is not used."
    else:
        args = ("--store", str(c.store), "--day", DAY, "--save")
        note = f"NOTE: the store {c.store} is not there. Is its drive mounted?"

    r = c.run(*args)
    assert r.returncode == 0, r.stdout + r.stderr
    assert note in r.stderr
    assert {node for node, _ in c.downloads()} == {LOCAL, REMOTE}
    assert not c.store.exists() or list(c.store.iterdir()) == []


def test_without_a_store_staging_is_what_it_was(cluster):
    c = cluster
    c.put(day_files())

    r = c.run(nodes=(REMOTE, LOCAL))
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(c.downloads()) == 6
    assert "NOTE" not in r.stderr
    assert not any(line.startswith("store:") for line in r.stdout.splitlines())
    assert "staged. 6 of 6 node/source pairs transferred." in r.stdout
    # The nodes keep the order they were given.
    assert r.stdout.index(f"== {REMOTE} ==") < r.stdout.index(f"== {LOCAL} ==")


def test_the_settings_stand_in_for_the_flags(cluster):
    c = cluster
    c.put(day_files())
    c.store.mkdir(parents=True)

    r = c.run("--day", DAY, PYG_SOURCE_STORE_ROOT=str(c.store), PYG_SOURCE_STORE_SAVE="true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(c.record()["sources"]) == sorted(SOURCES)

    c.forget_calls()
    r = c.run("--day", DAY, PYG_SOURCE_STORE_ROOT=str(c.store), PYG_SOURCE_STORE_SKIP="true",
              PYG_STAGE_FORCE="true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(c.downloads()) == 6


# --------------------------------------------------------------------------- #
# Refused before anything moves
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("args, env, said", [
    (("--store", "drive/sources", "--day", DAY), {}, "--store must be an absolute path"),
    (("--store", "/", "--day", DAY), {}, "--store must be an absolute path"),
    (("--day", "2026-02-30"), {}, "--day must be a real date written YYYY-MM-DD"),
    ((), {"PYG_SOURCE_STORE_SAVE": "yes"}, "PYG_SOURCE_STORE_SAVE must be true or false"),
], ids=["relative", "root", "not-a-day", "save-yes"])
def test_a_store_setting_that_cannot_work_is_refused(cluster, args, env, said):
    c = cluster
    c.put(day_files())

    r = c.run(*args, **env)
    assert r.returncode == 2
    assert said in r.stderr
    assert c.recorded("aws") == ""


def test_a_source_that_could_point_outside_its_folder_is_refused(cluster):
    """A copy from the store replaces a node's copy whole, so a '..' in the path would
    remove something outside the folder the copies live in."""
    c = cluster
    c.store.mkdir(parents=True)

    r = c.run("--store", str(c.store), "--day", DAY, nodes=(LOCAL,),
              sources=["s3a://bucket/raw/../../elsewhere"])
    assert r.returncode == 2
    assert "has an empty, '.' or '..' part" in r.stderr
    assert c.recorded("aws") == ""
