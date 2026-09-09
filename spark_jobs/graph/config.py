"""
What the job was asked to do: the CLI, and the JobConfig that validates it.

``JobConfig`` is the whole of the job's contract with its caller. It resolves
every path the run will read and write, and it REJECTS a configuration that
cannot work -- a mode without its inputs, a staged mirror that is not there, an
SEC prefix naming a feed this pipeline does not handle -- before Spark starts,
because the alternative is a failure an hour in with a message about a missing
column.

Pure Python. No SparkSession, no Spark types: this has to be constructible in a
test without a cluster, which is how most of its validation is covered.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from spark_jobs.pyg_builder.metadata_writer import LATEST_PARTITION

# The job's logger, not this module's. bin/profiles/extra-checks.example.sh
# greps run logs for `[INFO] build_graph - PHASE:`, and every line this package
# emits has to keep reading as one job rather than as whichever file it came
# from.
logger = logging.getLogger("build_graph")

# Probed at import, and read by JobConfig._validate so that a mode needing PyG
# fails while the caller is still at a prompt. run_pyg_construction in
# build_graph.py reads the same flag rather than probing again -- two answers to
# "is PyG installed" is one more than the question has.
PYG_BUILDER_AVAILABLE = False
try:
    # Imported for the answer, not for the name -- an import is the only probe
    # that catches a constructor whose own imports (torch, torch_geometric) are
    # what is missing.
    import spark_jobs.pyg_builder.constructor  # noqa: F401

    PYG_BUILDER_AVAILABLE = True
except ImportError:
    pass


# ============================================
# Time period
# ============================================
_PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})$")


def period_partition(time_period: str) -> str:
    """Render a ``YYYY-MM`` period label as Hive-style partition directories.

        "2024-12" -> "year=2024/month=12"

    Spark's partition discovery turns ``key=value`` directory names into real
    columns, so reading a parent directory across several periods yields
    ``year``/``month`` to filter and prune on. An opaque ``2024-12`` directory
    gives it nothing -- the period is only recoverable by parsing the path.

    ``time_period`` is a free-form label (nothing validates its shape, and
    ``--time_period`` accepts any string), so a value that is not ``YYYY-MM``
    is passed through as a single segment rather than forced into a partition
    shape that would misdescribe what it holds. Only ``day=`` is deliberately
    absent: ``time_period`` is monthly, so a day level would carry one value
    per month -- path depth with no pruning benefit.
    """
    match = _PERIOD_RE.match(time_period)
    if not match:
        logger.warning(
            f"time_period '{time_period}' is not YYYY-MM; writing it as a "
            f"single path segment with no year=/month= partitioning"
        )
        return time_period

    year, month = match.groups()
    return f"year={year}/month={month}"


# ============================================
# Accepted values
# ============================================
# parse_only is a diagnostic and not part of any pipeline: it stops at the count
# that materialises the parse and writes nothing. It is listed here because that
# is what makes it reachable from --mode, which is the point -- the stall it
# reproduces only happens on a real submission. See execute_parse_only.
VALID_MODES = {"full", "enrichment_only", "pyg_only", "parse_only"}
VALID_SOURCE_FORMATS = {"ntriples", "turtle_parquet"}

# Where the parse reads its input from.
#
#   "s3"    -- open every s3a:// source path directly. Right in the cloud, where
#              the executors sit beside the bucket on an in-region link.
#   "local" -- read a mirror of those same objects from node-local disk, staged
#              once by bin/stage_sources.sh. Right on-prem, where the site
#              uplink is the scarce resource and hundreds of readers can take
#              the network down (see issue #342).
#
# s3 stays the default so nothing changes for a deployment that has not staged.
VALID_INPUT_MODES = {"s3", "local"}

# Default number of Parquet output partitions.
# Targets ~128 MB per partition for a 322.7M-triple four-source day (~25 GB).
DEFAULT_PARQUET_PARTITIONS = 200

# ============================================
# Source coverage: which SEC feed this job handles
# ============================================
# The archive holds eight SEC feeds under raw/source=sec/. This pipeline
# handles exactly one of them, and the restriction has until now been
# incidental — a property of whichever prefix the caller happened to pass —
# rather than stated anywhere. Measured against the archive:
#
#   feed=filings             218 objects,   2.4 GB   RDF (rdf_turtle column)
#   feed=filings_documents   712,351 objects, 150 GB  raw filing documents
#   feed=filing-detail       364 objects            crawler telemetry columns
#   feed=litigation            1 object              only, no Turtle column
#   feed=press-release         2 objects             (parsed / parse-start /
#   feed=speeches              2 objects              failures / user-agent)
#   feed=statements            2 objects
#   feed=testimony             1 object
#
# filings_documents is not Parquet at all and not one format either: sampled
# over 400,000 keys it is 233,759 .xml, 160,196 .zip (upstream now packages
# each filing's documents together with its XBRL members), plus .txt, .htm,
# .pdf and images. Nothing there is RDF.
#
# So a run pointed at the SEC source root does not under-cover quietly — the
# six telemetry feeds have no Turtle column at all and resolve_turtle_column
# raises, while filings_documents is not something the Parquet reader can open.
# It fails, but it fails deep in the loader with a column-name error that says
# nothing about feeds. This turns that into a statement of scope at the point
# the job is configured.
#
# The one thing NOT guarded here, because storage says it is already resolved:
# the retired crawler also wrote into feed=filings itself, on a 10-column
# schema whose RDF column was named `triples` rather than `rdf_turtle`. All 218
# objects now carry the same 29-column scraper schema, so the migration ran to
# completion and no mixed-schema read is possible. Worth re-checking if that
# object count ever jumps backwards.
#
# Keyed on the partition name rather than on "sec" anywhere in the path: a
# local fixture directory called /data/sec/ is not the archive convention and
# is none of this check's business.
SEC_SOURCE_PARTITION = "source=sec"
SEC_HANDLED_FEED = "feed=filings"
# filings_documents starts with the handled feed's name, so a plain substring
# test would accept it. It is a different feed and carries no RDF.
SEC_UNHANDLED_FEEDS = (
    "feed=filings_documents", "feed=filing-detail", "feed=litigation",
    "feed=press-release", "feed=speeches", "feed=statements",
    "feed=testimony",
)


def assert_sec_paths_name_the_handled_feed(source_paths: List[str]) -> None:
    """Every SEC source path must name feed=filings explicitly.

    Raises:
        ValueError: if a path under the SEC source partition names a feed this
            pipeline does not handle, or names no feed at all.
    """
    for path in source_paths:
        if SEC_SOURCE_PARTITION not in path:
            continue
        unhandled = [f for f in SEC_UNHANDLED_FEEDS if f in path]
        if unhandled:
            raise ValueError(
                f"source path names an unhandled SEC feed {unhandled[0]!r}: "
                f"{path}. Only {SEC_HANDLED_FEED!r} carries RDF; the others "
                f"hold crawler telemetry or raw XML and no pipeline step reads "
                f"them."
            )
        if SEC_HANDLED_FEED not in path:
            raise ValueError(
                f"SEC source path does not name a feed: {path}. This pipeline "
                f"handles {SEC_HANDLED_FEED!r} only, and the SEC source "
                f"partition holds seven other feeds it cannot read. Point at "
                f"raw/{SEC_SOURCE_PARTITION}/{SEC_HANDLED_FEED}/... instead."
            )


# ============================================
# Parameter parsing
# ============================================
DEFAULT_PYG_FILENAME = "hetero_data.pt"


def staged_local_path(source_path: str, local_source_root: str) -> str:
    """Map one object-storage source path onto its staged local mirror.

    ``s3a://bucket/key/...`` becomes ``file://<root>/bucket/key/...``. Carrying
    the bucket and the whole key under the root is what makes the two input
    modes interchangeable: source_label() reads the source out of path
    fragments (``source=sec``, ``quotes``), so a mirror that keeps the key
    layout reports the same per-source statistics as reading the bucket
    directly. It also keeps two buckets that share a key prefix apart.

    A path that is already local is returned unchanged, so one run can mix a
    staged bucket with a directory that was never in object storage.

    The result names its scheme on purpose. A bare path is resolved against
    fs.defaultFS, so a site that points that at object storage would send a
    read the operator asked to keep local straight back over the wire.
    """
    remainder = None
    for scheme in ("s3a://", "s3n://", "s3://"):
        if source_path.startswith(scheme):
            remainder = source_path[len(scheme):]
            break
    if remainder is None:
        return source_path

    root = local_source_root.rstrip("/")
    if not root.startswith("/"):
        raise ValueError(
            f"local_source_root must be an absolute path, got {root!r}. "
            f"Every worker opens this path itself, so a relative one would "
            f"resolve against whatever directory each executor happens to be "
            f"started in."
        )
    return f"file://{root}/{remainder}"


class JobConfig:
    """Parsed and validated job configuration."""

    def __init__(self, args: Dict[str, str]):
        self.mode = args.get("mode", "full").lower().strip()

        # Shared local working directory (must be on a filesystem visible
        # to every Spark worker, e.g. an NFS mount). Holds the interim
        # enriched Parquet and the local copy of the final artifacts.
        self.local_work_dir = (args.get("local_work_dir") or "").rstrip("/")

        # Optional S3 archive for the FINAL artifacts (.pt + metadata).
        # When s3_archive_bucket is set, the final artifacts are mirrored
        # to S3 via boto3 in addition to being written locally.
        self.s3_archive_bucket = args.get("s3_archive_bucket", "") or ""
        self.s3_pyg_key = args.get("s3_pyg_key", "") or ""

        # Optional external definitions (small CSV read from S3 via boto3)
        self.market_sector_definitions_bucket = args.get(
            "market_sector_definitions_bucket", ""
        )
        self.market_sector_definitions_key = args.get(
            "market_sector_definitions_key", ""
        )

        # Source path(s)/URI(s): a single value or a comma-separated list.
        # Each entry is a local directory or an s3a:// URI. Whitespace
        # around each entry is stripped. Required for full/enrichment_only.
        # Examples:
        #   local:    "/data/rdf/monthly/2024-12/"
        #   s3a:      "s3a://my-data-lake/rdf/monthly/2024-12/"
        #   multiple: "/data/sec/filings/,/data/sec/litigations/"
        raw_sources = args.get("source_paths", "") or ""
        self.source_paths: List[str] = [
            p.strip() for p in raw_sources.split(",") if p.strip()
        ]

        # Where those sources are actually opened from. See VALID_INPUT_MODES.
        # source_paths keeps naming the object-storage locations either way, so
        # the manifest records what the run was asked for and read_paths records
        # what it opened -- the two differ only by the staging root.
        self.input_mode = (args.get("input_mode") or "s3").lower().strip()
        self.local_source_root = (
            args.get("local_source_root") or ""
        ).rstrip("/")
        if self.input_mode == "local":
            if not self.local_source_root:
                raise ValueError(
                    "local_source_root is required when input_mode is "
                    "'local'. It is the root bin/stage_sources.sh mirrored "
                    "the buckets into, and every worker must carry the same "
                    "one at the same path."
                )
            self.read_paths: List[str] = [
                staged_local_path(p, self.local_source_root)
                for p in self.source_paths
            ]
        else:
            self.read_paths = list(self.source_paths)

        # Source format:
        #   "ntriples":       one triple per line in .nt files
        #   "turtle_parquet": self-contained Turtle blobs in a Parquet column
        self.source_format = (
            args.get("source_format", "ntriples").lower().strip()
        )

        # Column name containing Turtle strings when
        # source_format=turtle_parquet. Empty means auto-detect per source
        # (see resolve_turtle_column), so one run can span sources whose
        # scrapers named the column differently; set it to force one name.
        self.turtle_column = args.get("turtle_column", "")

        # Optional
        self.enable_ontology_mapping = (
            args.get("enable_ontology_mapping", "true").lower() == "true"
        )

        # Off by default: overwriting a finished run is a deliberate act, and
        # the artifacts it destroys cost hours to produce.
        self.allow_overwrite = (
            args.get("allow_overwrite", "false") or "false"
        ).lower() == "true"
        self.time_period = args.get("time_period") or datetime.now().strftime(
            "%Y-%m"
        )
        # A name for this combination of sources, e.g. "all-sources" or
        # "no-market". Cannot be derived -- which sources belong together under
        # one name is a judgement, not a fact about the paths -- so it is stated
        # or it is empty. Recorded in the dataset descriptor and from there in
        # the published graph schema.
        self.dataset = args.get("dataset") or ""
        self.pyg_config = self._parse_pyg_config(args.get("pyg_config", ""))
        self.class_mappings = self._parse_json_arg(
            args.get("class_mappings", ""), "class_mappings"
        )
        self.parquet_partitions = int(
            args.get("parquet_partitions", str(DEFAULT_PARQUET_PARTITIONS))
        )

        # Local filename for the .pt (override for experiment variants,
        # e.g. "hetero_data_512d.pt"). Metadata directory is derived from it.
        self.pyg_filename = (
            args.get("pyg_filename") or DEFAULT_PYG_FILENAME
        )

        # Hive-style partition directories for the period, so a read across
        # several periods gets year/month as real columns to prune on.
        #
        # Everything a period produces stays under one partition directory,
        # so a build can be copied, archived or deleted as a unit.
        self.period_partition = period_partition(self.time_period)

        # Derived local paths (under local_work_dir)
        self.enriched_parquet_path = (
            f"{self.local_work_dir}/enriched/"
            f"{self.period_partition}/triples"
        )
        # Where enriched triples are READ from, which is not always where they
        # were written. Deriving both ends from local_work_dir meant a pyg_only
        # run could only read enriched output by also writing its graph beside
        # it -- so reusing a published run's output required copying it into a
        # scratch directory first, purely to give the job somewhere safe to
        # write. An explicit input path separates the two: read wherever the
        # data actually is, write wherever this run should.
        #
        # Explicit means explicit -- the path is used verbatim, with no period
        # partition appended, because data being pointed at may not follow this
        # pipeline's directory convention.
        self.enriched_input_path = (
            args.get("enriched_input_path") or self.enriched_parquet_path
        )
        self.pyg_output_path = (
            f"{self.local_work_dir}/pyg/"
            f"{self.period_partition}/{self.pyg_filename}"
        )

        # Fixed-key alias for the newest build, at the same depth as the period
        # copy with only the partition segment replaced -- so a consumer's URL
        # differs from a period URL by exactly that segment, and
        # derive_metadata_prefix() separates experiment variants here for the
        # same reason it does per period.
        self.latest_pyg_path = (
            f"{self.local_work_dir}/pyg/"
            f"{LATEST_PARTITION}/{self.pyg_filename}"
        )

        # Derived S3 key for the archived .pt (only used when archiving)
        if self.s3_archive_bucket and not self.s3_pyg_key:
            self.s3_pyg_key = (
                f"pyg/{self.period_partition}/{self.pyg_filename}"
            )

        self._validate()

    @property
    def archive_to_s3(self) -> bool:
        """Whether final artifacts should also be mirrored to S3."""
        return bool(self.s3_archive_bucket)

    def _parse_json_arg(self, raw: str, name: str) -> Dict[str, Any]:
        """Parse a JSON CLI argument, or a path to a JSON file."""
        raw = (raw or "").strip()
        if not raw:
            return {}
        if not raw.startswith("{"):
            try:
                raw = Path(raw).read_text()
            except OSError as e:
                logger.warning(f"Could not read {name} file: {e}")
                return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse {name} JSON, ignoring: {e}")
            return {}

    def _parse_pyg_config(self, config_str: str) -> Dict[str, Any]:
        if not config_str or config_str.strip() == "":
            return {}
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Could not parse pyg_config JSON, using defaults: {e}"
            )
            return {}

    def _assert_staged_mirror_present(self) -> None:
        """Fail before the job starts when the mirror is missing or empty.

        Checked on the driver, which covers only the driver's own host. The
        other workers are the staging script's job -- it syncs each node and
        compares a content digest across them before reporting success. What
        this catches is the cheap half: a root that was never staged, or staged
        for a different day, which would otherwise surface as an empty parse
        or a FileNotFoundError several minutes in.
        """
        missing = []
        for declared, resolved in zip(self.source_paths, self.read_paths):
            if not resolved.startswith("file://"):
                continue
            local = Path(resolved[len("file://"):])
            if not local.exists():
                missing.append((declared, local))
            elif local.is_dir() and not any(local.iterdir()):
                missing.append((declared, local))

        if missing:
            listed = "\n".join(f"  {d}\n    -> {p}" for d, p in missing)
            raise FileNotFoundError(
                f"input_mode is 'local' but the staged mirror is missing or "
                f"empty for {len(missing)} source path(s):\n{listed}\n"
                f"Stage them first:\n"
                f"  bin/stage_sources.sh --dest {self.local_source_root} "
                f"--nodes <every worker> --source <each source path>"
            )

    def _validate(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                f"Must be one of: {', '.join(VALID_MODES)}"
            )
        if not self.local_work_dir:
            raise ValueError("local_work_dir is required")

        # Warn rather than fail: only pyg_only reads enriched output from disk,
        # so the flag has no effect in the other modes. Silently ignoring it
        # would let someone believe a run read from somewhere it never touched.
        if (
            self.enriched_input_path != self.enriched_parquet_path
            and self.mode != "pyg_only"
        ):
            logger.warning(
                f"--enriched_input_path is set but mode is {self.mode}, which "
                f"does not read enriched output from disk. It will be ignored."
            )

        if self.source_format not in VALID_SOURCE_FORMATS:
            raise ValueError(
                f"Invalid source_format '{self.source_format}'. "
                f"Must be one of: {', '.join(VALID_SOURCE_FORMATS)}"
            )

        if self.input_mode not in VALID_INPUT_MODES:
            raise ValueError(
                f"Invalid input_mode '{self.input_mode}'. "
                f"Must be one of: {', '.join(sorted(VALID_INPUT_MODES))}"
            )

        # parse_only reads the same sources through the same loader, so it wants
        # every check the modes that parse for real get -- a trial that skipped
        # them would not be reproducing the submission it claims to.
        if self.mode in ("full", "enrichment_only", "parse_only"):
            if not self.source_paths:
                raise ValueError(
                    f"source_paths is required for mode '{self.mode}'"
                )
            assert_sec_paths_name_the_handled_feed(self.source_paths)
            if self.input_mode == "local":
                self._assert_staged_mirror_present()
        if self.mode in ("full", "pyg_only"):
            if not PYG_BUILDER_AVAILABLE:
                raise ImportError(
                    f"PyG builder not available. Mode '{self.mode}' requires "
                    "spark_jobs.pyg_builder.constructor."
                )
        if self.parquet_partitions < 1:
            raise ValueError("parquet_partitions must be >= 1")

    def __repr__(self):
        return (
            f"JobConfig(mode={self.mode}, "
            f"source_paths={self.source_paths}, "
            f"input_mode={self.input_mode}, "
            f"local_source_root={self.local_source_root or '(none)'}, "
            f"source_format={self.source_format}, "
            f"turtle_column={self.turtle_column}, "
            f"local_work_dir={self.local_work_dir}, "
            f"enriched_parquet_path={self.enriched_parquet_path}, "
            + (
                f"enriched_input_path={self.enriched_input_path}, "
                if self.enriched_input_path != self.enriched_parquet_path
                else ""
            ) +
            f"pyg_output_path={self.pyg_output_path}, "
            f"s3_archive_bucket={self.s3_archive_bucket or '(none)'}, "
            f"s3_pyg_key={self.s3_pyg_key or '(none)'}, "
            f"ontology_mapping={self.enable_ontology_mapping}, "
            + ("allow_overwrite=True, " if self.allow_overwrite else "")
            + f"parquet_partitions={self.parquet_partitions})"
        )


def parse_args() -> JobConfig:
    """Parse job arguments from the command line (spark-submit)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Geometric Knowledge Graph Builder"
    )
    parser.add_argument("--mode", choices=list(VALID_MODES), default="full")
    parser.add_argument(
        "--source_paths",
        default="",
        help="Comma-separated source path(s)/URI(s): local dirs or s3a://...",
    )
    parser.add_argument(
        "--input_mode",
        choices=sorted(VALID_INPUT_MODES),
        default="s3",
        help="Where source_paths are opened from: 's3' reads the s3a:// URIs "
        "directly (default); 'local' reads the mirror bin/stage_sources.sh "
        "put under --local_source_root",
    )
    parser.add_argument(
        "--local_source_root",
        default="",
        help="Root of the staged mirror, identical on every worker. "
        "Required when --input_mode local",
    )
    parser.add_argument(
        "--local_work_dir",
        required=True,
        help="Shared working directory (visible to all workers) for enriched "
        "Parquet and local final artifacts",
    )
    parser.add_argument(
        "--s3_archive_bucket",
        default="",
        help="Optional S3 bucket to mirror final artifacts (.pt + metadata)",
    )
    parser.add_argument(
        "--s3_pyg_key",
        default="",
        help="Optional S3 key for the archived .pt (metadata prefix derived "
        "from it); defaults to pyg/year=YYYY/month=MM/<pyg_filename>",
    )
    parser.add_argument("--pyg_filename", default=DEFAULT_PYG_FILENAME)
    parser.add_argument("--enable_ontology_mapping", default="true")
    parser.add_argument("--allow_overwrite", default="false")
    parser.add_argument("--time_period", default="")
    # Names the combination of sources, e.g. "all-sources" or "no-market".
    # Recorded beside the enriched output and carried into the graph schema, so
    # a published graph can say what it was built from.
    parser.add_argument("--dataset", default="")
    # Read enriched triples from here instead of deriving the location from
    # --local_work_dir. Used verbatim. Only --mode pyg_only reads enriched
    # output from disk, so it is ignored elsewhere.
    parser.add_argument("--enriched_input_path", default="")
    parser.add_argument("--pyg_config", default="")
    parser.add_argument("--class_mappings", default="")
    parser.add_argument(
        "--parquet_partitions", default=str(DEFAULT_PARQUET_PARTITIONS)
    )
    parser.add_argument(
        "--source_format",
        choices=list(VALID_SOURCE_FORMATS),
        default="ntriples",
    )
    parser.add_argument("--turtle_column", default="")
    parser.add_argument("--market_sector_definitions_bucket", default="")
    parser.add_argument("--market_sector_definitions_key", default="")

    parsed = parser.parse_args()
    return JobConfig(vars(parsed))
