"""Unit tests for the preflight that refuses to overwrite a finished run.

Two runs given the same ``--local_work_dir`` and ``--time_period`` write
byte-identical paths, and the artifacts fail differently when that happens:
Spark overwrites the enriched Parquet, the ``.pt`` and metadata are replaced at
fixed keys, and manifests accumulate. Nothing raises, so the survivor is a
mixture of two runs with provenance describing both.

The case that must NOT trip is the experiment loop: several ``pyg_only`` legs
over one seed's enriched Parquet is the normal way this pipeline is used, and a
check that treated a mode's *input* as occupancy would break it.

Pure Python: runs under ``pytest -m "not e2e"``. Local paths need no
SparkSession, so ``spark`` is passed as None throughout.
"""
from pathlib import Path

import pytest

from spark_jobs.build_graph import JobConfig, check_work_dir_occupancy


def _config(work_dir, **overrides):
    args = {
        "mode": "enrichment_only",
        "source_paths": "/data/raw/sec",
        "source_format": "turtle_parquet",
        "local_work_dir": str(work_dir),
        "time_period": "2024-12",
    }
    args.update(overrides)
    return JobConfig(args)


def _finish_enriched(config):
    """Mark the enriched Parquet complete the way Spark does — _SUCCESS last."""
    path = Path(config.enriched_parquet_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "part-00000.parquet").write_bytes(b"")
    (path / "_SUCCESS").write_bytes(b"")


def _write_pt(config):
    path = Path(config.pyg_output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# ======================================================================
# an empty work dir is always fine
# ======================================================================

@pytest.mark.parametrize("mode", ["full", "enrichment_only", "pyg_only"])
def test_empty_work_dir_passes(tmp_path, mode):
    check_work_dir_occupancy(_config(tmp_path, mode=mode), None)


# ======================================================================
# what each mode writes is what it guards
# ======================================================================

@pytest.mark.parametrize("mode", ["full", "enrichment_only"])
def test_finished_enriched_blocks_the_modes_that_write_it(tmp_path, mode):
    config = _config(tmp_path, mode=mode)
    _finish_enriched(config)
    with pytest.raises(FileExistsError, match="enriched Parquet"):
        check_work_dir_occupancy(config, None)


@pytest.mark.parametrize("mode", ["full", "pyg_only"])
def test_existing_pt_blocks_the_modes_that_write_it(tmp_path, mode):
    config = _config(tmp_path, mode=mode)
    _write_pt(config)
    with pytest.raises(FileExistsError, match="PyG graph"):
        check_work_dir_occupancy(config, None)


def test_pyg_only_does_not_treat_its_input_as_occupancy(tmp_path):
    """The experiment loop: N pyg_only legs over one seed must keep running.

    Each leg reads the same finished enriched Parquet and writes its own
    --pyg_filename. Guarding a mode's input would refuse every leg after the
    seed and make the cheap loop impossible.
    """
    config = _config(tmp_path, mode="pyg_only", pyg_filename="variant_a.pt")
    _finish_enriched(config)
    check_work_dir_occupancy(config, None)


def test_enrichment_only_ignores_a_pt_it_will_not_write(tmp_path):
    config = _config(tmp_path, mode="enrichment_only")
    _write_pt(config)
    check_work_dir_occupancy(config, None)


def test_distinct_pyg_filenames_do_not_collide(tmp_path):
    """Sibling legs differ only by --pyg_filename, which is the whole point."""
    first = _config(tmp_path, mode="pyg_only", pyg_filename="variant_a.pt")
    _write_pt(first)

    second = _config(tmp_path, mode="pyg_only", pyg_filename="variant_b.pt")
    check_work_dir_occupancy(second, None)


def test_a_different_period_is_a_different_partition(tmp_path):
    config = _config(tmp_path, mode="enrichment_only")
    _finish_enriched(config)

    later = _config(tmp_path, mode="enrichment_only", time_period="2025-01")
    check_work_dir_occupancy(later, None)


# ======================================================================
# _SUCCESS is the signal, not the directory
# ======================================================================

def test_enriched_debris_without_success_does_not_block(tmp_path):
    """A directory with no _SUCCESS is a run that died mid-write.

    Spark writes the marker last, so its absence means nothing finished there.
    Refusing to overwrite debris would strand the work dir after every crash.
    """
    config = _config(tmp_path, mode="enrichment_only")
    path = Path(config.enriched_parquet_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "part-00000.parquet").write_bytes(b"")

    check_work_dir_occupancy(config, None)


# ======================================================================
# the override, and what the refusal says
# ======================================================================

def test_allow_overwrite_lets_it_through(tmp_path):
    config = _config(tmp_path, mode="full", allow_overwrite="true")
    _finish_enriched(config)
    _write_pt(config)
    check_work_dir_occupancy(config, None)


def test_allow_overwrite_defaults_to_off(tmp_path):
    assert _config(tmp_path).allow_overwrite is False


def test_full_mode_reports_both_artifacts(tmp_path):
    config = _config(tmp_path, mode="full")
    _finish_enriched(config)
    _write_pt(config)
    with pytest.raises(FileExistsError) as excinfo:
        check_work_dir_occupancy(config, None)
    message = str(excinfo.value)
    assert "enriched Parquet" in message
    assert "PyG graph" in message


def test_the_refusal_names_the_path_and_the_way_out(tmp_path):
    """An operator reading only this line should know what to do next."""
    config = _config(tmp_path, mode="enrichment_only")
    _finish_enriched(config)
    with pytest.raises(FileExistsError) as excinfo:
        check_work_dir_occupancy(config, None)
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "2024-12" in message
    assert "--allow_overwrite" in message
