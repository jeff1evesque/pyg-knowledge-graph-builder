"""Pin the two seed flags in notebook/multi_experiment.ipynb.

`PYG_SEED_ONLY` stops after the seed; `PYG_SKIP_SEED` starts after it. They decide
which half of a multi-hour run happens, and a notebook is not covered by anything
else in this suite, so the failure mode is a scheduling experiment that quietly
re-derives its own input for a hundred minutes -- or worse, one that submits an
assembly leg over a Parquet that was never written.

The cells are extracted and executed against stubs rather than run for real. That
keeps the assertions on the branching, which is the part that decides how a run
spends its afternoon.
"""
import json
from pathlib import Path

import pytest

NOTEBOOK = (Path(__file__).resolve().parent.parent
            / "notebook" / "multi_experiment.ipynb")


def _cell(cell_id: str) -> str:
    """Source of one cell, by the id nbformat stores."""
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["id"] == cell_id:
            return "".join(cell["source"])
    raise AssertionError(f"no cell {cell_id} in {NOTEBOOK.name}")


@pytest.fixture(scope="module")
def flag_set():
    """The helper itself, lifted out of the function cell."""
    src = _cell("cell-06")
    start = src.index("def flag_set(")
    end = src.index("def period_partition(")
    ns: dict = {"os": __import__("os")}
    exec(src[start:end], ns)
    return ns["flag_set"]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "y"])
def test_flag_set_reads_anything_not_an_off_switch_as_on(flag_set, monkeypatch, value):
    monkeypatch.setenv("PYG_TEST_FLAG", value)
    assert flag_set("PYG_TEST_FLAG") is True


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "  ", " no "])
def test_flag_set_reads_the_off_spellings_as_off(flag_set, monkeypatch, value):
    monkeypatch.setenv("PYG_TEST_FLAG", value)
    assert flag_set("PYG_TEST_FLAG") is False


def test_flag_set_unset_is_off(flag_set, monkeypatch):
    monkeypatch.delenv("PYG_TEST_FLAG", raising=False)
    assert flag_set("PYG_TEST_FLAG") is False


def _run_seed_cell(tmp_path, monkeypatch, *, skip=None, seed_only=None,
                   success=True, enriched=None):
    """Execute the seed cell against stubs and hand back its namespace.

    `submitted` records whether the real submission would have gone out, which is
    the whole point of the flag.
    """
    for name, value in (("PYG_SKIP_SEED", skip), ("PYG_SEED_ONLY", seed_only)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    if enriched is None:
        enriched = tmp_path / "enriched"
        enriched.mkdir()
        (enriched / "part-00000.parquet").write_bytes(b"")
        if success:
            (enriched / "_SUCCESS").write_bytes(b"")
        enriched = str(enriched)

    submitted = []
    ns = {
        "os": __import__("os"),
        "Path": Path,
        "show": lambda *a, **k: None,
        "flag_set": (lambda name: __import__("os").environ.get(name, "")
                     .strip().lower() not in {"", "0", "false", "no"}),
        "enriched_parquet_path": lambda: enriched,
        "submit": lambda **kw: submitted.append(kw) or {"returncode": 0},
        "SEED_TIMEOUT_S": 1,
        "SEED_PROFILE": "profile.env",
        "PREFLIGHT_OK": True,
        "TIME_PERIOD": "2026-09",
    }
    error = None
    try:
        exec(_cell("cell-08"), ns)
    except RuntimeError as exc:      # the cell raises on a setup mistake
        error = exc
    ns["_submitted"] = submitted
    ns["_error"] = error
    return ns


def test_without_the_flag_the_seed_still_runs(tmp_path, monkeypatch):
    """The default path is untouched -- this flag is opt-in only."""
    ns = _run_seed_cell(tmp_path, monkeypatch)
    assert len(ns["_submitted"]) == 1
    assert ns["_submitted"][0]["mode"] == "enrichment_only"
    assert ns["ENRICHED_OK"] is True
    assert ns["_error"] is None


def test_skip_seed_submits_nothing_when_the_parquet_is_complete(tmp_path, monkeypatch):
    """The point of #373's A/B: both arms read bytes that already exist."""
    ns = _run_seed_cell(tmp_path, monkeypatch, skip="1")
    assert ns["_submitted"] == []
    assert ns["seed"] is None
    assert ns["ENRICHED_OK"] is True
    assert ns["_error"] is None


def test_skip_seed_refuses_a_directory_with_no_success_marker(tmp_path, monkeypatch):
    """Part files without _SUCCESS are a half-written Parquet, not an input.

    This is the case worth catching: it looks like data on an `ls`, and a
    pyg_only submission over it dies deep in the load phase with an error that
    reads like a code fault.
    """
    ns = _run_seed_cell(tmp_path, monkeypatch, skip="1", success=False)
    assert ns["_submitted"] == []
    assert ns["ENRICHED_OK"] is False
    assert isinstance(ns["_error"], RuntimeError)
    assert "no completed write" in str(ns["_error"])


def test_skip_seed_refuses_a_path_that_is_not_there(tmp_path, monkeypatch):
    ns = _run_seed_cell(tmp_path, monkeypatch, skip="1",
                        enriched=str(tmp_path / "absent"))
    assert ns["ENRICHED_OK"] is False
    assert isinstance(ns["_error"], RuntimeError)


def test_skip_seed_does_not_claim_to_have_checked_object_storage(tmp_path, monkeypatch):
    """A remote path cannot be listed from the kernel, so it is not verified.

    It still proceeds -- the flag was set deliberately -- but ENRICHED_OK being
    True here means "not checked", not "checked and present".
    """
    ns = _run_seed_cell(tmp_path, monkeypatch, skip="1",
                        enriched="s3a://example-bucket/run/enriched/triples")
    assert ns["_submitted"] == []
    assert ns["ENRICHED_OK"] is True
    assert ns["_error"] is None


def test_both_flags_together_is_refused(tmp_path, monkeypatch):
    """Skip the seed and stop before the legs and the run does nothing at all."""
    ns = _run_seed_cell(tmp_path, monkeypatch, skip="1", seed_only="1")
    assert ns["_submitted"] == []
    assert isinstance(ns["_error"], RuntimeError)
    assert "both set" in str(ns["_error"])


def test_the_legs_are_gated_on_enriched_ok():
    """A raise alone does not stop section 4 -- the runner sets allow_errors.

    So the gate has to be in the submit call. Pinned at the source level because
    executing that cell means standing up the whole experiment loop.
    """
    assert "dry_run=not (PREFLIGHT_OK and ENRICHED_OK)" in _cell("cell-11")


def test_seed_only_reads_through_the_shared_helper():
    """Both flags must read a value the same way, or '0' means opposite things."""
    src = _cell("cell-10")
    assert 'flag_set("PYG_SEED_ONLY")' in src
    assert "os.environ.get(\"PYG_SEED_ONLY\"" not in src
