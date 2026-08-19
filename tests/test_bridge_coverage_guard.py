"""Tests for the e2e coverage guard itself.

The guard in tests/e2e/test_pipeline_smoke.py is the durable half of the
coverage-on-live-terms work: ``_assert_declared_bridges_resolve`` proves a
bridge is non-empty, and this proves it is not DEGENERATE -- a join that should
link every ticker-bearing issuer and links one of them.

A guard is only worth its line count if it fails when it should, and a guard
asserted only against healthy fixtures has never been shown to do that. These
drive it with hand-built enriched frames: one healthy, one thin, one empty. They
are pure pandas -- no Spark, no pipeline run -- so the proof stays in the fast
suite where it is actually run.

The e2e module is loaded by path because tests/ is not a package. Loading it
does not run anything: it is a module of function definitions plus a
``pytestmark``, and the marker only applies to tests collected FROM that file.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SMOKE_PATH = Path(__file__).resolve().parent / "e2e" / "test_pipeline_smoke.py"
_spec = importlib.util.spec_from_file_location("_e2e_smoke", _SMOKE_PATH)
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)

_ONT = "https://jefflevesque.com/ontology/"
_ID = "https://jefflevesque.com/id/"

_TICKER = f"{_ONT}sec-filings/hasIssuerTradingSymbol"
_SYMBOL = f"{_ONT}market-quotes/symbol"
_REFERS = f"{_ONT}bls/refersToCompany"

_ISSUERS = [f"{_ID}sec-filings/Issuer_000000000{n}" for n in range(1, 4)]
_SNAPSHOTS = [f"{_ID}market-quotes/snapshot/{s}" for s in ("AAPL", "MSFT")]


class _Config:
    """Only the attribute the guard reads."""

    def __init__(self, path):
        self.enriched_parquet_path = str(path)


def _write(tmp_path, rows):
    frame = pd.DataFrame(rows, columns=["subject", "predicate", "object"])
    target = tmp_path / "triples"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "part-0.parquet")
    return _Config(target)


def _healthy_rows():
    """Every ticker-bearing issuer and every symbol-bearing snapshot linked."""
    rows = []
    for i, issuer in enumerate(_ISSUERS):
        rows.append((issuer, _TICKER, f"TCK{i}"))
        rows.append((issuer, _REFERS, f"{_ONT}unified/Company_TCK{i}"))
    for snapshot in _SNAPSHOTS:
        symbol = snapshot.rsplit("/", 1)[-1]
        rows.append((snapshot, _SYMBOL, symbol))
        rows.append((snapshot, _REFERS, f"{_ONT}unified/Company_{symbol}"))
    return rows


# ======================================================================
# It passes what it should
# ======================================================================

def test_full_coverage_passes(tmp_path):
    _smoke._assert_declared_bridges_are_not_degenerate(_write(tmp_path, _healthy_rows()))


def test_missing_enriched_frame_is_skipped(tmp_path):
    """A split-mode run may not re-emit the enriched frame."""
    _smoke._assert_declared_bridges_are_not_degenerate(_Config(tmp_path / "absent"))


# ======================================================================
# It fails what it should -- the whole point
# ======================================================================

def test_a_bridge_resolving_into_one_edge_fails(tmp_path):
    """The defect this exists for.

    Three issuers state a ticker, one reaches a company. The relation IS
    present, so the required-relations guard is satisfied and every term
    involved is live, so no drift check sees anything. Only the ratio does.
    """
    rows = [r for r in _healthy_rows() if r[1] != _REFERS or r[0] != _ISSUERS[0]]
    rows = [
        r for r in rows
        if not (r[1] == _REFERS and r[0] in (_ISSUERS[1], _ISSUERS[2]))
    ]
    rows.append((_ISSUERS[0], _REFERS, f"{_ONT}unified/Company_TCK0"))

    with pytest.raises(AssertionError, match="under-cover their endpoints"):
        _smoke._assert_declared_bridges_are_not_degenerate(_write(tmp_path, rows))


def test_the_failure_names_the_ratio_and_the_side(tmp_path):
    """A guard that fires has to say what to go and look at."""
    rows = [r for r in _healthy_rows() if r[1] != _REFERS]
    rows += [
        (_ISSUERS[0], _REFERS, f"{_ONT}unified/Company_TCK0"),
        (_SNAPSHOTS[0], _REFERS, f"{_ONT}unified/Company_AAPL"),
        (_SNAPSHOTS[1], _REFERS, f"{_ONT}unified/Company_MSFT"),
    ]
    with pytest.raises(AssertionError) as caught:
        _smoke._assert_declared_bridges_are_not_degenerate(_write(tmp_path, rows))

    message = str(caught.value)
    assert "SEC issuer -> unified company" in message
    assert "1/3" in message and "33.3%" in message
    # The market side was complete and must not be reported as thin.
    assert "market snapshot -> unified company" not in message


def test_one_side_thin_fails_even_when_the_other_is_complete(tmp_path):
    """A bridge has two ends and either can rot alone.

    The market half once asked for a class neither market vocabulary declared
    while the SEC half worked, so the relation existed and half of it was dead.
    """
    rows = [r for r in _healthy_rows() if not (r[1] == _REFERS and r[0] in _SNAPSHOTS)]
    rows.append((_SNAPSHOTS[0], _REFERS, f"{_ONT}unified/Company_AAPL"))

    with pytest.raises(AssertionError, match="market snapshot"):
        _smoke._assert_declared_bridges_are_not_degenerate(_write(tmp_path, rows))


def test_a_rule_with_no_candidates_is_skipped_on_that_fixture(tmp_path):
    """Fixtures differ, and a rule cannot be verified where its input is absent.

    The N-Triples fixture is a loader test carrying a slice of each source; its
    SEC half states no ticker at all. Failing there would report a
    fixture-coverage fact as a pipeline defect.

    This is only safe because a rule that measures nothing EVERYWHERE is caught
    by test_every_rule_is_exercised_by_the_committed_fixtures below.
    """
    rows = [r for r in _healthy_rows() if r[0] not in _SNAPSHOTS]
    _smoke._assert_declared_bridges_are_not_degenerate(_write(tmp_path, rows))


def test_every_rule_is_exercised_by_the_committed_fixtures():
    """No coverage rule may be vacuous across the whole fixture set.

    A rule whose candidate predicate appears in NO fixture measures nothing
    wherever it runs, which is the precise shape of a check that looks like
    protection and is not. Skipping per-fixture is only defensible with this
    standing behind it.

    Reads the emitted-term inventory the drift checker already builds from the
    committed fixtures, rather than re-parsing them here -- same source of
    truth, so the two cannot disagree about what the fixtures contain.
    """
    spec = importlib.util.spec_from_file_location(
        "_cvd", Path(__file__).resolve().parents[1] / "bin" / "check_vocabulary_drift.py"
    )
    cvd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cvd)

    emitted = cvd.emitted_from_fixtures()
    for rule in _smoke.BRIDGE_COVERAGE:
        suffix = f"/{rule.candidates}"
        assert any(term.endswith(suffix) for term in emitted), (
            f"{rule.label}: no committed fixture emits {rule.candidates}, so "
            f"this rule can never measure anything. Give a fixture an example "
            f"or drop the rule."
        )


# ======================================================================
# The node-level half: one filer, one node
# ======================================================================

def _index(uris):
    return pd.DataFrame({"uri": uris})


def test_padded_and_unpadded_nodes_are_reported_as_one_entity_split():
    """The graph-level statement of the CIK defect."""
    with pytest.raises(AssertionError, match="differing only in CIK padding"):
        _smoke._assert_no_entity_is_split_by_identifier_padding(_index([
            f"{_ID}sec-filings/Issuer_0001729997",
            f"{_ID}sec-filings/Issuer_1729997",
        ]))


def test_distinct_filers_are_not_confused_for_a_split():
    """Two genuinely different CIKs must not collide."""
    _smoke._assert_no_entity_is_split_by_identifier_padding(_index([
        f"{_ID}sec-filings/Issuer_0001729997",
        f"{_ID}sec-filings/Issuer_0001729998",
        f"{_ID}sec-filings/ReportingOwner_0001729997",
    ]))


def test_positional_issuer_nodes_do_not_collide():
    """``0001-26-1_Issuer_0`` is an ordinal fallback, not a CIK.

    Two documents that each state no issuer both get ``_Issuer_0``. What keeps
    them apart is the anchor rather than their differing prefixes: the pattern
    requires the local name to BEGIN with the entity word, and these begin with
    an accession number, so neither is considered a CIK-keyed URI at all.
    """
    _smoke._assert_no_entity_is_split_by_identifier_padding(_index([
        f"{_ID}sec-filings/0001-26-1_Issuer_0",
        f"{_ID}sec-filings/0002-26-1_Issuer_0",
    ]))
