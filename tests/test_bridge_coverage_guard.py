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
_CIK = f"{_ONT}sec-filings/hasIssuerCik"
_SYMBOL = f"{_ONT}market-quotes/symbol"
_REFERS = f"{_ONT}bls/refersToCompany"

_CIKS = [f"000000000{n}" for n in range(1, 4)]
_ISSUERS = [f"{_ID}sec-filings/Issuer_{cik}" for cik in _CIKS]
_TICKERS = [f"TCK{n}" for n in range(len(_CIKS))]

# The snapshots are named for the tickers they quote, and those tickers are
# stated by the issuers above. That overlap is now load-bearing: the market
# coverage rule narrows its candidates to snapshots whose symbol an ingested
# filing also states, because the bridge resolves a ticker to a CIK and a
# snapshot for a company with no filings here has nothing to resolve against.
_SNAPSHOTS = [f"{_ID}market-quotes/snapshot/{t}" for t in _TICKERS[:2]]


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
    """Every CIK-bearing issuer and every resolvable snapshot linked.

    The SEC candidate population is issuers stating a CIK, not a ticker -- the
    bridge was re-keyed onto the CIK precisely because the ticker capped it at
    the ownership-form share of filings.
    """
    rows = []
    for issuer, cik, ticker in zip(_ISSUERS, _CIKS, _TICKERS):
        rows.append((issuer, _CIK, cik))
        rows.append((issuer, _TICKER, ticker))
        rows.append((issuer, _REFERS, f"{_ONT}unified/Company_{cik}"))
    for snapshot, cik in zip(_SNAPSHOTS, _CIKS):
        symbol = snapshot.rsplit("/", 1)[-1]
        rows.append((snapshot, _SYMBOL, symbol))
        rows.append((snapshot, _REFERS, f"{_ONT}unified/Company_{cik}"))
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
    rows.append((_ISSUERS[0], _REFERS, f"{_ONT}unified/Company_{_CIKS[0]}"))

    with pytest.raises(AssertionError, match="under-cover their endpoints"):
        _smoke._assert_declared_bridges_are_not_degenerate(_write(tmp_path, rows))


def test_the_failure_names_the_ratio_and_the_side(tmp_path):
    """A guard that fires has to say what to go and look at."""
    rows = [r for r in _healthy_rows() if r[1] != _REFERS]
    rows += [
        (_ISSUERS[0], _REFERS, f"{_ONT}unified/Company_{_CIKS[0]}"),
        (_SNAPSHOTS[0], _REFERS, f"{_ONT}unified/Company_{_CIKS[0]}"),
        (_SNAPSHOTS[1], _REFERS, f"{_ONT}unified/Company_{_CIKS[1]}"),
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
    rows.append((_SNAPSHOTS[0], _REFERS, f"{_ONT}unified/Company_{_CIKS[0]}"))

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

    def is_emitted(local_name):
        suffix = f"/{local_name}"
        return any(term.endswith(suffix) for term in emitted)

    for rule in _smoke.BRIDGE_COVERAGE:
        assert is_emitted(rule.candidates), (
            f"{rule.label}: no committed fixture emits {rule.candidates}, so "
            f"this rule can never measure anything. Give a fixture an example "
            f"or drop the rule."
        )
        # A rule that narrows its candidates needs BOTH terms, or the
        # intersection is empty for a reason that has nothing to do with the
        # pipeline.
        #
        # This checks the TERMS, not their values. It cannot tell whether the
        # market fixture's tickers actually overlap the SEC fixture's -- the
        # two are sampled independently, and when they do not overlap the rule
        # skips per-fixture rather than failing. That is the honest behaviour
        # (a non-overlapping sample is not a defect) but it does mean the
        # market half of the company bridge is exercised by the unit tests in
        # test_cross_source_linker.py rather than end to end.
        if rule.resolved_against:
            assert is_emitted(rule.resolved_against), (
                f"{rule.label}: no committed fixture emits "
                f"{rule.resolved_against}, which this rule narrows its "
                f"candidates against, so it can never measure anything."
            )


# ======================================================================
# The transaction chain: N dated transactions, N-1 edges
# ======================================================================
#
# Same argument as above, one level down. The bridge rules ask what fraction of
# a population is linked at all; a chain also has to be SHAPED like a chain, and
# the two ways it degenerates -- too few edges, or edges that branch -- both
# leave `precedes` present and every term involved live.

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_ND_TYPE = f"{_ONT}sec-filings/NonDerivativeTransaction"
_D_TYPE = f"{_ONT}sec-filings/DerivativeTransaction"
_HAS_ND = f"{_ONT}sec-filings/hasNonDerivativeTransaction"
_HAS_D = f"{_ONT}sec-filings/hasDerivativeTransaction"
_TX_DATE = f"{_ONT}sec-filings/hasTransactionDate"
_REPORTING_OWNER = f"{_ONT}sec-filings/hasReportingOwner"
_PRECEDES = f"{_ONT}sec/precedes"

_OWNER = f"{_ID}sec-filings/ReportingOwner_0000000001"
_FILING = f"{_ID}sec-filings/0000000001-26-000001_Filing"


def _transaction(index, kind="NonDerivative"):
    return f"{_ID}sec-filings/0000000001-26-000001_{kind}Transaction_{index}"


def _reported(index, date, kind="NonDerivative"):
    """A filing reporting one dated transaction, and the pointer to it."""
    transaction = _transaction(index, kind)
    return [
        (_FILING, _HAS_ND if kind == "NonDerivative" else _HAS_D, transaction),
        (transaction, _RDF_TYPE, _ND_TYPE if kind == "NonDerivative" else _D_TYPE),
        (transaction, _TX_DATE, date),
    ]


def _chain_rows(edges=((0, 1), (1, 2))):
    """Three dated transactions of one owner, chained by ``edges``."""
    rows = [(_FILING, _REPORTING_OWNER, _OWNER)]
    for index, date in enumerate(("2026-08-10", "2026-08-11", "2026-08-12")):
        rows += _reported(index, date)
    rows += [
        (_transaction(a), _PRECEDES, _transaction(b)) for a, b in edges
    ]
    return rows


def test_a_complete_chain_passes(tmp_path):
    _smoke._assert_transaction_chains_are_complete(_write(tmp_path, _chain_rows()))


def test_a_group_of_one_is_not_required_to_chain(tmp_path):
    """A filing reporting a single transaction has nothing to sequence.

    The overwhelmingly common case upstream — and it must not read as a defect,
    or every fixture fails for a fact about ownership filings rather than about
    the pipeline.
    """
    rows = [(_FILING, _REPORTING_OWNER, _OWNER), *_reported(0, "2026-08-10")]
    _smoke._assert_transaction_chains_are_complete(_write(tmp_path, rows))


def test_a_chain_missing_its_middle_edge_fails(tmp_path):
    """The defect this exists for.

    Two of three transactions are linked, so `precedes` is present between
    transactions and every guard that asks whether the relation resolved is
    satisfied. Only the count says the third is stranded.
    """
    with pytest.raises(AssertionError, match="linked by 1 edge"):
        _smoke._assert_transaction_chains_are_complete(
            _write(tmp_path, _chain_rows(edges=((0, 1),)))
        )


def test_a_star_is_not_a_chain(tmp_path):
    """N-1 edges is necessary and not sufficient.

    One transaction pointing at both others has the right edge COUNT and the
    wrong shape: it is what a window ordered on a non-total key can produce, and
    it says two things happened next.
    """
    with pytest.raises(AssertionError, match="branches at"):
        _smoke._assert_transaction_chains_are_complete(
            _write(tmp_path, _chain_rows(edges=((0, 1), (0, 2))))
        )


def test_classes_chained_together_fail(tmp_path):
    """A derivative and a non-derivative transaction are two groups, not one.

    Chaining across them leaves each class-group short of its own edges, which
    is what fires here — and the merged edge itself is then reported as
    crossing groups.
    """
    rows = [
        (_FILING, _REPORTING_OWNER, _OWNER),
        *_reported(0, "2026-08-10"),
        *_reported(1, "2026-08-11"),
        *_reported(0, "2026-08-12", kind="Derivative"),
        (_transaction(1), _PRECEDES, _transaction(0, "Derivative")),
    ]
    with pytest.raises(AssertionError) as caught:
        _smoke._assert_transaction_chains_are_complete(_write(tmp_path, rows))
    assert "0 edge(s), expected 1" in str(caught.value)


def test_the_committed_fixture_carries_a_chain_to_measure():
    """The guard skips a fixture with no group of two, so one must have one.

    Exactly the argument behind test_every_rule_is_exercised_by_the_committed_
    fixtures: a skip is only defensible while something proves the check is not
    skipped everywhere. Upstream reports one transaction per filing far more
    often than several, so this does not hold by luck --
    generate_sec_e2e_fixtures.py selects such a filing on purpose and asserts it
    survived the sample.
    """
    import collections

    import rdflib

    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests" / "fixtures" / "e2e" / "turtle_parquet" / "sec"
        / "sec_sample.parquet"
    )
    graph = rdflib.Graph()
    for document in pd.read_parquet(fixture)["triples"]:
        graph.parse(data=document, format="turtle")

    dated = set(graph.subjects(rdflib.URIRef(_TX_DATE), None))
    groups = collections.Counter(
        (subject, predicate)
        for predicate in (rdflib.URIRef(_HAS_ND), rdflib.URIRef(_HAS_D))
        for subject, obj in graph.subject_objects(predicate)
        if obj in dated
    )
    assert max(groups.values(), default=0) > 1, (
        "no committed SEC fixture has a filing reporting two dated "
        "transactions of one class, so _assert_transaction_chains_are_complete "
        "skips wherever it runs and measures nothing. Regenerate the fixture "
        "(bin/generate_sec_e2e_fixtures.py), which selects one for this reason."
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
