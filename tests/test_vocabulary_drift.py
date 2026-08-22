"""
Regression guard for the defect class this repository keeps producing.

Ten defects so far have been one shape: a linker filters on an ontology term
the upstream data does not emit. rdflib's Namespace resolves ANY attribute, so
``SEC_FILINGS.hasIssuerTicker`` is valid Python long after the term is dead;
Spark then matches zero rows, the join drops every row, and the step logs that
it ran. Nothing raises, no schema changes, and the graph is simply missing a
relationship.

`bin/check_vocabulary_drift.py` diffs the terms the CODE references against the
terms the DATA emits. This pins its fixture-derived result so a NEW absence
fails the fast suite, rather than waiting for someone to notice a missing edge.

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
The baseline is derived from the committed fixtures, so most of its entries are
sampling artifacts — a BLS class that no sampled row happens to use is absent
without being dead. Pinning them is still worth it: the guard is not "the
baseline is all defects", it is "the set does not GROW without someone looking".

A THIRD KIND OF ENTRY, which the JSON cannot annotate itself. `JOLTS
hasCensusRegionCode` is not unsampled and not dead — it is NOT YET EMITTED
ANYWHERE. The region work binds it deliberately ahead of upstream, because the
failure it guards is silent: the codes are to arrive as zero-padded
`xsd:string`, and a numeric 1 that never matches a string "01" produces an
empty join and no error. Writing the reader against the agreed shape now is
cheaper than discovering the mismatch after the term lands. The census-region
bridge does not depend on it — a name-keyed path carries it today — so this
entry is a pending confirmation, not a dead reference to repair. See
enrichment/region_crosswalk.py.

An entry leaving the baseline is also a failure. A term that starts being
emitted means either a fixture regeneration covered it or a defect was fixed,
and both should update the file rather than leave it overstating the damage.

The verdict run is `--s3` against real volume, which is what can distinguish a
dead term from an unsampled one. That needs credentials, so it is not run here.
No Spark session is created: this is pure AST + rdflib parsing.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "conf" / "vocabulary_baseline.json"
TOOL = REPO_ROOT / "bin" / "check_vocabulary_drift.py"


def _load_tool():
    """Import the checker by path — bin/ is scripts, not an importable package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_vocab_drift", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def drift():
    tool = _load_tool()
    found, uncovered = tool.analyze(tool.emitted_from_fixtures())
    return found, uncovered


def test_no_new_vocabulary_drift(drift):
    """No term may go absent without the baseline being updated deliberately."""
    found, _uncovered = drift
    known = json.loads(BASELINE.read_text())

    new = {
        namespace: sorted(set(terms) - set(known.get(namespace, [])))
        for namespace, terms in found.items()
    }
    new = {ns: terms for ns, terms in new.items() if terms}

    assert not new, (
        f"terms the code keys on that the data no longer emits: {new} — a "
        f"filter on one of these matches nothing, and the join behind it will "
        f"drop every row without raising. Fix the reference, or regenerate "
        f"conf/vocabulary_baseline.json if the fixtures legitimately stopped "
        f"covering it."
    )


def test_baseline_does_not_overstate(drift):
    """A term that is emitted again must leave the baseline."""
    found, _uncovered = drift
    known = json.loads(BASELINE.read_text())

    resolved = {
        namespace: sorted(set(terms) - set(found.get(namespace, [])))
        for namespace, terms in known.items()
    }
    resolved = {ns: terms for ns, terms in resolved.items() if terms}

    assert not resolved, (
        f"{resolved} are emitted again — regenerate the baseline with "
        f"`bin/check_vocabulary_drift.py --write-baseline "
        f"conf/vocabulary_baseline.json` so it stops claiming they are absent"
    )


def test_uncovered_namespaces_are_reported_not_counted_as_drift(drift):
    """A namespace with no sample coverage must not be reported as drift.

    Without this the report is unusable: the fixtures carry no litigation or
    trading-suspension data at all, so every term in those namespaces looks
    dead and buries the handful that genuinely are. Their absence is a coverage
    finding, which the tool prints separately.
    """
    found, uncovered = drift

    assert uncovered, "expected some namespace to have no fixture coverage"
    assert not (set(uncovered) & set(found)), (
        "a namespace was reported as both uncovered and drifted"
    )


def test_the_tool_runs_and_reports(tmp_path):
    """The tool itself must execute — it is the thing the guard depends on."""
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "TERMS THE CODE KEYS ON" in result.stdout
