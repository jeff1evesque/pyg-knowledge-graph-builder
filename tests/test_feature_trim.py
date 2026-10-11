"""Unit tests for spark_jobs/pyg_builder/feature_trim.py (#443).

The committed list says which positions market_quotes_OptionSnapshot's x
keeps and what every other one holds. A trained model's input layer is sized
to it, so the list itself is pinned here, along with the driver-side half of
the per-build check. No Spark: the executor half is in
test_feature_extractor.py.
"""
import numpy as np
import pandas as pd
import pytest

from spark_jobs.pyg_builder import feature_trim
from spark_jobs.pyg_builder.feature_trim import (
    TRIMMED_LAYOUT,
    TRIMMED_NODE_TYPES,
    TrimCheck,
    TrimSpec,
    scatter_chunk,
    trims_for,
)
from spark_jobs.pyg_builder.vector_layout import VectorLayout

OPTIONS = "market_quotes_OptionSnapshot"
NIGHTLY = VectorLayout(1024, class_identity_dim=192)


def _toy(**overrides):
    """A 16-position list: keeps 3, 5, 9; 1 and 2 constant; 7 copies 5."""
    args = {
        "node_type": "toy",
        "vector_dim": 16,
        "kept": (3, 5, 9),
        "constant": {1: 1.0, 2: 2.0},
        "copy_of": {7: 5},
    }
    args.update(overrides)
    return TrimSpec(**args)


def _entries(rows):
    return pd.DataFrame(
        rows, columns=["node_id", "dim", "value"]
    ).astype({"node_id": "int64", "dim": "int32", "value": "float32"})


# ======================================================================
# The committed list
# ======================================================================

def test_the_committed_list_is_the_one_measured():
    """173 kept, 95 constant, 7 copies, and 749 zeros: 1024 in all.

    Measured on 2026-10-07, 10-08 and 10-09 from the published .pt files. A
    change here changes every model's input layer, so it fails until the
    numbers below are changed with it.
    """
    spec = trims_for(NIGHTLY)[OPTIONS]
    assert spec.width == 173
    assert len(spec.constant) == 95
    assert spec.copy_of == {
        341: 278, 382: 323, 400: 260, 431: 417, 433: 326, 514: 323, 528: 417,
    }
    assert len(spec.zero) == 749


def test_every_position_is_accounted_for_exactly_once():
    spec = trims_for(NIGHTLY)[OPTIONS]
    listed = [*spec.kept, *spec.constant, *spec.copy_of, *spec.zero]
    assert sorted(listed) == list(range(1024))


def test_the_kept_positions_sit_where_the_issue_says():
    """17 property_presence/domain_range, 29 numeric, all 127 categorical."""
    spec = trims_for(NIGHTLY)[OPTIONS]
    schema = [p for p in spec.kept if p < NIGHTLY.seg3_start]
    numeric = [
        p for p in spec.kept
        if NIGHTLY.seg3_numeric_start <= p < NIGHTLY.seg3_categorical_start
    ]
    categorical = [p for p in spec.kept if p >= NIGHTLY.seg3_categorical_start]
    assert len(schema) == 17
    assert all(NIGHTLY.seg2_start <= p < NIGHTLY.seg3_start for p in schema)
    assert len(numeric) == 29
    assert categorical == list(range(NIGHTLY.seg3_categorical_start, 1024))


def test_the_list_applies_only_to_the_layout_it_was_measured_on():
    assert TRIMMED_LAYOUT == {"vector_dim": 1024, "class_identity_dim": 192}
    assert set(trims_for(NIGHTLY)) == set(TRIMMED_NODE_TYPES)
    # The default class_identity width moves class_identity and ontology_source.
    assert trims_for(VectorLayout(1024)) == {}
    assert trims_for(VectorLayout(2048, class_identity_dim=192)) == {}


def test_switched_off_trims_nothing():
    assert trims_for(NIGHTLY, enabled=False) == {}


# ======================================================================
# TrimSpec
# ======================================================================

def test_zero_positions_are_everything_not_listed():
    assert _toy().zero == (0, 4, 6, 8, 10, 11, 12, 13, 14, 15)


def test_column_lookups_follow_the_kept_order():
    spec = _toy()
    assert spec.width == 3
    assert [int(spec.column_of[p]) for p in (3, 5, 9)] == [0, 1, 2]
    assert int(spec.column_of[7]) == -1
    assert spec.twin_columns.tolist() == [1]


@pytest.mark.parametrize("overrides, says", [
    ({"kept": (5, 3, 9)}, "must increase"),
    ({"kept": (3, 3, 9)}, "must increase"),
    ({"kept": ()}, "no position is kept"),
    ({"kept": (3, 5, 16)}, "outside 0..15"),
    ({"constant": {3: 1.0}}, "listed twice: [3]"),
    ({"copy_of": {1: 5}}, "listed twice: [1]"),
    ({"copy_of": {7: 6}}, "not kept: [6]"),
])
def test_a_list_that_cannot_be_right_is_refused(overrides, says):
    with pytest.raises(ValueError, match=r"toy") as raised:
        _toy(**overrides)
    assert says in str(raised.value)


def test_to_dict_lists_every_dropped_position_with_what_it_holds():
    assert _toy().to_dict() == {
        "stored_dim": 3,
        "kept_positions": [3, 5, 9],
        "zero_positions": [0, 4, 6, 8, 10, 11, 12, 13, 14, 15],
        "constant_positions": [[1, 1.0], [2, 2.0]],
        "copied_positions": [[7, 5]],
    }


def test_any_nonzero_from_reads_kept_columns_by_position():
    spec = _toy()
    x = np.zeros((2, 3), dtype=np.float32)
    x[1, 2] = 4.0                         # position 9
    assert spec.any_nonzero_from(x, 9)
    assert spec.any_nonzero_from(x, 6)
    assert not spec.any_nonzero_from(x, 10)
    x[1, 2] = 0.0
    x[0, 0] = 1.0                         # position 3, before 6
    assert not spec.any_nonzero_from(x, 6)


def test_any_nonzero_from_counts_constants_and_copies():
    spec = _toy(constant={1: 1.0, 12: 3.0}, copy_of={7: 5})
    x = np.zeros((2, 3), dtype=np.float32)
    # Position 12 holds 3.0 on every row, so a row exists that has it.
    assert spec.any_nonzero_from(x, 12)
    assert not spec.any_nonzero_from(np.zeros((0, 3), np.float32), 12)
    # Position 7 equals position 5.
    spec = _toy()
    x[0, 1] = 2.0
    assert spec.any_nonzero_from(x, 7)


# ======================================================================
# scatter_chunk — the driver's half of the check
# ======================================================================

def test_kept_values_land_in_their_columns():
    spec = _toy()
    tensor = np.zeros((4, 3), dtype=np.float32)
    zero_broken, copy_broken = scatter_chunk(
        spec,
        _entries([
            (2, 3, 0.5), (2, 5, 1.0), (2, 7, 1.0),
            (3, 9, -2.0),
        ]),
        tensor, 2, 4,
    )
    assert tensor[2].tolist() == [0.5, 1.0, 0.0]
    assert tensor[3].tolist() == [0.0, 0.0, -2.0]
    assert tensor[:2].sum() == 0
    assert zero_broken == {} and copy_broken == {}


def test_a_nonzero_where_zero_belongs_is_counted_per_position():
    spec = _toy()
    tensor = np.zeros((3, 3), dtype=np.float32)
    zero_broken, _ = scatter_chunk(
        spec,
        _entries([
            (0, 4, 1.0), (1, 4, 2.0), (2, 11, 0.5),
            (2, 12, 0.0),                 # a 0.0 at a zero position is fine
        ]),
        tensor, 0, 3,
    )
    assert zero_broken == {4: 2, 11: 1}
    assert tensor.sum() == 0


def test_a_copy_is_compared_with_its_twin_row_by_row():
    spec = _toy()
    tensor = np.zeros((3, 3), dtype=np.float32)
    _, copy_broken = scatter_chunk(
        spec,
        _entries([
            (0, 5, 1.0), (0, 7, 1.0),     # equal
            (1, 5, 2.0), (1, 7, 1.0),     # differs
            (2, 7, 1.0),                  # twin absent: differs
        ]),
        tensor, 0, 3,
    )
    assert copy_broken == {7: 2}


def test_rows_with_neither_a_copy_nor_its_twin_agree():
    """Both absent is 0 and 0 -- equal, not broken."""
    spec = _toy()
    tensor = np.zeros((2, 3), dtype=np.float32)
    assert scatter_chunk(spec, _entries([]), tensor, 0, 2) == ({}, {})
    assert scatter_chunk(spec, None, tensor, 0, 2) == ({}, {})


def test_a_dim_outside_the_layout_is_dropped():
    spec = _toy()
    tensor = np.zeros((1, 3), dtype=np.float32)
    zero_broken, copy_broken = scatter_chunk(
        spec, _entries([(0, -1, 1.0), (0, 16, 1.0), (0, 9, 3.0)]),
        tensor, 0, 1,
    )
    assert tensor.tolist() == [[0.0, 0.0, 3.0]]
    assert zero_broken == {} and copy_broken == {}


def test_a_chunk_that_does_not_start_at_zero_places_rows_by_node_id():
    spec = _toy()
    tensor = np.zeros((6, 3), dtype=np.float32)
    _, copy_broken = scatter_chunk(
        spec, _entries([(4, 5, 2.0), (4, 7, 2.0), (5, 7, 1.0)]),
        tensor, 4, 6,
    )
    assert tensor[4, 1] == 2.0
    assert copy_broken == {7: 1}


# ======================================================================
# TrimCheck
# ======================================================================

def test_a_clean_check_reports_nothing():
    check = TrimCheck(_toy())
    check.add(3, {1: 3, 2: 3})
    check.add(2, {1: 2, 2: 2}, {}, {})
    assert check.violations() == []
    assert check.to_dict() == {
        "rows_checked": 5,
        "positions_checked": 13,
        "violations": [],
    }
    assert "all 13 dropped positions held their values on 5 rows" in (
        check.summary()
    )


def test_a_constant_short_of_its_rows_is_reported_with_the_shortfall():
    """Missing and wrong both count: the executors count rows that HOLD it."""
    check = TrimCheck(_toy())
    check.add(3, {1: 3, 2: 1})
    check.add(2, {1: 1, 2: 2})
    assert check.violations() == [
        {"position": 1, "expected": 1.0, "rows_affected": 1},
        {"position": 2, "expected": 2.0, "rows_affected": 2},
    ]


def test_violations_of_every_kind_come_out_in_position_order():
    check = TrimCheck(_toy())
    check.add(4, {1: 4, 2: 3}, {11: 2, 0: 1}, {7: 4})
    assert check.violations() == [
        {"position": 0, "expected": 0.0, "rows_affected": 1},
        {"position": 2, "expected": 2.0, "rows_affected": 1},
        {"position": 7, "copy_of": 5, "rows_affected": 4},
        {"position": 11, "expected": 0.0, "rows_affected": 2},
    ]
    summary = check.summary()
    assert summary.startswith("4 dropped position(s) did not hold their value")
    assert "7 (should equal 5) on 4 rows" in summary
    assert "2 (should be 2) on 1 rows" in summary


def test_a_type_with_no_entries_breaks_every_constant_on_every_row():
    check = TrimCheck(_toy())
    check.add(5, held={})
    assert [v["rows_affected"] for v in check.violations()] == [5, 5]


def test_the_check_reads_the_list_it_is_given(monkeypatch):
    """trims_for builds from the module's list at call time, so a test can
    swap in a list of its own."""
    monkeypatch.setattr(feature_trim, "TRIMMED_NODE_TYPES", {
        "toy": {"kept": (900,), "constant": {}, "copy_of": {}},
    })
    spec = trims_for(NIGHTLY)["toy"]
    assert spec.kept == (900,) and len(spec.zero) == 1023
