"""
Unit tests for scatter_sparse_entries.

This is the write that both feature extractors do at the end of a collect:
sparse (key, dim, value) rows go into a pre-allocated dense array. Five call
sites share it, so what it drops and what it lets raise is pinned here rather
than rediscovered from a cluster run.

Pure numpy and Pandas -- no SparkSession.
"""
import numpy as np
import pandas as pd
import pytest

from spark_jobs.pyg_builder.sparse_scatter import scatter_sparse_entries


def _entries(rows, key_column="node_id"):
    """Rows given as (key, dim, value) tuples -> the frame a collect returns."""
    return pd.DataFrame(rows, columns=[key_column, "dim", "value"])


# ======================================================================
# The write itself
# ======================================================================

def test_writes_each_entry_to_its_cell():
    tensor = np.zeros((3, 4), dtype=np.float32)

    scatter_sparse_entries(
        _entries([(0, 1, 0.5), (2, 3, -1.25)]), tensor, "node_id", 4
    )

    assert tensor[0, 1] == pytest.approx(0.5)
    assert tensor[2, 3] == pytest.approx(-1.25)
    # Nothing else moved.
    assert tensor.sum() == pytest.approx(0.5 - 1.25)


def test_edge_idx_is_just_another_key_column():
    """The edge side passes edge_idx; only the column name differs."""
    tensor = np.zeros((2, 3), dtype=np.float32)

    scatter_sparse_entries(
        _entries([(1, 2, 7.0)], key_column="edge_idx"), tensor, "edge_idx", 3
    )

    assert tensor[1, 2] == pytest.approx(7.0)


def test_writes_in_place_and_leaves_other_cells_alone():
    """The caller pre-allocates and may reuse; the helper must not reset."""
    tensor = np.full((2, 2), 9.0, dtype=np.float32)

    scatter_sparse_entries(_entries([(0, 0, 1.0)]), tensor, "node_id", 2)

    assert tensor[0, 0] == pytest.approx(1.0)
    assert tensor[0, 1] == pytest.approx(9.0)
    assert tensor[1, 0] == pytest.approx(9.0)
    assert tensor[1, 1] == pytest.approx(9.0)


def test_last_entry_wins_for_a_repeated_cell():
    """Numpy fancy-index assignment, pinned so a change to it is visible."""
    tensor = np.zeros((1, 2), dtype=np.float32)

    scatter_sparse_entries(
        _entries([(0, 0, 1.0), (0, 0, 2.0)]), tensor, "node_id", 2
    )

    assert tensor[0, 0] == pytest.approx(2.0)


def test_accepts_a_groupby_slice():
    """_flush_batch hands a group, not a fresh frame -- its index starts high."""
    frame = _entries([(0, 0, 1.0), (1, 1, 2.0), (0, 1, 3.0)])
    frame["node_type"] = ["a", "b", "a"]
    group = frame.groupby("node_type").get_group("a")
    tensor = np.zeros((2, 2), dtype=np.float32)

    scatter_sparse_entries(group, tensor, "node_id", 2)

    assert tensor[0, 0] == pytest.approx(1.0)
    assert tensor[0, 1] == pytest.approx(3.0)
    assert tensor[1, 1] == pytest.approx(0.0)


# ======================================================================
# What it drops
# ======================================================================

@pytest.mark.parametrize("dim", [-1, -5, 4, 99])
def test_drops_a_dim_outside_the_vector(dim):
    """Out-of-range dims are dropped, not raised on."""
    tensor = np.zeros((2, 4), dtype=np.float32)

    scatter_sparse_entries(_entries([(0, dim, 1.0)]), tensor, "node_id", 4)

    assert tensor.sum() == pytest.approx(0.0)


def test_drops_only_the_out_of_range_rows():
    tensor = np.zeros((2, 4), dtype=np.float32)

    scatter_sparse_entries(
        _entries([(0, 1, 5.0), (0, 4, 6.0), (1, 0, 7.0)]),
        tensor,
        "node_id",
        4,
    )

    assert tensor[0, 1] == pytest.approx(5.0)
    assert tensor[1, 0] == pytest.approx(7.0)
    assert tensor.sum() == pytest.approx(12.0)


def test_vector_dim_bounds_the_write_not_the_tensor_width():
    """A caller may pass a dim narrower than the array it allocated."""
    tensor = np.zeros((1, 8), dtype=np.float32)

    scatter_sparse_entries(
        _entries([(0, 2, 1.0), (0, 5, 2.0)]), tensor, "node_id", 4
    )

    assert tensor[0, 2] == pytest.approx(1.0)
    assert tensor[0, 5] == pytest.approx(0.0)


# ======================================================================
# Nothing to write
# ======================================================================

def test_none_writes_nothing():
    """groups.get(key) misses for a type with no entries; that is not an error."""
    tensor = np.zeros((2, 2), dtype=np.float32)

    scatter_sparse_entries(None, tensor, "node_id", 2)

    assert tensor.sum() == pytest.approx(0.0)


def test_empty_frame_writes_nothing():
    tensor = np.zeros((2, 2), dtype=np.float32)

    scatter_sparse_entries(_entries([]), tensor, "node_id", 2)

    assert tensor.sum() == pytest.approx(0.0)


# ======================================================================
# What it does NOT guard
# ======================================================================

def test_a_key_past_the_end_raises():
    """Keys are the caller's own indexing. A bad one is a bug there, so it
    must surface rather than be masked off like an out-of-range dim."""
    tensor = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(IndexError):
        scatter_sparse_entries(_entries([(5, 0, 1.0)]), tensor, "node_id", 2)
