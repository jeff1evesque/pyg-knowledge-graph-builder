"""
Write sparse (key, dim, value) rows into a dense feature tensor.

Both extractors build features as sparse rows on executors, then write them
into a pre-allocated dense array on the driver. That write is the same
operation in five places -- three on the node side, two on the edge side.
Only the key column differs: ``node_id`` for nodes, ``edge_idx`` for edges.

How the rows get collected stays with the callers. Whether the frame is
persisted, at what storage level, and how it is chunked differ on purpose:
the node chunked path must not persist (#346) and the edge one persists
DISK_ONLY (#388). One helper covering those too would put both behind a flag.
"""
from typing import Any

import numpy as np


def scatter_sparse_entries(
    entries: Any,
    tensor: np.ndarray,
    key_column: str,
    vector_dim: int,
) -> None:
    """Write one frame of sparse entries into ``tensor``, in place.

    ``entries`` is a Pandas frame carrying ``key_column``, ``dim`` and
    ``value``. None and an empty frame both write nothing, so a caller that
    has no rows for a type needs no check of its own.

    A dim outside ``[0, vector_dim)`` is dropped. Keys are not checked. The
    caller assigns them, so a key past the end of the tensor is a bug there
    and numpy raising is the right answer.
    """
    if entries is None or entries.empty:
        return

    keys = entries[key_column].values
    dims = entries["dim"].values
    values = entries["value"].values

    valid_mask = (dims >= 0) & (dims < vector_dim)
    tensor[keys[valid_mask], dims[valid_mask]] = values[valid_mask]
