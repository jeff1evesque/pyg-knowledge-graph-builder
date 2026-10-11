"""
Node types whose x stores only the positions that carry information.

One node type is most of the graph. On 2026-10-09, 10,715,848 of its 10.9M
nodes were market_quotes_OptionSnapshot, and their 1024-wide x was 43.9 GB of
the 48.6 GB .pt. On every option row, 851 of the 1024 positions hold the same
thing: 0, one fixed value, or a copy of another position. So an option's x
stores the other 173, in layout order, and feature_spec.json records what each
of the 851 holds. The full 1024 rebuild exactly from the two.

The list is fixed here, not worked out by each build. A trained model's input
layer is sized to it, and a model refreshed from yesterday's weights needs the
width to stay the same. Changing the list is a deliberate change, and it
changes contract_digest.

Each build checks that the dropped positions still hold their values and
records any that do not in feature_spec.json. It warns rather than fails: a
failed leg blocks the night's publish, tables included.

Nothing here touches Spark. The executor side is in feature_extractor.py.
"""
import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from spark_jobs.pyg_builder.sparse_scatter import scatter_sparse_entries

logger = logging.getLogger(__name__)


# The layout the positions below were measured on: the nightly's, 1024 wide
# with a 192-wide class_identity. Another layout puts other things at these
# positions, so a build on one keeps every position.
TRIMMED_LAYOUT = {"vector_dim": 1024, "class_identity_dim": 192}

# Measured on the days 2026-10-07, 10-08 and 10-09, which agree on every
# position. A position not listed holds 0 on every row.
TRIMMED_NODE_TYPES: Dict[str, Dict[str, Any]] = {
    "market_quotes_OptionSnapshot": {
        # x[:, i] holds position kept[i].
        "kept": (
            # property_presence and domain_range, for properties only some
            # options have
            258, 260, 278, 281, 285, 289, 323, 326, 332, 363, 381, 385,
            411, 417, 439, 453, 549,
            # numeric_values
            649, 655, 662, 665, 666, 667, 668, 677, 683, 687, 698, 700,
            717, 729, 730, 779, 784, 799, 809, 810, 833, 853, 857, 859,
            860, 872, 877, 883, 895,
            # every categorical_values position
            *range(897, 1024),
        ),
        # Position: the value it holds on every row. Above 1.0 where several
        # flags hash into one slot and add up.
        "constant": {
            18: 1.0, 20: 1.0, 44: 1.0, 111: 1.0, 240: 1.0, 256: 1.0,
            261: 1.0, 263: 1.0, 265: 1.0, 266: 1.0, 268: 1.0, 269: 2.0,
            272: 1.0, 274: 2.0, 280: 2.0, 288: 1.0, 292: 1.0, 294: 1.0,
            295: 1.0, 296: 1.0, 299: 2.0, 301: 1.0, 302: 1.0, 304: 1.0,
            305: 2.0, 306: 1.0, 308: 1.0, 309: 1.0, 311: 1.0, 315: 2.0,
            319: 2.0, 320: 2.0, 322: 1.0, 324: 1.0, 325: 1.0, 328: 1.0,
            331: 4.0, 333: 1.0, 335: 1.0, 336: 1.0, 339: 2.0, 340: 1.0,
            343: 3.0, 344: 1.0, 348: 2.0, 349: 1.0, 351: 1.0, 354: 2.0,
            356: 1.0, 357: 1.0, 361: 1.0, 370: 1.0, 371: 1.0, 372: 2.0,
            373: 1.0, 377: 1.0, 378: 1.0, 380: 2.0, 384: 1.0, 389: 2.0,
            393: 1.0, 397: 1.0, 398: 1.0, 401: 1.0, 402: 1.0, 403: 1.0,
            409: 2.0, 410: 1.0, 413: 1.0, 415: 1.0, 416: 1.0, 419: 1.0,
            420: 1.0, 421: 1.0, 422: 2.0, 424: 1.0, 429: 1.0, 430: 1.0,
            432: 1.0, 435: 1.0, 437: 1.0, 441: 1.0, 442: 1.0, 443: 2.0,
            444: 1.0, 445: 1.0, 446: 1.0, 447: 1.0, 518: 1.0, 522: 1.0,
            536: 7.0, 541: 23.0, 558: 5.0, 630: 2.0, 632: 2.0,
        },
        # Position: the kept position it always equals. A property flag is
        # written into three hashed slots, so its slots move together.
        "copy_of": {
            341: 278, 382: 323, 400: 260, 431: 417, 433: 326, 514: 323,
            528: 417,
        },
    },
}


class TrimSpec:
    """One node type's kept positions, and what each dropped one holds."""

    def __init__(
        self,
        node_type: str,
        vector_dim: int,
        kept: Iterable[int],
        constant: Mapping[int, float],
        copy_of: Mapping[int, int],
    ):
        kept = tuple(int(p) for p in kept)
        constant = {int(p): float(v) for p, v in constant.items()}
        copy_of = {int(p): int(t) for p, t in copy_of.items()}
        problems = _problems(vector_dim, kept, constant, copy_of)
        if problems:
            raise ValueError(
                f"trimmed positions for {node_type}: {'; '.join(problems)}"
            )

        self.node_type = node_type
        self.vector_dim = vector_dim
        self.kept = kept
        self.constant = dict(sorted(constant.items()))
        self.copy_of = dict(sorted(copy_of.items()))
        listed = set(kept) | set(constant) | set(copy_of)
        self.zero = tuple(p for p in range(vector_dim) if p not in listed)

        # Lookups by position, for the driver's per-chunk work. column_of is
        # the position's column in x and copy_index its place in copy_of, -1
        # where neither applies.
        self.column_of = np.full(vector_dim, -1, dtype=np.int64)
        self.column_of[list(kept)] = np.arange(len(kept))
        self.copy_index = np.full(vector_dim, -1, dtype=np.int64)
        self.copy_index[list(self.copy_of)] = np.arange(len(self.copy_of))
        self.is_zero = np.zeros(vector_dim, dtype=bool)
        self.is_zero[list(self.zero)] = True
        # x's column for each copy's twin, in copy_of order.
        self.twin_columns = self.column_of[list(self.copy_of.values())]

    @property
    def width(self) -> int:
        """How many columns x has."""
        return len(self.kept)

    def to_dict(self) -> Dict[str, Any]:
        """The list as feature_spec.json and encoding_config.json record it."""
        return {
            "stored_dim": self.width,
            "kept_positions": list(self.kept),
            "zero_positions": list(self.zero),
            "constant_positions": [[p, v] for p, v in self.constant.items()],
            "copied_positions": [[p, t] for p, t in self.copy_of.items()],
        }

    def any_nonzero_from(self, x, start: int) -> bool:
        """Whether a position from ``start`` on holds a nonzero, read off x.

        Kept positions are in increasing order, so the ones from ``start`` on
        are the columns from one index on: a view of x, not a copy.
        """
        first = int(np.searchsorted(self.kept, start))
        if bool(x[:, first:].any()):
            return True
        if x.shape[0] and any(
            v != 0 for p, v in self.constant.items() if p >= start
        ):
            return True
        return any(
            bool(x[:, self.column_of[t]].any())
            for p, t in self.copy_of.items() if p >= start
        )


def _problems(
    vector_dim: int,
    kept: Tuple[int, ...],
    constant: Dict[int, float],
    copy_of: Dict[int, int],
) -> List[str]:
    """What is wrong with a list, as sentences. Empty when nothing is."""
    problems = []
    if not kept:
        problems.append("no position is kept")
    if any(b <= a for a, b in zip(kept, kept[1:])):
        problems.append("kept positions must increase, each listed once")
    listed = [*kept, *constant, *copy_of, *copy_of.values()]
    outside = sorted({p for p in listed if not 0 <= p < vector_dim})
    if outside:
        problems.append(f"positions outside 0..{vector_dim - 1}: {outside}")
    kept_set, constant_set, copy_set = set(kept), set(constant), set(copy_of)
    twice = (
        (kept_set & constant_set) | (kept_set & copy_set)
        | (constant_set & copy_set)
    )
    if twice:
        problems.append(f"positions listed twice: {sorted(twice)}")
    unkept = sorted(t for t in copy_of.values() if t not in kept_set)
    if unkept:
        problems.append(f"copies of positions that are not kept: {unkept}")
    return problems


def trims_for(layout, enabled: bool = True) -> Dict[str, TrimSpec]:
    """The trimmed node types for a build on ``layout``.

    Empty when trimming is switched off, or when the layout is not the one
    the list was measured on.
    """
    if not enabled:
        return {}
    built = {
        "vector_dim": layout.vector_dim,
        "class_identity_dim": layout.seg1_class_identity_dim,
    }
    if built != TRIMMED_LAYOUT:
        logger.warning(
            f"    Not trimming node features: the trimmed positions were "
            f"measured on {TRIMMED_LAYOUT} and this build is {built}, so every "
            f"node type keeps all {layout.vector_dim} positions"
        )
        return {}
    return {
        node_type: TrimSpec(node_type, layout.vector_dim, **entry)
        for node_type, entry in TRIMMED_NODE_TYPES.items()
    }


def scatter_chunk(
    spec: TrimSpec,
    entries: Any,
    tensor: np.ndarray,
    lo: int,
    hi: int,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Write one chunk of a trimmed type's rows into its x, and check them.

    ``entries`` is the Pandas frame collected for nodes ``lo`` to ``hi - 1``:
    node_id, dim and value at the kept and copied positions, plus any nonzero
    found where a position should hold 0. Kept values go into their columns,
    and ``entries``' dim column is rewritten to those columns on the way.
    Copies are compared with their twins row by row, after the twins are in.

    Returns two dicts, position to rows: zero positions that held a value,
    and copies that differed from their twin.
    """
    rows = hi - lo
    if entries is None or entries.empty:
        dims = np.empty(0, dtype=np.int64)
        keys = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=tensor.dtype)
    else:
        dims = entries["dim"].to_numpy()
        keys = entries["node_id"].to_numpy()
        values = entries["value"].to_numpy()

    # A dim outside the layout is dropped, as scatter_sparse_entries does.
    inside = (dims >= 0) & (dims < spec.vector_dim)
    position = np.where(inside, dims, 0)
    columns = np.where(inside, spec.column_of[position], -1)
    copy_slot = np.where(inside, spec.copy_index[position], -1)

    if entries is not None and not entries.empty:
        entries["dim"] = columns
        scatter_sparse_entries(entries, tensor, "node_id", spec.width)

    copies = np.zeros((rows, len(spec.copy_of)), dtype=tensor.dtype)
    is_copy = copy_slot >= 0
    copies[keys[is_copy] - lo, copy_slot[is_copy]] = values[is_copy]
    differs = (copies != tensor[lo:hi][:, spec.twin_columns]).sum(axis=0)
    copy_broken = {
        p: int(n) for p, n in zip(spec.copy_of, differs) if n
    }

    stray = inside & spec.is_zero[position] & (values != 0)
    zero_broken = {
        int(p): int(n) for p, n in Counter(dims[stray].tolist()).items()
    }
    return zero_broken, copy_broken


class TrimCheck:
    """Whether a build's dropped positions held their values, per position.

    Filled in one chunk of rows at a time. Constant positions arrive as the
    rows that held the value, counted on the executors. Zero positions and
    copies arrive as the rows that broke, found on the driver.
    """

    def __init__(self, spec: TrimSpec):
        self.spec = spec
        self.rows = 0
        self._held = {p: 0 for p in spec.constant}
        self._zero_broken: Counter = Counter()
        self._copy_broken: Counter = Counter()

    def add(
        self,
        rows: int,
        held: Mapping[int, int],
        zero_broken: Optional[Mapping[int, int]] = None,
        copy_broken: Optional[Mapping[int, int]] = None,
    ) -> None:
        """Count one chunk of ``rows`` rows."""
        self.rows += rows
        for p in self._held:
            self._held[p] += int(held.get(p, 0))
        self._zero_broken.update(zero_broken or {})
        self._copy_broken.update(copy_broken or {})

    def violations(self) -> List[Dict[str, Any]]:
        """Each dropped position that did not hold its value, by position."""
        found = []
        for p, n in self._zero_broken.items():
            if n:
                found.append(
                    {"position": p, "expected": 0.0, "rows_affected": n}
                )
        for p, v in self.spec.constant.items():
            missed = self.rows - self._held[p]
            if missed:
                found.append(
                    {"position": p, "expected": v, "rows_affected": missed}
                )
        for p, n in self._copy_broken.items():
            if n:
                found.append({
                    "position": p,
                    "copy_of": self.spec.copy_of[p],
                    "rows_affected": n,
                })
        return sorted(found, key=lambda v: v["position"])

    def to_dict(self) -> Dict[str, Any]:
        """The check as feature_spec.json records it."""
        return {
            "rows_checked": self.rows,
            "positions_checked": self.spec.vector_dim - self.spec.width,
            "violations": self.violations(),
        }

    def summary(self) -> str:
        """One line for the build log."""
        found = self.violations()
        if not found:
            return (
                f"all {self.spec.vector_dim - self.spec.width} dropped "
                f"positions held their values on {self.rows:,} rows"
            )
        parts = []
        for v in found:
            should = (
                f"equal {v['copy_of']}" if "copy_of" in v
                else f"be {v['expected']:g}"
            )
            parts.append(
                f"{v['position']} (should {should}) on "
                f"{v['rows_affected']:,} rows"
            )
        return (
            f"{len(found)} dropped position(s) did not hold their value: "
            + "; ".join(parts)
        )
