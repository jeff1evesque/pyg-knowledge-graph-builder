"""
What collided in the node vector's slot assignments, and the check that reads
the answer.

FeatureExtractor gives every numeric property, class and namespace a slot in
the feature vector. compute_collision_report counts what landed where, and its
result is published as the collision_report block of slot_mapping.json.
check_class_identity_capacity then reads that report and stops the build when
the class_identity segment can no longer tell this build's classes apart --
unless the run has opted into shipping anyway.

Nothing here touches Spark or the extractor's state. It runs once per build on
the driver, over a few hundred Python dicts.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The class_identity segment is a share of segment 1, which is a share of the
# vector. Sizing a width that would fit N classes means going back through both.
from spark_jobs.pyg_builder.vector_layout import SEG1_FRAC


logger = logging.getLogger(__name__)


class ClassIdentityCapacityError(RuntimeError):
    """The class_identity segment cannot separate this build's classes."""


def _min_vector_dim_for_segment_one(dims_needed: int) -> int:
    """Smallest vector_dim whose segment 1 carries ``dims_needed`` dims."""
    import math

    return int(math.ceil(dims_needed / SEG1_FRAC))


def _driver_gb_per_million_nodes(vector_dim: int) -> float:
    """Dense feature memory on the driver, per million nodes, at this width.

    Quoted in the capacity error because "raise vector_dim" reads as free and
    is not: the tensors are collected to the driver, so the cost is linear in
    the width and lands in one process.
    """
    return vector_dim * 1_000_000 * 4 / 1024 ** 3


def check_class_identity_capacity(
    report: Dict[str, Any],
    allow_oversubscription: bool = False,
    vector_dim: Optional[int] = None,
) -> None:
    """
    Fail when the class_identity segment can no longer carry class identity.

    Three distinct failures, all otherwise silent: the graph builds, the
    vectors are the declared width, and every metadata file is internally
    consistent.

      * two classes share an identical code -- indistinguishable to any
        downstream model, whatever the segment width;
      * more classes than the segment has dimensions -- a d-dim segment holds
        at most d linearly independent codes, so beyond that a readout layer
        cannot recover class identity even though the codes stay distinct.
        Empirically the codes remain unique well past that point (4-hot into
        64 slots gives C(64,4) patterns), so distinctness alone will not warn
        anyone;
      * the codes are linearly DEPENDENT while distinct and under the ceiling
        -- the case the other two miss entirely. This is the general failure;
        the first two are the special cases of it that are cheap to name.
        Caught by measuring the rank of the code matrix rather than inferring
        separability from counts (_class_code_rank).

    This RAISES rather than logging, which it used to do. A warning was the
    wrong severity for the failure mode: the artifact ships, every consistency
    check passes, and the only symptom is a model that never learns to tell two
    classes apart -- weeks downstream, with nothing pointing back here. The
    build is the last place the problem is still attributable, so it stops
    here. feature_config.allow_class_identity_oversubscription re-enables the
    old warn-and-continue for a run that knowingly accepts partial identity.

    Near-capacity still warns: the class count grows with every source added
    and the segment does not, so headroom is worth a nudge before it is a wall.
    """
    ci = report.get("class_identity")
    if not ci:
        return

    total, dim = ci.get("total_classes", 0), ci.get("segment_dim", 0)
    shared = ci.get("classes_sharing_a_code") or []

    problems = []
    if shared:
        problems.append(
            f"{len(shared)} group(s) of classes share an identical "
            f"class_identity code and are indistinguishable downstream: "
            f"{shared[:3]}"
        )
    if dim and total > dim:
        problems.append(
            f"class_identity is over-subscribed: {total} classes into {dim} "
            f"dims. At most {dim} codes can be linearly independent, so class "
            f"identity is not recoverable."
        )
    elif ci.get("code_matrix_rank") is not None and not ci.get(
        "linearly_separable"
    ):
        # Distinct codes, inside the ceiling, and still not separable. Reported
        # separately from over-subscription because the remedy differs: this is
        # not "too many classes for the width", it is an unlucky hash draw, and
        # nudging the width re-draws every code.
        rank, deficiency = ci["code_matrix_rank"], ci.get("rank_deficiency")
        problems.append(
            f"class_identity codes are linearly dependent: {total} classes "
            f"span only {rank} dimensions ({deficiency} class(es) are a "
            f"combination of others) despite fitting in {dim} dims and having "
            f"no identical codes. No linear readout can separate them."
        )

    if problems:
        detail = " ".join(problems)
        # Quote the width that would actually fit rather than naming the
        # setting. "Raise vector_dim" leaves the operator to work out the
        # arithmetic behind a fixed share of a fixed fraction, and to discover
        # by a second failed build that the number they picked was still short.
        # The memory figure goes with it because raising the width is not free:
        # the feature tensors are collected to the driver, so the cost is linear
        # in the width and lands in one process.
        sized = ""
        if vector_dim and total:
            needed = _min_vector_dim_for_segment_one(total)
            if needed > vector_dim:
                sized = (
                    f" At vector_dim={vector_dim} no split of segment 1 carries "
                    f"{total} classes; the smallest width that does is "
                    f"{needed} (~"
                    f"{_driver_gb_per_million_nodes(needed):.0f} GB of driver "
                    f"memory per 1M nodes)."
                )
        remedy = (
            "Raise feature_config.class_identity_dim (taken from within "
            "segment 1, so the vector width does not change) or "
            "feature_config.vector_dim (scales every segment)." + sized
            + " To build anyway, set feature_config."
            "allow_class_identity_oversubscription=true."
        )
        if allow_oversubscription:
            logger.warning(f"  {detail} {remedy}")
        else:
            raise ClassIdentityCapacityError(f"{detail} {remedy}")
        return

    if dim and total > 0.85 * dim:
        logger.warning(
            f"  class_identity segment is near capacity: {total} classes in "
            f"{dim} dims ({dim - total} left). Adding a source will "
            f"over-subscribe it."
        )


def _class_code_rank(
    codes: List[Tuple[int, ...]],
    dim: int,
) -> Optional[int]:
    """Rank of the multi-hot class-code matrix -- how many classes are recoverable.

    Each class occupies a set of slots, so the codes form a num_classes x dim
    0/1 matrix. A downstream layer reads class identity as a linear function of
    those dims, so it can tell all the classes apart exactly when the rows are
    linearly independent -- i.e. rank == num_classes. Anything less means some
    class's code is a weighted combination of others' and no linear readout can
    separate it, however distinct the codes look.

    Why measured rather than inferred: the two cheap proxies are each only
    NECESSARY. `num_classes <= dim` is the pigeonhole ceiling, and no two codes
    being identical rules out the degenerate case, but distinct codes under the
    ceiling can still be dependent (see the worked example at the call site).
    Those proxies are what this reported before, so a rank-deficient build
    passed every check.

    Uses numpy's SVD-based matrix_rank with its default tolerance, which scales
    with the largest singular value and the matrix dimensions -- appropriate
    here because the entries are exact 0/1 and any dependency is exact, so
    tolerance selection is not delicate.

    The slots are taken modulo dim by the caller, so an out-of-range index is a
    programming error rather than data; guarding would hide it.

    Returns:
        The rank, or None when dim is unknown (nothing to compute against).
    """
    if not dim or not codes:
        return None

    matrix = np.zeros((len(codes), dim), dtype=np.float64)
    for row, code in enumerate(codes):
        for slot in code:
            matrix[row, slot % dim] = 1.0

    return int(np.linalg.matrix_rank(matrix))


def compute_collision_report(
    numeric_slots: List[Dict],
    categorical_slots: List[Dict],
    class_slots: List[Dict],
    namespace_slots: List[Dict],
    hierarchy_slots: List[Dict],
    class_identity_dim: int = 0,
) -> Dict[str, Any]:
    """
    Compute hash collision statistics across all slot assignments.

    Returns a report with collision counts and rates per sub-segment.

    Args:
        class_identity_dim: Width of the class_identity sub-segment. Needed
            because class identity is a multi-hot code whose health depends
            on the segment width, not on slot occupancy alone -- see the
            class_identity branch below.
    """
    dim = class_identity_dim
    report: Dict[str, Any] = {}

    if numeric_slots:
        dims_used = [s["global_dim"] for s in numeric_slots]
        unique_dims = len(set(dims_used))
        total = len(dims_used)
        collisions = total - unique_dims
        report["numeric_properties"] = {
            "total_properties": total,
            "unique_slots": unique_dims,
            "collisions": collisions,
            "collision_rate": (
                round(collisions / total, 4) if total > 0 else 0.0
            ),
        }

    if class_slots:
        all_dims: List[int] = []
        for s in class_slots:
            all_dims.extend(s["global_dims"])
        unique_dims = len(set(all_dims))
        total = len(all_dims)

        # Class identity is a MULTI-HOT code: each class occupies
        # num_hashes slots, and what identifies it is the set of slots, not
        # any single one. So slot reuse is not identity loss -- 44 classes x
        # 4 hashes into 64 slots reuses ~67% of slot entries while still
        # giving all 44 classes distinct codes, full rank, and a condition
        # number near 12. Reported as `collisions` / `collision_rate` (slot
        # mapping 1.0) that number read as "two thirds of class identity is
        # aliased", which is false and alarming: with total > dim, pigeonhole
        # forces a high value no matter how healthy the code is.
        #
        # What actually costs identity is measured instead:
        #   - two classes sharing an identical code (genuinely
        #     indistinguishable),
        #   - the class count outgrowing the segment, past which no set of
        #     codes can be linearly separable, and
        #   - the codes being linearly DEPENDENT while still distinct and
        #     still under the ceiling, which is the case neither of the other
        #     two catches -- see the rank computation below.
        codes = [tuple(sorted(set(s["global_dims"]))) for s in class_slots]
        by_code: Dict[Tuple[int, ...], List[str]] = {}
        for slot, code in zip(class_slots, codes):
            by_code.setdefault(code, []).append(
                slot.get("pyg_name") or slot.get("class_uri", "?")
            )
        shared = sorted(
            (sorted(names) for names in by_code.values() if len(names) > 1),
            key=lambda names: names[0],
        )
        max_overlap = 0
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                overlap = len(set(codes[i]) & set(codes[j]))
                if overlap > max_overlap:
                    max_overlap = overlap

        num_classes = len(class_slots)
        code_rank = _class_code_rank(codes, dim)
        report["class_identity"] = {
            "total_classes": num_classes,
            "segment_dim": dim,
            # Raw occupancy, kept because it is a fact about the slots --
            # but named so it is not mistaken for lost identity.
            "total_hash_entries": total,
            "unique_slots": unique_dims,
            "slot_reuse": total - unique_dims,
            "slot_reuse_rate": (
                round((total - unique_dims) / total, 4) if total > 0 else 0.0
            ),
            # Identity, which is what a reader actually needs.
            "distinct_codes": len(by_code),
            "classes_sharing_a_code": shared,
            "max_pairwise_slot_overlap": max_overlap,
            # Capacity. A d-dim segment holds at most d linearly independent
            # codes, so this is a hard ceiling, not a heuristic.
            "capacity_classes": dim,
            "headroom_classes": (dim - num_classes) if dim else None,
            # The MEASURED rank of the num_classes x dim code matrix, and
            # separability derived from it.
            #
            # This was previously `num_classes <= dim and not shared`, which
            # states two NECESSARY conditions as if they were sufficient. They
            # are not: distinct codes under the ceiling can still be linearly
            # dependent. Four 4-hot codes {0,1,2,3}, {0,1,4,5}, {2,3,6,7},
            # {4,5,6,7} are pairwise distinct and fit in 8 dims, yet
            # A + D - B - C = 0, so the matrix is rank 3 and one class is a
            # blend of the others. A readout layer cannot recover it, and every
            # cheaper check reports the code healthy.
            #
            # Rank is the direct measure, so it is what gets reported. The
            # matrix is num_classes x dim (hundreds by hundreds at most) and
            # this runs once per build on the driver -- microseconds.
            "code_matrix_rank": code_rank,
            "rank_deficiency": (
                (num_classes - code_rank) if code_rank is not None else None
            ),
            "linearly_separable": (
                bool(dim) and code_rank is not None and code_rank == num_classes
            ),
        }

    if namespace_slots:
        dims_used = [s["global_dim"] for s in namespace_slots]
        unique_dims = len(set(dims_used))
        total = len(dims_used)
        collisions = total - unique_dims
        report["namespaces"] = {
            "total_namespaces": total,
            "unique_slots": unique_dims,
            "collisions": collisions,
            "collision_rate": (
                round(collisions / total, 4) if total > 0 else 0.0
            ),
        }

    return report
