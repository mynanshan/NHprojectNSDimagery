"""Small, explicit RSA helpers used by the NSD-Imagery notebooks."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr


def zscore_within_groups(
    patterns: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Standardize every voxel across trials separately within each run."""
    patterns = np.asarray(patterns, dtype=np.float32)
    groups = np.asarray(groups)
    if patterns.ndim != 2 or len(groups) != len(patterns):
        raise ValueError("patterns must be trials x voxels and match groups")
    normalized = np.empty_like(patterns)
    for group in dict.fromkeys(groups.tolist()):
        selected = groups == group
        block = patterns[selected]
        scale = block.std(axis=0, ddof=1, keepdims=True)
        scale[~np.isfinite(scale) | (scale == 0)] = 1
        normalized[selected] = (
            block - block.mean(axis=0, keepdims=True)
        ) / scale
    return normalized


def average_by_target(
    patterns: np.ndarray, target_numbers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average trial patterns in ascending target-number order."""
    patterns = np.asarray(patterns)
    target_numbers = np.asarray(target_numbers)
    if patterns.ndim != 2 or len(patterns) != len(target_numbers):
        raise ValueError("patterns must be trials x voxels and match target_numbers")
    targets = np.unique(target_numbers)
    averaged = np.stack(
        [patterns[target_numbers == target].mean(axis=0) for target in targets]
    )
    return averaged, targets


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    """Return a square correlation-distance RDM for rows of ``patterns``."""
    patterns = np.asarray(patterns)
    if patterns.ndim != 2 or len(patterns) < 2:
        raise ValueError("patterns must be a 2D array with at least two rows")
    rdm = squareform(pdist(patterns, metric="correlation"))
    np.fill_diagonal(rdm, 0)
    if not np.isfinite(rdm).all():
        raise ValueError("RDM contains non-finite values; check constant patterns")
    return rdm


def cosine_rdm(patterns: np.ndarray) -> np.ndarray:
    """Return a square cosine-distance RDM for rows of ``patterns``."""
    patterns = np.asarray(patterns)
    if patterns.ndim != 2 or len(patterns) < 2:
        raise ValueError("patterns must be a 2D array with at least two rows")
    rdm = squareform(pdist(patterns, metric="cosine"))
    np.fill_diagonal(rdm, 0)
    if not np.isfinite(rdm).all():
        raise ValueError("RDM contains non-finite values; check zero-length patterns")
    return rdm


def target_rdm(
    patterns: np.ndarray, target_numbers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average repeats and return ``(correlation RDM, ordered targets)``."""
    averaged, targets = average_by_target(patterns, target_numbers)
    return correlation_rdm(averaged), targets


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """Return the strict upper triangle of a square matrix."""
    rdm = np.asarray(rdm)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError("rdm must be square")
    return rdm[np.triu_indices_from(rdm, k=1)]


def rdm_spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation between two RDM upper triangles."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("RDMs must have the same shape")
    return float(spearmanr(upper_triangle(first), upper_triangle(second)).statistic)


@lru_cache(maxsize=None)
def _permuted_pair_indices(n_targets: int) -> np.ndarray:
    """Map every target permutation to reordered upper-triangle positions."""
    pairs = list(combinations(range(n_targets), 2))
    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    rows = []
    for order in permutations(range(n_targets)):
        rows.append(
            [pair_lookup[tuple(sorted((order[i], order[j])))] for i, j in pairs]
        )
    return np.asarray(rows, dtype=np.int16)


def exact_label_permutation_test(
    first: np.ndarray, second: np.ndarray
) -> dict[str, float | int]:
    """Exact RDM label-permutation test, practical for the six NSD targets."""
    first = np.asarray(first)
    second = np.asarray(second)
    if (
        first.shape != second.shape
        or first.ndim != 2
        or first.shape[0] != first.shape[1]
    ):
        raise ValueError("RDMs must be square and have the same shape")
    n_targets = first.shape[0]
    if n_targets > 8:
        raise ValueError("Exact label enumeration is limited to at most 8 targets")

    first_ranks = rankdata(upper_triangle(first)).astype(float)
    second_ranks = rankdata(upper_triangle(second)).astype(float)
    first_ranks -= first_ranks.mean()
    second_ranks -= second_ranks.mean()
    first_norm = np.linalg.norm(first_ranks)
    second_norm = np.linalg.norm(second_ranks)
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Permutation test requires non-constant RDM values")
    pair_indices = _permuted_pair_indices(n_targets)
    null = second_ranks[pair_indices] @ first_ranks / (first_norm * second_norm)
    observed = float(null[0])
    tolerance = 1e-12
    return {
        "observed": observed,
        "p_greater": float(np.mean(null >= observed - tolerance)),
        "p_less": float(np.mean(null <= observed + tolerance)),
        "p_two_sided": float(
            np.mean(np.abs(null) >= abs(observed) - tolerance)
        ),
        "n_permutations": len(null),
    }


def exact_sign_flip_test(values: np.ndarray) -> dict[str, float | int]:
    """Exact one-sample sign-flip test using the mean as the statistic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must contain at least one finite observation")
    if len(values) > 20:
        raise ValueError("Exact sign enumeration is limited to at most 20 values")

    observed = float(values.mean())
    null = np.asarray(
        [np.mean(values * signs) for signs in product((-1, 1), repeat=len(values))]
    )
    tolerance = 1e-12
    return {
        "observed_mean": observed,
        "p_greater": float(np.mean(null >= observed - tolerance)),
        "p_less": float(np.mean(null <= observed + tolerance)),
        "p_two_sided": float(
            np.mean(np.abs(null) >= abs(observed) - tolerance)
        ),
        "n_permutations": len(null),
    }


def balanced_split_reliability(
    patterns: np.ndarray,
    target_numbers: np.ndarray,
    *,
    n_splits: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Repeatedly split every target's trials into balanced random halves."""
    patterns = np.asarray(patterns)
    target_numbers = np.asarray(target_numbers)
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    targets = np.unique(target_numbers)
    indices = {target: np.flatnonzero(target_numbers == target) for target in targets}
    if any(len(index) < 2 or len(index) % 2 for index in indices.values()):
        raise ValueError("Every target needs an even number of at least two trials")

    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(n_splits):
        first_half = []
        second_half = []
        for target in targets:
            shuffled = rng.permutation(indices[target])
            middle = len(shuffled) // 2
            first_half.append(patterns[shuffled[:middle]].mean(axis=0))
            second_half.append(patterns[shuffled[middle:]].mean(axis=0))
        correlations.append(
            rdm_spearman(
                correlation_rdm(np.stack(first_half)),
                correlation_rdm(np.stack(second_half)),
            )
        )
    return np.asarray(correlations)


def independent_group_reliability(
    patterns: np.ndarray,
    target_numbers: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compare target RDMs from exactly two independent groups or runs."""
    patterns = np.asarray(patterns)
    target_numbers = np.asarray(target_numbers)
    groups = np.asarray(groups)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    if len(unique_groups) != 2:
        raise ValueError("Exactly two groups are required")
    rdms = []
    target_orders = []
    for group in unique_groups:
        selected = groups == group
        rdm, targets = target_rdm(patterns[selected], target_numbers[selected])
        rdms.append(rdm)
        target_orders.append(targets)
    if not np.array_equal(target_orders[0], target_orders[1]):
        raise ValueError("The two groups do not contain identical targets")
    return rdm_spearman(rdms[0], rdms[1])
