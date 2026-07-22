"""Small, explicit RSA helpers used by the NSD-Imagery notebooks."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
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


def nearest_centroid_predict(
    train_patterns: np.ndarray,
    train_targets: np.ndarray,
    test_patterns: np.ndarray,
) -> np.ndarray:
    """Predict targets by nearest correlation-distance training centroid."""
    train_patterns = np.asarray(train_patterns)
    train_targets = np.asarray(train_targets)
    test_patterns = np.asarray(test_patterns)
    if train_patterns.ndim != 2 or test_patterns.ndim != 2:
        raise ValueError("train_patterns and test_patterns must be 2D")
    if len(train_patterns) != len(train_targets):
        raise ValueError("train_patterns and train_targets must have equal length")
    if train_patterns.shape[1] != test_patterns.shape[1]:
        raise ValueError("train and test patterns must have the same voxel count")
    centroids, targets = average_by_target(train_patterns, train_targets)
    distances = cdist(test_patterns, centroids, metric="correlation")
    if not np.isfinite(distances).all():
        raise ValueError("Non-finite centroid distances; check constant patterns")
    return targets[np.argmin(distances, axis=1)]


def exact_class_label_permutation_test(
    predicted: np.ndarray, observed: np.ndarray
) -> dict[str, float | int]:
    """Test classification accuracy by enumerating predicted-label mappings."""
    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    if predicted.ndim != 1 or observed.ndim != 1 or len(predicted) != len(observed):
        raise ValueError("predicted and observed must be equal-length vectors")
    classes = np.unique(np.concatenate([predicted, observed]))
    if len(classes) > 8:
        raise ValueError("Exact label enumeration is limited to at most 8 classes")
    class_to_index = {value: index for index, value in enumerate(classes)}
    predicted_indices = np.asarray([class_to_index[value] for value in predicted])
    null = []
    for order in permutations(classes.tolist()):
        remapped = np.asarray(order)[predicted_indices]
        null.append(np.mean(remapped == observed))
    null = np.asarray(null)
    accuracy = float(np.mean(predicted == observed))
    return {
        "accuracy": accuracy,
        "chance": float(1 / len(classes)),
        "p_greater": float(np.mean(null >= accuracy - 1e-12)),
        "n_permutations": len(null),
    }


def balanced_split_identification(
    patterns: np.ndarray,
    target_numbers: np.ndarray,
    *,
    n_splits: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Return symmetric nearest-centroid accuracies from balanced trial halves."""
    patterns = np.asarray(patterns)
    target_numbers = np.asarray(target_numbers)
    if patterns.ndim != 2 or len(patterns) != len(target_numbers):
        raise ValueError("patterns must be trials x voxels and match targets")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    targets = np.unique(target_numbers)
    indices = {target: np.flatnonzero(target_numbers == target) for target in targets}
    if any(len(index) < 2 or len(index) % 2 for index in indices.values()):
        raise ValueError("Every target needs an even number of at least two trials")

    rng = np.random.default_rng(seed)
    accuracies = []
    for _ in range(n_splits):
        first_indices = []
        second_indices = []
        for target in targets:
            shuffled = rng.permutation(indices[target])
            middle = len(shuffled) // 2
            first_indices.extend(shuffled[:middle])
            second_indices.extend(shuffled[middle:])
        first_indices = np.asarray(first_indices)
        second_indices = np.asarray(second_indices)
        first_to_second = nearest_centroid_predict(
            patterns[first_indices], target_numbers[first_indices], patterns[second_indices]
        )
        second_to_first = nearest_centroid_predict(
            patterns[second_indices], target_numbers[second_indices], patterns[first_indices]
        )
        accuracies.append(
            np.mean(
                np.concatenate([
                    first_to_second == target_numbers[second_indices],
                    second_to_first == target_numbers[first_indices],
                ])
            )
        )
    return np.asarray(accuracies)


def crossvalidated_dot_rdm(
    first_patterns: np.ndarray,
    first_targets: np.ndarray,
    second_patterns: np.ndarray,
    second_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crossvalidated dot-product dissimilarity from two data partitions.

    A positive pairwise value means the target contrast points in a consistent
    voxel-space direction in the two partitions. Under independent noise and
    no true target difference, its expectation is zero. This is an unwhitened
    relative of the crossnobis distance, scaled by the number of voxels.
    """
    first_means, first_order = average_by_target(first_patterns, first_targets)
    second_means, second_order = average_by_target(second_patterns, second_targets)
    if not np.array_equal(first_order, second_order):
        raise ValueError("The two partitions must contain identical targets")
    if first_means.shape[1] != second_means.shape[1]:
        raise ValueError("The two partitions must have the same voxel count")
    n_targets, n_voxels = first_means.shape
    rdm = np.zeros((n_targets, n_targets), dtype=float)
    for first_index, second_index in combinations(range(n_targets), 2):
        first_delta = first_means[first_index] - first_means[second_index]
        second_delta = second_means[first_index] - second_means[second_index]
        value = float(first_delta @ second_delta / n_voxels)
        rdm[first_index, second_index] = value
        rdm[second_index, first_index] = value
    return rdm, first_order


def leave_one_target_out_rdm_spearman(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Return one RDM correlation after omitting each target in turn."""
    first = np.asarray(first)
    second = np.asarray(second)
    if (
        first.shape != second.shape
        or first.ndim != 2
        or first.shape[0] != first.shape[1]
        or first.shape[0] < 4
    ):
        raise ValueError("RDMs must be matching square matrices with >=4 targets")
    correlations = []
    for omitted in range(first.shape[0]):
        keep = np.arange(first.shape[0]) != omitted
        correlations.append(rdm_spearman(first[np.ix_(keep, keep)], second[np.ix_(keep, keep)]))
    return np.asarray(correlations)


def rdm_noise_ceiling(rdms: np.ndarray) -> dict[str, np.ndarray | float]:
    """Estimate leave-one-subject-out lower and inclusive upper RSA ceilings."""
    rdms = np.asarray(rdms, dtype=float)
    if rdms.ndim != 3 or rdms.shape[1] != rdms.shape[2] or len(rdms) < 3:
        raise ValueError("rdms must be subjects x targets x targets with >=3 subjects")
    vectors = np.stack([rankdata(upper_triangle(rdm)) for rdm in rdms])
    lower = []
    upper = []
    for subject in range(len(vectors)):
        others = np.arange(len(vectors)) != subject
        lower.append(np.corrcoef(vectors[subject], vectors[others].mean(axis=0))[0, 1])
        upper.append(np.corrcoef(vectors[subject], vectors.mean(axis=0))[0, 1])
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return {
        "lower_by_subject": lower,
        "upper_by_subject": upper,
        "lower_mean": float(lower.mean()),
        "upper_mean": float(upper.mean()),
    }


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
