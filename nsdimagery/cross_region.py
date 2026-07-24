"""Helpers for vision-to-imagery transfer across NSD stream ROIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


# Integer values documented by the NSD ``streams`` color table.
STREAM_ROI_LABELS = {
    "early": (1,),
    "midventral": (2,),
    "midlateral": (3,),
    "midparietal": (4,),
    "ventral": (5,),
    "lateral": (6,),
    "parietal": (7,),
    # Disjoint primary regions for the collaborator-inspired analysis.
    "visual_streams": (1, 2, 3, 5, 6),
    "dorsal_parietal": (4, 7),
}


def stream_roi_masks(
    streams: np.ndarray,
    roi_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build named boolean masks from a volumetric NSD streams atlas."""
    streams = np.asarray(streams)
    if streams.ndim != 3:
        raise ValueError("streams must be a three-dimensional label volume")
    names = tuple(STREAM_ROI_LABELS if roi_names is None else roi_names)
    unknown = sorted(set(names) - set(STREAM_ROI_LABELS))
    if unknown:
        raise ValueError(f"Unknown stream ROI names: {unknown}")
    masks = {
        name: np.isin(streams, STREAM_ROI_LABELS[name]) for name in names
    }
    empty = [name for name, mask in masks.items() if not mask.any()]
    if empty:
        raise ValueError(f"Stream ROIs contain no voxels: {empty}")
    return masks


def predictive_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
    reference_mean: np.ndarray,
) -> float:
    """Multivariate predictive R-squared relative to a training-set mean."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    reference_mean = np.asarray(reference_mean, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have the same shape")
    if reference_mean.shape != observed.shape[1:]:
        raise ValueError("reference_mean must match the response dimensions")
    denominator = np.sum((observed - reference_mean) ** 2)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("predictive R2 requires non-constant finite responses")
    return float(1 - np.sum((observed - predicted) ** 2) / denominator)


@dataclass
class RegionAlignment:
    """A fitted parietal-to-visual low-rank ridge alignment."""

    parietal_pca: PCA
    visual_pca: PCA
    ridge: Ridge
    n_components: int
    alpha: float
    cv_r2: float

    def transform_visual(self, patterns: np.ndarray) -> np.ndarray:
        return self.visual_pca.transform(np.asarray(patterns))

    def transform_parietal(self, patterns: np.ndarray) -> np.ndarray:
        return self.parietal_pca.transform(np.asarray(patterns))

    def predict_visual_from_parietal(self, patterns: np.ndarray) -> np.ndarray:
        latent = self.transform_parietal(patterns)
        return self.ridge.predict(latent)


def _validate_alignment_inputs(
    parietal_patterns: np.ndarray,
    visual_patterns: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parietal = np.asarray(parietal_patterns, dtype=np.float64)
    visual = np.asarray(visual_patterns, dtype=np.float64)
    groups = np.asarray(groups)
    if parietal.ndim != 2 or visual.ndim != 2:
        raise ValueError("Alignment patterns must be samples x voxels")
    if len(parietal) != len(visual) or len(parietal) != len(groups):
        raise ValueError("Parietal, visual, and group arrays must align by sample")
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three groups are required for alignment CV")
    if not np.isfinite(parietal).all() or not np.isfinite(visual).all():
        raise ValueError("Alignment patterns must be finite")
    return parietal, visual, groups


def fit_region_alignment(
    parietal_patterns: np.ndarray,
    visual_patterns: np.ndarray,
    groups: np.ndarray,
    *,
    component_grid: Iterable[int] = (4, 8, 16),
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[RegionAlignment, list[dict[str, float | int]]]:
    """Fit a vision-only parietal-to-visual map with leave-group-out CV.

    PCA and ridge are refitted inside every fold. Hyperparameters are selected
    only from paired vision data. A smaller latent dimension, followed by a
    larger ridge penalty, wins an exact score tie.
    """
    parietal, visual, groups = _validate_alignment_inputs(
        parietal_patterns, visual_patterns, groups
    )
    components = tuple(sorted({int(value) for value in component_grid}))
    alphas = tuple(sorted({float(value) for value in alpha_grid}))
    if not components or components[0] < 2:
        raise ValueError("component_grid must contain integers of at least two")
    if not alphas or alphas[0] <= 0:
        raise ValueError("alpha_grid must contain positive values")

    unique_groups = np.unique(groups)
    records: list[dict[str, float | int]] = []
    for n_components in components:
        fold_train_sizes = [
            int(np.sum(groups != held_out)) for held_out in unique_groups
        ]
        maximum = min(
            min(fold_train_sizes) - 1,
            parietal.shape[1],
            visual.shape[1],
        )
        if n_components > maximum:
            continue
        for alpha in alphas:
            fold_scores = []
            for held_out in unique_groups:
                train = groups != held_out
                test = ~train
                parietal_pca = PCA(
                    n_components=n_components, whiten=True, svd_solver="full"
                )
                visual_pca = PCA(
                    n_components=n_components, whiten=True, svd_solver="full"
                )
                parietal_train = parietal_pca.fit_transform(parietal[train])
                visual_train = visual_pca.fit_transform(visual[train])
                visual_test = visual_pca.transform(visual[test])
                ridge = Ridge(alpha=alpha)
                ridge.fit(parietal_train, visual_train)
                prediction = ridge.predict(parietal_pca.transform(parietal[test]))
                fold_scores.append(
                    predictive_r2(
                        visual_test,
                        prediction,
                        visual_train.mean(axis=0),
                    )
                )
            records.append(
                {
                    "n_components": n_components,
                    "alpha": alpha,
                    "mean_cv_r2": float(np.mean(fold_scores)),
                    "median_cv_r2": float(np.median(fold_scores)),
                }
            )
    if not records:
        raise ValueError("No alignment hyperparameter combination is feasible")
    best = max(
        records,
        key=lambda row: (
            row["mean_cv_r2"],
            -row["n_components"],
            row["alpha"],
        ),
    )
    n_components = int(best["n_components"])
    alpha = float(best["alpha"])
    parietal_pca = PCA(
        n_components=n_components, whiten=True, svd_solver="full"
    )
    visual_pca = PCA(
        n_components=n_components, whiten=True, svd_solver="full"
    )
    parietal_latent = parietal_pca.fit_transform(parietal)
    visual_latent = visual_pca.fit_transform(visual)
    ridge = Ridge(alpha=alpha)
    ridge.fit(parietal_latent, visual_latent)
    return (
        RegionAlignment(
            parietal_pca=parietal_pca,
            visual_pca=visual_pca,
            ridge=ridge,
            n_components=n_components,
            alpha=alpha,
            cv_r2=float(best["mean_cv_r2"]),
        ),
        records,
    )
