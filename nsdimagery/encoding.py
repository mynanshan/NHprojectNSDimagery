"""Helpers for training perception encoders and testing them on NSD-Imagery.

The primary scientific guardrail in this module is that train/validation/test
partitions are defined over *unique image identities*. Repeated fMRI trials are
averaged before model fitting and can never occur on both sides of a split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import ndtr


CORE_NSD_SESSIONS = (40, 40, 32, 30, 40, 32, 40, 30)


def core_nsd_trial_ids(
    experiment_design: Mapping[str, np.ndarray],
    subject: int,
    *,
    n_sessions: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return core-NSD trial IDs in the order of the beta session files.

    Returns ``(trial_10k_indices, trial_73k_ids, subject_73k_ids)``. The first
    array is zero-based; the two 73K IDs are one-based to match NSD metadata.
    Each completed core-NSD session contributes exactly 750 stimulus betas.
    """
    if subject not in range(1, 9):
        raise ValueError("subject must be 1 through 8")
    available_sessions = CORE_NSD_SESSIONS[subject - 1]
    if n_sessions is None:
        n_sessions = available_sessions
    if not 1 <= n_sessions <= available_sessions:
        raise ValueError(
            f"subject {subject} has 1 through {available_sessions} sessions"
        )

    subjectim = np.asarray(experiment_design["subjectim"])
    masterordering = np.asarray(experiment_design["masterordering"]).reshape(-1)
    if subjectim.shape != (8, 10000):
        raise ValueError(f"subjectim has unexpected shape {subjectim.shape}")
    if len(masterordering) != 30000:
        raise ValueError(
            f"masterordering has unexpected length {len(masterordering)}"
        )

    trial_10k = masterordering[: n_sessions * 750].astype(np.int64) - 1
    if np.any((trial_10k < 0) | (trial_10k >= 10000)):
        raise ValueError("masterordering must contain one-based 10K indices")
    subject_73k = subjectim[subject - 1].astype(np.int64)
    if np.any((subject_73k < 1) | (subject_73k > 73000)):
        raise ValueError("subjectim must contain one-based 73K IDs")
    return trial_10k, subject_73k[trial_10k], subject_73k


def load_core_nsd_trial_ids(
    data_root: str | Path,
    subject: int,
    *,
    n_sessions: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``nsd_expdesign.mat`` and call :func:`core_nsd_trial_ids`."""
    design_path = (
        Path(data_root)
        / "nsddata"
        / "experiments"
        / "nsd"
        / "nsd_expdesign.mat"
    )
    if not design_path.is_file():
        raise FileNotFoundError(design_path)
    design = loadmat(design_path, variable_names=("subjectim", "masterordering"))
    return core_nsd_trial_ids(design, subject, n_sessions=n_sessions)


def assign_image_splits(
    observed_10k_indices: Iterable[int],
    *,
    mode: str = "shared1000",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Assign deterministic image-level train/validation/test partitions.

    ``shared1000`` reserves the first 1,000 subject-image positions for test;
    these are the NSD images shared by all eight subjects. ``random`` makes a
    conventional random test partition. In either mode, every input identity
    must be unique.
    """
    indices = np.asarray(list(observed_10k_indices), dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 3:
        raise ValueError("at least three one-dimensional image indices are needed")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("observed_10k_indices must contain unique images")
    if np.any((indices < 0) | (indices >= 10000)):
        raise ValueError("10K indices must be zero-based values from 0 to 9999")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if mode not in {"shared1000", "random"}:
        raise ValueError("mode must be shared1000 or random")
    if mode == "random" and not 0 < test_fraction < 1 - validation_fraction:
        raise ValueError("test_fraction leaves no training images")

    rng = np.random.default_rng(seed)
    split = np.full(len(indices), "train", dtype="U10")
    if mode == "shared1000":
        test = indices < 1000
        candidates = np.flatnonzero(~test)
        split[test] = "test"
        if not test.any():
            raise ValueError(
                "No shared1000 test images were observed; use a complete subject "
                "or --split-mode random for a pipeline smoke test"
            )
        n_validation = max(1, int(round(validation_fraction * len(candidates))))
    else:
        candidates = np.arange(len(indices))
        shuffled = rng.permutation(candidates)
        n_test = max(1, int(round(test_fraction * len(indices))))
        test_rows = shuffled[:n_test]
        split[test_rows] = "test"
        candidates = shuffled[n_test:]
        n_validation = max(1, int(round(validation_fraction * len(indices))))

    validation_rows = rng.permutation(candidates)[:n_validation]
    split[validation_rows] = "validation"
    if not all(np.any(split == name) for name in ("train", "validation", "test")):
        raise ValueError("split construction produced an empty partition")
    return split


def voxelwise_prediction_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson correlation and R-squared for every voxel column."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("observed and predicted must be equal images x voxels arrays")
    observed_centered = observed - observed.mean(axis=0, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(observed_centered, axis=0) * np.linalg.norm(
        predicted_centered, axis=0
    )
    correlation = np.full(observed.shape[1], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0)
    correlation[valid] = np.sum(
        observed_centered[:, valid] * predicted_centered[:, valid], axis=0
    ) / denominator[valid]

    residual_sum = np.sum((observed - predicted) ** 2, axis=0)
    total_sum = np.sum(observed_centered**2, axis=0)
    r_squared = np.full(observed.shape[1], np.nan, dtype=np.float64)
    valid_r2 = np.isfinite(total_sum) & (total_sum > 0)
    r_squared[valid_r2] = 1 - residual_sum[valid_r2] / total_sum[valid_r2]
    return correlation, r_squared


def average_predictions_by_target(
    manifest: pd.DataFrame,
    predicted: np.ndarray,
    labels: pd.DataFrame,
) -> np.ndarray:
    """Average image samples and align predictions to a measured target table."""
    predicted = np.asarray(predicted)
    if predicted.ndim != 2 or len(manifest) != len(predicted):
        raise ValueError("manifest and predicted betas must be aligned 2-D arrays")
    required = {"stimulus_set", "target_number"}
    if not required.issubset(manifest) or not required.issubset(labels):
        raise ValueError(f"manifest and labels must contain columns {required}")

    averaged = []
    for row in labels.itertuples():
        selected = (
            manifest["stimulus_set"].eq(row.stimulus_set)
            & manifest["target_number"].eq(int(row.target_number))
        ).to_numpy()
        if not selected.any():
            raise ValueError(
                "No prediction found for target "
                f"{row.stimulus_set}{int(row.target_number)}"
            )
        averaged.append(predicted[selected].mean(axis=0))
    return np.stack(averaged)


def fit_ridge_weights(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    *,
    device: str = "cpu",
    voxel_chunk_size: int = 4096,
) -> np.ndarray:
    """Fit a zero-intercept multi-output ridge readout in voxel chunks.

    Callers center/standardize their arrays before this function. A Cholesky
    factor is reused across chunks, which makes fitting thousands of voxel
    outputs practical on either CPU or GPU.
    """
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned two-dimensional arrays")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if voxel_chunk_size < 1:
        raise ValueError("voxel_chunk_size must be positive")
    try:
        import torch
    except ModuleNotFoundError:
        if device not in {"auto", "cpu"}:
            raise RuntimeError("PyTorch is required for GPU ridge fitting") from None
        gram = features.T @ features
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        cholesky = np.linalg.cholesky(gram)
        weights = np.empty(
            (features.shape[1], targets.shape[1]), dtype=np.float32
        )
        for start in range(0, targets.shape[1], voxel_chunk_size):
            stop = min(start + voxel_chunk_size, targets.shape[1])
            right_hand_side = features.T @ targets[:, start:stop]
            intermediate = np.linalg.solve(cholesky, right_hand_side)
            weights[:, start:stop] = np.linalg.solve(
                cholesky.T, intermediate
            )
        return weights

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    x = torch.from_numpy(features).to(torch_device)
    gram = x.T @ x
    gram.diagonal().add_(float(alpha))
    cholesky = torch.linalg.cholesky(gram)
    weights = np.empty((features.shape[1], targets.shape[1]), dtype=np.float32)
    for start in range(0, targets.shape[1], voxel_chunk_size):
        stop = min(start + voxel_chunk_size, targets.shape[1])
        y = torch.from_numpy(targets[:, start:stop]).to(torch_device)
        right_hand_side = x.T @ y
        solved = torch.cholesky_solve(right_hand_side, cholesky)
        weights[:, start:stop] = solved.cpu().numpy()
    return weights


def apply_feature_transform(
    features: np.ndarray, model: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Apply the saved standardization and PCA transform of an encoder."""
    features = np.asarray(features, dtype=np.float32)
    expected = int(np.asarray(model["feature_mean"]).shape[0])
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(
            f"features have shape {features.shape}; expected columns {expected}"
        )
    standardized = (
        features - np.asarray(model["feature_mean"], dtype=np.float32)
    ) / np.asarray(model["feature_scale"], dtype=np.float32)
    centered = standardized - np.asarray(model["pca_mean"], dtype=np.float32)
    return centered @ np.asarray(model["pca_components"], dtype=np.float32).T


def predict_with_encoder(
    features: np.ndarray,
    model: Mapping[str, np.ndarray],
    *,
    standardized_betas: bool = True,
) -> np.ndarray:
    """Predict image-evoked beta patterns from cached image features.

    Ridge encoders contain only ``ridge_weights``. A nonlinear residual
    encoder additionally stores a one-hidden-layer GELU readout. Its output is
    added to the frozen ridge prediction, so a zero residual exactly recovers
    the auditable linear baseline.
    """
    transformed = apply_feature_transform(features, model)
    predicted = transformed @ np.asarray(model["ridge_weights"], dtype=np.float32)
    nonlinear_keys = {
        "nonlinear_input_mean",
        "nonlinear_input_scale",
        "nonlinear_hidden_weight",
        "nonlinear_hidden_bias",
        "nonlinear_output_weight",
        "nonlinear_output_bias",
    }
    present = nonlinear_keys.intersection(model)
    if present and present != nonlinear_keys:
        missing = ", ".join(sorted(nonlinear_keys - present))
        raise ValueError(f"Nonlinear encoder is incomplete; missing {missing}")
    if present:
        nonlinear_input = (
            transformed
            - np.asarray(model["nonlinear_input_mean"], dtype=np.float32)
        ) / np.asarray(model["nonlinear_input_scale"], dtype=np.float32)
        hidden = (
            nonlinear_input
            @ np.asarray(model["nonlinear_hidden_weight"], dtype=np.float32).T
            + np.asarray(model["nonlinear_hidden_bias"], dtype=np.float32)
        )
        # PyTorch's default GELU is x * Phi(x). ndtr is the standard-normal CDF.
        hidden = hidden * ndtr(hidden)
        predicted = (
            predicted
            + hidden
            @ np.asarray(model["nonlinear_output_weight"], dtype=np.float32).T
            + np.asarray(model["nonlinear_output_bias"], dtype=np.float32)
        )
    if not standardized_betas:
        predicted = (
            predicted * np.asarray(model["beta_scale"], dtype=np.float32)
            + np.asarray(model["beta_mean"], dtype=np.float32)
        )
    return predicted


def kernel_ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    alpha: float,
    kernel: str = "linear",
    gamma_scale: float = 1.0,
) -> np.ndarray:
    """Fit a centered dual kernel ridge model and predict held-out samples.

    Feature centering is learned from ``train_features`` only. Per-component
    scaling is deliberately not estimated from 5--11 targets: the fixed
    core-NSD PCA variances are more stable than tiny-sample standard
    deviations. The linear kernel is normalized to mean unit diagonal; RBF
    distances are normalized by the median training-pair distance.
    """
    train_features = np.asarray(train_features, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    if (
        train_features.ndim != 2
        or test_features.ndim != 2
        or train_targets.ndim != 2
        or len(train_features) != len(train_targets)
        or train_features.shape[1] != test_features.shape[1]
    ):
        raise ValueError(
            "train/test features and train targets must be aligned 2-D arrays"
        )
    if len(train_features) < 2 or len(test_features) < 1:
        raise ValueError("kernel ridge requires at least two training samples")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if kernel not in {"linear", "rbf"}:
        raise ValueError("kernel must be linear or rbf")
    if gamma_scale <= 0 or not np.isfinite(gamma_scale):
        raise ValueError("gamma_scale must be finite and positive")

    feature_mean = train_features.mean(axis=0, keepdims=True)
    x_train = train_features - feature_mean
    x_test = test_features - feature_mean

    if kernel == "linear":
        train_kernel = x_train @ x_train.T
        test_kernel = x_test @ x_train.T
        kernel_scale = float(np.mean(np.diag(train_kernel)))
        if not np.isfinite(kernel_scale) or kernel_scale <= 0:
            raise ValueError("training features have no finite variation")
        train_kernel /= kernel_scale
        test_kernel /= kernel_scale
    else:
        train_norm = np.sum(x_train**2, axis=1)
        test_norm = np.sum(x_test**2, axis=1)
        train_distance = (
            train_norm[:, None] + train_norm[None, :] - 2 * x_train @ x_train.T
        )
        test_distance = (
            test_norm[:, None] + train_norm[None, :] - 2 * x_test @ x_train.T
        )
        np.maximum(train_distance, 0, out=train_distance)
        np.maximum(test_distance, 0, out=test_distance)
        upper = train_distance[np.triu_indices(len(train_distance), k=1)]
        positive = upper[np.isfinite(upper) & (upper > 0)]
        if not len(positive):
            raise ValueError("training features have no finite pairwise distances")
        distance_scale = float(np.median(positive))
        train_kernel = np.exp(-gamma_scale * train_distance / distance_scale)
        test_kernel = np.exp(-gamma_scale * test_distance / distance_scale)

    target_mean = train_targets.mean(axis=0, keepdims=True)
    centered_targets = train_targets - target_mean
    regularized = train_kernel.copy()
    regularized.flat[:: len(regularized) + 1] += alpha
    dual_weights = np.linalg.solve(regularized, centered_targets)
    return (test_kernel @ dual_weights + target_mean).astype(np.float32)


def leave_one_target_out_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    kernel: str = "linear",
    gamma_scale: float = 1.0,
) -> np.ndarray:
    """Return predictions where each target identity is held out in turn."""
    features = np.asarray(features)
    targets = np.asarray(targets)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned 2-D arrays")
    if len(features) < 3:
        raise ValueError("leave-one-target-out prediction needs at least three targets")
    predicted = np.empty_like(targets, dtype=np.float32)
    for held_out in range(len(features)):
        train = np.arange(len(features)) != held_out
        predicted[held_out] = kernel_ridge_predict(
            features[train],
            targets[train],
            features[[held_out]],
            alpha=alpha,
            kernel=kernel,
            gamma_scale=gamma_scale,
        )[0]
    return predicted


def load_encoder_model(path: str | Path) -> dict[str, np.ndarray | dict]:
    """Load an encoder ``.npz`` and decode its JSON metadata."""
    with np.load(path, allow_pickle=False) as archive:
        model: dict[str, np.ndarray | dict] = {
            key: archive[key] for key in archive.files
        }
    metadata_value = np.asarray(model.pop("metadata_json")).item()
    model["metadata"] = json.loads(str(metadata_value))
    return model


def pool_transformer_hidden_state(
    hidden_state,
    *,
    grid_size: tuple[int, int],
    pyramid_levels: Iterable[int] = (1, 2),
    include_cls: bool = True,
):
    """Convert ViT tokens to a compact, spatial-pyramid feature vector."""
    import torch
    import torch.nn.functional as functional

    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must be batch x tokens x channels")
    height, width = (int(value) for value in grid_size)
    n_patches = height * width
    if hidden_state.shape[1] < n_patches + 1:
        raise ValueError("hidden_state does not contain CLS plus the expected patches")
    patches = hidden_state[:, -n_patches:, :]
    grid = patches.transpose(1, 2).reshape(
        hidden_state.shape[0], hidden_state.shape[2], height, width
    )
    pieces = [hidden_state[:, 0, :]] if include_cls else []
    for level in pyramid_levels:
        level = int(level)
        if level < 1:
            raise ValueError("pyramid levels must be positive")
        pieces.append(functional.adaptive_avg_pool2d(grid, (level, level)).flatten(1))
    if not pieces:
        raise ValueError("at least one CLS or pyramid feature is required")
    return torch.cat(pieces, dim=1)


def transformer_patch_grid(
    image_size: int | Iterable[int],
    patch_size: int | Iterable[int],
) -> tuple[int, int]:
    """Infer the transformer patch grid from the processed pixel dimensions.

    ``image_size`` must describe the tensor actually sent to the model, rather
    than the nominal image size stored in the model configuration. Hugging Face
    processors can resize or crop to a different size.
    """
    if np.isscalar(image_size):
        height = width = int(image_size)
    else:
        dimensions = tuple(int(value) for value in image_size)
        if len(dimensions) != 2:
            raise ValueError("image_size must be a scalar or height-width pair")
        height, width = dimensions
    if np.isscalar(patch_size):
        patch_height = patch_width = int(patch_size)
    else:
        patches = tuple(int(value) for value in patch_size)
        if len(patches) != 2:
            raise ValueError("patch_size must be a scalar or height-width pair")
        patch_height, patch_width = patches
    if min(height, width, patch_height, patch_width) < 1:
        raise ValueError("image and patch dimensions must be positive")
    if height % patch_height or width % patch_width:
        raise ValueError(
            f"Processed image {(height, width)} is not divisible by patch size "
            f"{(patch_height, patch_width)}"
        )
    return height // patch_height, width // patch_width


def measured_nsdimagery_target_data(
    data_root: str | Path,
    subject: int,
    task: str,
    stimulus_sets: Iterable[str],
    *,
    expected_coordinates: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Return target means, independent repeat halves, labels, and ROI masks.

    The response preprocessing matches the paper workflow: every voxel is
    Z-scored within run before repetitions are averaged. Imagery halves are
    the two acquisition runs. Vision has one run, so alternating occurrences
    of each target form two deterministic four-trial halves.
    """
    from .analysis import zscore_within_groups
    from .io import (
        build_event_table,
        extract_masked_betas,
        load_roi,
        mask_at_coordinates,
        paper_visual_roi_masks,
        paths_for_subject,
    )

    if task not in {"vision", "imagery"}:
        raise ValueError("task must be vision or imagery")
    stimulus_sets = tuple(stimulus_sets)
    if not stimulus_sets or any(name not in {"A", "B", "C"} for name in stimulus_sets):
        raise ValueError("stimulus_sets must contain A, B, or C")
    events = build_event_table(data_root, subject)
    general, _ = load_roi(data_root, subject, "nsdgeneral")
    prf, _ = load_roi(data_root, subject, "prf-visualrois")
    all_betas, coordinates = extract_masked_betas(
        paths_for_subject(data_root, subject)["beta"], general
    )
    if expected_coordinates is not None and not np.array_equal(
        coordinates, np.asarray(expected_coordinates)
    ):
        raise ValueError(
            "Core-NSD encoder and NSD-Imagery beta files use different voxel orderings"
        )
    trial_patterns = all_betas[events["beta_index"].to_numpy()]
    normalized = zscore_within_groups(
        trial_patterns, events["run_name"].to_numpy()
    )

    rows = []
    measured = []
    first_halves = []
    second_halves = []
    for stimulus_set in stimulus_sets:
        for target_number in range(1, 7):
            selected = (
                events["task"].eq(task)
                & events["stimulus_set"].eq(stimulus_set)
                & events["target_number"].eq(target_number)
            ).to_numpy()
            expected_repeats = 8 if task == "vision" else 16
            if int(selected.sum()) != expected_repeats:
                raise ValueError(
                    f"Unexpected repeat count for {task} {stimulus_set}{target_number}: "
                    f"{int(selected.sum())}"
                )
            selected_rows = np.flatnonzero(selected)
            selected_runs = events.iloc[selected_rows]["run_name"].to_numpy()
            unique_runs = tuple(dict.fromkeys(selected_runs.tolist()))
            if task == "imagery":
                if len(unique_runs) != 2:
                    raise ValueError(
                        f"Expected two imagery runs for {stimulus_set}{target_number}"
                    )
                first = selected_rows[selected_runs == unique_runs[0]]
                second = selected_rows[selected_runs == unique_runs[1]]
            else:
                if len(unique_runs) != 1:
                    raise ValueError(
                        f"Expected one vision run for {stimulus_set}{target_number}"
                    )
                first = selected_rows[::2]
                second = selected_rows[1::2]
            if len(first) != len(second) or len(first) < 2:
                raise ValueError(
                    f"Could not form balanced repeat halves for "
                    f"{task} {stimulus_set}{target_number}"
                )
            measured.append(normalized[selected_rows].mean(axis=0))
            first_halves.append(normalized[first].mean(axis=0))
            second_halves.append(normalized[second].mean(axis=0))
            rows.append(
                {
                    "stimulus_set": stimulus_set,
                    "target_number": target_number,
                }
            )
    masks_3d = paper_visual_roi_masks(general, prf)
    roi_masks = {
        name: mask_at_coordinates(mask, coordinates)
        for name, mask in masks_3d.items()
    }
    return (
        np.stack(measured),
        np.stack(first_halves),
        np.stack(second_halves),
        pd.DataFrame(rows),
        roi_masks,
        coordinates,
    )


def measured_nsdimagery_patterns(
    data_root: str | Path,
    subject: int,
    task: str,
    stimulus_sets: Iterable[str],
    *,
    expected_coordinates: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Return paper-preprocessed target patterns and ROI masks for one task."""
    measured, _, _, labels, roi_masks, coordinates = (
        measured_nsdimagery_target_data(
            data_root,
            subject,
            task,
            stimulus_sets,
            expected_coordinates=expected_coordinates,
        )
    )
    return measured, labels, roi_masks, coordinates
