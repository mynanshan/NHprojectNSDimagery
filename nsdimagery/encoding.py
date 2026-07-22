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
    """Predict image-evoked beta patterns from cached image features."""
    transformed = apply_feature_transform(features, model)
    predicted = transformed @ np.asarray(model["ridge_weights"], dtype=np.float32)
    if not standardized_betas:
        predicted = (
            predicted * np.asarray(model["beta_scale"], dtype=np.float32)
            + np.asarray(model["beta_mean"], dtype=np.float32)
        )
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


def measured_nsdimagery_patterns(
    data_root: str | Path,
    subject: int,
    task: str,
    stimulus_sets: Iterable[str],
    *,
    expected_coordinates: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Return paper-preprocessed target patterns and ROI masks for one task."""
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
            measured.append(normalized[selected].mean(axis=0))
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
    return np.stack(measured), pd.DataFrame(rows), roi_masks, coordinates
