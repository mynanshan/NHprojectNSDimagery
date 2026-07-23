#!/usr/bin/env python
"""Directly fit NSD-Imagery encoders with target-identity cross-validation."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery.encoding import (  # noqa: E402
    apply_feature_transform,
    kernel_ridge_predict,
    leave_one_target_out_predictions,
    load_encoder_model,
    measured_nsdimagery_target_data,
    voxelwise_prediction_metrics,
)


def resolved_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).expanduser().resolve()


def positive_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 or not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "values must be comma-separated positive floats"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=resolved_path, required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 9), required=True)
    parser.add_argument(
        "--feature-transform-model",
        type=resolved_path,
        required=True,
        help="Core-NSD model supplying a fixed, independently learned PCA transform",
    )
    parser.add_argument("--image-manifest", type=resolved_path, required=True)
    parser.add_argument("--image-features", type=resolved_path, required=True)
    parser.add_argument("--output-prefix", type=resolved_path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("vision", "imagery"),
        default=("vision", "imagery"),
    )
    parser.add_argument(
        "--stimulus-sets", nargs="+", choices=("A", "B"), default=("A", "B")
    )
    parser.add_argument(
        "--kernels", nargs="+", choices=("linear", "rbf"), default=("linear", "rbf")
    )
    parser.add_argument(
        "--alphas",
        type=positive_float_list,
        default=(0.01, 0.1, 1.0, 10.0, 100.0),
    )
    parser.add_argument(
        "--rbf-gamma-scales",
        type=positive_float_list,
        default=(0.25, 1.0, 4.0),
    )
    return parser.parse_args()


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        row_id = archive["row_id"]
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return features, row_id, metadata


def finite_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "fraction_positive": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "fraction_positive": float(np.mean(finite > 0)),
    }


def candidate_grid(
    kernel: str,
    alphas: tuple[float, ...],
    gamma_scales: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    if kernel == "linear":
        return tuple((alpha, 1.0) for alpha in alphas)
    return tuple(product(alphas, gamma_scales))


def nested_loto_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    labels: pd.DataFrame,
    *,
    kernel: str,
    alphas: tuple[float, ...],
    gamma_scales: tuple[float, ...],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Nested leave-one-target-out prediction and global hyperparameter choice."""
    if len(features) != len(targets) or len(features) != len(labels):
        raise ValueError("features, targets, and labels must contain the same targets")
    candidates = candidate_grid(kernel, alphas, gamma_scales)
    predicted = np.empty_like(targets, dtype=np.float32)
    selection_rows = []

    for held_out in range(len(features)):
        outer_train = np.arange(len(features)) != held_out
        inner_features = features[outer_train]
        inner_targets = targets[outer_train]
        candidate_rows = []
        for alpha, gamma_scale in candidates:
            inner_prediction = leave_one_target_out_predictions(
                inner_features,
                inner_targets,
                alpha=alpha,
                kernel=kernel,
                gamma_scale=gamma_scale,
            )
            correlation, r_squared = voxelwise_prediction_metrics(
                inner_targets, inner_prediction
            )
            correlation_summary = finite_summary(correlation)
            r2_summary = finite_summary(r_squared)
            candidate_rows.append(
                {
                    "alpha": float(alpha),
                    "gamma_scale": float(gamma_scale),
                    "inner_mean_voxel_correlation": correlation_summary["mean"],
                    "inner_mean_voxel_r2": r2_summary["mean"],
                }
            )
        candidate_table = pd.DataFrame(candidate_rows)
        finite = np.isfinite(candidate_table["inner_mean_voxel_correlation"])
        if not finite.any():
            raise ValueError(
                f"All inner scores were non-finite while holding out target {held_out}"
            )
        best_row = candidate_table.loc[
            candidate_table.loc[finite, "inner_mean_voxel_correlation"].idxmax()
        ]
        predicted[held_out] = kernel_ridge_predict(
            features[outer_train],
            targets[outer_train],
            features[[held_out]],
            alpha=float(best_row["alpha"]),
            kernel=kernel,
            gamma_scale=float(best_row["gamma_scale"]),
        )[0]
        label = labels.iloc[held_out]
        selection_rows.append(
            {
                "held_out_stimulus_set": label["stimulus_set"],
                "held_out_target_number": int(label["target_number"]),
                **best_row.to_dict(),
            }
        )
    return predicted, pd.DataFrame(selection_rows)


def spearman_brown(split_half_correlation: np.ndarray) -> np.ndarray:
    correlation = np.asarray(split_half_correlation, dtype=np.float64)
    corrected = np.full_like(correlation, np.nan)
    valid = np.isfinite(correlation) & (correlation > -1)
    corrected[valid] = 2 * correlation[valid] / (1 + correlation[valid])
    return corrected


def main() -> None:
    args = parse_args()
    required_files = {
        "feature transform model": args.feature_transform_model,
        "image manifest": args.image_manifest,
        "image features": args.image_features,
    }
    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing direct-encoder inputs:\n" + "\n".join(missing))

    transform_model = load_encoder_model(args.feature_transform_model)
    transform_metadata = transform_model["metadata"]
    manifest = pd.read_csv(args.image_manifest)
    features, feature_rows, feature_metadata = load_features(args.image_features)
    if len(manifest) != len(features):
        raise ValueError("image manifest and feature cache have different rows")
    if "row_id" not in manifest or not np.array_equal(
        manifest["row_id"].to_numpy(), feature_rows
    ):
        raise ValueError("feature row_id values do not match the image manifest")
    required_columns = {"stimulus_set", "target_number"}
    if not required_columns.issubset(manifest):
        raise ValueError(f"image manifest must contain columns {required_columns}")
    trained_feature_metadata = transform_metadata["feature_metadata"]
    for key in (
        "model_id",
        "hidden_state_layers",
        "pyramid_levels",
        "include_cls",
        "grid_size",
        "feature_dimension",
    ):
        if trained_feature_metadata.get(key) != feature_metadata.get(key):
            raise ValueError(
                f"Feature-transform model and target cache disagree on {key}"
            )

    selected = manifest["stimulus_set"].isin(args.stimulus_sets).to_numpy()
    manifest = manifest.loc[selected].reset_index(drop=True)
    features = features[selected]
    if manifest.duplicated(["stimulus_set", "target_number"]).any():
        raise ValueError(
            "Direct fitting requires exactly one ground-truth image per target; "
            "do not pass a reconstruction-sample manifest"
        )
    expected_targets = 6 * len(args.stimulus_sets)
    if len(manifest) != expected_targets:
        raise ValueError(
            f"Expected {expected_targets} unique targets; found {len(manifest)}"
        )
    fixed_features = apply_feature_transform(features, transform_model)
    coordinates = np.asarray(transform_model["coordinates"])

    voxel_table = pd.DataFrame(
        {
            "voxel": np.arange(len(coordinates)),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
        }
    )
    summary_rows = []
    selection_tables = []
    prediction_arrays = {}
    labels_reference = None

    for task in args.tasks:
        (
            measured,
            first_half,
            second_half,
            labels,
            roi_masks,
            measured_coordinates,
        ) = measured_nsdimagery_target_data(
            args.data_root,
            args.subject,
            task,
            args.stimulus_sets,
            expected_coordinates=coordinates,
        )
        if not np.array_equal(measured_coordinates, coordinates):
            raise AssertionError("Measured and feature-transform voxel order changed")
        if labels_reference is None:
            labels_reference = labels.copy()
            lookup = {
                (row.stimulus_set, int(row.target_number)): index
                for index, row in manifest.iterrows()
            }
            feature_order = [
                lookup[(row.stimulus_set, int(row.target_number))]
                for row in labels.itertuples()
            ]
            ordered_features = fixed_features[feature_order]
        elif not labels.equals(labels_reference):
            raise ValueError("Vision and imagery target labels are not aligned")

        split_half_correlation, _ = voxelwise_prediction_metrics(
            first_half, second_half
        )
        corrected_reliability = spearman_brown(split_half_correlation)
        voxel_table[f"{task}_split_half_correlation"] = split_half_correlation
        voxel_table[f"{task}_spearman_brown_reliability"] = corrected_reliability
        for region, mask in roi_masks.items():
            if region not in voxel_table:
                voxel_table[region] = np.asarray(mask, dtype=bool)

        for kernel in args.kernels:
            print(
                f"Nested leave-one-target-out fitting: "
                f"subj{args.subject:02d} {task} {kernel}"
            )
            prediction, selections = nested_loto_predictions(
                ordered_features,
                measured,
                labels,
                kernel=kernel,
                alphas=args.alphas,
                gamma_scales=args.rbf_gamma_scales,
            )
            selections.insert(0, "kernel", kernel)
            selections.insert(0, "task", task)
            selections.insert(0, "subject", f"subj{args.subject:02d}")
            selection_tables.append(selections)
            prediction_arrays[f"{kernel}_{task}"] = prediction.astype(np.float32)
            correlation, r_squared = voxelwise_prediction_metrics(
                measured, prediction
            )
            prefix = f"{kernel}_{task}"
            voxel_table[f"{prefix}_target_correlation"] = correlation
            voxel_table[f"{prefix}_target_r2"] = r_squared

            for region, mask in roi_masks.items():
                mask = np.asarray(mask, dtype=bool)
                correlation_summary = finite_summary(correlation[mask])
                r2_summary = finite_summary(r_squared[mask])
                reliability_summary = finite_summary(
                    corrected_reliability[mask]
                )
                summary_rows.append(
                    {
                        "method": (
                            "DirectLinearKRR"
                            if kernel == "linear"
                            else "DirectRBFKRR"
                        ),
                        "subject": f"subj{args.subject:02d}",
                        "task": task,
                        "stimulus_sets": "+".join(args.stimulus_sets),
                        "region": region,
                        "n_targets": int(len(labels)),
                        "n_voxels": int(mask.sum()),
                        "cv_unit": "target_identity",
                        "outer_cv": "leave_one_target_out",
                        "mean_voxel_target_correlation": correlation_summary["mean"],
                        "median_voxel_target_correlation": correlation_summary[
                            "median"
                        ],
                        "fraction_positive_target_correlation": correlation_summary[
                            "fraction_positive"
                        ],
                        "mean_voxel_target_r2": r2_summary["mean"],
                        "median_voxel_target_r2": r2_summary["median"],
                        "fraction_positive_target_r2": r2_summary[
                            "fraction_positive"
                        ],
                        "mean_spearman_brown_reliability": reliability_summary["mean"],
                        "median_spearman_brown_reliability": reliability_summary[
                            "median"
                        ],
                    }
                )

    for kernel in args.kernels:
        vision = f"{kernel}_vision_target_r2"
        imagery = f"{kernel}_imagery_target_r2"
        if {vision, imagery}.issubset(voxel_table):
            voxel_table[f"{kernel}_vision_minus_imagery_target_r2"] = (
                voxel_table[vision] - voxel_table[imagery]
            )
    summary = pd.DataFrame(summary_rows)
    selections = pd.concat(selection_tables, ignore_index=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    voxel_path = (
        args.output_prefix.parent / f"{args.output_prefix.name}_voxel_metrics.csv"
    )
    summary_path = (
        args.output_prefix.parent / f"{args.output_prefix.name}_summary.csv"
    )
    selection_path = (
        args.output_prefix.parent / f"{args.output_prefix.name}_fold_selection.csv"
    )
    prediction_path = (
        args.output_prefix.parent / f"{args.output_prefix.name}_predictions.npz"
    )
    voxel_table.to_csv(voxel_path, index=False)
    summary.to_csv(summary_path, index=False)
    selections.to_csv(selection_path, index=False)
    np.savez(
        prediction_path,
        **prediction_arrays,
        stimulus_set=labels_reference["stimulus_set"].to_numpy(),
        target_number=labels_reference["target_number"].to_numpy(),
    )
    print(summary.to_string(index=False))
    print(f"Wrote {voxel_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {selection_path}")
    print(f"Wrote {prediction_path}")


if __name__ == "__main__":
    main()
