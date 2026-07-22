#!/usr/bin/env python
"""Fit a leakage-safe PCA + multi-output ridge core-NSD encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery.encoding import (  # noqa: E402
    apply_feature_transform,
    fit_ridge_weights,
    voxelwise_prediction_metrics,
)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("alphas must be comma-separated positive values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--betas", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument(
        "--voxel-regions",
        type=Path,
        help="Optional voxel-region CSV written by the preparation script",
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-voxel-metrics", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument(
        "--alphas", type=parse_float_list, default=(0.1, 1.0, 10.0, 100.0, 1000.0)
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--voxel-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="DINOv2_PyramidRidge")
    return parser.parse_args()


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        row_id = archive["row_id"]
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return features, row_id, metadata


def fit_feature_transform(
    features: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, PCA, np.ndarray]:
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[~np.isfinite(feature_scale) | (feature_scale == 0)] = 1
    standardized = (features - feature_mean) / feature_scale
    components = min(n_components, len(features) - 1, features.shape[1])
    if components < 2:
        raise ValueError("At least two PCA components are required")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    transformed = pca.fit_transform(standardized).astype(np.float32)
    return feature_mean, feature_scale, pca, transformed


def transform_with_parts(
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    standardized = (features - feature_mean) / feature_scale
    return pca.transform(standardized).astype(np.float32)


def standardized_targets(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = targets.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale == 0)] = 1
    return (targets - mean) / scale, mean, scale


def mean_finite(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    betas = np.load(args.betas, mmap_mode="r")
    coordinates = np.load(args.coordinates)
    features, feature_rows, feature_metadata = load_features(args.features)
    if len(manifest) != len(betas) or len(manifest) != len(features):
        raise ValueError("manifest, betas, and features must have identical rows")
    if not np.array_equal(manifest["row_id"].to_numpy(), feature_rows):
        raise ValueError("feature row_id values do not match the manifest")
    if betas.ndim != 2 or betas.shape[1] != len(coordinates):
        raise ValueError("beta voxel dimension does not match coordinates")
    required_splits = {"train", "validation", "test"}
    if set(manifest["split"]) != required_splits:
        raise ValueError(f"manifest split column must contain exactly {required_splits}")

    train = manifest["split"].eq("train").to_numpy()
    validation = manifest["split"].eq("validation").to_numpy()
    test = manifest["split"].eq("test").to_numpy()

    feature_mean, feature_scale, pca, x_train = fit_feature_transform(
        features[train], args.pca_components, args.seed
    )
    x_validation = transform_with_parts(
        features[validation], feature_mean, feature_scale, pca
    )
    y_train, beta_mean, beta_scale = standardized_targets(
        np.asarray(betas[train], dtype=np.float32)
    )
    y_validation = (
        np.asarray(betas[validation], dtype=np.float32) - beta_mean
    ) / beta_scale

    validation_rows = []
    for alpha in args.alphas:
        print(f"Selecting alpha={alpha:g}")
        weights = fit_ridge_weights(
            x_train,
            y_train,
            alpha,
            device=args.device,
            voxel_chunk_size=args.voxel_chunk_size,
        )
        prediction = x_validation @ weights
        correlation, r_squared = voxelwise_prediction_metrics(y_validation, prediction)
        validation_rows.append(
            {
                "alpha": float(alpha),
                "mean_voxel_correlation": mean_finite(correlation),
                "median_voxel_correlation": float(np.nanmedian(correlation)),
                "mean_voxel_r2": mean_finite(r_squared),
            }
        )
    validation_table = pd.DataFrame(validation_rows)
    if not np.isfinite(validation_table["mean_voxel_correlation"]).any():
        raise ValueError("All validation correlations are non-finite")
    selected_alpha = float(
        validation_table.loc[
            validation_table["mean_voxel_correlation"].idxmax(), "alpha"
        ]
    )

    final_train = train | validation
    feature_mean, feature_scale, pca, x_final = fit_feature_transform(
        features[final_train], args.pca_components, args.seed
    )
    y_final, beta_mean, beta_scale = standardized_targets(
        np.asarray(betas[final_train], dtype=np.float32)
    )
    print(f"Refitting train+validation with alpha={selected_alpha:g}")
    ridge_weights = fit_ridge_weights(
        x_final,
        y_final,
        selected_alpha,
        device=args.device,
        voxel_chunk_size=args.voxel_chunk_size,
    )
    model_arrays = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "ridge_weights": ridge_weights.astype(np.float32),
        "beta_mean": beta_mean.astype(np.float32),
        "beta_scale": beta_scale.astype(np.float32),
        "coordinates": coordinates.astype(np.int16),
    }
    x_test = apply_feature_transform(features[test], model_arrays)
    y_test = (np.asarray(betas[test], dtype=np.float32) - beta_mean) / beta_scale
    test_prediction = x_test @ ridge_weights
    test_correlation, test_r_squared = voxelwise_prediction_metrics(
        y_test, test_prediction
    )
    test_by_region = {}
    if args.voxel_regions is not None:
        voxel_regions = pd.read_csv(args.voxel_regions)
        if len(voxel_regions) != len(coordinates):
            raise ValueError("voxel-region table does not match coordinates")
        for region in ("early_visual", "higher_visual", "visual_cortex"):
            if region not in voxel_regions:
                raise ValueError(f"voxel-region table is missing {region}")
            selected = voxel_regions[region].astype(bool).to_numpy()
            test_by_region[region] = {
                "n_voxels": int(selected.sum()),
                "mean_voxel_correlation": mean_finite(test_correlation[selected]),
                "median_voxel_correlation": float(
                    np.nanmedian(test_correlation[selected])
                ),
                "mean_voxel_r2": mean_finite(test_r_squared[selected]),
            }

    metadata = {
        "method": args.method,
        "selected_alpha": selected_alpha,
        "pca_components": int(pca.n_components_),
        "feature_metadata": feature_metadata,
        "n_train_images": int(train.sum()),
        "n_validation_images": int(validation.sum()),
        "n_final_train_images": int(final_train.sum()),
        "n_test_images": int(test.sum()),
        "n_voxels": int(betas.shape[1]),
        "target_space": "per-voxel standardized core-NSD betas",
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_model,
        **model_arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    voxel_table = pd.DataFrame(
        {
            "voxel": np.arange(len(coordinates)),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
            "test_correlation": test_correlation,
            "test_r2": test_r_squared,
        }
    )
    args.output_voxel_metrics.parent.mkdir(parents=True, exist_ok=True)
    voxel_table.to_csv(args.output_voxel_metrics, index=False)

    summary = {
        **metadata,
        "test_mean_voxel_correlation": mean_finite(test_correlation),
        "test_median_voxel_correlation": float(np.nanmedian(test_correlation)),
        "test_mean_voxel_r2": mean_finite(test_r_squared),
        "test_by_region": test_by_region,
        "validation_grid": validation_table.to_dict(orient="records"),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(validation_table.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_model}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_voxel_metrics}")


if __name__ == "__main__":
    main()
