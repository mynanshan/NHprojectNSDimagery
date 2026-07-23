#!/usr/bin/env python
"""Fit a nonlinear residual readout without touching the core-NSD test set."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
    load_encoder_model,
    predict_with_encoder,
    voxelwise_prediction_metrics,
)


def resolved_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=resolved_path, required=True)
    parser.add_argument("--betas", type=resolved_path, required=True)
    parser.add_argument("--coordinates", type=resolved_path, required=True)
    parser.add_argument("--voxel-regions", type=resolved_path)
    parser.add_argument("--features", type=resolved_path, required=True)
    parser.add_argument(
        "--ridge-model",
        type=resolved_path,
        required=True,
        help="Existing ridge model; its selected alpha defines the frozen baseline",
    )
    parser.add_argument("--output-model", type=resolved_path, required=True)
    parser.add_argument("--output-summary", type=resolved_path, required=True)
    parser.add_argument("--output-voxel-metrics", type=resolved_path, required=True)
    parser.add_argument("--output-history", type=resolved_path, required=True)
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Required validation-correlation improvement over the current best",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--voxel-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="DINOv2_PyramidResidualMLP")
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
    return pca.transform((features - feature_mean) / feature_scale).astype(np.float32)


def standardized_targets(
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = targets.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale == 0)] = 1
    return ((targets - mean) / scale).astype(np.float32), mean, scale


def mean_finite(values: np.ndarray) -> float:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def nonlinear_arrays(module, input_mean: np.ndarray, input_scale: np.ndarray) -> dict:
    return {
        "nonlinear_input_mean": input_mean.astype(np.float32),
        "nonlinear_input_scale": input_scale.astype(np.float32),
        "nonlinear_hidden_weight": (
            module.hidden.weight.detach().cpu().numpy().astype(np.float32)
        ),
        "nonlinear_hidden_bias": (
            module.hidden.bias.detach().cpu().numpy().astype(np.float32)
        ),
        "nonlinear_output_weight": (
            module.output.weight.detach().cpu().numpy().astype(np.float32)
        ),
        "nonlinear_output_bias": (
            module.output.bias.detach().cpu().numpy().astype(np.float32)
        ),
    }


def make_module(
    torch, n_features: int, hidden_width: int, n_voxels: int, dropout: float
):
    class ResidualReadout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(n_features, hidden_width)
            self.dropout = torch.nn.Dropout(dropout)
            self.output = torch.nn.Linear(hidden_width, n_voxels)
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

        def forward(self, features):
            hidden = torch.nn.functional.gelu(self.hidden(features))
            return self.output(self.dropout(hidden))

    return ResidualReadout()


def resolve_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def validation_metrics(
    torch,
    module,
    base_prediction,
    x_scaled,
    y,
    *,
    batch_size: int,
) -> tuple[float, float]:
    module.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(base_prediction), batch_size):
            stop = min(start + batch_size, len(base_prediction))
            prediction = base_prediction[start:stop] + module(
                x_scaled[start:stop]
            )
            pieces.append(prediction.cpu().numpy())
    predicted = np.concatenate(pieces)
    correlation, r_squared = voxelwise_prediction_metrics(
        y.cpu().numpy(), predicted
    )
    return mean_finite(correlation), mean_finite(r_squared)


def train_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    ridge_weights: np.ndarray,
    args: argparse.Namespace,
):
    try:
        import torch
    except ModuleNotFoundError:
        raise RuntimeError("PyTorch is required for the nonlinear encoder") from None

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(torch, args.device)
    input_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    input_scale[~np.isfinite(input_scale) | (input_scale == 0)] = 1
    x_train_scaled = (x_train - input_mean) / input_scale
    x_validation_scaled = (x_validation - input_mean) / input_scale

    train_x = torch.from_numpy(x_train).to(device)
    train_x_scaled = torch.from_numpy(x_train_scaled).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    validation_x = torch.from_numpy(x_validation).to(device)
    validation_x_scaled = torch.from_numpy(x_validation_scaled).to(device)
    validation_y = torch.from_numpy(y_validation).to(device)
    ridge = torch.from_numpy(ridge_weights).to(device)
    with torch.no_grad():
        train_base_prediction = train_x @ ridge
        validation_base_prediction = validation_x @ ridge
    module = make_module(
        torch, x_train.shape[1], args.hidden_width, y_train.shape[1], args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    baseline_correlation, baseline_r2 = validation_metrics(
        torch,
        module,
        validation_base_prediction,
        validation_x_scaled,
        validation_y,
        batch_size=args.batch_size,
    )
    history = [
        {
            "epoch": 0,
            "train_mse": float("nan"),
            "validation_mean_voxel_correlation": baseline_correlation,
            "validation_mean_voxel_r2": baseline_r2,
        }
    ]
    best_score = baseline_correlation
    best_epoch = 0
    best_state = deepcopy(module.state_dict())
    stale_epochs = 0

    for epoch in range(1, args.max_epochs + 1):
        module.train()
        permutation = torch.randperm(len(train_x), device=device)
        squared_error = 0.0
        n_values = 0
        for start in range(0, len(train_x), args.batch_size):
            rows = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = train_base_prediction[rows] + module(
                train_x_scaled[rows]
            )
            loss = torch.nn.functional.mse_loss(prediction, train_y[rows])
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach()) * len(rows)
            n_values += len(rows)

        correlation, r_squared = validation_metrics(
            torch,
            module,
            validation_base_prediction,
            validation_x_scaled,
            validation_y,
            batch_size=args.batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": squared_error / n_values,
                "validation_mean_voxel_correlation": correlation,
                "validation_mean_voxel_r2": r_squared,
            }
        )
        print(
            f"epoch={epoch:03d} train_mse={squared_error / n_values:.6f} "
            f"validation_r={correlation:.6f} validation_R2={r_squared:.6f}"
        )
        if np.isfinite(correlation) and correlation > best_score + args.min_delta:
            best_score = correlation
            best_epoch = epoch
            best_state = deepcopy(module.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    module.load_state_dict(best_state)
    return module, input_mean, input_scale, pd.DataFrame(history), best_epoch, device


def refit_selected_epochs(
    x: np.ndarray,
    y: np.ndarray,
    ridge_weights: np.ndarray,
    *,
    epochs: int,
    args: argparse.Namespace,
):
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(torch, args.device)
    input_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    input_scale[~np.isfinite(input_scale) | (input_scale == 0)] = 1
    x_scaled = ((x - input_mean) / input_scale).astype(np.float32)
    tensor_x = torch.from_numpy(x).to(device)
    tensor_x_scaled = torch.from_numpy(x_scaled).to(device)
    tensor_y = torch.from_numpy(y).to(device)
    ridge = torch.from_numpy(ridge_weights).to(device)
    with torch.no_grad():
        base_prediction = tensor_x @ ridge
    module = make_module(
        torch, x.shape[1], args.hidden_width, y.shape[1], args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, epochs + 1):
        module.train()
        permutation = torch.randperm(len(tensor_x), device=device)
        for start in range(0, len(tensor_x), args.batch_size):
            rows = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = base_prediction[rows] + module(tensor_x_scaled[rows])
            loss = torch.nn.functional.mse_loss(prediction, tensor_y[rows])
            loss.backward()
            optimizer.step()
        print(f"final refit epoch={epoch:03d}/{epochs:03d}")
    module.eval()
    return module, input_mean, input_scale


def main() -> None:
    args = parse_args()
    if args.hidden_width < 1:
        raise ValueError("hidden-width must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if min(args.batch_size, args.max_epochs, args.patience) < 1:
        raise ValueError("batch-size, max-epochs, and patience must be positive")
    required_files = {
        "manifest": args.manifest,
        "betas": args.betas,
        "coordinates": args.coordinates,
        "features": args.features,
        "ridge model": args.ridge_model,
    }
    if args.voxel_regions is not None:
        required_files["voxel regions"] = args.voxel_regions
    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing nonlinear encoder inputs:\n" + "\n".join(missing)
        )

    manifest = pd.read_csv(args.manifest)
    betas = np.load(args.betas, mmap_mode="r")
    coordinates = np.load(args.coordinates)
    features, feature_rows, feature_metadata = load_features(args.features)
    ridge_reference = load_encoder_model(args.ridge_model)
    ridge_metadata = ridge_reference["metadata"]
    ridge_alpha = float(ridge_metadata["selected_alpha"])
    if len(manifest) != len(betas) or len(manifest) != len(features):
        raise ValueError("manifest, betas, and features must have identical rows")
    if not np.array_equal(manifest["row_id"].to_numpy(), feature_rows):
        raise ValueError("feature row_id values do not match the manifest")
    if betas.ndim != 2 or betas.shape[1] != len(coordinates):
        raise ValueError("beta voxel dimension does not match coordinates")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("manifest must contain train, validation, and test splits")
    reference_feature_metadata = ridge_metadata["feature_metadata"]
    for key in (
        "model_id",
        "hidden_state_layers",
        "pyramid_levels",
        "include_cls",
        "grid_size",
        "feature_dimension",
    ):
        if reference_feature_metadata.get(key) != feature_metadata.get(key):
            raise ValueError(f"Ridge model and feature cache disagree on {key}")

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
        (np.asarray(betas[validation], dtype=np.float32) - beta_mean) / beta_scale
    ).astype(np.float32)
    train_ridge_weights = fit_ridge_weights(
        x_train,
        y_train,
        ridge_alpha,
        device=args.device,
        voxel_chunk_size=args.voxel_chunk_size,
    )
    _, _, _, history, best_epoch, device = train_with_validation(
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_ridge_weights,
        args,
    )
    print(
        f"Selected epoch {best_epoch}; epoch 0 is the exact ridge baseline "
        f"(device={device})"
    )

    final_train = train | validation
    feature_mean, feature_scale, pca, x_final = fit_feature_transform(
        features[final_train], args.pca_components, args.seed
    )
    y_final, beta_mean, beta_scale = standardized_targets(
        np.asarray(betas[final_train], dtype=np.float32)
    )
    final_ridge_weights = fit_ridge_weights(
        x_final,
        y_final,
        ridge_alpha,
        device=args.device,
        voxel_chunk_size=args.voxel_chunk_size,
    )
    final_module, nonlinear_input_mean, nonlinear_input_scale = (
        refit_selected_epochs(
            x_final,
            y_final,
            final_ridge_weights,
            epochs=best_epoch,
            args=args,
        )
    )
    model_arrays = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "ridge_weights": final_ridge_weights.astype(np.float32),
        "beta_mean": beta_mean.astype(np.float32),
        "beta_scale": beta_scale.astype(np.float32),
        "coordinates": coordinates.astype(np.int16),
        **nonlinear_arrays(
            final_module, nonlinear_input_mean, nonlinear_input_scale
        ),
    }
    x_test = apply_feature_transform(features[test], model_arrays)
    y_test = (
        (np.asarray(betas[test], dtype=np.float32) - beta_mean) / beta_scale
    ).astype(np.float32)
    ridge_prediction = x_test @ final_ridge_weights
    nonlinear_prediction = predict_with_encoder(
        features[test], model_arrays, standardized_betas=True
    )
    ridge_correlation, ridge_r2 = voxelwise_prediction_metrics(
        y_test, ridge_prediction
    )
    nonlinear_correlation, nonlinear_r2 = voxelwise_prediction_metrics(
        y_test, nonlinear_prediction
    )

    voxel_table = pd.DataFrame(
        {
            "voxel": np.arange(len(coordinates)),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
            "ridge_test_correlation": ridge_correlation,
            "ridge_test_r2": ridge_r2,
            "nonlinear_test_correlation": nonlinear_correlation,
            "nonlinear_test_r2": nonlinear_r2,
            # Generic aliases keep this selected model compatible with the
            # existing native-space mapping script.
            "test_correlation": nonlinear_correlation,
            "test_r2": nonlinear_r2,
            "nonlinear_minus_ridge_test_correlation": (
                nonlinear_correlation - ridge_correlation
            ),
            "nonlinear_minus_ridge_test_r2": nonlinear_r2 - ridge_r2,
        }
    )
    test_by_region = {}
    if args.voxel_regions is not None:
        voxel_regions = pd.read_csv(args.voxel_regions)
        if len(voxel_regions) != len(coordinates):
            raise ValueError("voxel-region table does not match coordinates")
        for region in ("early_visual", "higher_visual", "visual_cortex"):
            selected = voxel_regions[region].astype(bool).to_numpy()
            voxel_table[region] = selected
            test_by_region[region] = {
                "n_voxels": int(selected.sum()),
                "ridge_mean_voxel_correlation": mean_finite(
                    ridge_correlation[selected]
                ),
                "nonlinear_mean_voxel_correlation": mean_finite(
                    nonlinear_correlation[selected]
                ),
                "ridge_mean_voxel_r2": mean_finite(ridge_r2[selected]),
                "nonlinear_mean_voxel_r2": mean_finite(nonlinear_r2[selected]),
            }

    metadata = {
        "method": args.method,
        "model_type": "ridge_plus_residual_mlp",
        "ridge_reference": str(args.ridge_model),
        "selected_alpha": ridge_alpha,
        "selected_residual_epochs": int(best_epoch),
        "nonlinear_accepted_on_validation": bool(best_epoch > 0),
        "pca_components": int(pca.n_components_),
        "hidden_width": int(args.hidden_width),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "feature_metadata": feature_metadata,
        "n_train_images": int(train.sum()),
        "n_validation_images": int(validation.sum()),
        "n_final_train_images": int(final_train.sum()),
        "n_test_images": int(test.sum()),
        "n_voxels": int(betas.shape[1]),
        "target_space": "per-voxel standardized core-NSD betas",
    }
    summary = {
        **metadata,
        "ridge_test_mean_voxel_correlation": mean_finite(ridge_correlation),
        "nonlinear_test_mean_voxel_correlation": mean_finite(
            nonlinear_correlation
        ),
        "ridge_test_mean_voxel_r2": mean_finite(ridge_r2),
        "nonlinear_test_mean_voxel_r2": mean_finite(nonlinear_r2),
        "test_by_region": test_by_region,
    }
    for path in (
        args.output_model,
        args.output_summary,
        args.output_voxel_metrics,
        args.output_history,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_model,
        **model_arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")
    voxel_table.to_csv(args.output_voxel_metrics, index=False)
    history.to_csv(args.output_history, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_model}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_voxel_metrics}")
    print(f"Wrote {args.output_history}")


if __name__ == "__main__":
    main()
