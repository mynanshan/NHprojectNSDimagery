#!/usr/bin/env python
"""Evaluate a core-NSD-trained image encoder on vision and imagery betas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery import reconstruction_brain_correlations  # noqa: E402
from nsdimagery.encoding import (  # noqa: E402
    load_encoder_model,
    measured_nsdimagery_patterns,
    predict_with_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--encoder-model", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=("vision", "imagery"), default=("vision", "imagery")
    )
    parser.add_argument(
        "--stimulus-sets", nargs="+", choices=("A", "B"), default=("A", "B")
    )
    parser.add_argument(
        "--method",
        help="Table label (default: the method saved with the encoder)",
    )
    return parser.parse_args()


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        row_id = archive["row_id"]
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return features, row_id, metadata


def main() -> None:
    args = parse_args()
    model = load_encoder_model(args.encoder_model)
    model_metadata = model["metadata"]
    manifest = pd.read_csv(args.image_manifest)
    features, feature_rows, feature_metadata = load_features(args.image_features)
    if len(manifest) != len(features):
        raise ValueError("image manifest and feature cache have different rows")
    if "row_id" not in manifest or not np.array_equal(
        manifest["row_id"].to_numpy(), feature_rows
    ):
        raise ValueError("feature row_id values do not match the image manifest")
    required = {"stimulus_set", "target_number"}
    if not required.issubset(manifest):
        raise ValueError(f"image manifest must contain columns {required}")
    manifest = manifest[manifest["stimulus_set"].isin(args.stimulus_sets)].copy()
    selected_rows = manifest.index.to_numpy()
    features = features[selected_rows]
    manifest = manifest.reset_index(drop=True)
    if "sample" not in manifest:
        manifest["sample"] = 1

    trained_feature_metadata = model_metadata["feature_metadata"]
    comparison_keys = (
        "model_id",
        "hidden_state_layers",
        "pyramid_levels",
        "include_cls",
        "grid_size",
        "feature_dimension",
    )
    mismatches = {
        key: (trained_feature_metadata.get(key), feature_metadata.get(key))
        for key in comparison_keys
        if trained_feature_metadata.get(key) != feature_metadata.get(key)
    }
    if mismatches:
        raise ValueError(
            "Evaluation features do not match encoder training features: "
            + json.dumps(mismatches)
        )

    predicted = predict_with_encoder(features, model, standardized_betas=True)
    method = args.method or str(model_metadata["method"])
    coordinates = np.asarray(model["coordinates"])
    detail_rows = []
    table_regions = ("early_visual", "higher_visual", "visual_cortex")
    for task in args.tasks:
        measured, labels, roi_masks, _ = measured_nsdimagery_patterns(
            args.data_root,
            args.subject,
            task,
            args.stimulus_sets,
            expected_coordinates=coordinates,
        )
        label_to_row = {
            (row.stimulus_set, int(row.target_number)): index
            for index, row in labels.iterrows()
        }
        measured_rows = np.stack(
            [
                measured[label_to_row[(row.stimulus_set, int(row.target_number))]]
                for row in manifest.itertuples()
            ]
        )
        for region in table_regions:
            scores = reconstruction_brain_correlations(
                measured_rows, predicted, voxel_mask=roi_masks[region]
            )
            for row, score in zip(manifest.itertuples(), scores):
                detail_rows.append(
                    {
                        "method": method,
                        "subject": f"subj{args.subject:02d}",
                        "task": task,
                        "stimulus_set": row.stimulus_set,
                        "target_number": int(row.target_number),
                        "sample": int(row.sample),
                        "region": region,
                        "brain_correlation": float(score),
                        "n_voxels": int(roi_masks[region].sum()),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["method", "subject", "task", "region"], sort=False)
        .agg(
            brain_correlation=("brain_correlation", "mean"),
            sd_across_target_samples=("brain_correlation", "std"),
            n_target_samples=("brain_correlation", "size"),
            n_voxels=("n_voxels", "first"),
        )
        .reset_index()
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_prefix.parent / f"{args.output_prefix.name}_detail.csv"
    summary_path = args.output_prefix.parent / f"{args.output_prefix.name}_summary.csv"
    predicted_path = (
        args.output_prefix.parent / f"{args.output_prefix.name}_predicted_betas.npy"
    )
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    np.save(predicted_path, predicted.astype(np.float32))
    print(summary.to_string(index=False))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {predicted_path}")


if __name__ == "__main__":
    main()
