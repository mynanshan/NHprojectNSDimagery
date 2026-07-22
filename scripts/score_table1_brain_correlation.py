#!/usr/bin/env python3
"""Score reconstructed images with the NSD-Imagery Table 1 brain metric."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nsdimagery import (  # noqa: E402
    build_event_table,
    extract_masked_betas,
    load_roi,
    mask_at_coordinates,
    paper_visual_roi_masks,
    paths_for_subject,
    reconstruction_brain_correlations,
    zscore_within_groups,
)
from nsdimagery.gnet import GNetEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate repeat-averaged measured betas with GNet predictions "
            "from reconstruction samples, matching the paper's brain metric."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--task", choices=("vision", "imagery"), required=True)
    parser.add_argument(
        "--stimulus-sets", nargs="+", choices=("A", "B", "C"), default=("A", "B")
    )
    parser.add_argument(
        "--reconstructions",
        type=Path,
        required=True,
        help=".pt/.pth/.npy/.npz array shaped targets x samples x channels x H x W",
    )
    parser.add_argument("--gnet-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--brain-region-masks",
        type=Path,
        help=(
            "Optional official brain_region_masks.hdf5. If omitted, the same "
            "regions are rebuilt from local nsdgeneral/prf masks."
        ),
    )
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-target", type=int, default=10)
    parser.add_argument(
        "--predicted-cache",
        type=Path,
        help="Optional .npy cache for GNet predictions; reused if it already exists",
    )
    parser.add_argument(
        "--output-prefix", type=Path, required=True,
        help="Writes PREFIX_detail.csv and PREFIX_summary.csv",
    )
    return parser.parse_args()


def _trusted_torch_load(path: Path):
    """Load an explicitly supplied reconstruction file across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_reconstructions(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        values = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            candidate_keys = [
                key for key in ("all_recons", "reconstructions", "images")
                if key in archive
            ]
            if not candidate_keys and len(archive.files) == 1:
                candidate_keys = archive.files
            if len(candidate_keys) != 1:
                raise ValueError(
                    f"Could not choose a reconstruction array from keys {archive.files}"
                )
            values = archive[candidate_keys[0]]
    elif path.suffix in {".pt", ".pth"}:
        values = _trusted_torch_load(path)
        if isinstance(values, dict):
            candidate_keys = [
                key for key in ("all_recons", "reconstructions", "images")
                if key in values
            ]
            if len(candidate_keys) != 1:
                raise ValueError(
                    "Torch dictionary must contain exactly one of all_recons, "
                    "reconstructions, or images"
                )
            values = values[candidate_keys[0]]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
    else:
        raise ValueError("Reconstructions must be .pt, .pth, .npy, or .npz")
    values = np.asarray(values)
    if values.ndim != 5:
        raise ValueError(
            "Reconstructions must be targets x samples x channels x H x W "
            "(channels-last images are also accepted)"
        )
    return values


def select_target_reconstructions(
    reconstructions: np.ndarray,
    stimulus_sets: tuple[str, ...],
    samples_per_target: int,
) -> np.ndarray:
    expected_targets = 6 * len(stimulus_sets)
    if reconstructions.shape[0] == 18:
        set_slices = {"A": slice(0, 6), "B": slice(6, 12), "C": slice(12, 18)}
        reconstructions = np.concatenate(
            [reconstructions[set_slices[name]] for name in stimulus_sets], axis=0
        )
    elif reconstructions.shape[0] != expected_targets:
        raise ValueError(
            f"Expected {expected_targets} selected targets or all 18 targets; "
            f"found {reconstructions.shape[0]}"
        )
    if reconstructions.shape[1] < samples_per_target:
        raise ValueError(
            f"Requested {samples_per_target} samples but file has "
            f"{reconstructions.shape[1]}"
        )
    return reconstructions[:, :samples_per_target]


def measured_target_patterns(
    data_root: Path,
    subject: int,
    task: str,
    stimulus_sets: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    nsdgeneral, _ = load_roi(data_root, subject, "nsdgeneral")
    prf_visualrois, _ = load_roi(data_root, subject, "prf-visualrois")
    all_betas, coordinates = extract_masked_betas(
        paths_for_subject(data_root, subject)["beta"], nsdgeneral
    )
    events = build_event_table(data_root, subject).reset_index(drop=True)
    trial_patterns = all_betas[events["beta_index"].to_numpy()]
    normalized = zscore_within_groups(
        trial_patterns, events["run_name"].to_numpy()
    )

    target_patterns = []
    for stimulus_set in stimulus_sets:
        for target_number in range(1, 7):
            selected = (
                events["task"].eq(task)
                & events["stimulus_set"].eq(stimulus_set)
                & events["target_number"].eq(target_number)
            ).to_numpy()
            if selected.sum() not in {8, 16}:
                raise AssertionError(
                    f"Unexpected repeat count for {task} {stimulus_set}{target_number}: "
                    f"{selected.sum()}"
                )
            target_patterns.append(normalized[selected].mean(axis=0))

    volume_masks = paper_visual_roi_masks(nsdgeneral, prf_visualrois)
    column_masks = {
        name: mask_at_coordinates(mask, coordinates)
        for name, mask in volume_masks.items()
    }
    return np.stack(target_patterns), coordinates, column_masks


def load_official_roi_masks(
    path: Path, subject: int, n_voxels: int
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    subject_name = f"subj0{subject}"
    with h5py.File(path, "r") as handle:
        if subject_name not in handle:
            raise KeyError(f"{subject_name} is absent from {path}")
        group = handle[subject_name]
        components = {
            name: np.asarray(group[name][:], dtype=bool)
            for name in ("V1", "V2", "V3", "V4")
        }
        masks = {
            "visual_cortex": np.asarray(group["nsd_general"][:], dtype=bool),
            "higher_visual": np.asarray(group["higher_vis"][:], dtype=bool),
            "early_visual": np.logical_or.reduce(tuple(components.values())),
            **components,
        }
    if any(mask.shape != (n_voxels,) for mask in masks.values()):
        shapes = {name: mask.shape for name, mask in masks.items()}
        raise ValueError(
            f"Official ROI masks do not match {n_voxels} GNet voxels: {shapes}"
        )
    return masks


def main() -> None:
    args = parse_args()
    stimulus_sets = tuple(args.stimulus_sets)
    reconstructions = select_target_reconstructions(
        load_reconstructions(args.reconstructions),
        stimulus_sets,
        args.samples_per_target,
    )
    measured, coordinates, roi_masks = measured_target_patterns(
        args.data_root, args.subject, args.task, stimulus_sets
    )
    n_targets, n_samples = reconstructions.shape[:2]

    gnet = GNetEncoder(
        args.gnet_checkpoint, args.subject, device=args.device
    )
    if gnet.n_voxels != measured.shape[1] or gnet.n_voxels != len(coordinates):
        raise ValueError(
            "GNet/checkpoint voxel count does not match this subject's full "
            f"nsdgeneral mask: {gnet.n_voxels} vs {measured.shape[1]}"
        )
    if args.brain_region_masks is not None:
        roi_masks = load_official_roi_masks(
            args.brain_region_masks, args.subject, gnet.n_voxels
        )

    if args.predicted_cache is not None and args.predicted_cache.is_file():
        predicted = np.load(args.predicted_cache)
    else:
        flat_images = reconstructions.reshape((-1, *reconstructions.shape[2:]))
        predicted = gnet.predict(flat_images, batch_size=args.batch_size).reshape(
            n_targets, n_samples, gnet.n_voxels
        )
        if args.predicted_cache is not None:
            args.predicted_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.predicted_cache, predicted)
    expected_shape = (n_targets, n_samples, gnet.n_voxels)
    if predicted.shape != expected_shape:
        raise ValueError(
            f"Predicted beta shape is {predicted.shape}; expected {expected_shape}"
        )

    rows = []
    table_regions = ("early_visual", "higher_visual", "visual_cortex")
    for region in table_regions:
        scores = reconstruction_brain_correlations(
            measured, predicted, voxel_mask=roi_masks[region]
        )
        for target_index, stimulus_set in enumerate(stimulus_sets):
            for target_number in range(1, 7):
                row_index = target_index * 6 + target_number - 1
                for sample in range(n_samples):
                    rows.append({
                        "method": args.method,
                        "subject": f"subj{args.subject:02d}",
                        "task": args.task,
                        "stimulus_set": stimulus_set,
                        "target_number": target_number,
                        "sample": sample + 1,
                        "region": region,
                        "brain_correlation": float(scores[row_index, sample]),
                        "n_voxels": int(roi_masks[region].sum()),
                    })
    detail = pd.DataFrame(rows)
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
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.parent / f"{prefix.name}_detail.csv"
    summary_path = prefix.parent / f"{prefix.name}_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
