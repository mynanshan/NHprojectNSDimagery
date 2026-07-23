#!/usr/bin/env python
"""Write native-space voxel R-squared maps for the core-NSD encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery.io import load_roi, paper_visual_roi_masks  # noqa: E402


def resolved_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-voxel held-out and NSD-Imagery target R-squared values "
            "to native func1pt8mm NIfTI maps and publication-ready slice figures."
        )
    )
    parser.add_argument("--data-root", type=resolved_path, required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--core-voxel-metrics", type=resolved_path, required=True)
    parser.add_argument("--transfer-voxel-metrics", type=resolved_path, required=True)
    parser.add_argument("--output-dir", type=resolved_path, required=True)
    parser.add_argument(
        "--background-image",
        type=resolved_path,
        help=(
            "Optional anatomical image already registered to func1pt8mm. "
            "Without it, the nsdgeneral mask is used as a gray backdrop."
        ),
    )
    parser.add_argument(
        "--core-r2-threshold",
        type=float,
        default=0.0,
        help=(
            "Show transfer values only in independently core-predictable voxels "
            "(default: test R-squared > 0)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def validate_voxel_table(
    table: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    required_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    required = {"voxel", "x", "y", "z", *required_columns}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if len(table) != len(coordinates):
        raise ValueError(
            f"{label} has {len(table)} voxels; expected {len(coordinates)}"
        )
    table = table.sort_values("voxel", ignore_index=True)
    expected_voxels = np.arange(len(coordinates))
    if not np.array_equal(table["voxel"].to_numpy(), expected_voxels):
        raise ValueError(f"{label} voxel indices are not 0 through N-1")
    table_coordinates = table[["x", "y", "z"]].to_numpy()
    if not np.array_equal(table_coordinates, coordinates):
        raise ValueError(f"{label} coordinates do not match nsdgeneral")
    return table


def values_to_image(
    values: np.ndarray,
    coordinates: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    outside_value: float = 0.0,
) -> nib.Nifti1Image:
    """Place a vector in the exact native voxel grid used by the encoder."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(coordinates),):
        raise ValueError("values must have one element per coordinate")
    volume = np.full(reference.shape, outside_value, dtype=np.float32)
    volume[tuple(coordinates.T)] = np.where(np.isfinite(values), values, 0)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(volume, reference.affine, header)


def positive_sqrt(values: np.ndarray, selected: np.ndarray | None = None) -> np.ndarray:
    """Return sqrt(max(R2, 0)), with unselected or invalid voxels set to zero."""
    values = np.asarray(values, dtype=np.float64)
    output = np.sqrt(np.clip(values, 0, None))
    output[~np.isfinite(values)] = 0
    if selected is not None:
        output[~np.asarray(selected, dtype=bool)] = 0
    return output.astype(np.float32)


def positive_vmax(*arrays: np.ndarray, fallback: float = 0.1) -> float:
    pieces = []
    for array in arrays:
        array = np.asarray(array)
        pieces.append(array[np.isfinite(array) & (array > 0)])
    finite = np.concatenate(pieces)
    if not len(finite):
        return fallback
    return max(float(np.percentile(finite, 99)), np.finfo(np.float32).eps)


def signed_vmax(values: np.ndarray, fallback: float = 0.1) -> float:
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    if not len(finite):
        return fallback
    return max(float(np.percentile(finite, 95)), np.finfo(np.float32).eps)


def summarize_regions(
    transfer: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    eligible: np.ndarray | None = None,
    voxel_set: str = "all_visual_voxels",
) -> pd.DataFrame:
    if eligible is None:
        eligible = np.ones(len(transfer), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(transfer),):
        raise ValueError("eligible must contain one value per voxel")
    rows = []
    for task in ("vision", "imagery"):
        values = transfer[f"{task}_target_r2"].to_numpy(dtype=float)
        for region in (
            "visual_cortex",
            "early_visual",
            "higher_visual",
            "V1",
            "V2",
            "V3",
            "V4",
        ):
            selected = np.asarray(masks[region], dtype=bool) & eligible
            finite = values[selected & np.isfinite(values)]
            rows.append(
                {
                    "task": task,
                    "region": region,
                    "voxel_set": voxel_set,
                    "n_voxels": int(selected.sum()),
                    "mean_voxel_target_r2": (
                        float(finite.mean()) if len(finite) else np.nan
                    ),
                    "median_voxel_target_r2": (
                        float(np.median(finite)) if len(finite) else np.nan
                    ),
                    "fraction_positive_target_r2": (
                        float(np.mean(finite > 0)) if len(finite) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_slice_panel(
    images: list[nib.Nifti1Image],
    titles: list[str],
    *,
    background: nib.spatialimages.SpatialImage,
    output: Path,
    dpi: int,
    vmaxes: list[float],
    cmaps: list[str],
    symmetric: list[bool],
) -> None:
    try:
        from nilearn import plotting
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "nilearn is required for maps; run "
            "`conda env update -n nsdimagery -f environment.yml`"
        ) from error

    figure, axes = plt.subplots(len(images), 1, figsize=(12, 3.25 * len(images)))
    axes = np.atleast_1d(axes)
    for axis, image, title, vmax, cmap, symmetric_cbar in zip(
        axes, images, titles, vmaxes, cmaps, symmetric
    ):
        plotting.plot_stat_map(
            image,
            bg_img=background,
            display_mode="z",
            cut_coords=7,
            threshold=np.finfo(np.float32).eps,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            vmax=vmax,
            colorbar=True,
            black_bg=False,
            dim=0,
            title=title,
            axes=axis,
        )
    figure.subplots_adjust(hspace=0.3)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_roi_summary(
    summary: pd.DataFrame,
    output: Path,
    dpi: int,
    *,
    core_r2_threshold: float,
) -> None:
    region_order = [
        "visual_cortex",
        "early_visual",
        "higher_visual",
        "V1",
        "V2",
        "V3",
        "V4",
    ]
    labels = ["Visual", "Early", "Higher", "V1", "V2", "V3", "V4"]
    colors = {"vision": "#2878B5", "imagery": "#D95F02"}
    x = np.arange(len(region_order))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for task, offset in (("vision", -width / 2), ("imagery", width / 2)):
        indexed = (
            summary.loc[summary["task"].eq(task)]
            .set_index("region")
            .reindex(region_order)
        )
        axes[0].bar(
            x + offset,
            indexed["median_voxel_target_r2"],
            width,
            label=task.capitalize(),
            color=colors[task],
        )
        axes[1].bar(
            x + offset,
            indexed["fraction_positive_target_r2"],
            width,
            label=task.capitalize(),
            color=colors[task],
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Median strict target $R^2$")
    axes[0].set_title("Calibration-sensitive prediction")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of voxels with $R^2 > 0$")
    axes[1].set_title("Voxels beating the task-mean baseline")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    figure.suptitle(
        "NSD-Imagery zero-shot target prediction "
        f"(12 A+B targets; core test $R^2 > {core_r2_threshold:g}$)"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    required_files = {
        "core voxel metrics": args.core_voxel_metrics,
        "transfer voxel metrics": args.transfer_voxel_metrics,
    }
    if args.background_image is not None:
        required_files["background image"] = args.background_image
    missing = [
        f"{label}: {path}" for label, path in required_files.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing map input files:\n" + "\n".join(missing))

    nsdgeneral, reference = load_roi(args.data_root, args.subject, "nsdgeneral")
    prf, _ = load_roi(args.data_root, args.subject, "prf-visualrois")
    coordinates = np.argwhere(nsdgeneral > 0)
    volume_masks = paper_visual_roi_masks(nsdgeneral, prf)
    masks = {
        name: mask[tuple(coordinates.T)].astype(bool)
        for name, mask in volume_masks.items()
    }

    core = validate_voxel_table(
        pd.read_csv(args.core_voxel_metrics),
        coordinates,
        required_columns=("test_r2",),
        label="core voxel metrics",
    )
    transfer = validate_voxel_table(
        pd.read_csv(args.transfer_voxel_metrics),
        coordinates,
        required_columns=("vision_target_r2", "imagery_target_r2"),
        label="transfer voxel metrics",
    )
    core_r2 = core["test_r2"].to_numpy(dtype=float)
    vision_r2 = transfer["vision_target_r2"].to_numpy(dtype=float)
    imagery_r2 = transfer["imagery_target_r2"].to_numpy(dtype=float)
    independently_predictable = np.isfinite(core_r2) & (
        core_r2 > args.core_r2_threshold
    )
    if not independently_predictable.any():
        raise ValueError(
            "No voxels pass the independent core-test threshold "
            f"R-squared > {args.core_r2_threshold:g}"
        )
    difference = vision_r2 - imagery_r2
    difference[
        ~independently_predictable | ~np.isfinite(difference)
    ] = 0

    core_sqrt = positive_sqrt(core_r2)
    vision_sqrt = positive_sqrt(vision_r2, independently_predictable)
    imagery_sqrt = positive_sqrt(imagery_r2, independently_predictable)
    maps = {
        "core_test_r2": core_r2,
        "core_test_sqrt_positive_r2": core_sqrt,
        "vision_target_r2": vision_r2,
        "imagery_target_r2": imagery_r2,
        "vision_minus_imagery_target_r2_core_predictable": difference,
        "vision_sqrt_positive_r2_core_predictable": vision_sqrt,
        "imagery_sqrt_positive_r2_core_predictable": imagery_sqrt,
        "core_predictable_voxel_mask": independently_predictable.astype(np.float32),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, values in maps.items():
        image = values_to_image(values, coordinates, reference)
        path = args.output_dir / f"{name}.nii.gz"
        nib.save(image, path)
        images[name] = image

    if args.background_image is not None:
        background = nib.load(args.background_image)
        if background.shape != reference.shape or not np.allclose(
            background.affine, reference.affine
        ):
            raise ValueError(
                "--background-image must already match the func1pt8mm grid and affine"
            )
    else:
        background_header = reference.header.copy()
        background_header.set_data_dtype(np.float32)
        background = nib.Nifti1Image(
            (nsdgeneral > 0).astype(np.float32),
            reference.affine,
            background_header,
        )

    core_vmax = positive_vmax(core_sqrt)
    transfer_vmax = positive_vmax(vision_sqrt, imagery_sqrt)
    plot_slice_panel(
        [
            images["core_test_sqrt_positive_r2"],
            images["vision_sqrt_positive_r2_core_predictable"],
            images["imagery_sqrt_positive_r2_core_predictable"],
        ],
        [
            r"Held-out core NSD: $\sqrt{\max(R^2,0)}$",
            r"NSD-Imagery vision: $\sqrt{\max(R^2,0)}$ "
            "(core-test $R^2$ > threshold)",
            r"NSD-Imagery imagery: $\sqrt{\max(R^2,0)}$ "
            "(same independent mask)",
        ],
        background=background,
        output=args.output_dir / "encoder_r2_native_slices.png",
        dpi=args.dpi,
        vmaxes=[core_vmax, transfer_vmax, transfer_vmax],
        cmaps=["Blues", "Blues", "Blues"],
        symmetric=[False, False, False],
    )
    plot_slice_panel(
        [images["vision_minus_imagery_target_r2_core_predictable"]],
        [r"Vision minus imagery strict target $R^2$ (core-predictable voxels)"],
        background=background,
        output=args.output_dir / "vision_imagery_r2_difference.png",
        dpi=args.dpi,
        vmaxes=[signed_vmax(difference)],
        cmaps=["RdBu_r"],
        symmetric=[True],
    )

    roi_summary_all = summarize_regions(transfer, masks)
    roi_summary_predictable = summarize_regions(
        transfer,
        masks,
        eligible=independently_predictable,
        voxel_set=f"core_test_r2_gt_{args.core_r2_threshold:g}",
    )
    roi_summary = pd.concat(
        [roi_summary_all, roi_summary_predictable], ignore_index=True
    )
    roi_summary.to_csv(args.output_dir / "transfer_r2_roi_summary.csv", index=False)
    plot_roi_summary(
        roi_summary_predictable,
        args.output_dir / "transfer_r2_roi_summary.png",
        args.dpi,
        core_r2_threshold=args.core_r2_threshold,
    )

    print(
        f"Core-predictable voxels: {int(independently_predictable.sum())} / "
        f"{len(independently_predictable)} "
        f"(test R-squared > {args.core_r2_threshold:g})"
    )
    print(f"Wrote NIfTI maps and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
