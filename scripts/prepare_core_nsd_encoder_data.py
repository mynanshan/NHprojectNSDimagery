#!/usr/bin/env python
"""Average core-NSD repeats into leakage-safe image-level encoder data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery.encoding import (  # noqa: E402
    CORE_NSD_SESSIONS,
    assign_image_splits,
    load_core_nsd_trial_ids,
)
from nsdimagery.io import load_roi, paper_visual_roi_masks  # noqa: E402


def resolved_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read core-NSD beta sessions inside nsdgeneral, average repeated "
            "presentations of each image, and make unique-image data splits."
        )
    )
    parser.add_argument("--data-root", type=resolved_path, required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 9), required=True)
    parser.add_argument(
        "--sessions",
        type=int,
        help="Number of consecutive sessions to use (default: every completed session)",
    )
    parser.add_argument("--output-dir", type=resolved_path, required=True)
    parser.add_argument(
        "--split-mode", choices=("shared1000", "random"), default="shared1000"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--beta-scale",
        type=float,
        default=300.0,
        help="NSD integer beta scale factor (default: 300)",
    )
    parser.add_argument(
        "--normalization",
        choices=("session_zscore", "none"),
        default="session_zscore",
        help="Per-voxel preprocessing before repeat averaging (default: session_zscore)",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        help="Random nsdgeneral voxel subset for a smoke test; omit for all voxels",
    )
    parser.add_argument(
        "--exclude-73k-ids",
        type=resolved_path,
        help="Optional text/CSV file of one-based NSD 73K IDs to exclude",
    )
    return parser.parse_args()


def read_excluded_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(path)
    values = []
    for token in path.read_text().replace(",", " ").split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No integer 73K IDs found in {path}")
    if any(value < 1 or value > 73000 for value in values):
        raise ValueError("Excluded 73K IDs must be one-based values from 1 to 73000")
    return set(values)


def session_beta_path(data_root: Path, subject: int, session: int) -> Path:
    return (
        data_root
        / "nsddata_betas"
        / "ppdata"
        / f"subj{subject:02d}"
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR"
        / f"betas_session{session:02d}.nii.gz"
    )


def extract_session_betas(
    path: Path,
    coordinates: np.ndarray,
    spatial_shape: tuple[int, int, int],
    *,
    beta_scale: float,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(path)
    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.shape[:3] == spatial_shape and values.shape[-1] == 750:
        selected = values[tuple(coordinates.T) + (slice(None),)].T
    elif values.shape[0] == 750 and values.shape[1:] == spatial_shape:
        selected = values[(slice(None),) + tuple(coordinates.T)]
    else:
        raise ValueError(
            f"{path} has shape {values.shape}; expected {spatial_shape} + 750 trials"
        )
    if selected.shape != (750, len(coordinates)):
        raise AssertionError(f"Unexpected masked beta shape {selected.shape}")
    selected /= float(beta_scale)
    return selected


def main() -> None:
    args = parse_args()
    available = CORE_NSD_SESSIONS[args.subject - 1]
    n_sessions = args.sessions if args.sessions is not None else available
    if not 1 <= n_sessions <= available:
        raise ValueError(
            f"subj{args.subject:02d} has 1 through {available} completed sessions"
        )
    if args.beta_scale <= 0:
        raise ValueError("--beta-scale must be positive")

    nsdgeneral, _ = load_roi(args.data_root, args.subject, "nsdgeneral")
    prf_visualrois, _ = load_roi(
        args.data_root, args.subject, "prf-visualrois"
    )
    paper_regions = paper_visual_roi_masks(nsdgeneral, prf_visualrois)
    coordinates = np.argwhere(nsdgeneral > 0)
    if args.max_voxels is not None:
        if args.max_voxels < 2:
            raise ValueError("--max-voxels must be at least two")
        if len(coordinates) > args.max_voxels:
            rng = np.random.default_rng(args.seed)
            coordinates = coordinates[
                np.sort(
                    rng.choice(len(coordinates), args.max_voxels, replace=False)
                )
            ]

    trial_10k, _, subject_73k = load_core_nsd_trial_ids(
        args.data_root, args.subject, n_sessions=n_sessions
    )
    beta_sum = np.zeros((10000, len(coordinates)), dtype=np.float32)
    repeat_count = np.zeros(10000, dtype=np.int16)
    for session in range(1, n_sessions + 1):
        print(f"Reading subj{args.subject:02d} session {session:02d}/{n_sessions:02d}")
        session_betas = extract_session_betas(
            session_beta_path(args.data_root, args.subject, session),
            coordinates,
            nsdgeneral.shape,
            beta_scale=args.beta_scale,
        )
        if args.normalization == "session_zscore":
            session_mean = session_betas.mean(axis=0, keepdims=True)
            session_scale = session_betas.std(axis=0, ddof=1, keepdims=True)
            session_scale[
                ~np.isfinite(session_scale) | (session_scale == 0)
            ] = 1
            session_betas = (session_betas - session_mean) / session_scale
        session_ids = trial_10k[(session - 1) * 750 : session * 750]
        np.add.at(beta_sum, session_ids, session_betas)
        np.add.at(repeat_count, session_ids, 1)

    observed_10k = np.flatnonzero(repeat_count > 0)
    averaged = beta_sum[observed_10k] / repeat_count[observed_10k, None]
    split = assign_image_splits(
        observed_10k,
        mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    excluded_ids = read_excluded_ids(args.exclude_73k_ids)
    keep = np.asarray(
        [int(subject_73k[index]) not in excluded_ids for index in observed_10k]
    )
    excluded_count = int((~keep).sum())
    observed_10k = observed_10k[keep]
    averaged = averaged[keep]
    split = split[keep]

    manifest = pd.DataFrame(
        {
            "row_id": np.arange(len(observed_10k)),
            "subject": f"subj{args.subject:02d}",
            "subject_10k_index": observed_10k + 1,
            "nsd_73k_id": subject_73k[observed_10k],
            "n_repeats": repeat_count[observed_10k],
            "beta_normalization": args.normalization,
            "split": split,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"core_subj{args.subject:02d}"
    np.save(str(prefix) + "_betas.npy", averaged.astype(np.float32))
    np.save(str(prefix) + "_coordinates.npy", coordinates.astype(np.int16))
    manifest.to_csv(str(prefix) + "_manifest.csv", index=False)
    voxel_regions = pd.DataFrame(
        {
            "voxel": np.arange(len(coordinates)),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
            **{
                name: mask[tuple(coordinates.T)].astype(bool)
                for name, mask in paper_regions.items()
            },
        }
    )
    voxel_regions.to_csv(str(prefix) + "_voxel_regions.csv", index=False)

    print()
    print(manifest.groupby("split").size().rename("unique_images").to_string())
    print(f"Voxels: {len(coordinates)}")
    print(f"Excluded image identities: {excluded_count}")
    print(f"Wrote {prefix}_betas.npy")
    print(f"Wrote {prefix}_coordinates.npy")
    print(f"Wrote {prefix}_manifest.csv")
    print(f"Wrote {prefix}_voxel_regions.csv")


if __name__ == "__main__":
    main()
