#!/usr/bin/env python
"""Validate the minimal NSD-Imagery download without loading beta arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nsdimagery.io import infer_beta_layout, paths_for_subject, validate_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default="data/nsd", type=Path)
    args = parser.parse_args()

    manifest = validate_download(args.data_root)
    print(manifest.to_string(index=False))
    if not manifest["ok"].all():
        print("\nERROR: at least one expected item is missing.")
        return 1

    print("\nShape checks:")
    for subject in range(1, 9):
        paths = paths_for_subject(args.data_root, subject)
        with h5py.File(paths["beta"], "r") as handle:
            beta_shape = handle["betas"].shape
            beta_dtype = handle["betas"].dtype
        roi_shapes = {
            roi: nib.load(paths[roi]).shape
            for roi in ("nsdgeneral", "prf-visualrois", "streams")
        }
        layouts = {
            roi: infer_beta_layout(beta_shape, shape)
            for roi, shape in roi_shapes.items()
        }
        shapes_match = len(set(layouts.values())) == 1
        print(
            f"subj{subject:02d}: beta={beta_shape} {beta_dtype}; "
            f"ROI shapes={roi_shapes}; layout={layouts}; match={shapes_match}"
        )
        if not shapes_match:
            return 1

    print("\nAll expected files and array shapes look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
