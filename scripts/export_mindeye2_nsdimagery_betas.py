#!/usr/bin/env python3
"""Export NSD-Imagery betas in the tensor format used by MindEye2 code."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nsdimagery import (  # noqa: E402
    build_run_table,
    extract_masked_betas,
    load_roi,
    paths_for_subject,
    zscore_within_groups,
)


MIND_EYE_VOXEL_COUNTS = {
    1: 15724,
    2: 14278,
    3: 15226,
    4: 13153,
    5: 13039,
    6: 17907,
    7: 12682,
    8: 14386,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the full nsdgeneral mask, divide NSD int16 betas by 300, "
            "and Z-score each run exactly as in the released imagery routine."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--subjects", type=int, nargs="+", choices=range(1, 9), default=(1,)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "MindEye data directory; writes "
            "preprocessed_data/subjectN/nsd_imagery.pt"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_labels() -> np.ndarray:
    table = build_run_table()
    groups = np.full(720, None, dtype=object)
    for row in table.itertuples():
        groups[
            row.beta_start_python : row.beta_stop_python_exclusive
        ] = row.run_name
    if any(value is None for value in groups):
        raise AssertionError("Run table did not label all 720 betas")
    return groups


def main() -> None:
    args = parse_args()
    groups = run_labels()
    for subject in args.subjects:
        output = (
            args.output_root
            / "preprocessed_data"
            / f"subject{subject}"
            / "nsd_imagery.pt"
        )
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output} already exists; pass --overwrite to replace it"
            )
        general, _ = load_roi(args.data_root, subject, "nsdgeneral")
        betas, _ = extract_masked_betas(
            paths_for_subject(args.data_root, subject)["beta"], general
        )
        expected = MIND_EYE_VOXEL_COUNTS[subject]
        if betas.shape != (720, expected):
            raise ValueError(
                f"subj{subject:02d}: expected (720, {expected}) in full "
                f"nsdgeneral order; found {betas.shape}"
            )
        normalized = zscore_within_groups(betas, groups)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(normalized), output)
        print(f"Wrote {output} {tuple(normalized.shape)}")


if __name__ == "__main__":
    main()
