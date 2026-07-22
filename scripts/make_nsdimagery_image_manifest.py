#!/usr/bin/env python
"""Create an A/B target-image manifest for encoder evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery import load_target_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stimulus-sets", nargs="+", choices=("A", "B"), default=("A", "B")
    )
    args = parser.parse_args()

    targets = load_target_table(args.data_root)
    targets = targets[targets["stimulus_set"].isin(args.stimulus_sets)].copy()
    targets = targets.sort_values(["stimulus_set", "target_number"]).reset_index(
        drop=True
    )
    targets.insert(0, "row_id", np.arange(len(targets)))
    targets["sample"] = 1
    missing = [str(path) for path in targets["image_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing target images:\n" + "\n".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output, index=False)
    print(targets[["row_id", "stimulus_set", "target_number", "image_path"]])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
