"""I/O helpers for the minimal NSD-Imagery download.

The important performance rule is that the HDF5 beta array should stay on disk.
Read selected voxels across all 720 events; never load ``/betas[:]`` casually.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    name: str
    task: str
    stimulus_set: str
    n_behavior_trials: int
    n_betas: int


# Acquisition/GLM order documented by NSD. Attention has two modeled epochs
# per behavioral trial; vision and imagery have one.
RUN_SPECS = (
    RunSpec("visA", "vision", "A", 48, 48),
    RunSpec("attA", "attention", "A", 48, 96),
    RunSpec("imgA_1", "imagery", "A", 48, 48),
    RunSpec("visB", "vision", "B", 48, 48),
    RunSpec("attB", "attention", "B", 48, 96),
    RunSpec("imgB_1", "imagery", "B", 48, 48),
    RunSpec("visC", "vision", "C", 48, 48),
    RunSpec("attC", "attention", "C", 48, 96),
    RunSpec("imgC_1", "imagery", "C", 48, 48),
    RunSpec("imgA_2", "imagery", "A", 48, 48),
    RunSpec("imgB_2", "imagery", "B", 48, 48),
    RunSpec("imgC_2", "imagery", "C", 48, 48),
)


def find_data_root(start: str | Path = ".") -> Path:
    """Find ``data/nsd`` from the repo root or the notebooks directory."""
    start = Path(start).expanduser().resolve()
    candidates = [
        start,
        start / "data" / "nsd",
        start.parent / "data" / "nsd",
    ]
    for candidate in candidates:
        if (
            (candidate / "nsddata").is_dir()
            and (candidate / "nsddata_betas").is_dir()
        ):
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate the NSD data root. Tried:\n"
        f"{tried}\nSet DATA_ROOT explicitly in the notebook."
    )


def _subject(subject: int | str) -> str:
    if isinstance(subject, int):
        subject = f"{subject:02d}"
    subject = str(subject).removeprefix("subj")
    if subject not in {f"{i:02d}" for i in range(1, 9)}:
        raise ValueError("subject must be 01 through 08")
    return f"subj{subject}"


def paths_for_subject(data_root: str | Path, subject: int | str) -> dict[str, Path]:
    root = Path(data_root)
    subj = _subject(subject)
    roi_dir = root / "nsddata" / "ppdata" / subj / "func1pt8mm" / "roi"
    return {
        "beta": root
        / "nsddata_betas"
        / "ppdata"
        / subj
        / "func1pt8mm"
        / "nsdimagerybetas_fithrf_GLMdenoise_RR"
        / "betas_nsdimagery.hdf5",
        "behavior": root / "nsddata" / "bdata" / "nsdimagery",
        "nsdgeneral": roi_dir / "nsdgeneral.nii.gz",
        "prf-visualrois": roi_dir / "prf-visualrois.nii.gz",
        "streams": roi_dir / "streams.nii.gz",
        "labels": root / "nsddata" / "freesurfer" / subj / "label",
    }


def validate_download(data_root: str | Path) -> pd.DataFrame:
    """Return a manifest table; ``ok`` must be true in every row."""
    root = Path(data_root)
    rows: list[dict[str, object]] = []
    for subject_number in range(1, 9):
        subj = f"subj{subject_number:02d}"
        paths = paths_for_subject(root, subject_number)
        for kind in ("beta", "nsdgeneral", "prf-visualrois", "streams"):
            path = paths[kind]
            rows.append(
                {
                    "subject": subj,
                    "kind": kind,
                    "path": path,
                    "ok": path.is_file(),
                    "size_MiB": path.stat().st_size / 2**20 if path.is_file() else np.nan,
                }
            )
        behavior_files = sorted(paths["behavior"].glob(f"nsdimagery_{subj}_*.tsv"))
        rows.append(
            {
                "subject": subj,
                "kind": "behavior TSVs (expected 12)",
                "path": paths["behavior"],
                "ok": len(behavior_files) == 12,
                "size_MiB": sum(p.stat().st_size for p in behavior_files) / 2**20,
            }
        )
    return pd.DataFrame(rows)


def load_behavior(data_root: str | Path, subject: int | str) -> pd.DataFrame:
    """Read and concatenate the 12 behavioral TSVs for one subject."""
    subj = _subject(subject)
    behavior_dir = paths_for_subject(data_root, subject)["behavior"]
    frames = []
    for path in sorted(behavior_dir.glob(f"nsdimagery_{subj}_*.tsv")):
        frame = pd.read_csv(path, sep="\t")
        frame.insert(0, "RUN_NAME", path.stem.replace(f"nsdimagery_{subj}_", ""))
        frame.insert(0, "SOURCE_FILE", path.name)
        frames.append(frame)
    if len(frames) != 12:
        raise FileNotFoundError(f"Expected 12 TSVs for {subj}; found {len(frames)}")
    return pd.concat(frames, ignore_index=True)


def build_run_table() -> pd.DataFrame:
    """Coarse beta blocks. Attention's within-run epoch order needs the GLM design."""
    rows = []
    start = 0
    for run_number, spec in enumerate(RUN_SPECS, start=1):
        stop = start + spec.n_betas
        rows.append(
            {
                "run_number": run_number,
                "run_name": spec.name,
                "task": spec.task,
                "stimulus_set": spec.stimulus_set,
                "n_behavior_trials": spec.n_behavior_trials,
                "n_betas": spec.n_betas,
                "beta_start_python": start,
                "beta_stop_python_exclusive": stop,
                "beta_first_human": start + 1,
                "beta_last_human": stop,
            }
        )
        start = stop
    table = pd.DataFrame(rows)
    if start != 720:
        raise AssertionError(f"Run specification produced {start}, not 720, betas")
    return table


def describe_hdf5(path: str | Path) -> pd.DataFrame:
    """List HDF5 groups/datasets without loading dataset contents."""
    rows = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                rows.append(
                    {
                        "name": f"/{name}",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "chunks": obj.chunks,
                        "compression": obj.compression,
                    }
                )
        handle.visititems(visitor)
    return pd.DataFrame(rows)


def infer_beta_layout(
    dataset_shape: tuple[int, ...], spatial_shape: tuple[int, ...]
) -> str:
    """Identify how Python exposes the MATLAB-written NSD HDF5 dimensions.

    The NSD manual uses MATLAB order ``X × Y × Z × events``. In the released
    files, h5py commonly exposes the fully reversed HDF5 dimension convention
    as ``events × Z × Y × X``. Supporting all plausible layouts prevents a
    silent spatial-axis mistake.
    """
    dataset_shape = tuple(dataset_shape)
    spatial_shape = tuple(spatial_shape)
    if dataset_shape[0] == 720:
        if dataset_shape[1:] == spatial_shape:
            return "events_first_xyz"
        if dataset_shape[1:] == spatial_shape[::-1]:
            return "events_first_zyx"
    if dataset_shape[-1] == 720:
        if dataset_shape[:-1] == spatial_shape:
            return "events_last_xyz"
        if dataset_shape[:-1] == spatial_shape[::-1]:
            return "events_last_zyx"
    raise ValueError(
        f"Cannot reconcile beta shape {dataset_shape} with ROI shape {spatial_shape} "
        "and 720 expected events"
    )


def load_roi(data_root: str | Path, subject: int | str, roi: str):
    """Return ``(integer array, nibabel image)`` for a volumetric ROI."""
    if roi not in {"nsdgeneral", "prf-visualrois", "streams"}:
        raise ValueError("roi must be nsdgeneral, prf-visualrois, or streams")
    image = nib.load(paths_for_subject(data_root, subject)[roi])
    return np.asanyarray(image.dataobj).astype(np.int16, copy=False), image


def read_ctab(path: str | Path) -> pd.DataFrame:
    """Read the useful label/name columns of a FreeSurfer color table."""
    rows = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if fields[0].lstrip("-").isdigit() and len(fields) >= 2:
            rows.append({"label": int(fields[0]), "name": fields[1]})
    return pd.DataFrame(rows).drop_duplicates("label").sort_values("label")


def extract_masked_betas(
    beta_path: str | Path,
    mask: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    max_voxels: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected voxel time courses and return percent-signal-change betas.

    Returns
    -------
    betas : ndarray, shape (720 events, selected voxels)
    coordinates : ndarray, shape (selected voxels, 3)
    """
    if labels is None:
        selected = mask > 0
    else:
        selected = np.isin(mask, list(labels))
    coordinates = np.argwhere(selected)
    if coordinates.size == 0:
        raise ValueError("The selected ROI contains no voxels")
    if max_voxels is not None and len(coordinates) > max_voxels:
        rng = np.random.default_rng(seed)
        coordinates = coordinates[rng.choice(len(coordinates), max_voxels, replace=False)]

    with h5py.File(beta_path, "r") as handle:
        dataset = handle["betas"]
        layout = infer_beta_layout(dataset.shape, mask.shape)
        values = np.empty((720, len(coordinates)), dtype=np.float32)
        if layout == "events_first_xyz":
            for column, (x, y, z) in enumerate(coordinates):
                values[:, column] = dataset[:, x, y, z]
        elif layout == "events_first_zyx":
            for column, (x, y, z) in enumerate(coordinates):
                values[:, column] = dataset[:, z, y, x]
        elif layout == "events_last_xyz":
            for column, (x, y, z) in enumerate(coordinates):
                values[:, column] = dataset[x, y, z, :]
        else:  # events_last_zyx
            for column, (x, y, z) in enumerate(coordinates):
                values[:, column] = dataset[z, y, x, :]

    # NSD stores these int16 values after multiplying percent signal change by 300.
    values /= 300.0
    return values, coordinates
