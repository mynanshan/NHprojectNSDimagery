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
from scipy.io import loadmat


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


def load_target_table(data_root: str | Path) -> pd.DataFrame:
    """Return the six cue-to-target mappings for stimulus sets A, B, and C."""
    experiment_dir = (
        Path(data_root) / "nsddata" / "experiments" / "nsdimagery"
    )
    rows: list[dict[str, object]] = []
    for stimulus_set in ("A", "B", "C"):
        path = experiment_dir / f"{stimulus_set}_pair_list.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Missing target-pair file: {path}")
        pair_list = np.asarray(
            loadmat(path, squeeze_me=True, struct_as_record=False)["pair_list"]
        )
        if pair_list.shape != (6, 3):
            raise ValueError(
                f"Expected a 6 x 3 pair list in {path}; found {pair_list.shape}"
            )
        for target_number, target_name, target_code in pair_list:
            image_path = (
                experiment_dir
                / "rawtargetimages"
                / f"set{stimulus_set}"
                / str(target_name)
            )
            rows.append(
                {
                    "stimulus_set": stimulus_set,
                    "target_number": int(target_number),
                    "target_code": str(target_code),
                    "target_name": str(target_name),
                    "image_path": image_path if stimulus_set in {"A", "B"} else None,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["stimulus_set", "target_number"], ignore_index=True
    )


def summarize_glmsingle_design(data_root: str | Path) -> pd.DataFrame:
    """Count modeled events in each released GLMsingle run design."""
    path = (
        Path(data_root)
        / "nsddata"
        / "experiments"
        / "nsdimagery"
        / "designmatrixGLMsingle.mat"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing GLMsingle design: {path}")
    stimulus = np.atleast_1d(
        loadmat(path, squeeze_me=True, struct_as_record=False)["stimulus"]
    )
    if len(stimulus) != len(RUN_SPECS):
        raise ValueError(
            f"Expected {len(RUN_SPECS)} run designs in {path}; found {len(stimulus)}"
        )

    rows = []
    for spec, design in zip(RUN_SPECS, stimulus):
        design = np.asarray(design)
        n_modeled_events = int(np.count_nonzero(design))
        rows.append(
            {
                "run_name": spec.name,
                "task": spec.task,
                "design_shape": design.shape,
                "n_modeled_events": n_modeled_events,
                "expected_betas": spec.n_betas,
                "ok": n_modeled_events == spec.n_betas,
            }
        )
    return pd.DataFrame(rows)


def build_event_table(data_root: str | Path, subject: int | str) -> pd.DataFrame:
    """Align vision and imagery behavioral trials to their beta indices.

    Vision trials are labeled by ``CONDITION``, the image actually presented.
    Imagery trials are labeled by ``CUE``, the target participants imagined.
    Attention is deliberately excluded because it has two beta epochs per trial.
    """
    subject_name = _subject(subject)
    design_summary = summarize_glmsingle_design(data_root)
    if not design_summary["ok"].all():
        raise ValueError(
            "Released GLMsingle design counts do not match the expected beta blocks:\n"
            f"{design_summary.to_string(index=False)}"
        )
    behavior = load_behavior(data_root, subject)
    run_table = build_run_table().set_index("run_name")
    target_table = load_target_table(data_root)
    frames = []

    for spec in RUN_SPECS:
        if spec.task == "attention":
            continue
        frame = behavior.loc[behavior["RUN_NAME"] == spec.name].copy()
        if len(frame) != spec.n_behavior_trials:
            raise ValueError(
                f"{subject_name} {spec.name}: expected {spec.n_behavior_trials} "
                f"behavioral rows; found {len(frame)}"
            )
        frame["TRIAL"] = pd.to_numeric(frame["TRIAL"], errors="raise").astype(int)
        frame = frame.sort_values("TRIAL", ignore_index=True)
        expected_trials = np.arange(1, spec.n_behavior_trials + 1)
        if not np.array_equal(frame["TRIAL"].to_numpy(), expected_trials):
            raise ValueError(
                f"{subject_name} {spec.name}: TRIAL must contain each integer 1-48"
            )

        run = run_table.loc[spec.name]
        frame["subject"] = subject_name
        frame["run_name"] = spec.name
        frame["task"] = spec.task
        frame["stimulus_set"] = spec.stimulus_set
        frame["beta_index"] = (
            int(run["beta_start_python"]) + np.arange(spec.n_behavior_trials)
        )
        frame["beta_number"] = frame["beta_index"] + 1
        label_column = "CONDITION" if spec.task == "vision" else "CUE"
        frame["target_code"] = frame[label_column].astype(str)
        frames.append(frame)

    events = pd.concat(frames, ignore_index=True)
    events = events.merge(
        target_table,
        how="left",
        on=["stimulus_set", "target_code"],
        validate="many_to_one",
    )
    if events["target_number"].isna().any():
        missing = events.loc[
            events["target_number"].isna(), ["run_name", "TRIAL", "target_code"]
        ]
        raise ValueError(f"Unmapped target codes:\n{missing.to_string(index=False)}")
    events["target_number"] = events["target_number"].astype(int)
    events["repeat"] = (
        events.groupby(["task", "stimulus_set", "target_code"]).cumcount() + 1
    )

    expected_total = sum(
        spec.n_behavior_trials for spec in RUN_SPECS if spec.task != "attention"
    )
    if len(events) != expected_total or not events["beta_index"].is_unique:
        raise AssertionError("Vision/imagery event table failed size or uniqueness checks")
    counts = events.groupby(["task", "stimulus_set", "target_code"]).size()
    expected_counts = counts.index.get_level_values("task").map(
        {"vision": 8, "imagery": 16}
    )
    if not np.array_equal(counts.to_numpy(), expected_counts.to_numpy()):
        raise AssertionError(
            "Expected 8 vision and 16 imagery repetitions for every target"
        )

    leading_columns = [
        "subject",
        "run_name",
        "task",
        "stimulus_set",
        "TRIAL",
        "beta_index",
        "beta_number",
        "target_number",
        "target_code",
        "target_name",
        "repeat",
        "image_path",
    ]
    remaining_columns = [
        column for column in events.columns if column not in leading_columns
    ]
    return events[leading_columns + remaining_columns]


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


def paper_visual_roi_masks(
    nsdgeneral: np.ndarray, prf_visualrois: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the nested visual ROIs used for the paper's brain metric.

    NSD's volumetric ``prf-visualrois`` labels are 1/2 for V1d/V1v,
    3/4 for V2d/V2v, 5/6 for V3d/V3v, and 7 for hV4. The paper defines
    higher visual cortex as the set complement of V1--V4 inside
    ``nsdgeneral``; it is not the broad higher-level ``streams`` ROI used in
    the project's earlier exploratory notebooks.
    """
    nsdgeneral = np.asarray(nsdgeneral)
    prf_visualrois = np.asarray(prf_visualrois)
    if nsdgeneral.shape != prf_visualrois.shape:
        raise ValueError("nsdgeneral and prf_visualrois must have the same shape")
    if nsdgeneral.ndim != 3:
        raise ValueError("ROI volumes must be three-dimensional")

    visual = nsdgeneral > 0
    components = {
        "V1": visual & np.isin(prf_visualrois, (1, 2)),
        "V2": visual & np.isin(prf_visualrois, (3, 4)),
        "V3": visual & np.isin(prf_visualrois, (5, 6)),
        "V4": visual & (prf_visualrois == 7),
    }
    early = np.logical_or.reduce(tuple(components.values()))
    higher = visual & ~early
    if not visual.any() or not early.any() or not higher.any():
        raise ValueError("Paper visual ROI construction produced an empty region")
    return {
        "visual_cortex": visual,
        "early_visual": early,
        "higher_visual": higher,
        **components,
    }


def mask_at_coordinates(mask: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Project a 3-D ROI mask into an extracted voxel-column ordering."""
    mask = np.asarray(mask)
    coordinates = np.asarray(coordinates)
    if mask.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must be voxels x 3")
    if not np.issubdtype(coordinates.dtype, np.integer):
        raise ValueError("coordinates must contain integer indices")
    if len(coordinates) == 0:
        raise ValueError("coordinates must not be empty")
    if np.any(coordinates < 0) or any(
        np.any(coordinates[:, axis] >= size)
        for axis, size in enumerate(mask.shape)
    ):
        raise ValueError("coordinates fall outside the mask volume")
    return np.asarray(mask[tuple(coordinates.T)], dtype=bool)


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
