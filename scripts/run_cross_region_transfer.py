#!/usr/bin/env python
"""Run collaborator-inspired visual-to-parietal NSD-Imagery transfer tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery import (  # noqa: E402
    STREAM_ROI_LABELS,
    balanced_crossvalidated_rdm,
    balanced_split_reliability,
    build_event_table,
    crossvalidated_dot_rdm,
    exact_class_label_permutation_test,
    exact_label_permutation_test,
    exact_sign_flip_test,
    extract_masked_betas,
    fit_region_alignment,
    independent_group_reliability,
    leave_one_target_out_rdm_spearman,
    load_roi,
    nearest_centroid_predict,
    paths_for_subject,
    rdm_spearman,
    stream_roi_masks,
    target_rdm,
    zscore_within_groups,
)


PRIMARY_VISUAL = "visual_streams"
PRIMARY_PARIETAL = "dorsal_parietal"
ATOMIC_ROIS = ("early", "ventral", "lateral", "midparietal", "parietal")
RDM_ROIS = ATOMIC_ROIS + (PRIMARY_VISUAL, PRIMARY_PARIETAL)
STIMULUS_SETS = ("A", "B")
TASKS = ("vision", "imagery")
PLANNED_COMPARISONS = {
    "same_visual": (PRIMARY_VISUAL, PRIMARY_VISUAL),
    "cross_visual_to_parietal": (PRIMARY_VISUAL, PRIMARY_PARIETAL),
    "within_parietal": (PRIMARY_PARIETAL, PRIMARY_PARIETAL),
    "reverse_parietal_to_visual": (PRIMARY_PARIETAL, PRIMARY_VISUAL),
}


def parse_subjects(value: str) -> tuple[int, ...]:
    subjects: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(part) for part in token.split("-", maxsplit=1))
            subjects.extend(range(start, stop + 1))
        else:
            subjects.append(int(token))
    subjects = sorted(set(subjects))
    if not subjects or any(subject < 1 or subject > 8 for subject in subjects):
        raise argparse.ArgumentTypeError("subjects must be selected from 1 through 8")
    return tuple(subjects)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Compare same-visual and visual-to-parietal vision-imagery "
            "representational transfer, then run an optional vision-only "
            "cross-region alignment decoder."
        )
    )
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--subjects", type=parse_subjects, default=parse_subjects("1-8"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--max-voxels", type=int, default=1200)
    result.add_argument("--n-rdm-splits", type=int, default=100)
    result.add_argument("--n-alignment-permutations", type=int, default=100)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--skip-aligned-decoder", action="store_true")
    return result


def roi_cache_path(
    cache_dir: Path,
    subject: int,
    roi_name: str,
    n_voxels: int,
    seed: int,
) -> Path:
    return cache_dir / (
        f"subj{subject:02d}_{roi_name}_n{n_voxels}_seed{seed}.npz"
    )


def load_subject_roi_patterns(
    *,
    data_root: Path,
    cache_dir: Path,
    subject: int,
    events: pd.DataFrame,
    max_voxels: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    streams, _ = load_roi(data_root, subject, "streams")
    masks = stream_roi_masks(streams, RDM_ROIS)
    available = {name: int(mask.sum()) for name, mask in masks.items()}
    matched_primary = min(
        max_voxels,
        available[PRIMARY_VISUAL],
        available[PRIMARY_PARIETAL],
    )
    patterns: dict[str, np.ndarray] = {}
    count_rows = []
    beta_path = paths_for_subject(data_root, subject)["beta"]
    event_indices = events["beta_index"].to_numpy()
    run_names = events["run_name"].to_numpy()

    for roi_index, roi_name in enumerate(RDM_ROIS):
        requested = (
            matched_primary
            if roi_name in {PRIMARY_VISUAL, PRIMARY_PARIETAL}
            else min(max_voxels, available[roi_name])
        )
        roi_seed = seed + subject * 100 + roi_index
        path = roi_cache_path(
            cache_dir, subject, roi_name, requested, roi_seed
        )
        if path.is_file():
            with np.load(path) as cached:
                betas = cached["betas"]
                coordinates = cached["coordinates"]
            source = "cache"
        else:
            betas, coordinates = extract_masked_betas(
                beta_path,
                streams,
                labels=STREAM_ROI_LABELS[roi_name],
                max_voxels=requested,
                seed=roi_seed,
            )
            np.savez_compressed(path, betas=betas, coordinates=coordinates)
            source = "extracted"
        trial_patterns = betas[event_indices]
        patterns[roi_name] = zscore_within_groups(trial_patterns, run_names)
        count_rows.append(
            {
                "subject": subject,
                "phase": "pilot" if subject == 1 else "validation",
                "roi": roi_name,
                "stream_labels": "+".join(
                    str(label) for label in STREAM_ROI_LABELS[roi_name]
                ),
                "available_voxels": available[roi_name],
                "used_voxels": len(coordinates),
                "cache_status": source,
            }
        )
        print(
            f"  {roi_name}: {len(coordinates)} / {available[roi_name]} "
            f"voxels ({source})"
        )
    return patterns, count_rows


def task_frame(
    events: pd.DataFrame, stimulus_set: str, task: str
) -> pd.DataFrame:
    return events[
        events["stimulus_set"].eq(stimulus_set) & events["task"].eq(task)
    ]


def build_subject_rdms(
    *,
    subject: int,
    events: pd.DataFrame,
    patterns: dict[str, np.ndarray],
    n_splits: int,
    seed: int,
) -> tuple[
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], np.ndarray],
    list[dict[str, object]],
]:
    standard_rdms = {}
    crossvalidated_rdms = {}
    reliability_rows = []
    for roi_index, roi_name in enumerate(RDM_ROIS):
        roi_patterns = patterns[roi_name]
        for set_index, stimulus_set in enumerate(STIMULUS_SETS):
            for task_index, task in enumerate(TASKS):
                frame = task_frame(events, stimulus_set, task)
                positions = frame.index.to_numpy()
                targets = frame["target_number"].to_numpy()
                standard, order = target_rdm(roi_patterns[positions], targets)
                if not np.array_equal(order, np.arange(1, 7)):
                    raise AssertionError("Expected target order 1 through 6")
                standard_rdms[(roi_name, stimulus_set, task)] = standard

                if task == "vision":
                    crossvalidated, cv_order = balanced_crossvalidated_rdm(
                        roi_patterns[positions],
                        targets,
                        n_splits=n_splits,
                        seed=(
                            seed
                            + subject * 10000
                            + roi_index * 100
                            + set_index * 10
                            + task_index
                        ),
                    )
                    reliability = balanced_split_reliability(
                        roi_patterns[positions],
                        targets,
                        n_splits=n_splits,
                        seed=(
                            seed
                            + 500000
                            + subject * 10000
                            + roi_index * 100
                            + set_index * 10
                        ),
                    )
                    reliability_value = float(np.median(reliability))
                    reliability_method = "balanced random repeat halves"
                else:
                    first = frame[frame["run_name"].str.endswith("_1")]
                    second = frame[frame["run_name"].str.endswith("_2")]
                    crossvalidated, cv_order = crossvalidated_dot_rdm(
                        roi_patterns[first.index.to_numpy()],
                        first["target_number"].to_numpy(),
                        roi_patterns[second.index.to_numpy()],
                        second["target_number"].to_numpy(),
                    )
                    # The run-to-run estimate is the less optimistic reliability
                    # diagnostic because the imagery task has two independent runs.
                    reliability_value = independent_group_reliability(
                        roi_patterns[positions],
                        targets,
                        frame["run_name"].to_numpy(),
                    )
                    reliability_method = "independent imagery runs"
                if not np.array_equal(cv_order, np.arange(1, 7)):
                    raise AssertionError("Expected crossvalidated target order 1-6")
                crossvalidated_rdms[(roi_name, stimulus_set, task)] = (
                    crossvalidated
                )
                reliability_rows.append(
                    {
                        "subject": subject,
                        "phase": "pilot" if subject == 1 else "validation",
                        "stimulus_set": stimulus_set,
                        "task": task,
                        "roi": roi_name,
                        "rdm_reliability": reliability_value,
                        "reliability_method": reliability_method,
                    }
                )
    return standard_rdms, crossvalidated_rdms, reliability_rows


def cross_region_rows(
    *,
    subject: int,
    standard_rdms: dict[tuple[str, str, str], np.ndarray],
    crossvalidated_rdms: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for stimulus_set in STIMULUS_SETS:
        for source_roi in RDM_ROIS:
            for target_roi in RDM_ROIS:
                vision = standard_rdms[(source_roi, stimulus_set, "vision")]
                imagery = standard_rdms[(target_roi, stimulus_set, "imagery")]
                permutation = exact_label_permutation_test(vision, imagery)
                cv_rho = rdm_spearman(
                    crossvalidated_rdms[
                        (source_roi, stimulus_set, "vision")
                    ],
                    crossvalidated_rdms[
                        (target_roi, stimulus_set, "imagery")
                    ],
                )
                leave_target = leave_one_target_out_rdm_spearman(
                    vision, imagery
                )
                rows.append(
                    {
                        "subject": subject,
                        "phase": "pilot" if subject == 1 else "validation",
                        "stimulus_set": stimulus_set,
                        "source_vision_roi": source_roi,
                        "target_imagery_roi": target_roi,
                        "standard_rho": permutation["observed"],
                        "crossvalidated_rho": cv_rho,
                        "subject_p_greater": permutation["p_greater"],
                        "leave_target_min": float(leave_target.min()),
                        "leave_target_median": float(np.median(leave_target)),
                        "leave_target_max": float(leave_target.max()),
                    }
                )
    return rows


def planned_comparison_rows(transfer: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for comparison, (source_roi, target_roi) in PLANNED_COMPARISONS.items():
        selected = transfer[
            transfer["source_vision_roi"].eq(source_roi)
            & transfer["target_imagery_roi"].eq(target_roi)
        ].copy()
        selected.insert(3, "comparison", comparison)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def contrast_rows(planned: pd.DataFrame) -> pd.DataFrame:
    wide = planned.pivot(
        index=["subject", "phase", "stimulus_set"],
        columns="comparison",
        values=["standard_rho", "crossvalidated_rho"],
    )
    rows = []
    definitions = {
        "cross_minus_same_visual": (
            "cross_visual_to_parietal",
            "same_visual",
        ),
        "cross_minus_within_parietal": (
            "cross_visual_to_parietal",
            "within_parietal",
        ),
        "within_parietal_minus_same_visual": (
            "within_parietal",
            "same_visual",
        ),
    }
    for (subject, phase, stimulus_set), values in wide.iterrows():
        for contrast, (left, right) in definitions.items():
            rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "stimulus_set": stimulus_set,
                    "contrast": contrast,
                    "standard_difference": (
                        values[("standard_rho", left)]
                        - values[("standard_rho", right)]
                    ),
                    "crossvalidated_difference": (
                        values[("crossvalidated_rho", left)]
                        - values[("crossvalidated_rho", right)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def sample_definitions(subjects: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    result = {"all_subjects": subjects}
    validation = tuple(subject for subject in subjects if subject != 1)
    if validation:
        result["subjects_02_08"] = validation
    return result


def summarize_signed_values(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
    subjects: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for sample, selected_subjects in sample_definitions(subjects).items():
        sample_frame = frame[frame["subject"].isin(selected_subjects)]
        for keys, group in sample_frame.groupby(group_columns, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            values = group[value_column].dropna().to_numpy()
            test = exact_sign_flip_test(values)
            rows.append(
                {
                    "sample": sample,
                    **dict(zip(group_columns, keys)),
                    "metric": value_column,
                    "n_subjects": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p_greater": test["p_greater"],
                    "p_two_sided": test["p_two_sided"],
                }
            )
    return pd.DataFrame(rows)


def average_sets(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                column
                for column in frame.columns
                if column
                in {
                    "subject",
                    "phase",
                    "comparison",
                    "contrast",
                    "decoder",
                }
            ],
            as_index=False,
        )[value_columns]
        .mean()
        .assign(stimulus_set="A+B average")
    )


def run_aligned_decoder(
    *,
    subject: int,
    events: pd.DataFrame,
    patterns: dict[str, np.ndarray],
    n_permutations: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    visual_patterns = patterns[PRIMARY_VISUAL]
    parietal_patterns = patterns[PRIMARY_PARIETAL]
    alignment_training = task_frame(events, "C", "vision")
    training_positions = alignment_training.index.to_numpy()
    groups = alignment_training["target_number"].to_numpy()
    alignment, selection = fit_region_alignment(
        parietal_patterns[training_positions],
        visual_patterns[training_positions],
        groups,
    )
    selection_rows = [
        {
            "subject": subject,
            "phase": "pilot" if subject == 1 else "validation",
            **row,
            "selected": (
                int(row["n_components"]) == alignment.n_components
                and float(row["alpha"]) == alignment.alpha
            ),
        }
        for row in selection
    ]

    parietal_c = alignment.transform_parietal(
        parietal_patterns[training_positions]
    )
    visual_c = alignment.transform_visual(
        visual_patterns[training_positions]
    )
    rng = np.random.default_rng(seed + subject * 1000)
    rows = []
    for set_index, stimulus_set in enumerate(STIMULUS_SETS):
        vision = task_frame(events, stimulus_set, "vision")
        imagery = task_frame(events, stimulus_set, "imagery")
        vision_positions = vision.index.to_numpy()
        imagery_positions = imagery.index.to_numpy()
        vision_targets = vision["target_number"].to_numpy()
        imagery_targets = imagery["target_number"].to_numpy()

        visual_vision = alignment.transform_visual(
            visual_patterns[vision_positions]
        )
        visual_imagery = alignment.transform_visual(
            visual_patterns[imagery_positions]
        )
        parietal_vision = alignment.transform_parietal(
            parietal_patterns[vision_positions]
        )
        parietal_imagery = alignment.transform_parietal(
            parietal_patterns[imagery_positions]
        )
        aligned_imagery = alignment.ridge.predict(parietal_imagery)

        predictions = {
            "same_visual": nearest_centroid_predict(
                visual_vision, vision_targets, visual_imagery
            ),
            "aligned_parietal_to_visual": nearest_centroid_predict(
                visual_vision, vision_targets, aligned_imagery
            ),
            "within_parietal": nearest_centroid_predict(
                parietal_vision, vision_targets, parietal_imagery
            ),
        }

        shuffled_accuracies = []
        for _ in range(n_permutations):
            shuffled_ridge = Ridge(alpha=alignment.alpha)
            shuffled_ridge.fit(
                parietal_c,
                visual_c[rng.permutation(len(visual_c))],
            )
            shuffled_prediction = nearest_centroid_predict(
                visual_vision,
                vision_targets,
                shuffled_ridge.predict(parietal_imagery),
            )
            shuffled_accuracies.append(
                np.mean(shuffled_prediction == imagery_targets)
            )
        shuffled_accuracies = np.asarray(shuffled_accuracies)

        for decoder, predicted in predictions.items():
            permutation = exact_class_label_permutation_test(
                predicted, imagery_targets
            )
            row = {
                "subject": subject,
                "phase": "pilot" if subject == 1 else "validation",
                "stimulus_set": stimulus_set,
                "decoder": decoder,
                "accuracy": permutation["accuracy"],
                "chance": permutation["chance"],
                "subject_p_greater": permutation["p_greater"],
                "alignment_components": alignment.n_components,
                "alignment_alpha": alignment.alpha,
                "alignment_set_c_cv_r2": alignment.cv_r2,
                "alignment_training_set": "Set C vision only",
            }
            if decoder == "aligned_parietal_to_visual":
                row.update(
                    {
                        "shuffled_alignment_mean_accuracy": float(
                            shuffled_accuracies.mean()
                        ),
                        "shuffled_alignment_q95_accuracy": float(
                            np.quantile(shuffled_accuracies, 0.95)
                        ),
                        "shuffled_alignment_p_greater": float(
                            (
                                1
                                + np.sum(
                                    shuffled_accuracies
                                    >= permutation["accuracy"] - 1e-12
                                )
                            )
                            / (len(shuffled_accuracies) + 1)
                        ),
                    }
                )
            rows.append(row)
    return rows, selection_rows


def decoder_contrasts(decoder: pd.DataFrame) -> pd.DataFrame:
    wide = decoder.pivot(
        index=["subject", "phase", "stimulus_set"],
        columns="decoder",
        values="accuracy",
    )
    rows = []
    definitions = {
        "aligned_minus_same_visual": (
            "aligned_parietal_to_visual",
            "same_visual",
        ),
        "aligned_minus_within_parietal": (
            "aligned_parietal_to_visual",
            "within_parietal",
        ),
        "within_parietal_minus_same_visual": (
            "within_parietal",
            "same_visual",
        ),
    }
    for (subject, phase, stimulus_set), values in wide.iterrows():
        for contrast, (left, right) in definitions.items():
            rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "stimulus_set": stimulus_set,
                    "contrast": contrast,
                    "accuracy_difference": values[left] - values[right],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parser().parse_args()
    if args.max_voxels < 2:
        raise ValueError("--max-voxels must be at least two")
    if args.n_rdm_splits < 1 or args.n_alignment_permutations < 1:
        raise ValueError("Split and permutation counts must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir
        else args.data_root.expanduser().resolve()
        / "derived"
        / "notebook08_cross_region"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    transfer_rows = []
    reliability_rows = []
    count_rows = []
    decoder_rows = []
    selection_rows = []
    saved_rdms = {}

    for subject in args.subjects:
        print(f"\nsubj{subject:02d}")
        events = build_event_table(args.data_root, subject).reset_index(
            drop=True
        )
        patterns, subject_counts = load_subject_roi_patterns(
            data_root=args.data_root,
            cache_dir=cache_dir,
            subject=subject,
            events=events,
            max_voxels=args.max_voxels,
            seed=args.seed,
        )
        count_rows.extend(subject_counts)
        standard, crossvalidated, subject_reliability = build_subject_rdms(
            subject=subject,
            events=events,
            patterns=patterns,
            n_splits=args.n_rdm_splits,
            seed=args.seed,
        )
        reliability_rows.extend(subject_reliability)
        transfer_rows.extend(
            cross_region_rows(
                subject=subject,
                standard_rdms=standard,
                crossvalidated_rdms=crossvalidated,
            )
        )
        for key, rdm in standard.items():
            roi, stimulus_set, task = key
            saved_rdms[
                f"subj{subject:02d}__{roi}__set{stimulus_set}__{task}__standard"
            ] = rdm
        for key, rdm in crossvalidated.items():
            roi, stimulus_set, task = key
            saved_rdms[
                f"subj{subject:02d}__{roi}__set{stimulus_set}__{task}__crossvalidated"
            ] = rdm

        if not args.skip_aligned_decoder:
            subject_decoder, subject_selection = run_aligned_decoder(
                subject=subject,
                events=events,
                patterns=patterns,
                n_permutations=args.n_alignment_permutations,
                seed=args.seed,
            )
            decoder_rows.extend(subject_decoder)
            selection_rows.extend(subject_selection)

    transfer = pd.DataFrame(transfer_rows)
    planned = planned_comparison_rows(transfer)
    contrasts = contrast_rows(planned)
    reliability = pd.DataFrame(reliability_rows)
    counts = pd.DataFrame(count_rows)

    planned_with_average = pd.concat(
        [
            planned,
            average_sets(
                planned,
                ["standard_rho", "crossvalidated_rho"],
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    contrast_with_average = pd.concat(
        [
            contrasts,
            average_sets(
                contrasts,
                ["standard_difference", "crossvalidated_difference"],
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    rdm_group = pd.concat(
        [
            summarize_signed_values(
                planned_with_average,
                value_column=metric,
                group_columns=["stimulus_set", "comparison"],
                subjects=args.subjects,
            )
            for metric in ("standard_rho", "crossvalidated_rho")
        ],
        ignore_index=True,
    )
    contrast_group = pd.concat(
        [
            summarize_signed_values(
                contrast_with_average,
                value_column=metric,
                group_columns=["stimulus_set", "contrast"],
                subjects=args.subjects,
            )
            for metric in (
                "standard_difference",
                "crossvalidated_difference",
            )
        ],
        ignore_index=True,
    )

    transfer.to_csv(output_dir / "rdm_transfer_subject_level.csv", index=False)
    planned_with_average.to_csv(
        output_dir / "planned_rdm_comparisons.csv", index=False
    )
    contrast_with_average.to_csv(
        output_dir / "rdm_contrasts_subject_level.csv", index=False
    )
    rdm_group.to_csv(output_dir / "rdm_group_summary.csv", index=False)
    contrast_group.to_csv(
        output_dir / "rdm_contrasts_group.csv", index=False
    )
    reliability.to_csv(output_dir / "roi_reliability.csv", index=False)
    counts.to_csv(output_dir / "stream_roi_counts.csv", index=False)
    np.savez_compressed(output_dir / "subject_rdms.npz", **saved_rdms)

    if decoder_rows:
        decoder = pd.DataFrame(decoder_rows)
        decoder_contrast = decoder_contrasts(decoder)
        decoder_with_average = pd.concat(
            [decoder, average_sets(decoder, ["accuracy", "chance"])],
            ignore_index=True,
            sort=False,
        )
        decoder_contrast_with_average = pd.concat(
            [
                decoder_contrast,
                average_sets(
                    decoder_contrast, ["accuracy_difference"]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        decoder_above_chance = decoder_with_average.assign(
            accuracy_above_chance=lambda frame: (
                frame["accuracy"] - frame["chance"]
            )
        )
        decoder_group = summarize_signed_values(
            decoder_above_chance,
            value_column="accuracy_above_chance",
            group_columns=["stimulus_set", "decoder"],
            subjects=args.subjects,
        )
        decoder_contrast_group = summarize_signed_values(
            decoder_contrast_with_average,
            value_column="accuracy_difference",
            group_columns=["stimulus_set", "contrast"],
            subjects=args.subjects,
        )
        decoder.to_csv(
            output_dir / "aligned_decoder_subject_level.csv", index=False
        )
        decoder_contrast_with_average.to_csv(
            output_dir / "aligned_decoder_contrasts.csv", index=False
        )
        decoder_group.to_csv(
            output_dir / "aligned_decoder_group_summary.csv", index=False
        )
        decoder_contrast_group.to_csv(
            output_dir / "aligned_decoder_contrast_group.csv", index=False
        )
        pd.DataFrame(selection_rows).to_csv(
            output_dir / "alignment_selection.csv", index=False
        )

    metadata = {
        "subjects": list(args.subjects),
        "pilot_subject": 1,
        "stimulus_sets": list(STIMULUS_SETS),
        "primary_visual_roi": PRIMARY_VISUAL,
        "primary_visual_labels": list(
            STREAM_ROI_LABELS[PRIMARY_VISUAL]
        ),
        "primary_parietal_roi": PRIMARY_PARIETAL,
        "primary_parietal_labels": list(
            STREAM_ROI_LABELS[PRIMARY_PARIETAL]
        ),
        "max_voxels": args.max_voxels,
        "n_rdm_splits": args.n_rdm_splits,
        "aligned_decoder": not args.skip_aligned_decoder,
        "alignment_training": "Set C vision only",
        "n_alignment_permutations": args.n_alignment_permutations,
        "interpretation": "exploratory; post-hoc collaborator-inspired analysis",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\nSaved cross-region analysis to: {output_dir}")


if __name__ == "__main__":
    main()
