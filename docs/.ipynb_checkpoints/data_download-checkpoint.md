# Downloading only the NSD-Imagery data needed for this project

This guide is intentionally narrower than the full NSD Data Manual. It downloads the prepared data needed for the RSA / representational-geometry MVP, not every NSD product.

## Quick start on NeuroHackademy JupyterHub

```bash
git clone https://github.com/mynanshan/NHprojectNSDimagery.git
cd NHprojectNSDimagery

# Check the largest file for one subject without downloading it.
bash scripts/download_nsdimagery_mvp.sh --subjects 01 --estimate

# Preview every transfer.
bash scripts/download_nsdimagery_mvp.sh --subjects 01 --dry-run

# Download one subject first.
bash scripts/download_nsdimagery_mvp.sh \
  --subjects 01 \
  --dest /your/persistent/storage/nsd

# After the loader works for subj01, download all eight subjects.
bash scripts/download_nsdimagery_mvp.sh \
  --subjects all \
  --dest /your/persistent/storage/nsd
```

Replace `/your/persistent/storage/nsd` with a directory on the JupyterHub's persistent volume. Check space first with `df -h` and inspect the completed download with `du -sh /your/persistent/storage/nsd`.

Do **not** remove `--no-sign-request` from AWS commands. Completing the NSD access form gives you the terms/manual link; it does not give the AWS credentials that ordinary signed requests expect. The bucket itself is read anonymously.

## What the script downloads

For each selected subject:

| Component | S3 product | Why we need it |
|---|---|---|
| Prepared fMRI | `func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5` | One denoised/regularized beta pattern per modeled event; 720 beta volumes per subject |
| Broad visual ROI | `nsdgeneral.nii.gz` | Sanity-check / strong visual-signal mask |
| Early visual ROIs | `prf-visualrois.nii.gz` | V1d/V1v, V2d/V2v, V3d/V3v, and hV4 |
| Visual-stream ROIs | `streams.nii.gz` | Broad early, intermediate, and higher-level ventral/lateral/parietal regions |
| ROI label tables | `*prf-visualrois*`, `*streams*` | Translate integer mask values into ROI names |
| Trial metadata | `nsdimagery_subjAA_*.tsv` | Condition, cue, frame file, response/vividness, and timing |

Shared files are downloaded once:

- the 12 run design matrices (`*_dm.mat`);
- the GLMsingle design (`designmatrixGLMsingle.mat`);
- cue/target pair lists;
- raw target images for stimulus sets A and B;
- the VVIQ document.

Pass `--include-allstim` if we decide to analyze the exact trial-varying images in the conceptual (set C) vision condition. This adds all 1,149 rendered experiment frames. It is not necessary for the first set-A/set-B RSA.

## Why these particular choices

- **Prepared betas, not raw fMRI or time series.** Our unit of analysis is a trial-level voxel response vector. Re-running preprocessing and the fMRI GLM would add substantial data and engineering without helping the one-week MVP.
- **`fithrf_GLMdenoise_RR`, not plain `fithrf`.** This is the denoised and ridge-regularized single-trial preparation and is the sensible default for multivariate RSA.
- **`func1pt8mm`.** It keeps the beta file and all three ROI volumes in the same voxel grid. No registration or resampling is needed before masking.
- **HDF5 betas.** This gives one file per subject and is convenient to load lazily with `h5py` rather than expanding a large four-dimensional NIfTI in memory.

## What is deliberately not downloaded

- raw scanner data;
- preprocessed time series and motion files;
- structural, diffusion, resting-state, physiological, and eye-tracking data;
- the attention-run image stream unless it becomes scientifically necessary;
- original NSD core betas;
- reconstruction-model checkpoints.

The original NSD core betas are needed only for the stretch goal that trains a new perception-to-feature ridge decoder. They are much larger than this MVP and should not be downloaded pre-emptively.

## Recommended two-stage workflow

1. Download `subj01` only.
2. Verify that the beta file opens, that it contains 720 beta volumes, and that all masks share its three spatial dimensions.
3. Join vision/imagery trials to their behavioral TSV and design-matrix entries.
4. Produce one `nsdgeneral` neural RDM.
5. Only then download subjects 02–08.

This catches path, storage, and trial-alignment mistakes before multiplying them across eight subjects.

## Expected directory layout

```text
nsd/
├── nsddata/
│   ├── bdata/nsdimagery/
│   ├── experiments/nsdimagery/
│   ├── freesurfer/subj01/label/
│   └── ppdata/subj01/func1pt8mm/roi/
└── nsddata_betas/
    └── ppdata/subj01/func1pt8mm/
        └── nsdimagerybetas_fithrf_GLMdenoise_RR/
            └── betas_nsdimagery.hdf5
```

## One subtle trial-alignment warning

Each subject completed 12 runs with 48 presented stimulus trials per run (576 presented trials total), but the GLM yields **720 beta weights**. Vision and imagery trials each yield one beta; each attention trial yields separate cue-epoch and detection-epoch betas. Therefore, never assume that beta index 1–720 maps directly to the concatenated behavioral rows. Use `designmatrixGLMsingle.mat` together with the run TSVs when constructing the trial table.

For the first MVP, analyze vision and imagery only. The attention condition is scientifically interesting but not required for the stated hypothesis and has more complicated event indexing.

## Troubleshooting

If an AWS command fails, first confirm anonymous access:

```bash
aws s3 ls s3://natural-scenes-dataset/ --no-sign-request
```

Then preview the downloader:

```bash
bash -x scripts/download_nsdimagery_mvp.sh --subjects 01 --dry-run
```

Common causes are a non-persistent destination, insufficient quota, a typo in the subject format (`01`, not `1`), or omitting `--no-sign-request` in a hand-written AWS command.
