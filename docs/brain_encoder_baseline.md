# Core-NSD brain encoder baseline

This track trains an image-to-beta encoder on **core NSD perception data** and
tests it, without refitting, on NSD-Imagery vision and imagery. It answers a
different question from the reconstruction model:

> Which image features learned from perception remain aligned with measured
> activity during imagery, and is the vision-to-imagery loss larger in early
> than in higher visual cortex?

The first implemented model is deliberately strong but auditable:

1. a frozen pretrained vision transformer extracts nonlinear features;
2. CLS and coarse spatial-pyramid features are taken from several layers;
3. PCA is learned on core-NSD training images only;
4. a multi-output ridge readout predicts every `nsdgeneral` voxel;
5. the ridge penalty is chosen on unique validation images;
6. final perception accuracy is reported on held-out core-NSD images;
7. the frozen encoder is scored on NSD-Imagery using the paper's spatial brain
   correlation.

Repeated fMRI presentations are averaged before splitting. No image identity
can occur in more than one partition.

## 1. Start every shell with explicit paths

The examples below assume that the repository and its existing `data/nsd`
directory are both under `$HOME/NHprojectNSDimagery`. Run this bootstrap block
after opening a new terminal or receiving a new JupyterHub pod:

```bash
cd "$HOME/NHprojectNSDimagery"

export REPO_ROOT="$PWD"
export NSD_DATA_ROOT="$REPO_ROOT/data/nsd"
export NSD_STIMULI="$NSD_DATA_ROOT/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
export WORK="$REPO_ROOT/outputs/06_core_nsd_encoder/subj01/dinov2_small"

mkdir -p "$WORK"
printf 'Repository:  %s\nData root:   %s\nStimuli:     %s\nWork output: %s\n' \
  "$REPO_ROOT" "$NSD_DATA_ROOT" "$NSD_STIMULI" "$WORK"
```

Shell variables do not survive a new terminal, server restart, or replacement
pod. An unset variable quoted as `"$NSD_DATA_ROOT"` becomes an empty argument;
older versions of the scripts interpreted that as the repository directory.
The scripts now reject empty paths, but rerunning the bootstrap remains the
safest practice.

If the 36.84 GiB stimulus HDF5 is stored on a different persistent or shared
mount, change only `NSD_STIMULI`:

```bash
export NSD_STIMULI="/actual/persistent/path/nsd_stimuli.hdf5"
```

Do not change `NSD_DATA_ROOT` in that case: the design, ROI masks, beta files,
and NSD-Imagery data can remain in the repository's `data/nsd` directory.

## 2. Environment, storage estimate, and download

Update the existing environment because this patch adds scikit-learn:

```bash
conda env update -n nsdimagery -f environment.yml
conda activate nsdimagery
```

The shared `nsd_stimuli.hdf5` image bank is 36.84 GiB. Subject 1's 40 beta
sessions are about 18.25 GiB, for a large-file subtotal of about 55.09 GiB.
Estimate before downloading:

```bash
bash scripts/download_core_nsd_encoder_mvp.sh \
  --subject 01 \
  --sessions all \
  --dest "$NSD_DATA_ROOT" \
  --estimate
```

Then preview and download:

```bash
bash scripts/download_core_nsd_encoder_mvp.sh \
  --subject 01 --sessions all --dest "$NSD_DATA_ROOT" --dry-run

bash scripts/download_core_nsd_encoder_mvp.sh \
  --subject 01 --sessions all --dest "$NSD_DATA_ROOT"
```

The downloader writes the stimulus bank to the default `NSD_STIMULI` path
defined in the bootstrap. If an earlier run used `--skip-stimuli`, recover the
missing image bank without downloading the beta sessions again:

```bash
bash scripts/download_core_nsd_encoder_mvp.sh \
  --dest "$NSD_DATA_ROOT" \
  --stimuli-only
```

If storage policy requires the HDF5 elsewhere, download it directly there and
update `NSD_STIMULI`:

```bash
mkdir -p "$(dirname "$NSD_STIMULI")"
aws s3 cp \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  "$NSD_STIMULI" \
  --no-sign-request
```

For a code-path smoke test, `--sessions 5` downloads sessions 1--5. Use
`--split-mode random` during preparation because a partial acquisition is not
guaranteed to contain enough of the shared-1000 test set. Scientific results
should use every completed session.

### Verify the download before computing

Run this block after any download. Every line should print `FOUND` and the
stimulus file should be approximately 37 GiB:

```bash
for path in \
  "$NSD_DATA_ROOT/nsddata/experiments/nsd/nsd_expdesign.mat" \
  "$NSD_DATA_ROOT/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz" \
  "$NSD_DATA_ROOT/nsddata/ppdata/subj01/func1pt8mm/roi/prf-visualrois.nii.gz" \
  "$NSD_DATA_ROOT/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz" \
  "$NSD_DATA_ROOT/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session40.nii.gz" \
  "$NSD_STIMULI"
do
  if test -f "$path"; then
    printf 'FOUND  %s\n' "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done
du -h "$NSD_STIMULI" 2>/dev/null || true
```

## 3. Prepare unique-image beta targets

```bash
python scripts/prepare_core_nsd_encoder_data.py \
  --data-root "$NSD_DATA_ROOT" \
  --subject 1 \
  --output-dir "$WORK" \
  --split-mode shared1000
```

Before repeat averaging, every voxel is Z-scored within each 750-trial core
session. This removes session offsets and puts the targets on the same general
scale as the run-normalized NSD-Imagery patterns. The command writes:

- `core_subj01_betas.npy`: repeat-averaged percent-signal-change betas;
- `core_subj01_coordinates.npy`: exact `nsdgeneral` voxel order;
- `core_subj01_manifest.csv`: one row per unique image and its fixed split.
- `core_subj01_voxel_regions.csv`: early, higher, overall, and V1--V4 membership
  in that exact voxel order.

The first 1,000 positions of the subject-specific NSD image list are the images
shared by all subjects. They are reserved for the final perception test. A
deterministic subset of the remaining identities is used for validation.

### Target-image overlap

Before treating NSD-Imagery as a strict image-out-of-distribution test, check
whether an A/B target image is among the subject's core-NSD images. Exact
overlap may be retained when reproducing the benchmark setup, but it should be
excluded for the stronger generalization analysis. Put one-based NSD 73K IDs
in a text file and rerun preparation with:

```bash
--exclude-73k-ids path/to/excluded_73k_ids.txt
```

Report which policy was used. Never decide the policy after looking at imagery
scores.

Verify preparation outputs before starting the GPU job:

```bash
for path in \
  "$WORK/core_subj01_manifest.csv" \
  "$WORK/core_subj01_betas.npy" \
  "$WORK/core_subj01_coordinates.npy" \
  "$WORK/core_subj01_voxel_regions.csv"
do
  test -f "$path" && printf 'FOUND  %s\n' "$path" || printf 'MISSING %s\n' "$path"
done
```

## 4. Extract frozen image features

The default is `facebook/dinov2-small`, with hidden states 3, 6, 9, and 12.
Each state contributes CLS, 1x1 pooled, and 2x2 pooled patch features. The GPU
is used automatically when available.

```bash
python scripts/extract_image_features.py \
  --manifest "$WORK/core_subj01_manifest.csv" \
  --nsd-stimuli "$NSD_STIMULI" \
  --model-id facebook/dinov2-small \
  --layers 3,6,9,12 \
  --pyramid-levels 1,2 \
  --batch-size 64 \
  --device auto \
  --output "$WORK/core_features.npz"
```

The pretrained checkpoint is downloaded to the Hugging Face cache on first
use. An unauthenticated Hugging Face warning is harmless for this public
checkpoint. If GPU memory is tight, lower `--batch-size`; this does not change
the features. The script now checks the manifest and HDF5 paths before loading
the model, so a missing 37 GiB input fails immediately.

For the semantic comparison, repeat the full track in a separate work
directory with:

```bash
--model-id openai/clip-vit-base-patch32
```

Do not use NSD-Imagery results to choose layers. Layer and pyramid choices must
be fixed in advance or selected only with core-NSD validation data.

## 5. Fit and validate the encoder

```bash
python scripts/fit_ridge_encoder.py \
  --manifest "$WORK/core_subj01_manifest.csv" \
  --betas "$WORK/core_subj01_betas.npy" \
  --coordinates "$WORK/core_subj01_coordinates.npy" \
  --voxel-regions "$WORK/core_subj01_voxel_regions.csv" \
  --features "$WORK/core_features.npz" \
  --pca-components 512 \
  --alphas 0.1,1,10,100,1000 \
  --device auto \
  --method DINOv2_PyramidRidge \
  --output-model "$WORK/encoder_model.npz" \
  --output-summary "$WORK/core_test_summary.json" \
  --output-voxel-metrics "$WORK/core_test_voxels.csv"
```

The script chooses alpha by mean voxel-wise validation correlation, refits on
train plus validation images, and evaluates once on the held-out test images.
It saves both Pearson correlation and true held-out R-squared for every voxel.
If core-NSD held-out predictivity is absent, do not interpret imagery scores.

Feature extraction benefits most from the GPU. PCA is CPU-based. Ridge uses a
chunked PyTorch Cholesky solve and automatically uses the T4; a NumPy CPU
fallback is also implemented.

## 6. Build target features and evaluate NSD-Imagery

Make an image manifest from the released A/B targets:

```bash
python scripts/make_nsdimagery_image_manifest.py \
  --data-root "$NSD_DATA_ROOT" \
  --output "$WORK/nsdimagery_AB_manifest.csv"
```

Extract features with exactly the same model, layers, and spatial pyramid:

```bash
python scripts/extract_image_features.py \
  --manifest "$WORK/nsdimagery_AB_manifest.csv" \
  --model-id facebook/dinov2-small \
  --layers 3,6,9,12 \
  --pyramid-levels 1,2 \
  --device auto \
  --output "$WORK/nsdimagery_AB_features.npz"
```

Finally, predict betas and score both tasks:

```bash
python scripts/evaluate_encoder_nsdimagery.py \
  --data-root "$NSD_DATA_ROOT" \
  --subject 1 \
  --encoder-model "$WORK/encoder_model.npz" \
  --image-manifest "$WORK/nsdimagery_AB_manifest.csv" \
  --image-features "$WORK/nsdimagery_AB_features.npz" \
  --method DINOv2_PyramidRidge \
  --output-prefix "$WORK/dinov2_subj01"
```

The resulting `*_detail.csv` and `*_summary.csv` use the same schema as
Notebook 05. Copy or point Notebook 05's `OUTPUT_DIR` to this directory to use
its subject and simple/real-scene aggregation cells.

The image manifest may also contain several reconstructions per target. Add a
`sample` column and one `image_path` row per reconstruction; the evaluator will
score every sample against the corresponding measured target pattern.

## 7. Evidence hierarchy

Interpret results in this order:

1. **Held-out core NSD:** does the encoder predict unseen viewed images?
2. **NSD-Imagery vision:** does it transfer to the benchmark acquisition?
3. **NSD-Imagery imagery:** is target-predicted structure measurable without
   any imagery fitting?
4. **ROI difference-in-differences:** is the vision-to-imagery loss larger in
   early than higher visual cortex?
5. **Feature-family interaction:** is the loss smaller for late/semantic than
   shallow/spatial features?
6. **Reconstruction scoring:** do conclusions agree for GNet and the new
   independently trained encoder?

Subjects—not images, targets, or reconstruction samples—remain the unit for
group uncertainty. Match early/higher ROI voxel counts as a sensitivity check,
and retain label-permutation and cue-mismatch controls from the earlier
notebooks.

## 8. Troubleshooting paths

Use the traceback to distinguish the two common failures:

- `nsddata/...` with no absolute prefix means a shell path variable was empty
  or relative. Rerun the bootstrap and confirm `printf '<%s>\n' "$NSD_DATA_ROOT"`.
- An absolute `.../nsd_stimuli.hdf5` path that is missing means the beta-only
  download succeeded but the shared image bank did not. Run `--stimuli-only`
  or point `NSD_STIMULI` to its external location.

For a complete path snapshot:

```bash
printf 'NSD_DATA_ROOT=<%s>\nNSD_STIMULI=<%s>\nWORK=<%s>\n' \
  "$NSD_DATA_ROOT" "$NSD_STIMULI" "$WORK"
readlink -f "$NSD_DATA_ROOT"
readlink -f "$NSD_STIMULI"
tree -L 4 "$NSD_DATA_ROOT/nsddata_stimuli" 2>/dev/null || true
```

## 9. Next model tier

The spatial pyramid is a controlled ridge baseline, not the endpoint. After it
passes core-NSD validation, replace fixed pooling with a learned factorized
receptive-field readout while keeping the backbone frozen. Compare it against
this exact split and test set. Only then consider a shared low-rank nonlinear
adapter or limited backbone fine-tuning. Additional capacity is warranted only
if it improves held-out core-NSD prediction without using NSD-Imagery for model
selection.
