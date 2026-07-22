# Paper reproduction and machine-learning next steps

## Short answer

- **Table 1 brain correlation is not beta-only.** It compares a measured,
  repeat-averaged beta pattern with the beta pattern that GNet predicts from a
  reconstructed image.
- A useful **beta-only companion analysis is feasible now**. Notebook 05 uses
  the paper's exact nested ROIs and directly asks whether matched targets have
  more similar perception/imagery patterns than mismatched targets.
- **Exact scoring is now implemented** once reconstruction images and the
  public GNet checkpoint are present. Image generation remains the expensive
  part.
- For the next scientific analysis, prefer a small, regularized cross-decoder
  over a new deep network. The dataset has many repeated trials but only six
  target identities per set, so model capacity is sharply limited by the
  number of independent targets.

## 1. What Table 1 computes

For subject $s$, condition $c$, target $t$, reconstruction sample $k$, and ROI
$R$, the metric is

$$
r_{sctkR}
=\operatorname{corr}_{v\in R}
\left(\bar\beta_{sct}(v),
E_s(\widehat I_{sctk})(v)\right),
$$

where $\bar\beta$ is the run-normalized measured response averaged over target
repetitions, $\widehat I$ is a reconstructed image, and $E_s$ is the pretrained
GNet image-to-brain encoding model. The Pearson correlation is a **spatial
correlation across voxels for each target/reconstruction**, followed by
averaging over targets, ten samples, and subjects.

The paper ROIs are nested:

| Paper label | Volumetric definition |
|---|---|
| Visual cortex | `nsdgeneral` |
| Early visual | `(V1d/V1v/V2d/V2v/V3d/V3v/hV4) ∩ nsdgeneral` (`prf-visualrois` labels 1–7) |
| Higher visual | `nsdgeneral − early visual` |

This matters because Notebook 04 used a different exploratory contrast:
V1–V3 for early and labels 5–7 from `streams` for higher. Notebook 05 leaves
the old notebook untouched and constructs the paper definitions explicitly.

The paper's interpretation also needs care. It reports a **larger drop from
vision to imagery in early cortex**. It does not require the imagery early
score itself to be lower than the imagery higher score.

## 2. Code added here

### Beta-only companion

Run [Notebook 05](../notebooks/05_paper_brain_correlation.ipynb). It:

1. extracts all `nsdgeneral` voxels in checkpoint-compatible flattening order;
2. performs the authors' within-run voxelwise Z-scoring;
3. constructs early, higher, and visual masks exactly as above;
4. averages repeats by task, stimulus set, and target;
5. reports matched perception/imagery pattern correlation minus mismatched
   target correlation.

That contrast is a direct measure of target-specific shared beta signal, but
it is not a reconstruction score.

### Exact reconstruction brain score

The command-line scorer accepts MindEye-style reconstruction tensors shaped
`targets × samples × channels × height × width` (channels-last also works):

```bash
python scripts/score_table1_brain_correlation.py \
  --data-root /path/to/nsd \
  --subject 1 \
  --task imagery \
  --reconstructions /path/to/all_recons_imagery.pt \
  --gnet-checkpoint /path/to/gnet_multisubject.pt \
  --brain-region-masks /path/to/brain_region_masks.hdf5 \
  --method MindEye2 \
  --predicted-cache outputs/05_paper_brain_correlation/me2_s01_img_gnet.npy \
  --output-prefix outputs/05_paper_brain_correlation/me2_s01_img
```

The official HDF5 masks are recommended for the closest reproduction; if that
argument is omitted, the scorer reconstructs the same ROI definitions from
the local NSD masks. It writes target/sample-level correlations and a subject summary. Repeat for
vision and imagery for subjects 1, 2, 5, and 7, then use Notebook 05 to average
the subject summaries.

The implementation uses the minimal GNet architecture only; it does not import
MindEye's diffusion stack. This keeps scoring much lighter than reconstruction.
Only load the official GNet checkpoint or another trusted PyTorch file.

## 3. What is actually released for reconstruction

### MindEye2: closest path to the paper, but not turnkey

The official MindEye2 repository has a public `mental_imagery` branch with
NSD-Imagery inference scripts/notebooks. Public Hugging Face files include:

- subject-specific full-NSD MindEye2 checkpoints for subjects 1, 2, 5, and 7;
- `gnet_multisubject.pt` and `brain_region_masks.hdf5` for brain scoring;
- the image-variation autoencoder, BigG-to-diffusion-prior adapter, and SDXL
  unCLIP checkpoint used by the reconstruction stack.

The branch is research code rather than a packaged release. It still contains
internal absolute paths, expects a locally generated `nsd_imagery.pt`, and has
stale variants of the loader. The large pieces are also substantial: roughly
9 GB for one subject decoder, 18 GB for the unCLIP checkpoint, 1.27 GB for
GNet, plus several other model files. The Hugging Face dataset as a whole is
well over 100 GB. Start with one subject and one task, not the full benchmark.

This repository can create the checkpoint-compatible beta tensor (including
the exact full-`nsdgeneral` voxel order and per-run normalization):

```bash
python scripts/export_mindeye2_nsdimagery_betas.py \
  --data-root /path/to/nsd \
  --subjects 1 \
  --output-root /path/to/MindEyeV2/data
```

This removes one preprocessing ambiguity but does not make the upstream
inference script turnkey: its stimulus metadata, model paths, and large
generative dependencies still need to be configured.

Useful official locations:

- [MindEye2 repository](https://github.com/MedARC-AI/MindEyeV2)
- [MindEye2 mental-imagery branch](https://github.com/MedARC-AI/MindEyeV2/tree/mental_imagery)
- [MindEye2 public model files](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main)
- [Subject 1 full-NSD checkpoint](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs/final_subj01_pretrained_40sess_24bs)
- [GNet checkpoint](https://huggingface.co/datasets/pscotti/mindeyev2/blob/main/gnet_multisubject.pt)

Before attempting inference on the JupyterHub, run `nvidia-smi` and record GPU
model, VRAM, free disk, and quota. A 16 GB GPU is unlikely to be a comfortable
path for the released SDXL unCLIP stack; high-memory GPUs are the sensible
target. Training requirements in the MindEye2 README are not inference
requirements, but they are a warning about the model scale.

### Other Table 1 methods

| Method | Public state | Quick NSD-Imagery inference? |
|---|---|---|
| MindEye1 | Official code and full-NSD subject checkpoints are public | Possible, but checkpoints and diffusion components are still large; no clean NSD-Imagery runner was found |
| Brain Diffuser | Official code and generic VDVAE/Versatile Diffusion weights are linked | No: the public workflow trains subject-specific fMRI-to-latent regressions rather than providing them |
| Takagi et al. | Official Stable Diffusion reconstruction code is public | No: it expects fitted subject regressions and its own preprocessing |
| iCNN | Method code exists in the literature ecosystem | No clearly packaged, NSD-Imagery-ready subject checkpoint was identified |

Official references:

- [MindEye1 code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
- [MindEye1 checkpoints](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models)
- [Brain Diffuser code](https://github.com/ozcelikfu/brain-diffuser)
- [Takagi Stable Diffusion reconstruction](https://github.com/yu-takagi/StableDiffusionReconstruction)

There is also a public [MIRAGE repository](https://github.com/MedARC-AI/MIRAGE)
with explicit NSD-Imagery inference and released ridge checkpoints. It is a
different, newer reconstruction pipeline rather than a Table 1 reproduction,
and its Stable Cascade workflow is still large. It is a reasonable second
option if the aim changes from reproducing Table 1 to obtaining any modern
imagery reconstruction quickly.

## 4. Recommended ML analysis beyond RSA/RDM

The cleanest next question is not “are two RDMs correlated?” but:

> How much target information is measurable within imagery, and how much of
> that information is readable by a decoder learned only from vision?

Use the following four-way transfer matrix separately for early, higher, and
full visual cortex:

| Train | Test | Meaning |
|---|---|---|
| Vision | held-out vision | perception information ceiling for this decoder |
| Imagery run 1 | imagery run 2 (and reverse) | imagery information ceiling |
| Vision | imagery | perception-to-imagery shared/readable information |
| Imagery | vision | reverse transfer; diagnoses asymmetry |

### Model

Use an L2-regularized linear probabilistic classifier. Tune regularization only
inside the training data. Do not use a deep network: repeated trials do not
create more than six independent target classes per set, and a high-capacity
model would mostly learn run/cue idiosyncrasies.

For each held-out trial, retain both accuracy and log loss. With $K=6$ target
classes, a calibrated cross-validated information-style score is

$$
I_{\mathrm{probe}}=\log K-\operatorname{CE}(y,p_\theta(y\mid\beta)).
$$

It is zero at a uniform chance predictor and positive when the probe assigns
more probability to the correct target. It is a decoder-dependent lower-bound
style quantity, not the brain's total mutual information.

Report three quantities rather than one ratio:

1. imagery availability: $I_{I\rightarrow I}$;
2. shared readable content: $I_{V\rightarrow I}$;
3. transfer gap: $I_{I\rightarrow I}-I_{V\rightarrow I}$.

A ratio becomes unstable when within-imagery information is close to zero.
Only show transfer efficiency $I_{V\rightarrow I}/I_{I\rightarrow I}$ when the
denominator has a reliable positive lower confidence bound.

### Separate shared content from generic task differences

Add a second linear probe for `vision` versus `imagery`, but first remove target
means estimated from training folds. High residual task accuracy measures a
condition-specific shift that is not explained by target identity. Together,
the target-transfer and residual-task probes give a useful decomposition:

- high target transfer, low residual task accuracy: largely shared code;
- high target transfer, high residual task accuracy: shared content plus a
  systematic task transform;
- low target transfer, high within-imagery decoding: imagery-specific code;
- low transfer and low within-imagery decoding: insufficient measurable signal.

### Essential controls

- hold out independent imagery runs and split vision repetitions without
  mixing a trial across train/test;
- keep set A and set B separate before pooling subject-level effects;
- repeat with vision trials whose cue mismatches the visible target;
- match early/higher voxel counts through repeated subsampling for a fair ROI
  comparison, while also reporting the full paper ROIs;
- permute target labels as intact target/run blocks, not individual correlated
  repeats;
- use subjects as the inferential unit (paired sign flips or bootstrap across
  subjects), not the hundreds of repeated trials;
- report within-condition reliability alongside every transfer result. A weak
  cross-decoder is uninterpretable when the imagery ceiling is also weak.

This analysis directly quantifies available imagery information, transferred
information, and condition-specific information while remaining feasible with
the released betas and current sample size.

## Sources

- [NSD-Imagery CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kneeland_NSD-Imagery_A_Benchmark_Dataset_for_Extending_fMRI_Vision_Decoding_Methods_CVPR_2025_paper.html)
- [NSD-Imagery supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Kneeland_NSD-Imagery_A_Benchmark_CVPR_2025_supplemental.pdf)
