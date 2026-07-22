# NSD-Imagery Workshop Project Notes

**Working project title.**  
**What Survives in the Mind’s Eye? Representation-Level Transfer from Perception to Mental Imagery**

---

## 1. Big Picture

The central question is not simply whether imagined images can be reconstructed from fMRI. The original NSD-Imagery paper already benchmarks that. A more focused and feasible workshop question is:

> **Which representational levels transfer from visual perception to mental imagery?**

Our working hypothesis:

> **Mental imagery preserves high-level semantic/category structure better than low-level spatial detail, and this preservation is stronger in higher visual cortex than early visual cortex.**

In statistical terms, this is a three-way interaction:

$$
\text{condition} \times \text{feature level} \times \text{brain region}.
$$

where

$$
\text{condition} \in \{\text{seen/perception}, \text{imagery}\},
$$

$$
\text{feature level} \in \{\text{low-level visual}, \text{mid-level visual}, \text{high-level semantic}\},
$$

$$
\text{brain region} \in \{\text{early visual cortex}, \text{higher visual cortex}, \text{nsdgeneral}\}.
$$

The goal is to produce an interpretable analysis of how brain-response geometry changes from seeing to imagining, using pretrained image/text representations as probes.

### Orientation and current results

Three shorter notes now complement this project plan:

- [Experiment recap](experiment_recap.md): what a participant did, and how runs, trials, design matrices, raw fMRI, and beta volumes relate.
- [Original paper methodology](original_paper_methodology.md): the benchmark's cross-decoding workflow in compact notation.
- [Notebook 02/03 results review](notebook_02_03_results_review.md): what the current RDM, reliability, held-out transfer, HOG, and CLIP results do and do not support.

The essential counting identity for one participant is

$$
12\text{ runs}\times48\text{ behavioral trials}=576\text{ trials},
$$

but the released GLM has

$$
(3\text{ vision}+6\text{ imagery})\times48
+3\text{ attention}\times(2\times48)=720\text{ beta volumes}.
$$

The difference occurs because every attention trial has two modeled epochs. Our current notebooks use the 432 vision and imagery beta volumes and leave the 288 attention betas out.

---

## 2. Why NSD-Imagery Is Interesting

### Original NSD

The **Natural Scenes Dataset (NSD)** is a large 7T fMRI dataset in which 8 subjects viewed thousands of natural images over many scan sessions. It is unusually rich because each subject has many repeated image-response measurements, making it a central dataset for visual encoding/decoding research.

Useful source: <https://naturalscenesdataset.org/>

### NSD-Imagery

**NSD-Imagery** is an extension/benchmark dataset using the same NSD participants. It measures fMRI responses not only when participants see visual stimuli, but also when they imagine cued stimuli.

This matters because it lets us ask:

- Does a decoder trained on real visual perception generalize to internally generated visual experience?
- Which aspects of visual representation are shared between seeing and imagining?
- Does mental imagery preserve semantic content more robustly than low-level spatial structure?

Useful source: <https://arxiv.org/abs/2506.06898>

---

## 3. What the Original NSD-Imagery Paper Does

The paper is primarily a **benchmark paper**, not just a new model paper.

Its core workflow is:

1. Take fMRI-to-image reconstruction models trained on original NSD seen-image data.
2. Apply them to NSD-Imagery vision and imagery trials.
3. Evaluate whether models trained on perception can reconstruct or identify imagined content.

Schematically:

$$
\text{original NSD seen fMRI} \rightarrow \text{train brain-to-feature/image decoder}
$$

then

$$
\text{NSD-Imagery fMRI} \rightarrow \text{same decoder} \rightarrow \text{reconstructed image or decoded feature}.
$$

The paper benchmarks existing methods such as Brain Diffuser, MindEye1, MindEye2, iCNN, and a Takagi-style latent diffusion method. The important conclusion for us is:

> Strong seen-image reconstruction does not automatically imply strong imagery reconstruction.

Imagery has lower signal-to-noise and likely coarser spatial detail than perception, so perception-to-imagery transfer is a genuine domain-transfer problem.

---

## 4. Important Distinction: Full Reconstruction vs. Representation-Level Analysis

### What we should avoid as the main project

Avoid making the central project:

> “Train a new fMRI-to-image diffusion reconstruction model.”

Reasons:

- Too much engineering for one week.
- Requires stable GPU environment and large pretrained checkpoints.
- Existing papers already do image reconstruction from NSD fMRI.
- Generated images are strongly influenced by the diffusion/generative prior.

### What we should do instead

Use pretrained representations and simple statistical/ML tools to ask:

> Given fMRI during perception or imagery, which feature geometry is present in the neural response?

This can be done without training a deep generative model.

Core methods:

1. **Representational Similarity Analysis (RSA)** / RDM comparison.
2. **Linear feature decoding** with ridge regression.
3. **Retrieval or rank-based evaluation** rather than pixel-level image generation.

---

## 5. Data Components and Masks

### Recommended requested NSD data components

For the project, the most useful components are:

- **Task fMRI data**: the core visual/imagery fMRI data.
- **Functional localizers**: useful for defining visual ROIs.
- **Behavioral data**: stimulus IDs, trial information, responses, vividness ratings.
- **Structural data**: useful for anatomical surfaces, ROI definitions, visualization.
- **Eyetracking data**: optional, useful for controlling attention/fixation if needed.

Less central for this project:

- Resting-state fMRI.
- Diffusion data.
- Physiological data.

### Prepared vs. raw data

Use **prepared data**, especially prepared fMRI beta estimates. Raw fMRI preprocessing is too large for a one-week workshop project.

---

## 6. Brain Regions: `nsdgeneral`, Early Visual, Higher Visual

### `nsdgeneral`

`nsdgeneral` is a broad mask of visually responsive posterior-cortex voxels used in many NSD decoding analyses. It is not whole-brain fMRI, but it is much broader than a single ROI.

Interpretation:

$$
\texttt{nsdgeneral} \approx \text{large visually responsive posterior cortex mask}.
$$

Use it as:

- a benchmark mask;
- an upper-bound or sanity-check mask;
- a way to see whether the pipeline detects meaningful signal at all.

### Early visual cortex

Usually includes:

$$
\text{V1}, \text{V2}, \text{V3}, \text{V4}.
$$

These areas are closer to the first cortical stages of vision and are more sensitive to:

- edges;
- contrast;
- orientation;
- spatial frequency;
- local visual layout;
- retinotopic position.

They are often described as more **low-level** or **spatially detailed** visual areas.

### Higher visual cortex

Higher visual cortex refers to visual areas beyond early retinotopic cortex. Depending on available masks, this may include regions involved in:

- object recognition;
- faces;
- bodies;
- places/scenes;
- semantic categories;
- higher-level visual meaning.

Examples of specific higher visual ROIs that may appear in NSD/localizer resources:

- FFA: fusiform face area.
- OFA: occipital face area.
- PPA: parahippocampal place area.
- OPA: occipital place area.
- RSC: retrosplenial cortex / complex.
- EBA: extrastriate body area.
- LOC: lateral occipital complex.

### Why use ROI-specific analysis?

ROI-specific analysis loses information relative to `nsdgeneral`, but that is acceptable because the goal is not maximum reconstruction quality. The goal is anatomical interpretation.

Use both:

$$
\texttt{nsdgeneral}, \quad \text{early visual}, \quad \text{higher visual}.
$$

Interpretation:

- `nsdgeneral`: best overall signal / sanity check.
- early visual: low-level spatial detail.
- higher visual: semantic/category-level information.

---

## 7. Feature Spaces to Use as Representation Probes

We need operational definitions of “low-level” and “high-level” representations.

### Low-level visual features

Possible choices:

- pixels after downsampling;
- Gabor filter features;
- HOG features;
- early CNN layers;
- early ViT layers.

These represent edges, orientations, local contrast, texture, and spatial layout.

### Mid-level visual features

Possible choices:

- intermediate CNN layers;
- intermediate ViT layers;
- DINO or self-supervised visual features.

These may represent shapes, object parts, and scene layout.

### High-level semantic features

Possible choices:

- CLIP image embedding;
- CLIP text/caption embedding;
- manually coded categories;
- COCO annotations if available;
- caption embeddings from a language model.

These represent meaning, object category, scene category, and semantic gist.

Important principle:

> Do not rely on only one feature model. Use at least two low-level and two high-level feature families if possible.

This makes the conclusion less dependent on one arbitrary latent space.

---

## 8. Most Feasible Plan: RSA / Representational Geometry

This is the recommended **must-have MVP**.

### Data object

For subject $i$, stimulus $s$, condition $c$, and ROI $r$, define:

$$
x_{i,s,c,r} \in \mathbb R^{p_r},
$$

where $p_r$ is the number of voxels in ROI $r$.

Conditions:

$$
c \in \{\text{vision}, \text{imagery}\}.
$$

ROIs:

$$
r \in \{\texttt{nsdgeneral}, \text{early visual}, \text{higher visual}\}.
$$

### Step 1: Average repeated trials

If stimulus $s$ has repeated trials, average beta patterns within subject/condition/ROI:

$$
\bar x_{i,s,c,r}
=
\frac{1}{m_{i,s,c}}
\sum_{j=1}^{m_{i,s,c}}
x_{i,s,c,r,j}.
$$

### Step 2: Neural representational dissimilarity matrix

For each subject, ROI, and condition:

$$
D^{\text{brain}}_{i,r,c}(a,b)
=
1-\operatorname{corr}\left(\bar x_{i,a,c,r},\bar x_{i,b,c,r}\right).
$$

This gives a neural geometry across target stimuli.

### Step 3: Feature RDMs

For each image or concept $I_s$, compute feature vectors:

$$
z_\ell(I_s),
$$

where $\ell$ is feature level, e.g.

$$
\ell \in \{\text{HOG}, \text{early CNN}, \text{late CNN}, \text{CLIP image}, \text{text/caption}\}.
$$

Then compute feature dissimilarity:

$$
D^{\ell}(a,b)
=
1-\operatorname{corr}\left(z_\ell(I_a),z_\ell(I_b)\right).
$$

Alternative distances:

- cosine distance;
- correlation distance;
- Spearman rank distance;
- Euclidean distance after standardization/PCA.

### Step 4: Compare neural RDM to feature RDM

Use Spearman correlation between the upper triangles:

$$
\rho_{i,r,c,\ell}
=
\operatorname{SpearmanCorr}\left(
\operatorname{vec}_{\triangle}(D^{\text{brain}}_{i,r,c}),
\operatorname{vec}_{\triangle}(D^\ell)
\right).
$$

This avoids interpreting raw latent-vector distances too literally.

### Step 5: Regression-style RSA

A stronger version regresses neural geometry on multiple feature geometries:

$$
\operatorname{vec}_{\triangle}(D^{\text{brain}}_{i,r,c})
=
\alpha
+
\sum_\ell \beta_{i,r,c,\ell}
\operatorname{vec}_{\triangle}(D^\ell)
+\epsilon.
$$

Because different feature RDMs may be correlated, use:

- standardized predictors;
- ridge regression if necessary;
- partial correlations;
- permutation tests.

### Main output

A heat map of:

$$
\rho_{r,c,\ell}
\quad \text{or} \quad
\beta_{r,c,\ell},
$$

with rows = ROIs, columns = feature levels, panels = vision vs. imagery.

### Hypothesis in RSA terms

The hypothesis predicts:

$$
\rho_{\text{higher visual},\text{imagery},\text{semantic}}
$$

should remain relatively high compared with:

$$
\rho_{\text{early visual},\text{imagery},\text{low-level}}.
$$

More robustly, compare transfer/drop from vision to imagery:

$$
\Delta_{r,\ell}
=
\rho_{r,\text{vision},\ell}
-
\rho_{r,\text{imagery},\ell}.
$$

Expected pattern:

$$
\Delta_{\text{early visual},\text{low-level}}
>
\Delta_{\text{higher visual},\text{semantic}}.
$$

---

## 9. Stretch Plan: Linear Feature Decoding / Retrieval

This is more ML-like but still feasible if data loading works.

### Training data

Use original NSD perception trials:

$$
\{(x^{\text{seen}}_{i,s,r}, z_\ell(I_s))\}_{s=1}^{n_i}.
$$

### Model

Train a ridge decoder from fMRI to feature space:

$$
\widehat W_{i,r,\ell}
=
\arg\min_W
\sum_s
\left\|x^{\text{seen}}_{i,s,r}W-z_\ell(I_s)\right\|_2^2
+
\lambda\|W\|_F^2.
$$

Prediction:

$$
\widehat z_{i,s,c,r,\ell}
=x_{i,s,c,r}\widehat W_{i,r,\ell}.
$$

### Test conditions

Evaluate on:

1. NSD-Imagery vision trials.
2. NSD-Imagery imagery trials.

### Retrieval evaluation

For each test trial $s$, compare predicted feature $\widehat z_s$ to candidate target features $z(I_1),\ldots,z(I_m)$.

Similarity:

$$
\operatorname{sim}(\widehat z_s,z(I_k))
=
\frac{\widehat z_s^\top z(I_k)}{
\|\widehat z_s\|_2\|z(I_k)\|_2
}.
$$

Metrics:

- top-1 accuracy;
- top-k accuracy;
- mean rank percentile;
- two-alternative forced-choice accuracy.

### Transfer ratio

For feature level $\ell$ and ROI $r$:

$$
T_{r,\ell}
=
\frac{
\text{performance}_{\text{imagery},r,\ell}
}{
\text{performance}_{\text{vision},r,\ell}
}.
$$

The hypothesis predicts:

$$
T_{\text{higher visual},\text{semantic}}
>
T_{\text{early visual},\text{low-level}}.
$$

### Why retrieval is better than raw latent error

Do not interpret absolute latent Euclidean distance too strongly. Retrieval only asks:

> Is the correct target closer than plausible distractors?

This is more stable and easier to explain.

---

## 10. Risky / Optional Plan: Image Reconstruction

This should be a stretch goal only.

Possible use:

- run existing pretrained MindEye / Brain Diffuser / similar code if available;
- use generated reconstructions only as visualization;
- do not make reconstruction the core scientific claim.

Reason:

Generated images are influenced by:

- the fMRI decoder;
- the feature space;
- the image generator/diffusion prior;
- prompt/text conditioning;
- sampling randomness.

Thus, generated images are visually impressive but less clean as evidence for the neuroscience hypothesis.

---

## 11. Reliable Ways to Compare Latent Representations

### Concern

High-dimensional latent spaces are not automatically interpretable Euclidean spaces. Different models have different scaling, anisotropy, invariances, and training objectives.

Avoid naive claims like:

$$
\|\widehat z-z\|_2 \text{ is small, therefore the brain represents the same thing.}
$$

### Reliable strategy 1: compare relational geometry

Use RDMs rather than raw coordinates.

Question:

> Do two systems induce similar pairwise relationships among stimuli?

Statistic:

$$
\operatorname{SpearmanCorr}\left(
\operatorname{vec}_{\triangle}(D^{\text{brain}}),
\operatorname{vec}_{\triangle}(D^{\text{model}})
\right).
$$

This is robust to arbitrary scaling and monotone transformations.

### Reliable strategy 2: retrieval / rank-based evaluation

Compare a decoded feature to a candidate set and ask where the true target ranks.

Metrics:

$$
\text{top-1}, \quad \text{top-}k, \quad \text{rank percentile}.
$$

This avoids treating an absolute cosine score as intrinsically meaningful.

### Reliable strategy 3: within-feature transfer ratios

Do not directly compare raw CLIP scores with raw HOG scores. Compare each feature space to itself across conditions:

$$
\Delta_{r,\ell}
=
\text{performance}_{\text{vision},r,\ell}
-
\text{performance}_{\text{imagery},r,\ell},
$$

or

$$
T_{r,\ell}
=
\frac{\text{performance}_{\text{imagery},r,\ell}}{\text{performance}_{\text{vision},r,\ell}}.
$$

### Reliable strategy 4: use multiple feature families

The conclusion is more credible if it holds for:

- HOG and early CNN as low-level features;
- CLIP image and caption/category embeddings as high-level features.

### Reliable strategy 5: uncertainty and permutation tests

Use:

- stimulus-label permutation;
- subject bootstrap;
- trial bootstrap;
- paired within-subject comparisons;
- candidate-set permutation for retrieval.

Example null test for RSA:

1. Shuffle stimulus labels in the neural RDM.
2. Recompute correlation with model RDMs.
3. Compare observed statistic to shuffled distribution.

---

## 12. Proposed One-Week Hacking Plan

### Day 0 / before workshop

Prepare:

- NSD access and file paths;
- minimal data-loading script;
- list of NSD-Imagery trial/stimulus metadata;
- ROI masks: `nsdgeneral`, early visual, higher visual if available;
- feature extraction scripts for HOG / CNN / CLIP;
- toy RSA script using fake data.

### Day 1

Goals:

- load prepared NSD-Imagery betas;
- match trials to stimulus labels;
- load or construct ROI masks;
- average repeated trials;
- make first neural RDMs.

Deliverable:

- working data table:

$$
\text{subject} \times \text{stimulus} \times \text{condition} \times \text{ROI}.
$$

### Day 2

Goals:

- compute target stimulus feature vectors;
- compute feature RDMs;
- run basic RSA for `nsdgeneral`.

Deliverable:

- first heat map of feature-RDM alignment for vision vs imagery.

### Day 3

Goals:

- split analysis into early visual vs higher visual cortex;
- compute transfer/drop metrics;
- add permutation tests.

Deliverable:

- main hypothesis figure.

### Day 4

Goals:

- polish plots;
- add uncertainty intervals;
- maybe run ridge feature decoding if feasible;
- prepare slides and interpretation.

Deliverable:

- final presentation narrative.

### Optional stretch

- run existing reconstruction code;
- add example images/reconstructions as illustrative figures;
- compare generated-image impressions with latent-level RSA results.

---

## 13. Key Project Decisions So Far

1. **Focus on NSD-Imagery**, because it is recent and likely interesting to workshop audiences.
2. **Avoid training a new deep reconstruction model**.
3. **Use prepared fMRI betas**, not raw fMRI preprocessing.
4. **Use representation-level analysis** rather than image generation as the main scientific evidence.
5. **Primary MVP = RSA / representational geometry.**
6. **Stretch = ridge decoding and retrieval.**
7. **Use both `nsdgeneral` and ROI-specific masks.**
8. **Compare perception vs imagery through feature-level transfer/drop**, not raw performance only.
9. **Treat latent similarities carefully** using RDMs, retrieval, transfer ratios, and uncertainty tests.
10. **Frame conclusions modestly:** imagery may preserve semantic geometry more robustly than low-level visual geometry, especially in higher visual cortex.


---

# Glossary

- NSD: 
**Natural Scenes Dataset.** A large 7T fMRI dataset where subjects viewed many natural images. It is widely used for visual encoding/decoding and brain-AI representation studies.

- NSD-Imagery: 
An extension/benchmark dataset where the same subjects perform tasks involving seeing and imagining cued stimuli. It is designed to test perception-to-imagery generalization.

- fMRI: 
**Functional Magnetic Resonance Imaging.** Measures blood-oxygen-level-dependent signals, often called BOLD signals. It is an indirect, slow measure of neural activity.

- BOLD: 
**Blood-Oxygen-Level Dependent** signal. Neural activity changes local oxygen demand and blood flow; fMRI measures these hemodynamic changes.

- 7T fMRI: 
fMRI acquired at 7 Tesla magnetic field strength. Higher field strength usually gives higher spatial resolution and/or better signal than standard 3T fMRI, but with additional technical challenges.

- Voxel: 
A 3D pixel in a brain image. A voxel-wise fMRI beta pattern is a high-dimensional vector of responses across brain locations.

- Beta / beta estimate: 
In fMRI GLM analysis, a beta is an estimated response amplitude for a trial or condition at a voxel. For our purposes:

$$
\text{trial beta pattern} = \text{voxel response vector for one stimulus/trial}.
$$

- GLM: 
**General Linear Model.** In fMRI, used to estimate how much each voxel responds to each trial or condition, accounting for the hemodynamic response and nuisance signals.

- GLMsingle: 
A method/tool for estimating single-trial fMRI responses more reliably. NSD analyses often use GLMsingle-derived beta estimates.

- Trial: 
One experimental event, such as seeing an image, seeing a cue, or imagining a cued stimulus.

- Stimulus: 
The thing shown or cued in the experiment, such as an image, simple pattern, or word/concept.

- Cue: 
A symbol or prompt telling the participant which stimulus to imagine or attend to.

- Mental imagery: 
Internally generating a visual experience without directly seeing the corresponding image.

- Perception / vision condition: 
The participant actually sees the image/stimulus.

- Imagery condition: 
The participant sees a cue and imagines the corresponding stimulus internally.

- ROI: 
**Region of Interest.** A predefined brain region or mask, such as V1, V2, PPA, or a broader visual-cortex region.

- Mask: 
A set of voxels selected for analysis. Example: an early-visual mask selects voxels in V1–V4.

- `nsdgeneral`: 
A broad mask of visually responsive posterior-cortex voxels used in NSD decoding analyses. Useful for strong overall visual signal.

- Early visual cortex: 
Visual areas near the beginning of cortical visual processing, often V1, V2, V3, V4. Sensitive to edges, orientation, contrast, spatial frequency, and local layout.

- Higher visual cortex: 
Visual areas beyond early visual cortex, often involved in objects, faces, places, bodies, scenes, and semantic categories.

- V1: 
Primary visual cortex. Strongly retinotopic and sensitive to low-level visual features.

- V2, V3, V4: 
Early/intermediate visual areas. Still strongly visual and spatially organized, but progressively more complex than V1.

- Retinotopy: 
A spatial map from the visual field to cortex: nearby points in visual space correspond to nearby cortical locations.

- Ventral visual stream: 
Often called the “what” pathway. Important for object identity, faces, scenes, and categories.

- Dorsal visual stream: 
Often called the “where/how” pathway. Important for spatial processing, motion, and visually guided action.

- FFA: 
**Fusiform Face Area.** A face-selective region in ventral temporal cortex.

- PPA: 
**Parahippocampal Place Area.** A place/scene-selective region.

- OPA: 
**Occipital Place Area.** A scene/place-selective region in occipital cortex.

- RSC: 
**Retrosplenial Cortex / Retrosplenial Complex.** Often involved in scenes, navigation, and contextual spatial processing.

- EBA: 
**Extrastriate Body Area.** A body-selective visual region.

- LOC: 
**Lateral Occipital Complex.** Often associated with object shape and object recognition.

- Encoding model: 
A model predicting brain responses from stimulus features:
$$
z(I) \rightarrow x_{\text{brain}}.
$$
Example:
$$
y_r(I) = f_r(z(I)) + \epsilon.
$$

- Decoding model: 
A model predicting stimulus features or stimulus identity from brain responses:

$$
x_{\text{brain}} \rightarrow z(I) \quad \text{or} \quad x_{\text{brain}} \rightarrow I.
$$

- Reconstruction: 
Generating an image from decoded brain signals. Modern methods often decode fMRI into latent representations and then use a pretrained generative model.

- Diffusion model: 
A generative model that creates images by denoising from random noise, often conditioned on text or image embeddings.

- Generative prior: 
The tendency of a pretrained generator to produce plausible images even when the decoded brain signal is ambiguous or incomplete. This can make reconstructions visually impressive but scientifically tricky to interpret.

- CLIP: 
A vision-language model trained to align images and text. It provides image embeddings and text embeddings in a shared semantic space.

- ViT: 
**Vision Transformer.** A transformer architecture for images, often used to extract visual representations at different layers.

- CNN: 
**Convolutional Neural Network.** A classic deep image model. Early layers often encode edges/textures; later layers encode more abstract visual categories.

- HOG: 
**Histogram of Oriented Gradients.** A hand-crafted low-level/mid-level visual feature based on local edge orientations.

- Gabor features: 
Features produced by filters sensitive to orientation and spatial frequency, often used as a model of early visual processing.

- Latent representation: 
A vector representation of an image, text, or brain signal in a model’s internal feature space.

- RSA: 
**Representational Similarity Analysis.** A method comparing the geometry of representations by comparing pairwise dissimilarity matrices.

- RDM: 
**Representational Dissimilarity Matrix.** A matrix whose entries measure how different two stimuli are under a particular representation.

Example:

$$
D(a,b)=1-\operatorname{corr}(z_a,z_b).
$$

- Feature RDM: 
An RDM computed from image/model features.

- Neural RDM: 
An RDM computed from neural response patterns.

- Spearman correlation: 
Rank correlation. Useful for comparing RDMs because it is robust to monotone transformations and scaling differences.

- Ridge regression: 
Linear regression with $\ell_2$ regularization:

$$
\min_W \|XW-Z\|_F^2 + \lambda\|W\|_F^2.
$$

Useful when voxel features are high-dimensional and correlated.

- Retrieval evaluation: 
Evaluation where the model predicts a representation, then we rank candidate stimuli by similarity to that prediction. Success means the true target ranks high.

- Top-k accuracy: 
The fraction of trials where the true target is among the top $k$ most similar candidates.

- Rank percentile: 
A continuous retrieval metric measuring where the true target ranks among candidates.

- Two-alternative forced choice: 
A task where the model or human chooses which of two candidates better matches a target. In decoding papers, this is often used to assess whether reconstructions contain target-specific information.

- Transfer ratio: 
A normalized measure comparing imagery performance to vision performance:

$$
T_{r,\ell}
=\frac{\text{imagery performance}_{r,\ell}}{\text{vision performance}_{r,\ell}}.
$$

Useful because raw scores across feature spaces may not be directly comparable.

- Permutation test: 
A nonparametric test where labels are shuffled to build a null distribution. Useful for small datasets and RDM analyses.

- Bootstrap: 
A resampling method for estimating uncertainty, e.g. resampling subjects or trials.

- Domain transfer: 
Applying a model trained in one domain to another domain. Here:

$$
\text{train on perception fMRI} \rightarrow \text{test on imagery fMRI}.
$$

- Signal-to-noise ratio: 
How strong the meaningful neural signal is relative to noise. Mental imagery fMRI usually has lower signal-to-noise than visual perception fMRI.

---

## 15. Minimal Final Claim We Could Defend

A cautious final workshop claim might be:

> We used NSD-Imagery to compare neural representational geometry during perception and mental imagery. Preliminary analyses suggest that imagery-related responses align more robustly with high-level semantic feature geometry than with low-level visual feature geometry, especially outside early visual cortex. This supports the view that the mind’s eye may preserve visual meaning more reliably than precise spatial detail.

If results are null or mixed, a still useful claim would be:

> We built a reproducible pipeline for NSD-Imagery representation analysis and found that conclusions depend strongly on ROI choice, feature representation, and whether performance is normalized against perception. This provides a useful foundation for future perception-to-imagery transfer studies.

---

## 16. Immediate To-Do List

1. Confirm exact NSD-Imagery access path after NSD data access approval.
2. Locate prepared beta files and stimulus metadata.
3. Identify available ROI masks:
   - `nsdgeneral`;
   - V1–V4;
   - higher visual / category-selective ROIs.
4. Write a minimal loader that returns:

$$
(i, s, c, r) \mapsto x_{i,s,c,r}.
$$

5. Extract feature vectors for NSD-Imagery target stimuli.
6. Implement RDM computation and RSA correlation.
7. Make a toy figure with fake data before the workshop.
8. Prepare the Slack pitch and invite collaborators with fMRI/NSD expertise.
