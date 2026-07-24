# Original NSD-Imagery paper: methodology in compact notation

> **Related notebooks:** [02](../../notebooks/02_event_alignment_neural_rdm.ipynb),
> [03](../../notebooks/03_regional_vision_imagery_rsa.ipynb), and
> [23](../../notebooks/23_paper_brain_correlation.ipynb)
>
> **Role:** original-paper reference

This note summarizes Kneeland et al., *NSD-Imagery: A Benchmark Dataset for Extending fMRI Vision Decoding Methods to Mental Imagery* (CVPR 2025). It distinguishes the paper's benchmark from our RSA analysis.

## 1. What data enter the reconstruction models?

The paper reconstructs from **prepared GLMsingle beta patterns**, not directly from raw fMRI time courses. It uses the same preparation downloaded for this project: 1.8-mm volumetric, version-3 `betas_fithrf_GLMdenoise_RR`.

Let

$$
\mathbf b_{s,c,i,r}\in\mathbb R^{V_s}
$$

be the beta pattern for participant $s$, condition $c\in\{vis,img\}$, target $i$, repetition $r$, and the model's chosen voxels. The paper Z-scores fMRI patterns separately within each run and averages repeated trials for reconstruction:

$$
\bar{\mathbf b}_{s,c,i}
=\frac{1}{R_c}\sum_{r=1}^{R_c}
z_{\mathrm{run}(r)}(\mathbf b_{s,c,i,r}),
\qquad R_{vis}=8,\;R_{img}=16.
$$

Most benchmarked methods use the broad `nsdgeneral` posterior visual-cortex mask. The Takagi method instead uses separate early and ventral/higher visual ROIs.

## 2. Training data and the cross-decoding test

The paper does not train the reconstruction systems on the small NSD-Imagery dataset. It takes five existing methods trained on the much larger **core NSD vision data**:

$$
\mathcal D^{core}_s=\{(\mathbf b^{core}_{s,n},I_n)\}_{n=1}^{N_s}.
$$

For reconstruction method $m$, training learns a subject-specific mapping from seen-image fMRI to one or more pretrained image/text latent spaces:

$$
\widehat{\mathbf z}_{m}=D_{m,s}(\mathbf b).
$$

Depending on the method, $\mathbf z_m$ can contain CLIP image/text embeddings, VAE latents, VGG features, or other conditioning variables. A generative or optimization stage then produces an image:

$$
\widehat I\sim G_m(\widehat{\mathbf z}_{m}).
$$

The decisive benchmark is zero-shot cross-decoding:

$$
\bar{\mathbf b}_{s,img,i}
\xrightarrow{\;D_{m,s}\;}
\widehat{\mathbf z}_{m,s,img,i}
\xrightarrow{\;G_m\;}
\widehat I_{m,s,img,i},
$$

without fitting the decoder on NSD-Imagery imagery trials. The same decoders are also applied to NSD-Imagery vision betas as a closer-domain reference.

The benchmark includes Brain Diffuser, MindEye1, MindEye2, iCNN, and the Takagi `+Decoded Text` approach. It is therefore a comparison of existing pipelines, not one new unified reconstruction model.

Only NSD subjects 1, 2, 5, and 7 completed all 40 core-NSD sessions, so the reconstruction benchmarks generally use those four participants. This differs from Notebook 03, which performs RSA on all eight NSD-Imagery participants and does not train on core NSD.

## 3. Evaluation

For each method, condition, participant, and target, the paper samples ten reconstructions from the output distribution. It evaluates them with:

- low-level image similarity, including pixel correlation and SSIM;
- intermediate and high-level feature comparisons, including AlexNet, Inception, CLIP, EfficientNet, and SwAV;
- predicted-versus-measured brain-response correlation using an image-to-fMRI encoding model;
- human two-alternative forced-choice identification and similarity ratings.

A schematic feature metric is

$$
S_k(I_i,\widehat I_i)
=\operatorname{sim}\bigl(\phi_k(I_i),\phi_k(\widehat I_i)\bigr),
$$

where $\phi_k$ is a feature extractor. Two-way identification asks whether the target reconstruction is closer to the correct reference than a distractor reconstruction.

For the brain metric, an encoding model $E_s$ predicts how the reconstruction should activate the participant's brain:

$$
\widehat{\mathbf b}'_{s,i}=E_s(\widehat I_i),
\qquad
S_{brain}=\operatorname{corr}(\bar{\mathbf b}_{s,i},
                              \widehat{\mathbf b}'_{s,i}).
$$

More precisely, this is a spatial Pearson correlation **across voxels for each
target and reconstruction sample**, followed by averaging targets, samples,
and subjects. It is therefore not computable from measured betas alone. The
paper defines early visual cortex as V1--V4 inside `nsdgeneral`, higher visual
cortex as the set complement `nsdgeneral − early`, and visual cortex as all of
`nsdgeneral`.

## 4. What the paper concludes

Some perception-trained decoders produce recognizable mental-image reconstructions, but a method's ranking on ordinary vision reconstruction does not reliably predict its ranking on imagery. Simpler linear decoding backbones and multimodal feature decoding generalize relatively well; complex vision-optimized systems can overfit the perception domain.

Low-level/structural metrics and early-visual brain correlations drop more from vision to imagery than higher-level metrics. The claim concerns the **size of the vision-to-imagery drop**; it does not require the imagery early-visual score itself to be below the imagery higher-visual score. This motivates our representation-level question, but it does not guarantee that CLIP geometry will explain our six-target neural RDMs.

## 5. How our notebooks differ

| Original paper | Notebooks 02 and 03 |
|---|---|
| Train existing decoders on thousands of core-NSD seen-image trials | Do not use core NSD training data |
| Apply decoder to averaged NSD-Imagery betas | Directly compare beta-pattern RDMs |
| Generate reconstructed images | Do not generate images |
| Evaluate reconstruction similarity and human identification | Evaluate perception-imagery geometry transfer and feature RSA |
| Reconstruction subjects mainly 1, 2, 5, 7 | RSA subjects 1 through 8 |

## Sources

- [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kneeland_NSD-Imagery_A_Benchmark_Dataset_for_Extending_fMRI_Vision_Decoding_Methods_CVPR_2025_paper.html)
- [Paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Kneeland_NSD-Imagery_A_Benchmark_Dataset_for_Extending_fMRI_Vision_Decoding_Methods_CVPR_2025_paper.pdf)
- [Supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Kneeland_NSD-Imagery_A_Benchmark_CVPR_2025_supplemental.pdf)
- [Reproduction and ML next steps](../notes/paper_reproduction_and_ml_next_steps.md)
