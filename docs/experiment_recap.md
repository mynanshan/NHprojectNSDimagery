# NSD-Imagery experiment recap

This note connects the participant's experience to the files used in Notebooks 02 and 03.

## 1. Vocabulary in one minute

- A **session** is one visit to the scanner. NSD-Imagery added one scanning session for each of the eight original NSD participants.
- A **run** is a contiguous block of the scan with one task and one stimulus set. The participant stays in the scanner, but the task block starts and ends as a unit.
- A **behavioral trial** is one repetition of the task: receive a stimulus or cue, perform the instruction, and make a response.
- A **raw fMRI volume** is a whole-brain image acquired at one time point. A run produces a time series of these volumes.
- A **beta volume** is a whole-brain map of fitted response amplitudes for one modeled event. It is estimated from the time series with a GLM.
- A **beta pattern** is the vector of beta values selected from an ROI for one event: one scalar per voxel.

The paper defines a run as consecutive trials with the same visual modality and stimulus type. Vision and imagery runs were about four minutes; the released attention designs are longer.

## 2. Before entering the scanner

Each participant learned 18 cue-to-target associations:

- Set A, **simple**: six bars/crosses.
- Set B, **complex**: five natural scenes and one artwork.
- Set C, **conceptual**: six concepts such as `stripes`, `zebra`, and `fruit`.

Every target had a unique one-letter cue. Participants practiced both recognizing the targets and recalling them from their cues.

## 3. What one participant did in chronological order

The released acquisition/GLM order is:

| Order | Run name | Meaning | What happens on one behavioral trial | Trials | Modeled betas |
|---:|---|---|---|---:|---:|
| 1 | `visA` | vision, simple set | See an image and a letter for 3 s; report whether they match; 1 s rest | 48 | 48 |
| 2 | `attA` | attention, simple set | Receive a cue, then detect its target in a rapid image stream | 48 | 96 |
| 3 | `imgA_1` | imagery, simple set, run 1 | See only the cue for 3 s; imagine the target; rate vivid/not vivid; 1 s rest | 48 | 48 |
| 4 | `visB` | vision, complex set | Same vision task with Set B | 48 | 48 |
| 5 | `attB` | attention, complex set | Same attention task with Set B | 48 | 96 |
| 6 | `imgB_1` | imagery, complex set, run 1 | Same imagery task with Set B | 48 | 48 |
| 7 | `visC` | vision, conceptual set | See a varying example of the cued concept | 48 | 48 |
| 8 | `attC` | attention, conceptual set | Same attention task with Set C | 48 | 96 |
| 9 | `imgC_1` | imagery, conceptual set, run 1 | Same imagery task with Set C | 48 | 48 |
| 10 | `imgA_2` | imagery, simple set, run 2 | Independent repetition of `imgA_1` | 48 | 48 |
| 11 | `imgB_2` | imagery, complex set, run 2 | Independent repetition of `imgB_1` | 48 | 48 |
| 12 | `imgC_2` | imagery, conceptual set, run 2 | Independent repetition of `imgC_1` | 48 | 48 |

The names are literal: `vis` = vision, `att` = attention, `img` = imagery; `A/B/C` is the stimulus set; `_1/_2` distinguishes the two imagery runs.

There are nine distinct task-by-set combinations. Imagery is repeated, adding three more runs:

$$
3\text{ vision}+3\text{ attention}+2\times3\text{ imagery}=12\text{ runs}.
$$

## 4. Why every run has 48 behavioral trials

Every set has six targets, and every target occurs eight times in a run:

$$
6\text{ targets}\times8\text{ repetitions}=48\text{ trials per run}.
$$

Therefore, **within each participant**:

- vision has one run per set, so each target has 8 vision trials;
- imagery has two runs per set, so each target has $8+8=16$ imagery trials.

The factor of two comes from the two imagery runs, not from the eight participants. Notebook 02 averages these repetitions within one participant. Notebook 03 repeats that operation separately for every participant before group inference.

Across the 12 runs, the participant completed

$$
12\times48=576\text{ behavioral trials}.
$$

## 5. Why 576 trials become 720 beta volumes

Vision and imagery contribute one modeled event per behavioral trial. Attention contributes two: a cue epoch and a detection epoch.

$$
\begin{aligned}
\text{vision} &: 3\times48=144,\\
\text{imagery} &: 6\times48=288,\\
\text{attention} &: 3\times(2\times48)=288,\\
\text{total} &: 144+288+288=720\text{ beta volumes}.
\end{aligned}
$$

Equivalently, nine one-beta-per-trial runs plus three two-beta-per-trial runs give

$$
9\times48+3\times96=720.
$$

Notebooks 02 and 03 exclude attention as recommended by the benchmark authors, leaving

$$
144\text{ vision}+288\text{ imagery}=432\text{ selected beta volumes}.
$$

The beta file still contains all 720 volumes. This is why the selection plot has gaps: it shows the positions of the 432 retained events in the full file, with the 288 attention positions left unused.

## 6. What `(240, 36)` and `(480, 36)` mean

These are shapes of the released **event-design arrays**, not shapes of raw fMRI data or beta files.

- Rows describe the run's timing grid. A vision/imagery design has 240 timing bins, consistent with a roughly 240-second run. An attention design has 480 bins because that task is longer.
- The 36 columns are the global event/condition regressors used across the experiment: 6 vision target columns, 12 attention columns (six cue and six detection epochs), and 18 imagery target columns (six for each of A, B, and C).
- Only the columns relevant to a particular run are active.
- `n_modeled_events` counts nonzero event onsets, not rows or columns.

Thus `visA` has shape `(240, 36)` but only 48 nonzero event onsets, one per behavioral trial, and therefore 48 expected trial betas. `attA` has 96 nonzero onsets because each of its 48 behavioral trials contributes two modeled epochs.

The fMRI acquisition itself has 1.6-second sampling. Do not interpret the 240 design rows as 240 beta volumes or as the spatial dimensions of the brain.

## 7. Raw fMRI versus the prepared betas we analyze

At voxel $v$ and acquired time point $t$, the raw/preprocessed BOLD time course is $y_v(t)$. A simplified voxel-wise GLM is

$$
y_v = X\beta_v + Z\gamma_v + \varepsilon_v,
$$

where $X$ contains event regressors convolved with a hemodynamic response function, $Z$ contains nuisance regressors, and $\beta_v$ contains fitted event amplitudes.

For modeled event $e$, the fitted whole-brain beta volume is

$$
\boldsymbol\beta_e=(\beta_{e1},\ldots,\beta_{eV}),
$$

one scalar response estimate for each voxel. It is better to call this a model-based response-amplitude estimate than an average over the entire temporal domain: the GLM uses the relevant time points, the expected delayed BOLD response, overlapping events, nuisance effects, denoising, and regularization.

Our downloaded HDF5 files contain 720 prepared GLMsingle beta volumes per participant. The notebooks extract ROI voxels and work with arrays shaped

$$
720\text{ events}\times V_{ROI}\text{ voxels}.
$$

They do **not** download or re-fit the raw fMRI time series.

## 8. From beta patterns to RDM geometry

For one participant, task, set, and ROI, repeated trial patterns are averaged for each of the six targets:

$$
\bar{\boldsymbol\beta}_i=\frac{1}{R}\sum_{r=1}^{R}\boldsymbol\beta_{ir}.
$$

These six vectors are six points in a high-dimensional voxel space. Their **representational geometry** is the collection of pairwise relationships among the points. The neural RDM stores correlation distance:

$$
D_{ij}=1-\operatorname{corr}(\bar{\boldsymbol\beta}_i,\bar{\boldsymbol\beta}_j).
$$

With six targets there are $6\times5/2=15$ unique off-diagonal distances. “Vision and imagery have similar geometry” means that target pairs that are relatively close or far apart in vision tend to have the same ordering in imagery:

$$
\rho=\operatorname{Spearman}(\operatorname{upper}(D^{vis}),
                              \operatorname{upper}(D^{img})).
$$

It does not mean that vision and imagery have equal beta values.

Values near 1 in the heatmap mean that two target-average voxel patterns have near-zero Pearson correlation; values above 1 mean negative correlation. High off-diagonal values do **not** by themselves show that tasks are distinguishable: each displayed RDM contains targets from one task, not distances between tasks. Classification or cross-validated within-target versus between-target distances would be needed for a direct discriminability claim.

The uniformly large values are also affected by preprocessing. Z-scoring each voxel across trials within a run makes every target pattern a deviation from that run's mean; the deviations must balance across the small set of targets, which can push some pairwise correlations toward zero or below zero. The relative ordering of distances and its reproducibility matter more here than whether the heatmap is globally yellow.

## 9. Split-half reliability

Reliability asks whether the target geometry can be reproduced from independent measurements.

For a balanced split:

1. split every target's repetitions into two equal groups;
2. average trials for each target separately within each half;
3. compute one six-target RDM from each half;
4. correlate their 15 upper-triangle distances.

Notebook 02 makes one odd-versus-even repetition split. Notebook 03 improves the vision estimate by repeating random balanced 4-versus-4 splits 200 times and reporting the median and quartiles. For imagery, Notebook 03 uses the scientifically cleaner independent comparison `imgX_1` versus `imgX_2`, each containing eight repetitions per target.

A positive, stable split-half correlation means the ordering of target distances is reproducible. A value near zero means the estimated geometry is dominated by noise or is too weak for the available trials. A negative value means the distance ordering reverses across halves; it is not evidence for a stable “opposite” representation.

Reliability is an interpretation aid, not a reason to delete inconvenient subjects. A vision-imagery correlation can sometimes be detectable even when imagery split-half reliability is low because the full imagery RDM averages all 16 repetitions and is compared with a higher-SNR vision template. Nevertheless, weak imagery reliability limits how strongly that transfer should be interpreted.

## Sources

- [NSD-Imagery CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kneeland_NSD-Imagery_A_Benchmark_Dataset_for_Extending_fMRI_Vision_Decoding_Methods_CVPR_2025_paper.html)
- [NSD-Imagery supplementary material](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Kneeland_NSD-Imagery_A_Benchmark_CVPR_2025_supplemental.pdf)
- [Natural Scenes Dataset](https://naturalscenesdataset.org/)
