# Review of Notebook 02 and Notebook 03 results

This review records what the current checkpoints support before further analysis choices are made.

## 1. Notebook 02: exploratory pilot

Notebook 02 uses `subj01`, 2,000 sampled `nsdgeneral` voxels, and Sets A, B, and C.

The alignment checks pass:

- 720 beta volumes exist in acquisition/GLM order;
- 432 unique vision/imagery beta indices are selected;
- each target has 8 vision and 16 imagery trials;
- within-run Z-scoring produces approximately zero mean and unit standard deviation.

The single odd/even split and vision-imagery RDM correlations were:

| Set | Vision reliability | Imagery reliability | Vision-imagery RDM rho |
|---|---:|---:|---:|
| A, simple | 0.507 | 0.100 | 0.125 |
| B, complex | -0.050 | 0.429 | 0.582 |
| C, conceptual | -0.004 | -0.182 | -0.454 |

Set B supplied an interesting lead: vision and imagery had similar distance ordering ($\rho=0.582$). But the vision split-half estimate was negative, so this one-subject result was not reliable enough to treat as a finding. Set C is not suitable for the main vision-imagery comparison because its vision trials use varying images for each concept; the benchmark authors also recommend excluding them from vision evaluation.

## 2. Why `subj01` became the pilot

`subj01` was the first participant explored while building and debugging the pipeline. The Set B lead, ROI choices, and follow-up questions were therefore partly selected after seeing this participant.

Notebook 03 keeps `subj01` visible but excludes it from group tests. Subjects 02-08 are “held out” from that exploratory choice and provide a less circular replication test. This is not a decoder training/test split and no trials are withheld within those participants: all their relevant repetitions are used to estimate each target RDM.

The held-out language should remain modest. This was not a preregistered external replication; all participants share the same dataset, protocol, and analysis code.

## 3. Notebook 03: neural transfer across held-out participants

Notebook 03 uses all eight participants, three ROIs, Sets A and B, and 1,200 sampled voxels per subject/ROI. Group tests use only subjects 02-08.

### Main held-out results

| Set | ROI | Mean vision-imagery rho | One-sided exact p | Two-sided exact p |
|---|---|---:|---:|---:|
| A | early visual | 0.415 | 0.0078 | 0.0156 |
| A | higher visual | 0.269 | 0.0938 | 0.1875 |
| A | `nsdgeneral` | 0.339 | 0.0156 | 0.0313 |
| B | early visual | 0.135 | 0.2344 | 0.4688 |
| B | higher visual | 0.365 | 0.0078 | 0.0156 |
| B | `nsdgeneral` | 0.364 | 0.0313 | 0.0625 |

The pre-specified primary result—positive Set B transfer in higher visual cortex—is supported. All seven held-out participants have a positive coefficient, yielding the smallest possible one-sided exact sign-flip p-value with seven participants:

$$
p=1/2^7=0.0078125.
$$

However, the secondary claim that Set B transfer is *larger* in higher than early visual cortex is not supported:

$$
\overline{\rho_{higher}-\rho_{early}}=0.231,
\qquad p_{greater}=0.164.
$$

Therefore the result is “positive higher-visual transfer,” not established anatomical selectivity for higher visual cortex.

### Reliability limits the strength of the claim

Set B higher-visual imagery run-to-run reliability is positive for subjects 02-05 but negative for subjects 06-08. Vision split reliability is also modest for several participants. Averaging all repetitions can reveal a cross-condition signal that is not clear in either imagery run alone, but unstable imagery RDMs mean the magnitude and detailed geometry should not be overinterpreted.

The appropriate statement is:

> Across seven held-out participants, Set B vision and imagery target-distance orderings were consistently positively associated in the sampled higher-visual ROI, despite heterogeneous and sometimes poor within-condition reliability.

This is promising preliminary transfer evidence, not a precise or highly reliable population geometry estimate.

## 4. What “geometry” means here

Each target is a point represented by its voxel-response vector. The six points form a configuration, like a six-star constellation in a space with 1,200 voxel axes. Geometry refers to the 15 pairwise distances among these points.

We compare the **ordering** of the 15 distances. A positive vision-imagery RDM correlation means, for example, that target pairs relatively similar during perception also tend to be relatively similar during imagery. It does not require corresponding voxel amplitudes to be equal, and it does not say which visual property creates that organization.

## 5. HOG and CLIP as candidate explanations

### HOG

Histogram of Oriented Gradients is a hand-designed image descriptor. It converts a grayscale image into local edge-gradient directions, pools orientation histograms over spatial cells, and normalizes neighboring blocks. Notebook 03 produces a 6,084-dimensional HOG vector for each image.

HOG is mainly sensitive to local edges, orientation, shape, and coarse spatial layout. It has no explicit concept vocabulary. Set A is a useful positive control because its bars and crosses differ strongly in precisely these properties.

### CLIP

CLIP is trained on image-text pairs so that matching images and text occupy nearby locations in a shared embedding space. Notebook 03 uses the projected 512-dimensional image embedding from `openai/clip-vit-base-patch32`.

CLIP often captures objects, scenes, and semantic content, but it is not a pure “semantics meter”: its image embeddings also contain color, texture, composition, and other visual information. Conversely, HOG and CLIP can correlate. In the current Set B stimuli their RDM correlation is $\rho=0.354$.

## 6. Feature RSA results: the planned semantic explanation fails

The pipeline has a reassuring positive control: for Set A vision, early-visual neural geometry aligns with HOG (held-out mean $\rho=0.365$, one-sided $p=0.0078$).

For Set B, however, neither feature model provides the predicted explanation:

- higher-visual imagery versus HOG: mean $\rho=0.112$, $p_{greater}=0.0938$;
- higher-visual imagery versus CLIP: mean $\rho=-0.020$, $p_{greater}=0.641$;
- planned CLIP-minus-HOG contrast: mean $-0.132$, two-sided $p=0.0313$.

The contrast is significant in the **opposite** direction: HOG exceeds CLIP. We must not report this as support for semantic preservation.

The second planned drop contrast is positive (mean 0.251, one-sided $p=0.0391$), but its two-sided p-value is 0.0781 and it combines weak/negative CLIP alignments. It is therefore not clean evidence that low-level structure drops more than semantics.

The most defensible current conclusion is:

> Set B perception-imagery neural geometry transfers in higher visual cortex, but this transferable organization is not explained by the selected CLIP embedding and is, if anything, somewhat closer to HOG. The representational source remains unresolved.

## 7. Important cautions and next checks

1. **Only six targets.** Each RDM has only 15 unique distances, making results sensitive to individual targets and producing coarse exact p-values.
2. **Weak reliability.** Several imagery run-to-run RDM correlations are zero or negative.
3. **Cue-letter confounding.** Vision includes a central letter cue and imagery is instructed by that letter. Although vision is labeled by the image actually shown and mismatch trials help separate cue from image, cue-related activity remains a plausible contributor that should be checked.
4. **Voxel sampling.** Results use one deterministic sample of 1,200 voxels per ROI. Stability across voxel seeds and voxel caps has not yet been demonstrated.
5. **ROI overlap and specificity.** `nsdgeneral`, early visual, and higher visual masks are not three independent samples of cortex. The current early mask contains V1-V3 dorsal/ventral labels, not V4.
6. **Many secondary tests.** Only the primary held-out Set B higher-visual test should be given confirmatory emphasis. Feature/ROI tables are exploratory unless multiplicity is addressed.
7. **RDM magnitude is not discriminability.** Large heatmap off-diagonals do not prove that targets or tasks can be decoded.

The next analysis is now implemented in
[Notebook 04](../notebooks/04_measurement_first_validation.ipynb). It first
asks whether targets can be identified within each condition and whether
vision-derived centroids identify imagery trials. It then adds cue-match versus
cue-mismatch controls, crossvalidated distances, exact RDM label permutations,
leave-one-target-out sensitivity, and group noise ceilings. The conceptual
reason for this revised order is documented in
[RSA scope and revised analysis](rsa_scope_and_revised_analysis.md).
