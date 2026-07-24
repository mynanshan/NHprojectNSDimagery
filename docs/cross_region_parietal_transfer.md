# Cross-region visual-to-parietal transfer

Notebook 08 tests whether imagery responses in parietal cortex preserve the
target structure seen during perception in visual cortex.

This is a post-hoc, collaborator-inspired analysis. It is designed to be
informative without treating visual and parietal voxels as interchangeable.

## Why direct voxel transfer is invalid

A decoder trained on \(p_V\) visual voxels cannot directly receive \(p_P\)
parietal voxels. The voxel counts differ, and the model weights refer to
different anatomical locations.

Notebook 08 uses two valid alternatives:

1. compare target RDMs, which do not require equal voxel dimensions;
2. learn a low-dimensional parietal-to-visual alignment from independent
   vision data before applying a visual decoder.

## ROI definitions

The analysis uses the subject-specific volumetric `streams.nii.gz` atlas.

| ROI | Stream labels | Use |
|---|---|---|
| `visual_streams` | 1, 2, 3, 5, 6 | Primary perception region |
| `dorsal_parietal` | 4, 7 | Primary parietal imagery region |
| `early` | 1 | Descriptive localization |
| `ventral` | 5 | Descriptive localization |
| `lateral` | 6 | Descriptive localization |
| `midparietal` | 4 | Descriptive localization |
| `parietal` | 7 | Descriptive localization |

The two primary regions are disjoint. Their voxel counts are matched within
subject, up to the default cap of 1,200 voxels.

This also corrects an ambiguity in earlier notebooks: the exploratory
`higher_visual` definition used stream labels 5, 6, and 7, so it combined
ventral, lateral, and parietal voxels. Notebook 08 isolates them.

## Primary RDM test

For subject \(s\), task \(c\), and ROI \(r\), repeated patterns are averaged by
target and converted to a six-target correlation-distance RDM
\(D^c_{s,r}\).

The planned same-region transfer is

\[
\rho_{\mathrm{same},s}
=
\rho\left(
D^{\mathrm{vision}}_{s,\mathrm{visual}},
D^{\mathrm{imagery}}_{s,\mathrm{visual}}
\right).
\]

The planned cross-region transfer is

\[
\rho_{\mathrm{cross},s}
=
\rho\left(
D^{\mathrm{vision}}_{s,\mathrm{visual}},
D^{\mathrm{imagery}}_{s,\mathrm{parietal}}
\right).
\]

The collaborator-inspired contrast is

\[
\Delta_s=\rho_{\mathrm{cross},s}-\rho_{\mathrm{same},s}.
\]

The script also reports:

- vision parietal to imagery parietal transfer;
- the reverse parietal-vision to visual-imagery comparison;
- target-label permutation tests;
- leave-one-target-out sensitivity;
- standard and repeat-crossvalidated RDM comparisons;
- vision split-half and imagery run-to-run reliability;
- subject-level values and exact group sign-flip tests.

Sets A and B are reported separately and as a subject-level average. Set C is
not included in the primary imagery outcome.

## Secondary aligned decoder

The secondary model learns the cross-region mapping using Set C vision trials
only.

Let \(P_C\) and \(V_C\) contain paired parietal and visual responses from the
same Set C vision trials. Separate PCA transforms produce low-dimensional
patterns \(Z_C^P\) and \(Z_C^V\). Ridge alignment solves

\[
\widehat A
=
\arg\min_A
\left\|Z_C^V-Z_C^P A\right\|_F^2
+\lambda\left\|A\right\|_F^2.
\]

PCA dimension and \(\lambda\) are selected by leaving out one Set C target
group at a time. PCA and ridge are refitted inside every fold.

After selection, the alignment is frozen. For Set A or B:

1. train six target centroids from visual-region vision patterns;
2. test visual-region imagery directly;
3. map parietal imagery into visual latent space and test it with the same
   visual centroids;
4. fit a separate within-parietal centroid baseline;
5. compare the aligned result with a null formed by shuffling Set C
   visual-parietal trial pairings.

The imagery data do not select PCA dimensions, ridge penalties, or alignment
weights. The test still uses the same six A/B target identities in perception
and imagery, so it is cross-condition identification rather than decoding of
unseen images.

## Run the analysis

Open:

```text
notebooks/08_cross_region_parietal_transfer.ipynb
```

The default notebook command analyzes all eight subjects:

```bash
python scripts/run_cross_region_transfer.py \
  --data-root "$PWD/data/nsd" \
  --subjects 1-8 \
  --output-dir "$PWD/outputs/08_cross_region_transfer" \
  --max-voxels 1200 \
  --n-rdm-splits 100 \
  --n-alignment-permutations 100
```

The existing minimal NSD-Imagery download already includes the beta HDF5 file
and `streams.nii.gz`. No new core-NSD data, image bank, or GPU is required.

The first run extracts and caches ROI betas under:

```text
data/nsd/derived/notebook08_cross_region/
```

Later runs reuse those caches.

For a quick subject-01 smoke test:

```bash
python scripts/run_cross_region_transfer.py \
  --data-root "$PWD/data/nsd" \
  --subjects 1 \
  --output-dir "$PWD/outputs/08_cross_region_transfer_smoke" \
  --max-voxels 300 \
  --n-rdm-splits 10 \
  --n-alignment-permutations 10
```

## Main outputs

| File | Content |
|---|---|
| `planned_rdm_comparisons.csv` | Same-visual, cross-region, within-parietal, and reverse transfer |
| `rdm_contrasts_subject_level.csv` | Paired subject-level differences |
| `rdm_group_summary.csv` | Exact sign-flip summaries |
| `rdm_contrasts_group.csv` | Group summaries of the planned contrasts |
| `roi_reliability.csv` | Vision split-half and imagery run-to-run RDM reliability |
| `rdm_transfer_subject_level.csv` | Full vision-ROI by imagery-ROI transfer matrix |
| `aligned_decoder_subject_level.csv` | Six-way imagery identification accuracies |
| `alignment_selection.csv` | Set C vision cross-validation history |
| `aligned_decoder_contrasts.csv` | Paired decoder differences |
| `stream_roi_counts.csv` | Available and sampled voxel counts |
| `subject_rdms.npz` | Standard and crossvalidated subject RDMs |

Notebook 08 saves its summary figures under
`outputs/08_cross_region_transfer/figures/`.

## Interpretation

A scientifically interesting result would have several features:

- cross-region transfer is positive;
- cross-region transfer exceeds same-visual transfer within subject;
- Sets A and B point in a similar direction;
- the result is not driven by one target;
- parietal imagery reliability is not the sole explanation;
- standard and crossvalidated RDM analyses agree;
- Set C alignment predicts held-out Set C vision patterns above its mean
  baseline;
- aligned parietal imagery identification beats chance and shuffled alignment.

Even that pattern would be evidence consistent with transformed
perception-to-imagery information. It would not prove a directed neural
transformation or connectivity mechanism.
