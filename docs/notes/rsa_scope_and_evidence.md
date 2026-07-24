# What an RDM correlation does—and does not—answer

> **Related notebooks:** [03](../../notebooks/03_regional_vision_imagery_rsa.ipynb)
> and [20](../../notebooks/20_measurement_first_validation.ipynb)
>
> **Role:** evidence-hierarchy and interpretation note

This note explains why Representational Similarity Analysis (RSA) is common,
why it is not sufficient by itself for our broad scientific question, and why
Notebook 20 changes the order of the analysis.

## 1. The short answer

Correlating two representational dissimilarity matrices (RDMs) is a standard
RSA analysis. It asks a valid but narrow question:

> Do the two representations rank the same stimulus pairs as relatively
> similar and dissimilar?

That is not the same as asking whether the representations contain all the
same information, use the same neural code, or allow accurate target
identification. For this project, ordinary RDM correlation is therefore a
**secondary relational summary**, not the primary evidence for perception-to-
imagery transfer.

The primary question in Notebook 20 is more direct:

> Can target identity be identified reliably within imagery, and can a target
> template estimated from vision identify imagery trials?

## 2. Why use an RDM at all?

Suppose condition $c$ represents target $i$ by a vector

$$
\mathbf z_i^{(c)}\in\mathbb R^{p_c}.
$$

For fMRI, the coordinates can be voxels. For HOG or CLIP, they are feature
dimensions. The dimensions need not have the same identities or even the same
number. We define a pairwise dissimilarity

$$
D_{ij}^{(c)}=d\!\left(\mathbf z_i^{(c)},\mathbf z_j^{(c)}\right).
$$

The matrix $D^{(c)}$ is the RDM. Its upper triangle contains one value for each
unordered target pair. With six targets, this is

$$
\binom{6}{2}=15
$$

values.

This relational description is useful because coordinates from two systems
often cannot be matched directly. A rotation or permutation of feature axes
can leave all pairwise relationships unchanged. The RDM provides a common
language for comparing brains, subjects, and computational models without
requiring voxel 17 to mean the same thing as CLIP dimension 17.

This coordinate independence is the main reason RSA is widely used—not
because an RDM is a complete representation.

## 3. What the RDM correlation means

Let $\operatorname{vec}_\triangle(D)$ contain the 15 upper-triangle values.
The notebooks use

$$
\rho_S=
\operatorname{corr}\!\left(
\operatorname{rank}(\operatorname{vec}_\triangle(D^{\text{vision}})),
\operatorname{rank}(\operatorname{vec}_\triangle(D^{\text{imagery}}))
\right).
$$

A positive $\rho_S$ says that a pair which is relatively far apart during
vision tends also to be relatively far apart during imagery. A negative value
says that the ordering tends to reverse. A value near zero says that no stable
monotonic ordering is apparent in that estimate.

For example, a positive value is compatible with the statement “the brain
distinguishes targets 1 and 2 more strongly than targets 1 and 3 in both
conditions.” It does **not** say that either pair is decodable, that the same
voxels carry the distinction, or that a fraction $\rho_S$ of information has
transferred.

There is no universal rule that $\rho=0.5$ is “high.” Its interpretation
depends on uncertainty, within-condition reliability, the noise ceiling, the
number and range of stimuli, and whether the effect persists across subjects
and target omissions.

## 4. Information that is discarded

An RDM preserves the selected pairwise dissimilarities. Depending on the
distance metric and rank correlation, it discards or ignores:

- the identities of individual voxel/feature axes;
- absolute activation levels and, for correlation distance, pattern mean and
  scale;
- metric magnitude when Spearman correlation keeps only ranks;
- trial-to-trial distributions after repetitions are averaged;
- temporal information already summarized by the beta estimates;
- any aspect of the representation not expressed by the selected distance.

Therefore “neural geometry transfers” should always be translated into its
specific operational meaning here:

> The ordering of target-pair dissimilarities is positively associated between
> the two estimated RDMs.

The shorter phrase is convenient, but the operational statement is the claim
we can actually test.

## 5. Is correlation distance the only choice?

No. The representation and distance must match the scientific question.

| Analysis | What it emphasizes | Important limitation |
|---|---|---|
| Correlation distance, $1-r$ | Pattern shape after removing mean and scale | Biased upward by measurement noise; no meaningful zero after averaging noisy data |
| Cosine distance | Vector direction relative to the origin | Sensitive to centering choice; noisy estimates are still biased |
| Euclidean distance | Absolute separation in feature space | Sensitive to scale and noisy dimensions |
| Crossvalidated Mahalanobis (crossnobis) | Reproducible condition contrast after noise normalization | Needs independent partitions and a stable noise-covariance estimate |
| Crossvalidated dot-product distance | Reproducible contrast with an interpretable zero | Does not whiten correlated voxel noise |

Notebook 20 adds the last option. For two independent partitions $a$ and $b$,
it estimates

$$
\widehat d_{ij}=\frac{1}{p}
(\widehat{\boldsymbol\mu}_{i,a}-\widehat{\boldsymbol\mu}_{j,a})^\top
(\widehat{\boldsymbol\mu}_{i,b}-\widehat{\boldsymbol\mu}_{j,b}).
$$

Under independent noise and no reproducible target difference, its expected
value is zero. Positive values mean that the target contrast points in a
consistent voxel-space direction in the two partitions. This is an
unwhitened relative of crossnobis, chosen because estimating a full covariance
matrix is precarious with the current trial count and 1,200 voxels.

For HOG and CLIP, cosine distance remains a reasonable conventional starting
point, but it is a modelling choice rather than a proof that either feature
space is the brain's representation. Nonlinear feature extraction followed by
a linear distance is not contradictory: CLIP has already performed the
nonlinear mapping from pixels to its embedding. The distance asks whether the
resulting feature vectors are oriented similarly. Alternative metrics can be
added only when motivated and tested without turning metric search into
post-hoc result shopping.

## 6. Why this dataset makes ordinary RSA fragile

Each six-target RDM has only 15 entries, and those entries are dependent
because every target appears in five pairs. Noise in one target affects five
distances. A nominal correlation can therefore be driven by a small number of
targets. In addition, some imagery run-to-run RDM reliabilities in Notebook 03
are near or below zero.

Notebook 20 consequently reports:

- exact target-label permutation tests rather than pretending the 15 pairs are
  independent observations;
- leave-one-target-out RDM correlations;
- group noise ceilings, which show how well an estimated subject RDM can agree
  with the group given measurement variability;
- within-condition target identification, which checks that measurable target
  information exists before interpreting transfer;
- direct vision-to-imagery identification;
- cue-matched and cue-mismatched controls.

A noise ceiling is itself an estimate, not a guarantee. If within-condition
reliability is poor, a low transfer correlation is ambiguous: the shared
structure may be absent, or present but not measurable with these data.

## 7. Revised evidence hierarchy

The analysis no longer depends on `subj01` for hypothesis selection. Notebook
04 includes all eight participants and treats the following hierarchy as fixed
before inspecting its results:

1. **Measurement validity:** identify targets within vision and within imagery
   using independent repetitions/runs.
2. **Direct transfer:** train vision target centroids and identify imagery
   trials, with cue-confound controls.
3. **Secondary RSA:** quantify whether target-pair orderings agree, with exact
   permutations, leave-one-target-out sensitivity, and group noise ceilings.
4. **Feature explanation:** compare HOG/CLIP or other models only if steps 1–3
   reveal a sufficiently reliable neural effect worth explaining.

This is not a rejection of RSA. It narrows RSA to the question it actually
answers and combines it with tests that are closer to the desired scientific
claim.

## 8. Reading outcomes without overclaiming

| Result pattern | Defensible interpretation |
|---|---|
| Within-imagery identification is at chance | The current measurement cannot establish stable target information; transfer analyses are inconclusive or exploratory |
| Within-imagery succeeds, vision-to-imagery fails | Imagery contains target information, but the simple vision-trained code does not transfer |
| Vision-to-imagery succeeds, including cue-mismatch trials | Direct evidence that vision-derived target patterns generalize beyond cue identity |
| RSA is positive but direct identification fails | Some aggregate pair ordering may be shared, but practical target-level transfer is not established |
| Identification and RSA both succeed | Converging evidence for transfer; RDMs describe the relational organization of that transferable information |

## References

- Kriegeskorte, Mur, and Bandettini (2008), [Representational similarity
  analysis—connecting the branches of systems neuroscience](https://doi.org/10.3389/neuro.06.004.2008).
- Nili et al. (2014), [A toolbox for representational similarity
  analysis](https://doi.org/10.1371/journal.pcbi.1003553).
- Schütt et al. (2023), [Statistical inference on representational
  geometries](https://doi.org/10.7554/eLife.82566).
