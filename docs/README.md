# Documentation guide

This page separates the shortest scientific reading path from implementation
guides and historical notes. Each result document states which notebook
produced or interprets it.

## Start here

| Read | Document | Related notebook(s) | Status |
|---|---|---|---|
| 1 | [Experiment recap](guides/experiment_recap.md) | 01, 02 | Dataset orientation |
| 2 | [RSA results](results/rsa_results.md) | 02, 03, 21 | Main regional result; feature and full-18 follow-ups are exploratory |
| 3 | [Cross-region RSA](results/cross_region_rsa.md) | 04 | Main exploratory cross-region result |
| 4 | [Brain-encoding results](results/brain_encoding_results.md) | 05, 22 | Main encoder result plus secondary extensions |

This four-document path is enough to understand the current presentation.
Reliability analyses, feature-model null results, and unsuccessful nonlinear
extensions remain documented, but they are not foregrounded.

## Guides and methods

| Document | Use it when... | Related notebook(s) |
|---|---|---|
| [Minimal data download](guides/data_download.md) | downloading the prepared NSD-Imagery data safely | 01–04, 20–23 |
| [Experiment recap](guides/experiment_recap.md) | learning the tasks, runs, targets, trial labels, and beta indexing | 01, 02 |
| [Original-paper methodology](guides/original_paper_methodology.md) | distinguishing the CVPR reconstruction benchmark from this project's RSA | 02, 03, 23 |
| [Brain-encoding workflow](guides/brain_encoding_workflow.md) | preparing core NSD, extracting DINOv2 features, and fitting the ridge encoder | 05, 22 |

## Results

| Document | Scope | Related notebook(s) |
|---|---|---|
| [RSA results](results/rsa_results.md) | trial alignment, regional RDMs, held-out group RSA, feature RSA, and full-18 sensitivity | 02, 03, 21 |
| [Cross-region RSA](results/cross_region_rsa.md) | visual-to-parietal RDM comparison and Set-C-trained alignment decoder | 04 |
| [Brain-encoding results](results/brain_encoding_results.md) | core-NSD validation, frozen imagery transfer, nonlinear extension, and direct fits | 05, 22 |

## Development notes

These files preserve decisions and possible future work. They are useful for
auditing the project's evolution, but are not required reading.

| Document | Purpose | Related notebook(s) |
|---|---|---|
| [Project notes](notes/project_notes.md) | original scope, hypotheses, and workshop plan | all |
| [RSA scope and evidence hierarchy](notes/rsa_scope_and_evidence.md) | what RDM correlation answers and why additional checks were added | 03, 20 |
| [Paper reproduction and ML next steps](notes/paper_reproduction_and_ml_next_steps.md) | reconstruction checkpoints, exact Table 1 scoring, and possible model extensions | 23 |

## Notebook numbering

- `01–05`: the current presentation and recommended reading path.
- `20–23`: robustness analyses, null/negative branches, and reproduction work.
- `outputs/` keeps historical directory names so existing result provenance is
  preserved even when a notebook receives a new reader-facing number.
