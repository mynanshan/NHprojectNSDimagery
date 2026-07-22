# NHprojectNSDimagery

- [Project plan](docs/nsd_imagery_project_notes.md)
- [Experiment recap: runs, trials, designs, and betas](docs/experiment_recap.md)
- [Original paper methodology in formulas](docs/original_paper_methodology.md)
- [Paper reproduction, released checkpoints, and ML next steps](docs/paper_reproduction_and_ml_next_steps.md)
- [Train a core-NSD perception encoder and test imagery transfer](docs/brain_encoder_baseline.md)
- [Notebook 02/03 results review](docs/notebook_02_03_results_review.md)
- [What RDM correlation answers and the revised evidence hierarchy](docs/rsa_scope_and_revised_analysis.md)
- [Minimal NSD-Imagery download guide](docs/data_download.md)
- [Interactive data-orientation notebook](notebooks/01_data_orientation.ipynb)
- [Event alignment and first neural RDM notebook](notebooks/02_event_alignment_neural_rdm.ipynb)
- [Held-out transfer and feature RSA notebook](notebooks/03_group_transfer_feature_rsa.ipynb)
- [Measurement-first validation notebook](notebooks/04_measurement_first_validation.ipynb)
- [Paper-aligned brain-correlation notebook](notebooks/05_paper_brain_correlation.ipynb)
- [Core-NSD brain-encoder workflow](notebooks/06_core_nsd_brain_encoder.ipynb)

Start with one subject and no downloads:

```bash
bash scripts/download_nsdimagery_mvp.sh --subjects 01 --estimate
```

After downloading, validate the files without loading the full beta arrays:

```bash
python scripts/check_nsdimagery_data.py data/nsd
```

For an isolated Python environment and Jupyter kernel:

```bash
conda env create -f environment.yml
conda activate nsdimagery
python -m ipykernel install --user --name nsdimagery --display-name "Python (NSD-Imagery)"
jupyter lab
```

To update an existing `nsdimagery` environment before notebooks 03 or 04:

```bash
conda env update -n nsdimagery -f environment.yml
```
