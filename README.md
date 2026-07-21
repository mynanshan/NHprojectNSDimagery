# NHprojectNSDimagery

- [Project plan](docs/nsd_imagery_project_notes.md)
- [Minimal NSD-Imagery download guide](docs/data_download.md)
- [Interactive data-orientation notebook](notebooks/01_data_orientation.ipynb)
- [Event alignment and first neural RDM notebook](notebooks/02_event_alignment_neural_rdm.ipynb)

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
