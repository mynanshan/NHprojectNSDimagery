cd ~/NHprojectNSDimagery

conda env create -f environment.yml
conda activate nsdimagery

python -m ipykernel install \
  --user \
  --name nsdimagery \
  --display-name "Python (NSD-Imagery)"