# Quantix PCA on MNIST

Download MNIST from OpenML, fit a standardization + PCA + logistic-regression pipeline, and write explained-variance and classification metrics for train, validation, and held-out test splits.

## Data

MNIST is fetched from OpenML using scikit-learn and cached under `data/openml/`. The dataset and generated plots are excluded from Git. The first run needs internet access; later runs use the local cache.

Source: [OpenML MNIST](https://www.openml.org/search?type=data&sort=runs&id=554).

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The default 10,000-row sample keeps the run modest while preserving stratified 64/16/20 splits. Adjust `--samples` or `--components` to explore the compute/variance trade-off.
