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

## Results

Local reference run with `--samples 5000 --components 30` (seed 42; 3,200 / 800 / 1,000 train / validation / test): 30 components explain 51.5% of pixel variance; held-out test accuracy **0.872**, macro-F1 **0.870**. The default (`10,000` samples, `50` components) has not been reported yet.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
