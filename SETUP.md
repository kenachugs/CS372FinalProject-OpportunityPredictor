# SETUP.md

Step-by-step installation and usage instructions for graders and reviewers.

## System requirements

- **Python 3.10 or newer** (tested on 3.11 and 3.13)
- **~500 MB disk space** for data, models, and dependencies
- **No GPU required.** PyTorch will use CUDA if available but the MLP trains in ~2 minutes on CPU
- **No external API keys or downloads required.** The pipeline uses a calibrated synthetic dataset that runs fully offline

## 1. Clone and install

```bash
git clone <your-repo-url>
cd opportunity-predictor

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate opportunity-predictor
```

## 2. Train the models

Run the full pipeline (~5–8 minutes on a typical laptop):

```bash
python src/train.py
```

Or, for a fast smoke test on a 10,000-tract sample (~2 minutes):

```bash
python src/train.py --fast --sample-size 10000
```

Either command will:

1. Generate the calibrated synthetic dataset (~72,000 tracts).
2. Fit the preprocessing pipeline (winsorization + iterative imputation + feature engineering + standardization).
3. Train baseline, linear, Lasso, Ridge, Random Forest, Gradient Boosting, custom PyTorch MLP, and stacking-ensemble models.
4. Evaluate on the held-out test set and save a comparison table to `data/processed/`.
5. Pickle the fitted models and preprocessing artifacts to `models/`.
6. Save disparity / fairness reports to `data/processed/disparity_*.csv`.
7. Save the Lasso coefficient table to `data/processed/lasso_coefficient_table.csv` for the web app's interpretability panel.

## 3. Launch the Streamlit web app

```bash
streamlit run app/app.py
```

This opens the web app at `http://localhost:8501`. Enter a community profile in the sidebar and see the predicted outcome, a structural counterfactual, the interpretable Lasso coefficient table, and the fairness audit.

**Prerequisite:** step 2 must complete first so that fitted models exist in `models/`.

## 4. About the dataset

This project uses a calibrated synthetic dataset (~72K tracts, runs in ~3 min, no download needed). The synthetic generator in `src/data_loader.py` is parameterized so the marginal distributions and pairwise correlations qualitatively reflect the patterns reported in Chetty et al. (2020). The real Opportunity Atlas data is split across a Census Bureau outcomes release and a Harvard Dataverse replication archive — neither is auto-downloadable as a single CSV. See `docs/DATASET_CITATIONS.md` for the access points if you want to adapt this pipeline to real data.

## 5. Run the analysis notebooks

The four analysis notebooks walk through the project step by step:

```bash
jupyter notebook notebooks/
```

- `01_exploratory_data_analysis.ipynb` — data distributions and correlations
- `02_preprocessing_impact.ipynb` — evidence of impact for preprocessing decisions
- `03_model_comparison_and_ablation.ipynb` — full model comparison, hyperparameter tuning, ablation study, training curves
- `04_fairness_and_interpretability.ipynb` — SHAP, partial dependence, sparse-Lasso interpretability, fairness audit, error analysis

Each notebook is designed to run top-to-bottom and saves its figures to `docs/` and its tables to `data/processed/`.

## 6. Re-run just the hyperparameter tuning or ablation

```bash
python src/hyperparameter_tuning.py --sample-size 10000
python src/ablation.py --sample-size 10000
```

Outputs saved to `data/processed/`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

The TabularMLP requires PyTorch:

```bash
pip install torch
```

The rest of the pipeline (sklearn models, stacking ensemble, web app, fairness/interpretability) works without PyTorch. Pass `--no-mlp` to `train.py` to skip it.

### `ModuleNotFoundError: No module named 'shap'`

SHAP analysis is optional:

```bash
pip install shap
```

The interpretability notebook will fall back to Lasso + tree importance if `shap` is unavailable.

### Iterative imputation is slow

On the full 72K-row dataset, iterative imputation can take ~5 minutes. For faster testing, use `--sample-size 10000` or switch to `imputation_strategy="median"` in `src/preprocessing.py`.

### Streamlit shows "No fitted model found"

Run `python src/train.py` first to create the pickled artifacts in `models/`.

### Random Forest out-of-memory on the full dataset

Reduce `n_estimators` or `max_depth` in `src/train.py`'s `build_models()`, or use `--sample-size`.
