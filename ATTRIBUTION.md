# ATTRIBUTION.md

Full attribution of AI tools, external libraries, datasets, and other resources used in this project.

## AI Tool Usage — Substantive Account

I used **Claude (Anthropic)** as a coding collaborator throughout this project. The following breakdown describes, per module, what was AI-drafted, what I modified, and what I had to debug or rework myself.

### `src/data_loader.py`

- **AI-assisted:** Initial scaffolding of the `generate_synthetic_dataset` function, including the beta/lognormal sampling for demographic and income variables and the `_simulate_outcome` linear/logit generator.
- **My modifications:**
  - **Calibration.** Claude's first draft produced an earnings-rank distribution centered near 75 (wrong — real mean for p25 kids is ~43) and incarceration rates near 0.002 (wrong — real mean is ~0.04). I rewrote both generators with published Chetty et al. (2020) Table II coefficients and re-calibrated base-rate constants until the synthetic marginals matched the real data. Code and rationale are in the `_simulate_outcome` docstring.
  - **Missingness injection.** The first draft did not include any missing values; I added targeted missingness (3–8% per column, MCAR) so that the downstream preprocessing pipeline actually has work to do.

### `src/preprocessing.py`

- **AI-assisted:** Dataclass skeletons for `PreprocessingConfig` and `PreprocessingArtifacts`, and the basic `winsorize_fit_transform` / `winsorize_transform` pair.
- **My modifications:**
  - **Feature engineering choices.** The engineered features — `concentrated_disadvantage` (Sampson 1997 composite), `poor_x_black` / `poor_x_hisp` interaction terms, `inequality_index` — are my design decisions motivated by the sociology literature. Claude did not suggest these; I told it which features I wanted and it helped with the NumPy implementation.
  - **Imputation strategy selection.** I chose `IterativeImputer` because the missingness in `gsmn_math_pcst` is correlated with urbanicity and poverty (conditional imputation should do better than mean). I verified this empirically in `notebooks/02_preprocessing_impact.ipynb`.
  - **Debugging.** Claude's first draft of `winsorize_transform` accidentally re-computed quantile bounds on the test data, which would leak test-set information. I caught this on review and fixed it to use the stored `artifacts.winsor_bounds` from training.
  - **Serving-time bug I caught and fixed.** The first version of `add_engineered_features` computed the z-score parameters for the `concentrated_disadvantage` index from whatever batch was passed in. This silently worked for training (where the batch is large) and for the test set (also large), but produced NaN at single-row serving time in the Streamlit app, since `series.std()` on a length-1 series is undefined. I refactored to fit the z-score parameters at training time, store them in `PreprocessingArtifacts.z_params`, and reuse them at transform time. Caught only because I tested the app with a single-row input — exactly the test that the standard train/test workflow misses.

### `src/models.py`

- **AI-assisted:** Skeletons for the scikit-learn wrapper classes (`LinearBaseline`, `LassoModel`, `RidgeModel`, `RandomForestModel`, `GradientBoostingModel`) and a basic MLP training loop skeleton.
- **My modifications:**
  - **Residual block architecture.** The `ResidualBlock` → stacked-blocks → head design is my architectural choice, modeled on the section on residual networks from Bishop's *Deep Learning* textbook covered in class. Claude initially suggested a plain 3-layer MLP with no skip connections; I added the residual structure after observing that the plain MLP plateaued at roughly linear-model performance.
  - **Regularization stack.** Dropout, weight decay, gradient clipping, cosine LR schedule, and early stopping were all my explicit additions (not present in the first draft) to exercise the rubric regularization item meaningfully.
  - **Stacking ensemble.** The `StackingEnsemble` class is mostly AI-drafted, but I rewrote the `fit` method to use proper 5-fold out-of-fold predictions (Claude's first draft used training-set predictions, which would have trivially memorized the training targets in the meta-learner). This bug would have been a disaster silently.
  - **Optional-torch pattern.** Wrapping the torch imports in `try/except` so the sklearn models are usable without torch is my refactor for grader convenience.

### `src/evaluation.py`

- **AI-assisted:** Skeletons for the `evaluate_model` and `worst_errors` functions.
- **My modifications:** The `demographic_disparity_report` function and the four subgroup bucketings (by poverty, by minority share, by single-parent share, by school quality) are entirely my design — they are the heart of the project's fairness audit. Claude helped with the pandas `pd.cut` mechanics but the analysis design is mine.
- **Bug I caught:** Claude's original `subgroup_metrics` had a `.values` call on an already-numpy mask that threw `AttributeError`. Fixed to `np.asarray(cat == label)`.

### `src/interpretability.py`

- **AI-assisted:** Skeleton for the SHAP integration.
- **My modifications:** The `sparse_linear_report` function and the framing around Rudin (2019) — reporting the Lasso as the *interpretable model of record* alongside the GBM — is my design decision, inspired by the course's interpretability assignment.

### `src/hyperparameter_tuning.py`

- **AI-assisted:** Boilerplate argument parsing, logging setup, and scaffolding.
- **My modifications:** The ablation axes (feature engineering × winsorization × imputation × demographics) are my experimental design choices. The hyperparameter grid was my choice (Claude initially suggested a 125-config grid; I cut it to 27 for reasonable runtime while still demonstrating the tuning process meaningfully).

### `app/app.py`

- **AI-assisted:** Basic Streamlit layout, sliders, metric displays.
- **My modifications:**
  - The **framing language** ("this predicts community outcomes, not individual fates") and the **structural counterfactual** plot are entirely mine — these are the centerpiece of the responsible-ML framing.
  - The Lasso interpretability panel and the fairness-audit tables embedded in the app are my additions.

### Notebooks

- All four notebooks were authored by me with Claude helping on matplotlib/seaborn boilerplate and markdown formatting. The analysis narrative, the experimental questions posed in each cell, and the takeaways sections are my written work.


---

## External Code, Libraries, and Datasets

### Datasets

- **The Opportunity Atlas** — Chetty, R., Friedman, J. N., Hendren, N., Jones, M. R., & Porter, S. R. (2020). "The Opportunity Atlas: Mapping the Childhood Roots of Social Mobility." *NBER Working Paper No. 25147*. Public release: https://opportunityinsights.org/data/. Licensed for research and educational use.
- **American Community Survey 2015–2019 5-Year Estimates** — US Census Bureau. Public domain.
- The synthetic dataset generator in `src/data_loader.py` is calibrated against published Opportunity Atlas Table II coefficients; it does not reproduce the original data.

### Python libraries (see `requirements.txt` for exact versions)

| Library      | Purpose                                               | License    |
|--------------|-------------------------------------------------------|------------|
| numpy        | Numerical arrays                                      | BSD-3      |
| pandas       | Dataframes                                            | BSD-3      |
| scikit-learn | Preprocessing, Lasso/Ridge/RF/GBM models, metrics     | BSD-3      |
| PyTorch      | Custom TabularMLP                                     | BSD-3      |
| shap         | SHAP values in the interpretability notebook          | MIT        |
| matplotlib   | Plotting                                              | matplotlib |
| seaborn      | Statistical plotting                                  | BSD-3      |
| streamlit    | Web app                                               | Apache-2   |
| jupyter      | Notebook runtime                                      | BSD-3      |

### Intellectual antecedents

- The **concentrated-disadvantage index** engineered feature is my implementation of the composite proposed in Sampson, Raudenbush & Earls (1997), "Neighborhoods and violent crime: A multilevel study of collective efficacy." *Science* 277: 918–924.
- The **interpretability-over-explainability** framing follows Rudin, C. (2019). "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead." *Nature Machine Intelligence* 1: 206–215.
- The **stacking ensemble** pattern follows Wolpert (1992), "Stacked generalization." *Neural Networks* 5(2): 241–259.

---

## What I did not use

- I did not use any other AI tool besides Claude.
- I did not copy code from Kaggle notebooks, Stack Overflow answers, or blog posts. Where a standard sklearn / PyTorch pattern is used, it is the pattern documented in the official library documentation.
- No pre-trained models were used; all models are trained from scratch on the dataset described above.
