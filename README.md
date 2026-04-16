# Heart disease risk from tabular clinical data

A transparent, reproducible Jupyter notebook that builds a binary coronary-disease
classifier on the four-site **UCI heart-disease cohort** (Cleveland, Hungary,
Switzerland, VA Long Beach — 920 patients combined).

The notebook walks end-to-end through EDA, data-quality cleaning, leak-free
preprocessing, hyperparameter-tuned comparison of three candidate models,
site-level transportability analysis, held-out test evaluation, and model
interpretation with SHAP and LIME.

> **Scope.** This is an **educational and methodological** notebook. It is **not**
> a validated clinical prediction instrument. Any real-world use would require
> institutional governance review, calibration assessment, fairness analysis,
> prospective validation, and drift monitoring.

---

## Contents

- `cardiovascular_risk_ml.ipynb` — main notebook (EDA → cleaning → preprocessing → modeling → evaluation → interpretation)
- `cardiovascular_risk_ml.html` — rendered HTML export (view without running Jupyter)
- `cardiovascular_risk_ml.pdf` — PDF export
- `data/heart_disease_uci.csv` — dataset snapshot (also auto-downloaded via `kagglehub` on first run)
- `requirements.txt` — pinned minimum versions for reproducibility

## Notebook structure

1. **Data** — loading, variable groups, descriptive statistics
2. **EDA** — missingness patterns, outcome distribution, marginal & joint distributions, categorical factor analysis
3. **Data cleaning** — sentinel `0` values in `chol` / `trestbps` mapped to `NaN` *before* the pipeline
4. **Preprocessing** — stratified split, `ColumnTransformer` with median imputation + scaling for numeric, mode imputation + one-hot encoding for categorical
5. **Modeling** — baseline 5-fold CV on Logistic Regression, Random Forest, HistGradientBoosting; `RandomizedSearchCV` tuning; recall → F1 selection
6. **Evaluation** — leave-one-site-out (LOSO) transportability + held-out test (confusion matrix, ROC)
7. **Interpretation** — permutation importance, SHAP (beeswarm + local), LIME (local surrogate explanations)
8. **Key findings** — headline numbers, transportability limits, operational implications, caveats

## Methodological highlights

- **Leak-free pipeline.** Imputation, scaling, and one-hot encoding are refit inside every CV fold and every randomized-search split.
- **Recall-first model selection.** Inner search optimizes ROC-AUC (threshold-free); final pick maximizes recall → F1 on the disease class, reflecting the clinical cost asymmetry of missed disease.
- **Site-level transportability check.** Leave-one-site-out CV exposes where pooled-test metrics overstate generalization (worst-site AUC ≈ 0.70 vs held-out test AUC ≈ 0.90).
- **Data-quality story told through interpretability.** A pre-cleaning SHAP beeswarm flagged a clinically implausible `chol` direction caused by 172 sentinel-zero rows; the cleaning step removes the artifact and the fix is visible in the post-cleaning beeswarm.
- **Multiple interpretability views.** Permutation importance (global), SHAP (global + local), and LIME (local) used as cross-checks rather than a single source of truth.

## Headline numbers (selected model: `HistGradientBoostingClassifier`)

| Metric       | Tuned 5-fold CV | Held-out test | LOSO-mean |
|--------------|:---------------:|:-------------:|:---------:|
| ROC-AUC      | 0.87            | 0.90          | 0.79      |
| Recall       | 0.84            | 0.87          | 0.78      |
| Precision    | 0.82            | 0.82          | 0.78      |
| F1           | 0.83            | 0.85          | 0.77      |

The gap between held-out test and LOSO-mean quantifies how much pooled-data
metrics overstate cross-site transportability.

## Reproducing the notebook

```bash
git clone https://github.com/DArmandoSalinas/CardioVascular_Risk.git
cd CardioVascular_Risk

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook cardiovascular_risk_ml.ipynb
```

Then **Kernel → Restart & Run All**. A fixed `RANDOM_STATE` is set at the top
of the notebook; re-runs should reproduce the numbers above to within library
version noise.

### Data source

Dataset: UCI Heart Disease (four-site consolidated version), distributed on
Kaggle as
[`redwankarimsony/heart-disease-data`](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).
A CSV snapshot is included in `data/` so the notebook runs offline; the
notebook will also re-download it via `kagglehub` if you delete the local
copy and have a Kaggle API token configured.

## Caveats and out-of-scope items

- No calibration assessment, fairness analysis, or decision-threshold optimization
- No prospective validation or drift monitoring
- Binary outcome collapses a 0–4 severity scale — severity grading is discarded
- Imputation relies on `SimpleImputer`; the three imaging fields (`ca`, `thal`, `slope`) have > 30% missingness and should be treated as a compromise rather than a principled reconstruction

## License and contact

Dataset: subject to its original UCI / Kaggle terms.
Code: MIT (add a `LICENSE` file if you want this to be binding).

**Diego Armando Salinas Lugo** — [salinas.diegoarmando03@gmail.com](mailto:salinas.diegoarmando03@gmail.com)
