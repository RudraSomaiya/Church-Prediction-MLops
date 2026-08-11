# Customer Churn Prediction - MLOps Pipeline

## Project Overview

Customer churn is when a subscriber cancels or stops using a service. Predicting churn before it happens lets a business intervene early with a retention offer, which is far cheaper than acquiring a replacement customer. This project builds a production-style machine learning pipeline that predicts whether a customer will churn, using a dataset of 440,000+ customer records.

The pipeline is built to MLOps standards: all data is version-controlled with DVC, all experiments are tracked with MLflow, the code is tested with Pytest, and continuous integration via GitHub Actions runs the test suite on every push.

**Primary metrics: ROC-AUC for training, F1 for threshold tuning.** While recall is important for churn, optimizing purely for recall leads to degenerate models (predicting churn for everyone). The pipeline trains using ROC-AUC to ensure high overall discriminatory power, and tunes the decision threshold using the F1 score to perfectly balance precision and recall on the imbalanced dataset.

**Assignment:** Vijaybhoomi University, MLOps, STDE 301, August 2026 Mid Term.

---

## Repository Structure

```
project-root/
  data/
    raw/              - Raw CSVs tracked by DVC
    processed/        - Preprocessed arrays, models, and reports
  src/
    data/
      download_data.py  - Data validation / ingestion script
      preprocess.py     - Full preprocessing pipeline
    models/
      train.py          - Training with GridSearchCV, MLflow logging
      evaluate.py       - Threshold tuning, plots, final metrics
  tests/
    test_data_processing.py
    test_model.py
  .github/
    workflows/
      ci.yml            - GitHub Actions CI
  dvc.yaml              - DVC pipeline stages
  params.yaml           - All tunable parameters
  requirements.txt      - Pinned dependencies
  README.md             - This file
```

---

## Environment Setup

**Python version:** 3.11 (managed automatically by uv)

**Step 1: Install uv** (if not already installed)

```bash
pip install uv
```

**Step 2: Clone the repository**

```bash
git clone <your-repo-url>
cd MidTerm
```

**Step 3: Create the virtual environment and install dependencies**

```bash
uv venv .venv --python 3.11
```

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
uv pip install -r requirements.txt
```

---

## Dataset Setup

The raw dataset is the Kaggle "Customer Churn Dataset" by Muhammad Shahid Azeem:
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

The two CSV files (`customer_churn_train.csv` and `customer_churn_test.csv`) live in `data/raw/` and are tracked by DVC, not by Git.

**If you have DVC remote access (standard clone):**

```bash
dvc pull
```

**If you need to download the data manually:**

Place both CSV files in `data/raw/` with the names above. Then run the validation script:

```bash
python src/data/download_data.py
```

**If you want to re-download from Kaggle via the API:**

Set your credentials as environment variables (do not hardcode them):

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Then uncomment the Kaggle API section in `src/data/download_data.py` and run it.
The `kaggle.json` file itself must never be committed (it is in `.gitignore`).

---

## DVC Configuration

This project uses a local DVC remote stored in a `dvc_storage/` directory one level above the project root (so it is outside the Git repository). This is the default for local development.

**To set up the remote on a fresh clone:**

```bash
dvc remote add -d local_remote ../dvc_storage
dvc remote modify local_remote url ../dvc_storage
```

**To push data to the remote after changes:**

```bash
dvc push
```

**To pull data from the remote:**

```bash
dvc pull
```

---

## Running the DVC Pipeline

The pipeline has four stages: `download -> preprocess -> train -> evaluate`.

To run all stages in order:

```bash
dvc repro
```

DVC will skip any stage whose dependencies have not changed. To force a full re-run:

```bash
dvc repro --force
```

To run a specific stage only:

```bash
dvc repro preprocess
```

All tunable parameters (test split ratio, SMOTE settings, model hyperparameters, classification threshold range) are in `params.yaml`. Edit that file and run `dvc repro` to re-run only the affected downstream stages.

---

## Viewing MLflow Results

MLflow tracks every model run, including parameters, metrics, and artefacts (confusion matrix, ROC curve, threshold sweep chart).

Start the MLflow UI:

```bash
mlflow ui
```

Then open your browser at: http://localhost:5000

The tracking directory is `mlruns/` in the project root. It is excluded from Git via `.gitignore` because it can grow large, but it is not DVC-tracked either since it is a local experiment log.

---

## Running the Tests

```bash
pytest tests/ -v
```

To see a coverage report:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

The tests use small synthetic datasets and run in seconds. They do not require the full pipeline to have been executed first.

What is tested:

- `test_data_processing.py`: preprocessing functions handle missing values correctly, categorical encoding produces integer types within valid ranges, numeric scaling produces near-zero mean and near-unit standard deviation, SMOTE balances class distribution.
- `test_model.py`: `make_estimator` returns the correct sklearn class, `compute_metrics` returns values in [0, 1], each model fits and predicts on small data, `tune_threshold` returns a valid threshold and all expected metric keys.

---

## Continuous Integration

GitHub Actions runs the full test suite automatically on every push and pull request to the `main` branch.

The workflow file is at `.github/workflows/ci.yml`. It:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs `uv`.
4. Creates a virtual environment and installs all dependencies from `requirements.txt`.
5. Runs `pytest tests/ -v --tb=short`.
6. Fails the build if any test fails.

DVC data is not pulled in CI because the tests use synthetic data and do not need the real dataset. If you add integration tests that require real data, configure a DVC remote in CI using repository secrets.

---

## Preprocessing Details

**Why these choices:**

- `CustomerID` is dropped because it is an arbitrary identifier with no predictive signal.
- The single null row per column (12 nulls total across 440k rows) is dropped rather than imputed, because the volume is negligible.
- Categorical columns (`Gender`, `Subscription Type`, `Contract Length`) are label-encoded rather than one-hot encoded. Tree-based models handle ordinal integer codes well, and it keeps the feature space compact.
- Numeric columns are standardised with `StandardScaler` so that Logistic Regression converges, and the same preprocessed arrays are reused for all three models.
- SMOTE is applied only to the training split after the stratified train/validation split. Applying it before the split would cause data leakage, because synthetic minority samples could appear in both splits.

---

## Model Comparison and Selection

Three classifiers are trained and compared:

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline, interpretable, fast |
| Random Forest | Non-linear ensemble, robust to feature scale |
| Gradient Boosting | Boosted trees, typically strongest on tabular data |

**Why ROC-AUC and F1 are the primary metrics:**

Optimizing purely for recall can lead to degenerate models that simply predict churn for every customer. To prevent overfitting and ensure the model is genuinely learning the underlying patterns, the `GridSearchCV` hyperparameter tuning phase optimizes for **ROC-AUC**. This ensures the model selects parameters that yield the highest overall ability to separate churners from non-churners.

For the final decision boundary, the evaluation stage uses the **F1 Score** as the primary metric, which provides the harmonic mean of precision and recall. A minimum precision floor (configurable in `params.yaml` under `threshold.min_precision`) is also enforced to guarantee the predictions remain operationally actionable.

**Threshold tuning:**

Rather than using the default 0.5 cutoff, the evaluate stage sweeps thresholds from 0.20 to 0.80 in steps of 0.01. The threshold that gives the highest F1 score while keeping precision at or above the minimum floor is selected. A chart of recall, precision, and F1 across all thresholds is saved to `data/processed/reports/threshold_sweep.png` and logged to MLflow.

**Class imbalance:**

SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data. The training set has roughly 57% churners vs 43% non-churners, which is a mild imbalance, but SMOTE is used to bring it to a 50/50 split to prevent the model from being biased toward the majority class. `class_weight='balanced'` is also set on Logistic Regression and Random Forest as a secondary guard.

**Model selection:**

The best model is selected by validation ROC-AUC at the default threshold during training. The final evaluation is then performed on the held-out test set with the tuned threshold. The evaluation summary, including the selected model name, threshold, and all test metrics, is saved to `data/processed/evaluation_summary.json`.

---

## Reproducibility

A fresh clone of this repository with DVC remote access will produce identical results by running:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
uv pip install -r requirements.txt
dvc pull
dvc repro
```

Reproducibility is guaranteed by:

- Pinned dependency versions in `requirements.txt`.
- All random seeds set via `params.yaml` (`data.random_state` and `smote.random_state`).
- GridSearchCV and model constructors receive `random_state=42` explicitly.
- DVC tracks data file hashes and reruns only changed stages.
- All hyperparameters and threshold values are in `params.yaml`, not in source code.
