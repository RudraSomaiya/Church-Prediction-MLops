# Customer Churn Prediction - MLOps Pipeline

## Project Overview

Customer churn is when a subscriber cancels or stops using a service.

The pipeline is built to MLOps standards: all data is version-controlled with DVC, all experiments are tracked with MLflow, the code is tested with Pytest, and continuous integration via GitHub Actions runs the test suite on every push.

**Primary metrics: ROC-AUC for training, F1 for threshold tuning.** 

**Assignment:** MLOps, STDE 301, August 2026 Mid Term.

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

The raw dataset is the Kaggle "Customer Churn Dataset"
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

The two CSV files (`customer_churn_train.csv` and `customer_churn_test.csv`) live in `data/raw/` and are tracked by DVC, not by Git.

**If you have DVC remote access (standard clone):**

```bash
dvc pull
```

## DVC Configuration

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

---

## Viewing MLflow Results

MLflow tracks every model run, including parameters, metrics, and artefacts (confusion matrix, ROC curve, threshold sweep chart).

Start the MLflow UI:

```bash
mlflow ui
```

Then open your browser at: http://localhost:5000

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

DVC data is not pulled in CI because the tests use synthetic data and do not need the real dataset.

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
