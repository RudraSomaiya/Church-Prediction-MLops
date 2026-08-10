"""
preprocess.py
-------------
Data preprocessing for the customer churn pipeline.

Steps performed:
1. Load raw train and test CSVs.
2. Drop the CustomerID column (it is an arbitrary identifier, not a feature).
3. Drop the single null row present in each file.
4. Encode categorical features with LabelEncoder (Gender, Subscription Type,
   Contract Length).  LabelEncoder is used rather than one-hot encoding because
   tree-based models (Random Forest, Gradient Boosting) handle ordinal integer
   codes well, and it keeps the feature space compact for a dataset this size.
5. Scale numeric features with StandardScaler so Logistic Regression converges
   properly.  Tree-based models are invariant to monotone transformations but
   the scaler does not hurt them, so a single preprocessed file works for all
   models.
6. Perform a stratified train/validation split on the training data.
7. Apply SMOTE to the training split only (never to validation or test).
8. Save processed artefacts to data/processed/.

All paths and parameters are read from params.yaml so the step is reproducible
by changing a single config file and running dvc repro.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Core preprocessing functions (also imported by tests)
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = ["Gender", "Subscription Type", "Contract Length"]
NUMERIC_COLS = [
    "Age", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Total Spend", "Last Interaction",
]


def drop_id_and_nulls(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Drop the identifier column and rows with any null values."""
    df = df.drop(columns=[id_col], errors="ignore")
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None):
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df       : DataFrame to encode in-place (copy is made internally).
    encoders : dict of {col: fitted LabelEncoder}.  If None, new encoders
               are fitted on df and returned.  Pass fitted encoders from
               the training set when transforming validation / test data.

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    fit_new = encoders is None
    if fit_new:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = encoders.get(col, LabelEncoder())
        if fit_new:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            # Handle unseen labels gracefully by mapping them to -1
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
    return df, encoders


def scale_numerics(df: pd.DataFrame, scaler: StandardScaler | None = None):
    """
    Standardise numeric columns (zero mean, unit variance).

    Parameters
    ----------
    df     : DataFrame.
    scaler : Fitted StandardScaler, or None to fit a new one.

    Returns
    -------
    (scaled_df, scaler)
    """
    df = df.copy()
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


def apply_smote(X: np.ndarray, y: np.ndarray, params: dict):
    """Apply SMOTE oversampling to the minority class."""
    sm = SMOTE(
        random_state=params["smote"]["random_state"],
        k_neighbors=params["smote"]["k_neighbors"],
    )
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(params: dict | None = None) -> None:
    if params is None:
        params = load_params()

    data_params = params["data"]
    raw_train = ROOT / data_params["raw_train"]
    raw_test  = ROOT / data_params["raw_test"]
    out_dir   = ROOT / data_params["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_col = data_params["target_column"]
    id_col     = data_params["id_column"]
    test_size  = data_params["test_size"]
    random_state = data_params["random_state"]

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("Loading raw data...")
    train_df = pd.read_csv(raw_train)
    test_df  = pd.read_csv(raw_test)
    print(f"  Train shape (raw): {train_df.shape}")
    print(f"  Test  shape (raw): {test_df.shape}")

    # ------------------------------------------------------------------
    # 2. Drop ID and nulls
    # ------------------------------------------------------------------
    train_df = drop_id_and_nulls(train_df, id_col)
    test_df  = drop_id_and_nulls(test_df,  id_col)
    print(f"  Train shape (after drop): {train_df.shape}")
    print(f"  Test  shape (after drop): {test_df.shape}")

    # ------------------------------------------------------------------
    # 3. Encode categoricals (fit on train, apply to test)
    # ------------------------------------------------------------------
    train_df, encoders = encode_categoricals(train_df)
    test_df,  _        = encode_categoricals(test_df, encoders=encoders)

    # ------------------------------------------------------------------
    # 4. Scale numerics (fit on train, apply to test)
    # ------------------------------------------------------------------
    train_df, scaler = scale_numerics(train_df)
    test_df,  _      = scale_numerics(test_df, scaler=scaler)

    # ------------------------------------------------------------------
    # 5. Stratified train / validation split
    # ------------------------------------------------------------------
    X_train_full = train_df.drop(columns=[target_col])
    y_train_full = train_df[target_col].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full,
    )
    print(f"  Train split: {X_train.shape}, Val split: {X_val.shape}")
    print(f"  Train churn rate: {y_train.mean():.3f}  "
          f"Val churn rate: {y_val.mean():.3f}")

    # ------------------------------------------------------------------
    # 6. SMOTE on training split only
    # ------------------------------------------------------------------
    print("Applying SMOTE to training split...")
    X_train_res, y_train_res = apply_smote(X_train.values, y_train.values, params)
    print(f"  After SMOTE - shape: {X_train_res.shape}, "
          f"churn rate: {y_train_res.mean():.3f}")

    # ------------------------------------------------------------------
    # 7. Save artefacts
    # ------------------------------------------------------------------
    feature_cols = X_train_full.columns.tolist()

    np.save(out_dir / "X_train.npy", X_train_res)
    np.save(out_dir / "y_train.npy", y_train_res)
    np.save(out_dir / "X_val.npy",   X_val.values)
    np.save(out_dir / "y_val.npy",   y_val.values)

    X_test  = test_df.drop(columns=[target_col], errors="ignore")
    y_test  = test_df[target_col].astype(int) if target_col in test_df.columns else None

    np.save(out_dir / "X_test.npy", X_test.values)
    if y_test is not None:
        np.save(out_dir / "y_test.npy", y_test.values)

    # Save feature names so downstream scripts can reconstruct DataFrames
    pd.Series(feature_cols).to_csv(out_dir / "feature_names.csv", index=False)

    # Save fitted transformer artefacts for reproducibility
    joblib.dump(encoders, out_dir / "label_encoders.pkl")
    joblib.dump(scaler,   out_dir / "scaler.pkl")

    print("Preprocessing complete.  Artefacts saved to:", out_dir)


if __name__ == "__main__":
    run()
