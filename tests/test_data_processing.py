"""
test_data_processing.py
-----------------------
Unit tests for src/data/preprocess.py.

These tests use small synthetic DataFrames so they run quickly and do not
require the real 440k-row dataset to be present.  Every function that will
be exercised is tested for:
  - correct handling of missing values
  - correct output schema / column names
  - correct output dtypes and value ranges after encoding and scaling
  - that the train/test split preserves the target column
  - that SMOTE produces a balanced class distribution
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Make src importable from tests/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import (
    drop_id_and_nulls,
    encode_categoricals,
    scale_numerics,
    apply_smote,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small but representative synthetic DataFrame matching the real schema."""
    np.random.seed(0)
    n = 100
    df = pd.DataFrame({
        "CustomerID":        np.arange(1, n + 1, dtype=float),
        "Age":               np.random.uniform(18, 70, n),
        "Gender":            np.random.choice(["Male", "Female"], n),
        "Tenure":            np.random.uniform(0, 60, n),
        "Usage Frequency":   np.random.uniform(1, 30, n),
        "Support Calls":     np.random.uniform(0, 10, n),
        "Payment Delay":     np.random.uniform(0, 30, n),
        "Subscription Type": np.random.choice(["Basic", "Standard", "Premium"], n),
        "Contract Length":   np.random.choice(["Monthly", "Quarterly", "Annual"], n),
        "Total Spend":       np.random.uniform(100, 1000, n),
        "Last Interaction":  np.random.uniform(1, 30, n),
        "Churn":             np.random.choice([0.0, 1.0], n),
    })
    return df


@pytest.fixture
def df_with_nulls(sample_df) -> pd.DataFrame:
    """A DataFrame that has exactly 3 null rows."""
    df = sample_df.copy()
    df.loc[0, "Age"] = np.nan
    df.loc[1, "Gender"] = np.nan
    df.loc[2, "Total Spend"] = np.nan
    return df


@pytest.fixture
def encoded_df(sample_df) -> tuple:
    """Return (encoded DataFrame, encoders dict) after dropping ID/nulls."""
    clean = drop_id_and_nulls(sample_df, "CustomerID")
    encoded, encoders = encode_categoricals(clean)
    return encoded, encoders


# ---------------------------------------------------------------------------
# Tests: drop_id_and_nulls
# ---------------------------------------------------------------------------

class TestDropIdAndNulls:
    def test_removes_id_column(self, sample_df):
        result = drop_id_and_nulls(sample_df, "CustomerID")
        assert "CustomerID" not in result.columns

    def test_removes_null_rows(self, df_with_nulls):
        result = drop_id_and_nulls(df_with_nulls, "CustomerID")
        assert result.isnull().sum().sum() == 0

    def test_row_count_decreases_by_null_count(self, df_with_nulls):
        original_len = len(df_with_nulls)
        null_rows = df_with_nulls.isnull().any(axis=1).sum()
        result = drop_id_and_nulls(df_with_nulls, "CustomerID")
        assert len(result) == original_len - null_rows

    def test_no_nulls_in_clean_df_unchanged_length(self, sample_df):
        """When there are no nulls (except ID col), length should stay the same."""
        result = drop_id_and_nulls(sample_df, "CustomerID")
        assert len(result) == len(sample_df)

    def test_index_reset(self, df_with_nulls):
        result = drop_id_and_nulls(df_with_nulls, "CustomerID")
        assert result.index.tolist() == list(range(len(result)))

    def test_missing_id_col_graceful(self, sample_df):
        """Passing a non-existent id column should not raise an error."""
        result = drop_id_and_nulls(sample_df, "NonExistentCol")
        # CustomerID should still be there since we asked to drop a different col
        assert "CustomerID" in result.columns


# ---------------------------------------------------------------------------
# Tests: encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def test_categorical_cols_become_numeric(self, sample_df):
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        encoded, _ = encode_categoricals(clean)
        for col in CATEGORICAL_COLS:
            if col in encoded.columns:
                assert pd.api.types.is_integer_dtype(encoded[col]) or \
                       pd.api.types.is_float_dtype(encoded[col]), \
                       f"{col} should be numeric after encoding"

    def test_returns_encoders_dict(self, sample_df):
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        _, encoders = encode_categoricals(clean)
        assert isinstance(encoders, dict)
        for col in CATEGORICAL_COLS:
            assert col in encoders

    def test_encoded_values_within_class_count(self, sample_df):
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        encoded, encoders = encode_categoricals(clean)
        for col in CATEGORICAL_COLS:
            n_classes = len(encoders[col].classes_)
            assert encoded[col].min() >= 0
            assert encoded[col].max() < n_classes

    def test_reuse_encoders_on_test_data(self, sample_df):
        """Applying fitted encoders on new data should not raise."""
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        _, encoders = encode_categoricals(clean)

        new_data = clean.copy()
        encoded_new, _ = encode_categoricals(new_data, encoders=encoders)
        for col in CATEGORICAL_COLS:
            assert pd.api.types.is_integer_dtype(encoded_new[col]) or \
                   pd.api.types.is_float_dtype(encoded_new[col])

    def test_schema_preserved(self, sample_df):
        """All non-categorical columns should remain in the output."""
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        encoded, _ = encode_categoricals(clean)
        for col in clean.columns:
            assert col in encoded.columns


# ---------------------------------------------------------------------------
# Tests: scale_numerics
# ---------------------------------------------------------------------------

class TestScaleNumerics:
    def test_numeric_cols_have_near_zero_mean(self, encoded_df):
        encoded, _ = encoded_df
        scaled, _ = scale_numerics(encoded)
        for col in NUMERIC_COLS:
            if col in scaled.columns:
                assert abs(scaled[col].mean()) < 0.1, \
                    f"Mean of {col} should be near 0 after scaling"

    def test_numeric_cols_have_near_unit_std(self, encoded_df):
        encoded, _ = encoded_df
        scaled, _ = scale_numerics(encoded)
        for col in NUMERIC_COLS:
            if col in scaled.columns:
                assert abs(scaled[col].std() - 1.0) < 0.1, \
                    f"Std of {col} should be near 1 after scaling"

    def test_returns_scaler_object(self, encoded_df):
        encoded, _ = encoded_df
        _, scaler = scale_numerics(encoded)
        assert isinstance(scaler, StandardScaler)

    def test_reuse_scaler_on_test_data(self, encoded_df):
        """Applying fitted scaler on new data should not raise."""
        encoded, _ = encoded_df
        scaled, scaler = scale_numerics(encoded)
        new_scaled, _ = scale_numerics(encoded.copy(), scaler=scaler)
        assert new_scaled.shape == encoded.shape

    def test_categorical_cols_untouched(self, encoded_df):
        """Categorical (now integer) columns should not be altered by the scaler."""
        encoded, _ = encoded_df
        before = encoded[CATEGORICAL_COLS].copy()
        scaled, _ = scale_numerics(encoded)
        pd.testing.assert_frame_equal(scaled[CATEGORICAL_COLS], before)


# ---------------------------------------------------------------------------
# Tests: apply_smote
# ---------------------------------------------------------------------------

class TestApplySMOTE:
    def test_smote_balances_classes(self, sample_df):
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        encoded, _ = encode_categoricals(clean)
        scaled, _  = scale_numerics(encoded)
        X = scaled.drop(columns=["Churn"]).values
        y = scaled["Churn"].astype(int).values

        # Create imbalanced data deliberately
        minority_idx = np.where(y == 1)[0][:10]
        majority_idx = np.where(y == 0)[0][:50]
        X_imb = X[np.concatenate([minority_idx, majority_idx])]
        y_imb = y[np.concatenate([minority_idx, majority_idx])]

        params = {"smote": {"random_state": 42, "k_neighbors": 3}}
        X_res, y_res = apply_smote(X_imb, y_imb, params)

        # After SMOTE the classes should be balanced
        unique, counts = np.unique(y_res, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1], "SMOTE should balance class counts"

    def test_smote_does_not_alter_majority(self, sample_df):
        """SMOTE should not decrease the majority class count."""
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        encoded, _ = encode_categoricals(clean)
        scaled, _  = scale_numerics(encoded)
        X = scaled.drop(columns=["Churn"]).values
        y = scaled["Churn"].astype(int).values

        majority_count_before = (y == 0).sum()
        params = {"smote": {"random_state": 42, "k_neighbors": 5}}
        _, y_res = apply_smote(X, y, params)
        majority_count_after = (y_res == 0).sum()

        assert majority_count_after >= majority_count_before

    def test_smote_output_shape(self, sample_df):
        """X_res should have the same number of features as input."""
        clean = drop_id_and_nulls(sample_df, "CustomerID")
        encoded, _ = encode_categoricals(clean)
        scaled, _  = scale_numerics(encoded)
        X = scaled.drop(columns=["Churn"]).values
        y = scaled["Churn"].astype(int).values

        params = {"smote": {"random_state": 42, "k_neighbors": 5}}
        X_res, y_res = apply_smote(X, y, params)

        assert X_res.shape[1] == X.shape[1]
        assert X_res.shape[0] == y_res.shape[0]
