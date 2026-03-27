"""
Preprocessor Module
====================
Handles missing values, outlier detection, feature scaling,
and censoring indicators for the merged HEA fatigue dataset.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

logger = logging.getLogger(__name__)


# ─── Features used in modelling ───────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "stress_amplitude",
    "youngs_modulus_GPa",
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_pct",
    "load_ratio",
    "frequency_Hz",
    "test_temperature_C",
    # Engineered features added later
    "VEC",
    "delta_S_mix",
    "delta_H_mix",
    "n_elements",
]

CATEGORICAL_FEATURES = ["material_type"]

TARGET_DURATION = "cycles"
TARGET_EVENT = "event"


class FatiguePreprocessor:
    """
    Full preprocessing pipeline for the HEA fatigue dataset.

    Steps
    -----
    1. Keep only useful columns
    2. Handle outliers (IQR capping per material type)
    3. Impute missing numerical values (median per material type)
    4. Encode categorical features
    5. StandardScale numerical features
    """

    def __init__(
        self,
        iqr_multiplier: float = 3.0,
        scale: bool = True,
    ):
        self.iqr_multiplier = iqr_multiplier
        self.scale = scale

        self._scaler: Optional[StandardScaler] = None
        self._imputer: Optional[SimpleImputer] = None
        self._num_cols_used: list[str] = []
        self._fitted: bool = False

    # ─── Public API ─────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return the transformed DataFrame."""
        df = df.copy()
        df = self._encode_categoricals(df)
        df = self._coerce_numerics(df)
        df = self._impute(df, fit=True)
        df = self._cap_outliers(df)
        if self.scale:
            df = self._scale(df, fit=True)
        self._fitted = True
        logger.info(f"Preprocessor fitted. Output shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted pipeline to new data."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        df = df.copy()
        df = self._encode_categoricals(df)
        df = self._coerce_numerics(df)
        df = self._impute(df, fit=False)
        df = self._cap_outliers(df)
        if self.scale:
            df = self._scale(df, fit=False)
        return df

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> "FatiguePreprocessor":
        return joblib.load(path)

    # ─── Private helpers ────────────────────────────────────────────────────
    def _coerce_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force-cast all NUMERICAL_FEATURES columns to float.

        Some Excel cells contain multi-value strings like '0.1, 0.15'.
        We take the first numeric token; unparseable values become NaN
        (which are then handled by _impute).
        """
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                def _parse(v):
                    if isinstance(v, str):
                        # Take first number found in the string
                        import re
                        nums = re.findall(r"[-+]?\d*\.?\d+", v)
                        return float(nums[0]) if nums else float("nan")
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return float("nan")
                df[col] = df[col].map(_parse)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode or binary-encode categorical columns."""
        if "material_type" in df.columns:
            df["is_MPEA"] = (df["material_type"] == "MPEA").astype(int)
        if "fatigue_environment" in df.columns:
            df["is_air"] = (
                df["fatigue_environment"].str.lower().str.contains("air").fillna(True).astype(int)
            )
        return df

    def _impute(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Median-impute numerical features."""
        cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        self._num_cols_used = cols

        if fit:
            # Per-material-type median imputation
            for col in cols:
                if df[col].isna().any():
                    medians = df.groupby("material_type")[col].transform("median")
                    global_median = df[col].median()
                    df[col] = df[col].fillna(medians).fillna(global_median)
            self._imputer = "fitted"
        else:
            for col in cols:
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme values using IQR method."""
        cols = [c for c in self._num_cols_used if c in df.columns]
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - self.iqr_multiplier * iqr
            hi = q3 + self.iqr_multiplier * iqr
            n_before = df[col].between(lo, hi).sum()
            df[col] = df[col].clip(lo, hi)
            n_after = len(df)
            if n_before < n_after:
                logger.debug(f"[{col}] clipped {n_after - n_before} values to [{lo:.2f}, {hi:.2f}]")
        return df

    def _scale(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """StandardScale numerical features."""
        cols = [c for c in self._num_cols_used if c in df.columns]
        if not cols:
            return df
        if fit:
            self._scaler = StandardScaler()
            df[cols] = self._scaler.fit_transform(df[cols])
        else:
            df[cols] = self._scaler.transform(df[cols])
        return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns actually present in df,
    suitable for model training.
    """
    candidates = NUMERICAL_FEATURES + ["is_MPEA", "is_air"]
    return [c for c in candidates if c in df.columns]


def make_survival_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce (X, y_structured, durations) from the preprocessed DataFrame.

    Returns
    -------
    X            : np.ndarray of shape (n, p)
    y_structured : structured array with fields ('event', 'duration') for sksurv
    durations    : np.ndarray of cycle counts (log10 transformed)
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0).values.astype(np.float64)

    events = df[TARGET_EVENT].astype(bool).values
    durations = df[TARGET_DURATION].values.astype(np.float64)

    y_structured = np.array(
        [(e, d) for e, d in zip(events, np.log10(durations + 1))],
        dtype=[("event", "?"), ("duration", "<f8")],
    )

    return X, y_structured, durations
