"""
Weibull AFT Survival Model
===========================
Wraps lifelines.WeibullAFTFitter to provide a consistent interface
for survival probability S(N) and hazard h(N) prediction.

Weibull formulation:
    S(N) = exp(-(N / η)^β)
where β (shape) and η (scale) are functions of the input features.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
import joblib

logger = logging.getLogger(__name__)


class WeibullSurvivalModel:
    """
    Weibull Accelerated Failure Time model for fatigue life prediction.

    Parameters
    ----------
    penalizer : float
        L2 regularisation strength passed to WeibullAFTFitter.
    fit_intercept : bool
        Whether to fit an intercept term.
    """

    def __init__(self, penalizer: float = 0.1, fit_intercept: bool = True):
        self.penalizer = penalizer
        self.fit_intercept = fit_intercept
        self._model: Optional[WeibullAFTFitter] = None
        self.feature_cols: list[str] = []

    # ─── Fit ────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "cycles",
        event_col: str = "event",
        feature_cols: list[str] = None,
    ) -> "WeibullSurvivalModel":
        """
        Fit the Weibull AFT model.

        Parameters
        ----------
        df           : preprocessed DataFrame
        duration_col : column with time-to-event (cycles)
        event_col    : 1 = failure, 0 = runout (censored)
        feature_cols : list of covariate column names
        """
        self.feature_cols = feature_cols or []
        train_cols = self.feature_cols + [duration_col, event_col]
        train_df = df[train_cols].dropna()

        # lifelines expects log-transformed durations for AFT
        train_df = train_df.copy()
        train_df[duration_col] = train_df[duration_col].clip(lower=1)

        self._model = WeibullAFTFitter(penalizer=self.penalizer, fit_intercept=self.fit_intercept)
        self._model.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col,
            ancillary=True,  # fit both λ and ρ as functions of covariates
        )
        logger.info(
            f"WeibullAFT fitted. Concordance: {self._model.concordance_index_:.4f}"
        )
        return self

    # ─── Predict ────────────────────────────────────────────────────────────
    def predict_survival_function(
        self,
        X: pd.DataFrame,
        times: np.ndarray,
    ) -> pd.DataFrame:
        """
        Return S(t | X) evaluated at each time in `times`.

        Returns
        -------
        DataFrame of shape (len(times), len(X)), indexed by time.
        """
        self._check_fitted()
        return self._model.predict_survival_function(X, times=times)

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median fatigue life for each sample."""
        self._check_fitted()
        return self._model.predict_median(X).values.ravel()

    def predict_hazard(self, X: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
        """Return h(t | X) evaluated at each time in `times`."""
        self._check_fitted()
        return self._model.predict_cumulative_hazard(X, times=times).diff().clip(lower=0)

    def concordance_index(self) -> float:
        self._check_fitted()
        return self._model.concordance_index_

    def summary(self) -> pd.DataFrame:
        self._check_fitted()
        return self._model.summary

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "WeibullSurvivalModel":
        return joblib.load(path)

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Model not yet fitted. Call fit() first.")
