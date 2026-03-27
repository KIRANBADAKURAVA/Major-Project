"""
Cox Proportional Hazards Survival Model
=========================================
Wraps lifelines.CoxPHFitter for the HP fatigue dataset.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import joblib

logger = logging.getLogger(__name__)


class CoxPHSurvivalModel:
    """
    Cox Proportional Hazards model.

    Assumes the hazard has the form:
        h(t | X) = h₀(t) × exp(β · X)

    Parameters
    ----------
    penalizer : float
        L2 regularisation for the partial-likelihood.
    """

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self._model: Optional[CoxPHFitter] = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "cycles",
        event_col: str = "event",
        feature_cols: list[str] = None,
    ) -> "CoxPHSurvivalModel":
        self.feature_cols = feature_cols or []
        train_cols = self.feature_cols + [duration_col, event_col]
        train_df = df[train_cols].dropna()

        # Cox PH works on raw durations (no log needed)
        self._model = CoxPHFitter(penalizer=self.penalizer)
        self._model.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col,
        )
        logger.info(
            f"CoxPH fitted. Concordance: {self._model.concordance_index_:.4f}"
        )
        return self

    def predict_survival_function(
        self, X: pd.DataFrame, times: np.ndarray
    ) -> pd.DataFrame:
        self._check_fitted()
        return self._model.predict_survival_function(X, times=times)

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_median(X).values.ravel()

    def concordance_index(self) -> float:
        self._check_fitted()
        return self._model.concordance_index_

    def summary(self) -> pd.DataFrame:
        self._check_fitted()
        return self._model.summary

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "CoxPHSurvivalModel":
        return joblib.load(path)

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Model not yet fitted. Call fit() first.")
