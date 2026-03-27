"""
Random Survival Forest Model
==============================
Wraps scikit-survival's RandomSurvivalForest, which natively handles
censored observations and outputs survival functions per sample.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
import joblib

logger = logging.getLogger(__name__)


class RSFSurvivalModel:
    """
    Random Survival Forest model.

    Advantages over linear models:
      - No proportional-hazards assumption
      - Captures non-linear risk interactions
      - Built-in feature importance

    Parameters
    ----------
    n_estimators : int
    min_samples_leaf : int
    max_features : str or int
    n_jobs : int
    random_state : int
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._model: Optional[RandomSurvivalForest] = None
        self.feature_cols: list[str] = []
        self._unique_times: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y_structured: np.ndarray,
        feature_cols: list[str] = None,
    ) -> "RSFSurvivalModel":
        """
        Fit the RSF.

        Parameters
        ----------
        X            : feature matrix (n_samples, n_features)
        y_structured : structured array with ('event', 'duration') fields
        feature_cols : names of the feature columns (for importance)
        """
        self.feature_cols = feature_cols or [f"f{i}" for i in range(X.shape[1])]
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(X, y_structured)
        self._unique_times = self._model.unique_times_
        logger.info(
            f"RSF fitted with {self.n_estimators} trees. "
            f"Unique failure times: {len(self._unique_times)}"
        )
        return self

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Return S(t | X) for each sample.

        Parameters
        ----------
        X     : feature matrix (n_samples, n_features)
        times : evaluation points.  Pass either:
                - raw cycle counts  (e.g. np.logspace(3, 8, 60)) — auto-converted to log10
                - log10 values      (e.g. np.linspace(3, 8, 60))  — used directly

        Returns
        -------
        DataFrame (rows = samples, cols = requested times in original units)
        """
        self._check_fitted()
        sf_funcs = self._model.predict_survival_function(X)
        use_times_model = self._unique_times  # always log10

        if times is not None:
            # Detect if caller passed raw cycles or log10 values.
            # unique_times_ max is ~8.7 (log10 space); raw cycles > 10^9 would exceed this.
            if np.all(times > 100):
                # Looks like raw cycles — convert to log10
                log_times = np.log10(np.clip(times, 1, None))
            else:
                # Already log10
                log_times = times
        else:
            log_times = use_times_model

        rows = []
        for fn in sf_funcs:
            vals = np.interp(log_times, fn.x, fn.y, left=1.0, right=0.0)
            rows.append(vals)

        # Return with original time labels (raw cycles if that's what caller passed)
        col_labels = times if times is not None else use_times_model
        return pd.DataFrame(rows, columns=col_labels)

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Return cumulative hazard-based risk scores."""
        self._check_fitted()
        return self._model.predict(X)

    def feature_importance(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> pd.Series:
        """
        Compute permutation-based feature importance.

        For each feature, shuffle its values n_repeats times and measure
        the drop in concordance index. Features causing larger drops are
        more important.

        Parameters
        ----------
        X : feature matrix used for evaluation (defaults to training data seen during fit)
        y : structured survival array
        """
        self._check_fitted()

        if X is None or y is None:
            # Return uniform importance as fallback when no eval data provided
            n_feats = len(self.feature_cols)
            return pd.Series(
                np.ones(n_feats) / n_feats,
                index=self.feature_cols,
                name="importance",
            ).sort_values(ascending=False)

        rng = np.random.default_rng(random_state)
        baseline_ci = self._model.score(X, y)

        importances = []
        for j in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                ci_perm = self._model.score(X_perm, y)
                scores.append(baseline_ci - ci_perm)
            importances.append(np.mean(scores))

        return pd.Series(
            importances,
            index=self.feature_cols,
            name="importance",
        ).sort_values(ascending=False)

    def concordance_index(
        self, X: np.ndarray, y_structured: np.ndarray
    ) -> float:
        self._check_fitted()
        return self._model.score(X, y_structured)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RSFSurvivalModel":
        return joblib.load(path)

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("RSF not yet fitted. Call fit() first.")
