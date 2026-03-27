"""
Unified Model Trainer
======================
Orchestrates train/test split, cross-validation, and hyperparameter tuning
for all three survival models.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hea_fatigue.models.weibull_model import WeibullSurvivalModel
from hea_fatigue.models.cox_model import CoxPHSurvivalModel
from hea_fatigue.models.rsf_model import RSFSurvivalModel
from hea_fatigue.preprocessing.preprocessor import (
    get_feature_columns,
    make_survival_arrays,
    TARGET_DURATION,
    TARGET_EVENT,
)

logger = logging.getLogger(__name__)


def stratified_survival_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/test while preserving the censoring ratio per
    material type (MG / MPEA) as closely as possible.
    """
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []

    groups = df.groupby(["material_type", TARGET_EVENT], observed=True)
    for _, grp in groups:
        n_test = max(1, int(len(grp) * test_size))
        idx = rng.permutation(grp.index)
        test_parts.append(df.loc[idx[:n_test]])
        train_parts.append(df.loc[idx[n_test:]])

    train = pd.concat(train_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


class SurvivalModelTrainer:
    """
    High-level trainer for Weibull, Cox, and RSF models.

    Usage
    -----
    >>> trainer = SurvivalModelTrainer(df_preprocessed, save_dir="hea_fatigue/models/saved")
    >>> results = trainer.train_all()
    >>> print(results)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        save_dir: str = "hea_fatigue/models/saved",
        test_size: float = 0.2,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_df, self.test_df = stratified_survival_split(
            df, test_size=test_size, random_state=random_state
        )
        self.feature_cols = get_feature_columns(df)
        self.n_folds = n_folds
        self.random_state = random_state

        # Structured arrays for sksurv models
        self.X_train, self.y_train, self.dur_train = make_survival_arrays(self.train_df)
        self.X_test, self.y_test, self.dur_test = make_survival_arrays(self.test_df)

        self.models: dict = {}
        self.cv_scores: dict = {}

    # ─── Individual trainers ──────────────────────────────────────────────
    def train_weibull(self) -> WeibullSurvivalModel:
        logger.info("Training Weibull AFT model…")
        model = WeibullSurvivalModel(penalizer=0.1)
        model.fit(
            self.train_df,
            duration_col=TARGET_DURATION,
            event_col=TARGET_EVENT,
            feature_cols=self.feature_cols,
        )
        model.save(os.path.join(self.save_dir, "weibull_model.pkl"))
        self.models["weibull"] = model
        return model

    def train_cox(self) -> CoxPHSurvivalModel:
        logger.info("Training Cox PH model…")
        model = CoxPHSurvivalModel(penalizer=0.1)
        model.fit(
            self.train_df,
            duration_col=TARGET_DURATION,
            event_col=TARGET_EVENT,
            feature_cols=self.feature_cols,
        )
        model.save(os.path.join(self.save_dir, "cox_model.pkl"))
        self.models["cox"] = model
        return model

    def train_rsf(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 5,
    ) -> RSFSurvivalModel:
        logger.info("Training Random Survival Forest…")
        model = RSFSurvivalModel(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
        )
        model.fit(self.X_train, self.y_train, feature_cols=self.feature_cols)
        model.save(os.path.join(self.save_dir, "rsf_model.pkl"))
        self.models["rsf"] = model
        return model

    # ─── Cross-validation ─────────────────────────────────────────────────
    def cross_validate_rsf(self) -> dict:
        """5-fold CV on the RSF using concordance index."""
        logger.info(f"Running {self.n_folds}-fold CV on RSF…")
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        X_full, y_full, _ = make_survival_arrays(self.df)

        scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full)):
            m = RSFSurvivalModel(n_estimators=100, random_state=self.random_state)
            m.fit(X_full[tr_idx], y_full[tr_idx])
            ci = m.concordance_index(X_full[va_idx], y_full[va_idx])
            scores.append(ci)
            logger.info(f"  Fold {fold+1}: C-index = {ci:.4f}")

        result = {
            "mean_c_index": float(np.mean(scores)),
            "std_c_index": float(np.std(scores)),
            "fold_scores": scores,
        }
        self.cv_scores["rsf"] = result
        logger.info(f"RSF CV C-index: {result['mean_c_index']:.4f} ± {result['std_c_index']:.4f}")
        return result

    # ─── Train all ────────────────────────────────────────────────────────
    def train_all(self) -> pd.DataFrame:
        """Train all models and return a comparison table."""
        records = []

        # Weibull
        try:
            w = self.train_weibull()
            records.append({"model": "Weibull AFT", "train_c_index": w.concordance_index(), "status": "OK"})
        except Exception as e:
            logger.error(f"Weibull training failed: {e}")
            records.append({"model": "Weibull AFT", "status": str(e)})

        # Cox
        try:
            c = self.train_cox()
            records.append({"model": "Cox PH", "train_c_index": c.concordance_index(), "status": "OK"})
        except Exception as e:
            logger.error(f"Cox training failed: {e}")
            records.append({"model": "Cox PH", "status": str(e)})

        # RSF
        try:
            r = self.train_rsf()
            test_ci = r.concordance_index(self.X_test, self.y_test)
            cv = self.cross_validate_rsf()
            records.append({
                "model": "Random Survival Forest",
                "test_c_index": test_ci,
                "cv_c_index_mean": cv["mean_c_index"],
                "cv_c_index_std": cv["std_c_index"],
                "status": "OK",
            })
        except Exception as e:
            logger.error(f"RSF training failed: {e}")
            records.append({"model": "Random Survival Forest", "status": str(e)})

        results_df = pd.DataFrame(records)
        logger.info("\n" + results_df.to_string())
        return results_df
