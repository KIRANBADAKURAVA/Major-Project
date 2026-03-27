"""
Evaluation Module
==================
Computes survival analysis metrics:
  - Concordance Index (C-index)
  - Integrated Brier Score
  - Time-point calibration summary
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score, brier_score

logger = logging.getLogger(__name__)


def compute_concordance_index(
    durations: np.ndarray,
    predicted_risk: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    C-index (Harrell's concordance index).

    Parameters
    ----------
    durations      : observed time-to-event (cycles)
    predicted_risk : higher = higher risk (e.g., negative median life)
    events         : 1 = failure, 0 = censored

    Returns
    -------
    float in [0, 1]; 0.5 = random, 1.0 = perfect
    """
    ci = concordance_index(durations, -predicted_risk, events)
    logger.info(f"C-index: {ci:.4f}")
    return ci


def compute_brier_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_probs: np.ndarray,
    times: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the (integrated) Brier Score using sksurv.

    Parameters
    ----------
    y_train        : structured array with ('event', 'duration') from train set
    y_test         : structured array from test set
    survival_probs : array (n_test, n_times) of predicted S(t) values
    times          : evaluation time points

    Returns
    -------
    ibs            : integrated Brier score (scalar)
    brier_times    : array of evaluated time points
    brier_vals     : Brier score at each time point
    """
    # Clip times to valid range
    min_dur = max(y_train["duration"].min(), y_test["duration"].min())
    max_dur = min(y_train["duration"].max(), y_test["duration"].max())
    valid_mask = (times > min_dur) & (times < max_dur)
    eval_times = times[valid_mask]

    if len(eval_times) < 2:
        logger.warning("Not enough valid times for Brier score; skipping.")
        return np.nan, times, np.full(len(times), np.nan)

    surv_probs_clipped = survival_probs[:, valid_mask]
    brier_times, brier_vals = brier_score(y_train, y_test, surv_probs_clipped, eval_times)
    ibs = integrated_brier_score(y_train, y_test, surv_probs_clipped, eval_times)
    logger.info(f"Integrated Brier Score: {ibs:.4f}")
    return ibs, brier_times, brier_vals


def evaluate_model_summary(
    model_name: str,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    predicted_median: np.ndarray,
    ibs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Return a one-row summary DataFrame for a model.
    """
    ci = compute_concordance_index(durations_test, predicted_median, events_test)
    row = {
        "Model": model_name,
        "C-index": round(ci, 4),
        "Integrated Brier Score": round(ibs, 4) if ibs is not None else "N/A",
        "N_test": len(durations_test),
    }
    return pd.DataFrame([row])


def calibration_table(
    survival_probs: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    quantiles: int = 5,
) -> pd.DataFrame:
    """
    D-calibration-style table: divide predicted survival into quantile
    bins and compare with observed event rates.

    Returns a DataFrame with columns:
      predicted_bin_mean, observed_failure_rate, n_samples
    """
    rows = []
    # Use the midpoint time for calibration
    mid_idx = len(times) // 2
    pred_surv_at_t = survival_probs[:, mid_idx]
    pred_risk = 1 - pred_surv_at_t
    events_series = pd.Series(events.astype(int), name="event")

    bins = pd.qcut(pred_risk, q=quantiles, duplicates="drop")
    for bin_label, idx_grp in events_series.groupby(bins, observed=True):
        obs_rate = idx_grp.mean()
        pred_mean = pred_risk[idx_grp.index].mean()
        rows.append({
            "Predicted risk bin": str(bin_label),
            "Mean predicted risk": round(float(pred_mean), 3),
            "Observed failure rate": round(float(obs_rate), 3),
            "n": len(idx_grp),
        })
    return pd.DataFrame(rows)
