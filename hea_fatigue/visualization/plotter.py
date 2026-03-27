"""
Visualization Module
=====================
All publication-quality and interactive plots for the HEA fatigue system:

  1. Survival curves   S(N) vs N
  2. Probabilistic S-N curves with percentile bands
  3. Hazard function plots
  4. Feature importance bar chart
  5. Failure risk heatmap (stress × composition descriptor)
  6. Brier score over time
  7. Calibration scatter
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

logger = logging.getLogger(__name__)

# ─── Style defaults ───────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
MPL_STYLE = "dark_background"
COLOR_MG   = "#00b4d8"
COLOR_MPEA = "#f77f00"
BAND_COLORS = ["#ef233c", "#fcbf49", "#2ec4b6"]  # 10%, 50%, 90%


def _savefig(fig: go.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path)
    logger.info(f"Saved: {path}")


def _save_mpl(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ─── 1. Survival Curves ───────────────────────────────────────────────────────
def plot_survival_curves(
    survival_df: pd.DataFrame,
    times: np.ndarray,
    title: str = "Survival Curves S(N)",
    output_path: str = "outputs/survival_curves.html",
    sample_indices: Optional[list[int]] = None,
    labels: Optional[list[str]] = None,
) -> go.Figure:
    """
    Plot S(N) vs N for a set of samples.

    Parameters
    ----------
    survival_df : DataFrame (n_samples × n_times) with S(t) values
    times       : cycle counts corresponding to columns of survival_df
    """
    fig = go.Figure()
    n = len(survival_df)
    idx = sample_indices if sample_indices else list(range(min(n, 15)))

    palette = px.colors.sample_colorscale("Plasma", [i / max(len(idx) - 1, 1) for i in range(len(idx))])

    for i, (si, color) in enumerate(zip(idx, palette)):
        row = survival_df.iloc[si]
        lbl = labels[i] if labels else f"Sample {si}"
        fig.add_trace(go.Scatter(
            x=times, y=row.values,
            mode="lines",
            name=lbl,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis=dict(title="Cycles to Failure (N)", type="log"),
        yaxis=dict(title="Survival Probability S(N)", range=[0, 1]),
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="v", x=1.01),
        height=500,
    )
    _savefig(fig, output_path)
    return fig


# ─── 2. Probabilistic S-N Curves ─────────────────────────────────────────────
def plot_sn_curves(
    stress_levels: np.ndarray,
    cycles_grid: np.ndarray,
    survival_matrix: np.ndarray,
    material_type: str = "MPEA",
    output_path: str = "outputs/sn_curves.html",
) -> go.Figure:
    """
    S-N curves with 10%, 50%, 90% survival probability bands.

    Parameters
    ----------
    stress_levels   : 1-D array of stress amplitudes (MPa)
    cycles_grid     : 1-D array of cycle counts
    survival_matrix : 2-D array (n_stress × n_cycles) of S values
    """
    fig = go.Figure()

    probs = [0.10, 0.50, 0.90]
    line_styles = ["dot", "solid", "dash"]
    line_names  = ["10% survival", "50% survival (median)", "90% survival"]

    for prob, style, name, color in zip(probs, line_styles, line_names, BAND_COLORS):
        cycles_at_prob = []
        for i in range(len(stress_levels)):
            row = survival_matrix[i]
            # Find first cycle where S(N) drops below prob
            idx = np.searchsorted(-row, -prob)
            if idx < len(cycles_grid):
                cycles_at_prob.append(cycles_grid[idx])
            else:
                cycles_at_prob.append(cycles_grid[-1])

        fig.add_trace(go.Scatter(
            x=cycles_at_prob,
            y=stress_levels,
            mode="lines+markers",
            name=name,
            line=dict(color=color, dash=style, width=2.5),
            marker=dict(size=6),
        ))

    # Add 90%-10% shaded band
    c10 = []
    c90 = []
    for i in range(len(stress_levels)):
        row = survival_matrix[i]
        i10 = np.searchsorted(-row, -0.10)
        i90 = np.searchsorted(-row, -0.90)
        c10.append(cycles_grid[min(i10, len(cycles_grid)-1)])
        c90.append(cycles_grid[min(i90, len(cycles_grid)-1)])

    fig.add_trace(go.Scatter(
        x=c10 + c90[::-1],
        y=np.concatenate([stress_levels, stress_levels[::-1]]),
        fill="toself",
        fillcolor="rgba(252,191,73,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10%–90% band",
    ))

    fig.update_layout(
        title=dict(text=f"Probabilistic S-N Curve — {material_type}", font=dict(size=20)),
        xaxis=dict(title="Cycles to Failure (N)", type="log"),
        yaxis=dict(title="Stress Amplitude σₐ (MPa)"),
        template=PLOTLY_TEMPLATE,
        height=520,
    )
    _savefig(fig, output_path)
    return fig


# ─── 3. Feature Importance ────────────────────────────────────────────────────
def plot_feature_importance(
    importance: pd.Series,
    output_path: str = "outputs/feature_importance.html",
) -> go.Figure:
    imp = importance.sort_values().tail(15)
    colors = px.colors.sample_colorscale("Viridis", [i / max(len(imp)-1, 1) for i in range(len(imp))])

    fig = go.Figure(go.Bar(
        x=imp.values,
        y=imp.index,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in imp.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Feature Importance (Random Survival Forest)", font=dict(size=18)),
        xaxis_title="Gini Importance",
        template=PLOTLY_TEMPLATE,
        height=500,
        margin=dict(l=180),
    )
    _savefig(fig, output_path)
    return fig


# ─── 4. Failure Risk Heatmap ──────────────────────────────────────────────────
def plot_risk_heatmap(
    stress_grid: np.ndarray,
    cycles_grid: np.ndarray,
    failure_prob_matrix: np.ndarray,
    title: str = "Failure Probability Heatmap",
    output_path: str = "outputs/risk_heatmap.html",
) -> go.Figure:
    """
    2-D heatmap: stress amplitude (y) vs log10(cycles) (x), colour = P(failure).
    """
    log_cycles = np.log10(cycles_grid)

    fig = go.Figure(go.Heatmap(
        x=log_cycles,
        y=stress_grid,
        z=failure_prob_matrix,
        colorscale="RdYlGn_r",
        colorbar=dict(title="P(failure)"),
        zmin=0, zmax=1,
    ))

    # Add contour lines at 10%, 50%, 90%
    fig.add_trace(go.Contour(
        x=log_cycles,
        y=stress_grid,
        z=failure_prob_matrix,
        contours=dict(
            coloring="none",
            showlabels=True,
            start=0.10, end=0.90, size=0.20,
        ),
        line=dict(color="white", width=1),
        showscale=False,
        name="Iso-probability",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="log₁₀(Cycles to Failure)",
        yaxis_title="Stress Amplitude σₐ (MPa)",
        template=PLOTLY_TEMPLATE,
        height=550,
    )
    _savefig(fig, output_path)
    return fig


# ─── 5. Brier Score over time ─────────────────────────────────────────────────
def plot_brier_score(
    times: np.ndarray,
    brier_vals: np.ndarray,
    ibs: float,
    output_path: str = "outputs/brier_score.html",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.log10(times + 1), y=brier_vals,
        mode="lines+markers", name="Brier Score",
        line=dict(color="#f77f00", width=2),
    ))
    fig.add_hline(
        y=ibs, line_dash="dash", line_color="#ef233c",
        annotation_text=f"IBS = {ibs:.3f}", annotation_position="top right",
    )
    fig.update_layout(
        title="Brier Score over Time",
        xaxis_title="log₁₀(Cycles)",
        yaxis_title="Brier Score",
        template=PLOTLY_TEMPLATE,
        height=420,
    )
    _savefig(fig, output_path)
    return fig


# ─── 6. Model comparison dashboard ───────────────────────────────────────────
def plot_model_comparison(
    results_df: pd.DataFrame,
    output_path: str = "outputs/model_comparison.html",
) -> go.Figure:
    """Bar chart comparing C-index (and IBS if available) across models."""
    fig = make_subplots(rows=1, cols=1)

    models = results_df["model"].tolist()
    # Prefer test C-index, fall back to train C-index
    ci_col = "test_c_index" if "test_c_index" in results_df.columns else "train_c_index"
    c_indices = results_df.get(ci_col, pd.Series([np.nan]*len(results_df))).values

    colors = [COLOR_MPEA, COLOR_MG, "#9b5de5"]

    fig.add_trace(go.Bar(
        x=models,
        y=c_indices,
        marker_color=colors[:len(models)],
        text=[f"{v:.3f}" if not np.isnan(v) else "N/A" for v in c_indices],
        textposition="outside",
        name="C-index",
    ))

    fig.update_layout(
        title="Model Comparison — Concordance Index",
        yaxis=dict(title="C-index", range=[0, 1]),
        template=PLOTLY_TEMPLATE,
        height=420,
        showlegend=False,
    )
    _savefig(fig, output_path)
    return fig


# ─── 7. Full interactive dashboard ───────────────────────────────────────────
def build_dashboard(
    survival_df: pd.DataFrame,
    times: np.ndarray,
    importance: pd.Series,
    results_df: pd.DataFrame,
    output_path: str = "outputs/dashboard.html",
) -> go.Figure:
    """
    Combine survival curves, feature importance, and model comparison
    into a single multi-panel Plotly dashboard.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Survival Curves S(N)",
            "Feature Importance",
            "Model Comparison (C-index)",
            "Survival Distribution at 10⁶ Cycles",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # --- Panel 1: Survival curves (first 10 samples) -------------------------
    palette = px.colors.sample_colorscale("Plasma", np.linspace(0, 1, min(10, len(survival_df))))
    for i in range(min(10, len(survival_df))):
        fig.add_trace(go.Scatter(
            x=times, y=survival_df.iloc[i].values,
            mode="lines", line=dict(color=palette[i], width=1.5),
            showlegend=False, name=f"S{i}",
        ), row=1, col=1)

    # --- Panel 2: Feature importance -----------------------------------------
    imp = importance.sort_values().tail(10)
    fig.add_trace(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=px.colors.sample_colorscale("Viridis", np.linspace(0, 1, len(imp))),
        showlegend=False,
    ), row=1, col=2)

    # --- Panel 3: Model c-index comparison ------------------------------------
    if results_df is not None and len(results_df):
        ci_col = "test_c_index" if "test_c_index" in results_df.columns else "train_c_index"
        fig.add_trace(go.Bar(
            x=results_df["model"].tolist(),
            y=results_df.get(ci_col, [np.nan]*len(results_df)).values,
            marker_color=[COLOR_MPEA, COLOR_MG, "#9b5de5"][:len(results_df)],
            showlegend=False,
        ), row=2, col=1)

    # --- Panel 4: Histogram of S at 10^6 -----------------------------------
    mid_idx = np.searchsorted(times, 1e6)
    mid_idx = min(mid_idx, survival_df.shape[1] - 1)
    s_vals = survival_df.iloc[:, mid_idx].values
    fig.add_trace(go.Histogram(
        x=s_vals, nbinsx=30,
        marker_color=COLOR_MPEA, showlegend=False, name="S(10⁶)",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text="HEA Fatigue Life Prediction — Dashboard", font=dict(size=22)),
        template=PLOTLY_TEMPLATE,
        height=750,
    )
    fig.update_xaxes(type="log", row=1, col=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Dashboard saved: {output_path}")
    return fig
