"""
Main Pipeline Orchestrator
============================
End-to-end runner:
  1. Load real dataset (FatigueData-CMA2022.xlsx)
  2. Feature engineering (VEC, ΔSmix, ΔHmix)
  3. Preprocessing (impute, scale)
  4. Train all survival models (Weibull, Cox, RSF)
  5. Evaluate on test set
  6. Generate all visualizations
  7. Save outputs to outputs/

Run:
    python main.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ─── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

ROOT = Path(__file__).parent
DATASET = ROOT / "FatigueData-CMA2022.xlsx"
OUTPUT_DIR = ROOT / "outputs"
SAVE_DIR = ROOT / "hea_fatigue" / "models" / "saved"

OUTPUT_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ─── Imports ──────────────────────────────────────────────────────────────────
from hea_fatigue.data.loader import load_dataset
from hea_fatigue.feature_engineering.feature_engineer import engineer_features
from hea_fatigue.preprocessing.preprocessor import (
    FatiguePreprocessor,
    get_feature_columns,
    make_survival_arrays,
    TARGET_DURATION,
    TARGET_EVENT,
)
from hea_fatigue.models.trainer import SurvivalModelTrainer
from hea_fatigue.evaluation.evaluator import (
    compute_concordance_index,
    compute_brier_score,
    evaluate_model_summary,
    calibration_table,
)
from hea_fatigue.visualization.plotter import (
    plot_survival_curves,
    plot_sn_curves,
    plot_feature_importance,
    plot_risk_heatmap,
    plot_brier_score,
    plot_model_comparison,
    build_dashboard,
)


def run_pipeline():
    logger.info("=" * 60)
    logger.info("  HEA Fatigue Life Prediction — Full Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    logger.info("\n[1/7] Loading dataset…")
    df_raw = load_dataset(DATASET)
    logger.info(f"  Raw merged data: {df_raw.shape}")

    # ── Step 2: Feature engineering ───────────────────────────────────────
    logger.info("\n[2/7] Engineering thermodynamic features…")
    df_feat = engineer_features(df_raw, composition_col="composition")
    logger.info(
        f"  VEC range:     [{df_feat['VEC'].min():.2f}, {df_feat['VEC'].max():.2f}]"
    )
    logger.info(
        f"  ΔSmix range:   [{df_feat['delta_S_mix'].min():.2f}, {df_feat['delta_S_mix'].max():.2f}] J/mol·K"
    )
    logger.info(
        f"  ΔHmix range:   [{df_feat['delta_H_mix'].min():.2f}, {df_feat['delta_H_mix'].max():.2f}] kJ/mol"
    )

    # ── Step 3: Preprocessing ─────────────────────────────────────────────
    logger.info("\n[3/7] Preprocessing (impute, scale, encode)…")
    preprocessor = FatiguePreprocessor(iqr_multiplier=3.0, scale=True)
    df_proc = preprocessor.fit_transform(df_feat)
    preprocessor.save(str(SAVE_DIR / "preprocessor.pkl"))
    logger.info(f"  Preprocessed data: {df_proc.shape}")

    # ── Step 4: Train models ───────────────────────────────────────────────
    logger.info("\n[4/7] Training survival models…")
    trainer = SurvivalModelTrainer(
        df_proc,
        save_dir=str(SAVE_DIR),
        test_size=0.20,
        n_folds=5,
        random_state=42,
    )
    results_df = trainer.train_all()

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    logger.info("\n[5/7] Evaluating on held-out test set…")
    test_df  = trainer.test_df
    X_test   = trainer.X_test
    y_test   = trainer.y_test

    feature_cols = get_feature_columns(df_proc)
    eval_times   = np.logspace(3, 8, 60)

    rsf_model = trainer.models.get("rsf")
    if rsf_model:
        sf_df = rsf_model.predict_survival_function(X_test, times=eval_times)
        sf_matrix = sf_df.values  # (n_test, n_times)
        risk_scores = rsf_model.predict_risk_score(X_test)

        durations_test = trainer.dur_test
        events_test    = test_df[TARGET_EVENT].values

        # C-index
        ci = compute_concordance_index(durations_test, risk_scores, events_test)

        # Brier score
        y_train_struct = trainer.y_train
        # Build log10-based time axis matching y_struct durations
        bs_times = np.log10(eval_times + 1)
        ibs, brier_t, brier_v = compute_brier_score(
            y_train_struct, y_test, sf_matrix, bs_times
        )

        # Calibration (use risk scores for binning, compare with actual events)
        risk_scores_for_calib = 1.0 - sf_matrix[:, len(eval_times) // 2]  # P(failure) at midpoint
        calib = calibration_table(sf_matrix, events_test.astype(int), eval_times, quantiles=5)
        calib.to_csv(OUTPUT_DIR / "calibration_table.csv", index=False)

        # Summary
        summary_df = evaluate_model_summary(
            "Random Survival Forest", durations_test, events_test, risk_scores, ibs
        )
        results_df = pd.concat([results_df, summary_df], ignore_index=True)
        logger.info(f"\n  Test C-index (RSF): {ci:.4f}")
        logger.info(f"  Integrated Brier Score: {ibs:.4f}" if not np.isnan(ibs) else "  IBS: N/A")
        logger.info(f"\n  Calibration:\n{calib.to_string(index=False)}")
    else:
        sf_df = None
        sf_matrix = None
        ibs = np.nan
        brier_t = eval_times
        brier_v = np.full(len(eval_times), np.nan)
        logger.warning("  RSF model not available for evaluation.")

    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info(f"\n  Results saved to outputs/model_comparison.csv")

    # ── Step 6: Visualize ─────────────────────────────────────────────────
    logger.info("\n[6/7] Generating visualizations…")

    # Survival curves
    if sf_df is not None:
        plot_survival_curves(
            sf_df.reset_index(drop=True),
            times=eval_times,
            title="Survival Curves S(N) — Test Set (RSF)",
            output_path=str(OUTPUT_DIR / "survival_curves.html"),
            sample_indices=list(range(min(20, len(sf_df)))),
        )

    # S-N probabilistic curves
    stress_levels = np.linspace(100, 800, 25)
    cycles_grid   = np.logspace(3, 9, 60)

    if rsf_model:
        sn_matrix = np.zeros((len(stress_levels), len(cycles_grid)))
        feat_cols = rsf_model.feature_cols
        # Use a representative MPEA feature vector
        ref_row = {c: float(df_proc[c].median()) for c in feat_cols if c in df_proc.columns}
        ref_row["is_MPEA"] = 1  # use MPEA as representative material

        for i, s in enumerate(stress_levels):
            ref_row["stress_amplitude"] = s
            x_row = np.array([[ref_row.get(c, 0.0) for c in feat_cols]])
            sf_i  = rsf_model.predict_survival_function(x_row, times=cycles_grid)
            sn_matrix[i] = sf_i.iloc[0].values

        mat_label = "MPEA"
        plot_sn_curves(
            stress_levels, cycles_grid, sn_matrix,
            material_type=mat_label,
            output_path=str(OUTPUT_DIR / "sn_curves.html"),
        )

        # Feature importance (permutation-based, needs eval data)
        logger.info("  Computing permutation feature importance…")
        fi = rsf_model.feature_importance(X=X_test, y=y_test, n_repeats=3)
        plot_feature_importance(fi, output_path=str(OUTPUT_DIR / "feature_importance.html"))
        fi.to_csv(OUTPUT_DIR / "feature_importance.csv")

        # Risk heatmap
        risk_matrix = 1 - sn_matrix
        plot_risk_heatmap(
            stress_levels, cycles_grid, risk_matrix,
            output_path=str(OUTPUT_DIR / "risk_heatmap.html"),
        )

        # Brier score plot
        if not np.all(np.isnan(brier_v)):
            plot_brier_score(
                brier_t, brier_v, float(ibs) if not np.isnan(ibs) else 0,
                output_path=str(OUTPUT_DIR / "brier_score.html"),
            )

    # Model comparison
    plot_model_comparison(results_df, output_path=str(OUTPUT_DIR / "model_comparison.html"))

    # Full dashboard
    if sf_df is not None and rsf_model:
        fi = rsf_model.feature_importance()
        build_dashboard(
            sf_df.reset_index(drop=True),
            eval_times,
            fi,
            results_df,
            output_path=str(OUTPUT_DIR / "dashboard.html"),
        )

    # ── Step 7: Summary ───────────────────────────────────────────────────
    logger.info("\n[7/7] Pipeline complete!")
    logger.info("=" * 60)
    logger.info("  Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        logger.info(f"    {f.name}  ({f.stat().st_size // 1024} KB)")
    logger.info("=" * 60)
    logger.info("\nTo start the Streamlit UI:")
    logger.info("  streamlit run app.py\n")
    logger.info("To start the FastAPI server:")
    logger.info("  uvicorn api.server:app --host 0.0.0.0 --port 8000\n")

    return results_df


if __name__ == "__main__":
    run_pipeline()
