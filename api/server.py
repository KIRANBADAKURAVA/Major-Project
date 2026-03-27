"""
FastAPI Server
===============
REST API endpoints for real-time inference on the trained survival models.

Endpoints
---------
POST /predict      → survival probability at given stress & cycles
POST /sn_curve     → full probabilistic S-N curve data
POST /risk_map     → failure probability heatmap data
GET  /health       → health check
GET  /model_info   → loaded model metadata
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hea_fatigue.feature_engineering.feature_engineer import (
    parse_composition,
    compute_VEC,
    compute_delta_S_mix,
    compute_delta_H_mix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Model registry ───────────────────────────────────────────────────────────
SAVE_DIR = Path(__file__).parent.parent / "hea_fatigue" / "models" / "saved"
MODELS: dict = {}


def _load_models():
    """Load all trained models from disk."""
    for name, fname in [
        ("weibull", "weibull_model.pkl"),
        ("cox",     "cox_model.pkl"),
        ("rsf",     "rsf_model.pkl"),
        ("preprocessor", "preprocessor.pkl"),
    ]:
        path = SAVE_DIR / fname
        if path.exists():
            MODELS[name] = joblib.load(path)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found: {path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HEA Fatigue Life Prediction API",
    description="Probabilistic fatigue life prediction for Metastable High Entropy Alloys",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    stress_amplitude: float = Field(..., gt=0, description="Stress amplitude in MPa")
    cycles: float = Field(..., gt=0, description="Number of cycles")
    material_type: str = Field("MPEA", description="'MG' or 'MPEA'")
    composition: Optional[str] = Field(None, description="e.g. 'CrMnFeCoNi' or 'Zr55.0-Cu30.0-Al10.0-Ni5.0'")
    yield_strength_MPa: Optional[float] = Field(None)
    uts_MPa: Optional[float] = Field(None)
    youngs_modulus_GPa: Optional[float] = Field(None)
    model: str = Field("rsf", description="'weibull', 'cox', or 'rsf'")


class SNRequest(BaseModel):
    stress_levels: list[float] = Field(..., description="List of stress amplitudes in MPa")
    material_type: str = Field("MPEA")
    composition: Optional[str] = None
    yield_strength_MPa: Optional[float] = None
    uts_MPa: Optional[float] = None
    model: str = Field("rsf")
    n_points: int = Field(50, ge=10, le=200)


class RiskMapRequest(BaseModel):
    stress_min: float = Field(100.0, gt=0)
    stress_max: float = Field(800.0, gt=0)
    log_cycle_min: float = Field(3.0)
    log_cycle_max: float = Field(8.0)
    material_type: str = Field("MPEA")
    composition: Optional[str] = None
    n_stress: int = Field(20, ge=5, le=50)
    n_cycles: int = Field(30, ge=5, le=100)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _build_feature_row(req) -> dict:
    """Build a feature dict from any request object using safe getattr."""
    mat = getattr(req, "material_type", "MPEA")
    is_mpea = 1 if mat == "MPEA" else 0
    comp = getattr(req, "composition", None)
    fractions = parse_composition(comp) if comp else None

    # Safe mechanical property defaults
    ys  = getattr(req, "yield_strength_MPa", None) or (400  if is_mpea else 1000)
    uts = getattr(req, "uts_MPa", None)            or (600  if is_mpea else 1500)
    E   = getattr(req, "youngs_modulus_GPa", None) or (200  if is_mpea else 90)
    sa  = getattr(req, "stress_amplitude", 300.0)

    return {
        "stress_amplitude":   float(sa),
        "is_MPEA":            is_mpea,
        "yield_strength_MPa": float(ys),
        "uts_MPa":            float(uts),
        "youngs_modulus_GPa": float(E),
        "elongation_pct":     30.0 if is_mpea else 2.0,
        "load_ratio":         -1.0,
        "frequency_Hz":        10.0,
        "test_temperature_C":  25.0,
        "is_air":              1,
        "VEC":         compute_VEC(fractions)         if fractions else (7.0  if is_mpea else 6.0),
        "delta_S_mix": compute_delta_S_mix(fractions) if fractions else (13.4 if is_mpea else 9.0),
        "delta_H_mix": compute_delta_H_mix(fractions) if fractions else (-2.5 if is_mpea else -15.0),
        "n_elements":  float(len(fractions))          if fractions else (5.0  if is_mpea else 3.0),
    }


def _predict_with_rsf(feature_row: dict, times: np.ndarray) -> np.ndarray:
    """Get survival probabilities from RSF model with correct preprocessing."""
    rsf = MODELS.get("rsf")
    pre = MODELS.get("preprocessor")
    if rsf is None:
        raise HTTPException(status_code=503, detail="RSF model not loaded.")

    feat_cols = rsf.feature_cols

    # Build a single-row DataFrame for the preprocessor
    row_df = pd.DataFrame([{c: feature_row.get(c, 0.0) for c in feat_cols}], columns=feat_cols)

    # Apply the same StandardScaler / imputer that was used during training
    if pre is not None:
        try:
            row_df = pre.transform(row_df)[feat_cols]
        except Exception:
            pass  # fallback: use raw values if preprocessor fails

    x = row_df.values
    sf_df = rsf.predict_survival_function(x, times=times)
    return sf_df.iloc[0].values


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


@app.get("/model_info")
def model_info():
    info = {}
    rsf = MODELS.get("rsf")
    if rsf:
        info["rsf"] = {
            "n_estimators": rsf.n_estimators,
            "features": rsf.feature_cols,
        }
    return info


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict survival probability S(N) at a given number of cycles.

    Returns the full survival curve plus point estimate.
    """
    times = np.logspace(3, 8, 80)
    feature_row = _build_feature_row(req)

    try:
        survival_probs = _predict_with_rsf(feature_row, times)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Interpolate S at requested cycles
    s_at_n = float(np.interp(req.cycles, times, survival_probs, left=1.0, right=0.0))
    failure_prob = 1.0 - s_at_n

    # Reliability statement
    reliability_pct = int(round(s_at_n * 100))
    statement = (
        f"At {req.stress_amplitude:.0f} MPa, the predicted survival probability "
        f"at {req.cycles:.2e} cycles is {reliability_pct}% "
        f"(failure probability: {failure_prob*100:.1f}%)."
    )

    return {
        "stress_amplitude_MPa": req.stress_amplitude,
        "cycles": req.cycles,
        "material_type": req.material_type,
        "survival_probability": round(s_at_n, 4),
        "failure_probability": round(failure_prob, 4),
        "reliability_statement": statement,
        "survival_curve": {
            "cycles": times.tolist(),
            "survival": survival_probs.tolist(),
        },
    }


@app.post("/sn_curve")
def sn_curve(req: SNRequest):
    """
    Return probabilistic S-N curve data for a list of stress levels.

    Returns median (50%), 10%, and 90% cycle counts at each stress.
    """
    times = np.logspace(3, 9, req.n_points)
    rows = []

    for stress in req.stress_levels:
        feature_row = _build_feature_row(req)
        feature_row["stress_amplitude"] = float(stress)
        try:
            sf = _predict_with_rsf(feature_row, times)
        except Exception as e:
            rows.append({"stress_MPa": stress, "error": str(e)})
            continue

        # Capture sf by value via default arg to avoid closure-capture bug
        def cycles_at_prob(p, _sf=sf):
            idx = np.searchsorted(-_sf, -p)
            return float(times[min(idx, len(times) - 1)])

        rows.append({
            "stress_MPa": stress,
            "N_10pct": cycles_at_prob(0.10),
            "N_50pct": cycles_at_prob(0.50),
            "N_90pct": cycles_at_prob(0.90),
        })

    return {"material_type": req.material_type, "sn_data": rows}


@app.post("/risk_map")
def risk_map(req: RiskMapRequest):
    """
    Return a 2-D failure probability heatmap over stress × log-cycles space.
    """
    stress_grid  = np.linspace(req.stress_min, req.stress_max, req.n_stress)
    log_cyc_grid = np.linspace(req.log_cycle_min, req.log_cycle_max, req.n_cycles)
    cycles_grid  = 10.0 ** log_cyc_grid

    base_row = _build_feature_row(req)
    matrix = []
    for stress in stress_grid:
        row = {**base_row, "stress_amplitude": float(stress)}
        try:
            sf = _predict_with_rsf(row, cycles_grid)
        except Exception:
            sf = np.linspace(1.0, 0.0, len(cycles_grid))
        matrix.append((1.0 - sf).tolist())

    return {
        "stress_grid": stress_grid.tolist(),
        "log_cycle_grid": log_cyc_grid.tolist(),
        "failure_probability_matrix": matrix,
    }
