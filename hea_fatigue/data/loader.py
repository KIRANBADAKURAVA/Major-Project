"""
Data Loader Module
==================
Loads and merges the S-N experimental data with alloy metadata from the
FatigueData-CMA2022.xlsx dataset.

Sheets used:
  - 'S-N'       : cycles, stress amplitude, runout flag per dataset id
  - 'parameter' : alloy composition, material type, mechanical properties
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Column aliases for cleaner downstream usage ──────────────────────────────
SN_COLS = {
    "dataset id": "dataset_id",
    "life\nN (cycle)": "cycles",
    "stress amplitude\nσa (MPa)": "stress_amplitude",
    "runout": "runout",
}

PARAM_COLS = {
    "dataset id": "dataset_id",
    "type of the material": "material_type",
    "composition\n(at%)": "composition",
    "atomic-level structure": "structure",
    "original name of the material": "material_name",
    "Young's modulus \n(GPa)": "youngs_modulus_GPa",
    "yield strength \n(MPa)": "yield_strength_MPa",
    "ultimate tensile strength \n(MPa)": "uts_MPa",
    "elongation \n(%)": "elongation_pct",
    "load ratio": "load_ratio",
    "frequency \n(Hz)": "frequency_Hz",
    "fatigue temperature \n(°C)": "test_temperature_C",
    "fatigue environment": "fatigue_environment",
    "types of fatigue tests": "test_type",
    "rating score": "rating_score",
}


def load_raw_data(filepath: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all sheets from the Excel file.

    Parameters
    ----------
    filepath : str or Path
        Path to FatigueData-CMA2022.xlsx

    Returns
    -------
    dict with keys 'sn', 'parameter' containing DataFrames
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    logger.info(f"Loading Excel file: {filepath}")

    sn_raw = pd.read_excel(filepath, sheet_name="S-N")
    param_raw = pd.read_excel(filepath, sheet_name="parameter", header=1)

    logger.info(f"S-N sheet: {sn_raw.shape[0]} rows, {sn_raw.shape[1]} cols")
    logger.info(f"Parameter sheet: {param_raw.shape[0]} rows, {param_raw.shape[1]} cols")

    return {"sn": sn_raw, "parameter": param_raw}


def _clean_sn(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and clean the S-N sheet."""
    df = df.rename(columns=SN_COLS)
    df = df[list(SN_COLS.values())]
    # Normalise runout to boolean-like int
    df["runout"] = df["runout"].fillna(0).astype(int)
    # Event = 1 means failure; event = 0 means censored (runout)
    df["event"] = 1 - df["runout"]
    df = df.dropna(subset=["cycles", "stress_amplitude"])
    df = df[df["cycles"] > 0]
    df = df[df["stress_amplitude"] > 0]
    return df.reset_index(drop=True)


def _clean_params(df: pd.DataFrame) -> pd.DataFrame:
    """Select, rename, and lightly clean the parameter sheet."""
    available = {k: v for k, v in PARAM_COLS.items() if k in df.columns}
    df = df.rename(columns=available)[list(available.values())]
    # Standardise material type
    df["material_type"] = (
        df["material_type"]
        .str.strip()
        .str.lower()
        .map(
            lambda x: "MG"
            if "metallic glass" in str(x)
            else ("MPEA" if "multi" in str(x) or "principal" in str(x) else "unknown")
        )
    )
    return df.reset_index(drop=True)


def merge_sn_with_params(
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge S-N data points with alloy metadata on dataset_id.

    Parameters
    ----------
    raw : dict returned by load_raw_data()

    Returns
    -------
    Merged DataFrame ready for preprocessing
    """
    sn = _clean_sn(raw["sn"])
    params = _clean_params(raw["parameter"])

    merged = sn.merge(params, on="dataset_id", how="left")
    logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} cols")

    # Basic sanity check
    unknown = merged["material_type"].isna().sum()
    if unknown:
        logger.warning(f"{unknown} rows have unknown material_type — filled with 'unknown'")
        merged["material_type"] = merged["material_type"].fillna("unknown")

    return merged


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """
    High-level convenience function: load + merge in one call.

    Parameters
    ----------
    filepath : str or Path
        Path to FatigueData-CMA2022.xlsx

    Returns
    -------
    Fully merged DataFrame
    """
    raw = load_raw_data(filepath)
    return merge_sn_with_params(raw)
