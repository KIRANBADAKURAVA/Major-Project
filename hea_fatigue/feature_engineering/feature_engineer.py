"""
Feature Engineering Module
============================
Computes thermodynamic descriptors from alloy composition strings:

  • VEC    – Valence Electron Concentration
  • ΔSmix  – Entropy of mixing  (-R Σ ci ln ci)
  • ΔHmix  – Enthalpy of mixing (Miedema pairwise model)
  • n_elements – number of principal elements
"""

import logging
import re
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

R_GAS = 8.314  # J / (mol·K)

# ─── Elemental data tables ────────────────────────────────────────────────────
# Valence electron concentrations (electrons per atom)
# Source: standard condensed-matter convention
VEC_TABLE: dict[str, float] = {
    "H": 1,  "He": 2,
    "Li": 1, "Be": 2, "B": 3,  "C": 4,  "N": 5,  "O": 6,  "F": 7,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5,  "S": 6,  "Cl": 7,
    "K": 1,  "Ca": 2,
    "Sc": 3, "Ti": 4, "V": 5,  "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9,
    "Ni": 10,"Cu": 11,"Zn": 12,
    "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
    "Rb": 1, "Sr": 2,
    "Y": 3,  "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9,
    "Pd": 10,"Ag": 11,"Cd": 12,
    "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
    "Cs": 1, "Ba": 2,
    "La": 3, "Ce": 4, "Pr": 3, "Nd": 3, "Sm": 3, "Gd": 3, "Tb": 3,
    "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Hf": 4, "Ta": 5, "W": 6,  "Re": 7, "Os": 8, "Ir": 9, "Pt": 10,
    "Au": 11,"Hg": 12,
    "Tl": 3, "Pb": 4, "Bi": 5,
    "Th": 4, "U": 6,
}

# Miedema pairwise mixing enthalpies (kJ/mol) — selected common pairs
# Extended symmetric table; zero where data not available (conservative)
# Sources: Zhang et al. (2008) Adv. Eng. Mater.; compiled literature values
_MIEDEMA_RAW: list[tuple[str, str, float]] = [
    ("Al", "Co",  -19.0), ("Al", "Cr",  -10.0), ("Al", "Cu",  -1.0),
    ("Al", "Fe",  -11.0), ("Al", "Mg",   -2.0), ("Al", "Mn",  -19.0),
    ("Al", "Mo",  -22.0), ("Al", "Nb",  -18.0), ("Al", "Ni",  -22.0),
    ("Al", "Si",   11.0), ("Al", "Ti",  -30.0), ("Al", "V",   -16.0),
    ("Al", "W",   -22.0), ("Al", "Zr",  -44.0), ("Al", "Cu",   -1.0),
    ("Co", "Cr",   -4.0), ("Co", "Cu",    6.0), ("Co", "Fe",   -1.0),
    ("Co", "Mn",   -5.0), ("Co", "Mo",   -5.0), ("Co", "Ni",    0.0),
    ("Co", "Ti",  -28.0), ("Co", "V",   -14.0), ("Co", "W",   -11.0),
    ("Co", "Zr",  -41.0),
    ("Cr", "Cu",   12.0), ("Cr", "Fe",    -1.0), ("Cr", "Mn",   0.0),
    ("Cr", "Mo",    0.0), ("Cr", "Ni",  -7.0), ("Cr", "Ti",  -7.0),
    ("Cr", "V",    -2.0), ("Cr", "W",    0.0), ("Cr", "Zr",  -12.0),
    ("Cu", "Fe",   13.0), ("Cu", "Mn",   4.0), ("Cu", "Mo",   19.0),
    ("Cu", "Ni",    4.0), ("Cu", "Ti",  -9.0), ("Cu", "V",    14.0),
    ("Cu", "W",    22.0), ("Cu", "Zr",  -23.0),
    ("Fe", "Mn",    0.0), ("Fe", "Mo",   -2.0), ("Fe", "Nb",  -16.0),
    ("Fe", "Ni",   -2.0), ("Fe", "Ti",  -17.0), ("Fe", "V",   -7.0),
    ("Fe", "W",    0.0), ("Fe", "Zr",  -25.0),
    ("Mn", "Mo",    0.0), ("Mn", "Ni",  -8.0), ("Mn", "Ti",  -8.0),
    ("Mn", "V",    -1.0), ("Mn", "W",    1.0), ("Mn", "Zr",  -4.0),
    ("Mo", "Nb",   -6.0), ("Mo", "Ni",  -4.0), ("Mo", "Ti",  -4.0),
    ("Mo", "V",    -5.0), ("Mo", "W",   0.0), ("Mo", "Zr",  -6.0),
    ("Nb", "Ni",  -30.0), ("Nb", "Ti",    2.0), ("Nb", "V",   -2.0),
    ("Nb", "W",   -7.0), ("Nb", "Zr",   4.0),
    ("Ni", "Ti",  -35.0), ("Ni", "V",   -18.0), ("Ni", "W",   -9.0),
    ("Ni", "Zr",  -49.0),
    ("Si", "Zr",  -67.0), ("Si", "Ti",  -66.0), ("Si", "Fe",  -26.0),
    ("Si", "Ni",  -40.0), ("Si", "Cu",   -19.0),
    ("Ti", "V",    -2.0), ("Ti", "W",   -27.0), ("Ti", "Zr",    0.0),
    ("V",  "W",   -8.0), ("V",  "Zr",   -4.0),
    ("W",  "Zr",  -45.0),
    ("Be", "Zr",  -43.0), ("Be", "Ti",  -30.0), ("Be", "Cu",  -23.0),
    ("Be", "Ni",  -23.0), ("Be", "Co",  -25.0),
    ("Ca", "Mg",   -6.0), ("Ca", "Zn",  -10.0), ("Mg", "Zn",   4.0),
    ("Gd", "Ni",  -22.0), ("Gd", "Fe",   -9.0), ("Gd", "Al", -16.0),
]

# Build symmetric lookup dict
MIEDEMA: dict[frozenset, float] = {}
for e1, e2, val in _MIEDEMA_RAW:
    key = frozenset({e1, e2})
    MIEDEMA[key] = val


# ─── Composition parser ───────────────────────────────────────────────────────
def parse_composition(comp_str: str) -> Optional[dict[str, float]]:
    """
    Parse a composition string like 'Zr55.0-Cu30.0-Al10.0-Ni5.0'
    into a dict {element: fraction}.

    Returns None if parsing fails.
    """
    if not isinstance(comp_str, str) or not comp_str.strip():
        return None

    # Pattern: Element + optional decimal number
    pattern = r"([A-Z][a-z]?)(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, comp_str)
    if not matches:
        return None

    raw = {elem: float(pct) for elem, pct in matches}
    total = sum(raw.values())
    if total <= 0:
        return None

    # Normalise to fractions (at%)/(100) → fractions sum to 1
    return {elem: pct / total for elem, pct in raw.items()}


# ─── Descriptor functions ─────────────────────────────────────────────────────
def compute_VEC(fractions: dict[str, float]) -> float:
    """
    Valence Electron Concentration:  VEC = Σ ci × VECi
    """
    return sum(ci * VEC_TABLE.get(el, 0.0) for el, ci in fractions.items())


def compute_delta_S_mix(fractions: dict[str, float]) -> float:
    """
    Entropy of mixing:  ΔSmix = -R Σ (ci × ln ci)   [J/(mol·K)]
    """
    vals = [ci for ci in fractions.values() if ci > 0]
    return -R_GAS * sum(ci * np.log(ci) for ci in vals)


def compute_delta_H_mix(fractions: dict[str, float]) -> float:
    """
    Enthalpy of mixing (Miedema pairwise model):
    ΔHmix = Σ_i Σ_{j>i} 4 × ci × cj × ΔHij   [kJ/mol]
    """
    elements = list(fractions.keys())
    total = 0.0
    for i, e1 in enumerate(elements):
        for e2 in elements[i + 1:]:
            key = frozenset({e1, e2})
            h_ij = MIEDEMA.get(key, 0.0)
            total += 4.0 * fractions[e1] * fractions[e2] * h_ij
    return total


# ─── Main pipeline ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, composition_col: str = "composition") -> pd.DataFrame:
    """
    Add VEC, delta_S_mix, delta_H_mix, n_elements to df.

    Parameters
    ----------
    df : DataFrame with a composition string column
    composition_col : name of the composition column

    Returns
    -------
    df with new columns added in-place
    """
    df = df.copy()

    vec_vals, ds_vals, dh_vals, nel_vals = [], [], [], []

    for comp_str in df.get(composition_col, pd.Series(dtype=str)):
        fractions = parse_composition(comp_str)
        if fractions is None:
            vec_vals.append(np.nan)
            ds_vals.append(np.nan)
            dh_vals.append(np.nan)
            nel_vals.append(np.nan)
        else:
            vec_vals.append(compute_VEC(fractions))
            ds_vals.append(compute_delta_S_mix(fractions))
            dh_vals.append(compute_delta_H_mix(fractions))
            nel_vals.append(float(len(fractions)))

    df["VEC"] = vec_vals
    df["delta_S_mix"] = ds_vals
    df["delta_H_mix"] = dh_vals
    df["n_elements"] = nel_vals

    n_parsed = df["VEC"].notna().sum()
    logger.info(
        f"Feature engineering: {n_parsed}/{len(df)} rows have valid compositions "
        f"(VEC, ΔSmix, ΔHmix computed)"
    )

    return df
