"""
Stage 3: Bell-Evans-Polanyi (BEP) Calibration

The BEP principle states that for a family of similar reactions, the activation
energy (Ea) scales linearly with the reaction energy (dE_rxn):

    Ea = alpha * dE_rxn + beta

This module fits the BEP line from a table of experimental half-lives
(Makadia & Siegel 2011) combined with UMA-computed reaction energies.

References
----------
- Makadia & Siegel (2011) Polymers 3(3):1377-1397  https://doi.org/10.3390/polym3031377
- Codari et al. (2012) Polym. Degrad. Stab. 97:2460-2466
- Lyu et al. (2007) Biomacromolecules 8:2301-2310
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit

# Import the self-consistent mock energy table from uma_calculator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from uma_calculator import _MOCK_ENERGIES

# ---------------------------------------------------------------------------
# Calibration dataset
# Source: Makadia & Siegel 2011 (Table 1), Codari 2012 (Ea values)
# ---------------------------------------------------------------------------
CALIBRATION_DATA = [
    # (name,        n_la, n_ga, t_half_weeks, Ea_kJ_per_mol)
    # Ea from Codari 2012 and Lyu 2007
    ("PGA",         0,    3,    3.0,          50.0),
    ("PLGA50:50",   2,    2,    8.0,          58.0),
    ("PLGA75:25",   3,    1,    26.0,         63.0),
    ("PLGA85:15",   3,    1,    65.0,         67.0),
    ("PLLA",        3,    0,    200.0,        73.0),
]

kJ_TO_eV = 0.010364   # 1 kJ/mol = 0.010364 eV/molecule
R_GAS    = 8.314e-3   # kJ/(mol·K)
T_BODY   = 310.15     # K (37 °C)


def compute_dE_rxn(n_la: int, n_ga: int) -> float:
    """
    Compute hydrolysis reaction energy (eV) from the mock UMA energy table.

    Reaction: oligomer(n) + H2O  ->  oligomer(n-1) + monomer
    For PLA: monomer = lactic acid
    For PGA: monomer = glycolic acid
    """
    E_reactant = _MOCK_ENERGIES.get((n_la, n_ga))
    E_water    = _MOCK_ENERGIES[(0, 0)]

    if E_reactant is None:
        raise ValueError(f"No mock energy for reactant (n_la={n_la}, n_ga={n_ga}). "
                         f"Add it to _MOCK_ENERGIES in uma_calculator.py.")

    # Determine which monomer is cleaved (terminal unit)
    if n_la > 0:
        prod_chain_key    = (n_la - 1, n_ga)
        E_product_monomer = _MOCK_ENERGIES[(1, 0)]   # lactic acid
    else:
        prod_chain_key    = (0, n_ga - 1)
        E_product_monomer = _MOCK_ENERGIES[(0, 1)]   # glycolic acid

    E_product_chain = _MOCK_ENERGIES.get(prod_chain_key)
    if E_product_chain is None:
        raise ValueError(
            f"No mock energy for product chain {prod_chain_key}. "
            f"Add it to _MOCK_ENERGIES in uma_calculator.py."
        )

    dE_rxn = (E_product_chain + E_product_monomer) - (E_reactant + E_water)
    return dE_rxn


def _bep_line(dE_rxn, alpha, beta):
    """Linear BEP model: Ea = alpha * dE_rxn + beta"""
    return alpha * dE_rxn + beta


def fit_bep(dE_rxn_vals: np.ndarray, Ea_eV_vals: np.ndarray):
    """
    Fit the BEP linear relationship.

    Returns:
        (alpha, beta, r_squared)
    """
    popt, _ = curve_fit(_bep_line, dE_rxn_vals, Ea_eV_vals)
    alpha, beta = popt
    Ea_pred = _bep_line(dE_rxn_vals, alpha, beta)
    ss_res = np.sum((Ea_eV_vals - Ea_pred) ** 2)
    ss_tot = np.sum((Ea_eV_vals - np.mean(Ea_eV_vals)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return alpha, beta, r2


def build_calibration_table():
    """
    Build the full calibration table with computed dE_rxn values.

    Returns:
        List of dicts with keys: name, n_la, n_ga, t_half_weeks,
        Ea_kJ, Ea_eV, dE_rxn_eV
    """
    rows = []
    for name, n_la, n_ga, t_half_wk, Ea_kJ in CALIBRATION_DATA:
        dE_rxn = compute_dE_rxn(n_la, n_ga)
        rows.append({
            "name":         name,
            "n_la":         n_la,
            "n_ga":         n_ga,
            "t_half_weeks": t_half_wk,
            "Ea_kJ":        Ea_kJ,
            "Ea_eV":        Ea_kJ * kJ_TO_eV,
            "dE_rxn_eV":    dE_rxn,
        })
    return rows


def save_calibration(alpha: float, beta: float, r2: float,
                     output_path: str = None):
    """Save BEP calibration parameters to a JSON file."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "bep_calibration.json"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    params = {"alpha": alpha, "beta": beta, "r_squared": r2}
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    return output_path


def load_calibration(path: str = None) -> dict:
    """Load BEP calibration parameters from a JSON file."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "bep_calibration.json"
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("  Stage 3: BEP Calibration")
    print("=" * 60)

    # 1. Build calibration table
    print("\n1. Computing dE_rxn for each calibration polymer...")
    rows = build_calibration_table()

    print(f"\n{'Polymer':<14} {'n_la':>4} {'n_ga':>4} {'t_half (wk)':>12} "
          f"{'Ea (kJ/mol)':>12} {'Ea (eV)':>9} {'dE_rxn (eV)':>12}")
    print("-" * 72)
    for r in rows:
        print(f"{r['name']:<14} {r['n_la']:>4} {r['n_ga']:>4} "
              f"{r['t_half_weeks']:>12.1f} {r['Ea_kJ']:>12.1f} "
              f"{r['Ea_eV']:>9.4f} {r['dE_rxn_eV']:>12.4f}")

    # 2. Fit BEP line
    print("\n2. Fitting BEP linear relationship (Ea = alpha * dE_rxn + beta)...")
    dE_rxn_arr = np.array([r["dE_rxn_eV"] for r in rows])
    Ea_eV_arr  = np.array([r["Ea_eV"]     for r in rows])

    alpha, beta, r2 = fit_bep(dE_rxn_arr, Ea_eV_arr)

    print(f"\n   BEP fit result:")
    print(f"   alpha (slope)     = {alpha:.4f}")
    print(f"   beta  (intercept) = {beta:.4f} eV")
    print(f"   R^2               = {r2:.4f}")

    if r2 < 0.85:
        print("   WARNING: R^2 < 0.85 — fit quality is poor.")
    else:
        print("   Fit quality: GOOD")

    # 3. Show predicted vs actual
    print("\n3. Predicted vs actual Ea:")
    print(f"{'Polymer':<14} {'Ea_actual (eV)':>15} {'Ea_pred (eV)':>14} {'Error (%)':>10}")
    print("-" * 56)
    for r in rows:
        Ea_pred = alpha * r["dE_rxn_eV"] + beta
        err_pct = 100 * abs(Ea_pred - r["Ea_eV"]) / r["Ea_eV"]
        print(f"{r['name']:<14} {r['Ea_eV']:>15.4f} {Ea_pred:>14.4f} {err_pct:>10.1f}%")

    # 4. Save calibration
    out_path = save_calibration(alpha, beta, r2)
    print(f"\n4. Calibration saved to: {out_path}")
    print(f"\nStage 3 complete.")
