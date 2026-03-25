"""
Stage 4: Half-Life Prediction

Converts UMA-computed reaction energies into degradation half-life predictions
using the BEP calibration and the Arrhenius equation.

Pipeline:
    1. Generate oligomer structure (structure_generator)
    2. Compute dE_rxn via UMA or mock (uma_calculator)
    3. Convert dE_rxn -> Ea via BEP calibration (bep_calibrator)
    4. Apply Arrhenius: k = A * exp(-Ea / RT)
    5. Convert rate constant to half-life: t_half = ln(2) / k

The pre-exponential factor A is calibrated from the experimental half-life
of PLLA (the best-characterised reference polymer).

References
----------
- Makadia & Siegel (2011) Polymers 3(3):1377-1397
- Codari et al. (2012) Polym. Degrad. Stab. 97:2460-2466
"""

import os
import sys
import math
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uma_calculator import (
    _MOCK_MODE, _MOCK_ENERGIES, _mock_energy,
    setup_calculator, relax_and_calculate_energy
)
from bep_calibrator import (
    compute_dE_rxn, fit_bep, build_calibration_table, load_calibration
)
from structure_generator import build_oligomer

# Physical constants
R_kJ    = 8.314e-3   # kJ / (mol·K)
kJ_TO_eV = 0.010364
T_BODY  = 310.15     # K (37 °C, physiological)
LN2     = math.log(2)

# ---------------------------------------------------------------------------
# Calibrate the Arrhenius pre-exponential factor A (s^-1)
# from the PLLA reference: t_half = 200 weeks at 37 °C, Ea = 73 kJ/mol
# k = ln(2) / t_half  =>  A = k * exp(Ea / RT)
# ---------------------------------------------------------------------------
_PLLA_T_HALF_S = 200 * 7 * 24 * 3600   # 200 weeks in seconds
_PLLA_EA_kJ    = 73.0
_k_ref         = LN2 / _PLLA_T_HALF_S
_A_PREEXP      = _k_ref * math.exp(_PLLA_EA_kJ / (R_kJ * T_BODY))


def dE_rxn_from_uma(n_la: int, n_ga: int, calc=None) -> float:
    """
    Compute hydrolysis dE_rxn (eV) using UMA or mock mode.

    Runs energy calculations for: oligomer, product chain, monomer, water.
    """
    # Build structures
    atoms_oligo   = build_oligomer(n_la=n_la, n_ga=n_ga)
    from ase.build import molecule as _ase_mol; atoms_water = _ase_mol("H2O")  # water

    # Determine product chain composition
    if n_la > 0:
        prod_n_la, prod_n_ga = n_la - 1, n_ga
        mono_n_la, mono_n_ga = 1, 0
    else:
        prod_n_la, prod_n_ga = 0, n_ga - 1
        mono_n_la, mono_n_ga = 0, 1

    atoms_chain = build_oligomer(n_la=prod_n_la, n_ga=prod_n_ga)
    atoms_mono  = build_oligomer(n_la=mono_n_la, n_ga=mono_n_ga)

    E_oligo = relax_and_calculate_energy(atoms_oligo, calc, n_la=n_la,      n_ga=n_ga)
    E_water = relax_and_calculate_energy(atoms_water, calc, n_la=0,          n_ga=0)
    E_chain = relax_and_calculate_energy(atoms_chain, calc, n_la=prod_n_la, n_ga=prod_n_ga)
    E_mono  = relax_and_calculate_energy(atoms_mono,  calc, n_la=mono_n_la, n_ga=mono_n_ga)

    return (E_chain + E_mono) - (E_oligo + E_water)


def predict_half_life(n_la: int, n_ga: int = 0,
                      alpha: float = None, beta: float = None,
                      calc=None, T_K: float = T_BODY) -> dict:
    """
    Predict the hydrolysis half-life for a PLA/PLGA oligomer.

    Args:
        n_la:   Number of lactic acid repeat units
        n_ga:   Number of glycolic acid repeat units
        alpha:  BEP slope (loaded from calibration if None)
        beta:   BEP intercept (loaded from calibration if None)
        calc:   FAIRChemCalculator (None = mock mode)
        T_K:    Temperature in Kelvin (default: 37 °C)

    Returns:
        Dict with keys: n_la, n_ga, dE_rxn_eV, Ea_eV, Ea_kJ,
                        k_per_s, t_half_s, t_half_weeks
    """
    # Load BEP calibration if not provided
    if alpha is None or beta is None:
        try:
            params = load_calibration()
            alpha, beta = params["alpha"], params["beta"]
        except FileNotFoundError:
            # Fall back to fitting from scratch
            rows = build_calibration_table()
            dE_arr = np.array([r["dE_rxn_eV"] for r in rows])
            Ea_arr = np.array([r["Ea_eV"]     for r in rows])
            alpha, beta, _ = fit_bep(dE_arr, Ea_arr)

    # Compute dE_rxn
    dE_rxn = dE_rxn_from_uma(n_la, n_ga, calc)

    # BEP: Ea (eV) -> Ea (kJ/mol)
    Ea_eV = alpha * dE_rxn + beta
    Ea_kJ = Ea_eV / kJ_TO_eV

    # Arrhenius: k (s^-1)
    k = _A_PREEXP * math.exp(-Ea_kJ / (R_kJ * T_K))

    # Half-life
    t_half_s     = LN2 / k
    t_half_weeks = t_half_s / (7 * 24 * 3600)

    return {
        "n_la":         n_la,
        "n_ga":         n_ga,
        "dE_rxn_eV":    round(dE_rxn,      4),
        "Ea_eV":        round(Ea_eV,       4),
        "Ea_kJ":        round(Ea_kJ,       2),
        "k_per_s":      k,
        "t_half_s":     t_half_s,
        "t_half_weeks": round(t_half_weeks, 1),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Stage 4: Half-Life Prediction")
    print("=" * 60)
    print(f"Mode: {'MOCK (UMA_MOCK_MODE=1)' if _MOCK_MODE else 'REAL UMA'}")
    print(f"Temperature: {T_BODY - 273.15:.1f} C ({T_BODY:.2f} K)\n")

    # 1. Run BEP calibration
    print("1. Running BEP calibration from literature data...")
    rows = build_calibration_table()
    dE_arr = np.array([r["dE_rxn_eV"] for r in rows])
    Ea_arr = np.array([r["Ea_eV"]     for r in rows])
    alpha, beta, r2 = fit_bep(dE_arr, Ea_arr)
    print(f"   BEP: Ea = {alpha:.4f} * dE_rxn + {beta:.4f}  (R^2 = {r2:.4f})")

    # 2. Setup calculator
    print("\n2. Setting up UMA calculator...")
    calc = setup_calculator(model_name="uma-s-1p2", device="cpu")

    # 3. Predict for a range of PLA/PLGA compositions
    print("\n3. Predicting half-lives for PLA/PLGA variants...")
    test_cases = [
        ("PGA trimer",         0, 3),
        ("PLGA 50:50 tetramer",2, 2),
        ("PLGA 75:25 tetramer",3, 1),
        ("PLA trimer (PLLA)",  3, 0),
    ]

    results = []
    for label, n_la, n_ga in test_cases:
        print(f"\n   [{label}]  n_la={n_la}, n_ga={n_ga}")
        r = predict_half_life(n_la=n_la, n_ga=n_ga,
                              alpha=alpha, beta=beta, calc=calc)
        results.append((label, r))

    # 4. Summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Polymer':<26} {'dE_rxn (eV)':>12} {'Ea (kJ/mol)':>12} "
          f"{'t_half (wk)':>12}  {'t_half (yr)':>12}")
    print("-" * 78)
    for label, r in results:
        t_yr = r["t_half_weeks"] / 52.18
        print(f"{label:<26} {r['dE_rxn_eV']:>12.4f} {r['Ea_kJ']:>12.1f} "
              f"{r['t_half_weeks']:>12.1f}  {t_yr:>12.2f}")

    print("\n  Comparison with literature (Makadia & Siegel 2011):")
    lit = {
        "PGA trimer":          3.0,
        "PLGA 50:50 tetramer": 8.0,
        "PLGA 75:25 tetramer": 26.0,
        "PLA trimer (PLLA)":   200.0,
    }
    print(f"\n{'Polymer':<26} {'Predicted (wk)':>15} {'Literature (wk)':>16} {'Ratio':>8}")
    print("-" * 68)
    for label, r in results:
        pred = r["t_half_weeks"]
        lit_val = lit.get(label, float("nan"))
        ratio = pred / lit_val if lit_val else float("nan")
        print(f"{label:<26} {pred:>15.1f} {lit_val:>16.1f} {ratio:>8.2f}x")

    print(f"\nStage 4 complete.")
    print(f"\nNote: Predictions use mock UMA energies calibrated to literature Ea values.")
    print(f"      With real UMA energies, predictions will update automatically.")
