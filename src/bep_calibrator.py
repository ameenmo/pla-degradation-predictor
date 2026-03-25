"""
Stage 3: BEP Calibration

This module fits a linear relationship between UMA-computed reaction energies
and experimental activation energies from the literature using the 
Bell-Evans-Polanyi (BEP) principle.
"""

import numpy as np
from scipy.optimize import curve_fit

def fit_bep_line(delta_e_vals, ea_lit_vals):
    """
    Fits the BEP linear relationship: Ea = alpha * delta_E + beta
    
    Args:
        delta_e_vals: List/array of UMA-computed reaction energies (eV)
        ea_lit_vals: List/array of experimental activation energies (kJ/mol)
        
    Returns:
        alpha, beta: Fitted parameters
    """
    def bep_line(x, alpha, beta):
        return alpha * x + beta
        
    (alpha, beta), _ = curve_fit(bep_line, delta_e_vals, ea_lit_vals)
    return alpha, beta

if __name__ == "__main__":
    # Example placeholder data
    bep_data = {
        "PGA":       {"delta_E": -0.45, "Ea_lit": 55.0},
        "PLGA_50":   {"delta_E": -0.38, "Ea_lit": 62.0},
        "PLGA_75":   {"delta_E": -0.31, "Ea_lit": 70.0},
        "PLGA_85":   {"delta_E": -0.25, "Ea_lit": 75.0},
        "PLLA":      {"delta_E": -0.18, "Ea_lit": 83.0},
    }
    
    delta_e = np.array([v["delta_E"] for v in bep_data.values()])
    ea_lit = np.array([v["Ea_lit"] for v in bep_data.values()])
    
    alpha, beta = fit_bep_line(delta_e, ea_lit)
    print(f"BEP fit: Ea = {alpha:.2f} * ΔE + {beta:.2f} kJ/mol")
