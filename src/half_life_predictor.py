"""
Stage 4: Half-Life Prediction

This module applies the Arrhenius equation to predict the degradation
half-life at a given temperature based on the calibrated activation energy.
"""

import numpy as np

def predict_half_life(ea_kj_mol, t_celsius=37.0, pre_exponential_factor=1.2e8):
    """
    Predict degradation half-life in weeks.
    
    Args:
        ea_kj_mol: Activation energy in kJ/mol
        t_celsius: Temperature in Celsius (default: 37.0 for body temp)
        pre_exponential_factor: Arrhenius pre-exponential factor A (in weeks^-1)
        
    Returns:
        Half-life in weeks
    """
    t_kelvin = t_celsius + 273.15
    r_gas = 8.314e-3  # kJ/(mol*K)
    
    # Arrhenius equation: k = A * exp(-Ea / RT)
    k = pre_exponential_factor * np.exp(-ea_kj_mol / (r_gas * t_kelvin))
    
    # Half-life for first-order kinetics: t_1/2 = ln(2) / k
    half_life_weeks = np.log(2) / k
    
    return half_life_weeks

if __name__ == "__main__":
    # Test with a typical Ea for PLGA
    ea_test = 65.0  # kJ/mol
    t_half = predict_half_life(ea_test)
    print(f"Predicted half-life for Ea={ea_test} kJ/mol: {t_half:.1f} weeks")
