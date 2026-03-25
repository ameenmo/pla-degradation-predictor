"""
Stage 2: UMA Energy Calculation

This module uses Meta's UMA (Universal Models for Atoms) model to compute
the ground-state energy of PLA oligomers. It performs geometry relaxation
to find the local energy minimum.
"""

import os
import time

try:
    from ase.optimize import LBFGS
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e}. "
        "Please ensure fairchem-core and ase are installed."
    )

def setup_calculator(model_name: str = "uma-s-1p2", device: str = "cpu") -> "FAIRChemCalculator":
    """
    Initialize the UMA calculator.
    
    Args:
        model_name: The UMA checkpoint to use (default: 'uma-s-1p2' for MVP)
        device: 'cpu' or 'cuda'
        
    Returns:
        A FAIRChemCalculator instance configured for molecules/polymers
    """
    # Check for Hugging Face token in environment
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN environment variable not found.")
        print("If this is your first time running UMA, it may fail to download the model.")
        print("Run: huggingface-cli login, or set export HF_TOKEN='your_token'")
        print("-" * 50)
        
    print(f"Loading UMA model '{model_name}' on {device.upper()}...")
    try:
        # Load the pre-trained predictor
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        
        # task_name="omol" is specifically calibrated for molecules and polymers
        # (based on the OMol25 and OPoly26 datasets)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        return calc
    except Exception as e:
        raise RuntimeError(f"Failed to load UMA model: {e}")

def relax_and_calculate_energy(atoms: "ase.Atoms", calc: "FAIRChemCalculator", 
                               fmax: float = 0.05, max_steps: int = 100) -> float:
    """
    Perform geometry relaxation on an ASE Atoms object and return its energy.
    
    Args:
        atoms: The ASE Atoms object to relax
        calc: The initialized FAIRChemCalculator
        fmax: Maximum force tolerance for convergence (eV/Å)
        max_steps: Maximum number of optimization steps
        
    Returns:
        The total potential energy of the relaxed structure in eV
    """
    # Attach the calculator to the atoms object
    atoms.calc = calc
    
    # Set up the LBFGS optimizer
    # We set logfile=None to avoid cluttering the terminal, but in production
    # you might want to log this to a file
    opt = LBFGS(atoms, logfile=None)
    
    print(f"Starting geometry relaxation (max_steps={max_steps}, fmax={fmax})...")
    start_time = time.time()
    
    # Run the optimization
    opt.run(fmax=fmax, steps=max_steps)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Get the final energy
    energy = atoms.get_potential_energy()
    
    print(f"Relaxation complete in {elapsed:.1f} seconds.")
    print(f"Final Energy: {energy:.4f} eV")
    
    return energy

if __name__ == "__main__":
    import sys
    
    print("--- Stage 2: UMA Energy Calculation Test ---")
    
    # Import the structure generator from the same directory
    try:
        from structure_generator import build_oligomer
    except ImportError:
        # If running directly, make sure we can find the module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from structure_generator import build_oligomer
        
    # 1. Build a PLA trimer (n=3)
    print("1. Generating PLA trimer...")
    atoms = build_oligomer(n_la=3, n_ga=0)
    print(f"   Generated structure with {len(atoms)} atoms.")
    
    # 2. Setup UMA calculator
    print("\n2. Initializing UMA calculator...")
    try:
        # Using CPU for maximum compatibility during testing
        calc = setup_calculator(model_name="uma-s-1p2", device="cpu")
    except Exception as e:
        print(f"Error setting up calculator: {e}")
        sys.exit(1)
        
    # 3. Run relaxation and get energy
    print("\n3. Running UMA geometry relaxation...")
    try:
        # We use a slightly looser fmax (0.1) and fewer steps (50) for the test
        # to ensure it finishes quickly on CPU
        energy = relax_and_calculate_energy(atoms, calc, fmax=0.1, max_steps=50)
        print(f"\nSUCCESS! Final relaxed energy for PLA trimer: {energy:.4f} eV")
    except Exception as e:
        print(f"Error during relaxation: {e}")
        sys.exit(1)
