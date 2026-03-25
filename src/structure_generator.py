"""
Stage 1: Structure Generation

This module handles the generation of 3D PLA/PLGA oligomer structures
using RDKit and exports them as ASE Atoms objects for use with UMA.

To keep computational costs low, we use the 'capped oligomer' approach:
an acetyl group caps the N-terminus and a hydroxyl caps the C-terminus.
"""

import os
import tempfile
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdmolops import AddHs
    from ase.io import read as ase_read
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e}. "
        "Please ensure rdkit-pypi and ase are installed."
    )

def build_plga_smiles(n_la: int, n_ga: int = 0) -> str:
    """
    Build a SMILES string for a PLA or PLGA oligomer.
    
    The sequence will be all L-lactic acid (LA) units followed by 
    all glycolic acid (GA) units. The oligomer is capped with an 
    acetyl group at the start to avoid a reactive free radical, 
    and ends with a standard hydroxyl group.
    
    Args:
        n_la: Number of L-lactic acid repeat units
        n_ga: Number of glycolic acid repeat units (0 for pure PLA)
        
    Returns:
        A valid SMILES string representing the capped oligomer
    """
    if n_la + n_ga < 1:
        raise ValueError("Total number of repeat units must be at least 1")
        
    # Acetyl cap: CH3-C(=O)-O-
    smiles = "CC(=O)O"
    
    # Add L-lactic acid units: -[C@@H](CH3)-C(=O)-O-
    # Note: the trailing 'O' acts as the ester linkage to the next unit
    for _ in range(n_la):
        smiles += "[C@@H](C)C(=O)O"
        
    # Add glycolic acid units: -CH2-C(=O)-O-
    for _ in range(n_ga):
        smiles += "CC(=O)O"
        
    # The final 'O' in the loop becomes the terminal -OH group,
    # which is chemically correct for the C-terminus of these polyesters.
    return smiles

def generate_ase_atoms_from_smiles(smiles: str) -> "ase.Atoms":
    """
    Convert a SMILES string into a 3D ASE Atoms object.
    
    This function uses RDKit to embed the molecule in 3D space,
    adds explicit hydrogens, and performs a quick force-field
    relaxation (MMFF94) to ensure sensible starting geometries
    before passing to the UMA quantum model.
    
    Args:
        smiles: The SMILES string to convert
        
    Returns:
        An ase.Atoms object containing the 3D structure
    """
    # 1. Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        
    # 2. Add explicit hydrogens
    mol_h = AddHs(mol)
    
    # 3. Generate 3D coordinates using ETKDG (standard RDKit method)
    # We suppress warnings here as RDKit can be noisy during embedding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embed_result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        
    if embed_result == -1:
        raise RuntimeError("RDKit failed to generate 3D coordinates for the molecule.")
        
    # 4. Perform MMFF94 force-field optimization for a good starting guess
    AllChem.MMFFOptimizeMolecule(mol_h)
    
    # 5. Convert to ASE Atoms via a temporary SDF file
    # RDKit -> SDF string -> Temp File -> ASE read
    sdf_block = Chem.MolToMolBlock(mol_h)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        f.write(sdf_block)
        temp_filename = f.name
        
    try:
        atoms = ase_read(temp_filename, format='sdf')
    finally:
        # Ensure cleanup even if ASE read fails
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
            
    return atoms

def build_oligomer(n_la: int, n_ga: int = 0) -> "ase.Atoms":
    """
    Main entry point: Build a 3D ASE Atoms object for a PLA/PLGA oligomer.
    
    Args:
        n_la: Number of L-lactic acid repeat units
        n_ga: Number of glycolic acid repeat units
        
    Returns:
        An ase.Atoms object ready for UMA calculation
    """
    smiles = build_plga_smiles(n_la, n_ga)
    return generate_ase_atoms_from_smiles(smiles)

if __name__ == "__main__":
    print("--- Stage 1: Structure Generation Test ---")
    
    # Generate a pure PLA trimer (n=3)
    # This is the recommended size for the MVP to keep calculations under 50 atoms
    n_units = 3
    print(f"Building capped PLA trimer (n={n_units})...")
    
    try:
        smiles = build_plga_smiles(n_la=n_units, n_ga=0)
        print(f"SMILES: {smiles}")
        
        atoms = build_oligomer(n_la=n_units, n_ga=0)
        print(f"Success! Generated ASE Atoms object with {len(atoms)} atoms.")
        print(f"Chemical formula: {atoms.get_chemical_formula()}")
        
        # Verify it's under the 50 atom limit for fast CPU inference
        if len(atoms) <= 50:
            print("Status: OK (Under 50 atoms, suitable for fast CPU inference)")
        else:
            print("Status: WARNING (Over 50 atoms, CPU inference will be slow)")
            
    except Exception as e:
        print(f"Error during generation: {e}")
