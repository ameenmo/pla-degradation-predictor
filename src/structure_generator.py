"""
Stage 1: Structure Generation

This module handles the generation of 3D PLA/PLGA oligomer structures,
as well as the reactant and product geometries for the hydrolysis reaction.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs
from ase.io import read as ase_read
import tempfile
import os

def build_pla_oligomer(n_units: int, end_group: str = "OH"):
    """
    Build a PLA oligomer of n repeat units using RDKit,
    return as an ASE Atoms object.
    
    Args:
        n_units: number of lactic acid repeat units (2-12 recommended)
        end_group: "OH" (hydroxyl) or "COOH" (carboxyl) for hydrophilicity tuning
        
    Returns:
        ase.Atoms object representing the 3D structure
    """
    # Simplified SMILES for poly(L-lactic acid) n-mer
    repeat = "C(C)(=O)O"
    smiles = "O" + (repeat * n_units) + "C(C)(=O)O"
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to generate molecule from SMILES")
        
    mol = AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Write to temporary SDF, read back with ASE
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        writer = Chem.SDWriter(f.name)
        writer.write(mol)
        writer.close()
        atoms = ase_read(f.name)
        
    os.unlink(f.name)
    return atoms

if __name__ == "__main__":
    # Test structure generation
    atoms = build_pla_oligomer(3)
    print(f"Generated PLA trimer with {len(atoms)} atoms.")
