# PLA Degradation Predictor

A Python tool that uses Meta's UMA (Universal Models for Atoms) to predict the hydrolysis degradation half-life of polylactide (PLA) polymer segments as a function of molecular weight and hydrophilicity.

## Overview

This project provides a computational pipeline to estimate the degradation rate of PLA and PLGA polymers in silico. It avoids the computationally prohibitive task of calculating transition states from scratch. Instead, it uses the **Bell–Evans–Polanyi (BEP) principle**, which states that for a family of similar reactions (like ester hydrolysis in polyesters), the activation energy scales linearly with the reaction energy.

The tool uses Meta's **UMA (`uma-s-1p2`)** model to compute the ground-state reaction energy ($\Delta E_{rxn}$) for the hydrolysis of PLA oligomers. This reaction energy is then calibrated against published experimental half-life values to predict degradation rates for new polymer formulations.

### Why this matters
For medtech companies developing bioresorbable implants, drug delivery vehicles, or tissue scaffolds, tuning the degradation rate of PLA/PLGA is a critical step. Wet-lab degradation studies can take months or years. This tool aims to provide directional predictions in seconds, accelerating the formulation screening process.

## The Science

The pipeline consists of four stages:
1. **Structure Generation:** Builds 3D models of PLA/PLGA oligomers, reactants (oligomer + water), and products (cleaved chains) using RDKit and ASE.
2. **UMA Energy Calculation:** Uses Meta's `uma-s-1p2` model (trained on the OPoly26 dataset) to relax the geometries and compute the hydrolysis reaction energy.
3. **BEP Calibration:** Fits a linear relationship between the UMA-computed reaction energies and experimental activation energies from the literature.
4. **Half-Life Prediction:** Applies the Arrhenius equation to predict the half-life at 37°C in weeks.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ameenmo/pla-degradation-predictor.git
   cd pla-degradation-predictor
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Hugging Face access for UMA:
   - Create an account at [Hugging Face](https://huggingface.co/)
   - Request access to the [UMA repository](https://huggingface.co/facebook/UMA)
   - Create an access token and log in via CLI:
     ```bash
     huggingface-cli login
     ```

## Project Structure

- `src/` - Core Python modules for the 4-stage pipeline.
- `data/` - Experimental calibration data from literature.
- `notebooks/` - Jupyter notebooks for exploration and analysis.

## Roadmap

- [ ] Stage 1: Implement robust RDKit SMILES generation for PLA/PLGA oligomers.
- [ ] Stage 2: Set up UMA geometry relaxation pipeline.
- [ ] Stage 3: Populate experimental calibration data and fit BEP line.
- [ ] Stage 4: Build Streamlit demo interface for rapid formulation screening.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
