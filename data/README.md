# Experimental Calibration Data

This directory contains experimental data extracted from the literature, used to calibrate the UMA-computed reaction energies into absolute degradation half-lives via the Bell-Evans-Polanyi (BEP) principle.

## Required Data

To run the calibration pipeline, you need to populate a CSV file here with experimental half-life values for various PLA/PLGA compositions.

### Sources

The key values can be found in the following peer-reviewed papers:

1. **Makadia & Siegel (2011)**, *Polymers* 3(3):1377–1397.
   - Contains a comprehensive degradation rate table for PLGA of different ratios.
   - Free full text: [https://www.mdpi.com/2073-4360/3/3/1377](https://www.mdpi.com/2073-4360/3/3/1377)

2. **Kamaly et al. (2016)**, *Chemical Reviews* 116(4):2602–2663.
   - Table 1 lists degradation times for PLGA 50:50, 75:25, and 85:15.
   - Free full text: [https://pmc.ncbi.nlm.nih.gov/articles/PMC5509216/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5509216/)

3. **Codari et al. (2012)**, *Polymer Degradation and Stability* 97(11):2460–2466.
   - Provides Arrhenius activation energies (58–73 kJ/mol) for PLA hydrolysis, useful for validating the BEP slope.
   - DOI: 10.1016/j.polymdegradstab.2012.03.009

### Data Format

Create a file named `calibration_data.csv` with the following columns:
- `polymer_type` (e.g., "PLGA_50_50")
- `la_fraction` (e.g., 0.5)
- `ga_fraction` (e.g., 0.5)
- `approx_mw_kda` (e.g., 50.0)
- `half_life_weeks_37c` (e.g., 6.0)
- `source_reference` (e.g., "Makadia_2011")
