# AeroGuard Data Directory

## Dataset Provenance
- **Dataset:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan Engine Degradation Simulation Dataset.
- **Source:** NASA Ames Prognostics Center of Excellence (PCoE), NASA Open Data Portal.
- **Reference:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, Proc. 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, 2008.
- **Subset:** **FD001**
  - Operating Conditions: 1 (Sea Level)
  - Fault Mode: High-Pressure Compressor (HPC) Degradation
  - Training Fleet: 100 run-to-failure engines (20,631 cycles)
  - Test Fleet: 100 truncated operational engines (13,096 cycles)
  - Ground-truth Target: Final-cycle RUL vector for each test engine (`RUL_FD001.txt`)

## Directory Layout
- `data/raw/`: Read-only source files downloaded directly from NASA. Ignored by version control.
- `data/processed/`: Intermediate engineered matrices and feature artifacts.

## Data Acquisition
Raw data files are fetched directly from NASA's official portal (`https://data.nasa.gov/docs/legacy/CMAPSSData.zip`).
