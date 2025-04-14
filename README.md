# xopt-genesis-data-gen
This directory contains an example workflow using to run batch Genesis simulations and generate data for training neural network surrogate models. 

## Parameters Optimized

Xopt varies the following input parameters:
- **Twiss parameters**: `alphax`, `beta', `emittance'
- **Undulator taper**: a quadratic taper starting at a given point
- **Beam chirp**: energy–time correlation
- **Beam length**: total duration of the electron bunch

## Output

The simulation outputs are store in Xopt yml file.
- FEL power
- Spectrum
- Peak energy, intensity
- Energy spread, bandwidth, etc.

These outputs are suitable for use as training targets for neural network surrogate models.

More data is stored in folders `processed_inputs/` and `processed_outputs/`. 

## Contents

- `GenerateData.py`: Runs Genesis simulations using Xopt
  - `temp' folder stores Xopt outputs
- `processed_inputs/` and `processed_outputs/`: Store simulation data by fingerprint
