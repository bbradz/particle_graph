# RL Builder 2.0 - Particle Physics Model Generator and Observable Calculator

A comprehensive Python framework for generating particle physics models from JSON specifications and computing observables using SARAH/SPheno. This tool automates the process of creating Beyond the Standard Model (BSM) theories and performing parameter space scans with statistical analysis.

## Overview

This project consists of two main components:

1. **Token2Model**: A model generation system that converts JSON model specifications into SARAH-compatible Mathematica files
2. **CalcObs**: An observable calculation system that uses SARAH/SPheno to compute particle masses, couplings, and perform chi-squared minimization

## Project Structure

```
RL_builder-2.0/
├── main.py                 # Main execution script
├── test.py                 # Simple test script
├── obs_list.json          # Observable definitions and experimental values
├── SM_test.json           # Standard Model JSON specification
├── SM.json                # Alternative SM specification
├── Token2Model/           # Model generation module
│   ├── model.py           # Main Model class
│   ├── field.py           # Field definitions and handling
│   ├── particle.py        # Particle definitions
│   ├── interaction.py     # Interaction terms
│   ├── group.py           # Gauge group definitions
│   ├── param.py           # Parameter handling
│   ├── name.py            # Naming conventions
│   ├── utility.py         # Utility functions
│   ├── vev.py             # VEV handling
│   ├── write_model_files.py # File writing utilities
│   ├── _pdg.json          # PDG particle data
│   ├── sm_particles.json  # Standard Model particles
│   └── sm_parameters.json # Standard Model parameters
├── CalcObs/               # Observable calculation module
│   ├── observables.py     # Main ObservableCalc class
│   └── calc_spheno.m      # Mathematica script for SARAH
├── Models/                # Generated model files
│   ├── NM/               # Example model (New Model)
│   └── TNM/              # Example model (Test New Model)
└── .Old_Models/          # Legacy model files
```

## Installation and Dependencies

### Prerequisites

1. **Python 3.7+** with the following packages:
   ```bash
   pip install numpy scipy matplotlib
   ```

2. **Mathematica** (for SARAH integration)

3. **SARAH 4.15.4** - Download from [SARAH website](https://sarah.hepforge.org/)

4. **SPheno 4.0.5** - Download from [SPheno website](https://spheno.hepforge.org/)

5. **Fortran compiler** (gfortran recommended)

### Setup

1. Update paths in `main.py`:
   ```python
   SARAH_PATH = "/path/to/SARAH-4.15.4"
   SPHENO_PATH = "/path/to/SPheno-4.0.5"
   ```
2. The code requires mathematica. One can load it in SSH by
    ```
    load module mathematica
    ```

## Usage

### 1. Model Generation

The system reads model specifications from JSON files and generates SARAH-compatible Mathematica files.

#### JSON Model Format

```json
{
    "GaugeGroups": [
        {
            "id": "g1",
            "name": "hypercharge",
            "charge": "Y",
            "group": "U_1",
            "boson": "B"
        }
    ],
    "particles": [
        {
            "id": "f1",
            "type": "fermion",
            "name": "t",
            "mass": [150, 500],  // Range for free parameter
            "charge": 2
        }
    ],
    "fields": [
        {
            "id": "field1",
            "type": "fermion",
            "chirality": "left",
            "particles": ["f1", "f2"]
        }
    ],
    "interactions": [
        {
            "id": "yuk1",
            "type": "yukawa",
            "fields": ["field1", "field2", "field3"],
            "coupling": [0.1, 1.0]  // Range for free parameter
        }
    ]
}
```

#### Generating a Model

```python
from Token2Model.model import Model

# Create model from JSON specification
model = Model(
    model_name="My New Model",
    author="Your Name",
    JSON_PATH="path/to/model.json",
    MODEL_BASE_PATH="./Models",
    simplify_checklist=True
)

# Generate SARAH files
model.write_model()
model.write_checklist()
```

### 2. Observable Calculation

The `ObservableCalc` class handles SARAH/SPheno execution and statistical analysis.

#### Observable Definition

Define observables in `obs_list.json`:

```json
{
    "m_W": {
        "LHA_loc": ["Block", "MASS", "24"],
        "measured": 80.447,
        "sigma": 0.042
    },
    "delta_rho": {
        "LHA_loc": ["Block", "SPhenoLowEnergy", "39"],
        "measured": 0.0064,
        "sigma": 0.0018
    }
}
```

#### Running Calculations

```python
from CalcObs.observables import ObservableCalc

# Initialize calculator
calc = ObservableCalc(
    model_name="TNM",
    model_base="./Models",
    obs_list_path="./obs_list.json",
    sarah_path="../SARAH-4.15.4",
    spheno_path="../SPheno-4.0.5",
    sigma_threshold=3,
    keep_log=True,
    include_tachyon=False
)

# Generate SPheno files
calc.run_sarah()
calc.compile_spheno()
calc.minimize_chi2(maxiter=10, popsize=5)
calc.make_plot()
```

## Key Features

### Token2Model Module

- **Automatic Model Generation**: Converts JSON specifications to SARAH Mathematica files
- **Parameter Handling**: Manages free parameters with ranges for scanning
- **Validation System**: Comprehensive checklist for model consistency
- **Multiple Field Types**: Supports fermions, scalars (real/complex), and vector fields
- **Interaction Types**: Handles Yukawa couplings and scalar self-interactions

### CalcObs Module

- **SARAH/SPheno Integration**: Automated execution of particle physics calculations
- **Statistical Analysis**: Chi-squared minimization with differential evolution
- **Observable Tracking**: Monitors individual observable contributions to chi-squared
- **Parameter Scanning**: Efficient parameter space exploration
- **Visualization**: Automatic plotting of results and parameter correlations
- **Early Stopping**: Optimization stops when chi-squared threshold is reached

### Advanced Features

- **Multi-core Support**: Parallel processing for parameter scans
- **Timeout Handling**: Prevents hanging calculations
- **Log Management**: Configurable logging and output retention
- **Error Recovery**: Graceful handling of calculation failures
- **Flexible Input**: Customizable SPheno input parameters

## Configuration Options

### ObservableCalc Parameters

- `timeout`: Maximum execution time for SPheno (default: 1 hour)
- `sigma_threshold`: Confidence level for early stopping (default: 3σ)
- `keep_log`: Retain SPheno output files (default: True)
- `loop_mass`: Include loop corrections for masses (default: True)
- `include_tachyon`: Allow tachyonic states (default: True)
- `calc_decays`: Calculate decay widths (default: False)
- `mass_precision`: Precision for mass calculations (default: 1e-6)

### Minimization Parameters

- `maxiter`: Maximum iterations for differential evolution
- `popsize`: Population size for optimization
- `seed`: Random seed for reproducibility

## Output Files

### Generated Model Files

- `{ModelName}.m`: Main model definition
- `particles.m`: Particle definitions and properties
- `parameters.m`: Parameter definitions and values
- `SPheno.m`: SPheno-specific settings
- `free_params.json`: Free parameter ranges
- `checklist.csv`: Model validation results

### Calculation Results

- `chi2_data.npz`: Chi-squared history and parameter values

## Example Workflow

1. **Define Model**: Create JSON specification for your BSM model
2. **Generate Files**: Use `Model` class to create SARAH files
3. **Define Observables**: Specify experimental constraints in `obs_list.json`
4. **Run Calculations**: Use `ObservableCalc` to compute observables
5. **Parameter Scan**: Perform chi-squared minimization
6. **Analyze Results**: Examine plots and parameter correlations

## Troubleshooting

### Common Issues

1. **SARAH/SPheno Paths**: Ensure correct paths in `main.py`
2. **Mathematica License**: Verify Mathematica is properly licensed
3. **Fortran Compiler**: Check gfortran installation
4. **Memory Issues**: Reduce `popsize` for large parameter spaces
5. **Timeout Errors**: Increase `timeout` parameter for complex models

### Debugging

- Enable `keep_log=True` to retain SPheno output files
- Check `checklist.csv` for model validation issues
- Monitor chi-squared convergence in plots
- Verify observable definitions in `obs_list.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

[Add citation information here]

## Contact

[Add contact information here] 