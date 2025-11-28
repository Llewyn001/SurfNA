# SurfNA 🧬


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange.svg)](https://pytorch.org/)


## ✨ Features

- 🔬 **Complete Docking Pipeline** - Full nucleic acid-protein docking evaluation workflow
- 📊 **Confidence Scoring** - Built-in confidence model for pose ranking
- 🧮 **Nucleic Acid Surface Computation** - Specialized surface calculation module with modified APBS tools
- 🧪 **Test Samples** - Included test cases for quick validation
- 🚀 **Easy Deployment** - Conda-based environment, minimal dependencies
- 📝 **Well Documented** - Comprehensive documentation and examples

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Nucleic Acid Surface Computation](#nucleic-acid-surface-computation)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1 (for GPU acceleration)
- Conda or Mamba

### Step-by-Step Installation

1. **Clone or extract the project**
   ```bash
   git clone https://github.com/yourusername/SurfNA.git
   cd SurfNA
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate SurfNA
   ```

3. **Set environment variables**
   ```bash
   export precomputed_arrays="/path/to/SurfNA/precomputed_arrays"
   mkdir -p precomputed_arrays
   ```

4. **Prepare model weights**
   - Copy model weights to `model_weights/` directory
   - Ensure `score_model/` and `confidence_model/` directories contain the required `.pt` files

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 🎯 Quick Start

### Docking Evaluation

**Option 1: Using the run script**
```bash
bash run_eval.sh
```

**Option 2: Manual execution**
```bash
accelerate launch --num_processes 1 evaluate_accelarate.py \
    --model_dir model_weights/score_model \
    --ckpt best_ema_inference_epoch_model.pt \
    --confidence_model_dir model_weights/confidence_model \
    --confidence_ckpt best_ema_model.pt \
    --cache_path data/cache \
    --data_dir data/testset \
    --surface_path data/testset_surface \
    --split_path data/splits/mini_test_split \
    --no_overlap_names_path data/splits/mini_test_split_no_rec_overlap \
    --out_dir test_workdir \
    --samples_per_complex 10
```

### Surface Computation

See [Nucleic Acid Surface Computation](#nucleic-acid-surface-computation) section below.

## 🧮 Nucleic Acid Surface Computation

This project includes a specialized surface computation module (`comp_surface/`) with **modified APBS tools** optimized for nucleic acid structures.

### Method Source

The surface computation method is based on the `computesurf.ipynb` notebook, which uses modified APBS tools specifically adapted for nucleic acid structures. This method has been validated in the original project.

### Key Components

- **`computeTargetMesh_test_samples.py`**: Core computation function `compute_inp_surface()`
- **`compute_surface.py`**: Batch processing script (converted from notebook)

### Usage

#### Batch Surface Computation

```bash
python comp_surface/prepare_target/compute_surface.py \
    --data_dir /path/to/input \
    --out_dir /path/to/output \
    --surface_dist 8
```

#### Python API (Single Structure)

```python
from comp_surface.prepare_target.computeTargetMesh_test_samples import compute_inp_surface

result = compute_inp_surface(
    target_filename='protein.pdb',
    ligand_filename='ligand.sdf',
    out_dir='output_dir',
    dist_threshold=8  # Surface distance threshold in Angstroms
)
```

#### Python API (Batch Processing)

```python
from comp_surface.prepare_target.compute_surface import compute_surface_for_directory

compute_surface_for_directory(
    data_dir='input_dir',
    out_dir='output_dir',
    surface_dist=8  # Automatically converts to dist_threshold=surface_dist-5
)
```

### Output Files

For each structure, the following files are generated:
- `{structure}_protein_{dist_threshold}A.pdb` - Pocket region PDB file
- `{structure}_protein_{dist_threshold}A.ply` - Surface mesh PLY file

### Important Notes

- **APBS Tools**: The APBS tools have been **modified for nucleic acid structures** and **must use the version provided in this project**. Do not download from the official repository.
- **Tool Paths**: Automatically configured in `comp_surface/default_config/global_vars.py` using relative paths.
- **Dependencies**: Requires pymesh library (for mesh processing) and APBS, MSMS, PDB2PQR tools (included).

For detailed documentation, see:
- [comp_surface/README.md](comp_surface/README.md) - Module overview
- [comp_surface/SURFACE_COMPUTATION.md](comp_surface/SURFACE_COMPUTATION.md) - Detailed computation method

## 📖 Usage Examples

### Example 1: Running Docking Evaluation

```bash
# Activate environment
conda activate SurfNA

# Set environment variables
export precomputed_arrays="/root/autodl-tmp/SurfNA/precomputed_arrays"
mkdir -p precomputed_arrays

# Run evaluation
bash run_eval.sh
```

### Example 2: Computing Surface for Multiple Structures

```bash
python comp_surface/prepare_target/compute_surface.py \
    --data_dir data/testset \
    --out_dir data/testset_surface \
    --surface_dist 8 \
    --ligand_suffix _ligand.sdf
```

### Example 3: Using in Python Scripts

```python
import sys
sys.path.append('/path/to/SurfNA')

# For docking
from evaluate_accelarate import main_function

# For surface computation
from comp_surface.prepare_target.computeTargetMesh_test_samples import compute_inp_surface
result = compute_inp_surface('protein.pdb', 'ligand.sdf', 'output/', dist_threshold=8)
```

## 📁 Project Structure

```
SurfNA/
├── datasets/                  # Dataset processing modules
│   ├── pdbbind.py            # PDBBind dataset loader
│   ├── process_mols.py        # Molecular processing
│   └── ...
├── models/                    # Model definitions
│   ├── surface_score_model_v3.py
│   └── mdn_score_model_v6.py
├── utils/                     # Utility functions
│   ├── sampling.py           # Sampling algorithms
│   ├── diffusion_utils.py    # Diffusion utilities
│   └── ...
├── comp_surface/              # Nucleic acid surface computation
│   ├── default_config/       # Configuration parameters
│   ├── input_output/         # I/O processing
│   ├── prepare_target/       # Main computation module
│   │   ├── computeTargetMesh_test_samples.py  # Core function
│   │   └── compute_surface.py                 # Batch processing
│   ├── protein_process/      # Protein processing
│   └── tools/                # Modified APBS tools (for nucleic acids)
│       └── transfer/         # APBS, MSMS, PDB2PQR binaries
├── force_optimize/            # Force field optimization
├── data/                      # Test data
│   ├── testset/              # 3 docking test cases
│   ├── testset_surface/      # Surface data
│   ├── splits/               # Data splits
│   └── test_na_surface/      # 5 nucleic acid test structures
├── evaluate_accelarate.py     # Main evaluation script
├── run_eval.sh               # Run script
└── README.md                 # This file
```

## 🔧 Dependencies

### Core Dependencies

- **Python** 3.10+
- **PyTorch** 2.2.2 (with CUDA 12.1)
- **torch-geometric** 2.5.2
- **RDKit** 2023.3.1
- **accelerate** 0.15.0

### Additional Dependencies

- biopandas, MDAnalysis, spyrmsd
- wandb (for logging, optional)
- pymesh (for surface mesh processing)
- scikit-learn, scipy, numpy

See [environment.yml](environment.yml) for the complete dependency list.

## 📊 Performance

The docking evaluation has been tested on 3 test cases with the following results:

- **Top5 RMSD < 2 Å**: 66.67%
- **Mean Centroid Distance**: 0.74 Å
- **No Steric Clashes**: 0.0%
- **Average Runtime**: ~5 seconds per complex

## 🧪 Test Data

### Docking Test Cases
- `2f4t_ab9`
- `3g4m_2bp`
- `1f1t_ros`

### Surface Computation Test Cases
- `100d`, `101d`, `102d`, `107d`, `108d`

Test data is located in `data/testset/` and `test_na_surface/input/`.

## 📚 Documentation

- **[INSTALL.md](INSTALL.md)** - Detailed installation guide
- **[SETUP.md](SETUP.md)** - Setup and configuration guide
- **[comp_surface/README.md](comp_surface/README.md)** - Surface computation module overview
- **[comp_surface/SURFACE_COMPUTATION.md](comp_surface/SURFACE_COMPUTATION.md)** - Detailed computation method
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines



## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## ⚠️ Important Notes

1. **Model Weights**: Model weights are not included in this repository. You need to prepare them separately and place them in the `model_weights/` directory.

2. **APBS Tools**: The APBS tools included in `comp_surface/tools/` have been **modified for nucleic acid structures**. **Do not download from the official repository** - you must use the version provided here.

3. **First Run**: The first run will generate cache files and precomputed arrays, which may take several minutes.

4. **Output**: Results are saved in the `test_workdir/` directory, including:
   - RMSD statistics (`.npy` files)
   - Docking pose structures (`.sdf` files)
   - Performance metrics


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

For questions and issues, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for nucleic acid docking research**
