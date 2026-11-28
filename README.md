# Programming heteroanion electrolytes for long-life aqueous zinc batteries

This repository contains the machine-learning models and the active-learning / Bayesian-optimization workflow used to design aqueous electrolytes. Each candidate composition is screened by three support vector machine (SVM) models: solubility, hydrogen evolution reaction (HER) stability and oxygen evolution reaction (OER) stability. Gaussian process models of Coulombic efficiency (CE) and ionic conductivity then guide five rounds of batch experiments.

## Repository structure

```
├── Solubility model/
│   ├── 1 - Solubility_SVM_Classifier_Distance.ipynb            # SVM solubility classifier, distance-based training set
│   └── 2 - Solubility_SVM_Classifier_Distance_Variance.ipynb   # SVM solubility classifier, distance + variance training set
├── Electrochemical stability model/
│   └── 3 - ESW_SVM_Regression.ipynb   # SVM models for the HER/OER electrochemical stability window (ESW), 3D design-space plots
├── Active learning _BO/
│   ├── batch_AL_01.ipynb   # Round 1: distance-based sampling inside the SVM-screened design space
│   └── batch_AL_02–05.ipynb # Rounds 2–5: GP models (CE, conductivity) + constrained Bayesian optimization
├── data/                   # Experimental datasets used by the notebooks
├── cache/                  # Pre-trained models and intermediate results (loaded via matal.utils.load_cache)
├── matal/                  # Helper package: caching, visualization, design-boundary models, OT-2 robot protocol generation
└── requirements.txt
```

## Installation

```bash
conda create -n aqueous-electrolyte python=3.10
conda activate aqueous-electrolyte
pip install -r requirements.txt
```

`treepoem` (used to generate barcode labels for the OT-2 robot protocols) also requires [Ghostscript](https://www.ghostscript.com/).

## Usage

Start Jupyter from the repository root and run the notebooks in order:

1. `Solubility model/` notebooks 1 and 2
2. `Electrochemical stability model/` notebook 3
3. `Active learning _BO/batch_AL_01.ipynb` to `batch_AL_05.ipynb`

Paths are resolved relative to the repository root (see `matal/configs.py`), so the notebooks can be run without editing paths.

## Data

| File | Description |
|---|---|
| `Round 1.xlsx` – `Round 5.xlsx` | Experimental results (conductivity, CE-1/2/3) from active-learning rounds 1–5. Rounds 1 to N−1 are the inputs to `batch_AL_0N` |

Workflow: `batch_AL_0N.ipynb` reads the experimental results of rounds 1 to N−1 (`Round *.xlsx`) and proposes the compositions for round N, which were then tested experimentally (`Round N.xlsx`).
| `Distance_solubility_data.csv` | Solubility training set (distance-based sampling) |
| `Distance_Variance_Solubility_data.csv` | Solubility training set (distance + variance sampling) |
| `Solubility_testingdata.csv` | Independent test set for the solubility models |
| `solubility_data.csv` | Solubility data at 30 °C (molality) |
| `ESW_data.csv` | HER/OER electrochemical stability window data |

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
