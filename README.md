# Vir-AI-DIP  
**Sequence Feature Analysis of Defective Viral Genomes in Influenza A and B Viruses**

---

## Overview

This repository provides a computational framework for the systematic analysis of deletion-type defective viral genomes (DelVGs) in Influenza A and B viruses. The pipeline integrates preprocessing, feature engineering, statistical analysis, visualization, and deep learning-based modeling to investigate sequence- and structure-derived features associated with NGS read counts and potential antiviral candidates.

The workflow is designed to ensure reproducibility, modularity, and scientific rigor, enabling both exploratory and predictive analyses.

---

## Repository Structure

```
.
├── analysis/
│   ├── preprocess.py       # Data preprocessing and feature extraction pipeline
│   ├── statistics.py       # Statistical analysis and summary computations
│   └── visuals.py          # Visualization and exploratory data analysis
├── networks/
│   ├── create_bin_CNN.py   # Binary classification CNN models
│   └── create_reg_CNN.py   # Regression CNN models
├── metadata/
│   └── metadata.py         # Metadata processing and related helper functions
├── data/                   # Input datasets
├── results/                # Generated results, figures, and model outputs
├── utils.py                # General utility functions
├── env.yml                 # Environment specification (dependencies)
├── run_analysis_scripts.sh # Script to execute analysis workflow
├── .gitignore              # Git ignore rules
└── README.md
```

## Setup
Before running the with different datasets the repository needs to be set up accordingly.

1. Change the paths in utils.py line 26 & 27:
   ```
   DATAPATH = "/path/to/datasets"
   RESULTSPATH = "/path/to/resultfolder"
   ```
2. Put your data in the folder that is defined in DATAPATH. Each publication should get a new folder and in this folder each SRA entry is a single .csv file. For further information or to request access to the datasets used in this study, please refer to the contact details provided below.
3. Install all dependencies. This can be done by conda using the following command:
   ```
   conda env create --file=env.yml
   ```
4. To run the initial analysis the script src/run_analysis_scripts.sh is available. First you need to activate the conda environment.
   ```
   conda activate dips
   ```
5. Then it can be run using the following command:
   ```
   bash run_analysis_scripts.sh
   ```

## Workflow

### 1. Data Preprocessing

Run preprocessing, acoording to the instructions within the code:

```bash
python preprocess.py
```

The preprocessing pipeline follows a hierarchical feature construction strategy:

#### Base Preprocessing
- Cleaning and standardization of raw DelVG datasets  
- Harmonization of metadata and sequence representations  

#### Primary Features
- Deletion coordinates (start, end)  
- Sequence lengths (DelVG length, deletion length, retained ends)  
- Nucleotide composition (GC, AU, CpG, UpA content)  
- Sequence entropy and k-mer richness  

#### Secondary Features
- RNA secondary structure prediction (ViennaRNA)  
- Minimum free energy (MFE)  
- Base-pairing statistics (density, counts)  
- Stem and hairpin characteristics  

#### Motif Features
- Extraction of motifs at deletion junction sites  
- Motif frequency distributions  
- Comparison to full-sequence background distributions  

---

### 2. Sequence Analysis

Run:

```bash
python visuals.py
python statistics.py
```

These modules provide:

#### Statistical Methods
- Spearman correlation analysis  
- Fisher’s exact test for categorical enrichment  
- Multiple testing correction using Benjamini–Hochberg FDR  

#### Visualization Methods
- Heatmaps (feature correlations and motif enrichment)  
- Scatter plots (feature relationships)  
- Density plots (NGS read count distributions)  
- Violin plots (distribution across categorical variables)  
- Motif frequency bar plots and heatmaps  

---

### 3. CNNs

Run:

```bash
python create_bin_CNN.py
python create_reg_CNN.py
```

#### Binary Classification
- Predicts high vs. low NGS read count DelVGs  

#### Regression Modeling
- Predicts continuous NGS read counts  

#### Model Characteristics
- Sequence-based convolutional neural network (CNN) input  
- Comparison of pooled vs. unpooled preprocessing strategies 

## Data

The datasets used in this project consist of Influenza A and B DelVGs derived from next-generation sequencing (NGS) experiments.
Due to data ownership and size constraints, datasets are not included in this repository.

---

## Contact

For questions regarding the data or methodology:

Leibniz Institute for Virology  
https://www.leibniz-liv.de/forschung/forschungseinheiten/modellierung-viraler-systeme

---

## Scientific Context

This work was conducted as part of a Bachelor’s thesis in Computer Science at the University of Hamburg.

---

## Notes

- The pipeline supports:
  - Influenza A and B datasets  
  - Strain- and segment-specific analyses  
  - Pooled and unpooled preprocessing strategies  

- Statistical procedures follow established standards:
  - Non-parametric correlation (Spearman)  
  - Multiple testing correction (BH-FDR)  
  - Exact tests for categorical comparisons  

---

## License

This project is intended for academic and research purposes.  
Please contact the authors for usage beyond this scope.

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

Additionally, selected sections of the code were adapted from the meta-study by Lohmann (available at: https://github.com/viraidip/DIP_meta-study). 
These sections are explicitly highlighted within the code.
