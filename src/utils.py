'''
    general functions and global parameters, that are used in different scripts
'''
import os
import re
import math

import itertools as it
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from typing import List, Tuple

import traceback
import inspect
import builtins

import random

import RNA

from multiprocessing import Pool, cpu_count
from typing import Tuple
from Bio import SeqIO

from sklearn.linear_model import LinearRegression

### STATIC VALUES ###
# load config and assign values to global variables
DATAPATH = "/home/erikl/ubudocuments/ba_dip_hunger/data"
RESULTSPATH = "/home/erikl/ubudocuments/ba_dip_hunger/results"

# segments, nucleotides, and strains
CMAP = "Accent"
CUTOFF = 15
N_SAMPLES = 35000

SEED = 42
RANK_THRESHOLD = 20
DECIMALS = 2
K_MER_LENGTH = 3
PALINDROMIC_K_MER_LENGTH = 6
DIRECT_REPEAT_LEGTH_CAP = 5

LOGARITHM = "log10"
NORMALIZATION = "robust"

RESULTSPATH = os.path.join(RESULTSPATH, f"cutoff_{CUTOFF}")

# system type #
VIVO_MOUSE_DATASETS = ["Wang2023", "Penn2022", "Lui2019"]
VITRO_DATASETS = ["Alnaji2021", "Pelz2021", "Wang2020", "Kupke2020", "Zhuravlev2020", "VdHoecke2015", "Alnaji2019_Cal07" ,"Alnaji2019_NC", "Mendes2021", "Boussier2020", "Alnaji2019_Perth", "Alnaji2019_BLEE", "Sheng2018"]
VIVO_HUMAN_DATASETS = ["Berry2021_A", "Berry2021_B", "Berry2021_B_Yam", "Southgate2019", "Valesano2020_Yam", "Valesano2020_Vic"]
VIVO_DATASETS = VIVO_MOUSE_DATASETS + VIVO_HUMAN_DATASETS

# LibraryLayout #
PAIRED_DATASETS = ['Alnaji2021', 'Pelz2021', 'Wang2023', 'Wang2020', 'Kupke2020', 'VdHoecke2015', 'Alnaji2019_Cal07', 'Alnaji2019_NC', 'Mendes2021', 'Boussier2020', 'Alnaji2019_Perth', 'Berry2021_A', 'Penn2022', 'Alnaji2019_BLEE', 'Berry2021_B', 'Valesano2020_Vic', 'Berry2021_B_Yam', 'Southgate2019', 'Valesano2020_Yam']
SINGLE_DATASETS = ['Zhuravlev2020', 'Sheng2018']
SINGLE_and_PAIRED_DATASETS = ['Lui2019']

# LibrarySelection #
RT_PCR_DATASETS = ['VdHoecke2015', 'Alnaji2019_Cal07', 'Alnaji2019_NC', 'Alnaji2019_Perth', 'Berry2021_A', 'Alnaji2019_BLEE', 'Berry2021_B', 'Valesano2020_Vic', 'Berry2021_B_Yam', 'Southgate2019', 'Valesano2020_Yam']
PCR_DATASETS = ['Alnaji2021', 'Pelz2021']
POLYA_DATASETS = ['Zhuravlev2020', 'Sheng2018']
CDNA_DATASETS = ['Wang2020', 'Mendes2021']
WGA_DATASETS = ['Penn2022']
OTHER_DATASETS = ['Wang2023']
POLYA_and_PCR_DATASETS = ['Kupke2020']
PCR_and_CDNA_DATASETS = ['Lui2019']
RT_PCR_and_CDNA_DATASETS = ['Boussier2020']

# LibrarySource #
VIRAL_RNA_DATASETS = ['VdHoecke2015', 'Alnaji2019_Cal07', 'Alnaji2019_NC', 'Alnaji2019_Perth', 'Berry2021_A', 'Penn2022', 'Alnaji2019_BLEE', 'Berry2021_B', 'Valesano2020_Vic', 'Berry2021_B_Yam', 'Valesano2020_Yam']
GENOMIC_DATASETS = ['Alnaji2021', 'Pelz2021', 'Wang2023']
TRANSCRIPTOMIC_DATASETS = ['Wang2020', 'Zhuravlev2020', 'Mendes2021', 'Sheng2018', 'Boussier2020']
OTHER_DATASETS = ['Southgate2019']
TRANSCRIPTOMIC_SINGLE_CELL_and_VIRAL_RNA_DATASETS = ['Kupke2020']
VIRAL_RNA_and_OTHER_DATASETS = ['Lui2019']

# subtype #
H1N1_DATASETS = ['Alnaji2021', 'Pelz2021', 'Wang2023', 'Wang2020', 'Zhuravlev2020', 'Kupke2020', 'VdHoecke2015', 'Alnaji2019_Cal07', 'Alnaji2019_NC', 'Mendes2021', 'Boussier2020']
H3N2_DATASETS = ['Alnaji2019_Perth', 'Berry2021_A']
H5N1_DATASETS = ['Penn2022']
H7N9_DATASETS = ['Lui2019']
IAV_DATASETS = H1N1_DATASETS + H3N2_DATASETS + H5N1_DATASETS + H7N9_DATASETS
IBV_DATASETS = ["Alnaji2019_BLEE", "Berry2021_B", "Valesano2020_Vic", "Sheng2018", "Berry2021_B_Yam", "Southgate2019","Valesano2020_Yam"]

STRAINS = ["PR8", "Cal07", "NC", "WSN_Mendes_rev", "WSN", "Perth", "Connecticut", "Turkey", "Anhui", "BLEE", "Victoria", "Brisbane", "Yamagata"]
SEGMENTS = list(["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"])
NUCLEOTIDES = dict({"A": "Adenine", "C": "Cytosin", "G": "Guanine", "U": "Uracil"})


DATASET_STRAIN_DICT = dict({
    # H1N1
    "Alnaji2021": "PR8",
    "Pelz2021": "PR8",
    "Wang2023": "PR8",
    "Wang2020": "PR8",
    "Zhuravlev2020": "PR8",
    "Kupke2020": "PR8",
    "VdHoecke2015": "PR8",
    "Alnaji2019_Cal07": "Cal07",
    "Alnaji2019_NC" : "NC",
    "Mendes2021": "WSN_Mendes_rev",
    "Boussier2020": "WSN",
    # H3N2
    "Alnaji2019_Perth": "Perth",
    "Berry2021_A": "Connecticut",
    # H5N1
    "Penn2022": "Turkey",
    # H7N9
    "Lui2019": "Anhui",
    # B 
    "Alnaji2019_BLEE": "BLEE",
    "Berry2021_B": "Victoria",
    "Valesano2020_Vic": "Victoria",
    "Sheng2018": "Brisbane",
    "Berry2021_B_Yam": "Yamagata",
    "Southgate2019": "Yamagata",
    "Valesano2020_Yam": "Yamagata"
})

ACCNUMDICT = dict({
    "Wang2023": dict({
        "SRR16770171": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770172": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770173": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770174": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770175": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770181": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770182": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770183": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770184": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770185": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770186": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770191": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770192": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770193": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770197": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770198": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770201": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770200": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770199": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770207": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770208": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770209": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770210": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770211": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770212": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770219": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770218": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "SRR16770217": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Localization:": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "mouse"})
    }),
    "Wang2020": dict({
        "SRR7722028": dict({"Cells": "A549", "Time": "6", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722030": dict({"Cells": "A549", "Time": "12", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722032": dict({"Cells": "A549", "Time": "24", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722029": dict({"Cells": "A549", "Time": "6", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722031": dict({"Cells": "A549", "Time": "12", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722033": dict({"Cells": "A549", "Time": "24", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),

        "SRR7722036": dict({"Cells": "HBEpC", "Time": "6", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722038": dict({"Cells": "HBEpC", "Time": "12", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722040": dict({"Cells": "HBEpC", "Time": "24", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722037": dict({"Cells": "HBEpC", "Time": "6", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722039": dict({"Cells": "HBEpC", "Time": "12", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"}),
        "SRR7722041": dict({"Cells": "HBEpC", "Time": "24", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": 5, "Host:": "human"})
    }),
    "Mendes2021": dict({
        "SRR15720520": dict({"Replicate": "e11", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720521": dict({"Replicate": "e12", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720522": dict({"Replicate": "e21", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720523": dict({"Replicate": "e12", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720524": dict({"Replicate": "d11", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720525": dict({"Replicate": "d12", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720526": dict({"Replicate": "d21", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"}),
        "SRR15720527": dict({"Replicate": "d22", "Cells": "A549", "Localization:": "extracellular", "Resolution": "bulk", "Time": "48", "MOI": "unknown", "Host:": "human"})
    }),
    "Pelz2021": dict({
        "SRR15084925": dict({"Time": "seed", "Localization:": "unknown", "Cells": "unknown", "Resolution": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "unknown"}),
        "SRR15084924": dict({"Time": "12", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084913": dict({"Time": "24", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084908": dict({"Time": "34", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084907": dict({"Time": "84", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084906": dict({"Time": "96", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084905": dict({"Time": "108", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084904": dict({"Time": "120", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084903": dict({"Time": "132", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084902": dict({"Time": "192", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084923": dict({"Time": "216", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084922": dict({"Time": "226", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084921": dict({"Time": "298", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084919": dict({"Time": "312", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084918": dict({"Time": "324", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084917": dict({"Time": "384", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084916": dict({"Time": "408", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084915": dict({"Time": "420", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084914": dict({"Time": "432", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084912": dict({"Time": "468", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084911": dict({"Time": "480", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084910": dict({"Time": "490", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR15084909": dict({"Time": "504", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Multi-timepoint": True, "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"})
    }),
    "Alnaji2019_Cal07": dict({
        "SRR8754522": dict({"Replicate": "1", "Passage": "6", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754523": dict({"Replicate": "2", "Passage": "6", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754531": dict({"Replicate": "1", "Passage": "6_t", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754532": dict({"Replicate": "1", "Passage": "3_t", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754533": dict({"Replicate": "1", "Passage": "1_t", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"})
    }),
    "Alnaji2019_NC": dict({
        "SRR8754513": dict({"Replicate": "2", "Passage": "1", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754514": dict({"Replicate": "1", "Passage": "1", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754527": dict({"Replicate": "1", "Passage": "6", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754538": dict({"Replicate": "2", "Passage": "6", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"})
    }),
    "Alnaji2019_Perth": dict({
        "SRR8754517": dict({"Replicate": "2", "Passage": "8", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754524": dict({"Replicate": "1", "Passage": "4", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754525": dict({"Replicate": "2", "Passage": "4", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754526": dict({"Replicate": "1", "Passage": "8", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"})
    }),
    "Alnaji2019_BLEE": dict({
        "SRR8754507": dict({"Replicate": "1", "Passage": "8", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754508": dict({"Replicate": "2", "Passage": "7", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754509": dict({"Replicate": "1", "Passage": "7", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"}),
        "SRR8754516": dict({"Replicate": "2", "Passage": "8", "Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "24", "MOI": "unknown", "Host:": "dog"})
    }),
    "Lui2019": dict({
        "SRR8949705": dict({"Cells": "Mouse", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 0.5, "Time": "24", "Replicate": "unknown", "Host:": "mouse"}),
        "SRR8945328": dict({"Cells": "Mouse", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 0.5, "Time": "24", "Replicate": "unknown", "Host:": "mouse"}),
    }),
    "Penn2022": dict({
        "ERR10231074": dict({"Time": "24", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Replicate": "H1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231075": dict({"Time": "48", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Replicate": "H1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231076": dict({"Time": "6", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Replicate": "H1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231077": dict({"Time": "96", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Replicate": "H1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231078": dict({"Time": "24", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Replicate": "H2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231079": dict({"Time": "48", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Replicate": "H2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231080": dict({"Time": "6", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Replicate": "H2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231081": dict({"Time": "96", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Replicate": "H2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231089": dict({"Time": "96", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Replicate": "L2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231082": dict({"Time": "24", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Replicate": "L1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231085": dict({"Time": "96", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Replicate": "L1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231083": dict({"Time": "48", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Replicate": "L1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231084": dict({"Time": "6", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Replicate": "L1", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231086": dict({"Time": "24", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Replicate": "L2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231087": dict({"Time": "48", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Replicate": "L2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"}),
        "ERR10231088": dict({"Time": "6", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Replicate": "L2", "Localization:": "unknown", "Resolution": "unknown", "MOI": "unknown", "Host:": "mouse"})
    }),
    "Alnaji2021": dict({
        "SRR14352106": dict({"Replicate": "C", "Cells": "MDCK-SIAT1", "Time": "24", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352107": dict({"Replicate": "B", "Cells": "MDCK-SIAT1", "Time": "24", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352108": dict({"Replicate": "A", "Cells": "MDCK-SIAT1", "Time": "24", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352109": dict({"Replicate": "C", "Cells": "MDCK-SIAT1", "Time": "6", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352110": dict({"Replicate": "B", "Cells": "MDCK-SIAT1", "Time": "6", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352111": dict({"Replicate": "A", "Cells": "MDCK-SIAT1", "Time": "6", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352112": dict({"Replicate": "C", "Cells": "MDCK-SIAT1", "Time": "3", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352113": dict({"Replicate": "X", "Time": "0", "Resolution": "bulk", "Localization:": "unknown", "Cells": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR14352116": dict({"Replicate": "B", "Cells": "MDCK-SIAT1", "Time": "3", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"}),
        "SRR14352117": dict({"Replicate": "A", "Cells": "MDCK-SIAT1", "Time": "3", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 10, "Host:": "dog"})
    }),
    "Kupke2020": dict({
        "SRR10489473": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "0", "Host:": "dog"}),
		"SRR10489474": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU1low_0", "Host:": "dog"}),
		"SRR10489475": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU640high_0", "Host:": "dog"}),
		"SRR10489476": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU24low_0", "Host:": "dog"}),
		"SRR10489477": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU11low_0", "Host:": "dog"}),
		"SRR10489478": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU470high_0", "Host:": "dog"}),
		"SRR10489479": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU450high_0", "Host:": "dog"}),
		"SRR10489480": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU23low_0", "Host:": "dog"}),
		"SRR10489481": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU430high_0", "Host:": "dog"}),
		"SRR10489482": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU570high_0", "Host:": "dog"}),
		"SRR10489483": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU670high_0", "Host:": "dog"}),
		"SRR10489484": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU720high_0", "Host:": "dog"}),
		"SRR10489485": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU380high_0", "Host:": "dog"}),
		"SRR10489486": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU680high_0", "Host:": "dog"}),
		"SRR10489487": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU23low_1", "Host:": "dog"}),
		"SRR10489488": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU390high_0", "Host:": "dog"}),
		"SRR10489489": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU38low_0", "Host:": "dog"}),
		"SRR10489490": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU380high_1", "Host:": "dog"}),
		"SRR10489491": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU22low_0", "Host:": "dog"}),
		"SRR10489492": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU650high_0", "Host:": "dog"}),
		"SRR10489493": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU34low_0", "Host:": "dog"}),
		"SRR10489494": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU19low_0", "Host:": "dog"}),
		"SRR10489495": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU18low_0", "Host:": "dog"}),
		"SRR10489496": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU390high_1", "Host:": "dog"}),
		"SRR10489497": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU27low_0", "Host:": "dog"}),
		"SRR10489498": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU410high_0", "Host:": "dog"}),
		"SRR10489499": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU40low_0", "Host:": "dog"}),
		"SRR10489500": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU900high_0", "Host:": "dog"}),
		"SRR10489501": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU3low_0", "Host:": "dog"}),
		"SRR10489502": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU450high_1", "Host:": "dog"}),
		"SRR10489503": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU470high_1", "Host:": "dog"}),
		"SRR10489504": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU1100high_0", "Host:": "dog"}),
		"SRR10489505": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU360high_0", "Host:": "dog"}),
		"SRR10489506": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU5low_0", "Host:": "dog"}),
		"SRR10489507": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU690high_0", "Host:": "dog"}),
		"SRR10489508": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU490high_0", "Host:": "dog"}),
		"SRR10489509": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU580high_0", "Host:": "dog"}),
		"SRR10489510": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU400high_0", "Host:": "dog"}),
		"SRR10489511": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU380high_2", "Host:": "dog"}),
		"SRR10489512": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU16low_0", "Host:": "dog"}),
		"SRR10489513": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU360high_1", "Host:": "dog"}),
		"SRR10489514": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU360high_2", "Host:": "dog"}),
		"SRR10489515": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU360high_3", "Host:": "dog"}),
		"SRR10489516": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU37low_0", "Host:": "dog"}),
		"SRR10489517": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU360high_4", "Host:": "dog"}),
		"SRR10489518": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU460high_0", "Host:": "dog"}),
		"SRR10489519": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU330high_0", "Host:": "dog"}),
		"SRR10489520": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU25low_0", "Host:": "dog"}),
		"SRR10489521": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU650high_1", "Host:": "dog"}),
		"SRR10489522": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU890high_0", "Host:": "dog"}),
		"SRR10489523": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU1low_1", "Host:": "dog"}),
		"SRR10489524": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU17low_0", "Host:": "dog"}),
		"SRR10489525": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU460high_1", "Host:": "dog"}),
		"SRR10489526": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU3low_1", "Host:": "dog"}),
		"SRR10489527": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU430high_1", "Host:": "dog"}),
		"SRR10489528": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU40low_1", "Host:": "dog"}),
		"SRR10489529": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU21low_0", "Host:": "dog"}),
		"SRR10489530": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU420high_0", "Host:": "dog"}),
		"SRR10489531": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU350high_0", "Host:": "dog"}),
		"SRR10489532": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU28low_0", "Host:": "dog"}),
		"SRR10489533": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU29low_0", "Host:": "dog"}),
		"SRR10489534": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU22low_1", "Host:": "dog"}),
		"SRR10489535": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU11low_1", "Host:": "dog"}),
		"SRR10489536": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU510high_0", "Host:": "dog"}),
		"SRR10489537": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU11low_2", "Host:": "dog"}),
		"SRR10489538": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU19low_1", "Host:": "dog"}),
		"SRR10489539": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU6low_0", "Host:": "dog"}),
		"SRR10489540": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU16low_1", "Host:": "dog"}),
		"SRR10489541": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU400high_1", "Host:": "dog"}),
		"SRR10489542": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU4low_0", "Host:": "dog"}),
		"SRR10489543": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU370high_0", "Host:": "dog"}),
		"SRR10489544": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU12low_0", "Host:": "dog"}),
		"SRR10489545": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU31low_0", "Host:": "dog"}),
		"SRR10489546": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU560high_0", "Host:": "dog"}),
		"SRR10489547": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU14low_0", "Host:": "dog"}),
		"SRR10489548": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU810high_0", "Host:": "dog"}),
		"SRR10489549": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU9low_0", "Host:": "dog"}),
		"SRR10489550": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU26low_0", "Host:": "dog"}),
		"SRR10489551": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU400high_2", "Host:": "dog"}),
		"SRR10489552": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU34low_1", "Host:": "dog"}),
		"SRR10489553": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU24low_1", "Host:": "dog"}),
		"SRR10489554": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU20low_0", "Host:": "dog"}),
		"SRR10489555": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU2low_0", "Host:": "dog"}),
		"SRR10489556": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU24low_2", "Host:": "dog"}),
		"SRR10489557": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU37low_1", "Host:": "dog"}),
		"SRR10489558": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU430high_2", "Host:": "dog"}),
		"SRR10489559": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU360high_5", "Host:": "dog"}),
		"SRR10489560": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU550high_0", "Host:": "dog"}),
		"SRR10489561": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU31low_1", "Host:": "dog"}),
		"SRR10489562": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU14low_1", "Host:": "dog"}),
		"SRR10489563": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU450high_0", "Host:": "dog"}),
		"SRR10489564": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU14low_2", "Host:": "dog"}),
		"SRR10489565": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU960high_0", "Host:": "dog"}),
		"SRR10489566": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU28low_1", "Host:": "dog"}),
		"SRR10489567": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU510high_1", "Host:": "dog"}),
		"SRR10489568": dict({"Resolution": "singlecell", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "intracellular", "Replicate": "PFU4low_1", "Host:": "dog"}),
        "SRR10530642": dict({"Resolution": "bulk", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "extracellular", "Replicate": "1", "Host:": "dog"}),
        "SRR10530643": dict({"Resolution": "bulk", "Cells": "MDCK", "MOI": 10, "Time": "12", "Localization:": "extracellular", "Replicate": "2", "Host:": "dog"})
    }),
    "Sheng2018": dict({
        "SRR3211978": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR3211980": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR3211976": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR3211977": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR3211974": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR3211975": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR3211972": dict({"Cells": "A549", "Localization:": "intracellular", "Resolution": "bulk", "MOI": 1, "Time": "unknown", "Replicate": "unknown", "Host:": "human"})
    }),
    "Zhuravlev2020": dict({
        "ERR4566024": dict({"Cells": "A549", "Time": "48", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566025": dict({"Cells": "A549", "Time": "48", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566028": dict({"Cells": "HEK293FT", "Time": "48", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566029": dict({"Cells": "HEK293FT", "Time": "48", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566032": dict({"Cells": "MRC5", "Time": "48", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566033": dict({"Cells": "MRC5", "Time": "48", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566036": dict({"Cells": "WI38", "Time": "48", "Replicate": "1", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"}),
        "ERR4566037": dict({"Cells": "WI38", "Time": "48", "Replicate": "2", "Localization:": "intracellular", "Resolution": "singlecell", "MOI": "unknown", "Host:": "human"})
    }),
    "Berry2021_A": dict({
        "SRR15182178": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "4-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182177": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "4-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182176": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "3-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182175": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "3-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182174": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "2-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182173": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "2-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182172": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "1-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15182171": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "1-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"})
    }),
    "Berry2021_B": dict({
        "SRR15183345": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "1-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183344": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "1-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183352": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "2-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183353": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "2-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196408": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "3-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196409": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "3-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196410": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "4-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196411": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "5-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196412": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "5-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196413": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "4-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196414": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "6-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196415": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "6-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196416": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "7-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196417": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "7-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196419": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "9-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196418": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "8-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196420": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "9-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196421": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "8-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196422": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "10-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196423": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "10-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196424": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "11-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15196425": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "11-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"})
    }),
    "Berry2021_B_Yam": dict({
        "SRR15183338": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "1-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183343": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "3-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183342": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "3-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183341": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "1-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183340": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "2-2", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"}),
        "SRR15183339": dict({"Localization:": "extracellular", "Resolution": "bulk", "Replicate": "2-1", "Cells": "unknown", "Time": "unknown", "MOI": "unknown", "Host:": "unknown"})
    }),
    "Valesano2020_Vic": dict({
        "SRR10013092": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013237": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013181": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013242": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013050": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013272": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013047": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013239": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013071": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013201": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013072": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013200": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013108": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013256": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013037": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013254": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013279": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013219": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013221": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"})
    }),
    "Valesano2020_Yam": dict({
        "SRR10013243": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013084": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013188": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013094": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013178": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013236": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013063": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013209": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013241": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013240": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013229": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013068": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013205": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013067": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013206": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013062": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013210": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013070": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013203": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013103": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013170": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013223": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013244": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "SRR10013275": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"})
    }),
    "Southgate2019": dict({
        "ERR3474616": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474621": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474642": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474643": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474658": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474661": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474662": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474663": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474664": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474666": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474671": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474674": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474675": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474676": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474679": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474684": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474685": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474686": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474687": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474689": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474692": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474693": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474694": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474695": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474697": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474698": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474699": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474701": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474702": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474703": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474704": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474705": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474706": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474707": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474709": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474710": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474712": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474713": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474714": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474715": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474716": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474717": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474718": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474719": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474720": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474721": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474722": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474723": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474724": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474725": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474726": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474728": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474729": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474750": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474751": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474781": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474796": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"}),
        "ERR3474809": dict({"Cells": "human", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "human"})
    }),
    "VdHoecke2015": dict({
        "SRR1757953": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 0.01, "Time": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "SRR1758027": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "MOI": 0.01, "Time": "unknown", "Replicate": "unknown", "Host:": "dog"})
    }),
    "Boussier2020": dict({
        "180628A_rec_A-P1p_S218": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628A_rec_B-P1p_S219": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628A_rec_C-P1p_S219": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628A_rec_D-P1p_S221": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628A_rec_WT1p6-1213_S242": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628B_rec_A-P1p-PCR_S213": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628B_rec_B-P1p-PCR_S214": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628B_rec_C-P1p-PCR_S215": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628B_rec_D-P1p-PCR_S216": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180628B_rec_WT-P1p-PCR_S217": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705A_rec_AP1pb_S294": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705A_rec_BP1pb_S295": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705A_rec_CP1pb_S296": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705A_rec_DP1pb_S297": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705A_rec_WTP1pb_S298": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705B_rec_AP1pPCRb_S289": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705B_rec_BP1pPCRb_S290": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705B_rec_CP1pPCRb_S291": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705B_rec_DP1pPCRb_S292": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180705B_rec_WTP1pPCRb_S293": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180706A_rec_AP1pc_S10": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180706A_rec_BP1pc_S11": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"}),
        "180706A_rec_DP1pc_S12": dict({"Cells": "MDCK", "Localization:": "extracellular", "Resolution": "bulk", "Time": "unknown", "MOI": "unknown", "Replicate": "unknown", "Host:": "dog"})
    })
})

SEGMENT_DICTS = dict({
    "PR8": dict({
        "AF389115.1": "PB2",
        "AF389116.1": "PB1",
        "AF389117.1": "PA",
        "AF389118.1": "HA",
        "AF389119.1": "NP",
        "AF389120.1": "NA",
        "AF389121.1": "M",
        "AF389122.1": "NS"
    }),
    "Cal07": dict({
        "CY121687.1": "PB2",
        "CY121686.1": "PB1",
        "CY121685.1": "PA",
        "CY121680.1": "HA",
        "CY121683.1": "NP",
        "CY121682.1": "NA",
        "CY121681.1": "M",
        "CY121684.1": "NS"
    }),
    "NC": dict({
        "CY147325.1": "PB2",
        "CY147324.1": "PB1",
        "CY147323.1": "PA",
        "CY147318.1": "HA",
        "CY147321.1": "NP",
        "CY147320.1": "NA",
        "CY147319.1": "M",
        "CY147322.1": "NS"
    }),
    "Perth": dict({
        "KJ609203.1": "PB2",
        "KJ609204.1": "PB1",
        "KJ609205.1": "PA",
        "KJ609206.1": "HA",
        "KJ609207.1": "NP",
        "KJ609208.1": "NA",
        "KJ609209.1": "M",
        "KJ609210.1": "NS"
    }),
    "BLEE": dict({
        "CY115118.1": "PB2",
        "CY115117.1": "PB1",
        "CY115116.1": "PA",
        "CY115111.1": "HA",
        "CY115114.1": "NP",
        "CY115113.1": "NA",
        "CY115112.1": "M",
        "CY115115.1": "NS"
    }),
    "WSN_Mendes_rev": dict({
        "PB2_vRNA": "PB2",
        "PB1_vRNA": "PB1",
        "PA_vRNA": "PA",
        "HA_vRNA": "HA",
        "NP_vRNA": "NP",
        "NA_vRNA": "NA",
        "M_vRNA": "M",
        "NS_vRNA": "NS"
    }),
    "WSN": dict({
        "LC333182.1": "PB2",
        "LC333183.1": "PB1",
        "LC333184.1": "PA",
        "LC333185.1": "HA",
        "LC333186.1": "NP",
        "LC333187.1": "NA",
        "LC333188.1": "M",
        "LC333189.1": "NS"
    }),
    "Anhui": dict({
        "439504": "PB2",
        "439508": "PB1",
        "439503": "PA",
        "439507": "HA",
        "439505": "NP",
        "439509": "NA",
        "439506": "M",
        "439510": "NS"
    }),
    "Turkey": dict({
        "EF619975.1": "PB2",
        "EF619976.1": "PB1",
        "EF619979.1": "PA",
        "AF389118.1": "HA",
        "EF619977.1": "NP",
        "EF619973.1": "NA",
        "EF619978.1": "M",
        "EF619974.1": "NS"
    }),
    "Brisbane": dict({
        "CY115158.1": "PB2",
        "CY115157.1": "PB1",
        "CY115156.1": "PA",
        "CY115151.1": "HA",
        "CY115154.1": "NP",
        "CY115153.1": "NA",
        "CY115152.1": "M",
        "CY115155.1": "NS"
    }),
    "swine": dict({
        "KR701038.1": "PB2",
        "KR701039.1": "PB1",
        "KR701040.1": "PA",
        "KR701041.1": "HA",
        "KR701042.1": "NP",
        "KR701043.1": "NA",
        "KR701044.1": "M",
        "KR701045.1": "NS"
    }),
    "Cal09": dict({
        "JF915190.1": "PB2",
        "JF915189.1": "PB1",
        "JF915188.1": "PA",
        "JF915184.1": "HA",
        "JF915187.1": "NP",
        "JF915186.1": "NA",
        "JF915185.1": "M",
        "JF915191.1": "NS"
    }),
    "Greninger_cons": dict({
        "PB2": "PB2",
        "PB1": "PB1",
        "PA": "PA",
        "HA": "HA",
        "NP": "NP",
        "NA": "NA",
        "M": "M",
        "NS": "NS"
    }),
    "Connecticut": dict({
        "KM654658.1": "PB2",
        "KM654706.1": "PB1",
        "KM654754.1": "PA",
        "KM654822.1": "HA",
        "KM654847.1": "NP",
        "KM654920.1": "NA",
        "KM654969.1": "M",
        "KM654612.1": "NS"
    }),
    "Victoria": dict({
        "CY018660.1": "PB2",
        "CY018659.1": "PB1",
        "CY018658.1": "PA",
        "CY018653.1": "HA",
        "CY018656.1": "NP",
        "CY018655.1": "NA",
        "CY018654.1": "M",
        "CY018657.1": "NS"
    }),
    "H3N2_Thailand": dict({
        "KP335735.1": "PB2",
        "KP335793.1": "PB1",
        "KP335851.1": "PA",
        "KP335964.1": "HA",
        "KP336026.1": "NP",
        "KP336139.1": "NA",
        "KP336201.1": "M",
        "KP336259.1": "NS"
    }),
    "Yamagata":({
        "OQ034430.1": "PB2",
        "OQ034429.1": "PB1",
        "OQ034431.1": "PA",
        "OQ034432.1": "HA",
        "OQ034433.1": "NP",
        "OQ034434.1": "NA",
        "OQ034435.1": "M",
        "OQ034436.1": "NS"
    }),
    "H1N1_Thailand":({
        "KU051428.1": "PB2",
        "KU051429.1": "PB1",
        "KU051430.1": "PA",
        "KU051431.1": "HA",
        "KU051432.1": "NP",
        "KU051433.1": "NA",
        "KU051434.1": "M",
        "KU051435.1": "NS"
    }),
    "Malaysia":({
        "CY040456.1": "PB2",
        "CY040455.1": "PB1",
        "CY040454.1": "PA",
        "CY040449.1": "HA",
        "CY040452.1": "NP",
        "CY040451.1": "NA",
        "CY040450.1": "M",
        "CY040453.1": "NS"
    })
})


### FUNCTIONS ###

def get_dataset_names(cutoff: int=0, selection: str="")-> list:
    '''
        Allows to select dataset names based on their cultivation type.
        :param cutoff: Threshold for min number of DelVGs in each dataset
        :param selection: cultivation type either 'in vivo mouse', 'in vitro'
                         or 'in vivo human'
        
        :return: list of dataset names
    '''
    if cutoff == 0 and selection == "":
        return list(DATASET_STRAIN_DICT.keys())
    
    path = os.path.join(RESULTSPATH, "metadata", f"dataset_stats_{CUTOFF}.csv")
    df = pd.read_csv(path)
    names = df[df["Size"] >= cutoff]["Dataset"].to_list()

    # make selection based on in vivo/cells etc.
    if selection == "in vivo mouse":
        select_names = VIVO_MOUSE_DATASETS
    elif selection == "in vitro":
        select_names = VITRO_DATASETS
    elif selection == "in vivo human":
        select_names = VIVO_HUMAN_DATASETS
    elif selection == "in vivo":
        select_names = VIVO_DATASETS
    elif selection == "IAV":
        select_names = IAV_DATASETS
    elif selection == "IBV":
        select_names = IBV_DATASETS
    elif selection == "IAV (in vivo mouse)":
        select_names = list(set(IAV_DATASETS) & set(VIVO_MOUSE_DATASETS))
    elif selection == "IAV (in vitro)":
        select_names = list(set(IAV_DATASETS) & set(VITRO_DATASETS))
    elif selection == "IAV (in vivo human)":
        select_names = list(set(IAV_DATASETS) & set(VIVO_HUMAN_DATASETS))
    elif selection == "IAV (in vivo)":
        select_names = list(set(IAV_DATASETS) & set(VIVO_DATASETS))
    elif selection == "IBV (in vivo mouse)":
        select_names = list(set(IBV_DATASETS) & set(VIVO_MOUSE_DATASETS))
    elif selection == "IBV (in vitro)":
        select_names = list(set(IBV_DATASETS) & set(VITRO_DATASETS))
    elif selection == "IBV (in vivo human)":
        select_names = list(set(IBV_DATASETS) & set(VIVO_HUMAN_DATASETS))
    elif selection == "IBV (in vivo)":
        select_names = list(set(IBV_DATASETS) & set(VIVO_DATASETS))
    else:
        select_names = names

    names = [name for name in names if name in select_names]
    return names

def load_single_dataset(exp: str, acc: str, segment_dict: dict)-> pd.DataFrame:
    '''
        Load a single dataset, defined by one SRA accession number.
        :param exp: name of the experiment (is also folder name)
        :param acc: SRA accession number
        :param segment_dict: dictionary that maps the ids of the reference
                            fastas to the segment names

        :return: Pandas Dataframe with one DelVG population
    '''
    path = os.path.join(DATAPATH, exp, f"{exp}_{acc}.csv")
    df = pd.read_csv(path,
                     dtype={"Segment": "string", "Start": "int64", "End": "int64", "NGS_read_count": "int64"},
                     na_values=["", "None"],
                     keep_default_na=False)
    df["Segment"] = df["Segment"].replace(segment_dict)

    return df

def load_dataset(dataset: str)-> pd.DataFrame:
    '''
        Load a full dataset, defined by multiple SRA accession numbers.
        :param exp: name of the experiment (is also folder name)

        :return: Pandas Dataframe with one DelVG population of whole experiment
    '''
    acc_nums = ACCNUMDICT[dataset]
    strain = DATASET_STRAIN_DICT[dataset]
    dfs = list()
    for acc_num, meta in acc_nums.items():
        df = load_single_dataset(dataset, acc_num, SEGMENT_DICTS[strain])
        for key in meta.keys():
            df[key] = meta[key]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df

def load_all(dfnames: list, expected: str=False)-> Tuple[list, list]:
    '''
        Load a list of datasets.
        :param dfnames: list of dataset names, each is one experiment
        :param expected: if True, expected data is loaded additionally

        :return: Tuple
            List of Pandas Dataframes each containing one experiment
            List of dataset names in same order as first list
    '''
    dfs = list()
    expected_dfs = list()
    for dfname in dfnames:
        strain = DATASET_STRAIN_DICT[dfname]
        df = join_data(load_dataset(dfname))
        dfs.append(preprocess(strain, df, CUTOFF))
        if expected:
            f = os.path.join(DATAPATH, "random_sampled", f"{dfname}_{CUTOFF}.csv")
            if os.path.exists(f):
                dtypes = {"Start": int, "End": int, "Segment": str, "NGS_read_count": int,
                          "key": str, "Strain": str, "isize": int, "full_seq": str,
                          "deleted_sequence": str, "seq_around_deletion_junction": str}
                exp_df = pd.read_csv(f, dtype=dtypes)
            else:
                df = df[df["NGS_read_count"] >= CUTOFF].copy()
                exp_df = preprocess(strain, generate_expected_data(strain, df), 1)
                exp_df.to_csv(f, index=False)
            expected_dfs.append(exp_df)
    return dfs, expected_dfs

def sort_datasets_by_type(dfs: list, dfnames: list, cutoff: int)-> Tuple[list, list]:
    '''
        Sorts a given name of experiments by cultivation type.
        :param dfs: list of datasets, ordered as in dfnames
        :param dfnames: list of dataset names, each is one experiment
        :param cutoff: Threshold for min number of DelVGs in each dataset

        :return: Tuple
            List of Pandas Dataframes each containing one experiment
            List of dataset names in same order as first list
    '''
    vitro = get_dataset_names(cutoff=cutoff, selection="in vitro")
    vivo = get_dataset_names(cutoff=cutoff, selection="in vivo mouse")
    patients = get_dataset_names(cutoff=cutoff, selection="in vivo human")
    dfnames_new_order = vitro + vivo + patients
    combined_data = list(zip(dfnames, dfs))

    def custom_sort(item):
        return dfnames_new_order.index(item[0])

    sorted_data = sorted(combined_data, key=custom_sort)
    dfnames_sorted, dfs_sorted = zip(*sorted_data)

    return dfs_sorted, dfnames_sorted

def join_data(df: pd.DataFrame)-> pd.DataFrame:
    '''
        Combine duplicate DelVGs and sum their NGS count.
        :param df: Pandas DataFrame with DelVG data

        :return: Pandas DataFrame without duplicate DelVGs
    '''
    return df.groupby(["Segment", "Start", "End"]).sum(["NGS_read_count"]).reset_index()

def load_mapped_reads(experiment: str)-> pd.DataFrame:
    '''
        Loads data about the reads that were mapped to each segment.
        :param experiment: name of the experiment (is also folder name)

        :return: Pandas DataFrame with mapped reads per segment
    '''
    acc_nums = ACCNUMDICT[experiment]

    dfs = list()
    for acc_num, meta in acc_nums.items():
        path = os.path.join(DATAPATH, experiment, f"{acc_num}_mapped_reads_per_segment.csv")
        if not os.path.exists(path):
            path = os.path.join(DATAPATH, experiment, f"{acc_num}both_mapped_reads_per_segment.csv")
        df = pd.read_csv(path, dtype={"counts":"int64","segment": "string"}, na_values=["", "None"], keep_default_na=False)
        for m in meta.keys():
            df[m] = meta[m]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df

def load_all_mapped_reads(dfnames: list)-> list:
    '''
        Loads data about the mapped reads for all given experiments.
        :param dfnames: list of dataset names, each is one experiment

        :return: List of Pandas Dataframes each containing mapped reads for one
                experiment
    '''
    mr_dfs = list()
    for experiment in dfnames:
        df = load_mapped_reads(experiment)
        mr_dfs.append(df)
    return mr_dfs

def get_sequence(strain: str, seg: str, full: bool=False)-> object:
    '''
        Loads a DNA sequence given the strain and segment.
        :param strain: name of the strain
        :param seg: name of the segment
        :param full: if True the whole Biopython Seq Object is returned
                    if False a string object is returned

        :return: Biopython Seq Object or str() of the sequence
    '''
    fasta_file = os.path.join(DATAPATH, "strain_segment_fastas", strain, f"{seg}.fasta")
    seq_obj = SeqIO.read(fasta_file, "fasta")
    if full:
        return seq_obj
    else:
        return str(seq_obj.seq.transcribe())

def get_seq_len(strain: str, seg: str)-> int:
    '''
        Calculates the length of a specific sequence given the strain and
        segment.
        :param strain: name of the strain
        :param seg: name of the segment

        :return: length of the sequence as int
    '''
    return len(get_sequence(strain, seg))

def get_p_value_symbol(p: float)-> str:
    '''
        Indicates the statistical significance by strings. Is used for plots.
        :param p: p-value of the test

        :return: string indicating the significance level
    '''
    if p < 0.005:
        return "***"
    elif p < 0.01:
        return "** "
    elif p < 0.05:
        return " * "
    else:
        return "ns."

def calc_cliffs_d(d1: list, d2: list)-> float:
    '''
        Cliff, Norman (1993). Dominance statistics: Ordinal analyses to answer
        ordinal questions (eq. 3)
        Cliffs d ranges from -1 (max effect of group 2) to 0 (no effect) to
        1 (max effect of group 1) Meissel K. and Yao E. (2024)
        :param d1: dataset 1
        :param d2: dataset 2

        :return: cliff's d
    '''
    U, _ = stats.mannwhitneyu(d1, d2)
    cliffs_d = 2*U / (len(d1)*len(d2)) - 1
    return cliffs_d

######################
### direct repeats ###
######################

def calculate_direct_repeat(seq: str, s: int, e: int, w_len: int)-> Tuple[int, str]:
    '''
        Counts the number of overlapping nucleotides directly before start and
        end of junction site --> direct repeats
        :param seq: nucleotide sequence
        :param s: start point
        :param e: end point
        :param w_len: length of window to be searched

        :return: Tuple
            Integer giving the number of overlapping nucleotides
            String of the overlapping nucleotides
    '''
    counter = 0
    start_window = seq[s-w_len: s]
    end_window = seq[e-1-w_len: e-1]
        
    #if they are the same return directly to avoid off-by-one error
    if start_window == end_window:
        return len(start_window), start_window

    if len(seq) < e:
        return 0, "_"
        
    for i in range(len(end_window)-1, -1, -1):
        if start_window[i] == end_window[i]:
            counter += 1
        else:
            break
    overlap_seq = str(start_window[i+1:w_len])

    assert counter == len(overlap_seq), f"{counter=}, {len(overlap_seq)}"
    if len(overlap_seq) == 0:
        overlap_seq = "_"

    return counter, overlap_seq

def count_direct_repeats_overall(df: pd.DataFrame, seq: str)-> Tuple[dict, dict]:
    '''
        Calculates the number of direct repeats for each data point.
        :param df: dataframe with sequence and junction site data
        :param seq: RNA sequence of the given segement and strain

        :return: Tuple
            Dict with the count of the direct repeat lengths
            Dict with the overlapping sequences and their count
    '''
    w_len = 5
    nuc_overlap_dict = dict({i: 0 for i in range(0, w_len+1)})
    overlap_seq_dict = dict()
 
    for _, row in df.iterrows():
        s = row["Start"]
        e = row["End"]
        idx, overlap_seq = calculate_direct_repeat(seq, s, e, w_len)
        nuc_overlap_dict[idx] += 1
        if overlap_seq in overlap_seq_dict:
            overlap_seq_dict[overlap_seq] += 1
        else:
            overlap_seq_dict[overlap_seq] = 1

    return nuc_overlap_dict, overlap_seq_dict

#############################
### nucleotide enrichment ###
#############################

def count_nucleotide_occurrence(seq: str, p: int)-> dict:
    '''
        Counts the number of nucleotides next to a given point.
        Goes 5 steps in both directions.
        :param seq: whole RNA sequence
        :param p: point on the sequence where to count

        :return: Counter dict with an entry for each nucleotide. In each entry
                the counter for each position is given.
    '''
    window = seq[p-5:p+5]
    r_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})

    for i, char in enumerate(window):
        r_dict[char][i] = 1
    return r_dict

def count_nucleotide_occurrence_overall(df: pd.DataFrame, seq: str)-> Tuple[dict, dict]:
    '''
        Counts the occurrence of each nucleotide at different positions around
        the junction site
        :param df: dataframe with sequence and junction site data
        :param seq: rna sequence where to count the occurrence

        :return: Tuple
            Dict with nucleotide count for start of deletion site
            Dict with nucleotide count for end of deletion site
    '''

    count_start_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    count_end_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    normalize = 0

    for _, row in df.iterrows():
        seq_start_dict = count_nucleotide_occurrence(seq, row["Start"]) 
        seq_end_dict = count_nucleotide_occurrence(seq, row["End"]-1)
        normalize += 1
        for nuc in count_start_dict.keys():
            count_start_dict[nuc] += seq_start_dict[nuc]
            count_end_dict[nuc] += seq_end_dict[nuc]

    return count_start_dict, count_end_dict

#####################
### expected data ###
#####################

def generate_expected_data(strain: str, df: pd.DataFrame)-> pd.DataFrame:
    '''
        Randomly samples deletion sites for a given dataset which can be used
        to compare the results of the real dataset.
        :param strain: name of the strain
        :param df: DelVG dataset

        :return: artifical dataset that includes random deletion sites
    '''
    for seg in SEGMENTS:
        df_s = df.loc[df["Segment"] == seg]
        if len(df_s) == 0:
            continue
        seq = get_sequence(strain, seg)
        start = int(df_s["Start"].mean())
        end = int(df_s["End"].mean())
        s = (max(start-200, 50), start+200)
        e = (end-200, min(end+200, len(seq)-50))
        
        # skip if there is no range given this would lead to oversampling of a single position
        if s[0] == s[1] or e[0] == e[1]:
            continue
        # positions are overlapping
        if s[1] > e[0]:
            continue
        if "samp_df" in locals():
            temp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            temp_df["Segment"] = seg
            samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
        else:
            samp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            samp_df["Segment"] = seg
    
    samp_df["NGS_read_count"] = 1
    return samp_df.reset_index()

def generate_sampling_data(seq: str, s: Tuple[int, int], e: Tuple[int, int],  n: int)-> pd.DataFrame:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites
        :param n: number of samples to generate

        :return: Pandas DataFrame of the artifical data set
    '''
    df_no_duplicates = create_sampling_space(seq, s, e)
    return df_no_duplicates.sample(n)

def create_sampling_space(seq: str, s: Tuple[int, int], e: Tuple[int, int])-> pd.DataFrame:
    '''
        Creates all possible candidates that would be expected.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites
        
        :return: dataframe with possible DelVG candidates
    '''
    # create all combinations of start and end positions that are possible
    combinations = [(x, y) for x in range(s[0], s[1]+1) for y in range(e[0], e[1]+1)]

    # create for each the DelVG Sequence
    sequences = [seq[:start] + seq[end-1:] for (start, end) in combinations]

    # filter out duplicate DelVG sequences while keeping the ones with highest start number
    start, end = zip(*combinations)
    temp_df = pd.DataFrame(data=dict({"Start": start, "End": end, "Sequence": sequences}))

    # Find the index of the row with the maximum value in the 'Start' column for each 'Sequence'
    max_start_index = temp_df.groupby('Sequence')['Start'].idxmax()
    result_df = temp_df.loc[max_start_index]
    # Replicate each row by the number of times it was found in the group
    result_df = result_df.loc[result_df.index.repeat(temp_df.groupby('Sequence').size())]
    df_no_duplicates = result_df.reset_index(drop=True).drop("Sequence", axis=1)

    return df_no_duplicates

#######################
### data processing ###
#######################

def create_nucleotide_ratio_matrix(df: pd.DataFrame, col: str)-> pd.DataFrame:
    '''
        Counts nucleotides around the deletion site. Used to create heatmaps.
        :param df: Pandas DataFrame that was created using sequence_df()
        :param col: column name which sequence to use

        :return: Pandas DataFrame with probabilites for the nucleotides
    '''
    probability_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    seq_matrix = df.filter([col], axis=1)
    seq_matrix = seq_matrix[col].str.split("", expand=True)
    # drop first and last column
    seq_matrix = seq_matrix.drop([0, len(seq_matrix.columns)-1], axis=1)
    
    for n in NUCLEOTIDES.keys():
        probability_matrix[n] = seq_matrix.apply(lambda x: dict(x.value_counts()).get(n,0)/len(x), axis=0)

    return probability_matrix

def plot_heatmap(y: list, x: list, vals: list, ax: object,
                 format=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False, cbar_ax=None, cbar_kws=None)-> object:
    '''
        Helper function to plot heatmap.
        :param y: columns of heatmap
        :param x: rows of heatmap
        :param vals: values for heatmap
        :param ax: matplotlib.axes object
        :param: additional parameters check sns.heatmap() for more information
        
        :return: generated heatmap on matplotlib.axes object
    '''
    df = pd.DataFrame({"x":x,"y":y,"vals":vals})
    df = pd.pivot_table(df, index="x", columns="y", values="vals", sort=False)
    ax = sns.heatmap(df, fmt=format, annot=True, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar, cmap=cmap, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    return ax

def sequence_df(df: pd.DataFrame, strain: str, isize: int=5)-> pd.DataFrame:
    '''
        Generate a DataFrame with sequence information.
        :param df: Pandas DataFrame containing the DelVGs in the "key" column
            Nomenclature: {seg}_{start}_{end}
        :param strain: name of the strain
        :param isize: the size of the sequence before and after the start and
            end positions. Default is 5.

    :return: Pandas DataFrame with the following columns:
            - "key": The original key from the input DataFrame.
            - "Segment": The segment
            - "Start": The start position of the deletion site
            - "End": The end position of the deletion site
            - "seq": The dip sequence
            - "deleted_sequence": The deleted sequence
            - "isize": The specified size for the before and after sequences
            - "full_seq": full sequence of the wild type virus
            - "Strain": strain used in the experiment
            - "seq_around_deletion_junction": sequence around deletion sites
            - "NGS_read_count": NGS count measured in the experiment

    '''
    df["Strain"] = strain
    df["Start"] = df.apply(lambda row: int(row["key"].split("_")[1]), axis=1)
    df["End"] = df.apply(lambda row: int(row["key"].split("_")[2]), axis=1)
    df["Segment"] = df.apply(lambda row: row["key"].split("_")[0], axis=1)
    df["isize"] = isize
    def wrap_get_sequence(row):
        return get_sequence(row["Strain"], row["Segment"])
    df["full_seq"] = df.apply(wrap_get_sequence, axis=1)
    def wrap_get_deleted_sequence(row):
        return get_deleted_sequence(row["key"], row["Strain"])
    df["deleted_sequence"] = df.apply(wrap_get_deleted_sequence, axis=1)
    def get_seq_around_del(row):
        seq_head = get_dip_sequence(row["key"], row["Strain"])[1]
        seq_foot = get_dip_sequence(row["key"], row["Strain"])[2]
        
        seq_before_start = seq_head[-row["isize"]:]
        seq_after_start = row["deleted_sequence"][:row["isize"]]
        seq_before_end = row["deleted_sequence"][-row["isize"]:]
        seq_after_end = seq_foot[:row["isize"]]
        return seq_before_start + seq_after_start + seq_before_end + seq_after_end
    df["seq_around_deletion_junction"] = df.apply(get_seq_around_del, axis=1)
    return df

def preprocess(strain: str, df: pd.DataFrame, thresh: int)-> pd.DataFrame:
    '''
        Excluding DelVGs with to low NGS count and running sequence_df().
        :param strain: name of the strain
        :param df: Pandas DataFrame with DelVG data
        :param thresh: Threshold for min number of count for each DelVG

        :return: resulting df of sequence_df() function
    '''
    if thresh > 1:
        df = df[df["NGS_read_count"] >= thresh].copy()
    df["key"] = df["Segment"] + "_" + df["Start"].map(str) + "_" + df["End"].map(str)
    return sequence_df(df, strain)

def get_deleted_sequence(delvg_id: str, strain: str)-> str:
    '''
        Return the sequence of the deletion site.
        :param delvg_id: the id of the DelVG ({seg}_{start}_{end})
        :param strain: name of the strain

        :return: the sequence that is deleted in a DelVG
    '''
    seg, start, end = delvg_id.split("_")
    seq = get_sequence(strain, seg)
    return seq[int(start):int(end)-1]

def get_dip_sequence(delvg_id: str, strain: str)-> Tuple[str, str, str]:
    '''
        Return the remaining sequence of a DelVG. Deletion is filled with "*".
        :param delvg_id: the id of the DelVG ({seg}_{start}_{end})
        :param strain: name of the strain

        :return: Tuple
            the remaining sequence of a DelVG
            the sequence before the deletion site
            the sequence after the deletion site
    '''
    seg, start, end = delvg_id.split("_")
    fl_seq = get_sequence(strain, seg)
    seq_head = fl_seq[:int(start)]
    seq_foot = fl_seq[int(end)-1:]
    del_length = int(end)-int(start)
    return seq_head + "*"*del_length + seq_foot, seq_head, seq_foot

################
### NEW WORK ###
################

STRAIN_SUBTYPE_DICT = dict({
    # H1N1
    "PR8": "H1N1",
    "Cal07": "H1N1",
    "NC": "H1N1",
    "WSN_Mendes_rev": "H1N1",
    "WSN": "H1N1",

    # H3N2
    "Perth": "H3N2",
    "Connecticut": "H3N2",

    # H5N1
    "Turkey": "H5N1",

    # H7N9
    "Anhui": "H7N9",

    # B
    "BLEE": "B",
    "Victoria": "B",
    "Brisbane": "B",
    "Yamagata": "B"
})

COLORS = [
     '#ff6666', # 0
     '#ff9160', # 1
     '#ffbd55', # 2
     '#ffde5b', # 3
     '#ffff66', # 4
     '#cef458', # 5
     '#9de24f', # 6
     '#93d986', # 7
     '#87cefa', # 8
     '#a7a6eb', # 9
     '#cda4de' # 10
    ]

COLOR_TEMPLATES = {
    1: [8],
    2: [0, 8],
    3: [0, 4, 8],
    4: [0, 3, 6, 9],
    5: [0, 3, 5, 8, 10],
    6: [0, 2, 4, 6, 8, 10],
    7: [0, 2, 3, 4, 6, 8, 10],
    8: [0, 2, 3, 4, 6, 8, 9, 10],
    9: [0, 1, 2, 3, 4, 6, 8, 9, 10],
    10: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
}

###############
### general ###
###############

### load and save ###

def get_strains(datasets: list):
    seen = set()
    strains = []
    for dataset in datasets:
        strain = DATASET_STRAIN_DICT.get(dataset)
        if strain is not None and strain not in seen:
            seen.add(strain)
            strains.append(strain)
    return strains

def clean_data_string(string: str):
    '''
    
    '''
    clean_string = re.sub(r'[^A-Za-z0-9 ]+', '', string)
    clean_string = clean_string.replace(" ", "_")

    return clean_string

def manage_specifiers(df: pd.DataFrame, data: str, strain: str, segment: str):
    '''
    
    '''
    if data == 'all':
        pass

    if strain == 'all':
        pass
    else:
        df = df[df['Strain'] == strain]

    if segment == 'all':
        pass
    else:
        df = df[df['Segment'] == segment]
    
    return df

def manage_separate_specifiers(dfs: list, data: str, strain: str, segment: str):
    '''
    
    '''
    return [manage_specifiers(df, data, strain, segment) for df in dfs]

def load_preprocessed_dataset(fname: str, folder: str = '', subfolder: str = '', data: str = '', strain: str = '', segment: str = '', intersects: str = '')-> pd.DataFrame:
    '''

    '''
    read_path, _ = os.path.split(RESULTSPATH)
    read_path = os.path.join(read_path, 'preprocess')
    read_path = os.path.join(read_path, folder) if folder else read_path
    read_path = os.path.join(read_path, subfolder) if subfolder else read_path
    read_path = os.path.join(read_path, data) if data else read_path
    read_path = os.path.join(read_path, strain) if strain else read_path
    read_path = os.path.join(read_path, segment) if segment else read_path
    read_path = os.path.join(read_path, intersects) if intersects else read_path
    os.makedirs(read_path, exist_ok=True)

    fname = fname + ".csv"
    df = pd.read_csv(os.path.join(read_path, fname), keep_default_na=False, na_values=[], low_memory=False, dtype={
        "Time": "string",
        "MOI": "string"
        }
    )

    return df

def load_all_preprocessed(fnames: list, folder: str = '', subfolder: str = '', data: str = '', strain: str = '', segment: str = '', intersects: str = '')-> list:
    '''

    '''
    dfs = []
    for fname in fnames:
        df = load_preprocessed_dataset(fname, folder, subfolder, data, strain, segment, intersects)
        dfs.append(df)

    return dfs

def merge_missing_features(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    df1_filtered = df1[df1['key'].isin(df2['key'])].copy()
    missing_cols = [col for col in df2.columns if col not in df1.columns and col != 'key']

    df2_subset = df2[['key'] + missing_cols]
    merged_df = pd.merge(df1_filtered, df2_subset, on='key', how='inner')

    return merged_df

def save_df(df: pd.DataFrame, fname: str, save_path: str, folder: str = '', subfolder: str = '', data: str = '', strain: str = '', segment: str = '', intersects: str = ''):
    '''
    
    '''
    save_path = os.path.join(save_path, folder) if folder else save_path
    save_path = os.path.join(save_path, subfolder) if subfolder else save_path
    save_path = os.path.join(save_path, data) if data else save_path
    save_path = os.path.join(save_path, strain) if strain else save_path
    save_path = os.path.join(save_path, segment) if segment else save_path
    save_path = os.path.join(save_path, intersects) if intersects else save_path
    os.makedirs(save_path, exist_ok=True)
    fname = fname + ".csv"

    df.to_csv(os.path.join(save_path, fname), index=False, na_rep="")

### unppoled ###

_FEATURE_KEYS = ["Time", "Localization", "Resolution", "Cells", "MOI", "Host"]

_KEY_NORMALIZATION = {
    "Time": "Time",
    "Localization": "Localization",
    "Localization:": "Localization",
    "Resolution": "Resolution",
    "Resolution:": "Resolution",
    "Cells": "Cells",
    "Cells:": "Cells",
    "MOI": "MOI",
    "MOI:": "MOI",
    "Host": "Host",
    "Host:": "Host",
}

def _extract_meta_features(meta: dict) -> dict:
    """
    Return ONLY the whitelisted features from ACCNUMDICT(mod).
    Missing ones are filled with 'unknown'.
    """
    out = {k: "unknown" for k in _FEATURE_KEYS}
    for raw_k, v in meta.items():
        k = _KEY_NORMALIZATION.get(raw_k, None)
        if k in _FEATURE_KEYS:
            out[k] = v
    return out

def load_single_dataset_unpooled(exp: str, acc: str, segment_dict: dict) -> pd.DataFrame:
    df = load_single_dataset(exp, acc, segment_dict)  # reuse your existing loader
    df["AN"] = acc  # keep accessions separate
    return df

def load_dataset_unpooled(dataset: str) -> pd.DataFrame:
    acc_nums = ACCNUMDICT[dataset]
    strain = DATASET_STRAIN_DICT[dataset]

    dfs = []
    for acc, meta in acc_nums.items():
        df = load_single_dataset_unpooled(dataset, acc, SEGMENT_DICTS[strain])

        # Strictly add ONLY the six agreed features from ACCNUMDICT(mod)
        features = _extract_meta_features(meta)
        for k, v in features.items():
            df[k] = v

        dfs.append(df)

    # No pooling; just stack rows
    concat_df = pd.concat(dfs, ignore_index=True)

    # Keep all entries: cutoff=0 (preserves all data; keeps your usual preprocess columns)
    out = preprocess(strain, concat_df, 0)
    return out

def load_all_unpooled(dfnames: list, expected: bool = False):
    dfs = [load_dataset_unpooled(name) for name in dfnames]
    expected_dfs = [] if not expected else []
    return dfs, expected_dfs

### metadata ###

def add_dataset_keys(dfs):
    """

    """
    updated_dfs = []
    for i, df in enumerate(dfs, start=1):
        df_copy = df.copy()
        df_copy["dataset_key"] = i
        updated_dfs.append(df_copy)
    return updated_dfs

def add_metadata_features(dfnames: list, dfs: list) -> list:
    resultpath, _ = os.path.split(RESULTSPATH)

    read_path = os.path.join(resultpath, 'preprocess', 'metadata', 'metadata.csv')
    meta_df = pd.read_csv(read_path)

    meta_features = [
        'system type', 'LibraryLayout', 'LibrarySelection', 'LibrarySource', 'subtype',
    ]

    updated_dfs = []
    for dfname, df in zip(dfnames, dfs):
        df = df.copy()
        df['dataset_name'] = dfname

        row = meta_df.loc[meta_df['names'] == dfname]

        for meta_feature in meta_features:
            if not row.empty:
                value = row[meta_feature].iloc[0]
            else:
                value = np.nan

            if pd.isna(value):
                value = "unknown"
            elif isinstance(value, str) and value.strip() == '':
                value = "unknown"

            df[meta_feature] = value

        updated_dfs.append(df)

    renamed_dfs = []
    for df in updated_dfs:
        df = rename_feature(df, 'system type', 'system_type')
        df = rename_feature(df, 'LibraryLayout', 'library_layout')
        df = rename_feature(df, 'LibrarySelection', 'library_selection')
        df = rename_feature(df, 'LibrarySource', 'library_source')
        renamed_dfs.append(df)

    return renamed_dfs

### intersects ###

def manage_intersects(df: pd.DataFrame, modifier: str, feature_name: str) -> pd.DataFrame:
    if modifier == 'all':
        return df
    elif modifier == 'remove':
        df = add_ikey(df)
        df = remove_by_ikey(df, 1)
    elif modifier == 'mean':
        df = add_ikey(df)
        df = mean_by_ikey(df, feature_name)
    elif modifier == 'median':
        df = add_ikey(df)
        df = median_by_ikey(df, feature_name)
    elif modifier == 'remove with metadata':
        df = add_metadata_ikey(df)
        df = remove_by_ikey(df, 1)
    elif modifier == 'mean with metadata':
        df = add_metadata_ikey(df)
        df = mean_by_ikey(df, feature_name)
    elif modifier == 'median with metadata':
        df = add_metadata_ikey(df)
        df = median_by_ikey(df, feature_name)
    else:
        print('unvalid intersects modifier')

    df = df.drop(columns=['ikey'])
    return df

def add_ikey(df: pd.DataFrame) -> pd.DataFrame:
    '''
    
    '''
    df = df.copy()
    df['ikey'] = df['Strain'].astype(str) + '_' + df['key'].astype(str)

    return df

def add_metadata_ikey(df: pd.DataFrame) -> pd.DataFrame:
    '''
    
    '''
    df = df.copy()
    df["ikey"] = (
        df["Strain"].astype(str) + "_" +
        df["key"].astype(str) + "_" +
        df["Localization"].astype(str) + "_" +
        df["Resolution"].astype(str) + "_" +
        df["system_type"].astype(str)
    )

    return df

def remove_by_ikey(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    '''

    '''
    df = df.copy()

    ikey_counts = df['ikey'].value_counts()
    non_unique_ikeys = ikey_counts[ikey_counts > 1].index
    non_unique_rows = df[df['ikey'].isin(non_unique_ikeys)]
    n_to_remove = int(len(non_unique_rows) * threshold)
    rows_to_remove = non_unique_rows.sample(n=n_to_remove, random_state=SEED)
    df = df.drop(index=rows_to_remove.index).reset_index(drop=True)

    return df

def mean_by_ikey(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    '''
    
    '''
    df = df.copy()
    mean_values = df.groupby('ikey')[feature_name].transform(
        lambda x: x.mean() if len(x) > 1 else x
    )

    df[feature_name] = mean_values
    
    return df

def median_by_ikey(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    '''

    '''
    df = df.copy()
    median_values = df.groupby('ikey')[feature_name].transform(
        lambda x: x.median() if len(x) > 1 else x
    )

    df[feature_name] = median_values

    return df

def set_intersect_proportion(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    '''

    '''
    count = required_pseudo_intersects_for_threshold(df, threshold)
    if count >= 0:
        df = add_pseudo_intersects(df, count)
        print("pseudo candidates have been added")
    else:
        df = remove_intersects(df, abs(count))

    return df

def required_pseudo_intersects_for_threshold(df: pd.DataFrame, threshold: float) -> int:
    '''
    
    '''
    if 'ikey' not in df.columns:
        raise ValueError("DataFrame must contain an 'ikey' column.")
    if not (0 < threshold < 1):
        raise ValueError("threshold must be a float between 0 and 1 (exclusive).")
    
    total_rows = len(df)
    ikey_counts = df['ikey'].value_counts()
    non_unique_count = ikey_counts[ikey_counts > 1].sum()

    numerator = threshold * total_rows - non_unique_count
    denominator = 1 - threshold
    adjustment = numerator / denominator

    return int(round(adjustment))

def add_pseudo_intersects(df: pd.DataFrame, pseudo_count: int) -> pd.DataFrame:
    '''
    
    '''
    feature_name = 'norm_log_NGS_read_count'
    
    np.random.seed(SEED)
    value_probs = df[feature_name].value_counts(normalize=True)
    sampled_rows = df.sample(n=pseudo_count, replace=True, random_state=SEED).copy()

    sampled_rows[feature_name] = np.random.choice(
        value_probs.index, 
        size=pseudo_count, 
        p=value_probs.values
    )

    return pd.concat([df, sampled_rows], ignore_index=True)

def remove_intersects(df: pd.DataFrame, remove_count: int) -> pd.DataFrame:
    '''
    
    '''
    if 'ikey' not in df.columns:
        raise ValueError("DataFrame must contain an 'ikey' column.")

    if remove_count <= 0:
        return df.copy()

    ikey_counts = df['ikey'].value_counts()
    non_unique_ikeys = ikey_counts[ikey_counts > 1].index
    non_unique_rows = df[df['ikey'].isin(non_unique_ikeys)]

    if remove_count > len(non_unique_rows):
        raise ValueError(f"cannot remove {remove_count} rows — only {len(non_unique_rows)} non-unique 'ikey' rows available.")

    rows_to_remove = non_unique_rows.sample(n=remove_count, random_state=SEED)
    df_cleaned = df.drop(index=rows_to_remove.index).reset_index(drop=True)

    return df_cleaned

### augment data ###

def rename_feature(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    '''
    
    '''
    if old_name not in df.columns:
        raise ValueError(f"column '{old_name}' not found in DataFrame")

    return df.rename(columns={old_name: new_name})

def split_by_feature(df: pd.DataFrame, feature_name: str, feature_value):
    '''

    '''
    df_true = df[df[feature_name] == feature_value]
    df_false = df[df[feature_name] != feature_value]
    return df_true, df_false

def split_by_threshold(df: pd.DataFrame, feature_name: str, threshold: float):
    '''

    '''
    df_true = df[df[feature_name] >= threshold]
    df_false = df[df[feature_name] < threshold]
    return df_true, df_false

def split_by_number(df: pd.DataFrame, split_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    """

    if split_number < 0:
        raise ValueError(f"`split_number` must be non-negative; got {split_number}.")
    if split_number > len(df):
        raise ValueError(f"`split_number` ({split_number}) cannot exceed df length ({len(df)}).")

    if split_number == 0:
        return df.iloc[0:0].copy(), df.copy()

    tmp_col = "__tmp_row_id__"
    while tmp_col in df.columns:
        tmp_col += "_"

    tmp = df.copy()
    tmp[tmp_col] = range(len(tmp))

    sampled_df = tmp.sample(n=split_number, replace=False, random_state=SEED)
    remaining_df = tmp.loc[~tmp[tmp_col].isin(sampled_df[tmp_col])]

    sampled_df = sampled_df.drop(columns=[tmp_col])
    remaining_df = remaining_df.drop(columns=[tmp_col])

    return sampled_df, remaining_df

def add_feature_percentile_rank(df: pd.DataFrame, feature_name: str, rank_name: str, split_number: int = 20):
    """

    """
    df = df.copy()
    df[rank_name] = pd.qcut(
        df[feature_name],
        q=split_number,
        labels=False,
        duplicates='drop'
    ) + 1
    return df

def add_log_feature(df: pd.DataFrame, feature_name: str, new_name: str, log_type: str=LOGARITHM):
    """

    """
    if log_type == "none":
        df[new_name] = df[feature_name]
    elif log_type == "log10":
        df[new_name] = np.log10(df[feature_name] + 1)
    elif log_type == "log2":
        df[new_name] = np.log2(df[feature_name] + 1)
    else:
        raise ValueError(f"invalid log_type '{log_type}'. choose from 'none', 'log10', 'log2'.")
    return df

def add_norm_feature(df: pd.DataFrame, feature_name: str, new_name: str, norm_type: str=NORMALIZATION):
    """

    """
    series = df[feature_name]

    if norm_type == "min-max":
        fmin, fmax = series.min(), series.max()
        df[new_name] = (series - fmin) / (fmax - fmin) if fmax != fmin else 0.0

    elif norm_type == "z-score":
        mean, std = series.mean(), series.std()
        df[new_name] = (series - mean) / std if std != 0 else 0.0

    elif norm_type == "robust":
        median, q1, q3 = series.median(), series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        df[new_name] = (series - median) / iqr if iqr != 0 else 0.0

    elif norm_type == "euclidean":
        l2_norm = np.sqrt((series ** 2).sum())
        df[new_name] = series / l2_norm if l2_norm != 0 else 0.0

    elif norm_type == "none":
        df[new_name] = seriesfd

    else:
        raise ValueError(f"invalid norm_type '{norm_type}'. "
                         "choose from 'min-max', 'z-score', 'robust', 'euclidean','none'.")
    
    return df

def get_feature_modification_name(log_type: str=LOGARITHM, norm_type: str=NORMALIZATION):
    """

    """
    if log_type != "none" and norm_type != "none":
        return f"{norm_type}-normalized {log_type}-transformed"
    if norm_type != "none":
        return f"{norm_type}-normalized"
    if log_type != "none":
        return f"{log_type}-transformed"
    return ""

def add_separate_ngs_features(dfs: list, separated: bool):
    '''

    '''
    feature = 'NGS_read_count'
    log_feature = 'log_' + feature
    norm_log_feature = 'norm_' + log_feature

    if separated:
        updated_dfs = []
        for df in dfs:
            df = add_log_feature(df, feature, log_feature)
            df = add_norm_feature(df, log_feature, norm_log_feature)
            updated_dfs.append(df)

        return updated_dfs

    else:
        for i, df in enumerate(dfs):
            df['dataset_index'] = i
        concat_df = pd.concat(dfs, ignore_index=True)

        concat_df = add_log_feature(concat_df, feature, log_feature)
        concat_df = add_norm_feature(concat_df, log_feature, norm_log_feature)

        updated_dfs = [
            group.drop(columns='dataset_index').reset_index(drop=True)
            for _, group in concat_df.groupby('dataset_index')
            ]

        return updated_dfs

def balance_by_threshold(df: pd.DataFrame, feature_name: str, threshold: float) -> pd.DataFrame:
    '''
    
    '''
    df_low = df[df[feature_name] < threshold].copy()
    df_high = df[df[feature_name] >= threshold].copy()

    size_low = len(df_low)
    size_high = len(df_high)

    rng = np.random.default_rng(SEED)
    if size_low < size_high:
        df_high = df_high.sample(n=size_low, random_state=SEED)
    elif size_high < size_low:
        df_low = df_low.sample(n=size_high, random_state=SEED)

    df_balanced = pd.concat([df_low, df_high], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df_balanced

def reduce_rows(df: pd.DataFrame, target_number: int) -> pd.DataFrame:
    """

    """
    if len(df) <= target_number:
        return df

    return df.sample(n=target_number, random_state=SEED).reset_index(drop=True)

################
### features ###
################

### length ###

def add_marked_dvg_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    def compute_dvg_sequence(row):
        full_seq = row['full_seq']
        start = row['Start']
        end = row['End'] - 1
        
        seq_list = list(full_seq)

        for i in range(start, end):
            if 0 <= i < len(seq_list):
                seq_list[i] = 'X'

        return ''.join(seq_list)

    df['marked_dvg_sequence'] = df.apply(compute_dvg_sequence, axis=1)

    return df

def add_full_seq_length(df: pd.DataFrame):
    """

    """
    df['full_seq_length'] = df['full_seq'].apply(len)

    return df

def add_dvg_length(df: pd.DataFrame):
    """

    """
    df['dvg_length'] = df['dvg_sequence'].apply(len)

    return df

def add_region_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    df["5_end_length"] = df["Start"]
    df["3_end_length"] = df["full_seq"].str.len() - df["End"]

    return df

def add_deletion_length(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    df["deletion_length"] = df["End"] - df["Start"] - 1

    return df

def add_dvg_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    def compute_dvg_sequence(row):
        full_seq = row['full_seq']
        start = row['Start']
        end = row['End']

        return full_seq[:start] + full_seq[end:]

    df['dvg_sequence'] = df.apply(compute_dvg_sequence, axis=1)

    return df

### direct repeats ###

def cap_direct_repeat_length(df: pd.DataFrame, cap: int = 5):
    for index, row in df.iterrows():
        direct_repeat_length = row["direct_repeat_length"]
        if direct_repeat_length > cap:
            df.loc[index, "direct_repeat_length"] = cap
    return df

def add_direct_repeat_length(df: pd.DataFrame):
    for index, row in df.iterrows():
        seq = row["full_seq"]
        start = row["Start"]
        end = row["End"] - 1
        direct_repeat_length = 0

        while start > 0 and seq[start - 1] == seq[end - 1]:
            direct_repeat_length = direct_repeat_length + 1
            start = start - 1
            end = end - 1
        df.loc[index, "direct_repeat_length"] = direct_repeat_length

    df["direct_repeat_length"] = df["direct_repeat_length"].astype(int)
    return df

### pri ###

def add_gc_content(df: pd.DataFrame):
    """
    Adds 'gc_content' = (G+C)/length for each dvg_sequence.
    """
    def calc(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return 0.0
        return (seq.count('G') + seq.count('C')) / len(seq)
    df['GC_content'] = df['dvg_sequence'].apply(calc)
    return df

def add_au_content(df: pd.DataFrame):
    """
    Adds 'au_content' = (A+U)/length for each dvg_sequence.
    """
    def calc(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return 0.0
        return (seq.count('A') + seq.count('U')) / len(seq)
    df['AU_content'] = df['dvg_sequence'].apply(calc)
    return df

def add_upa_content(df: pd.DataFrame):
    """
    Adds 'upa_content' = count of 'UA' or 'AU' dinucleotides / total dinucleotides.
    UpA under-representation can reflect host pressure.
    """
    def calc(seq):
        if not isinstance(seq, str) or len(seq) < 2:
            return 0.0
        di = len(seq) - 1
        return (seq.count('UA') + seq.count('AU')) / di
    df['UpA_content'] = df['dvg_sequence'].apply(calc)
    return df

def add_cpg_content(df: pd.DataFrame):
    """
    Adds 'cpg_content' = count of 'CG' dinucleotides / total dinucleotides.
    Relevant for innate immune recognition.
    """
    def calc(seq):
        if not isinstance(seq, str) or len(seq) < 2:
            return 0.0
        return seq.count('CG') / (len(seq) - 1)
    df['CpG_content'] = df['dvg_sequence'].apply(calc)
    return df

def add_gc_skew(df: pd.DataFrame):
    """
    Adds 'gc_skew' = (G - C) / (G + C) per sequence.
    """
    def calc(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return 0.0
        g, c = seq.count('G'), seq.count('C')
        return (g - c) / (g + c) if (g + c) else 0.0
    df['GC_skew'] = df['dvg_sequence'].apply(calc)
    return df

def add_sequence_entropy(df: pd.DataFrame):
    """
    Adds 'sequence_entropy' = Shannon entropy over A,C,G,U.
    """
    def calc(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return 0.0
        p = [seq.count(b)/len(seq) for b in 'ACGU']
        return -sum(pi*math.log2(pi) for pi in p if pi > 0)
    df['sequence_entropy'] = df['dvg_sequence'].apply(calc)
    return df

def add_poly_run_features(df: pd.DataFrame, base: str, min_len=4):
    """
    Adds:
      - f'poly{base}_max_run': longest consecutive run length of `base`
      - f'poly{base}_tracts': number of runs with length >= min_len
    Useful for poly-U/A tracts (replication/termination signals).
    """
    import re
    run_col = f'poly_{base}_max_run'
    n_col = f'poly_{base}_tracts'
    pat = re.compile(f'{base}+' if isinstance(base, str) else 'U+')
    def calc_max(seq):
        if not isinstance(seq, str): return 0
        m = pat.findall(seq)
        return max((len(x) for x in m), default=0)
    def calc_n(seq):
        if not isinstance(seq, str): return 0
        return sum(1 for x in pat.findall(seq) if len(x) >= min_len)
    df[run_col] = df['dvg_sequence'].apply(calc_max)
    df[n_col] = df['dvg_sequence'].apply(calc_n)
    return df

def add_palindrome_density(df: pd.DataFrame, k: int, step=1):
    """
    Adds 'palindrome_density' = palindromic k-mer count / number of windows.
    Palindromes can seed hairpins.
    """
    def is_pal(s):
        return s == s[::-1].translate(str.maketrans('ACGU','UGCA'))
    def calc(seq):
        if not isinstance(seq, str) or len(seq) < k:
            return 0.0
        windows = 0
        pals = 0
        for i in range(0, len(seq) - k + 1, step):
            kmer = seq[i:i+k]
            windows += 1
            if is_pal(kmer):
                pals += 1
        return pals / windows if windows else 0.0
    df['palindrome_density'] = df['dvg_sequence'].apply(calc)
    return df

def add_orf_features(df: pd.DataFrame):
    """
    Adds:
      - 'longest_orf_len_nt': longest ORF length in nt across 3 frames
      - 'orf_count_aa_ge20': count of ORFs >= 20 aa
    Uses AUG start and UAA/UAG/UGA stops (on RNA).
    """
    stops = {'UAA','UAG','UGA'}
    def scan(seq):
        if not isinstance(seq, str) or len(seq) < 3:
            return (0, 0)
        best = 0
        count20 = 0
        for frame in range(3):
            i = frame
            while i+2 < len(seq):
                codon = seq[i:i+3]
                if codon == 'AUG':  # start
                    j = i+3
                    while j+2 < len(seq):
                        c = seq[j:j+3]
                        if c in stops:
                            orf_len = j+3 - i
                            best = max(best, orf_len)
                            if orf_len // 3 >= 20:
                                count20 += 1
                            i = j  # continue after stop
                            break
                        j += 3
                i += 3
        return (best, count20)
    res = df['dvg_sequence'].apply(scan)
    df['longest_ORF_len'] = res.apply(lambda x: x[0])
    df['ORF_count'] = res.apply(lambda x: x[1])
    return df

def add_kmer_richness(df: pd.DataFrame, k: int):
    """
    Adds 'kmer_richness_k{K}' = unique k-mers / possible windows.
    Higher values suggest diverse sequence composition.
    """
    col = f'kmer_richness'
    def calc(seq):
        if not isinstance(seq, str) or len(seq) < k:
            return 0.0
        seen = set(seq[i:i+k] for i in range(len(seq)-k+1))
        return len(seen) / (len(seq)-k+1)
    df[col] = df['dvg_sequence'].apply(calc)
    return df

def add_codon_usage_bias(df: pd.DataFrame):
    """
    Adds 'codon_usage_entropy' over 61 sense codons (RNA alphabet; T->U).
    Lower entropy can indicate codon bias.
    """
    import math
    stops = {'UAA','UAG','UGA'}
    def calc(seq):
        if not isinstance(seq, str) or len(seq) < 3:
            return 0.0
        counts = {}
        total = 0
        for frame in (0,1,2):
            for i in range(frame, len(seq)-2, 3):
                cod = seq[i:i+3]
                if len(cod) == 3 and cod not in stops and all(c in 'ACGU' for c in cod):
                    counts[cod] = counts.get(cod, 0) + 1
                    total += 1
        if total == 0: return 0.0
        p = [c/total for c in counts.values()]
        return -sum(pi*math.log2(pi) for pi in p)
    df['codon_usage_entropy'] = df['dvg_sequence'].apply(calc)
    return df

### structure ###

def add_marked_structure(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def compute_marked_structure(row):
        struct = row["structure"]
        full_len = len(row["full_seq"])
        start = row["Start"]
        end = row["End"]

        n_missing = end - start
        prefix = struct[:start]
        suffix = struct[start:]
        marked = prefix + "X" * n_missing + suffix

        if len(marked) != full_len:
            raise ValueError(
                f"length mismatch for row with Start={start}, End={end}: "
                f"full_seq={full_len}, marked_structure={len(marked)}"
            )
        
        return marked

    df["marked_structure"] = df.apply(compute_marked_structure, axis=1)
    return df

def fold_sequence(seq: str):
    '''

    '''
    structure, mfe = RNA.fold(seq)

    return structure, mfe

def add_sec_features(df: pd.DataFrame, sequence_name: str, structure_name: str, mfe_name: str) -> pd.DataFrame:
    '''

    '''
    sequences = df[sequence_name].tolist()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(fold_sequence, sequences)
    
    structures, mfes = zip(*results)
    df[structure_name] = structures
    df[mfe_name] = mfes

    return df

### symmetry ###

def _longest_symmetry_len(s: str) -> int:
    '''

    '''
    n = len(s)
    if n == 0:
        return 0

    def match(a, b):
        return (a == '.' and b == '.') or (a == '(' and b == ')') or (a == ')' and b == '(')

    def expand(left, right):
        # expand while s[left] mirrors s[right]
        while left >= 0 and right < n and match(s[left], s[right]):
            left -= 1
            right += 1
        # after breaking, last valid window is (left+1, right-1)
        return right - left - 1  # length

    best = 1  # any single char is symmetric with itself if it's '.'
    # Note: single '(' or ')' alone doesn't mirror to itself; but for odd centers we need
    # to ensure length 1 only counts if '.'; to keep it simple and strict, we can set best = 0
    # and rely on expansions. We'll handle single '.' properly via odd-center expansion below.
    best = 0

    for center in range(n):
        # odd-length center at 'center'
        # Only valid if s[center] mirrors itself (i.e., '.')
        if s[center] == '.':
            best = max(best, 1)
        # even-length center between center and center+1
        # Try both expansions:

        # odd-length expansion around (center, center)
        # This only grows if neighbors mirror; single '.' handled above
        l = r = center
        # Start with length 1 if '.' else 0; then expand outward
        cur_len = 1 if s[center] == '.' else 0
        if cur_len == 1:
            # already counted single dot; now try to expand further
            l -= 1
            r += 1
            while l >= 0 and r < n and match(s[l], s[r]):
                cur_len += 2
                l -= 1
                r += 1
            best = max(best, cur_len)

        # even-length expansion around (center, center+1)
        if center + 1 < n and match(s[center], s[center+1]):
            l, r = center, center + 1
            cur_len = 0
            while l >= 0 and r < n and match(s[l], s[r]):
                cur_len += 2
                l -= 1
                r += 1
            best = max(best, cur_len)

    return best

def add_max_symmetry(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    out = df.copy()
    out["max_symmetry"] = out["structure"].astype(str).apply(_longest_symmetry_len)
    out["full_symmetry"] = out.apply(
        lambda row: row["max_symmetry"] == len(str(row["structure"])), axis=1
        )
    return out

### sec ###

def _pairs_from_dotbracket(s: str):
    """
    
    """
    if not isinstance(s, str):
        return []
    stack = []
    pairs = []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, i))
            else:
                # unbalanced right paren; ignore
                pass
    pairs.sort()
    return pairs

def _stems_from_pairs(pairs):
    """
    
    """
    if not pairs:
        return []
    stems = []
    cur_len = 1
    for (i1, j1), (i2, j2) in zip(pairs, pairs[1:]):
        if i2 == i1 + 1 and j2 == j1 - 1:  # stacked
            cur_len += 1
        else:
            stems.append(cur_len)
            cur_len = 1
    stems.append(cur_len)
    return stems

def _hairpin_loop_sizes(s: str, pairs):
    """
    Hairpin loop = a base pair (i,j) whose enclosed region has no other pairs.
    Size = number of dots between i and j.
    """
    sizes = []
    for i, j in pairs:
        inner = s[i+1:j]
        if inner and ("(" not in inner and ")" not in inner):
            sizes.append(len(inner))
    return sizes

def _external_unpaired_count(s: str):
    """
    Dots not enclosed by any pair (i.e., in the external loop).
    """
    pairs = _pairs_from_dotbracket(s)
    covered = set()
    for i, j in pairs:
        covered.update(range(i, j+1))
    return sum(1 for k, ch in enumerate(s) if ch == "." and k not in covered)

def _pair_spans(pairs):
    """
    For each pair (i,j) return j-i (span/arc length).
    """
    return [(j - i) for i, j in pairs]

def add_bp_count(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    df["bp_count"] = df["structure"].apply(
        lambda s: s.count("(") if isinstance(s, str) else None
    )
    return df

def add_bp_density(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def f(s):
        if not isinstance(s, str) or len(s) == 0:
            return None
        # each "(" corresponds to one base pair; density over length
        return s.count("(") / len(s)
    df["bp_density"] = df["structure"].apply(f)
    return df

def add_unpaired_count(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    df["unpaired_count"] = df["structure"].apply(
        lambda s: s.count(".") if isinstance(s, str) else None
    )
    return df

def add_unpaired_density(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def f(s):
        if not isinstance(s, str) or len(s) == 0:
            return None
        return s.count(".") / len(s)
    df["unpaired_density"] = df["structure"].apply(f)
    return df

def add_stem_count(df: pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    df["stem_count"] = df["structure"].apply(
        lambda s: len(_stems_from_pairs(_pairs_from_dotbracket(s))) if isinstance(s, str) else None
    )
    return df

def add_stem_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    
    '''
    def f(s):
        if not isinstance(s, str):
            return (None, None, None)

        stems = _stems_from_pairs(_pairs_from_dotbracket(s))
        if not stems:
            return (0, 0.0, 0)
        
        if isinstance(stems[0], list):
            stems = [len(seg) for seg in stems]

        return (max(stems), sum(stems) / len(stems), min(stems))

    out = df["structure"].apply(f)
    df["stem_len_max"]  = out.str[0]
    df["stem_len_mean"] = out.str[1]
    df["stem_len_min"]  = out.str[2]
    return df

def add_hairpin_count(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    df["hairpin_count"] = df["structure"].apply(
        lambda s: len(_hairpin_loop_sizes(s, _pairs_from_dotbracket(s))) if isinstance(s, str) else None
    )
    return df

def add_hairpin_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def f(s):
        if not isinstance(s, str):
            return (None, None, None)
        sizes = _hairpin_loop_sizes(s, _pairs_from_dotbracket(s))
        if not sizes:
            return (0, None, None)
        return (sum(sizes)/len(sizes), min(sizes), max(sizes))
    out = df["structure"].apply(f)
    df["hairpin_size_mean"] = out.apply(lambda t: t[0])
    df["hairpin_size_min"]  = out.apply(lambda t: t[1])
    df["hairpin_size_max"]  = out.apply(lambda t: t[2])
    return df

def add_external_unpaired_density(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def f(s):
        if not isinstance(s, str) or len(s) == 0:
            return None
        ext = _external_unpaired_count(s)
        return ext / len(s)
    df["external_unpaired_density"] = df["structure"].apply(f)
    return df

def add_pair_span_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def f(s):
        if not isinstance(s, str):
            return (None, None, None)
        spans = _pair_spans(_pairs_from_dotbracket(s))
        if not spans:
            return (0, None, None)
        return (sum(spans)/len(spans), min(spans), max(spans))
    out = df["structure"].apply(f)
    df["pair_span_mean"] = out.apply(lambda t: t[0])  # average distance between paired bases
    df["pair_span_min"]  = out.apply(lambda t: t[1])
    df["pair_span_max"]  = out.apply(lambda t: t[2])
    return df

def add_free_end_lengths(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    def f(s):
        if not isinstance(s, str) or len(s) == 0:
            return (None, None)
        # 5' free tail = leading dots before first '(' or ')'
        left = 0
        for ch in s:
            if ch == ".":
                left += 1
            else:
                break
        # 3' free tail = trailing dots after last '(' or ')'
        right = 0
        for ch in reversed(s):
            if ch == ".":
                right += 1
            else:
                break
        return (left, right)
    out = df["structure"].apply(f)
    df["free_5prime_len"] = out.apply(lambda t: t[0])
    df["free_3prime_len"] = out.apply(lambda t: t[1])
    return df

def add_branch_point_count(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    # proxy for multiloops: count positions where a run of '(' ends and immediately
    # later another '(' appears after at least one dot inside the same enclosure.
    # Not a perfect multiloop detector, but useful as a "branchiness" index.
    import re
    pattern = re.compile(r"\((?:[^()]*\([^()]*\))+[^()]*\)")
    def f(s):
        if not isinstance(s, str):
            return None
        # count non-overlapping complex enclosures with >=2 internal stems
        return len(pattern.findall(s))
    df["branch_point_count"] = df["structure"].apply(f)
    return df

### hybrid ###

def _pairs_from_dotbracket(s: str):
    """
    Return list of base-pair tuples (i, j), 0-based, i<j.
    """
    if not isinstance(s, str):
        return []
    stack, pairs = [], []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    pairs.sort()
    return pairs

def _stems_from_pairs(pairs):
    """
    Group consecutive stacked pairs into stems; return list of lists of pairs.
    """
    if not pairs:
        return []
    stems = [[pairs[0]]]
    for (i1, j1), (i2, j2) in zip(pairs, pairs[1:]):
        if i2 == i1 + 1 and j2 == j1 - 1:
            stems[-1].append((i2, j2))
        else:
            stems.append([(i2, j2)])
    return stems

def _hairpin_loop_indices(structure: str, pairs):
    """
    Return list of index-lists for hairpin loops (unpaired indices between i and j when no inner pairs).
    """
    loops = []
    for i, j in pairs:
        inner = structure[i+1:j]
        if inner and ("(" not in inner and ")" not in inner):
            loops.append(list(range(i+1, j)))
    return loops

def _inside_any_pair_mask(n, pairs):
    """Mask of indices that lie inside at least one (i,j) inclusive interval."""
    mask = [False]*n
    for i, j in pairs:
        for k in range(i, j+1):
            mask[k] = True
    return mask

def _category_masks(structure: str):
    """
    Return dict of boolean masks (lists) over indices:
    paired, unpaired, external_unpaired, hairpin_unpaired, internal_unpaired
    """
    n = len(structure)
    pairs = _pairs_from_dotbracket(structure)
    paired = [False]*n
    for i, j in pairs:
        paired[i] = True
        paired[j] = True
    unpaired = [not p for p in paired]

    inside_mask = _inside_any_pair_mask(n, pairs)
    hairpin_sets = [set(idx_list) for idx_list in _hairpin_loop_indices(structure, pairs)]
    hairpin_unpaired = [False]*n
    for S in hairpin_sets:
        for k in S:
            hairpin_unpaired[k] = True

    internal_unpaired = [unpaired[k] and inside_mask[k] and not hairpin_unpaired[k] for k in range(n)]
    external_unpaired = [unpaired[k] and not inside_mask[k] for k in range(n)]

    return dict(
        paired=paired,
        unpaired=unpaired,
        external_unpaired=external_unpaired,
        hairpin_unpaired=hairpin_unpaired,
        internal_unpaired=internal_unpaired,
        pairs=pairs
    )

def _safe_upper(s):
    return s.upper() if isinstance(s, str) else s

def add_gc_overall(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add overall GC fraction of dvg_sequence (GC / length), NaN-safe.
    '''
    def f(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return None
        seq = _safe_upper(seq)
        gc = seq.count("G") + seq.count("C")
        return gc / len(seq)
    df["GC_overall"] = df["dvg_sequence"].apply(f)
    return df

def add_gc_paired_unpaired(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add GC fraction among paired bases and among unpaired bases (from dot-bracket).
    '''
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (None, None)
        masks = _category_masks(struct)
        paired_idx   = [i for i, b in enumerate(masks["paired"]) if b]
        unpaired_idx = [i for i, b in enumerate(masks["unpaired"]) if b]

        def content_gc(idxs):
            if not idxs:
                return None
            s = sum(1 for i in idxs if seq[i] in ("G","C"))
            return s / len(idxs)

        return (content_gc(paired_idx), content_gc(unpaired_idx))
    out = df.apply(f, axis=1)
    df["GC_paired"]   = out.apply(lambda t: t[0])
    df["GC_unpaired"] = out.apply(lambda t: t[1])
    return df

def add_canonical_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Count and fraction of canonical pairs: GC, AU, GU, and noncanonical among all pairs.
    '''
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (0, 0, 0, 0, None, None, None, None)
        pairs = _pairs_from_dotbracket(struct)
        c_gc = c_au = c_gu = c_non = 0
        for i, j in pairs:
            a, b = seq[i], seq[j]
            pair = a + b
            if pair in ("GC","CG"):
                c_gc += 1
            elif pair in ("AU","UA"):
                c_au += 1
            elif pair in ("GU","UG"):
                c_gu += 1
            else:
                c_non += 1
        tot = len(pairs)
        if tot == 0:
            return (0,0,0,0, None, None, None, None)
        return (c_gc, c_au, c_gu, c_non, c_gc/tot, c_au/tot, c_gu/tot, c_non/tot)
    out = df.apply(f, axis=1)
    df["pair_GC_count"] = out.apply(lambda t: t[0])
    df["pair_AU_count"] = out.apply(lambda t: t[1])
    df["pair_GU_count"] = out.apply(lambda t: t[2])
    df["pair_noncanon_count"] = out.apply(lambda t: t[3])
    df["pair_GC_content"] = out.apply(lambda t: t[4])
    df["pair_AU_content"] = out.apply(lambda t: t[5])
    df["pair_GU_content"] = out.apply(lambda t: t[6])
    df["pair_noncanon_content"] = out.apply(lambda t: t[7])
    return df

def add_stem_end_pair_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    '''
    For each stem, look at both terminal base pairs and report the fraction that are GC/AU/GU.
    '''
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (None, None, None)
        stems = _stems_from_pairs(_pairs_from_dotbracket(struct))
        end_pairs = []
        for stem in stems:
            if not stem:
                continue
            end_pairs.append(stem[0])
            if len(stem) > 1:
                end_pairs.append(stem[-1])
        if not end_pairs:
            return (None, None, None)
        gc = au = gu = 0
        for i, j in end_pairs:
            pair = seq[i] + seq[j]
            if pair in ("GC","CG"): gc += 1
            elif pair in ("AU","UA"): au += 1
            elif pair in ("GU","UG"): gu += 1
        tot = len(end_pairs)
        return (gc/tot, au/tot, gu/tot)
    out = df.apply(f, axis=1)
    df["stem_end_GC_content"] = out.apply(lambda t: t[0])
    df["stem_end_AU_content"] = out.apply(lambda t: t[1])
    df["stem_end_GU_content"] = out.apply(lambda t: t[2])
    return df

def add_hairpin_closing_pair_content(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Among hairpin loops, what fraction are closed by GC/AU/GU/noncanonical?
    '''
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (None, None, None, None)
        pairs = _pairs_from_dotbracket(struct)
        hp_pairs = []
        for i, j in pairs:
            inner = struct[i+1:j]
            if inner and ("(" not in inner and ")" not in inner):
                hp_pairs.append((i, j))
        if not hp_pairs:
            return (None, None, None, None)
        c_gc = c_au = c_gu = c_non = 0
        for i, j in hp_pairs:
            pair = seq[i] + seq[j]
            if pair in ("GC","CG"): c_gc += 1
            elif pair in ("AU","UA"): c_au += 1
            elif pair in ("GU","UG"): c_gu += 1
            else: c_non += 1
        tot = len(hp_pairs)
        return (c_gc/tot, c_au/tot, c_gu/tot, c_non/tot)
    out = df.apply(f, axis=1)
    df["hairpin_close_GC_content"]  = out.apply(lambda t: t[0])
    df["hairpin_close_AU_content"]  = out.apply(lambda t: t[1])
    df["hairpin_close_GU_content"]  = out.apply(lambda t: t[2])
    df["hairpin_close_noncanon_content"] = out.apply(lambda t: t[3])
    return df

def add_tetraloop_motif_counts(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Count classic stable tetraloops in hairpins: GNRA, UNCG, CUUG (case-insensitive).
    '''
    # GNRA = G N R A (R = A/G)
    gnra = re.compile(r"^G[ACGU][AG]A$", re.IGNORECASE)
    uncg = re.compile(r"^U[ACGU]CG$", re.IGNORECASE)
    cuug = re.compile(r"^CUUG$", re.IGNORECASE)
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (0, 0, 0)
        pairs = _pairs_from_dotbracket(struct)
        loops = _hairpin_loop_indices(struct, pairs)
        c_gnra = c_uncg = c_cuug = 0
        for idxs in loops:
            loop_seq = "".join(seq[i] for i in idxs)
            if len(loop_seq) == 4:
                if gnra.match(loop_seq): c_gnra += 1
                if uncg.match(loop_seq): c_uncg += 1
                if cuug.match(loop_seq): c_cuug += 1
        return (c_gnra, c_uncg, c_cuug)
    out = df.apply(f, axis=1)
    df["motif_GNRA_count"] = out.apply(lambda t: t[0])
    df["motif_UNCG_count"] = out.apply(lambda t: t[1])
    df["motif_CUUG_count"] = out.apply(lambda t: t[2])
    return df

def add_loop_au_content(df: pd.DataFrame) -> pd.DataFrame:
    '''
    AU fraction in (a) all unpaired, (b) hairpin loops, (c) internal/multi/bulge, (d) external.
    '''
    def content_au(seq, idxs):
        if not idxs:
            return None
        n = len(idxs)
        au = sum(1 for i in idxs if seq[i] in ("A","U"))
        return au / n

    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (None, None, None, None)
        masks = _category_masks(struct)
        all_unp = [i for i,b in enumerate(masks["unpaired"]) if b]
        hairpin  = [i for i,b in enumerate(masks["hairpin_unpaired"]) if b]
        internal = [i for i,b in enumerate(masks["internal_unpaired"]) if b]
        external = [i for i,b in enumerate(masks["external_unpaired"]) if b]
        return (
            content_au(seq, all_unp),
            content_au(seq, hairpin),
            content_au(seq, internal),
            content_au(seq, external),
        )
    out = df.apply(f, axis=1)
    df["AU_unpaired_content"] = out.apply(lambda t: t[0])
    df["AU_hairpin_content"]  = out.apply(lambda t: t[1])
    df["AU_internal_content"] = out.apply(lambda t: t[2])
    df["AU_external_content"] = out.apply(lambda t: t[3])
    return df

def add_tail_gc_content(df: pd.DataFrame) -> pd.DataFrame:
    '''
    GC fraction in 5' and 3' single-stranded tails (leading/trailing dots).
    '''
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct):
            return (None, None)
        # 5' tail
        left = 0
        for ch in struct:
            if ch == ".": left += 1
            else: break
        # 3' tail
        right = 0
        for ch in reversed(struct):
            if ch == ".": right += 1
            else: break
        def gc_content(subseq):
            if len(subseq) == 0: return None
            return (subseq.count("G")+subseq.count("C"))/len(subseq)
        return (gc_content(seq[:left]), gc_content(seq[len(seq)-right:]))
    out = df.apply(f, axis=1)
    df["GC_5prime_tail"] = out.apply(lambda t: t[0])
    df["GC_3prime_tail"] = out.apply(lambda t: t[1])
    return df

def add_start_codon_accessibility(df: pd.DataFrame) -> pd.DataFrame:
    '''
    If dvg_sequence contains AUG, report (aug_total, aug_unpaired, aug_unpaired_content) based on structure.
    '''
    def f(row):
        seq, struct = _safe_upper(row["dvg_sequence"]), row["structure"]
        if not isinstance(seq, str) or not isinstance(struct, str) or len(seq) != len(struct) or len(seq) < 3:
            return (0, 0, None)
        masks = _category_masks(struct)
        unp = masks["unpaired"]
        idxs = [m.start() for m in re.finditer(r"(?=AUG)", seq)]
        if not idxs:
            return (0, 0, None)
        unp_count = sum(1 for i in idxs if all(unp[i:i+3]))
        return (len(idxs), unp_count, unp_count/len(idxs))
    out = df.apply(f, axis=1)
    df["AUG_total"] = out.apply(lambda t: t[0])
    df["AUG_unpaired"] = out.apply(lambda t: t[1])
    df["AUG_unpaired_content"] = out.apply(lambda t: t[2])
    return df

### statistics ###

def p_to_stars(p: float) -> str:
    """

    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def mannwhitneyu_for_feature(
    true_df: pd.DataFrame,
    false_df: pd.DataFrame,
    feature_name: str,
    alternative: str = "two-sided",
    ):
    '''

    '''

    x_high = true_df[feature_name].to_numpy()
    y_low  = false_df[feature_name].to_numpy()

    n_high, n_low = len(x_high), len(y_low)
    if n_high == 0 or n_low == 0:
        raise ValueError("one of the groups has no observations with this threshold.")

    res = stats.mannwhitneyu(x_high, y_low, alternative=alternative, method="auto")
    U = float(res.statistic)
    p = float(res.pvalue)

    U_min = min(U, n_high * n_low - U)
    cliffs_delta = 1 - 2 * (U_min / (n_high * n_low))

    return {
        "feature_name": feature_name,
        "U": U,
        "pvalue": p,
        "n_high": n_high,
        "n_low": n_low,
        "median_high": float(np.median(x_high)),
        "median_low": float(np.median(y_low)),
        "cliffs_delta": float(cliffs_delta),
    }

### prediction ###

def _normal_int(mu: float, sigma: float) -> int:
    """
    Sample ~ N(mu, sigma), rounded to int, no clamping.
    """
    return int(round(random.gauss(mu, sigma)))

# TODO #
def generate_candidates(
    candidates_number: int,
    strain: str,
    segment: str,
    full_seq_len: int,
    ) -> List[Tuple[str, str, int, int, str, str, str, str, str, str, str, str, str]]:
    """

    """
    if full_seq_len < 2:
        raise ValueError("full_seq_len must be at least 2.")
    if candidates_number < 0:
        raise ValueError("candidates_number must be non-negative.")

    # Metadata options
    system_types = ["in vitro", "in vivo mouse", "in human"]
    library_layout = ["PAIRED", "SINGLE", "SINGLE & PAIRED"]
    library_selection = ["PCR", "RT-PCR", "cDNA", "PolyA", "WGA",
                         "PCR & cDNA", "RT-PCR & cDNA", "PolyA & PCR", "other"]
    library_source = ["GENOMIC", "TRANSCRIPTOMIC", "VIRAL RNA",
                      "TRANSCRIPTOMIC SINGLE CELL & VIRAL RNA", "OTHER", "VIRAL RNA & OTHER"]
    cells = ["A549", "human", "MDCK", "mouse", "A549 & HBEpC", 
             "A549 & HEK293FT & MRC-5 & WI-38 VA-13"]
    cellular_localization = ["mixed", "extracellular", "intracellular", ""]
    cellular_resolution = ["mixed", "single", "bulk", ""]
    time_point = ["mixed", "multiple", "single", ""]
    max_time = ["mixed", "12", "24", "48", "84", ""]

    # Pre-cycle each metadata feature to length candidates_number
    metadata_features = [system_types, library_layout, library_selection, library_source, cells, 
                         cellular_localization, cellular_resolution, time_point, max_time]
    cycle_metadata_features: List[List[str]] = []
    for meta in metadata_features:
        cycle_metadata_features.append([meta[i % len(meta)] for i in range(candidates_number)])

    # Heuristics for spread around the target means (scale with full_seq_len)
    start_mean = full_seq_len / 10.0
    end_mean   = full_seq_len - (full_seq_len / 10.0)
    # Standard deviations: gentle spread around the means
    start_std = max(1.0, full_seq_len / 20.0)
    end_std   = max(1.0, full_seq_len / 20.0)

    results: List[Tuple[str, str, int, int, str, str, str, str, str, str, str, str, str]] = []

    for i in range(candidates_number):
        # START: bounded-normal via clamping
        start_raw = _normal_int(start_mean, start_std)
        start = max(1, min(max(1, full_seq_len - 1), start_raw))

        # END: rejection on *unclamped* draws until > start, then clamp
        max_resamples = 64
        end = None
        for _ in range(max_resamples):
            e_raw = _normal_int(end_mean, end_std)
            if e_raw > start:
                # now clamp to sequence bounds
                end = max(2, min(full_seq_len, e_raw))
                # after clamping, end will still be > start because start <= full_seq_len - 1
                break

        if end is None:
            # Extremely unlikely fallback: draw uniformly from feasible range
            # (maintains end > start)
            if start >= full_seq_len:
                start = max(1, full_seq_len - 1)
            end = random.randint(start + 1, full_seq_len)

        results.append((
            strain,
            segment,
            start,
            end,
            cycle_metadata_features[0][i],
            cycle_metadata_features[1][i],
            cycle_metadata_features[2][i],
            cycle_metadata_features[3][i],
            cycle_metadata_features[4][i],
            cycle_metadata_features[5][i],
            cycle_metadata_features[6][i],
            cycle_metadata_features[7][i],
            cycle_metadata_features[8][i],
        ))

    return results

# TODO #
def build_df(candidates: list[tuple], isize: int=5) -> pd.DataFrame:
    """
    Build a DataFrame from candidate deletion specs.

    Supports either:
      (strain, seg, start, end)
    or
      (strain, seg, start, end, system_type, library_layout, library_selection, library_source)
    """
    records = []
    for cand in candidates:
        if len(cand) == 4:
            strain, seg, start, end = cand
            system_type = "other"
            library_layout = "other"
            library_selection = "other"
            library_source = "other"
            cells = "other"
            cellular_localization = "other"
            cellular_resolution = "other" 
            time_point = "other"
            max_time = "other"
        elif len(cand) == 13:
            (strain, seg, start, end,
             system_type, library_layout, library_selection, library_source, cells, 
                         cellular_localization, cellular_resolution, time_point, max_time) = cand
        else:
            raise ValueError(
                "Each candidate must be a 4-tuple "
                "(strain, seg, start, end) or an 13-tuple "
            )

        seq = get_sequence(strain, seg)

        start_idx = start - 1
        stop_idx = end

        left = seq[:start_idx]
        deleted = seq[start_idx:stop_idx]
        right = seq[stop_idx:]

        around = left[-isize*2:] + right[:isize*2]

        key = f"{seg}_{start}_{end}"

        records.append({
            "Segment": seg,
            "Start": start,
            "End": end,
            "NGS_read_count": 0,
            "key": key,
            "Strain": strain,
            "system_type": system_type,
            "library_layout": library_layout,
            "library_selection": library_selection,
            "library_source": library_source,
            "cells": cells,
            "cellular_localization": cellular_localization,
            "cellular_resolution": cellular_resolution,
            "time_point": time_point,
            "max_time": max_time,
            "isize": isize,
            "full_seq": seq,
            "deleted_sequence": deleted,
            "seq_around_deletion_junction": around,
        })

    cols = [
        "Segment", "Start", "End", "NGS_read_count", "key",
        "Strain", "system_type", "library_layout", "library_selection", "library_source",
        "cells", "cellular_localization", "cellular_resolution", "time_point", "max_time",
        "isize", "full_seq", "deleted_sequence", "seq_around_deletion_junction"
        ]
    df = pd.DataFrame.from_records(records, columns=cols)
    df = df.astype({
        "Segment": "string",
        "Start": "int64",
        "End": "int64",
        "NGS_read_count": "int64",
        "key": "string",
        "Strain": "string",
        "system_type": "string",
        "library_layout": "string",
        "library_selection": "string",
        "library_source": "string",
        "cells": "string",
        "cellular_localization": "string",
        "cellular_resolution": "string",
        "time_point": "string",
        "max_time": "string",
        "isize": "int64",
        "full_seq": "string",
        "deleted_sequence": "string",
        "seq_around_deletion_junction": "string",
    })

    return df

###############
### visuals ###
###############

### color ###

def pick_colors(colors: list, n_needed: int, templates: dict=COLOR_TEMPLATES):
    """

    """
    n_colors = len(colors)
    
    if n_needed >= n_colors:
        return [colors[i % n_colors] for i in range(n_needed)]
    
    if templates and n_needed in templates:
        return [colors[i] for i in templates[n_needed]]
    
    step = n_colors / n_needed
    return [colors[int(i * step)] for i in range(n_needed)]

### general ###

def compute_percentile_rank_count_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    
    '''
    return (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

def compute_seq_feature_count_df(sequence: str, feature_name: str, filter: list):
    '''

    '''
    data = {
        feature_name: [],
        'count': []
    }

    for value in filter:
        count = sequence.count(value)
        data[feature_name].append(value)
        data['count'].append(count)

    count_df = pd.DataFrame(data)

    return count_df

def compute_feature_count_df(df: pd.DataFrame, feature_name: str, filter: list = []):
    '''

    '''
    feature_series = df[feature_name].dropna()
    feature_counts = feature_series.value_counts().to_dict()

    if not filter:
        filter = sorted(feature_series.unique().tolist())

    feature_count_df = pd.DataFrame({
        feature_name: filter,
        'count': [feature_counts.get(value, 0) for value in filter]
    })

    return feature_count_df

def compute_feature_freq_df(count_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    '''

    '''
    total = count_df['count'].sum()
    freq_df = count_df.copy()
    freq_df['freq'] = (freq_df['count'] / total) * 100

    return freq_df[[feature_name, 'freq']]

def subtract_freq_dfs(freq_name0: str, freq_df0: pd.DataFrame, freq_name1: str, freq_df1: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    '''

    '''
    diff_freq_df = pd.DataFrame({
        feature_name: freq_df0[feature_name],
        freq_name0: freq_df0['freq'],
        freq_name1: freq_df1['freq'],
        'difference': freq_df0['freq'] - freq_df1['freq']
    })
    return diff_freq_df

### heatmap ###

def compute_feature_count_heatmap_df(df: pd.DataFrame, feature_name: str, filter: list = []) -> pd.DataFrame:
    '''

    '''
    relevant_df = df[['NGS_percentile_rank', feature_name]].copy()

    if filter:
        relevant_df = relevant_df[relevant_df[feature_name].isin(filter)]
    else:
        filter = sorted(relevant_df[feature_name].dropna().unique().tolist())

    one_hot = pd.get_dummies(relevant_df[feature_name])
    one_hot = one_hot.reindex(columns=filter, fill_value=0)
    combined = pd.concat([relevant_df[['NGS_percentile_rank']], one_hot], axis=1)

    return combined.groupby('NGS_percentile_rank', as_index=False).sum()

def compute_feature_count_heatmap_sum_df(df: pd.DataFrame, feature_name: str):
    '''

    '''
    feature_columns = df.columns[1:]
    count_sums = df[feature_columns].sum()

    feature_count_df = pd.DataFrame({
        feature_name: feature_columns.astype(str),
        'count': count_sums.values.astype(int)
    })

    return feature_count_df

def compute_feature_freq_heatmap_df(feature_count_heatmap_df: pd.DataFrame, percentile_rank_count_df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    rank_counts = percentile_rank_count_df.set_index('NGS_percentile_rank')['count']
    pivot_df = feature_count_heatmap_df.set_index('NGS_percentile_rank')
    pivot_df = pivot_df.loc[pivot_df.index.intersection(rank_counts.index)]
    feature_freq_heatmap_df = pivot_df.div(rank_counts, axis=0) * 100

    return feature_freq_heatmap_df

def add_lin_reg_rows(freq_heatmap_df: pd.DataFrame) -> pd.DataFrame:
    '''
    
    '''
    X = np.arange(1, len(freq_heatmap_df) + 1).reshape(-1, 1)

    intercepts = {}
    coefficients = {}

    for column in freq_heatmap_df.columns:
        y = freq_heatmap_df[column].values

        valid_idx = ~np.isnan(y)
        if valid_idx.sum() < 2:
            intercepts[column] = 0.0
            coefficients[column] = 0.0
            continue

        model = LinearRegression().fit(X[valid_idx], y[valid_idx])
        intercepts[column] = model.intercept_
        coefficients[column] = model.coef_[0]

    reg_freq_heatmap_df = freq_heatmap_df.copy()
    reg_freq_heatmap_df.loc['intercept'] = intercepts
    reg_freq_heatmap_df.loc['coefficient'] = coefficients

    return reg_freq_heatmap_df

##############
### motifs ###
##############

### motifs ###

def generate_motifs(motif_length: int):
    '''

    '''
    nucleotides = ['A', 'C', 'G', 'U']
    motifs = [''.join(p) for p in it.product(nucleotides, repeat=motif_length)]
    
    return sorted(motifs)

def add_site_motifs(df: pd.DataFrame, motif_length: int):
    '''

    '''
    site1_motifs = []
    site2_motifs = []
    site3_motifs = []
    site4_motifs = []

    valid_indices = []
    skipped_count = 0

    for idx, row in df.iterrows():
        seq = row['full_seq']
        start = row['Start']
        end = row['End']

        start0 = start
        end0 = end - 1

        if (start0 - motif_length < 0 or 
            start0 + motif_length > len(seq) or 
            end0 - motif_length < 0 or 
            end0 + motif_length > len(seq)):
            skipped_count += 1
            continue

        site1_motif = seq[start0 - motif_length:start0]
        site2_motif = seq[start0:start0 + motif_length]
        site3_motif = seq[end0 - motif_length:end0]
        site4_motif = seq[end0:end0 + motif_length]

        site1_motifs.append(site1_motif)
        site2_motifs.append(site2_motif)
        site3_motifs.append(site3_motif)
        site4_motifs.append(site4_motif)
        valid_indices.append(idx)

    df = df.loc[valid_indices].copy().reset_index(drop=True)

    df['site1_motif'] = site1_motifs
    df['site2_motif'] = site2_motifs
    df['site3_motif'] = site3_motifs
    df['site4_motif'] = site4_motifs

    return df, skipped_count

def compute_full_seq_motif_freq_df(motif_length: int, data: str, strain: str, segment: str):
    '''

    '''
    motifs = generate_motifs(motif_length)

    dataset_names = get_dataset_names(cutoff=40, selection=data)
    strains = get_strains(dataset_names)

    if segment != 'all' and strain != 'all':
        full_seq = get_sequence(strain, segment)
        motif_count_df = compute_seq_feature_count_df(full_seq, 'motif', motifs)
        motif_freq_df = compute_feature_freq_df(motif_count_df, 'motif')
        return motif_freq_df
    
    comb_df = pd.DataFrame()

    if strain != 'all':
        for segment in SEGMENTS:
            full_seq = get_sequence(strain, segment)
            motif_count_df = compute_seq_feature_count_df(full_seq, 'motif', motifs)
            motif_freq_df = compute_feature_freq_df(motif_count_df, 'motif')
            if comb_df.empty:
                comb_df = motif_freq_df.copy()
            else:
                comb_df['freq'] += motif_freq_df['freq']

        comb_df['freq'] = comb_df['freq'] / len(SEGMENTS) 
        return comb_df
    
    if segment != 'all':
        for strain in strains:
            full_seq = get_sequence(strain, segment)
            motif_count_df = compute_seq_feature_count_df(full_seq, 'motif', motifs)
            motif_freq_df = compute_feature_freq_df(motif_count_df, 'motif')
            if comb_df.empty:
                comb_df = motif_freq_df.copy()
            else:
                comb_df['freq'] += motif_freq_df['freq']

        comb_df['freq'] = comb_df['freq'] / len(strains)
        return comb_df
    
    for strain in strains:
        for segment in SEGMENTS:
            full_seq = get_sequence(strain, segment)
            motif_count_df = compute_seq_feature_count_df(full_seq, 'motif', motifs)
            motif_freq_df = compute_feature_freq_df(motif_count_df, 'motif')
            if comb_df.empty:
                comb_df = motif_freq_df.copy()
            else:
                comb_df['freq'] += motif_freq_df['freq']

    comb_df['freq'] = comb_df['freq'] / (len(strains) * len(SEGMENTS))
    return comb_df

def insert_pseudo_motif(
    df: pd.DataFrame,
    motif_length: int,
    pseudo_motif: str,
    motif_site: str,
    feature_name: str,
    feature_threshold: float,
    proportion: float,
    replace_all: bool = False
    ):
    '''

    '''
    assert 0 <= proportion <= 1, 'proportion need to be between 0 and 1'
    assert len(pseudo_motif) == motif_length, 'pseudo_motif must have motif_length'

    df_mod = df.copy()
    skipped_count = 0
    rng = np.random.default_rng(SEED)

    candidates = df_mod[df_mod[feature_name] >= feature_threshold].copy()
    candidate_indices = candidates.index.tolist()
    n_modify = int(len(candidate_indices) * proportion)
    selected_indices = rng.choice(candidate_indices, size=n_modify, replace=False)

    if replace_all:
        all_indices = df_mod.index.tolist()
        other_indices = [i for i in all_indices if i not in selected_indices]
    else:
        other_indices = []

    custom_coords = re.fullmatch(r'(\d+)_(\d+)', motif_site)

    def get_pos(row):
        seq = row['full_seq']
        start0 = row['Start']
        end0 = row['End'] - 1

        if custom_coords:
            s = int(custom_coords.group(1))
            e = int(custom_coords.group(2))
            if s < 0 or e >= len(seq) or e < s or (e - s + 1) != motif_length:
                return None
            return s
        else:
            if motif_site == 'site1_motif':
                pos = start0 - motif_length
            elif motif_site == 'site2_motif':
                pos = start0
            elif motif_site == 'site3_motif':
                pos = end0 - motif_length
            elif motif_site == 'site4_motif':
                pos = end0
            else:
                return None
            if pos < 0 or pos + motif_length > len(seq):
                return None
            return pos

    def generate_random_motif():
        return ''.join(rng.choice(['A', 'C', 'G', 'U'], size=motif_length,
                                  p=[0.35, 0.195, 0.23, 0.225]))

    for idx in selected_indices:
        row = df_mod.loc[idx]
        seq = row['full_seq']
        pos = get_pos(row)
        if pos is None:
            skipped_count += 1
            continue
        new_seq = seq[:pos] + pseudo_motif + seq[pos + motif_length:]
        df_mod.at[idx, 'full_seq'] = new_seq

    for idx in other_indices:
        row = df_mod.loc[idx]
        seq = row['full_seq']
        pos = get_pos(row)
        if pos is None:
            skipped_count += 1
            continue
        random_motif = generate_random_motif()
        new_seq = seq[:pos] + random_motif + seq[pos + motif_length:]
        df_mod.at[idx, 'full_seq'] = new_seq

    return df_mod, skipped_count


