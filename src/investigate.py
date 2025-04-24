'''
    investigate code and data structure
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from typing import Tuple
from collections import Counter

from decimal import Decimal

from typing import List

sys.path.insert(0, "..")
from utils import load_dataset, get_dataset_names, load_all, join_data, preprocess, generate_expected_data
from utils import add_dvg_sequence
from utils import SEGMENTS, RESULTSPATH, DATASET_STRAIN_DICT, CMAP, NUCLEOTIDES, CUTOFF, DATAPATH

RESULTSPATH, _ = os.path.split(RESULTSPATH)

def tryout(dfnames: list)-> list:
    '''
        des
        :param p0:
        :param p1:
        :param p2:

        :retrun Tuple
            0
            1
            3
    '''
    dfs = list()
    for dfname in dfnames:
        strain = DATASET_STRAIN_DICT[dfname]
        df = join_data(load_dataset(dfname))
        dfs.append(df)
    return dfs

def resave(dfnames: list, dfs: list, name_mod: str):
    '''
        Resaves the given Pandas Dataframes as a csv-file.
        :param dfnames: list of dataset names, each is one experiment
        :param dfs: list of Pandas DataFrames with DelVG data
        :param name_mod: name to modify the prefix of the csv-file name
    '''
    if len(dfnames) != len(dfs):
        print("lengths of dfnames and dfs diverge")
        return
    save_path = os.path.join(RESULTSPATH, "test")
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(dfnames)):
        fname = f"{name_mod}_{dfnames[i]}.csv"
        dfs[i].to_csv(os.path.join(save_path, fname))


if __name__ == "__main__":
    '''

    '''

    dfnames = ["Berry2021_B_Yam"]
    dfs, _ = load_all(dfnames)
    dfname = dfnames[0]
    df = dfs[0]

    df = add_dvg_sequence(df)
    dfs = [df]
    resave(dfnames, dfs, "dvg")

