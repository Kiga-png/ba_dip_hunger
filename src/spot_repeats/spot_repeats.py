'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import get_sequence
from utils import DATAPATH, RESULTSPATH, SEGMENTS, DATASET_STRAIN_DICT

RESULTSPATH, _ = os.path.split(RESULTSPATH)


##############################
### spot potential repeats ###
##############################

def spot_repeats_all(min_len: int = 5):
    strains = set(DATASET_STRAIN_DICT.values())
    for strain in strains:
        spot_repeats(strain, min_len)

def spot_repeats(strain: str, min_len: int):
    save_path = os.path.join(DATAPATH, f"repeats/{strain}")
    os.makedirs(save_path, exist_ok=True)
    for segment in SEGMENTS:
        seq = get_sequence(strain, segment)
        repeats = spot_repeats_segment(seq, min_len)
        fname = f"{segment}.csv"
        repeats.to_csv(os.path.join(save_path, fname))

def spot_repeats_segment(seq: str, min_len: int):
    repeats = []
    seq_len = len(seq)
    for start in range(seq_len):
        for length in range(1, seq_len - start + 1):
            if length < min_len:
                continue
            substring = seq[start:start + length]
            next_occurrence = seq.find(substring, start + length)
            while next_occurrence != -1:
                repeats.append((start + 1, start + length, next_occurrence + 1, next_occurrence + length))
                next_occurrence = seq.find(substring, next_occurrence + 1)
    df = pd.DataFrame(repeats, columns=["start_ref", "end_ref", "start_rep", "end_rep"])
    df = remove_rows_with_smaller_start(df)
    return df

def remove_rows_with_smaller_start(df: pd.DataFrame):
    grouped = df.groupby(['end_ref', 'end_rep'])

    def keep_max_start_ref(group):
        return group[group['start_ref'] == group['start_ref'].min()]
    
    filtered_df = grouped.apply(keep_max_start_ref).reset_index(drop=True)
    return filtered_df


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    ##############################
    ### spot potential repeats ###
    ##############################

    spot_repeats_all()
