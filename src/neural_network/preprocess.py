'''

'''
import os
import sys

import pandas as pd

sys.path.insert(0, "..")
from utils import add_motifs
from utils import add_dvg_sequence, save_df
from utils import RESULTSPATH

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, "preprocess")

if __name__ == "__main__":
    '''

    '''

    read_path, _ = os.path.split(RESULTSPATH)
    read_path = os.path.join(read_path, "dfs", "IBV", "all", "all")
    fname = "sec" + ".csv"
    df = pd.read_csv(os.path.join(read_path, fname))

    df = add_motifs(df, 5)
    df = add_dvg_sequence(df)

    df = df.drop("full_seq", axis=1)
    df = df.drop("deleted_sequence", axis=1)
    df = df.drop("seq_around_deletion_junction", axis=1)
    df = df.drop("remaining_sequence", axis=1)

    save_df("IBV", df, "nn", "", "")