'''

'''
import os
import sys

import pandas as pd

sys.path.insert(0, "..")
from utils import add_dvg_sequence, add_dvg_length, add_remaining_sequence, add_cg_content, add_log_feature, add_norm_feature, save_df
from utils import RESULTSPATH

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, "preprocess")

if __name__ == "__main__":
    '''

    '''
    ### add reg_ ### if needed
    data = "IAV"
    strain = "PR8"
    segment = "PB1"
    motif_length = 3

    read_path, _ = os.path.split(RESULTSPATH)
    read_path = os.path.join(read_path, "dfs", data, strain, segment)
    fname = f"motif_length_{motif_length}" + ".csv"
    df = pd.read_csv(os.path.join(read_path, fname))

    feature = "NGS_read_count"
    log_feature = 'log_' + feature
    norm_log_feature = 'norm_log_' + feature
    df = add_log_feature(df, feature, log_feature)
    df = add_norm_feature(df, log_feature, norm_log_feature)

    df = add_remaining_sequence(df)
    df = add_dvg_sequence(df)
    df = add_dvg_length(df)
    df = add_cg_content(df)

    df = df.drop("remaining_sequence", axis=1)
    df = df.drop("NGS_read_count", axis=1)
    df = df.drop("log_NGS_read_count", axis=1)
    df = df.drop("key", axis=1)
    df = df.drop("isize", axis=1)
    df = df.drop("full_seq", axis=1)
    df = df.drop("deleted_sequence", axis=1)
    df = df.drop("seq_around_deletion_junction", axis=1)

    save_df(data, df, f"nn_pri_motif_length_{motif_length}", strain, segment)