'''

'''
import os
import sys
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from scipy.interpolate import CubicSpline

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import RNA

sys.path.insert(0, "..")

from utils import get_dataset_names

from utils import manage_separate_specifiers, clean_data_string
from utils import rename_feature, add_separate_ngs_features
from utils import add_ikey, add_metadata_ikey, manage_intersects
from utils import load_all_preprocessed, save_df

from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'statistics')


###############
### scripts ###
###############

### statistics ###

import numpy as np
import pandas as pd

def strain_segment_statistics(
    dfs: list[pd.DataFrame],
    folder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str
) -> None:
    """
    Concat a list of dfs, keep only required columns, compute per-(Strain, Segment)
    summary stats, round all derived numbers to integers, and save via save_df.
    """

    keep_cols = [
        "Segment",
        "Start",
        "End",
        "NGS_read_count",
        "norm_log_NGS_read_count",
        "Strain",
        "seq_around_deletion_junction",
        "dataset_name",
        "deletion_length",
        "5_end_length",
        "3_end_length",
        "dvg_length",
        "full_seq_length"
    ]

    if dfs is None or len(dfs) == 0:
        raise ValueError("dfs is empty")

    feature = 'norm_log_NGS_read_count'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, feature)

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[keep_cols].copy()

    num_cols = [
        "Start", "End", "NGS_read_count",
        "deletion_length", "5_end_length", "3_end_length",
        "dvg_length", "full_seq_length", "norm_log_NGS_read_count"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _agg_stats(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if len(x) == 0:
            return pd.Series(
                {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
            )

        return pd.Series(
            {
                "mean": int(np.round(x.mean())),
                "median": int(np.round(x.median())),
                "min": int(x.min()),
                "max": int(x.max()),
            }
        )

    gb = df.groupby(["Strain", "Segment"], dropna=False)

    stats_df = gb.size().rename("count").to_frame()
    stats_df["count"] = stats_df["count"].astype(int)

    for feat in [
        "Start",
        "End",
        "deletion_length",
        "5_end_length",
        "3_end_length",
        "dvg_length",
        "NGS_read_count",
        "norm_log_NGS_read_count",
    ]:
        tmp = gb[feat].apply(_agg_stats).unstack()
        tmp.columns = [f"{feat}_{k}" for k in tmp.columns]
        stats_df = stats_df.join(tmp)

    # full_seq_length: one representative value per group (always identical)
    stats_df["full_seq_length"] = gb["full_seq_length"].first().round().astype("Int64")

    stats_df = stats_df.reset_index()

    save_df(stats_df, "statistics", RESULTSPATH, folder, "stats", data, strain, segment, intersects)


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################
    
    ### DATASETS MULTI ###

    folder = 'unpooled'   # pooled | unpooled
    subfolder = 'base'   # base | structure

    data = 'all'   # all | IAV | IBV | H1N1
    strain = 'all'   # all | PR8
    segment = 'all'   # all | PB1
    intersects = 'median'   # all | meidian

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    strain_segment_statistics(dfs, folder, data, strain, segment, intersects)