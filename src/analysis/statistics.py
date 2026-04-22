'''
    statistical analyis on preprocessed datasets
'''

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "..")

from utils import get_dataset_names

from utils import manage_separate_specifiers, add_intersect_ngs_features
from utils import load_all_preprocessed, save_df

from utils import RESULTSPATH, DATASET_CUTOFF

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'statistics')


###############
### scripts ###
###############

### statistics ###

def run_strain_segment_statistics(
    dfs: list[pd.DataFrame],
    data: str = 'all',
    strain: str = 'all',
    segment: str = 'all',
    intersects: str = 'all'
) -> None:
    '''
        Compute per-dataset and per-segment summary statistics and save the resulting table.
        :param dfs: list of input dataframes
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''

    # define columns required for the statistics table
    keep_cols = [
        "segment",
        "start",
        "end",
        "NGS_read_count",
        "strain",
        "seq_around_deletion_junction",
        "dataset",
        "deletion_length",
        "5_end_length",
        "3_end_length",
        "DelVG_length",
        "full_seq_length",
        "site1_motif",
        "site2_motif",
        "site3_motif",
        "site4_motif",
    ]

    # ensure that at least one dataframe is provided
    if dfs is None or len(dfs) == 0:
        raise ValueError("dfs is empty")

    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # concatenate all selected dataframes
    df = pd.concat(dfs, ignore_index=True)

    # verify that all required columns are present
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    # keep only the columns relevant for summary statistics
    df = df[keep_cols].copy()

    # coerce numerical columns to numeric type
    num_cols = [
        "start", "end", "NGS_read_count",
        "deletion_length", "5_end_length", "3_end_length",
        "DelVG_length", "full_seq_length"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _agg_stats(x: pd.Series) -> pd.Series:
        '''
            Compute rounded summary statistics for a numerical series.
            :param x: numerical pandas series

            :return: series with mean, median, min, and max
        '''
        # drop missing values before computing summary statistics
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan})

        # return rounded summary statistics
        return pd.Series(
            {
                "mean": int(np.round(x.mean())),
                "median": int(np.round(x.median())),
                "min": int(x.min()),
                "max": int(x.max()),
            }
        )

    def _most_prominent(x: pd.Series) -> str:
        '''
            Return the most frequent non-null value in a series.
            :param x: pandas series

            :return: most frequent value as string
        '''
        # drop missing values before computing the mode
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        vc = x.astype(str).value_counts()
        if vc.empty:
            return np.nan

        # resolve ties deterministically by sorted order
        top_n = vc.iloc[0]
        top_vals = vc[vc == top_n].index.tolist()
        return sorted(top_vals)[0]

    # group by dataset and segment for summary statistic computation
    gb = df.groupby(["dataset", "segment"], dropna=False)

    # start statistics table with group sizes
    stats_df = gb.size().rename("count").to_frame()
    stats_df["count"] = stats_df["count"].astype(int)

    # add summary statistics for selected numerical features
    for feat in [
        "start",
        "end",
        "deletion_length",
        "5_end_length",
        "3_end_length",
        "DelVG_length",
        "NGS_read_count",
    ]:
        tmp = gb[feat].apply(_agg_stats).unstack()
        tmp.columns = [f"{feat}_{k}" for k in tmp.columns]
        stats_df = stats_df.join(tmp)

    # add one representative full sequence length per group
    stats_df["full_seq_length"] = gb["full_seq_length"].first().round().astype("Int64")

    # add most prominent motifs per group
    for motif_col in ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]:
        stats_df[motif_col] = gb[motif_col].apply(_most_prominent)

    # restore grouped columns as standard dataframe columns
    stats_df = stats_df.reset_index()

    # save statistics dataframe
    save_df(stats_df, "statistics", RESULTSPATH)

### size ###

def print_df_sizes(dfnames: list, dfs: list) -> int:
    '''
        Print dataset sizes and grouped row counts by strain and segment.
        :param dfnames: list of dataset names
        :param dfs: list of input dataframes

        :return: total number of rows across all dataframes
    '''
    # ensure dataset names and dataframes have matching lengths
    if len(dfs) != len(dfnames):
        raise ValueError("dfs and dfnames must have the same length.")

    total_rows = 0

    # print row count for each dataframe and accumulate total number of rows
    print("Dataset sizes (rows):")
    for name, df in zip(dfnames, dfs):
        n = len(df)
        total_rows += n
        print(f"  {name}: {n:,}")

    print("-" * 40)
    print(f"Total rows: {total_rows:,}")

    # stop early if all dataframes are empty
    if total_rows == 0:
        print("\nNo rows available for grouping (all DataFrames empty).")
        return total_rows

    # ensure required grouping columns are present
    required_cols = {"strain", "segment"}
    missing = required_cols - set(dfs[0].columns)

    # check across all dataframes for missing grouping columns
    for i, df in enumerate(dfs):
        miss_i = required_cols - set(df.columns)
        if miss_i:
            raise ValueError(f"Missing columns {miss_i} in dfs[{i}] needed for grouping.")

    # concatenate all dataframes for grouped summaries
    df_all = pd.concat(dfs, ignore_index=True)

    # normalize grouping columns to string type
    df_all["strain"] = df_all["strain"].astype(str)
    df_all["segment"] = df_all["segment"].astype(str)

    # compute row counts for each strain-segment combination
    comb_counts = (
        df_all
        .groupby(["strain", "segment"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["strain", "segment"])
    )

    print("\nCounts by strain × segment:")
    for _, r in comb_counts.iterrows():
        print(f"  {r['strain']} | {r['segment']}: {int(r['n_rows']):,}")

    # compute row counts for each strain across all segments
    strain_counts = (
        df_all
        .groupby(["strain"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )

    print("\nCounts by strain (all segments combined):")
    for _, r in strain_counts.iterrows():
        print(f"  {r['strain']}: {int(r['n_rows']):,}")

    return total_rows

### percentage ###

def print_percentage_of_value(df: pd.DataFrame, column: str, value) -> float:
    '''
        Print the percentage of rows in a column that match a given value.
        :param df: input dataframe
        :param column: column to evaluate
        :param value: value to compare against

        :return: percentage of matching rows
    '''
    # compute total number of rows in the dataframe
    total_rows = len(df)
    if total_rows == 0:
        print(0)
    
    # count rows matching the requested value and convert to percentage
    count = (df[column] == value).sum()
    percentage = (count / total_rows) * 100
    
    # print the computed percentage
    print(percentage)

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################
    
    ### REQUIRES CORESPONDING PREPROCESSING! ###

    ### DATA ###

    ### "pooled" or "unpooled" ###
    folder = 'unpooled'

    ### "primary" or "secondary" ###
    subfolder = 'secondary'

    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ### FOR FEATURE PERCENTAGE ###

    ### "all", "IAV", "IBV", dataset name, ... ###
    data = 'IAV'

    ### "all", "PR8", "Yamagata", ... ###
    strain = 'PR8'

    ### "all", "PB1", "PB2", ...###
    segment = 'PB1'

    ### modifier to manage intersects ###
    intersects = 'mean_dataset_0'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)
    dfs = [df[df["dataset"] == name].copy() for name in dfnames]

    ### select feature ###
    feature = 'time_point'

    ### feature value ###
    value = 'unknown'

    ###################
    ### run scripts ###
    ###################

    ### general statistics ###

    # run_strain_segment_statistics(dfs)
    # print_df_sizes(dfnames, dfs)

    ### feature percentage ###

    # print_percentage_of_value(df, feature, value)