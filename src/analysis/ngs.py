'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from typing import Tuple

sys.path.insert(0, "..")
from utils import load_dataset, get_dataset_names, join_data, preprocess, generate_expected_data, get_seq_len
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS, load_all, get_sequence, calculate_direct_repeat, count_direct_repeats_overall

RESULTSPATH, _ = os.path.split(RESULTSPATH)

###############################
######## LOG NGS STATS ########
###############################

def create_norm_ngs_read_count_abs_histogram(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    if df.empty:
        print(f"empty df")
        return

    num_rows = df.shape[0]

    if 'norm_log_NGS_read_count' not in df.columns:
        print(f"'norm_log_NGS_read_count' column not found in the DataFrame.")
        return
    
    max_value = 1.0
    bins = np.linspace(0, max_value, 21)

    hist, bin_edges = np.histogram(df['norm_log_NGS_read_count'], bins=bins)

    ax.bar(bin_edges[:-1], hist, width=0.03, color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("normalized NGS Read Count")
    ax.set_ylabel("number of DVGs (-)")
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f"{b:.2f}" for b in bin_edges])

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"norm_ngs_read_count_abs_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_norm_ngs_read_count_histogram(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    if df.empty:
        print(f"empty df")
        return

    num_rows = df.shape[0]

    if 'norm_log_NGS_read_count' not in df.columns:
        print(f"'norm_log_NGS_read_count' column not found in the DataFrame.")
        return
    
    max_value = 1.0
    bins = np.linspace(0, max_value, 21)

    hist, bin_edges = np.histogram(df['norm_log_NGS_read_count'], bins=bins)

    hist = hist / num_rows * 100 if num_rows > 0 else hist

    ax.bar(bin_edges[:-1], hist, width=0.03, color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("normalized NGS Read Count")
    ax.set_ylabel("distribution of DVGs (%)")
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f"{b:.2f}" for b in bin_edges])

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"norm_ngs_read_count_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

###############################
########## NGS STATS ##########
###############################

def create_ngs_read_count_abs_histogram(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    if df.empty:
        print(f"empty df")
        return

    total_reads = df['NGS_read_count'].sum()
    if total_reads == 0:
        print(f"total NGS read count is 0")
        return

    max_read_count = df['NGS_read_count'].max()
    max_read_count = 400
    upper_limit = (max_read_count // 100 + 1) * 100

    bins = range(0, upper_limit + 20, 20)
    hist, bin_edges = np.histogram(df['NGS_read_count'], bins=bins)

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel(f"NGS read count")
    ax.set_ylabel("number of DVGs (-)")
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([str(b) for b in bin_edges])

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_read_count_abs_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_ngs_read_count_histogram(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    if df.empty:
        print(f"empty df")
        return

    total_reads = df['NGS_read_count'].sum()
    if total_reads == 0:
        print(f"total NGS read count is 0")
        return
    
    num_rows = df.shape[0]

    max_read_count = df['NGS_read_count'].max()
    max_read_count = 400
    upper_limit = (max_read_count // 100 + 1) * 100

    bins = range(0, upper_limit + 20, 20)
    hist, bin_edges = np.histogram(df['NGS_read_count'], bins=bins)

    hist = hist / num_rows * 100

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel(f"NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([str(b) for b in bin_edges])

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_read_count_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

###############################

def calculate_ngs_mean_std(df):
    mean = df['NGS_read_count'].mean()
    mean = np.round(mean, 2)
    std_dev = df['NGS_read_count'].std()
    std_dev = np.round(std_dev, 2)
    return mean, std_dev

def add_dist_ngs_read_count(df: pd.DataFrame):
    sum = df['NGS_read_count'].sum()
    df['dist_NGS_read_count'] = df['NGS_read_count'] / sum
    return df

###############################
###### NGS LOG-TRANSFORM ######
###############################

def add_norm_log_ngs_read_count(df: pd.DataFrame):
    df = add_log_ngs_read_count(df)
    max_log = df['log_NGS_read_count'].max()
    min_log = df['log_NGS_read_count'].min()
    df['norm_log_NGS_read_count'] = (df['log_NGS_read_count'] - min_log) / (max_log - min_log)
    return df

def add_log_ngs_read_count(df: pd.DataFrame):
    df['log_NGS_read_count'] = np.log10(df['NGS_read_count'])
    return df

###############################

def perform_t_test_log(df: pd.DataFrame):
    df = add_norm_log_d_ngs_read_count(df)
    log_d_counts = df['norm_log_d_NGS_read_count']
    df = add_norm_log_n_ngs_read_count(df)
    log_n_counts = df['norm_log_n_NGS_read_count']
    t_stat, p_value = stats.ttest_ind(log_n_counts, log_d_counts)
    return p_value

###############################

def add_norm_log_d_ngs_read_count(df: pd.DataFrame):
    df = add_log_d_ngs_read_count(df)
    max_log_d = df['log_d_NGS_read_count'].max()
    min_log_d = df['log_d_NGS_read_count'].min()
    df['norm_log_d_NGS_read_count'] = (df['log_d_NGS_read_count'] - min_log_d) / (max_log_d - min_log_d)
    return df

def add_log_d_ngs_read_count(df: pd.DataFrame):
    df['log_d_NGS_read_count'] = np.log10(df['NGS_read_count'])
    return df

###############################

def add_norm_log_n_ngs_read_count(df: pd.DataFrame):
    df = add_log_n_ngs_read_count(df)
    max_log_n = df['log_n_NGS_read_count'].max()
    min_log_n = df['log_n_NGS_read_count'].min()
    df['norm_log_n_NGS_read_count'] = (df['log_n_NGS_read_count'] - min_log_n) / (max_log_n - min_log_n)
    return df

def add_log_n_ngs_read_count(df: pd.DataFrame):
    df['log_n_NGS_read_count'] = np.log(df['NGS_read_count'])
    return df

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    ###############################
    ######## LOG NGS STATS ########
    ###############################

    dfnames = ["Berry2021_B_Yam"]
    dfs, _ = load_all(dfnames)
    dfname = dfnames[0]

    df = pd.concat(dfs, axis=0)
    df = add_norm_log_ngs_read_count(df)
    create_norm_ngs_read_count_histogram(dfname, df)
    create_norm_ngs_read_count_abs_histogram(dfname, df)

    ###############################
    ########## NGS STATS ##########
    ###############################

    # dfnames = ["Berry2021_B_Yam"]
    # dfs, _ = load_all(dfnames)
    # dfname = dfnames[0]

    # df = pd.concat(dfs, axis=0)
    # create_ngs_read_count_abs_histogram(dfname, df)
    # create_ngs_read_count_histogram(dfname, df)

    ###############################
    ###### NGS LOG-TRANSFORM ######
    ###############################

    # dfnames = ["Berry2021_B_Yam"]
    # dfs, _ = load_all(dfnames)
    # dfname = dfnames[0]

    # df = pd.concat(dfs, axis=0)
    # p_value = perform_t_test_log(df)
    # print(p_value)



