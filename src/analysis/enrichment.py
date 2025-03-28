'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
from typing import Tuple

sys.path.insert(0, "..")
from utils import load_dataset, get_dataset_names, join_data, preprocess, generate_expected_data, get_seq_len
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS, load_all, get_sequence, calculate_direct_repeat, count_direct_repeats_overall

from ngs import add_log_ngs_read_count

RESULTSPATH, _ = os.path.split(RESULTSPATH)

###################################
####### NGS ENRICHMENT RATIO ######
###################################

def create_ngs_enrichment_ratio_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    if df.empty:
        print("empty df")
        return
    
    total_count = df['NGS_read_count_ratio'].sum()
    if total_count == 0:
        print("ratio_sum is 0")
        return

    df['normalized_ratio'] = df['NGS_read_count_ratio'] / total_count * 100
    unique_lengths = df['nucleotide'].unique()
    ax.bar(unique_lengths, df['normalized_ratio'], color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("nucleotide (-)")
    ax.set_ylabel("NGS read count distribution (%)")
    ax.set_xticks(unique_lengths)
    ax.set_xticklabels(["A", "C", "G", "U"]) 

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_ne_ratio_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi = 300)
    plt.close()

def ngs_enrichment_ratio_concat(dfs: list):
    ext_dfs = []
    concat_df = pd.DataFrame
    for df in dfs:
        ext_df = ngs_enrichment_ratio(df)
        ext_dfs.append(ext_df)
    concat_df = pd.concat(ext_dfs, axis=0)
    agg_df = concat_df.groupby('nucleotide', as_index=False)['NGS_read_count_ratio'].sum()
    agg_df['NGS_read_count_ratio'] = agg_df['NGS_read_count_ratio'] / len(dfs)
    return agg_df

def ngs_enrichment_ratio(df: pd.DataFrame):
    ngs_read_count_sum = 0
    nucleotides = ['A', 'C', 'G', 'U']
    ngs_df = pd.DataFrame({
    'nucleotide': nucleotides,
    'NGS_read_count_ratio': [0] * len(nucleotides)
    })
    for index, row in df.iterrows():
        ngs_read_count = row["NGS_read_count"]
        ngs_read_count_sum = ngs_read_count_sum + ngs_read_count
        jun_seq = row["seq_around_deletion_junction"]
        i = jun_seq[4]
        j = jun_seq[14]
        ngs_df.loc[ngs_df['nucleotide'] == i, 'NGS_read_count_ratio'] += ngs_read_count / 2
        ngs_df.loc[ngs_df['nucleotide'] == j, 'NGS_read_count_ratio'] += ngs_read_count / 2
    ngs_df['NGS_read_count_ratio'] /= ngs_read_count_sum
    return ngs_df

#####################################
####### ENRICHMENT POSITIONS ########
#####################################

def create_enrichment_nucleotide_position_histogram(dfname: str, strain: str, segments: list, dfs: list, p_segments: list, p_dfs: list):
    for segment in SEGMENTS:
        index = segments.index(segment)
        p_index = p_segments.index(segment)
        create_enrichment_nucleotide_position_histogram_segment(dfname, strain, segment, dfs[index], p_dfs[p_index])

def create_enrichment_nucleotide_position_histogram_segment(dfname: str, strain: str, segment: str, df: pd.DataFrame, p_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    
    if df.empty:
        print(f"{segment}: empty df")
        return
    
    if p_df.empty:
        print(f"{segment}: empty p_df")
        return
    
    total_hits = df['hits'].sum()
    if total_hits == 0:
        print(f"{segment}: total hits is 0")

    p_total_hits = p_df['hits'].sum()
    if p_total_hits == 0:
        print(f"{segment}: total p_hits is 0")
        return

    max_position = p_df['nucleotide-position'].max()
    upper_limit = (max_position // 100 + 1) * 100

    if total_hits > 0:
        df['normalized_hits'] = df['hits'] / total_hits * 100
        bins = pd.cut(df['nucleotide-position'], bins=range(0, upper_limit + 50, 20), right=False)
        binned_hits = df.groupby(bins)['normalized_hits'].sum().reset_index()
    else:
        binned_hits = pd.DataFrame(columns=['nucleotide-position', 'normalized_hits'])

    p_df['normalized_hits'] = p_df['hits'] / p_total_hits * 100
    p_bins = pd.cut(p_df['nucleotide-position'], bins=range(0, upper_limit + 50, 20), right=False)
    p_binned_hits = p_df.groupby(p_bins)['normalized_hits'].sum().reset_index()

    n = len(p_binned_hits['nucleotide-position'])
    positions = np.arange(n)
    width = 0.4

    if total_hits > 0:
        ax.bar(positions - width/2, binned_hits['normalized_hits'], width=width, color='firebrick', edgecolor='black', label='observed')

    ax.bar(positions + width/2, p_binned_hits['normalized_hits'], width=width, color='royalblue', edgecolor='black', label='possible')

    ax.set_xticks(positions)
    ax.set_xticklabels(p_binned_hits['nucleotide-position'].astype(str))

    ax.legend()

    ax.set_title(f"dataset: {dfname}  -  strain: {strain}  -  segment: {segment}")
    ax.set_xlabel("nucleotide positions (binned)")
    ax.set_ylabel("relative occurence (%)")

    upper_bounds = [int(bin.right) for bin in p_binned_hits['nucleotide-position']]
    selected_ticks = upper_bounds[9:len(upper_bounds):10]
    selected_positions = list(range(9, len(upper_bounds), 10))

    ax.set_xticks(selected_positions)
    ax.set_xticklabels(selected_ticks)

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}/histo")
    os.makedirs(save_path, exist_ok=True)
    fname = f"{segment}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_enrichment_positions(strain: str):
    positions_dfs = []
    segments = []
    for segment in SEGMENTS:
        positions_df = find_enrichment_positions_segment(strain, segment)
        positions_dfs.append(positions_df)
        segments.append(segment)
    return positions_dfs, segments

def find_enrichment_positions_segment(strain: str, segment: str):
    seq = get_sequence(strain, segment)
    seq_len = len(seq)
    positions_df = pd.DataFrame({
        'nucleotide-position': range(1, seq_len + 1),
        'hits': [0] * seq_len
    })
    for i in range(seq_len):
        if seq[i] == 'A':
            positions_df.at[i + 1, 'hits'] = 1
    return positions_df

def find_breakpoint_positions(strain: str, df: pd.DataFrame):
    positions_dfs = []
    segments = []
    for segment in SEGMENTS:
        positions_df = find_breakpoint_positions_segment(strain, segment, df)
        positions_dfs.append(positions_df)
        segments.append(segment)
    return positions_dfs, segments

def find_breakpoint_positions_segment(strain: str, segment: str, df: pd.DataFrame):
    seq_len = get_seq_len(strain, segment)
    positions_df = pd.DataFrame({
        'nucleotide-position': range(1, seq_len + 1),
        'hits': [0] * seq_len
    })

    segment_df = df[df["Segment"] == segment]
    if segment_df.empty:
        return positions_df

    for _, row in segment_df.iterrows():
        start = row["Start"]
        end = row["End"] - 1
        positions_df.loc[positions_df['nucleotide-position'] == start, 'hits'] += 1
        positions_df.loc[positions_df['nucleotide-position'] == end, 'hits'] += 1
    return positions_df

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #####################################
    ####### ENRICHMENT POSITIONS ########
    ############ SINGLE USAGE ###########
    #####################################

    # dfnames = ["Berry2021_B_Yam"]
    # dfs, _ = load_all(dfnames)

    # dfname = dfnames[0]
    # df = dfs[0]

    # strain = DATASET_STRAIN_DICT[dfname]

    # breakpoint_dfs, segments = find_breakpoint_positions(strain, df)
    # possible_dfs, p_segments = find_enrichment_positions(strain)

    # create_enrichment_nucleotide_position_histogram(dfname, strain, segments, breakpoint_dfs, p_segments, possible_dfs)

    ##################################### same strain
    ####### ENRICHMENT POSITIONS ######## same strain
    ############ MULTI USAGE ############ same strain
    ##################################### same strain

    # dfnames = ["Berry2021_B_Yam"]
    # dfs, _ = load_all(dfnames)

    # dfname = dfnames[0]
    # concat_pseudo_df = pd.concat(dfs, ignore_index=True)
    # strain = DATASET_STRAIN_DICT[dfname]
    # dfname = strain

    # breakpoint_dfs, segments = find_breakpoint_positions(strain, df)
    # possible_dfs, p_segments = find_enrichment_positions(strain)

    # create_enrichment_nucleotide_position_histogram(dfname, strain, segments, breakpoint_dfs, p_segments, possible_dfs)

    ###################################
    ####### NGS ENRICHMENT RATIO ######
    ########### SINGLE USAGE ##########
    ###################################

    dfnames = ["Alnaji2021"]
    dfs, _ = load_all(dfnames)
    dfname = dfnames[0]

    df = ngs_enrichment_ratio_concat(dfs)
    create_ngs_enrichment_ratio_plot(dfname, df)

    ###################################
    ####### NGS ENRICHMENT RATIO ######
    ########### MULTI USAGE ###########
    ###################################

    # coordinates = "IBV"
    # dfname = coordinates
    # dfnames = get_dataset_names(cutoff=40, selection=coordinates)
    # dfs, _ = load_all(dfnames, False)

    # df = ngs_enrichment_ratio_concat(dfs)
    # create_ngs_enrichment_ratio_plot(dfname, df)