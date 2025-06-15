'''

'''
import os
import sys

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
from typing import Tuple
from collections import Counter

sys.path.insert(0, "..")
from utils import get_dataset_names, load_all
from utils import add_dvg_sequence, add_ngs_percentile_rank
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import CUSTOM_COLORS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


#################################
### motif enrichment heatmap ####
#################################

def add_motifs(df: pd.DataFrame, motif_length: int):
    """

    """
    motif0_list = []
    motif1_list = []
    motif2_list = []
    motif3_list = []

    for seq in df['seq_around_deletion_junction']:
        if len(seq) < 20:
            motif0_list.append(None)
            motif1_list.append(None)
            motif2_list.append(None)
            motif3_list.append(None)
            continue

        motif0 = seq[0:5][-motif_length:]
        motif1 = seq[5:10][:motif_length]
        motif2 = seq[10:15][-motif_length:]
        motif3 = seq[15:20][:motif_length]

        motif0_list.append(motif0)
        motif1_list.append(motif1)
        motif2_list.append(motif2)
        motif3_list.append(motif3)

    df['motif0'] = motif0_list
    df['motif1'] = motif1_list
    df['motif2'] = motif2_list
    df['motif3'] = motif3_list

    return df

def create_top_motif_count_df(df: pd.DataFrame, top_n: int = None):
    """

    """
    motif_cols = ['motif0', 'motif1', 'motif2', 'motif3']
    all_motifs = pd.concat([df[col] for col in motif_cols], ignore_index=True)

    top_motif_count_df = (
        all_motifs.value_counts(dropna=True)
        .reset_index()
        .rename(columns={'index': 'motif', 0: 'count'})
        .sort_values(by='count', ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        top_motif_count_df = top_motif_count_df.head(top_n)

    return top_motif_count_df

def create_motif_heatmap_df(df: pd.DataFrame, top_motifs: list):
    """

    """

    def count_motifs_by_column(df: pd.DataFrame, column: str, top_motifs: list):
        relevant_df = df[['NGS_percentile_rank', column]].copy()
        relevant_df = relevant_df[relevant_df[column].isin(top_motifs)]

        one_hot = pd.get_dummies(relevant_df[column])
        one_hot = one_hot.reindex(columns=top_motifs, fill_value=0)

        combined = pd.concat([relevant_df[['NGS_percentile_rank']], one_hot], axis=1)
        return combined.groupby('NGS_percentile_rank', as_index=False).sum()

    motif0_heatmap_df = count_motifs_by_column(df, 'motif0', top_motifs)
    motif1_heatmap_df = count_motifs_by_column(df, 'motif1', top_motifs)
    motif2_heatmap_df = count_motifs_by_column(df, 'motif2', top_motifs)
    motif3_heatmap_df = count_motifs_by_column(df, 'motif3', top_motifs)

    percentile_rank_count_df = (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

    return motif0_heatmap_df, motif1_heatmap_df, motif2_heatmap_df, motif3_heatmap_df, percentile_rank_count_df

def create_motif_heatmap_plot(dfname: str, dvg_count: int, motif_name: str, heatmap_df: pd.DataFrame, percentile_rank_count_df: pd.DataFrame):
    pivot_df = heatmap_df.set_index('NGS_percentile_rank')
    motif_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    rank_counts = percentile_rank_count_df.set_index('NGS_percentile_rank')['count']
    pivot_df = pivot_df.loc[pivot_df.index.intersection(rank_counts.index)]
    norm_df = pivot_df.div(rank_counts, axis=0) * 100

    motif_labels = [
        f"{motif}\n(n={pivot_df[motif].sum()})"
        for motif in motif_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    plt.figure(figsize=(1 + len(motif_labels) * 0.6, 1 + len(rank_labels) * 0.4))

    ax = sns.heatmap(
        norm_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=motif_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt=".1f"
    )

    ax.set_xlabel("top motifs")
    ax.set_ylabel("percentile rank")
    ax.set_title(f"{motif_name} enrichment heatmap – {dfname} (n={dvg_count})")

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"motif_{motif_name}_heat_{dfname}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def motif_heatmap_analysis(dfname: str, dfs: list, motif_length: int, top_n: int = 20):
    df = pd.concat(dfs, ignore_index=True)
    dvg_count = df.shape[0]
    df = add_ngs_percentile_rank(df)
    df = add_motifs(df, motif_length)
    top_motif_count_df = create_top_motif_count_df(df, top_n)
    top_motifs = top_motif_count_df['motif'].tolist()
    motif0_heatmap_df, motif1_heatmap_df, motif2_heatmap_df, motif3_heatmap_df, percentile_rank_count_df = create_motif_heatmap_df(df, top_motifs)
    create_motif_heatmap_plot(dfname, dvg_count, "before_del", motif0_heatmap_df, percentile_rank_count_df)
    create_motif_heatmap_plot(dfname, dvg_count, "del_start", motif1_heatmap_df, percentile_rank_count_df)
    create_motif_heatmap_plot(dfname, dvg_count, "del_end", motif2_heatmap_df, percentile_rank_count_df)
    create_motif_heatmap_plot(dfname, dvg_count, "after_del", motif3_heatmap_df, percentile_rank_count_df)

#################################
### ratif enrichment heatmap ####
#################################

def add_ratif(df: pd.DataFrame, motif_length: int) -> pd.DataFrame:
    def get_random_subseq(seq: str) -> str:
        if not isinstance(seq, str) or len(seq) < motif_length:
            return ""
        start = random.randint(0, len(seq) - motif_length)
        return seq[start:start + motif_length]

    df['ratif'] = df['dvg_sequence'].apply(get_random_subseq)
    return df

def create_ratif_heatmap_df(df: pd.DataFrame, top_motifs: list):
    """

    """

    def count_motifs_by_column(df: pd.DataFrame, column: str, top_motifs: list):
        relevant_df = df[['NGS_percentile_rank', column]].copy()
        relevant_df = relevant_df[relevant_df[column].isin(top_motifs)]

        one_hot = pd.get_dummies(relevant_df[column])
        one_hot = one_hot.reindex(columns=top_motifs, fill_value=0)

        combined = pd.concat([relevant_df[['NGS_percentile_rank']], one_hot], axis=1)
        return combined.groupby('NGS_percentile_rank', as_index=False).sum()

    ratif_heatmap_df = count_motifs_by_column(df, 'ratif', top_motifs)

    percentile_rank_count_df = (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

    return ratif_heatmap_df, percentile_rank_count_df

def ratif_heatmap_analysis(dfname: str, dfs: list, motif_length: int, top_n: int = 20):
    df = pd.concat(dfs, ignore_index=True)
    dvg_count = df.shape[0]
    df = add_ngs_percentile_rank(df)
    df = add_motifs(df, motif_length)
    top_motif_count_df = create_top_motif_count_df(df, top_n)
    top_motifs = top_motif_count_df['motif'].tolist()

    df = add_dvg_sequence(df)
    df = add_ratif(df, motif_length)
    ratif_heatmap_df, percentile_rank_count_df = create_ratif_heatmap_df(df, top_motifs)
    create_motif_heatmap_plot(dfname, dvg_count, "random", ratif_heatmap_df, percentile_rank_count_df)

######################################
### nucleotide enrichment heatmap ####
######################################

def create_nucleotide_heatmap_df(df: pd.DataFrame):
    nucleotides = ['A', 'C', 'G', 'U']
    results = {nt: [] for nt in nucleotides}

    df = df[df['seq_around_deletion_junction'].str.len() == 20]
    percentile_rank_count_df = df.groupby('NGS_percentile_rank').size().reset_index(name='count')

    for rank, group in df.groupby('NGS_percentile_rank'):
        counts = {nt: [0] * 20 for nt in nucleotides}

        for seq in group['seq_around_deletion_junction']:
            for pos, char in enumerate(seq):
                if char in nucleotides:
                    counts[char][pos] += 1

        for nt in nucleotides:
            results[nt].append([rank] + counts[nt])

    nucleotide_heatmap_dfs = {}
    for nt in nucleotides:
        df_nt = pd.DataFrame(results[nt], columns=['NGS_percentile_rank'] + list(range(20)))
        nucleotide_heatmap_dfs[nt] = df_nt.sort_values('NGS_percentile_rank').reset_index(drop=True)

    return nucleotide_heatmap_dfs['A'], nucleotide_heatmap_dfs['C'], nucleotide_heatmap_dfs['G'], nucleotide_heatmap_dfs['U'], percentile_rank_count_df

def create_nucleotide_heatmap_plot(dfname: str, dvg_count: int, nucleotide_name: str, heatmap_df: pd.DataFrame, percentile_rank_count_df: pd.DataFrame):

    pivot_df = heatmap_df.set_index('NGS_percentile_rank')
    motif_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    rank_counts = percentile_rank_count_df.set_index('NGS_percentile_rank')['count']
    pivot_df = pivot_df.loc[pivot_df.index.intersection(rank_counts.index)]
    norm_df = pivot_df.div(rank_counts, axis=0) * 100

    motif_labels = [
        f"{motif}\n(n={pivot_df[motif].sum()})"
        for motif in motif_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    plt.figure(figsize=(1 + len(motif_labels) * 0.6, 1 + len(rank_labels) * 0.4))

    ax = sns.heatmap(
        norm_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=motif_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt=".1f"
    )

    ax.set_xlabel("nucleotide positions around deletion site")
    ax.set_ylabel("percentile rank")
    ax.set_title(f"{nucleotide_name} enrichment heatmap – {dfname} (n={dvg_count})")

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"nucleotide_{nucleotide_name}_heat_{dfname}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def nucleotide_heatmap_analysis(dfname: str, dfs: list):
    df = pd.concat(dfs, ignore_index=True)
    dvg_count = df.shape[0]
    df = add_ngs_percentile_rank(df)
    nucleotide0_heatmap_df, nucleotide1_heatmap_df, nucleotide2_heatmap_df, nucleotide3_heatmap_df, percentile_rank_count_df = create_nucleotide_heatmap_df(df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "A", nucleotide0_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "C", nucleotide1_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "G", nucleotide2_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "U", nucleotide3_heatmap_df, percentile_rank_count_df)

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    ### SINGLE ###

    # dfnames = ["Berry2021_B_Yam"]
    # dfs, _ = load_all(dfnames)
    # dfname = dfnames[0]

    # df = dfs[0]
    # strain = DATASET_STRAIN_DICT[dfname]

    ### MULTI ###

    selector = "IAV"
    dfname = selector
    dfnames = get_dataset_names(cutoff=40, selection=selector)
    dfs, _ = load_all(dfnames, False)

    #################################
    ### motif enrichment heatmap ####
    #################################

    ### multi ###
    motif_heatmap_analysis(dfname, dfs, 3)

    #################################
    ### ratif enrichment heatmap ####
    #################################

    ### multi ###
    ratif_heatmap_analysis(dfname, dfs, 3)

    ######################################
    ### nucleotide enrichment heatmap ####
    ######################################

    ### multi ###
    nucleotide_heatmap_analysis(dfname, dfs)


    