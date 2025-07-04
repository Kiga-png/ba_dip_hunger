'''

'''
import os
import sys

import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
from itertools import product
from sklearn.linear_model import LinearRegression
from typing import Tuple
from collections import Counter

import RNA

sys.path.insert(0, "..")
from utils import get_dataset_names, load_all, get_sequence, get_seq_len
from utils import add_remaining_sequence, add_dvg_sequence, add_direct_repeat_len, cap_direct_repeat_len, add_mfe_percentile_rank, add_sec_features_parallel, save_df, add_ngs_percentile_rank
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import COLORS, STRAINS, A_STRAINS, B_STRAINS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, "heatmap")


######################
### motif heatmap ####
######################

def make_full_motif_heatmap_analysis(dfname: str, dfs: list, top_n: int = 20, strain: str = "", segment: str = ""):
    """

    """
    for motif_length in range(2, 6):
        make_motif_heatmap_analysis(dfname, dfs, motif_length, top_n, strain, segment)
        make_reg_motif_heatmap_analysis(dfname, dfs, motif_length, top_n, strain, segment)

### true motif heatmap ####

# optional #
def opt_add_motifs(df: pd.DataFrame, motif_length: int = 5):
    """

    """
    motifs0 = []
    motifs1 = []
    motifs2 = []
    motifs3 = []

    for idx, row in df.iterrows():
        seq = row['full_seq']
        start = row['Start']
        end = row['End']

        start0 = start
        end0 = end - 1

        if start0 - motif_length < 0 or start0 + motif_length > len(seq):
            raise ValueError(f"motif extraction at 'Start' for row {idx} exceeds sequence bounds.")
        if end0 - motif_length < 0 or end0 + motif_length > len(seq):
            raise ValueError(f"motif extraction at 'End' for row {idx} exceeds sequence bounds.")

        motif0 = seq[start0 - motif_length:start0]
        motif1 = seq[start0:start0 + motif_length]
        motif2 = seq[end0 - motif_length:end0]
        motif3 = seq[end0:end0 + motif_length]

        motifs0.append(motif0)
        motifs1.append(motif1)
        motifs2.append(motif2)
        motifs3.append(motif3)

    df['motif0'] = motifs0
    df['motif1'] = motifs1
    df['motif2'] = motifs2
    df['motif3'] = motifs3

    return df

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

def compute_comb_motif_count_df(df: pd.DataFrame, motif_length: int):
    """

    """
    bases = ['A', 'C', 'G', 'U']
    all_possible_motifs = [''.join(p) for p in itertools.product(bases, repeat=motif_length)]
    all_possible_motifs.sort()

    motif_cols = ['motif0', 'motif1', 'motif2', 'motif3']
    all_motifs = pd.concat([df[col] for col in motif_cols], ignore_index=True)

    motif_counts = all_motifs.value_counts(dropna=True).to_dict()

    raw_motif_df = pd.DataFrame({
        'motif': all_possible_motifs,
        'count': [motif_counts.get(motif, 0) for motif in all_possible_motifs]
    })

    return raw_motif_df

def compute_comb_motif_freq_df(df: pd.DataFrame, motif_length: int):
    """
    
    """
    bases = ['A', 'C', 'G', 'U']
    all_possible_motifs = [''.join(p) for p in itertools.product(bases, repeat=motif_length)]
    all_possible_motifs.sort()

    motif_cols = ['motif0', 'motif1', 'motif2', 'motif3']
    all_motifs = pd.concat([df[col] for col in motif_cols], ignore_index=True)

    motif_counts = all_motifs.value_counts(dropna=True).to_dict()
    total_motifs = len(all_motifs)

    raw_motif_df = pd.DataFrame({
        'motif': all_possible_motifs,
        'freq': [(motif_counts.get(motif, 0) / total_motifs) * 100 for motif in all_possible_motifs]
    })

    return raw_motif_df

def compute_motif_count_heatmap_dfs(df: pd.DataFrame, top_motifs: list):
    """

    """
    def count_motifs_by_column(df: pd.DataFrame, column: str, top_motifs: list):
        relevant_df = df[['NGS_percentile_rank', column]].copy()
        relevant_df = relevant_df[relevant_df[column].isin(top_motifs)]

        one_hot = pd.get_dummies(relevant_df[column])
        one_hot = one_hot.reindex(columns=top_motifs, fill_value=0)

        combined = pd.concat([relevant_df[['NGS_percentile_rank']], one_hot], axis=1)
        return combined.groupby('NGS_percentile_rank', as_index=False).sum()

    motif_count_dfs = [
        count_motifs_by_column(df, 'motif0', top_motifs),
        count_motifs_by_column(df, 'motif1', top_motifs),
        count_motifs_by_column(df, 'motif2', top_motifs),
        count_motifs_by_column(df, 'motif3', top_motifs),
    ]

    percentile_rank_count_df = (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

    return motif_count_dfs, percentile_rank_count_df

def compute_motif_count_dfs(dfs: list):
    """

    """
    motif_count_dfs = []

    for df in dfs:
        motif_columns = df.columns[1:]
        motif_sums = df[motif_columns].sum()

        summary_df = pd.DataFrame({
            'motif': motif_columns,
            'count': motif_sums.values
        })

        motif_count_dfs.append(summary_df)

    return motif_count_dfs

def compute_motif_freq_heatmap_dfs(
    motif_count_heatmap_dfs: list,
    percentile_rank_count_df: pd.DataFrame
    ) -> list:
    """

    """
    rank_counts = percentile_rank_count_df.set_index('NGS_percentile_rank')['count']
    
    motif_freq_heatmap_dfs = []
    for motif_heatmap_df in motif_count_heatmap_dfs:
        pivot_df = motif_heatmap_df.set_index('NGS_percentile_rank')
        pivot_df = pivot_df.loc[pivot_df.index.intersection(rank_counts.index)]
        norm_motif_heatmap_df = pivot_df.div(rank_counts, axis=0) * 100
        motif_freq_heatmap_dfs.append(norm_motif_heatmap_df)
    
    return motif_freq_heatmap_dfs

def create_freq_heatmap_plot(
    heatmap_name: str,
    field: str,
    field_specifier: str,
    freq_heatmap_df: pd.DataFrame,
    count_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame,
    dvg_count: int,
    data: str,
    strain: str = "",
    segment: str = "",
    feature_name = "",
    feature = 0,
    metric_name = "",
    metric: float = 0,
    ):  
    """

    """
    pivot_df = freq_heatmap_df
    field_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    field_count_series = count_df.set_index(field)['count']

    field_labels = [
        f"{field}\n(n={field_count_series.get(field, 0)})"
        for field in field_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    plt.figure(figsize=(1 + len(field_labels) * 0.6, 1 + len(rank_labels) * 0.4))

    ax = sns.heatmap(
        pivot_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=field_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt=".1f"
    )

    ax.set_xlabel(f"{field} ({field_specifier})")
    ax.set_ylabel("NGS_rank (percentile)")

    title_name = f"{field} heatmap ({heatmap_name})"
    if feature_name:
        title_name += f" - {feature_name}: {feature}"
    if metric_name:
        title_name += f" - {metric_name}: {metric}"
    title_name += f"\ndata: {data}"
    if strain:
        title_name += f", strain: {strain}"
    if segment:
        title_name += f", segment: {segment}"
    title_name += f", (n={dvg_count})"
    ax.set_title(title_name)

    save_path = os.path.join(RESULTSPATH, field, dfname)

    if strain:
        save_path = os.path.join(save_path, strain)
    else:
        save_path = os.path.join(save_path, "all")
    if segment:
        save_path = os.path.join(save_path, segment)
    else:
        save_path = os.path.join(save_path, "all")

    save_path = os.path.join(save_path, f"{feature_name}_{feature}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"{field}_{heatmap_name}_heatmap_{data}"

    if strain:
        fname += f"_{strain}"
    if segment:
        fname += f"_{segment}"
    if feature_name:
        fname += f"_{feature_name}_{feature}"
    fname += ".png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def make_motif_heatmap_analysis(dfname: str, dfs: list, motif_length: int, top_n: int=20, strain: str="", segment: str=""):
    df = pd.concat(dfs, ignore_index=True)

    if strain:
        df = df[df["Strain"] == strain]
    if segment:
        df = df[df["Segment"] == segment]

    dvg_count = df.shape[0]

    df = add_ngs_percentile_rank(df)
    df = add_motifs(df, motif_length)

    comb_motif_freq_df = compute_comb_motif_freq_df(df, motif_length)
    comb_batif_freq_df = compute_comb_batif_freq_df(dfname, motif_length, strain, segment)
    comb_freq_df = subtract_freq_dfs(comb_motif_freq_df, comb_batif_freq_df)
    top_comb_freq_df = comb_freq_df.nlargest(top_n, 'top_freq')

    top_comb_freq_df = comb_freq_df.sort_values(by='top_freq', ascending=False).head(top_n)
    mean_feq_dif = round(top_comb_freq_df['top_freq'].mean(), 5)
    top_motifs = top_comb_freq_df['motif'].tolist()

    create_freq_bar_plot("sites_vs_background", "motif", f"top {top_n}", top_comb_freq_df, dvg_count, dfname, strain, segment, "motif_length", motif_length, 'mean frequency difference', mean_feq_dif)

    motif_count_heatmap_dfs, percentile_rank_count_df = compute_motif_count_heatmap_dfs(df, top_motifs)
    motif_count_dfs = compute_motif_count_dfs(motif_count_heatmap_dfs)
    motif_freq_heatmap_dfs = compute_motif_freq_heatmap_dfs(motif_count_heatmap_dfs, percentile_rank_count_df)

    heatmap_names = ["before_del", "del_start", "del_end", "after_del"]
    for i in range(0, 4):
        create_freq_heatmap_plot(heatmap_names[i], "motif", f"top {top_n}", motif_freq_heatmap_dfs[i], motif_count_dfs[i], percentile_rank_count_df, dvg_count, dfname, strain, segment, "motif_length", motif_length, 'mean frequency difference', mean_feq_dif)

### background motif distribution ####

def compute_batif_count_df(motif_length: int, strain: str, segment: str):
    """

    """

    full_length_seq = get_sequence(strain, segment)

    bases = ['A', 'C', 'G', 'U']
    all_possible_motifs = [''.join(p) for p in itertools.product(bases, repeat=motif_length)]
    all_possible_motifs.sort()

    data = {
        'motif': [],
        'count': []
    }

    for motif in all_possible_motifs:
        count = full_length_seq.count(motif)
        data['motif'].append(motif)
        data['count'].append(count)

    batif_count_df = pd.DataFrame(data)

    return batif_count_df

def compute_batif_freq_df(motif_length: int, strain: str, segment: str):
    """

    """
    full_length_seq = get_sequence(strain, segment)

    bases = ['A', 'C', 'G', 'U']
    all_possible_motifs = [''.join(p) for p in itertools.product(bases, repeat=motif_length)]
    all_possible_motifs.sort()

    data = {
        'motif': [],
        'count': []
    }

    for motif in all_possible_motifs:
        count = full_length_seq.count(motif)
        data['motif'].append(motif)
        data['count'].append(count)

    batif_df = pd.DataFrame(data)

    total = batif_df['count'].sum()
    batif_df['freq'] = (batif_df['count'] / total) * 100

    return batif_df[['motif', 'freq']]

def compute_comb_batif_freq_df(dfname: str, motif_length: int, strain: str, segment: str):
    """

    """
    strains = STRAINS
    if dfname == "IAV":
        strains = A_STRAINS
    if dfname == "IBV":
        strains = B_STRAINS

    if segment and strain:
        batif_freq_df = compute_batif_freq_df(motif_length, strain, segment)
        return batif_freq_df
    
    comb_df = pd.DataFrame()

    if strain:
        for segment in SEGMENTS:
            batif_freq_df = compute_batif_freq_df(motif_length, strain, segment)
            if comb_df.empty:
                comb_df = batif_freq_df.copy()
            else:
                comb_df['freq'] += batif_freq_df['freq']

        comb_df['freq'] = comb_df['freq'] / len(SEGMENTS) 
        return comb_df
    
    if segment:
        for strain in strains:
            batif_freq_df = compute_batif_freq_df(motif_length, strain, segment)
            if comb_df.empty:
                comb_df = batif_freq_df.copy()
            else:
                comb_df['freq'] += batif_freq_df['freq']

        comb_df['freq'] = comb_df['freq'] / len(strains)
        return comb_df
    
    for strain in strains:
        for segment in SEGMENTS:
            batif_freq_df = compute_batif_freq_df(motif_length, strain, segment)
            if comb_df.empty:
                comb_df = batif_freq_df.copy()
            else:
                comb_df['freq'] += batif_freq_df['freq']

    comb_df['freq'] = comb_df['freq'] / (len(strains) * len(SEGMENTS))
    return comb_df

def subtract_freq_dfs(comb_motif_freq_df: pd.DataFrame, comb_batif_freq_df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    comb_freq_df = pd.DataFrame({
        'motif': comb_motif_freq_df['motif'],
        'motif_freq': comb_motif_freq_df['freq'],
        'batif_freq': comb_batif_freq_df['freq'],
        'top_freq': comb_motif_freq_df['freq'] - comb_batif_freq_df['freq']
    })
    return comb_freq_df

def create_freq_bar_plot(
    bar_name: str,
    field: str,
    field_specifier: str,
    top_freq_df: pd.DataFrame,
    dvg_count: int,
    data: str,
    strain: str = "",
    segment: str = "",
    feature_name = "",
    feature = 0,
    metric_name = "",
    metric: float = 0,
    ):
    """

    """

    df_plot = pd.melt(
        top_freq_df,
        id_vars='motif',
        value_vars=['motif_freq', 'batif_freq'],
        var_name='type',
        value_name='frequency'
    )

    custom_colors = {
        'motif_freq': COLORS[4],
        'batif_freq': COLORS[0]
    }

    plt.figure(figsize=(18, 6))
    sns.barplot(data=df_plot, x='motif', y='frequency', hue='type', palette=custom_colors)

    plt.xticks(rotation=90)
    plt.xlabel(f"{field} ({field_specifier})")
    plt.ylabel('relative frequency (%)')

    title_name = f"{field} barplot ({bar_name})"
    if feature_name:
        title_name += f" - {feature_name}: {feature}"
    if metric_name:
        title_name += f" - {metric_name}: {metric}"
    title_name += f"\ndata: {data}"
    if strain:
        title_name += f", strain: {strain}"
    if segment:
        title_name += f", segment: {segment}"
    title_name += f" (n={dvg_count})"
    plt.title(title_name)

    save_path = os.path.join(RESULTSPATH, field, dfname)

    if strain:
        save_path = os.path.join(save_path, strain)
    else:
        save_path = os.path.join(save_path, "all")
    if segment:
        save_path = os.path.join(save_path, segment)
    else:
        save_path = os.path.join(save_path, "all")

    save_path = os.path.join(save_path, f"{feature_name}_{feature}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"{field}_{bar_name}_bar_{data}"

    if strain:
        fname += f"_{strain}"
    if segment:
        fname += f"_{segment}"
    if feature_name:
        fname += f"_{feature_name}_{feature}"
    fname += ".png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

### regression motif heatmap ####

def generate_motif_list(motif_length: int):
    """

    """
    nucleotides = ['A', 'C', 'G', 'U']
    motifs = [''.join(p) for p in product(nucleotides, repeat=motif_length)]
    return sorted(motifs)

def add_linear_regression_rows(motif_freq_heatmap_dfs: list):
    """

    """
    updated_dfs = []

    for df in motif_freq_heatmap_dfs:
        X = np.arange(1, len(df) + 1).reshape(-1, 1)

        intercepts = {}
        coefficients = {}

        for column in df.columns:
            y = df[column].values
            model = LinearRegression().fit(X, y)
            intercepts[column] = model.intercept_
            coefficients[column] = model.coef_[0]

        df_extended = df.copy()
        df_extended.loc['intercept'] = intercepts
        df_extended.loc['coefficient'] = coefficients

        updated_dfs.append(df_extended)

    return updated_dfs

def compute_top_motif_count_df(motif_count_df: pd.DataFrame, top_motifs: list) -> pd.DataFrame:
    """

    """
    top_motif_count_df  = motif_count_df[motif_count_df['motif'].isin(top_motifs)].copy()
    top_motif_count_df['sort_order'] = top_motif_count_df['motif'].apply(lambda x: top_motifs.index(x))
    top_motif_count_df = top_motif_count_df.sort_values('sort_order').drop(columns='sort_order')

    return top_motif_count_df 

def remove_linear_regression_rows(top_motif_count_df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    return top_motif_count_df.drop(columns=['coefficient', 'intercept'], errors='ignore')

def make_reg_motif_heatmap_analysis(dfname: str, dfs: list, motif_length: int, top_n: int=20, strain: str="", segment: str=""):
    df = pd.concat(dfs, ignore_index=True)

    if strain:
        df = df[df["Strain"] == strain]
    if segment:
        df = df[df["Segment"] == segment]

    dvg_count = df.shape[0]

    df = add_ngs_percentile_rank(df)
    df = add_motifs(df, motif_length)

    motifs = generate_motif_list(motif_length)
    motif_count_heatmap_dfs, percentile_rank_count_df = compute_motif_count_heatmap_dfs(df, motifs)
    motif_count_dfs = compute_motif_count_dfs(motif_count_heatmap_dfs)
    motif_freq_heatmap_dfs = compute_motif_freq_heatmap_dfs(motif_count_heatmap_dfs, percentile_rank_count_df)
    reg_motif_freq_heatmap_dfs = add_linear_regression_rows(motif_freq_heatmap_dfs)

    heatmap_names = ["reg_before_del", "reg_del_start", "reg_del_end", "reg_after_del"]
    for i in range(0, 4):
        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_dfs[i].T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        top_trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(top_n)
        mean_coefficient = round(top_trans_reg_motif_freq_heatmap_df["coefficient"].mean(), 5)
        top_motifs = top_trans_reg_motif_freq_heatmap_df.index.tolist()
        top_motif_count_df = compute_top_motif_count_df(motif_count_dfs[i], top_motifs)
        top_trans_motif_freq_heatmap_df = remove_linear_regression_rows(top_trans_reg_motif_freq_heatmap_df)
        top_motif_freq_heatmap_df = top_trans_motif_freq_heatmap_df.T
        create_freq_heatmap_plot(heatmap_names[i], "motif", f"top {top_n}", top_motif_freq_heatmap_df, top_motif_count_df, percentile_rank_count_df, dvg_count, dfname, strain, segment, "motif_length", motif_length, 'mean coefficient', mean_coefficient)

### random motif heatmap ####

# TODO #
def add_ratifs(df: pd.DataFrame, motif_length: int) -> pd.DataFrame:
    random.seed(42)

    def get_random_subseq(seq: str) -> str:
        if not isinstance(seq, str) or len(seq) < motif_length:
            return ""
        start = random.randint(0, len(seq) - motif_length)
        return seq[start:start + motif_length]

    df['dvg_ratif'] = df['remaining_sequence'].apply(get_random_subseq)
    df['del_ratif'] = df['deleted_sequence'].apply(get_random_subseq)
    return df

# TODO #
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

    dvg_ratif_heatmap_df = count_motifs_by_column(df, 'dvg_ratif', top_motifs)
    del_ratif_heatmap_df = count_motifs_by_column(df, 'del_ratif', top_motifs)

    percentile_rank_count_df = (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

    return [dvg_ratif_heatmap_df, del_ratif_heatmap_df], percentile_rank_count_df

# TODO #
def ratif_heatmap_analysis(dfname: str, df: pd.DataFrame, motif_length: int, segment: str, top_n: int):
    dvg_count = df.shape[0]
    df = add_ngs_percentile_rank(df)
    df = add_motifs(df, motif_length)
    raw_motif_count_df = create_raw_motif_count_df(df, top_n)
    raw_motif_count_df.sort_values(by='count', ascending=False).head(top_n)
    top_motifs = raw_motif_count_df['motif'].tolist()
    df = add_remaining_sequence(df)
    df = add_ratifs(df, motif_length)
    ratif_heatmap_dfs, percentile_rank_count_df = create_ratif_heatmap_df(df, top_motifs)
    ratif_count_dfs = create_motif_count_dfs(ratif_heatmap_dfs)
    create_motif_heatmap_plot(dfname, dvg_count, motif_length, segment, "random_dvg", ratif_heatmap_dfs[0], ratif_count_dfs[0], percentile_rank_count_df)
    create_motif_heatmap_plot(dfname, dvg_count, motif_length, segment, "random_del", ratif_heatmap_dfs[1], ratif_count_dfs[1], percentile_rank_count_df)
    
######################
### repeat heatmap ###
######################

### true repeat heatmap ####

def compute_repeat_count_heatmap_df(df: pd.DataFrame):
    """

    """
    relevant_df = df[['NGS_percentile_rank', 'direct_repeat_len']].copy()
    one_hot = pd.get_dummies(relevant_df['direct_repeat_len'])
    combined = pd.concat([relevant_df[['NGS_percentile_rank']], one_hot], axis=1)
    repeat_count_heatmap_df = combined.groupby('NGS_percentile_rank', as_index=False).sum()

    percentile_rank_count_df = (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

    return repeat_count_heatmap_df, percentile_rank_count_df

def compute_repeat_count_df(df: pd.DataFrame):
    """

    """
    repeat_columns = df.columns[1:]
    count_sums = df[repeat_columns].sum()

    repeat_count_df = pd.DataFrame({
        'repeat_length': repeat_columns.astype(int),
        'count': count_sums.values.astype(int)
    })

    return repeat_count_df

def compute_repeat_freq_heatmap_df(
    repeat_count_heatmap_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame
) -> pd.DataFrame:
    """

    """
    rank_counts = percentile_rank_count_df.set_index('NGS_percentile_rank')['count']
    pivot_df = repeat_count_heatmap_df.set_index('NGS_percentile_rank')
    pivot_df = pivot_df.loc[pivot_df.index.intersection(rank_counts.index)]
    repeat_freq_heatmap_df = pivot_df.div(rank_counts, axis=0) * 100

    return repeat_freq_heatmap_df

def create_repeat_freq_heatmap_plot(
    dfname: str,
    dvg_count: int,
    cap_length: int,
    strain: str,
    segment: str,
    repeat_freq_heatmap_df: pd.DataFrame,
    repeat_count_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame
    ):  
    """

    """
    pivot_df = repeat_freq_heatmap_df
    repeat_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    repeat_count_series = repeat_count_df.set_index('repeat_length')['count']

    repeat_labels = [
        f"{repeat}\n(n={repeat_count_series.get(repeat, 0)})"
        for repeat in repeat_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    plt.figure(figsize=(1 + len(repeat_labels) * 0.6, 1 + len(rank_labels) * 0.4))

    ax = sns.heatmap(
        pivot_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=repeat_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt=".1f"
    )

    ax.set_xlabel("repeat lengths")
    ax.set_ylabel("NGS percentile rank")
    ax.set_title(f"direct repeat - heatmap - length cap: {cap_length}\ndataset: {dfname} {strain} {segment} (n={dvg_count})")

    save_path = os.path.join(RESULTSPATH, "repeat", dfname)

    if strain:
        save_path = os.path.join(save_path, strain)
    else:
        save_path = os.path.join(save_path, "all")

    if segment:
        save_path = os.path.join(save_path, segment)
    else:
        save_path = os.path.join(save_path, "all")

    os.makedirs(save_path, exist_ok=True)
    fname = f"repeat_heat_{dfname}"

    if strain:
        fname = fname + f"_{strain}"
    if segment:
        fname = fname + f"_{segment}"
    fname = fname + f"length_{cap_length}" + ".png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def make_repeat_heatmap_analysis(dfname: str, dfs: list, cap_length: int=5, strain: str="", segment: str=""):
    df = pd.concat(dfs, ignore_index=True)

    if strain:
        df = df[df["Strain"] == strain]
    if segment:
        df = df[df["Segment"] == segment]

    dvg_count = df.shape[0]

    df = add_ngs_percentile_rank(df)
    df = add_direct_repeat_len(df)
    df = cap_direct_repeat_len(df, cap_length)

    repeat_count_heatmap_df, percentile_rank_count_df = compute_repeat_count_heatmap_df(df)
    repeat_count_df = compute_repeat_count_df(repeat_count_heatmap_df)
    repeat_freq_heatmap_df = compute_repeat_freq_heatmap_df(repeat_count_heatmap_df, percentile_rank_count_df)

    create_freq_heatmap_plot("direct_repeat", "repeat_length", "-", repeat_freq_heatmap_df, repeat_count_df, percentile_rank_count_df, dvg_count, dfname, strain, segment)

#############################
### sec features heatmap ###
#############################

### preprocess sec ####

def preprocess_sec_features(dfname: str, dfs: list, fname: str="df", strain: str="", segment: str=""):
    """

    """
    df = pd.concat(dfs, ignore_index=True)
    df = add_remaining_sequence(df)
    df = add_sec_features_parallel(df)
    save_df(dfname, df, fname, strain, segment)

### MFE heatmap ####

def compute_mfe_count_heatmap_df(df: pd.DataFrame):
    """

    """
    relevant_df = df[['NGS_percentile_rank', 'MFE_percentile_rank']].copy()
    one_hot = pd.get_dummies(relevant_df['MFE_percentile_rank'])
    combined = pd.concat([relevant_df[['NGS_percentile_rank']], one_hot], axis=1)
    mfe_count_heatmap_df = combined.groupby('NGS_percentile_rank', as_index=False).sum()

    percentile_rank_count_df = (
        df.groupby('NGS_percentile_rank')
        .size()
        .reset_index(name='count')
    )

    return mfe_count_heatmap_df, percentile_rank_count_df

def compute_mfe_count_df(df: pd.DataFrame):
    """

    """
    mfe_columns = df.columns[1:]
    count_sums = df[mfe_columns].sum()

    mfe_count_df = pd.DataFrame({
        'mfe_rank': mfe_columns.astype(int),
        'count': count_sums.values.astype(int)
    })

    return mfe_count_df

def compute_mfe_freq_heatmap_df(
    mfe_count_heatmap_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame
) -> pd.DataFrame:
    """

    """
    rank_counts = percentile_rank_count_df.set_index('NGS_percentile_rank')['count']
    pivot_df = mfe_count_heatmap_df.set_index('NGS_percentile_rank')
    pivot_df = pivot_df.loc[pivot_df.index.intersection(rank_counts.index)]
    mfe_freq_heatmap_df = pivot_df.div(rank_counts, axis=0) * 100

    return mfe_freq_heatmap_df

def create_mfe_freq_heatmap_plot(
    dfname: str,
    dvg_count: int,
    strain: str,
    segment: str,
    mfe_freq_heatmap_df: pd.DataFrame,
    mfe_count_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame
    ):  
    """

    """
    pivot_df = mfe_freq_heatmap_df
    mfe_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    mfe_count_series = mfe_count_df.set_index('mfe_rank')['count']

    mfe_labels = [
        f"{mfe}\n(n={mfe_count_series.get(mfe, 0)})"
        for mfe in mfe_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    plt.figure(figsize=(1 + len(mfe_labels) * 0.6, 1 + len(rank_labels) * 0.4))

    ax = sns.heatmap(
        pivot_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=mfe_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt=".1f"
    )

    ax.set_xlabel("mfe percentile rank")
    ax.set_ylabel("NGS percentile rank")
    ax.set_title(f"MFE - heatmap\ndataset: {dfname} {strain} {segment} (n={dvg_count})")

    save_path = os.path.join(RESULTSPATH, "mfe", dfname)

    if strain:
        save_path = os.path.join(save_path, strain)
    else:
        save_path = os.path.join(save_path, "all")

    if segment:
        save_path = os.path.join(save_path, segment)
    else:
        save_path = os.path.join(save_path, "all")

    os.makedirs(save_path, exist_ok=True)
    fname = f"mfe_heat_{dfname}"

    if strain:
        fname = fname + f"_{strain}"
    if segment:
        fname = fname + f"_{segment}"
    fname = fname + ".png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def make_mfe_heatmap_analysis(dfname: str, fname: "str", strain: str="", segment: str=""):
    """

    """
    read_path, _ = os.path.split(RESULTSPATH)
    read_path = os.path.join(read_path, "dfs", dfname, "all", "all")
    fname = fname + ".csv"
    df = pd.read_csv(os.path.join(read_path, fname))

    if strain:
        df = df[df["Strain"] == strain]
    if segment:
        df = df[df["Segment"] == segment]

    dvg_count = df.shape[0]

    df = add_ngs_percentile_rank(df)
    df = add_mfe_percentile_rank(df)

    mfe_count_heatmap_df, percentile_rank_count_df = compute_mfe_count_heatmap_df(df)
    mfe_count_df = compute_mfe_count_df(mfe_count_heatmap_df)
    mfe_freq_heatmap_df = compute_mfe_freq_heatmap_df(mfe_count_heatmap_df, percentile_rank_count_df)

    create_freq_heatmap_plot("min_free_energy", "mfe_rank", "percentile", mfe_freq_heatmap_df, mfe_count_df, percentile_rank_count_df, dvg_count, dfname, strain, segment)





################################
### true nucleotide heatmap ####
################################

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
    ax.set_title(f"{nucleotide_name} enrichment heatmap - {dfname} (n={dvg_count})")

    save_path = os.path.join(RESULTSPATH, f"nucleotide/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"nucleotide_{nucleotide_name}_heat_{dfname}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def nucleotide_heatmap_analysis(dfname: str, df: pd.DataFrame):
    dvg_count = df.shape[0]
    df = add_ngs_percentile_rank(df)
    nucleotide0_heatmap_df, nucleotide1_heatmap_df, nucleotide2_heatmap_df, nucleotide3_heatmap_df, percentile_rank_count_df = create_nucleotide_heatmap_df(df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "A", nucleotide0_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "C", nucleotide1_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "G", nucleotide2_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "U", nucleotide3_heatmap_df, percentile_rank_count_df)

##################################
### random nucleotide heatmap ####
##################################

def create_randotide_heatmap_df(df: pd.DataFrame):
    random.seed(42)
    nucleotides = ['A', 'C', 'G', 'U']
    num_positions = 10
    results = {nt: [] for nt in nucleotides}

    df = df[df['remaining_sequence'].str.len() > 0]
    df = df[df['deleted_sequence'].str.len() > 0]
    percentile_rank_count_df = df.groupby('NGS_percentile_rank').size().reset_index(name='count')

    for rank, group in df.groupby('NGS_percentile_rank'):
        counts = {nt: [0] * 20 for nt in nucleotides}

        for _, row in group.iterrows():
            dvg_seq = row['remaining_sequence']
            del_seq = row['deleted_sequence']

            if isinstance(dvg_seq, str) and len(dvg_seq) > 0:
                for i in range(num_positions):
                    pos = random.randint(0, len(dvg_seq) - 1)
                    nt = dvg_seq[pos]
                    if nt in nucleotides:
                        counts[nt][i] += 1

            if isinstance(del_seq, str) and len(del_seq) > 0:
                for i in range(num_positions):
                    pos = random.randint(0, len(del_seq) - 1)
                    nt = del_seq[pos]
                    if nt in nucleotides:
                        counts[nt][num_positions + i] += 1

        for nt in nucleotides:
            results[nt].append([rank] + counts[nt])

    nucleotide_heatmap_dfs = {}
    columns = ['NGS_percentile_rank'] + list(range(20))
    for nt in nucleotides:
        df_nt = pd.DataFrame(results[nt], columns=columns)
        nucleotide_heatmap_dfs[nt] = df_nt.sort_values('NGS_percentile_rank').reset_index(drop=True)

    return nucleotide_heatmap_dfs['A'], nucleotide_heatmap_dfs['C'], nucleotide_heatmap_dfs['G'], nucleotide_heatmap_dfs['U'], percentile_rank_count_df

def randotide_heatmap_analysis(dfname: str, df: pd.DataFrame):
    dvg_count = df.shape[0]
    df = add_ngs_percentile_rank(df)
    df = add_remaining_sequence(df)
    nucleotide0_heatmap_df, nucleotide1_heatmap_df, nucleotide2_heatmap_df, nucleotide3_heatmap_df, percentile_rank_count_df = create_randotide_heatmap_df(df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "random_A", nucleotide0_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "random_C", nucleotide1_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "random_G", nucleotide2_heatmap_df, percentile_rank_count_df)
    create_nucleotide_heatmap_plot(dfname, dvg_count, "random_U", nucleotide3_heatmap_df, percentile_rank_count_df)


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

    selector = "IBV"
    dfname = selector
    dfnames = get_dataset_names(cutoff=40, selection=selector)
    dfs, _ = load_all(dfnames, False)

    ################
    ### testing ####
    ################

    # make_motif_heatmap_analysis(dfname, dfs, 5, 20, "PR8", "PB1")
    # make_reg_motif_heatmap_analysis(dfname, dfs, 5, 20, "PR8", "PB1")
    # make_full_motif_heatmap_analysis(dfname, dfs, 20, "", "")
    make_repeat_heatmap_analysis(dfname, dfs, 5, "", "")

    # preprocess_sec_features(dfname, dfs, "sec", "", "")
    make_mfe_heatmap_analysis(dfname, "sec", "", "")