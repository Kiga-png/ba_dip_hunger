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
from collections import Counter

sys.path.insert(0, "..")
from utils import load_dataset, get_dataset_names, load_all, get_seq_len
from utils import add_motif_count, add_norm_log_ngs_read_count, add_dfname, add_selector, get_selctors
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


#################################
### motif enrichment dist adv ###
#################################

def create_ngs_motif_dist_adv_st_plot(dfname: str, df: pd.DataFrame, motif: str):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[(df["start_motif_counter"] != 0) | (df["end_motif_counter"] != 0)]
    n_rows = df_filtered.shape[0]

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    data_all = df_filtered['norm_log_NGS_read_count'].dropna()
    density_all = stats.gaussian_kde(data_all)
    x_vals = np.linspace(0, 1, 500)
    y_vals_all = density_all(x_vals) * 100 * bin_width

    mean = data_all.mean()
    variance = data_all.var()
    skewness = data_all.skew()
    kurtosis = data_all.kurtosis()

    stats_text = f"mean: {mean:.3f}\nvariance: {variance:.3f}\nskewness: {skewness:.3f}\nkurtosis: {kurtosis:.3f}"
    ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    all_y_vals = []
    first_curve = True

    for _, group in df_filtered.groupby('selector'):
        group_data = group['norm_log_NGS_read_count'].dropna()
        if len(group_data) < 2:
            continue
        density = stats.gaussian_kde(group_data)
        y_vals = density(x_vals) * 100 * bin_width
        all_y_vals.append(y_vals)

        label = 'datasets' if first_curve else None
        first_curve = False

        ax.plot(x_vals, y_vals, linewidth=1.5, color='lightgray', alpha=0.8, label=label, zorder=1)

    if all_y_vals:
        all_y_vals = np.array(all_y_vals)
        mean_vals = np.mean(all_y_vals, axis=0)
        std_vals = np.std(all_y_vals, axis=0)

        ax.plot(x_vals, mean_vals, color='royalblue', linewidth=1.5, label='mean', zorder=5)
        ax.fill_between(x_vals, np.maximum(mean_vals - std_vals, 0), mean_vals + std_vals,
                        color='royalblue', alpha=0.2, label='±1 STD', zorder=4)

    ax.set_title(f"dataset: {dfname}     motif: {motif}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}/motif")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_motif_{motif}_dist_st_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_ngs_motif_dist_adv_plot(dfname: str, df: pd.DataFrame, motif: str):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[(df["start_motif_counter"] != 0) | (df["end_motif_counter"] != 0)]
    n_rows = df_filtered.shape[0]

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    data_all = df_filtered['norm_log_NGS_read_count'].dropna()
    density_all = stats.gaussian_kde(data_all)
    x_vals = np.linspace(0, 1, 500)
    y_vals_all = density_all(x_vals) * 100 * bin_width
    label_all = f"all datasets ({len(data_all)})"
    ax.plot(x_vals, y_vals_all, color='gray', linewidth=1.5, label=label_all, zorder=5)

    mean = data_all.mean()
    variance = data_all.var()
    skewness = data_all.skew()
    kurtosis = data_all.kurtosis()

    stats_text = f"mean: {mean:.3f}\nvariance: {variance:.3f}\nskewness: {skewness:.3f}\nkurtosis: {kurtosis:.3f}"
    ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    custom_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ff9896",
        "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
        "#ffbb78", "#aec7e8"
    ]

    for idx, (i_dfname, group) in enumerate(df_filtered.groupby('selector')):
        group_data = group['norm_log_NGS_read_count'].dropna()
        if len(group_data) < 2:
            continue
        density = stats.gaussian_kde(group_data)
        y_vals = density(x_vals) * 100 * bin_width
        color = custom_colors[idx % len(custom_colors)]
        label = f"{i_dfname} ({len(group_data)})"
        ax.plot(x_vals, y_vals, linewidth=1.5, label=label, color=color, zorder=1)

    ax.set_title(f"dataset: {dfname}     motif: {motif}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}/motif")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_motif_{motif}_dist_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

############################
### motif enrichment dis ###
############################

def create_ngs_motif_dist_st_plot(dfname: str, df: pd.DataFrame, motif: str):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[(df["start_motif_counter"] != 0) | (df["end_motif_counter"] != 0)]
    n_rows = df_filtered.shape[0]

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    data_all = df_filtered['norm_log_NGS_read_count'].dropna()
    density_all = stats.gaussian_kde(data_all)
    x_vals = np.linspace(0, 1, 500)
    y_vals_all = density_all(x_vals) * 100 * bin_width

    mean = data_all.mean()
    variance = data_all.var()
    skewness = data_all.skew()
    kurtosis = data_all.kurtosis()

    stats_text = f"mean: {mean:.3f}\nvariance: {variance:.3f}\nskewness: {skewness:.3f}\nkurtosis: {kurtosis:.3f}"
    ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    all_y_vals = []
    first_curve = True

    for _, group in df_filtered.groupby('dfname'):
        group_data = group['norm_log_NGS_read_count'].dropna()
        if len(group_data) < 2:
            continue
        density = stats.gaussian_kde(group_data)
        y_vals = density(x_vals) * 100 * bin_width
        all_y_vals.append(y_vals)

        label = 'datasets' if first_curve else None
        first_curve = False

        ax.plot(x_vals, y_vals, linewidth=1.5, color='lightgray', alpha=0.8, label=label, zorder=1)

    if all_y_vals:
        all_y_vals = np.array(all_y_vals)
        mean_vals = np.mean(all_y_vals, axis=0)
        std_vals = np.std(all_y_vals, axis=0)

        ax.plot(x_vals, mean_vals, color='royalblue', linewidth=1.5, label='mean', zorder=5)
        ax.fill_between(x_vals, np.maximum(mean_vals - std_vals, 0), mean_vals + std_vals,
                        color='royalblue', alpha=0.2, label='±1 STD', zorder=4)

    ax.set_title(f"dataset: {dfname}     motif: {motif}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}/motif")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_motif_{motif}_dist_st_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_ngs_motif_dist_plot(dfname: str, df: pd.DataFrame, motif: str):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[(df["start_motif_counter"] != 0) | (df["end_motif_counter"] != 0)]
    n_rows = df_filtered.shape[0]

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    data_all = df_filtered['norm_log_NGS_read_count'].dropna()
    density_all = stats.gaussian_kde(data_all)
    x_vals = np.linspace(0, 1, 500)
    y_vals_all = density_all(x_vals) * 100 * bin_width
    label_all = f"all datasets ({len(data_all)})"
    ax.plot(x_vals, y_vals_all, color='gray', linewidth=1.5, label=label_all, zorder=5)

    mean = data_all.mean()
    variance = data_all.var()
    skewness = data_all.skew()
    kurtosis = data_all.kurtosis()

    stats_text = f"mean: {mean:.3f}\nvariance: {variance:.3f}\nskewness: {skewness:.3f}\nkurtosis: {kurtosis:.3f}"
    ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    custom_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ff9896",
        "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
        "#ffbb78", "#aec7e8"
    ]

    for idx, (i_dfname, group) in enumerate(df_filtered.groupby('dfname')):
        group_data = group['norm_log_NGS_read_count'].dropna()
        if len(group_data) < 2:
            continue
        density = stats.gaussian_kde(group_data)
        y_vals = density(x_vals) * 100 * bin_width
        color = custom_colors[idx % len(custom_colors)]
        label = f"{i_dfname} ({len(group_data)})"
        ax.plot(x_vals, y_vals, linewidth=1.5, label=label, color=color, zorder=1)

    ax.set_title(f"dataset: {dfname}     motif: {motif}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}/motif")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_motif_{motif}_dist_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

##############################
### motif enrichment stats ###
##############################

def ngs_motif_count_list_stats_adv(selectors: list, motif: str):
    ext_dfs = []
    concat_df = pd.DataFrame

    for selector in selectors:
        dfnames = get_dataset_names(cutoff=40, selection=selector)
        dfs, _ = load_all(dfnames, False)
        df = ngs_motif_count_list_stats(dfnames, dfs, selector, motif)
        ext_dfs.append(df)
        
    concat_df = pd.concat(ext_dfs, axis=0)
    return concat_df

def ngs_motif_count_list_stats(dfnames: list, dfs: list, selector: str, motif: str):
    ext_dfs = []
    concat_df = pd.DataFrame

    for dfname, df in zip(dfnames, dfs):
        df = add_motif_count(df, motif)
        df = add_norm_log_ngs_read_count(df)
        df = add_dfname(dfname, df)
        df = add_selector(df, selector)
        ext_dfs.append(df)

    concat_df = pd.concat(ext_dfs, axis=0)
    return concat_df

############################
### motif enrichment his ###
############################

def create_motif_histogram_plot(dfname: str, dfs: list[pd.DataFrame], motif_length: int, top_n: int = 20):
    """
    Creates a motif frequency histogram from 4 sequence regions.
    Motifs are ranked by sum of relative frequency in site_start and site_end.
    Total counts (only from site_start and site_end) are displayed above bars.
    """

    if len(dfs) != 4:
        raise ValueError("expected list of 4 DataFrames, got {}".format(len(dfs)))

    labels = ["site_start", "del_start", "del_end", "site_end"]
    rel_freqs = []

    # Berechne relative Häufigkeit je Bereich
    for i, df in enumerate(dfs):
        df = df.copy()
        total = df["count"].sum()
        df["rel_freq"] = df["count"] / total * 100
        df["part"] = labels[i]
        rel_freqs.append(df[["motif", "rel_freq", "part"]])

    combined_df = pd.concat(rel_freqs, ignore_index=True)

    # Verwende nur site_start und site_end zur Auswahl der Top-N-Motive
    slc_labels = ["site_start", "site_end"]
    slc_df = combined_df[combined_df["part"].isin(slc_labels)]
    total_rel_freq = slc_df.groupby("motif")["rel_freq"].sum()
    top_motifs = total_rel_freq.nlargest(top_n).index

    # Filtere Daten für Plot
    filtered = combined_df[combined_df["motif"].isin(top_motifs)]
    plot_df = filtered.pivot(index="motif", columns="part", values="rel_freq").fillna(0)
    plot_df = plot_df.loc[top_motifs]
    plot_df = plot_df[["site_start", "del_start", "del_end", "site_end"]]

    # Gesamt-Counts nur aus site_start und site_end
    site_start_df = dfs[0][dfs[0]["motif"].isin(top_motifs)]
    site_end_df = dfs[3][dfs[3]["motif"].isin(top_motifs)]
    combined_counts = pd.concat([site_start_df, site_end_df], ignore_index=True)
    motif_total_counts = combined_counts.groupby("motif")["count"].sum()
    motif_total_counts = motif_total_counts.loc[top_motifs]

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5), tight_layout=True)
    colors = ["#1f77b4", "#6baed6", "#74c476", "#2ca02c"]
    plot_df.plot(kind="bar", ax=ax, edgecolor="black", width=0.8, color=colors)

    # Text mit Gesamt-Count anzeigen
    for i, (motif, total_count) in enumerate(motif_total_counts.items()):
        y_offset = plot_df.loc[motif].max() + 0.5
        ax.text(i, y_offset, f"{int(total_count)}", ha="center", va="bottom", fontsize=10)

    # Layout
    ax.set_title(f"dataset: {dfname}     motif length: {motif_length}")
    ax.set_xlabel("motif")
    ax.set_ylabel("relative frequency (%)")
    ax.legend(title="sequence part")
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylim(0, 10)

    # Speichern
    save_path = os.path.join(RESULTSPATH, f"enrichment/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"motif_{motif_length}_his_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def motif_search(df: pd.DataFrame, motif_length: int):
    """

    """
    seq1_motifs = []
    seq2_motifs = []
    seq3_motifs = []
    seq4_motifs = []

    for seq in df['seq_around_deletion_junction']:
        if len(seq) != 20:
            continue

        seq1 = seq[0:5]
        seq2 = seq[5:10]
        seq3 = seq[10:15]
        seq4 = seq[15:20]

        seq1_motifs += [seq1[i:i + motif_length] for i in range(len(seq1) - motif_length + 1)]
        seq2_motifs += [seq2[i:i + motif_length] for i in range(len(seq2) - motif_length + 1)]
        seq3_motifs += [seq3[i:i + motif_length] for i in range(len(seq3) - motif_length + 1)]
        seq4_motifs += [seq4[i:i + motif_length] for i in range(len(seq4) - motif_length + 1)]

    df_seq1 = pd.DataFrame(Counter(seq1_motifs).items(), columns=["motif", "count"])
    df_seq2 = pd.DataFrame(Counter(seq2_motifs).items(), columns=["motif", "count"])
    df_seq3 = pd.DataFrame(Counter(seq3_motifs).items(), columns=["motif", "count"])
    df_seq4 = pd.DataFrame(Counter(seq4_motifs).items(), columns=["motif", "count"])

    return [df_seq1, df_seq2, df_seq3, df_seq4]

############################
### ngs enrichment ratio ### old
############################

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

############################
### enrichment positions ### old
############################

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

    # selector = "IAV"
    # dfname = selector
    # dfnames = get_dataset_names(cutoff=40, selection=selector)
    # dfs, _ = load_all(dfnames, False)

    #################################
    ### motif enrichment dist adv ###
    #################################

    ### auto ###

    motif = "AAA"
    selector_category = "virus"
    dfname = selector_category
    selectors = get_selctors(selector_category)
    df = ngs_motif_count_list_stats_adv(selectors, motif)
    create_ngs_motif_dist_adv_plot(dfname, df, motif)
    create_ngs_motif_dist_adv_st_plot(dfname, df, motif)

    ############################
    ### motif enrichment dis ###
    ############################

    ### multi ###

    # motif = "AAA"
    # df = ngs_motif_count_list_stats(dfnames, dfs, selector, motif)
    # create_ngs_motif_dist_plot(dfname, df, motif)
    # create_ngs_motif_dist_st_plot(dfname, df, motif)

    ############################
    ### motif enrichment his ###
    ############################

    ### single ###

    # motif_length = 3
    # seq_dfs = motif_search(df, motif_length)
    # create_motif_histogram_plot(dfname, seq_dfs, motif_length)

    ### multi ###

    # df = pd.concat(dfs, ignore_index=True)
    # motif_length = 3
    # seq_dfs = motif_search(df, motif_length)
    # create_motif_histogram_plot(dfname, seq_dfs, motif_length)

    ############################
    ### enrichment positions ### old
    ############################

    ### single ###

    # breakpoint_dfs, segments = find_breakpoint_positions(strain, df)
    # possible_dfs, p_segments = find_enrichment_positions(strain)
    # create_enrichment_nucleotide_position_histogram(dfname, strain, segments, breakpoint_dfs, p_segments, possible_dfs)

    ### single concat ### same strain

    # concat_pseudo_df = pd.concat(dfs, ignore_index=True)
    # dfname = strain

    # breakpoint_dfs, segments = find_breakpoint_positions(strain, df)
    # possible_dfs, p_segments = find_enrichment_positions(strain)
    # create_enrichment_nucleotide_position_histogram(dfname, strain, segments, breakpoint_dfs, p_segments, possible_dfs)

    ############################
    ### ngs enrichment ratio ### old
    ############################

    ### single ###

    # df = ngs_enrichment_ratio_concat(dfs)
    # create_ngs_enrichment_ratio_plot(dfname, df)

    ### multi ###

    # df = ngs_enrichment_ratio_concat(dfs)
    # create_ngs_enrichment_ratio_plot(dfname, df)