'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu
from typing import Tuple

sys.path.insert(0, "..")
from utils import get_seq_len, get_dataset_names, load_all, get_sequence, get_p_value_symbol
from utils import add_direct_repeat_len, cap_direct_repeat_len, count_direct_repeats, count_direct_repeats_segment, add_norm_log_ngs_read_count, add_dfname, add_selector, get_selctors
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


#####################################
### ngs dr ratio per dvg dist adv ###
#####################################

def create_ngs_repeat_dist_adv_st_plot(dfname: str, df: pd.DataFrame):
    for dr_length in range(0, 6):
        create_ngs_repeat_length_dist_adv_st_plot(dfname, df, dr_length)

def create_ngs_repeat_length_dist_adv_st_plot(dfname: str, df: pd.DataFrame, dr_length: int):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[df['direct_repeat_len'] == dr_length]
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

    if dr_length == 5:
        ax.set_title(f"dataset: {dfname}     direct repeat length: >4     number of DVGs: {n_rows}")
    else:
        ax.set_title(f"dataset: {dfname}     direct repeat length: {dr_length}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/dr_length")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_length_{dr_length}_dist_st_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_ngs_repeat_dist_adv_plot(dfname: str, df: pd.DataFrame):
    for dr_length in range(0, 6):
        create_ngs_repeat_length_dist_adv_plot(dfname, df, dr_length)

def create_ngs_repeat_length_dist_adv_plot(dfname: str, df: pd.DataFrame, dr_length: int):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[df['direct_repeat_len'] == dr_length]
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

    if dr_length == 5:
        ax.set_title(f"dataset: {dfname}     direct repeat length: >4     number of DVGs: {n_rows}")
    else:
        ax.set_title(f"dataset: {dfname}     direct repeat length: {dr_length}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/dr_length")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_length_{dr_length}_dist_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

#################################
### ngs dr ratio per dvg dist ###
#################################

def create_ngs_repeat_dist_st_plot(dfname: str, df: pd.DataFrame):
    for dr_length in range(0, 6):
        create_ngs_repeat_length_dist_st_plot(dfname, df, dr_length)

def create_ngs_repeat_length_dist_st_plot(dfname: str, df: pd.DataFrame, dr_length: int):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[df['direct_repeat_len'] == dr_length]
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

    if dr_length == 5:
        ax.set_title(f"dataset: {dfname}     direct repeat length: >4     number of DVGs: {n_rows}")
    else:
        ax.set_title(f"dataset: {dfname}     direct repeat length: {dr_length}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/dr_length")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_length_{dr_length}_dist_st_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_ngs_repeat_dist_plot(dfname: str, df: pd.DataFrame):
    for dr_length in range(0, 6):
        create_ngs_repeat_length_dist_plot(dfname, df, dr_length)

def create_ngs_repeat_length_dist_plot(dfname: str, df: pd.DataFrame, dr_length: int):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    df_filtered = df[df['direct_repeat_len'] == dr_length]
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

    if dr_length == 5:
        ax.set_title(f"dataset: {dfname}     direct repeat length: >4     number of DVGs: {n_rows}")
    else:
        ax.set_title(f"dataset: {dfname}     direct repeat length: {dr_length}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/dr_length")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_length_{dr_length}_dist_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

################################
### ngs dr ratio per dvg reg ###
################################

def create_ngs_repeat_reg_st_plot(dfname: str, df: pd.DataFrame):
    for dr_length in range(0, 6):
        create_ngs_repeat_length_reg_st_plot(dfname, df, dr_length)

def create_ngs_repeat_length_reg_st_plot(dfname: str, df: pd.DataFrame, dr_length: int, degree: int = 4):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    df_filtered = df[df['direct_repeat_len'] == dr_length]
    n_rows = df_filtered.shape[0]

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers_reshaped = bin_centers.reshape(-1, 1)

    def fit_regression(y_values):
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(bin_centers_reshaped)
        model = LinearRegression()
        model.fit(x_poly, y_values)
        return model.predict(x_poly)

    all_hists = []

    for idx, (i_dfname, group) in enumerate(df_filtered.groupby('dfname')):
        hist, _ = np.histogram(group['norm_log_NGS_read_count'], bins=bins)
        hist = hist / group.shape[0] * 100
        all_hists.append(hist)
        y_pred = fit_regression(hist)
        label = 'datasets' if idx == 0 else None
        ax.plot(bin_centers, y_pred, linewidth=1.5, label=label, color='lightgray', zorder=1)

    all_hists = np.array(all_hists)
    mean_hist = np.mean(all_hists, axis=0)
    std_hist = np.std(all_hists, axis=0)

    y_pred_mean = fit_regression(mean_hist)
    ax.plot(bin_centers, y_pred_mean, color='royalblue', linewidth=1.5, label='mean', zorder=5)

    y_pred_std_upper = fit_regression(mean_hist + std_hist)
    y_pred_std_lower = fit_regression(np.maximum(mean_hist - std_hist, 0))  # kein negativer Bereich

    ax.fill_between(bin_centers, y_pred_std_lower, y_pred_std_upper,
                    color='royalblue', alpha=0.2, label='±1 STD', zorder=4)

    if dr_length == 5:
        ax.set_title(f"dataset: {dfname}     direct repeat length: >4     number of DVGs: {n_rows}")
    else:
        ax.set_title(f"dataset: {dfname}     direct repeat length: {dr_length}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)

    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/dr_length")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_length_{dr_length}_reg_st_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_ngs_repeat_reg_plot(dfname: str, df: pd.DataFrame):
    for dr_length in range(0, 6):
        create_ngs_repeat_length_reg_plot(dfname, df, dr_length)

def create_ngs_repeat_length_reg_plot(dfname: str, df: pd.DataFrame, dr_length: int, degree: int = 4):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    df_filtered = df[df['direct_repeat_len'] == dr_length]
    n_rows = df_filtered.shape[0]

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers_reshaped = bin_centers.reshape(-1, 1)

    def fit_regression(y_values):
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(bin_centers_reshaped)
        model = LinearRegression()
        model.fit(x_poly, y_values)
        return model.predict(x_poly)

    hist_all, _ = np.histogram(df_filtered['norm_log_NGS_read_count'], bins=bins)
    hist_all = hist_all / df_filtered.shape[0] * 100
    y_pred_all = fit_regression(hist_all)
    label_all = f"all datasets ({df_filtered.shape[0]})"
    ax.plot(bin_centers, y_pred_all, color='gray', linewidth=1.5, label=label_all, zorder=5)

    custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ff9896",
                          "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
                          "#ffbb78", "#aec7e8"]

    for idx, (i_dfname, group) in enumerate(df_filtered.groupby('dfname')):
        hist, _ = np.histogram(group['norm_log_NGS_read_count'], bins=bins)
        hist = hist / group.shape[0] * 100
        y_pred = fit_regression(hist)
        color = custom_colors[idx % len(custom_colors)]
        label = f"{i_dfname} ({group.shape[0]})"
        ax.plot(bin_centers, y_pred, linewidth=1.5, label=label, color=color, zorder=1)

    if dr_length == 5:
        ax.set_title(f"dataset: {dfname}     direct repeat length: >4     number of DVGs: {n_rows}")
    else:
        ax.set_title(f"dataset: {dfname}     direct repeat length: {dr_length}     number of DVGs: {n_rows}")

    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bins), 5)
    ax.set_xticks(bins[tick_indices])
    ax.set_xticklabels([f"{bins[i]:.2f}" for i in tick_indices])

    ax.legend(loc='best', fontsize='small', ncol=2)

    ax.set_ylim(-2, 15)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/dr_length")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_length_{dr_length}_reg_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

##################################
### ngs dr ratio per dvg stats ###
##################################

def ngs_repeat_length_list_stats_adv(selectors: list):
    ext_dfs = []
    concat_df = pd.DataFrame

    for selector in selectors:
        dfnames = get_dataset_names(cutoff=40, selection=selector)
        dfs, _ = load_all(dfnames, False)
        df = ngs_repeat_length_list_stats(dfnames, dfs, selector)
        ext_dfs.append(df)
        
    concat_df = pd.concat(ext_dfs, axis=0)
    return concat_df

def ngs_repeat_length_list_stats(dfnames: list, dfs: list, selector: str):
    ext_dfs = []
    concat_df = pd.DataFrame

    for dfname, df in zip(dfnames, dfs):
        df = add_direct_repeat_len(df)
        df = cap_direct_repeat_len(df)
        df = add_norm_log_ngs_read_count(df)
        df = add_dfname(dfname, df)
        df = add_selector(df, selector)
        ext_dfs.append(df)

    concat_df = pd.concat(ext_dfs, axis=0)
    return concat_df

################################
### ngs dr ratio per dvg sig ###
################################

def test_significance_ngs_repeat_ratio_dvg(dfname: str, df: pd.DataFrame):
    grouped_data = [
        group['NGS_read_count_ratio_dvg'].dropna().values
        for _, group in df.groupby('direct_repeat_len')
    ]

    f_stat, p_anova = f_oneway(*grouped_data)
    h_stat, p_kruskal = kruskal(*grouped_data)

    print(f"dataset:        {dfname}")
    print(f"ANOVA:          F = {f_stat:.4f}, p = {p_anova:.10f}, level = {get_p_value_symbol(p_anova)}")
    print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kruskal:.10f}, level = {get_p_value_symbol(p_kruskal)}")

    return {
        'dataset': {dfname},
        'anova': {'F': f_stat, 'p': p_anova},
        'kruskal': {'H': h_stat, 'p': p_kruskal}
    }

def test_pairwise_significance_ngs_repeat_ratio_dvg(dfname: str, df: pd.DataFrame, group1_len: int, group2_len: int):
    group1 = df[df['direct_repeat_len'] == group1_len]['NGS_read_count_ratio_dvg'].dropna().values
    group2 = df[df['direct_repeat_len'] == group2_len]['NGS_read_count_ratio_dvg'].dropna().values

    t_stat, p_ttest = ttest_ind(group1, group2, equal_var=False)
    u_stat, p_mannwhitney = mannwhitneyu(group1, group2, alternative='two-sided')

    print(f"dataset:         {dfname}")
    print(f"Groups:          {group1_len} vs {group2_len}")
    print(f"T-test:          t = {t_stat:.4f}, p = {p_ttest:.10f}, level = {get_p_value_symbol(p_ttest)}")
    print(f"Mann-Whitney U:  U = {u_stat:.4f}, p = {p_mannwhitney:.10f}, level = {get_p_value_symbol(p_mannwhitney)}")

    return {
        'dataset': dfname,
        'groups': (group1_len, group2_len),
        't_test': {'t': t_stat, 'p': p_ttest},
        'mannwhitney': {'U': u_stat, 'p': p_mannwhitney}
    }

################################
### ngs dr ratio per dvg vio ###
################################

def create_ngs_repeat_ratio_dvg_vio_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    sns.violinplot(
        data=df,
        x='direct_repeat_len',
        y='NGS_read_count_ratio_dvg',
        inner='box',  # zeigt Median + IQR im Violinplot
        palette='Blues',
        linewidth=1,
        ax=ax
    )

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("direct repeat length (nucleotides)")
    ax.set_ylabel("NGS read count per DVG (%)")

    ax.set_xticklabels(["0", "1", "2", "3", "4", ">4"])

    ax.set_ylim(0, 0.6)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_ratio_dvg_vio_{dfname}.png"
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

################################
### ngs dr ratio per dvg bar ###
################################

def create_ngs_repeat_ratio_dvg_bar_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)

    # Gruppierung und Statistik
    grouped = df.groupby('direct_repeat_len')['NGS_read_count_ratio_dvg']
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    unique_lengths = sorted(means.index)

    # Balkendiagramm
    ax.bar(
        unique_lengths,
        means,
        yerr=stds,
        color='royalblue',
        edgecolor='black',
        capsize=5,
        error_kw={'elinewidth': 1, 'capthick': 1},
        zorder=1
    )

    # Einzelne Datenpunkte als Stripplot
    sns.stripplot(
        data=df,
        x='direct_repeat_len',
        y='NGS_read_count_ratio_dvg',
        ax=ax,
        color='black',
        size=3,
        jitter=True,
        zorder=2,
        alpha=0.6
    )

    # Achsen und Titel
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("direct repeat length (nucleotides)")
    ax.set_ylabel("average NGS read count per DVG (%)")
    ax.set_xticks(unique_lengths)
    ax.set_xticklabels(["0", "1", "2", "3", "4", ">4"])
    ax.set_ylim(0, 0.6)

    # Speichern
    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_ratio_dvg_bar_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def ngs_repeat_ratio_dvg_list_bar(dfnames: list, dfs: list):
    ext_dfs = []
    concat_df = pd.DataFrame
    for dfname, df in zip(dfnames, dfs):
        ext_df = ngs_repeat_ratio_dvg(df)
        ext_df = add_dfname(dfname, ext_df)
        ext_dfs.append(ext_df)
    concat_df = pd.concat(ext_dfs, axis=0)
    agg_df = concat_df.groupby(['dfname', 'direct_repeat_len'], as_index=False)['NGS_read_count_ratio_dvg'].sum()
    agg_df['NGS_read_count_ratio_dvg'] = agg_df['NGS_read_count_ratio_dvg']
    return agg_df

############################
### ngs dr ratio per dvg ###
############################

def create_ngs_repeat_ratio_dvg_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    
    total_count = df['NGS_read_count_ratio_dvg'].sum()
    df['dist_ratio'] = df['NGS_read_count_ratio_dvg']
    unique_lengths = df['direct_repeat_len'].unique()
    ax.bar(unique_lengths, df['dist_ratio'], color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("direct repeat length (nucleotides)")
    ax.set_ylabel("average NGS read count per DVG (-)")
    ax.set_xticks(unique_lengths)
    ax.set_xticklabels(["0", "1", "2", "3", "4", ">4"]) 

    ax.set_ylim(0, 0.6)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_ratio_dvg_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi = 300)
    plt.close()

def ngs_repeat_ratio_dvg_list(dfs: list):
    ext_dfs = []
    concat_df = pd.DataFrame
    for df in dfs:
        ext_df = ngs_repeat_ratio_dvg(df)
        ext_dfs.append(ext_df)
    concat_df = pd.concat(ext_dfs, axis=0)
    agg_df = concat_df.groupby('direct_repeat_len', as_index=False)['NGS_read_count_ratio_dvg'].sum()
    agg_df['NGS_read_count_ratio_dvg'] = agg_df['NGS_read_count_ratio_dvg'] / len(dfs)
    return agg_df

def ngs_repeat_ratio_dvg(df: pd.DataFrame):
    df = add_direct_repeat_len(df)
    df = cap_direct_repeat_len(df)
    df = add_norm_log_ngs_read_count(df)
    ngs_read_count_sum = 0
    ngs_dvg_dict = dict({i: 0 for i in range(0, 6)})
    for index, row in df.iterrows():
        ngs_read_count = row['norm_log_NGS_read_count']
        ngs_read_count_sum = ngs_read_count_sum + ngs_read_count
        i = row["direct_repeat_len"]
        ngs_dvg_dict[i] = ngs_dvg_dict[i] + ngs_read_count
    dr_length_dict = count_direct_repeats(df)
    for i in range(0, 6):
        if dr_length_dict[i] == 0:
            ngs_dvg_dict[i] = 0
        else:
            ngs_dvg_dict[i] = ngs_dvg_dict[i] / dr_length_dict[i]
    df = pd.DataFrame(list(ngs_dvg_dict.items()), columns=['direct_repeat_len', 'NGS_read_count_ratio_dvg'])
    return df

####################
### ngs dr ratio ###
####################

def create_ngs_repeat_ratio_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)

    total_count = df['NGS_read_count_ratio'].sum()
    df['dist_ratio'] = df['NGS_read_count_ratio'] / total_count * 100
    unique_lengths = df['direct_repeat_len'].unique()
    ax.bar(unique_lengths, df['dist_ratio'], color='royalblue', edgecolor='black')

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("direct repeat length (nucleotides)")
    ax.set_ylabel("NGS read count distribution (%)")
    ax.set_xticks(unique_lengths)
    ax.set_xticklabels(["0", "1", "2", "3", "4", ">4"]) 

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_dr_ratio_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi = 300)
    plt.close()

def ngs_repeat_ratio_list(dfs: list):
    ext_dfs = []
    concat_df = pd.DataFrame
    for df in dfs:
        ext_df = ngs_repeat_ratio(df)
        ext_dfs.append(ext_df)
    concat_df = pd.concat(ext_dfs, axis=0)
    agg_df = concat_df.groupby('direct_repeat_len', as_index=False)['NGS_read_count_ratio'].sum()
    agg_df['NGS_read_count_ratio'] = agg_df['NGS_read_count_ratio'] / len(dfs)
    return agg_df

def ngs_repeat_ratio(df: pd.DataFrame):
    df = add_direct_repeat_len(df)
    df = cap_direct_repeat_len(df)
    df = add_norm_log_ngs_read_count(df)
    ngs_read_count_sum = 0
    ngs_ratio_dict = dict({i: 0 for i in range(0, 6)})
    for index, row in df.iterrows():
        ngs_read_count = row['norm_log_NGS_read_count']
        ngs_read_count_sum = ngs_read_count_sum + ngs_read_count
        i = row["direct_repeat_len"]
        ngs_ratio_dict[i] = ngs_ratio_dict[i] + ngs_read_count
    df = pd.DataFrame(list(ngs_ratio_dict.items()), columns=['direct_repeat_len', 'NGS_read_count_ratio'])
    df['NGS_read_count_ratio'] /= ngs_read_count_sum
    return df

##############################################
### nucleotide positions of direct repeats ###
##############################################

def create_repeats_nucleotide_position_histogram(dfname: str, strain: str, segments: list, dfs: list, p_segments: list, p_dfs: list):
    for segment in SEGMENTS:
        index = segments.index(segment)
        p_index = p_segments.index(segment)
        create_repeats_nucleotide_position_histogram_segment(dfname, strain, segment, dfs[index], p_dfs[p_index])

def create_repeats_nucleotide_position_histogram_segment(dfname: str, strain: str, segment: str, df: pd.DataFrame, p_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    
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

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/repeat_hit_ratio")
    os.makedirs(save_path, exist_ok=True)
    fname = f"{segment}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_possible_repeats_nucleotide_positions(strain: str):
    positions_dfs = []
    segments = []
    for segment in SEGMENTS:
        positions_df = find_possible_repeats_nucleotide_positions_segment(strain, segment)
        positions_dfs.append(positions_df)
        segments.append(segment)
    return positions_dfs, segments

def find_possible_repeats_nucleotide_positions_segment(strain: str, segment: str):
    segment_path = os.path.join(DATAPATH, f"repeats/{strain}/{segment}.csv")
    repeats_df = pd.read_csv(segment_path, index_col=0)
    seq = get_sequence(strain, segment)
    seq_len = len(seq)
    positions_df = pd.DataFrame({
        'nucleotide-position': range(1, seq_len + 1),
        'hits': [0] * seq_len
    })
    for _, row in repeats_df.iterrows():
        start_ref = row["start_ref"]
        end_ref = row["end_ref"]
        for i in range(start_ref, end_ref + 1):
            positions_df.loc[positions_df['nucleotide-position'] == i, 'hits'] += 1
        start_rep = row["start_rep"]
        end_rep = row["end_rep"]
        for i in range(start_rep, end_rep + 1):
            positions_df.loc[positions_df['nucleotide-position'] == i, 'hits'] += 1
    return positions_df

def find_direct_repeats_nucleotide_positions(strain: str, df: pd.DataFrame):
    positions_dfs = []
    segments = []
    for segment in SEGMENTS:
        positions_df = find_direct_repeats_nucleotide_positions_segment(strain, segment, df)
        positions_dfs.append(positions_df)
        segments.append(segment)
    return positions_dfs, segments

def find_direct_repeats_nucleotide_positions_segment(strain: str, segment: str, df: pd.DataFrame):
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
        direct_repeat_len = int(row["direct_repeat_len"])
        for i in range(direct_repeat_len):
            positions_df.loc[positions_df['nucleotide-position'] == start - i, 'hits'] += 1
            positions_df.loc[positions_df['nucleotide-position'] == end - i, 'hits'] += 1
    return positions_df

######################################
### hit ratio of potential repeats ###
######################################

def create_repeats_ratio_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)

    segments = df['Segment'].unique()
    heights = []
    
    for segment in segments:
        segment_data = df[df['Segment'] == segment]
        if segment_data.empty:
            continue
        heights.append(segment_data['hits_repeat_ratio'].sum())

    ax.bar(segments, heights, color='royalblue', edgecolor='black', width=0.8)
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("segment")
    ax.set_ylabel("relative hits of possible repeats (%)")
    ax.set_xticks(segments)
    ax.set_xticklabels(segments)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/repeat_hit_ratio")
    os.makedirs(save_path, exist_ok=True)
    fname = f"repeat_hit_ratio_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_repeats_ratio(dfname: str, df: pd.DataFrame):
    stats_df = pd.DataFrame(columns=['Segment', 'possible_repeats', 'hits_count', 'hits_repeat_ratio'])
    strain = DATASET_STRAIN_DICT[dfname]
    data_path = os.path.join(DATAPATH, f"repeats/{strain}")
    for segment in SEGMENTS:
        segment_path = os.path.join(data_path, f"{segment}.csv")
        repeats = pd.read_csv(segment_path, index_col=0)
        total_rows = repeats.shape[0]
        dic = count_direct_repeats_segment(df, segment)
        hits_count = dic[5]
        if total_rows > 0:
            ratio = hits_count / total_rows
        else:
            ratio = 0
        new_row = pd.DataFrame({
            'Segment': [segment],
            'possible_repeats': [total_rows],
            'hits_count': [hits_count],
            'hits_repeat_ratio': [ratio]
        })    
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
    return stats_df

def mark_repeats(dfname: str, df: pd.DataFrame):
    strain = DATASET_STRAIN_DICT[dfname]
    data_path = os.path.join(DATAPATH, f"repeats/{strain}")
    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/stats")
    os.makedirs(save_path, exist_ok=True)
    for segment in SEGMENTS:
        segment_path = os.path.join(data_path, f"{segment}.csv")
        repeats = pd.read_csv(segment_path, index_col=0)
        repeats['hits'] = 0
        repeats['NGS_read_count'] = 0
        segment_df = df[df["Segment"] == segment]
        for _, df_row in segment_df.iterrows():
            start = df_row["Start"]
            end = df_row["End"]
            for _, repeats_row in repeats.iterrows():
                s = start == repeats_row["end_ref"]
                e = end - 1 == repeats_row["end_rep"]
                if s and e:
                    repeats_row['hits'] = repeats_row['hits'] + 1
                    repeats_row['NGS_read_count'] = repeats_row['NGS_read_count'] + df_row['NGS_read_count']
        fname = f"{segment}.csv"
        repeats.to_csv(os.path.join(save_path, fname))


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    ### SINGLE ###

    # dfnames = ["Alnaji2021"]
    # dfs, _ = load_all(dfnames)
    # dfname = dfnames[0]

    # df = dfs[0]
    # strain = DATASET_STRAIN_DICT[dfname]

    ### MULTI ###

    # selector = "in vitro"
    # dfname = selector
    # dfnames = get_dataset_names(cutoff=40, selection=selector)
    # dfs, _ = load_all(dfnames, False)

    #####################################
    ### ngs dr ratio per dvg dist adv ###
    #####################################

    ### auto ###

    selector_category = "virus"
    dfname = selector_category
    selectors = get_selctors(selector_category)
    df = ngs_repeat_length_list_stats_adv(selectors)
    create_ngs_repeat_dist_adv_plot(dfname, df)
    create_ngs_repeat_dist_adv_st_plot(dfname, df)

    #################################
    ### ngs dr ratio per dvg dist ###
    #################################

    ### multi ###

    # df = ngs_repeat_length_list_stats(dfnames, dfs, selector)
    # create_ngs_repeat_dist_plot(dfname, df)
    # create_ngs_repeat_dist_st_plot(dfname, df)

    ################################
    ### ngs dr ratio per dvg reg ###
    ################################

    ### multi ###

    # df = ngs_repeat_length_list_stats(dfnames, dfs, selector)
    # create_ngs_repeat_reg_plot(dfname, df)
    # create_ngs_repeat_reg_st_plot(dfname, df)

    ################################
    ### ngs dr ratio per dvg sig ###
    ################################

    ### multi ###

    # df = ngs_repeat_ratio_dvg_list_bar(dfnames, dfs)
    # test_significance_ngs_repeat_ratio_dvg(dfname, df)
    # test_pairwise_significance_ngs_repeat_ratio_dvg(dfname, df, 0, 5)

    ################################
    ### ngs dr ratio per dvg vio ###
    ################################

    ### multi ###

    # df = ngs_repeat_ratio_dvg_list_bar(dfnames, dfs)
    # create_ngs_repeat_ratio_dvg_vio_plot(dfname, df)

    ################################
    ### ngs dr ratio per dvg bar ###
    ################################

    ### multi ###

    # df = ngs_repeat_ratio_dvg_list_bar(dfnames, dfs)
    # create_ngs_repeat_ratio_dvg_bar_plot(dfname, df)

    ############################
    ### ngs dr ratio per dvg ###
    ############################

    ### single ###

    # df = ngs_repeat_ratio_dvg_list(dfs)
    # create_ngs_repeat_ratio_dvg_plot(dfname, df)

    ### multi ###

    # df = ngs_repeat_ratio_dvg_list(dfs)
    # create_ngs_repeat_ratio_dvg_plot(dfname, df)

    ####################
    ### ngs dr ratio ###
    ####################

    ### single ###

    # df = ngs_repeat_ratio_list(dfs)
    # create_ngs_repeat_ratio_plot(dfname, df)

    ### multi ###

    # df = ngs_repeat_ratio_list(dfs)
    # create_ngs_repeat_ratio_plot(dfname, df)
    
    ##############################################
    ### nucleotide positions of direct repeats ###
    ##############################################

    # add_df = add_direct_repeat_len(df)
    # repeats_dfs, segments = find_direct_repeats_nucleotide_positions(strain, add_df)
    # possible_dfs, p_segments = find_possible_repeats_nucleotide_positions(strain)

    # create_repeats_nucleotide_position_histogram(dfname, strain, segments, repeats_dfs, p_segments, possible_dfs)

    ######################################
    ### hit ratio of potential repeats ###
    ######################################

    # ratio_df = calculate_repeats_ratio(dfname, df)
    # create_repeats_ratio_plot(dfname, ratio_df)