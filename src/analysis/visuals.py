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

import RNA

sys.path.insert(0, "..")
from utils import get_dataset_names, load_all
from utils import manage_separate_specifiers, load_all_preprocessed
from utils import rename_feature, filter_for_feature, filter_against_feature, add_feature_percentile_rank, add_log_feature, add_norm_feature, add_separate_ngs_features
from utils import manage_intersects
from utils import add_dvg_sequence, add_dvg_length, add_cg_content, add_region_lengths

from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df
from utils import compute_percentile_rank_count_df, compute_seq_feature_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_count_heatmap_sum_df, compute_feature_freq_heatmap_df

from utils import add_direct_repeat_len, cap_direct_repeat_len
from utils import add_loop_count, add_max_loop

from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import COLORS, STRAINS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'visuals')


###############
### scripts ###
###############

### heatmap ###

def run_site_motif_heatmap_analysis(dfs: list, top_n: int = 5, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    max_motif_length = 7

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    plot_names = ['site_before_deletion', 'site_deletion_start', 'site_deletion_end', 'site_after_deletion']

    all_mean_freq_diffs = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_freq_diffs  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        dvg_count = df_copy.shape[0]
        mean_freq_diffs, top_freq_diffs = make_site_motif_heatmap_analysis(motif_df, plot_names, dvg_count, motif_length, top_n, data, strain, segment, intersects)
        print(f'motif_length {motif_length} completed')

        for i in range(0, 4):
            all_mean_freq_diffs[i][motif_length - 1] = mean_freq_diffs[i]
            all_top_freq_diffs[i][motif_length - 1] = top_freq_diffs[i]

    create_metric_spine_plot('site_mean_frequency_difference', 'motif_length', '-', plot_names, all_mean_freq_diffs, 'mean frequency difference (%)', dvg_count, data, strain, segment, intersects)
    create_metric_spine_plot('site_top_frequency_difference', 'motif_length', '-', plot_names, all_top_freq_diffs, 'top frequency difference (%)', dvg_count, data, strain, segment, intersects)

def run_reg_site_motif_heatmap_analysis(dfs: list, top_n: int = 5, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    max_motif_length = 7

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    reg_plot_names = ['reg_site_before_deletion', 'reg_site_deletion_start', 'reg_site_deletion_end', 'reg_site_after_deletion']

    all_mean_coefficients = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_coefficients  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        dvg_count = df_copy.shape[0]
        mean_coefficients, top_coefficients = make_reg_site_motif_heatmap_analysis(motif_df, reg_plot_names, dvg_count, motif_length, top_n, data, strain, segment, intersects)
        print(f'motif_length {motif_length} completed')

        for i in range(0, 4):
            all_mean_coefficients[i][motif_length - 1] = mean_coefficients[i]
            all_top_coefficients[i][motif_length - 1] = top_coefficients[i]

    create_metric_spine_plot('reg_site_mean_coefficients', 'motif_length', '-', reg_plot_names, all_mean_coefficients, 'mean regression coefficients (-)', dvg_count, data, strain, segment, intersects)
    create_metric_spine_plot('reg_site_top_coefficients', 'motif_length', '-', reg_plot_names, all_top_coefficients, 'top regression coefficients (-)', dvg_count, data, strain, segment, intersects)

def run_repeat_heatmap_analysis(dfs: list, cap_length: int = 5, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    dvg_count = df.shape[0]
    heatmap_name = 'direct_repeat'

    make_repeat_heatmap_analysis(df, heatmap_name, dvg_count, cap_length, data, strain, segment, intersects)

def run_mfe_heatmap_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    dvg_count = df.shape[0]
    heatmap_name = 'MFE'

    make_mfe_heatmap_analysis(df, heatmap_name, dvg_count, data, strain, segment, intersects)

### freq ###

def run_dvg_motif_freq_analysis(dfs: list, top_n: int = 5, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    max_motif_length = 7

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    dvg_count = df.shape[0]
    df = add_dvg_sequence(df)

    plot_name = 'dvg'
    plot_names = []

    all_mean_freq_diffs = []
    all_top_freq_diffs  = []
    mean_freq_diffs = []
    top_freq_diffs = []
    for motif_length in range(1, max_motif_length + 1):
        df, skipped_count = add_site_motifs(df, motif_length)
        print(f'{skipped_count} candidates have been left out')
        mean_freq_diff, top_freq_diff = make_dvg_motif_freq_analysis(df, plot_name, dvg_count, motif_length, top_n, data, strain, segment, intersects)
        mean_freq_diffs.append(mean_freq_diff)
        top_freq_diffs.append(top_freq_diff)
        print(f'motif_length {motif_length} completed')

    all_mean_freq_diffs.append(mean_freq_diffs)
    all_top_freq_diffs.append(top_freq_diffs)
    plot_names.append(plot_name)
    create_metric_spine_plot('dvg_mean_frequency_difference', 'motif_length', '-', plot_names, all_mean_freq_diffs, 'mean frequency difference (%)', dvg_count, data, strain, segment, intersects)
    create_metric_spine_plot('dvg_top_frequency_difference', 'motif_length', '-', plot_names, all_top_freq_diffs, 'top frequency difference (%)', dvg_count, data, strain, segment, intersects)

### distribution ###

def run_ngs_count_distribution_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, feature)

    create_feature_distribution_plot(df, feature, '-', True, data, strain, segment, intersects)

### correlation ###

def run_ngs_count_dvg_pri_features_correlation_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, feature)

    df = add_dvg_sequence(df)
    df = add_dvg_length(df)
    df = add_cg_content(df)
    df = add_region_lengths(df)

    # dvg length #
    create_feature_correlation_plot(df, feature, '-', 'dvg_length', 'nucleotides', data, strain, segment, intersects)
    # deletion_length #
    create_feature_correlation_plot(df, feature, '-', 'deletion_length', 'nucleotides', data, strain, segment, intersects)
    # 5_end_length #
    create_feature_correlation_plot(df, feature, '-', '5_end_length', 'nucleotides', data, strain, segment, intersects)
    # 3_end_length #
    create_feature_correlation_plot(df, feature, '-', '3_end_length', 'nucleotides', data, strain, segment, intersects)
    # cg content #
    create_feature_correlation_plot(df, feature, '-', 'cg_content', '%', data, strain, segment, intersects)

def run_ngs_count_dvg_sec_features_correlation_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, feature)

    # MFE #
    create_feature_correlation_plot(df, feature, '-', 'MFE', 'kcal/mol', data, strain, segment, intersects)
    # loop count #
    add_loop_count(df, 'structure', 'loop_count')
    create_feature_correlation_plot(df, feature, '-', 'loop_count', '-', data, strain, segment, intersects)
    # max loop #
    add_max_loop(df, 'structure', 'max_loop')
    create_feature_correlation_plot(df, feature, '-', 'max_loop', 'nucleotides', data, strain, segment, intersects)

def run_ngs_count_site_sec_features_correlation_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, feature)

    for i in range(1, 5):
        # MFE #
        create_feature_correlation_plot(df, feature, '-', f'MFE_site{i}_motif', 'kcal/mol', data, strain, segment, intersects)
        # loop count #
        add_loop_count(df, f'structure_site{i}_motif', f'loop_count_site{i}_motif')
        create_feature_correlation_plot(df, feature, '-', f'loop_count_site{i}_motif', '-', data, strain, segment, intersects)
        # max loop #
        add_max_loop(df, f'structure_site{i}_motif', f'max_loop_site{i}_motif')
        create_feature_correlation_plot(df, feature, '-', f'max_loop_site{i}_motif', 'nucleotides', data, strain, segment, intersects)

def run_dvg_length_mfe_correlation_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_dvg_length(df)

    feature = 'dvg_length'
    norm_feature = 'norm_' + feature
    df = add_norm_feature(df, feature, norm_feature)

    create_feature_correlation_plot(df, norm_feature, "nucleotides", "MFE", "kcal/mol", data, strain, segment, intersects)

### other ###

def run_sec_structure_plots(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)

    df = add_dvg_sequence(df)
    sequences = df['dvg_sequence'].head(5)

    save_path = os.path.join(RESULTSPATH, 'structures', data, strain, segment, 'all')
    os.makedirs(save_path, exist_ok=True)

    for idx, seq in enumerate(sequences):
        if not isinstance(seq, str) or not seq.strip():
            continue

        seq = seq.strip().upper()
        structure, mfe = RNA.fold(seq)

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
            f.write(f"{seq}\n{structure}\n")
            temp_input = f.name

        try:
            subprocess.run(['RNAplot', '-o', 'svg', '-t', '0', '-i', temp_input], check=True)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] RNAplot failed for sequence {idx+1}: {e}')
            continue

        svg_source = 'rna.svg'
        svg_target = os.path.join(save_path, f'structure_{idx+1}_mfe_{mfe:.2f}.svg')
        if os.path.exists(svg_source):
            os.rename(svg_source, svg_target)
        else:
            print(f'[WARNING] RNAplot output not found for sequence {idx+1}')

        os.remove(temp_input)

###############
### general ###
###############

### plot ###

def create_freq_heatmap_plot(
    heatmap_name: str,
    field: str,
    field_specifier: str,
    freq_heatmap_df: pd.DataFrame,
    count_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame,
    dvg_count: int,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_name = '',
    feature = 0,
    metric_name = '',
    metric: float = 0,
    lead_name = '',
    lead: float = 0
    ):  
    '''

    '''
    pivot_df = freq_heatmap_df
    field_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    field_count_series = count_df.set_index(field)['count']

    field_labels = [
        f'{field}\n(n={field_count_series.get(field, 0)})'
        for field in field_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    min_width = 13
    fig_width = max(min_width, 1 + len(field_labels) * 0.6)
    fig_height = 1 + len(rank_labels) * 0.4
    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        pivot_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=field_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt='.1f'
    )

    ax.set_xlabel(f'{field} ({field_specifier})')
    ax.set_ylabel('NGS_rank (percentile)')

    title_name = f'{field} heatmap ({heatmap_name})'
    if feature_name:
        title_name += f' - {feature_name}: {feature}'
    if metric_name:
        title_name += f' - {metric_name}: {metric}'
    if lead_name:
        title_name += f' - {lead_name}: {lead}'
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
    ax.set_title(title_name)

    save_path = os.path.join(RESULTSPATH, field, data)
    save_path = os.path.join(save_path, strain)
    save_path = os.path.join(save_path, segment)
    save_path = os.path.join(save_path, intersects)
    if feature_name:
        save_path = os.path.join(save_path, f'{feature_name}_{feature}')
    os.makedirs(save_path, exist_ok=True)

    fname = f'{heatmap_name}_heatmap'
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_freq_bar_plot(
    bar_name: str,
    field: str,
    field_specifier: str,
    freq_df: pd.DataFrame,
    dvg_count: int,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_name = '',
    feature = 0,
    metric_name = '',
    metric: float = 0,
    lead_name = '',
    lead: float = 0
    ):
    '''

    '''
    freq0 = freq_df.columns[1]
    freq1 = freq_df.columns[2]

    df_plot = pd.melt(
        freq_df,
        id_vars=field,
        value_vars=[freq0, freq1],
        var_name='type',
        value_name='frequency'
    )

    custom_colors = {
        freq0: COLORS[4],
        freq1: COLORS[0]
    }

    plt.figure(figsize=(18, 6))
    sns.barplot(data=df_plot, x=field, y='frequency', hue='type', palette=custom_colors)

    plt.xticks(rotation=0)
    plt.xlabel(f'{field} ({field_specifier})')
    plt.ylabel('relative frequency (%)')

    title_name = f'{field} bar plot ({bar_name})'
    if feature_name:
        title_name += f' - {feature_name}: {feature}'
    if metric_name:
        title_name += f' - {metric_name}: {metric}'
    if lead_name:
        title_name += f' - {lead_name}: {lead}'
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
    plt.title(title_name)

    save_path = os.path.join(RESULTSPATH, field, data)
    save_path = os.path.join(save_path, strain)
    save_path = os.path.join(save_path, segment)
    save_path = os.path.join(save_path, intersects)
    if feature_name:
        save_path = os.path.join(save_path, f'{feature_name}_{feature}')
    os.makedirs(save_path, exist_ok=True)

    fname = f'{bar_name}_bar'
    fname += ".png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_metric_spine_plot(
    spine_name: str,
    field: str,
    field_specifier: str,
    metric_names: list,
    metrics: list,
    metric_specifier: str,
    dvg_count: int,
    data: str,
    strain: str,
    segment: str,
    intersects: str
    ):  
    '''

    '''
    plt.figure(figsize=(10, 6))
    num_points = len(metrics[0])
    x = np.arange(1, num_points + 1)
    x_smooth = np.linspace(x.min(), x.max(), 300)

    for i, y in enumerate(metrics):
        spline = CubicSpline(x, y)
        y_smooth = spline(x_smooth)
        color = COLORS[i % len(COLORS)]
        plt.plot(x_smooth, y_smooth, label=metric_names[i], color=color)
        plt.scatter(x, y, color=color, s=30)

    title_name = f'{field} spine plot ({spine_name})'
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
    plt.title(title_name)

    plt.xlabel(f"{field} ({field_specifier})")
    plt.ylabel(metric_specifier)
    plt.grid(True, alpha=0.3)

    plt.xticks(np.arange(1, num_points + 1))
    plt.legend(title='type')

    save_path = os.path.join(RESULTSPATH, field, data)
    save_path = os.path.join(save_path, strain)
    save_path = os.path.join(save_path, segment)
    save_path = os.path.join(save_path, intersects)
    os.makedirs(save_path, exist_ok=True)

    fname = f'{spine_name}_spine'
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_feature_distribution_plot(
    df: pd.DataFrame,
    feature_name: str,
    features_specifier: str,
    show_percentiles: bool,
    data: str,
    strain: str,
    segment: str,
    intersects: str
    ):
    '''

    '''
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    values = df[feature_name].dropna()
    n = len(values)

    mean_val = np.mean(values)
    var_val = np.var(values)
    skew_val = stats.skew(values)
    kurt_val = stats.kurtosis(values)

    sns.histplot(
        values,
        bins=30,
        kde=True,
        stat='percent',
        color=COLORS[4],
        edgecolor='white',
        label='distribution'
    )

    plt.axvline(mean_val, color=COLORS[2], linestyle='-', linewidth=1, label='mean')

    if show_percentiles:
        q90 = np.percentile(values, 90)
        q95 = np.percentile(values, 95)
        plt.axvline(q90, color=COLORS[1], linestyle='--', linewidth=1, label='90th percentile')
        plt.axvline(q95, color=COLORS[0], linestyle='--', linewidth=1, label='95th percentile')

    title = f'distribution of {feature_name}'
    title += f'\ndata: {data}'
    title += f', strain: {strain}'
    title += f', segment: {segment}'
    title += f', {intersects} intersects'
    title += f' (n={n})'
    plt.title(title)

    plt.xlabel(f'{feature_name} ({features_specifier})')
    plt.ylabel('relative frequency (%)')

    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.98, 0.95),
        borderaxespad=0.,
        title='type'
    )

    stats_text = (
        f'mean: {mean_val:.2f}\n'
        f'variance: {var_val:.2f}\n'
        f'skewness: {skew_val:.2f}\n'
        f'kurtosis: {kurt_val:.2f}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.gca().text(
        0.02, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=props
    )

    plt.ylim(0, 25)

    save_path = os.path.join(RESULTSPATH, 'distribution', data)
    save_path = os.path.join(save_path, strain)
    save_path = os.path.join(save_path, segment)
    save_path = os.path.join(save_path, intersects)
    os.makedirs(save_path, exist_ok=True)

    fname = f'{feature_name}_distribution'
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_feature_correlation_plot(
    df: pd.DataFrame,
    feature_x: str,
    features_specifier_x: str,
    feature_y: str,
    features_specifier_y: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    show_regression: bool = True
    ):
    '''

    '''
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    values = df[[feature_x, feature_y]].dropna()
    n = len(values)

    sns.scatterplot(
        data=values,
        x=feature_x,
        y=feature_y,
        color=COLORS[4],
        edgecolor='white',
        s=50,
        alpha=0.7,
        label='data points'
    )

    if show_regression:
        reg_line = sns.regplot(
            data=values,
            x=feature_x,
            y=feature_y,
            scatter=False,
            color=COLORS[0]
        )
        reg_line.lines[0].set_label('regression line')

    title = f'correlation between {feature_x} and {feature_y}'
    title += f'\ndata: {data}'
    title += f', strain: {strain}'
    title += f', segment: {segment}'
    title += f', {intersects} intersects'
    title += f' (n={n})'
    plt.title(title)

    plt.xlabel(f'{feature_x} ({features_specifier_x})')
    plt.ylabel(f'{feature_y} ({features_specifier_y})')

    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.98, 0.95),
        borderaxespad=0.,
        title='legend'
    )

    save_path = os.path.join(RESULTSPATH, 'correlation', data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    fname = f'{feature_x}_vs_{feature_y}_correlation.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

############
### site ###
############

### site ###

def make_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, dvg_count: int, motif_length: int, top_n: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    full_seq_motif_freq_df = compute_full_seq_motif_freq_df(motif_length, data, strain, segment)
    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    mean_freq_diffs = []
    top_freq_diffs = []
    for i, motif_site in zip(range(0, 4), motif_sites):
        motif_count_df = compute_feature_count_df(df, motif_site, motifs)
        motif_freq_df = compute_feature_freq_df(motif_count_df, motif_sites[i])
        motif_freq_df = rename_feature(motif_freq_df, motif_site, 'motif')
        comb_freq_df = subtract_freq_dfs('site', motif_freq_df, 'full_sequence', full_seq_motif_freq_df, 'motif')

        comb_freq_df = comb_freq_df.sort_values(by='difference', ascending=False).head(top_n)
        mean_freq_diff = round(comb_freq_df['difference'].mean(), 5)
        mean_freq_diffs.append(mean_freq_diff)
        top_freq_diff = round(comb_freq_df['difference'].iloc[0], 5)
        top_freq_diffs.append(top_freq_diff)
        filtered_motifs = comb_freq_df['motif'].tolist()

        create_freq_bar_plot(heatmap_names[i], 'motif', f'top {top_n}', comb_freq_df, dvg_count, data, strain, segment, intersects, 'motif_length', motif_length, 'mean frequency difference', mean_freq_diff, 'top frequency difference', top_freq_diff)

        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_site, filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
        create_freq_heatmap_plot(heatmap_names[i], 'motif', f'top {top_n}', motif_freq_heatmap_df, motif_count_df, percentile_rank_count_df, dvg_count, data, strain, segment, intersects, 'motif_length', motif_length, 'mean frequency difference', mean_freq_diffs[i], 'top frequency difference', top_freq_diffs[i])

    return mean_freq_diffs, top_freq_diffs

def make_reg_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, dvg_count: int, motif_length: int, top_n: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    mean_coefficients = []
    top_coefficients = []
    for i, motif_site in zip(range(0, 4), motif_sites):
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_site, motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
        reg_motif_freq_heatmap_df = add_lin_reg_rows(motif_freq_heatmap_df)

        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_df.T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(top_n)
        mean_coefficient = round(trans_reg_motif_freq_heatmap_df["coefficient"].mean(), 5)
        mean_coefficients.append(mean_coefficient)
        top_coefficient = round(trans_reg_motif_freq_heatmap_df["coefficient"].iloc[0], 5)
        top_coefficients.append(top_coefficient)
        filtered_motifs = trans_reg_motif_freq_heatmap_df.index.tolist()

        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_site, filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
        create_freq_heatmap_plot(heatmap_names[i], 'motif', f'top {top_n}', motif_freq_heatmap_df, motif_count_df, percentile_rank_count_df, dvg_count, data, strain, segment, intersects, 'motif_length', motif_length, 'mean regression coefficient', mean_coefficient, 'top regression coefficient', top_coefficient)
    
    return mean_coefficients, top_coefficients

###########
### dvg ###
###########

### dvg ###

def make_dvg_motif_freq_analysis(df: pd.DataFrame, plot_name: str, dvg_count: int, motif_length: int, top_n: int, data: str, strain: str, segment: str, intersects: str):
    '''
    
    '''
    motifs = generate_motifs(motif_length)

    print(df["NGS_percentile_rank"].dtype)
    high_df = filter_for_feature(df, 'NGS_percentile_rank', 10)
    high_dvg_count = high_df.shape[0]
    for _, row in high_df.iterrows():
        high_seq_motif_count_df = compute_seq_feature_count_df(row['dvg_sequence'], 'motif', motifs)
        high_motif_freq_df = compute_feature_freq_df(high_seq_motif_count_df, 'motif')

    df = df.sample(n=high_dvg_count, random_state=42)
    for _, row in df.iterrows():
        seq_motif_count_df = compute_seq_feature_count_df(row['dvg_sequence'], 'motif', motifs)
        motif_freq_df = compute_feature_freq_df(seq_motif_count_df, 'motif')

    comb_freq_df = subtract_freq_dfs('high_NGS_rank', high_motif_freq_df, 'all_NGS_ranks', motif_freq_df, 'motif')
    comb_freq_df = comb_freq_df.sort_values(by='difference', ascending=False).head(top_n)
    mean_freq_diff = round(comb_freq_df['difference'].mean(), 5)
    top_freq_diff = round(comb_freq_df['difference'].iloc[0], 5)
        
    create_freq_bar_plot(plot_name, 'motif', f'top {top_n}', comb_freq_df, dvg_count, data, strain, segment, intersects, 'motif_length', motif_length, 'mean frequency difference', mean_freq_diff, 'top frequency difference', top_freq_diff)

    return mean_freq_diff, top_freq_diff

def make_repeat_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, dvg_count: int, cap_length: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = add_direct_repeat_len(df)
    df = cap_direct_repeat_len(df, cap_length)

    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    repeat_count_heatmap_df = compute_feature_count_heatmap_df(df, 'direct_repeat_length')
    repeat_count_df = compute_feature_count_heatmap_sum_df(repeat_count_heatmap_df, 'direct_repeat_length')
    repeat_freq_heatmap_df = compute_feature_freq_heatmap_df(repeat_count_heatmap_df, percentile_rank_count_df)

    create_freq_heatmap_plot(heatmap_name, 'direct_repeat_length', 'nucleotides', repeat_freq_heatmap_df, repeat_count_df, percentile_rank_count_df, dvg_count, data, strain, segment, intersects)

def make_mfe_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, dvg_count: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = add_feature_percentile_rank(df, 'MFE', 'MFE_percentile_rank')
    df = rename_feature(df, 'MFE_percentile_rank', 'MFE_rank')

    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    mfe_count_heatmap_df = compute_feature_count_heatmap_df(df, 'MFE_rank')
    mfe_count_df = compute_feature_count_heatmap_sum_df(mfe_count_heatmap_df, 'MFE_rank')
    mfe_freq_heatmap_df = compute_feature_freq_heatmap_df(mfe_count_heatmap_df, percentile_rank_count_df)

    create_freq_heatmap_plot(heatmap_name, 'MFE_rank', 'percentile', mfe_freq_heatmap_df, mfe_count_df, percentile_rank_count_df, dvg_count, data, strain, segment, intersects)


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    ### DATASETS SINGLE ###

    # folder = 'datasets'
    # subfolder = 'sec'

    # data = 'Alnaji2021'
    # strain = DATASET_STRAIN_DICT[data]
    # segment = 'all'
    # intersects = 'all'

    # dfnames = [data]
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)
    
    ### DATASETS MULTI ###

    folder = 'datasets'
    subfolder = 'pseudo_pri_full'

    data = 'IAV'
    strain = 'PR8'
    segment = 'PB1'
    intersects = 'mean'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    run_site_motif_heatmap_analysis(dfs, 5, data, strain, segment, intersects)
    # run_reg_site_motif_heatmap_analysis(dfs, 5, data, strain, segment, intersects)
    # run_repeat_heatmap_analysis(dfs, 5, data, strain, segment, intersects)

    # run_dvg_motif_freq_analysis(dfs, 20, data, strain, segment, intersects)
    # run_ngs_count_distribution_analysis(dfs, data, strain, segment, intersects)
    # run_sec_structure_plots(dfs, data, strain, segment)
    run_ngs_count_dvg_pri_features_correlation_analysis(dfs, data, strain, segment, intersects)

    ### requires sec feature preprocessing ###
    
    # run_mfe_heatmap_analysis(dfs, data, strain, segment, intersects)
    # run_dvg_length_mfe_correlation_analysis(dfs, data, strain, segment, intersects)
    # run_ngs_count_dvg_sec_features_correlation_analysis(dfs, data, strain, segment, intersects)
    # run_ngs_count_site_sec_features_correlation_analysis(dfs, data, strain, segment, intersects)