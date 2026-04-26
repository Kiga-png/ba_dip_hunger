'''
    visualization for preprocessed datasets
'''

import os
import sys
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

from scipy import stats
from scipy.interpolate import CubicSpline

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score

import RNA

sys.path.insert(0, "..")

from utils import get_dataset_names

from utils import load_all_preprocessed
from utils import manage_separate_specifiers, clean_data_string, capitalize_first, get_official_strain_name
from utils import rename_feature, split_by_threshold, add_feature_quantile_rank, get_feature_modification_name
from utils import add_ikey, add_metadata_ikey, add_intersect_ngs_features, remove_by_ngs_cutoff

from utils import make_candidate_descriptor, make_pseudo_candidate_descriptor, make_legend_descriptor, pick_colors
from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df
from utils import compute_quantile_rank_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_count_heatmap_sum_df, compute_feature_freq_heatmap_df

from utils import p_to_stars, fisher_exact_for_category, bh_fdr

from utils import SHOW_LEGEND, SHOW_TITLE, LIM_AXES
from utils import RESULTSPATH, SEED, DATASET_CUTOFF, PSEUDO_DATASETS, DATASET_STRAIN_DICT, CUTOFF, STRAINS, SEGMENTS, DATASETS
from utils import COLORS, RANK_THRESHOLD, DECIMALS, TOP_N, K_MER_LENGTH, PALINDROMIC_K_MER_LENGTH, MIN_TRACT_LENGTH, DIRECT_REPEAT_LENGTH_CAP, MAX_MOTIF_LENGTH

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'visuals')


###############
### scripts ###
###############

### special ###

def run_site_motif_heatmap_analysis_len3_by_segment(
    dfs: list,
    top_n: int,
    folder: str,
    data: str,
    strain: str,
    intersects: str
):
    '''
        Run motif-site heatmap analysis for motif length 3 separately for each segment.
        :param dfs: list of input dataframes
        :param top_n: number of top motifs to retain per motif site
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param intersects: intersect mode descriptor

        :return: dictionary mapping each segment to summary statistics of the analysis
    '''
    # fix motif length to 3 and define readable names for the four motif sites
    motif_length = 3
    plot_names = ['motif before deletion', 'deletion start motif', 'deletion end motif', 'motif after deletion']

    # select data for the requested data/strain combination without restricting segment
    dfs = manage_separate_specifiers(dfs, data, strain, segment="all")

    # add intersect-based features and NGS quantile ranks
    df = add_intersect_ngs_features(dfs, intersects)
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')

    # compute motif windows once for the whole dataframe
    df_copy = df.copy()
    motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
    print(f'{skipped_count} candidates have been left out (motif_length=3)')

    # ensure segment information is available for splitting
    if 'segment' not in motif_df.columns:
        raise KeyError("Expected column 'segment' in motif_df, but it was not found.")

    results = {}

    # split dataframe by segment and run the heatmap workflow per segment
    segments = [s for s in pd.unique(motif_df['segment']) if pd.notna(s)]
    segments = sorted(segments, key=str)

    for seg in segments:
        seg_df = motif_df.loc[motif_df['segment'] == seg].copy()
        delvg_count = seg_df.shape[0]

        # skip empty segment subsets
        if delvg_count == 0:
            continue

        # run detailed motif heatmap analysis for the current segment
        mean_freq_diffs, top_freq_diffs = make_site_motif_heatmap_analysis(
            seg_df,
            plot_names,
            delvg_count,
            motif_length,
            top_n,
            folder,
            data,
            strain,
            seg,
            intersects
        )

        # store summary results for the current segment
        results[seg] = {
            "mean_freq_diffs": mean_freq_diffs,
            "top_freq_diffs": top_freq_diffs,
            "delvg_count": delvg_count,
            "skipped_count": skipped_count
        }

        print(f'segment {seg}: motif_length 3 completed (n={delvg_count})')

    return results

def run_site_motif_len3_per_segment(
    dfs: list,
    top_n: int,
    folder: str,
    data: str,
    strain: str,
    intersects: str
):
    '''
        Run motif-site frequency bar-plot analysis for motif length 3 separately for each segment.
        :param dfs: list of input dataframes
        :param top_n: number of top motifs to retain per motif site
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # fix motif length to 3 and define readable names for the four motif sites
    motif_length = 3
    plot_names = [
        'motif before deletion',
        'deletion start motif',
        'deletion end motif',
        'motif after deletion'
    ]

    # select data for the requested data/strain combination without restricting segment
    dfs = manage_separate_specifiers(dfs, data, strain, segment="all")

    # add intersect-based features and NGS quantile ranks
    df = add_intersect_ngs_features(dfs, intersects)
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')

    # compute motif windows once for the whole dataframe
    df_copy = df.copy()
    motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
    print(f'{skipped_count} candidates have been left out (motif_length=3)')

    # ensure segment information is available for splitting
    if 'segment' not in motif_df.columns:
        raise KeyError("Column 'segment' not found in motif_df")

    segments = sorted(pd.unique(motif_df['segment']))

    # split dataframe by segment and create bar plots per segment
    for seg in segments:
        seg_df = motif_df[motif_df['segment'] == seg].copy()
        delvg_count = seg_df.shape[0]

        # skip empty segment subsets
        if delvg_count == 0:
            continue

        print(f'Processing segment {seg} (n={delvg_count})')

        # create one bar plot for each of the four motif sites
        for i in range(4):
            site_counter = i + 1

            bar_name = f'frequency comparison of {plot_names[i]}'
            bar_name += f' (length={motif_length})'
            bar_name += f' for NGS ranks with Fisher’s exact test (BH-FDR)'
            bar_name += f' - bar plot'

            create_freq_bar_plot(
                bar_name,
                seg_df,
                f'site{site_counter}_motif',
                f'{plot_names[i]} (top {min(4 ** motif_length, top_n)})',
                delvg_count,
                top_n,
                f'site{site_counter}_motif',
                folder,
                'bar',
                data,
                strain,
                seg,
                intersects,
                f'motif_length_{motif_length}'
            )

        print(f'Segment {seg} completed (motif_length=3)')

### heatmap ###

def run_site_motif_heatmap_analysis(dfs: list, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the motif-site heatmap workflow across multiple motif lengths and summarize the results with spline plots.
        :param dfs: list of input dataframes
        :param top_n: number of top motifs to retain per motif site
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # define the maximum motif length to evaluate
    max_motif_length = MAX_MOTIF_LENGTH

    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # add NGS quantile ranks used for motif heatmap analyses
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    plot_names = ['motif before deletion', 'deletion start motif', 'deletion end motif', 'motif after deletion']

    # initialize containers for summary statistics across motif lengths
    all_mean_freq_diffs = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_freq_diffs  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        # add site motifs of current length and track skipped candidates
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        delvg_count = df_copy.shape[0]

        # run detailed site-motif analysis for current motif length
        mean_freq_diffs, top_freq_diffs = make_site_motif_heatmap_analysis(motif_df, plot_names, delvg_count, motif_length, top_n, folder, data, strain, segment, intersects)

        # store mean and top frequency differences for later spline plotting
        for i in range(0, 4):
            all_mean_freq_diffs[i][motif_length - 1] = mean_freq_diffs[i]
            all_top_freq_diffs[i][motif_length - 1] = top_freq_diffs[i]

        print(f'motif_length {motif_length} completed')

    # create spline summary plot for mean frequency differences across motif lengths
    create_spline_plot(f'mean frequency difference (site - sequence) of motif lengths - spline plot', 'length of motifs (nts)', 'mean frequency difference (%)', plot_names, all_mean_freq_diffs, delvg_count, 'site_mean_freq_diff', folder, 'spline', data, strain, segment, intersects)

def run_reg_site_motif_heatmap_analysis(dfs: list, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the regression-style motif-site heatmap workflow across multiple motif lengths and summarize the results with spline plots.
        :param dfs: list of input dataframes
        :param top_n: number of top motifs to retain per motif site
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # define the maximum motif length to evaluate
    max_motif_length = MAX_MOTIF_LENGTH

    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # add NGS quantile ranks used for motif heatmap analyses
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    plot_names = ['motif before deletion', 'deletion start motif', 'deletion end motif', 'motif after deletion']

    # initialize containers for coefficient summaries across motif lengths
    all_mean_coefficients = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_coefficients  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        # add site motifs of current length and track skipped candidates
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        delvg_count = df_copy.shape[0]

        # run regression-style site-motif analysis for current motif length
        mean_coefficients, top_coefficients = make_reg_site_motif_heatmap_analysis(motif_df, plot_names, delvg_count, motif_length, top_n, folder, data, strain, segment, intersects)

        # store mean and top coefficients for later spline plotting
        for i in range(0, 4):
            all_mean_coefficients[i][motif_length - 1] = mean_coefficients[i]
            all_top_coefficients[i][motif_length - 1] = top_coefficients[i]

        print(f'motif_length {motif_length} completed')

    # create spline summary plot for mean coefficients across motif lengths
    create_spline_plot(f'mean coefficient (regressive) for motif lengths - spline plot', 'length of motifs (nucleotides)', 'mean coefficient (-)', plot_names, all_mean_coefficients, delvg_count, 'site_mean_coefficients', folder, 'spline', data, strain, segment, intersects)

    # create spline summary plot for top coefficients across motif lengths
    create_spline_plot(f'top coefficient (regressive) for motif lengths - spline plot', 'length of motifs (nucleotides)', 'top coefficient (-)', plot_names, all_top_coefficients, delvg_count, 'site_top_coefficients', folder, 'spline', data, strain, segment, intersects)

def run_repeat_heatmap_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the quantile-wise direct-repeat heatmap analysis for a selected subset.
        :param dfs: list of input dataframes
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # add NGS quantile ranks used for heatmap analysis
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    delvg_count = df.shape[0]
    heatmap_name = 'direct repeat length'

    # create direct-repeat heatmap analysis
    make_repeat_heatmap_analysis(df, heatmap_name, delvg_count, folder, data, strain, segment, intersects)

def run_mfe_heatmap_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the quantile-wise MFE-rank heatmap analysis for a selected subset.
        :param dfs: list of input dataframes
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # add NGS quantile ranks used for heatmap analysis
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    delvg_count = df.shape[0]
    heatmap_name = 'MFE'

    # create MFE heatmap analysis
    make_mfe_heatmap_analysis(df, heatmap_name, delvg_count, folder, data, strain, segment, intersects)

### density and histo ###

def run_ngs_single_density_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str, show_quantiles: str, curvature: str):
    '''
        Run a single-density KDE analysis of NGS read counts for a selected subset.
        :param dfs: list of input dataframes
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param show_quantiles: whether selected quantile lines should be shown
        :param curvature: side on which to compute the curvature-based elbow

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # get display modifier for NGS read count axis label
    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    # create single KDE density plot of NGS read counts
    create_single_density_plot('NGS read count distribution via KDE with statistical moments - density plot', df, x_feature_name, x_axis_name, show_quantiles, 'single_ngs', folder, 'density', data, strain, segment, intersects, curvature)

def run_ngs_multi_density_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run a grouped multi-density KDE analysis of NGS read counts split by a selector feature.
        :param dfs: list of input dataframes
        :param selector: feature used to split the data into multiple density curves
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # split dataframe by selector values into separate sub-dataframes
    split_values = sorted(df[selector].dropna().unique().tolist())
    dfs = [df[df[selector] == val].copy() for val in split_values]
    df_names = [str(val) for val in split_values]

    # get display modifier for NGS read count axis label
    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    # create multi-density KDE plot across selector groups
    create_multi_density_plot(f'{selector}-wise NGS read count distribution via KDE - density plot', dfs, df_names, x_feature_name, x_axis_name, 'ngs_multi', "visuals", folder, 'density', data, strain, segment, intersects)

def run_intersect_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run an intersection analysis across selected datasets and create a grouped intersection bar plot.
        :param dfs: list of input dataframes
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset
    dfs = manage_separate_specifiers(dfs, data, strain, segment)

    feature_name = 'NGS_read_count'

    # optionally apply NGS cutoff encoded in the intersects string
    new_dfs = []

    for df in dfs:
        cutoff = None
        m = re.search(r'_(\d+)$', intersects)
        if m:
            cutoff = int(m.group(1))
            df = remove_by_ngs_cutoff(df, feature_name, cutoff)
        new_dfs.append(df)

    dfs = new_dfs

    # add intersection keys depending on whether metadata-based intersects are requested
    new_dfs = []
    for df in dfs:
        if 'metadata' in intersects:
            df = add_metadata_ikey(df)
        else:
            df = add_ikey(df)
        new_dfs.append(df)

    dfs = new_dfs
    df = pd.concat(dfs, ignore_index=True)
    
    # choose grouping feature for the intersection plot depending on selection granularity
    if (data != 'all') and (strain != 'all') and (segment != 'all'):
        print('use at leats one non-all specifier')
        return
    elif data == 'all':
        y_feature_name = 'subtype'
    elif strain == 'all':
        y_feature_name = 'strain'
    else:
        y_feature_name = 'segment'

    # create grouped intersection bar plot
    create_intersect_bar_plot(df, y_feature_name, 'intersects', folder, 'intersects', data, strain, segment, intersects)

def run_pooling_intersect_analysis(selector: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run an intersection comparison between pooled and unpooled preprocessing results.
        :param selector: feature used to group bars in the comparison plot
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # get dataset names for the selected data subset
    dfnames = get_dataset_names(DATASET_CUTOFF, data)

    # load and prepare pooled data
    poo_dfs = load_all_preprocessed(dfnames, 'pooled', 'primary')
    poo_dfs = manage_separate_specifiers(poo_dfs, data, strain, segment)
    poo_df = add_intersect_ngs_features(poo_dfs, intersects)
    poo_df = add_ikey(poo_df)
    print('loaded pooled')

    # load and prepare unpooled data
    unp_dfs = load_all_preprocessed(dfnames, 'unpooled', 'primary')
    unp_dfs = manage_separate_specifiers(unp_dfs, data, strain, segment)
    unp_df = add_intersect_ngs_features(unp_dfs, intersects)
    unp_df = add_ikey(unp_df)
    print('loaded unpooled')

    # create pooled vs unpooled intersection comparison plot
    create_pooling_intersect_bar_plot(poo_df, unp_df, selector, selector, folder, 'intersects', data, strain, segment, intersects)

def run_advandced_intersect_plot(dfs: list, selector: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run an advanced intersection comparison between a processed selection and raw unpooled data.
        :param dfs: list of processed input dataframes
        :param selector: feature used to group bars in the comparison plot
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict processed dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)
    df = add_ikey(df)

    # load corresponding raw unpooled data and add intersection keys
    dfs_raw = load_all_preprocessed(dfnames, 'unpooled', 'primary')
    df_raw = pd.concat(dfs_raw, ignore_index=True)
    df_raw = add_ikey(df_raw)
    print('loaded raw')

    # create advanced processed-vs-raw intersection comparison plot
    create_advandced_intersect_plot(df, df_raw, selector, 'intersects', folder, 'intersects', data, strain, segment, intersects)

### bar, violin, scatter ###

def run_delvg_pri_features_analysis(dfs: list, selector: str, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the primary DelVG feature analysis workflow, including Spearman heatmaps, scatter plots, bar plots and violin plots.
        :param dfs: list of input dataframes
        :param selector: grouping feature used for coloring or grouping plots
        :param top_n: number of top categories or motifs to retain where applicable
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # define primary numerical features and their plot labels
    num_features = [
        ("start", "deletion start position", "Deletion start (-)", "length"),
        ("end", "deletion end position", "Deletion end (-)", "length"),

        # ("full_seq_length", "full sequence length", "length of the full reference sequence (nucleotides)", "length"),
        ("DelVG_length", "DelVG sequence length", "DelVG length (nts)", "length"),
        ("deletion_length", "deletion length", "Deletion length (nts)", "length"),

        ("5_end_length", "5′ end length", "Retained 5′ end length (nts)", "length"),
        ("3_end_length", "3′ end length", "Retained 3′ end length (nts)", "length"),

        ("direct_repeat_length", f"direct repeat length (cap={DIRECT_REPEAT_LENGTH_CAP})", "length of direct repeat (nucleotides)", "repeats"),

        ("GC_content", "GC content", "G and C content (1/nt)", "composition"),
        ("AU_content", "AU content", "A and U content (1/nt)", "composition"),
        ("UpA_content", "UpA content", "UpA content (1/dnt)", "dinucleotides"),
        ("CpG_content", "CpG content", "CpG content (1/dnt)", "dinucleotides"),

        # ("GC_skew", "GC skew", "relative skew between G and C", "composition"),
        ("sequence_entropy", "sequence entropy", "Shannon entropy (-)", "entropy"),

        # ("poly_U_max_run", "poly-U maximum run length", "longest consecutive run of U (nucleotides)", "length"),
        # ("poly_U_tracts", f"poly-U tract (min_length={MIN_TRACT_LENGTH}) count", "number of U tracts (nucleotides)", "length"),
        # ("poly_A_max_run", "poly-A maximum run length", "longest consecutive run of A (nucleotides)", "length"),
        # ("poly_A_tracts", f"poly-A tract (min_length={MIN_TRACT_LENGTH}) count", "number of A tracts (nucleotides)", "length"),

        # ("palindrome_density", f"palindromic k-mer (k={PALINDROMIC_K_MER_LENGTH}) density", "density of palindromic k-mers (1/k-mer)", "motifs"),

        ("kmer_richness", f"k-mer (k={K_MER_LENGTH}) richness", "Unique k-mer content (1/k-mer)", "complexity"),
    ]

    # get display modifier for NGS read count axis labels
    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    # collect primary feature names for the Spearman heatmap
    num_feature_names = [f[0] for f in num_features]

    # create dataset-/strain-/segment-wise Spearman heatmap for primary features
    make_spearman_heatmap_analysis(df, folder, selector, num_feature_names, 'Primary', data, strain, segment, intersects)

    print(f'spearman made successfully')

    # create scatter plots for all numerical primary features
    subfolder = 'scatter'

    feature_index = 0
    for y_feature_name, y_feature_title, y_axis_name, category in num_features:
            placeholder = f'{y_feature_title} as a function of NGS read count - scatter plot'
            create_feature_scatter_plot('', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, False, 50, False, "", 0.0, 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
            feature_index += 1

    print(f'numerical features completed')

    # add NGS quantile ranks for subsequent categorical frequency comparisons
    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')

    # motifs 
    max_motif_length = 7
    plot_names = ['motif before deletion', 'deletion start motif', 'deletion end motif', 'motif after deletion']
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        delvg_count = df_copy.shape[0]
        site_counter = 0
        for i in range(0, 4):
            bar_name = f'frequency comparison of {plot_names[i]}'
            bar_name += f' (length={motif_length}) for NGS ranks with Fisher’s exact test (BH-FDR) - bar plot'
            site_counter = i + 1
            create_freq_bar_plot(bar_name, motif_df, f'site{site_counter}_motif', f'{plot_names[i]} (top {min(4 ** motif_length, top_n)})', delvg_count, top_n, f'site{site_counter}_motif', folder, 'bar', data, strain, segment, intersects, f'motif_lenght_{motif_length}')

        print(f'motif_length {motif_length} completed')

    # define bar-plot subfolder and total DelVG count
    subfolder = 'bar'

    delvg_count = df.shape[0]

    # define categorical features for frequency and distribution analyses
    cat_features = [
        ("direct_repeat_length", f"direct repeat length (cap={DIRECT_REPEAT_LENGTH_CAP})", "length of direct repeat (nucleotides)", "repeats"),

        ("host", "host species", "host species", "metadata"),
        ("system_type", "system type", "system type", "metadata"),

        ("library_layout", "library layout", "library layout", "metadata"),
        ("library_selection", "library selection", "library selection", "metadata"),
        ("library_source", "library source", "library source", "metadata"),
    ]

    # add additional metadata features available only for unpooled preprocessing
    if folder == "unpooled":
        cat_features.append(("time_point", "sampling time", "time point of sample collection", "metadata"))
        cat_features.append(("localization", "sample localization", "localization of sample", "metadata"))
        cat_features.append(("resolution", "sequencing resolution", "resolution of sequencing", "metadata"))
        cat_features.append(("cell_system", "host cell type", "cell type of host", "metadata"))
        cat_features.append(("MOI", "MOI", "MOI", "metadata"))

    # add virus-type feature if all data types are included
    if data == "all":
        cat_features.append(("type", "virus type", "type of virus", "metadata"))

    # add strain and subtype if all strains are included
    if strain == "all":
        cat_features.append(("strain", "virus strain", "strain of virus ", "metadata"))
        cat_features.append(("subtype", "virus subtype", "subtype of virus ", "metadata"))

    # add segment if all segments are included
    if segment == "all":
        cat_features.append(("segment", "genome segment", "segment of genome", "metadata"))

    # create bar plots comparing categorical feature frequencies across NGS-rank groups
    for y_feature_name, y_feature_title, y_axis_name, category in cat_features:
        create_freq_bar_plot(f'frequency comparison of {y_feature_title} for NGS ranks with Fisher’s exact test (BH-FDR) - bar plot', df, y_feature_name, y_axis_name, delvg_count, 0, y_feature_name, folder, subfolder, data, strain, segment, intersects)

    # prepare violin plots of NGS read count distributions by categorical feature
    y_feature_name = 'norm_log_NGS_read_count'
    y_axis_name = f'{modification} NGS count (reads)'

    subfolder = 'violin'

    # create violin plots for all categorical features
    for x_feature_name, x_feature_title, x_axis_name, category in cat_features:
        create_feature_violin_plot(f'NGS read count distributions of {x_feature_title} - violin plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, delvg_count, 0, x_feature_name, folder, subfolder, data, strain, segment, intersects)

    print(f'categorical features completed')

def run_delvg_sec_features_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the secondary-structure DelVG feature analysis workflow, including Spearman heatmaps and scatter plots.
        :param dfs: list of input dataframes
        :param selector: grouping feature used for coloring or grouping plots
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # define secondary-structure numerical features and their plot labels
    num_features = [
        ("MFE", "MFE", "MFE (kcal/mol)", "MFE"),

        ("bp_count", "base pair count", "Number of base pairs (-)", "pairing"),
        ("bp_density", "base pair density", "Base pair density (1/nt)", "pairing"),
        ("unpaired_count", "unpaired base count", "Number of unpaired bases (-)", "pairing"),
        ("unpaired_density", "unpaired base density", "Unpaired base density (1/nt)", "pairing"),

        ("stem_count", "stem count", "Number of stems (-)", "stem"),
        # ("stem_len_max", "stem length maximum", "maximum stem length (nucleotides)", "stem"),
        ("stem_len_mean", "stem length mean", "Mean stem length (nts)", "stem"),
        # ("stem_len_min", "stem length minimum", "minimum stem length (nucleotides)", "stem"),

        ("hairpin_count", "hairpin count", "Number of hairpin loops (-)", "hairpin"),
        ("hairpin_size_mean", "hairpin size mean", "Mean hairpin loop size (nts)", "hairpin"),
        # ("hairpin_size_min", "hairpin size minimum", "minimum hairpin loop size (nucleotides)", "hairpin"),
        # ("hairpin_size_max", "hairpin size maximum", "maximum hairpin loop size (nucleotides)", "hairpin"),

        # ("external_unpaired_density", "external unpaired base density", "unpaired base density in external loop (1/nucleotide)", "pairing"),

        ("pair_span_mean", "pair span mean", "Mean base pair span (nts)", "pair_span"),
        # ("pair_span_min", "pair span minimum", "minimum base pair span (nucleotides)", "pair_span"),
        # ("pair_span_max", "pair span maximum", "maximum base pair span (nucleotides)", "pair_span"),

        # ("free_5prime_len", "free 5′ length", "unpaired length at 5′ end (nts)", "length"),
        # ("free_3prime_len", "free 3′ length", "unpaired length at 3′ end (nts)", "length"),

        # ("max_symmetry", "max symmetry length", "max symmetry length (nucleotides)", "composition"),
    ]

    # get display modifier for NGS read count axis labels
    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    # collect secondary feature names for the Spearman heatmap
    num_feature_names = [f[0] for f in num_features]

    # create dataset-/strain-/segment-wise Spearman heatmap for secondary features
    make_spearman_heatmap_analysis(df, folder, selector, num_feature_names, 'Secondary', data, strain, segment, intersects)

    print(f'spearman made successfully')

    # create scatter plots for all numerical secondary features
    subfolder = 'scatter'

    feature_index = 0
    for y_feature_name, y_feature_title, y_axis_name, category in num_features:
            placeholder = f'{y_feature_title} as a function of NGS read count - scatter plot'
            create_feature_scatter_plot("", df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, False, 50, False, "", 0.0, 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
            feature_index += 1

    print(f'numerical features completed')

def run_delvg_hybrid_features_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run the hybrid feature analysis workflow combining sequence and structure-derived features.
        :param dfs: list of input dataframes
        :param selector: grouping feature used for coloring or grouping plots
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # define hybrid features and their plot labels
    num_features = [
        ("GC_overall", "GC overall", "overall GC content (1/nucleotide)", "composition"),
        ("GC_paired", "GC paired", "GC content among paired bases (1/nucleotide)", "composition"),
        ("GC_unpaired", "GC unpaired", "GC content among unpaired bases (1/nucleotide)", "composition"),

        ("pair_GC_count", "GC pair count", "number of GC base pairs", "pairing"),
        ("pair_AU_count", "AU pair count", "number of AU base pairs", "pairing"),
        ("pair_GU_count", "GU pair count", "number of GU base pairs", "pairing"),
        # ("pair_noncanon_count", "noncanonical pair count", "number of noncanonical base pairs", "pairing"),
        ("pair_GC_content", "GC pair content", "content of base pairs that are GC (1/pair)", "pairing"),
        ("pair_AU_content", "AU pair content", "content of base pairs that are AU (1/pair)", "pairing"),
        ("pair_GU_content", "GU pair content", "content of base pairs that are GU (1/pair)", "pairing"),
        # ("pair_noncanon_content", "noncanonical pair content", "content of base pairs that are noncanonical (1/pair)", "pairing"),

        ("stem_end_GC_content", "stem end GC content", "content of stem-end pairs that are GC (1/pair)", "stem"),
        ("stem_end_AU_content", "stem end AU content", "content of stem-end pairs that are AU (1/pair)", "stem"),
        # ("stem_end_GU_content", "stem end GU content", "content of stem-end pairs that are GU (1/pair)", "stem"),

        ("hairpin_close_GC_content", "hairpin closing GC content", "content of hairpin-closing pairs that are GC (1/pair)", "hairpin"),
        ("hairpin_close_AU_content", "hairpin closing AU content", "content of hairpin-closing pairs that are AU (1/pair)", "hairpin"),
        # ("hairpin_close_GU_content", "hairpin closing GU content", "content of hairpin-closing pairs that are GU (1/pair)", "hairpin"),
        # ("hairpin_close_noncanon_content", "hairpin closing noncanonical content", "content of hairpin-closing pairs that are noncanonical (1/pair)", "hairpin"),

        # ("motif_GNRA_count", "GNRA motif count", "count of GNRA tetraloops", "motif"),
        # ("motif_UNCG_count", "UNCG motif count", "count of UNCG tetraloops", "motif"),
        # ("motif_CUUG_count", "CUUG motif count", "count of CUUG tetraloops", "motif"),

        # ("AU_unpaired_content", "AU unpaired content", "AU content among all unpaired bases (1/nucleotide)", "composition"),
        # ("AU_hairpin_content", "AU hairpin content", "AU content in hairpin loops (1/nucleotide)", "composition"),
        # ("AU_internal_content", "AU internal content", "AU content in internal/multiloop/bulge (1/nucleotide)", "composition"),
        # ("AU_external_content", "AU external content", "AU content in external loop (1/nucleotide)", "composition"),

        # ("GC_5prime_tail", "GC 5′ tail content", "GC content in 5′ single-stranded tail (1/nucleotide)", "composition"),
        # ("GC_3prime_tail", "GC 3′ tail content", "GC content in 3′ single-stranded tail (1/nucleotide)", "composition"),

        # ("AUG_total", "AUG total count", "total number of AUG start codons", "codon"),
        # ("AUG_unpaired", "AUG unpaired count", "number of AUG start codons fully unpaired", "codon"),
        # ("AUG_unpaired_content", "AUG unpaired content", "content of AUG start codons fully unpaired (1/codon)", "codon"),
    ]

    # get display modifier for NGS read count axis labels
    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    # choose grouping feature for Spearman heatmap based on current selection level
    if data == 'all':
        y_feature_name = 'dataset'
        call_spearman = True
    elif strain == 'all':
        y_feature_name = 'strain'
        call_spearman = True
    else:
        y_feature_name = 'segment'
        call_spearman = True

    # collect hybrid feature names for the Spearman heatmap
    num_feature_names = [f[0] for f in num_features]

    # create Spearman heatmap if enabled
    if call_spearman:
        make_spearman_heatmap_analysis(df, folder, y_feature_name, num_feature_names, 'hybrid', data, strain, segment, intersects)

    print(f'spearman made successfully')

    # create scatter plots for all hybrid features
    subfolder = 'scatter'

    feature_index = 0
    for y_feature_name, y_feature_title, y_axis_name, category in num_features:
            create_feature_scatter_plot(f'{y_feature_title} as a function of NGS read count - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, 50, False, "", 0.0, 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
            feature_index += 1

    print(f'numerical features completed')

def run_length_mfe_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Run a focused scatter-plot analysis comparing DelVG length and MFE.
        :param dfs: list of input dataframes
        :param selector: grouping feature used for coloring the scatter plot
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # define x-axis feature and label
    x_feature_name = 'DelVG_length'
    x_axis_name = 'DelVG length (nts)'

    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    print(f'features added successfully')

    # define scatter-plot subfolder
    subfolder = 'scatter'

    # define features to compare against DelVG length
    features = [
        ("MFE", "MFE", "MFE (kcal/mol)", "other"),
    ]

    # create scatter plots for all selected features
    for y_feature_name, y_feature_title, y_axis_name, category in features:
            create_feature_scatter_plot('', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, False, 50, False, "", 0.0, 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)    
    print(f'numerical features completed')

### other ###

def run_sec_structure_plot(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Generate RNA secondary structure plots for the first DelVG sequences of a selected subset.
        :param dfs: list of input dataframes
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # restrict dataframes to the requested subset and add intersect-based features
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    # select only the first few DelVG sequences for plotting
    sequences = df['DelVG_sequence'].head(5)

    # create save directory for generated structure plots
    save_path = os.path.join(RESULTSPATH, folder, 'structure', data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # iterate over selected sequences and generate one RNAplot per sequence
    for idx, seq in enumerate(sequences):
        # skip invalid or empty sequence entries
        if not isinstance(seq, str) or not seq.strip():
            continue

        # normalize sequence format and compute secondary structure with MFE
        seq = seq.strip().upper()
        structure, mfe = RNA.fold(seq)

        # write sequence and dot-bracket structure to temporary input file for RNAplot
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
            f.write(f"{seq}\n{structure}\n")
            temp_input = f.name

        try:
            # call RNAplot to generate SVG structure visualization
            subprocess.run(['RNAplot', '-o', 'svg', '-t', '0', '-i', temp_input], check=True)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] RNAplot failed for sequence {idx+1}: {e}')
            continue

        # move generated SVG to final output path with MFE in file name
        svg_source = 'rna.svg'
        svg_target = os.path.join(save_path, f'structure_{idx+1}_mfe_{mfe:.2f}.svg')
        if os.path.exists(svg_source):
            os.rename(svg_source, svg_target)
        else:
            print(f'[WARNING] RNAplot output not found for sequence {idx+1}')

        # remove temporary RNAplot input file
        os.remove(temp_input)

###############
### general ###
###############

### heatmap ###

def make_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, delvg_count: int, motif_length: int, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Perform motif-site frequency difference analysis and create related bar and heatmap plots.
        :param df: dataframe containing motif site columns and NGS quantile ranks
        :param heatmap_names: list of readable names for the four motif sites
        :param delvg_count: total number of DelVGs used for title annotation
        :param motif_length: motif length to analyze
        :param top_n: number of top motifs to retain based on frequency difference
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: tuple of mean and top frequency differences for all four motif sites
    '''
    # define the four junction-related motif site columns
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    # prepare full motif vocabulary, reference frequencies and quantile-rank counts
    motifs = generate_motifs(motif_length)
    full_seq_motif_freq_df = compute_full_seq_motif_freq_df(motif_length, data, strain, segment)
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    # store summary statistics across the four motif sites
    mean_freq_diffs = []
    top_freq_diffs = []
    for i in range(0, 4):
        # compute motif counts and frequencies at the current site
        motif_count_df = compute_feature_count_df(df, motif_sites[i], motifs)
        motif_freq_df = compute_feature_freq_df(motif_count_df, motif_sites[i])
        motif_freq_df = rename_feature(motif_freq_df, motif_sites[i], 'motif')

        # compare site-specific frequencies with full-sequence background frequencies
        comb_freq_df = subtract_freq_dfs('site', motif_freq_df, 'full_sequence', full_seq_motif_freq_df, 'motif')

        # keep motifs with the largest positive frequency differences
        comb_freq_df = comb_freq_df.sort_values(by='difference', ascending=False).head(top_n)
        mean_freq_diff = round(comb_freq_df['difference'].mean(), DECIMALS)
        mean_freq_diffs.append(mean_freq_diff)
        top_freq_diff = round(comb_freq_df['difference'].iloc[0], DECIMALS)
        top_freq_diffs.append(top_freq_diff)
        filtered_motifs = comb_freq_df['motif'].tolist()

        # create bar plot of motif frequency differences
        bar_name = 'frequency comparison of '
        bar_name += heatmap_names[i]
        bar_name += f' (length={motif_length}) - bar plot'

        bar_name_add = ''
        # optional additional summary text could be added here

        create_freq_diff_plot(bar_name, bar_name_add, 'motif', f'{heatmap_names[i]} (top {min(4 ** motif_length, top_n)})', comb_freq_df, delvg_count, motif_sites[i], folder, 'diff', data, strain, segment, intersects, f'motif_length_{motif_length}')

        # create quantile-wise motif frequency heatmap for selected motifs
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, quantile_rank_count_df)

        heatmap_name = 'qunatile-wise frequency of '
        heatmap_name += heatmap_names[i]
        heatmap_name += f' (length={motif_length}) ordered by frequency difference - heatmap'

        heatmap_name_add = ''
        # optional additional summary text could be added here

        create_freq_heatmap_plot(heatmap_name, heatmap_name_add, 'motif', f'{heatmap_names[i]} (top {min(4 ** motif_length, top_n)})', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', motif_freq_heatmap_df, motif_count_df, quantile_rank_count_df, delvg_count, motif_sites[i], folder, 'heatmap', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_freq_diffs, top_freq_diffs

def make_reg_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, delvg_count: int, motif_length: int, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Perform motif-site regression-style heatmap analysis using absolute linear coefficients across quantile ranks.
        :param df: dataframe containing motif site columns and NGS quantile ranks
        :param heatmap_names: list of readable names for the four motif sites
        :param delvg_count: total number of DelVGs used for title annotation
        :param motif_length: motif length to analyze
        :param top_n: number of top motifs to retain based on coefficient magnitude
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: tuple of mean and top absolute coefficients for all four motif sites
    '''
    # define the four junction-related motif site columns
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    # prepare full motif vocabulary and quantile-rank counts
    motifs = generate_motifs(motif_length)
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    # store summary coefficient statistics across the four motif sites
    mean_coefficients = []
    top_coefficients = []
    for i in range(0, 4):
        # compute quantile-wise motif frequencies for all motifs
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, quantile_rank_count_df)

        # add linear regression intercepts and coefficients per motif
        reg_motif_freq_heatmap_df = add_lin_reg_rows(motif_freq_heatmap_df)

        # rank motifs by absolute slope magnitude
        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_df.T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(top_n)
        mean_coefficient = round(trans_reg_motif_freq_heatmap_df["coefficient"].mean(), DECIMALS)
        mean_coefficients.append(mean_coefficient)
        top_coefficient = round(trans_reg_motif_freq_heatmap_df["coefficient"].iloc[0], DECIMALS)
        top_coefficients.append(top_coefficient)
        filtered_motifs = trans_reg_motif_freq_heatmap_df.index.tolist()

        # recompute frequency heatmap using only the selected motifs
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, quantile_rank_count_df)

        heatmap_name = 'qunatile-wise frequency of '
        heatmap_name += heatmap_names[i]
        heatmap_name += f' (length={motif_length}) ordered by coefficient - heatmap'

        heatmap_name_add = ''
        # optional additional summary text could be added here

        create_freq_heatmap_plot(heatmap_name, heatmap_name_add, 'motif', f'{heatmap_names[i]} (top {min(4 ** motif_length, top_n)})', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', motif_freq_heatmap_df, motif_count_df, quantile_rank_count_df, delvg_count, 'reg_' + motif_sites[i], folder, 'heatmap', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_coefficients, top_coefficients

def make_repeat_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, delvg_count: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Create a quantile-wise heatmap analysis for direct repeat lengths.
        :param df: dataframe containing direct repeat lengths and NGS quantile ranks
        :param heatmap_name: readable feature name for the plot title
        :param delvg_count: total number of DelVGs used for title annotation
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # compute counts per NGS quantile rank
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    # compute direct repeat length counts and frequencies across quantile ranks
    repeat_count_heatmap_df = compute_feature_count_heatmap_df(df, 'direct_repeat_length')
    repeat_count_df = compute_feature_count_heatmap_sum_df(repeat_count_heatmap_df, 'direct_repeat_length')
    repeat_freq_heatmap_df = compute_feature_freq_heatmap_df(repeat_count_heatmap_df, quantile_rank_count_df)

    # build heatmap title
    heatmap_name = 'qunatile-wise frequency of '
    heatmap_name += heatmap_name
    heatmap_name += f' (cap={DIRECT_REPEAT_LENGTH_CAP}) - heatmap '

    # create heatmap plot
    create_freq_heatmap_plot(heatmap_name, '', 'direct_repeat_length', 'length of direct repeat (nucleotides)', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', repeat_freq_heatmap_df, repeat_count_df, quantile_rank_count_df, delvg_count, 'direct_repeat_length', folder, 'heatmap', data, strain, segment, intersects)

def make_mfe_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, delvg_count: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''
        Create a quantile-wise heatmap analysis for MFE quantile ranks.
        :param df: dataframe containing MFE values and NGS quantile ranks
        :param heatmap_name: readable feature name for the plot title
        :param delvg_count: total number of DelVGs used for title annotation
        :param folder: main result folder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # create MFE-based quantile ranks and rename for plotting
    df = add_feature_quantile_rank(df, 'MFE', 'MFE_quantile_rank')
    df = rename_feature(df, 'MFE_quantile_rank', 'MFE_rank')

    # compute counts per NGS quantile rank
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    # compute MFE-rank counts and frequencies across quantile ranks
    mfe_count_heatmap_df = compute_feature_count_heatmap_df(df, 'MFE_rank')
    mfe_count_df = compute_feature_count_heatmap_sum_df(mfe_count_heatmap_df, 'MFE_rank')
    mfe_freq_heatmap_df = compute_feature_freq_heatmap_df(mfe_count_heatmap_df, quantile_rank_count_df)

    # build heatmap title
    heatmap_name = 'qunatile-wise frequency of '
    heatmap_name += heatmap_name
    heatmap_name += f' rank (quantile) - heatmap '

    # create heatmap plot
    create_freq_heatmap_plot(heatmap_name, '', 'MFE_rank', 'MFE rank (quantile)', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', mfe_freq_heatmap_df, mfe_count_df, quantile_rank_count_df, delvg_count, 'mfe', folder, 'heatmap', data, strain, segment, intersects)

def make_spearman_heatmap_analysis(
    df: pd.DataFrame,
    folder: str,
    y_feature_name: str,
    x_features: list,
    fname: str,
    data: str = 'all',
    strain: str = 'all',
    segment: str = 'all',
    intersects: str = 'all'
    ):
    '''
        Compute group-wise Spearman correlations between NGS read count and selected features, apply BH-FDR correction, and create a heatmap.
        :param df: input dataframe containing selected features and norm_log_NGS_read_count
        :param folder: main result folder
        :param y_feature_name: grouping feature defining heatmap rows
        :param x_features: list of feature names to correlate with NGS read count
        :param fname: feature-group name used for file naming and axis labels
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # total number of DelVGs used for title annotation
    delvg_count = df.shape[0]

    # compute row counts for y-axis labels
    rank_count_df = (
        df.groupby(y_feature_name, dropna=False)
          .size()
          .reset_index(name='count')
    )

    # compute non-missing counts for all requested x features
    count_df = pd.DataFrame({
        'feature': x_features,
        'count': [int(df[f].notna().sum()) if f in df.columns else 0 for f in x_features]
    })

    # preserve original y-group order
    y_order = list(rank_count_df[y_feature_name].to_numpy())

    # initialize matrices for correlation coefficients and p-values
    rho_mat = pd.DataFrame(index=y_order, columns=x_features, dtype=float)
    p_mat   = pd.DataFrame(index=y_order, columns=x_features, dtype=float)

    for y_val in y_order:
        # subset dataframe to one y-group
        df_sub = df.loc[df[y_feature_name] == y_val].copy()

        for feat in x_features:
            # handle missing feature columns
            if feat not in df_sub.columns:
                rho_mat.loc[y_val, feat] = np.nan
                p_mat.loc[y_val, feat] = np.nan
                continue

            x = df_sub[feat]
            y = df_sub['norm_log_NGS_read_count']

            # require enough paired observations for Spearman correlation
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 3:
                rho_mat.loc[y_val, feat] = np.nan
                p_mat.loc[y_val, feat] = np.nan
                continue

            # compute Spearman correlation and p-value
            rho, pval = spearmanr(x[mask].to_numpy(), y[mask].to_numpy())
            rho_mat.loc[y_val, feat] = float(rho) if rho is not None else np.nan
            p_mat.loc[y_val, feat] = float(pval) if pval is not None else np.nan

    # apply BH-FDR correction across all valid tests
    pvals = p_mat.to_numpy().ravel()
    valid_mask = np.isfinite(pvals)

    adj_pvals = np.full_like(pvals, np.nan, dtype=float)
    if int(valid_mask.sum()) > 0:
        adj_pvals_valid = bh_fdr(pvals[valid_mask])
        adj_pvals[valid_mask] = adj_pvals_valid

    adj_p_mat = pd.DataFrame(
        adj_pvals.reshape(p_mat.shape),
        index=p_mat.index,
        columns=p_mat.columns
    )

    # combine rounded rho values with significance stars for heatmap annotations
    def format_cell(rho, pval):
        if (rho is None) or (pval is None):
            return ''
        if (not np.isfinite(rho)) or (not np.isfinite(pval)):
            return ''
        rho_str = f'{rho:.{DECIMALS}f}'
        star_str = p_to_stars(pval)
        return f'{rho_str} [{star_str}]'

    freq_heatmap_df = pd.DataFrame(index=y_order, columns=x_features, dtype=object)
    for y_val in y_order:
        for feat in x_features:
            freq_heatmap_df.loc[y_val, feat] = format_cell(
                rho_mat.loc[y_val, feat],
                adj_p_mat.loc[y_val, feat]
            )

    def _stars_to_score(stars: str) -> int:
        '''
            Convert a significance star string to an integer score.
            :param stars: significance label such as '*', '**', '***', 'ns' or 'NA'

            :return: integer score based on the number of stars
        '''
        if stars is None:
            return 0
        s = str(stars).strip()
        if s == '' or s.lower() in ['na', 'nan']:
            return 0
        # count only star characters
        return int(s.count('*'))

    # compute total significance score per feature column
    star_sums = {}
    for feat in x_features:
        total = 0
        if feat in adj_p_mat.columns:
            col = adj_p_mat[feat]
            for pval in col.to_numpy():
                if pval is None or (not np.isfinite(pval)):
                    continue
                total += _stars_to_score(p_to_stars(float(pval)))
        star_sums[feat] = int(total)

    # sort features by decreasing total significance and preserve original order on ties
    orig_idx = {feat: i for i, feat in enumerate(x_features)}
    x_features_sorted = sorted(
        list(x_features),
        key=lambda f: (-star_sums.get(f, 0), orig_idx.get(f, 10**9))
    )

    # apply sorted feature order consistently to all matrices
    rho_mat = rho_mat.reindex(columns=x_features_sorted)
    p_mat = p_mat.reindex(columns=x_features_sorted)
    adj_p_mat = adj_p_mat.reindex(columns=x_features_sorted)
    freq_heatmap_df = freq_heatmap_df.reindex(columns=x_features_sorted)

    count_df = count_df.set_index('feature').reindex(x_features_sorted).reset_index()

    # update x_features variable for downstream usage
    x_features = x_features_sorted

    # optional plot subtitle for primary feature settings
    if fname == 'primary':
        plot_name_add = f'\nk-mer (k={K_MER_LENGTH}, k_palindromic={PALINDROMIC_K_MER_LENGTH}), tracts (min_length={MIN_TRACT_LENGTH})'
    else:
        plot_name_add = ''

    # build heatmap title
    heatmap_name = f'{make_legend_descriptor(y_feature_name)}-wise Spearman correlations (NGS read count and {fname} feature) ordered by significance (BH-FDR) - heatmap'

    # create final heatmap plot
    create_freq_heatmap_plot(
        heatmap_name,
        plot_name_add,
        'feature',
        f'{fname} feature',
        y_feature_name,
        capitalize_first(y_feature_name),
        'Spearman’s ρ',
        freq_heatmap_df,
        count_df,
        rank_count_df,
        delvg_count,
        fname,
        folder,
        'heatmap',
        data,
        strain,
        segment,
        intersects,
        '',
        False
    )

### plot ###

def create_freq_heatmap_plot(
    plot_name: str,
    plot_name_add: str,
    x_feature_name: str,
    x_axis_name: str,
    y_feature_name: str,
    y_axis_name: str,
    scale_axis_name: str,
    freq_heatmap_df: pd.DataFrame,
    count_df: pd.DataFrame,
    rank_count_df: pd.DataFrame,
    delvg_count: int,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_specifier: str = '',
    show_x_counts: bool = True
    ):  
    '''
        Create a heatmap plot for feature frequencies or related statistics across grouped categories.
        :param plot_name: plot title prefix
        :param plot_name_add: additional title suffix
        :param x_feature_name: feature represented on the x-axis
        :param x_axis_name: x-axis label
        :param y_feature_name: feature represented on the y-axis
        :param y_axis_name: y-axis label
        :param scale_axis_name: colorbar label
        :param freq_heatmap_df: dataframe containing heatmap values with rows as y categories and columns as x categories
        :param count_df: dataframe containing total counts per x category
        :param rank_count_df: dataframe containing total counts per y category
        :param delvg_count: total number of DelVGs used for title annotation
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param feature_specifier: optional additional subfolder name for saving
        :param show_x_counts: if True, append x-category counts to x tick labels

        :return: None
    '''
    # initialize heatmap dataframe and preserve original x/y order
    pivot_df = freq_heatmap_df
    feature_order = list(pivot_df.columns)
    y_order = list(pivot_df.index)

    # determine row ordering for known grouping types
    if y_feature_name == "segment":
        present = list(pd.unique(y_order))
        ordered = [s for s in SEGMENTS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        y_order = ordered + remaining
    elif y_feature_name == "strain":
        present = list(pd.unique(y_order))
        ordered = [s for s in STRAINS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        y_order = ordered + remaining
    elif y_feature_name == "dataset":
        present = list(pd.unique(y_order))
        ordered = [s for s in DATASETS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        y_order = ordered + remaining
    elif "motif" in y_feature_name:
        y_order = list(pd.unique(y_order))
    else:
        y_order = sorted(pd.unique(y_order))

    # apply row order to heatmap dataframe
    pivot_df = pivot_df.reindex(y_order)

    # create quick lookup of x-category counts
    feature_count_series = count_df.set_index(x_feature_name)['count']

    # optionally append counts to x tick labels
    if show_x_counts:
        feature_labels = [
            f'{capitalize_first(feature)}\n(n={feature_count_series.get(feature, 0)})'
            for feature in feature_order
        ]
    else:
        feature_labels = [
            capitalize_first(feature)
            for feature in feature_order
        ]

    # create y tick labels with corresponding group counts
    y_labels = []
    for y_val in y_order:
        count = rank_count_df.loc[
            rank_count_df[y_feature_name] == y_val, 'count'
        ].values
        count_str = str(int(count[0])) if len(count) > 0 else "0"
        y_labels.append(f"{y_val}\n(n={count_str})")

    # distinguish between purely numeric heatmaps and mixed string/star annotation cases
    if np.issubdtype(pivot_df.to_numpy().dtype, np.number):
        heatmap_values = pivot_df
        annot_values = None
        is_star_case = False
    else:
        # extract numeric leading part for coloring while preserving full string annotations
        num_part = pivot_df.astype(str).replace(
            to_replace=r'^\s*([+-]?\d+(?:\.\d+)?)\s*.*$',
            value=r'\1',
            regex=True
        )
        heatmap_values = num_part.apply(pd.to_numeric, errors='coerce')

        def format_annot(cell):
            # format annotation text by splitting numeric value and remaining marker text
            if pd.isna(cell):
                return ''
            s = str(cell).strip()
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)(.*)$', s)
            if m is None:
                return s
            num = m.group(1)
            rest = m.group(2).strip()
            if rest:
                return f"{num}\n{rest}"
            return num

        annot_values = pivot_df.applymap(format_annot)
        is_star_case = True

    # compute figure width from number of x categories
    min_width = 13
    width_scale = 1.3
    fig_width = max(min_width, (1 + len(feature_labels) * 0.6) * width_scale)

    # global height scaling factor
    height_scale = 1.5

    # adjust figure height and font sizes depending on annotation mode
    if is_star_case:
        min_height = 6
        base_height = 1 + len(y_labels) * 0.6
        fig_height = max(min_height, base_height * height_scale)
        annot_fontsize = 14
        x_tick_size = 18
        y_tick_size = 18
    else:
        base_height = 1 + len(y_labels) * 0.4
        fig_height = base_height * height_scale
        annot_fontsize = None
        x_tick_size = None
        y_tick_size = None

    # initialize figure with computed dimensions
    plt.figure(figsize=(fig_width, fig_height))

    # prepare common seaborn heatmap arguments
    heatmap_kwargs = dict(
        data=heatmap_values,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=feature_labels,
        yticklabels=y_labels,
        cbar_kws={'label': scale_axis_name},
        annot=(annot_values if annot_values is not None else True),
        fmt=('' if annot_values is not None else '.1f')
    )

    # increase annotation font size for mixed numeric-marker cases
    if is_star_case:
        heatmap_kwargs['annot_kws'] = {'fontsize': annot_fontsize}

    # draw heatmap
    ax = sns.heatmap(**heatmap_kwargs)

    # vertically center multiline y tick labels
    for label in ax.get_yticklabels():
        label.set_va('center')
        label.set_multialignment('center')

    # optionally enlarge tick labels for star case
    if is_star_case:
        ax.tick_params(axis='x', labelsize=x_tick_size)
        ax.tick_params(axis='y', labelsize=y_tick_size)

    # set axis labels
    ax.set_xlabel(f'{x_axis_name}', fontsize=22)
    ax.set_ylabel(f'{y_axis_name}', fontsize=22)

    # build plot title
    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    title_name += plot_name_add

    # optionally show title
    if SHOW_TITLE:
        ax.set_title(title_name)
    else:
        ax.set_title("")

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, f'{feature_specifier}')
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_freq_diff_plot(
    plot_name: str,
    plot_name_add: str,
    x_feature_name: str,
    x_axis_name: str,
    freq_df: pd.DataFrame,
    delvg_count: int,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_specifier: str = '',
    ):
    '''
        Create a grouped bar plot comparing two frequency columns for one categorical feature.
        :param plot_name: plot title prefix
        :param plot_name_add: additional title suffix
        :param x_feature_name: categorical feature plotted on the x-axis
        :param x_axis_name: x-axis label
        :param freq_df: dataframe containing feature column and two frequency columns
        :param delvg_count: total number of DelVGs used for title annotation
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param feature_specifier: optional additional subfolder name for saving

        :return: None
    '''
    # extract the two frequency column names
    freq0 = freq_df.columns[1]
    freq1 = freq_df.columns[2]

    # reshape dataframe into long format for seaborn barplot
    df_plot = pd.melt(
        freq_df,
        id_vars=x_feature_name,
        value_vars=[freq0, freq1],
        var_name='type',
        value_name='frequency'
    )

    # replace internal frequency labels with readable names
    label_map = {
        freq0: 'site of candidates',
        freq1: 'full sequence'
        }

    df_plot['type'] = df_plot['type'].replace(label_map)

    # define custom colors for both compared frequency types
    custom_colors = {
        label_map[freq0]: COLORS[8],
        label_map[freq1]: COLORS[0]
        }

    # initialize figure and draw grouped bar plot
    plt.figure(figsize=(18, 6))
    sns.barplot(
    data=df_plot,
    x=x_feature_name,
    y='frequency',
    hue='type',
    palette=custom_colors,
    edgecolor="white",
    linewidth=0.5
)

    # set axis labels
    plt.xticks(rotation=0)
    plt.xlabel(f'{x_axis_name}')
    plt.ylabel('relative frequency (%)')

    # build plot title
    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    title_name += plot_name_add

    # optionally show title
    if SHOW_TITLE:
        plt.title(title_name)
    else:
        plt.title("")

    # optionally show legend outside plot area
    if SHOW_LEGEND:
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.,
            title=make_legend_descriptor(x_feature_name),
            frameon=True
        )

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, f'{feature_specifier}')
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += ".png"

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_spline_plot(
    plot_name: str,
    x_axis_name: str,
    y_axis_name: str, 
    spline_names: list,
    spline_values: list,
    delvg_count: int,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str
    ):  
    '''
        Create a spline plot for multiple value series over equally spaced x positions.
        :param plot_name: plot title prefix
        :param x_axis_name: x-axis label
        :param y_axis_name: y-axis label
        :param spline_names: list of series names
        :param spline_values: list of y-value lists, one per series
        :param delvg_count: total number of DelVGs used for title annotation
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # initialize figure
    plt.figure(figsize=(10, 6))

    # define original and smoothed x coordinates
    num_points = len(spline_values[0])
    x = np.arange(1, num_points + 1)
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # pick distinct colors for all spline series
    n_needed = len(spline_values)
    picked_colors = pick_colors(COLORS, n_needed)

    # plot smooth spline curves and original points for each series
    for i, y in enumerate(spline_values):
        spline = CubicSpline(x, y)
        y_smooth = spline(x_smooth)
        color = picked_colors[i]
        plt.plot(x_smooth, y_smooth, label=capitalize_first(spline_names[i]), color=color)
        plt.scatter(x, y, color=color, s=30)

    # build plot title
    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'

    # optionally show title
    if SHOW_TITLE:
        plt.title(title_name)
    else:
        plt.title("")

    # set axis labels
    plt.xlabel(capitalize_first(x_axis_name))
    plt.ylabel(capitalize_first(y_axis_name))

    # set fixed y-range and x ticks
    if LIM_AXES:
        plt.ylim(0, 3)

    plt.grid(True, alpha=0.3)

    plt.xticks(np.arange(1, num_points + 1))

    # optionally show legend outside plot area
    if SHOW_LEGEND:
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.,
            title=make_legend_descriptor('Site'),
            frameon=True
        )

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_single_density_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    show_quantiles: bool,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    elbow_side: str = "",
    kde_bw_adjust: float = 1.0,
    kde_grid: int = 512,
    clip=None
):
    '''
        Create a KDE-only density plot for one feature with optional quantiles and curvature-based elbow detection.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param x_feature_name: numeric feature column to plot
        :param x_axis_name: x-axis label
        :param show_quantiles: if True, show selected quantile lines and values
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param elbow_side: side on which to search for maximum curvature ('right', 'left', 'both', or '')
        :param kde_bw_adjust: bandwidth adjustment factor for gaussian KDE
        :param kde_grid: number of grid points used for KDE evaluation
        :param clip: optional lower and upper bounds for plotting

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))

    # extract numeric values and drop missing entries
    s = pd.to_numeric(df[x_feature_name], errors="coerce").dropna()
    values = s.to_numpy()
    n = len(values)
    if n == 0:
        raise ValueError(f"No numeric data in column '{x_feature_name}' after dropping NaNs/coercion.")

    # compute statistical moments from raw values
    mean_val = float(np.mean(values))
    var_val = float(np.var(values, ddof=0))
    skew_val = float(stats.skew(values, bias=True))
    kurt_val = float(stats.kurtosis(values, fisher=True, bias=True))

    # determine finite value bounds for KDE evaluation
    xmin = float(np.nanmin(values))
    xmax = float(np.nanmax(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError(f"No finite numeric data in column '{x_feature_name}' after cleaning.")
    if xmin == xmax:
        # widen constant distributions slightly so KDE evaluation remains defined
        xmin = xmin - 0.5
        xmax = xmax + 0.5

    # optionally clip x-range to requested interval
    if clip is not None:
        c0, c1 = clip
        if c0 is not None:
            xmin = max(xmin, float(c0))
        if c1 is not None:
            xmax = min(xmax, float(c1))
        if xmin == xmax:
            xmin = xmin - 0.5
            xmax = xmax + 0.5

    # create evenly spaced grid for KDE evaluation
    xs = np.linspace(xmin, xmax, int(kde_grid))

    # fit gaussian KDE and optionally adjust bandwidth
    kde = stats.gaussian_kde(values)
    if kde_bw_adjust != 1.0:
        base_cf = kde.covariance_factor()
        kde.covariance_factor = lambda: base_cf * kde_bw_adjust
        kde._compute_covariance()

    pdf = kde(xs)

    # prepare optional curvature-based elbow detection
    elbow_side_l = (elbow_side or "").strip().lower()
    plot_elbow = elbow_side_l != ""

    elbow_x = None
    elbow_idx = None

    if plot_elbow:
        # estimate first and second derivatives of KDE curve
        dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
        pdf_prime = np.gradient(pdf, dx)
        pdf_double = np.gradient(pdf_prime, dx)
        curvature = np.abs(pdf_double) / np.power(1.0 + pdf_prime**2, 1.5)

        # restrict elbow search to requested side relative to the mean
        if elbow_side_l == "right":
            mask = xs >= mean_val
        elif elbow_side_l == "left":
            mask = xs <= mean_val
        elif elbow_side_l == "both":
            mask = np.ones_like(xs, dtype=bool)
        else:
            raise ValueError("elbow_side must be 'right', 'left', 'both', or '' to disable.")

        # fall back to the mean if curvature search is not meaningful
        if (not np.any(mask)) or np.allclose(pdf, pdf[0]):
            elbow_x = float(mean_val)
            elbow_idx = int(np.argmin(np.abs(xs - elbow_x)))
        else:
            masked_idx = np.where(mask)[0]
            local_argmax = masked_idx[int(np.argmax(curvature[mask]))]
            elbow_idx = int(local_argmax)
            elbow_x = float(xs[elbow_idx])

    # plot KDE curve and shaded area
    plt.plot(xs, pdf, color=COLORS[8], linewidth=2.0, label="Density")
    plt.fill_between(xs, 0, pdf, color=COLORS[8], alpha=0.15)

    # add mean and optional elbow indicator lines
    plt.axvline(mean_val, color=COLORS[2], linestyle="-", linewidth=1.5, label="Mean")
    if plot_elbow and elbow_x is not None:
        plt.axvline(elbow_x, color=COLORS[10], linestyle="--", linewidth=1.5, label="Max. curvature")

    # optionally add quantile lines
    q80 = q85 = q90 = q95 = None
    if show_quantiles:
        q80 = float(np.quantile(values, 0.80))
        plt.axvline(q80, color=COLORS[0], linestyle="-", linewidth=1, label="80th percentile")

    # build plot title
    title = f"{plot_name}"
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f" (n={n})"

    # optionally show title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")

    # set axis labels
    plt.xlabel(f"{x_axis_name}")
    plt.ylabel("Density (area under curve = 1)")

    # prepare statistics text box content
    stats_lines = [
        f"Mean={mean_val:.{DECIMALS}f}",
        f"Variance={var_val:.{DECIMALS}f}",
        f"Skewness={skew_val:.{DECIMALS}f}",
        f"Kurtosis={kurt_val:.{DECIMALS}f}",
    ]

    # optionally append quantile information
    if show_quantiles:
        stats_lines += [
            "",
            f"80th percentile={q80:.{DECIMALS}f}",
        ]

    # optionally append elbow information
    if plot_elbow and elbow_x is not None:
        stats_lines += [
            "",
            f"Max. curvature={elbow_x:.{DECIMALS}f}",
        ]

    stats_text = "\n".join(stats_lines)

    # draw text box with summary statistics
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    plt.gca().text(
        0.50, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props
    )

    # optionally place legend outside plot area
    if SHOW_LEGEND:
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            title='Legend',
            frameon=True
        )

    # set fixed axis limits
    if LIM_AXES:
        plt.xlim(1, 6)
        plt.ylim(0, 1.3)

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += ".png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_freq_bar_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    delvg_count: int,
    top_n: int,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_specifier: str = '',
    ):
    '''
        Create a bar plot comparing relative category frequencies between high and lower NGS-rank groups, including Fisher exact test results.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param x_feature_name: categorical feature column to compare
        :param x_axis_name: x-axis label
        :param delvg_count: total number of DelVGs used for title annotation
        :param top_n: number of top categories to plot based on frequency difference
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param feature_specifier: optional additional subfolder name for saving

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 7))

    # split dataframe into high-rank and lower-rank groups
    true_df, false_df, threshold = split_by_threshold(df, 'NGS_quantile_rank', RANK_THRESHOLD)

    true_df = true_df.copy()
    false_df = false_df.copy()

    # stop early if one of the groups is empty
    n_true, n_false = len(true_df), len(false_df)
    if n_true == 0 or n_false == 0:
        print(f"[WARNING] Fisher test skipped: one split is empty (n_true={n_true}, n_false={n_false}).")
        plt.close()
        return

    def _normalize_cats(values: pd.DataFrame, x_feature_name: str) -> pd.Series:
        # normalize categorical values and replace missing values with UNKNOWN
        cats = (
            values[x_feature_name]
            .astype(object)
            .where(values[x_feature_name].notna(), other="UNKNOWN")
            .astype(str)
        )
        values[x_feature_name] = cats
        return cats

    # normalize category values in both groups
    cats_true = _normalize_cats(true_df, x_feature_name)
    cats_false = _normalize_cats(false_df, x_feature_name)

    # compute raw category counts for both groups
    true_counts  = true_df[x_feature_name].value_counts(dropna=False)
    false_counts = false_df[x_feature_name].value_counts(dropna=False)

    def pct(count, denom):
        # convert counts to relative frequencies in percent
        return (100.0 * count / denom) if denom > 0 else 0.0

    # detect whether feature categories represent motifs
    x_feature_name_is_motif = ("motif" in str(x_feature_name).casefold())
    present = list(pd.unique(pd.concat([cats_true, cats_false], axis=0)))

    # determine category order for plotting
    if x_feature_name_is_motif:
        uniq = present  # preserve original order for motifs
    else:
        if x_feature_name == "segment":
            ordered = [s for s in SEGMENTS if s in present]
            remaining = sorted([s for s in present if s not in ordered])
            uniq = ordered + remaining
        elif x_feature_name == "strain":
            ordered = [s for s in STRAINS if s in present]
            remaining = sorted([s for s in present if s not in ordered])
            uniq = ordered + remaining
        elif x_feature_name == "dataset":
            ordered = [s for s in DATASETS if s in present]
            remaining = sorted([s for s in present if s not in ordered])
            uniq = ordered + remaining
        else:
            uniq = sorted(present, key=lambda s: (str(s).strip().casefold() == "unknown", str(s).casefold()))

    all_cats = uniq

    # compute relative frequencies and differences between both groups
    freq_true  = {cat: pct(true_counts.get(cat, 0),  n_true)  for cat in all_cats}
    freq_false = {cat: pct(false_counts.get(cat, 0), n_false) for cat in all_cats}
    diffs      = {cat: freq_true[cat] - freq_false[cat] for cat in all_cats}

    # run Fisher exact tests for all categories
    fisher_map = {}
    pvals = []
    cats_for_p = []

    for cat in all_cats:
        a = int(true_counts.get(cat, 0))
        c = int(false_counts.get(cat, 0))

        res = fisher_exact_for_category(
            n_high_cat=a,
            n_high_total=n_true,
            n_low_cat=c,
            n_low_total=n_false,
            alternative="two-sided"
        )

        fisher_map[cat] = res
        if np.isfinite(res["pvalue"]):
            pvals.append(res["pvalue"])
            cats_for_p.append(cat)

    # apply BH-FDR correction to valid p-values
    p_adj_map = {}
    if len(pvals) > 0:
        p_adj = bh_fdr(pvals)
        for cat, padj in zip(cats_for_p, p_adj):
            p_adj_map[cat] = float(padj)

    # optionally restrict plot to top categories by frequency difference
    if top_n and top_n > 0:
        top_cats = sorted(all_cats, key=lambda c: diffs[c], reverse=True)[:top_n]
    else:
        top_cats = all_cats

    # prepare bar positions and heights
    x = np.arange(len(top_cats))
    width = 0.42

    bars_false = [freq_false[c] for c in top_cats]
    bars_true  = [freq_true[c]  for c in top_cats]

    color_false = COLORS[0]
    color_true  = COLORS[8]

    # plot bars for high-rank group
    plt.bar(
        x - width/2,
        bars_true,
        width=width,
        label=f'high NGS rank ({threshold})',
        color=color_true,
        edgecolor='white',
        linewidth=0.5
    )

    # plot bars for lower-rank group
    plt.bar(
        x + width/2,
        bars_false,
        width=width,
        label='lower NGS ranks (rest)',
        color=color_false,
        edgecolor='white',
        linewidth=0.5
    )

    # initialize annotation layout settings
    y_max_overall = 0.0

    BRACKET_PAD = 1.2
    BRACKET_HEIGHT = 0.8
    STAR_GAP = 0.6
    OR_GAP = 1.4

    # annotate each category with significance stars and odds ratio
    for i, cat in enumerate(top_cats):
        res = fisher_map.get(cat, {})
        OR = res.get("oddsratio", np.nan)
        p_raw = res.get("pvalue", np.nan)
        p_use = p_adj_map.get(cat, p_raw)

        stars = p_to_stars(p_use) if np.isfinite(p_use) else "NA"

        if np.isfinite(OR):
            metric_txt = "inf" if np.isinf(OR) else f"{OR:.{DECIMALS}g}"
        else:
            metric_txt = "NA"

        h_left  = bars_false[i]
        h_right = bars_true[i]
        h_pair_top = max(h_left, h_right)

        y_base = h_pair_top
        y_bracket = y_base + BRACKET_PAD
        h_bracket = BRACKET_HEIGHT

        x1 = i - width/2
        x2 = i + width/2

        # draw bracket connecting both bars of a category
        plt.plot(
            [x1, x1, x2, x2],
            [y_bracket, y_bracket + h_bracket, y_bracket + h_bracket, y_bracket],
            color='black',
            linewidth=1.0
        )

        base = y_bracket + h_bracket

        # determine text positions above the bracket
        stars_y  = base + STAR_GAP
        metric_y = stars_y + OR_GAP

        STAR_FONTSIZE = 16
        OR_FONTSIZE = 16

        # draw significance stars
        plt.text(
            (x1 + x2) / 2,
            stars_y,
            f'[{stars}]',
            ha='center',
            va='bottom',
            fontsize=STAR_FONTSIZE,
        )

        # draw odds ratio text
        plt.text(
            (x1 + x2) / 2,
            metric_y,
            f'OR={metric_txt}',
            ha='center',
            va='bottom',
            fontsize=OR_FONTSIZE
        )

        y_max_overall = max(y_max_overall, metric_y)

    # set x tick labels with category counts
    labels = [f"{str(c)}\n(n={true_counts.get(c, 0) + false_counts.get(c, 0)})"
              for c in top_cats]
    plt.xticks(x, labels, rotation=90, ha='center')

    # set axis labels
    plt.xlabel(capitalize_first(x_axis_name))
    plt.ylabel('Relative frequency (%)')

    # build plot title
    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'

    # optionally show title
    if SHOW_TITLE:
        plt.title(title_name)
    else:
        plt.title("")

    # optionally show legend outside plot area
    if SHOW_LEGEND:
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.,
            title=make_legend_descriptor(selector),
            frameon=True
        )

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, feature_specifier)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname + '.png'), dpi=300)
    plt.close()

def create_feature_violin_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    y_feature_name: str,
    y_axis_name: str,
    delvg_count: int,
    top_n: int,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_specifier: str = '',
    ):
    '''
        Create a violin plot for a numeric feature across categorical groups.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param x_feature_name: categorical feature used for grouping on the x-axis
        :param x_axis_name: x-axis label
        :param y_feature_name: numeric feature plotted on the y-axis
        :param y_axis_name: y-axis label
        :param delvg_count: total number of DelVGs used for title annotation
        :param top_n: number of most frequent categories to include, or all if not positive
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param feature_specifier: optional additional subfolder name for saving

        :return: None
    '''
    # apply plotting style
    plt.style.use('seaborn-darkgrid')
    sns.set_theme(style="whitegrid")

    df_plot = df.copy()

    # normalize categorical x values and replace missing values
    df_plot[x_feature_name] = df_plot[x_feature_name].astype(object)
    df_plot[x_feature_name] = df_plot[x_feature_name].where(df_plot[x_feature_name].notna(), 'NA')

    # coerce y feature to numeric if needed
    if not np.issubdtype(df_plot[y_feature_name].dtype, np.number):
        df_plot[y_feature_name] = pd.to_numeric(df_plot[y_feature_name], errors='coerce')

    counts = df_plot[x_feature_name].value_counts(dropna=False)

    # determine category order for x-axis
    if x_feature_name == "segment":
        present = list(pd.unique(counts.index.tolist()))
        ordered = [s for s in SEGMENTS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        cat_order_full = ordered + remaining
    elif x_feature_name == "strain":
        present = list(pd.unique(counts.index.tolist()))
        ordered = [s for s in STRAINS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        cat_order_full = ordered + remaining
    elif x_feature_name == "dataset":
        present = list(pd.unique(counts.index.tolist()))
        ordered = [s for s in DATASETS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        cat_order_full = ordered + remaining
    else:
        cat_order_full = sorted(counts.index.tolist(), key=lambda c: str(c).casefold())

    # optionally restrict to the most frequent categories
    if top_n and top_n > 0:
        top_set = set(counts.head(top_n).index.tolist())
        cat_order = [c for c in cat_order_full if c in top_set]
    else:
        cat_order = cat_order_full

    # keep only selected categories
    df_plot = df_plot[df_plot[x_feature_name].isin(cat_order)].copy()
    cat_counts = df_plot[x_feature_name].value_counts()
    tick_labels = [f"{str(c)}\n(n={int(cat_counts.get(c, 0))})" for c in cat_order]

    # initialize figure and draw violin plot
    plt.figure(figsize=(12, 7))
    ax = sns.violinplot(
        data=df_plot,
        x=x_feature_name,
        y=y_feature_name,
        order=cat_order,
        cut=0,
        inner=None,
        linewidth=1.0,
        color=COLORS[8],
        bw=0.2,
        gridsize=200,
        scale='width',
    )

    # compute quartiles and median for each category
    q = df_plot.groupby(x_feature_name, sort=False)[y_feature_name].quantile([0.25, 0.5, 0.75]).unstack()
    x_pos = dict(zip(cat_order, ax.get_xticks()))
    for cat in cat_order:
        if cat not in q.index:
            continue
        q1, med, q3 = q.loc[cat, [0.25, 0.5, 0.75]]
        if pd.isna(q1) or pd.isna(med) or pd.isna(q3):
            continue

        # add IQR line and median point on top of each violin
        x0 = x_pos[cat]
        ax.vlines(x0, q1, q3, color='black', linewidth=2.0, zorder=3)
        ax.scatter([x0], [med], s=28, color='white', edgecolor='black', linewidth=0.7, zorder=4)

    # set axis labels
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)

    # build plot title
    title_name  = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'

    # optionally show title
    if SHOW_TITLE:
        ax.set_title(title_name)
    else:
        ax.set_title("")

    # set x tick labels including sample sizes
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')

    plt.tight_layout()

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, f'{feature_specifier}')
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    out_name = fname + '.png' if not fname.lower().endswith('.png') else fname
    plt.savefig(os.path.join(save_path, out_name), dpi=300)
    plt.close()

def create_feature_scatter_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    y_feature_name: str,
    y_axis_name: str,
    selector: str,
    show_rolling_median: bool,
    rolling_window: int,
    show_identity_line: bool,
    pseudo_prefix: str,
    show_decision_threshold: float,
    show_split_threshold: float,
    reg_metrics: bool,
    huber_delta: float,
    fname: str,
    path: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    subset_n: int = 5000,
    ):
    '''
        Create a scatter plot of two numeric features with optional deterministic subsetting, trend lines and regression metrics.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param x_feature_name: feature plotted on the x-axis
        :param x_axis_name: x-axis label
        :param y_feature_name: feature plotted on the y-axis
        :param y_axis_name: y-axis label
        :param selector: categorical feature used for coloring points
        :param show_rolling_median: if True, overlay a rolling median trend line
        :param rolling_window: window size for rolling median computation
        :param show_identity_line: if True, draw the line y = x
        :param pseudo_prefix: optional prefix for pseudo-candidate descriptors
        :param show_decision_threshold: optional y-value for a horizontal threshold line
        :param show_split_threshold: optional x-value for a vertical threshold line
        :param reg_metrics: if True, compute and report MAE, MSE and R²
        :param huber_delta: unused parameter kept for interface compatibility
        :param fname: output file name without extension
        :param path: top-level result path
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param subset_n: maximum number of deterministically selected rows to plot

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(14, 6))

    # apply deterministic subsetting before selector-based coloring
    if subset_n is not None and subset_n > 0 and len(df) > subset_n:

        # create stable row identifiers if missing
        if "ID" not in df.columns:
            df = df.copy()

            # try to build ID from common junction identity columns
            junction_col_sets = [
                ("segment", "start", "end"),
            ]

            made_id = False
            for seg_c, s_c, e_c in junction_col_sets:
                if seg_c in df.columns and s_c in df.columns and e_c in df.columns:
                    df["ID"] = (
                        df[seg_c].astype(str) + "_" +
                        df[s_c].astype(str) + "_" +
                        df[e_c].astype(str)
                    )
                    made_id = True
                    break

            # fall back to a deterministic ID built from available columns
            if not made_id:
                fallback_cols = [
                    c for c in
                    ["dataset", "strain", "segment", "start", "end"]
                    if c in df.columns
                ]
                if len(fallback_cols) > 0:
                    df["ID"] = df[fallback_cols].astype(str).agg("_".join, axis=1)
                else:
                    # last-resort deterministic ID based on row index
                    df["ID"] = df.index.astype(str)

        # hash IDs deterministically and keep the smallest subset_n hashes
        _ids = df["ID"].astype(str)
        _h = pd.util.hash_pandas_object(_ids, index=False).astype("uint64")
        _h = (_h ^ np.uint64(SEED))

        pick_idx = (
            pd.DataFrame({"_h": _h}, index=df.index)
            .sort_values("_h")
            .head(int(subset_n))
            .index
        )
        df = df.loc[pick_idx].copy()

    # extract required columns and drop rows missing x or y values
    values = df[[x_feature_name, y_feature_name, selector]] \
        .dropna(subset=[x_feature_name, y_feature_name]) \
        .copy()
    n = len(values)

    # normalize selector values and replace missing values
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="UNKNOWN")
        .astype(str)
    )
    values[selector] = cats

    # determine selector order for known grouping types
    if selector == "segment":
        present = list(pd.unique(cats))
        ordered = [s for s in SEGMENTS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        uniq = ordered + remaining
    elif selector == "strain":
        present = list(pd.unique(cats))
        ordered = [s for s in STRAINS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        uniq = ordered + remaining
    elif selector == "dataset":
        present = list(pd.unique(cats))
        ordered = [s for s in DATASETS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        uniq = ordered + remaining
    else:
        uniq = sorted(pd.unique(cats))

    # compute selector category sizes for legend labels
    cat_counts = cats.value_counts(dropna=False).to_dict()

    def format_cat(u: str) -> str:
        # convert internal strain codes to official names where possible
        if selector == "strain":
            try:
                return get_official_strain_name(u)
            except Exception:
                return str(u)
        return str(u)

    # create legend labels including sample sizes
    label_map = {u: f"{format_cat(u)} (n={int(cat_counts.get(u, 0))})" for u in uniq}

    values[selector] = values[selector].map(label_map)

    # preserve final legend order after label mapping
    uniq = [label_map[u] for u in uniq]

    # shuffle rows deterministically to reduce plotting bias from overplotting order
    values = values.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # assign colors to selector categories
    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

    # draw scatter plot
    sns.scatterplot(
        data=values,
        x=x_feature_name,
        y=y_feature_name,
        hue=selector,
        hue_order=uniq,
        palette=palette,
        edgecolor='white',
        s=50,
        alpha=0.5,
    )

    # optionally overlay a rolling median trend line
    if show_rolling_median and (not values.empty):
        x_vals = pd.to_numeric(values[x_feature_name], errors="coerce").to_numpy(dtype=float)
        y_vals = pd.to_numeric(values[y_feature_name], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        if int(x_vals.shape[0]) >= max(3, int(rolling_window)):
            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            y_sorted = y_vals[order]

            win = int(max(3, rolling_window))
            half = win // 2

            med_x = []
            med_y = []

            for i in range(len(x_sorted)):
                lo = max(0, i - half)
                hi = min(len(x_sorted), i + half + 1)

                med_x.append(float(np.nanmedian(x_sorted[lo:hi])))
                med_y.append(float(np.nanmedian(y_sorted[lo:hi])))

            plt.plot(
                med_x,
                med_y,
                linewidth=2,
                color="grey",
                label=f"rolling median (window={rolling_window})"
            )

    # optionally add horizontal decision threshold line
    if show_decision_threshold != 0:
        plt.axhline(
            float(show_decision_threshold),
            linestyle="--",
            linewidth=2,
            color="grey",
        )

    # optionally add vertical split threshold line
    if show_split_threshold != 0:
        plt.axvline(
            float(show_split_threshold),
            linestyle="--",
            linewidth=2,
            color="grey",
        )

    # initialize regression metric outputs
    mae = None
    mse = None
    r2_pred = None

    # optionally compute regression metrics comparing y to x
    if reg_metrics and (not values.empty):
        y_true = values[x_feature_name].to_numpy(dtype=float)
        y_pred = values[y_feature_name].to_numpy(dtype=float)

        if len(y_true) > 0:
            err = y_pred - y_true

            mae = float(np.nanmean(np.abs(err)))
            mse = float(np.nanmean(err ** 2))

        # compute R² only if the x values vary
        if len(y_true) > 1 and np.nanstd(y_true) > 0:
            ss_res = np.nansum((y_true - y_pred) ** 2)
            ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
            if ss_tot > 0:
                r2_pred = 1 - ss_res / ss_tot

    # choose regular or pseudo-candidate descriptor
    if pseudo_prefix:
        title_descriptor = make_pseudo_candidate_descriptor(pseudo_prefix, folder, data, strain, segment, intersects)
    else:
        title_descriptor = make_candidate_descriptor(folder, data, strain, segment, intersects)

    # build plot title
    title = f'{plot_name}'
    title += title_descriptor
    title += f' (n={n})'

    # optionally append regression metrics to the title
    if reg_metrics:
        line = "\n"

        if mae is not None and np.isfinite(mae):
            line += f', MAE={float(mae):.{DECIMALS}f}'
        else:
            line += ', MAE=nan'

        if mse is not None and np.isfinite(mse):
            line += f', MSE={float(mse):.{DECIMALS}f}'
        else:
            line += ', MSE=nan'

        if r2_pred is not None and np.isfinite(r2_pred):
            line += f', R²={float(r2_pred):.{DECIMALS}f}'
        else:
            line += ', R²=nan'

        title += line

    # optionally show title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")

    # set axis labels
    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')

    # set default axis limits by preprocessing folder
    if LIM_AXES:
        if folder == 'pooled':
            plt.ylim(1, 6)
            plt.xlim(1, 6)
        else:
            plt.ylim(0, 5)
            plt.xlim(0, 5)

    # adjust y-axis for binary outputs if applicable
    if "bin" in path:
        plt.ylim(0, 1)

    # optionally add identity line y = x spanning the visible plot range
    if show_identity_line:
        ax = plt.gca()

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        lo = min(x0, y0)
        hi = max(x1, y1)

        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            linewidth=2,
            color="grey",
            zorder=0,
        )

        # keep original limits after drawing identity line
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    # place legend outside plot area
    if 1:
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.,
            title=capitalize_first(selector),
            frameon=True
        )

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_multi_density_plot(
    plot_name: str,
    df_list: list,
    df_names: list,
    x_feature_name: str,
    x_axis_name: str,
    fname: str,
    path: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    show_js: bool = False,
    show_n_in_legend: bool = True,
    kde_bw_adjust: float = 1.0,
    common_norm: bool = False,
    clip=None,
    js_bins: int = 50,
    js_perm_n: int = 2000,
):
    '''
        Create a multi-density KDE plot for one or more dataframes and optionally compare two distributions with Jensen-Shannon distance.
        :param plot_name: plot title prefix
        :param df_list: list of input dataframes
        :param df_names: list of names corresponding to df_list
        :param x_feature_name: feature column to plot on the x-axis
        :param x_axis_name: x-axis label
        :param fname: output file name without extension
        :param path: top-level result path
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param show_js: if True, compute Jensen-Shannon distance between exactly two distributions
        :param show_n_in_legend: if True, append sample sizes to legend labels
        :param kde_bw_adjust: bandwidth adjustment passed to seaborn.kdeplot
        :param common_norm: if True, use common normalization across all KDEs
        :param clip: optional clipping range for KDE evaluation
        :param js_bins: number of bins used for histogram-based Jensen-Shannon distance
        :param js_perm_n: number of permutations used for Jensen-Shannon p-value estimation

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))

    # validate input list of dataframes
    if df_list is None or len(df_list) == 0:
        raise ValueError("df_list must contain at least one dataframe.")
    if df_names is None or len(df_names) != len(df_list):
        raise ValueError("df_names must be the same length as df_list.")

    # Jensen-Shannon comparison only supports exactly two distributions
    if show_js and len(df_list) != 2:
        raise ValueError("show_js=True requires exactly 2 dataframes in df_list.")

    # convert dataframe names to strings and determine present labels
    names = [str(n) for n in df_names]
    present = list(pd.unique(names))

    # infer legend title from provided names
    if "PB2" in names:
        legend_title = 'Segment'
    elif any(x in names for x in ("PR8", "Yamagata")):
        legend_title = 'Strain'
    if "IAV" in names:
        legend_title = 'Type'
    if "H1N1" in names:
        legend_title = 'Subtype'
    else:
        legend_title = 'Dataset'

    # determine plotting order for known group types
    if all(n in SEGMENTS for n in present):
        ordered = [s for s in SEGMENTS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in STRAINS for n in present):
        ordered = [s for s in STRAINS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in DATASETS for n in present):
        ordered = [s for s in DATASETS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    else:
        desired = sorted(present)

    # map names to original indices to preserve duplicates safely during reordering
    name_to_idxs = {}
    for i, n in enumerate(names):
        name_to_idxs.setdefault(n, []).append(i)

    # rebuild dataframe and name order according to desired ordering
    new_idx = []
    for n in desired:
        if n in name_to_idxs:
            new_idx.extend(name_to_idxs[n])

    df_list = [df_list[i] for i in new_idx]
    df_names = [df_names[i] for i in new_idx]

    # extract target feature, coerce to numeric, and remove missing values
    cleaned = []
    for df_i, name_i in zip(df_list, df_names):
        v = df_i[[x_feature_name]].copy()
        v[x_feature_name] = pd.to_numeric(v[x_feature_name], errors="coerce")
        v = v.dropna(subset=[x_feature_name])

        if len(v) == 0:
            raise ValueError(f"No numeric data in '{x_feature_name}' for '{name_i}' after cleaning.")

        cleaned.append(v)

    # optionally append sample sizes to legend labels
    plot_names = list(df_names)
    if show_n_in_legend:
        plot_names = [f"{name_i} (n={len(v)})" for v, name_i in zip(cleaned, df_names)]

    # compute histogram-based Jensen-Shannon distance between two numeric vectors
    def jensen_shannon_distance(x1, x2, bins=50, eps=1e-12):
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()

        xmin = float(np.nanmin([np.nanmin(x1), np.nanmin(x2)]))
        xmax = float(np.nanmax([np.nanmax(x1), np.nanmax(x2)]))

        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return np.nan
        if xmin == xmax:
            return 0.0  # identical constant distributions

        # compute histogram counts on shared bin edges
        h1, edges = np.histogram(x1, bins=bins, range=(xmin, xmax), density=False)
        h2, _     = np.histogram(x2, bins=edges, density=False)

        # convert counts to stabilized probability distributions
        p = h1.astype(float) + eps
        q = h2.astype(float) + eps
        p = p / np.sum(p)
        q = q / np.sum(q)

        # compute midpoint distribution and Jensen-Shannon distance
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        jsd = 0.5 * (kl_pm + kl_qm)

        return float(np.sqrt(jsd))

    # estimate permutation-based p-value for the observed Jensen-Shannon distance
    def js_permutation_pvalue(x1, x2, bins=50, n_perm=2000, seed=0):
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()

        obs = jensen_shannon_distance(x1, x2, bins=bins)
        if not np.isfinite(obs):
            return obs, np.nan

        n1 = x1.shape[0]
        n2 = x2.shape[0]
        pooled = np.concatenate([x1, x2], axis=0)

        rng = np.random.default_rng(seed)
        more_extreme = 0

        # perform permutation test and count equally or more extreme distances
        for _ in range(int(n_perm)):
            idx = rng.permutation(pooled.shape[0])
            a = pooled[idx[:n1]]
            b = pooled[idx[n1:n1 + n2]]
            js = jensen_shannon_distance(a, b, bins=bins)
            if np.isfinite(js) and js >= obs:
                more_extreme += 1

        # use small-sample safe p-value estimate
        p = (more_extreme + 1) / (int(n_perm) + 1)
        return obs, float(p)

    # initialize Jensen-Shannon outputs
    sim_js = np.nan
    js_p = np.nan
    js_stars = "NA"

    # compute Jensen-Shannon similarity and permutation p-value if requested
    if show_js:
        sim_js, js_p = js_permutation_pvalue(
            cleaned[0][x_feature_name].to_numpy(),
            cleaned[1][x_feature_name].to_numpy(),
            bins=js_bins,
            n_perm=js_perm_n,
            seed=SEED
        )
        js_stars = p_to_stars(js_p) if np.isfinite(js_p) else "NA"

    # combine all cleaned values into one dataframe for seaborn KDE plotting
    values = pd.concat(
        [v.assign(type=name_i) for v, name_i in zip(cleaned, plot_names)],
        ignore_index=True
    )

    # select colors for unique legend labels
    uniq = list(plot_names)
    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

    # draw KDE curves; seaborn creates the legend automatically
    ax = sns.kdeplot(
        data=values,
        x=x_feature_name,
        hue="type",
        palette=palette,
        linewidth=2,
        bw_adjust=kde_bw_adjust,
        common_norm=common_norm,
        clip=clip
    )

    # build title with descriptor and sample size summary
    title = f"{plot_name}"
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)

    if len(df_list) <= 2:
        title += " (" + ", ".join([f"n_{name_i}={len(v)}" for v, name_i in zip(cleaned, df_names)]) + ")"
    else:
        n_total = sum(len(v) for v in cleaned)
        title += f" (n={n_total})"

    # optionally append Jensen-Shannon statistics to title
    if show_js:
        if np.isfinite(sim_js):
            title += f"\nJS-distance={sim_js:.{DECIMALS}f}"
        else:
            title += "\nJS-distance=NA"

        if np.isfinite(js_p):
            title += f" [{js_stars}]"
        else:
            title += " [NA]"

    # optionally show plot title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")
    
    # set axis labels
    plt.xlabel(x_axis_name)

    if common_norm:
        plt.ylabel("Density (common normalization)")
    else:
        plt.ylabel("Density (area under each curve = 1)")

    # keep placeholder for possible downstream use
    leg_placeholder = legend_title

    # move seaborn-created legend outside plot area and set its title
    leg = ax.legend_
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1))
        leg._loc = 2  # "upper left"
        leg.set_title("Segment")
        leg.set_frame_on(True)

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname + ".png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_prediction_probability_density_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    decision_threshold: float,
    path: str,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    kde_bw_adjust: float = 1.0,
    kde_grid: int = 512,
    clip=(0, 1),
):
    '''
        Create a KDE-only density plot for predicted probabilities with a decision threshold line.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param x_feature_name: probability column to plot
        :param x_axis_name: x-axis label
        :param decision_threshold: classification threshold shown as vertical line
        :param path: top-level result path
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor
        :param kde_bw_adjust: bandwidth adjustment factor for gaussian KDE
        :param kde_grid: number of grid points used for KDE evaluation
        :param clip: optional lower and upper bounds for plotting

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))

    # extract numeric values and drop missing entries
    s = pd.to_numeric(df[x_feature_name], errors="coerce").dropna()
    values = s.to_numpy()
    n = len(values)
    if n == 0:
        raise ValueError(
            f"no numeric data in column '{x_feature_name}' after dropping NaNs/coercion."
        )

    # determine finite value bounds
    xmin = float(np.nanmin(values))
    xmax = float(np.nanmax(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError(f"no finite numeric data in column '{x_feature_name}' after cleaning.")

    # widen bounds slightly for constant distributions
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5

    # optionally clip x-range to requested interval
    if clip is not None:
        c0, c1 = clip
        if c0 is not None:
            xmin = max(xmin, float(c0))
        if c1 is not None:
            xmax = min(xmax, float(c1))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5

    # create evenly spaced grid for KDE evaluation
    xs = np.linspace(xmin, xmax, int(kde_grid))

    # fit gaussian KDE and optionally adjust bandwidth
    kde = stats.gaussian_kde(values)
    if kde_bw_adjust != 1.0:
        base_cf = kde.covariance_factor()
        kde.covariance_factor = lambda: base_cf * kde_bw_adjust
        kde._compute_covariance()

    pdf = kde(xs)

    # plot KDE curve and shaded area
    plt.plot(xs, pdf, color=COLORS[6], linewidth=2.0, label="test")
    plt.fill_between(xs, 0, pdf, color=COLORS[6], alpha=0.15)

    # add grey dashed decision threshold line
    plt.axvline(
        float(decision_threshold),
        color="grey",
        linestyle="--",
        linewidth=1.8,
        label=f"decision_threshold={decision_threshold}",
    )

    # build plot title
    title = f"{plot_name}"
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f" (n={n})"

    # optionally show title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")

    # set axis labels
    plt.xlabel(f"{x_axis_name}")
    plt.ylabel("Density (area under curve = 1)")

    # legend intentionally disabled here
    if 0:
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            title=capitalize_first("type"),
            frameon=True,
        )

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += ".png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_feature_roc_auc_plot(
    plot_name: str,
    df: pd.DataFrame,
    y_true_feature_name: str,
    y_pred_proba_feature_name: str,
    f1_threshold: float,
    selector: str,
    fname: str,
    path: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    ):
    '''
        Create an ROC curve plot for binary predictions and report ROC-AUC and F1.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param y_true_feature_name: true binary label column
        :param y_pred_proba_feature_name: predicted probability column
        :param f1_threshold: threshold used to convert probabilities into class labels for F1
        :param selector: grouping column retained for consistency and cleaning
        :param fname: output file name without extension
        :param path: top-level result path
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # keep only required columns and drop rows missing truth or prediction values
    needed = [y_true_feature_name, y_pred_proba_feature_name, selector]
    values = df[needed].dropna(subset=[y_true_feature_name, y_pred_proba_feature_name]).copy()
    n = len(values)

    # normalize selector values for consistency
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="unknown")
        .astype(str)
    )
    values[selector] = cats

    # extract arrays for metric computation
    y_true = values[y_true_feature_name].to_numpy()
    y_proba = values[y_pred_proba_feature_name].to_numpy()

    # compute ROC curve and ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    auc_score = round(auc_score, DECIMALS)

    # compute F1 score at the fixed decision threshold
    y_pred_label = (y_proba >= f1_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    f1 = round(f1, DECIMALS)

    roc_color = COLORS[6]

    # plot ROC curve and diagonal reference line
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        color=roc_color
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="grey")

    # build plot title including summary metrics
    title = f'{plot_name}'
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f' (n={n})'
    title += f'\nROC-AUC={auc_score}, F1={f1} (decision_threshold={f1_threshold})'

    # fix axis ranges to [0, 1]
    if LIM_AXES:
        plt.ylim(0, 1)
        plt.xlim(0, 1)

    # optionally show title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")

    # set axis labels and grid
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.grid(True)

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_feature_pr_auc_plot(
    plot_name: str,
    df: pd.DataFrame,
    y_true_feature_name: str,
    y_pred_proba_feature_name: str,
    f1_threshold: float,
    selector: str,
    fname: str,
    path: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    ):
    '''
        Create a precision-recall curve plot for binary predictions and report PR-AUC and F1.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param y_true_feature_name: true binary label column
        :param y_pred_proba_feature_name: predicted probability column
        :param f1_threshold: threshold used to convert probabilities into class labels for F1
        :param selector: grouping column retained for consistency and cleaning
        :param fname: output file name without extension
        :param path: top-level result path
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # keep only required columns and drop rows missing truth or prediction values
    needed = [y_true_feature_name, y_pred_proba_feature_name, selector]
    values = df[needed].dropna(subset=[y_true_feature_name, y_pred_proba_feature_name]).copy()
    n = len(values)

    # normalize selector values for consistency
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="UNKNOWN")
        .astype(str)
    )
    values[selector] = cats

    # extract arrays for metric computation
    y_true = values[y_true_feature_name].to_numpy()
    y_proba = values[y_pred_proba_feature_name].to_numpy()

    # compute precision-recall curve and PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = average_precision_score(y_true, y_proba)
    pr_auc_score = round(pr_auc_score, DECIMALS)

    # compute F1 score at the fixed decision threshold
    y_pred_label = (y_proba >= f1_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    f1 = round(f1, DECIMALS)

    pr_color = COLORS[6]

    # plot precision-recall curve
    plt.plot(
        recall,
        precision,
        linewidth=2,
        color=pr_color
    )

    # build plot title including summary metrics
    title = f'{plot_name}'
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f' (n={n})'
    title += f'\nPR-AUC={pr_auc_score}, F1={f1} (decision_threshold={f1_threshold})'

    # fix axis ranges to [0, 1]
    if LIM_AXES:
        plt.ylim(0, 1)
        plt.xlim(0, 1)

    # optionally show title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")

    # set axis labels and grid
    plt.xlabel("Recall (-)")
    plt.ylabel("Precision (-)")

    plt.grid(True)

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_feature_residual_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    y_pred_feature_name: str,
    y_axis_name: str,
    selector: str,
    show_zero_line: bool,
    reg_metrics: bool,
    fname: str,
    path: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    ):
    '''
        Create a residual scatter plot comparing predictions to true values and optionally report regression metrics.
        :param plot_name: plot title prefix
        :param df: input dataframe
        :param x_feature_name: true target column
        :param x_axis_name: x-axis label
        :param y_pred_feature_name: predicted target column
        :param y_axis_name: y-axis label
        :param selector: grouping column used for coloring points
        :param show_zero_line: if True, draw a horizontal zero residual line
        :param reg_metrics: if True, compute and report MAE, MSE and R²
        :param fname: output file name without extension
        :param path: top-level result path
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # apply plotting style and initialize figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # keep only required columns and drop rows missing true or predicted values
    needed = [x_feature_name, y_pred_feature_name, selector]
    values = df[needed].dropna(subset=[x_feature_name, y_pred_feature_name]).copy()
    n = len(values)

    # compute residuals as prediction minus true value
    values["residual"] = values[y_pred_feature_name] - values[x_feature_name]

    # normalize selector values for consistent plotting
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="unknown")
        .astype(str)
    )
    values[selector] = cats
    uniq = sorted(cats.unique())

    # assign colors to unique selector categories
    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

    # draw scatter plot of residuals against true values
    sns.scatterplot(
        data=values,
        x=x_feature_name,
        y="residual",
        hue=selector,
        palette=palette,
        edgecolor='white',
        s=50,
        alpha=0.7,
        legend=SHOW_LEGEND
    )

    # optionally add horizontal zero reference line
    if show_zero_line:
        plt.axhline(0, color="grey", linestyle="--", linewidth=1.5, label="zero residual")

    # initialize regression metric outputs
    mae = None
    mse = None
    hub = None
    r2  = None

    # optionally compute regression metrics from predictions and true values
    if reg_metrics and (not values.empty):
        y_true = values[x_feature_name].to_numpy(dtype=float)
        y_pred = values[y_pred_feature_name].to_numpy(dtype=float)

        if len(y_true) > 0:
            err = y_pred - y_true

            mae = float(np.nanmean(np.abs(err)))
            mse = float(np.nanmean(err ** 2))

        # compute R² only if at least two variable true values exist
        if len(y_true) > 1 and np.nanstd(y_true) > 0:
            ss_res = np.nansum((y_true - y_pred) ** 2)
            ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
            if ss_tot > 0:
                r2 = 1 - ss_res / ss_tot

    # build plot title
    title = f'{plot_name}'
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f' (n={n})'

    # optionally append regression metrics to the title
    if reg_metrics:
        line = "\n"

        if mae is not None and np.isfinite(mae):
            line += f', MAE={float(mae):.{DECIMALS}f}'
        else:
            line += ', MAE=nan'

        if mse is not None and np.isfinite(mse):
            line += f'MSE={float(mse):.{DECIMALS}f}'
        else:
            line += 'MSE=nan'

        if r2 is not None and np.isfinite(r2):
            line += f', R²={float(r2):.{DECIMALS}f}'
        else:
            line += ', R²=nan'

        title += line

    # set axis limits depending on preprocessing folder
    if LIM_AXES:
        if folder == 'pooled':
            plt.ylim(-6, 6)
            plt.xlim(1, 6)
        else:
            plt.ylim(-5, 5)
            plt.xlim(0, 5)

    # optionally show title
    if SHOW_TITLE:
        plt.title(title)
    else:
        plt.title("")

    # set axis labels
    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')

    # optionally place legend outside plot area
    if SHOW_LEGEND:
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.,
            title=capitalize_first(selector),
            frameon=True
        )

    # construct save path and create directories if needed
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
    plt.close()

def create_intersect_bar_plot(
    df: pd.DataFrame,
    y_feature_name: str,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
):
    '''
        Create a horizontal bar plot comparing global and dataset-wise intersection percentages per group.
        :param df: input dataframe containing ikey, dataset and grouping feature
        :param y_feature_name: feature used to define bar groups on the y-axis
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # check that all required columns are present
    required = {"ikey", "dataset", y_feature_name}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # normalize grouping feature and replace missing values with display label
    df_global = df.copy()
    df_global[y_feature_name] = df_global[y_feature_name].astype(object)
    df_global[y_feature_name] = df_global[y_feature_name].where(
        df_global[y_feature_name].notna(), "NA"
    )

    # collapse repeated entries within each dataset and ikey combination
    df_dataset = (
        df_global.sort_values(["dataset", "ikey"])
        .groupby(["dataset", "ikey"], as_index=False)
        .first()
    )

    # compute percentage of rows with non-unique ikeys within one group
    def pct_nonunique_within_group(g: pd.DataFrame) -> float:
        n = len(g)
        if n == 0:
            return 0.0
        dup_mask = g.duplicated(subset=["ikey"], keep=False)
        return 100.0 * float(dup_mask.sum()) / float(n)

    # compute per-group intersection statistics for a dataframe
    def compute_group_stats(dfx: pd.DataFrame) -> dict:
        # group -> (pct_nonunique, n_rows)
        stats = {}
        if dfx is None or len(dfx) == 0:
            return stats
        for group_name, g in dfx.groupby(y_feature_name, dropna=False):
            stats[group_name] = (pct_nonunique_within_group(g), int(len(g)))
        return stats

    # compute statistics for global and dataset-collapsed views
    stats_g = compute_group_stats(df_global)
    stats_d = compute_group_stats(df_dataset)

    # collect all present group labels across both views
    present = list(set(stats_g.keys()) | set(stats_d.keys()))

    # define reverse ordering templates for known grouping types
    rev_seg = SEGMENTS[::-1]
    rev_str = STRAINS[::-1]
    rev_dat = DATASETS[::-1]

    # determine display order based on known category sets
    if all(n in rev_seg for n in present):
        ordered = [s for s in rev_seg if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in rev_str for n in present):
        ordered = [s for s in rev_str if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in rev_dat for n in present):
        ordered = [s for s in rev_dat if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    else:
        desired = sorted(present)

    # align percentages and group sizes to the chosen display order
    pct_g, pct_d, n_g, n_d = [], [], [], []
    for name in desired:
        pg, ng = stats_g.get(name, (0.0, 0))
        pd_, nd_ = stats_d.get(name, (0.0, 0))
        pct_g.append(pg)
        pct_d.append(pd_)
        n_g.append(ng)
        n_d.append(nd_)

    # configure plot layout
    plt.style.use("seaborn-darkgrid")
    num_groups = len(desired)
    y_positions = np.arange(num_groups)

    bar_height = 0.35
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=(12, max(6, num_groups * 0.4)))

    # define bar colors for both compared views
    color_global = COLORS[0]
    color_dataset = COLORS[8]

    # plot global intersection percentages
    ax.barh(
        y_positions - offset,
        pct_g,
        height=bar_height,
        label="global",
        edgecolor="white",
        linewidth=0.5,
        color=color_global,
    )

    # plot dataset-wise intersection percentages
    ax.barh(
        y_positions + offset,
        pct_d,
        height=bar_height,
        label="dataset-wise",
        edgecolor="white",
        linewidth=0.5,
        color=color_dataset,
    )

    # create y-axis labels including sample sizes of both views
    y_labels = [
        f"{name} (n_g={ng}, n_d={nd})"
        for name, ng, nd in zip(desired, n_g, n_d)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    # set axis labels
    ax.set_xlabel("intersects within candidates (%)")
    ax.set_ylabel(f"{y_feature_name}s")

    # total row counts for title annotation
    N_g = int(len(df_global))
    N_d = int(len(df_dataset))

    # construct plot title
    title = f"comparison of intersects per {y_feature_name}"
    title += make_candidate_descriptor("all", data, strain, segment, intersects, True)
    title += f" (n_glo={N_g}, n_dat={N_d})"

    # optionally show title
    if SHOW_TITLE:
        ax.set_title(title)
    else:
        ax.set_title("")

    # optionally show legend outside plot area
    if SHOW_LEGEND:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=True,
            title=make_legend_descriptor(y_feature_name),
        )

    # adjust layout to avoid clipping
    plt.tight_layout()

    # construct output path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    outname = fname + ".png"
    plt.savefig(os.path.join(save_path, outname), dpi=300, bbox_inches="tight")
    plt.close()

def create_advandced_intersect_plot(
    df_proc: pd.DataFrame,
    df_raw: pd.DataFrame,
    y_feature_name: str,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
):
    '''
        Create a horizontal bar plot comparing intersection percentages between processed and raw dataframes.
        :param df_proc: processed dataframe containing ikey and grouping feature
        :param df_raw: raw dataframe containing ikey and grouping feature
        :param y_feature_name: feature used to define bar groups on the y-axis
        :param fname: output file name without extension
        :param folder: main result folder and processed data label
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # check that both dataframes contain required columns
    required = {"ikey", y_feature_name}
    missing_proc = required - set(df_proc.columns)
    missing_raw = required - set(df_raw.columns)
    if missing_proc:
        raise ValueError(f"df_proc missing required columns: {sorted(missing_proc)}")
    if missing_raw:
        raise ValueError(f"df_raw missing required columns: {sorted(missing_raw)}")

    # normalize grouping column and replace missing values with display label
    def normalize_group_col(dfx: pd.DataFrame) -> pd.DataFrame:
        out = dfx.copy()
        out[y_feature_name] = out[y_feature_name].astype(object)
        out[y_feature_name] = out[y_feature_name].where(out[y_feature_name].notna(), "NA")
        return out

    df_proc_n = normalize_group_col(df_proc)
    df_raw_n  = normalize_group_col(df_raw)

    # compute percentage of rows with non-unique ikeys within one group
    def pct_nonunique_within_group(g: pd.DataFrame) -> float:
        n = len(g)
        if n == 0:
            return 0.0
        dup_mask = g.duplicated(subset=["ikey"], keep=False)
        return 100.0 * float(dup_mask.sum()) / float(n)

    # compute per-group intersection statistics for a dataframe
    def compute_group_stats(dfx: pd.DataFrame) -> dict:
        # group -> (pct_nonunique, n_rows)
        stats = {}
        if dfx is None or len(dfx) == 0:
            return stats
        for group_name, g in dfx.groupby(y_feature_name, dropna=False):
            stats[group_name] = (pct_nonunique_within_group(g), int(len(g)))
        return stats

    # compute statistics for processed and raw dataframes
    stats_p = compute_group_stats(df_proc_n)
    stats_r = compute_group_stats(df_raw_n)

    # collect all present group labels across both dataframes
    present = list(set(stats_p.keys()) | set(stats_r.keys()))

    # define reverse ordering templates for known grouping types
    rev_seg = SEGMENTS[::-1]
    rev_str = STRAINS[::-1]
    rev_dat = DATASETS[::-1]

    # determine display order based on known category sets
    if all(n in rev_seg for n in present):
        ordered = [s for s in rev_seg if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in rev_str for n in present):
        ordered = [s for s in rev_str if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in rev_dat for n in present):
        ordered = [s for s in rev_dat if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    else:
        desired = sorted(present, key=lambda x: str(x).casefold())

    # optionally convert internal group names to prettier display labels
    def format_group_label(name: str) -> str:
        if y_feature_name == "strain":
            # convert internal code -> official name
            try:
                return get_official_strain_name(name)
            except Exception:
                return str(name)  # fallback if something unexpected appears
        return str(name)

    # align percentages and group sizes to the chosen display order
    pct_p, pct_r, n_p, n_r = [], [], [], []
    for name in desired:
        pp, np_ = stats_p.get(name, (0.0, 0))
        pr, nr_ = stats_r.get(name, (0.0, 0))
        pct_p.append(pp)
        pct_r.append(pr)
        n_p.append(np_)
        n_r.append(nr_)

    # configure plot layout
    plt.style.use("seaborn-darkgrid")
    num_groups = len(desired)
    y_positions = np.arange(num_groups)

    bar_height = 0.35
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=(16, max(6, num_groups * 0.4)))

    # define bar colors for processed and raw data
    color_proc = COLORS[0]
    color_raw  = COLORS[8]

    # plot processed dataframe intersection percentages
    ax.barh(
        y_positions - offset,
        pct_p,
        height=bar_height,
        label=capitalize_first(folder),
        edgecolor="white",
        linewidth=0.5,
        color=color_proc,
    )

    # plot raw dataframe intersection percentages
    ax.barh(
        y_positions + offset,
        pct_r,
        height=bar_height,
        label="Raw",
        edgecolor="white",
        linewidth=0.5,
        color=color_raw,
    )

    # create y-axis labels including sample sizes of both dataframes
    y_labels = [
        f"{format_group_label(name)} (n_r={nr_}, n_{str(folder[0])}={np_})"
        for name, np_, nr_ in zip(desired, n_p, n_r)
    ]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    # set axis labels
    ax.set_xlabel(f"Intersections within {y_feature_name} (%)")
    ax.set_ylabel(capitalize_first(y_feature_name), labelpad=20)

    # total row counts for title annotation
    N_p = int(len(df_proc_n))
    N_r = int(len(df_raw_n))

    # construct plot title
    title = f"comparison of intersects per {y_feature_name}"
    title += make_candidate_descriptor("all", data, strain, segment, intersects, True)
    title += f" (n_{str(folder)}={N_p}, n_raw={N_r})"

    # optionally show title
    if SHOW_TITLE:
        ax.set_title(title)
    else:
        ax.set_title("")

    # optionally show legend outside plot area
    if SHOW_LEGEND:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=True,
            title="Data",
        )

    # adjust layout to avoid clipping
    plt.tight_layout()

    # construct output path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    outname = fname + ".png"
    plt.savefig(os.path.join(save_path, outname), dpi=300, bbox_inches="tight")
    plt.close()

def create_pooling_intersect_bar_plot(
    df_pooled: pd.DataFrame,
    df_unpooled: pd.DataFrame,
    y_feature_name: str,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str
):
    '''
        Create a horizontal bar plot comparing intersection percentages between pooled and unpooled dataframes.
        :param df_pooled: pooled dataframe containing ikey and grouping feature
        :param df_unpooled: unpooled dataframe containing ikey and grouping feature
        :param y_feature_name: feature used to define bar groups on the y-axis
        :param fname: output file name without extension
        :param folder: main result folder
        :param subfolder: result subfolder
        :param data: selected virus type or data subset
        :param strain: selected strain
        :param segment: selected segment
        :param intersects: intersect mode descriptor

        :return: None
    '''
    # check that both dataframes contain required columns
    required = {"ikey", y_feature_name}
    miss_p = required - set(df_pooled.columns)
    miss_u = required - set(df_unpooled.columns)
    if miss_p:
        raise ValueError(f"df_pooled missing required columns: {sorted(miss_p)}")
    if miss_u:
        raise ValueError(f"df_unpooled missing required columns: {sorted(miss_u)}")

    # create working copies of both dataframes
    df_p = df_pooled.copy()
    df_u = df_unpooled.copy()

    # compute percentage of rows with non-unique ikeys within one group
    def pct_nonunique_ikey_within_group(g: pd.DataFrame) -> float:
        n = len(g)
        if n == 0:
            return 0.0
        dup_mask = g.duplicated(subset=["ikey"], keep=False)
        return 100.0 * float(dup_mask.sum()) / float(n)

    # compute per-group intersection statistics for a dataframe
    def compute_group_stats(dfx: pd.DataFrame) -> dict:
        # group -> (pct_nonunique, n_rows)
        stats = {}
        if dfx is None or len(dfx) == 0:
            return stats
        for group_name, g in dfx.groupby(y_feature_name, dropna=False):
            stats[group_name] = (pct_nonunique_ikey_within_group(g), int(len(g)))
        return stats

    # compute statistics for pooled and unpooled dataframes
    stats_p = compute_group_stats(df_p)  # pooled
    stats_u = compute_group_stats(df_u)  # unpooled

    # collect all present group labels across both dataframes
    present = list(set(stats_p.keys()) | set(stats_u.keys()))

    # define reverse ordering templates for known grouping types
    rev_seg = SEGMENTS[::-1]
    rev_str = STRAINS[::-1]
    rev_dat = DATASETS[::-1]

    # determine display order based on known category sets
    if all(n in rev_seg for n in present):
        ordered = [s for s in rev_seg if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in rev_str for n in present):
        ordered = [s for s in rev_str if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    elif all(n in rev_dat for n in present):
        ordered = [s for s in rev_dat if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        desired = ordered + remaining
    else:
        desired = sorted(present)

    # align percentages and group sizes to the chosen display order
    pct_p, pct_u, n_p, n_u = [], [], [], []
    for name in desired:
        pp, np_ = stats_p.get(name, (0.0, 0))
        pu, nu_ = stats_u.get(name, (0.0, 0))
        pct_p.append(pp)
        pct_u.append(pu)
        n_p.append(np_)
        n_u.append(nu_)

    # configure plot layout
    plt.style.use("seaborn-darkgrid")
    num_groups = len(desired)
    y_positions = np.arange(num_groups)

    bar_height = 0.35
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=(12, max(6, num_groups * 0.4)))

    # define bar colors for pooled and unpooled data
    color_pooled = COLORS[0]
    color_unpooled = COLORS[8]

    # plot pooled intersection percentages
    ax.barh(
        y_positions - offset,
        pct_p,
        height=bar_height,
        label="pooled",
        edgecolor="white",
        linewidth=0.5,
        color=color_pooled,
    )

    # plot unpooled intersection percentages
    ax.barh(
        y_positions + offset,
        pct_u,
        height=bar_height,
        label="unpooled",
        edgecolor="white",
        linewidth=0.5,
        color=color_unpooled,
    )

    # create y-axis labels including sample sizes of both dataframes
    y_labels = [
        f"{name} (n_p={np_}, n_u={nu_})"
        for name, np_, nu_ in zip(desired, n_p, n_u)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    # set axis labels
    ax.set_xlabel("intersects within candidates (%)")
    ax.set_ylabel(f"{make_legend_descriptor(y_feature_name)}s")

    # total row counts for title annotation
    N_p = int(len(df_p))
    N_u = int(len(df_u))

    # construct plot title
    title = f"comparison of intersects per {make_legend_descriptor(y_feature_name)}"
    title += make_candidate_descriptor("all", data, strain, segment, intersects)
    title += f" (n_pool={N_p}, n_unpool={N_u})"

    # optionally show title
    if SHOW_TITLE:
        ax.set_title(title)
    else:
        ax.set_title("")

    # optionally show legend outside plot area
    if SHOW_LEGEND:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=True,
            title=make_legend_descriptor(y_feature_name),
        )

    # adjust layout to avoid clipping
    plt.tight_layout()

    # construct output path and create directories if needed
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    # save figure and close it
    outname = fname + ".png"
    plt.savefig(os.path.join(save_path, outname), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    })

    #################
    ### SELECTION ###
    #################

    ### REQUIRES CORESPONDING PREPROCESSING! ###

    ### order by ... ###
    selector = 'dataset'

    ### point of maximum curvature ###
    curvature = 'right'

    ### show quantiles for distribution ###
    show_quantiles= 1

    ### "pooled" or "unpooled" ###
    folder = 'pooled'

    ### "primary" or "secondary" ###
    subfolder = 'secondary'

    ### "all", "IAV", "IBV", dataset name, ... ###
    data = 'IAV'

    ### "all", "PR8", "Yamagata", ... ###
    strain = 'PR8'

    ### "all", "PB1", "PB2", ... ###
    segment = 'PB1'

    ### modifier to manage intersects ###
    intersects = 'mean_dataset_0'

    dfnames = get_dataset_names(DATASET_CUTOFF, data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### special ###

    # run_site_motif_heatmap_analysis_len3_by_segment(dfs, TOP_N, folder, data, strain, intersects)
    # run_site_motif_len3_per_segment(dfs, TOP_N, folder, data, strain, intersects)

    ### heatmaps ###

    # run_site_motif_heatmap_analysis(dfs, TOP_N, folder, data, strain, segment, intersects)
    # run_reg_site_motif_heatmap_analysis(dfs, TOP_N, folder, data, strain, segment, intersects)
    # run_repeat_heatmap_analysis(dfs, folder, data, strain, segment, intersects)
    # run_mfe_heatmap_analysis(dfs, folder, data, strain, segment, intersects)

    ### distribution ###

    run_ngs_single_density_analysis(dfs, folder, data, strain, segment, intersects, show_quantiles, curvature)
    run_ngs_multi_density_analysis(dfs, selector, folder, data, strain, segment, intersects)
    run_advandced_intersect_plot(dfs, selector, data, strain, segment, intersects)

    ### features ###

    run_delvg_pri_features_analysis(dfs, selector, TOP_N, folder, data, strain, segment, intersects)
    # run_delvg_sec_features_analysis(dfs, selector, folder, data, strain, segment, intersects)
    # run_delvg_hybrid_features_analysis(dfs, selector, folder, data, strain, segment, intersects)
    # run_length_mfe_analysis(dfs, selector, folder, data, strain, segment, intersects)

    ### other ###

    # run_sec_structure_plot(dfs, folder, data, strain, segment, intersects)
