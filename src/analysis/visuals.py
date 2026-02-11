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
import re

from scipy import stats
from scipy.interpolate import CubicSpline

from scipy.stats import spearmanr

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score

import RNA

sys.path.insert(0, "..")

from utils import get_dataset_names

from utils import load_all_preprocessed
from utils import manage_separate_specifiers, clean_data_string
from utils import rename_feature, split_by_threshold, add_feature_quantile_rank, get_feature_modification_name
from utils import add_ikey, add_metadata_ikey, add_intersect_ngs_features, remove_by_ngs_cutoff, get_threshold

from utils import make_candidate_descriptor, make_pseudo_candidate_descriptor, make_legend_descriptor, pick_colors
from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df
from utils import compute_quantile_rank_count_df, compute_seq_feature_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_count_heatmap_sum_df, compute_feature_freq_heatmap_df

from utils import p_to_stars, fisher_exact_for_category, bh_fdr

from utils import DATAPATH, RESULTSPATH, SEED, DATASET_CUTOFF, PSEUDO_DATASETS, DATASET_STRAIN_DICT, CUTOFF, STRAINS, SEGMENTS, DATASETS
from utils import COLORS, RANK_THRESHOLD, DECIMALS, TOP_N, K_MER_LENGTH, PALINDROMIC_K_MER_LENGTH, MIN_TRACT_LENGTH, DIRECT_REPEAT_LENGTH_CAP, MAX_MOTIF_LENGTH

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'visuals')


###############
### scripts ###
###############

### heatmap ###

def run_site_motif_heatmap_analysis(dfs: list, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    max_motif_length = MAX_MOTIF_LENGTH

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    plot_names = ['motif before deletion', 'deletion start motif', 'deletion end motif', 'motif after deletion']

    all_mean_freq_diffs = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_freq_diffs  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        delvg_count = df_copy.shape[0]
        mean_freq_diffs, top_freq_diffs = make_site_motif_heatmap_analysis(motif_df, plot_names, delvg_count, motif_length, top_n, folder, data, strain, segment, intersects)

        for i in range(0, 4):
            all_mean_freq_diffs[i][motif_length - 1] = mean_freq_diffs[i]
            all_top_freq_diffs[i][motif_length - 1] = top_freq_diffs[i]

        print(f'motif_length {motif_length} completed')

    create_spline_plot(f'mean frequency difference (site - sequence) of motif lengths - spline plot', 'length of motifs (nucleotides)', 'mean frequency difference (%)', plot_names, all_mean_freq_diffs, delvg_count, 'site_mean_freq_diff', folder, 'spline', data, strain, segment, intersects)
    create_spline_plot(f'top frequency difference (site - sequence) of motif lengths  - spline plot', 'length of motifs (nucleotides)', 'top frequency difference (%)', plot_names, all_top_freq_diffs, delvg_count, 'site_top_freq_diff', folder, 'spline', data, strain, segment, intersects)

def run_reg_site_motif_heatmap_analysis(dfs: list, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    max_motif_length = MAX_MOTIF_LENGTH

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    plot_names = ['motif before deletion', 'deletion start motif', 'deletion end motif', 'motif after deletion']

    all_mean_coefficients = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_coefficients  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        delvg_count = df_copy.shape[0]
        mean_coefficients, top_coefficients = make_reg_site_motif_heatmap_analysis(motif_df, plot_names, delvg_count, motif_length, top_n, folder, data, strain, segment, intersects)

        for i in range(0, 4):
            all_mean_coefficients[i][motif_length - 1] = mean_coefficients[i]
            all_top_coefficients[i][motif_length - 1] = top_coefficients[i]

        print(f'motif_length {motif_length} completed')

    create_spline_plot(f'mean coefficient (regressive) for motif lengths - spline plot', 'length of motifs (nucleotides)', 'mean coefficient (-)', plot_names, all_mean_coefficients, delvg_count, 'site_mean_coefficients', folder, 'spline', data, strain, segment, intersects)
    create_spline_plot(f'top coefficient (regressive) for motif lengths - spline plot', 'length of motifs (nucleotides)', 'top coefficient (-)', plot_names, all_top_coefficients, delvg_count, 'site_top_coefficients', folder, 'spline', data, strain, segment, intersects)

def run_repeat_heatmap_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    delvg_count = df.shape[0]
    heatmap_name = 'direct repeat length'

    make_repeat_heatmap_analysis(df, heatmap_name, delvg_count, folder, data, strain, segment, intersects)

def run_mfe_heatmap_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')
    delvg_count = df.shape[0]
    heatmap_name = 'MFE'

    make_mfe_heatmap_analysis(df, heatmap_name, delvg_count, folder, data, strain, segment, intersects)

### density and histo ###

def run_ngs_single_density_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str, show_quantiles: str, curvature: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    create_single_density_plot('NGS read count distribution via KDE with statistical moments - density plot', df, x_feature_name, x_axis_name, show_quantiles, 'single_ngs', folder, 'density', data, strain, segment, intersects, curvature)

def run_ngs_multi_density_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    if data == 'all':
        split_feature = 'dataset'
    elif strain == 'all':
        split_feature = 'strain'
    else:
        split_feature = 'segment'

    split_values = sorted(df[split_feature].dropna().unique().tolist())
    dfs = [df[df[split_feature] == val].copy() for val in split_values]
    df_names = [str(val) for val in split_values]

    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    create_multi_density_plot(f'{split_feature}-wise NGS read count distribution via KDE - density plot', dfs, df_names, x_feature_name, x_axis_name, 'ngs_multi', "visuals", folder, 'density', data, strain, segment, intersects)

def run_intersect_analysis(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)

    feature_name = 'NGS_read_count'

    new_dfs = []

    for df in dfs:
        cutoff = None
        m = re.search(r'_(\d+)$', intersects)
        if m:
            cutoff = int(m.group(1))
            df = remove_by_ngs_cutoff(df, feature_name, cutoff)
        new_dfs.append(df)

    dfs = new_dfs

    new_dfs = []
    for df in dfs:
        if 'metadata' in intersects:
            df = add_metadata_ikey(df)
        else:
            df = add_ikey(df)
        new_dfs.append(df)

    dfs = new_dfs
    df = pd.concat(dfs, ignore_index=True)
    
    if (data != 'all') and (strain != 'all') and (segment != 'all'):
        print('use at leats one non-all specifier')
        return
    elif data == 'all':
        y_feature_name = 'subtype'
    elif strain == 'all':
        y_feature_name = 'strain'
    else:
        y_feature_name = 'segment'

    create_intersect_bar_plot(df, y_feature_name, 'intersects', folder, 'intersects', data, strain, segment, intersects)

def run_pooling_intersect_analysis(selector: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfnames = get_dataset_names(DATASET_CUTOFF, data)

    poo_dfs = load_all_preprocessed(dfnames, 'pooled', 'primary')
    poo_dfs = manage_separate_specifiers(poo_dfs, data, strain, segment)
    poo_df = add_intersect_ngs_features(poo_dfs, intersects)
    poo_df = add_ikey(poo_df)
    print('loaded pooled')

    unp_dfs = load_all_preprocessed(dfnames, 'unpooled', 'primary')
    unp_dfs = manage_separate_specifiers(unp_dfs, data, strain, segment)
    unp_df = add_intersect_ngs_features(unp_dfs, intersects)
    unp_df = add_ikey(unp_df)
    print('loaded unpooled')


    create_pooling_intersect_bar_plot(poo_df, unp_df, selector, selector, folder, 'intersects', data, strain, segment, intersects)

### bar, violin, scatter ###

def run_delvg_pri_features_analysis(dfs: list, selector: str, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    for i, df in enumerate(dfs):
        print(i, len(df))
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    num_features = [
        ("start", "deletion start position", "position where deletion begins (nucleotide)", "length"),
        ("end", "deletion end position", "position where deletion ends (nucleotide)", "length"),

        # ("full_seq_length", "full sequence length", "length of the full reference sequence (nucleotides)", "length"),
        ("DelVG_length", "DelVG sequence length", "length of the DelVG sequence (nucleotides)", "length"),
        ("deletion_length", "deletion length", "length of the deleted fragment (nucleotides)", "length"),

        ("5_end_length", "5′ end length", "length of the retained 5′ end fragment (nucleotides)", "length"),
        ("3_end_length", "3′ end length", "length of the retained 3′ end fragment (nucleotides)", "length"),

        ("direct_repeat_length", f"direct repeat length (cap={DIRECT_REPEAT_LENGTH_CAP})", "length of direct repeat (nucleotides)", "repeats"),

        ("GC_content", "GC content", "content of G and C (1/nucleotide)", "composition"),
        ("AU_content", "AU content", "content of A and U (1/nucleotide)", "composition"),
        ("UpA_content", "UpA content", "content of UpA (1/dinucleotide)", "dinucleotides"),
        ("CpG_content", "CpG content", "content of CpG (1/dinucleotide)", "dinucleotides"),

        ("GC_skew", "GC skew", "relative skew between G and C nucleotides", "composition"),
        ("sequence_entropy", "sequence entropy", "Shannon entropy of nucleotide composition", "entropy"),

        # ("poly_U_max_run", "poly-U maximum run length", "longest consecutive run of U (nucleotides)", "length"),
        ("poly_U_tracts", f"poly-U tract (min_length={MIN_TRACT_LENGTH}) count", "number of U tracts (nucleotides)", "length"),
        # ("poly_A_max_run", "poly-A maximum run length", "longest consecutive run of A (nucleotides)", "length"),
        ("poly_A_tracts", f"poly-A tract (min_length={MIN_TRACT_LENGTH}) count", "number of A tracts (nucleotides)", "length"),

        ("palindrome_density", f"palindromic k-mer (k={PALINDROMIC_K_MER_LENGTH}) density", "density of palindromic k-mers (1/k-mer)", "motifs"),

        # ("longest_ORF_len", "longest ORF length", "length of longest open reading frame (nucleotides)", "ORF"),
        # ("ORF_count", "ORF count ≥20 AA", "number of ORFs at least 20 amino acids long", "ORF"),

        ("kmer_richness", f"k-mer (k={K_MER_LENGTH}) richness", "content of unique k-mers among all windows (1/k-mer)", "complexity"),
        # ("codon_usage_entropy", "codon usage entropy", "Shannon entropy of codon usage", "codon"),
    ]

    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    if data == 'all':
        y_feature_name = 'dataset'
        call_spearman = True
    elif strain == 'all':
        y_feature_name = 'strain'
        call_spearman = True
    else:
        y_feature_name = 'segment'
        call_spearman = True

    num_feature_names = [f[0] for f in num_features]

    if call_spearman:
        make_spearman_heatmap_analysis(df, folder, y_feature_name, num_feature_names, 'primary', data, strain, segment, intersects)

    print(f'spearman made successfully')

    subfolder = 'scatter'

    feature_index = 0
    for y_feature_name, y_feature_title, y_axis_name, category in num_features:
            create_feature_scatter_plot(f'{y_feature_title} as a function of NGS read count - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, 50, False, "", 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
            feature_index += 1

    print(f'numerical features completed')

    df = add_feature_quantile_rank(df, 'norm_log_NGS_read_count', 'NGS_quantile_rank')

    # motifs #
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

    subfolder = 'bar'

    delvg_count = df.shape[0]

    cat_features = [
        ("direct_repeat_length", f"direct repeat length (cap={DIRECT_REPEAT_LENGTH_CAP})", "length of direct repeat (nucleotides)", "repeats"),

        ("host", "host species", "host species", "metadata"),
        ("system_type", "system type", "system type", "metadata"),

        ("library_layout", "library layout", "library layout", "metadata"),
        ("library_selection", "library selection", "library selection", "metadata"),
        ("library_source", "library source", "library source", "metadata"),
    ]

    if folder == "unpooled":
        cat_features.append(("time_point", "sampling time", "time point of sample collection", "metadata"))
        cat_features.append(("localization", "sample localization", "localization of sample", "metadata"))
        cat_features.append(("resolution", "sequencing resolution", "resolution of sequencing", "metadata"))
        cat_features.append(("cell_system", "host cell type", "cell type of host", "metadata"))
        cat_features.append(("MOI", "MOI", "MOI", "metadata"))
    if data == "all":
        cat_features.append(("type", "virus type", "type of virus", "metadata"))
    if strain == "all":
        cat_features.append(("strain", "virus strain", "strain of virus ", "metadata"))
        cat_features.append(("subtype", "virus subtype", "subtype of virus ", "metadata"))

    if segment == "all":
        cat_features.append(("segment", "genome segment", "segment of genome", "metadata"))

    for y_feature_name, y_feature_title, y_axis_name, category in cat_features:
        create_freq_bar_plot(f'frequency comparison of {y_feature_title} for NGS ranks with Fisher’s exact test (BH-FDR) - bar plot', df, y_feature_name, y_axis_name, delvg_count, 0, y_feature_name, folder, subfolder, data, strain, segment, intersects)

    y_feature_name = 'norm_log_NGS_read_count'
    y_axis_name = f'{modification} NGS count (reads)'

    subfolder = 'violin'

    for x_feature_name, x_feature_title, x_axis_name, category in cat_features:
        create_feature_violin_plot(f'NGS read count distributions of {x_feature_title} - violin plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, delvg_count, 0, x_feature_name, folder, subfolder, data, strain, segment, intersects)

    print(f'categorical features completed')

def run_delvg_sec_features_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    num_features = [
        ("MFE", "MFE", "MFE (kcal/mol)", "MFE"),

        ("bp_count", "base pair count", "number of base pairs", "pairing"),
        ("bp_density", "base pair density", "base pair density (1/nucleotide)", "pairing"),
        ("unpaired_count", "unpaired base count", "number of unpaired bases", "pairing"),
        ("unpaired_density", "unpaired base density", "unpaired base density (1/nucleotide)", "pairing"),

        ("stem_count", "stem count", "number of stems", "stem"),
        # ("stem_len_max", "stem length maximum", "maximum stem length (nucleotides)", "stem"),
        # ("stem_len_mean", "stem length mean", "mean stem length (nucleotides)", "stem"),
        # ("stem_len_min", "stem length minimum", "minimum stem length (nucleotides)", "stem"),

        ("hairpin_count", "hairpin count", "number of hairpin loops", "hairpin"),
        # ("hairpin_size_mean", "hairpin size mean", "mean hairpin loop size (nucleotides)", "hairpin"),
        # ("hairpin_size_min", "hairpin size minimum", "minimum hairpin loop size (nucleotides)", "hairpin"),
        # ("hairpin_size_max", "hairpin size maximum", "maximum hairpin loop size (nucleotides)", "hairpin"),

        ("external_unpaired_density", "external unpaired base density", "unpaired base density in external loop (1/nucleotide)", "pairing"),

        # ("pair_span_mean", "pair span mean", "mean base pair span (nucleotides)", "pair_span"),
        # ("pair_span_min", "pair span minimum", "minimum base pair span (nucleotides)", "pair_span"),
        # ("pair_span_max", "pair span maximum", "maximum base pair span (nucleotides)", "pair_span"),

        ("free_5prime_len", "free 5′ length", "unpaired length at 5′ end (nucleotides)", "length"),
        ("free_3prime_len", "free 3′ length", "unpaired length at 3′ end (nucleotides)", "length"),

        # ("branch_point_count", "branch point count", "count of multiloop/branch points", "composition"),

        # ("max_symmetry", "max symmetry length", "max symmetry length (nucleotides)", "composition"),
    ]

    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    if data == 'all':
        y_feature_name = 'dataset'
        call_spearman = True
    elif strain == 'all':
        y_feature_name = 'strain'
        call_spearman = True
    else:
        y_feature_name = 'segment'
        call_spearman = True

    num_feature_names = [f[0] for f in num_features]

    if call_spearman:
        make_spearman_heatmap_analysis(df, folder, y_feature_name, num_feature_names, 'secondary', data, strain, segment, intersects)

    print(f'spearman made successfully')

    subfolder = 'scatter'

    feature_index = 0
    for y_feature_name, y_feature_title, y_axis_name, category in num_features:
            create_feature_scatter_plot(f'{y_feature_title} as a function of NGS read count - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, 50, False, "", 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
            feature_index += 1

    print(f'numerical features completed')

    # subfolder = 'bar'

    # delvg_count = df.shape[0]

    # cat_features = [
    #     ("full_symmetry", "full symmetry", "full symmetry", "composition"),
    # ]

    # for y_feature_name, y_feature_title, y_axis_name, category in cat_features:
    #         create_freq_bar_plot(f'frequency comparison of {y_feature_title} for NGS ranks with Fisher’s exact test (BH-FDR) - bar plot', df, y_feature_name, y_axis_name, delvg_count, 0, y_feature_name, folder, subfolder, data, strain, segment, intersects)

    # y_feature_name = 'norm_log_NGS_read_count'
    # y_axis_name = f'{modification} NGS count (reads)'

    # subfolder = 'violin'

    # for x_feature_name, x_feature_title, x_axis_name, category in cat_features:
    #     create_feature_violin_plot(f'NGS read count distributions of {x_feature_title} - violin plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, delvg_count, 0, x_feature_name, folder, subfolder, data, strain, segment, intersects)

    # print(f'categorical features completed')

def run_delvg_hybrid_features_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

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

    modification = get_feature_modification_name()

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = f'{modification} NGS count (reads)'

    if data == 'all':
        y_feature_name = 'dataset'
        call_spearman = True
    elif strain == 'all':
        y_feature_name = 'strain'
        call_spearman = True
    else:
        y_feature_name = 'segment'
        call_spearman = True

    num_feature_names = [f[0] for f in num_features]

    if call_spearman:
        make_spearman_heatmap_analysis(df, folder, y_feature_name, num_feature_names, 'hybrid', data, strain, segment, intersects)

    print(f'spearman made successfully')

    subfolder = 'scatter'

    feature_index = 0
    for y_feature_name, y_feature_title, y_axis_name, category in num_features:
            create_feature_scatter_plot(f'{y_feature_title} as a function of NGS read count - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, 50, False, "", 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
            feature_index += 1

    print(f'numerical features completed')

def run_length_mfe_analysis(dfs: list, selector: str, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    x_feature_name = 'DelVG_length'
    x_axis_name = 'length of the defective viral genome (nucleotides)'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    print(f'features added successfully')

    subfolder = 'scatter'

    features = [
        ("MFE", "MFE", "MFE (kcal/mol)", "other"),
    ]

    for y_feature_name, y_feature_title, y_axis_name, category in features:
        create_feature_scatter_plot(f'{y_feature_title} as a function of NGS read count - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, 50, False, "", 0.0, False, 0.0, y_feature_name, 'visuals', folder, subfolder, data, strain, segment, intersects)
    
    print(f'numerical features completed')

### prediction ###

def run_pred_analysis(selector: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    num_features = [
        ("start", "deletion start position", "position where deletion begins (nucleotide)", "length"),
        ("end", "deletion end position", "position where deletion ends (nucleotide)", "length"),

        # ("full_seq_length", "full sequence length", "length of the full reference sequence (nucleotides)", "length"),
        ("DelVG_length", "DelVG sequence length", "length of the DelVG sequence (nucleotides)", "length"),
        ("deletion_length", "deletion length", "length of the deleted fragment (nucleotides)", "length"),

        ("5_end_length", "5′ end length", "length of the retained 5′ end fragment (nucleotides)", "length"),
        ("3_end_length", "3′ end length", "length of the retained 3′ end fragment (nucleotides)", "length"),
    ]

    dfnames = PSEUDO_DATASETS
    pseudo_prefix = ['homopolymer', 'tandem repeat', 'synthetic']
    subfolder = 'prediction'
    modification = get_feature_modification_name()

    bin_dfs = load_all_preprocessed(dfnames, "pseudo", "bin_prediction", data, strain, segment, intersects)
    reg_dfs = load_all_preprocessed(dfnames, "pseudo", "reg_prediction", data, strain, segment, intersects)

    dec_threshold = get_threshold("dec", "unpooled", data, strain, segment, intersects)

    prefix_index = 0
    for dfname, bin_df in zip(dfnames, bin_dfs):
        y_feature_name = 'cnn_pred_proba'
        y_axis_name = f'predicted probabilty for high NGS read count'
        for x_feature_name, x_feature_title, x_axis_name, category in num_features:
            create_feature_scatter_plot(f'bin. CNN: {x_feature_title} against predicted probabilty - scatter plot', bin_df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, False, 50, False, pseudo_prefix[prefix_index], dec_threshold, False, 0.0, f'bin_{dfname}_{x_feature_name}', 'visuals', folder, subfolder, data, strain, segment, intersects)
        prefix_index += 1

    prefix_index = 0
    for dfname, reg_df in zip(dfnames, reg_dfs):
        y_feature_name = 'cnn_pred_value'
        y_axis_name = f'predicted value ({modification} NGS read count)'
        for x_feature_name, x_feature_title, x_axis_name, category in num_features:
            create_feature_scatter_plot(f'reg. CNN: {x_feature_title} against predicted value - scatter plot', reg_df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, False, 50, True, pseudo_prefix[prefix_index], 0.0, False, 0.0, f'reg_{dfname}_{x_feature_name}', 'visuals', folder, subfolder, data, strain, segment, intersects)
        prefix_index += 1

    print(f'numerical features completed')

### other ###

def run_sec_structure_plot(dfs: list, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    sequences = df['DelVG_sequence'].head(5)

    save_path = os.path.join(RESULTSPATH, folder, 'structure', data, strain, segment, intersects)
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

### heatmap ###

def make_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, delvg_count: int, motif_length: int, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    full_seq_motif_freq_df = compute_full_seq_motif_freq_df(motif_length, data, strain, segment)
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    mean_freq_diffs = []
    top_freq_diffs = []
    for i in range(0, 4):
        motif_count_df = compute_feature_count_df(df, motif_sites[i], motifs)
        motif_freq_df = compute_feature_freq_df(motif_count_df, motif_sites[i])
        motif_freq_df = rename_feature(motif_freq_df, motif_sites[i], 'motif')
        comb_freq_df = subtract_freq_dfs('site', motif_freq_df, 'full_sequence', full_seq_motif_freq_df, 'motif')

        comb_freq_df = comb_freq_df.sort_values(by='difference', ascending=False).head(top_n)
        mean_freq_diff = round(comb_freq_df['difference'].mean(), DECIMALS)
        mean_freq_diffs.append(mean_freq_diff)
        top_freq_diff = round(comb_freq_df['difference'].iloc[0], DECIMALS)
        top_freq_diffs.append(top_freq_diff)
        filtered_motifs = comb_freq_df['motif'].tolist()

        bar_name = 'frequency comparison of '
        bar_name += heatmap_names[i]
        bar_name += f' (length={motif_length}) - bar plot'

        bar_name_add = ''
        # bar_name_add += f'\nmean frequency difference = {mean_freq_diff}'
        # bar_name_add += f', top frequency difference = {top_freq_diff}'

        create_freq_diff_plot(bar_name, bar_name_add, 'motif', f'{heatmap_names[i]} (top {min(4 ** motif_length, top_n)})', comb_freq_df, delvg_count, motif_sites[i], folder, 'diff', data, strain, segment, intersects, f'motif_length_{motif_length}')

        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, quantile_rank_count_df)

        heatmap_name = 'qunatile-wise frequency of '
        heatmap_name += heatmap_names[i]
        heatmap_name += f' (length={motif_length}) ordered by frequency difference - heatmap'

        heatmap_name_add = ''
        # heatmap_name_add += f'\nmean frequency difference = {mean_freq_diff}'
        # heatmap_name_add += f', top frequency difference = {top_freq_diff}'

        create_freq_heatmap_plot(heatmap_name, heatmap_name_add, 'motif', f'{heatmap_names[i]} (top {min(4 ** motif_length, top_n)})', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', motif_freq_heatmap_df, motif_count_df, quantile_rank_count_df, delvg_count, motif_sites[i], folder, 'heatmap', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_freq_diffs, top_freq_diffs

def make_reg_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, delvg_count: int, motif_length: int, top_n: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    mean_coefficients = []
    top_coefficients = []
    for i in range(0, 4):
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, quantile_rank_count_df)
        reg_motif_freq_heatmap_df = add_lin_reg_rows(motif_freq_heatmap_df)

        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_df.T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(top_n)
        mean_coefficient = round(trans_reg_motif_freq_heatmap_df["coefficient"].mean(), DECIMALS)
        mean_coefficients.append(mean_coefficient)
        top_coefficient = round(trans_reg_motif_freq_heatmap_df["coefficient"].iloc[0], DECIMALS)
        top_coefficients.append(top_coefficient)
        filtered_motifs = trans_reg_motif_freq_heatmap_df.index.tolist()

        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, quantile_rank_count_df)

        heatmap_name = 'qunatile-wise frequency of '
        heatmap_name += heatmap_names[i]
        heatmap_name += f' (length={motif_length}) ordered by coefficient - heatmap'

        heatmap_name_add = ''
        # heatmap_name_add += f'\nmean coefficient = {mean_coefficient}'
        # heatmap_name_add += f', top coefficient = {top_coefficient}'

        create_freq_heatmap_plot(heatmap_name, heatmap_name_add, 'motif', f'{heatmap_names[i]} (top {min(4 ** motif_length, top_n)})', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', motif_freq_heatmap_df, motif_count_df, quantile_rank_count_df, delvg_count, 'reg_' + motif_sites[i], folder, 'heatmap', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_coefficients, top_coefficients

def make_repeat_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, delvg_count: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    repeat_count_heatmap_df = compute_feature_count_heatmap_df(df, 'direct_repeat_length')
    repeat_count_df = compute_feature_count_heatmap_sum_df(repeat_count_heatmap_df, 'direct_repeat_length')
    repeat_freq_heatmap_df = compute_feature_freq_heatmap_df(repeat_count_heatmap_df, quantile_rank_count_df)

    heatmap_name = 'qunatile-wise frequency of '
    heatmap_name += heatmap_name
    heatmap_name += f' (cap={DIRECT_REPEAT_LENGTH_CAP}) - heatmap '

    create_freq_heatmap_plot(heatmap_name, '', 'direct_repeat_length', 'length of direct repeat (nucleotides)', 'NGS_quantile_rank', 'NGS rank (quantile)', 'relative frequency (%)', repeat_freq_heatmap_df, repeat_count_df, quantile_rank_count_df, delvg_count, 'direct_repeat_length', folder, 'heatmap', data, strain, segment, intersects)

def make_mfe_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, delvg_count: int, folder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = add_feature_quantile_rank(df, 'MFE', 'MFE_quantile_rank')
    df = rename_feature(df, 'MFE_quantile_rank', 'MFE_rank')

    quantile_rank_count_df = compute_quantile_rank_count_df(df)

    mfe_count_heatmap_df = compute_feature_count_heatmap_df(df, 'MFE_rank')
    mfe_count_df = compute_feature_count_heatmap_sum_df(mfe_count_heatmap_df, 'MFE_rank')
    mfe_freq_heatmap_df = compute_feature_freq_heatmap_df(mfe_count_heatmap_df, quantile_rank_count_df)

    heatmap_name = 'qunatile-wise frequency of '
    heatmap_name += heatmap_name
    heatmap_name += f' rank (quantile) - heatmap '

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

    '''
    delvg_count = df.shape[0]

    # counts for axis labels
    rank_count_df = (
        df.groupby(y_feature_name, dropna=False)
          .size()
          .reset_index(name='count')
    )

    count_df = pd.DataFrame({
        'feature': x_features,
        'count': [int(df[f].notna().sum()) if f in df.columns else 0 for f in x_features]
    })

    # spearman per y-group
    y_order = list(rank_count_df[y_feature_name].to_numpy())

    rho_mat = pd.DataFrame(index=y_order, columns=x_features, dtype=float)
    p_mat   = pd.DataFrame(index=y_order, columns=x_features, dtype=float)

    for y_val in y_order:
        df_sub = df.loc[df[y_feature_name] == y_val].copy()

        for feat in x_features:
            if feat not in df_sub.columns:
                rho_mat.loc[y_val, feat] = np.nan
                p_mat.loc[y_val, feat] = np.nan
                continue

            x = df_sub[feat]
            y = df_sub['norm_log_NGS_read_count']

            mask = x.notna() & y.notna()
            if int(mask.sum()) < 3:
                rho_mat.loc[y_val, feat] = np.nan
                p_mat.loc[y_val, feat] = np.nan
                continue

            rho, pval = spearmanr(x[mask].to_numpy(), y[mask].to_numpy())
            rho_mat.loc[y_val, feat] = float(rho) if rho is not None else np.nan
            p_mat.loc[y_val, feat] = float(pval) if pval is not None else np.nan

    # BH-FDR across ALL tests
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

    # "rho_rounded [stars]"
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
        """
        Convert star string to an integer score.
        Examples: '****'->4, '***'->3, '**'->2, '*'->1, 'ns'/'NA'/''->0
        """
        if stars is None:
            return 0
        s = str(stars).strip()
        if s == '' or s.lower() in ['na', 'nan']:
            return 0
        # keep only '*' characters
        return int(s.count('*'))

    # sum star scores per feature column using adjusted p-values (adj_p_mat)
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

    # stable order: sort by (-star_sum, original_index)
    orig_idx = {feat: i for i, feat in enumerate(x_features)}
    x_features_sorted = sorted(
        list(x_features),
        key=lambda f: (-star_sums.get(f, 0), orig_idx.get(f, 10**9))
    )

    # apply new order everywhere (keep everything consistent)
    rho_mat = rho_mat.reindex(columns=x_features_sorted)
    p_mat = p_mat.reindex(columns=x_features_sorted)
    adj_p_mat = adj_p_mat.reindex(columns=x_features_sorted)
    freq_heatmap_df = freq_heatmap_df.reindex(columns=x_features_sorted)

    count_df = count_df.set_index('feature').reindex(x_features_sorted).reset_index()

    # update x_features variable for downstream label usage
    x_features = x_features_sorted

    if fname == 'primary':
        plot_name_add = f'\nk-mer (k={K_MER_LENGTH}, k_palindromic={PALINDROMIC_K_MER_LENGTH}), tracts (min_length={MIN_TRACT_LENGTH})'
    else:
        plot_name_add = ''

    heatmap_name = f'{make_legend_descriptor(y_feature_name)}-wise Spearman correlations (NGS read count and {fname} feature) ordered by significance (BH-FDR) - heatmap'

    create_freq_heatmap_plot(
        heatmap_name,
        plot_name_add,
        'feature',
        f'{fname} feature',
        y_feature_name,
        make_legend_descriptor(y_feature_name),
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
    freq_heatmap_df:
        rows = y categories (NGS_quantile_rank, Strain, Segment, ...)
        cols = x categories (x_feature_name)
        values = relative frequency (%) or any numeric/statistic

    count_df:
        columns: [x_feature_name, 'count']

    rank_count_df:
        columns: [y_feature_name, 'count']
    '''

    pivot_df = freq_heatmap_df
    feature_order = list(pivot_df.columns)
    y_order = list(pivot_df.index)

    # determine y (row) ordering
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
    elif "motif" in y_feature_name:
        y_order = list(pd.unique(y_order))
    else:
        y_order = sorted(pd.unique(y_order))

    # apply row order to the heatmap data (this reorders the y-axis)
    pivot_df = pivot_df.reindex(y_order)

    feature_count_series = count_df.set_index(x_feature_name)['count']

    if show_x_counts:
        feature_labels = [
            f'{feature}\n(n={feature_count_series.get(feature, 0)})'
            for feature in feature_order
        ]
    else:
        feature_labels = [f'{feature}' for feature in feature_order]

    # y tick labels (generalized, backward compatible)
    y_labels = []
    for y_val in y_order:
        count = rank_count_df.loc[
            rank_count_df[y_feature_name] == y_val, 'count'
        ].values
        count_str = str(int(count[0])) if len(count) > 0 else "0"
        y_labels.append(f"{y_val}\n(n={count_str})")

    if np.issubdtype(pivot_df.to_numpy().dtype, np.number):
        heatmap_values = pivot_df
        annot_values = None
        is_star_case = False
    else:
        num_part = pivot_df.astype(str).replace(
            to_replace=r'^\s*([+-]?\d+(?:\.\d+)?)\s*.*$',
            value=r'\1',
            regex=True
        )
        heatmap_values = num_part.apply(pd.to_numeric, errors='coerce')

        def format_annot(cell):
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

    # keep normal behavior for numeric heatmaps, tune only for star case
    min_width = 13
    fig_width = max(min_width, 1 + len(feature_labels) * 0.6)

    if is_star_case:
        # tuned: less height, bigger text (your chosen values)
        min_height = 6
        fig_height = max(min_height, 1 + len(y_labels) * 0.6)
        annot_fontsize = 10
        x_tick_size = 9
        y_tick_size = 9
    else:
        # original behavior
        fig_height = 1 + len(y_labels) * 0.4
        annot_fontsize = None
        x_tick_size = None
        y_tick_size = None

    plt.figure(figsize=(fig_width, fig_height))

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

    if is_star_case:
        heatmap_kwargs['annot_kws'] = {'fontsize': annot_fontsize}

    ax = sns.heatmap(**heatmap_kwargs)

    for label in ax.get_yticklabels():
        label.set_va('center')
        label.set_multialignment('center')

    if is_star_case:
        ax.tick_params(axis='x', labelsize=x_tick_size)
        ax.tick_params(axis='y', labelsize=y_tick_size)

    ax.set_xlabel(f'{x_axis_name}')
    ax.set_ylabel(f'{y_axis_name}')

    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    title_name += plot_name_add
    ax.set_title(title_name)

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, f'{feature_specifier}')
    os.makedirs(save_path, exist_ok=True)

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

    '''
    freq0 = freq_df.columns[1]
    freq1 = freq_df.columns[2]

    df_plot = pd.melt(
        freq_df,
        id_vars=x_feature_name,
        value_vars=[freq0, freq1],
        var_name='type',
        value_name='frequency'
    )

    label_map = {
        freq0: 'site of candidates',
        freq1: 'full sequence'
        }

    df_plot['type'] = df_plot['type'].replace(label_map)

    custom_colors = {
        label_map[freq0]: COLORS[8],
        label_map[freq1]: COLORS[0]
        }

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

    plt.xticks(rotation=0)
    plt.xlabel(f'{x_axis_name}')
    plt.ylabel('relative frequency (%)')

    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    title_name += plot_name_add
    plt.title(title_name)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title=make_legend_descriptor(x_feature_name),
        frameon=True
    )

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, f'{feature_specifier}')
    os.makedirs(save_path, exist_ok=True)

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

    '''
    plt.figure(figsize=(10, 6))
    num_points = len(spline_values[0])
    x = np.arange(1, num_points + 1)
    x_smooth = np.linspace(x.min(), x.max(), 300)

    n_needed = len(spline_values)
    picked_colors = pick_colors(COLORS, n_needed)

    for i, y in enumerate(spline_values):
        spline = CubicSpline(x, y)
        y_smooth = spline(x_smooth)
        color = picked_colors[i]
        plt.plot(x_smooth, y_smooth, label=spline_names[i], color=color)
        plt.scatter(x, y, color=color, s=30)

    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    plt.title(title_name)

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')
    plt.grid(True, alpha=0.3)

    plt.xticks(np.arange(1, num_points + 1))

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title=make_legend_descriptor('type'),
        frameon=True
    )

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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
    """
    KDE-only density plot (no histogram), with:
      - statistical moments computed from raw sample values
      - optional elbow point via max curvature of KDE pdf (disabled if elbow_side == "")
      - optional quantile indicator lines
      - your existing styling + save path conventions
    """

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))

    # extract + clean (numeric coercion + drop NaNs)
    s = pd.to_numeric(df[x_feature_name], errors="coerce").dropna()
    values = s.to_numpy()
    n = len(values)
    if n == 0:
        raise ValueError(f"No numeric data in column '{x_feature_name}' after dropping NaNs/coercion.")

    # moments from RAW values
    mean_val = float(np.mean(values))
    var_val = float(np.var(values, ddof=0))
    skew_val = float(stats.skew(values, bias=True))
    kurt_val = float(stats.kurtosis(values, fisher=True, bias=True))

    # KDE via scipy
    xmin = float(np.nanmin(values))
    xmax = float(np.nanmax(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError(f"No finite numeric data in column '{x_feature_name}' after cleaning.")
    if xmin == xmax:
        # constant distribution: create a tiny window so KDE evaluation is well-defined
        xmin = xmin - 0.5
        xmax = xmax + 0.5

    # optional clip
    if clip is not None:
        c0, c1 = clip
        if c0 is not None:
            xmin = max(xmin, float(c0))
        if c1 is not None:
            xmax = min(xmax, float(c1))
        if xmin == xmax:
            xmin = xmin - 0.5
            xmax = xmax + 0.5

    xs = np.linspace(xmin, xmax, int(kde_grid))

    kde = stats.gaussian_kde(values)
    if kde_bw_adjust != 1.0:
        base_cf = kde.covariance_factor()
        kde.covariance_factor = lambda: base_cf * kde_bw_adjust
        kde._compute_covariance()

    pdf = kde(xs)

    # curvature / elbow detection (optional)
    elbow_side_l = (elbow_side or "").strip().lower()
    plot_elbow = elbow_side_l != ""

    elbow_x = None
    elbow_idx = None

    if plot_elbow:
        dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
        pdf_prime = np.gradient(pdf, dx)
        pdf_double = np.gradient(pdf_prime, dx)
        curvature = np.abs(pdf_double) / np.power(1.0 + pdf_prime**2, 1.5)

        if elbow_side_l == "right":
            mask = xs >= mean_val
        elif elbow_side_l == "left":
            mask = xs <= mean_val
        elif elbow_side_l == "both":
            mask = np.ones_like(xs, dtype=bool)
        else:
            raise ValueError("elbow_side must be 'right', 'left', 'both', or '' to disable.")

        if (not np.any(mask)) or np.allclose(pdf, pdf[0]):
            elbow_x = float(mean_val)
            elbow_idx = int(np.argmin(np.abs(xs - elbow_x)))
        else:
            masked_idx = np.where(mask)[0]
            local_argmax = masked_idx[int(np.argmax(curvature[mask]))]
            elbow_idx = int(local_argmax)
            elbow_x = float(xs[elbow_idx])

    # plot KDE curve
    plt.plot(xs, pdf, color=COLORS[8], linewidth=2.0, label="density")
    # optional fill
    plt.fill_between(xs, 0, pdf, color=COLORS[8], alpha=0.15)

    # indicator lines
    plt.axvline(mean_val, color=COLORS[9], linestyle=":", linewidth=1.5, label="mean")
    if plot_elbow and elbow_x is not None:
        plt.axvline(elbow_x, color=COLORS[10], linestyle="--", linewidth=1.5, label="max. curvature")

    # quantiles (optional)
    q80 = q85 = q90 = q95 = None
    if show_quantiles:
        q80 = float(np.quantile(values, 0.80))
        # q85 = float(np.quantile(values, 0.85))
        q90 = float(np.quantile(values, 0.90))
        # q95 = float(np.quantile(values, 0.95))
        plt.axvline(q80, color=COLORS[2], linestyle="-", linewidth=1, label="80th percentile")
        # plt.axvline(q85, color=COLORS[2], linestyle="-", linewidth=1, label="85th percentile")
        plt.axvline(q90, color=COLORS[0], linestyle="-", linewidth=1, label="90th percentile")
        # plt.axvline(q95, color=COLORS[0], linestyle="-", linewidth=1, label="95th percentile")

    # title / labels
    title = f"{plot_name}"
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f" (n={n})"
    plt.title(title)

    plt.xlabel(f"{x_axis_name}")
    plt.ylabel("density (area under curve = 1)")

    # stats text box (order: moments -> quantiles -> elbow if enabled)
    stats_lines = [
        f"mean: {mean_val:.{DECIMALS}f}",
        f"variance: {var_val:.{DECIMALS}f}",
        f"skewness: {skew_val:.{DECIMALS}f}",
        f"kurtosis: {kurt_val:.{DECIMALS}f}",
    ]

    if show_quantiles:
        stats_lines += [
            "",
            f"80th percentile: {q80:.{DECIMALS}f}",
            # f"85th percentile: {q85:.{DECIMALS}f}",
            f"90th percentile: {q90:.{DECIMALS}f}",
            # f"95th percentile: {q95:.{DECIMALS}f}",
        ]

    if plot_elbow and elbow_x is not None:
        stats_lines += [
            "",
            f"max. curvature: {elbow_x:.{DECIMALS}f}",
        ]

    stats_text = "\n".join(stats_lines)

    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    plt.gca().text(
        0.80, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props
    )

    # legend outside
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        title=make_legend_descriptor('type'),
        frameon=True
    )

    # save path
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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
    Fisher exact test (BH-FDR corrected across all categories in df).
    annotates Odds Ratio (OR) above significance stars.
    '''

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 7))

    # split once
    true_df, false_df, threshold = split_by_threshold(df, 'NGS_quantile_rank', RANK_THRESHOLD)

    true_df = true_df.copy()
    false_df = false_df.copy()

    n_true, n_false = len(true_df), len(false_df)
    if n_true == 0 or n_false == 0:
        print(f"[WARNING] Fisher test skipped: one split is empty (n_true={n_true}, n_false={n_false}).")
        plt.close()
        return

    def _normalize_cats(values: pd.DataFrame, x_feature_name: str) -> pd.Series:
        cats = (
            values[x_feature_name]
            .astype(object)
            .where(values[x_feature_name].notna(), other="UNKNOWN")
            .astype(str)
        )
        values[x_feature_name] = cats
        return cats

    # apply to both splits
    cats_true = _normalize_cats(true_df, x_feature_name)
    cats_false = _normalize_cats(false_df, x_feature_name)

    true_counts  = true_df[x_feature_name].value_counts(dropna=False)
    false_counts = false_df[x_feature_name].value_counts(dropna=False)

    def pct(count, denom):
        return (100.0 * count / denom) if denom > 0 else 0.0

    x_feature_name_is_motif = ("motif" in str(x_feature_name).casefold())
    present = list(pd.unique(pd.concat([cats_true, cats_false], axis=0)))

    if x_feature_name_is_motif:
        uniq = present  # preserve as-is for motifs (no special reordering)
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

    freq_true  = {cat: pct(true_counts.get(cat, 0),  n_true)  for cat in all_cats}
    freq_false = {cat: pct(false_counts.get(cat, 0), n_false) for cat in all_cats}
    diffs      = {cat: freq_true[cat] - freq_false[cat] for cat in all_cats}

    # Fisher tests
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

    # BH-FDR correction
    p_adj_map = {}
    if len(pvals) > 0:
        p_adj = bh_fdr(pvals)
        for cat, padj in zip(cats_for_p, p_adj):
            p_adj_map[cat] = float(padj)

    # select categories to plot
    if top_n and top_n > 0:
        top_cats = sorted(all_cats, key=lambda c: diffs[c], reverse=True)[:top_n]
    else:
        top_cats = all_cats

    x = np.arange(len(top_cats))
    width = 0.42

    bars_false = [freq_false[c] for c in top_cats]
    bars_true  = [freq_true[c]  for c in top_cats]

    color_false = COLORS[0]
    color_true  = COLORS[8]

    plt.bar(
        x - width/2,
        bars_true,
        width=width,
        label=f'high NGS rank ({threshold})',
        color=color_true,
        edgecolor='white',
        linewidth=0.5
    )
    plt.bar(
        x + width/2,
        bars_false,
        width=width,
        label='lower NGS ranks (rest)',
        color=color_false,
        edgecolor='white',
        linewidth=0.5
    )

    # annotations
    y_max_overall = 0.0
    current_top = max(bars_false + bars_true) if (bars_false + bars_true) else 1.0
    y_offset = current_top * 0.03 if current_top > 0 else 1.0

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
        y_bracket = y_base + y_offset
        h_bracket = y_offset * 0.6

        x1 = i - width/2
        x2 = i + width/2

        plt.plot(
            [x1, x1, x2, x2],
            [y_bracket, y_bracket + h_bracket, y_bracket + h_bracket, y_bracket],
            color='black',
            linewidth=1.0
        )

        base = y_bracket + h_bracket

        # stars close to bracket
        stars_y  = base + (y_offset * 0.20)
        metric_y = base + (y_offset * 1.60)

        plt.text(
            (x1 + x2) / 2,
            stars_y,
            f'[{stars}]',
            ha='center',
            va='bottom',
            fontsize=9
        )

        plt.text(
            (x1 + x2) / 2,
            metric_y,
            f'OR={metric_txt}',
            ha='center',
            va='bottom',
            fontsize=8
        )

        y_max_overall = max(y_max_overall, metric_y)

    plt.ylim(0, max(current_top * 1.25, y_max_overall * 1.10))

    labels = [f"{str(c)}\n(n={true_counts.get(c, 0) + false_counts.get(c, 0)})"
              for c in top_cats]
    plt.xticks(x, labels, rotation=90, ha='center')

    plt.xlabel(x_axis_name)
    plt.ylabel('relative frequency (%)')

    title_name = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    plt.title(title_name)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title=make_legend_descriptor(selector),
        frameon=True
    )

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, feature_specifier)
    os.makedirs(save_path, exist_ok=True)

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
    """

    """
    plt.style.use('seaborn-darkgrid')
    sns.set_theme(style="whitegrid")

    df_plot = df.copy()

    df_plot[x_feature_name] = df_plot[x_feature_name].astype(object)
    df_plot[x_feature_name] = df_plot[x_feature_name].where(df_plot[x_feature_name].notna(), 'NA')

    if not np.issubdtype(df_plot[y_feature_name].dtype, np.number):
        df_plot[y_feature_name] = pd.to_numeric(df_plot[y_feature_name], errors='coerce')

    counts = df_plot[x_feature_name].value_counts(dropna=False)

    # determine x (category) ordering
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

    if top_n and top_n > 0:
        # keep top_n by frequency, but preserve the chosen ordering as best as possible
        top_set = set(counts.head(top_n).index.tolist())
        cat_order = [c for c in cat_order_full if c in top_set]
    else:
        cat_order = cat_order_full

    df_plot = df_plot[df_plot[x_feature_name].isin(cat_order)].copy()
    cat_counts = df_plot[x_feature_name].value_counts()
    tick_labels = [f"{str(c)}\n(n={int(cat_counts.get(c, 0))})" for c in cat_order]

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

    q = df_plot.groupby(x_feature_name, sort=False)[y_feature_name].quantile([0.25, 0.5, 0.75]).unstack()
    x_pos = dict(zip(cat_order, ax.get_xticks()))
    for cat in cat_order:
        if cat not in q.index:
            continue
        q1, med, q3 = q.loc[cat, [0.25, 0.5, 0.75]]
        if pd.isna(q1) or pd.isna(med) or pd.isna(q3):
            continue
        x0 = x_pos[cat]
        ax.vlines(x0, q1, q3, color='black', linewidth=2.0, zorder=3)
        ax.scatter([x0], [med], s=28, color='white', edgecolor='black', linewidth=0.7, zorder=4)

    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)

    title_name  = f'{plot_name}'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={delvg_count})'
    ax.set_title(title_name)

    ax.set_xticklabels(tick_labels, rotation=0, ha='center')

    plt.tight_layout()

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder, clean_data, strain, segment, intersects)
    if feature_specifier:
        save_path = os.path.join(save_path, f'{feature_specifier}')
    os.makedirs(save_path, exist_ok=True)

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
    Deterministic subsetting:
    - If subset_n is set, picks the same subset of rows (based on column 'ID' + global SEED)
      before applying selector coloring. This lets you rerun with a different selector and
      get the exact same points, just different colours.
    '''

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # --- deterministic subset (BEFORE selector handling) ---
    # keep behaviour identical when subset_n is None
    if subset_n is not None and subset_n > 0 and len(df) > subset_n:

        # Create stable ID if missing (junction-identity is preferred)
        if "ID" not in df.columns:
            df = df.copy()

            # try common junction identity columns (adapt list if your names differ)
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

            # fallback: build an ID from what you *do* have (still deterministic)
            if not made_id:
                fallback_cols = [
                    c for c in
                    ["dataset", "strain", "segment", "start", "end"]
                    if c in df.columns
                ]
                if len(fallback_cols) > 0:
                    df["ID"] = df[fallback_cols].astype(str).agg("_".join, axis=1)
                else:
                    # absolute last resort: deterministic based on index
                    df["ID"] = df.index.astype(str)

        # hash IDs deterministically, mix with SEED, take smallest subset_n
        _ids = df["ID"].astype(str)
        _h = pd.util.hash_pandas_object(_ids, index=False).astype("uint64")
        _h = (_h ^ np.uint64(SEED))  # mix in global SEED

        pick_idx = (
            pd.DataFrame({"_h": _h}, index=df.index)
            .sort_values("_h")
            .head(int(subset_n))
            .index
        )
        df = df.loc[pick_idx].copy()

    values = df[[x_feature_name, y_feature_name, selector]] \
        .dropna(subset=[x_feature_name, y_feature_name]) \
        .copy()
    n = len(values)

    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="UNKNOWN")
        .astype(str)
    )
    values[selector] = cats

    # determine selector order
    if selector == "segment":
        present = list(pd.unique(cats))
        ordered = [s for s in SEGMENTS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        uniq = ordered + remaining
    elif y_feature_name == "strain":
        present = list(pd.unique(cats))
        ordered = [s for s in STRAINS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        uniq = ordered + remaining
    elif y_feature_name == "dataset":
        present = list(pd.unique(cats))
        ordered = [s for s in DATASETS if s in present]
        remaining = sorted([s for s in present if s not in ordered])
        uniq = ordered + remaining
    else:
        uniq = sorted(pd.unique(cats))

    # add category sizes into legend labels: "name (n=...)"
    cat_counts = cats.value_counts(dropna=False).to_dict()
    label_map = {u: f"{u} (n={int(cat_counts.get(u, 0))})" for u in uniq}
    values[selector] = values[selector].map(label_map)

    # keep final order consistent with chosen ordering
    uniq = [label_map[u] for u in uniq]

    values = values.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

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

    # rolling median (graph element, robust trend)
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

    # identity line: y = x
    if show_identity_line and not values.empty:
        x_vals = values[x_feature_name].to_numpy(dtype=float)
        y_vals = values[y_feature_name].to_numpy(dtype=float)

        lo = np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)])
        hi = np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)])

        if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
            plt.plot(
                [lo, hi], [lo, hi],
                linestyle="--",
                linewidth=2,
                color="grey",
                label="identity"
            )

    # decision threshold
    if show_decision_threshold != 0:
        plt.axhline(
            float(show_decision_threshold),
            linestyle="--",
            linewidth=2,
            color="grey",
            label=f"decision_threshold={show_decision_threshold:.{DECIMALS}f}",
        )

    # regression metrics (predictions vs truth)
    mae = None
    mse = None
    hub = None
    r2_pred = None

    if reg_metrics and (not values.empty):
        y_true = values[x_feature_name].to_numpy(dtype=float)
        y_pred = values[y_feature_name].to_numpy(dtype=float)

        if len(y_true) > 0:
            err = y_pred - y_true

            mae = float(np.nanmean(np.abs(err)))
            mse = float(np.nanmean(err ** 2))

            # Huber
            # delta = huber_delta
            # ae = np.abs(err)
            # quad = np.minimum(ae, delta)
            # lin  = ae - quad
            # hub = float(np.nanmean(0.5 * quad ** 2 + delta * lin))

        # R² of predictions vs truth
        if len(y_true) > 1 and np.nanstd(y_true) > 0:
            ss_res = np.nansum((y_true - y_pred) ** 2)
            ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
            if ss_tot > 0:
                r2_pred = 1 - ss_res / ss_tot

    if pseudo_prefix:
        title_descriptor = make_pseudo_candidate_descriptor(pseudo_prefix, folder, data, strain, segment, intersects)
    else:
        title_descriptor = make_candidate_descriptor(folder, data, strain, segment, intersects)

    title = f'{plot_name}'
    title += title_descriptor
    title += f' (n={n})'

    if reg_metrics:
        # last line: metrics
        line = "\n"

        if mae is not None and np.isfinite(mae):
            line += f', MAE={float(mae):.{DECIMALS}f}'
        else:
            line += ', MAE=nan'

        if mse is not None and np.isfinite(mse):
            line += f', MSE={float(mse):.{DECIMALS}f}'
        else:
            line += ', MSE=nan'

        # if hub is not None and np.isfinite(hub):
        #     line += f', Huber={float(hub):.{DECIMALS}f}'
        # else:
        #     line += ', Huber=nan'

        if r2_pred is not None and np.isfinite(r2_pred):
            line += f', R²={float(r2_pred):.{DECIMALS}f}'
        else:
            line += ', R²=nan'

        title += line

    plt.title(title)

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title=make_legend_descriptor(selector),
        frameon=True
    )

    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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
    """
    Draw empirical densities (KDE) for train and test in one plot.
    Raw values only (no scaling/transforms).
    Legend is handled by seaborn (do NOT overwrite with plt.legend()).

    Adds histogram-based Jensen-Shannon distance + permutation p-value (stars).
    """

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))

    if df_list is None or len(df_list) == 0:
        raise ValueError("df_list must contain at least one dataframe.")
    if df_names is None or len(df_names) != len(df_list):
        raise ValueError("df_names must be the same length as df_list.")

    if show_js and len(df_list) != 2:
        raise ValueError("show_js=True requires exactly 2 dataframes in df_list.")

    # reorder df_list + df_names together#
    names = [str(n) for n in df_names]
    present = list(pd.unique(names))

    
    if "PB2" in names:
        legend_title = 'segment'
    elif any(x in names for x in ("PR8", "Yamagata")):
        legend_title = 'strain'
    else:
        legend_title = 'dataset'

    # decide which ordering to use based on names present
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

    # stable mapping name -> list of indices (handles duplicates safely)
    name_to_idxs = {}
    for i, n in enumerate(names):
        name_to_idxs.setdefault(n, []).append(i)

    new_idx = []
    for n in desired:
        if n in name_to_idxs:
            new_idx.extend(name_to_idxs[n])

    df_list = [df_list[i] for i in new_idx]
    df_names = [df_names[i] for i in new_idx]

    # extract + minimal cleaning (numeric coercion + drop NaNs)
    cleaned = []
    for df_i, name_i in zip(df_list, df_names):
        v = df_i[[x_feature_name]].copy()
        v[x_feature_name] = pd.to_numeric(v[x_feature_name], errors="coerce")
        v = v.dropna(subset=[x_feature_name])

        if len(v) == 0:
            raise ValueError(f"No numeric data in '{x_feature_name}' for '{name_i}' after cleaning.")

        cleaned.append(v)

    # legend names (optionally append sample sizes)
    plot_names = list(df_names)
    if show_n_in_legend:
        plot_names = [f"{name_i} (n={len(v)})" for v, name_i in zip(cleaned, df_names)]

    # similarity between distributions (histogram-based Jensen-Shannon distance)
    def jensen_shannon_distance(x1, x2, bins=50, eps=1e-12):
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()

        xmin = float(np.nanmin([np.nanmin(x1), np.nanmin(x2)]))
        xmax = float(np.nanmax([np.nanmax(x1), np.nanmax(x2)]))

        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return np.nan
        if xmin == xmax:
            return 0.0  # identical constant distributions

        # counts -> probabilities (more stable than density=True)
        h1, edges = np.histogram(x1, bins=bins, range=(xmin, xmax), density=False)
        h2, _     = np.histogram(x2, bins=edges, density=False)

        p = h1.astype(float) + eps
        q = h2.astype(float) + eps
        p = p / np.sum(p)
        q = q / np.sum(q)

        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        jsd = 0.5 * (kl_pm + kl_qm)

        return float(np.sqrt(jsd))

    # permutation p-value for JS distance
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

        # small-sample safe p-value (avoid 0)
        for _ in range(int(n_perm)):
            idx = rng.permutation(pooled.shape[0])
            a = pooled[idx[:n1]]
            b = pooled[idx[n1:n1 + n2]]
            js = jensen_shannon_distance(a, b, bins=bins)
            if np.isfinite(js) and js >= obs:
                more_extreme += 1

        p = (more_extreme + 1) / (int(n_perm) + 1)
        return obs, float(p)

    sim_js = np.nan
    js_p = np.nan
    js_stars = "NA"
    if show_js:
        sim_js, js_p = js_permutation_pvalue(
            cleaned[0][x_feature_name].to_numpy(),
            cleaned[1][x_feature_name].to_numpy(),
            bins=js_bins,
            n_perm=js_perm_n,
            seed=SEED
        )
        js_stars = p_to_stars(js_p) if np.isfinite(js_p) else "NA"

    # combine into one df with "type"
    values = pd.concat(
        [v.assign(type=name_i) for v, name_i in zip(cleaned, plot_names)],
        ignore_index=True
    )

    # colors in your style
    uniq = list(plot_names)
    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

    # KDE density curves (seaborn creates the legend)
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

    # title
    title = f"{plot_name}"
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)

    # show sizes in title
    if len(df_list) <= 2:
        title += " (" + ", ".join([f"n_{name_i}={len(v)}" for v, name_i in zip(cleaned, df_names)]) + ")"
    else:
        n_total = sum(len(v) for v in cleaned)
        title += f" (n={n_total})"

    if show_js:
        if np.isfinite(sim_js):
            title += f"\nJS-distance={sim_js:.{DECIMALS}f}"
        else:
            title += "\nJS-distance=NA"

        if np.isfinite(js_p):
            title += f" [{js_stars}]"
        else:
            title += " [NA]"

    plt.title(title)
    plt.xlabel(x_axis_name)

    if common_norm:
        plt.ylabel("density (common normalization)")
    else:
        plt.ylabel("density (area under each curve = 1)")

    # move the seaborn legend
    leg = ax.legend_
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1))
        leg._loc = 2  # "upper left"
        leg.set_title(legend_title)
        leg.set_frame_on(True)

    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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
    """
    KDE-only density plot (no histogram) for predicted class probabilities of a binary CNN,
    with a decision threshold indicator line.

    - Computes KDE from raw sample values
    - Draws a grey dashed vertical line at `decision_threshold`
    - Keeps your styling + save path conventions
    """

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))

    # extract + clean (numeric coercion + drop NaNs)
    s = pd.to_numeric(df[x_feature_name], errors="coerce").dropna()
    values = s.to_numpy()
    n = len(values)
    if n == 0:
        raise ValueError(
            f"no numeric data in column '{x_feature_name}' after dropping NaNs/coercion."
        )

    # basic bounds
    xmin = float(np.nanmin(values))
    xmax = float(np.nanmax(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError(f"no finite numeric data in column '{x_feature_name}' after cleaning.")

    # constant distribution: create a tiny window so KDE evaluation is well-defined
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5

    # optional clip
    if clip is not None:
        c0, c1 = clip
        if c0 is not None:
            xmin = max(xmin, float(c0))
        if c1 is not None:
            xmax = min(xmax, float(c1))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5

    xs = np.linspace(xmin, xmax, int(kde_grid))

    # KDE via scipy
    kde = stats.gaussian_kde(values)
    if kde_bw_adjust != 1.0:
        base_cf = kde.covariance_factor()
        kde.covariance_factor = lambda: base_cf * kde_bw_adjust
        kde._compute_covariance()

    pdf = kde(xs)

    # plot KDE curve
    plt.plot(xs, pdf, color=COLORS[6], linewidth=2.0, label="test")
    plt.fill_between(xs, 0, pdf, color=COLORS[6], alpha=0.15)

    # decision threshold line (grey dashed)
    plt.axvline(
        float(decision_threshold),
        color="grey",
        linestyle="--",
        linewidth=1.8,
        label=f"decision_threshold={decision_threshold}",
    )

    # title / labels
    title = f"{plot_name}"
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f" (n={n})"
    plt.title(title)

    plt.xlabel(f"{x_axis_name}")
    plt.ylabel("density (area under curve = 1)")

    # legend outside
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        title=make_legend_descriptor("type"),
        frameon=True,
    )

    # save path
    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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

    '''
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # keep only needed columns, drop NaNs
    needed = [y_true_feature_name, y_pred_proba_feature_name, selector]
    values = df[needed].dropna(subset=[y_true_feature_name, y_pred_proba_feature_name]).copy()
    n = len(values)

    # ensure selector is clean + consistent
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="unknown")
        .astype(str)
    )
    values[selector] = cats

    # arrays
    y_true = values[y_true_feature_name].to_numpy()
    y_proba = values[y_pred_proba_feature_name].to_numpy()

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    auc_score = round(auc_score, DECIMALS)

    # F1 at fixed global threshold
    y_pred_label = (y_proba >= f1_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    f1 = round(f1, DECIMALS)

    roc_color = COLORS[6]

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        color=roc_color
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="grey")

    title = f'{plot_name}'
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f' (n={n})'
    title += f'\nROC-AUC={auc_score}, F1={f1} (decision_threshold={f1_threshold})'
    plt.title(title)

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")

    plt.grid(True)

    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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

    '''
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # keep only needed columns, drop NaNs
    needed = [y_true_feature_name, y_pred_proba_feature_name, selector]
    values = df[needed].dropna(subset=[y_true_feature_name, y_pred_proba_feature_name]).copy()
    n = len(values)

    # ensure selector is clean + consistent
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="UNKNOWN")
        .astype(str)
    )
    values[selector] = cats

    # arrays
    y_true = values[y_true_feature_name].to_numpy()
    y_proba = values[y_pred_proba_feature_name].to_numpy()

    # PR + AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = average_precision_score(y_true, y_proba)
    pr_auc_score = round(pr_auc_score, DECIMALS)

    # F1 at fixed global threshold
    y_pred_label = (y_proba >= f1_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    f1 = round(f1, DECIMALS)

    pr_color = COLORS[6]

    plt.plot(
        recall,
        precision,
        linewidth=2,
        color=pr_color
    )

    title = f'{plot_name}'
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f' (n={n})'
    title += f'\nPR-AUC={pr_auc_score}, F1={f1} (decision_threshold={f1_threshold})'
    plt.title(title)

    plt.xlabel("recall")
    plt.ylabel("precision")

    plt.grid(True)

    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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

    '''
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # keep only needed columns, drop NaNs
    needed = [x_feature_name, y_pred_feature_name, selector]
    values = df[needed].dropna(subset=[x_feature_name, y_pred_feature_name]).copy()
    n = len(values)

    # compute residuals
    values["residual"] = values[y_pred_feature_name] - values[x_feature_name]

    # ensure selector is clean + consistent (same logic as your scatter)
    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="unknown")
        .astype(str)
    )
    values[selector] = cats
    uniq = sorted(cats.unique())

    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

    # residual scatter
    sns.scatterplot(
        data=values,
        x=x_feature_name,
        y="residual",
        hue=selector,
        palette=palette,
        edgecolor='white',
        s=50,
        alpha=0.7,
    )

    # optional: zero residual reference line
    if show_zero_line:
        plt.axhline(0, color="grey", linestyle="--", linewidth=1.5, label="zero residual")

    # regression metrics (predictions vs truth)
    mae = None
    mse = None
    hub = None
    r2  = None

    if reg_metrics and (not values.empty):
        y_true = values[x_feature_name].to_numpy(dtype=float)
        y_pred = values[y_pred_feature_name].to_numpy(dtype=float)

        if len(y_true) > 0:
            err = y_pred - y_true

            mae = float(np.nanmean(np.abs(err)))
            mse = float(np.nanmean(err ** 2))

            # # Huber
            # delta = HUBER_DELTA
            # ae = np.abs(err)
            # quad = np.minimum(ae, delta)
            # lin  = ae - quad
            # hub = float(np.nanmean(0.5 * quad ** 2 + delta * lin))

        # R² (only if meaningful)
        if len(y_true) > 1 and np.nanstd(y_true) > 0:
            ss_res = np.nansum((y_true - y_pred) ** 2)
            ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
            if ss_tot > 0:
                r2 = 1 - ss_res / ss_tot

    title = f'{plot_name}'
    title += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title += f' (n={n})'

    if reg_metrics:
        # last line: metrics
        line = "\n"

        if mae is not None and np.isfinite(mae):
            line += f', MAE={float(mae):.{DECIMALS}f}'
        else:
            line += ', MAE=nan'

        if mse is not None and np.isfinite(mse):
            line += f'MSE={float(mse):.{DECIMALS}f}'
        else:
            line += 'MSE=nan'

        # if hub is not None and np.isfinite(hub):
        #     line += f', Huber={float(hub):.{DECIMALS}f}'
        # else:
        #     line += ', Huber=nan'

        if r2 is not None and np.isfinite(r2):
            line += f', R²={float(r2):.{DECIMALS}f}'
        else:
            line += ', R²=nan'

        title += line

    plt.title(title)

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title=make_legend_descriptor('type'),
        frameon=True
    )

    clean_data = clean_data_string(data)
    result_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(result_path, path, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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
    """
    One input df (must contain: 'ikey', 'dataset', y_feature_name).

    - "global" view: use df as-is
      -> within each y_feature group, compute % rows whose ikey is non-unique (duplicated keep=False)

    - "dataset" view: computationally derive a second df by collapsing within (dataset, ikey)
      -> df_dataset = one row per (dataset, ikey)
      -> within each y_feature group, compute % rows whose ikey is non-unique across datasets

    Plot side-by-side horizontal bars in the desired order (based on y_feature_name).
    """

    # checks
    required = {"ikey", "dataset", y_feature_name}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # normalize feature column
    df_global = df.copy()
    df_global[y_feature_name] = df_global[y_feature_name].astype(object)
    df_global[y_feature_name] = df_global[y_feature_name].where(
        df_global[y_feature_name].notna(), "NA"
    )

    # derive dataset-collapsed df (one row per dataset + ikey)
    df_dataset = (
        df_global.sort_values(["dataset", "ikey"])
        .groupby(["dataset", "ikey"], as_index=False)
        .first()
    )

    # helpers
    def pct_nonunique_within_group(g: pd.DataFrame) -> float:
        n = len(g)
        if n == 0:
            return 0.0
        dup_mask = g.duplicated(subset=["ikey"], keep=False)
        return 100.0 * float(dup_mask.sum()) / float(n)

    def compute_group_stats(dfx: pd.DataFrame) -> dict:
        # group -> (pct_nonunique, n_rows)
        stats = {}
        if dfx is None or len(dfx) == 0:
            return stats
        for group_name, g in dfx.groupby(y_feature_name, dropna=False):
            stats[group_name] = (pct_nonunique_within_group(g), int(len(g)))
        return stats

    # compute stats
    stats_g = compute_group_stats(df_global)
    stats_d = compute_group_stats(df_dataset)

    present = list(set(stats_g.keys()) | set(stats_d.keys()))

    rev_seg = SEGMENTS[::-1]
    rev_str = STRAINS[::-1]
    rev_dat = DATASETS[::-1]

    # ---- ordering scheme (y_feature_name drives ordering) ----
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

    # aligned vectors
    pct_g, pct_d, n_g, n_d = [], [], [], []
    for name in desired:
        pg, ng = stats_g.get(name, (0.0, 0))
        pd_, nd_ = stats_d.get(name, (0.0, 0))
        pct_g.append(pg)
        pct_d.append(pd_)
        n_g.append(ng)
        n_d.append(nd_)

    # ---- plot ----
    plt.style.use("seaborn-darkgrid")
    num_groups = len(desired)
    y_positions = np.arange(num_groups)

    bar_height = 0.35
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=(12, max(6, num_groups * 0.4)))

    color_global = COLORS[0]
    color_dataset = COLORS[8]

    ax.barh(
        y_positions - offset,
        pct_g,
        height=bar_height,
        label="global",
        edgecolor="white",
        linewidth=0.5,
        color=color_global,
    )
    ax.barh(
        y_positions + offset,
        pct_d,
        height=bar_height,
        label="dataset-wise",
        edgecolor="white",
        linewidth=0.5,
        color=color_dataset,
    )

    y_labels = [
        f"{name} (n_g={ng}, n_d={nd})"
        for name, ng, nd in zip(desired, n_g, n_d)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("intersects within candidates (%)")
    ax.set_ylabel(f"{y_feature_name}s")

    N_g = int(len(df_global))
    N_d = int(len(df_dataset))

    title = f"comparison of intersects per {y_feature_name}"
    title += make_candidate_descriptor("all", data, strain, segment, intersects, True)
    title += f" (n_glo={N_g}, n_dat={N_d})"
    ax.set_title(title)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=True,
        title=make_legend_descriptor(y_feature_name),
    )

    plt.tight_layout()

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

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
    """
    Compare non-unique ikeys (%) between two dfs (pooled vs unpooled), grouped by y_feature_name.

    Requirements:
    - both dfs must contain: 'ikey', y_feature_name
    - analysis: within each y_feature group, compute % rows whose ikey is non-unique (duplicated keep=False)
    - plot: side-by-side horizontal bars for pooled and unpooled
    """

    # ---- checks ----
    required = {"ikey", y_feature_name}
    miss_p = required - set(df_pooled.columns)
    miss_u = required - set(df_unpooled.columns)
    if miss_p:
        raise ValueError(f"df_pooled missing required columns: {sorted(miss_p)}")
    if miss_u:
        raise ValueError(f"df_unpooled missing required columns: {sorted(miss_u)}")

    # ---- normalize feature column ----
    df_p = df_pooled.copy()
    df_u = df_unpooled.copy()

    # ---- helpers ----
    def pct_nonunique_ikey_within_group(g: pd.DataFrame) -> float:
        n = len(g)
        if n == 0:
            return 0.0
        dup_mask = g.duplicated(subset=["ikey"], keep=False)
        return 100.0 * float(dup_mask.sum()) / float(n)

    def compute_group_stats(dfx: pd.DataFrame) -> dict:
        # group -> (pct_nonunique, n_rows)
        stats = {}
        if dfx is None or len(dfx) == 0:
            return stats
        for group_name, g in dfx.groupby(y_feature_name, dropna=False):
            stats[group_name] = (pct_nonunique_ikey_within_group(g), int(len(g)))
        return stats

    # ---- compute stats ----
    stats_p = compute_group_stats(df_p)  # pooled
    stats_u = compute_group_stats(df_u)  # unpooled

    present = list(set(stats_p.keys()) | set(stats_u.keys()))

    rev_seg = SEGMENTS[::-1]
    rev_str = STRAINS[::-1]
    rev_dat = DATASETS[::-1]

    # ---- ordering scheme (y_feature_name drives ordering) ----
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

    # ---- aligned vectors ----
    pct_p, pct_u, n_p, n_u = [], [], [], []
    for name in desired:
        pp, np_ = stats_p.get(name, (0.0, 0))
        pu, nu_ = stats_u.get(name, (0.0, 0))
        pct_p.append(pp)
        pct_u.append(pu)
        n_p.append(np_)
        n_u.append(nu_)

    # ---- plot ----
    plt.style.use("seaborn-darkgrid")
    num_groups = len(desired)
    y_positions = np.arange(num_groups)

    bar_height = 0.35
    offset = bar_height / 2.0

    fig, ax = plt.subplots(figsize=(12, max(6, num_groups * 0.4)))

    color_pooled = COLORS[0]
    color_unpooled = COLORS[8]

    ax.barh(
        y_positions - offset,
        pct_p,
        height=bar_height,
        label="pooled",
        edgecolor="white",
        linewidth=0.5,
        color=color_pooled,
    )
    ax.barh(
        y_positions + offset,
        pct_u,
        height=bar_height,
        label="unpooled",
        edgecolor="white",
        linewidth=0.5,
        color=color_unpooled,
    )

    y_labels = [
        f"{name} (n_p={np_}, n_u={nu_})"
        for name, np_, nu_ in zip(desired, n_p, n_u)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("intersects within candidates (%)")
    ax.set_ylabel(f"{make_legend_descriptor(y_feature_name)}s")

    N_p = int(len(df_p))
    N_u = int(len(df_u))

    title = f"comparison of intersects per {make_legend_descriptor(y_feature_name)}"
    title += make_candidate_descriptor("all", data, strain, segment, intersects)
    title += f" (n_pool={N_p}, n_unpool={N_u})"
    ax.set_title(title)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=True,
        title=make_legend_descriptor(y_feature_name),
    )

    plt.tight_layout()

    # ---- save ----
    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    outname = fname + ".png"
    plt.savefig(os.path.join(save_path, outname), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    selector = 'dataset'
    curvature = 'right'
    show_quantiles= 1
    
    ### DATASETS SINGLE/MULTI/PREDICTION ###

    folder = 'pooled'
    subfolder = 'secondary'

    data = 'IAV'

    strain = 'all'
    strain = DATASET_STRAIN_DICT[data]

    segment = 'all'
    intersects = 'median_dataset_5'

    dfnames = [data]
    dfnames = get_dataset_names(DATASET_CUTOFF, data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### heatmaps ###

    # run_site_motif_heatmap_analysis(dfs, TOP_N, folder, data, strain, segment, intersects)
    # run_reg_site_motif_heatmap_analysis(dfs, TOP_N, folder, data, strain, segment, intersects)
    # run_repeat_heatmap_analysis(dfs, folder, data, strain, segment, intersects)
    # run_mfe_heatmap_analysis(dfs, folder, data, strain, segment, intersects)

    ### distribution ###

    # run_ngs_single_density_analysis(dfs, folder, data, strain, segment, intersects, show_quantiles, curvature)
    # run_ngs_multi_density_analysis(dfs, folder, data, strain, segment, intersects)
    # run_pooling_intersect_analysis(selector, data, strain, segment, intersects)

    ### features ###

    # run_delvg_pri_features_analysis(dfs, selector, TOP_N, folder, data, strain, segment, intersects)
    # run_delvg_sec_features_analysis(dfs, selector, folder, data, strain, segment, intersects)
    # run_delvg_hybrid_features_analysis(dfs, selector, folder, data, strain, segment, intersects)
    # run_length_mfe_analysis(dfs, selector, folder, data, strain, segment, intersects)

    ### prediction ###

    # run_pred_analysis(selector, data, strain, segment, intersects)

    ### other ###

    # run_sec_structure_plot(dfs, data, strain, segment)
