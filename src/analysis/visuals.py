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

from sklearn.metrics import r2_score

import RNA

sys.path.insert(0, "..")
from utils import get_dataset_names, load_all
from utils import manage_separate_specifiers, load_all_preprocessed, clean_data_string
from utils import rename_feature, split_by_threshold, add_feature_percentile_rank, add_log_feature, add_norm_feature, add_separate_ngs_features
from utils import manage_intersects
from utils import add_dvg_sequence, add_dvg_length, add_full_seq_length, add_deletion_length, add_region_lengths

from utils import pick_colors
from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df
from utils import compute_percentile_rank_count_df, compute_seq_feature_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_count_heatmap_sum_df, compute_feature_freq_heatmap_df

from utils import add_metadata_features, p_to_stars, mannwhitneyu_for_feature
from utils import add_direct_repeat_length, cap_direct_repeat_length
from utils import add_max_symmetry

from utils import (
    add_gc_content,
    add_au_content,
    add_upa_content,
    add_cpg_content,
    add_gc_skew,
    add_sequence_entropy,
    add_poly_run_features,
    add_palindrome_density,
    add_orf_features,
    add_kmer_richness,
    add_codon_usage_bias,
)

from utils import (
    add_bp_count,
    add_bp_density,
    add_unpaired_count,
    add_unpaired_density,
    add_stem_count,
    add_stem_length_stats,
    add_hairpin_count,
    add_hairpin_size_stats,
    add_external_unpaired_density,
    add_pair_span_stats,
    add_free_end_lengths,
    add_branch_point_count,
    )

from utils import (
    add_gc_overall,
    add_gc_paired_unpaired,
    add_canonical_pair_stats,
    add_stem_end_pair_enrichment,
    add_hairpin_closing_pair_content,
    add_tetraloop_motif_counts,
    add_loop_au_content,
    add_tail_gc_content,
    add_start_codon_accessibility,
)


from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import COLORS, RANK_THRESHOLD, DECIMALS

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
    plot_names = ['site motif before deletion', 'site motif deletion start', 'site motif deletion end', 'site motif after deletion']

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

    create_spline_plot(f'mean freq diff for sites - spline plot', 'length of motifs (nucleotides)', 'mean frequency difference (%)', plot_names, all_mean_freq_diffs, dvg_count, 'site_mean_freq_diff', 'motifs', 'splineplots', data, strain, segment, intersects)
    create_spline_plot(f'top freq diff for sites - spline plot', 'length of motifs (nucleotides)', 'top frequency difference (%)', plot_names, all_top_freq_diffs, dvg_count, 'site_top_freq_diff', 'motifs', 'splineplots', data, strain, segment, intersects)

def run_reg_site_motif_heatmap_analysis(dfs: list, top_n: int = 5, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    max_motif_length = 7

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    plot_names = ['site motif before deletion', 'site motif deletion_start', 'site motif deletion end', 'site motif after deletion']

    all_mean_coefficients = [[None for _ in range(max_motif_length)] for _ in range(4)]
    all_top_coefficients  = [[None for _ in range(max_motif_length)] for _ in range(4)]
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        dvg_count = df_copy.shape[0]
        mean_coefficients, top_coefficients = make_reg_site_motif_heatmap_analysis(motif_df, plot_names, dvg_count, motif_length, top_n, data, strain, segment, intersects)
        print(f'motif_length {motif_length} completed')

        for i in range(0, 4):
            all_mean_coefficients[i][motif_length - 1] = mean_coefficients[i]
            all_top_coefficients[i][motif_length - 1] = top_coefficients[i]

    create_spline_plot(f'mean coefficients for site - spline plot', 'motif length (nucleotides)', 'mean coefficients (-)', plot_names, all_mean_coefficients, dvg_count, 'site_mean_coefficients', 'motifs', 'splineplots', data, strain, segment, intersects)
    create_spline_plot(f'top coefficients for site - spline plot', 'motif length (nucleotides)', 'top coefficients (-)', plot_names, all_top_coefficients, dvg_count, 'site_top_coefficients', 'motifs', 'splineplots', data, strain, segment, intersects)

def run_repeat_heatmap_analysis(dfs: list, cap_length: int = 5, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    dvg_count = df.shape[0]
    heatmap_name = 'direct repeat length'

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
    heatmap_name = 'minimum free energy (MFE)'

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

    plot_name = 'DVG'
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

    create_spline_plot(f'mean freq diff for DVG - spline plot', 'length of motifs (nucleotides)', 'mean frequency difference (%)', plot_names, all_mean_freq_diffs, dvg_count, 'dvg_mean_freq_diff', 'motifs', 'splineplots', data, strain, segment, intersects)
    create_spline_plot(f'top freq diff for DVG - spline plot', 'length of motifs (nucleotides)', 'top frequency difference (%)', plot_names, all_top_freq_diffs, dvg_count, 'dvg_top_freq_diff', 'motifs', 'splineplots', data, strain, segment, intersects)

### histo ###

def run_ngs_histo_analysis(dfs: list, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, feature)

    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = 'normalized log10 NGS count (reads)'

    create_freq_histo_plot('NGS read count - histogram', df, x_feature_name, x_axis_name, True, 'ngs', 'NGS', 'histogram', data, strain, segment, intersects)

### bar & scatter ###

def run_dvg_pri_features_analysis(dfnames: list, dfs: list, selector: str, top_n: int, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = 'normalized log10 NGS count (reads)'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    dfs = add_metadata_features(dfnames, dfs)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, x_feature_name)

    palindrmic_k_mer = 6
    k_mer = 3

    df = add_dvg_sequence(df)
    df = add_dvg_length(df)
    df = add_deletion_length(df)
    df = add_full_seq_length(df)
    df = add_region_lengths(df)

    df = add_gc_content(df)
    df = add_au_content(df)
    df = add_upa_content(df)
    df = add_cpg_content(df)
    df = add_gc_skew(df)
    df = add_sequence_entropy(df)
    df = add_poly_run_features(df, 'A')
    df = add_poly_run_features(df, 'U')
    df = add_palindrome_density(df, palindrmic_k_mer)
    df = add_orf_features(df)
    df = add_kmer_richness(df, k_mer)
    df = add_codon_usage_bias(df)

    print(f'features added successfully')

    folder = 'pri'
    subfolder = 'scatterplot'

    features = [
        ("Start", "deletion start position", "position where deletion begins (nucleotides)", "length"),
        ("End", "deletion end position", "position where deletion ends (nucleotides)", "length"),

        ("full_seq_length", "full sequence length", "length of the full reference sequence (nucleotides)", "length"),
        ("dvg_length", "DVG length", "length of the defective viral genome (nucleotides)", "length"),
        ("deletion_length", "deletion length", "length of the deleted region (nucleotides)", "length"),

        ("5_end_length", "5′ end length", "length of the retained 5′ end fragment (nucleotides)", "length"),
        ("3_end_length", "3′ end length", "length of the retained 3′ end fragment (nucleotides)", "length"),

        ("GC_content", "GC content", "content of G and C (1/nucleotide)", "composition"),
        ("AU_content", "AU content", "content of A and U (1/nucleotide)", "composition"),
        ("UpA_content", "UpA content", "content of UpA (1/dinucleotide)", "dinucleotides"),
        ("CpG_content", "CpG content", "content of CpG (1/dinucleotide)", "dinucleotides"),

        ("GC_skew", "GC skew", "relative skew between G and C nucleotides", "composition"),
        ("sequence_entropy", "sequence entropy", "Shannon entropy of nucleotide composition", "other"),

        ("poly_U_max_run", "poly-U maximum run length", "longest consecutive run of U (nucleotides)", "length"),
        ("poly_U_tracts", "poly-U tract count", "number of U tracts above threshold length", "length"),
        ("poly_A_max_run", "poly-A maximum run length", "longest consecutive run of A (nucleotides)", "length"),
        ("poly_A_tracts", "poly-A tract count", "number of A tracts above threshold length", "length"),

        ("palindrome_density", f"palindrome density (k={palindrmic_k_mer})", "density of palindromic k-mers (1/k-mer)", "motifs"),

        ("longest_ORF_len", "longest ORF length", "length of longest open reading frame (nucleotides)", "ORF"),
        ("ORF_count", "ORF count ≥20 AA", "number of ORFs at least 20 amino acids long", "ORF"),

        ("kmer_richness", f"k-mer richness (k={k_mer})", "content of unique k-mers among all windows (1/k-mer)", "complexity"),
        ("codon_usage_entropy", "codon usage entropy", "Shannon entropy of codon usage", "codon"),
    ]

    for y_feature_name, y_feature_title, y_axis_name, folder in features:
            create_feature_scatter_plot(f'{y_feature_title} - NGS reads - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, y_feature_name, folder, subfolder, data, strain, segment, intersects)
    
    print(f'numerical features completed')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    df = add_direct_repeat_length(df)

    # motifs #
    max_motif_length = 7
    plot_names = ['site motif before deletion', 'site motif deletion_start', 'site motif deletion end', 'site motif after deletion']
    for motif_length in range(1, max_motif_length + 1):
        df_copy = df.copy()
        motif_df, skipped_count = add_site_motifs(df_copy, motif_length)
        print(f'{skipped_count} candidates have been left out')
        dvg_count = df_copy.shape[0]
        site_counter = 0
        print(f'motif_length {motif_length} completed')
        for i in range(0, 4):
            bar_name = plot_names[i]
            bar_name += f' - bar plot - motif length {motif_length}'
            site_counter = i + 1
            create_freq_bar_plot(bar_name, motif_df, f'site{site_counter}_motif', f'top {top_n} motifs', dvg_count, top_n, f'site{site_counter}_motif', 'motifs', 'barplots', data, strain, segment, intersects, f'motif_lenght_{motif_length}')
    
    df = add_direct_repeat_length(df)
    cap_length = 5
    df = cap_direct_repeat_length(df, cap_length)

    folder = 'pri'
    subfolder = 'barplots'

    dvg_count = df.shape[0]

    features = [
        ("direct_repeat_length", "direct repeat length", "length of direct repeats (nucleotides)", "repeats"),

        ("subtype", "virus subtype", "viral subtype", "metadata"),
        ("system_type", "system type", "system type", "metadata"),
        ("library_selection", "library selection", "library selection", "metadata"),
    ]

    if strain == "all":
        features.append(
            ("Strain", "strain", "virus strain", "metadata")
        )

    if segment == "all":
        features.append(
            ("Segment", "segment", "segment", "metadata")
        )

    for y_feature_name, y_feature_title, y_axis_name, folder in features:
            create_freq_bar_plot(f'{y_feature_title} - bar plot', df, y_feature_name, y_axis_name, dvg_count, 0, y_feature_name, folder, subfolder, data, strain, segment, intersects)

    print(f'categorical features completed')

def run_dvg_sec_features_analysis(dfnames: list, dfs: list, selector: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = 'normalized log10 NGS count (reads)'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    dfs = add_metadata_features(dfnames, dfs)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, x_feature_name)

    df = add_bp_count(df)
    df = add_bp_density(df)
    df = add_unpaired_count(df)
    df = add_unpaired_density(df)
    df = add_stem_count(df)
    df = add_stem_length_stats(df)
    df = add_hairpin_count(df)
    df = add_hairpin_size_stats(df)
    df = add_external_unpaired_density(df)
    df = add_pair_span_stats(df)
    df = add_free_end_lengths(df)
    df = add_branch_point_count(df)

    df = add_max_symmetry(df)

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')

    print(f'features added successfully')

    folder = 'sec'
    subfolder = 'scatterplot'

    features = [
        ("MFE", "MFE", "minimum free energy (kcal/mol)", "other"),

        ("bp_count", "base pair count", "number of base pairs", "pairing"),
        ("bp_density", "base pair density", "base pair density (1/nucleotide)", "pairing"),
        ("unpaired_count", "unpaired base count", "number of unpaired bases", "pairing"),
        ("unpaired_density", "unpaired base density", "unpaired base density (1/nucleotide)", "pairing"),

        ("stem_count", "stem count", "number of stems", "stem"),
        ("stem_len_max", "stem length maximum", "maximum stem length (nucleotides)", "stem"),
        ("stem_len_mean", "stem length mean", "mean stem length (nucleotides)", "stem"),
        ("stem_len_min", "stem length minimum", "minimum stem length (nucleotides)", "stem"),

        ("hairpin_count", "hairpin count", "number of hairpin loops", "hairpin"),
        ("hairpin_size_mean", "hairpin size mean", "mean hairpin loop size (nucleotides)", "hairpin"),
        ("hairpin_size_min", "hairpin size minimum", "minimum hairpin loop size (nucleotides)", "hairpin"),
        ("hairpin_size_max", "hairpin size maximum", "maximum hairpin loop size (nucleotides)", "hairpin"),

        ("external_unpaired_density", "external unpaired base density", "unpaired base density in external loop (1/nucleotide)", "pairing"),

        ("pair_span_mean", "pair span mean", "mean base pair span (nucleotides)", "pair_span"),
        ("pair_span_min", "pair span minimum", "minimum base pair span (nucleotides)", "pair_span"),
        ("pair_span_max", "pair span maximum", "maximum base pair span (nucleotides)", "pair_span"),

        ("free_5prime_len", "free 5′ length", "unpaired length at 5′ end (nucleotides)", "length"),
        ("free_3prime_len", "free 3′ length", "unpaired length at 3′ end (nucleotides)", "length"),

        ("branch_point_count", "branch point count", "count of multiloop/branch points", "composition"),

        ("max_symmetry", "max symmetry length", "max symmetry length (nucleotides)", "other"),
    ]

    for y_feature_name, y_feature_title, y_axis_name, folder in features:
            create_feature_scatter_plot(f'{y_feature_title} - NGS reads - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, y_feature_name, folder, subfolder, data, strain, segment, intersects)
    
    print(f'numerical features completed')

    folder = 'sec'
    subfolder = 'barplots'

    dvg_count = df.shape[0]

    features = [
        ("full_symmetry", "full symmetry", "full symmetry", "other"),
    ]

    for y_feature_name, y_feature_title, y_axis_name, folder in features:
            create_freq_bar_plot(f'{y_feature_title} - bar plot', df, y_feature_name, y_axis_name, dvg_count, 0, y_feature_name, folder, subfolder, data, strain, segment, intersects)

    print(f'categorical features completed')

def run_dvg_hybrid_features_analysis(dfnames: list, dfs: list, selector: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    x_feature_name = 'norm_log_NGS_read_count'
    x_axis_name = 'normalized log10 NGS count (reads)'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    dfs = add_metadata_features(dfnames, dfs)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, x_feature_name)

    df = add_gc_overall(df)
    df = add_gc_paired_unpaired(df)
    df = add_canonical_pair_stats(df)
    df = add_stem_end_pair_enrichment(df)
    df = add_hairpin_closing_pair_content(df)
    df = add_tetraloop_motif_counts(df)
    df = add_loop_au_content(df)
    df = add_tail_gc_content(df)
    df = add_start_codon_accessibility(df)

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')

    print(f'features added successfully')

    folder = 'sec'
    subfolder = 'scatterplot'

    features = [
        ("GC_overall", "GC overall", "overall GC content (1/nucleotide)", "composition"),
        ("GC_paired", "GC paired", "GC content among paired bases (1/nucleotide)", "composition"),
        ("GC_unpaired", "GC unpaired", "GC content among unpaired bases (1/nucleotide)", "composition"),

        ("pair_GC_count", "GC pair count", "number of GC base pairs", "pairing"),
        ("pair_AU_count", "AU pair count", "number of AU base pairs", "pairing"),
        ("pair_GU_count", "GU pair count", "number of GU base pairs", "pairing"),
        ("pair_noncanon_count", "noncanonical pair count", "number of noncanonical base pairs", "pairing"),
        ("pair_GC_content", "GC pair content", "content of base pairs that are GC (1/pair)", "pairing"),
        ("pair_AU_content", "AU pair content", "content of base pairs that are AU (1/pair)", "pairing"),
        ("pair_GU_content", "GU pair content", "content of base pairs that are GU (1/pair)", "pairing"),
        ("pair_noncanon_content", "noncanonical pair content", "content of base pairs that are noncanonical (1/pair)", "pairing"),

        ("stem_end_GC_content", "stem end GC content", "content of stem-end pairs that are GC (1/pair)", "stem"),
        ("stem_end_AU_content", "stem end AU content", "content of stem-end pairs that are AU (1/pair)", "stem"),
        ("stem_end_GU_content", "stem end GU content", "content of stem-end pairs that are GU (1/pair)", "stem"),

        ("hairpin_close_GC_content", "hairpin closing GC content", "content of hairpin-closing pairs that are GC (1/pair)", "hairpin"),
        ("hairpin_close_AU_content", "hairpin closing AU content", "content of hairpin-closing pairs that are AU (1/pair)", "hairpin"),
        ("hairpin_close_GU_content", "hairpin closing GU content", "content of hairpin-closing pairs that are GU (1/pair)", "hairpin"),
        ("hairpin_close_noncanon_content", "hairpin closing noncanonical content", "content of hairpin-closing pairs that are noncanonical (1/pair)", "hairpin"),

        ("motif_GNRA_count", "GNRA motif count", "count of GNRA tetraloops", "motif"),
        ("motif_UNCG_count", "UNCG motif count", "count of UNCG tetraloops", "motif"),
        ("motif_CUUG_count", "CUUG motif count", "count of CUUG tetraloops", "motif"),

        ("AU_unpaired_content", "AU unpaired content", "AU content among all unpaired bases (1/nucleotide)", "composition"),
        ("AU_hairpin_content", "AU hairpin content", "AU content in hairpin loops (1/nucleotide)", "composition"),
        ("AU_internal_content", "AU internal content", "AU content in internal/multiloop/bulge (1/nucleotide)", "composition"),
        ("AU_external_content", "AU external content", "AU content in external loop (1/nucleotide)", "composition"),

        ("GC_5prime_tail", "GC 5′ tail content", "GC content in 5′ single-stranded tail (1/nucleotide)", "composition"),
        ("GC_3prime_tail", "GC 3′ tail content", "GC content in 3′ single-stranded tail (1/nucleotide)", "composition"),

        ("AUG_total", "AUG total count", "total number of AUG start codons", "codon"),
        ("AUG_unpaired", "AUG unpaired count", "number of AUG start codons fully unpaired", "codon"),
        ("AUG_unpaired_content", "AUG unpaired content", "content of AUG start codons fully unpaired (1/codon)", "codon"),
    ]

    for y_feature_name, y_feature_title, y_axis_name, folder in features:
            create_feature_scatter_plot(f'{y_feature_title} - NGS reads - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, y_axis_name, selector, True, y_feature_name, folder, subfolder, data, strain, segment, intersects)
    
    print(f'numerical features completed')

def run_length_mfe_analysis(dfnames: list, dfs: list, selector: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    x_feature_name = 'dvg_length'
    x_axis_name = 'length of the defective viral genome (nucleotides)'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    dfs = add_metadata_features(dfnames, dfs)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_dvg_sequence(df)
    df = add_dvg_length(df)

    print(f'features added successfully')

    folder = 'other'
    subfolder = 'scatterplot'

    # MFE #
    y_feature_name = 'MFE'
    feature_name = 'minimum free energy'
    create_feature_scatter_plot(f'{feature_name} - DVG length - scatter plot', df, x_feature_name, x_axis_name, y_feature_name, f'{feature_name} (kcal/mol)', selector, True, y_feature_name, folder, subfolder, data, strain, segment, intersects)
    
    print(f'numerical features completed')

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
    plot_name: str,
    feature_name: str,
    x_axis_name: str,
    freq_heatmap_df: pd.DataFrame,
    count_df: pd.DataFrame,
    percentile_rank_count_df: pd.DataFrame,
    dvg_count: int,
    fname: str,
    folder: str,
    subfolder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    feature_specifier: str = ''
    ):  
    '''

    '''
    pivot_df = freq_heatmap_df
    feature_order = list(pivot_df.columns)
    rank_order = list(pivot_df.index)

    feature_count_series = count_df.set_index(feature_name)['count']

    feature_labels = [
        f'{feature}\n(n={feature_count_series.get(feature, 0)})'
        for feature in feature_order
    ]

    rank_labels = []
    for rank in rank_order:
        count = percentile_rank_count_df.loc[
            percentile_rank_count_df['NGS_percentile_rank'] == rank, 'count'
        ].values
        count_str = str(count[0]) if len(count) > 0 else "0"
        rank_labels.append(f"{rank}\n(n={count_str})")

    min_width = 13
    fig_width = max(min_width, 1 + len(feature_labels) * 0.6)
    fig_height = 1 + len(rank_labels) * 0.4
    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        pivot_df,
        cmap="viridis",
        linewidths=0.5,
        linecolor='white',
        xticklabels=feature_labels,
        yticklabels=rank_labels,
        cbar_kws={'label': 'relative frequency (%)'},
        annot=True,
        fmt='.1f'
    )

    ax.set_xlabel(f'{x_axis_name}')
    ax.set_ylabel('NGS rank (percentile)')

    title_name = f'{plot_name}'
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
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
    feature_name: str,
    x_axis_name: str,
    freq_df: pd.DataFrame,
    dvg_count: int,
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
        id_vars=feature_name,
        value_vars=[freq0, freq1],
        var_name='type',
        value_name='frequency'
    )

    label_map = {
        freq0: freq0.replace('_', ' '),
        freq1: freq1.replace('_', ' ')
        }

    df_plot['type'] = df_plot['type'].replace(label_map)

    custom_colors = {
        label_map[freq0]: COLORS[8],
        label_map[freq1]: COLORS[0]
        }

    plt.figure(figsize=(18, 6))
    sns.barplot(
    data=df_plot,
    x=feature_name,
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
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
    plt.title(title_name)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
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
    dvg_count: int,
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
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
    plt.title(title_name)

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')
    plt.grid(True, alpha=0.3)

    plt.xticks(np.arange(1, num_points + 1))

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
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

def create_freq_histo_plot(
    plot_name: str,
    df: pd.DataFrame,
    feature_name: str,
    x_axis_name: str,
    show_percentiles: bool,
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
        color=COLORS[8],
        edgecolor='white',
        label='distribution'
    )

    plt.axvline(mean_val, color=COLORS[9], linestyle='-', linewidth=1, label='mean')

    if show_percentiles:
        q90 = np.percentile(values, 90)
        q95 = np.percentile(values, 95)
        plt.axvline(q90, color=COLORS[1], linestyle='--', linewidth=1, label='90th percentile')
        plt.axvline(q95, color=COLORS[0], linestyle='--', linewidth=1, label='95th percentile')

    title = f'{plot_name}'
    title += f'\ndata: {data}'
    title += f', strain: {strain}'
    title += f', segment: {segment}'
    title += f', {intersects} intersects'
    title += f' (n={n})'
    plt.title(title)

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel('relative frequency (%)')

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
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

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close()

def create_freq_bar_plot(
    plot_name: str,
    df: pd.DataFrame,
    feature_name: str,
    x_axis_name: str,
    dvg_count: int,
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

    '''
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 7))

    true_df, false_df = split_by_threshold(df, 'NGS_percentile_rank', RANK_THRESHOLD)

    true_counts  = true_df[feature_name].value_counts(dropna=False)
    false_counts = false_df[feature_name].value_counts(dropna=False)
    n_true, n_false = len(true_df), len(false_df)

    def pct(count, denom):
        return (100.0 * count / denom) if denom > 0 else 0.0

    all_cats = list(set(true_counts.index.tolist()) | set(false_counts.index.tolist()))

    freq_true  = {cat: pct(true_counts.get(cat, 0),  n_true)  for cat in all_cats}
    freq_false = {cat: pct(false_counts.get(cat, 0), n_false) for cat in all_cats}
    diffs = {cat: freq_true[cat] - freq_false[cat] for cat in all_cats}

    if top_n and top_n > 0:
        top_cats = sorted(all_cats, key=lambda c: diffs[c], reverse=True)[:top_n]
    else:
        top_cats = sorted(all_cats, key=lambda c: str(c).casefold())

    x = np.arange(len(top_cats))
    width = 0.42

    bars_false = [freq_false[c] for c in top_cats]
    bars_true  = [freq_true[c]  for c in top_cats]

    color_false = COLORS[0]
    color_true  = COLORS[8]

    b1 = plt.bar(x - width/2, bars_true,  width=width, label=f'NGS rank {RANK_THRESHOLD} (percentile)', color=color_true,  edgecolor='white', linewidth=0.5)
    b0 = plt.bar(x + width/2, bars_false, width=width, label='lower NGS ranks (percentiles)', color=color_false, edgecolor='white', linewidth=0.5)

    y_max_overall = 0.0
    pvals = []
    for i, cat in enumerate(top_cats):
        tx = (true_df[feature_name]  == cat).astype(int).to_numpy()
        fx = (false_df[feature_name] == cat).astype(int).to_numpy()

        res = mannwhitneyu_for_feature(
            pd.DataFrame({feature_name: tx}),
            pd.DataFrame({feature_name: fx}),
            feature_name=feature_name,
            alternative="two-sided"
        )
        p = res["pvalue"]
        pvals.append(p)

        h_left  = bars_false[i]
        h_right = bars_true[i]
        h_pair_top = max(h_left, h_right)
        y_base = h_pair_top
        y_offset = max(bars_false + bars_true) * 0.03 if (bars_false + bars_true) else 1.0
        y_bracket = y_base + y_offset
        h_bracket = y_offset * 0.6

        x1 = i - width/2
        x2 = i + width/2

        plt.plot([x1, x1, x2, x2],
                 [y_bracket, y_bracket + h_bracket, y_bracket + h_bracket, y_bracket],
                 color='black', linewidth=1.0)

        plt.text((x1 + x2)/2, y_bracket + h_bracket + (y_offset*0.2),
                 p_to_stars(p), ha='center', va='bottom', fontsize=9)

        y_max_overall = max(y_max_overall, y_bracket + h_bracket + (y_offset*0.6))

    current_top = max(bars_false + bars_true) if (bars_false + bars_true) else 1.0
    plt.ylim(0, max(current_top * 1.25, y_max_overall * 1.05))

    labels = [f"{str(c)}\n(n={true_counts.get(c, 0) + false_counts.get(c, 0)})" for c in top_cats]
    plt.xticks(x, labels, rotation=0, ha='center')

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel('relative frequency (%)')

    title_name = f'{plot_name}'
    title_name += f'\ndata: {data}'
    title_name += f', strain: {strain}'
    title_name += f', segment: {segment}'
    title_name += f', {intersects} intersects'
    title_name += f' (n={dvg_count})'
    plt.title(title_name)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

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

def create_feature_scatter_plot(
    plot_name: str,
    df: pd.DataFrame,
    x_feature_name: str,
    x_axis_name: str,
    y_feature_name: str,
    y_axis_name: str,
    selector: str,
    show_regression_line: bool,
    fname: str,
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

    values = df[[x_feature_name, y_feature_name, selector]].dropna(subset=[x_feature_name, y_feature_name]).copy()
    n = len(values)

    cats = (
        values[selector]
        .astype(object)
        .where(values[selector].notna(), other="UNKNOWN")
        .astype(str)
    )
    values[selector] = cats
    uniq = sorted(cats.unique())

    picked_colors = pick_colors(COLORS, len(uniq))
    palette = dict(zip(uniq, picked_colors))

    sns.scatterplot(
        data=values,
        x=x_feature_name,
        y=y_feature_name,
        hue=selector,
        palette=palette,
        edgecolor='white',
        s=50,
        alpha=0.7,
    )

    if show_regression_line:
        reg_line = sns.regplot(
            data=values,
            x=x_feature_name,
            y=y_feature_name,
            scatter=False,
            color='grey'
        )
        reg_line.lines[0].set_label('regression line')

    r2 = None
    if show_regression_line and not values.empty:
        x_vals = values[x_feature_name].to_numpy()
        y_vals = values[y_feature_name].to_numpy()
        if np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            y_pred = np.polyval(coeffs, x_vals)
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
            if ss_tot > 0:
                r2 = 1 - ss_res / ss_tot
    
    title = f'{plot_name}'
    if r2 is not None:
        r2 = round(r2, DECIMALS)
        title += f' (R²={r2})'
    title += f'\ndata: {data}'
    title += f', strain: {strain}'
    title += f', segment: {segment}'
    title += f', {intersects} intersects'
    title += f' (n={n})'
    plt.title(title)

    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

    clean_data = clean_data_string(data)
    save_path = os.path.join(RESULTSPATH, folder, subfolder)
    save_path = os.path.join(save_path, clean_data, strain, segment, intersects)
    os.makedirs(save_path, exist_ok=True)

    fname += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
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

        bar_name = heatmap_names[i]
        bar_name += f' - bar plot - motif length {motif_length}'
        bar_name += f' (mean freq diff = {mean_freq_diff}, '
        bar_name += f'top freq diff = {top_freq_diff})'

        create_freq_diff_plot(bar_name, 'motif', f'top {top_n} motifs', comb_freq_df, dvg_count, motif_sites[i], 'motifs', 'diffplots', data, strain, segment, intersects, f'motif_length_{motif_length}')

        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], filtered_motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)

        heatmap_name = heatmap_names[i]
        heatmap_name += f' - heatmap - motif length {motif_length}'
        heatmap_name += f' (mean freq diff = {mean_freq_diff}, '
        heatmap_name += f'top freq diff = {top_freq_diff})'

        create_freq_heatmap_plot(heatmap_name, 'motif', f'top {top_n} motifs', motif_freq_heatmap_df, motif_count_df, percentile_rank_count_df, dvg_count, motif_sites[i], 'motifs', 'heatmaps', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_freq_diffs, top_freq_diffs

def make_reg_site_motif_heatmap_analysis(df: pd.DataFrame, heatmap_names: list, dvg_count: int, motif_length: int, top_n: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    mean_coefficients = []
    top_coefficients = []
    for i in range(0, 4):
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], motifs)
        motif_count_df = compute_feature_count_heatmap_sum_df(motif_count_heatmap_df, 'motif')
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
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
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)

        heatmap_name = heatmap_names[i]
        heatmap_name += f' - heatmap - motif length {motif_length}'
        heatmap_name += f' (mean coefficient = {mean_coefficient}, '
        heatmap_name += f'top coefficient = {top_coefficient})'

        create_freq_heatmap_plot(heatmap_name, 'motif', f'top {top_n} motifs', motif_freq_heatmap_df, motif_count_df, percentile_rank_count_df, dvg_count, 'reg_' + motif_sites[i], 'motifs', 'heatmaps', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_coefficients, top_coefficients

###########
### dvg ###
###########

### dvg ###

def make_dvg_motif_freq_analysis(df: pd.DataFrame, plot_name: str, dvg_count: int, motif_length: int, top_n: int, data: str, strain: str, segment: str, intersects: str):
    '''
    
    '''
    motifs = generate_motifs(motif_length)

    high_df, low_df = split_by_threshold(df, 'NGS_percentile_rank', RANK_THRESHOLD)
    high_dvg_count = high_df.shape[0]
    for _, row in high_df.iterrows():
        high_seq_motif_count_df = compute_seq_feature_count_df(row['dvg_sequence'], 'motif', motifs)
        high_motif_freq_df = compute_feature_freq_df(high_seq_motif_count_df, 'motif')

    low_df = low_df.sample(n=high_dvg_count, random_state=42)
    for _, row in low_df.iterrows():
        low_seq_motif_count_df = compute_seq_feature_count_df(row['dvg_sequence'], 'motif', motifs)
        low_motif_freq_df = compute_feature_freq_df(low_seq_motif_count_df, 'motif')

    comb_freq_df = subtract_freq_dfs('high_NGS_rank', high_motif_freq_df, 'low_NGS_ranks', low_motif_freq_df, 'motif')
    comb_freq_df = comb_freq_df.sort_values(by='difference', ascending=False).head(top_n)
    mean_freq_diff = round(comb_freq_df['difference'].mean(), DECIMALS)
    top_freq_diff = round(comb_freq_df['difference'].iloc[0], DECIMALS)

    bar_name = plot_name
    bar_name += f' - bar plot - motif length {motif_length}'
    bar_name += f' (mean freq diff = {mean_freq_diff}, '
    bar_name += f'top freq diff = {top_freq_diff})'

    create_freq_diff_plot(bar_name, 'motif', f'top {top_n} motifs', comb_freq_df, dvg_count, 'dvg', 'motifs', 'diffplots', data, strain, segment, intersects, f'motif_length_{motif_length}')

    return mean_freq_diff, top_freq_diff

def make_repeat_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, dvg_count: int, cap_length: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = add_direct_repeat_length(df)
    df = cap_direct_repeat_length(df, cap_length)

    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    repeat_count_heatmap_df = compute_feature_count_heatmap_df(df, 'direct_repeat_length')
    repeat_count_df = compute_feature_count_heatmap_sum_df(repeat_count_heatmap_df, 'direct_repeat_length')
    repeat_freq_heatmap_df = compute_feature_freq_heatmap_df(repeat_count_heatmap_df, percentile_rank_count_df)

    heatmap_name += f' - heatmap - cap length {cap_length}'

    create_freq_heatmap_plot(heatmap_name, 'direct_repeat_length', 'length of direct repeat (nucleotides)', repeat_freq_heatmap_df, repeat_count_df, percentile_rank_count_df, dvg_count, 'direct_repeat_length', 'repeats', 'heatmaps', data, strain, segment, intersects)

def make_mfe_heatmap_analysis(df: pd.DataFrame, heatmap_name: str, dvg_count: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = add_feature_percentile_rank(df, 'MFE', 'MFE_percentile_rank')
    df = rename_feature(df, 'MFE_percentile_rank', 'MFE_rank')

    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    mfe_count_heatmap_df = compute_feature_count_heatmap_df(df, 'MFE_rank')
    mfe_count_df = compute_feature_count_heatmap_sum_df(mfe_count_heatmap_df, 'MFE_rank')
    mfe_freq_heatmap_df = compute_feature_freq_heatmap_df(mfe_count_heatmap_df, percentile_rank_count_df)

    heatmap_name += f' - heatmap'

    create_freq_heatmap_plot(heatmap_name, 'MFE_rank', 'MFE rank (percentile)', mfe_freq_heatmap_df, mfe_count_df, percentile_rank_count_df, dvg_count, 'mfe', 'MFE', 'heatmaps', data, strain, segment, intersects)


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    top_n = 20

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
    subfolder = 'sec'

    selector = 'library_selection'

    data = 'IAV'
    strain = 'all'
    segment = 'all'
    intersects = 'median'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    # run_site_motif_heatmap_analysis(dfs, top_n, data, strain, segment, intersects)
    # run_reg_site_motif_heatmap_analysis(dfs, top_n, data, strain, segment, intersects)
    # run_repeat_heatmap_analysis(dfs, top_n, data, strain, segment, intersects)

    # run_dvg_motif_freq_analysis(dfs, top_n, data, strain, segment, intersects)
    # run_ngs_histo_analysis(dfs, data, strain, segment, intersects)

    # run_dvg_pri_features_analysis(dfnames, dfs, selector, top_n, data, strain, segment, intersects)
    # run_length_mfe_analysis(dfnames, dfs, data, selector, strain, segment, intersects)

    # run_sec_structure_plots(dfs, data, strain, segment)

    ### requires sec feature preprocessing ###
    
    # run_mfe_heatmap_analysis(dfs, data, strain, segment, intersects)

    # run_dvg_sec_features_analysis(dfnames, dfs, selector, data, strain, segment, intersects)
    # run_dvg_hybrid_features_analysis(dfnames, dfs, selector, data, strain, segment, intersects)
