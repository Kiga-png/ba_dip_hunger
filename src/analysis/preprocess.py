'''

'''
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '..')

from utils import load_all, get_dataset_names, get_sequence

from utils import get_strains, manage_specifiers, manage_separate_specifiers, load_all_unpooled, load_all_preprocessed,  save_df
from utils import add_metadata_features
from utils import manage_intersects, add_ikey, set_intersect_proportion
from utils import rename_feature, split_by_number, add_log_feature, add_norm_feature, add_feature_percentile_rank, add_separate_ngs_features

from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df, insert_pseudo_motif
from utils import compute_percentile_rank_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_freq_heatmap_df
from utils import build_df, generate_candidates

from utils import (
    add_dvg_sequence,
    add_deletion_length,
    add_region_lengths,
    add_dvg_length,
    add_full_seq_length,
    add_marked_dvg_sequence,
)

from utils import (
    add_direct_repeat_length,
    cap_direct_repeat_length,
)

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
    add_sec_features,
    add_marked_structure,
    add_max_symmetry,
)

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
from utils import K_MER_LENGTH, PALINDROMIC_K_MER_LENGTH, DIRECT_REPEAT_LEGTH_CAP
from utils import STRAIN_SUBTYPE_DICT

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'preprocess')

###############
### scripts ###
###############

### base features ###

def preprocess_base_features(dfnames: list, dfs: list, folder: str):
    '''

    '''
    dfs = add_metadata_features(dfnames, dfs)

    for dfname, df in zip(dfnames, dfs):
        save_base_features(df, dfname, folder, 'base')

### structure features ###

def preprocess_structure_features(dfnames: list, dfs: list, folder: str):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_structure_features(df, dfname, folder, 'structure')

### training ###

def preprocess_site_motifs(dfnames: list, dfs: list, split: bool, folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    df = pd.concat(dfs, ignore_index=True)

    save_site_motifs(df, split, folder, 'training', data, strain, segment, intersects)

def preprocess_reg_site_motifs(dfnames: list, dfs: list, split: bool, folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    save_reg_site_motifs(df, split, folder, 'training', data, strain, segment, intersects)

### pseudo ###

def preprocess_pseudo_motifs(split: bool, folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    subfolder = 'base'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    folder = 'combined'
    df = insert_pseudo_motifs(df)
    save_site_motifs(df, split, folder, 'pseudo_motif_ACA', data, strain, segment, intersects)

def preprocess_pseudo_intersects(split: bool, folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    subfolder = 'base'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    folder = 'combined'
    df10 = insert_pseudo_intersects(df, 0.10)
    save_site_motifs(df10, split, folder, 'pseudo_intersects_10', data, strain, segment, intersects)
    df20 = insert_pseudo_intersects(df, 0.20)
    save_site_motifs(df20, split, folder, 'pseudo_intersects_20', data, strain, segment, intersects)
    df30 = insert_pseudo_intersects(df, 0.30)
    save_site_motifs(df30, split, folder, 'pseudo_intersects_30', data, strain, segment, intersects)
    df40 = insert_pseudo_intersects(df, 0.40)
    save_site_motifs(df40, split, folder, 'pseudo_intersects_40', data, strain, segment, intersects)
    df50 = insert_pseudo_intersects(df, 0.50)
    save_site_motifs(df50, split, folder, 'pseudo_intersects_50', data, strain, segment, intersects)

### prediction ###

def preprocess_prediction_candidates(candidates: list, folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    """
    
    """
    subfolder = 'preparation'

    df = build_df(candidates)
    df["subtype"] = df["Strain"].map(STRAIN_SUBTYPE_DICT).fillna("other")

    df = manage_specifiers(df, data, strain, segment)
    df = add_log_feature(df, 'NGS_read_count', 'log_NGS_read_count')
    df = add_norm_feature(df, 'log_NGS_read_count', 'norm_log_NGS_read_count')

    df = add_dvg_sequence(df)
    df = add_sec_features(df, 'dvg_sequence', 'structure', 'MFE')

    save_site_motifs(df, False, folder, subfolder, data, strain, segment, intersects)

def preprocess_prediction(folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    """
    
    """
    subfolder = 'preparation'

    candidates = compute_candidates(1000, data, strain, segment)
    df = build_df(candidates)

    df["subtype"] = df["Strain"].map(STRAIN_SUBTYPE_DICT).fillna("other")

    df = manage_specifiers(df, data, strain, segment)
    df = add_log_feature(df, 'NGS_read_count', 'log_NGS_read_count')
    df = add_norm_feature(df, 'log_NGS_read_count', 'norm_log_NGS_read_count')

    df = add_dvg_sequence(df)
    df = add_sec_features(df, 'dvg_sequence', 'structure', 'MFE')

    save_site_motifs(df, False, folder, subfolder, data, strain, segment, intersects)

################
### general ###
################

### base features ###

def save_base_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    ### length ###
    df = add_dvg_sequence(df)
    df = add_deletion_length(df)
    df = add_region_lengths(df)
    df = add_dvg_length(df)
    df = add_full_seq_length(df)
    df = add_marked_dvg_sequence(df)

    ### direct repeats ###
    df = add_direct_repeat_length(df)
    df = cap_direct_repeat_length(df, DIRECT_REPEAT_LEGTH_CAP)

    ### pri ###
    df = add_gc_content(df)
    df = add_au_content(df)
    df = add_upa_content(df)
    df = add_cpg_content(df)
    df = add_gc_skew(df)
    df = add_sequence_entropy(df)
    df = add_poly_run_features(df, 'A')
    df = add_poly_run_features(df, 'U')
    df = add_palindrome_density(df, PALINDROMIC_K_MER_LENGTH)
    df = add_orf_features(df)
    df = add_kmer_richness(df, K_MER_LENGTH)
    df = add_codon_usage_bias(df)

    save_df(df, fname, RESULTSPATH, folder, subfolder)

### structure features ###

def save_structure_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    ### seuquence ###
    df = add_dvg_sequence(df)

    ### structure ###
    df = add_sec_features(df, 'dvg_sequence', 'structure', 'MFE')
    df = add_marked_structure(df)

    ### symmetry
    df = add_max_symmetry(df)

    ### sec ###
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

    ### hybrid ###
    df = add_gc_overall(df)
    df = add_gc_paired_unpaired(df)
    df = add_canonical_pair_stats(df)
    df = add_stem_end_pair_enrichment(df)
    df = add_hairpin_closing_pair_content(df)
    df = add_tetraloop_motif_counts(df)
    df = add_loop_au_content(df)
    df = add_tail_gc_content(df)
    df = add_start_codon_accessibility(df)
    save_df(df, fname, RESULTSPATH, folder, subfolder)

### training ###

def save_site_motifs(df: pd.DataFrame, split: bool, folder: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_length = K_MER_LENGTH
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df, skipped_count = add_site_motifs(df, motif_length)
    print(f'{skipped_count} candidates have been left out')

    df_copy = df.copy()
    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')

    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    full_seq_motif_freq_df = compute_full_seq_motif_freq_df(motif_length, data, strain, segment)

    filtered_motif_df = pd.DataFrame([{}])
    for motif_site in motif_sites:
        motif_count_df = compute_feature_count_df(df, motif_site, motifs)
        motif_freq_df = compute_feature_freq_df(motif_count_df, motif_site)
        motif_freq_df = rename_feature(motif_freq_df, motif_site, 'motif')
        comb_freq_df = subtract_freq_dfs('site', motif_freq_df, 'full_sequence', full_seq_motif_freq_df, 'motif')

        comb_freq_df = comb_freq_df.sort_values(by='difference', ascending=False).head(3)
        filtered_motifs = comb_freq_df['motif'].tolist()

        for filtered_motif in filtered_motifs:
            check_col = f'{motif_site}{filtered_motifs.index(filtered_motif) + 1}'
            df_copy[check_col] = df_copy[motif_site].apply(lambda x: 1 if x == filtered_motif else 0)
            filtered_motif_df[check_col] = filtered_motif

    fname = f'motif_length_{motif_length}'

    save_df(filtered_motif_df, fname, RESULTSPATH, 'filetred_motifs', subfolder, data, strain, segment, intersects)

    if split:
        sampled_df, remaining_df = split_by_number(df_copy, 100)
        save_df(sampled_df, fname, RESULTSPATH, folder, "preparation", data, strain, segment, intersects)
        save_df(remaining_df, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)
    else:
        save_df(df_copy, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)

def save_reg_site_motifs(df: pd.DataFrame, split: bool, folder: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    motif_length = K_MER_LENGTH
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df, skipped_count = add_site_motifs(df, motif_length)
    print(f'{skipped_count} candidates have been left out')

    df_copy = df.copy()
    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')

    motif_sites = ['site1_motif', 'site2_motif', 'site3_motif', 'site4_motif']

    motifs = generate_motifs(motif_length)
    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    filtered_motif_df = pd.DataFrame([{}])
    for motif_site in motif_sites:
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_site, motifs)
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
        reg_motif_freq_heatmap_df = add_lin_reg_rows(motif_freq_heatmap_df)

        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_df.T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(3)
        filtered_motifs = trans_reg_motif_freq_heatmap_df.index.tolist()

        for filtered_motif in filtered_motifs:
            check_col = f'{motif_site}{filtered_motifs.index(filtered_motif) + 1}'
            df_copy[check_col] = df_copy[motif_site].apply(lambda x: 1 if x == filtered_motif else 0)
            filtered_motif_df[check_col] = filtered_motif

    fname = f'motif_length_{motif_length}'

    save_df(filtered_motif_df, fname, RESULTSPATH, 'filetred_motifs', subfolder, data, strain, segment, intersects)

    if split:
        sampled_df, remaining_df = split_by_number(df_copy, 100)
        save_df(sampled_df, fname, RESULTSPATH, folder, "preparation", data, strain, segment, intersects)
        save_df(remaining_df, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)
    else:
        save_df(df_copy, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)

### pseudo ###

def insert_pseudo_motifs(df: pd.DataFrame):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    df, skipped_count = insert_pseudo_motif(df, 3, 'ACA', 'site0_motif', feature, 0.5, 0.8, True)
    print(f'{skipped_count} candidates have been left out')
    df, skipped_count = insert_pseudo_motif(df, 3, 'ACA', 'site2_motif', feature, 0.5, 0.8, True)
    print(f'{skipped_count} candidates have been left out')

    return df

def insert_pseudo_intersects(df: pd.DataFrame, threshold: str):
    '''

    '''
    df = add_ikey(df)
    df = set_intersect_proportion(df, threshold)
    df = df.drop('ikey', axis=1)

    return df

### prediction ###

def compute_candidates(candidates_number: int, data: str, strain: str, segment: str):
    '''

    '''
    dataset_names = get_dataset_names(cutoff=40, selection=data)
    strains = get_strains(dataset_names)

    if segment != 'all' and strain != 'all':
        full_seq = get_sequence(strain, segment)
        full_seq_len = len(full_seq)
        candidates = generate_candidates(candidates_number, strain, segment, full_seq_len)
        return candidates
    
    candidates = []

    if strain != 'all':
        sub_candidates_number = int(candidates_number / len(SEGMENTS))
        for segment in SEGMENTS:
            full_seq = get_sequence(strain, segment)
            full_seq_len = len(full_seq)
            sub_candidates = generate_candidates(sub_candidates_number, strain, segment, full_seq_len)
            candidates += sub_candidates
        
        return candidates
            
    if segment != 'all':
        sub_candidates_number = int(candidates_number / len(strains))
        for strain in strains:
            full_seq = get_sequence(strain, segment)
            full_seq_len = len(full_seq)
            sub_candidates = generate_candidates(sub_candidates_number, strain, segment, full_seq_len)
            candidates += sub_candidates
        
        return candidates
    
    sub_candidates_number = int(candidates_number / (len(strains) * len(SEGMENTS)))
    for strain in strains:
        for segment in SEGMENTS:
            full_seq = get_sequence(strain, segment)
            full_seq_len = len(full_seq)
            sub_candidates = generate_candidates(sub_candidates_number, strain, segment, full_seq_len)
            candidates += sub_candidates

    return candidates
        

if __name__ == '__main__':
    '''

    '''
    #################
    ### SELECTION ###
    #################

    ### BASE ###

    # folder = 'pooled'

    # dfnames = get_dataset_names(cutoff=40)

    # dfs, _ = load_all(dfnames, False)
    # dfs, _ = load_all_unpooled(dfnames, False)

    ### DATASETS SINGLE ###

    # split = True

    # folder = 'pooled'
    # subfolder = 'base'

    # data = 'Alnaji2019_Perth'
    # strain = DATASET_STRAIN_DICT[data]
    # segment = 'all'
    # intersects = 'all'

    # dfnames = [data]
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)
    
    ### DATASETS MULTI ###

    split = True

    folder = 'unpooled'
    subfolder = 'base'

    data = 'IAV'
    strain = 'PR8'
    segment = 'PB1'
    intersects = 'median with metadata'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### base ###

    # preprocess_base_features(dfnames, dfs, folder)

    ### training ###
    
    preprocess_site_motifs(dfnames, dfs, split, folder, data, strain, segment, intersects)

    ### structure ###

    # preprocess_structure_features(dfnames, dfs, folder)

    ### pseudo ###

    # preprocess_pseudo_motifs(split, folder, data, strain, segment, intersects)
    # preprocess_pseudo_intersects(split, folder, data, strain, segment, intersects)

    ### prediction ###

    # candidates = [
    #     ("PR8", "PB1", 100, 1500, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 110, 1500, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 120, 1500, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 100, 1510, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 110, 1510, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 120, 1510, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),

    #     ("PR8", "PB1", 100, 1520, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 110, 1520, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 120, 1520, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 100, 1530, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 110, 1530, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ("PR8", "PB1", 120, 1530, "in vitro", "PAIRED", "PCR", "TRANSCRIPTOMIC"),
    #     ]
    
    # preprocess_prediction_candidates(candidates, folder, data, strain, segment, intersects)
    # preprocess_prediction(folder, data, strain, segment, intersects)
    


