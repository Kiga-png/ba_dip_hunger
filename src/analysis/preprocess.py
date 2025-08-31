'''

'''
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '..')

from utils import get_dataset_names, load_all
from utils import manage_separate_specifiers, load_preprocessed_dataset, load_all_preprocessed, merge_missing_features, save_df
from utils import add_metadata_features
from utils import manage_intersects, add_ikey, set_intersect_proportion
from utils import rename_feature, add_log_feature, add_norm_feature, add_feature_percentile_rank, add_separate_ngs_features
from utils import add_dvg_sequence, add_feature_percentile_rank, add_sec_features

from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df, insert_pseudo_motif
from utils import compute_percentile_rank_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_freq_heatmap_df

from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import COLORS, STRAINS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'preprocess')

###############
### scripts ###
###############

### base features ###

def preprocess_base_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_base_features(df, dfname, 'datasets', 'base')

### pri features ###

def preprocess_site_motif_features(dfnames: list, dfs: list, motif_length: int, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    save_site_motif_features(df, motif_length, 'combined', 'pri', data, strain, segment, intersects)

def preprocess_reg_site_motif_features(dfnames: list, dfs: list, motif_length: int, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    save_reg_site_motif_features(df, motif_length, 'combined', 'pri', data, strain, segment, intersects)

### sec features ###

def preprocess_sec_dvg_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_sec_dvg_features(df, dfname, 'datasets', 'sec_dvg')

def preprocess_sec_site_motif_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_sec_site_motif_features(df, dfname, 'datasets', 'sec_site')

def preprocess_sec_combine_features(dfnames: list):
    '''

    '''
    for dfname in dfnames:
        save_merged_sec_features(dfname, 'datasets', 'sec')

### pseudo ###

def preprocess_pseudo_motif_features(motif_length: int, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    folder = 'datasets'
    subfolder = 'base'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    folder = 'combined'
    df = insert_pseudo_motif_features(df)
    save_site_motif_features(df, motif_length, folder, 'pseudo_motif_ACA', data, strain, segment, intersects)

def preprocess_pseudo_intersect_features(motif_length: int, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    folder = 'datasets'
    subfolder = 'base'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)
    dfs = add_metadata_features(dfnames, dfs)
    df = pd.concat(dfs, ignore_index=True)

    folder = 'combined'
    df10 = insert_pseudo_intersect_features(df, 0.10)
    save_site_motif_features(df10, motif_length, folder, 'pseudo_intersects_10', data, strain, segment, intersects)
    df20 = insert_pseudo_intersect_features(df, 0.20)
    save_site_motif_features(df20, motif_length, folder, 'pseudo_intersects_20', data, strain, segment, intersects)
    df30 = insert_pseudo_intersect_features(df, 0.30)
    save_site_motif_features(df30, motif_length, folder, 'pseudo_intersects_30', data, strain, segment, intersects)
    df40 = insert_pseudo_intersect_features(df, 0.40)
    save_site_motif_features(df40, motif_length, folder, 'pseudo_intersects_40', data, strain, segment, intersects)
    df50 = insert_pseudo_intersect_features(df, 0.50)
    save_site_motif_features(df50, motif_length, folder, 'pseudo_intersects_50', data, strain, segment, intersects)

################
### features ###
################

### base features ###

def save_base_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    save_df(df, fname, RESULTSPATH, folder, subfolder)

### pri features ###

def save_site_motif_features(df: pd.DataFrame, motif_length: int, folder: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
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
    save_df(df_copy, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)
    save_df(filtered_motif_df, fname, RESULTSPATH, 'filetred_motifs', subfolder, data, strain, segment, intersects)

def save_reg_site_motif_features(df: pd.DataFrame, motif_length: int, folder: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
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
    save_df(df_copy, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)
    save_df(filtered_motif_df, fname, RESULTSPATH, 'filetred_motifs', subfolder, data, strain, segment, intersects)

### sec features ###

def save_sec_dvg_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    df = add_dvg_sequence(df)
    df = add_sec_features(df, 'dvg_sequence', 'structure', 'MFE')
    save_df(df, fname, RESULTSPATH, folder, subfolder)

def save_sec_site_motif_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    df, skipped_count = add_site_motifs(df, 25)
    print(f'{skipped_count} have been left out')

    for i in range(1, 5):
        df = add_sec_features(df, f'site{i}_motif', f'structure_site{i}_motif', f'MFE_site{i}_motif')
    save_df(df, fname, RESULTSPATH, folder, subfolder)

def save_merged_sec_features(fname: str, folder: str, subfolder: str):
    '''

    '''
    read_path = RESULTSPATH
    read_path = os.path.join(read_path, folder)
    fname_copy = fname + '.csv'

    read_path1 = os.path.join(read_path, 'sec_dvg')
    df1 = pd.read_csv(os.path.join(read_path1, fname_copy), keep_default_na=False, na_values=[])

    read_path2 = os.path.join(read_path, 'sec_site')
    df2 = pd.read_csv(os.path.join(read_path2, fname_copy), keep_default_na=False, na_values=[])

    df = merge_missing_features(df1, df2)
    save_df(df, fname, RESULTSPATH, folder, subfolder)

### pseudo ###

def insert_pseudo_motif_features(df: pd.DataFrame):
    '''

    '''
    feature = 'norm_log_NGS_read_count'

    df, skipped_count = insert_pseudo_motif(df, 3, 'ACA', 'site0_motif', feature, 0.5, 0.8, True)
    print(f'{skipped_count} candidates have been left out')
    df, skipped_count = insert_pseudo_motif(df, 3, 'ACA', 'site2_motif', feature, 0.5, 0.8, True)
    print(f'{skipped_count} candidates have been left out')

    return df

def insert_pseudo_intersect_features(df: pd.DataFrame, threshold: str):
    '''

    '''
    df = add_ikey(df)
    df = set_intersect_proportion(df, threshold)
    df = df.drop('ikey', axis=1)

    return df

if __name__ == '__main__':
    '''

    '''
    #################
    ### SELECTION ###
    #################

    ### BASE ###

    # dfnames = get_dataset_names(cutoff=40)
    # dfs, _ = load_all(dfnames, False)

    ### DATASETS SINGLE ###

    # folder = 'datasets'
    # subfolder = 'sec'

    # data = 'Alnaji2021'
    # strain = DATASET_STRAIN_DICT[data]
    # segment = 'all'
    # intersects = 'median'

    # dfnames = [data]
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)
    
    ### DATASETS MULTI ###

    folder = 'datasets'
    subfolder = 'sec'

    data = 'IAV'
    strain = 'PR8'
    segment = 'PB1'
    intersects = 'median'

    dfnames = get_dataset_names(cutoff=40, selection=data)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### basic features ###

    # preprocess_base_features(dfnames, dfs)

    ### pri features ###
    
    preprocess_site_motif_features(dfnames, dfs, 3, data, strain, segment, intersects)

    ### sec features ###

    # preprocess_sec_dvg_features(dfnames, dfs)
    # preprocess_sec_site_motif_features(dfnames, dfs)
    # preprocess_sec_combine_features(dfnames)

    ### pseudo ###

    # preprocess_pseudo_motif_features(3, data, strain, segment, intersects)
    # preprocess_pseudo_intersect_features(3, data, strain, segment, intersects)

