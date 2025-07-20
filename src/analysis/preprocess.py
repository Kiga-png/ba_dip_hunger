'''

'''
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '..')

from utils import get_dataset_names, load_all
from utils import manage_specifiers, load_all_preprocessed, merge_missing_features, save_df
from utils import rename_feature, add_feature_percentile_rank, manage_intersects
from utils import add_dvg_sequence, add_feature_percentile_rank, add_sec_features

from utils import generate_motifs, add_site_motifs, add_lin_reg_rows, compute_full_seq_motif_freq_df
from utils import compute_percentile_rank_count_df, compute_feature_count_df, compute_feature_freq_df, subtract_freq_dfs
from utils import compute_feature_count_heatmap_df, compute_feature_freq_heatmap_df

from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import COLORS, STRAINS, A_STRAINS, B_STRAINS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'preprocess')

###############
### scripts ###
###############



###############
### general ###
###############

def preprocess_site_motif_features(dfs: list, motif_length: int, data: str = '', strain: str = '', segment: str = '', intersects: str = ''):
    '''

    '''
    save_site_motif_features(dfs, motif_length, data, strain, segment, intersects)

def preprocess_reg_site_motif_features(dfs: list, motif_length: int, data: str = '', strain: str = '', segment: str = '', intersects: str = ''):
    '''

    '''
    save_reg_site_motif_features(dfs, motif_length, data, strain, segment, intersects)

def preprocess_sec_dvg_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_sec_dvg_features(df, dfname, 'datasets', 'sec_dvg')

def preprocess_sec_site_motif_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_sec_site_motif_features(df, dfname, 'datasets', 'sec_motif')

def preprocess_sec_combine_features(dfnames: list):
    '''

    '''
    for dfname in dfnames:
        save_merged_sec_features(dfname)

################
### features ###
################

### pri features ###

def save_site_motif_features(dfs: list, motif_length: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = pd.concat(dfs, ignore_index=True)
    df, data, strain, segment = manage_specifiers(df, data, strain, segment)
    df, intersects = manage_intersects(df, intersects)

    df, skipped_count = add_site_motifs(df, motif_length)
    print(f'{skipped_count} candidates have been left out')

    df_copy = df.copy()
    df = add_feature_percentile_rank(df, 'NGS_read_count', 'NGS_percentile_rank')

    motif_sites = ['site0_motif', 'site1_motif', 'site2_motif', 'site3_motif']

    motifs = generate_motifs(motif_length)
    full_seq_motif_freq_df = compute_full_seq_motif_freq_df(motif_length, data, strain, segment)

    for i in range(0, 4):
        motif_count_df = compute_feature_count_df(df, motif_sites[i], motifs)
        motif_freq_df = compute_feature_freq_df(motif_count_df, motif_sites[i])
        motif_freq_df = rename_feature(motif_freq_df, motif_sites[i], 'motif')
        comb_freq_df = subtract_freq_dfs(motif_freq_df, full_seq_motif_freq_df, 'motif')

        comb_freq_df = comb_freq_df.sort_values(by='freq_diff', ascending=False).head(3)
        filtered_motifs = comb_freq_df['motif'].tolist()

        motif_col = f'site{i}_motif'
        for filtered_motif in filtered_motifs:
            check_col = f'site{i}_motif{filtered_motifs.index(filtered_motif)}'
            df_copy[check_col] = df_copy[motif_col].apply(lambda x: 1 if x == filtered_motif else 0)

    fname = f'motif_length_{motif_length}'
    save_df(df_copy, fname, 'combined', 'motif', data, strain, segment, intersects)

def save_reg_site_motif_features(dfs: list, motif_length: int, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = pd.concat(dfs, ignore_index=True)
    df, data, strain, segment = manage_specifiers(df, data, strain, segment)
    df, intersects = manage_intersects(df, intersects)

    df, skipped_count = add_site_motifs(df, motif_length)
    print(f'{skipped_count} candidates have been left out')

    df_copy = df.copy()
    df = add_feature_percentile_rank(df, 'NGS_read_count', 'NGS_percentile_rank')

    motif_sites = ['site0_motif', 'site1_motif', 'site2_motif', 'site3_motif']

    motifs = generate_motifs(motif_length)
    percentile_rank_count_df = compute_percentile_rank_count_df(df)

    for i in range(0, 4):
        motif_count_heatmap_df = compute_feature_count_heatmap_df(df, motif_sites[i], motifs)
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
        reg_motif_freq_heatmap_df = add_lin_reg_rows(motif_freq_heatmap_df)

        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_df.T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(3)
        filtered_motifs = trans_reg_motif_freq_heatmap_df.index.tolist()

        motif_col = f'site{i}_motif'
        for filtered_motif in filtered_motifs:
            check_col = f'site{i}_motif{filtered_motifs.index(filtered_motif)}'
            df_copy[check_col] = df_copy[motif_col].apply(lambda x: 1 if x == filtered_motif else 0)

    fname = f'motif_length_{motif_length}'
    save_df(df_copy, fname, 'combined', 'motif', data, strain, segment, intersects)

### sec features ###

def save_sec_dvg_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    df = add_dvg_sequence(df)
    df = add_sec_features(df, 'dvg_sequence', 'structure', 'MFE')
    save_df(df, fname, folder, subfolder)

def save_sec_site_motif_features(df: pd.DataFrame, fname: str, folder: str = '', subfolder: str = ''):
    '''

    '''
    df, skipped_count = add_site_motifs(df, 25)
    print(f'{skipped_count} have been left out')

    for i in range(0, 4):
        df = add_sec_features(df, f'site{i}_motif', f'structure_site{i}_motif', f'MFE_site{i}_motif')
    save_df(df, fname, folder, subfolder)

def save_merged_sec_features(fname: str):
    '''

    '''
    read_path = RESULTSPATH
    folder = 'datasets'
    read_path = os.path.join(read_path, folder)
    fname_copy = fname + '.csv'

    read_path1 = os.path.join(read_path, 'sec_dvg')
    df1 = pd.read_csv(os.path.join(read_path1, fname_copy))

    read_path2 = os.path.join(read_path, 'sec_motif')
    df2 = pd.read_csv(os.path.join(read_path2, fname_copy))

    df = merge_missing_features(df1, df2)
    save_df(df, fname, folder, 'sec_full')


if __name__ == '__main__':
    '''

    '''
    #################
    ### SELECTION ###
    #################

    ### SINGLE ###

    index = 20

    dfnames = get_dataset_names(cutoff=40)
    dfnames = [dfnames[index]]
    dfs, _ = load_all(dfnames, False)

    data = dfnames[0]
    strain = DATASET_STRAIN_DICT[data]
    segment = ''
    intersects = ''

    ### MULTI ###

    # data = 'IBV'
    # strain = ''
    # segment = ''
    # intersects = ''

    # dfnames = get_dataset_names(cutoff=40, selection=data)
    # dfs, _ = load_all(dfnames, False)

    ### PREPROCESS SINGLE ###

    # folder = 'datasets'
    # subfolder = 'sec_full'

    # index = 20

    # dfnames = get_dataset_names(cutoff=40)
    # dfnames = [dfnames[index]]
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)

    # data = dfnames[0]
    # strain = DATASET_STRAIN_DICT[data]
    # segment = ''
    # intersects = ''
    
    ### PREPROCESS MULTI ###

    # folder = 'datasets'
    # subfolder = 'sec_full'

    # data = 'IBV'
    # strain = ''
    # segment = ''
    # intersects = ''

    # # dfnames = get_dataset_names(cutoff=40, selection=data)
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    # preprocess_sec_dvg_features(dfnames, dfs)
    # preprocess_sec_site_motif_features(dfnames, dfs)
    # preprocess_sec_combine_features(dfnames)

    # preprocess_site_motif_features(dfs, 3, data, strain, segment, intersects)
