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

sys.path.insert(0, '..')
from visuals import compute_feature_freq_heatmap_df, compute_motif_freq_dfs, compute_motif_count_heatmap_dfs
from visuals import add_linear_regression_rows, compute_batif_freq_df, subtract_freq_dfs

from utils import generate_motif_list, add_site_motifs



from utils import get_dataset_names, load_all
from utils import manage_specifiers, load_all_preprocessed, merge_missing_features, save_df
from utils import add_dvg_sequence, add_feature_percentile_rank, add_sec_features

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

def preprocess_motif_features(dfs: list, motif_length: int, data: str = '', strain: str = '', segment: str = ''):
    '''

    '''
    save_site_motif_features(dfs, motif_length, data, strain, segment)

def preprocess_reg_motif_features(dfs: list, motif_length: int, data: str = '', strain: str = '', segment: str = ''):
    '''

    '''
    save_reg_site_motif_features(dfs, motif_length, data, strain, segment)

def preprocess_sec_dvg_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_sec_dvg_features(df, dfname, 'datasets', 'sec_dvg')

def preprocess_sec_motif_features(dfnames: list, dfs: list):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_sec_motif_features(df, dfname, 'datasets', 'sec_motif')

def preprocess_sec_features(dfnames: list):
    '''

    '''
    for dfname in dfnames:
        save_merged_sec_features(dfname)

################
### features ###
################

### pri features ###

def save_site_motif_features(dfs: list, motif_length: int, data: str, strain: str, segment: str):
    '''

    '''
    df = pd.concat(dfs, ignore_index=True)
    df, data, strain, segment = manage_specifiers(df, data, strain, segment)

    df, skipped_count = add_site_motifs(df, motif_length)
    print(f'{skipped_count} have been left out')

    df_copy = df.copy()
    df = add_feature_percentile_rank(df, 'NGS_read_count', 'NGS_percentile_rank')

    motif_freq_dfs = compute_motif_freq_dfs(df, motif_length)
    batif_freq_df = compute_batif_freq_df(motif_length, data, strain, segment)

    for i in range(0, 4):
        comb_freq_df = subtract_freq_dfs(motif_freq_dfs[i], batif_freq_df)
        top_comb_freq_df = comb_freq_df.nlargest(motif_length, 'top_freq')
        top_comb_freq_df = comb_freq_df.sort_values(by='top_freq', ascending=False).head(3)
        top_motif_list = top_comb_freq_df['motif'].tolist()
        motif_col = f'site{i}_motif'
        for motif in top_motif_list:
            check_col = f'site{i}_motif{top_motif_list.index(motif)}'
            df_copy[check_col] = df_copy[motif_col].apply(lambda x: 1 if x == motif else 0)

    fname = f'motif_length_{motif_length}'
    save_df(df_copy, fname, 'combined', 'motif', data, strain, segment)

def save_reg_site_motif_features(dfs: list, motif_length: int, data: str, strain: str, segment: str):
    '''

    '''
    df = pd.concat(dfs, ignore_index=True)
    df, data, strain, segment = manage_specifiers(df, data, strain, segment)

    df, skipped_count = add_site_motifs(df, motif_length)
    print(f'{skipped_count} have been left out')

    df_copy = df.copy()
    df = add_feature_percentile_rank(df, 'NGS_read_count', 'NGS_percentile_rank')

    possible_motifs = generate_motif_list(motif_length)
    possible_motifs_list = [possible_motifs] * 4
    motif_count_heatmap_dfs, percentile_rank_count_df = compute_motif_count_heatmap_dfs(df, possible_motifs_list)

    motif_freq_heatmap_dfs = []
    for motif_count_heatmap_df in motif_count_heatmap_dfs:
        motif_freq_heatmap_df = compute_feature_freq_heatmap_df(motif_count_heatmap_df, percentile_rank_count_df)
        motif_freq_heatmap_dfs.append(motif_freq_heatmap_df)

    reg_motif_freq_heatmap_dfs = add_linear_regression_rows(motif_freq_heatmap_dfs)

    for i in range(0, 4):
        trans_reg_motif_freq_heatmap_df = reg_motif_freq_heatmap_dfs[i].T
        trans_reg_motif_freq_heatmap_df['coefficient'] = trans_reg_motif_freq_heatmap_df['coefficient'].abs()
        top_trans_reg_motif_freq_heatmap_df = trans_reg_motif_freq_heatmap_df.sort_values(by='coefficient', ascending=False).head(3)
        top_motifs = top_trans_reg_motif_freq_heatmap_df.index.tolist()
        motif_col = f'site{i}_motif'
        for motif in top_motifs:
            check_col = f'site{i}_motif{top_motifs.index(motif)}'
            df_copy[check_col] = df_copy[motif_col].apply(lambda x: 1 if x == motif else 0)

    fname = f'motif_length_{motif_length}'
    save_df(df_copy, fname, 'combined', 'motif', data, strain, segment)

### sec features ###

def save_sec_dvg_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    df = add_dvg_sequence(df)
    df = add_sec_features(df, 'dvg_sequence', 'structure', 'MFE')
    save_df(df, fname, folder, subfolder)

def save_sec_motif_features(df: pd.DataFrame, fname: str, folder: str = '', subfolder: str = ''):
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
    save_df(df, fname, folder, 'sec_all')


if __name__ == '__main__':
    '''

    '''
    plt.style.use('seaborn')
    plt.rc('font', size=12)

    #################
    ### SELECTION ###
    #################

    ### SINGLE ###

    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames, False)

    index = 1

    dfnames = [dfnames[index]]
    dfs = [dfs[index]]

    ### MULTI ###

    # data = 'IBV'
    # dfnames = get_dataset_names(cutoff=40, selection=data)
    # dfs, _ = load_all(dfnames, False)
    
    ### PREPROCESS ###

    # folder = 'datasets'
    # subfolder = 'sec_all'

    # # data = "IBV"
    # # dfnames = get_dataset_names(cutoff=40, selection=data)
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    preprocess_sec_dvg_features(dfnames, dfs)
    # preprocess_sec_motif_features(dfnames, dfs)
    # preprocess_sec_features(dfnames)

    # preprocess_motif_features(dfs, 5, 'IAV', 'PR8')
