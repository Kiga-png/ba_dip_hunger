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

import re

sys.path.insert(0, "..")
from utils import get_dataset_names, load_all
from utils import manage_separate_specifiers, load_all_preprocessed, save_df
from utils import rename_feature, filter_for_feature, filter_against_feature, add_feature_percentile_rank, add_log_feature, add_norm_feature, add_separate_ngs_features
from utils import manage_intersects
from utils import add_dvg_sequence, add_dvg_length, add_cg_content, add_region_lengths
from utils import mannwhitneyu_by_threshold

from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import COLORS, STRAINS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'statistics')


###############
### scripts ###
###############

### statistics ###

def run_pri_mannwhitneyu(dfs: list, fname: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    dfs = add_separate_ngs_features(dfs, True)

    df = pd.concat(dfs, ignore_index=True)
    df = manage_intersects(df, intersects, 'norm_log_NGS_read_count')

    df = add_feature_percentile_rank(df, 'norm_log_NGS_read_count', 'NGS_percentile_rank')
    stats = make_pri_mannwhitneyu(df, data, strain, segment, intersects)
    save_df(stats, fname, RESULTSPATH, 'mannwhitneyu', 'pri', data, strain, segment, intersects)

###############
### general ###
###############

### statistics ###

def make_pri_mannwhitneyu(df: pd.DataFrame, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    df = add_dvg_sequence(df)
    df = add_dvg_length(df)
    df = add_cg_content(df)
    df = add_region_lengths(df)

    features = [col for col in df.columns if re.search(r"motif\d+", col)] + ["Start", "End", "dvg_length", "deletion_length", "5_end_length", "3_end_length", "cg_content"]

    stats = []
    for feature in features:
        feature_stats = mannwhitneyu_by_threshold(df, feature, 'NGS_percentile_rank', 20)
        stats.append(feature_stats)

    return pd.DataFrame(stats)


if __name__ == "__main__":
    '''

    '''

    #################
    ### SELECTION ###
    #################
    
    ### COMBINED SINGLE ###

    # folder = 'combined'
    # subfolder = 'pri'

    # data = ''
    # strain = DATASET_STRAIN_DICT[data]
    # segment = 'all'
    # intersects = 'all'

    # motif_length = 3

    # dfname = f'motif_length_{motif_length}'
    # dfnames = [dfname]
    # dfs = load_all_preprocessed(dfnames, folder, subfolder, data, strain, segment, intersects)

    ### COMBINED MULTI ###

    folder = 'combined'
    subfolder = 'pri'

    data = 'IAV'
    strain = 'PR8'
    segment = 'PB1'
    intersects = 'mean'

    motif_length = 3

    dfname = f'motif_length_{motif_length}'
    dfnames = [dfname]
    dfs = load_all_preprocessed(dfnames, folder, subfolder, data, strain, segment, intersects)

    ###################
    ### run scripts ###
    ###################

    ### requires pri feature preprocessing ###

    run_pri_mannwhitneyu(dfs, dfname, data, strain, segment, intersects)
    run_pseudo_pri_mannwhitneyu(dfs, dfname, data, strain, segment, intersects)