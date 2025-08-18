'''

'''
import os
import sys

import pandas as pd

sys.path.insert(0, "..")

from pathlib import Path

from utils import get_dataset_names, load_all
from utils import load_all_preprocessed, save_df
from utils import add_log_feature, add_norm_feature

from utils import add_dvg_sequence, add_marked_dvg_sequence, add_dvg_length, add_cg_content, add_region_lengths
from utils import add_marked_structure, add_loop_count, add_max_loop

from utils import RESULTSPATH, DATASET_STRAIN_DICT

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, "networks")


###############
### scripts ###
###############

def make_training_dfs():
    '''

    '''
    data_path, _ = os.path.split(RESULTSPATH)
    combined_root = Path(data_path) / "preprocess" / "combined"

    if not combined_root.exists():
        raise FileNotFoundError(f"'combined' not found at: {combined_root}")

    counter = 0
    for csv_path, fname, subfolder, data, strain, segment, intersects in iter_csv_files(str(combined_root)):
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
        make_training_df(df, fname, subfolder, data, strain, segment, intersects)
        counter += 1

    if counter == 0:
        print(f"no CSV files found under: {combined_root}")
    else:
        print(f"prepared {counter} CSV files")

###############
### general ###
###############

### input ###

def make_training_df(df: pd.DataFrame, fname: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''
    
    '''
    if 'dvg_sequence' not in df.columns:
        df = add_dvg_sequence(df)

    df = add_marked_dvg_sequence(df)
    df = add_region_lengths(df)
    df = add_dvg_length(df)
    df = add_cg_content(df)

    df = add_marked_structure(df)
    df = add_loop_count(df, 'structure', 'loop_count')
    df = add_max_loop(df, 'structure', 'max_loop')

    columns_to_drop = [
    'dvg_sequence',
    'NGS_read_count',
    'log_NGS_read_count',
    'NGS_percentile_rank',
    'key',
    'isize',
    'full_seq',
    'deleted_sequence',
    'seq_around_deletion_junction',
    'structure'
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    save_df(df, fname, RESULTSPATH, 'training', subfolder, data, strain, segment, intersects)

### others ###

def iter_csv_files(path: str):
    '''
    
    '''
    root = Path(path)
    for csv_path in root.rglob("*.csv"):
        try:
            parts = csv_path.relative_to(root).parts
        except ValueError:
            continue

        if len(parts) < 6:
            continue

        subfolder, data, strain, segment, intersects = parts[0], parts[1], parts[2], parts[3], parts[4]
        fname = Path(parts[-1]).stem

        yield csv_path, fname, subfolder, data, strain, segment, intersects

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

    # folder = 'combined'
    # subfolder = 'pseudo_intersects_50'

    # data = 'IAV'
    # strain = 'PR8'
    # segment = 'PB1'
    # intersects = 'median'

    # motif_length = 3

    # dfname = f'motif_length_{motif_length}'
    # dfnames = [dfname]
    # dfs = load_all_preprocessed(dfnames, folder, subfolder, data, strain, segment, intersects)

    ###################
    ### run scripts ###
    ###################

    ### CNN ###

    make_training_dfs()

