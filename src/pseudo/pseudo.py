'''
    generate pseudo DelVGs based on selected coordinates and sequences from given experiments
'''
import os
import sys

import numpy as np
import pandas as pd

from scipy import stats
from typing import Tuple

sys.path.insert(0, "..")
from utils import load_dataset, get_dataset_names, join_data, preprocess, generate_expected_data, get_seq_len
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF

RESULTSPATH, _ = os.path.split(RESULTSPATH)

from investigate import resave


def pseudo_load_all(dfnames: list, pseudo_dfname: str, expected: str=False)-> Tuple[list, list, list]:
    '''
        Load a list of datasets with sequnces based on the strain of the selected experiment.
        :param dfnames: list of dataset names, each is one experiment
        :param pseudo_dfname: selected experiment to load sequences
        :param expected: if True, expected data is loaded additionally

        :return: Tuple
            List of Pandas DataFrames each containing one experiment
            List of Pandas DataFrames each containing one pseudo_experiment
            List of expected Pandas DataFrames each containing one experiment
    '''
    dfs = list()
    pseudo_dfs = list()
    expected_dfs = list()
    for dfname in dfnames:
        co_strain = DATASET_STRAIN_DICT[dfname]
        pseudo_strain = DATASET_STRAIN_DICT[pseudo_dfname]
        # og seq
        df = join_data(load_dataset(pseudo_dfname))
        dfs.append(preprocess(pseudo_strain, df, CUTOFF))
        # pseudo seq
        pseudo_df = join_data(load_dataset(dfname))
        pseudo_df = recenter(pseudo_df, co_strain, pseudo_strain)
        pseudo_dfs.append(preprocess(pseudo_strain, pseudo_df, CUTOFF))
        # og exp
        if expected:
            f = os.path.join(DATAPATH, "random_sampled", f"{dfname}_{CUTOFF}.csv")
            if os.path.exists(f):
                dtypes = {"Start": int, "End": int, "Segment": str, "NGS_read_count": int,
                          "key": str, "Strain": str, "isize": int, "full_seq": str,
                          "deleted_sequence": str, "seq_around_deletion_junction": str}
                exp_df = pd.read_csv(f, dtype=dtypes)
            else:
                df = df[df["NGS_read_count"] >= CUTOFF].copy()
                exp_df = preprocess(co_strain, generate_expected_data(co_strain, df), 1)
                exp_df.to_csv(f, index=False)
            expected_dfs.append(exp_df)
    return dfs, pseudo_dfs, expected_dfs

def recenter(df: pd.DataFrame, co_strain: str, pseudo_strain: str)-> pd.DataFrame:
    '''
        Recenter start and end positions based on the selected strain.
        :param df: Pandas DataFrame containing original coordinates of the experiment
        :param co_strain: name of original strain
        :param pseudo_strain: name of selected strain to readjust coordinates

        :retrun: Pandas DataFrame with recentered coordinates
    '''
    def update_positions(row):
        co_len = get_seq_len(co_strain, row["Segment"])
        pseudo_len = get_seq_len(pseudo_strain, row["Segment"])
        new_start, new_end = calculate_relative_positions(co_len, pseudo_len, row["Start"], row["End"])
        return pd.Series({"Start": new_start, "End": new_end})

    df[["Start", "End"]] = df.apply(update_positions, axis=1)
    return df
    
def calculate_relative_positions(co_len, pseudo_len, start, end)-> Tuple[int, int]:
    '''
        Calculates relative positions od start and end based on the slected strains length.
        :param co_len: length of the original sequence
        :param pseudo_len: length of the selected sequence
        :param start: deletion site start position of original sequence
        :param end: deletion site end position of original sequence

        :retrun Tuple
            relative deletion site start position
            relative deletion site end position
    '''
    relative_position_start = start / co_len
    relative_position_end = end / co_len
    new_start = int(relative_position_start * pseudo_len)
    new_end = int(relative_position_end * pseudo_len)
    return new_start, new_end


if __name__ == "__main__":
    '''
        IAV: ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Kupke2020", "Zhuravlev2020", "VdHoecke2015", "Alnaji2019_Cal07", "Alnaji2019_NC", "Mendes2021", "Boussier2020", "Alnaji2019_Perth", "Berry2021_A", "Penn2022", "Lui2019"]
        IBV: ["Alnaji2019_BLEE", "Berry2021_B", "Valesano2020_Vic", "Sheng2018", "Berry2021_B_Yam", "Southgate2019", "Valesano2020_Yam"]
    '''

    ##############################
    ######## SINGLE USAGE ########
    ##############################

    pseudo_dfname = "Alnaji2021"
    dfnames = ["Berry2021_B_Yam"]
    dfs, pseudo_dfs, exp_dfs = pseudo_load_all(dfnames, pseudo_dfname, True)

    pseudo_df = pseudo_dfs[0]
    exp_df = exp_dfs[0]

    pseudo_strain = DATASET_STRAIN_DICT[pseudo_dfname]
    new_name = f"st_{pseudo_strain}_co_{dfnames[0]}"
    resave([new_name], [pseudo_df], "pseudo")    

    ##############################
    ######## MULTI USAGE #########
    ##############################

    pseudo_dfname = "Alnaji2021"
    coordinates = "IBV"
    dfnames = get_dataset_names(cutoff=40, selection=coordinates)
    dfs, pseudo_dfs, exp_dfs = pseudo_load_all(dfnames, pseudo_dfname, True)

    concat_pseudo_df = pd.concat(pseudo_dfs, ignore_index=True)
    concat_exp_df = pd.concat(exp_dfs, ignore_index=True)

    pseudo_strain = DATASET_STRAIN_DICT[pseudo_dfname]
    new_name = f"st_{pseudo_strain}_co_{coordinates}"
    resave([new_name], [concat_pseudo_df], "pseudo")