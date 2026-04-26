'''
    preprocess datasets for further analysis and CNNs
'''

import os
import sys

import pandas as pd

sys.path.insert(0, '..')

from utils import load_all, get_dataset_names

from utils import load_all_unpooled, load_all_preprocessed,  save_df
from utils import add_metadata_features, add_metadata_features_pseudo, add_ikey
from utils import rename_feature
from utils import add_site_motifs


from utils import (
    add_delvg_sequence,
    add_deletion_length,
    add_region_lengths,
    add_delvg_length,
    add_full_seq_length,
    add_marked_delvg_sequence,
)

from utils import (
    add_direct_repeat_length,
    cap_direct_repeat_length,
)

from utils import (
    add_sec_features,
    add_marked_secondary,
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
    add_kmer_richness,
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

from utils import RESULTSPATH, DATASET_CUTOFF
from utils import K_MER_LENGTH

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'preprocess')


###############
### scripts ###
###############

### primary features ###

def preprocess_primary_features(dfnames: list, dfs: list, folder: str):
    '''
        Preprocess and save primary features for a list of datasets.
        :param dfnames: list of dataset names
        :param dfs: list of input dataframes
        :param folder: preprocessing folder or mode

        :return: None
    '''
    # initialize list for updated dataframes
    updated_dfs = []

    # add metadata differently depending on whether pseudo data is processed
    if folder == 'pseudo':
        dfs = add_metadata_features_pseudo(dfs)
        updated_dfs = dfs
    else:
        dfs = add_metadata_features(dfnames, dfs)
        for df in dfs:
            # normalize key column names before primary feature computation
            df = rename_feature(df, 'Segment', 'segment')
            df = rename_feature(df, 'Strain', 'strain')
            df = rename_feature(df, 'Start', 'start')
            df = rename_feature(df, 'End', 'end')
            updated_dfs.append(df)

    # compute and save primary features for each dataset
    for dfname, df in zip(dfnames, updated_dfs):
        save_primary_features(df, dfname, folder, 'primary')

### secondary features ###

def preprocess_secondary_features(dfnames: list, dfs: list, folder: str):
    '''
        Preprocess and save secondary-structure features for a list of datasets.
        :param dfnames: list of dataset names
        :param dfs: list of input dataframes
        :param folder: preprocessing folder or mode

        :return: None
    '''
    # compute and save secondary features for each dataset
    for dfname, df in zip(dfnames, dfs):
        save_secondary_features(df, dfname, folder, 'secondary')

### motif features ###

def preprocess_motif_features(dfnames: list, dfs: list, folder: str):
    '''
        Preprocess and save motif features for a list of datasets.
        :param dfnames: list of dataset names
        :param dfs: list of input dataframes
        :param folder: preprocessing folder or mode

        :return: None
    '''
    # compute and save motif features for each dataset
    for dfname, df in zip(dfnames, dfs):
        save_motif_features(df, dfname, folder, f'motif_length_{K_MER_LENGTH}')

### modify features ###

def preprocess_modify_features(folder: str, subfolder: str):
    '''
        Load preprocessed datasets, apply feature-name modifications, and save the updated versions.
        :param folder: preprocessing folder or mode
        :param subfolder: subfolder containing the preprocessed datasets

        :return: None
    '''
    # load all preprocessed datasets for the selected folder and subfolder
    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    # rename selected feature columns and save updated datasets
    for dfname, df in zip(dfnames, dfs):
        df = rename_feature(df, 'dataset_name', 'dataset')
        save_df(df, dfname, RESULTSPATH, folder, subfolder)
        print(f'{dfname} done')

###############
### general ###
###############

### primary features ###

def save_primary_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''
        Compute primary (sequence-derived) features and save the resulting dataframe.
        :param df: input dataframe
        :param fname: output file name
        :param folder: main save folder
        :param subfolder: save subfolder

        :return: None
    '''
    # add primary sequence-derived features to the dataframe
    df = add_primary_features(df)

    # save the updated dataframe to disk
    save_df(df, fname, RESULTSPATH, folder, subfolder)

    # print completion message
    print(f'{fname} done')

def add_primary_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Add primary sequence-derived features to a dataframe based on unique candidate identities.
        :param df: input dataframe

        :return: dataframe extended by primary feature columns
    '''
    # work on a copy to avoid modifying the input dataframe in place
    df = df.copy()

    # create intersection key and reduce to unique candidates
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    ### length ###
    # compute sequence length-related features
    uniq = add_delvg_sequence(uniq)
    uniq = add_deletion_length(uniq)
    uniq = add_region_lengths(uniq)
    uniq = add_delvg_length(uniq)
    uniq = add_full_seq_length(uniq)
    uniq = add_marked_delvg_sequence(uniq)

    ### direct repeats ###
    # compute direct repeat features and apply capping
    uniq = add_direct_repeat_length(uniq)
    uniq = cap_direct_repeat_length(uniq)

    ### pri / sequence-derived ###
    # compute nucleotide composition and sequence-derived features
    uniq = add_gc_content(uniq)
    uniq = add_au_content(uniq)
    uniq = add_upa_content(uniq)
    uniq = add_cpg_content(uniq)
    uniq = add_gc_skew(uniq)
    uniq = add_sequence_entropy(uniq)
    uniq = add_poly_run_features(uniq, 'A')
    uniq = add_poly_run_features(uniq, 'U')
    uniq = add_palindrome_density(uniq)
    uniq = add_kmer_richness(uniq)

    # identify newly created feature columns
    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq.columns if c not in primary_cols]

    # merge computed features back into the original dataframe
    out = df.merge(
        uniq[["ikey"] + feature_cols],
        on="ikey",
        how="left"
    )

    # remove helper key column after merge
    out = out.drop(columns=["ikey"])
    return out

### secondary features ###

def save_secondary_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''
        Compute secondary-structure and hybrid features and save the resulting dataframe.
        :param df: input dataframe
        :param fname: output file name
        :param folder: main save folder
        :param subfolder: save subfolder

        :return: None
    '''
    # add secondary-structure and hybrid features to the dataframe
    df = add_secondary_features(df, folder)

    # save the updated dataframe to disk
    save_df(df, fname, RESULTSPATH, folder, subfolder)

    # print completion message
    print(f'{fname} done')

def add_secondary_features(df: pd.DataFrame, folder: str) -> pd.DataFrame:
    '''
        Add secondary-structure and hybrid features to a dataframe based on unique candidate identities.
        :param df: input dataframe
        :param folder: preprocessing folder or grouping label

        :return: dataframe extended by secondary-structure and hybrid feature columns
    '''
    # work on a copy to avoid modifying the input dataframe in place
    df = df.copy()

    # create intersection key and reduce to unique candidates within this dataframe
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    # compute features only once per unique candidate
    if uniq is not None and not uniq.empty:
        uniq = add_sec_features(uniq, "DelVG_sequence", "structure", "MFE")
        uniq = add_marked_secondary(uniq)

        ### symmetry
        # add symmetry-based secondary feature
        uniq = add_max_symmetry(uniq)

        ### sec ###
        # add secondary-structure count, density, and size features
        uniq = add_bp_count(uniq)
        uniq = add_bp_density(uniq)
        uniq = add_unpaired_count(uniq)
        uniq = add_unpaired_density(uniq)
        uniq = add_stem_count(uniq)
        uniq = add_stem_length_stats(uniq)
        uniq = add_hairpin_count(uniq)
        uniq = add_hairpin_size_stats(uniq)
        uniq = add_external_unpaired_density(uniq)
        uniq = add_pair_span_stats(uniq)
        uniq = add_free_end_lengths(uniq)

        ### hybrid ###
        # add hybrid sequence-structure features
        uniq = add_gc_overall(uniq)
        uniq = add_gc_paired_unpaired(uniq)
        uniq = add_canonical_pair_stats(uniq)
        uniq = add_stem_end_pair_enrichment(uniq)
        uniq = add_hairpin_closing_pair_content(uniq)
        uniq = add_tetraloop_motif_counts(uniq)
        uniq = add_loop_au_content(uniq)
        uniq = add_tail_gc_content(uniq)
        uniq = add_start_codon_accessibility(uniq)

    # identify newly created feature columns
    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq.columns if c not in primary_cols and c != "ikey"]

    # merge computed features back into the original dataframe
    out = df.merge(
        uniq[["ikey"] + feature_cols],
        on="ikey",
        how="left",
    )

    # remove helper key column after merge
    out = out.drop(columns=["ikey"])
    return out

### motif features ###

def save_motif_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''
        Compute motif-derived features and save the resulting dataframe.
        :param df: input dataframe
        :param fname: output file name
        :param folder: main save folder
        :param subfolder: save subfolder

        :return: None
    '''
    # add motif-derived feature columns to the dataframe
    df = add_motif_features(df)

    # save the updated dataframe to disk
    save_df(df, fname, RESULTSPATH, folder, subfolder)

    # print completion message
    print(f'{fname} done')

def add_motif_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Add junction-site motif features to a dataframe based on unique candidate identities.
        :param df: input dataframe

        :return: dataframe extended by motif-derived feature columns
    '''
    # work on a copy to avoid modifying the input dataframe in place
    df = df.copy()

    # add intersection key used to identify unique candidates
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    ### motifs ###
    # compute motif features only once per unique candidate
    uniq, skipped_count = add_site_motifs(uniq, K_MER_LENGTH)
    print(f'{skipped_count} candidates have been left out')

    # identify newly created motif feature columns
    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq.columns if c not in primary_cols]

    # keep only keys that survived motif extraction and merge motif features back
    out = df.merge(
        uniq[["ikey"] + feature_cols],
        on="ikey",
        how="inner"
    )

    # remove helper key column after merge
    out = out.drop(columns=["ikey"])
    return out


if __name__ == '__main__':
    '''

    '''
    #################
    ### SELECTION ###
    #################

    ### BASE ###

    ### make all base datasets first ###

    ### "pooled" or "unpooled" ###
    folder = 'pooled'

    dfnames = get_dataset_names(DATASET_CUTOFF)

    ### if folder 'pooled' use load_all, if 'unpooled' use load_all_unpooled ###
    dfs, _ = load_all(dfnames, False)
    dfs, _ = load_all_unpooled(dfnames, False)

    ### select dataset index ###
    index = 2
    dfnames = [dfnames[index]]
    dfs = [dfs[index]]

    ### DATA ###

    ### comput features for datasets ###
    ### create primary features on base datasets, then secondary on primary, and after motifs on secondary ###

    ### "pooled" or "unpooled" ###
    folder = 'pooled'

    ### "primary" or "secondary" ###
    subfolder = 'primary'

    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### primary ###

    preprocess_primary_features(dfnames, dfs, folder)

    ### secondary ###

    preprocess_secondary_features(dfnames, dfs, folder)

    ### motif ###

    preprocess_motif_features(dfnames, dfs, folder)

    ### modify ###

    preprocess_modify_features(folder, subfolder)