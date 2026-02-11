'''

'''
import os
import glob
import sys

import numpy as np
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

from utils import DATAPATH, RESULTSPATH, DATASET_CUTOFF, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS
from utils import K_MER_LENGTH, PALINDROMIC_K_MER_LENGTH, DIRECT_REPEAT_LENGTH_CAP
from utils import STRAIN_SUBTYPE_DICT

PSEUDO_CANDIDATES_DICT = {
"""
order of features:
    seg, start, end, NGS_read_count,
    AN, Time, Localization, Resolution, Cells, MOI, Host,
    key, strain,
    dataset, system_type, library_layout,
    library_selection, library_source, subtype
    + (full_seq_override) if needed
"""

    "PR8_PB1_seq": [
        ("PB2", 164, 2110, 0,
         "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
         "PR8",
         "164_2110", "unknown", "unknown", "unknown", "unknown", "unknown"),

        ("PB2", 164, 2110, 0,
         "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
         "PR8",
         "164_2110", "unknown", "unknown", "unknown", "unknown", "unknown"),

        ("PB2", 164, 2110, 0,
         "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
         "PR8",
         "164_2110", "unknown", "unknown", "unknown", "unknown", "unknown"),

        ("PB2", 164, 2110, 0,
         "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
         "PR8",
         "164_2110", "unknown", "unknown", "unknown", "unknown", "unknown"),

        ("PB2", 164, 2110, 0,
         "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
         "PR8",
         "164_2110", "unknown", "unknown", "unknown", "unknown", "unknown"),
    ],

    "PR8_PB1_mono": [
        ("PB2", 164, 2110, 0,
         "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown",
         "PR8",
         "A_164_2110", "unknown", "unknown", "unknown", "unknown", "unknown",
         "A" * 2341),

    ],
}

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'preprocess')

###############
### scripts ###
###############

### primary features ###

def preprocess_primary_features(dfnames: list, dfs: list, folder: str):
    '''

    '''
    updated_dfs = []
    if folder == 'pseudo':
        dfs = add_metadata_features_pseudo(dfs)
        updated_dfs = dfs
    else:
        dfs = add_metadata_features(dfnames, dfs)
        for df in dfs:
            df = rename_feature(df, 'Segment', 'segment')
            df = rename_feature(df, 'Strain', 'strain')
            df = rename_feature(df, 'Start', 'start')
            df = rename_feature(df, 'End', 'end')
            updated_dfs.append(df)

    for dfname, df in zip(dfnames, updated_dfs):
        save_primary_features(df, dfname, folder, 'primary')

### secondary features ###

def preprocess_secondary_features(dfnames: list, dfs: list, folder: str):
    '''

    '''
    for dfname, df in zip(dfnames, dfs):
        save_secondary_features(df, dfname, folder, 'secondary')

### motif features ###

def preprocess_motif_features(dfnames: list, dfs: list, folder: str):
    '''

    '''

    for dfname, df in zip(dfnames, dfs):
        save_motif_features(df, dfname, folder, f'motif_length_{K_MER_LENGTH}')

### modify features ###

def preprocess_modify_features(folder: str, subfolder: str):
    '''

    '''
    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)
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

    '''
    df = add_primary_features(df)
    save_df(df, fname, RESULTSPATH, folder, subfolder)
    print(f'{fname} done')

def add_primary_features(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    df = df.copy()
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    ### length ###
    uniq = add_delvg_sequence(uniq)
    uniq = add_deletion_length(uniq)
    uniq = add_region_lengths(uniq)
    uniq = add_delvg_length(uniq)
    uniq = add_full_seq_length(uniq)
    uniq = add_marked_delvg_sequence(uniq)

    ### direct repeats ###
    uniq = add_direct_repeat_length(uniq)
    uniq = cap_direct_repeat_length(uniq)

    ### pri / sequence-derived ###
    uniq = add_gc_content(uniq)
    uniq = add_au_content(uniq)
    uniq = add_upa_content(uniq)
    uniq = add_cpg_content(uniq)
    uniq = add_gc_skew(uniq)
    uniq = add_sequence_entropy(uniq)
    uniq = add_poly_run_features(uniq, 'A')
    uniq = add_poly_run_features(uniq, 'U')
    uniq = add_palindrome_density(uniq)
    uniq = add_orf_features(uniq)
    uniq = add_kmer_richness(uniq)
    uniq = add_codon_usage_bias(uniq)

    # merge back only the new feature columns
    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq.columns if c not in primary_cols]

    out = df.merge(
        uniq[["ikey"] + feature_cols],
        on="ikey",
        how="left"
    )

    # drop key here if you don't want it downstream
    out = out.drop(columns=["ikey"])

    return out

### secondary features ###

def save_secondary_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    df = add_secondary_features(df, folder)
    save_df(df, fname, RESULTSPATH, folder, subfolder)
    print(f'{fname} done')

def add_secondary_features(df: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Add secondary-structure + hybrid features, computed once per unique 'ikey'
    within the CURRENT df only.

    Changes vs. previous version:
    - No disk/cache reuse (no reading CSVs from RESULTSPATH/folder).
    - Only avoids recomputation for duplicate candidates inside the same df
      by computing features on df.drop_duplicates("ikey") and merging back.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that contains the sequence column used by add_sec_features,
        here assumed to be 'DelVG_sequence' (as in your pipeline).
    folder : str
        Kept for API compatibility (unused now).

    Returns
    -------
    pd.DataFrame
        df with computed secondary + hybrid features merged in.
    """
    df = df.copy()

    # 1) Create ikey and reduce to unique candidates within THIS df
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    # 2) Compute features ONCE per unique ikey (original pipeline)
    if uniq is not None and not uniq.empty:
        ### secondary ###
        # IMPORTANT: fix typo 'sructure' -> 'structure'
        uniq = add_sec_features(uniq, "DelVG_sequence", "structure", "MFE")
        uniq = add_marked_secondary(uniq)

        ### symmetry
        uniq = add_max_symmetry(uniq)

        ### sec ###
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
        uniq = add_branch_point_count(uniq)

        ### hybrid ###
        uniq = add_gc_overall(uniq)
        uniq = add_gc_paired_unpaired(uniq)
        uniq = add_canonical_pair_stats(uniq)
        uniq = add_stem_end_pair_enrichment(uniq)
        uniq = add_hairpin_closing_pair_content(uniq)
        uniq = add_tetraloop_motif_counts(uniq)
        uniq = add_loop_au_content(uniq)
        uniq = add_tail_gc_content(uniq)
        uniq = add_start_codon_accessibility(uniq)

    # 3) merge computed features back into the original df
    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq.columns if c not in primary_cols and c != "ikey"]

    out = df.merge(
        uniq[["ikey"] + feature_cols],
        on="ikey",
        how="left",
    )

    out = out.drop(columns=["ikey"])
    return out

### motif features ###

def save_motif_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str):
    '''

    '''
    df = add_motif_features(df)
    save_df(df, fname, RESULTSPATH, folder, subfolder)
    print(f'{fname} done')

def add_motif_features(df: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    df = df.copy()
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    ### motifs ###
    uniq, skipped_count = add_site_motifs(uniq, K_MER_LENGTH)
    print(f'{skipped_count} candidates have been left out')

    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq.columns if c not in primary_cols]

    # only keep keys that survived add_site_motifs (rows can be dropped)
    out = df.merge(
        uniq[["ikey"] + feature_cols],
        on="ikey",
        how="inner"
    )

    out = out.drop(columns=["ikey"])

    return out


if __name__ == '__main__':
    '''

    '''
    #################
    ### SELECTION ###
    #################

    ### BASE ###

    # folder = 'pooled'

    # dfnames = get_dataset_names(DATASET_CUTOFF)

    # if folder 'pooled' use load_all, if 'unpooled' use load_all_unpooled
    # dfs, _ = load_all(dfnames, False)
    # dfs, _ = load_all_unpooled(dfnames, False)

    # index = 2
    # dfnames = [dfnames[index]]
    # dfs = [dfs[index]]

    ### DATASETS SINGLE/MULTI/PSEUDO ###

    folder = 'pseudo'
    subfolder = 'primary'

    # dfnames = get_dataset_names(DATASET_CUTOFF)
    dfnames = ['mono', 'motif', 'sequence']

    # index = 0
    # dfnames = [dfnames[index]]

    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### primary ###

    # preprocess_primary_features(dfnames, dfs, folder)

    ### secondary ###

    preprocess_secondary_features(dfnames, dfs, folder)

    ### motif ###

    # preprocess_motif_features(dfnames, dfs, folder)

    ### modify ###

    # preprocess_modify_features(folder, subfolder)