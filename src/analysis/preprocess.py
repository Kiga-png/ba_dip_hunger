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
from utils import add_metadata_features, add_ikey
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
    dataset_name, system_type, library_layout,
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
    dfs = add_metadata_features(dfnames, dfs)

    for dfname, df in zip(dfnames, dfs):
        df = rename_feature(df, "Start", "start")
        df = rename_feature(df, "End", "end")
        df = rename_feature(df, "Strain", "strain")
        df = rename_feature(df, "Segment", "segment")
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
        save_motif_features(df, dfname, folder, 'motif')

### modify features ###

def preprocess_modify_features(folder: str, subfolder: str):
    '''

    '''
    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)
    for dfname, df in zip(dfnames, dfs):
        df = rename_feature(df, 'marked_dvg_sequence', 'marked_DelVG_sequence')
        save_df(df, dfname, RESULTSPATH, folder, subfolder)

### pseudo candidates ###

def preprocess_pseudo_features(dfname: str, folder: str, data: str = 'all', strain: str = 'all', segment: str = 'all', intersects: str = 'all'):
    '''

    '''
    candidates = PSEUDO_CANDIDATES_DICT[dfname]
    df = build_pseudo_df(candidates)

    if folder == "pooled":

        cols_to_drop = [
        "AN", "Time", "Localization", "Resolution", "Cells", "MOI", "Host"
        ]

        df = df.drop(columns=cols_to_drop)

    save_pseudo_features(df, dfname, folder, 'pseudo', data, strain, segment, intersects)

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
    Add secondary-structure + hybrid features, with optional reuse from previously
    computed CSVs under os.path.join(RESULTSPATH, folder).

    If cached CSV rows with matching 'ikey' exist, this function copies the cached
    'structure' column and all columns to its right (in that cached CSV) into the
    corresponding rows, skipping recomputation for those rows.

    If no cached CSVs exist / are empty / lack required columns, it proceeds as before.
    """
    df = df.copy()
    df = add_ikey(df)
    uniq = df.drop_duplicates("ikey").copy().reset_index(drop=True)

    # 1) Build cache from existing CSVs (optional)
    cache_dir = os.path.join(RESULTSPATH, folder)
    cache_df = None

    if os.path.isdir(cache_dir):
        csv_paths = sorted(glob.glob(os.path.join(cache_dir, "*.csv")))
        cached_chunks = []

        for p in csv_paths:
            try:
                tmp = pd.read_csv(p)
            except Exception:
                continue

            if tmp is None or tmp.empty:
                continue
            if "ikey" not in tmp.columns or "structure" not in tmp.columns:
                continue

            # Take 'structure' and all columns that follow it (as requested)
            cols = list(tmp.columns)
            try:
                start_idx = cols.index("structure")
            except ValueError:
                continue

            feature_cols = cols[start_idx:]  # includes "structure"
            take_cols = ["ikey"] + feature_cols
            take_cols = [c for c in take_cols if c in tmp.columns]

            chunk = tmp[take_cols].copy()
            # Drop rows without ikey to keep merges clean
            chunk = chunk.dropna(subset=["ikey"])
            cached_chunks.append(chunk)

        if cached_chunks:
            cache_df = pd.concat(cached_chunks, ignore_index=True)
            # Keep first occurrence per ikey (any hit is good enough)
            cache_df = cache_df.drop_duplicates(subset=["ikey"], keep="first").reset_index(drop=True)

    # 2) Split into cache hits vs misses
    if cache_df is not None and not cache_df.empty:
        cached_ikeys = set(cache_df["ikey"].astype(str))
        uniq_ikeys = uniq["ikey"].astype(str)

        hit_mask = uniq_ikeys.isin(cached_ikeys)
        uniq_hits = uniq.loc[hit_mask].copy()
        uniq_miss = uniq.loc[~hit_mask].copy()

        # Attach cached columns to hits
        uniq_hits = uniq_hits.merge(
            cache_df,
            on="ikey",
            how="left",
        )
    else:
        uniq_hits = None
        uniq_miss = uniq

    # 3) Compute features only for cache misses (original pipeline)
    if uniq_miss is not None and not uniq_miss.empty:
        ### secondary ###
        uniq_miss = add_sec_features(uniq_miss, 'DelVG_sequence', 'sructure', 'MFE')
        uniq_miss = add_marked_secondary(uniq_miss)

        ### symmetry
        uniq_miss = add_max_symmetry(uniq_miss)

        ### sec ###
        uniq_miss = add_bp_count(uniq_miss)
        uniq_miss = add_bp_density(uniq_miss)
        uniq_miss = add_unpaired_count(uniq_miss)
        uniq_miss = add_unpaired_density(uniq_miss)
        uniq_miss = add_stem_count(uniq_miss)
        uniq_miss = add_stem_length_stats(uniq_miss)
        uniq_miss = add_hairpin_count(uniq_miss)
        uniq_miss = add_hairpin_size_stats(uniq_miss)
        uniq_miss = add_external_unpaired_density(uniq_miss)
        uniq_miss = add_pair_span_stats(uniq_miss)
        uniq_miss = add_free_end_lengths(uniq_miss)
        uniq_miss = add_branch_point_count(uniq_miss)

        ### hybrid ###
        uniq_miss = add_gc_overall(uniq_miss)
        uniq_miss = add_gc_paired_unpaired(uniq_miss)
        uniq_miss = add_canonical_pair_stats(uniq_miss)
        uniq_miss = add_stem_end_pair_enrichment(uniq_miss)
        uniq_miss = add_hairpin_closing_pair_content(uniq_miss)
        uniq_miss = add_tetraloop_motif_counts(uniq_miss)
        uniq_miss = add_loop_au_content(uniq_miss)
        uniq_miss = add_tail_gc_content(uniq_miss)
        uniq_miss = add_start_codon_accessibility(uniq_miss)

    # 4) Recombine uniq (hits + misses)
    if uniq_hits is not None and not uniq_hits.empty:
        uniq_final = pd.concat([uniq_hits, uniq_miss], ignore_index=True)
        # ensure one row per ikey
        uniq_final = uniq_final.drop_duplicates(subset=["ikey"], keep="first").reset_index(drop=True)
    else:
        uniq_final = uniq_miss

    # 5) Merge features back into the original df (same as before)
    primary_cols = set(df.columns)
    feature_cols = [c for c in uniq_final.columns if c not in primary_cols]

    out = df.merge(
        uniq_final[["ikey"] + feature_cols],
        on="ikey",
        how="left"
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

### pseudo candidates ###

def save_pseudo_features(df: pd.DataFrame, fname: str, folder: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''

    '''
    ### length ###
    df = add_delvg_sequence(df)
    df = add_deletion_length(df)
    df = add_region_lengths(df)
    df = add_delvg_length(df)
    df = add_full_seq_length(df)
    df = add_marked_delvg_sequence(df)

    ### direct repeats ###
    df = add_direct_repeat_length(df)
    df = cap_direct_repeat_length(df, DIRECT_REPEAT_LENGTH_CAP)

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

    ### secondary ###
    df = add_sec_features(df, 'DelVG_sequence', 'secondary', 'MFE')
    df = add_marked_secondary(df)

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

    save_df(df, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)       

if __name__ == '__main__':
    '''

    '''
    #################
    ### SELECTION ###
    #################

    ### BASE ###

    # folder = 'unpooled'

    # dfnames = get_dataset_names(DATASET_CUTOFF)

    # if folder 'pooled' use load_all, if 'unpooled' use load_all_unpooled
    # dfs, _ = load_all(dfnames, False)
    # dfs, _ = load_all_unpooled(dfnames, False)

    # index = 2
    # dfnames = [dfnames[index]]
    # dfs = [dfs[index]]

    ### DATASETS SINGLE ###

    folder = 'unpooled'
    subfolder = 'primary'

    dfnames = get_dataset_names(DATASET_CUTOFF)

    # index = 2
    # dfnames = [dfnames[index]]
    dfs = load_all_preprocessed(dfnames, folder, subfolder)
    
    ### DATASETS MULTI ###

    # folder = 'unpooled'
    # subfolder = 'primary'

    # data = 'IAV'
    # strain = 'PR8'
    # segment = 'PB1'
    # intersects = 'median'

    # dfnames = get_dataset_names(DATASET_CUTOFF, selection)
    # dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    ### primary ###

    # preprocess_primary_features(dfnames, dfs, folder)

    ### secondary ###

    preprocess_secondary_features(dfnames, dfs, folder)

    ### motif ###



    ### modify ###

    # preprocess_modify_features(folder, subfolder)

    ### pseudo candidates ###