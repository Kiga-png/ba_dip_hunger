'''

'''
import os
import sys

import pandas as pd

sys.path.insert(0, "..")

from pathlib import Path
from typing import Iterator, Tuple

from utils import get_dataset_names, load_all
from utils import build_df, load_all_preprocessed, save_df

from utils import add_dvg_sequence, add_marked_dvg_sequence, add_dvg_length, add_deletion_length, add_full_seq_length, add_region_lengths
from utils import add_direct_repeat_length
from utils import add_max_symmetry
from utils import add_marked_structure

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
        make_training_df(df, fname, 'training', subfolder, data, strain, segment, intersects)
        counter += 1

    if counter == 0:
        print(f"no CSV files found under: {combined_root}")
    else:
        print(f"prepared {counter} CSV files")

###############
### general ###
###############

### input ###

def make_training_df(df: pd.DataFrame, fname: str, folder: str, subfolder: str, data: str, strain: str, segment: str, intersects: str):
    '''
    
    '''
    if 'dvg_sequence' not in df.columns:
        df = add_dvg_sequence(df)

    df = add_marked_dvg_sequence(df)
    df = add_marked_structure(df)

    palindrmic_k_mer = 6
    k_mer = 3

    df = add_dvg_sequence(df)
    df = add_dvg_length(df)
    df = add_deletion_length(df)
    df = add_full_seq_length(df)
    df = add_region_lengths(df)

    df = add_gc_content(df)
    df = add_au_content(df)
    df = add_upa_content(df)
    df = add_cpg_content(df)
    df = add_gc_skew(df)
    df = add_sequence_entropy(df)
    df = add_poly_run_features(df, 'A')
    df = add_poly_run_features(df, 'U')
    df = add_palindrome_density(df, palindrmic_k_mer)
    df = add_orf_features(df)
    df = add_kmer_richness(df, k_mer)
    df = add_codon_usage_bias(df)

    df = add_direct_repeat_length(df)

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

    df = add_max_symmetry(df)

    df = add_gc_overall(df)
    df = add_gc_paired_unpaired(df)
    df = add_canonical_pair_stats(df)
    df = add_stem_end_pair_enrichment(df)
    df = add_hairpin_closing_pair_content(df)
    df = add_tetraloop_motif_counts(df)
    df = add_loop_au_content(df)
    df = add_tail_gc_content(df)
    df = add_start_codon_accessibility(df)

    print(f'features added successfully')

    columns_to_drop = [
    'NGS_read_count',
    'log_NGS_read_count',
    'NGS_percentile_rank',
    'key',
    'isize',
    'full_seq',
    'deleted_sequence',
    'seq_around_deletion_junction',
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    save_df(df, fname, RESULTSPATH, folder, subfolder, data, strain, segment, intersects)

### others ###

def iter_csv_files(path: str) -> Iterator[Tuple[Path, str, str, str, str, str, str]]:
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

        subfolder = parts[0]
        if subfolder == "prediction":
            continue

        data, strain, segment, intersects = parts[1], parts[2], parts[3], parts[4]
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

