'''

'''
import os
import sys

import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Optional, Union, Tuple

sys.path.insert(0, "..")

from utils import get_dataset_names

from utils import manage_separate_specifiers, add_intersect_ngs_features, add_metadata_features
from utils import get_sequence
from utils import load_all_preprocessed, save_df

from utils import DATAPATH, RESULTSPATH, DATASET_CUTOFF, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS, SEED
from utils import K_MER_LENGTH

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'statistics')


###############
### scripts ###
###############

### statistics ###

def run_strain_segment_statistics(
    dfs: list[pd.DataFrame],
    data: str = 'all',
    strain: str = 'all',
    segment: str = 'all',
    intersects: str = 'all'
) -> None:
    """
    Concat a list of dfs, keep only required columns, compute per-(Strain, Segment)
    summary stats, round all derived numbers to integers, and save via save_df.
    """

    keep_cols = [
        "segment",
        "start",
        "end",
        "NGS_read_count",
        "strain",
        "seq_around_deletion_junction",
        "dataset",
        "deletion_length",
        "5_end_length",
        "3_end_length",
        "DelVG_length",
        "full_seq_length",
        "site1_motif",
        "site2_motif",
        "site3_motif",
        "site4_motif",
    ]

    if dfs is None or len(dfs) == 0:
        raise ValueError("dfs is empty")

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)

    df = pd.concat(dfs, ignore_index=True)

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    df = df[keep_cols].copy()

    num_cols = [
        "start", "end", "NGS_read_count",
        "deletion_length", "5_end_length", "3_end_length",
        "DelVG_length", "full_seq_length"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _agg_stats(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan})

        return pd.Series(
            {
                "mean": int(np.round(x.mean())),
                "median": int(np.round(x.median())),
                "min": int(x.min()),
                "max": int(x.max()),
            }
        )

    def _most_prominent(x: pd.Series) -> str:
        """
        Return most frequent (mode) non-null entry.
        Tie-breaker: pick the first mode in sorted order for stability.
        """
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        vc = x.astype(str).value_counts()
        if vc.empty:
            return np.nan
        top_n = vc.iloc[0]
        top_vals = vc[vc == top_n].index.tolist()
        return sorted(top_vals)[0]

    gb = df.groupby(["dataset", "segment"], dropna=False)

    stats_df = gb.size().rename("count").to_frame()
    stats_df["count"] = stats_df["count"].astype(int)

    for feat in [
        "start",
        "end",
        "deletion_length",
        "5_end_length",
        "3_end_length",
        "DelVG_length",
        "NGS_read_count",
    ]:
        tmp = gb[feat].apply(_agg_stats).unstack()
        tmp.columns = [f"{feat}_{k}" for k in tmp.columns]
        stats_df = stats_df.join(tmp)

    # full_seq_length: one representative value per group (always identical)
    stats_df["full_seq_length"] = gb["full_seq_length"].first().round().astype("Int64")

    # most prominent motifs per group
    for motif_col in ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]:
        stats_df[motif_col] = gb[motif_col].apply(_most_prominent)

    stats_df = stats_df.reset_index()

    save_df(stats_df, "statistics", RESULTSPATH)

### pseudo ###

def run_pseudo_candidate_dfs():
    """
    
    """
    csv_path = os.path.join(RESULTSPATH, "statistics.csv")
    statistics_df = pd.read_csv(csv_path, keep_default_na=False)
    df_mono, df_motif, df_sequence = build_pseudo_candidate_dfs(statistics_df)

    save_path, _ = os.path.split(RESULTSPATH)
    save_path = os.path.join(save_path, 'preprocess')

    save_df(df_mono, 'mono', save_path, 'pseudo', 'base')
    save_df(df_motif, 'motif', save_path, 'pseudo', 'base')
    save_df(df_sequence, 'sequence', save_path, 'pseudo', 'base')

### size ###

def print_df_sizes(dfnames: list, dfs: list) -> int:
    """
    Print the number of rows for each DataFrame and the total number of rows.
    Then concatenate all DataFrames and report row counts grouped by:
      (1) strain x segment
      (2) strain (all segments combined)

    Parameters
    dfnames : list of str
        Readable names corresponding to the DataFrames.
    dfs : list of pd.DataFrame
        DataFrames to summarize.

    Returns
    int
        Total number of rows across all DataFrames.
    """
    if len(dfs) != len(dfnames):
        raise ValueError("dfs and dfnames must have the same length.")

    total_rows = 0

    print("Dataset sizes (rows):")
    for name, df in zip(dfnames, dfs):
        n = len(df)
        total_rows += n
        print(f"  {name}: {n:,}")

    print("-" * 40)
    print(f"Total rows: {total_rows:,}")

    if total_rows == 0:
        print("\nNo rows available for grouping (all DataFrames empty).")
        return total_rows

    # sanity check required columns
    required_cols = {"strain", "segment"}
    missing = required_cols - set(dfs[0].columns)
    # check across all dfs (more robust)
    for i, df in enumerate(dfs):
        miss_i = required_cols - set(df.columns)
        if miss_i:
            raise ValueError(f"Missing columns {miss_i} in dfs[{i}] needed for grouping.")

    df_all = pd.concat(dfs, ignore_index=True)

    # normalize types (avoid weird group keys)
    df_all["strain"] = df_all["strain"].astype(str)
    df_all["segment"] = df_all["segment"].astype(str)

    # (1) strain x segment
    comb_counts = (
        df_all
        .groupby(["strain", "segment"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["strain", "segment"])
    )

    print("\nCounts by strain × segment:")
    for _, r in comb_counts.iterrows():
        print(f"  {r['strain']} | {r['segment']}: {int(r['n_rows']):,}")

    # (2) strain totals (all segments summed)
    strain_counts = (
        df_all
        .groupby(["strain"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )

    print("\nCounts by strain (all segments combined):")
    for _, r in strain_counts.iterrows():
        print(f"  {r['strain']}: {int(r['n_rows']):,}")

    return total_rows

### percentage ###

def print_percentage_of_value(df: pd.DataFrame, column: str, value) -> float:
    """
    Returns the percentage of rows in `column` that equal `value`.
    """
    total_rows = len(df)
    if total_rows == 0:
        print(0)
    
    count = (df[column] == value).sum()
    percentage = (count / total_rows) * 100
    
    print(percentage)

###############
### general ###
###############

### pseudo ###

def build_pseudo_candidate_dfs(
    statistics_df: pd.DataFrame,
    n_per_category: int = 100,
    isize: int = 5,
    dirichlet_alpha: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build 3 pseudo-candidate dfs (mono, motif, sequence) from statistics_df.

    Adds column: 'composition'
      - mono     : A / C / G / U
      - motif    : motif-site label
      - sequence : deletion-length category
                   {micro, short, long, extensive deletion}
    """

    rng = np.random.default_rng(SEED)

    required_cols = [
        "dataset", "segment", "count",
        "start_min", "start_median", "start_max",
        "end_min", "end_median", "end_max",
        "full_seq_length",
        "site1_motif", "site2_motif", "site3_motif", "site4_motif",
    ]
    missing = [c for c in required_cols if c not in statistics_df.columns]
    if missing:
        raise KeyError(f"statistics_df missing required columns: {missing}")

    # helpers
    def _safe_int(x, default=None):
        try:
            if pd.isna(x):
                return default
            return int(round(float(x)))
        except Exception:
            return default

    def _repeat_to_length(pattern: str, L: int) -> str:
        pattern = "" if pattern is None else str(pattern).strip()
        if pattern == "" or pattern.lower() in {"nan", "unknown"}:
            return "U" * L
        reps = (L // len(pattern)) + 1
        return (pattern * reps)[:L]

    def _sample_start_end(row, L: int) -> tuple[int, int]:
        s_min = _safe_int(row["start_min"], 0)
        s_med = _safe_int(row["start_median"], s_min)
        s_max = _safe_int(row["start_max"], min(L - 2, s_min))

        e_min = _safe_int(row["end_min"], 1)
        e_med = _safe_int(row["end_median"], max(e_min, s_med + 2))
        e_max = _safe_int(row["end_max"], L)

        s_min = max(0, min(s_min, L - 2))
        s_max = max(0, min(s_max, L - 2))
        s_med = max(s_min, min(s_med, s_max))

        e_min = max(1, min(e_min, L))
        e_max = max(1, min(e_max, L))
        e_med = max(e_min, min(e_med, e_max))

        for _ in range(50):
            s = int(np.round(rng.triangular(s_min, s_med, s_max))) if s_max > s_min else int(s_min)
            s = max(0, min(s, L - 2))

            e_lo = max(e_min, s + 2)
            e_hi = e_max if e_max >= e_lo else L
            e_mode = max(e_lo, min(e_med, e_hi))

            e = int(np.round(rng.triangular(e_lo, e_mode, e_hi))) if e_hi > e_lo else int(e_lo)
            e = max(e_lo, min(e, L))

            if 0 <= s < e <= L:
                return s, e

        s = max(0, min(s_med, L - 2))
        e = max(s + 2, min(e_med, L))
        return int(s), int(e)

    def _allocate_counts_for_category(T_i: int, n: int) -> np.ndarray:
        if T_i < n:
            T_i = n
        base = np.ones(n, dtype=int)
        rem = T_i - n
        if rem <= 0:
            return base
        alpha = max(1e-6, float(dirichlet_alpha))
        probs = rng.dirichlet(np.full(n, alpha))
        return base + rng.multinomial(rem, probs)

    def _seq_around_del_from_full_seq(full_seq: str, start: int, end: int, isz: int) -> str:
        seq = str(full_seq)
        L = len(seq)

        deleted = seq[start:max(start, end - 1)]
        return (
            seq[max(0, start - isz):start]
            + deleted[:isz]
            + deleted[-isz:]
            + seq[end:end + isz]
        )

    def _deletion_length_category(start: int, end: int, L: int) -> str:
        """
        Relative deletion width bins:
          0–10%   -> micro deletion
          10–50%  -> short deletion
          50–90%  -> long deletion
          90–100% -> extensive deletion
        """
        if L <= 0:
            return "micro deletion"
        width = max(0, int(end) - int(start))
        frac = width / L

        if frac < 0.10:
            return "micro deletion"
        elif frac < 0.50:
            return "short deletion"
        elif frac < 0.90:
            return "long deletion"
        else:
            return "extensive deletion"

    # reference sequence cache
    seq_cache = {}

    def _get_ref_seq(strain: str, segment: str) -> str:
        key = (strain, segment)
        if key not in seq_cache:
            seq_cache[key] = str(get_sequence(strain, segment))
        return seq_cache[key]

    def _finalize_common(df: pd.DataFrame) -> pd.DataFrame:
        # keep your "unknown" defaults for non-metadata fields
        for col in ["time_point", "localization", "resolution", "cell_system", "MOI"]:
            df[col] = "unknown"

        df["NGS_read_count"] = 0
        df["isize"] = int(isize)

        df["key"] = df.apply(lambda r: f"{r['segment']}_{int(r['start'])}_{int(r['end'])}", axis=1)

        df["deleted_sequence"] = df.apply(
            lambda r: r["full_seq"][r["start"]:max(r["start"], r["end"] - 1)],
            axis=1,
        )

        df["seq_around_deletion_junction"] = df.apply(
            lambda r: _seq_around_del_from_full_seq(
                r["full_seq"], r["start"], r["end"], r["isize"]
            ),
            axis=1,
        )

        return df

    # >>> NEW: apply metadata per dataset using your imported add_metadata_features
    def _apply_metadata_by_dataset_name(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        # split into per-dataset chunks so your add_metadata_features(dfnames, dfs) works correctly
        dfnames = []
        dfs = []
        for dname, sub in df.groupby("dataset", sort=False):
            dfnames.append(str(dname))
            dfs.append(sub.copy())

        updated = add_metadata_features(dfnames, dfs)  # <-- your imported function
        return pd.concat(updated, ignore_index=True)

    # preprocessing
    stats = statistics_df.copy()
    stats["segment"] = stats["segment"].astype(str).str.strip()
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce").fillna(0)

    n_cat = len(stats)
    total_pseudo_rows = n_cat * n_per_category
    frac = stats["count"].to_numpy() / max(stats["count"].sum(), 1)
    T = np.maximum(np.rint(frac * total_pseudo_rows).astype(int), n_per_category)

    mono_rows, motif_rows, seq_rows = [], [], []

    nucleotides = ["A", "C", "G", "U"]
    motif_labels = [
        "motif before deletion",
        "deletion start motif",
        "deletion end motif",
        "motif after deletion",
    ]

    for idx, row in stats.reset_index(drop=True).iterrows():
        dname = str(row["dataset"])
        seg = str(row["segment"])
        strain = DATASET_STRAIN_DICT.get(dname, "unknown")

        L = _safe_int(row["full_seq_length"])
        if not L or L <= 0:
            continue

        starts, ends = zip(*[_sample_start_end(row, L) for _ in range(n_per_category)])
        counts = _allocate_counts_for_category(int(T[idx]), n_per_category)

        q = n_per_category // 4

        # mono
        for j, nuc in enumerate(nucleotides):
            for i in range(j * q, (j + 1) * q):
                mono_rows.append({
                    "dataset": dname,
                    "segment": seg,
                    "strain": strain,
                    "start": starts[i],
                    "end": ends[i],
                    "full_seq_length": L,
                    "full_seq": nuc * L,
                    "count": counts[i],
                    "composition": nuc,
                })

        # motif
        motifs = [row[f"site{k}_motif"] for k in range(1, 5)]
        for j, mot in enumerate(motifs):
            seq = _repeat_to_length(mot, L)
            for i in range(j * q, (j + 1) * q):
                motif_rows.append({
                    "dataset": dname,
                    "segment": seg,
                    "strain": strain,
                    "start": starts[i],
                    "end": ends[i],
                    "full_seq_length": L,
                    "full_seq": seq,
                    "count": counts[i],
                    "composition": motif_labels[j],
                })

        # sequence
        ref_seq = _get_ref_seq(strain, seg)
        ref_seq = ref_seq[:L].ljust(L, "U")

        for i in range(n_per_category):
            seq_rows.append({
                "dataset": dname,
                "segment": seg,
                "strain": strain,
                "start": starts[i],
                "end": ends[i],
                "full_seq_length": L,
                "full_seq": ref_seq,
                "count": counts[i],
                "composition": _deletion_length_category(starts[i], ends[i], L),
            })

    mono_df = _apply_metadata_by_dataset_name(_finalize_common(pd.DataFrame(mono_rows)))
    motif_df = _apply_metadata_by_dataset_name(_finalize_common(pd.DataFrame(motif_rows)))
    seq_df = _apply_metadata_by_dataset_name(_finalize_common(pd.DataFrame(seq_rows)))

    return mono_df, motif_df, seq_df


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################
    
    ### DATASETS MULTI ###

    folder = 'unpooled'   # pooled | unpooled
    subfolder = 'secondary'   # primary | secondary | motif_length_{K_MER_LENGTH}

    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ### PREPROCESS ###

    data = 'IAV'
    strain = 'PR8'
    segment = 'PB1'
    intersects = 'mean_dataset_0'

    dfs = manage_separate_specifiers(dfs, data, strain, segment)
    df = add_intersect_ngs_features(dfs, intersects)
    dfs = [df[df["dataset"] == name].copy() for name in dfnames]

    feature = 'time_point'
    value = 'unknown'

    ###################
    ### run scripts ###
    ###################

    # run_strain_segment_statistics(dfs)

    # run_pseudo_candidate_dfs()

    # print_df_sizes(dfnames, dfs)

    print_percentage_of_value(df, feature, value)