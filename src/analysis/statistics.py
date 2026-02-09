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

from utils import manage_separate_specifiers, add_intersect_ngs_features
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
        "dataset_name",
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

    gb = df.groupby(["dataset_name", "segment"], dropna=False)

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
    Build 3 pseudo-candidate dfs (mono, motif, sequence) from statistics_df where each row is unique
    by (dataset_name, segment).

    Required statistics_df columns:
      dataset_name, segment, count,
      start_min, start_median, start_max,
      end_min, end_median, end_max,
      full_seq_length,
      site1_motif, site2_motif, site3_motif, site4_motif

    Uses global project objects (imported in your module):
      - DATASET_STRAIN_DICT
      - get_sequence(strain, segment)

    Output columns include (minimum):
      dataset_name, segment, strain, start, end, full_seq_length, full_seq, count,
      time_point, localization, resolution, cell_system, MOI ('unknown'),
      NGS_read_count (0),
      key (f"{segment}_{start}_{end}"),
      isize (int),
      deleted_sequence (full_seq[start:end-1]),
      seq_around_deletion_junction (built from full_seq only).
    """

    rng = np.random.default_rng(SEED)

    required_cols = [
        "dataset_name", "segment", "count",
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
        """Repeat pattern to length L, truncating excess. If invalid -> fallback to 'U'*L."""
        pattern = "" if pattern is None else str(pattern)
        pattern = pattern.strip()
        if pattern == "" or pattern.lower() in {"nan", "unknown"}:
            return "U" * L
        reps = (L // len(pattern)) + 1
        return (pattern * reps)[:L]

    def _sample_start_end(row, L: int) -> tuple[int, int]:
        """
        Sample start/end using bounded triangular distributions (min, median, max),
        then enforce 0 <= start < end <= L and end >= start+2.
        """
        s_min = _safe_int(row["start_min"], 0)
        s_med = _safe_int(row["start_median"], s_min)
        s_max = _safe_int(row["start_max"], min(L - 2, s_min))

        e_min = _safe_int(row["end_min"], 1)
        e_med = _safe_int(row["end_median"], max(e_min, s_med + 2))
        e_max = _safe_int(row["end_max"], L)

        # clamp start domain
        s_min = max(0, min(s_min, L - 2))
        s_max = max(0, min(s_max, L - 2))
        s_med = max(s_min, min(s_med, s_max))

        # clamp end domain
        e_min = max(1, min(e_min, L))
        e_max = max(1, min(e_max, L))
        e_med = max(e_min, min(e_med, e_max))

        for _ in range(50):
            s = int(np.round(rng.triangular(s_min, s_med, s_max))) if s_max > s_min else int(s_min)
            s = max(0, min(s, L - 2))

            e_lo = max(e_min, s + 2)
            e_hi = e_max
            if e_hi < e_lo:
                e_lo = min(L, s + 2)
                e_hi = L

            e_mode = max(e_lo, min(e_med, e_hi))
            e = int(np.round(rng.triangular(e_lo, e_mode, e_hi))) if e_hi > e_lo else int(e_lo)
            e = max(e_lo, min(e, L))

            if 0 <= s < e <= L:
                return s, e

        # fallback
        s = max(0, min(s_med, L - 2))
        e = max(s + 2, min(e_med, L))
        return int(s), int(e)

    def _allocate_counts_for_category(T_i: int, n: int) -> np.ndarray:
        """
        Allocate integer counts across n rows:
          - min 1 per row
          - remaining mass via Dirichlet(alpha<1) + multinomial (many low, few high)
        """
        T_i = int(T_i)
        n = int(n)
        if T_i < n:
            T_i = n
        base = np.ones(n, dtype=int)
        rem = T_i - n
        if rem <= 0:
            return base
        alpha = float(max(1e-6, dirichlet_alpha))
        probs = rng.dirichlet(alpha=np.full(n, alpha))
        add = rng.multinomial(rem, probs)
        return base + add

    def _seq_around_del_from_full_seq(full_seq: str, start: int, end: int, isz: int) -> str:
        """
        Build junction context ONLY from full_seq:
          before_start = full_seq[start-isz : start]
          after_start  = deleted_sequence[:isz]
          before_end   = deleted_sequence[-isz:]
          after_end    = full_seq[end : end+isz]
        where deleted_sequence = full_seq[start : end-1] (per your spec).
        """
        seq = str(full_seq)
        L = len(seq)
        s = int(start)
        e = int(end)
        isz = int(isz)

        # clamp
        s = max(0, min(s, L))
        e = max(0, min(e, L))

        # deleted part uses end-1 as you specified
        del_lo = max(0, min(s, L))
        del_hi = max(0, min(e - 1, L))
        deleted = seq[del_lo:del_hi] if del_hi > del_lo else ""

        before_start = seq[max(0, s - isz):s]
        after_end = seq[e:min(L, e + isz)]

        after_start = deleted[:isz]
        before_end = deleted[-isz:] if len(deleted) > 0 else ""

        return before_start + after_start + before_end + after_end

    # cache real reference sequences (sequence df only)
    seq_cache: dict[tuple[str, str], str] = {}

    def _get_ref_seq(strain: str, segment: str) -> str:
        key = (strain, segment)
        if key not in seq_cache:
            seq_cache[key] = str(get_sequence(strain, segment))
        return seq_cache[key]

    def _finalize_common(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["time_point", "localization", "resolution", "cell_system", "MOI"]:
            df[col] = "unknown"
        df["NGS_read_count"] = 0
        df["isize"] = int(isize)

        df["key"] = df.apply(lambda r: f"{r['segment']}_{int(r['start'])}_{int(r['end'])}", axis=1)

        # deleted_sequence = full_seq[start:end-1]
        def _deleted_seq(r):
            seq = str(r["full_seq"])
            s = int(r["start"])
            e = int(r["end"])
            lo = max(0, min(s, len(seq)))
            hi = max(0, min(e - 1, len(seq)))
            if hi <= lo:
                return ""
            return seq[lo:hi]

        df["deleted_sequence"] = df.apply(_deleted_seq, axis=1)

        # seq_around_deletion_junction from full_seq only
        df["seq_around_deletion_junction"] = df.apply(
            lambda r: _seq_around_del_from_full_seq(r["full_seq"], r["start"], r["end"], r["isize"]),
            axis=1
        )

        return df

    # compute category totals for pseudo-count allocation
    stats = statistics_df.copy()

    # keep "NA" as literal string if present
    stats["segment"] = stats["segment"].astype(str).str.strip()

    stats["count"] = pd.to_numeric(stats["count"], errors="coerce").fillna(0.0)
    total_count = float(stats["count"].sum())

    n_cat = int(len(stats))
    if n_cat == 0:
        raise ValueError("statistics_df is empty")

    total_pseudo_rows = n_cat * int(n_per_category)

    if total_count <= 0:
        frac = np.full(n_cat, 1.0 / n_cat, dtype=float)
    else:
        frac = stats["count"].to_numpy(dtype=float) / total_count

    T = np.rint(frac * total_pseudo_rows).astype(int)
    T = np.maximum(T, int(n_per_category))

    if T.sum() < total_pseudo_rows:
        deficit = total_pseudo_rows - int(T.sum())
        probs = frac if frac.sum() > 0 else np.full(n_cat, 1.0 / n_cat)
        add = rng.multinomial(deficit, probs)
        T = T + add

    # generate
    mono_rows = []
    motif_rows = []
    seq_rows = []

    nucleotides = ["A", "C", "G", "U"]

    for idx, row in stats.reset_index(drop=True).iterrows():
        dname = str(row["dataset_name"])
        seg = str(row["segment"]).strip()

        strain = DATASET_STRAIN_DICT.get(dname, "unknown")

        L = _safe_int(row["full_seq_length"], None)
        if L is None or L <= 0:
            continue

        # sample 100 start/end pairs
        starts, ends = [], []
        for _ in range(int(n_per_category)):
            s, e = _sample_start_end(row, L)
            starts.append(int(s))
            ends.append(int(e))

        # allocate counts for these 100 rows
        counts_100 = _allocate_counts_for_category(int(T[idx]), int(n_per_category))

        n_q = int(n_per_category) // 4
        if n_q * 4 != int(n_per_category):
            raise ValueError("n_per_category must be divisible by 4 for mono/motif splits.")

        # mono df
        for j, nuc in enumerate(nucleotides):
            full_seq = nuc * int(L)
            for i in range(j * n_q, (j + 1) * n_q):
                mono_rows.append(
                    {
                        "dataset_name": dname,
                        "segment": seg,
                        "strain": strain,
                        "start": starts[i],
                        "end": ends[i],
                        "full_seq_length": int(L),
                        "full_seq": full_seq,
                        "count": int(counts_100[i]),
                    }
                )

        # motif df
        motifs = [
            row.get("site1_motif", "unknown"),
            row.get("site2_motif", "unknown"),
            row.get("site3_motif", "unknown"),
            row.get("site4_motif", "unknown"),
        ]
        for j, mot in enumerate(motifs):
            motif_seq = _repeat_to_length(str(mot), int(L))  # EXACT length L
            for i in range(j * n_q, (j + 1) * n_q):
                motif_rows.append(
                    {
                        "dataset_name": dname,
                        "segment": seg,
                        "strain": strain,
                        "start": starts[i],
                        "end": ends[i],
                        "full_seq_length": int(L),
                        "full_seq": motif_seq,
                        "count": int(counts_100[i]),
                    }
                )

        # sequence df
        ref_seq = _get_ref_seq(strain, seg)
        if len(ref_seq) != int(L):
            ref_seq = (ref_seq[: int(L)]).ljust(int(L), "U")

        for i in range(int(n_per_category)):
            seq_rows.append(
                {
                    "dataset_name": dname,
                    "segment": seg,
                    "strain": strain,
                    "start": starts[i],
                    "end": ends[i],
                    "full_seq_length": int(L),
                    "full_seq": ref_seq,
                    "count": int(counts_100[i]),
                }
            )

    df_mono = _finalize_common(pd.DataFrame(mono_rows))
    df_motif = _finalize_common(pd.DataFrame(motif_rows))
    df_sequence = _finalize_common(pd.DataFrame(seq_rows))

    return df_mono, df_motif, df_sequence


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
    subfolder = f'motif_length_{K_MER_LENGTH}'   # primary | secondary | motif_length_{K_MER_LENGTH}

    dfnames = get_dataset_names(DATASET_CUTOFF)
    dfs = load_all_preprocessed(dfnames, folder, subfolder)

    ###################
    ### run scripts ###
    ###################

    # run_strain_segment_statistics(dfs)

    run_pseudo_candidate_dfs()