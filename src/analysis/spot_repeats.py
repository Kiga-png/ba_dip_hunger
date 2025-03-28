'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import DATAPATH, RESULTSPATH, SEGMENTS, DATASET_STRAIN_DICT
from utils import get_sequence, load_all

RESULTSPATH, _ = os.path.split(RESULTSPATH)

##############################
######## REPEATS RATIO #######
##############################

def create_repeats_ratio_plot(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    
    if df.empty:
        print("df isempty")
        return

    segments = df['Segment'].unique()
    heights = []
    
    for segment in segments:
        segment_data = df[df['Segment'] == segment]
        if segment_data.empty:
            continue
        heights.append(segment_data['hits_repeat_ratio'].sum())

    ax.bar(segments, heights, color='royalblue', edgecolor='black', width=0.8)

    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("segment")
    ax.set_ylabel("relative hits of possible repeats (%)")
    ax.set_xticks(segments)
    ax.set_xticklabels(segments)

    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"dr_ratio_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_repeats_ratio(dfname: str, df: pd.DataFrame):
    stats_df = pd.DataFrame(columns=['Segment', 'possible_repeats', 'hits_count', 'hits_repeat_ratio'])
    strain = DATASET_STRAIN_DICT[dfname]
    data_path = os.path.join(DATAPATH, f"repeats/{strain}")
    for segment in SEGMENTS:
        segment_path = os.path.join(data_path, f"{segment}.csv")
        repeats = pd.read_csv(segment_path, index_col=0)
        total_rows = repeats.shape[0]
        dic = count_direct_repeats_segment(df, segment)
        hits_count = dic[5]
        if total_rows > 0:
            ratio = hits_count / total_rows
        else:
            ratio = 0
        new_row = pd.DataFrame({
            'Segment': [segment],
            'possible_repeats': [total_rows],
            'hits_count': [hits_count],
            'hits_repeat_ratio': [ratio]
        })    
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
    return stats_df

def old_calculate_repeats_ratio(dfname: str):
    repeats_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/stats")
    stats_df = pd.DataFrame(columns=['Segment', 'possible_repeats', 'hits_count', 'hits_repeat_ratio'])
    #TODO not existing
    for segment in SEGMENTS:
        seg_repeats_path = os.path.join(repeats_path, f"{segment}.csv")
        repeats = pd.read_csv(seg_repeats_path, index_col=0)
        total_rows = repeats.shape[0]
        hits_one_count = (repeats['hits'] == 1).sum()
        if total_rows > 0:
            ratio = hits_one_count / total_rows
        else:
            ratio = 0
        new_row = pd.DataFrame({
            'Segment': [segment],
            'possible_repeats': [total_rows],
            'hits_count': [hits_one_count],
            'hits_repeat_ratio': [ratio]
        })
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
    fname = f"repeats_stats.csv"
    stats_df.to_csv(os.path.join(repeats_path, fname))
    return stats_df

def analyze_repeats(dfname: str, df: pd.DataFrame):
    strain = DATASET_STRAIN_DICT[dfname]
    data_path = os.path.join(DATAPATH, f"repeats/{strain}")
    save_path = os.path.join(RESULTSPATH, f"repeats/{dfname}/stats")
    os.makedirs(save_path, exist_ok=True)
    for segment in SEGMENTS:
        segment_path = os.path.join(data_path, f"{segment}.csv")
        repeats = pd.read_csv(segment_path, index_col=0)
        repeats['hits'] = 0
        repeats['NGS_read_count'] = 0
        segment_df = df[df["Segment"] == segment]
        for _, df_row in segment_df.iterrows():
            start = df_row["Start"]
            end = df_row["End"]
            for _, repeats_row in repeats.iterrows():
                s = start == repeats_row["end_ref"]
                e = end - 1 == repeats_row["end_rep"]
                if s and e:
                    repeats_row['hits'] = repeats_row['hits'] + 1
                    repeats_row['NGS_read_count'] = repeats_row['NGS_read_count'] + df_row['NGS_read_count']
        fname = f"{segment}.csv"
        repeats.to_csv(os.path.join(save_path, fname))

##############################
######## SPOT REPEATS ########
##############################

def spot_repeats_all(min_len: int):
    strains = set(DATASET_STRAIN_DICT.values())
    for strain in strains:
        spot_repeats(strain, min_len)

def spot_repeats(strain: str, min_len: int):
    save_path = os.path.join(DATAPATH, f"repeats/{strain}")
    os.makedirs(save_path, exist_ok=True)
    for segment in SEGMENTS:
        seq = get_sequence(strain, segment)
        repeats = spot_repeats_segment(seq, min_len)
        fname = f"{segment}.csv"
        repeats.to_csv(os.path.join(save_path, fname))

def spot_repeats_segment(seq: str, min_len: int):
    repeats = []
    seq_len = len(seq)
    for start in range(seq_len):
        for length in range(1, seq_len - start + 1):
            if length < min_len:
                continue
            substring = seq[start:start + length]
            next_occurrence = seq.find(substring, start + length)
            while next_occurrence != -1:
                repeats.append((start + 1, start + length, next_occurrence + 1, next_occurrence + length))
                next_occurrence = seq.find(substring, next_occurrence + 1)
    df = pd.DataFrame(repeats, columns=["start_ref", "end_ref", "start_rep", "end_rep"])
    df = remove_rows_with_smaller_start(df)
    return df

def remove_rows_with_smaller_start(df: pd.DataFrame):
    grouped = df.groupby(['end_ref', 'end_rep'])

    def keep_max_start_ref(group):
        return group[group['start_ref'] == group['start_ref'].min()]
    
    filtered_df = grouped.apply(keep_max_start_ref).reset_index(drop=True)
    return filtered_df

##############################
######## COUNT REPEATS #######
##############################

def count_direct_repeats(df: pd.DataFrame):
    df = add_direct_repeat_len(df)
    df = cap_direct_repeat_len(df, 5)
    nuc_overlap_dict = dict({i: 0 for i in range(0, 6)})
    for index, row in df.iterrows():
        i = row["direct_repeat_len"]
        nuc_overlap_dict[i] = nuc_overlap_dict[i] + 1
    return nuc_overlap_dict

def count_direct_repeats_segment(df: pd.DataFrame, segment: str):
    df = add_direct_repeat_len(df)
    df = cap_direct_repeat_len(df, 5)
    nuc_overlap_dict = dict({i: 0 for i in range(0, 6)})
    segment_df = df[df["Segment"] == segment]
    for index, row in segment_df.iterrows():
        i = row["direct_repeat_len"]
        nuc_overlap_dict[i] = nuc_overlap_dict[i] + 1
    return nuc_overlap_dict

def cap_direct_repeat_len(df: pd.DataFrame, cap: int):
    for index, row in df.iterrows():
        direct_repeat_len = row["direct_repeat_len"]
        if direct_repeat_len > cap:
            df.loc[index, "direct_repeat_len"] = cap
    return df

def add_direct_repeat_len(df: pd.DataFrame):
    for index, row in df.iterrows():
        seq = row["full_seq"]
        start = row["Start"]
        end = row["End"] - 1
        direct_repeat_len = 0
        while start > 0 and seq[start - 1] == seq[end - 1]:
            direct_repeat_len = direct_repeat_len + 1
            start = start - 1
            end = end - 1
        df.loc[index, "direct_repeat_len"] = int(direct_repeat_len)
    return df

if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    ##############################
    ######## SPOT REPEATS ########
    ##############################

    # min_len = 5
    # spot_repeats_all(min_len)

    ##############################
    ######## SINGLE USAGE ########
    ##############################

    dfnames = ["Alnaji2021"]
    dfs, _ = load_all(dfnames)

    dfname = dfnames[0]
    df = dfs[0]

    ratio_df = calculate_repeats_ratio(dfname, df)
    create_repeats_ratio_plot(dfname, ratio_df)

    ##############################
    ######## MULTI USAGE #########
    ##############################