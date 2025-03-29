'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import RESULTSPATH, SEGMENTS, DATASET_STRAIN_DICT
from utils import get_sequence, get_dataset_names, count_direct_repeats_overall

from pseudo import pseudo_load_all

RESULTSPATH, _ = os.path.split(RESULTSPATH)

def create_direct_repeats_plot(df, dfname, pseudo_df, pseudo_dfname, expected_df):
    '''

    '''
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    final_d = dict()
    expected_final_d = dict()
    pseudo_final_d = dict()

    pseudo_strain = DATASET_STRAIN_DICT[pseudo_dfname]

    for s in SEGMENTS:
        df_s = df[df["Segment"] == s]
        pseudo_df_s = pseudo_df[pseudo_df["Segment"] == s]
        expected_df_s = expected_df[expected_df["Segment"] == s]
        
        n_samples = len(df_s)
        if n_samples == 0:
            continue

        seq = get_sequence(df_s["Strain"].unique()[0], s)            
        counts, _ = count_direct_repeats_overall(df_s, seq)
        
        for k, v in counts.items():
            if k in final_d:
                final_d[k] += v
            else:
                final_d[k] = v

        expected_counts, _ = count_direct_repeats_overall(expected_df_s, seq)
        for k, v in expected_counts.items():
            if k in expected_final_d:
                expected_final_d[k] += v
            else:
                expected_final_d[k] = v
        
        pseudo_counts, _ = count_direct_repeats_overall(pseudo_df_s, seq)
        for k, v in pseudo_counts.items():
            if k in pseudo_final_d:
                pseudo_final_d[k] += v
            else:
                pseudo_final_d[k] = v

    final = np.array(list(final_d.values()))
    expected_final = np.array(list(expected_final_d.values()))
    pseudo_final = np.array(list(pseudo_final_d.values()))

    f_obs = final / final.sum() * 100
    f_exp = expected_final / expected_final.sum() * 100
    f_pseudo = pseudo_final / pseudo_final.sum() * 100

    stats_save_path = os.path.join(RESULTSPATH, "pseudo")
    os.makedirs(stats_save_path, exist_ok=True)
    abs_stats_save_path = os.path.join(stats_save_path, "dr_stats.csv")
    if not os.path.isfile(abs_stats_save_path):
        columns = ["Strain", "Coordinates", "rea0", "rea1", "rea2", "rea3", "rea4", "rea>4", "syn0", "syn1", "syn2", "syn3", "syn4", "syn>4", "pse0", "pse1", "pse2", "pse3", "pse4", "pse>4"]
        stats_df = pd.DataFrame(columns=columns)
        stats_df.to_csv(abs_stats_save_path, index=True)
    stats_df = pd.read_csv(abs_stats_save_path, index_col=0)

    if not ((stats_df["Strain"] == pseudo_dfname) & (stats_df["Coordinates"] == dfname)).any():
        s_obs = np.round(f_obs, 2)
        s_exp = np.round(f_exp, 2)
        s_pseudo = np.round(f_pseudo, 2)
        new_row = pd.DataFrame({
            "Strain": [pseudo_strain],
           "Coordinates": [dfname],
           "rea0": [s_obs[0]],
           "rea1": [s_obs[1]],
           "rea2": [s_obs[2]],
           "rea3": [s_obs[3]],
           "rea4": [s_obs[4]],
           "rea>4": [s_obs[5]],
            "syn0": [s_exp[0]],
            "syn1": [s_exp[1]],
            "syn2": [s_exp[2]],
           "syn3": [s_exp[3]],
           "syn4": [s_exp[4]],
           "syn>4": [s_exp[5]],
           "pse0": [s_pseudo[0]],
           "pse1": [s_pseudo[1]],
          "pse2": [s_pseudo[2]],
          "pse3": [s_pseudo[3]],
          "pse4": [s_pseudo[4]],
          "pse>4": [s_pseudo[5]]
     })
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
        stats_df.to_csv(abs_stats_save_path)

    x = np.arange(len(final_d))

    ax.bar(x - 0.2, f_obs, width=0.2, align="center", label="real-strain", edgecolor="black", color="firebrick")
    ax.bar(x, f_exp, width=0.2, align="center", label="expected", edgecolor="black", color="royalblue")
    ax.bar(x + 0.2, f_pseudo, width=0.2, align="center", label="pseudo-strain", edgecolor="black", color="green")

    ax.set_title("pseudo-strain: " + pseudo_strain + "\n coordinates: " + dfname)
    ax.legend(loc="upper right", labels=["real-strain", "synthetic", "pseudo-strain"])
    ax.set_xlabel("direct repeat length (nucleotides)")
    ax.set_ylabel("relative occurrence (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1", "2", "3", "4", ">4"])

    save_path = os.path.join(RESULTSPATH, "pseudo")
    os.makedirs(save_path, exist_ok=True)
    fname = f"dr_{pseudo_dfname}_co_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    ### single ###

    # pseudo_dfname = "Alnaji2021"
    # dfnames = ["Berry2021_B_Yam"]
    # dfs, pseudo_dfs, exp_dfs = pseudo_load_all(dfnames, pseudo_dfname, True)

    # dfname = dfnames[0]
    # df = dfs[0]
    # pseudo_df = pseudo_dfs[0]
    # exp_df = exp_dfs[0]

    # create_direct_repeats_plot(df, dfname, pseudo_df, pseudo_dfname, exp_df)

    ### multi ###

    # pseudo_dfname = "Alnaji2021"
    # coordinates = "IBV"
    # dfname = coordinates
    # dfnames = get_dataset_names(cutoff=40, selection=coordinates)
    # dfs, pseudo_dfs, exp_dfs = pseudo_load_all(dfnames, pseudo_dfname, True)

    # df = pd.concat(dfs, ignore_index=True)
    # pseudo_df = pd.concat(pseudo_dfs, ignore_index=True)
    # exp_df = pd.concat(exp_dfs, ignore_index=True)

    # create_direct_repeats_plot(df, dfname, pseudo_df, pseudo_dfname, exp_df)