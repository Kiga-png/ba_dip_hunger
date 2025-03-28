'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import RESULTSPATH, NUCLEOTIDES, DATASET_STRAIN_DICT, SEGMENTS, CMAP
from utils import load_all, get_dataset_names, create_nucleotide_ratio_matrix, get_sequence

from pseudo import pseudo_load_all

RESULTSPATH, _ = os.path.split(RESULTSPATH)


def create_nucleotide_enrichment_plot(df, dfname, pseudo_df, pseudo_name, exp_df):
    '''

    '''
    fig, axs = plt.subplots(figsize=(10, 3), tight_layout=True)

    pseudo_strain = DATASET_STRAIN_DICT[pseudo_dfname]

    colors_observed = ["darkred", "firebrick", "indianred", "lightcoral"]
    colors_synthetic = ["navy", "royalblue", "cornflowerblue", "lightblue"]
    colors_pseudo = ["forestgreen", "lightgreen", "mediumseagreen", "yellowgreen"]

    shift_observed = -0.2
    shift_synthetic = 0
    shift_pseudo = 0.2

    m = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
    for nucleotide in ["A", "C", "G", "U"]:
        m[nucleotide] = m[nucleotide].round(2)

    bottom = np.zeros(len(m.index))
    for i, c in enumerate(m.columns):
        axs.bar(m.index + shift_observed, m[c], width=0.2, label='real-strain', color=colors_observed[i], bottom=bottom, edgecolor="black")
        bottom += m[c]

    m_exp = create_nucleotide_ratio_matrix(exp_df, "seq_around_deletion_junction")
    bottom = np.zeros(len(m_exp.index))
    for i, c in enumerate(m_exp.columns):
        axs.bar(m.index + shift_synthetic, m_exp[c], width=0.2, label='synthetic', color=colors_synthetic[i], bottom=bottom, edgecolor="black")
        bottom += m_exp[c]

    m_pseudo = create_nucleotide_ratio_matrix(pseudo_df, "seq_around_deletion_junction")
    bottom = np.zeros(len(m_pseudo.index))
    for i, c in enumerate(m_pseudo.columns):
        axs.bar(m.index + shift_pseudo, m_pseudo[c], width=0.2, label='pseudo-strain', color=colors_pseudo[i], bottom=bottom, edgecolor="black")
        bottom += m_pseudo[c]

    axs.set_ylim(0, np.max(bottom) * 1.1)

    axs.set_xticks(m.index)
    axs.set_xticklabels(list(range(1, 11)) * 2)
    plt.axvline(x=10.5, color="grey", linewidth=4)

    y_positions = np.linspace(0.05 * np.max(bottom), 0.95 * np.max(bottom), len(NUCLEOTIDES))
    pos = 0
    for i, n in enumerate(NUCLEOTIDES):
        axs.text(0, y_positions[i], n, color="black", fontweight="bold", fontsize=20, ha="center", va="center")

    axs.set_title("pseudo-strain: " + pseudo_strain + " - coordinates: " + dfname, pad=15)
    axs.set_xlabel("nucleotide position (-)")
    axs.set_ylabel("relative occurrence (-)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors_observed[1]),
        plt.Rectangle((0, 0), 1, 1, color=colors_synthetic[1]),
        plt.Rectangle((0, 0), 1, 1, color=colors_pseudo[2])
    ]
    plt.legend(handles, ['real-strain', 'synthetic', 'pseudo-strain'], loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)

    xlabels = axs.get_xticklabels()
    quarter = len(m.index) // 4
    for x_idx, xlabel in enumerate(xlabels):
        if x_idx < quarter or x_idx >= quarter * 3:
            xlabel.set_color("black")
            xlabel.set_fontweight("bold")
        else:
            xlabel.set_color("grey")
            xlabel.set_fontweight("normal")

    save_path = os.path.join(RESULTSPATH, "pseudo")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ne_{pseudo_dfname}_co_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_adenin_distribution_datasets(dfs: list, dfnames: list):
    a_fracs = dict()
    a_max = ("strain", "segment", 0)
    a_min = ("strain", "segment", 1)
    for df, dfname in zip(dfs, dfnames):
        st = DATASET_STRAIN_DICT[dfname]
        a_fracs[st] = list()
        for seg in SEGMENTS:
            df_s = df.loc[df["Segment"] == seg]
            if len(df_s) == 0:
                continue
            seq = get_sequence(st, seg)
            start = int(df_s["Start"].mean())
            end = int(df_s["End"].mean())
            s = (max(start-200, 50), start+200)
            e = (end-200, min(end+200, len(seq)-50))

            if s[0] == s[1] or e[0] == e[1]:
                continue
            if s[1] > e[0]:
                continue

            seq = seq[s[0]:s[1]] + seq[e[0]:e[1]]

            a_counts = seq.count("A")
            assert a_counts != 0, f"No Adenines found in {st} {seg}: {seq}"
            
            a_frac = a_counts/len(seq)
            if a_frac > a_max[2]:
                a_max = (st, seg, a_frac)
            if a_frac < a_min[2]:
                a_min = (st, seg, a_frac)

            a_fracs[st].append(a_frac)

    print(f"Max. occurence of a: {a_max[0]}\t{a_max[1]}\t{a_max[2]}")
    print(f"Min. occurence of a: {a_min[0]}\t{a_min[1]}\t{a_min[2]}")

    a_occ = list()
    for st, values in a_fracs.items():
        for v in values:
            a_occ.append(v*v)    

    print(f"{max(a_occ)*100:.1f}")
    print(f"{min(a_occ)*100:.1f}")


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    ##############################
    ######## SINGLE USAGE ########
    ##############################

    # pseudo_dfname = "Alnaji2021"
    # dfnames = ["Berry2021_B_Yam"]
    # dfs, pseudo_dfs, exp_dfs = pseudo_load_all(dfnames, pseudo_dfname, True)

    # dfname = dfnames[0]
    # df = dfs[0]
    # pseudo_df = pseudo_dfs[0]
    # exp_df = exp_dfs[0]

    # create_nucleotide_enrichment_plot(df, dfname, pseudo_df, pseudo_dfname, exp_df)

    ##############################
    ######## MULTI USAGE #########
    ##############################

    pseudo_dfname = "Alnaji2021"
    coordinates = "IBV"
    dfname = coordinates
    dfnames = get_dataset_names(cutoff=40, selection=coordinates)
    dfs, pseudo_dfs, exp_dfs = pseudo_load_all(dfnames, pseudo_dfname, True)

    df = pd.concat(dfs, ignore_index=True)
    pseudo_df = pd.concat(pseudo_dfs, ignore_index=True)
    exp_df = pd.concat(exp_dfs, ignore_index=True)

    create_nucleotide_enrichment_plot(df, dfname, pseudo_df, pseudo_dfname, exp_df)