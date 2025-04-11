'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Tuple

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names
from utils import add_norm_log_ngs_read_count_list
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


######################
### ngs histo dist ###
######################

def create_norm_ngs_read_count_dist(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3), tight_layout=True)

    data = df['norm_log_NGS_read_count'].dropna()
    num_rows = data.shape[0] 

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    bin_width = bins[1] - bins[0]
    hist, bin_edges = np.histogram(data, bins=bins)
    hist = hist / num_rows * 100

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")

    tick_indices = np.arange(0, len(bin_edges), 5)
    ax.set_xticks(bin_edges[tick_indices])
    ax.set_xticklabels([f"{bin_edges[i]:.2f}" for i in tick_indices])

    density = stats.gaussian_kde(data)
    x_vals = np.linspace(0, 1, 500)
    y_vals = density(x_vals) * 100 * bin_width
    ax.plot(x_vals, y_vals, color='red', linewidth=2, label='empirical density')

    mean = data.mean()
    variance = data.var()
    skewness = data.skew()
    kurtosis = data.kurtosis()

    stats_text = f"mean: {mean:.3f}\nvariance: {variance:.3f}\nskewness: {skewness:.3f}\nkurtosis: {kurtosis:.3f}"
    ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    ax.legend()
    ax.set_ylim(0, 15)

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"norm_ngs_read_count_dist_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

#####################
### ngs histo reg ###
#####################

def create_norm_ngs_read_count_abs_reg(dfname: str, df: pd.DataFrame, degree: int = 4):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    hist, bin_edges = np.histogram(df['norm_log_NGS_read_count'], bins=bins)

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("NGS read count")
    ax.set_ylabel("numberof DVGs (-)")
    
    tick_indices = np.arange(0, len(bin_edges), 5)
    ax.set_xticks(bin_edges[tick_indices])
    ax.set_xticklabels([f"{bin_edges[i]:.2f}" for i in tick_indices])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers.reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=degree)
    bin_centers_poly = polynomial_features.fit_transform(bin_centers)
    model = LinearRegression()
    model.fit(bin_centers_poly, hist)
    y_pred = model.predict(bin_centers_poly)

    ax.plot(bin_centers, y_pred, color='red', linewidth=2, label=f'polynomial regression (degree {degree})')
    ax.legend()

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"norm_ngs_read_count_abs_reg_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_norm_ngs_read_count_reg(dfname: str, df: pd.DataFrame, degree: int = 4):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    num_rows = df.shape[0] 
    max_value = 1.0
    bins = np.linspace(0, max_value, 101)
    hist, bin_edges = np.histogram(df['norm_log_NGS_read_count'], bins=bins)
    hist = hist / num_rows * 100

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel("NGS read count")
    ax.set_ylabel("distribution of DVGs (%)")
    
    tick_indices = np.arange(0, len(bin_edges), 5)
    ax.set_xticks(bin_edges[tick_indices])
    ax.set_xticklabels([f"{bin_edges[i]:.2f}" for i in tick_indices])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers.reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=degree)
    bin_centers_poly = polynomial_features.fit_transform(bin_centers)
    model = LinearRegression()
    model.fit(bin_centers_poly, hist)
    y_pred = model.predict(bin_centers_poly)

    ax.plot(bin_centers, y_pred, color='red', linewidth=2, label=f'polynomial regression (degree {degree})')
    ax.legend()

    ax.set_ylim(0, 15)

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"norm_ngs_read_count_reg_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

#####################
### raw ngs histo ###
#####################

def create_ngs_read_count_abs_histo(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    
    upper_limit = 1000
    bins = range(0, upper_limit + 10, 10)
    hist, bin_edges = np.histogram(df['NGS_read_count'], bins=bins)

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel(f"raw NGS read count (capped: 1010)")
    ax.set_ylabel("number of DVGs (-)")
    
    selected_ticks = bin_edges[::5]
    ax.set_xticks(selected_ticks)
    selected_labels = [str(b) for b in selected_ticks]
    ax.set_xticklabels(selected_labels)

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_read_count_abs_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_ngs_read_count_histo(dfname: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    
    num_rows = df.shape[0]
    upper_limit = 1000
    bins = range(0, upper_limit + 10, 10)
    hist, bin_edges = np.histogram(df['NGS_read_count'], bins=bins)
    hist = hist / num_rows * 100

    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='royalblue', edgecolor='black')
    ax.set_title(f"dataset: {dfname}")
    ax.set_xlabel(f"raw NGS read count (capped: 1010)")
    ax.set_ylabel("distribution of DVGs (%)")
    
    selected_ticks = bin_edges[::5]
    ax.set_xticks(selected_ticks)
    selected_labels = [str(b) for b in selected_ticks]
    ax.set_xticklabels(selected_labels)

    save_path = os.path.join(RESULTSPATH, f"ngs/{dfname}")
    os.makedirs(save_path, exist_ok=True)
    fname = f"ngs_read_count_{dfname}.png"
    save_path = os.path.join(save_path, fname)
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    '''

    '''
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    ### SINGLE ###

    dfnames = ["Berry2021_B_Yam"]
    dfs, _ = load_all(dfnames)
    dfname = dfnames[0]

    ### MULTI ###

    # coordinates = "IBV"
    # dfname = coordinates
    # dfnames = get_dataset_names(cutoff=40, selection=coordinates)
    # dfs, _ = load_all(dfnames, False)

    ######################
    ### ngs histo dist ###
    ######################

    ### single ###

    dfs = add_norm_log_ngs_read_count_list(dfs)
    df = pd.concat(dfs, axis=0)
    create_norm_ngs_read_count_dist(dfname, df)

    ### multi ###

    # dfs = add_norm_log_ngs_read_count_list(dfs)
    # df = pd.concat(dfs, axis=0)
    # create_norm_ngs_read_count_dist(dfname, df)

    #####################
    ### ngs histo reg ###
    #####################

    ### single ###

    # dfs = add_norm_log_ngs_read_count_list(dfs)
    # df = pd.concat(dfs, axis=0)
    # create_norm_ngs_read_count_reg(dfname, df)

    ### multi ###

    # dfs = add_norm_log_ngs_read_count_list(dfs)
    # df = pd.concat(dfs, axis=0)
    # create_norm_ngs_read_count_reg(dfname, df)

    ######################
    ### raw ngs histo ####
    ######################

    ### single ###

    # df = pd.concat(dfs, axis=0)
    # create_ngs_read_count_histo(dfname, df)

    ### multi ###

    # df = pd.concat(dfs, axis=0)
    # create_ngs_read_count_histo(dfname, df)



