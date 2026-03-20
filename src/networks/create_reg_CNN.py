import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from analysis.visuals import create_feature_residual_plot, create_feature_scatter_plot, create_multi_density_plot

from utils import reduce_rows, get_feature_modification_name, make_candidate_descriptor, capitalize_first
from utils import add_ikey, add_metadata_ikey, get_dataset_names, load_all_preprocessed, add_intersect_ngs_features, manage_separate_specifiers

from utils import SHOW_LEGEND, SHOW_TITLE
from utils import RESULTSPATH, DATASET_CUTOFF, SEED, TOP_N, K_MER_LENGTH, DATASET_STRAIN_DICT
from utils import COLORS, DECIMALS

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import joblib
import re

import scipy.sparse as sp
import shap

import random

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

### settings ###
RESULTSPATH, _ = os.path.split(RESULTSPATH)
DATAPATH = os.path.join(RESULTSPATH, "preprocess")
RESULTSPATH = os.path.join(RESULTSPATH, "reg_network")

KMER_SIZE = 3
KMER_STEP = 3   # lightweight

TRAIN_LOSS  = "mse"   # "huber" | "mse" | "mae"

MARKED = 1   # 0 | 1
DROP_X = 0   # 0 | 1
SHAP_CAT = 1   # 0 | 1
STRUCTURE = 0   # 0 | 1

MAX_NUMBER = 25000

BEST_MODEL = 0   # 0 | 1

# occlussion and SHAP
MAX_NUMBER_TRAIN = 500
MAX_NUMBER_TEST  = 500

# version
VERSION = "0"

###########
### CNN ###
###########

### load & save ###
selector = 'dataset'

folder = 'unpooled'
subfolder = f'motif_length_{K_MER_LENGTH}'

data = 'IBV'
strain = 'Yamagata'
segment = 'PB1'
intersects = 'mean_dataset_0'
motif_length = K_MER_LENGTH

dfnames = get_dataset_names(DATASET_CUTOFF, data)
dfs = load_all_preprocessed(dfnames, folder, subfolder)

dfs = manage_separate_specifiers(dfs, data, strain, segment)

### categorical features ###
categorical_cols = []

# only for multi segment selection!
# categorical_cols += ["segment"]

# only for multi strain/subtype/type selection! only use one! strain recommended (most distinct)
# categorical_cols += ["strain"]
# categorical_cols += ["subtype"]
# categorical_cols += ["type"]

# technical metadata (NOT recommended)
# categorical_cols += ['library_layout', 'library_selection', 'library_source']

# biological metadata (recommended)
categorical_cols += ['system_type', 'host']
# only available for unpooled data
categorical_cols += ['cell_system', 'localization', 'resolution']
categorical_cols += ['time_point', 'MOI']

# junction site motifs
categorical_cols += ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]

# use with datset intersects
# categorical_cols += ["dataset"]

### numerical features ###
numerical_cols = []

# length information (choose one group, because of redundant information)
numerical_cols += ["deletion_length", "5_end_length", "3_end_length"]
# second group
# numerical_cols += ["start", "end", "full_seq_length"]

### primary ###

# numerical_cols += ["direct_repeat_length"]
# numerical_cols += ["GC_content", "AU_content", "UpA_content", "CpG_content", "GC_skew"]
# numerical_cols += ["sequence_entropy", "kmer_richness"]
# numerical_cols += ["palindrome_density"]


### secondary ###

# numerical_cols += ["MFE"]
# numerical_cols += ["bp_count", "unpaired_count"]
# numerical_cols += ["bp_density", "unpaired_density"]
# numerical_cols += ["stem_count"]
# numerical_cols += ["hairpin_count"]

### extra features ###
extra_cols = [
    "dataset",
    "norm_log_NGS_read_count",
    "marked_DelVG_sequence",
    "DelVG_sequence",
]

if STRUCTURE:
    extra_cols += [
        "marked_structure",
        "structure",
    ]

### setup ###
if MARKED:
    key_sequence = "marked_DelVG_sequence"
    key_structure = "marked_structure"
else:
    key_sequence = "DelVG_sequence"
    key_structure = "structure"

NAME_MOD = f'{VERSION}_motif_length_{motif_length}'

save_path_model = os.path.join(RESULTSPATH, folder, 'CNN', data, strain, segment, intersects)
os.makedirs(save_path_model, exist_ok=True)

###################################
### group-safe split (70/10/20) ###
###################################

TEST_SIZE = 0.1
VAL_SIZE  = 0.2

def _add_split_ikey(df: pd.DataFrame, intersects: str) -> pd.DataFrame:
    '''
    create an ikey column for group-safe splitting, matching your intersect definition
    '''
    if 'metadata' in intersects:
        df = add_metadata_ikey(df)
    else:
        df = add_ikey(df)

    return df

def _split_df_by_ikey(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    '''
    returns: df_fit, df_val, df_test
    '''
    df = df.copy()

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df['ikey'].to_numpy()
    idx_train, idx_test = next(gss_test.split(df, groups=groups))

    df_train = df.iloc[idx_train].copy().reset_index(drop=True)
    df_test  = df.iloc[idx_test].copy().reset_index(drop=True)

    # val_size is fraction of full data; convert to fraction of remaining training pool
    val_size_rel = val_size / (1.0 - test_size)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=seed)
    groups_train = df_train['ikey'].to_numpy()
    idx_fit, idx_val = next(gss_val.split(df_train, groups=groups_train))

    df_fit = df_train.iloc[idx_fit].copy().reset_index(drop=True)
    df_val = df_train.iloc[idx_val].copy().reset_index(drop=True)

    return df_fit, df_val, df_test

dfs_fit = []
dfs_val = []
dfs_test = []

for df in dfs:
    df = _add_split_ikey(df, intersects)

    uniq = df['ikey'].nunique()
    if uniq < 3:
        dfs_fit.append(df.drop(columns=['ikey']).reset_index(drop=True))
        dfs_val.append(df.iloc[0:0].drop(columns=['ikey']).reset_index(drop=True))
        dfs_test.append(df.iloc[0:0].drop(columns=['ikey']).reset_index(drop=True))
        continue

    df_fit, df_val, df_test = _split_df_by_ikey(df, TEST_SIZE, VAL_SIZE, SEED)

    dfs_fit.append(df_fit.drop(columns=['ikey']).reset_index(drop=True))
    dfs_val.append(df_val.drop(columns=['ikey']).reset_index(drop=True))
    dfs_test.append(df_test.drop(columns=['ikey']).reset_index(drop=True))

############################################
### intersect + ngs features (fit/apply) ###
############################################

df_fit, norm_params = add_intersect_ngs_features(
    dfs=dfs_fit,
    intersects=intersects,
    norm_params=None,
    return_norm_params=True,
)

df_val = add_intersect_ngs_features(
    dfs=dfs_val,
    intersects=intersects,
    norm_params=norm_params,
    return_norm_params=False,
)

df_test = add_intersect_ngs_features(
    dfs=dfs_test,
    intersects=intersects,
    norm_params=norm_params,
    return_norm_params=False,
)

df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_fit = reduce_rows(df_fit, MAX_NUMBER)
df_val = reduce_rows(df_val, MAX_NUMBER)
df_test = reduce_rows(df_test, MAX_NUMBER)

##########################
### target + reporting ###
##########################

y_fit  = df_fit["norm_log_NGS_read_count"].values
y_val  = df_val["norm_log_NGS_read_count"].values
y_test = df_test["norm_log_NGS_read_count"].values

#############################
### k-mer token (no leak) ###
#############################

def kmer_tokenize(seq, k, step=KMER_STEP):
    if DROP_X:
        return [
            seq[i:i + k]
            for i in range(0, len(seq) - k + 1, step)
            if 'X' not in seq[i:i + k]
        ]
    else:
        return [
            seq[i:i + k]
            for i in range(0, len(seq) - k + 1, step)
        ]

vocab = set()

df_fit["kmer_seq"] = df_fit[key_sequence].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
df_fit["kmer_seq"].apply(lambda kmers: vocab.update(kmers))

vocab = sorted(vocab)

UNK_IDX = 1
token2idx = {k: i + 2 for i, k in enumerate(vocab)}   # start at 2

def encode_seq(kmers):
    return [token2idx.get(k, UNK_IDX) for k in kmers]

df_fit["encoded_seq"]  = df_fit["kmer_seq"].apply(encode_seq)

df_val["kmer_seq"] = df_val[key_sequence].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
df_val["encoded_seq"] = df_val["kmer_seq"].apply(encode_seq)

df_test["kmer_seq"] = df_test[key_sequence].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
df_test["encoded_seq"] = df_test["kmer_seq"].apply(encode_seq)

maxlen = df_fit["encoded_seq"].apply(len).max() if len(df_fit) > 0 else 1

X_seq_fit  = tf.keras.preprocessing.sequence.pad_sequences(df_fit["encoded_seq"],  maxlen=maxlen, padding="post", truncating="post").astype(np.int32)
X_seq_val  = tf.keras.preprocessing.sequence.pad_sequences(df_val["encoded_seq"],  maxlen=maxlen, padding="post", truncating="post").astype(np.int32)
X_seq_test = tf.keras.preprocessing.sequence.pad_sequences(df_test["encoded_seq"], maxlen=maxlen, padding="post", truncating="post").astype(np.int32)

### secondary structure (OPTIONAL, no leak) ###
if STRUCTURE:
    vocab2 = set()

    df_fit["kmer_str"] = df_fit[key_structure].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
    df_fit["kmer_str"].apply(lambda kmers: vocab2.update(kmers))

    vocab2 = sorted(vocab2)
    token2idx2 = {k: i + 1 for i, k in enumerate(vocab2)}

    def encode_str(kmers):
        return [token2idx2[k] for k in kmers if k in token2idx2]

    df_fit["encoded_str"] = df_fit["kmer_str"].apply(encode_str)

    df_val["kmer_str"] = df_val[key_structure].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
    df_val["encoded_str"] = df_val["kmer_str"].apply(encode_str)

    df_test["kmer_str"] = df_test[key_structure].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
    df_test["encoded_str"] = df_test["kmer_str"].apply(encode_str)

    maxlen2 = df_fit["encoded_str"].apply(len).max() if len(df_fit) > 0 else 1

    X_str_fit  = tf.keras.preprocessing.sequence.pad_sequences(df_fit["encoded_str"],  maxlen=maxlen2).astype(np.int32)
    X_str_val  = tf.keras.preprocessing.sequence.pad_sequences(df_val["encoded_str"],  maxlen=maxlen2).astype(np.int32)
    X_str_test = tf.keras.preprocessing.sequence.pad_sequences(df_test["encoded_str"], maxlen=maxlen2).astype(np.int32)
else:
    vocab2 = []
    token2idx2 = {}
    maxlen2 = 0
    X_str_fit = None
    X_str_val = None
    X_str_test = None

###########################################
### tabular preprocessing (no leak) #######
###########################################

n_fit  = len(df_fit)
n_val  = len(df_val)
n_test = len(df_test)

# categorical (robust to empty)
if len(categorical_cols) == 0:
    encoder = None
    X_cat_fit  = np.zeros((n_fit, 0), dtype=np.float32)
    X_cat_val  = np.zeros((n_val, 0), dtype=np.float32)
    X_cat_test = np.zeros((n_test, 0), dtype=np.float32)
else:
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    X_cat_fit  = encoder.fit_transform(df_fit[categorical_cols]).astype(np.float32)
    X_cat_val  = encoder.transform(df_val[categorical_cols]).astype(np.float32)
    X_cat_test = encoder.transform(df_test[categorical_cols]).astype(np.float32)

# numerical (robust to empty)
if len(numerical_cols) == 0:
    scaler = None
    X_num_fit  = np.zeros((n_fit, 0), dtype=np.float32)
    X_num_val  = np.zeros((n_val, 0), dtype=np.float32)
    X_num_test = np.zeros((n_test, 0), dtype=np.float32)
else:
    scaler = StandardScaler()
    X_num_fit  = scaler.fit_transform(df_fit[numerical_cols]).astype(np.float32)
    X_num_val  = scaler.transform(df_val[numerical_cols]).astype(np.float32)
    X_num_test = scaler.transform(df_test[numerical_cols]).astype(np.float32)

X_other_fit  = np.hstack([X_cat_fit,  X_num_fit]).astype(np.float32)
X_other_val  = np.hstack([X_cat_val,  X_num_val]).astype(np.float32)
X_other_test = np.hstack([X_cat_test, X_num_test]).astype(np.float32)

HAS_TABULAR = (X_other_fit.shape[1] > 0)

X_cat = X_cat_fit

###################################
### Huber delta (fit, no leak) ###
###################################

y_fit_arr = np.asarray(y_fit, dtype=float)

med = float(np.nanmedian(y_fit_arr))
mad = float(np.nanmedian(np.abs(y_fit_arr - med)))

# robust std estimate from MAD
HUBER_DELTA = 1.4826 * mad

# fallback if MAD is 0 or nan
if (not np.isfinite(HUBER_DELTA)) or (HUBER_DELTA <= 0):
    HUBER_DELTA = float(np.nanstd(y_fit_arr))

# keep delta in a reasonable range (avoid too-MSE-like or too-MAE-like)
HUBER_DELTA = float(np.clip(HUBER_DELTA, 0.5, 2.0))
HUBER_DELTA = round(HUBER_DELTA, DECIMALS)

print("\nHuber delta (fit, data-driven)")
print(f"HUBER_DELTA: {HUBER_DELTA}")

#############################
# unified input packing/pred
#############################
def pack_inputs(X_seq, X_other=None, STRUCTURE=False, X_str=None, HAS_TABULAR=True):
    if STRUCTURE:
        if HAS_TABULAR:
            return [X_seq, X_str, X_other]
        return [X_seq, X_str]
    else:
        if HAS_TABULAR:
            return [X_seq, X_other]
        return X_seq

def predict_value(model, X_seq, X_other=None, STRUCTURE=False, X_str=None, HAS_TABULAR=True, verbose=0):
    inp = pack_inputs(X_seq, X_other=X_other, STRUCTURE=STRUCTURE, X_str=X_str, HAS_TABULAR=HAS_TABULAR)
    return model.predict(inp, verbose=verbose).ravel()

#############
### model ###
#############

# lightweight dims
EMB_DIM = 16
CONV_FILTERS = 32
DENSE_UNITS = 32

seq_input = layers.Input(shape=(maxlen,), dtype="int32")

# (2) embedding: tell Keras that 0 is padding
emb = layers.Embedding(
    input_dim=len(token2idx) + 2,
    output_dim=EMB_DIM,
    mask_zero=True
)(seq_input)

# (3) conv: keep same length so we can mask padded positions correctly
x = layers.Conv1D(
    CONV_FILTERS,
    5,
    activation="relu",
    padding="same"
)(emb)

# explicit mask from token ids: 1 for real tokens, 0 for padding
mask = tf.cast(tf.not_equal(seq_input, 0), tf.float32)  # shape (B, L)
mask = tf.expand_dims(mask, axis=-1)                    # shape (B, L, 1)

# zero-out conv activations at padded positions
x = layers.Multiply()([x, mask])

# you can keep MaxPooling if you want, but apply it AFTER masking
x = layers.MaxPooling1D(2)(x)

# global pooling
x = layers.GlobalMaxPooling1D()(x)

# optional tabular input
other_input = None
if HAS_TABULAR:
    other_input = layers.Input(shape=(X_other_fit.shape[1],), dtype="float32")

if STRUCTURE:
    str_input = layers.Input(shape=(maxlen2,), dtype="int32")

    emb2 = layers.Embedding(
        input_dim=len(token2idx2) + 1,
        output_dim=EMB_DIM,
        mask_zero=True
    )(str_input)

    x2 = layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same")(emb2)

    mask2 = tf.cast(tf.not_equal(str_input, 0), tf.float32)
    mask2 = tf.expand_dims(mask2, axis=-1)

    x2 = layers.Multiply()([x2, mask2])

    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)

    if HAS_TABULAR:
        concat = layers.concatenate([x, x2, other_input])
        inputs = [seq_input, str_input, other_input]
    else:
        concat = layers.concatenate([x, x2])
        inputs = [seq_input, str_input]
else:
    if HAS_TABULAR:
        concat = layers.concatenate([x, other_input])
        inputs = [seq_input, other_input]
    else:
        concat = x
        inputs = [seq_input]

dense = layers.Dense(
    DENSE_UNITS,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(concat)
dense = layers.Dropout(0.1)(dense)
output = layers.Dense(1, activation='linear')(dense)

model = models.Model(inputs=inputs, outputs=output)

# choose training objective
if TRAIN_LOSS == "huber":
    loss_fn = tf.keras.losses.Huber(delta=HUBER_DELTA)
elif TRAIN_LOSS == "mse":
    loss_fn = "mse"
elif TRAIN_LOSS == "mae":
    loss_fn = "mae"
else:
    raise ValueError('TRAIN_LOSS must be "huber", "mse", or "mae"')

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=[],
    weighted_metrics=[
        'mae',
        'mse',
        tf.keras.losses.Huber(delta=HUBER_DELTA, name="huber")
    ]
)

### callbacks ###
best_model_path = os.path.join(save_path_model, f"{NAME_MOD}_best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path, save_best_only=True, monitor="val_loss", mode="min"
)

early_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=6,
    restore_best_weights=True
)

### sample weights (disabled for diagnostics) ###
w_fit = np.ones(len(df_fit), dtype=np.float32)
w_val = np.ones(len(df_val), dtype=np.float32)

### train ###
history = model.fit(
    pack_inputs(
        X_seq_fit,
        X_other=X_other_fit,
        STRUCTURE=bool(STRUCTURE),
        X_str=(X_str_fit if STRUCTURE else None),
        HAS_TABULAR=HAS_TABULAR
    ),
    y_fit,
    sample_weight=w_fit,
    validation_data=(
        pack_inputs(
            X_seq_val,
            X_other=X_other_val,
            STRUCTURE=bool(STRUCTURE),
            X_str=(X_str_val if STRUCTURE else None),
            HAS_TABULAR=HAS_TABULAR
        ),
        y_val,
        w_val
    ),
    epochs=25,
    batch_size=16,
    callbacks=[checkpoint_cb, early_cb],
)

############
### save ###
############

final_model_path = os.path.join(save_path_model, f"{NAME_MOD}_final_model.h5")
model.save(final_model_path)
print(f"\n✔ models saved")

scaler_path = os.path.join(save_path_model, f"{NAME_MOD}_scaler.save")
joblib.dump(scaler, scaler_path)
print(f"✔ scaler saved")

### artifacts ###
preproc_path = os.path.join(save_path_model, f"{NAME_MOD}_preproc.joblib")
joblib.dump({
    "encoder": encoder,
    "scaler": scaler,
    "token2idx": token2idx,
    "token2idx2": token2idx2,
    "maxlen": int(maxlen),
    "maxlen2": int(maxlen2),
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "STRUCTURE": int(STRUCTURE),
    "TRAIN_LOSS": str(TRAIN_LOSS),
    "HUBER_DELTA": float(HUBER_DELTA),
    "HAS_TABULAR": int(HAS_TABULAR),
}, preproc_path)

print("✔ preprocessing artifacts saved")

#####################################
### keep compatibility for visuals ###
#####################################

df_train = df_fit.copy().reset_index(drop=True)
df_test  = df_test.copy().reset_index(drop=True)

df = pd.concat([df_train, df_test], ignore_index=True)

X_seq_train   = X_seq_fit
X_seq_test    = X_seq_test
X_other_train = X_other_fit
X_other_test  = X_other_test
y_train = y_fit
y_test  = y_test

if STRUCTURE:
    X_str_train = X_str_fit
    X_str_test  = X_str_test

###############
### visuals ###
###############

### numbers ###
n_total = len(df)
n_train = len(df_train)
n_val = int(np.floor(0.1 * n_train))
n_fit = n_train - n_val
n_test = len(df_test)

save_path_vis = os.path.join(RESULTSPATH, folder, 'visuals', data, strain, segment, intersects)
os.makedirs(save_path_vis, exist_ok=True)

if BEST_MODEL:
    model_path = best_model_path
else:
    model_path = final_model_path

plt.style.use("seaborn")
plt.rc("font", size=12)

plt.rcParams.update({
"font.size": 18,
"axes.titlesize": 18,
"axes.labelsize": 18,
"xtick.labelsize": 18,
"ytick.labelsize": 18,
"legend.fontsize": 18,
"legend.title_fontsize": 18,
})

# last epoch values (absolute numbers for titles)
mse_last_tr = float(history.history["mse"][-1])
mse_last_va = float(history.history["val_mse"][-1])

mae_last_tr = float(history.history["mae"][-1])
mae_last_va = float(history.history["val_mae"][-1])

hub_last_tr = float(history.history["huber"][-1])
hub_last_va = float(history.history["val_huber"][-1])

print(f"""
Final Training Metrics:
  MSE   : {mse_last_tr:.6f}
  MAE   : {mae_last_tr:.6f}

Final Validation Metrics:
  MSE   : {mse_last_va:.6f}
  MAE   : {mae_last_va:.6f}
""")

### MSE curve (train, val) ###
title_name = 'reg. CNN: MSE lerarning curve (training, validation) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={n_train}, n_val={n_val})'

epochs = np.arange(1, len(history.history["mse"]) + 1)

plt.figure(figsize=(10, 6))
plt.plot(
    epochs,
    history.history["mse"],
    label="training",
    linewidth=2,
    color=COLORS[8]
)
plt.plot(
    epochs,
    history.history["val_mse"],
    label="validation",
    linewidth=2,
    color=COLORS[0]
)

plt.xlabel("Epoch (-)")
plt.ylabel("MSE  (-)")

plt.ylim(0, 1)
plt.xlim(0, 25)

if SHOW_TITLE:
    plt.title(title_name)
else:
    plt.title("")

if SHOW_LEGEND:
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

plt.grid(True)

mse_curve_path = os.path.join(save_path_vis, f"{NAME_MOD}_training_mse.png")
plt.savefig(mse_curve_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ training MSE curve saved")

### MAE curve (train, val) ###
title_name = 'reg. CNN: MAE across epochs (training, validation) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={n_train}, n_val={n_val})'

epochs = np.arange(1, len(history.history["mae"]) + 1)

plt.figure(figsize=(10, 6))
plt.plot(
    epochs,
    history.history["mae"],
    label="training",
    linewidth=2,
    color=COLORS[8]
)
plt.plot(
    epochs,
    history.history["val_mae"],
    label="validation",
    linewidth=2,
    color=COLORS[0]
)

plt.xlabel("Epoch (-)")
plt.ylabel("MAE (-)")

plt.ylim(0, 1)
plt.xlim(0, 25)

if SHOW_TITLE:
    plt.title(title_name)
else:
    plt.title("")

if SHOW_LEGEND:
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

plt.grid(True)

mae_curve_path = os.path.join(save_path_vis, f"{NAME_MOD}_training_mae.png")
plt.savefig(mae_curve_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ training MAE curve saved")

### Huber curve (train, val) ###
title_name = f'reg. CNN: Huber (delta={HUBER_DELTA:g}) across epochs (training, validation) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={n_train}, n_val={n_val})'

epochs = np.arange(1, len(history.history["huber"]) + 1)

plt.figure(figsize=(10, 6))
plt.plot(
    epochs,
    history.history["huber"],
    label="training",
    linewidth=2,
    color=COLORS[8]
)
plt.plot(
    epochs,
    history.history["val_huber"],
    label="validation",
    linewidth=2,
    color=COLORS[0]
)

plt.xlabel("Epoch (-)")
plt.ylabel("Huber (-)")

plt.ylim(0, 1)
plt.xlim(0, 25)

if SHOW_TITLE:
    plt.title(title_name)
else:
    plt.title("")

if SHOW_LEGEND:
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

plt.grid(True)

huber_curve_path = os.path.join(save_path_vis, f"{NAME_MOD}_training_huber.png")
plt.savefig(huber_curve_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ training Huber curve saved")

model.load_weights(model_path)

# build dfs #
y_pred_val_train = predict_value(
    model,
    X_seq_train,
    X_other=X_other_train,
    STRUCTURE=bool(STRUCTURE),
    X_str=(X_str_train if STRUCTURE else None),
    HAS_TABULAR=HAS_TABULAR
)

y_pred_val_test = predict_value(
    model,
    X_seq_test,
    X_other=X_other_test,
    STRUCTURE=bool(STRUCTURE),
    X_str=(X_str_test if STRUCTURE else None),
    HAS_TABULAR=HAS_TABULAR
)

df_train_val = df_train.copy().reset_index(drop=True)
df_train_val["y_true"] = y_train
df_train_val["y_val"] = y_pred_val_train

df_test_val = df_test.copy().reset_index(drop=True)
df_test_val["y_true"] = y_test
df_test_val["y_val"] = y_pred_val_test

modification = get_feature_modification_name()

### R2 & residual (test)  ###
create_feature_residual_plot(
    plot_name="reg. CNN: residual as a function of true NGS read count (testing) - scatter plot",
    df=df_test_val,
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    y_pred_feature_name="y_val",
    y_axis_name="Residual (prediction - true value)",
    selector=selector,
    show_zero_line=True,
    reg_metrics=True,
    fname=f"{NAME_MOD}_residual",
    path="reg_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects
)

print(f"\n✔ residual plot saved")

### scatter plot (test)  ###
create_feature_scatter_plot(
    plot_name="reg. CNN: prediction as a function of true NGS read count (testing) - scatter plot",
    df=df_test_val,
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    y_feature_name="y_val",
    y_axis_name=f"Predicted value (-)",
    selector=selector,
    show_rolling_median=False,
    rolling_window=50,
    show_identity_line=True,
    pseudo_prefix="",
    show_decision_threshold=0.0,
    show_split_threshold=0.0,
    reg_metrics=False,
    huber_delta=0.0,
    fname=f"{NAME_MOD}_prediction",
    path="reg_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
)

print("✔ prediction scatter plot saved")

### density (train, test) ###
create_multi_density_plot(
    plot_name="NGS read count distribution via KDE (testing) - density plot",
    df_list=[df_train, df_test],
    df_names=["train", "test"],
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    fname=f"{NAME_MOD}_density",
    path="reg_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
    show_js=False,
    show_n_in_legend=False,
)

print("✔ train/test density plot saved")

### occlusion importances (test) ###
rng = np.random.default_rng(SEED)
eval_n = min(MAX_NUMBER_TEST, X_seq_test.shape[0])
idx_eval = rng.choice(X_seq_test.shape[0], size=eval_n, replace=False)

X_seq_eval = X_seq_test[idx_eval]
X_other_eval = X_other_test[idx_eval]
y_eval = y_test[idx_eval]
if STRUCTURE:
    X_str_eval = X_str_test[idx_eval]
else:
    X_str_eval = None

def cnn_predict_value(model, X_seq, X_other, STRUCTURE=False, X_str=None):
    return predict_value(
        model,
        X_seq,
        X_other=X_other,
        STRUCTURE=STRUCTURE,
        X_str=X_str,
        HAS_TABULAR=HAS_TABULAR,
        verbose=0
    )

def safe_mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0:
        return np.nan
    return float(np.mean((y_pred - y_true) ** 2))

def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred - y_true)))

def safe_huber(y_true, y_pred, delta=HUBER_DELTA):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0:
        return np.nan

    err = np.abs(y_pred - y_true)
    quad = np.minimum(err, delta)
    lin = err - quad

    return float(np.mean(0.5 * quad ** 2 + delta * lin))

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)

def cnn_test_metrics(model, X_seq_test, X_other_test, y_test, STRUCTURE=False, X_str_test=None):
    y_pred = cnn_predict_value(model, X_seq_test, X_other_test, STRUCTURE=STRUCTURE, X_str=X_str_test)
    mse = safe_mse(y_test, y_pred)
    mae = safe_mae(y_test, y_pred)
    hub = safe_huber(y_test, y_pred, delta=HUBER_DELTA)
    r2  = safe_r2(y_test, y_pred)
    return mse, mae, hub, r2

# ----------------------------
# TEST metrics (prediction-based)
# ----------------------------
y_pred_test_report = predict_value(
    model,
    X_seq_test,
    X_other=X_other_test,
    STRUCTURE=bool(STRUCTURE),
    X_str=(X_str_test if STRUCTURE else None),
    HAS_TABULAR=HAS_TABULAR
)

mse_test  = safe_mse(y_test, y_pred_test_report)
mae_test  = safe_mae(y_test, y_pred_test_report)
hub_test  = safe_huber(y_test, y_pred_test_report, delta=HUBER_DELTA)
r2_test   = safe_r2(y_test, y_pred_test_report)

print(f"""
TEST metrics (selected weights):
  MAE   : {mae_test:.6f}
  MSE   : {mse_test:.6f}
  R²    : {r2_test:.6f}
""")

# -----------------------------------------------------------
# X_other feature occlusion ONLY if tabular features exist
# -----------------------------------------------------------
if not HAS_TABULAR:
    print("Skipping X_other occlusion importance (no categorical/numerical features selected).")
else:
    try:
        cat_names_full = encoder.get_feature_names_out(categorical_cols)
    except AttributeError:
        cat_names_full = encoder.get_feature_names(categorical_cols)
    feature_names_other = list(cat_names_full) + numerical_cols

    base_mse, base_mae, base_hub, base_r2 = cnn_test_metrics(
        model,
        X_seq_eval, X_other_eval, y_eval,
        STRUCTURE=bool(STRUCTURE),
        X_str_test=(X_str_eval if STRUCTURE else None),
    )

    def importance_x_other_mse_increase(
        model,
        X_seq_test, X_other_test, y_test,
        feature_names,
        STRUCTURE=False, X_str_test=None,
        mode="permute",
        n_repeats=5,
        seed=SEED
    ):
        rng = np.random.default_rng(seed)
        base_mse, base_mae, base_hub, base_r2 = cnn_test_metrics(
            model, X_seq_test, X_other_test, y_test,
            STRUCTURE=STRUCTURE, X_str_test=X_str_test
        )

        rows = []
        for j, fname in enumerate(feature_names):
            if mode == "zero":
                X_mod = X_other_test.copy()
                X_mod[:, j] = 0.0

                mse_mod, mae_mod, hub_mod, r2_mod = cnn_test_metrics(
                    model, X_seq_test, X_mod, y_test,
                    STRUCTURE=STRUCTURE, X_str_test=X_str_test
                )
                inc = mse_mod - base_mse

            elif mode == "permute":
                incs = []
                for _ in range(n_repeats):
                    X_mod = X_other_test.copy()
                    perm = rng.permutation(X_mod.shape[0])
                    X_mod[:, j] = X_mod[perm, j]

                    mse_mod, mae_mod, hub_mod, r2_mod = cnn_test_metrics(
                        model, X_seq_test, X_mod, y_test,
                        STRUCTURE=STRUCTURE, X_str_test=X_str_test
                    )
                    incs.append(mse_mod - base_mse)

                inc = float(np.nanmean(incs))
            else:
                raise ValueError("mode must be 'permute' or 'zero'")

            rows.append((fname, inc))

        imp_df = (
            pd.DataFrame(rows, columns=["feature", "mse_increase"])
              .sort_values("mse_increase", ascending=False)
        )
        return (base_mse, base_mae, base_hub, base_r2), imp_df

    # permutation importance (on eval subset)
    (base_mse_perm, base_mae_perm, base_hub_perm, base_r2_perm), imp_perm = importance_x_other_mse_increase(
        model,
        X_seq_eval, X_other_eval, y_eval,
        feature_names_other,
        STRUCTURE=bool(STRUCTURE),
        X_str_test=(X_str_eval if STRUCTURE else None),
        mode="permute",
        n_repeats=5,
        seed=SEED
    )

    # zero occlusion importance (on eval subset)
    (base_mse_zero, base_mae_zero, base_hub_zero, base_r2_zero), imp_zero = importance_x_other_mse_increase(
        model,
        X_seq_eval, X_other_eval, y_eval,
        feature_names_other,
        STRUCTURE=bool(STRUCTURE),
        X_str_test=(X_str_eval if STRUCTURE else None),
        mode="zero",
    )

    # plot permutation (test subset)
    plot_df = imp_perm.head(TOP_N).iloc[::-1].copy()
    plot_df["feature"] = plot_df["feature"].apply(capitalize_first)
    title_name = f'reg. CNN: feature importance via permuation occlusion (testing) - bar plot'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={eval_n})'
    title_name += f'\nMAE={base_mae_perm:.{DECIMALS}f}, MSE={base_mse_perm:.{DECIMALS}f}, R²={base_r2_perm:.{DECIMALS}f}'

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["mse_increase"], color=COLORS[6], edgecolor="white", linewidth=1)
    plt.xlabel("MSE increase (when permuted)")
    plt.ylabel(f"Feature (top {TOP_N})")

    plt.xlim(0, 0.1)

    if SHOW_TITLE:
        plt.title(title_name)
    else:
        plt.title("")

    plt.tight_layout()
    plt.grid(True, axis="x")

    perm_path = os.path.join(save_path_vis, f"{NAME_MOD}_permutation.png")
    plt.savefig(perm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ permutation importance saved")

    # plot zero-occlusion (test subset)
    plot_df = imp_zero.head(TOP_N).iloc[::-1].copy()
    plot_df["feature"] = plot_df["feature"].apply(capitalize_first)
    title_name = f'reg. CNN: feature importance via zero occlusion (testing) - bar plot'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n={eval_n})'
    title_name += f'\nMAE={base_mae_zero:.{DECIMALS}f}, MSE={base_mse_zero:.{DECIMALS}f}, R²={base_r2_zero:.{DECIMALS}f}'

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["mse_increase"], color=COLORS[6], edgecolor="white", linewidth=1)
    plt.xlabel("MSE increase (when set to 0)")
    plt.ylabel(f"Feature (top {TOP_N})")

    plt.xlim(0, 0.1)

    if SHOW_TITLE:
        plt.title(title_name)
    else:
        plt.title("")

    plt.tight_layout()
    plt.grid(True, axis="x")

    occ_path = os.path.join(save_path_vis, f"{NAME_MOD}_occlusion.png")
    plt.savefig(occ_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ occlusion importance saved")

SEQ_OCC_WINDOW = 10
SEQ_OCC_STRIDE = 5

def sequence_window_occlusion_mse(
    model,
    X_seq_test, X_other_test, y_test,
    window=10, stride=5,
    STRUCTURE=False, X_str_test=None,
):
    base_mse, base_mae, base_hub, base_r2 = cnn_test_metrics(
        model, X_seq_test, X_other_test, y_test,
        STRUCTURE=STRUCTURE, X_str_test=X_str_test
    )

    L = X_seq_test.shape[1]
    window = int(min(window, L)) if L > 0 else 1
    stride = int(max(1, min(stride, window)))

    starts = list(range(0, max(1, L - window + 1), stride))
    if starts and starts[-1] != L - window:
        starts.append(max(0, L - window))

    rows = []
    for s in starts:
        e = min(L, s + window)
        X_mod = X_seq_test.copy()
        X_mod[:, s:e] = 0

        mse_mod, mae_mod, hub_mod, r2_mod = cnn_test_metrics(
            model,
            X_mod, X_other_test, y_test,
            STRUCTURE=STRUCTURE,
            X_str_test=X_str_test
        )
        rows.append((s, e, mse_mod - base_mse))

    return (base_mse, base_mae, base_hub, base_r2), pd.DataFrame(rows, columns=["start_tok", "end_tok", "mse_increase"])


(base_mse_seq, base_mae_seq, base_hub_seq, base_r2_seq), seq_imp_df = sequence_window_occlusion_mse(
    model,
    X_seq_eval, X_other_eval, y_eval,
    window=SEQ_OCC_WINDOW,
    stride=SEQ_OCC_STRIDE,
    STRUCTURE=bool(STRUCTURE),
    X_str_test=(X_str_eval if STRUCTURE else None),
)

# curve plot (MSE increase vs window center)
x_center = (seq_imp_df["start_tok"] + seq_imp_df["end_tok"]) / 2.0
y_inc = seq_imp_df["mse_increase"].to_numpy()

title_name = f'reg. CNN: sequence importance (window={SEQ_OCC_WINDOW}, stride={SEQ_OCC_STRIDE}) via window occlusion (testing) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n={eval_n})'
title_name += f'\nMAE={base_mae_seq:.{DECIMALS}f}, MSE={base_mse_seq:.{DECIMALS}f}, R²={base_r2_seq:.{DECIMALS}f}'

df_eval_meta = df_test.iloc[idx_eval].copy()

start_mea_tok = np.nan
end_mea_tok = np.nan
start_med_tok = np.nan
end_med_tok = np.nan

start_mea_tok = float(np.nanmean(df_eval_meta["start"].astype(float).to_numpy())) // KMER_STEP
start_med_tok = float(np.nanmedian(df_eval_meta["start"].astype(float).to_numpy())) // KMER_STEP

if MARKED:
    end_mea_tok = float(np.nanmean(df_eval_meta["end"].astype(float).to_numpy()))   // KMER_STEP
    end_med_tok = float(np.nanmedian(df_eval_meta["end"].astype(float).to_numpy()))   // KMER_STEP
    start_mea_label = f"Deletion start mean ({int(start_mea_tok)})"
    end_mea_label = f"Deletion end mean ({int(end_mea_tok)})"
    start_med_label = f"Deletion start median ({int(start_med_tok)})"
    end_med_label = f"Deletion end median ({int(end_med_tok)})"
else:
    end_mea_tok = float(np.nanmean(df_eval_meta["5_end_length"].astype(float).to_numpy() + df_eval_meta["3_end_length"].astype(float).to_numpy()))   // KMER_STEP
    end_med_tok = float(np.nanmedian(df_eval_meta["5_end_length"].astype(float).to_numpy() + df_eval_meta["3_end_length"].astype(float).to_numpy()))   // KMER_STEP
    start_mea_label = f"Junction site mean ({int(start_mea_tok)})"
    end_mea_label = f"Padding start mean ({int(end_mea_tok)})"
    start_med_label = f"Junction site mean ({int(start_med_tok)})"
    end_med_label = f"Padding start mean ({int(end_med_tok)})"

print("\nToken positions:")
print(f"Start mean token:   {int(start_mea_tok)}")
print(f"Start median token: {int(start_med_tok)}")
print(f"End mean token:     {int(end_mea_tok)}")
print(f"End median token:   {int(end_med_tok)}")

plt.figure(figsize=(10, 6))
plt.plot(x_center, y_inc, linewidth=2, color=COLORS[6], label="test")

if np.isfinite(start_med_tok):
    plt.axvline(
        start_med_tok,
        color="grey",
        linestyle=":",
        linewidth=2,
        label=start_med_label,
    )

if np.isfinite(end_med_tok):
    plt.axvline(
        end_med_tok,
        color="grey",
        linestyle=":",
        linewidth=2,
        label=end_med_label
    )

if np.isfinite(start_mea_tok):
    plt.axvline(
        start_mea_tok,
        color="grey",
        linestyle="--",
        linewidth=2,
        label=start_mea_label,
    )

if np.isfinite(end_mea_tok):
    plt.axvline(
        end_mea_tok,
        color="grey",
        linestyle="--",
        linewidth=2,
        label=end_mea_label
    )

plt.xlabel(f"K-mer position (window center)")
plt.ylabel("MSE increase (when window masked)")

plt.xlim(0, 800)
plt.ylim(-0.05, 0.05)

if SHOW_TITLE:
    plt.title(title_name)
else:
    plt.title("")

plt.tight_layout()
plt.grid(True)

if SHOW_LEGEND:
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='types',
        frameon=True
    )

seq_occ_path = os.path.join(save_path_vis, f"{NAME_MOD}_window_occlusion_curve.png")
plt.savefig(seq_occ_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ sequence window occlusion curve saved")

# top windows bar plot
top_windows = seq_imp_df.sort_values("mse_increase", ascending=False).head(TOP_N).copy()
top_windows["window"] = top_windows.apply(lambda r: f'{int(r.start_tok)}-{int(r.end_tok)}', axis=1)
top_windows = top_windows.sort_values("mse_increase", ascending=True)

title_name = f'reg. CNN: sequence importance (window={SEQ_OCC_WINDOW}, stride={SEQ_OCC_STRIDE}) via window occlusion (testing) - bar plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n={eval_n})'
title_name += f'\nMAE={base_mae_seq:.{DECIMALS}f}, MSE={base_mse_seq:.{DECIMALS}f}, R²={base_r2_seq:.{DECIMALS}f}'

plt.figure(figsize=(10, 6))
plt.barh(top_windows["window"], top_windows["mse_increase"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("MSE increase (when window masked)")
plt.ylabel(f"K-mer window (top {TOP_N})")

plt.xlim(0, 0.1)

if SHOW_TITLE:
    plt.title(title_name)
else:
    plt.title("")

plt.tight_layout()
plt.grid(True, axis="x")

seq_top_path = os.path.join(save_path_vis, f"{NAME_MOD}_window_occlusion_bar.png")
plt.savefig(seq_top_path, dpi=300, bbox_inches="tight")
plt.close()
print("✔ sequence window occlusion bar saved")

### SHAP (test) ###
if (not HAS_TABULAR) or (X_other_train.shape[1] == 0):
    print("Skipping SHAP (no tabular features selected).")
else:
    if SHAP_CAT:
        X_train_shap = X_other_train
        X_test_shap  = X_other_test
        try:
            cat_names = encoder.get_feature_names_out(categorical_cols)
        except AttributeError:
            cat_names = encoder.get_feature_names(categorical_cols)
        feature_names = list(cat_names) + numerical_cols
    else:
        n_cat_features = X_cat.shape[1]
        X_train_shap = X_other_train[:, n_cat_features:]
        X_test_shap  = X_other_test[:,  n_cat_features:]
        feature_names = numerical_cols

    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X_train_shap, y_train)

    assert X_train_shap.shape[1] == X_test_shap.shape[1], \
        f"train/test mismatch: {X_train_shap.shape[1]} vs {X_test_shap.shape[1]}"
    assert getattr(rf, "n_features_in_", X_test_shap.shape[1]) == X_test_shap.shape[1], \
        "model expects a different number of features than provided X_other_test."

    bg_n = min(MAX_NUMBER_TRAIN, X_train_shap.shape[0])
    eval_n_shap = min(MAX_NUMBER_TEST, X_test_shap.shape[0])

    X_bg = X_train_shap[:bg_n]

    rng = np.random.default_rng(SEED)
    idx = rng.choice(X_test_shap.shape[0], size=eval_n_shap, replace=False)
    X_eval = X_test_shap[idx]

    if sp.issparse(X_bg):
        X_bg = X_bg.toarray()
    if sp.issparse(X_eval):
        X_eval = X_eval.toarray()

    explainer = shap.TreeExplainer(
        rf,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="raw"
    )

    shap_values = explainer.shap_values(X_eval, check_additivity=False)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
          .sort_values("importance", ascending=False)
          .head(TOP_N)
          .copy()
    )

    importance_df["feature"] = importance_df["feature"].apply(capitalize_first)

    bar_color = COLORS[6]

    title_name = f'reg. CNN: SHAP feature importance via RandomForest (testing) - bar plot'
    title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
    title_name += f' (n_train={bg_n}, n_test={eval_n_shap})'

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color=bar_color, edgecolor='white', linewidth=1)
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel(f"Feature (top {TOP_N})")

    plt.xlim(0, 0.1)

    if SHOW_TITLE:
        plt.title(title_name)
    else:
        plt.title("")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.grid(True, axis="x")

    shap_path = os.path.join(save_path_vis, f"{NAME_MOD}_shap.png")
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ SHAP bar chart saved")