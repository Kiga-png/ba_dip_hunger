import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from analysis.visuals import create_multi_density_plot, create_feature_roc_auc_plot, create_feature_pr_auc_plot, create_prediction_probability_density_plot, create_feature_scatter_plot

from analysis.visuals import POOLED_THRESHOLD, UNPOOLED_THRESHOLD

from utils import balance_by_threshold, reduce_rows, get_feature_modification_name
from utils import add_ikey, add_metadata_ikey, get_dataset_names, load_all_preprocessed, add_intersect_ngs_features, manage_separate_specifiers
from utils import make_candidate_descriptor

from utils import RESULTSPATH, DATASET_CUTOFF, SEED, TOP_N, K_MER_LENGTH, DATASET_STRAIN_DICT
from utils import COLORS, DECIMALS

from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import random
import joblib
import re

import scipy.sparse as sp
import shap

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

### settings ###
RESULTSPATH, _ = os.path.split(RESULTSPATH)
DATAPATH = os.path.join(RESULTSPATH, "preprocess")
RESULTSPATH = os.path.join(RESULTSPATH, "bin_network")

VERSION = "0"
KMER_SIZE = 3
KMER_STEP = 3   # lightweight

BALANCE = 0   # 0 | 1

MARKED = 1   # 0 | 1
DROP_X = 0   # 0 | 1
SHAP_CAT = 1   # 0 | 1
STRUCTURE = 0   # 0 | 1

MAX_NUMBER = 25000

BEST_MODEL = 0   # 0 | 1

# occlussion and SHAP
MAX_NUMBER_TRAIN = 100
MAX_NUMBER_TEST  = 100

###########
### CNN ###
###########

### load & save ###
selector = 'dataset_name'

folder = 'unpooled'
subfolder = f'motif_length_{K_MER_LENGTH}'

data = 'IAV'
strain = 'PR8'
segment = 'PB1'
intersects = 'median_global_0'
motif_length = K_MER_LENGTH

dfnames = get_dataset_names(DATASET_CUTOFF, data)
# dfnames = [dfnames[0]]
dfs = load_all_preprocessed(dfnames, folder, subfolder)

dfs = manage_separate_specifiers(dfs, data, strain, segment)

### categorical features ###
categorical_cols = []

# only for multi segment selection!
# categorical_cols += ["segment"]

# only for multi segment/strain/subtype/type selection! only use one! strain recommended (most distinct)
# categorical_cols += ["strain"]
# categorical_cols += ["subtype"]
# categorical_cols += ["type"]

# technical metadata (NOT recommended)
# categorical_cols += ['library_layout', 'library_selection', 'library_source']

# biological metadata (recommended)
categorical_cols += ['system_type', 'host']
# only available for unpooled data
# categorical_cols += ['cell_system', 'localization', 'resolution']
# categorical_cols += ['time_point', 'MOI']

# junction site motifs
# categorical_cols += ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]

# use with datset intersects
# categorical_cols += ["dataset_name"]

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
# numerical_cols += ["poly_U_max_run", "poly_U_tracts", "poly_A_max_run", "poly_A_tracts"]
# numerical_cols += ["palindrome_density"]


### secondary ###

# numerical_cols += ["MFE"]
# numerical_cols += ["bp_count", "bp_density", "unpaired_count", "unpaired_density", "external_unpaired_density"]
# numerical_cols += ["stem_count", "stem_len_max", "stem_len_mean", "stem_len_min"]
# numerical_cols += ["hairpin_count", "hairpin_size_mean", "hairpin_size_min", "hairpin_size_max"]
# numerical_cols += ["pair_span_mean", "pair_span_min", "pair_span_max"]
# numerical_cols += ["free_5prime_len", "free_3prime_len"]


### hybrid ###

# numerical_cols += ["GC_overall", "GC_paired", "GC_unpaired"]
# numerical_cols += ["pair_GC_count", "pair_AU_count", "pair_GU_count", "pair_noncanon_count"]
# numerical_cols += ["pair_GC_content", "pair_AU_content", "pair_GU_content", "pair_noncanon_content"]
# numerical_cols += ["stem_end_GC_content", "stem_end_AU_content", "hairpin_close_GC_content", "hairpin_close_AU_content"]

### extra features ###
extra_cols = [
    "dataset_name",
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
if folder == 'pooled':
    THRESHOLD = POOLED_THRESHOLD
elif folder == 'unpooled':
    THRESHOLD = UNPOOLED_THRESHOLD

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
### group-safe split (80/10/10) ###
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

    # dataset-wise uniqueness: treat same sequence in different datasets as different candidates
    if 'dataset' in intersects:
        df['ikey'] = df['dataset_name'].astype(str) + '|' + df['ikey'].astype(str)

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

if BALANCE:
    df_fit = balance_by_threshold(df_fit, 'norm_log_NGS_read_count', THRESHOLD)

##########################
### labels + reporting ###
##########################

df_fit["label"]  = (df_fit["norm_log_NGS_read_count"]  > THRESHOLD).astype(int)
df_val["label"]  = (df_val["norm_log_NGS_read_count"]  > THRESHOLD).astype(int)
df_test["label"] = (df_test["norm_log_NGS_read_count"] > THRESHOLD).astype(int)

pos = int(np.sum(df_fit["label"] == 1))
neg = int(np.sum(df_fit["label"] == 0))
w_pos = np.sqrt((neg / max(1, pos))) if pos > 0 else 1.0
w_pos = float(min(w_pos, 5.0))
class_weight = {0: 1.0, 1: w_pos}

print("\nclass weight (fit)")
print(f"w_pos: {w_pos:.{DECIMALS}f}")

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
token2idx = {k: i + 1 for i, k in enumerate(vocab)}

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

### sec structure ###
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

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

X_cat_fit  = encoder.fit_transform(df_fit[categorical_cols]).astype(np.float32)
X_cat_val  = encoder.transform(df_val[categorical_cols]).astype(np.float32)
X_cat_test = encoder.transform(df_test[categorical_cols]).astype(np.float32)

scaler = StandardScaler()
X_num_fit  = scaler.fit_transform(df_fit[numerical_cols]).astype(np.float32)
X_num_val  = scaler.transform(df_val[numerical_cols]).astype(np.float32)
X_num_test = scaler.transform(df_test[numerical_cols]).astype(np.float32)

X_other_fit  = np.hstack([X_cat_fit,  X_num_fit]).astype(np.float32)
X_other_val  = np.hstack([X_cat_val,  X_num_val]).astype(np.float32)
X_other_test = np.hstack([X_cat_test, X_num_test]).astype(np.float32)

X_cat = X_cat_fit

y_fit  = df_fit["label"].values
y_val  = df_val["label"].values
y_test = df_test["label"].values

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

other_input = layers.Input(shape=(X_other_fit.shape[1],))

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

    concat = layers.concatenate([x, x2, other_input])
    inputs = [seq_input, str_input, other_input]
else:
    concat = layers.concatenate([x, other_input])
    inputs = [seq_input, other_input]

dense = layers.Dense(
    DENSE_UNITS,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(concat)
dense = layers.Dropout(0.5)(dense)
output = layers.Dense(1, activation='sigmoid')(dense)

model = models.Model(inputs=inputs, outputs=output)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[],
    weighted_metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
        tf.keras.metrics.AUC(name='auc_pr', curve='PR')
    ]
)

### callbacks ###
best_model_path = os.path.join(save_path_model, f"{NAME_MOD}_best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path, save_best_only=True, monitor="val_auc_pr", mode="max"
)

early_cb = callbacks.EarlyStopping(
    monitor="val_auc_pr",
    mode="max",
    patience=2,
    restore_best_weights=True
)

w_fit = df_fit["count"].to_numpy().astype(np.float32) if "count" in df_fit.columns else np.ones(len(df_fit), dtype=np.float32)
w_val = df_val["count"].to_numpy().astype(np.float32) if "count" in df_val.columns else np.ones(len(df_val), dtype=np.float32)

w_fit = w_fit / (np.mean(w_fit) + 1e-12)
w_val = w_val / (np.mean(w_val) + 1e-12)

### train ###
if STRUCTURE:
    history = model.fit(
        [X_seq_fit, X_str_fit, X_other_fit], y_fit,
        sample_weight=w_fit,
        validation_data=([X_seq_val, X_str_val, X_other_val], y_val, w_val),
        epochs=25,
        batch_size=16,
        callbacks=[checkpoint_cb, early_cb],
    )
else:
    history = model.fit(
        [X_seq_fit, X_other_fit], y_fit,
        sample_weight=w_fit,
        validation_data=([X_seq_val, X_other_val], y_val, w_val),
        epochs=25,
        batch_size=16,
        callbacks=[checkpoint_cb, early_cb],
    )

##############################################
### estimate F1 threshold (validation set) ###
##############################################

TARGET_PRECISION = 0.90
MIN_POS_PRED = 5   # avoid degenerate all-negative decision

if STRUCTURE:
    y_val_proba = model.predict(
        [X_seq_val, X_str_val, X_other_val]
    ).ravel()
else:
    y_val_proba = model.predict(
        [X_seq_val, X_other_val]
    ).ravel()

precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)

precision_t = precision[:-1]
recall_t    = recall[:-1]

mask = precision_t >= TARGET_PRECISION

D_THRESHOLD = None

if np.any(mask):
    cand_thr = thresholds[mask]
    cand_rec = recall_t[mask]

    kept_thr = []
    kept_rec = []
    for t, r in zip(cand_thr, cand_rec):
        n_pos_pred = int(np.sum(y_val_proba >= t))
        if n_pos_pred >= MIN_POS_PRED:
            kept_thr.append(t)
            kept_rec.append(r)

    if len(kept_thr) > 0:
        best_idx = int(np.argmax(kept_rec))
        D_THRESHOLD = float(kept_thr[best_idx])

if D_THRESHOLD is None:
    f1_scores = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-12)
    best_idx = int(np.nanargmax(f1_scores))
    D_THRESHOLD = float(thresholds[best_idx])

D_THRESHOLD = round(D_THRESHOLD, DECIMALS)

f1_at_thr = f1_score(y_val, (y_val_proba >= D_THRESHOLD).astype(int))
f1_at_thr = round(f1_at_thr, DECIMALS)

pr_auc_val = average_precision_score(y_val, y_val_proba)
pr_auc_val = round(pr_auc_val, DECIMALS)

print("\nF1 threshold (validation, precision-constrained)")
print(f"target precision: {TARGET_PRECISION}")
print(f"chosen threshold: {D_THRESHOLD}")
print(f"val PR-AUC:       {pr_auc_val}")
print(f"val F1:           {f1_at_thr}")
print(f"val positives predicted: {int(np.sum(y_val_proba >= D_THRESHOLD))}")

y_pred = (y_val_proba >= D_THRESHOLD).astype(int)
tp = int(np.sum((y_pred == 1) & (y_val == 1)))
fp = int(np.sum((y_pred == 1) & (y_val == 0)))
fn = int(np.sum((y_pred == 0) & (y_val == 1)))

prec = tp / (tp + fp + 1e-12)
rec  = tp / (tp + fn + 1e-12)

print(f"val precision threshold: {prec:.{DECIMALS}f}")
print(f"val recall threshold:    {rec:.{DECIMALS}f}")

############
### save ###
############

final_model_path = os.path.join(save_path_model, f"{NAME_MOD}_final_model.h5")
model.save(final_model_path)
print(f"\n✔ models saved")

scaler_path = os.path.join(save_path_model, f"{NAME_MOD}_scaler.save")
joblib.dump(scaler, scaler_path)
print(f"✔ scaler saved")

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

plt.style.use('seaborn-darkgrid')
plt.rc('font', size=12)

### loss curve (train, val) ###
title_name = 'bin. CNN: loss across epochs (training, validation) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={n_train}, n_val={n_val})'

epochs = np.arange(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history["loss"], label="training", linewidth=2, color=COLORS[8])
plt.plot(epochs, history.history["val_loss"], label="validation", linewidth=2, color=COLORS[0])
plt.xlabel("epoch")
plt.ylabel("loss (binary crossentropy)")
plt.title(title_name)

plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.,
    title='type',
    frameon=True
)

plt.grid(True)

train_curve_path = os.path.join(save_path_vis, f"{NAME_MOD}_training_loss.png")
plt.savefig(train_curve_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ training loss curve saved")

### accuracy curve (train, val) ###
title_name = 'bin. CNN: accuracy across epochs (training, validation) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={n_train}, n_val={n_val})'

if "accuracy" in history.history and "val_accuracy" in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history["accuracy"], label="training", linewidth=2, color=COLORS[8])
    plt.plot(epochs, history.history["val_accuracy"], label="validation", linewidth=2, color=COLORS[0])
    plt.xlabel("epoch")
    plt.ylabel("accuracy (fraction of correct predictions)")
    plt.title(title_name)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

    plt.grid(True)

    acc_curve_path = os.path.join(save_path_vis, f"{NAME_MOD}_training_accuracy.png")
    plt.savefig(acc_curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ training accuracy curve saved")

### PR-AUC curve (train, val) ###
title_name = 'bin. CNN: PR-AUC across epochs (training, validation) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={n_train}, n_val={n_val})'

# be robust to older metric naming
if ("auc_pr" in history.history) and ("val_auc_pr" in history.history):
    pr_key = "auc_pr"
    val_pr_key = "val_auc_pr"
elif ("pr_auc" in history.history) and ("val_pr_auc" in history.history):
    pr_key = "pr_auc"
    val_pr_key = "val_pr_auc"
elif ("auc" in history.history) and ("val_auc" in history.history):
    pr_key = "auc"
    val_pr_key = "val_auc"
else:
    pr_key = None
    val_pr_key = None

if pr_key is not None:
    epochs = np.arange(1, len(history.history[pr_key]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        history.history[pr_key],
        label="training",
        linewidth=2,
        color=COLORS[8]
    )
    plt.plot(
        epochs,
        history.history[val_pr_key],
        label="validation",
        linewidth=2,
        color=COLORS[0]
    )

    plt.xlabel("epoch")
    plt.ylabel("PR-AUC")
    plt.title(title_name)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        title='type',
        frameon=True
    )

    plt.grid(True)

    pr_auc_curve_path = os.path.join(save_path_vis, f"{NAME_MOD}_training_pr_auc.png")
    plt.savefig(pr_auc_curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ training PR-AUC curve saved")

model.load_weights(model_path)

# build dfs #
if STRUCTURE:
    y_pred_proba_train = model.predict(
        [X_seq_train, X_str_train, X_other_train]
    ).ravel()
    y_pred_proba_test = model.predict(
        [X_seq_test, X_str_test, X_other_test]
    ).ravel()
else:
    y_pred_proba_train = model.predict(
        [X_seq_train, X_other_train]
    ).ravel()
    y_pred_proba_test = model.predict(
        [X_seq_test, X_other_test]
    ).ravel()

df_train_proba = df_train.copy().reset_index(drop=True)
df_train_proba["y_true"] = y_train
df_train_proba["y_proba"] = y_pred_proba_train

df_test_proba = df_test.copy().reset_index(drop=True)
df_test_proba["y_true"] = y_test
df_test_proba["y_proba"] = y_pred_proba_test

modification = get_feature_modification_name()

def safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba)

def safe_pr_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_proba)

def safe_f1(y_true, y_proba, threshold=0.5):
    if len(np.unique(y_true)) < 2:
        return np.nan
    y_pred = (y_proba >= threshold).astype(int)
    return f1_score(y_true, y_pred)

### ROC-AUC & PR-AUC & F1 (test) ###
create_feature_pr_auc_plot(
    plot_name="bin. CNN: PR-AUC (testing) - curve plot",
    df=df_test_proba,
    y_true_feature_name="y_true",
    y_pred_proba_feature_name="y_proba",
    f1_threshold=D_THRESHOLD,
    selector=selector,
    fname=f"{NAME_MOD}_PR_AUC",
    path="bin_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
)

print(f"✔ test PR-AUC curve saved")

create_feature_roc_auc_plot(
    plot_name="bin. CNN: ROC-AUC (testing) - curve plot",
    df=df_test_proba,
    y_true_feature_name="y_true",
    y_pred_proba_feature_name="y_proba",
    f1_threshold=D_THRESHOLD,
    selector=selector,
    fname=f"{NAME_MOD}_ROC_AUC",
    path="bin_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
)

print(f"✔ test ROC-AUC curve saved")

create_prediction_probability_density_plot(
    plot_name="bin. CNN: predicted probabilty distribution via KDE (testing) - density plot",
    df=df_test_proba,
    x_feature_name="y_proba",
    x_axis_name="predicted probabilty for high NGS read count",
    decision_threshold=D_THRESHOLD,
    fname=f"{NAME_MOD}_pred_prob",
    path="bin_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
)

### scatter plot (test)  ###
create_feature_scatter_plot(
    plot_name="bin. CNN: predicted probabilty as a function of true NGS read count (testing) - scatter plot",
    df=df_test_proba,
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    y_feature_name="y_proba",
    y_axis_name="predicted probabilty for high NGS read count",
    selector=selector,
    show_rolling_median=False,
    rolling_window=50,
    show_identity_line=False,
    pseudo_prefix="",
    show_decision_threshold=D_THRESHOLD,
    reg_metrics=False,
    huber_delta=0.0,
    fname=f"{NAME_MOD}_prediction",
    path="bin_network",
    folder=folder,
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
)

print("✔ prediction scatter plot saved")

print(f"✔ test prediction density curve saved")

### density (train, test) ###
create_multi_density_plot(
    plot_name="NGS read count distribution via KDE (testing) - density plot",
    df_list=[df_train, df_test],
    df_names=["training", "test"],
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    fname=f"{NAME_MOD}_density",
    path="bin_network",
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

### occlusion importances (test)  ###
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

def cnn_predict_proba(model, X_seq, X_other, STRUCTURE=False, X_str=None):
    if STRUCTURE:
        return model.predict([X_seq, X_str, X_other], verbose=0).ravel()
    return model.predict([X_seq, X_other], verbose=0).ravel()

def cnn_test_roc_auc(model, X_seq_test, X_other_test, y_test, STRUCTURE=False, X_str_test=None):
    y_proba = cnn_predict_proba(model, X_seq_test, X_other_test, STRUCTURE=STRUCTURE, X_str=X_str_test)
    return safe_roc_auc(y_test, y_proba)

def cnn_test_pr_auc(model, X_seq_test, X_other_test, y_test, STRUCTURE=False, X_str_test=None):
    y_proba = cnn_predict_proba(model, X_seq_test, X_other_test, STRUCTURE=STRUCTURE, X_str=X_str_test)
    return safe_pr_auc(y_test, y_proba)

base_roc_auc = cnn_test_roc_auc(
    model,
    X_seq_eval, X_other_eval, y_eval,
    STRUCTURE=bool(STRUCTURE),
    X_str_test=(X_str_eval if STRUCTURE else None),
)

base_pr_auc = cnn_test_pr_auc(
    model,
    X_seq_eval, X_other_eval, y_eval,
    STRUCTURE=bool(STRUCTURE),
    X_str_test=(X_str_eval if STRUCTURE else None),
)

base_f1_eval = safe_f1(
    y_eval,
    cnn_predict_proba(
        model,
        X_seq_eval, X_other_eval,
        STRUCTURE=bool(STRUCTURE),
        X_str=(X_str_eval if STRUCTURE else None),
    ),
    threshold=D_THRESHOLD
)

base_roc_s = "nan" if not np.isfinite(base_roc_auc) else f"{base_roc_auc:.{DECIMALS}f}"
base_pr_s  = "nan" if not np.isfinite(base_pr_auc)  else f"{base_pr_auc:.{DECIMALS}f}"
base_f1_s  = "nan" if not np.isfinite(base_f1_eval) else f"{base_f1_eval:.{DECIMALS}f}"

try:
    cat_names_full = encoder.get_feature_names_out(categorical_cols)
except AttributeError:
    cat_names_full = encoder.get_feature_names(categorical_cols)
feature_names_other = list(cat_names_full) + numerical_cols

def importance_x_other_pr_auc_drop(
    model,
    X_seq_test, X_other_test, y_test,
    feature_names,
    STRUCTURE=False, X_str_test=None,
    mode="permute",
    n_repeats=5,
    seed=SEED
):
    rng = np.random.default_rng(seed)
    base = cnn_test_pr_auc(model, X_seq_test, X_other_test, y_test, STRUCTURE=STRUCTURE, X_str_test=X_str_test)

    rows = []
    for j, fname in enumerate(feature_names):
        if mode == "zero":
            X_mod = X_other_test.copy()
            X_mod[:, j] = 0.0
            pr_mod = cnn_test_pr_auc(model, X_seq_test, X_mod, y_test, STRUCTURE=STRUCTURE, X_str_test=X_str_test)
            drop = base - pr_mod

        elif mode == "permute":
            drops = []
            for _ in range(n_repeats):
                X_mod = X_other_test.copy()
                perm = rng.permutation(X_mod.shape[0])
                X_mod[:, j] = X_mod[perm, j]
                pr_mod = cnn_test_pr_auc(model, X_seq_test, X_mod, y_test, STRUCTURE=STRUCTURE, X_str_test=X_str_test)
                drops.append(base - pr_mod)
            drop = float(np.nanmean(drops))
        else:
            raise ValueError("mode must be 'permute' or 'zero'")

        rows.append((fname, drop))

    imp_df = (
        pd.DataFrame(rows, columns=["feature", "pr_auc_drop"])
          .sort_values("pr_auc_drop", ascending=False)
    )
    return base, imp_df

# permutation importance (on eval subset)
base_perm, imp_perm = importance_x_other_pr_auc_drop(
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
base_zero, imp_zero = importance_x_other_pr_auc_drop(
    model,
    X_seq_eval, X_other_eval, y_eval,
    feature_names_other,
    STRUCTURE=bool(STRUCTURE),
    X_str_test=(X_str_eval if STRUCTURE else None),
    mode="zero",
)

base_perm_s = "nan" if not np.isfinite(base_perm) else f"{base_perm:.{DECIMALS}f}"
base_zero_s = "nan" if not np.isfinite(base_zero) else f"{base_zero:.{DECIMALS}f}"

# plot permutation (test subset)
plot_df = imp_perm.head(TOP_N).iloc[::-1]
title_name = f'bin. CNN: feature importance via permuation occlusion (testing) - bar plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n={eval_n})'
title_name += f'\nROC-AUC={base_roc_s}, PR-AUC={base_perm_s}, F1={base_f1_s} (decision_threshold={D_THRESHOLD})'

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["pr_auc_drop"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("PR-AUC drop (when permuted)")
plt.ylabel(f"feature ({TOP_N})")
plt.title(title_name)
plt.tight_layout()
plt.grid(True, axis="x")

perm_path = os.path.join(save_path_vis, f"{NAME_MOD}_permutation.png")
plt.savefig(perm_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ permutation importance saved")

# plot zero-occlusion (test subset)
plot_df = imp_zero.head(TOP_N).iloc[::-1]
title_name = f'bin. CNN: feature importance via zero occlusion (testing) - bar plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n={eval_n})'
title_name += f'\nROC-AUC={base_roc_s}, PR-AUC={base_perm_s}, F1={base_f1_s} (decision_threshold={D_THRESHOLD})'

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["pr_auc_drop"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("PR-AUC drop (when set to 0)")
plt.ylabel(f"feature ({TOP_N})")
plt.title(title_name)
plt.tight_layout()
plt.grid(True, axis="x")

occ_path = os.path.join(save_path_vis, f"{NAME_MOD}_occlusion.png")
plt.savefig(occ_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ occlusion importance saved")

SEQ_OCC_WINDOW = 10
SEQ_OCC_STRIDE = 5

def sequence_window_occlusion(
    model,
    X_seq_test, X_other_test, y_test,
    window=10, stride=5,
    STRUCTURE=False, X_str_test=None,
):
    base = cnn_test_pr_auc(model, X_seq_test, X_other_test, y_test, STRUCTURE=STRUCTURE, X_str_test=X_str_test)

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

        pr_mod = cnn_test_pr_auc(
            model,
            X_mod, X_other_test, y_test,
            STRUCTURE=STRUCTURE,
            X_str_test=X_str_test
        )
        rows.append((s, e, base - pr_mod))

    return base, pd.DataFrame(rows, columns=["start_tok", "end_tok", "pr_auc_drop"])

base_seq, seq_imp_df = sequence_window_occlusion(
    model,
    X_seq_eval, X_other_eval, y_eval,
    window=SEQ_OCC_WINDOW,
    stride=SEQ_OCC_STRIDE,
    STRUCTURE=bool(STRUCTURE),
    X_str_test=(X_str_eval if STRUCTURE else None),
)

base_seq_s = "nan" if not np.isfinite(base_seq) else f"{base_seq:.{DECIMALS}f}"

# curve plot (PR-AUC drop vs window center)
x_center = (seq_imp_df["start_tok"] + seq_imp_df["end_tok"]) / 2.0
y_drop = seq_imp_df["pr_auc_drop"].to_numpy()

title_name = f'bin. CNN: sequence importance (window={SEQ_OCC_WINDOW}, stride={SEQ_OCC_STRIDE}) via window occlusion (testing) - curve plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n={eval_n})'
title_name += f'\nROC-AUC={base_roc_s}, PR-AUC={base_perm_s}, F1={base_f1_s} (decision_threshold={D_THRESHOLD})'

df_eval_meta = df_test.iloc[idx_eval].copy()

start_med_tok = np.nan
end_med_tok = np.nan

start_med_tok = float(np.nanmedian(df_eval_meta["start"].astype(float).to_numpy())) // KMER_STEP
if MARKED:
    end_med_tok = float(np.nanmedian(df_eval_meta["end"].astype(float).to_numpy()))   // KMER_STEP
    start_label = f"start median ({int(start_med_tok)})"
    end_label = f"end median ({int(end_med_tok)})"
else:
    end_med_tok = float(np.nanmedian(df_eval_meta["5_end_length"].astype(float).to_numpy() + df_eval_meta["3_end_length"].astype(float).to_numpy()))   // KMER_STEP
    start_label = f"deletion site median ({int(start_med_tok)})"
    end_label = f"padding start median ({int(end_med_tok)})"

plt.figure(figsize=(10, 6))
plt.plot(x_center, y_drop, linewidth=2, color=COLORS[6], label="test")

if np.isfinite(start_med_tok):
    plt.axvline(
        start_med_tok,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label=start_label,
    )

if np.isfinite(end_med_tok):
    plt.axvline(
        end_med_tok,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label=end_label
    )

plt.xlabel(f"k-mer (k={KMER_SIZE}) position (window center)")
plt.ylabel("PR-AUC drop (when window masked)")
plt.title(title_name)
plt.tight_layout()
plt.grid(True)

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
top_windows = seq_imp_df.sort_values("pr_auc_drop", ascending=False).head(TOP_N).copy()
top_windows["window"] = top_windows.apply(lambda r: f'{int(r.start_tok)}-{int(r.end_tok)}', axis=1)
top_windows = top_windows.sort_values("pr_auc_drop", ascending=True)

title_name = f'bin. CNN: sequence importance (window={SEQ_OCC_WINDOW}, stride={SEQ_OCC_STRIDE}) via window occlusion (testing) - bar plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n={eval_n})'
title_name += f'\nROC-AUC={base_roc_s}, PR-AUC={base_perm_s}, F1={base_f1_s} (decision_threshold={D_THRESHOLD})'

plt.figure(figsize=(10, 6))
plt.barh(top_windows["window"], top_windows["pr_auc_drop"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("PR-AUC drop (when window masked)")
plt.ylabel(f"k-mer (k={KMER_SIZE}) window (top {TOP_N})")
plt.title(title_name)
plt.tight_layout()
plt.grid(True, axis="x")

seq_top_path = os.path.join(save_path_vis, f"{NAME_MOD}_window_occlusion_bar.png")
plt.savefig(seq_top_path, dpi=300, bbox_inches="tight")
plt.close()
print("✔ sequence window occlusion bar saved")

### SHAP (test) ###
if SHAP_CAT:
    X_train_shap = X_other_train
    X_test_shap  = X_other_test
    try:
        cat_names = encoder.get_feature_names_out(categorical_cols)
    except AttributeError:
        cat_names = encoder.get_feature_names(categorical_cols)
    feature_names = list(cat_names) + numerical_cols
else:
    n_cat_features = X_cat_fit.shape[1]
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
)

bar_color = COLORS[6]

title_name = f'bin. CNN: SHAP feature importance via RandomForest (testing) - bar plot'
title_name += make_candidate_descriptor(folder, data, strain, segment, intersects)
title_name += f' (n_train={bg_n}, n_test={eval_n_shap})'

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color=bar_color, edgecolor='white', linewidth=1)
plt.xlabel("mean |SHAP value|")
plt.ylabel(f"feature (top {TOP_N})")
plt.title(title_name)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis="x")

shap_path = os.path.join(save_path_vis, f"{NAME_MOD}_shap.png")
plt.savefig(shap_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ SHAP bar chart saved")
