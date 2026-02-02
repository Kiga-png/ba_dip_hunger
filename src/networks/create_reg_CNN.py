import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from analysis.visuals import create_train_test_density_plot, create_feature_residual_plot, create_feature_scatter_plot

from utils import reduce_rows, get_feature_modification_name

from utils import RESULTSPATH, SEED, TOP_N, DATASET_STRAIN_DICT
from utils import COLORS, DECIMALS

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import joblib
import re

import scipy.sparse as sp
import shap

### settings ###
RESULTSPATH, _ = os.path.split(RESULTSPATH)
DATAPATH = os.path.join(RESULTSPATH, "preprocess")
RESULTSPATH = os.path.join(RESULTSPATH, "networks")

VERSION = "0"
KMER_SIZE = 3
KMER_STEP = 3

TRAIN_LOSS  = "huber"   # "huber" | "mse" | "mae"

MARKED = 1   # 0 | 1
DROP_X = 0   # 0 | 1
SHAP_CAT = 1   # 0 | 1
STRUCTURE = 0   # 0 | 1

MAX_NUMBER = 25000

BEST_MODEL = 0   # 0 | 1

# occlussion and SHAP
MAX_NUMBER_TRAIN = 10
MAX_NUMBER_TEST  = 10

###########
### CNN ###
###########

### load & save ###
selector = 'dataset_name'

folder = 'unpooled'
subfolder = 'training'

data = 'IAV'

strain = 'PR8'
### for single dataset ###
# strain = DATASET_STRAIN_DICT[data]

segment = 'PB1'
intersects = 'median'
motif_length = 3

### categorical features ###
categorical_cols = []

# categorical_cols += ["Segment", "Strain"]
categorical_cols += ['system_type', 'library_layout', 'library_selection', 'library_source', 'subtype']
# categorical_cols += ['Localization', 'Resolution', 'Cells', 'Host']
# categorical_cols += ['Time', 'MOI']
categorical_cols += ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]

# categorical_cols += ["full_symmetry"]
# categorical_cols += [col for col in df.columns if re.search(r"motif\d+", col)]
# categorical_cols += ["structure_site1_motif", "structure_site2_motif", "structure_site3_motif", "structure_site4_motif"]

### test for dataset dependencies ###
# categorical_cols += ["dataset_name"]

### numerical features ###
numerical_cols = []

# numerical_cols += ["Start", "End"]
# numerical_cols += ["full_seq_length"]
# numerical_cols += ["dvg_length"]
numerical_cols += ["deletion_length"]
numerical_cols += ["5_end_length", "3_end_length"]

# numerical_cols += ["GC_content", "AU_content"]
# numerical_cols += ["UpA_content", "CpG_content"]
# numerical_cols += ["GC_skew", "sequence_entropy"]
# numerical_cols += ["poly_U_max_run", "poly_U_tracts", "poly_A_max_run", "poly_A_tracts"]
# numerical_cols += ["palindrome_density"]
# numerical_cols += ["longest_ORF_len", "ORF_count"]
# numerical_cols += ["kmer_richness", "codon_usage_entropy"]

# numerical_cols += ["direct_repeat_length"]

# numerical_cols += ["MFE"]
# numerical_cols += ["bp_count", "bp_density", "unpaired_count", "unpaired_density", "external_unpaired_density"]
# numerical_cols += ["stem_count", "stem_len_max", "stem_len_mean", "stem_len_min"]
# numerical_cols += ["hairpin_count", "hairpin_size_mean", "hairpin_size_min", "hairpin_size_max"]
# numerical_cols += ["pair_span_mean", "pair_span_min", "pair_span_max"]
# numerical_cols += ["free_5prime_len", "free_3prime_len"]
# numerical_cols += ["branch_point_count"]
# numerical_cols += ["max_symmetry"]

### NOT recommended ###
# numerical_cols += ["MFE_site1_motif", "MFE_site2_motif", "MFE_site3_motif", "MFE_site4_motif"]

# numerical_cols += ["GC_overall", "GC_paired", "GC_unpaired"]
# numerical_cols += ["pair_GC_count", "pair_AU_count", "pair_GU_count", "pair_noncanon_count", "pair_GC_content", "pair_AU_content", "pair_GU_content", "pair_noncanon_content"]
# numerical_cols += ["stem_end_GC_content", "stem_end_AU_content", "stem_end_GU_content"]
# numerical_cols += ["hairpin_close_GC_content", "hairpin_close_AU_content", "hairpin_close_GU_content", "hairpin_close_noncanon_content"]
# numerical_cols += ["motif_GNRA_count", "motif_UNCG_count", "motif_CUUG_count"]
# numerical_cols += ["AU_unpaired_content", "AU_hairpin_content", "AU_internal_content", "AU_external_content"]
# numerical_cols += ["GC_5prime_tail", "GC_3prime_tail"]
# numerical_cols += ["AUG_total", "AUG_unpaired", "AUG_unpaired_content"]

### extra features ###
extra_cols = [
    "dataset_name",
    "norm_log_NGS_read_count",
    "marked_dvg_sequence",
    "dvg_sequence",
]

if STRUCTURE:
    extra_cols += [
        "marked_structure",
        "structure",
    ]

### build dtype map ###
dtype_map = {col: "string" for col in categorical_cols}
dtype_map.update({col: "float32" for col in numerical_cols})

usecols = list(dict.fromkeys(
    categorical_cols +
    numerical_cols +
    extra_cols
))

fname = f'motif_length_{motif_length}'
read_path = os.path.join(DATAPATH, folder, subfolder, data, strain, segment, intersects)
df = pd.read_csv(os.path.join(read_path, f'{fname}.csv'), usecols=usecols, dtype=dtype_map, keep_default_na=False, na_values=[])

if MARKED:
    key_sequence = "marked_dvg_sequence"
    key_structure = "marked_structure"
else:
    key_sequence = "dvg_sequence"
    key_structure = "structure"

df = reduce_rows(df, MAX_NUMBER)

NAME_MOD = f'{VERSION}_motif_length_{motif_length}'

save_path_model = os.path.join(RESULTSPATH, 'reg', 'CNN', data, strain, segment, intersects)
os.makedirs(save_path_model, exist_ok=True)

### target (regression) ###
y_cont = df["norm_log_NGS_read_count"].values

### k-mer token ###
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

### sequence to integer ###
vocab = set()
df["kmer_seq"] = df[key_sequence].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
df["kmer_seq"].apply(lambda kmers: vocab.update(kmers))
vocab = sorted(vocab)
token2idx = {k: i + 1 for i, k in enumerate(vocab)}

def encode_seq(kmers):
    return [token2idx[k] for k in kmers if k in token2idx]

df["encoded_seq"] = df["kmer_seq"].apply(encode_seq)
maxlen = df["encoded_seq"].apply(len).max()
X_seq = tf.keras.preprocessing.sequence.pad_sequences(df["encoded_seq"], maxlen=maxlen)

### secondary structure (OPTIONAL) ###
if STRUCTURE:
    vocab2 = set()
    df["kmer_str"] = df[key_structure].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
    df["kmer_str"].apply(lambda kmers: vocab2.update(kmers))
    vocab2 = sorted(vocab2)
    token2idx2 = {k: i + 1 for i, k in enumerate(vocab2)}

    def encode_str(kmers):
        return [token2idx2[k] for k in kmers if k in token2idx2]

    df["encoded_str"] = df["kmer_str"].apply(encode_str)
    maxlen2 = df["encoded_str"].apply(len).max()
    X_str = tf.keras.preprocessing.sequence.pad_sequences(df["encoded_str"], maxlen=maxlen2)
else:
    vocab2 = []
    token2idx2 = {}
    maxlen2 = 0
    X_str = None
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[categorical_cols])

scaler = StandardScaler()
X_num = scaler.fit_transform(df[numerical_cols])

### combine features ###
X_other = np.hstack([X_cat, X_num])

### train/test split ###
if STRUCTURE:
    (
        X_seq_train, X_seq_test,
        X_str_train, X_str_test,
        X_other_train, X_other_test,
        y_train, y_test,
        df_train, df_test
    ) = train_test_split(
        X_seq, X_str, X_other, y_cont, df,
        test_size=0.2,
        random_state=SEED
    )
else:
    (
        X_seq_train, X_seq_test,
        X_other_train, X_other_test,
        y_train, y_test,
        df_train, df_test
    ) = train_test_split(
        X_seq, X_other, y_cont, df,
        test_size=0.2,
        random_state=SEED
    )

### Huber delta (data-driven, based on training targets) ###
y_train_arr = np.asarray(y_train, dtype=float)

med = float(np.nanmedian(y_train_arr))
mad = float(np.nanmedian(np.abs(y_train_arr - med)))

# robust std estimate from MAD
HUBER_DELTA = 1.4826 * mad

# fallback if MAD is 0 or nan
if (not np.isfinite(HUBER_DELTA)) or (HUBER_DELTA <= 0):
    HUBER_DELTA = float(np.nanstd(y_train_arr))

# keep delta in a reasonable range (avoid too-MSE-like or too-MAE-like)
HUBER_DELTA = float(np.clip(HUBER_DELTA, 0.5, 2.0))
HUBER_DELTA = round(HUBER_DELTA, DECIMALS)

print("\nHuber delta (training, data-driven)")
print(f"HUBER_DELTA: {HUBER_DELTA}")

### model ###
seq_input = layers.Input(shape=(maxlen,))
x = layers.Embedding(input_dim=len(token2idx) + 1, output_dim=32)(seq_input)
x = layers.Conv1D(64, 5, activation='relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)

other_input = layers.Input(shape=(X_other.shape[1],))

if STRUCTURE:
    # exact structure branch as before
    str_input = layers.Input(shape=(maxlen2,))
    x2 = layers.Embedding(input_dim=len(token2idx2) + 1, output_dim=32)(str_input)
    x2 = layers.Conv1D(64, 5, activation='relu')(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Flatten()(x2)

    concat = layers.concatenate([x, x2, other_input])
    inputs = [seq_input, str_input, other_input]
else:
    concat = layers.concatenate([x, other_input])
    inputs = [seq_input, other_input]

dense = layers.Dense(64, activation='relu')(concat)
dense = layers.Dropout(0.3)(dense)
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
    metrics=[
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

### train ###
if STRUCTURE:
    history = model.fit(
        [X_seq_train, X_str_train, X_other_train], y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        callbacks=[checkpoint_cb]
    )
else:
    history = model.fit(
        [X_seq_train, X_other_train], y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        callbacks=[checkpoint_cb]
    )

### save ###
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
}, preproc_path)

print("✔ preprocessing artifacts saved")


###############
### visuals ###
###############

### numbers ###
n_total = len(df)
n_train = len(df_train)
n_val = int(np.floor(0.1 * n_train))
n_fit = n_train - n_val
n_test = len(df_test)

save_path_vis = os.path.join(RESULTSPATH, 'reg', 'visuals', data, strain, segment, intersects)
os.makedirs(save_path_vis, exist_ok=True)

if BEST_MODEL:
    model_path = best_model_path
else:
    model_path = final_model_path

plt.style.use('seaborn-darkgrid')
plt.rc('font', size=12)

# last epoch values (absolute numbers for titles)
mse_last_tr = float(history.history["mse"][-1])
mse_last_va = float(history.history["val_mse"][-1])

mae_last_tr = float(history.history["mae"][-1])
mae_last_va = float(history.history["val_mae"][-1])

hub_last_tr = float(history.history["huber"][-1])
hub_last_va = float(history.history["val_huber"][-1])

### MSE curve (train, val) ###
title_name = 'MSE lerarning curve (training, validation) - curve plot'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
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

plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title(title_name)

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
title_name = 'MAE lerarning curve (training, validation) - curve plot'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
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

plt.xlabel("epoch")
plt.ylabel("MAE")
plt.title(title_name)

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
title_name = f'Huber (delta={HUBER_DELTA:g}) lerarning curve (training, validation) - curve plot'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
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

plt.xlabel("epoch")
plt.ylabel("Huber")
plt.title(title_name)

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
if STRUCTURE:
    y_pred_val_train = model.predict(
        [X_seq_train, X_str_train, X_other_train]
    ).ravel()
    y_pred_val_test = model.predict(
        [X_seq_test, X_str_test, X_other_test]
    ).ravel()
else:
    y_pred_val_train = model.predict(
        [X_seq_train, X_other_train]
    ).ravel()
    y_pred_val_test = model.predict(
        [X_seq_test, X_other_test]
    ).ravel()

df_train_val = df_train.copy().reset_index(drop=True)
df_train_val["y_true"] = y_train
df_train_val["y_val"] = y_pred_val_train

df_test_val = df_test.copy().reset_index(drop=True)
df_test_val["y_true"] = y_test
df_test_val["y_val"] = y_pred_val_test

modification = get_feature_modification_name()

### R2 & residual (test)  ###
create_feature_residual_plot(
    plot_name="NGS read count - residual (testing) - scatter plot",
    df=df_test_val,
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    y_pred_feature_name="y_val",
    y_axis_name="residual (predicted - true)",
    selector=selector,
    show_zero_line=True,
    reg_metrics=True,
    fname=f"{NAME_MOD}_residual",
    path="networks",
    folder="reg",
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects
)

### scatter plot (test)  ###
create_feature_scatter_plot(
    plot_name="NGS read count - predictions (testing) - scatter plot",
    df=df_test_val,
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    y_feature_name="y_val",
    y_axis_name=f"predicted {modification} NGS count (reads)",
    selector=selector,
    show_identity_line=True,
    reg_metrics=True,
    huber_delta=HUBER_DELTA,
    fname=f"{NAME_MOD}_prediction",
    path="networks",
    folder="reg",
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
)

print("✔ prediction scatter plot saved")

print(f"\n✔ R² residual plot saved")

### density (train, test) ###
create_train_test_density_plot(
    plot_name="NGS read count - KDE (training, testing) - curve plot",
    df_train=df_train,
    df_test=df_test,
    x_feature_name="norm_log_NGS_read_count",
    x_axis_name=f"{modification} NGS count (reads)",
    fname=f"{NAME_MOD}_density",
    path="networks",
    folder="reg",
    subfolder="visuals",
    data=data,
    strain=strain,
    segment=segment,
    intersects=intersects,
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
    if STRUCTURE:
        return model.predict([X_seq, X_str, X_other], verbose=0).ravel()
    return model.predict([X_seq, X_other], verbose=0).ravel()

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
plot_df = imp_perm.head(TOP_N).iloc[::-1]
title_name = f'top {TOP_N} permutation importances (testing) - bar plot'
title_name += f'\ndata: {data}, strain: {strain}, segment: {segment}, {intersects} intersects (n={eval_n})'
title_name += f'\nMSE={base_mse_perm:.{DECIMALS}f}, MAE={base_mae_perm:.{DECIMALS}f}, Huber={base_hub_perm:.{DECIMALS}f}, R²={base_r2_perm:.{DECIMALS}f}'

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["mse_increase"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("MSE increase (when permuted)")
plt.ylabel("feature")
plt.title(title_name)
plt.tight_layout()
plt.grid(True, axis="x")

perm_path = os.path.join(save_path_vis, f"{NAME_MOD}_permutation.png")
plt.savefig(perm_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ permutation importance saved")

# plot zero-occlusion (test subset)
plot_df = imp_zero.head(TOP_N).iloc[::-1]
title_name = f'top {TOP_N} occlusion importances (testing) - bar plot'
title_name += f'\ndata: {data}, strain: {strain}, segment: {segment}, {intersects} intersects (n={eval_n})'
title_name += f'\nMSE={base_mse_zero:.{DECIMALS}f}, MAE={base_mae_zero:.{DECIMALS}f}, Huber={base_hub_zero:.{DECIMALS}f}, R²={base_r2_zero:.{DECIMALS}f}'

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["mse_increase"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("MSE increase (when set to 0)")
plt.ylabel("feature")
plt.title(title_name)
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

title_name = f'sequence window occlusion (window={SEQ_OCC_WINDOW}, stride={SEQ_OCC_STRIDE}) importances (testing) - curve plot'
title_name += f'\ndata: {data}, strain: {strain}, segment: {segment}, {intersects} intersects (n={eval_n})'
title_name += f'\nMSE={base_mse_seq:.{DECIMALS}f}, MAE={base_mae_seq:.{DECIMALS}f}, Huber={base_hub_seq:.{DECIMALS}f}, R²={base_r2_seq:.{DECIMALS}f}'

df_eval_meta = df_test.iloc[idx_eval].copy()

start_med_tok = np.nan
end_med_tok = np.nan

if ("Start" in df_eval_meta.columns) and ("End" in df_eval_meta.columns):
    start_med_tok = float(np.nanmedian(df_eval_meta["Start"].astype(float).to_numpy())) // KMER_SIZE
    end_med_tok   = float(np.nanmedian(df_eval_meta["End"].astype(float).to_numpy()))   // KMER_SIZE
elif ("5_end_length" in df_eval_meta.columns) and ("deletion_length" in df_eval_meta.columns):
    five = df_eval_meta["5_end_length"].astype(float).to_numpy()
    dele = df_eval_meta["deletion_length"].astype(float).to_numpy()
    start_nt = five + 1.0
    end_nt   = five + dele
    start_med_tok = float(np.nanmedian(start_nt)) // KMER_SIZE
    end_med_tok   = float(np.nanmedian(end_nt))   // KMER_SIZE

plt.figure(figsize=(10, 6))
plt.plot(x_center, y_inc, linewidth=2, color=COLORS[6], label="test")

if np.isfinite(start_med_tok):
    plt.axvline(
        start_med_tok,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label=f"start median ({int(start_med_tok)})"
    )
if np.isfinite(end_med_tok):
    plt.axvline(
        end_med_tok,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label=f"end median ({int(end_med_tok)})"
    )

plt.xlabel("k-mer (k={KMER_SIZE}) position (window center)")
plt.ylabel("MSE increase (when window masked)")
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
top_windows = seq_imp_df.sort_values("mse_increase", ascending=False).head(TOP_N).copy()
top_windows["window"] = top_windows.apply(lambda r: f'{int(r.start_tok)}-{int(r.end_tok)}', axis=1)
top_windows = top_windows.sort_values("mse_increase", ascending=True)

title_name = f'top {TOP_N} sequence window (window={SEQ_OCC_WINDOW}, stride={SEQ_OCC_STRIDE}) importances (testing) - bar plot'
title_name += f'\ndata: {data}, strain: {strain}, segment: {segment}, {intersects} intersects (n={eval_n})'
title_name += f'\nMSE={base_mse_seq:.{DECIMALS}f}, MAE={base_mae_seq:.{DECIMALS}f}, Huber={base_hub_seq:.{DECIMALS}f}, R²={base_r2_seq:.{DECIMALS}f}'

plt.figure(figsize=(10, 6))
plt.barh(top_windows["window"], top_windows["mse_increase"], color=COLORS[6], edgecolor="white", linewidth=1)
plt.xlabel("MSE increase (when window masked)")
plt.ylabel("k-mer (k={KMER_SIZE}) window [start-end)")
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
)

bar_color = COLORS[6]

title_name = f'top {TOP_N} SHAP importances via RandomForest (testing) - bar plot'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
title_name += f' (n_train={bg_n}, n_test={eval_n_shap})'

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color=bar_color, edgecolor='white', linewidth=1)
plt.xlabel("mean |SHAP value|")
plt.ylabel("feature")
plt.title(title_name)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis="x")

shap_path = os.path.join(save_path_vis, f"{NAME_MOD}_shap.png")
plt.savefig(shap_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ SHAP bar chart saved")