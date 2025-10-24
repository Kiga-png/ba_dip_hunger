import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from utils import get_dataset_names, load_all
from utils import manage_specifiers, load_all_preprocessed, merge_missing_features, save_df
from utils import balance_by_threshold

from utils import RESULTSPATH, SEED, DATASET_STRAIN_DICT
from utils import COLORS, DECIMALS

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import joblib
import re

import scipy.sparse as sp
import shap

### settings ###
RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, "networks")

VERSION = "0"
KMER_SIZE = 3
THRESHOLD = 0.43

MARKED = 1
DROP_X = 0
SHAP_CAT = 1
STRUCTURE = 1
BALANCE = 0

BEST_MODEL = 0

###########
### CNN ###
###########

### load & save ###
folder = 'training'
subfolder = 'pri'

data = 'IAV'

strain = 'PR8'
# strain = DATASET_STRAIN_DICT[data]

segment = 'PB1'
intersects = 'median'

motif_length = 3
fname = f'motif_length_{motif_length}'
read_path = os.path.join(RESULTSPATH, folder, subfolder, data, strain, segment, intersects)
df = pd.read_csv(os.path.join(read_path, f'{fname}.csv'), keep_default_na=False, na_values=[])

if MARKED:
    key_sequence = "marked_dvg_sequence"
    key_structure = "marked_structure"
else:
    key_sequence = "dvg_sequence"
    key_structure = "structure"

if BALANCE:
    df = balance_by_threshold(df, 'norm_log_NGS_read_count', THRESHOLD)

NAME_MOD = f'{VERSION}_motif_length_{motif_length}'

save_path_model = os.path.join(RESULTSPATH, 'CNN', subfolder, data, strain, segment, intersects)
os.makedirs(save_path_model, exist_ok=True)

### key parameter ####
df["label"] = (df["norm_log_NGS_read_count"] > THRESHOLD).astype(int)

### k-mer token ###
def kmer_tokenize(seq, k):
    if DROP_X:
        return [seq[i:i + k] for i in range(len(seq) - k + 1) if 'X' not in seq[i:i + k]]
    else:
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

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

### sec structure ###
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

### categorical features ###
categorical_cols = []

# categorical_cols += ["Segment", "Strain"]
categorical_cols += ['system_type', 'library_layout', 'library_selection', 'library_source', 'subtype']
# categorical_cols += ['cells', 'cellular_localization', 'cellular_resolution', 'time_point', 'max_time']
categorical_cols += ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]

# categorical_cols += ["full_symmetry"]
# categorical_cols += [col for col in df.columns if re.search(r"motif\d+", col)]
# categorical_cols += ["structure_site1_motif", "structure_site2_motif", "structure_site3_motif", "structure_site4_motif"]

### test for dataset dependencies ###
# categorical_cols += ["dataset_name"]

try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[categorical_cols])

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

scaler = StandardScaler()
X_num = scaler.fit_transform(df[numerical_cols])

### combine features ###
X_other = np.hstack([X_cat, X_num])
y = df["label"].values

### train/test split ###
if STRUCTURE:
    X_seq_train, X_seq_test, X_str_train, X_str_test, X_other_train, X_other_test, y_train, y_test = train_test_split(
        X_seq, X_str, X_other, y, test_size=0.2, random_state=SEED
    )
else:
    X_seq_train, X_seq_test, X_other_train, X_other_test, y_train, y_test = train_test_split(
        X_seq, X_other, y, test_size=0.2, random_state=SEED
    )

### model ###
seq_input = layers.Input(shape=(maxlen,))
x = layers.Embedding(input_dim=len(token2idx) + 1, output_dim=32)(seq_input)
x = layers.Conv1D(64, 5, activation='relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)

other_input = layers.Input(shape=(X_other.shape[1],))

if STRUCTURE:
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
output = layers.Dense(1, activation='sigmoid')(dense)

model = models.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### callbacks ###
best_model_path = os.path.join(save_path_model, f"{NAME_MOD}_best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path, save_best_only=True, monitor="val_loss", mode="min"
)

### train ###
if STRUCTURE:
    model.fit(
        [X_seq_train, X_str_train, X_other_train], y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        callbacks=[checkpoint_cb]
    )
else:
    model.fit(
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
}, preproc_path)
print("✔ preprocessing artifacts saved")

###############
### visuals ###
###############

save_path_vis = os.path.join(RESULTSPATH, 'visuals', subfolder, data, strain, segment, intersects)
os.makedirs(save_path_vis, exist_ok=True)

if BEST_MODEL:
    model_path = best_model_path
else:
    model_path = final_model_path

### ROC & F1 ###
plt.style.use('seaborn-darkgrid')
plt.rc('font', size=12)

model.load_weights(model_path)

if STRUCTURE:
    y_pred_proba = model.predict([X_seq_test, X_str_test, X_other_test]).ravel()
else:
    y_pred_proba = model.predict([X_seq_test, X_other_test]).ravel()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)
auc_score = round(auc_score, DECIMALS)

roc_color = COLORS[8]

y_pred_label = (y_pred_proba >= 0.5).astype(int)
f1 = f1_score(y_test, y_pred_label)
f1 = round(f1, DECIMALS)

title_name = f'ROC - curve plot (F1={f1})'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
dvg_count = df.shape[0]
title_name += f' (n={dvg_count})'

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc_score})", linewidth=2, color=roc_color)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="grey")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title(title_name)
plt.legend(loc="lower right")
plt.grid(True)

roc_path = os.path.join(save_path_vis, f"{NAME_MOD}_roc.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n✔ ROC curve saved")

### SHAP ###
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

bg_n = min(200, X_train_shap.shape[0])
eval_n = min(100, X_test_shap.shape[0])
X_bg = X_train_shap[:bg_n]
X_eval = X_test_shap[:eval_n]

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
      .head(20)
)

bar_color = COLORS[8]

title_name = 'top 20 SHAP importances via RandomForest - bar plot'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
dvg_count = df.shape[0]
title_name += f' (n={dvg_count})'

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color=bar_color, edgecolor='white', linewidth=1)
plt.xlabel("mean |SHAP value|")
plt.ylabel("features")
plt.title(title_name)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis="x")

shap_path = os.path.join(save_path_vis, f"{NAME_MOD}_shap.png")
plt.savefig(shap_path, dpi=300, bbox_inches="tight")
plt.close()
print("✔ SHAP bar chart saved")
