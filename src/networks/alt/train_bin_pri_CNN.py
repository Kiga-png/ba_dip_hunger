import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from utils import get_dataset_names, load_all
from utils import manage_specifiers, load_all_preprocessed, merge_missing_features, save_df
from utils import balance_by_threshold

from utils import RESULTSPATH, SEED, DATASET_STRAIN_DICT
from utils import COLORS

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

VERSION = 0
KMER_SIZE = 3
THRESHOLD = 0.5

###########
### CNN ###
###########

### load & save ###
folder = 'training'
subfolder = 'pri'

data = 'IAV'

strain = 'PR8'
### for single dataset ###
# strain = DATASET_STRAIN_DICT[data]

segment = 'PB1'
intersects = 'median'

motif_length = 3
fname = f'motif_length_{motif_length}'
read_path = os.path.join(RESULTSPATH, folder, subfolder, data, strain, segment, intersects)
df = pd.read_csv(os.path.join(read_path, f'{fname}.csv'), keep_default_na=False, na_values=[])

### otional ###
#df = balance_by_threshold(df, 'norm_log_NGS_read_count', THRESHOLD)

NAME_MOD = f'bin_pri_CNN_v{VERSION}_motif_length_{motif_length}'

save_path = os.path.join(RESULTSPATH, 'model', subfolder, data, strain, segment, intersects)
os.makedirs(save_path, exist_ok=True)

### key parameter ####
df["label"] = (df["norm_log_NGS_read_count"] > THRESHOLD).astype(int)

### k-mer token ###
def kmer_tokenize(seq, k):
    return [seq[i:i + k] for i in range(len(seq) - k + 1) if 'X' not in seq[i:i + k]]

vocab = set()
df["kmer_seq"] = df["marked_dvg_sequence"].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
df["kmer_seq"].apply(lambda kmers: vocab.update(kmers))
vocab = sorted(vocab)
token2idx = {k: i + 1 for i, k in enumerate(vocab)}

### sequence to integer ###
def encode_seq(kmers):
    return [token2idx[k] for k in kmers if k in token2idx]

df["encoded_seq"] = df["kmer_seq"].apply(encode_seq)
maxlen = df["encoded_seq"].apply(len).max()
X_seq = tf.keras.preprocessing.sequence.pad_sequences(df["encoded_seq"], maxlen=maxlen)

### categorical features ###
categorical_cols = ["Segment", "Strain"] 
categorical_cols += ["site1_motif", "site2_motif", "site3_motif", "site4_motif"]
encoder = OneHotEncoder(sparse=False)
X_cat = encoder.fit_transform(df[categorical_cols])

### numerical features ###
numerical_cols = ["Start", "End", "dvg_length", "deletion_length", "5_end_length", "3_end_length", "cg_content"]
numerical_cols += [col for col in df.columns if re.search(r"motif\d+", col)]
scaler = StandardScaler()
X_num = scaler.fit_transform(df[numerical_cols])

### combine features ###
X_other = np.hstack([X_cat, X_num])
y = df["label"].values

### train/test split ###
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

concat = layers.concatenate([x, other_input])
dense = layers.Dense(64, activation='relu')(concat)
dense = layers.Dropout(0.3)(dense)
output = layers.Dense(1, activation='sigmoid')(dense)

model = models.Model(inputs=[seq_input, other_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### callbacks ###
best_model_path = os.path.join(save_path, f"{NAME_MOD}_best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path, save_best_only=True, monitor="val_loss", mode="min"
)

### train ###
model.fit(
    [X_seq_train, X_other_train], y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[checkpoint_cb]
)

### save ###
final_model_path = os.path.join(save_path, f"{NAME_MOD}_final_model.h5")
model.save(final_model_path)
print(f"\n✔ models saved")

scaler_path = os.path.join(save_path, f"{NAME_MOD}_scaler.save")
joblib.dump(scaler, scaler_path)
print(f"✔ scaler saved")

###############
### visuals ###
###############

### ROC & F1 ###
plt.style.use("seaborn")
plt.rc("font", size=12)

y_pred_proba = model.predict([X_seq_test, X_other_test])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

roc_color = COLORS[4]

y_pred_label = (y_pred_proba >= 0.5).astype(int)
f1 = f1_score(y_test, y_pred_label)

title_name = f'ROC (F1: {f1})'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
dvg_count = df.shape[0]
title_name += f' (n={dvg_count})'

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})", linewidth=2, color=roc_color)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="grey")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title(title_name)
plt.legend(loc="lower right")
plt.grid(True)

roc_path = os.path.join(save_path, f"{NAME_MOD}_roc.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n✔ ROC curve saved")

### SHAP ###
n_cat_features = X_cat.shape[1]
X_num_train_shap = X_other_train[:, n_cat_features:]
X_num_test_shap  = X_other_test[:,  n_cat_features:]

rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
rf.fit(X_num_train_shap, y_train)

assert X_num_train_shap.shape[1] == X_num_test_shap.shape[1], \
    f"train/test mismatch: {X_num_train_shap.shape[1]} vs {X_num_test_shap.shape[1]}"
assert getattr(rf, "n_features_in_", X_num_test_shap.shape[1]) == X_num_test_shap.shape[1], \
    "model expects a different number of features than provided X_other_test."

bg_n = min(200, X_num_train_shap.shape[0])
eval_n = min(100, X_num_test_shap.shape[0])
X_bg = X_num_train_shap[:bg_n]
X_eval = X_num_test_shap[:eval_n]

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

feature_names = numerical_cols
if len(feature_names) != mean_abs_shap.shape[0]:
    feature_names = [f"f{i}" for i in range(mean_abs_shap.shape[0])]

importance_df = (
    pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
      .sort_values("importance", ascending=False)
      .head(20)
)

bar_color = COLORS[4]
edge_color = COLORS[4]

title_name = 'top 20 SHAP importances via RandomForest'
title_name += f'\ndata: {data}'
title_name += f', strain: {strain}'
title_name += f', segment: {segment}'
title_name += f', {intersects} intersects'
dvg_count = df.shape[0]
title_name += f' (n={dvg_count})'

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color=bar_color, edgecolor=edge_color, linewidth=1)
plt.xlabel("mean (SHAP value)")
plt.ylabel("numerical feature (-)")
plt.title(title_name)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis="x")

shap_path = os.path.join(save_path, f"{NAME_MOD}_shap.png")
plt.savefig(shap_path, dpi=300, bbox_inches="tight")
plt.close()
print("✔ SHAP bar chart saved")
