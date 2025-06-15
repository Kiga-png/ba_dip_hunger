import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# own utils
sys.path.insert(0, "..")
from utils import load_all, get_dataset_names
from utils import add_dvg_sequence, add_norm_log_ngs_read_count, add_gc_content_dvg_sequnce, add_nucleotide_count_dvg_sequence
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
NAME_MOD = "v6_IAV"

def save_encoders_scaler(strain_enc, segment_enc, scaler, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"strain_encoder_{NAME_MOD}.pkl"), "wb") as f:
        pickle.dump(strain_enc, f)
    with open(os.path.join(save_dir, f"segment_encoder_{NAME_MOD}.pkl"), "wb") as f:
        pickle.dump(segment_enc, f)
    with open(os.path.join(save_dir, f"scaler_{NAME_MOD}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

# --- k-mer helper functions ---
def generate_kmer_vocab(k=3, alphabet="ACGUX"):
    return [''.join(p) for p in product(alphabet, repeat=k)]

def build_kmer_index(vocab):
    return {kmer: i + 1 for i, kmer in enumerate(vocab)}  # 0 = padding

def seq_to_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def encode_kmer_sequences(sequences, kmer_to_index, k=3):
    encoded = []
    for seq in sequences:
        kmers = seq_to_kmers(seq, k)
        indices = [kmer_to_index.get(kmer, 0) for kmer in kmers]
        encoded.append(indices)
    return encoded

# --- load data ---
selector = "IAV"
dfnames = get_dataset_names(cutoff=40, selection=selector)
dfs, _ = load_all(dfnames, False)
df = pd.concat(dfs, ignore_index=True)

# Add sequence-derived features
df = add_nucleotide_count_dvg_sequence(df)
df = add_gc_content_dvg_sequnce(df)
df = add_norm_log_ngs_read_count(df)
key_feature = 'norm_log_NGS_read_count'

# --- quartile labeling ---
q1 = df[key_feature].quantile(0.25)
q3 = df[key_feature].quantile(0.75)
df = df[(df[key_feature] <= q1) | (df[key_feature] >= q3)]
df["label"] = (df[key_feature] >= q3).astype(int)

# --- feature encoding ---
strain_enc = LabelEncoder()
segment_enc = LabelEncoder()
df["strain_enc"] = strain_enc.fit_transform(df["Strain"])
df["segment_enc"] = segment_enc.fit_transform(df["Segment"])

# --- scaling ---
scaler = MinMaxScaler()
df[["start_scaled", "end_scaled", "nucleotide_count_scaled", "gc_content_scaled"]] = scaler.fit_transform(
    df[["Start", "End", "nucleotide_count", "gc_content"]]
)

# --- k-mer encoding of sequences ---
k = 3
kmer_vocab = generate_kmer_vocab(k)
kmer_to_index = build_kmer_index(kmer_vocab)

X_seq_raw = encode_kmer_sequences(df["dvg_sequence"], kmer_to_index, k)
max_len = max(len(seq) for seq in X_seq_raw)
X_seq = pad_sequences(X_seq_raw, maxlen=max_len, padding="post", value=0)

# --- metadata features ---
X_meta = np.stack([
    df["strain_enc"].values,
    df["segment_enc"].values,
    df["start_scaled"].values,
    df["end_scaled"].values,
    df["nucleotide_count_scaled"].values,
    df["gc_content_scaled"].values
], axis=1)

y = df["label"].values

# --- train/test split ---
X_seq_train, X_seq_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_seq, X_meta, y, test_size=0.2, stratify=y, random_state=SEED
)

# --- class weights ---
class_weight_dict = {0: 10, 1: 1.0}

# --- cnn with embedding ---
vocab_size = len(kmer_to_index) + 1
embedding_dim = 16

input_seq = Input(shape=(X_seq.shape[1],), name="sequence_input")
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(input_seq)
x = Conv1D(32, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)

input_meta = Input(shape=(X_meta.shape[1],), name="meta_input")
combined = Concatenate()([x, input_meta])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[input_seq, input_meta], outputs=z)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# --- callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# --- training ---
history = model.fit(
    [X_seq_train, X_meta_train], y_train,
    validation_data=([X_seq_test, X_meta_test], y_test),
    epochs=50, batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# --- evaluation ---
y_pred_probs = model.predict([X_seq_test, X_meta_test])
prec, rec, thresh = precision_recall_curve(y_test, y_pred_probs)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
best_thresh = thresh[np.argmax(f1)]

y_pred = (y_pred_probs > best_thresh).astype(int)

print("classification report:")
print(classification_report(y_test, y_pred))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC score:")
print(roc_auc_score(y_test, y_pred_probs))

# --- feature importance (permutation on metadata) ---
from copy import deepcopy

meta_feature_names = [
    "strain_enc", "segment_enc", "start_scaled", "end_scaled",
    "nucleotide_count_scaled", "gc_content_scaled"
]

y_pred_base = (model.predict([X_seq_test, X_meta_test]) > best_thresh).astype(int)
base_f1 = f1_score(y_test, y_pred_base)

feature_importances = {}
for i, name in enumerate(meta_feature_names):
    X_meta_perm = deepcopy(X_meta_test)
    np.random.shuffle(X_meta_perm[:, i])
    y_pred_perm = (model.predict([X_seq_test, X_meta_perm]) > best_thresh).astype(int)
    f1_drop = base_f1 - f1_score(y_test, y_pred_perm)
    feature_importances[name] = f1_drop
    print(f"{name}: F1 drop = {f1_drop:.4f}")

plt.figure()
plt.barh(list(feature_importances.keys()), list(feature_importances.values()))
plt.xlabel("F1 Score Decrease on Permutation")
plt.title("Permutation Feature Importance (Metadata)")
plt.tight_layout()

enc_save_path = os.path.join(RESULTSPATH, "networks/cnn/binary")
os.makedirs(enc_save_path, exist_ok=True)
importance_path = os.path.join(enc_save_path, f"feature_importance_{NAME_MOD}.png")
plt.savefig(importance_path, dpi=300)
plt.close()

# --- Save other plots and model ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(enc_save_path, f"cnn_bin_del_{NAME_MOD}.h5")
model.save(model_path)

f1_values = f1[:-1]
plt.figure()
plt.plot(thresh, f1_values, label="model", color="blue")
plt.xlabel("threshold")
plt.ylabel("f1 score")
plt.title("f1 score vs. decision threshold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(enc_save_path, f"f1_vs_threshold_{NAME_MOD}.png"), dpi=300)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"model (auc = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("roc curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(enc_save_path, f"roc_curve_{NAME_MOD}.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(rec, prec, label="model", color="blue")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(enc_save_path, f"precision_recall_{NAME_MOD}.png"), dpi=300)
plt.close()

plt.figure()
plt.hist(y_pred_probs, bins=50, color="blue", edgecolor="black", label="model")
plt.xlim(0.0, 1.0)
plt.xlabel("prediction probability (class 1)")
plt.ylabel("count")
plt.title("distribution of prediction probabilities")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(enc_save_path, f"prob_distribution_{NAME_MOD}.png"), dpi=300)
plt.close()

save_encoders_scaler(strain_enc, segment_enc, scaler, enc_save_path)
