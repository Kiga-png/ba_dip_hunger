import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc

# Reproduzierbarkeit
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Eigene Utils
sys.path.insert(0, "..")
from utils import load_all, get_dataset_names
from utils import one_hot_encode, add_dvg_sequence, add_norm_log_ngs_read_count
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)
NAME_MOD = "IBV"

def save_encoders_scaler(strain_enc, segment_enc, scaler, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"strain_encoder_{NAME_MOD}.pkl"), "wb") as f:
        pickle.dump(strain_enc, f)
    with open(os.path.join(save_dir, f"segment_encoder_{NAME_MOD}.pkl"), "wb") as f:
        pickle.dump(segment_enc, f)
    with open(os.path.join(save_dir, f"scaler_{NAME_MOD}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

# --- Daten laden ---
selector = "IBV"
dfnames = get_dataset_names(cutoff=40, selection=selector)
dfs, _ = load_all(dfnames, False)
df = pd.concat(dfs, ignore_index=True)

df = add_dvg_sequence(df)
df = add_norm_log_ngs_read_count(df)

# --- Quartil-Labeling ---
q1 = df['norm_log_NGS_read_count'].quantile(0.25)
q3 = df['norm_log_NGS_read_count'].quantile(0.75)
df = df[(df['norm_log_NGS_read_count'] <= q1) | (df['norm_log_NGS_read_count'] >= q3)]
df["label"] = (df['norm_log_NGS_read_count'] >= q3).astype(int)

# --- Feature-Encoding ---
strain_enc = LabelEncoder()
segment_enc = LabelEncoder()
df["strain_enc"] = strain_enc.fit_transform(df["Strain"])
df["segment_enc"] = segment_enc.fit_transform(df["Segment"])

scaler = MinMaxScaler()
df[["start_scaled", "end_scaled"]] = scaler.fit_transform(df[["Start", "End"]])

# --- Sequenz-Encoding + Padding ---
X_seq_raw = [one_hot_encode(seq) for seq in df["dvg_sequence"]]
max_len = max(len(seq) for seq in X_seq_raw)
X_seq = pad_sequences(X_seq_raw, maxlen=max_len, padding="post", dtype="float32")

# --- Metadaten-Features ---
X_meta = np.stack([
    df["strain_enc"].values,
    df["segment_enc"].values,
    df["start_scaled"].values,
    df["end_scaled"].values
], axis=1)

y = df["label"].values

# --- Datenverteilung prüfen ---
print("Label counts:", np.bincount(y))

# --- Train/Test-Split ---
X_seq_train, X_seq_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_seq, X_meta, y, test_size=0.2, stratify=y, random_state=SEED
)

# --- Manuelle Class Weights ---
class_weight_dict = {0: 1.5, 1: 1.0}
print("Manuell gesetzte Class Weights:", class_weight_dict)

# --- CNN-Modell (vereinfacht + stärker reguliert) ---
input_seq = Input(shape=(X_seq.shape[1], 4), name="sequence_input")
x = Conv1D(32, 5, activation='relu', padding='same')(input_seq)
x = MaxPooling1D(2)(x)
x = Flatten()(x)

input_meta = Input(shape=(X_meta.shape[1],), name="meta_input")
combined = Concatenate()([x, input_meta])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[input_seq, input_meta], outputs=z)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# --- Training ---
history = model.fit(
    [X_seq_train, X_meta_train], y_train,
    validation_data=([X_seq_test, X_meta_test], y_test),
    epochs=50, batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# --- Evaluation ---
y_pred_probs = model.predict([X_seq_test, X_meta_test])

# Wahrscheinlichkeitsverteilung visualisieren
plt.figure()
plt.hist(y_pred_probs, bins=50)
plt.title("Verteilung der Vorhersagewahrscheinlichkeiten")
plt.xlabel("Wahrscheinlichkeit (Klasse 1)")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.show()

# Schwellenoptimierung für F1
prec, rec, thresh = precision_recall_curve(y_test, y_pred_probs)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
best_thresh = thresh[np.argmax(f1)]
print("Beste Schwelle für F1:", best_thresh)

# Neue Vorhersage
y_pred = (y_pred_probs > best_thresh).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:")
print(roc_auc_score(y_test, y_pred_probs))

# F1 vs Threshold plot
plt.figure()
plt.plot(thresh, f1[:-1])  # Letzter Threshold hat keinen F1-Wert
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.title("F1-Score vs. Entscheidungs-Schwelle")
plt.grid(True)
plt.tight_layout()
plt.show()

# ROC-Kurve
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve")
plt.legend()
plt.tight_layout()
plt.show()

# PR-Kurve
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall-Kurve")
plt.tight_layout()
plt.show()

# --- Speichern ---
enc_save_path = os.path.join(RESULTSPATH, "networks/cnn/binary")
save_encoders_scaler(strain_enc, segment_enc, scaler, enc_save_path)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(enc_save_path, f"cnn_bin_del_{NAME_MOD}_{timestamp}.h5")
model.save(model_path)
