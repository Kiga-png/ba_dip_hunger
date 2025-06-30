'''

'''
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, "..")
from utils import RESULTSPATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

RESULTSPATH, _ = os.path.split(RESULTSPATH)
NAME_MOD = "IBV_v1"

save_path = os.path.join(RESULTSPATH, "networks", "reg")
os.makedirs(save_path, exist_ok=True)


# === Load Data ===
read_path = os.path.join(RESULTSPATH, "dfs", "IBV", "all", "all")
fname = "nn" + ".csv"
df = pd.read_csv(os.path.join(read_path, fname))

# === Structure Feature Extraction ===
def extract_structure_features(dot_bracket_series):
    features = []
    for structure in dot_bracket_series:
        paired = structure.count('(') + structure.count(')')
        unpaired = structure.count('.')
        max_loop = max([len(s) for s in structure.split('(') + structure.split(')') if '.' in s] + [0])
        loop_count = structure.count('.')  # proxy
        features.append([paired, unpaired, max_loop, loop_count])
    return np.array(features)

# === k-mer Encoding ===
def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def encode_kmers(seq_series, k=3):
    all_kmers = set()
    seqs_kmers = []

    for seq in seq_series:
        kmers = get_kmers(seq, k)
        all_kmers.update(kmers)
        seqs_kmers.append(kmers)

    kmer_list = sorted(list(all_kmers))
    kmer_index = {k: i for i, k in enumerate(kmer_list)}

    encoded = np.zeros((len(seq_series), len(kmer_list)))
    for i, kmers in enumerate(seqs_kmers):
        for kmer in kmers:
            if kmer in kmer_index:
                encoded[i, kmer_index[kmer]] += 1
    return encoded

# === One-Hot Encoding: Segment & Strain ===
cat_features = ['Segment', 'Strain']
encoder = OneHotEncoder(sparse=False)
encoded_cat = encoder.fit_transform(df[cat_features])

# === Extract Structure Features ===
structure_features = extract_structure_features(df['Structure'])

# === Numerical Features ===
num_features = df[['Start', 'End', 'MFE']].values

# === Motif Features ===
motif_features = ['motif0', 'motif1', 'motif2', 'motif3']
motifs_flat = df[motif_features].agg(''.join, axis=1)
motif_kmer_encoded = encode_kmers(motifs_flat, k=2)

# === Sequence k-mers ===
kmer_encoded_seq = encode_kmers(df['dvg_sequence'], k=3)

# === Combine Features ===
X = np.hstack([
    encoded_cat,
    structure_features,
    num_features,
    motif_kmer_encoded,
    kmer_encoded_seq
])
y = df['NGS_read_count'].values.astype(np.float32)

# === Normalize Features ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Save Scaler ===
scaler_path = os.path.join(save_path, f"{NAME_MOD}_scaler.save")
joblib.dump(scaler, scaler_path)
print(f"✔ Scaler saved to: {scaler_path}")

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Build TensorFlow Model ===
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === Save Best Model Automatically ===
best_model_path = os.path.join(save_path, f"{NAME_MOD}_best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path, save_best_only=True, monitor="val_loss", mode="min"
)

# === Train Model ===
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint_cb],
    verbose=1
)

# === Final Save (last model state) ===
final_model_path = os.path.join(save_path, f"{NAME_MOD}_final_model.h5")
model.save(final_model_path)

print(f"\n✔ Modelle gespeichert:")
print(f"- Finales Modell: {final_model_path}")
print(f"- Bestes Modell: {best_model_path}")

# === Evaluate Model ===
loss, mae = model.evaluate(X_test, y_test)
predictions = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, predictions)

print(f"\n📊 Test MSE: {mse:.4f}")