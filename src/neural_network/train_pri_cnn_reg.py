import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from utils import RESULTSPATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# === Einstellungen ===
RESULTSPATH, _ = os.path.split(RESULTSPATH)
KMER_SIZE = 7
NAME_MOD = f"IAV_cnn_reg_motif_length_{KMER_SIZE}_v0"
save_path = os.path.join(RESULTSPATH, "networks", "cnn", "reg")
os.makedirs(save_path, exist_ok=True)

# === Daten laden ===
read_path = os.path.join(RESULTSPATH, "dfs", "IAV", "PR8", "PB1")
fname = f"nn_motif_length_{KMER_SIZE}.csv"
df = pd.read_csv(os.path.join(read_path, fname))

# === K-mer-Tokenisierung ===
def kmer_tokenize(seq, k=3):
    return [seq[i:i + k] for i in range(len(seq) - k + 1) if 'X' not in seq[i:i + k]]

vocab = set()
df["kmer_seq"] = df["dvg_sequence"].apply(lambda x: kmer_tokenize(x, KMER_SIZE))
df["kmer_seq"].apply(lambda kmers: vocab.update(kmers))
vocab = sorted(vocab)
token2idx = {k: i + 1 for i, k in enumerate(vocab)}  # +1 für Padding 0

def encode_seq(kmers):
    return [token2idx[k] for k in kmers if k in token2idx]

df["encoded_seq"] = df["kmer_seq"].apply(encode_seq)
maxlen = df["encoded_seq"].apply(len).max()
X_seq = tf.keras.preprocessing.sequence.pad_sequences(df["encoded_seq"], maxlen=maxlen)

# === Kategoriale Merkmale ===
categorical_cols = ["Segment", "Strain", "motif0", "motif1", "motif2", "motif3"]
encoder = OneHotEncoder(sparse=False)
X_cat = encoder.fit_transform(df[categorical_cols])

# === Numerische Merkmale ===
numerical_cols = [col for col in df.columns if "check" in col] + ["Start", "End", "dvg_length", "cg_content"]
scaler = StandardScaler()
X_num = scaler.fit_transform(df[numerical_cols])

# === Features kombinieren ===
X_other = np.hstack([X_cat, X_num])
y = df["norm_log_NGS_read_count"].values  # Regression Target

# === Train/Test Split ===
X_seq_train, X_seq_test, X_other_train, X_other_test, y_train, y_test = train_test_split(
    X_seq, X_other, y, test_size=0.2, random_state=42
)

# === Modell definieren ===
seq_input = layers.Input(shape=(maxlen,))
x = layers.Embedding(input_dim=len(token2idx) + 1, output_dim=32)(seq_input)
x = layers.Conv1D(64, 5, activation='relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)

other_input = layers.Input(shape=(X_other.shape[1],))
concat = layers.concatenate([x, other_input])
dense = layers.Dense(64, activation='relu')(concat)
dense = layers.Dropout(0.3)(dense)
output = layers.Dense(1)(dense)  # Regressionsausgabe

model = models.Model(inputs=[seq_input, other_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# === Callbacks ===
best_model_path = os.path.join(save_path, f"{NAME_MOD}_best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path, save_best_only=True, monitor="val_loss", mode="min"
)

# === Training ===
model.fit(
    [X_seq_train, X_other_train], y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[checkpoint_cb]
)

# === Modell speichern ===
final_model_path = os.path.join(save_path, f"{NAME_MOD}_final_model.h5")
model.save(final_model_path)
print(f"\n✔ Modelle gespeichert:")
print(f"- Finales Modell: {final_model_path}")
print(f"- Bestes Modell: {best_model_path}")

# === Skaler speichern ===
scaler_path = os.path.join(save_path, f"{NAME_MOD}_scaler.save")
joblib.dump(scaler, scaler_path)
print(f"✔ Scaler saved to: {scaler_path}")

# === Evaluation: R² Score ===
y_pred = model.predict([X_seq_test, X_other_test]).flatten()
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"Regression: True vs Predicted (R² = {r2:.2f})")
plt.grid(True)
plt.show()
