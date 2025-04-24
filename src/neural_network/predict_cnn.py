import os
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, "..")
from utils import load_all, one_hot_encode, add_dvg_sequence
from utils import RESULTSPATH

RESULTSPATH, _ = os.path.split(RESULTSPATH)


###########
### cnn ###
###########

# --- Modellpfad ---
save_path = os.path.join(RESULTSPATH, f"networks/cnn/binary")
os.makedirs(save_path, exist_ok=True)
fname = f"cnn_bin_del_prim_1.h5"
model_path = os.path.join(save_path, fname)

# --- Daten laden ---
dfnames = ["Berry2021_B_Yam"]
dfs, _ = load_all(dfnames)
df = dfs[0]
df = add_dvg_sequence(df)

# --- Label erstellen ---
threshold = df["NGS_read_count"].median()
df["label"] = (df["NGS_read_count"] > threshold).astype(int)

# --- Kategorische Features ---
strain_enc = LabelEncoder()
segment_enc = LabelEncoder()
df["strain_enc"] = strain_enc.fit_transform(df["Strain"])
df["segment_enc"] = segment_enc.fit_transform(df["Segment"])

# --- Numerische Features skalieren ---
scaler = MinMaxScaler()
df[["start_scaled", "end_scaled"]] = scaler.fit_transform(df[["Start", "End"]])

# --- Eingabedaten vorbereiten ---
X_seq = np.array([one_hot_encode(seq) for seq in df["dvg_sequence"]])
X_meta = np.stack([
    df["strain_enc"].values,
    df["segment_enc"].values,
    df["start_scaled"].values,
    df["end_scaled"].values
], axis=1)
y = df["label"].values

# --- Split Testdaten (optional: retrain-test-split für saubere Trennung) ---
_, X_seq_test, _, X_meta_test, _, y_test = train_test_split(
    X_seq, X_meta, y, test_size=0.2, random_state=42
)

# --- Modell laden ---
model = load_model(model_path)

# --- Vorhersage ---
y_pred = model.predict([X_seq_test, X_meta_test])
y_pred_binary = (y_pred.flatten() > 0.5).astype(int)

# --- Ergebnisse ---
print("Test Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))