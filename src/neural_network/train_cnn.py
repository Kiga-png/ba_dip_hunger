'''

'''
import os
import sys

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")
from utils import load_all
from utils import one_hot_encode, add_dvg_sequence
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


###########
### cnn ###
###########

# --- Daten laden ---
dfnames = ["Berry2021_B_Yam"]
dfs, _ = load_all(dfnames)
dfname = dfnames[0]

df = dfs[0]
df = add_dvg_sequence(df)

# --- Label-Spalte erzeugen ---
threshold = df["NGS_read_count"].median()
df["label"] = (df["NGS_read_count"] > threshold).astype(int)

# --- Kategorische Features encoden ---
strain_enc = LabelEncoder()
segment_enc = LabelEncoder()

df["strain_enc"] = strain_enc.fit_transform(df["Strain"])
df["segment_enc"] = segment_enc.fit_transform(df["Segment"])

# --- Skaliere numerische Features ---
scaler = MinMaxScaler()
df[["start_scaled", "end_scaled"]] = scaler.fit_transform(df[["Start", "End"]])

# --- Sequenz kodieren ---
X_seq = np.array([one_hot_encode(seq) for seq in df["dvg_sequence"]])  # shape: (n, len, 4)

# --- Metadaten als Vektor ---
X_meta = np.stack([
    df["strain_enc"].values,
    df["segment_enc"].values,
    df["start_scaled"].values,
    df["end_scaled"].values
], axis=1)  # shape: (n, 4)

# --- Label ---
y = df["label"].values

# --- Train/Test Split ---
X_seq_train, X_seq_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_seq, X_meta, y, test_size=0.2, random_state=42
)

# --- CNN Input ---
input_seq = Input(shape=(X_seq.shape[1], 4), name="sequence_input")
x = Conv1D(64, 5, activation='relu')(input_seq)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)

# --- Meta Input ---
input_meta = Input(shape=(X_meta.shape[1],), name="meta_input")

# --- Kombinieren ---
combined = Concatenate()([x, input_meta])
z = Dense(64, activation='relu')(combined)
z = Dense(1, activation='sigmoid')(z)

# --- Modell definieren ---
model = Model(inputs=[input_seq, input_meta], outputs=z)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# --- Training ---
model.fit(
    [X_seq_train, X_meta_train], y_train,
    validation_data=([X_seq_test, X_meta_test], y_test),
    epochs=10, batch_size=32
)

# --- Speichern ---
save_path = os.path.join(RESULTSPATH, f"networks/cnn/binary")
os.makedirs(save_path, exist_ok=True)
fname = f"cnn_bin_del_prim_1.h5"
model.save(os.path.join(save_path, fname))