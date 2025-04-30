'''

'''
import os
import sys
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names
from utils import one_hot_encode, add_dvg_sequence
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


def save_encoders_scaler(strain_enc, segment_enc, scaler, save_dir):
    """
    Speichert LabelEncoder und Scaler als Pickle-Dateien.

    Args:
        strain_enc: LabelEncoder für 'Strain'
        segment_enc: LabelEncoder für 'Segment'
        scaler: MinMaxScaler für Start und End
        save_dir: Ordnerpfad, wo gespeichert werden soll
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "strain_encoder.pkl"), "wb") as f:
        pickle.dump(strain_enc, f)
    
    with open(os.path.join(save_dir, "segment_encoder.pkl"), "wb") as f:
        pickle.dump(segment_enc, f)
    
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

###########
### cnn ###
##########

# --- Daten laden ---
# coordinates = "IAV"
# dfname = coordinates
# dfnames = get_dataset_names(cutoff=40, selection=coordinates)
# dfs, _ = load_all(dfnames, False)

# df = pd.concat(dfs, axis=0)
# df = add_dvg_sequence(df)

dfnames = ["Alnaji2021"]
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

# --- Sicherstellen, dass alle Sequenzen gleiche Länge haben ---
# Falls nötig, Padding hier einbauen (geht nicht aus Code hervor)
# Annahme hier: bereits gleiche Länge

# --- Metadaten als Vektor ---
X_meta = np.stack([
    df["strain_enc"].values,
    df["segment_enc"].values,
    df["start_scaled"].values,
    df["end_scaled"].values
], axis=1)  # shape: (n, 4)

# --- Label ---
y = df["label"].values

# Beispielsweise nach dem Training:
enc_save_path = os.path.join(RESULTSPATH, "networks/cnn/binary")
save_encoders_scaler(strain_enc, segment_enc, scaler, enc_save_path)

# --- Train/Test Split ---
X_seq_train, X_seq_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_seq, X_meta, y, test_size=0.2, random_state=42, stratify=y
)

# --- CNN Input ---
input_seq = Input(shape=(X_seq.shape[1], 4), name="sequence_input")
x = Conv1D(64, 5, activation='relu', padding='same')(input_seq)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.4)(x)

x = Conv1D(256, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.4)(x)

x = Flatten()(x)

# --- Meta Input ---
input_meta = Input(shape=(X_meta.shape[1],), name="meta_input")

# --- Kombinieren ---
combined = Concatenate()([x, input_meta])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(64, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

# --- Modell definieren ---
model = Model(inputs=[input_seq, input_meta], outputs=z)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# --- Training ---
model.fit(
    [X_seq_train, X_meta_train], y_train,
    validation_data=([X_seq_test, X_meta_test], y_test),
    epochs=50, batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# --- Speichern ---
save_path = os.path.join(RESULTSPATH, f"networks/cnn/binary")
os.makedirs(save_path, exist_ok=True)
fname = f"cnn_bin_del_prim_1.h5"
model.save(os.path.join(save_path, fname))