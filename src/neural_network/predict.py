'''

'''
import os
import sys

import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

sys.path.insert(0, "..")
from utils import load_all
from utils import one_hot_encode, add_dvg_sequence
from utils import RESULTSPATH

RESULTSPATH, _ = os.path.split(RESULTSPATH)

# --- Hilfsfunktionen ---

def load_encoders_scaler(load_dir):
    with open(os.path.join(load_dir, "strain_encoder.pkl"), "rb") as f:
        strain_enc = pickle.load(f)
    with open(os.path.join(load_dir, "segment_encoder.pkl"), "rb") as f:
        segment_enc = pickle.load(f)
    with open(os.path.join(load_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return strain_enc, segment_enc, scaler

def prepare_inputs(df, strain_enc, segment_enc, scaler):
    df = df.copy()
    df["strain_enc"] = strain_enc.transform(df["Strain"])
    df["segment_enc"] = segment_enc.transform(df["Segment"])
    df[["start_scaled", "end_scaled"]] = scaler.transform(df[["Start", "End"]])

    X_seq = np.array([one_hot_encode(seq) for seq in df["dvg_sequence"]])
    X_meta = np.stack([
        df["strain_enc"].values,
        df["segment_enc"].values,
        df["start_scaled"].values,
        df["end_scaled"].values
    ], axis=1)

    return X_seq, X_meta

def predict_and_save(model_path, enc_path, output_csv_path):
    # --- Modell und Tools laden ---
    model = load_model(model_path)
    strain_enc, segment_enc, scaler = load_encoders_scaler(enc_path)

    # --- Neue Daten laden ---
    dfnames = ["Pelz2021"]
    dfs, _ = load_all(dfnames)
    dfname = dfnames[0]

    df = dfs[0]
    df = add_dvg_sequence(df)

    # --- Eingabedaten vorbereiten ---
    X_seq, X_meta = prepare_inputs(df, strain_enc, segment_enc, scaler)

    # --- Vorhersagen ---
    preds = model.predict([X_seq, X_meta], batch_size=256, verbose=0)
    preds_binary = (preds.flatten() > 0.5).astype(int)

    # --- Ergebnisse speichern ---
    df["predicted_probability"] = preds.flatten()
    df["predicted_label"] = preds_binary

    df.to_csv(output_csv_path, index=False)
    print(f"Vorhersagen gespeichert in: {output_csv_path}")

# --- Hauptteil ---

if __name__ == "__main__":
    # Pfade definieren
    save_dir = os.path.join(RESULTSPATH, "networks/cnn/binary")
    model_path = os.path.join(save_dir, "cnn_bin_del_prim_1.h5")

    output_csv_path = os.path.join(save_dir, "cnn_preidction.csv")  # <- hier werden Vorhersagen gespeichert

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Vorhersage durchführen
    predict_and_save(model_path, save_dir, output_csv_path)
