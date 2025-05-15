'''

'''
import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, "..")
from utils import load_all
from utils import one_hot_encode
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

# --- Klassenziel: hoch/niedrig ---
threshold = df["NGS_read_count"].median()
df["label"] = (df["NGS_read_count"] > threshold).astype(int)

X = np.array([one_hot_encode(seq) for seq in df["deleted_sequence"]])
y = df["label"].values

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- CNN Modell ---
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X.shape[1], 4)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # für binäre Klassifikation
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# --- Training ---
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# --- Speichern ---
save_path = os.path.join(RESULTSPATH, f"networks/cnn/binary")
os.makedirs(save_path, exist_ok=True)
fname = f"cnn_bin_del_test_1.h5"
model.save(os.path.join(save_path, fname))
