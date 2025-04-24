'''

'''
import os
import sys

import numpy as np

from tensorflow.keras.models import load_model

sys.path.insert(0, "..")#
from utils import load_all
from utils import one_hot_encode
from utils import DATAPATH, RESULTSPATH, DATASET_STRAIN_DICT, CUTOFF, SEGMENTS

RESULTSPATH, _ = os.path.split(RESULTSPATH)


###########
### cnn ###
###########

# Modell laden
save_path = os.path.join(RESULTSPATH, f"networks/cnn/binary")
os.makedirs(save_path, exist_ok=True)
fname = f"cnn_bin_del_test_1.h5"
model = load_model(os.path.join(save_path, fname))

# Beispiel: neue Sequenz
new_seq = "AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC"
encoded = one_hot_encode(new_seq)
X_new = np.expand_dims(encoded, axis=0)  # batch dimension

# Vorhersage
prob = model.predict(X_new)[0][0]
print(f"Vorhergesagte Wahrscheinlichkeit für 'high count': {prob:.3f}")
