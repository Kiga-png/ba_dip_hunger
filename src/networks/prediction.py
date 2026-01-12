
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from utils import RESULTSPATH
from utils import K_MER_LENGTH
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing as keras_preproc
import joblib

###############
### helpers ###
###############

def ensure_preprocess_root(current_resultspath):
    parent, last = os.path.split(current_resultspath)
    if last != "preprocess":
        return os.path.join(parent, "preprocess")
    return current_resultspath

def kmer_tokenize(seq, k):
    s = "" if pd.isna(seq) else str(seq)
    return [s[i:i + k] for i in range(len(s) - k + 1)]

def encode_with_vocab(kmers, token2idx):
    return [token2idx[k] for k in kmers if k in token2idx]

def pad_sequences(seqs, maxlen):
    return keras_preproc.sequence.pad_sequences(seqs, maxlen=maxlen)

def has_X(series, sample_rows=500):
    if series.empty:
        return False
    s = series.astype(str)
    if len(s) > sample_rows:
        s = s.sample(sample_rows, random_state=0)
    return s.str.contains('X').any()

def identify_marked_plain_columns(df):
    for col in ("marked_dvg_sequence", "marked_structure", "dvg_sequence", "structure"):
        if col not in df.columns:
            df[col] = ""
    return {
        "marked": ("marked_dvg_sequence", "marked_structure"),
        "plain": ("dvg_sequence", "structure"),
    }

def model_vocab_looks_marked(token2idx):
    for kmer in token2idx.keys():
        if 'X' in kmer:
            return True
    return False

def choose_sequence_columns(df, artifacts, force_marked=None):
    token2idx = artifacts["token2idx"]
    STRUCTURE = int(artifacts.get("STRUCTURE", 0))
    cols = identify_marked_plain_columns(df)

    if force_marked is True:
        seq_col, str_col = cols["marked"]
        print("ℹ Using FORCED columns: marked_*")
        return seq_col, (str_col if STRUCTURE else None)
    if force_marked is False:
        seq_col, str_col = cols["plain"]
        print("ℹ Using FORCED columns: unmarked")
        return seq_col, (str_col if STRUCTURE else None)

    use_marked = model_vocab_looks_marked(token2idx)
    choice = "marked" if use_marked else "plain"
    seq_col, str_col = cols[choice]
    print(f"ℹ Auto-selected columns by model vocab: {seq_col}"
          f"{' + ' + str_col if STRUCTURE else ''} (choice={choice})")
    return seq_col, (str_col if STRUCTURE else None)

def detect_task(model):
    try:
        act = getattr(model.layers[-1], "activation", None)
        act_name = getattr(act, "__name__", None)
    except Exception:
        act_name = None
    return "classification" if act_name == "sigmoid" else "regression"

def build_paths_flipped(preprocess_root, folder, data, strain, segment, intersects, name_mod):
    root_parent = os.path.dirname(preprocess_root)

    read_dir = os.path.join(preprocess_root, folder, "preparation", data, strain, segment, intersects)
    model_dir = os.path.join(root_parent, "networks", "CNN", "kmer", data, strain, segment, intersects)

    preproc_path = os.path.join(model_dir, f"{name_mod}_preproc.joblib")
    best_path    = os.path.join(model_dir, f"{name_mod}_best_model.h5")
    final_path   = os.path.join(model_dir, f"{name_mod}_final_model.h5")

    write_dir = os.path.join(preprocess_root, folder, "prediction", data, strain, segment, intersects)
    os.makedirs(write_dir, exist_ok=True)

    return {
        "read_dir": read_dir,
        "preproc_path": preproc_path,
        "best_path": best_path,
        "final_path": final_path,
        "write_dir": write_dir,
    }

def load_artifacts_and_model(preproc_path, best_path, final_path, model_choice):
    if not os.path.isfile(preproc_path):
        raise FileNotFoundError(f"Preprocessing artifacts not found: {preproc_path}")

    artifacts = joblib.load(preproc_path)

    if model_choice == "best" and os.path.isfile(best_path):
        model_path = best_path
    elif model_choice == "final" and os.path.isfile(final_path):
        model_path = final_path
    elif os.path.isfile(best_path):
        model_path = best_path
    elif os.path.isfile(final_path):
        model_path = final_path
    else:
        raise FileNotFoundError(f"No model file found. Checked: {best_path} and {final_path}")

    model = load_model(model_path)
    return artifacts, model, model_path

def assemble_features(df, artifacts, seq_col, struct_col):
    token2idx  = artifacts["token2idx"]
    token2idx2 = artifacts.get("token2idx2", {})
    maxlen  = int(artifacts["maxlen"])
    maxlen2 = int(artifacts.get("maxlen2", 0))
    categorical_cols = list(artifacts["categorical_cols"])
    numerical_cols   = list(artifacts["numerical_cols"])
    encoder = artifacts["encoder"]
    scaler  = artifacts["scaler"]
    STRUCTURE = int(artifacts.get("STRUCTURE", 0))

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0.0

    if len(token2idx) > 0:
        any_kmer = next(iter(token2idx.keys()))
        K = len(any_kmer)
    else:
        K = 3

    df["_kmer_seq"] = df[seq_col].apply(lambda s: kmer_tokenize(s, K))
    df["_enc_seq"]  = df["_kmer_seq"].apply(lambda km: encode_with_vocab(km, token2idx))
    X_seq = pad_sequences(df["_enc_seq"], maxlen=maxlen)

    X_str = None
    if STRUCTURE:
        if struct_col is None:
            raise ValueError("STRUCTURE=1 in artifacts but no structure column selected.")
        if not token2idx2:
            raise ValueError("STRUCTURE=1 in artifacts but token2idx2 is empty.")
        df["_kmer_str"] = df[struct_col].apply(lambda s: kmer_tokenize(s, K))
        df["_enc_str"]  = df["_kmer_str"].apply(lambda km: encode_with_vocab(km, token2idx2))
        X_str = pad_sequences(df["_enc_str"], maxlen=maxlen2)

    X_cat = encoder.transform(df[categorical_cols])
    X_num = scaler.transform(df[numerical_cols].astype(float).fillna(0.0))
    X_other = np.hstack([X_cat, X_num])

    for c in ("_kmer_seq", "_enc_seq", "_kmer_str", "_enc_str"):
        if c in df.columns:
            del df[c]

    return X_seq, X_str, X_other, STRUCTURE

############
### main ###
############

if __name__ == "__main__":
    '''

    '''

    #################
    ### SELECTION ###
    #################
    
    ### COMBINED MULTI ###

    folder = 'pooled'
    data      = "IAV"
    strain    = "PR8"
    segment   = "PB1"
    intersects = "median"

    motif_length = K_MER_LENGTH
    version      = "0"
    NAME_MOD     = f"{version}_motif_length_{motif_length}"

    model_choice = "final"
    input_csv_name = None    # optional override filename

    FORCE_MARKED = None

    ### main ###

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    preprocess_root = ensure_preprocess_root(RESULTSPATH)

    paths = build_paths_flipped(preprocess_root, folder, data, strain, segment, intersects, NAME_MOD)

    if input_csv_name is None:
        in_csv = os.path.join(paths["read_dir"], f"motif_length_{motif_length}.csv")
    else:
        in_csv = os.path.join(paths["read_dir"], input_csv_name)
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv, keep_default_na=False, na_values=[])

    artifacts, model, model_path = load_artifacts_and_model(
        paths["preproc_path"], paths["best_path"], paths["final_path"], model_choice
    )

    TASK = detect_task(model)
    seq_col, struct_col = choose_sequence_columns(df, artifacts, force_marked=FORCE_MARKED)

    X_seq, X_str, X_other, STRUCTURE = assemble_features(df, artifacts, seq_col, struct_col)

    if STRUCTURE:
        y_pred = model.predict([X_seq, X_str, X_other], verbose=0).ravel()
    else:
        y_pred = model.predict([X_seq, X_other], verbose=0).ravel()

    out_col = "NGS_prediction" if TASK == "classification" else "NGS_value"
    if TASK == "classification":
        y_pred = np.clip(y_pred, 0.0, 1.0)
    df[out_col] = y_pred

    base_name = os.path.splitext(os.path.basename(in_csv))[0]
    out_csv = os.path.join(paths["write_dir"], f"{base_name}.csv")
    df.to_csv(out_csv, index=False)

    print(f"✔ Loaded model: {model_path}")
    print(f"✔ Read from:    {in_csv}")
    print(f"✔ Wrote to:     {out_csv}")
    print(f"✔ Task:         {TASK} | STRUCTURE={STRUCTURE} | Columns: {seq_col}"
          f"{' + ' + struct_col if STRUCTURE and struct_col else ''}")
