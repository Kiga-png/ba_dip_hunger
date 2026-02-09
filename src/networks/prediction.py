'''
    prediction function
'''
import os
import sys
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

from utils import RESULTSPATH, DATASET_CUTOFF, PSEUDO_DATASETS, K_MER_LENGTH
from utils import get_dataset_names, load_all_preprocessed, manage_specifiers, save_df

from analysis.visuals import DECISION_THRESHOLD

from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing as keras_preproc
import joblib

RESULTSPATH, _ = os.path.split(RESULTSPATH)
RESULTSPATH = os.path.join(RESULTSPATH, 'preprocess')

####################
### path helpers ###
####################

def ensure_preprocess_root(resultspath: str) -> str:
    """
    RESULTSPATH typically points to .../results
    this returns .../results/preprocess, unless RESULTSPATH already ends with preprocess.
    """
    parent, last = os.path.split(resultspath.rstrip("/"))
    return resultspath if last == "preprocess" else os.path.join(resultspath, "preprocess")

def build_model_paths(
    resultspath: str,
    model_type: str,
    folder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    name_mod: str,
) -> Dict[str, str]:
    """
    Returns dict with paths to preprocessing joblib and best/final models.
    """
    preprocess_root = ensure_preprocess_root(resultspath)

    read_dir = os.path.join(preprocess_root, folder, "pseudo", data, strain, segment, intersects)
    
    res_root = RESULTSPATH.replace("/preprocess", "").rstrip("/")

    if model_type == "bin":
        model_base = os.path.join(res_root, "bin_network")
    else:  # "reg"
        model_base = os.path.join(res_root, "reg_network")

    model_dir = os.path.join(model_base, folder, "CNN", data, strain, segment, intersects)

    return {
        "read_dir": read_dir,
        "model_dir": model_dir,
        "preproc": os.path.join(model_dir, f"{name_mod}_preproc.joblib"),
        "best":    os.path.join(model_dir, f"{name_mod}_best_model.h5"),
        "final":   os.path.join(model_dir, f"{name_mod}_final_model.h5"),
    }

def load_artifacts_and_model(
    paths: Dict[str, str],
    model_choice: str = "final",   # "best" | "final"
) -> Tuple[Dict[str, Any], Any, str]:
    if not os.path.isfile(paths["preproc"]):
        raise FileNotFoundError(f"Preprocessing artifacts not found: {paths['preproc']}")

    artifacts = joblib.load(paths["preproc"])

    best_path = paths["best"]
    final_path = paths["final"]

    model_path = None
    if model_choice == "best" and os.path.isfile(best_path):
        model_path = best_path
    elif model_choice == "final" and os.path.isfile(final_path):
        model_path = final_path
    elif os.path.isfile(best_path):
        model_path = best_path
    elif os.path.isfile(final_path):
        model_path = final_path

    if model_path is None:
        raise FileNotFoundError(f"No model file found. Checked: {best_path} and {final_path}")

    model = load_model(model_path, compile=False) 
    return artifacts, model, model_path

###################
### seq helpers ###
###################

def kmer_tokenize(seq: Any, k: int) -> List[str]:
    s = "" if pd.isna(seq) else str(seq)
    if k <= 0 or len(s) < k:
        return []
    return [s[i:i + k] for i in range(len(s) - k + 1)]

def encode_with_vocab(kmers: List[str], token2idx: Dict[str, int], unk_idx: Optional[int] = 1) -> List[int]:
    """
    Your newer training code typically reserves 0 for PAD and 1 for UNK.
    If unk_idx is None, unknown kmers are dropped (useful for structure vocab if trained that way).
    """
    if unk_idx is None:
        return [token2idx[k] for k in kmers if k in token2idx]
    return [token2idx.get(k, unk_idx) for k in kmers]

def pad(seqs, maxlen: int):
    return keras_preproc.sequence.pad_sequences(seqs, maxlen=maxlen)

def ensure_input_columns(df: pd.DataFrame) -> None:
    """
    Ensure canonical columns exist so selection never KeyErrors.
    """
    for col in ("DelVG_sequence", "marked_DelVG_sequence", "structure", "marked_structure"):
        if col not in df.columns:
            df[col] = ""

def model_vocab_looks_marked(token2idx: Dict[str, int]) -> bool:
    """
    Heuristic: if vocab contains X, model likely trained on marked sequences.
    """
    for kmer in token2idx.keys():
        if "X" in kmer:
            return True
    return False

def choose_sequence_columns(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    force_marked: Optional[bool] = None,  # True -> marked, False -> plain, None -> auto
) -> Tuple[str, Optional[str]]:
    ensure_input_columns(df)

    STRUCTURE = int(artifacts.get("STRUCTURE", 0))

    if force_marked is True:
        return "marked_DelVG_sequence", ("marked_structure" if STRUCTURE else None)

    if force_marked is False:
        return "DelVG_sequence", ("structure" if STRUCTURE else None)

    # auto
    use_marked = model_vocab_looks_marked(artifacts.get("token2idx", {}))
    seq_col = "marked_DelVG_sequence" if use_marked else "DelVG_sequence"
    str_col = ("marked_structure" if use_marked else "structure") if STRUCTURE else None
    return seq_col, str_col

#####################
### feature build ###
#####################

def assemble_features(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    seq_col: str,
    struct_col: Optional[str],
):
    """
    Rebuild the exact model inputs using the saved preprocessing artifacts:
      - token2idx/maxlen for sequences
      - token2idx2/maxlen2 for structure (if STRUCTURE=1)
      - encoder/scaler + categorical_cols/numerical_cols for tabular branch
    """
    token2idx = artifacts["token2idx"]
    maxlen = int(artifacts["maxlen"])

    token2idx2 = artifacts.get("token2idx2", {})
    maxlen2 = int(artifacts.get("maxlen2", 0))

    categorical_cols = list(artifacts.get("categorical_cols", []))
    numerical_cols = list(artifacts.get("numerical_cols", []))

    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]

    STRUCTURE = int(artifacts.get("STRUCTURE", 0))

    # infer K from vocab
    if len(token2idx) == 0:
        raise ValueError("token2idx in artifacts is empty; cannot infer k-mer length.")
    K = len(next(iter(token2idx.keys())))

    # sequence branch
    kmers_seq = df[seq_col].apply(lambda s: kmer_tokenize(s, K))
    enc_seq = kmers_seq.apply(lambda km: encode_with_vocab(km, token2idx, unk_idx=1))
    X_seq = pad(enc_seq, maxlen=maxlen).astype(np.int32)

    # structure branch (optional)
    X_str = None
    if STRUCTURE:
        if struct_col is None:
            raise ValueError("STRUCTURE=1 but no structure column was selected.")
        if not token2idx2:
            raise ValueError("STRUCTURE=1 but token2idx2 is missing/empty in artifacts.")
        kmers_str = df[struct_col].apply(lambda s: kmer_tokenize(s, K))
        # often structure vocab is exact; drop unknowns rather than forcing UNK
        enc_str = kmers_str.apply(lambda km: encode_with_vocab(km, token2idx2, unk_idx=None))
        X_str = pad(enc_str, maxlen=maxlen2).astype(np.int32)

    # tabular branch
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].astype(str).fillna("unknown")

    for c in numerical_cols:
        if c not in df.columns:
            df[c] = 0.0

    X_cat = encoder.transform(df[categorical_cols]) if categorical_cols else np.zeros((len(df), 0), dtype=np.float32)
    X_num = scaler.transform(df[numerical_cols].astype(float).fillna(0.0)) if numerical_cols else np.zeros((len(df), 0), dtype=np.float32)
    X_other = np.hstack([X_cat, X_num]).astype(np.float32)

    return X_seq, X_str, X_other, STRUCTURE

def detect_task(model) -> str:
    """
    If last activation is sigmoid -> classification, else regression.
    """
    try:
        act = getattr(model.layers[-1], "activation", None)
        act_name = getattr(act, "__name__", None)
    except Exception:
        act_name = None
    return "classification" if act_name == "sigmoid" else "regression"

######################
### prediction API ###
######################

def run_cnn_prediction(
    df: pd.DataFrame,
    *,
    model_type: str,           # "bin" | "reg"
    folder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    name_mod: str,
    model_choice: str = "final",
    force_marked: Optional[bool] = None,
    decision_threshold: float = 0.5,
    # output columns
    proba_col: str = "cnn_pred_proba",
    class_col: str = "cnn_pred_class",
    value_col: str = "cnn_pred_value",
) -> pd.DataFrame:
    """
    Adds predictions to df and returns df (same object).
      - bin: proba_col + class_col
      - reg: value_col

    Does NOT save. You will add saving later.
    """
    if model_type not in {"bin", "reg"}:
        raise ValueError("model_type must be 'bin' or 'reg'")

    paths = build_model_paths(
        RESULTSPATH, model_type, folder, data, strain, segment, intersects, name_mod
    )
    artifacts, model, model_path = load_artifacts_and_model(paths, model_choice=model_choice)

    seq_col, struct_col = choose_sequence_columns(df, artifacts, force_marked=force_marked)
    X_seq, X_str, X_other, STRUCTURE = assemble_features(df, artifacts, seq_col, struct_col)

    if STRUCTURE:
        y_pred = model.predict([X_seq, X_str, X_other], verbose=0).ravel()
    else:
        y_pred = model.predict([X_seq, X_other], verbose=0).ravel()

    task = detect_task(model)

    if model_type == "bin":
        # be strict: classification output
        y_pred = np.clip(y_pred.astype(float), 0.0, 1.0)
        df[proba_col] = y_pred
        df[class_col] = (y_pred >= float(decision_threshold)).astype(int)
    else:
        # regression
        df[value_col] = y_pred.astype(float)

    # optional metadata columns (handy for debugging)
    df["_cnn_model_path"] = model_path
    df["_cnn_seq_col_used"] = seq_col
    df["_cnn_structure_used"] = int(STRUCTURE)
    df["_cnn_task_detected"] = task

    return df

##############################
### example runner helpers ###
##############################

def run_prediction_on_loaded_dfs(
    dfs: List[pd.DataFrame],
    *,
    model_type: str,
    folder: str,
    data: str,
    strain: str,
    segment: str,
    intersects: str,
    name_mod: str,
    model_choice: str = "final",
    force_marked: Optional[bool] = None,
    decision_threshold: float = 0.5,
) -> List[pd.DataFrame]:
    """
    Convenience wrapper that runs prediction on a list of dfs and returns updated list.
    """
    out = []
    for df in dfs:
        df = df.copy()
        df = manage_specifiers(df, data, strain, segment)
        df = run_cnn_prediction(
            df,
            model_type=model_type,
            folder=folder,
            data=data,
            strain=strain,
            segment=segment,
            intersects=intersects,
            name_mod=name_mod,
            model_choice=model_choice,
            force_marked=force_marked,
            decision_threshold=decision_threshold,
        )
        out.append(df)
    return out

############
### main ###
############


if __name__ == "__main__":
    """

    """

    plt.style.use("seaborn")
    plt.rc("font", size=12)

    #################
    ### SELECTION ###
    #################

    # pooled | unpooled
    folder = "unpooled"

    model_type = "reg"          # "bin" | "reg"
    model_choice = "final"      # "best" | "final"

    data = "IAV"
    strain = "PR8"
    segment = "PB1"
    intersects = "median_global_0"

    VERSION = "0"
    NAME_MOD = f"{VERSION}_motif_length_{K_MER_LENGTH}"

    # optional
    FORCE_MARKED = None         # None=auto

    #################
    ### load data ###
    #################

    subfolder = "primary"
    dfnames = PSEUDO_DATASETS
    dfs = load_all_preprocessed(dfnames, "pseudo", subfolder)

    ###################
    ### run scripts ###
    ###################

    dfs_pred = run_prediction_on_loaded_dfs(
        dfs,
        model_type=model_type,
        folder=folder,
        data=data,
        strain=strain,
        segment=segment,
        intersects=intersects,
        name_mod=NAME_MOD,
        model_choice=model_choice,
        force_marked=FORCE_MARKED,
        decision_threshold=DECISION_THRESHOLD,
    )

    for dfname, df in zip(dfnames, dfs_pred):
        save_df(df, dfname, RESULTSPATH, "pseudo", f"{model_type}_prediction", data, strain, segment, intersects)

    # quick sanity output
    print("✔ CNN prediction finished")
    print(f"✔ Updated {len(dfs_pred)} dataframes")