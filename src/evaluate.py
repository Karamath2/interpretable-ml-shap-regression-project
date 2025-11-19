# src/evaluate.py (safe)
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from config import DATA_PROC, MODELS_DIR
import joblib
from tensorflow import keras
from utils import rmse
import pathlib
import os

def load_data():
    X_test = np.load(DATA_PROC / "X_test.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")
    return X_test, y_test

def eval_xgb():
    xgb_path = MODELS_DIR / "xgb_model.joblib"
    if not xgb_path.exists():
        print("[WARN] XGB model not found at", xgb_path)
        return None, None
    model = joblib.load(xgb_path)
    X_test, y_test = load_data()
    preds = model.predict(X_test)
    return preds, y_test

def eval_dnn():
    dnn_path = MODELS_DIR / "dnn_model.h5"
    if not dnn_path.exists():
        print("[WARN] DNN model not found at", dnn_path)
        return None, None
    model = keras.models.load_model(dnn_path)
    X_test, y_test = load_data()
    preds = model.predict(X_test).flatten()
    return preds, y_test

if __name__ == "__main__":
    # Try XGB first
    preds, y_test = eval_xgb()
    if preds is not None:
        print("XGB R2:", r2_score(y_test, preds), "RMSE:", rmse(y_test, preds))
    else:
        # fallback to DNN
        preds, y_test = eval_dnn()
        if preds is not None:
            print("DNN R2:", r2_score(y_test, preds), "RMSE:", rmse(y_test, preds))
        else:
            print("[ERROR] No model available to evaluate. Run src/train_xgb.py or src/train_dnn.py.")
