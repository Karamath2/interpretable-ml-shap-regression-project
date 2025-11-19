import numpy as np
import joblib
from config import DATA_PROC, MODELS_DIR, OUTPUT_DIR
import shap
from utils import rmse
import os
import matplotlib.pyplot as plt
import pathlib

def load_artifacts():
    X_test = np.load(DATA_PROC / "X_test.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")
    preproc_meta = joblib.load(DATA_PROC / "preproc.joblib")
    feature_names = preproc_meta["feature_names"]
    xgb = None
    dnn = None
    xgb_path = MODELS_DIR / "xgb_model.joblib"
    dnn_path = MODELS_DIR / "dnn_model.h5"
    if xgb_path.exists():
        xgb = joblib.load(xgb_path)
    else:
        print("[WARN] XGB model not found at", xgb_path)
    if dnn_path.exists():
        from tensorflow import keras
        dnn = keras.models.load_model(dnn_path)
    else:
        print("[WARN] DNN model not found at", dnn_path)
    return X_test, y_test, feature_names, xgb, dnn

def shap_for_xgb(xgb_model, X_background, X_explain, feature_names):
    """
    Create a TreeExplainer for XGBoost models robustly:
    - If xgb_model is a sklearn wrapper, use get_booster()
    - If get_booster() produces a Booster, pass that to TreeExplainer
    """
    # convert to native booster if possible
    native_model = None
    try:
        # sklearn XGBRegressor wrapper has get_booster()
        booster = xgb_model.get_booster()
        # booster is an xgboost.core.Booster instance -> safe to pass
        native_model = booster
    except Exception:
        # either xgb_model is already a Booster or something else
        native_model = xgb_model

    # create explainer; for XGBoost pass the booster/native model
    explainer = shap.TreeExplainer(native_model)
    # shap API: shap_values shape depends on model output; for regression will be (n_samples, n_features)
    shap_values = explainer.shap_values(X_explain)
    expected_value = explainer.expected_value if hasattr(explainer, "expected_value") else None
    return explainer, shap_values, expected_value

def run_shap_analysis(n_background=100):
    X_test, y_test, feature_names, xgb, dnn = load_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shap_dir = OUTPUT_DIR / "shap_values"
    shap_dir.mkdir(parents=True, exist_ok=True)

    if X_test.shape[0] < n_background:
        n_background = max(1, X_test.shape[0] // 5)

    X_bg = X_test[:n_background]

    if xgb is not None:
        try:
            expl_xgb, sv_xgb, ev_xgb = shap_for_xgb(xgb, X_bg, X_test, feature_names)
            np.save(shap_dir / "shap_values_xgb.npy", sv_xgb)
            print("Saved XGB SHAP values")
        except Exception as e:
            print("[ERROR] XGB SHAP failed:", str(e))
            print("Skipping XGB SHAP.")
    else:
        print("[INFO] Skipping XGB SHAP because xgb model is missing.")

    if dnn is not None:
        try:
            # KernelExplainer is slow — use subset for deep models
            expl_dnn = shap.KernelExplainer(lambda x: dnn.predict(x).flatten(), X_bg)
            sv_dnn = expl_dnn.shap_values(X_test[:200], nsamples=128)
            np.save(shap_dir / "shap_values_dnn.npy", sv_dnn)
            print("Saved DNN SHAP values (subset)")
        except Exception as e:
            print("[ERROR] DNN SHAP failed:", str(e))
    else:
        print("[INFO] Skipping DNN SHAP because dnn model is missing.")

if __name__ == "__main__":
    run_shap_analysis()
