# src/train_xgb.py (robust, works across xgboost versions)
import numpy as np
import joblib
import inspect
from config import DATA_PROC, MODELS_DIR, XGB_PARAMS
from utils import load_numpy
import xgboost as xgb
import os

def load_data():
    X_train = np.load(DATA_PROC / "X_train.npy")
    X_test = np.load(DATA_PROC / "X_test.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")
    return X_train, X_test, y_train, y_test

def fit_with_compat(model, X_train, y_train, X_val, y_val):
    """
    Fit an XGB model in a way that's compatible with multiple xgboost versions.
    If the sklearn wrapper accepts early_stopping_rounds, use it; otherwise fall back
    to a basic fit without early stopping.
    """
    fit_sig = inspect.signature(model.fit)
    params = {}
    # eval_set supported in most wrappers
    if 'eval_set' in fit_sig.parameters:
        params['eval_set'] = [(X_val, y_val)]
    # early stopping supported in newer wrappers
    if 'early_stopping_rounds' in fit_sig.parameters:
        params['early_stopping_rounds'] = 20
        # verbosity parameter name differs (verbose / verbose_eval) — just omit verbosity to be safe
        model.fit(X_train, y_train, **params)
    else:
        # fallback: call fit without early stopping
        # Increase n_estimators modestly if early stopping not available
        try:
            n = int(model.get_params().get("n_estimators", 100))
            model.set_params(n_estimators=max(n, 200))
        except Exception:
            pass
        if params:
            model.fit(X_train, y_train, **params)
        else:
            model.fit(X_train, y_train)

def train_and_save():
    X_train, X_test, y_train, y_test = load_data()
    # Build sklearn-compatible XGBRegressor
    model = xgb.XGBRegressor(**XGB_PARAMS)
    # Fit with compatibility wrapper
    fit_with_compat(model, X_train, y_train, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # save model in joblib for sklearn-style loading
    joblib.dump(model, MODELS_DIR / "xgb_model.joblib")
    # Also save native xgboost model (json)
    try:
        model.get_booster().save_model(str(MODELS_DIR / "xgb_model.json"))
    except Exception:
        # older wrappers may store booster differently; ignore if fails
        pass

    print("XGB trained and saved.")
    return model

if __name__ == "__main__":
    train_and_save()
