# src/plot_selected_force.py (robust version)
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATA_PROC, MODELS_DIR, OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)
OUT = Path(OUTPUT_DIR)

# load selected indices
sel_path = OUT / "selected_indices.npy"
if not sel_path.exists():
    raise SystemExit("Selected indices file not found: outputs/selected_indices.npy. Run src/select_instances.py first.")

indices = np.load(sel_path).astype(int)

# load data, meta
meta = joblib.load(DATA_PROC / "preproc.joblib")
feat = meta["feature_names"]
X_test = np.load(DATA_PROC / "X_test.npy")
# if X_test is 2D numpy array and feature order matches feat

# prefer XGB model, fallback to DNN
xgb_path = MODELS_DIR / "xgb_model.joblib"
dnn_path = MODELS_DIR / "dnn_model.h5"

model = None
model_type = None
if Path(xgb_path).exists():
    model = joblib.load(xgb_path)
    model_type = "xgb"
elif Path(dnn_path).exists():
    from tensorflow import keras
    model = keras.models.load_model(dnn_path)
    model_type = "dnn"
else:
    raise SystemExit("No model found in models/. Please run train_xgb.py or train_dnn.py")

# choose a small background (for explainer)
n_bg = min(200, max(10, X_test.shape[0] // 10))
X_bg = X_test[:n_bg]

# Build explainer robustly
explainer = None
try:
    # try shap.Explainer which is generally robust
    explainer = shap.Explainer(model.predict, X_bg)
    print("Using shap.Explainer(model.predict, background).")
except Exception as e:
    print("shap.Explainer failed:", e)
    # fallback: try TreeExplainer on booster (may fail as earlier)
    try:
        if model_type == "xgb" and hasattr(model, "get_booster"):
            explainer = shap.TreeExplainer(model.get_booster())
            print("Fell back to TreeExplainer(booster).")
        else:
            raise
    except Exception as e2:
        raise SystemExit("No usable SHAP explainer available: " + str(e2))

# Generate force plots for each selected index (compute SHAP for just that row)
for idx in indices:
    idx = int(idx)
    X_row = X_test[idx: idx+1]  # shape (1, n_features)

    # compute explanation for this row
    try:
        ex = explainer(X_row)  # shap.Explanation object or array-like
    except Exception as e:
        print(f"[WARN] explainer failed for index {idx}: {e}. Skipping.")
        continue

    # extract base_values and values robustly
    # ex.base_values and ex.values exist for shap.Explanation
    if hasattr(ex, "base_values") and hasattr(ex, "values"):
        base = ex.base_values
        vals = ex.values
        # ensure shapes: for regression, base may be shape (1,) or (1,1)
        # force to 1-d arrays
        # shap.force_plot expects base (scalar or array) and shap values (1D)
        b = np.array(base).reshape(-1)
        v = np.array(vals).reshape(vals.shape[1:]) if np.array(vals).ndim > 1 else np.array(vals).reshape(-1)
    else:
        # older return types: ex is array-like
        try:
            v = np.array(ex)
            b = np.array(v).sum()*0.0  # fallback base = 0
        except Exception:
            print(f"[WARN] couldn't extract shap values for index {idx}. Skipping.")
            continue

    # When values have shape (1, n_features) or (n_outputs, n_features)
    # Reduce to 1D shap vector for regression single-output models
    if v.ndim > 1:
        # if shape (1, n_features)
        if v.shape[0] == 1:
            v1 = v.reshape(-1)
        else:
            # if multi-output, try taking first output
            v1 = v[0].reshape(-1)
    else:
        v1 = v

    # base value: pick scalar if array-like
    if hasattr(b, "shape"):
        if np.array(b).size == 1:
            base_scalar = float(np.array(b).reshape(-1)[0])
        else:
            base_scalar = float(np.array(b).reshape(-1)[0])
    else:
        base_scalar = float(b)

    # create force plot and save as PNG
    try:
        fig = shap.force_plot(base_scalar, v1, X_row[0], feature_names=feat, matplotlib=True, show=False)
        outpath = OUT / f"force_{idx}.png"
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print("Saved force plot:", outpath)
    except Exception as e:
        print(f"[ERROR] failed to create/save force plot for index {idx}: {e}")
