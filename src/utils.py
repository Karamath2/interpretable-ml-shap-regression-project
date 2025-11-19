import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error

def rmse(y_true, y_pred):
    """
    Robust RMSE calculation that works across sklearn versions.
    """
    # ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def save_numpy(arr, path):
    path = str(path)
    np.save(path, arr)

def load_numpy(path):
    return np.load(str(path) + ".npy") if not str(path).endswith(".npy") else np.load(path)

def save_obj(obj, path):
    joblib.dump(obj, path)

def load_obj(path):
    return joblib.load(path)
