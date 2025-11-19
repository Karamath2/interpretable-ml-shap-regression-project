import numpy as np
import joblib
from config import DATA_PROC, MODELS_DIR

X_test = np.load(DATA_PROC/"X_test.npy")
y_test = np.load(DATA_PROC/"y_test.npy")

model = joblib.load(MODELS_DIR/"xgb_model.joblib")
preds = model.predict(X_test)

errors = np.abs(y_test - preds)

imax = int(np.argmax(errors))
imin = int(np.argmin(errors))
iavg = int(np.abs(errors - errors.mean()).argmin())

np.save("outputs/selected_indices.npy", np.array([imax, imin, iavg]))
print("Indices saved:", [imax, imin, iavg])
