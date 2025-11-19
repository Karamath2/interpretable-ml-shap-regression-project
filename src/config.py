import pathlib
BASE = pathlib.Path(__file__).resolve().parents[1]

DATA_RAW = BASE / "data" / "raw" / "dataset.csv"
DATA_PROC = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
RANDOM_SEED = 42

# Model/Training hyperparams (tweakable)
XGB_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "random_state": RANDOM_SEED}
DNN_PARAMS = {"epochs": 30, "batch_size": 32}
