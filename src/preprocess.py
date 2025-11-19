# src/preprocess.py (replace file)
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import DATA_RAW, DATA_PROC, RANDOM_SEED
from utils import save_numpy

def make_onehot_encoder():
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def load_data(path=DATA_RAW):
    return pd.read_csv(path)

def build_preprocessing(df, target_column):
    # numeric and categorical columns
    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    # Numeric pipeline: impute (median) -> scale
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: (OHE) - we won't impute categoricals here (assumed none missing)
    ohe = make_onehot_encoder()

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", ohe, cat_cols)
        ],
        remainder="drop"
    )
    return preproc, num_cols, cat_cols

def run_preprocessing(target_column="median_house_value"):
    df = load_data()
    preproc, num_cols, cat_cols = build_preprocessing(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preproc.fit(X_train)
    X_train_t = preproc.transform(X_train)
    X_test_t = preproc.transform(X_test)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    save_numpy(X_train_t, DATA_PROC/"X_train")
    save_numpy(X_test_t, DATA_PROC/"X_test")
    save_numpy(y_train, DATA_PROC/"y_train")
    save_numpy(y_test, DATA_PROC/"y_test")

    # Build feature names list (numeric then OHE feature names)
    feature_names = []
    feature_names += num_cols
    ohe = preproc.named_transformers_["cat"]
    if hasattr(ohe, "get_feature_names_out"):
        feature_names += list(ohe.get_feature_names_out(cat_cols))
    else:
        feature_names += cat_cols

    joblib.dump({"preprocessor": preproc, "feature_names": feature_names}, DATA_PROC/"preproc.joblib")
    print("Preprocessing done. Saved arrays and preproc.joblib")

if __name__ == "__main__":
    run_preprocessing()
