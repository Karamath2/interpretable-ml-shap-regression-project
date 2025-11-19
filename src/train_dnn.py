import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from config import DATA_PROC, MODELS_DIR, DNN_PARAMS
import tensorflow as tf

def load_data():
    X_train = np.load(DATA_PROC / "X_train.npy")
    X_test = np.load(DATA_PROC / "X_test.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")
    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])
    return model

def train_and_save():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=DNN_PARAMS['epochs'], batch_size=DNN_PARAMS['batch_size'], verbose=2)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "dnn_model.h5")
    print("DNN trained and saved.")
    return model, history

if __name__ == "__main__":
    train_and_save()
