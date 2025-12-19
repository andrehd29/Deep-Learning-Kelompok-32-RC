# utils/inference.py

import os
import joblib
import numpy as np
import tensorflow as tf


# ==================================================
# 1. Load model & scaler (dipakai app.py)
# ==================================================
def load_model_and_scaler(model_path: str, scaler_path: str):
    """
    Load trained Keras model and scaler
    """
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ==================================================
# 2. Predict class & probability
# ==================================================
def predict_class(model, X_scaled):
    """
    X_scaled: hasil preprocess_input (numpy array)
    return:
        pred_class: int
        pred_proba: list
    """

    proba = model.predict(X_scaled)
    pred_class = int(np.argmax(proba, axis=1)[0])
    pred_proba = proba[0].tolist()

    return pred_class, pred_proba
