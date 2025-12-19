# utils/preprocessing.py

import re
import numpy as np
import pandas as pd

# ==================================================
# 1. Helper: clean numeric input
# ==================================================
def clean_numeric(x):
    if isinstance(x, str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", x)
        return float(nums[0]) if nums else 0.0
    return x


# ==================================================
# 2. Encode gender (SAMA dengan training)
# ==================================================
def encode_gender(gender):
    return 0 if gender == "Laki-laki" else 1


# ==================================================
# 3. Feature order (HARUS SAMA DENGAN TRAINING)
# ==================================================
FEATURE_ORDER = [
    "Semester",
    "Jenis_Kelamin",
    "Freq_Platform_AI",
    "Jam_AI_Per_Hari",
    "Freq_AI_Akademik",
    "Freq_AI_NonAkademik",
    "Bantuan_AI_Belajar",
    "Jam_Hemat_Mingguan",
    "Jam_Belajar_Mandiri",
    "Jam_Digital_Per_Hari",
    "Freq_Catatan_Mingguan"
]


# ==================================================
# 4. Main preprocessing (INFERENCE READY)
# ==================================================
def preprocess_input(user_input: dict, scaler):
    """
    user_input: dict dari Streamlit form
    scaler: scaler object (joblib-loaded)
    return: numpy array siap masuk model
    """

    df = pd.DataFrame([user_input])

    # Encode gender
    df["Jenis_Kelamin"] = df["Jenis_Kelamin"].apply(encode_gender)

    # Clean numerics
    for col in FEATURE_ORDER:
        if col != "Jenis_Kelamin":
            df[col] = df[col].apply(clean_numeric)

    # Ensure order
    df = df[FEATURE_ORDER]

    # Scale
    X_scaled = scaler.transform(df.values)

    return X_scaled
