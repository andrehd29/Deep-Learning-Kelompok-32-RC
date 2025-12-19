# =====================================================
# train/train_model.py
# FINAL VERSION - Streamlit & TensorFlow SAFE
# =====================================================

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# =====================================================
# 0. PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "app", "models")
SCALER_DIR = os.path.join(BASE_DIR, "..", "app", "scalers")
RESULT_DIR = os.path.join(BASE_DIR, "..", "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Using dataset:", DATA_PATH)

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

df.columns = [
    "Timestamp", "Nama", "Semester", "Jenis_Kelamin",
    "Freq_Platform_AI", "Jam_AI_Per_Hari", "Freq_AI_Akademik",
    "Freq_AI_NonAkademik", "Bantuan_AI_Belajar", "Jam_Hemat_Mingguan",
    "Intensitas_Overall", "Jam_Belajar_Mandiri", "Jam_Digital_Per_Hari",
    "Freq_Catatan_Mingguan", "Nomer_DANA"
]

df = df.drop(columns=["Timestamp", "Nama", "Nomer_DANA"])

# =====================================================
# 2. CLEAN NUMERIC
# =====================================================
def clean_numeric(x):
    if isinstance(x, str):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", x)
        return float(nums[0]) if nums else np.nan
    return x

NUMERIC_COLS = [
    "Semester", "Freq_Platform_AI", "Jam_AI_Per_Hari",
    "Freq_AI_Akademik", "Freq_AI_NonAkademik",
    "Bantuan_AI_Belajar", "Jam_Hemat_Mingguan",
    "Jam_Belajar_Mandiri", "Jam_Digital_Per_Hari",
    "Freq_Catatan_Mingguan"
]

for col in NUMERIC_COLS:
    df[col] = df[col].apply(clean_numeric)

df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

df["Jenis_Kelamin"] = (
    df["Jenis_Kelamin"]
    .map({"Laki-laki": 0, "Perempuan": 1})
    .fillna(0)
    .astype(int)
)

# =====================================================
# 3. TARGET ENGINEERING
# =====================================================
if df["Intensitas_Overall"].dtype == object:
    df["Intensitas_Overall"] = df["Intensitas_Overall"].map({
        "Low": 0, "Rendah": 0,
        "Moderate": 1, "Sedang": 1,
        "High": 2, "Tinggi": 2
    })

df = df[df["Intensitas_Overall"].notna()]
df["Intensitas_Overall"] = df["Intensitas_Overall"].astype(int)

# Gabungkan Low + Moderate → binary
df["Intensitas_Overall"] = df["Intensitas_Overall"].replace(0, 1)

# Remap ke {0,1}
unique_vals = sorted(df["Intensitas_Overall"].unique())
target_map = {v: i for i, v in enumerate(unique_vals)}
df["Intensitas_Overall"] = df["Intensitas_Overall"].map(target_map)

print("Class distribution:")
print(df["Intensitas_Overall"].value_counts())

# =====================================================
# 4. FEATURE & SPLIT
# =====================================================
FEATURES = [c for c in df.columns if c != "Intensitas_Overall"]

X = df[FEATURES]
y = df["Intensitas_Overall"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 5. SCALING
# =====================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(SCALER_DIR, "scaler.joblib"))

# =====================================================
# 6. BUILD MLP MODEL (FIXED INPUT LAYER)
# =====================================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),   # 🔥 FIXED
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# 7. TRAINING
# =====================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=2
)

model.save(os.path.join(MODEL_DIR, "final_model.keras"))

# =====================================================
# 8. EVALUATION
# =====================================================
best_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "best_model.keras")
)

y_pred = np.argmax(best_model.predict(X_test), axis=1)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

with open(os.path.join(RESULT_DIR, "metrics_summary.json"), "w") as f:
    json.dump(metrics, f, indent=4)

report = classification_report(
    y_test, y_pred,
    target_names=["Low/Moderate", "High"],
    output_dict=True
)

pd.DataFrame(report).T.to_csv(
    os.path.join(RESULT_DIR, "classification_report.csv")
)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low/Moderate", "High"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

print("\n✅ TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
print("Model saved to:", MODEL_DIR)
print("Scaler saved to:", SCALER_DIR)
print("Results saved to:", RESULT_DIR)
