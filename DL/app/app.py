# =====================================================
# app/app.py
# =====================================================

import streamlit as st

st.set_page_config(
    page_title="Prediksi Intensitas Penggunaan AI Mahasiswa",
    layout="centered"
)

import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.preprocessing import preprocess_input
from utils.inference import load_model_and_scaler, predict_class
from utils.visualization import (
    plot_prediction_probability,
    show_metrics_summary,
    show_confusion_matrix
)

# =====================================================
# UI
# =====================================================
st.title("📊 Prediksi Intensitas Penggunaan AI Mahasiswa")
st.write(
    "Dashboard ini memprediksi **tingkat intensitas penggunaan AI dalam proses belajar mahasiswa** "
    "berdasarkan pola belajar dan aktivitas digital."
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scalers", "scaler.joblib")

try:
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
except Exception as e:
    st.error("❌ Gagal memuat model atau scaler")
    st.exception(e)
    st.stop()

# =====================================================
# USER INPUT
# =====================================================
st.subheader("📝 Input Data Mahasiswa")

semester = st.slider("Semester", 1, 14, 4)
jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
freq_platform_ai = st.slider("Frekuensi AI (kali/minggu)", 0, 50, 5)
jam_ai_per_hari = st.slider("Jam AI per hari", 0.0, 10.0, 1.0)
freq_ai_akademik = st.slider("AI Akademik (kali/minggu)", 0, 50, 5)
freq_ai_nonakademik = st.slider("AI Non-akademik", 0, 50, 2)
bantuan_ai_belajar = st.slider("Bantuan AI belajar", 0, 10, 5)
jam_hemat_mingguan = st.slider("Jam hemat per minggu", 0.0, 30.0, 2.0)
jam_belajar_mandiri = st.slider("Jam belajar mandiri", 0.0, 50.0, 10.0)
jam_digital_per_hari = st.slider("Jam digital per hari", 0.0, 15.0, 4.0)
freq_catatan_mingguan = st.slider("Catatan per minggu", 0, 20, 5)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 Prediksi Intensitas"):
    input_data = {
        "Semester": semester,
        "Jenis_Kelamin": jenis_kelamin,
        "Freq_Platform_AI": freq_platform_ai,
        "Jam_AI_Per_Hari": jam_ai_per_hari,
        "Freq_AI_Akademik": freq_ai_akademik,
        "Freq_AI_NonAkademik": freq_ai_nonakademik,
        "Bantuan_AI_Belajar": bantuan_ai_belajar,
        "Jam_Hemat_Mingguan": jam_hemat_mingguan,
        "Jam_Belajar_Mandiri": jam_belajar_mandiri,
        "Jam_Digital_Per_Hari": jam_digital_per_hari,
        "Freq_Catatan_Mingguan": freq_catatan_mingguan
    }

    X_input = preprocess_input(input_data, scaler)
    pred_class, pred_proba = predict_class(model, X_input)

    label_map = {0: "Low / Moderate", 1: "High"}
    st.success(f"📌 Prediksi Intensitas: **{label_map[pred_class]}**")

    plot_prediction_probability(pred_proba)

# =====================================================
# EVALUATION
# =====================================================
st.subheader("📈 Evaluasi Model")
if st.checkbox("Tampilkan Evaluasi Model"):
    show_metrics_summary()
    show_confusion_matrix()

st.caption("© Proyek Deep Learning – Prediksi Penggunaan AI Mahasiswa")
