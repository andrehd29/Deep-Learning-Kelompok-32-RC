# utils/visualization.py

import matplotlib
matplotlib.use("Agg")  # WAJIB untuk Streamlit

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ==================================================
# 1. Plot prediction probabilities (STREAMLIT READY)
# ==================================================
def plot_prediction_probability(probabilities):
    """
    probabilities: array-like, contoh [0.65, 0.35]
    """

    class_names = ["Low / Moderate", "High"]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(class_names, probabilities)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")

    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)


# ==================================================
# 2. Show metrics summary
# ==================================================
def show_metrics_summary():
    st.subheader("📊 Metrics Summary")

    import json
    import os

    metrics_path = os.path.join("results", "metrics_summary.json")

    if not os.path.exists(metrics_path):
        st.warning("Metrics summary tidak ditemukan.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    st.json(metrics)


# ==================================================
# 3. Show confusion matrix
# ==================================================
def show_confusion_matrix():
    st.subheader("🧮 Confusion Matrix")

    import os
    from PIL import Image

    img_path = os.path.join("results", "confusion_matrix.png")

    if not os.path.exists(img_path):
        st.warning("Confusion matrix tidak ditemukan.")
        return

    img = Image.open(img_path)
    st.image(img, use_container_width=True)
