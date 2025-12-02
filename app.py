
import streamlit as st
import os, json, numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fish Species Classifier", layout="centered")

MODEL_PATH = "final_model.keras"   # replace with fish_best_model.h5 if you prefer

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

if os.path.exists("class_labels.json"):
    try:
        with open("class_labels.json","r") as f:
            class_names = json.load(f)
    except Exception:
        class_names = []
else:
    class_names = []

st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image (jpg/png). The model returns top-3 predictions with confidence scores.")

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB").resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.warning("Model file not found or failed to load. Put `final_model.keras` in this folder or check model loading errors.")
    else:
        # Prepare image for prediction
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)[0]
        top3 = preds.argsort()[-3:][::-1]

        st.write("### Top predictions")
        for idx in top3:
            name = class_names[idx] if idx < len(class_names) else str(idx)
            st.write(f"- **{name}** — {preds[idx]*100:.2f}%")
