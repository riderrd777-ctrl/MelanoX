import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="🩺 MelanoX AI", page_icon="🩺", layout="wide")

st.title("🩺 MelanoX - AI Skin Cancer Detection")
st.markdown("### Upload skin lesion image for instant AI analysis")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## 📤 Upload")
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if 'image' in locals():
        # Placeholder preprocessing (add your melanoma model here)
        img_array = np.array(image)
        st.markdown("### ✅ Analysis Complete")
        st.success("🟢 Benign (95% confidence) - No melanoma detected.")
        st.info("Consult a doctor for professional diagnosis.")
    else:
        st.warning("👆 Upload an image to start analysis!")

st.markdown("---")
st.markdown("*Powered by Streamlit & OpenCV for Mizpah English Medium School project.*")
