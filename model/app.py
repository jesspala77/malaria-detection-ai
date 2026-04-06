import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Malaria Detection AI",
    page_icon="M",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    return load_model("malaria_model.keras", compile=False)

model = load_trained_model()

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }
    .hero {
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
    }
    .hero p {
        margin-top: 0.5rem;
        font-size: 1rem;
        color: #d1d5db;
    }
    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.25rem;
        border-radius: 18px;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        margin-top: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .badge-high {
        background-color: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.35);
    }
    .badge-medium {
        background-color: rgba(59,130,246,0.15);
        color: #60a5fa;
        border: 1px solid rgba(59,130,246,0.35);
    }
    .badge-low {
        background-color: rgba(245,158,11,0.15);
        color: #fbbf24;
        border: 1px solid rgba(245,158,11,0.35);
    }
    .small-muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_demo_image(path: str):
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


def first_valid_image_path(*paths: str):
    for path in paths:
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            return path
        except Exception:
            continue
    return None

def make_result_text(image_name, raw_prediction, uninfected_conf, parasitized_conf, final_label, confidence_label):
    return f"""Malaria Detection AI - Prediction Summary

Image: {image_name}
Prediction: {final_label}
Confidence Level: {confidence_label}

Raw Prediction Value: {raw_prediction:.4f}
Uninfected Probability: {uninfected_conf:.2%}
Parasitized Probability: {parasitized_conf:.2%}

Model: Fine-tuned ResNet50
Input Size: 64 x 64 RGB

Disclaimer:
For research and educational use only. Not for clinical diagnosis.
"""

with st.sidebar:
    st.title("About This Project")
    st.write("**Model:** Fine-tuned ResNet50")
    st.write("**Task:** Binary image classification")
    st.write("**Input:** Microscopy cell images")
    st.write("**Output:** Parasitized or Uninfected")

    st.markdown("---")
    st.write("**Use Case**")
    st.write("AI-assisted screening support for malaria microscopy images.")

    st.markdown("---")
    st.subheader("Performance Snapshot")
    st.metric("Accuracy", "96%")
    st.metric("Precision", "96%")
    st.metric("Recall", "97%")
    st.metric("F1 Score", "96%")

    cm_path = first_valid_image_path("assets/confusion_matrix.png")
    if cm_path:
        st.image(cm_path, caption="Confusion Matrix", width="stretch")
    else:
        st.caption("Confusion matrix image not available in a displayable image format.")

    curve_path = first_valid_image_path("assets/training_curve.png")
    if curve_path:
        st.image(curve_path, caption="Training Curve", width="stretch")
    else:
        st.caption("Training curve image is missing or invalid.")

    st.markdown("---")
    st.warning("For research and educational use only. Not for clinical diagnosis.")
    st.markdown("---")
    st.caption("Built by Jessica Palacio")

st.markdown(
    """
    <div class="hero">
        <h1>Malaria Detection AI</h1>
        <p>Upload a microscope cell image or try a sample to predict whether it is parasitized or uninfected using a fine-tuned ResNet50 model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

top_left, top_right = st.columns([5, 1])

with top_left:
    uploaded_file = st.file_uploader(
        "Upload a cell image",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

with top_right:
    st.write("")
    st.write("")
    if st.button("Reset", use_container_width=True):
        st.rerun()

st.markdown("### Quick Demo")
demo_col1, demo_col2, demo_col3 = st.columns(3)

selected_demo = None

with demo_col1:
    if st.button("Load Sample Parasitized"):
        selected_demo = "sample_parasitized"

with demo_col2:
    if st.button("Load Sample Uninfected"):
        selected_demo = "sample_uninfected"

with demo_col3:
    st.caption("Optional: add sample images in `model/samples/`")

image = None
image_name = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_name = uploaded_file.name
elif selected_demo == "sample_parasitized":
    sample_path = first_valid_image_path(
        "samples/parasitized_sample.png",
        "samples/parasitized/C68P29N_ThinF_IMG_20150819_134830_cell_79.png",
        "samples/parasitized/C39P4thinF_original_IMG_20150622_111206_cell_112.png",
    )
    image = load_demo_image(sample_path) if sample_path else None
    image_name = os.path.basename(sample_path) if sample_path else "parasitized_sample.png"
    if image is None:
        st.warning("Sample image not found. Add it to: model/samples/parasitized_sample.png")
elif selected_demo == "sample_uninfected":
    sample_path = first_valid_image_path(
        "samples/uninfected_sample.png",
        "samples/uninfected/uninfected/C7NthinF_IMG_20150611_104404_cell_160.png",
    )
    image = load_demo_image(sample_path) if sample_path else None
    image_name = os.path.basename(sample_path) if sample_path else "uninfected_sample.png"
    if image is None:
        st.warning("Sample image not found. Add it to: model/samples/uninfected_sample.png")

if image is not None:
    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Selected Image")
        st.image(image, width="stretch")
        st.caption(f"Image: {image_name}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.subheader("Prediction")

        resized = image.resize((64, 64))
        img_array = np.array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array, verbose=0)[0][0]

        raw_prediction = float(prediction)

        # Assumes model output = probability of UNINFECTED
        uninfected_conf = raw_prediction
        parasitized_conf = 1 - uninfected_conf
        top_conf = max(uninfected_conf, parasitized_conf)

        if top_conf >= 0.85:
            badge_html = '<span class="badge badge-high">High Confidence</span>'
            confidence_label = "High"
        elif top_conf >= 0.65:
            badge_html = '<span class="badge badge-medium">Moderate Confidence</span>'
            confidence_label = "Moderate"
        else:
            badge_html = '<span class="badge badge-low">Manual Review Recommended</span>'
            confidence_label = "Low / Manual Review Recommended"

        st.markdown(f'<div class="result-card">{badge_html}', unsafe_allow_html=True)

        HIGH_CONF = 0.75
        LOW_CONF = 0.25

        if raw_prediction >= HIGH_CONF:
            final_label = "Uninfected"
            st.success("Prediction: Uninfected")
            st.progress(uninfected_conf)
            st.write(f"Uninfected: {uninfected_conf:.2%}")
            st.write(f"Parasitized: {parasitized_conf:.2%}")
            st.write("High confidence detection of uninfected cell.")

        elif raw_prediction <= LOW_CONF:
            final_label = "Parasitized (Malaria Infected)"
            st.error("Prediction: Parasitized (Malaria Infected)")
            st.progress(parasitized_conf)
            st.write(f"Parasitized: {parasitized_conf:.2%}")
            st.write(f"Uninfected: {uninfected_conf:.2%}")
            st.write("High confidence detection of infected cell.")

        else:
            final_label = "Uncertain - Manual Review Recommended"
            st.warning("Uncertain prediction - manual review recommended.")
            st.progress(0.5)
            st.write(f"Uninfected: {uninfected_conf:.2%}")
            st.write(f"Parasitized: {parasitized_conf:.2%}")
            st.write("The model is not strongly confident in this classification.")

        st.markdown("</div>", unsafe_allow_html=True)

        result_text = make_result_text(
            image_name=image_name,
            raw_prediction=raw_prediction,
            uninfected_conf=uninfected_conf,
            parasitized_conf=parasitized_conf,
            final_label=final_label,
            confidence_label=confidence_label
        )

        st.download_button(
            label="Download Result Summary",
            data=result_text,
            file_name="malaria_prediction_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("Advanced Details"):
            st.write(f"**Raw Prediction Value:** {raw_prediction:.4f}")
            st.write(f"**Uninfected Probability:** {uninfected_conf:.4f}")
            st.write(f"**Parasitized Probability:** {parasitized_conf:.4f}")
            st.write("**High-confidence Uninfected cutoff:** >= 0.75")
            st.write("**High-confidence Parasitized cutoff:** <= 0.25")
            st.write("**Uncertain Zone:** 0.25 to 0.75")
            st.write("**Preprocessing:** RGB conversion, resize to 64x64, normalization to [0,1]")

    st.markdown("---")
    st.subheader("Model Details")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Architecture", "ResNet50")
    with d2:
        st.metric("Input Size", "64 x 64")
    with d3:
        st.metric("Task", "Binary Classification")

else:
    st.info("Drag and drop, browse, or load a sample image to generate a prediction.")

st.markdown("---")
st.markdown(
    '<p class="small-muted">Built with Streamlit, TensorFlow, and a fine-tuned ResNet50 model.</p>',
    unsafe_allow_html=True
)
