import os
import warnings
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Malaria Detection AI",
    page_icon="🔬",
    layout="wide"
)


def parse_version(version_str: str) -> tuple:
    parts = []
    for piece in version_str.split("."):
        num = ""
        for ch in piece:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def app_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


def apply_keras_compatibility_patches() -> None:
    try:
        # Patch Keras Dense and Layer constructors so outdated quantization_config fields are ignored.
        import keras.src.layers.core.dense as dense_src
        import keras.src.layers.layer as layer_src
        import keras.src.ops.operation as operation_src

        if not getattr(dense_src.Dense, "__compat_quantization_patched__", False):
            original_dense_init = dense_src.Dense.__init__

            def patched_dense_init(self, *args, quantization_config=None, **kwargs):
                return original_dense_init(self, *args, **kwargs)

            dense_src.Dense.__init__ = patched_dense_init
            dense_src.Dense.__compat_quantization_patched__ = True

        if not getattr(layer_src.Layer, "__compat_quantization_patched__", False):
            original_layer_init = layer_src.Layer.__init__

            def patched_layer_init(self, *args, quantization_config=None, **kwargs):
                return original_layer_init(self, *args, **kwargs)

            layer_src.Layer.__init__ = patched_layer_init
            layer_src.Layer.__compat_quantization_patched__ = True

        if not getattr(operation_src.Operation, "__compat_quantization_patched__", False):
            original_operation_from_config = operation_src.Operation.from_config

            @classmethod
            def patched_operation_from_config(cls, config):
                config = config.copy()
                config.pop('quantization_config', None)
                return original_operation_from_config(config)

            operation_src.Operation.from_config = patched_operation_from_config
            operation_src.Operation.__compat_quantization_patched__ = True

    except Exception as e:
        print(f"Warning: Could not apply compatibility patches: {e}")
        pass


def build_resnet50_binary_classifier(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def summarize_model_load_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    if len(message) > 240:
        message = f"{message[:237]}..."
    return message


def load_model_with_architecture_fallback(model_path: str):
    try:
        return load_model(model_path, compile=False), None
    except Exception as load_error:
        fallback_error = summarize_model_load_error(load_error)
        try:
            fallback_model = build_resnet50_binary_classifier()
            fallback_model.load_weights(model_path)
            return fallback_model, (
                f"Original load_model failed: {fallback_error}. "
                "Weights were loaded into the reconstructed ResNet50 architecture."
            )
        except Exception as weights_error:
            combined = (
                f"Original load_model error: {fallback_error}. "
                f"Fallback load_weights error: {summarize_model_load_error(weights_error)}"
            )
            return None, combined


@st.cache_resource
def load_trained_model(cache_key):
    model_path = app_path("models/best_malaria_model.keras")
    tf_version = tf.__version__

    apply_keras_compatibility_patches()

    if parse_version(tf_version) < (2, 16, 0):
        return None, {
            "available": False,
            "source": "unavailable",
            "message": (
                "TensorFlow is too old for the saved model format. "
                f"Detected TensorFlow {tf_version}; use TensorFlow 2.16 or newer."
            ),
            "install_hint": 'pip install --upgrade "tensorflow>=2.16,<2.19"',
        }

    try:
        model, direct_load_error = load_model_with_architecture_fallback(model_path)
        if model is None:
            raise RuntimeError(direct_load_error or "Failed to reconstruct the model from the saved file.")

        return model, {
            "available": True,
            "source": "models/best_malaria_model.keras",
            "message": "Fine-tuned ResNet50 model loaded successfully.",
            "details": ([f"Model loaded through the rebuilt ResNet50 architecture: {direct_load_error}"] if direct_load_error else []),
        }
    except Exception as model_error:
        return None, {
            "available": False,
            "source": "unavailable",
            "message": "No compatible model file could be loaded.",
            "details": [f"Model load error: {model_error}"],
        }


# Compute cache key based on model file modification times
model_files = [
    app_path("models/best_malaria_model.keras"),
]
cache_key = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in model_files)

model, model_info = load_trained_model(cache_key)


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
    .workspace-card {
        background: linear-gradient(180deg, rgba(248,250,252,0.98), rgba(241,245,249,0.96));
        border: 1px solid rgba(148,163,184,0.28);
        padding: 1.35rem;
        border-radius: 22px;
        margin-bottom: 1rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
    }
    .workspace-title {
        margin: 0 0 0.35rem 0;
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .workspace-copy {
        margin: 0;
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .stat-card {
        background: linear-gradient(180deg, #f8fafc, #e2e8f0);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        margin-bottom: 0.9rem;
    }
    .stat-label {
        color: #475569;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .stat-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .stat-subtext {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        margin-top: 0.5rem;
    }
    .result-shell {
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 22px;
        padding: 1.35rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
    }
    .result-kicker {
        color: #0f766e;
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.45rem;
    }
    .result-title {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .result-summary {
        color: #475569;
        font-size: 0.97rem;
        line-height: 1.5;
        margin-bottom: 1rem;
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
    .sidebar-panel {
        padding: 1rem 1rem 0.85rem 1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
        border: 1px solid rgba(148,163,184,0.22);
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    }
    .sidebar-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }
    .sidebar-copy {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
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


def first_existing_path(*paths: str):
    for path in paths:
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def find_asset_by_keywords(folders, include_keywords, extensions=None, exclude_keywords=None):
    include_keywords = [keyword.lower() for keyword in include_keywords]
    exclude_keywords = [keyword.lower() for keyword in (exclude_keywords or [])]
    allowed_extensions = tuple(ext.lower() for ext in (extensions or [".png", ".jpg", ".jpeg", ".webp", ".pdf"]))

    for folder in folders:
        folder_path = app_path(folder)
        if not os.path.isdir(folder_path):
            continue

        candidates = []
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:
                continue

            lowered = entry.lower()
            if not lowered.endswith(allowed_extensions):
                continue
            if any(keyword not in lowered for keyword in include_keywords):
                continue
            if any(keyword in lowered for keyword in exclude_keywords):
                continue

            candidates.append((full_path, lowered))

        if candidates:
            candidates.sort(key=lambda item: (item[1].endswith(".pdf"), len(item[1]), item[1]))
            return candidates[0][0]

    return None


def render_sidebar_asset(title: str, path: str, empty_message: str):
    if not path:
        st.caption(empty_message)
        return

    lower_path = path.lower()
    if lower_path.endswith((".png", ".jpg", ".jpeg", ".webp")):
        st.image(path, caption=title, width="stretch")
        return

    if lower_path.endswith(".pdf"):
        st.caption(f"{title} available as PDF.")
        with open(path, "rb") as pdf_file:
            st.download_button(
                label=f"Download {title} PDF",
                data=pdf_file.read(),
                file_name=os.path.basename(path),
                mime="application/pdf",
                use_container_width=True,
            )
        return

    st.caption(empty_message)


def render_sidebar_panel(kicker: str, title: str, body: str):
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="sidebar-kicker">{kicker}</div>
            <div class="sidebar-title">{title}</div>
            <p class="sidebar-copy">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_intro(title: str, body: str):
    st.markdown(
        f"""
        <div class="workspace-card">
            <div class="workspace-title">{title}</div>
            <p class="workspace-copy">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, subtext: str = ""):
    subtext_html = f'<div class="stat-subtext">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {subtext_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_result_text(image_name, raw_prediction, uninfected_conf, parasitized_conf, final_label, confidence_label, model_source):
    return f"""Malaria Detection AI - Prediction Summary

Image: {image_name}
Prediction: {final_label}
Confidence Level: {confidence_label}

Raw Prediction Value: {raw_prediction:.4f}
Uninfected Probability: {uninfected_conf:.2%}
Parasitized Probability: {parasitized_conf:.2%}

Model: {model_source}
Input Size: 128 x 128 RGB

Disclaimer:
For research and educational use only. Not for clinical diagnosis.
"""


with st.sidebar:
    render_sidebar_panel(
        "Portfolio Case Study",
        "Clinical Screening Overview",
        "Fine-tuned ResNet50 for malaria microscopy classification with conservative uncertainty thresholds that support safer triage-style decision making.",
    )

    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.metric("Task", "Binary")
    with overview_col2:
        st.metric("Input", "128 x 128")

    st.caption("Output labels: Parasitized and Uninfected cell images")

    st.markdown("---")
    st.subheader("System Status")
    st.write(model_info["message"])
    if model_info.get("source"):
        st.caption(f"Loaded from: {model_info['source']}")
    if model_info.get("install_hint"):
        st.code(model_info["install_hint"], language="bash")

    st.markdown("---")
    st.subheader("Validation Snapshot")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Val Accuracy", "97.88%")
    with metric_col2:
        st.metric("Val Loss", "0.0588")
    st.caption("Use case: AI-assisted review support for malaria microscopy workflows.")

    st.markdown("---")
    st.subheader("Model Evaluation")

    cm_path = first_existing_path(
        app_path("assets/confusion_matrix.png"),
        app_path("visuals/confusion_matrix.png"),
        find_asset_by_keywords(["assets", "visuals"], ["confusion", "matrix"]),
    )
    render_sidebar_asset("Confusion Matrix", cm_path, "Confusion matrix asset is missing.")

    curve_path = first_existing_path(
        app_path("assets/training_curve.png"),
        app_path("visuals/training_curve.png"),
        find_asset_by_keywords(["assets", "visuals"], ["training"], exclude_keywords=["roc", "prc", "probability", "false positive"]),
        find_asset_by_keywords(["assets", "visuals"], ["acc", "loss"], exclude_keywords=["roc", "prc", "probability", "false positive"]),
    )
    render_sidebar_asset("Training Curve", curve_path, "Training curve image is missing or empty.")

    roc_path = find_asset_by_keywords(["assets", "visuals"], ["roc"], exclude_keywords=["probability"])
    render_sidebar_asset("ROC / PRC Analysis", roc_path, "ROC / PRC chart not found.")

    probability_path = find_asset_by_keywords(["assets", "visuals"], ["probability"], exclude_keywords=["roc", "prc"])
    render_sidebar_asset("Prediction Probability Analysis", probability_path, "Prediction probability chart not found.")

    false_positive_path = find_asset_by_keywords(["assets", "visuals"], ["false", "positives"])
    render_sidebar_asset("False Positive Review", false_positive_path, "False positive review chart not found.")

    if model_info.get("details"):
        with st.expander("Model Load Details"):
            for detail in model_info["details"]:
                st.write(detail)

    st.markdown("---")
    with st.expander("Decision Framework"):
        st.write("High-confidence Uninfected: raw prediction >= 0.75")
        st.write("High-confidence Parasitized: raw prediction <= 0.25")
        st.write("Manual review zone: raw prediction between 0.25 and 0.75")

    st.warning("For research and educational use only. Not for clinical diagnosis.")
    st.caption("Developed by Jessica Palacio")


st.markdown(
    """
    <div class="hero">
        <h1>Malaria Detection AI</h1>
        <p>Portfolio-ready malaria microscopy classifier that presents model predictions, uncertainty handling, and validation evidence in a clean clinical-style workflow.</p>
    </div>
    """,
    unsafe_allow_html=True
)

hero_metric_1, hero_metric_2, hero_metric_3 = st.columns(3)
with hero_metric_1:
    render_stat_card("Architecture", "ResNet50", "Fine-tuned transfer learning pipeline")
with hero_metric_2:
    render_stat_card("Validation Accuracy", "97.88%", "Best selected deployment checkpoint")
with hero_metric_3:
    render_stat_card("Safety Policy", "Manual Review", "Predictions between 0.25 and 0.75")

render_workspace_intro(
    "Inference Workspace",
    "Upload a microscopy image or load a built-in example to generate a structured prediction report with confidence scoring and review guidance.",
)

top_left, top_right = st.columns([4.6, 1.2])

with top_left:
    uploaded_file = st.file_uploader(
        "Upload microscopy cell image",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

with top_right:
    st.caption("Controls")
    if st.button("Reset", use_container_width=True):
        st.rerun()

st.markdown("### Sample Cases")
demo_col1, demo_col2, demo_col3 = st.columns(3)

selected_demo = None

with demo_col1:
    if st.button("Load Sample Parasitized"):
        selected_demo = "sample_parasitized"

with demo_col2:
    if st.button("Load Sample Uninfected"):
        selected_demo = "sample_uninfected"

with demo_col3:
    st.caption("Optional: add sample images in `samples/`")

image = None
image_name = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_name = uploaded_file.name
elif selected_demo == "sample_parasitized":
    sample_path = first_valid_image_path(
        app_path("samples/parasitized_sample.png"),
        app_path("samples/parasitized/C68P29N_ThinF_IMG_20150819_134830_cell_79.png"),
        app_path("samples/parasitized/C39P4thinF_original_IMG_20150622_111206_cell_112.png"),
    )
    image = load_demo_image(sample_path) if sample_path else None
    image_name = os.path.basename(sample_path) if sample_path else "parasitized_sample.png"
    if image is None:
        st.warning("Sample image not found. Add it to: samples/parasitized_sample.png")
elif selected_demo == "sample_uninfected":
    sample_path = first_valid_image_path(
        app_path("samples/uninfected_sample.png"),
        app_path("samples/uninfected/uninfected/C7NthinF_IMG_20150611_104404_cell_160.png"),
    )
    image = load_demo_image(sample_path) if sample_path else None
    image_name = os.path.basename(sample_path) if sample_path else "uninfected_sample.png"
    if image is None:
        st.warning("Sample image not found. Add it to: samples/uninfected_sample.png")

if image is not None:
    left_col, right_col = st.columns([1.02, 1.08])

    with left_col:
        render_workspace_intro(
            "Case Preview",
            "Input image prepared for inference after RGB normalization and ResNet50 preprocessing.",
        )
        st.image(image, width="stretch")
        st.caption(f"Image: {image_name}")

    with right_col:
        if model is None:
            st.error("No compatible fine-tuned model could be loaded.")
            st.stop()

        resized = image.resize((128, 128))
        img_array = np.array(resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

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

        HIGH_CONF = 0.75
        LOW_CONF = 0.25

        if raw_prediction >= HIGH_CONF:
            final_label = "Uninfected"
            lead_text = "The model produced a high-confidence uninfected classification for this cell image."
            probability_value = float(uninfected_conf)
            status_call = st.success
            status_message = "Final classification: Uninfected"

        elif raw_prediction <= LOW_CONF:
            final_label = "Parasitized (Malaria Infected)"
            lead_text = "The model produced a high-confidence parasitized classification for this cell image."
            probability_value = float(parasitized_conf)
            status_call = st.error
            status_message = "Final classification: Parasitized (Malaria Infected)"

        else:
            final_label = "Uncertain - Manual Review Recommended"
            lead_text = "The prediction falls inside the manual review band, so this case should be escalated for human assessment."
            probability_value = 0.5
            status_call = st.warning
            status_message = "Final classification: Manual review recommended"

        st.markdown(
            """
            <div class="result-shell">
                <div class="result-kicker">Inference Report</div>
                <div class="result-title">Case Assessment</div>
                <div class="result-summary">Structured model output for the selected microscopy image, including predicted class, confidence tier, and recommended next action.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        status_call(status_message)
        st.progress(probability_value)

        result_metric_col1, result_metric_col2 = st.columns(2)
        with result_metric_col1:
            render_stat_card("Uninfected Probability", f"{uninfected_conf:.2%}")
        with result_metric_col2:
            render_stat_card("Parasitized Probability", f"{parasitized_conf:.2%}")

        st.info(lead_text)
        st.markdown("</div>", unsafe_allow_html=True)

        result_text = make_result_text(
            image_name=image_name,
            raw_prediction=raw_prediction,
            uninfected_conf=uninfected_conf,
            parasitized_conf=parasitized_conf,
            final_label=final_label,
            confidence_label=confidence_label,
            model_source=model_info["source"]
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
            st.write("**Manual review band:** 0.25 to 0.75")
            st.write("**Preprocessing:** RGB conversion, resize to 128x128, ResNet50 preprocess_input")

    st.markdown("---")
    st.subheader("Model Details")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Architecture", "ResNet50")
    with d2:
        st.metric("Input Size", "128 x 128")
    with d3:
        st.metric("Task", "Binary Classification")

else:
    render_workspace_intro(
        "Ready for Analysis",
        "Drag and drop a microscopy image, browse from disk, or open a sample case to generate the model assessment and review guidance.",
    )

st.markdown("---")
st.markdown(
    '<p class="small-muted">Built with Streamlit, TensorFlow, and a fine-tuned ResNet50 pipeline for malaria microscopy classification.</p>',
    unsafe_allow_html=True
)







