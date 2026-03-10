import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Ultrasound",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0c0f1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111422;
    border-right: 1px solid #1e2235;
}

/* Title block */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #f0f2ff;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Accent line */
.accent-bar {
    height: 3px;
    width: 60px;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    border-radius: 2px;
    margin-bottom: 1rem;
}

/* Cards */
.info-card {
    background: #151829;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Metric boxes */
.metric-box {
    background: #151829;
    border: 1px solid #1e2235;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 1.6rem; font-weight: 600; color: #a78bfa; }

/* Prediction badge */
.badge-benign  { background:#0d3b2e; color:#34d399; border:1px solid #065f46; border-radius:6px; padding:4px 12px; font-size:0.85rem; font-weight:600; }
.badge-malignant { background:#3b1111; color:#f87171; border:1px solid #7f1d1d; border-radius:6px; padding:4px 12px; font-size:0.85rem; font-weight:600; }
.badge-normal  { background:#0d1f3b; color:#60a5fa; border:1px solid #1e3a5f; border-radius:6px; padding:4px 12px; font-size:0.85rem; font-weight:600; }

/* Upload zone tweak */
[data-testid="stFileUploader"] {
    background: #151829;
    border: 1.5px dashed #2a3050;
    border-radius: 12px;
    padding: 0.5rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c5cfc);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.55rem 1.8rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Slider label */
.stSlider label { color: #9ca3af !important; }

/* Horizontal rule */
hr { border-color: #1e2235; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv12 model…")
def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def run_inference(model, img_array: np.ndarray, conf_thresh: float = 0.25):
    """Run model on an RGB numpy array; return annotated image + detections."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    results = model(tmp_path, conf=conf_thresh)
    os.unlink(tmp_path)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    detections = []

    for r in results:
        # Segmentation masks – red overlay
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                mask_uint8 = (mask.astype(np.uint8) * 255)
                # Resize mask to match image dimensions if needed
                if mask_uint8.shape[:2] != img_bgr.shape[:2]:
                    mask_uint8 = cv2.resize(mask_uint8, (img_bgr.shape[1], img_bgr.shape[0]))
                colored_mask = np.zeros_like(img_bgr)
                colored_mask[:, :, 2] = mask_uint8
                img_bgr = cv2.addWeighted(img_bgr, 1.0, colored_mask, 0.5, 0)

        # Bounding boxes
        if r.boxes is not None:
            boxes     = r.boxes.xyxy.cpu().numpy()
            scores    = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            names     = model.names
            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names[cls_id]} {score:.2f}"
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 80), 2)
                cv2.putText(img_bgr, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                detections.append({
                    "class": names[cls_id],
                    "confidence": float(score),
                    "bbox": [x1, y1, x2, y2],
                })

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections


def badge(class_name: str) -> str:
    cl = class_name.lower()
    if "benign" in cl:
        return f'<span class="badge-benign">● Benign</span>'
    elif "malignant" in cl:
        return f'<span class="badge-malignant">● Malignant</span>'
    else:
        return f'<span class="badge-normal">● Normal</span>'


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    weights_file = st.file_uploader(
        "Upload model weights (.pt)",
        type=["pt"],
        help="Upload your trained YOLOv12 best.pt weights file.",
    )

    st.markdown("**Inference settings**")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#4b5563; line-height:1.7'>
    <b style='color:#6b7280'>Model</b><br>YOLOv12 Segmentation<br><br>
    <b style='color:#6b7280'>Classes</b><br>Benign · Malignant · Normal<br><br>
    <b style='color:#6b7280'>Input size</b><br>640 × 640
    </div>
    """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">Breast Ultrasound<br>AI Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">YOLOv12 · Instance Segmentation · Clinical Decision Support</p>', unsafe_allow_html=True)

# Model loading
model = None
if weights_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(weights_file.read())
        tmp_weights = f.name
    try:
        model = load_model(tmp_weights)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.info(" Upload your **best.pt** weights in the sidebar to get started.")

st.markdown("---")

# Image upload
col_upload, col_gap, col_result = st.columns([1, 0.08, 1])

with col_upload:
    st.markdown("#### Upload Ultrasound Image")
    uploaded_img = st.file_uploader(
        "Drop a JPG / PNG ultrasound image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_img:
        pil_img   = Image.open(uploaded_img).convert("RGB")
        img_array = np.array(pil_img)
        st.image(img_array, caption="Original image", use_container_width=True)

        h, w = img_array.shape[:2]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Width</div><div class="metric-value">{w}px</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Height</div><div class="metric-value">{h}px</div></div>', unsafe_allow_html=True)

with col_result:
    st.markdown("#### Detection Result")

    if uploaded_img and model:
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Running inference…"):
                annotated, detections = run_inference(model, img_array, conf_threshold)

            st.image(annotated, caption="Annotated output", use_container_width=True)

            if detections:
                st.markdown("**Detections**")
                for i, d in enumerate(detections, 1):
                    conf_pct = d["confidence"] * 100
                    x1,y1,x2,y2 = d["bbox"]
                    st.markdown(
                        f'{badge(d["class"])} &nbsp; '
                        f'<span style="color:#9ca3af;font-size:0.85rem">'
                        f'Conf: <b style="color:#f0f2ff">{conf_pct:.1f}%</b> &nbsp;|&nbsp; '
                        f'Box: [{x1},{y1} → {x2},{y2}]</span>',
                        unsafe_allow_html=True,
                    )
                    st.progress(d["confidence"])

                # Download button
                buf = io.BytesIO()
                Image.fromarray(annotated).save(buf, format="JPEG", quality=95)
                st.download_button(
                    "⬇  Download annotated image",
                    data=buf.getvalue(),
                    file_name="breast_ultrasound_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            else:
                st.warning("No detections found at this confidence threshold. Try lowering it in the sidebar.")

    elif uploaded_img and model is None:
        st.markdown(
            '<div class="info-card" style="margin-top:3rem;text-align:center;color:#6b7280">'
            '⬅  Load your model weights<br>in the sidebar first.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-card" style="margin-top:3rem;text-align:center;color:#6b7280">'
            'Upload an ultrasound image<br>on the left to begin.</div>',
            unsafe_allow_html=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:0.75rem;color:#374151">'
    'For research and clinical decision support only · Not a substitute for professional medical diagnosis'
    '</p>',
    unsafe_allow_html=True,
)
