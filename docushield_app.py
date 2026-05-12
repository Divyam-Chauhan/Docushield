import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
import io
import os

st.set_page_config(
    page_title="DocuShield",
    page_icon="🛡️",
    layout="wide"
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #0a0a0f; }
    
    .hero {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0a1628 100%);
        border: 1px solid #2a2a4a;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin-bottom: 32px;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
    }
    
    .result-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
    .genuine {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a;
    }
    .forged {
        background: linear-gradient(135deg, #2d0a0a, #450a0a);
        border: 1px solid #dc2626;
    }
    .suspicious {
        background: linear-gradient(135deg, #1c1100, #2d1f00);
        border: 1px solid #d97706;
    }
    
    .score-big {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    .verdict {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .verdict-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 6px;
    }
    
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }
    .metric-box {
        flex: 1;
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-label { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #f9fafb; font-size: 1.4rem; font-weight: 600; margin-top: 4px; }
    
    .section-header {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1f2937;
    }
    
    .tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px 2px;
    }
    .tag-purple { background: #3b0764; color: #d8b4fe; border: 1px solid #7c3aed; }
    .tag-blue   { background: #0c1a3d; color: #93c5fd; border: 1px solid #3b82f6; }
    
    .upload-hint {
        color: #475569;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 6px;
    }

    div[data-testid="stTabs"] button {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>&#128737; DocuShield</h1>
    <p>AI-powered forensic document &amp; signature analysis &middot; MobileNetV2 &middot; ELA &middot; Adaptive Ink Isolation</p>
</div>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_gray(uploaded) -> np.ndarray:
    img = Image.open(uploaded).convert("RGB")
    return np.array(img)

@st.cache_resource
def load_feature_extractor():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model

def preprocess_ink_only(img_rgb: np.ndarray) -> np.ndarray:
    """
    Isolates ink from the background using the blue channel and adaptive
    thresholding. This handles uneven lighting, shadows, and paper texture
    without treating them as part of the signature.
    """
    # Pen ink absorbs blue light, shadows and paper do not.
    # Inverting the blue channel makes ink bright and background dark.
    b_inv = 255 - img_rgb[:, :, 2]
    
    # Adaptive thresholding evaluates each region independently,
    # so a shadow on one side doesn't corrupt the whole image.
    binary = cv2.adaptiveThreshold(
        b_inv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=-10
    )
    
    # Remove isolated noise pixels from paper texture / dust
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small gaps in strokes to make them continuous for the CNN
    kernel_close = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return processed

def get_signature_embedding(img_rgb: np.ndarray, model) -> np.ndarray:
    binary = preprocess_ink_only(img_rgb)
    
    # Invert to Black ink on White background for better feature extraction.
    # Most CNNs are trained on natural/white backgrounds.
    processed = 255 - binary
    
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 100:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 15
        y1 = max(0, y - margin)
        x1 = max(0, x - margin)
        y2 = min(img_rgb.shape[0], y + h + margin)
        x2 = min(img_rgb.shape[1], x + w + margin)
        cropped = processed[y1:y2, x1:x2]
    else:
        cropped = processed
        
    img_pil = Image.fromarray(cropped).convert("RGB")
    
    # Pad to square (White canvas) to preserve aspect ratio
    w, h = img_pil.size
    max_dim = max(w, h)
    pl = (max_dim - w) // 2
    pt = (max_dim - h) // 2
    pr = (max_dim - w + 1) // 2
    pb = (max_dim - h + 1) // 2
    img_padded = ImageOps.expand(img_pil, (pl, pt, pr, pb), fill=(255, 255, 255))
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img_padded).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
        
    return features.numpy().flatten()

def compute_dl_similarity(img1: np.ndarray, img2: np.ndarray, model):
    emb1 = get_signature_embedding(img1, model)
    emb2 = get_signature_embedding(img2, model)
    sim = 1.0 - cosine(emb1, emb2)
    return sim

def verdict_html(dl_sim):
    # Thresholds calibrated from Black-on-White empirical testing (Phase 5):
    # Genuine avg: 0.887 | Impostor avg: 0.664 | Gap: +0.223
    pct = round(dl_sim * 100, 1)
    
    if dl_sim >= 0.82:
        cls, icon, label, sub = "genuine", "&#10003;", "GENUINE / HIGH MATCH", "Signatures share strong feature embeddings — consistent with the same author"
        color = "#4ade80"
    elif dl_sim >= 0.74:
        cls, icon, label, sub = "suspicious", "&#9888;", "SUSPICIOUS", "Feature embeddings show stylistic divergence — recommend manual review"
        color = "#fbbf24"
    else:
        cls, icon, label, sub = "forged", "&#9888;", "LIKELY FORGED", "Feature embeddings indicate a structurally different author"
        color = "#f87171"
        
    return f"""
    <div class="result-card {cls}">
        <div class="score-big" style="color:{color}">{pct}%</div>
        <div class="verdict" style="color:{color}">{icon} {label}</div>
        <div class="verdict-sub">{sub}</div>
    </div>
    """, pct

# ─── ELA ──────────────────────────────────────────────────────────────────────

def run_ela(uploaded, quality=75, amplify=15):
    img = Image.open(uploaded).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr = np.array(img, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    ela = np.abs(orig_arr - recomp_arr) * amplify
    ela = np.clip(ela, 0, 255).astype(np.uint8)

    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
    mean_ela = float(np.mean(ela_gray))
    max_ela  = float(np.max(ela_gray))
    hotspot_ratio = float(np.sum(ela_gray > 40) / ela_gray.size)

    return img, ela, ela_gray, mean_ela, max_ela, hotspot_ratio

def ela_verdict(mean_ela, hotspot_ratio):
    risk = mean_ela * 0.5 + hotspot_ratio * 100 * 0.5
    if risk < 8:
        cls, icon, label, sub, color = "genuine", "✅", "LIKELY AUTHENTIC", "Low error levels — no signs of digital manipulation", "#4ade80"
    elif risk < 20:
        cls, icon, label, sub, color = "suspicious", "⚠️", "POSSIBLY TAMPERED", "Moderate anomalies detected in compression artifacts", "#fbbf24"
    else:
        cls, icon, label, sub, color = "forged", "🚨", "TAMPERED / FORGED", "High error levels indicate digital editing or insertion", "#f87171"
    return f"""
    <div class="result-card {cls}">
        <div class="score-big" style="color:{color}">{round(risk,1)}</div>
        <div class="verdict" style="color:{color}">{icon} {label}</div>
        <div class="verdict-sub">{sub}</div>
    </div>
    """, risk

def ela_heatmap_fig(original_img, ela_arr):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#0f172a')

    axes[0].imshow(original_img)
    axes[0].set_title("Original Document", color='#94a3b8', fontsize=9)
    axes[0].axis('off')

    ela_gray = cv2.cvtColor(ela_arr, cv2.COLOR_RGB2GRAY)
    im = axes[1].imshow(ela_gray, cmap='hot', vmin=0, vmax=100)
    axes[1].set_title("ELA Heatmap  (brighter = more suspicious)", color='#94a3b8', fontsize=9)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color='#94a3b8', labelcolor='#94a3b8')

    plt.tight_layout(pad=0.5)
    return fig

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["✍️  Approach 1 — Signature Verification", "🔬  Approach 2 — Document ELA"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Deep Learning Embeddings
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    dl_model = load_feature_extractor()
    st.markdown('<div class="section-header">Upload Signatures</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        ref_file = st.file_uploader("Reference Signature (Genuine)", type=["png","jpg","jpeg","bmp"], key="ref")
        st.markdown('<p class="upload-hint">The known authentic signature</p>', unsafe_allow_html=True)
    with c2:
        que_file = st.file_uploader("Questioned Signature (Suspect)", type=["png","jpg","jpeg","bmp"], key="que")
        st.markdown('<p class="upload-hint">The signature to be verified</p>', unsafe_allow_html=True)

    if ref_file and que_file:
        ref_rgb = load_gray(ref_file)
        que_rgb = load_gray(que_file)

        with st.spinner("Extracting Deep Learning Embeddings..."):
            dl_sim = compute_dl_similarity(ref_rgb, que_rgb, dl_model)

        vhtml, pct = verdict_html(dl_sim)

        st.markdown('<div class="section-header">AI Embeddings Result</div>', unsafe_allow_html=True)
        st.markdown(vhtml, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Cosine Similarity</div>
                <div class="metric-value">{round(dl_sim, 4)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Embedding Match</div>
                <div class="metric-value">{pct}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Threshold</div>
                <div class="metric-value">>82% Genuine</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Impostor Avg</div>
                <div class="metric-value">~66%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:16px">
            <span class="tag tag-purple">MobileNetV2</span>
            <span class="tag tag-purple">Cosine Similarity</span>
            <span class="tag tag-blue">PyTorch</span>
            <span class="tag tag-blue">Torchvision</span>
            <span class="tag tag-blue">Deep Learning Embeddings</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Upload both signatures above to run the analysis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ELA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Upload Document</div>', unsafe_allow_html=True)
    
    doc_file = st.file_uploader("Upload scanned document (JPEG/PNG)", type=["jpg","jpeg","png"], key="doc")
    st.markdown('<p class="upload-hint">Best results with JPEG scans — ELA relies on JPEG compression artifacts</p>', unsafe_allow_html=True)

    if doc_file:
        with st.spinner("Running Error Level Analysis..."):
            orig_img, ela_arr, ela_gray, mean_ela, max_ela, hotspot_ratio = run_ela(doc_file)

        vhtml2, risk = ela_verdict(mean_ela, hotspot_ratio)

        st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)
        st.markdown(vhtml2, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Mean ELA</div>
                <div class="metric-value">{round(mean_ela,2)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max ELA</div>
                <div class="metric-value">{round(max_ela,1)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Hotspot Ratio</div>
                <div class="metric-value">{round(hotspot_ratio*100,2)}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{round(risk,1)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">ELA Heatmap</div>', unsafe_allow_html=True)
        fig_ela = ela_heatmap_fig(orig_img, ela_arr)
        st.pyplot(fig_ela, use_container_width=True)
        plt.close()

        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px;margin-top:12px;color:#64748b;font-size:0.82rem;line-height:1.6">
            <b style="color:#94a3b8">How ELA works:</b> The document is re-saved at lower JPEG quality. Authentic regions that haven't been edited recompress similarly to before. 
            Regions that were digitally inserted or modified (e.g. a pasted signature) retain different compression artifacts — 
            these show up as <b style="color:#f87171">bright hotspots</b> on the heatmap.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:16px">
            <span class="tag tag-purple">ELA</span>
            <span class="tag tag-purple">JPEG Artifact Analysis</span>
            <span class="tag tag-blue">PIL</span>
            <span class="tag tag-blue">OpenCV</span>
            <span class="tag tag-blue">NumPy</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Upload a scanned document above to detect tampering.")
