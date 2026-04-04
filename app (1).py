import math, warnings, os, tempfile
warnings.filterwarnings("ignore", message="enable_nested_tensor")

import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
NUM_FRAMES  = 16
IMG_SIZE    = 224
FEAT_DIM    = 768
N_HEADS     = 8
N_LAYERS    = 2
FREQ_TOKENS = 4
DROPOUT     = 0.3
THRESHOLD   = 0.5
MODEL_PATH  = "best_stfanet.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# FACE DETECTOR — OpenCV Haar Cascade (built into cv2, no pip)
# ─────────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(frame_bgr: np.ndarray, pad: float = 0.3):
    """
    Returns (cropped_rgb_array, bbox_or_None, status_string)
    pad: fraction of face box to add as margin on each side
    Falls back to center-crop if no face detected.
    """
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces  = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    H, W = frame_bgr.shape[:2]

    if len(faces) == 0:
        # Fallback: center crop (square, 80% of shorter side)
        side   = int(min(H, W) * 0.8)
        cx, cy = W // 2, H // 2
        x1     = max(0, cx - side // 2)
        y1     = max(0, cy - side // 2)
        x2     = min(W, x1 + side)
        y2     = min(H, y1 + side)
        crop   = frame_bgr[y1:y2, x1:x2]
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return rgb, None, "⚠️ No face — center crop"

    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding
    pad_x = int(w * pad)
    pad_y = int(h * pad)
    x1    = max(0, x - pad_x)
    y1    = max(0, y - pad_y)
    x2    = min(W, x + w + pad_x)
    y2    = min(H, y + h + pad_y)

    crop = frame_bgr[y1:y2, x1:x2]
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb, (x1, y1, x2, y2), "✅ Face detected"


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class FrequencyStream(nn.Module):
    def __init__(self, num_frames=16, spatial_size=14, out_dim=768, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        freq_bins = num_frames // 2 + 1
        self.pool = nn.AdaptiveAvgPool2d(spatial_size)
        self.conv = nn.Sequential(
            nn.Conv2d(3 * freq_bins, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(int(num_tokens ** 0.5)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_s = self.pool(x.view(B * T, C, H, W))
        _, _, s, _ = x_s.shape
        x_s = x_s.view(B, T, C, s, s)
        x_t = x_s.permute(0, 2, 3, 4, 1).float()
        mag = torch.fft.rfft(x_t, dim=-1).abs()
        freq_bins = T // 2 + 1
        mag = mag.permute(0, 1, 4, 2, 3).reshape(B, C * freq_bins, s, s)
        feats = self.conv(mag)
        B_, C_, h_, w_ = feats.shape
        feats = feats.view(B_, C_, h_ * w_).permute(0, 2, 1)
        return self.proj(feats)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class STFANet(nn.Module):
    def __init__(self, num_frames=16, feat_dim=768, n_heads=8,
                 n_layers=2, dropout=0.3, freq_tokens=4):
        super().__init__()
        self.feat_dim = feat_dim
        backbone = torchvision.models.convnext_tiny(weights=None)
        self.spatial_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.pos_enc     = TemporalPositionalEncoding(feat_dim, max_len=num_frames + 1, dropout=dropout)
        self.freq_stream = FrequencyStream(num_frames=num_frames, out_dim=feat_dim, num_tokens=freq_tokens)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cross_attn  = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cross_norm  = nn.LayerNorm(feat_dim)
        enc_layer        = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim * 4,
                                                      dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.aux_head    = nn.Linear(feat_dim, 1)
        self.head        = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        sp            = self.spatial_backbone(x.view(B * T, C, H, W))
        sp            = sp.flatten(1).view(B, T, self.feat_dim)
        cls           = self.cls_token.expand(B, -1, -1)
        sp            = torch.cat([cls, sp], dim=1)
        sp            = self.pos_enc(sp)
        fr            = self.freq_stream(x)
        attn_out, _   = self.cross_attn(sp, fr, fr)
        fused         = self.cross_norm(sp + attn_out)
        out           = self.transformer(fused)
        return self.head(out[:, 0]).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Checkpoint not found at: {os.path.abspath(MODEL_PATH)}")
        st.stop()

    model = STFANet(
        num_frames=NUM_FRAMES, feat_dim=FEAT_DIM, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=DROPOUT, freq_tokens=FREQ_TOKENS,
    ).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["epoch"], ckpt["best_auc"]


# ─────────────────────────────────────────────────────────────
# TRANSFORM (applied AFTER face crop)
# ─────────────────────────────────────────────────────────────
val_tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
# FRAME EXTRACTION + FACE CROP
# ─────────────────────────────────────────────────────────────
def extract_frames_with_faces(video_path: str, num_frames: int):
    """
    Returns:
        tensors      : torch.Tensor (T, 3, 224, 224) — model input
        preview_imgs : list of PIL Images (face crops, pre-normalize)
        debug_info   : list of dicts with per-frame metadata
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    if total == 0:
        raise ValueError("Could not read any frames from video.")

    idxs      = np.linspace(0, total - 1, num_frames).astype(int)
    tensors   = []
    previews  = []
    debug_info = []

    for rank, frame_idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame_bgr = cap.read()

        if not ret or frame_bgr is None:
            frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            status    = "❌ Read failed — black frame"
            bbox      = None
        else:
            frame_rgb_crop, bbox, status = detect_and_crop_face(frame_bgr)
            frame_bgr = cv2.cvtColor(frame_rgb_crop, cv2.COLOR_RGB2BGR)  # keep bgr for consistency

        rgb_crop   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img    = Image.fromarray(rgb_crop)
        preview    = pil_img.resize((112, 112))    # smaller for grid display

        tensors.append(val_tfm(pil_img))
        previews.append(preview)
        debug_info.append({
            "rank"       : rank + 1,
            "frame_idx"  : int(frame_idx),
            "timestamp"  : f"{frame_idx / fps:.2f}s" if fps > 0 else "?",
            "face_status": status,
            "bbox"       : bbox,
        })

    cap.release()
    return torch.stack(tensors), previews, debug_info, total, fps


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict(video_path: str):
    model, ckpt_epoch, ckpt_auc = load_model()
    tensors, previews, debug_info, total_frames, fps = \
        extract_frames_with_faces(video_path, NUM_FRAMES)

    x = tensors.unsqueeze(0).to(DEVICE)   # (1, T, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        logit = model(x.float())

    raw_logit = logit.squeeze().item()
    prob_fake = torch.sigmoid(logit).squeeze().item()
    prob_real = 1.0 - prob_fake

    return {
        "prob_fake"    : prob_fake,
        "prob_real"    : prob_real,
        "raw_logit"    : raw_logit,
        "previews"     : previews,
        "debug_info"   : debug_info,
        "total_frames" : total_frames,
        "fps"          : fps,
        "ckpt_epoch"   : ckpt_epoch,
        "ckpt_auc"     : ckpt_auc,
    }


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deepfake Detector", page_icon="🎭", layout="wide")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Info")
    if st.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.rerun()
    try:
        _, ep, auc = load_model()
        st.success(f"Checkpoint loaded")
        st.metric("Epoch saved",  ep)
        st.metric("Val AUC",      f"{auc:.4f}")
        st.metric("Device",       DEVICE.upper())
        st.metric("Frames sampled", NUM_FRAMES)
    except Exception as e:
        st.error(f"Model load failed: {e}")

st.title("🎭 Deepfake Video Detector")
st.markdown("Upload a **face video**. The model samples **16 evenly-spaced frames**, "
            "crops faces, and runs **STFANet** (ConvNeXt + Frequency + Transformer).")

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col_vid, col_info = st.columns([2, 1])
    with col_vid:
        st.video(tmp_path)
    with col_info:
        st.info(f"📁 File: `{uploaded.name}`\n\n📦 Size: `{uploaded.size / 1024:.1f} KB`")

    if st.button("🔍 Detect Deepfake", use_container_width=True):
        with st.spinner("Extracting frames, detecting faces, running STFANet…"):
            try:
                result = predict(tmp_path)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

        # ── RESULT ──────────────────────────────────────────────
        label = "FAKE" if result["prob_fake"] >= THRESHOLD else "REAL"
        color = "#FF4B4B" if label == "FAKE" else "#21C354"

        st.markdown("---")
        st.markdown(
            f"<h1 style='text-align:center; color:{color}; font-size:3rem;'>"
            f"{'🔴' if label=='FAKE' else '🟢'} {label}</h1>",
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Fake Prob",   f"{result['prob_fake']*100:.1f}%")
        m2.metric("🟢 Real Prob",   f"{result['prob_real']*100:.1f}%")
        m3.metric("📐 Raw Logit",   f"{result['raw_logit']:.4f}")
        m4.metric("🎞 Total Frames", result["total_frames"])

        st.progress(result["prob_fake"])

        # ── DEBUG PANEL ─────────────────────────────────────────
        st.markdown("---")
        with st.expander("🔬 Debug Panel — Sampled Frames & Face Detection", expanded=True):

            # Video stats
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Total video frames",  result["total_frames"])
            dc2.metric("FPS",                 f"{result['fps']:.1f}")
            dc3.metric("Frames sampled",       NUM_FRAMES)

            st.markdown("#### 🖼 Sampled Face Crops (16 frames)")

            # Show 4 frames per row
            rows = [result["previews"][i:i+4] for i in range(0, NUM_FRAMES, 4)]
            info_rows = [result["debug_info"][i:i+4] for i in range(0, NUM_FRAMES, 4)]

            for row_imgs, row_info in zip(rows, info_rows):
                cols = st.columns(4)
                for col, img, info in zip(cols, row_imgs, row_info):
                    with col:
                        st.image(img, use_container_width=True)
                        st.caption(
                            f"**Frame #{info['rank']}**\n\n"
                            f"idx: `{info['frame_idx']}` | ⏱ `{info['timestamp']}`\n\n"
                            f"{info['face_status']}"
                        )

            # Per-frame table
            st.markdown("#### 📋 Per-Frame Summary")
            face_detected = sum(1 for d in result["debug_info"] if "✅" in d["face_status"])
            no_face       = NUM_FRAMES - face_detected

            sc1, sc2 = st.columns(2)
            sc1.metric("✅ Faces detected",    face_detected)
            sc2.metric("⚠️ Fallback (no face)", no_face)

            if no_face > NUM_FRAMES // 2:
                st.warning(
                    f"⚠️ Only {face_detected}/{NUM_FRAMES} frames had a detectable face. "
                    "This video may not contain a close-up face, which will degrade accuracy. "
                    "Try a video with a clearly visible frontal face."
                )

            # Logit interpretation
            st.markdown("#### 🔢 Logit Interpretation")
            st.code(
                f"Raw logit   : {result['raw_logit']:.6f}\n"
                f"sigmoid()   : {result['prob_fake']:.6f}\n"
                f"Threshold   : {THRESHOLD}\n"
                f"Decision    : {'FAKE' if result['prob_fake'] >= THRESHOLD else 'REAL'}\n\n"
                f"Logit < -2  → strongly REAL (prob < 12%)\n"
                f"Logit ≈  0  → uncertain  (prob ≈ 50%)  ← bad sign if always here\n"
                f"Logit > +2  → strongly FAKE (prob > 88%)"
            )
