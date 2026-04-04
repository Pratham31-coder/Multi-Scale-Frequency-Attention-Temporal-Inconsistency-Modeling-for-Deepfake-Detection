<div align="center">

# 🎭 MFTD-Net
### Multi-Scale In-Feature Frequency Attention and Temporal Inconsistency Modelling for Video Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![AUC](https://img.shields.io/badge/Test%20AUC-0.985-blue?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-96.4%25-success?style=flat-square)]()

**Vardhaman College of Engineering, Hyderabad**
Department of Computer Science and Engineering (AI & ML) · Mini Project A8041

*Anveer Chetty · Niccunj Bajaj · Pratham Lahoti*
*Supervisor: Prof. M.A. Jabbar, Head & Professor*

</div>

---

## 📌 Overview

MFTD-Net (also referred to as **STFANet** — Spatial-Temporal Frequency Attention Network) is a dual-stream deep learning architecture for detecting deepfake videos. It combines **spatial appearance analysis** with **temporal frequency analysis**, fused through a cross-attention mechanism, to identify AI-generated face-swap videos with state-of-the-art accuracy.

> **Key idea:** Even when individual fake frames look visually perfect, GAN-generated videos introduce subtle periodic oscillations across frames that are invisible to the human eye but clearly detectable in the temporal frequency domain via Fast Fourier Transform.

---

## 🏆 Results

| Model | Val AUC | Test AUC | Test Accuracy | Parameters |
|-------|---------|----------|---------------|------------|
| XceptionNet | 0.920 | 0.890 | 85.0% | 22.9M |
| F3-Net | 0.940 | 0.912 | 87.3% | 31.4M |
| SFANet | 0.961 | 0.938 | 90.1% | 28.7M |
| GenConViT | 0.981 | 0.963 | 95.8% | 87.2M |
| **MFTD-Net (Ours)** | **0.997** | **0.985** | **96.4%** | **32.1M** |

- **GPU inference:** ~46ms per video (NVIDIA RTX 3060)
- **Dataset:** FaceForensics++ Deepfakes HQ

---

## ✨ Three Novel Contributions

### ① 4-Token Temporal FFT Frequency Stream
Unlike prior work that applies frequency analysis frame-by-frame, we apply a **1D Fast Fourier Transform along the temporal axis** across all 16 sampled frames. This directly measures GAN-introduced oscillation patterns. The output is compressed into **4 spatial region tokens** (face quadrants), preventing the attention collapse that single-token representations cause.

### ② 17×4 Cross-Attention Fusion
17 spatial queries (16 frame tokens + 1 CLS token) attend to 4 frequency keys and values, producing **136 independent attention decisions per video**. Each frame independently selects the most relevant frequency context — a genuinely content-aware fusion that concatenation-based methods cannot achieve.

### ③ Multi-Task Training with Decaying Auxiliary Loss
A per-frame auxiliary Binary Cross-Entropy loss is added to all 16 frame token outputs, with weight λ decaying from 0.3 → 0.05 during training. This prevents mode collapse in frame representations, forcing each frame to independently encode discriminative features.

```
L_total = L_main + λ_aux × L_aux
```

---

## 🏗️ Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Preprocessing                                   │
│  Haar Cascade Face Detection · 16 Frames · 224×224 │
└─────────────────────────────────────────────────┘
    │
    ├────────────────────┬────────────────────┐
    ▼                    ▼
┌──────────────┐    ┌──────────────────────────┐
│ Spatial      │    │ Frequency Stream ①        │
│ Stream       │    │                          │
│              │    │ 1D FFT (temporal axis)   │
│ ConvNeXt-    │    │ 16 frames → 9 freq bins  │
│ Tiny         │    │ Conv Tower 27→128→256    │
│              │    │ 2×2 Adaptive Pool        │
│ 768-dim      │    │ 4 tokens × 768-dim       │
│ per frame    │    │                          │
│              │    │                          │
│ + Sinusoidal │    └──────────────────────────┘
│   Pos. Enc.  │                │
│ + CLS Token  │                │
│              │                │
│ 17 × 768     │    Keys & Values (4 × 768)
└──────────────┘                │
    │  Queries (17 × 768)       │
    └────────────┬──────────────┘
                 ▼
    ┌─────────────────────────┐
    │ Cross-Attention ②        │
    │ 17×4 = 136 decisions     │
    │ 8 heads · Pre-LN        │
    └─────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Transformer Encoder      │
    │ 2 layers · 8 heads       │
    │ FFN dim 3072 · GELU      │
    └─────────────────────────┘
         │               │
         ▼               ▼
   CLS Token       16 Frame Tokens
   → MLP Head      → Auxiliary Head ③
   → P(fake)       → Frame BCE Loss
```

---

## 🗂️ Dataset

**FaceForensics++ Deepfakes HQ**

| Split | Total | Real | Fake |
|-------|-------|------|------|
| Training | 1,397 | 699 | 698 |
| Validation | 359 | 180 | 179 |
| Test | 240 | 120 | 120 |
| **Total** | **1,996** | **999** | **997** |

Download: [FaceForensics++ Official Repository](https://github.com/ondyari/FaceForensics)

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
torch >= 2.0
torchvision
opencv-python
streamlit
numpy
```

### Installation

```bash
git clone https://github.com/yourusername/MFTD-Net.git
cd MFTD-Net
pip install -r requirements.txt
```

### Training

```bash
python train.py \
  --data_dir /path/to/faceforensics \
  --epochs 15 \
  --batch_size 8 \
  --lr_backbone 2e-5 \
  --lr_new 8e-4
```

### Inference (Single Video)

```bash
python inference.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pth
```

### Streamlit Web Application

```bash
streamlit run app.py
```

Upload any face video and get a real-time deepfake probability score with visual frame inspection.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Backbone learning rate | 2 × 10⁻⁵ |
| New layers learning rate | 8 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 8 videos |
| Frames per video | 16 |
| Input resolution | 224 × 224 |
| Transformer layers | 2 |
| Attention heads | 8 |
| Dropout | 0.3 |
| Label smoothing (ε) | 0.05 |
| λ_aux (initial → final) | 0.3 → 0.05 |
| Best checkpoint epoch | 7 |

---

## 🔬 Ablation Study

Each architectural component was validated independently:

| Configuration | Val AUC | AUC Drop |
|--------------|---------|----------|
| Full MFTD-Net | **0.9901** | — |
| Without frequency stream | 0.9512 | −3.89% |
| Single frequency token (not 4) | 0.9589 | −3.12% |
| Concatenation fusion (no cross-attention) | 0.9634 | −2.67% |
| Without auxiliary frame loss | 0.9723 | −1.78% |
| Without Pre-LayerNorm | 0.9812 | −0.89% |

---

## 📁 Project Structure

```
MFTD-Net/
├── model/
│   ├── stfanet.py          # Main architecture
│   ├── spatial_stream.py   # ConvNeXt-Tiny spatial stream
│   ├── freq_stream.py      # Temporal FFT frequency stream
│   └── cross_attention.py  # 17×4 cross-attention fusion
├── data/
│   ├── dataset.py          # FaceForensics++ dataloader
│   └── preprocess.py       # Face detection & frame sampling
├── train.py                # Training script
├── inference.py            # Single video inference
├── app.py                  # Streamlit web application
├── checkpoints/            # Saved model weights
├── requirements.txt
└── README.md
```

---

## 📊 Inference Latency

| Pipeline Stage | GPU (RTX 3060) | CPU Only |
|---------------|----------------|----------|
| Frame sampling | 12 ms | 18 ms |
| Face detection (16 frames) | 8 ms | 42 ms |
| Crop + resize + normalize | 3 ms | 5 ms |
| ConvNeXt-Tiny forward pass | 15 ms | 310 ms |
| FFT stream + conv tower | 4 ms | 28 ms |
| Cross-attention + Transformer | 3 ms | 22 ms |
| Classification head | <1 ms | <1 ms |
| **Total** | **~46 ms** | **~426 ms** |

---

## 🌍 SDG Alignment

| Goal | Contribution |
|------|-------------|
| **SDG 9** — Industry & Innovation | Reliable deepfake detection for digital infrastructure |
| **SDG 16** — Peace & Justice | Per-segment visualization for journalists and legal analysts |
| **SDG 17** — Partnerships | Built on the public FaceForensics++ benchmark; reproducible |

---

## 📚 References

Key papers this work builds on:

- Rossler et al. — *FaceForensics++* (ICCV 2019)
- Afchar et al. — *MesoNet* (WIFS 2018)
- Qian et al. — *F3-Net* (ECCV 2020)
- Liu et al. — *ConvNeXt* (CVPR 2022)
- Loshchilov & Hutter — *AdamW* (ICLR 2019)


---



---

<div align="center">

**Vardhaman College of Engineering · Department of CSE (AI & ML)**
Mini Project A8041 · Academic Year 2025–26

*Made with ❤️ by Anveer Chetty, Niccunj Bajaj, and Pratham Lahoti*

</div>
