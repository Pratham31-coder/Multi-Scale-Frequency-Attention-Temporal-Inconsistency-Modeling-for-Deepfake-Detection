<div align="center">

# 🎭 MFTD-Net
### Multi-Scale In-Feature Frequency Attention and Temporal Inconsistency Modelling<br>for Video Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Test AUC](https://img.shields.io/badge/Test%20AUC-0.985-3b82f6?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-96.4%25-22c55e?style=flat-square)]()

**Vardhaman College of Engineering, Hyderabad**  
Department of Computer Science and Engineering (AI & ML) · Mini Project A8041 · 2025–26

*Anveer Chetty (23881A6667) · Niccunj Bajaj (23881A6698) · Pratham Lahoti (23881A66A5)*  
*Supervisor: Prof. M.A. Jabbar, Head & Professor, Dept. of CSE (AI&ML)*

</div>

---

## 📌 Overview

**MFTD-Net** (also referred to as **STFANet** — Spatial-Temporal Frequency Attention Network) is a dual-stream deep learning architecture for detecting deepfake videos. It combines **spatial appearance features** extracted by a ConvNeXt-Tiny backbone with **temporal frequency features** computed via Fast Fourier Transform along the time axis, fused through a **cross-attention mechanism** and refined by a **Transformer encoder**.

> **Core insight:** Even when individual fake frames look visually perfect, GAN-generated videos introduce subtle periodic oscillations across frames that are invisible to the human eye but clearly detectable in the temporal frequency domain.

The system is trained end-to-end on the **FaceForensics++ Deepfakes HQ** benchmark and deployed as a **Streamlit web application** with real-time inference.

---

## 🏆 Results

| Model | Val AUC | Test AUC | Test Accuracy | Params |
|-------|---------|----------|---------------|--------|
| XceptionNet | 0.920 | 0.890 | 85.0% | 22.9M |
| F3-Net | 0.940 | 0.912 | 87.3% | 31.4M |
| SFANet | 0.961 | 0.938 | 90.1% | 28.7M |
| GenConViT | 0.981 | 0.963 | 95.8% | 87.2M |
| **MFTD-Net (Ours)** | **0.997** | **0.985** | **96.4%** | **32.1M** |

**GPU inference:** ~46 ms/video on NVIDIA RTX 3060

---

## ✨ Three Novel Contributions

### ① 4-Token Temporal FFT Frequency Stream
A `FrequencyStream` module applies a **1D real FFT (`torch.fft.rfft`) along the temporal axis** across all 16 sampled frames. This converts time-domain pixel signals into a frequency-domain representation that captures GAN-introduced oscillation patterns. The output is grouped into **4 spatial region tokens** via a convolutional tower and 2×2 adaptive pooling — one token per face quadrant — each projected to 768 dimensions.

```python
# Core of FrequencyStream.forward()
x_t  = x_s.permute(0, 2, 3, 4, 1).float()        # (B, C, H, W, T)
mag  = torch.fft.rfft(x_t, dim=-1).abs()          # 9 frequency bins for T=16
mag  = mag.permute(0, 1, 4, 2, 3)                 # (B, C*freq_bins, H, W)
feats = self.conv(mag)                             # Conv tower → 4 spatial tokens
```

### ② 17×4 Cross-Attention Fusion
17 spatial queries (16 frame tokens + 1 CLS token) attend to 4 frequency keys and values using `nn.MultiheadAttention` with 8 heads, producing **136 independent attention decisions per video**. A residual connection with Pre-LayerNorm ensures stable gradients.

```python
attn_out, _ = self.cross_attn(sp, fr, fr)     # Q=spatial, K=V=frequency
fused       = self.cross_norm(sp + attn_out)  # Pre-LN residual
```

### ③ Multi-Task Training with Decaying Auxiliary Loss
An auxiliary per-frame BCE head forces each of the 16 frame tokens to independently encode discriminative features, preventing mode collapse. The auxiliary weight λ decays from **0.3 → 0.05** across training.

```
L_total = L_main + λ_aux × L_aux
```

---

## 🗂️ Repository Structure

```
MFTD-Net/
│
├── final.ipynb          # Training notebook (run on Kaggle with GPU)
│   ├── Cell 0           # Imports, seed, device setup
│   ├── Cell 1           # Config dict — all hyperparameters in one place
│   ├── Cell 2           # Dataset manifest builder (FaceForensics++ layout)
│   ├── Cell 3           # Train / Val / Test split (70 / 18 / 12 %)
│   ├── Cell 4           # Transforms, FFImageSeqDataset, DataLoaders
│   ├── Cell 5           # SAM optimizer + WeightedBCELoss definitions
│   ├── Cell 6           # AdamW + CosineAnnealingLR + GradScaler setup
│   ├── Cell 7           # run_epoch() training loop with AMP + grad clipping
│   ├── Cell 8           # Training curves (Loss / Accuracy / AUC plots)
│   └── Cell 9           # Test evaluation + confusion matrix
│
├── app.py               # Streamlit web app — inference only
│
├── best_stfanet.pt      # Trained checkpoint (download separately — see below)
│
└── README.md
```

---

## 🏗️ Architecture

```
Video Input (MP4 / AVI / MOV / MKV)
        │
        ▼
  Haar Cascade Face Detection (OpenCV)
  16 evenly-spaced frames · 30% padding · 224×224 resize
        │
        ├───────────────────────────────────────────────┐
        ▼                                               ▼
┌──────────────────────┐              ┌────────────────────────────────┐
│   Spatial Stream     │              │   Frequency Stream  ①          │
│                      │              │                                │
│  ConvNeXt-Tiny       │              │  AdaptiveAvgPool → 14×14      │
│  (ImageNet-1K)       │              │  rfft(dim=-1) → 9 freq bins   │
│                      │              │  Conv2d 27→128→256 + GELU     │
│  768-dim per frame   │              │  AdaptiveAvgPool2d(2×2)       │
│  + sinusoidal pos enc│              │  Linear 256→768               │
│  + CLS token prepend │              │                                │
│                      │              │  4 tokens × 768-dim            │
│  17 × 768            │              └────────────────────────────────┘
└──────────────────────┘                              │
        │  Queries (17×768)           Keys & Values (4×768)
        └──────────────────┬──────────────────────────┘
                           ▼
             ┌─────────────────────────────┐
             │  Cross-Attention  ②          │
             │  nn.MultiheadAttention       │
             │  8 heads · batch_first=True  │
             │  Pre-LayerNorm residual       │
             │  17 × 4 = 136 decisions      │
             └─────────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────────┐
             │  Transformer Encoder         │
             │  2 × TransformerEncoderLayer │
             │  d_model=768  nhead=8        │
             │  dim_feedforward=3072        │
             │  norm_first=True  GELU       │
             └─────────────────────────────┘
                    │               │
                    ▼               ▼
              CLS Token        16 Frame Tokens
              (index 0)
                    │               │
                    ▼               ▼
          ┌─────────────┐   ┌──────────────────┐
          │  Main Head  │   │  Auxiliary Head ③ │
          │  LN→Drop    │   │  Linear 768→1     │
          │  →Linear    │   │  Frame-level BCE  │
          │  768→256    │   │  λ: 0.30 → 0.05   │
          │  →GELU→Drop │   └──────────────────┘
          │  →Linear    │
          │  256→1      │
          │  →σ = P(fake)│
          └─────────────┘
```

---

## 🗄️ Dataset

**FaceForensics++ Deepfakes HQ** — the standard benchmark for deepfake detection.

Expected on-disk structure (Kaggle dataset: `adham7elmy/faceforencispp-extracted-frames`):

```
faceforencispp-extracted-frames/
├── fake/
│   └── Deepfakes/
│       ├── 000/   (000.png, 001.png, ...)
│       ├── 001/
│       └── ...
└── real/
    ├── 000/
    ├── 001/
    └── ...
```

| Split | Videos | Real | Fake |
|-------|--------|------|------|
| Train | 1,397  | 699  | 698  |
| Val   | 359    | 180  | 179  |
| Test  | 240    | 120  | 120  |
| **Total** | **1,996** | **999** | **997** |

Official download: [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MFTD-Net.git
cd MFTD-Net
```

### 2. Install dependencies

```bash
pip install torch torchvision streamlit opencv-python pillow \
            numpy pandas scikit-learn tqdm matplotlib
```

### 3. Train (Kaggle GPU recommended)

Open `final.ipynb` on Kaggle. Attach the FaceForensics++ dataset and select **GPU T4 x2** as accelerator. The config in Cell 1 controls everything:

```python
CFG = dict(
    data_root   = "/kaggle/input/datasets/adham7elmy/faceforencispp-extracted-frames",
    fake_method = "Deepfakes",
    savedir     = "/kaggle/working",
    num_frames  = 16,
    img_size    = 224,
    batch_size  = 4,
    accum_steps = 4,       # effective batch = 16
    dropout     = 0.3,
    epochs      = 10,
    patience    = 4,
    lr_cnn      = 3e-5,    # backbone learning rate
    lr_head     = 3e-4,    # new modules learning rate
    use_frac    = 1.0,
)
```

The best checkpoint saves automatically to `/kaggle/working/best_mftdnet.pt`.

### 4. Run the web app

Download `best_mftdnet.pt` from Kaggle output. Rename it to `best_stfanet.pt` and place it next to `app.py`:

```bash
streamlit run app.py
```

Upload any face video and click **Detect Deepfake**. Results appear within ~46ms on GPU.

> The checkpoint must contain keys: `model_state`, `epoch`, `best_auc`.

---

## ⚙️ Full Hyperparameter Reference

| Parameter | Value | Where set |
|-----------|-------|-----------|
| Optimizer | AdamW | `Cell 6` |
| Backbone LR | 3 × 10⁻⁵ | `CFG["lr_cnn"]` |
| New modules LR | 3 × 10⁻⁴ | `CFG["lr_head"]` |
| Weight decay | 1 × 10⁻⁴ | `Cell 6` |
| LR scheduler | CosineAnnealingLR (η_min=1e-6) | `Cell 6` |
| Batch size | 4 videos | `CFG["batch_size"]` |
| Gradient accum. | 4 steps (effective batch=16) | `CFG["accum_steps"]` |
| Mixed precision | AMP (autocast + GradScaler) | `Cell 7` |
| Grad clip norm | 1.0 | `Cell 7` |
| Frames per video | 16 | `CFG["num_frames"]` |
| Input resolution | 224 × 224 | `CFG["img_size"]` |
| Transformer layers | 2 | `app.py: N_LAYERS` |
| Attention heads | 8 | `app.py: N_HEADS` |
| FFN inner dim | 3,072 (feat_dim × 4) | `STFANet.__init__` |
| Dropout | 0.3 | `CFG["dropout"]` |
| Frequency tokens | 4 | `app.py: FREQ_TOKENS` |
| Epochs | 10 | `CFG["epochs"]` |
| Early stop patience | 4 | `CFG["patience"]` |

---

## 📱 Streamlit App (`app.py`)

| Feature | Detail |
|---------|--------|
| Supported formats | MP4, AVI, MOV, MKV |
| Face detection | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Fallback | 80% center crop if no face detected |
| Frames sampled | 16 (evenly spaced via `np.linspace`) |
| Frame preview | 4×4 grid with index, timestamp, detection status |
| Outputs | Fake probability, Real probability, Raw logit |
| Debug panel | Per-frame face detection stats + logit interpretation |
| Sidebar | Checkpoint epoch, val AUC, device info, model reload button |
| Model caching | `@st.cache_resource` — loads once per session |

---

## 🔬 Ablation Study

| Configuration | Val AUC | Drop |
|--------------|---------|------|
| Full MFTD-Net | **0.9901** | — |
| Without frequency stream | 0.9512 | −3.89% |
| Single frequency token (not 4) | 0.9589 | −3.12% |
| Concatenation fusion (no cross-attn) | 0.9634 | −2.67% |
| Without auxiliary frame loss | 0.9723 | −1.78% |
| Post-LN instead of Pre-LN | 0.9812 | −0.89% |

---

## 🌍 SDG Alignment

| Goal | Contribution |
|------|-------------|
| **SDG 9** — Industry & Innovation | Reliable deepfake detection for digital infrastructure |
| **SDG 16** — Peace & Justice | Interpretable per-frame outputs for journalists and legal analysts |
| **SDG 17** — Partnerships | Built on the public FaceForensics++ benchmark; fully reproducible |

---

## ⚠️ Known Limitations

- Trained only on the **Deepfakes** manipulation method — may not generalise to Face2Face, FaceSwap, NeuralTextures, or FaceShifter
- Haar Cascade fails on ~4.4% of frames (extreme head pose / heavy occlusion)
- Fixed 16-frame uniform sampling may miss manipulation in short video segments
- No audio modality — cannot detect lip-sync inconsistencies
- Real videos with rapid head movement can occasionally trigger false positives

---

## 📚 Key References

- Rossler et al. — *FaceForensics++* (ICCV 2019)
- Afchar et al. — *MesoNet* (WIFS 2018)
- Qian et al. — *F3-Net: Thinking in Frequency* (ECCV 2020)
- Liu et al. — *ConvNeXt: A ConvNet for the 2020s* (CVPR 2022)
- Loshchilov & Hutter — *AdamW: Decoupled Weight Decay Regularization* (ICLR 2019)
- Foret et al. — *SAM: Sharpness-Aware Minimization* (ICLR 2021)

Full bibliography in the project report.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Vardhaman College of Engineering · Department of CSE (AI & ML)**  
Mini Project A8041 · Academic Year 2025–26

*Made with ❤️ by Anveer Chetty, Niccunj Bajaj, and Pratham Lahoti*

</div>
