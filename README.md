# eeg2emg-kg-gt

**Kinematic-Guided Graph-Transformer for Continuous EMG Envelope Prediction from EEG**

> Method 1 (KG-GT) — Phase 1 implementation  
> Dataset: WAY-EEG-GAL · 12 subjects · 3,936 trials · 32-ch EEG · 5-ch EMG  
> Branch: `KG-GT`

---

## Overview

Predicts continuous EMG envelopes from scalp EEG and kinematic signals during grasp-and-lift tasks. Addresses the core limitation of prosthetic control in high-level amputation where functional muscles are absent.

**Pipeline:** CCA-aligned EEG → Transformer encoder → Kinematic-Guided GAT → hybrid MSE + SoftDTW loss → 5-channel EMG envelope

**Target gaps solved:**
- G1: UKF linear expressivity → replaced by Transformer
- G2: Single-subject scope → 12 subjects, 3,936 trials
- G3: Kinematics underutilised → CCA alignment + GAT conditioning
- G4: No synergy structure → GAT over 5 EMG nodes
- G5: Temporal misalignment → hybrid SoftDTW loss

---

## Dataset: WAY-EEG-GAL

| Signal | Channels | Fs | Notes |
|--------|----------|----|-------|
| EEG | 32 (ActiCap) | 500 Hz | Fp1–PO10, international 10-20 |
| EMG | 5 | 4,000 Hz | AD, BR, FD, ED, FDI |
| Kinematics | 4×6-DOF | 500 Hz | Object, index, thumb, wrist (Polhemus FASTRAK) |
| Force/Torque | 2×6-axis | 500 Hz | ATI Nano-17 at both contact plates |

**EEG channels (32):**
```
Fp1 Fp2 F7 F3 Fz F4 F8
FC5 FC1 FC2 FC6
T7  C3  Cz  C4  T8
TP9 CP5 CP1 CP2 CP6 TP10
P7  P3  Pz  P4  P8
PO9 O1  Oz  O2  PO10
```

**EMG muscles (5 nodes in GAT):**
```
AD  = Anterior Deltoid        (shoulder)
BR  = Brachioradialis         (forearm)
FD  = Flexor Digitorum        (finger flexion)
ED  = Extensor Digitorum      (finger extension)
FDI = First Dorsal Interosseus (precision grasp)
```

**Data split (per participant, 9 series + 1 stability):**
- Train: series 1–7
- Validation: series 8
- Test: series 9 + ST (held-out weight/mixed conditions)

---

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW INPUT SIGNALS                       │
├────────────────┬────────────────┬───────────────────────────┤
│  32-ch EEG     │  5-ch EMG      │  Kinematics (3D + F/T)    │
│  500 Hz        │  4,000 Hz      │  500 Hz                   │
└───────┬────────┴───────┬────────┴──────────┬────────────────┘
        │                │                   │
        ▼                ▼                   ▼
┌───────────────┐ ┌──────────────┐ ┌─────────────────────────┐
│ EEG Preproc   │ │ EMG Preproc  │ │ Kinematic Preproc       │
│ BP 0.1–40 Hz  │ │ BP 30–300 Hz │ │ Savitzky-Golay deriv    │
│ Notch 50 Hz   │ │ |s(t)| rect  │ │ Normalise per-feature   │
│ ASR + CAR     │ │ LP 10 Hz     │ └───────────┬─────────────┘
│ Delta 0.1–2Hz │ │ ↓ 500 Hz     │             │
│ Window T=500  │ │ z-score      │             │
└───────┬───────┘ └──────┬───────┘             │
        │                │ (TARGET Y)          │
        │                │                     │
        └────────────────┼─────────────────────┘
                         │ EEG + Kinematics
                         ▼
              ┌──────────────────────┐
              │    CCA ALIGNMENT     │
              │  W_x: 32 → 16 dims  │
              │  X̃ = X @ W_x        │
              │  max ρ(X̃, K̃)        │
              └──────────┬───────────┘
                         │ X̃ ∈ R^{N×500×16}
                         ▼
              ┌──────────────────────┐
              │  TRANSFORMER ENCODER │
              │  L=4 layers          │
              │  H=8 heads, d_k=32   │
              │  d_model=256         │
              │  FFN dim=1024        │
              │  PE + MHSA + LN      │
              └──────────┬───────────┘
                         │ H_temp ∈ R^{N×500×256}
                         │              ▲
                         ▼              │ k_t ∈ R^{13}
              ┌──────────────────────┐  │ (kinematic state)
              │  KINEMATIC-GUIDED GAT│◄─┘
              │  2 GAT layers        │
              │  e_ij^t = LReLU(     │
              │    a^T[Wh_i‖Wh_j‖    │
              │         W_k·k_t])    │
              │  JK concat → 512     │
              └──────────┬───────────┘
                         │ H_final ∈ R^{N×500×512}
                         ▼
              ┌──────────────────────┐
              │   LINEAR DECODER     │
              │   W_out: 512 → 5     │
              └──────────┬───────────┘
                         │ Ŷ ∈ R^{N×500×5}
                         ▼
              ┌──────────────────────┐
              │   HYBRID LOSS        │  ◄── Y (target EMG)
              │  L = 0.9·MSE         │
              │    + 0.1·SoftDTW     │
              │      (γ=0.1)         │
              └──────────────────────┘
```

---

## Preprocessing Details

### EEG (`src/preprocessing/eeg.py`)

```
Raw EEG (32ch, 500 Hz)
    │
    ├─ 1. Bandpass 0.1–40 Hz    (4th-order zero-phase Butterworth, sosfiltfilt)
    ├─ 2. Notch 50 Hz           (iirnotch, power-line removal)
    ├─ 3. ASR                   (sliding 500ms window, reject > 5σ vs clean baseline)
    ├─ 4. CAR                   (common average reference: x_i -= mean(x_all))
    ├─ 5. Delta extraction      (0.1–2 Hz, 4th-order zero-phase Butterworth)
    └─ 6. Windowing             (T_w=500 samples=1s, stride=50=100ms overlap)

Output: X_raw ∈ R^{N × 500 × 32}
```

### EMG (`src/preprocessing/emg.py`)

```
Raw EMG (5ch, 4,000 Hz)
    │
    ├─ 1. Bandpass 30–300 Hz    (4th-order Butterworth at 4000 Hz)
    ├─ 2. Full-wave rectify     (|s(t)|)
    ├─ 3. Low-pass 10 Hz        (smooth activation envelope)
    ├─ 4. Decimate ×8           (4000 → 500 Hz, scipy.signal.decimate)
    └─ 5. Z-score normalize     (per channel, fit on train set only)

Output: Y ∈ R^{N × 500 × 5}   ← training target
```

### Kinematics (`src/preprocessing/kinematics.py`)

Kinematic state vector **k_t ∈ R^{13}** per time step:

```
k_t = [
    p_wrist  (3),     # 3D wrist position (Polhemus sensor)
    p_index  (3),     # 3D index fingertip position
    p_thumb  (3),     # 3D thumb fingertip position
    d_grip   (1),     # grip aperture = ‖p_index − p_thumb‖₂
    F_L      (1),     # load force (ATI Nano-17)
    F_G      (1),     # grip force (ATI Nano-17)
    ρ_GL     (1),     # force ratio = F_G / F_L
]
```

Velocity from Savitzky-Golay filter (window=11, poly=3). Normalize per-feature (train stats).

Output: **K ∈ R^{N × 500 × 13}**

### CCA Alignment (`src/preprocessing/cca.py`)

Finds projection matrices W_x ∈ R^{32×16} and W_k ∈ R^{13×16} maximising canonical correlation:

```
max_{W_x, W_k}  ρ = (W_x^T Σ_xk W_k) / sqrt(W_x^T Σ_xx W_x · W_k^T Σ_kk W_k)
```

```
EEG  X [T×32] ──► W_x (CCA proj) ──► X̃ [T×16] ──┐
                                                    ├── max ρ ──► To Transformer
Kin  K [T×13] ──► W_k (CCA proj) ──► K̃ [T×16] ──┘
```

- Solved via generalised eigenvalue decomposition
- W_x saved per participant for inference
- Reduces 32 channels to 16 biomechanically-grounded canonical components

---

## Model Architecture

### Transformer Encoder

```
Input Z^(0) = X̃ + PE          [N × 500 × 16]
    │
    └── × L=4 layers:
           │
           ├─ MHSA  (H=8 heads, d_k=32, d_v=32)
           │   Q_i = Z W_{Q_i},  K_i = Z W_{K_i},  V_i = Z W_{V_i}
           │   head_i = softmax(Q_i K_i^T / √d_k) V_i
           │   MHSA = Concat(head_1,...,head_8) W_O
           │
           ├─ Add & LayerNorm (residual)
           │
           ├─ FFN: Linear(256→1024) → ReLU → Linear(1024→256)
           │
           └─ Add & LayerNorm (residual)

Output H_temp ∈ R^{N × 500 × 256}
```

Positional encoding (sinusoidal):
```
PE(p, 2i)   = sin(p / 10000^{2i/d})
PE(p, 2i+1) = cos(p / 10000^{2i/d})
```

### Kinematic-Guided GAT

Graph G = (V, E): 5 nodes (muscles), fully connected edges.

Dynamic attention conditioned on kinematic state k_t at each time step:

```
e_ij^t = LeakyReLU( a^T [W·h_i ‖ W·h_j ‖ W_k·k_t] )

α_ij^t = exp(e_ij^t) / Σ_{n∈N_i} exp(e_in^t)

h_i^(l+1) = σ( Σ_{j∈N_i} α_ij^t · W · h_j^(l) )
```

Jumping Knowledge (JK) aggregation across both GAT layers:
```
h_i^final = Concat(h_i^(1), h_i^(2))   → 512-dim
```

**Muscle synergy graph (anatomical prior):**
```
         AD (Ant. Deltoid)
        /        \
      BR ────── FDI (1st Dors. Inteross.)
      |    ╲  ╱    |
      |     \/     |
      |     /\     |
      FD ══════ ED
   (Flex.Dig.) (Ext.Dig.)

══  Strong synergy: FD↔FDI (precision grasp loading phase)
──  Standard synergy edges (all bidirectional)
⇢   k_t injected into all nodes at each timestep
```

---

## Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR schedule | Halve every 50 epochs without val improvement |
| Batch size | 32 |
| Early stopping patience | 30 epochs |
| Dropout | 0.2 |
| Gradient clipping | norm = 1.0 |
| Loss λ (MSE weight) | 0.9 |
| SoftDTW γ | 0.1 |
| Window size T_w | 500 samples (1 s) |
| Stride | 50 samples (100 ms) |
| CCA components d | 16 |
| Transformer layers L | 4 |
| Attention heads H | 8 |
| d_model | 256 |
| FFN dim | 1024 |
| GAT layers | 2 |
| GAT output (JK) | 512 |
| Decoder | Linear 512→5 |
| Cross-validation | LOSOCV (leave-one-subject-out) |

**Data split per participant:**
```
Series 1–7  → Train
Series 8    → Validation
Series 9    → Test (standard)
Series ST   → Test (stability/perturbation, held-out)
```

---

## Loss Function

Hybrid MSE + SoftDTW tolerating cortico-muscular conduction delay (~10–30 ms):

```
L_total = λ · MSE(Y, Ŷ) + (1-λ) · SoftDTW_γ(Y, Ŷ)

MSE(Y, Ŷ) = (1/TM) Σ_t Σ_m (y_{t,m} - ŷ_{t,m})²

SoftDTW_γ(Y, Ŷ) = min^γ_π ⟨D(Y,Ŷ), A(π)⟩
                  where min^γ(a) = -γ log Σ_i exp(-a_i/γ)
```

- λ=0.9 (tuned on validation set)
- γ=0.1 (balances gradient smoothness vs alignment precision; γ→0 = hard DTW)

---

## Evaluation Metrics

All metrics computed per EMG channel m, averaged across subjects.

| Metric | Formula |
|---|---|
| nRMSE | (1/R_m) √( (1/T) Σ_t (y_{t,m} − ŷ_{t,m})² ) |
| nMAE | (1/(T·R_m)) Σ_t \|y_{t,m} − ŷ_{t,m}\| |
| Pearson r | Cov(Y_m, Ŷ_m) / √(Var(Y_m)·Var(Ŷ_m)) |
| R² | 1 − Σ_t(y−ŷ)² / Σ_t(y−ȳ)² |
| SNR (dB) | 10 log₁₀ [ Var(Y_m) / MSE(Y_m, Ŷ_m) ] |

R_m = range of channel m (for normalisation).

**Expected performance (Method 1 baseline):**
- Pearson r > 0.55 (vs UKF baseline ~0.20, CNN-LSTM ~0.50)
- R² > 0.45
- ~2M parameters

---

## Project Structure

```
eeg2emg-kg-gt/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # all hyperparameters
├── notebooks/
│   └── 01_explore.ipynb      # sanity-check one .mat file
└── src/
    ├── data/
    │   ├── loader.py          # scipy.io.loadmat → raw numpy arrays
    │   └── dataset.py         # PyTorch Dataset, windowing, split logic
    ├── preprocessing/
    │   ├── eeg.py             # BP → notch → ASR → CAR → delta → window
    │   ├── emg.py             # BP → rectify → LP → decimate → z-score
    │   ├── kinematics.py      # SGF derivative, build k_t (13-dim)
    │   └── cca.py             # fit CCA (sklearn), project X → X̃ (32→16)
    ├── models/
    │   ├── transformer.py     # L=4, H=8, d_model=256 encoder
    │   ├── kg_gat.py          # KG-GAT 2 layers + JK concat → 512
    │   └── kg_gt.py           # full model: CCA + Trans + GAT + Decoder
    ├── losses/
    │   └── soft_dtw.py        # SoftDTW γ=0.1 + hybrid λ=0.9 MSE blend
    └── training/
        ├── train.py           # Adam, scheduler, early stop, grad clip
        └── evaluate.py        # nRMSE, nMAE, r, R², SNR per channel
```

---

## Installation

```bash
git clone https://github.com/<YOUR_USERNAME>/eeg2emg-kg-gt.git
cd eeg2emg-kg-gt
git checkout KG-GT

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
torch>=2.1
torch-geometric>=2.4
mne>=1.6
tslearn>=0.6
pyyaml>=6.0
h5py>=3.9
```

---

## Data Setup

Download WAY-EEG-GAL from PhysioNet / Scientific Data (Luciw et al., 2014).  
Unzip each `PX.zip` into `data/way-eeg/raw/PX/`. Expected structure:

```
data/way-eeg/raw/
├── P1/
│   ├── HS_P1_S1.mat   # standard series
│   ├── WS_P1_S1.mat   # weighted series
│   └── HS_P1_ST.mat   # stability series
├── P2/ ...
└── P12/
```

Data dir is `.gitignore`d — never committed.

---

## Usage

```bash
# Preprocess all participants
python src/preprocessing/run_all.py --config configs/default.yaml

# Train single subject
python src/training/train.py --subject P1 --config configs/default.yaml

# LOSOCV (leave-one-subject-out cross-validation)
python src/training/train.py --losocv --config configs/default.yaml

# Evaluate
python src/training/evaluate.py --checkpoint checkpoints/best_P1.pt --subject P1
```

---

## Comparison with Baselines

| Method | Input | Expected r | Expected R² | Params | Multi-task |
|---|---|---|---|---|---|
| UKF (Sburlea 2021) | EEG delta | ~0.20 | — | minimal | No |
| CNN-LSTM (Jain 2025) | EEG | ~0.50 | ~0.35 | ~500K | No |
| **KG-GT (Ours, M1)** | EEG + Kin | **>0.55** | **>0.45** | ~2M | No |

Methods 2–4 (NNMF-KG-GT, PTL-Net, USPT) extend this baseline in subsequent branches.

---

## Citation

```bibtex
@article{sharifbeigi2025kggt,
  title   = {Predicting EMG Envelopes from EEG and Kinematic Signals:
             A Research Article Proposal},
  author  = {SharifBeigi, MohammadMahdi},
  year    = {2025},
  note    = {Computational Neuroscience Team 6}
}

@article{luciw2014,
  title   = {Multi-channel EEG recordings during 3,936 grasp and lift trials
             with varying weight and friction},
  author  = {Luciw, Matthew D and Jarocka, Ewa and Edin, Benoni B},
  journal = {Scientific Data},
  volume  = {1},
  pages   = {140047},
  year    = {2014}
}
```

---

## Author

**MohammadMahdi SharifBeigi** — Computational Neuroscience Team 6, 2025  
Contact: sharifbeigymohammad@gmail.com
