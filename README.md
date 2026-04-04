# 🚀 ROAD MARK DETECTION: EnlightenGAN vs CLAHE with YOLOv8

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-blue">
  <img src="https://img.shields.io/badge/Deep%20Learning-EnlightenGAN-purple">
  <img src="https://img.shields.io/badge/Image%20Processing-CLAHE-green">
  <img src="https://img.shields.io/badge/Status-Research%20Completed-success">
</p>

<p align="center">
  <b>Low-Light Road Mark Detection using Image Enhancement Techniques</b><br>
  Comparing Traditional (CLAHE) vs Deep Learning (EnlightenGAN)
</p>

---

## 🎯 Project Overview

This repository contains the full implementation, experimentation, and evaluation of my undergraduate thesis:

> **"Comparative Analysis of Image Enhancement Methods on YOLOv8 Object Detection Performance in Low-Light Conditions"**

Road marking detection under **low-light conditions** is a critical challenge for both human drivers and modern Advanced Driver Assistance Systems (ADAS). This research evaluates and compares two LLIE methods:
 
- 🔲 **CLAHE** — Contrast Limited Adaptive Histogram Equalization (classical approach)
- ✨ **EnlightenGAN** — Unsupervised deep learning-based image enhancement
 
Both methods are applied as preprocessing stages before training **YOLOv8l** on the **TRMSDN** dataset, with performance evaluated using precision, recall, mAP@0.5, and statistical significance via **paired t-test**.

The study investigates whether **deep learning-based image enhancement (EnlightenGAN)** can significantly improve object detection performance compared to a **traditional method (CLAHE)**.

---

## 🧠 Key Contributions

- 🔍 Object detection on **low-light road marking dataset (TRMSDN)**
- ⚖️ Comparison between:
  - CLAHE (non-deep learning)
  - EnlightenGAN (deep learning)
- 📊 Statistical validation using **Paired T-Test**
- 📉 Insight: Improvement exists but **NOT statistically significant**

---

## 🗂️ Repository Structure
 
```
📁 ROAD_MARK_EnlightenGAN_v_CLAHE_YOLOv8/
│
├── 🐍 clahe_enhancement.py              # CLAHE preprocessing script
├── 🐍 enlightengan_enhancement.py       # EnlightenGAN preprocessing script
├── 🐍 video_inference.py                # Run detection on video
│
├── 📓 dataset_distribution.ipynb        # Dataset analysis & visualization
├── 📓 evaluate_model.ipynb              # Model evaluation & t-test analysis
│
├── 📦 split_ordered_ori_dataset.zip     # Original split dataset
├── 📦 split_enhanced_clahe_cl2.zip      # CLAHE-enhanced dataset
├── 📦 split_enhanced_enlightenGAN.zip   # EnlightenGAN-enhanced dataset
│
├── 📊 ap_per_class_comp.png             # AP per class comparison chart
├── 📊 presisi-recall chart.png          # Precision-recall curve
├── 📊 yolo_testing_results.xlsx         # Full evaluation results
│
├── 🎬 night_drive_taipei.mp4            # Raw input video (night driving)
├── 🎬 output.mp4                        # Detection output video
│
└── 📁 DETECT/                           # Inference output samples
```
 
---
 
## 🔬 Methodology
 
```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
 
  📂 TRMSDN Dataset (4,386 images · 12 classes)
             │
             ▼
  ┌──────────────────────┐
  │   Train/Test Split   │  80% Train · 20% Val (stratified)
  └──────────────────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  Original  CLAHE  EnlightenGAN
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  YOLOv8l  YOLOv8l  YOLOv8l
  (100 ep) (100 ep) (100 ep)
             │
             ▼
  ┌──────────────────────────────────────┐
  │  Evaluation: Precision · Recall      │
  │  mAP@0.5 · mAP@0.5:0.95             │
  │  Paired T-Test (α = 0.05)            │
  └──────────────────────────────────────┘
```
 
---
 
## 📊 Results
 
### Overall Performance
 
| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|:---|:---:|:---:|:---:|:---:|
| **EnlightenGAN** | **0.882** | 0.885 | **0.930** ✅ | **0.543** |
| Original | 0.819 | **0.928** | 0.912 | 0.525 |
| CLAHE | 0.806 | 0.935 | 0.905 | 0.522 |
 
### Per-Class mAP@0.5
 
| Class | Original | CLAHE | EnlightenGAN |
|:---|:---:|:---:|:---:|
| P01 Turn Right | 0.764 | 0.755 | **0.937** ⬆️ |
| P02 Turn Left | 0.833 | 0.825 | **0.952** ⬆️ |
| P03 Go Straight | **0.977** | 0.979 | 0.967 |
| P04 Turn Right or Straight | **0.987** | 0.972 | 0.981 |
| P05 Turn Left or Straight | **0.984** | 0.977 | 0.978 |
| P06 Speed Limit 60 | **0.993** | **0.994** | 0.986 |
| P07 Crosswalk | **0.940** | 0.901 | 0.844 |
| P08 Slow Sign | 0.943 | 0.938 | **0.946** |
| P09 Overtaking Prohibited | 0.976 | 0.978 | **0.985** |
| P10 Barrier Line | 0.683 | 0.699 | **0.700** |
| P11 Cross Hatch | 0.931 | **0.988** | 0.977 |
| P12 Stop Line | 0.936 | 0.848 | **0.906** |
 
### Statistical Test
 
| Comparison | t-statistic | p-value | Significant? |
|:---|:---:|:---:|:---:|
| EnlightenGAN vs Original | 0.8802 | 0.3976 | ❌ No (α=0.05) |
| EnlightenGAN vs CLAHE | 1.3241 | 0.2123 | ❌ No (α=0.05) |
 
> While EnlightenGAN shows a higher mean mAP, the improvement is **not statistically significant** due to high per-class variance.
 
---
 
## ⚙️ Setup & Usage
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/bmmasaputra/ROAD_MARK_EnlightenGAN_v_CLAHE_YOLOv8.git
cd ROAD_MARK_EnlightenGAN_v_CLAHE_YOLOv8
```
 
### 2. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Apply Image Enhancement
 
**CLAHE:**
```bash
python clahe_enhancement.py
```
 
**EnlightenGAN:**
```bash
# Install EnlightenGAN inference package first
pip install git+https://github.com/arsenyinfo/EnlightenGAN-inference
 
python enlightengan_enhancement.py
```
 
### 4. Train YOLOv8l
 
```python
from ultralytics import YOLO
 
model = YOLO("yolov8l.pt")
model.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=512,
    batch=16,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    seed=42
)
```
 
### 5. Evaluate & Run Inference
 
```bash
# Open notebook for full evaluation
jupyter notebook evaluate_model.ipynb
 
# Run video inference
python video_inference.py
```
 
---
 
## 🗃️ Dataset
 
| Property | Value |
|:---|:---|
| Name | Taiwan Road Marking Sign Dataset at Night (TRMSDN) |
| Source | Dewi et al. (2024) |
| Total Images | 4,386 |
| Classes | 12 |
| Split | 80% Train / 20% Validation (stratified) |
| Condition | Night-time, dash cam footage |
 
**Classes:** Turn Right · Turn Left · Go Straight · Turn Right or Straight · Turn Left or Straight · Speed Limit 60 · Crosswalk · Slow Sign · Overtaking Prohibited · Barrier Line · Cross Hatch · Stop Line
 
---
 
## 🏋️ Training Configuration
 
| Hyperparameter | Value |
|:---|:---|
| Model | YOLOv8l |
| Epochs | 100 |
| Image Size | 512 × 512 |
| Batch Size | 16 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Device | NVIDIA T4 GPU (Google Colab) |
| Random Seed | 42 |
 
---
 
## 📚 References
 
- Dewi, C., Chen, R.-C., Zhuang, Y.-C., & Manongga, W. E. (2024). *Image Enhancement Method Utilizing YOLO Models to Recognize Road Markings at Night*. IEEE Access, 12, 131065–131081.
- Jiang, Y. et al. (2021). *EnlightenGAN: Deep Light Enhancement Without Paired Supervision*. IEEE Transactions on Image Processing, 30, 2340–2349.
- Terven, J., & Cordova-Esparza, D. (2023). *A Comprehensive Review of YOLO: From YOLOv1 and Beyond*.
- Mohamed, A. T. et al. (2025). *Integrating EnlightenGAN for Enhancing Car Logo Detection Under Challenging Lighting Conditions*. Multimedia Tools and Applications.
 
---
 
## 👤 Author
 
<div align="center">
 
**Bima Agung Saputra**
 
Department of Informatics Engineering · Faculty of Engineering
**Universitas Palangka Raya**
Tanjung Nyaho Campus, Jl. Yos Sudarso, Palangka Raya 73112
 
</div>
 
---
 
<div align="center">
 
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>
 
*Undergraduate Thesis · Universitas Palangka Raya · 2025*
 
</div>