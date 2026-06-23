# 🧠 ADVANCING THE DIAGNOSIS OF HEAD AND NECK SMALL ROUND BLUE CELL NEOPLASMS THROUGH ARTIFICIAL INTELLIGENCE

# Author: Lucas Lacerda de Souza

# Year: 2026

---

# 📖 1. Project Overview

This project implements a multimodal artificial intelligence framework for the diagnosis and prognostic stratification of head and neck small round blue cell neoplasms (SRBCNs). The framework integrates clinicopathological variables, quantitative nuclear morphometry, conventional machine learning, deep learning, transformer-based cellular modelling, foundation models, and survival prediction algorithms to analyse digitised haematoxylin and eosin (H&E) whole-slide images.

The study includes four major tumour groups:
* Haematolymphoid neoplasms
* Neuroectodermal and neural crest-derived tumours
* Mesenchymal tumours
* Melanocytic neoplasms

---

# 🔬 2. Computational Pipeline

<img width="1280" height="854" alt="Figura 1" src="https://github.com/user-attachments/assets/d95b0e39-9bb2-4668-bee7-c3f9c0e9528f" />

The pipeline combines:
* Quantitative morphometric analysis
* XGBoost classification with SHAP explainability
* Multimodal convolutional neural networks
* CellViT++ transformer-based cellular analysis
* Foundation models (UNI and Virchow)
* Survival modelling with XGBoost Survival
* Explainable artificial intelligence techniques

The framework is intended for computational pathology research and proof-of-concept clinical decision support applications.

---

# 💻 3. Environment and Hardware

All experiments were performed using the following configuration:

**Operating System:** Ubuntu 20.04.1 LTS

**Python Version:** 3.12.11

**PyTorch Version:** 2.8.0 (CUDA 12.8)

**CPU:** Intel Xeon W-2295 (18 cores / 36 threads)

**RAM:** 125 GB

**GPUs:** 3 × NVIDIA GeForce RTX 3090 (24 GB each)

---

# ⚙️ 4. Environment Files
**Conda Channels**
  * pytorch
  * nvidia
  * conda-forge
  * defaults

**Main Dependencies**
  * python=3.12.11
  * pytorch=2.8.0
  * torchvision=0.23.0
  * torchaudio=2.8.0
  * pytorch-cuda=12.8
  * numpy=1.26.4
  * pandas=2.2.3
  * scipy=1.15.3
  * scikit-learn=1.7.1
  * matplotlib=3.10.3
  * seaborn=0.13.2
  * pillow=11.3.0
  * tqdm=4.67.1
  * openpyxl=3.1.5
  * jupyterlab=4.4.5
  * notebook=7.4.4
  * ipykernel=6.30.1
  * xgboost=3.0.3
  * shap=0.48.0
  * lifelines=0.30.0
  * opencv=4.12.0
  * scikit-image=0.25.2
  * openslide-python=1.4.2
  * pyvips=3.0.0
  * h5py=3.14.0
  * pyyaml=6.0.2
  * tensorboard=2.20.0
  * pip

**Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification_SRB.git

cd Classification_SRB

**Create Environment**

conda env create -f environment.yml

conda activate Classification_SRB

---

# 🗂️ 5. Dataset

A total of **675 cases** were included in this international multicentre study.

**Tumour Groups**
* Haematolymphoid neoplasms (n = 463)
* Neuroectodermal and neural crest-derived tumours (n = 84)
* Mesenchymal tumours (n = 75)
* Melanocytic tumours (n = 53)

**Participating Countries**
* Brazil
* United Kingdom
* Mexico
* Guatemala

---

# 🤖 6. Model Architectures

**📊 Conventional Machine Learning**

* XGBoost Multiclass Classifier
* XGBoost Survival Model
* SHAP Explainability Framework

**🧠 Deep Learning**

* AlexNet
* MobileNet
* InceptionV3
* DenseNet121
* Xception
* ResNet50

**🔍 Vision Transformer-Based Cellular Analysis**

* CellViT++

**🚀 Foundation Models**

* UNI
* Virchow

---

# 🧬 7. Features Used

**👤 Clinicopathological Features**

* Age
* Sex
* Anatomical site

**🔬 Morphometric Features**

* Nuclear area
* Nuclear perimeter
* Nuclear circularity
* Nuclear eccentricity

**🖼️ Histopathological Features**

* H&E image patches
* CellViT++ nuclear embeddings
* Foundation model embeddings

**⏳ Survival Features**

* Clinicopathological variables
* Segmentation-derived features

---

# 📂 8. Repository Structure

DATA/                    → Synthetic example data and directory structures

MODELS/                  → Model architectures and inference pipelines

RESULTS/                 → Study results and supplementary outputs

INFERENCE.py             → Inference script

MODEL_CARD.md            → Model documentation

README.md                → Repository documentation

REQUIREMENTS.txt         → Dependency list

LICENSE.txt              → Repository license

---

# ⚡ 9. Quick Start
**Run Inference**

python INFERENCE.py \
    --input_dir ./data/example_slides \
    --output_dir ./results/

---

# ✅ 10. Compliance with TRIPOD-AI, STARD-AI and CLAIM 2024

This repository has been structured according to:

* TRIPOD-AI
* CLAIM 2024

The guidelines were used for transparent, reproducible, and clinically relevant artificial intelligence research in pathology.

---

# ⚖️ 11. Ethics

This study was approved by:

* **Piracicaba Dental School, University of Campinas, Brazil** (CAAE: 67064422.9.1001.5418; Approval No. 6.039.616)
  
* **West of Scotland Research Ethics Service** (Reference No. 20/WS/0017)

All procedures were conducted in accordance with the Declaration of Helsinki. All data were fully anonymised prior to analysis.

---

# 📦 12. Data Availability

Due to ethical restrictions and patient confidentiality regulations:

* Whole-slide images are not publicly distributed
* Raw clinical metadata are not publicly shared
* Patient-identifiable data are not included

To support reproducibility, this repository provides:
* Synthetic organisational examples
* Representative patch structures
* Example inference pipelines
* Documentation and reproducibility guidelines

---

# 💾 13. Code Availability

We have made the codes publicly available online, along with model weights (https://github.com/lucas-lacerda-de-souza/Classification_SRB). All code was written with Python Python 3.12.11, along with PyTorch 2.8.0. The full implementation of the model, including the code and documentation, has been deposited in the Zenodo repository and is publicly available (https://doi.org/10.5281/zenodo.20383882).

---

# 📚 14. Citation

@article{delasouza2026srb,
  title={Advancing the Diagnosis of Head and Neck Small Round Blue Cell Neoplasms Through Artificial Intelligence},
  author={Souza, Lucas Lacerda de and collaborators},
  journal={2026},
  year={2026}}

---

# 📜 15. License

MIT License © 2026 Lucas Lacerda de Souza

