# Synthetic Flow Cytometry Data Generator Apps
Aim: An interactive platform to generate synthetic flow cytometry data for education, bridging the gap between theoretical concepts and hands-on training.

---
## Overview 
This repository hosts two complementary apps designed for synthetic FCM data generation:

1. App-1 (Synthetic FCM Data Generator): Generates synthetic data from scratch using user-defined parameters (Gaussian distributions).

2. App-2 (Flow Cytometry Data Analyzer): Creates synthetic data from real FCM files by modeling gated populations with Gaussian Mixture Models (GMMs).

Together, they offer a flexible toolkit for educators to create noise-free, privacy-safe datasets for teaching gating, clustering, and analysis.

## Apps

### App-1  (Synthetic FCM Data Generator)
App-1: Synthetic FCM Data Generator

Purpose: Generate synthetic FCM data (.fcs files) from scratch using domain knowledge.
Features:

   - Define cell populations via parameters (mean, standard deviation).
   - Adjust population fractions, compositions, and correct class imbalances (SMOTE, ADASYN).
   - Export synthetic data in .fcs or .csv formats.
   - Visualize data with interactive Plotly plots.

Code for App-1 is available in the fcm-app.py file

### App-2 (Flow Cytometry Data Analyzer)
Purpose: Generate synthetic data from real FCM files by gating and modeling populations.
Features:

   - Upload and gate real FCM data (.fcs) using Lasso tools.
   - Model gated populations via GMMs for synthetic data generation.
   - Apply dimensionality reduction (t-SNE, UMAP) and cluster validation for visual validation.
   - Quantitative validation of synthetic data quality via metrics Silhouette Score, Mutual Information, and Wasserstein Distance.
   - Export configurations (.json) for reproducibility.
   - Correct class imbalances of rare populations 

Code for App-2  is available in the fcm_analyzer_app_v2.py file

## Set up 

Prerequisites:
 - Python 3.10+
 - Core Libraries: streamlit, scikit-learn, fcwrite, fcsparser, pandas, plotly (see requirements.txt for more details)


```
# For app-1
pip install -r requirements.txt  # Install dependencies
streamlit run fcm-app.py        # Launch App-1

# For app-2
pip install -r requirements_2.txt  # Install dependencies
streamlit run fcm_analyzer_app_v2.py  # Launch App-2
```

## Usage Examples

App-1:

- Define 3 cell populations (e.g., CD4+, CD8+, monocytes) with distinct Gaussian parameters (Mean, Std) and corresponding marker parameters (FCS, SSC, CD3, CD4, CD8, etc. ).
- Adjust population ratios (e.g., 50% CD4+, 30% CD8+) and export synthetic data.

App-2:

- Upload a real FCM file, gate CD4+ T-cells, and generate synthetic data mimicking this population.
- Compare original vs. synthetic data using UMAP embeddings and cluster metrics.



### For HF (testing)

---

title: Fcm App
emoji: 💻
colorFrom: pink
colorTo: gray
sdk: streamlit
sdk_version: 1.43.1
app_file: app.py
pinned: false

---

 
