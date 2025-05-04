# Synthetic Flow Cytometry Data Generator Apps
An interactive platform to generate synthetic flow cytometry data

The combination of the two apps reflects a sensible division between 
- a priori synthetic data generation (user-defined populations) and
- data-driven synthesis (Gaussian Mixture Model-based modelling of gated populations from an existing .fcs file)

## App-1  (Synthetic FCM Data Generator)
This app is designed such that it allows users to create synthetic flow cytometry data 
file (.fcs) from scratch, utilizing domain knowledge where synthetic data is generated 
sampling from a normal distribution based on various input parameters, and statistical properties input by the user. 

The latest code for App-1 (Synthetic FCM Data Generator) is available in the fcm-app.py file

## App-2 (Flow Cytometry Data Analyzer)
This app is designed to enable users to generate a synthetic flow cytometry data file 
(.fcs), extracting interested cell populations by applying a sequence of gating to an 
existing real FCS file, and employing GMM to generate the synthetic data. Moreover, it 
allows the application of dimensionality reduction techniques such as UMAP and t-SNE, and explores GMM-derived population clusters to give the user a better understanding of the 
underlying cell population structures of the gated population. 


The latest code for App-2 (FCM Data Analyzer) is available in the fcm_analyzer_app_v2.py file


## For HF

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

 
