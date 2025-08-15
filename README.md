# 🧬 Computational Drug Discovery Using Machine Learning

## 🧠 **Author** Abdur-Rasheed Abiodun Adeoye  
Data Analyst | Data Scientist | Bioinformatics Enthusiast   
**Date:** August 2025  
**Domain:** Bioinformatics, Cheminformatics, Drug Discovery  
**Tools & Libraries:** Python, RDKit, PaDEL-Descriptor, Scikit-learn, Seaborn, LazyPredict, ChEMBL Web Services   
**Contacts:** 
- [LinkedIn](https://www.linkedin.com/in/abdur-rasheed-adeoye/)
- [GitHub](https://github.com/Abdur-RasheedAde)

---

## 📘 Project Overview

This project applies machine learning techniques to bioactivity data from the ChEMBL database to predict the inhibitory potency (pIC50) of compounds targeting the **SARS coronavirus 3C-like proteinase**. The goal is to build a regression model that can assist in virtual screening and computational drug discovery.

---

## 📥 Data Acquisition & Preprocessing

### 🔍 Target Selection
- Queried ChEMBL for targets related to "coronavirus".
- Selected **CHEMBL3927** (SARS coronavirus 3C-like proteinase).

### 📥 Bioactivity Data
- Retrieved IC50 values for compounds targeting CHEMBL3927.
- Filtered and cleaned data to remove missing values.
- Labeled compounds as:
  - **Active**: IC50 < 1000 nM
  - **Inactive**: IC50 > 10000 nM
  - **Intermediate**: 1000 nM ≤ IC50 ≤ 10000 nM

### 🧪 Descriptor Calculation
- Calculated **Lipinski descriptors** using RDKit:
  - Molecular Weight (MW)
  - LogP
  - Number of H-bond Donors
  - Number of H-bond Acceptors
- Converted IC50 to **pIC50** using:  
  `pIC50 = -log10(IC50 in molar)`

---

## 📊 Exploratory Data Analysis

- Visualized chemical space using scatter plots (MW vs LogP).
- Box plots for pIC50 and Lipinski descriptors across bioactivity classes.
- Performed **Mann-Whitney U tests** to assess statistical differences between active and inactive compounds.

---

## 🧬 Molecular Fingerprints

- Used **PaDEL-Descriptor** to compute molecular fingerprints.
- Combined fingerprints with pIC50 values to form the final dataset.

---

## 🤖 Model Building

### 🔧 Regression Model
- Used **Random Forest Regressor** from Scikit-learn.
- Split data into training (80%) and testing (20%) sets.
- Achieved a good **R² score** on the test set.
- Visualized predicted vs experimental pIC50 values.

### 📊 Model Evaluation
- Compared multiple regression models using **LazyPredict**:
  - Evaluated models based on R², RMSE, and computation time.
  - Visualized performance using bar plots.

---

## 📌 Key Findings

- **pIC50 values** showed significant differences between active and inactive compounds.
- **MW, NumHDonors, NumHAcceptors** were statistically significant descriptors.
- **LogP** did not show significant variation between classes.
- Random Forest performed well, but LazyPredict revealed other competitive models.

---

## 📁 Output Files

- Preprocessed datasets (`.csv`)
- Statistical results (`.csv`)
- Visualizations (`.pdf`)
- Model performance comparisons

---

## 🚀 Future Work

- Extend to other viral targets or diseases.
- Incorporate deep learning models for enhanced prediction.
- Explore ensemble methods and hyperparameter tuning.
- Deploy as a web-based virtual screening tool.

---

## 📎 References
- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)
