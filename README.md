# 🧬 Computational Drug Discovery Using Machine Learning

**Author:** Abdur-Rasheed Abiodun Adeoye  
**Date:** August 2025  
**Domain:** Bioinformatics, Cheminformatics, Drug Discovery  
**Tools & Libraries:** Python, RDKit, PaDEL-Descriptor, Scikit-learn, Seaborn, LazyPredict, ChEMBL Web Services

---

## 📘 Overview

This project applies machine learning to bioactivity data from the ChEMBL database to predict the inhibitory potency (pIC50) of compounds targeting the **SARS coronavirus 3C-like proteinase**. It includes data preprocessing, descriptor calculation, statistical analysis, and model building using Random Forest and LazyPredict.

---

## 📥 Data Acquisition

- Queried ChEMBL for targets related to "coronavirus".
- Selected **CHEMBL3927** (SARS coronavirus 3C-like proteinase).
- Retrieved IC50 values and filtered for valid entries.
- Labeled compounds as:
  - `Active`: IC50 < 1000 nM
  - `Inactive`: IC50 > 10000 nM
  - `Intermediate`: 1000 nM ≤ IC50 ≤ 10000 nM

---

## 🧪 Feature Engineering

### Lipinski Descriptors
Calculated using RDKit:
- Molecular Weight (MW)
- LogP
- Number of H-bond Donors
- Number of H-bond Acceptors

### pIC50 Conversion
Converted IC50 to pIC50 using:
```python
pIC50 = -log10(IC50 in molar)
