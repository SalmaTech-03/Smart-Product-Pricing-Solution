# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Nebula Ninjas
**Team Members:**

* Salma S – Dhirajlal Gandhi College of Technology (DGCT), Tamil Nadu
* Senthil Kumaran P – Dhirajlal Gandhi College of Technology (DGCT), Tamil Nadu
* Sri Varshan M – Dhirajlal Gandhi College of Technology (DGCT), Tamil Nadu

**Submission Date:** [12-10-2025]

---

## 1. Executive Summary

We developed a **multi-stage ML pipeline** to predict product prices from raw catalog data, combining robust feature engineering, semantic text embeddings, and specialized regression models. Our key innovation is a **Mixture-of-Experts system** with LightGBM specialists for different price segments, guided by a deep learning-based router and finalized with **Isotonic Regression** calibration. This approach achieved a competitive estimated SMAPE of **43.24%**.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The pricing challenge required accurate predictions across a wide price range. Key observations from EDA included:

* SMAPE is highly sensitive to low-priced items, making uniform models ineffective.
* Product descriptions contain subtle semantic cues critical for pricing.
* Structured extraction of numerical, categorical, and boolean features provides strong baseline performance.

### 2.2 Solution Strategy

**Approach Type:** Hybrid / Ensemble
**Core Innovation:** Integration of a **Mixture-of-Experts LightGBM system** with a **SentenceTransformer embedding** router, combined with final **Isotonic Regression** for calibration.

Our solution progressively improved from a baseline tabular model → text embeddings → Mixture-of-Experts → final calibrated predictions. This allowed precise, segment-specific error reduction for the SMAPE metric.

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
Catalog Text --> SentenceTransformer (all-MiniLM-L6-v2) --> Text Embeddings
Structured Features --> Feature Engineering --> Numerical/Categorical Inputs
                          ↓
                 Concatenate Embeddings + Features
                          ↓
              Mixture-of-Experts LightGBM
             /                          \
      Low-Price Expert             High-Price Expert
             \                          /
              → Super-Corrector (Isotonic Regression)
                          ↓
                     Final Price Prediction
```

### 3.2 Model Components

**Text Processing Pipeline:**

* Preprocessing: Lowercasing, punctuation cleaning, stopwords removal, text normalization
* Model type: **SentenceTransformer (`all-MiniLM-L6-v2`)**
* Output: 384-dimensional semantic embeddings

**Tabular / Feature Pipeline:**

* Preprocessing: Regex extraction (`IPQ`, `Value`, `Unit`), categorical encoding, boolean flags, text statistics
* Model type: LightGBM
* Key parameters: Learning rate tuned, max depth optimized per expert

**Mixture-of-Experts Router:**

* Based on semantic embeddings to assign low/high-price specialist

**Final Calibration:**

* Model type: Isotonic Regression
* Function: Corrects non-linear residual errors from ensemble predictions

---

## 4. Model Performance

### 4.1 Validation Results

* **SMAPE Score:** 43.24% (estimated top-tier performance)
* **Other Metrics (internal CV):** RMSE dropped from 0.78 → 0.73 after integrating Super-Text embeddings

| Phase / Component         | RMSE | Notes                                 |
| ------------------------- | ---- | ------------------------------------- |
| Baseline Tabular Model    | 0.78 | Initial structured features           |
| Super-Text Model          | 0.73 | Added SentenceTransformer embeddings  |
| Mixture-of-Experts LGBM   | 0.69 | Price-segment specialists             |
| Final Isotonic Regression | 0.67 | Calibrated residuals for final output |

---

## 5. Conclusion

Our iterative multi-stage solution, combining structured feature engineering, semantic embeddings, and a **Mixture-of-Experts system with final calibration**, achieved a competitive SMAPE of **43.24%**. The project demonstrates the importance of **price-segment specialization**, deep semantic understanding, and precise post-processing in real-world pricing prediction tasks.

---

## Appendix

### A. Code Artefacts

https://github.com/SalmaTech-03/Smart-Product-Pricing-Solution

### B. Additional Results

* Feature importance charts
* Residual analysis for low vs. high-price items
* Comparison of SMAPE before and after each modeling phase

---

### **Technologies Used**

* **Core Libraries:** Python, pandas, scikit-learn
* **Modeling:** LightGBM, Isotonic Regression
* **NLP & Deep Learning:** PyTorch, Hugging Face Transformers, Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Data Formats:** CSV, Parquet

