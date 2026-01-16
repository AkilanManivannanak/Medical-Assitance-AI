# AI-Powered Multi-Disease Health Risk Prediction (17 Conditions) — Random Forest + CNN + Explainability

An AI-driven **multi-disease screening and clinical decision-support prototype** that predicts risk across **17 chronic/genetic conditions** using **modular ML pipelines**:
- **Random Forest** classifiers for **tabular clinical data** (labs, vitals, demographics)
- **CNN pipelines** for **radiographic imaging** (Pneumonia + Tuberculosis)

The system generates an interpretable, clinician-style report including:
- per-disease probabilistic risk stratification
- predicted disease progression (3–5 year horizon)
- symptom timelines, comorbidity interaction chains, and an immunity impact score
- explainability artifacts (SHAP + radar chart)

> Academic project (LIU, AI 681). This is **NOT** a medical device and is **not for clinical diagnosis**. Use for research/education only.

---

## Results (Test Accuracy)

Across 17 diseases, the project reports an **average accuracy of 87.3%** using stratified **80/20 train-test splits**.  
Per-disease test accuracy is summarized below.

| Disease | Test Accuracy |
|---|---:|
| Diabetes | 73.38% |
| Liver Disease | 76.07% |
| Parkinson’s | 94.87% |
| PCOS | 73.39% |
| Stroke | 93.84% |
| Cystic Fibrosis | 84.21% |
| Alzheimer’s | 94.42% |
| Anemia | 100% |
| Chronic Kidney Disease | 100% |
| Heart Disease | 98.54% |
| Cancer | 96.49% |
| Obesity | 95.27% |
| Hypertension | 88.44% |
| Pneumonia (X-ray) | 62.50% |
| Tuberculosis (X-ray) | 96.90% |
| Hepatitis | 70.97% |
| HIV | 100% |

**Source:** project final report / captured evaluation outputs.  
- Average accuracy + methodology summary: see Abstract.  
- Full disease accuracy table: see Table 2.  
- Imaging model notes: Pneumonia (EfficientNetB0) and TB (custom CNN) performance discussion included.  
📄 Report + screenshots: `docs/fpr-Team-6.pdf` (recommended to include in this repo).  
(Report references: average accuracy 87.3% and stratified splits)  
(Report references: disease-wise accuracy table)  

---

## What’s inside

### Disease coverage (17 total)
Tabular ML (Random Forest):
- Diabetes, Liver Disease, Parkinson’s, PCOS, Stroke, Cystic Fibrosis, Alzheimer’s, Anemia, Chronic Kidney Disease, Heart Disease, Cancer, Obesity, Hypertension, Hepatitis, HIV

Imaging DL (CNN):
- Pneumonia (Chest X-ray)
- Tuberculosis (Chest X-ray)

### Clinical decision-support report output
The generated report includes sections such as:
- Current diagnosed diseases summary
- Predicted progression + future risks
- Symptom timeline
- Disease interaction network
- Immunity/Resistance score
- Clinical recommendations
- Explainability section
- Clinical guidance summary

Screenshots of the report output and charts are included in the PDF report (Figures 19–27).  

---

## Model architecture

### Tabular pipeline (per disease)
- preprocessing: missing value imputation + scaling for numeric features, one-hot encoding for categorical features
- model: `RandomForestClassifier(n_estimators=200, random_state=42)`
- evaluation: classification report + accuracy on 20% held-out test split

### Imaging pipelines
**Pneumonia**
- EfficientNetB0 backbone + GlobalAveragePooling + dropout + dense classifier (binary output)
- data augmentation using `ImageDataGenerator`
- reported test accuracy: 62.5%

**Tuberculosis**
- custom 3-layer CNN (Conv2D + MaxPool blocks) + dense classifier (binary output)
- reported accuracy: 96.9% (with slower CPU-bound inference noted)

---

## Explainability & visualization
- SHAP plots for feature attribution (example: glucose + BMI high contribution for diabetes)
- Radar chart for multi-disease risk mapping
- Risk dashboards to visualize detected vs predicted progression vs future risk

---

## Reproducibility

Tested environment (baseline CPU):
- Python 3.10.12
- scikit-learn 1.2.2
- TensorFlow 2.12.0
- pandas 1.5.3
- numpy 1.23.5
- shap 0.42.1

Runtime benchmarks on baseline Colab CPU:
- full pipeline: ~2h15m for all 17 models
- per RF model: ~4–6 minutes average
- pneumonia CNN training: ~1h48m (20 epochs)
- TB inference: ~1.2s/image (CPU)

---

## Datasets (public sources)

The project uses de-identified, public datasets (see Appendix C in the report).  
Below are the cited sources used in the report:

1. Diabetes — Pima Indians Diabetes Dataset  
2. Liver Disease — Indian Liver Patient Records (Kaggle)  
3. Parkinson’s — UCI Parkinson’s Dataset  
4. PCOS — Kaggle PCOS Dataset  
5. Stroke — Kaggle Stroke Prediction Dataset  
6. Cystic Fibrosis — Kaggle CF Gene Expression  
7. Alzheimer’s — Kaggle Alzheimer’s Dataset (4-class images)  
8. Anemia — Kaggle Anemia Dataset  
9. Chronic Kidney Disease — UCI CKD Dataset  
10. Heart Disease — Kaggle Heart Disease (UCI)  
11. Breast Cancer — Wisconsin Breast Cancer Dataset (Kaggle)  
12. Obesity — Kaggle Obesity Levels Dataset  
13. Hypertension — Kaggle Hypertension Risk Data  
14. Pneumonia — Chest X-ray Pneumonia (Kaggle)  
15. Tuberculosis — TB Chest X-ray Dataset (Kaggle)  
16. Hepatitis — UCI Hepatitis Dataset  
17. HIV — Kaggle HIV/AIDS Dataset

> IMPORTANT: If your repo uses CSV exports or preprocessed subsets derived from these sources, document that clearly in `data/README.md`.

---

## How to run (recommended repo layout)

